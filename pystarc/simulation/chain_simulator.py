"""
PySTARC chain Brownian-dynamics simulator.

This module is a Brownian-dynamics simulator for systems made of one flexible
chain and one rigid target. It is the sister module to nam_simulator.py.

In each trajectory the chain undergoes three coupled Brownian motions: a
rigid-body translation of the chain centroid, a rigid-body rotation of the
chain, and a fluctuation of the internal coordinates under bonded and
non-bonded forces.

The internal motion runs on a smaller timestep (dt_chain) than the overall
translation and rotation (dt) because the bonded forces are stiff, so several
internal steps cover each outer step.

The frame is fixed to the target, centered at the origin. With
target_hydro_radius the target diffuses as well: its translation enters the
relative diffusion and its rotation turns the chain about the origin. The
chain starts on the b-sphere at r_start with a random orientation and
propagates until it reacts, escapes or reaches max_steps.
"""

from __future__ import annotations
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.motion.do_bd_step import (
    WATER_VISCOSITY,
    apply_frame_rotation,
    backstep_due_to_force,
    bd_step_wiener,
    bd_step_wiener_tensor,
    ermak_mccammon_rotation_tensor,
    ermak_mccammon_translation_tensor,
    receptor_rotation_vector,
)
from pystarc.simulation.coffdrop_chain import (
    ChainCommon,
    ChainForceEvaluator,
    ChainState,
    bonded_exclusions,
    compute_chain_forces,
    satisfy_constraints_hybrid,
)
from pystarc.forces.chain_gb import chain_full_gb_force
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.hydrodynamics.rotne_prager import (
    MobilityTensor,
    chain_diffusion_tensors,
    radial_D_gradient,
    stokes_rotational_diffusion,
    stokes_translational_diffusion,
)
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Atom, Molecule
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import multiprocessing as mp
import numpy as np
import math
import time
import warnings
from pystarc.global_defs.constants import (
    COULOMB_KCAL,
    KB_KCAL,
    KBT_KCAL,
    MOLAR_RATE_CONV,
    T_DEFAULT,
)
from pystarc.global_defs.defaults import (
    DESOLVATION_ALPHA,
    HYDRODYNAMIC_INTERACTIONS,
    SOLVENT_DIELECTRIC,
    VISCOSITY,
)

# Floor in A^2/ps for the eigenvalues of the relative translational diffusion
# tensor. It is far below any physical value and keeps the Cholesky factor
# finite when the tensor is not positive definite.
RELATIVE_DIFFUSION_FLOOR = 1e-6


def target_spheres(target) -> Tuple[np.ndarray, np.ndarray]:
    """Return the positions and radii of the target atoms with a radius of at
    least 1e-10 Å. Ghost atoms carry no excluded volume and are left out."""
    if target is None:
        return np.zeros((0, 3)), np.zeros(0)
    xyz = np.array([[a.x, a.y, a.z] for a in target.atoms], dtype=float)
    rad = np.array([a.radius for a in target.atoms], dtype=float)
    live = rad >= 1e-10
    return xyz.reshape(-1, 3)[live], rad[live]


def chain_target_steric_forces(
    chain_world_positions: np.ndarray,
    chain_radii: np.ndarray,
    target,
    eps: float = 1.0,
) -> np.ndarray:
    """Compute the WCA repulsion between every chain bead and every target atom.

    The function returns an (n_chain, 3) array of the per-bead force in kBT/Å.
    For each bead-atom pair the σ parameter is the sum of the bead and atom
    radii, and the ε parameter is uniform. The WCA potential and force are the
    same as in chain_intra_nonbonded_forces. The force on chain bead i pushes
    it away from target atom k whenever the two centers are closer than
    2^(1/6) σ, the cutoff where the potential and force reach zero.

    target is a Molecule or the (positions, radii) pair of target_spheres.
    Ghost atoms (radius < 1e-10) on either side are skipped.
    """
    n_chain = len(chain_radii)
    F = np.zeros((n_chain, 3))
    if target is None:
        return F
    target_pos, target_rad = (
        target if isinstance(target, tuple) else target_spheres(target)
    )
    chain_active = chain_radii >= 1e-10
    if target_rad.size == 0 or not chain_active.any():
        return F
    chain_pos_act = chain_world_positions[chain_active]
    chain_rad_act = chain_radii[chain_active]
    # dr[i, k] = chain bead i - target atom k, so the force points along +dr.
    dr = chain_pos_act[:, None, :] - target_pos[None, :, :]
    r = np.sqrt(np.einsum("ikd,ikd->ik", dr, dr))
    sig = chain_rad_act[:, None] + target_rad[None, :]
    active = (r > 1e-10) & (r < 2.0 ** (1.0 / 6.0) * sig)
    if not active.any():
        return F
    ia, ka = np.nonzero(active)
    sr6 = (sig[ia, ka] / r[ia, ka]) ** 6
    fmag_over_r = 4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r[ia, ka] ** 2
    fvec = fmag_over_r[:, None] * dr[ia, ka]
    F_act = np.zeros((chain_pos_act.shape[0], 3))
    for c in range(3):
        F_act[:, c] = np.bincount(ia, weights=fvec[:, c], minlength=len(F_act))
    F[chain_active] = F_act
    return F


def _check_chain_overlap(
    target,
    chain_world_positions: np.ndarray,
    chain_radii: np.ndarray,
) -> bool:
    """Return True if a chain bead overlaps a target atom.

    A bead and an atom overlap when their centers are closer than the sum of
    their radii. target is a Molecule, the (positions, radii) pair of
    target_spheres, or None. Ghost atoms and beads (radius < 1e-10) are
    skipped. Pairs within the chain are not tested; the bonded terms, the
    COFFDROP pair tables and the WCA repulsion hold their distances.
    """
    if target is None:
        return False
    target_pos, target_rad = (
        target if isinstance(target, tuple) else target_spheres(target)
    )
    live = chain_radii >= 1e-10
    if target_rad.size == 0 or not live.any():
        return False
    d = chain_world_positions[live][:, None, :] - target_pos[None, :, :]
    r2 = np.einsum("ikd,ikd->ik", d, d)
    return bool((r2 < (chain_radii[live][:, None] + target_rad[None, :]) ** 2).any())


@dataclass
class ChainBDParameters:
    """Simulation parameters for chain BD.

    The first block mirrors NAMParameters and holds the trajectory count, the
    integration timestep, the b-sphere radii, the RNG seed, the threading
    settings, and so on. The second block is chain-specific. It adds an inner
    timestep dt_chain for integrating the internal coordinates and a tolerance
    for the constraint solver.
    """

    n_trajectories: int = 1_000
    # Outer step in ps, and the outer step near the reaction surface, used
    # while any reaction pair is closer than 1.5 times its cutoff.
    dt: float = 0.2
    dt_rxn: float = 0.05
    minimum_core_dt: float = 0.0
    minimum_core_reaction_dt: float = 0.0
    max_steps: int = 1_000_000
    r_start: float = 100.0
    r_escape: float = 0.0
    # Debye screening length in Å. The default of 7.858 Å corresponds to about
    # 150 mM NaCl at 298 K, consistent with PySTARCConfig.debye_length. Only the
    # LMZ outer propagator uses this parameter. The chain-internal GB and COFFDROP
    # forces do not see it.
    debye_length: float = 7.858
    # Thermal energy, solvent permittivity and hydrodynamics for the LMZ outer
    # propagator, which normalises the encounter rate for the configured solvent
    # conditions.
    temperature_kT: float = KBT_KCAL
    dielectric: float = SOLVENT_DIELECTRIC
    hydrodynamic_interactions: bool = HYDRODYNAMIC_INTERACTIONS
    # When use_lmz is True (the default and correct setting), the LMZ outer
    # propagator handles the b-to-q diffusion zone analytically and returns
    # trajectories to the b-sphere with probability b/q. When False, the code
    # falls back to simple-escape behavior and terminates the trajectory at
    # r_escape. The simple-escape behavior is biased, giving a reaction
    # probability roughly ten times too low, but it is what the older tests were
    # calibrated against.
    use_lmz: bool = True
    seed: Optional[int] = None
    n_threads: int = 1
    verbose: bool = False
    # Chain-specific. The inner steps of an outer step of dt_rxn are scaled
    # by dt_rxn / dt, so the internal and the rigid-body clocks agree when
    # chain_steps_per_outer × dt_chain = dt.
    dt_chain: float = 0.05
    chain_steps_per_outer: int = 4
    # Number of internal-coordinate-only BD steps to run in bulk (with no
    # external forces) before each trajectory's main BD loop. The default of 0
    # means no equilibration, which is bit-exact with the prior code path and
    # consumes no RNG state.
    n_equilibration_steps: int = 0
    constraint_tol: float = 1e-6
    constraint_max_iter: int = 200
    # Whether to subdivide an outer BD step when the external force changes too
    # much across it. This uses NAM's criterion in
    # motion/do_bd_step.py:backstep_due_to_force. It is on by default because
    # without it, simulations crossing steep electrostatic gradients produce
    # systematic kinetic errors. In the worst case it costs one extra force
    # evaluation per outer step.
    force_change_backstep: bool = True
    # Whether to reject steps that overlap a chain bead with a target atom.
    # The outer move is redrawn up to three times, and when every draw
    # overlaps, the whole step, inner steps included, is rejected and the
    # chain stays where it was.
    use_hard_sphere: bool = True
    # Whether to add WCA forces with uniform ε between chain beads and target
    # atoms and between the chain bead pairs that have no bonded term and no
    # COFFDROP table. σ is the radius sum and ε is in kBT.
    use_soft_repulsion: bool = False
    soft_repulsion_eps: float = 1.0
    # Whether to use the Brownian bridge for reaction detection. When True, in
    # addition to the endpoint check, the code also evaluates the path-crossing
    # probability for every contact pair whose pre- and post-step distances are
    # both above its cutoff. This catches reactions that occur between discrete BD
    # steps, a real correctness fix because endpoint-only checks systematically
    # under-count fast reactions. It is on by default because it closes a method
    # gap. With an empty PathwaySet the code path is trivially skipped, so
    # existing tests are unaffected.
    use_brownian_bridge: bool = True
    # Whether to add generalized Born (self-Born) electrostatics on the chain.
    # The default of off preserves bit-exact prior behavior. When True, the chain
    # BD adds full GB forces (the diagonal self-Born term, the off-diagonal
    # cross-term, and the vacuum Coulomb interaction between non-bonded chain
    # pairs) on top of the existing external forces. The path B dispatch is keyed
    # on coffdrop_active. When the COFFDROP pair tables are active, only the
    # diagonal self-Born is added so the off-diagonal does not double-count
    # COFFDROP's pre-screened pair potentials. Effective radii come from OBC2
    # set-II (Onufriev, Bashford, Case 2004).
    use_self_born: bool = False
    # Interior and exterior dielectrics for GB. The defaults are a vacuum interior
    # and water at 300 K for the exterior. Tune them per system if needed.
    gb_eps_in: float = 1.0
    gb_eps_out: float = 78.5
    # OBC2 set-II coefficients. The defaults are the published set-II values. Do
    # not change them unless calibrating an alternative parameterization.
    gb_obc_alpha: float = 1.0
    gb_obc_beta: float = 0.8
    gb_obc_gamma: float = 4.85
    # Path B switch. This is True when the COFFDROP pair tables are loaded for the
    # chain, in which case GB is restricted to the diagonal term only to avoid
    # double-counting the empirical pre-screened electrostatics. The default of
    # False matches current production, where there are no COFFDROP pair forces in
    # the rigid-body benchmark path.
    coffdrop_active: bool = False
    # Whether to rotate the body frame diffusion tensors into the lab frame at
    # the start of each rigid-body step, D_lab = R D_body Rᵀ, with R taken from
    # the orientation at the start of the (sub)step. The default of False hands
    # the body tensors to the step unrotated, the prior behavior.
    rotate_body_tensors: bool = False
    # Number of outer steps between rebuilds of the body frame diffusion
    # tensors from the current internal configuration. A value of 0 never
    # rebuilds. A positive value requires auto_diffusion.
    diffusion_update_interval: int = 0
    # Whether the internal steps remove the rigid-body rotation of the chain
    # (constraint solver, then re-centering, then an unweighted Kabsch fit to
    # the positions before the step). The default of False keeps the prior
    # internal step.
    remove_internal_rotation: bool = False
    # Whether the inner steps feel the forces of the target, evaluated at the
    # start of the outer step and turned into the body frame. Needs
    # remove_internal_rotation, which leaves the rigid-body part of the forces
    # to the outer step.
    inner_field: bool = False

    def __post_init__(self):
        if self.r_escape == 0.0:
            self.r_escape = self.r_start * 1.1


def place_chain(
    body_positions: np.ndarray,
    com: np.ndarray,
    orientation: Quaternion,
) -> np.ndarray:
    """Map chain coordinates from the body frame to the world frame.

    The chain stores its atoms in a body frame whose origin is the chain center
    of mass and whose orientation defines the internal geometry. This function
    applies a rigid-body transformation to produce the world-frame positions used
    for force evaluation against the target and for reaction-criterion checks.

    The argument body_positions is the (n_atoms, 3) array of body-frame atom
    positions, which should already be centered so that their mean across atoms
    is at the origin of the body frame. The argument com is the length-3
    world-frame center of mass, and orientation is the quaternion describing the
    chain orientation in the world frame.

    The function returns the (n_atoms, 3) array of world-frame atom positions.
    """
    R = orientation.to_rotation_matrix()
    return (R @ body_positions.T).T + com


def rotate_tensor_to_lab(D_body, orientation: Quaternion):
    """Rotate a body frame diffusion tensor into the lab frame.

    The result is R D_body Rᵀ, where R is the rotation matrix of the
    orientation. R is taken at the start of the (sub)step that the tensor
    drives, the configuration at which force and torque are evaluated. A
    scalar passes through unchanged because an isotropic D is frame
    independent.
    """
    if np.ndim(D_body) == 0:
        return D_body
    R = orientation.to_rotation_matrix()
    return R @ np.asarray(D_body, dtype=float) @ R.T


def _isotropic_diffusion(D) -> float:
    """Return a scalar D, or tr(D)/3 for a (3, 3) tensor."""
    if np.ndim(D) == 0:
        return float(D)
    return float(np.trace(np.asarray(D, dtype=float)) / 3.0)


def initialize_bsphere(
    rng: np.random.Generator,
    r_start: float,
) -> Tuple[np.ndarray, Quaternion]:
    """Sample a uniformly random starting position and orientation on the b-sphere.

    The position is a vector of magnitude r_start pointing in a random direction
    that is uniformly distributed over the unit sphere. The orientation is a
    uniform-random unit quaternion. Together these define the initial state of
    the chain at the start of a trajectory.

    The argument rng is a numpy random Generator; pass an explicit one for
    reproducibility. The argument r_start is the b-sphere radius in Å.

    The function returns the (3,) chain center-of-mass position and the
    Quaternion giving the chain orientation.
    """
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    pos = v * r_start
    ori = random_quaternion(rng)
    return pos, ori


def check_escape(pos: np.ndarray, r_escape: float) -> bool:
    """Return True if the chain center of mass has strayed past r_escape.

    Used as the trajectory termination condition that stops sampling once
    the chain has diffused too far from the target. Distance is the L2
    norm of pos.
    """
    return float(np.linalg.norm(pos)) >= r_escape


def make_chain_scratch_molecule(chain_template: ChainCommon) -> Molecule:
    """Build a Molecule populated with one Atom per chain atom.

    The atoms get radius and charge from the template; their positions are
    initialized to zero and updated in place each step via
    update_chain_scratch_positions. Building the scratch once and reusing
    it avoids per-step allocation overhead.
    """
    atoms = []
    for i, ca in enumerate(chain_template.atoms):
        atom = Atom(
            index=i,
            name=ca.beadname if hasattr(ca, "beadname") else "",
            residue_name=ca.resname,
            residue_index=ca.resid,
            radius=ca.radius,
            charge=ca.charge,
        )
        atoms.append(atom)
    return Molecule(name=chain_template.name, atoms=atoms)


def update_chain_scratch_positions(
    scratch: Molecule, world_positions: np.ndarray
) -> None:
    """Copy world-frame positions into the scratch Molecule's atoms.

    Updates scratch.atoms[i].x, y, z in place. world_positions must be
    shape (n_atoms, 3) and have the same atom ordering as the scratch.
    """
    if world_positions.shape != (len(scratch.atoms), 3):
        raise ValueError(
            f"world_positions shape {world_positions.shape} does not match "
            f"({len(scratch.atoms)}, 3) for the scratch molecule"
        )
    for atom, p in zip(scratch.atoms, world_positions):
        atom.x = float(p[0])
        atom.y = float(p[1])
        atom.z = float(p[2])


def check_chain_reaction(
    target: Molecule,
    chain_scratch: Molecule,
    pathway_set: PathwaySet,
    rng: Optional[np.random.Generator] = None,
) -> Optional[str]:
    """Check whether any reaction in pathway_set fires for the current chain pose.

    The chain's world-frame positions must already have been written into
    chain_scratch (via update_chain_scratch_positions) before this call.
    Returns the name of the first reaction that fires, or None.
    """
    return pathway_set.check_all(target, chain_scratch, rng)


def compute_pair_distances(
    target: Optional[Molecule],
    chain_world_positions: np.ndarray,
    pathway_set: Optional[PathwaySet],
) -> "list[np.ndarray]":
    """Compute per-reaction contact-pair distances for Brownian bridge tracking.

    The function returns a list parallel to pathway_set.reactions, where each
    element is a 1D array of distances, one per ContactPair in that reaction. The
    distance for the pair (i_target, j_chain) is

        |target.atoms[i_target].position - chain_world_positions[j_chain]|

    Here the two terms are the world-frame position of the target atom and the
    world-frame position of the chain bead. If pathway_set is None or has no
    reactions, the function returns an empty list. The function is called by
    check_reaction_with_bridge to capture the pre-step distances, and it is
    called again after the step to compute the new distances for the bridge
    evaluation.

    Pass target as a Molecule with an .atoms attribute, or None to short-circuit.
    The argument chain_world_positions is the (n_chain, 3) array of world-frame
    bead positions, and pathway_set is the PathwaySet whose reactions are tracked.

    The function returns a list of (n_pairs_in_rxn,) distance arrays, one per
    reaction.
    """
    if pathway_set is None or target is None:
        return []
    out = []
    for rxn in pathway_set.reactions:
        pairs = rxn.criteria.pairs
        atoms = [target.atoms[p.mol1_atom_index] for p in pairs]
        t_pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float).reshape(-1, 3)
        c_idx = np.array([p.mol2_atom_index for p in pairs], dtype=np.int64)
        d = t_pos - chain_world_positions[c_idx]
        out.append(np.sqrt(np.einsum("ij,ij->i", d, d)))
    return out


def check_reaction_with_bridge(
    target: Optional[Molecule],
    chain_scratch: Molecule,
    pathway_set: PathwaySet,
    old_pair_dists_per_rxn: "list[np.ndarray]",
    new_pair_dists_per_rxn: "list[np.ndarray]",
    D_eff: float,
    dt: float,
    rng: np.random.Generator,
    rng_bb: Optional[np.random.Generator] = None,
) -> Optional[str]:
    """Check for reactions using the Brownian bridge.

    For each reaction this computes a per-pair "fired" flag from both the
    endpoint distance and the Brownian bridge crossing probability

        p_cross = exp(-x0 * x1 / (D_eff * dt))    when x0 > 0 and x1 > 0

    where x = pair_distance - cutoff is the signed height of the pair above the
    reaction surface, x0 is the pre-step value, and x1 is the post-step value. A
    pair is counted as fired in either of two cases. It fires if its endpoint
    distance is below the cutoff, meaning x1 < 0. It also fires if the bridge
    sample u is below p_cross, meaning the path crossed the surface during the
    step.

    A reaction fires when the count of fired pairs reaches n_needed for that
    reaction, matching the AND/OR semantics of pathway_set.check_all, where
    n_needed = -1 means all pairs must fire. The function returns the name of the
    first reaction that fires, or None.

    If the old and new distance lists are empty (no reactions) or have
    inconsistent lengths, the function falls back to the endpoint-only check in
    check_chain_reaction.

    The arguments target, chain_scratch, and pathway_set are the same as in
    check_chain_reaction. The argument old_pair_dists_per_rxn holds the pre-step
    distances per reaction (from compute_chain_pair_distances), and
    new_pair_dists_per_rxn holds the post-step distances per reaction. D_eff is
    the effective diffusion coefficient in Å²/ps for the bridge formula, one
    value for all pairs or a list with one array of per-pair values for each
    reaction. dt is the timestep in ps. rng is a numpy random Generator for the
    reaction-probability gate, consumed only when rxn.probability < 1.0.

    The argument rng_bb is an optional independent Generator for the bridge
    samples, drawing one uniform per pair where both x0 and x1 are above zero.
    When it is None (the default), the bridge samples come from rng. That
    preserves backward compatibility but causes the main rng stream to advance
    differently between bridge_on and bridge_off runs, which breaks strict
    monotonicity comparisons. Passing an independent rng_bb makes the bridge a
    purely additive operator on the trajectory.

    The function returns the reaction name as a string for the first firing
    reaction, otherwise None.
    """
    # Fall back to the endpoint-only check when we lack matching distance data.
    if (
        not old_pair_dists_per_rxn
        or not new_pair_dists_per_rxn
        or len(old_pair_dists_per_rxn) != len(new_pair_dists_per_rxn)
    ):
        return check_chain_reaction(target, chain_scratch, pathway_set, rng)

    for rxn_idx, rxn in enumerate(pathway_set.reactions):
        D = D_eff[rxn_idx] if isinstance(D_eff, (list, tuple)) else D_eff
        Ddt = np.maximum(np.asarray(D, dtype=float) * dt, 1e-30)
        old_d = old_pair_dists_per_rxn[rxn_idx]
        new_d = new_pair_dists_per_rxn[rxn_idx]
        cutoffs = np.array(
            [pair.distance_cutoff for pair in rxn.criteria.pairs],
            dtype=float,
        )
        x0 = old_d - cutoffs
        x1 = new_d - cutoffs

        # A pair fires at the endpoint when it is below the cutoff at the new
        # position.
        endpoint_fired = x1 < 0
        # A pair fires through the bridge when both x0 and x1 are above zero and
        # the sample says the path crossed the surface.
        both_above = (x0 > 0) & (x1 > 0)
        p_cross = np.zeros_like(x0)
        np.subtract(0.0, x0 * x1 / Ddt, out=p_cross, where=both_above)
        np.exp(p_cross, out=p_cross, where=both_above)
        # Where both_above is False, p_cross stays at its initialized value of 0.
        p_cross = np.where(both_above, p_cross, 0.0)
        # Draw the bridge sample. Use rng_bb when it is provided, since an
        # independent stream makes the bridge additive on the trajectory and
        # leaves the main rng untouched. Fall back to rng when rng_bb is None.
        bb_rng = rng_bb if rng_bb is not None else rng
        u = bb_rng.random(size=p_cross.shape)
        bridge_fired = u < p_cross
        pair_fired = endpoint_fired | bridge_fired
        # Apply the n_needed logic, where -1 means all pairs must fire.
        n_needed = rxn.criteria.n_needed
        threshold = len(rxn.criteria.pairs) if n_needed < 0 else n_needed
        if threshold == 0:
            # A reaction with no pair requirement would always fire. Guard
            # against this degenerate case so it does not trigger silently.
            continue
        n_fired = int(pair_fired.sum())
        if n_fired >= threshold:
            # Gate on the reaction probability, mirroring check_all semantics.
            if rxn.probability < 1.0:
                if rng.random() >= rxn.probability:
                    continue
            return rxn.name
    return None


def evaluate_target_grid_force_on_chain(
    chain_world_positions: np.ndarray,
    chain_charges: np.ndarray,
    grid: DXGrid,
) -> Tuple[np.ndarray, float]:
    """Evaluate the per-atom electrostatic force and total energy on the chain.

    The forces come from a precomputed Poisson-Boltzmann potential grid of the
    target. The grid is built once, typically by APBS, and represents the
    target's electrostatic potential φ as a 3D array of values in kBT/e. Each
    chain atom feels a force F_i = -q_i grad(φ)(r_i) and contributes q_i φ(r_i)
    to the total interaction energy. Here q_i is the partial charge of atom i and
    r_i is its position. Both quantities are computed by trilinear interpolation
    on the grid.

    For chain atoms whose positions fall outside the grid box, the grid routines
    return zero, so atoms far from the target contribute no electrostatic
    interaction.

    The argument chain_world_positions is the (n_atoms, 3) array of world-frame
    positions in Å, chain_charges is the (n_atoms,) array of partial charges in
    units of e, and grid is the DXGrid holding the target's PB potential.

    The function returns the (n_atoms, 3) array of forces in kBT/Å on each chain
    atom and the total interaction energy in kBT.
    """
    if chain_world_positions.shape[0] != chain_charges.shape[0]:
        raise ValueError(
            f"position count {chain_world_positions.shape[0]} does not "
            f"match charge count {chain_charges.shape[0]}"
        )
    forces = grid.batch_force_on_charges(chain_world_positions, chain_charges)
    potentials = grid.batch_interpolate(chain_world_positions)
    energy = float(np.sum(chain_charges * potentials))
    return forces, energy


DEFAULT_DESOLVATION_ALPHA = DESOLVATION_ALPHA


def evaluate_born_force_on_chain(
    chain_world_positions: np.ndarray,
    chain_charges: np.ndarray,
    grid: DXGrid,
    alpha: float = DEFAULT_DESOLVATION_ALPHA,
) -> Tuple[np.ndarray, float]:
    """Evaluate the per-atom Born desolvation force and total energy on the chain.

    The forces come from a precomputed Born desolvation potential grid of the
    target. The Born grid g(r) is built once and represents the target's
    desolvation field as a 3D array. Each chain atom feels a self-energy
    contribution and a corresponding force

        V_i  =  α * q_i^2 * g(r_i)
        F_i  = -α * q_i^2 * grad(g)(r_i)

    where q_i is the partial charge of atom i and r_i is its position. This
    matches the convention used in the rigid-body engine (engine.py). The
    prefactor α defaults to unity, because the grid holds the Kirkwood n = 1
    cavity self energy with the rigorous normalisation already folded in, so a
    charge sees dG = α q² a³ / r⁴ directly. Both the
    force and the energy are computed by trilinear interpolation on the grid.

    For chain atoms whose positions fall outside the grid box, the grid routines
    return zero, so atoms far from the target see no desolvation cost.

    The argument chain_world_positions is the (n_atoms, 3) array of world-frame
    positions in Å, chain_charges is the (n_atoms,) array of partial charges in
    units of e, and grid is the DXGrid holding the target's Born potential. The
    argument alpha is the desolvation prefactor in kBT/(e² × grid_unit) and
    defaults to DEFAULT_DESOLVATION_ALPHA. It is configurable so that chain-BD
    Born can be compared head-to-head with the rigid-body engine for the same
    system.

    The function returns the (n_atoms, 3) array of forces in kBT/Å on each chain
    atom and the total Born self-energy in kBT.
    """
    if chain_world_positions.shape[0] != chain_charges.shape[0]:
        raise ValueError(
            f"position count {chain_world_positions.shape[0]} does not "
            f"match charge count {chain_charges.shape[0]}"
        )
    # Per-atom gradient of the Born field, with shape (n_atoms, 3).
    grads = grid.batch_gradient(chain_world_positions)
    # Per-atom interpolated Born potential, with shape (n_atoms,).
    potentials = grid.batch_interpolate(chain_world_positions)
    # The force is F_i = -α q_i^2 grad(g)(r_i), so the per-atom coefficient is
    # -α q^2.
    coeffs = -alpha * chain_charges * chain_charges  # (n_atoms,)
    forces = coeffs[:, None] * grads  # (n_atoms, 3)
    # The energy is V = α sum_i q_i^2 g(r_i).
    energy = float(alpha * np.sum(chain_charges * chain_charges * potentials))
    return forces, energy


def chain_internal_bd_step(
    state: ChainState,
    dt: float,
    rng: np.random.Generator,
    kT: float = 1.0,
    viscosity: float = WATER_VISCOSITY,
    apply_constraints: bool = True,
    constraint_tol: float = 1e-6,
    constraint_max_iter: int = 200,
    soft_repulsion: bool = False,
    soft_repulsion_eps: float = 1.0,
    remove_rigid_rotation: bool = False,
    external_forces: Optional[np.ndarray] = None,
) -> None:
    """Take one Brownian-dynamics step on the chain's internal coordinates.

    This implements the Ermak-McCammon update per atom with independent
    Stokes-Einstein mobilities and no hydrodynamic coupling between chain atoms.
    For each atom i with radius a_i the update is

        mob_i = 1 / (6 π η a_i)
        Δr_i = mob_i * F_i * Δt + sqrt(2 kT mob_i Δt) * ξ_i

    where η is the solvent viscosity, F_i is the bonded force on atom i, and ξ_i
    is a unit Gaussian 3-vector. After all atoms are updated, the chain is
    re-centered at the origin so that the body frame stays valid, and the
    constraints are satisfied through the hybrid SHAKE/Newton solver.

    The argument state is the ChainState whose positions are advanced in place.
    dt is the integration timestep in ps, and rng is a numpy random Generator for
    the Wiener increments. kT is the thermal energy; PySTARC uses kT = 1
    throughout, and this parameter is exposed mainly for unit tests that need to
    vary it. viscosity is the solvent viscosity in kBT·ps/Å³ and defaults to the
    water value used elsewhere in the package. When apply_constraints is True
    (the default), the hybrid constraint solver is called after the integration
    step; set it to False only for diagnostic tests of the integrator.
    constraint_tol is the convergence tolerance for the constraint solver, and
    constraint_max_iter is the hard cap on solver iterations.

    With remove_rigid_rotation False the order is update, re-centering, and
    constraint solver. With True the order is update, constraint solver,
    re-centering, and a rotation of the new positions onto the old ones, the
    unweighted Kabsch fit. The rotation leaves every distance, angle, and
    dihedral unchanged, so the constraints stay satisfied and the internal
    coordinate distribution is the same. The fit is skipped for fewer than
    three atoms and for a collinear chain, where no rotation is defined.

    external_forces is an optional (n_atoms, 3) array of forces in the body
    frame, added to the internal forces. With remove_rigid_rotation the step
    keeps only their effect on the shape of the chain.
    """
    # Compute the internal forces from the bonded interactions, optionally adding
    # the intra-chain WCA non-bonded forces.
    compute_chain_forces(
        state,
        kT=kT,
        soft_repulsion=soft_repulsion,
        soft_repulsion_eps=soft_repulsion_eps,
    )
    if external_forces is not None:
        state.forces += external_forces
    # Apply the per-atom Ermak-McCammon update.
    n_atoms = state.n_atoms
    pos = state.positions
    old = pos.copy() if remove_rigid_rotation else None
    forces = state.forces
    radii = np.array([a.radius for a in state.common.atoms])
    mobilities = 1.0 / (6.0 * math.pi * viscosity * radii)
    drifts = mobilities[:, None] * forces * dt
    sigmas = np.sqrt(2.0 * kT * mobilities * dt)
    noise = sigmas[:, None] * rng.standard_normal((n_atoms, 3))
    # Clamp the per-atom drift defensively to prevent any single atom from taking
    # an unphysical displacement caused by near-singular intra-chain WCA forces
    # when chain atoms get too close. A typical inner step displaces atoms by less
    # than 0.1 Å, so a drift larger than MAX_INNER_DRIFT_A signals a near-singular
    # force kick. The offending atom's drift is rescaled to the cap while
    # preserving its direction, and the noise term is left unaffected.
    MAX_INNER_DRIFT_A = 5.0
    drift_mags = np.linalg.norm(drifts, axis=1)
    overshoot = drift_mags > MAX_INNER_DRIFT_A
    if overshoot.any():
        scale = MAX_INNER_DRIFT_A / drift_mags[overshoot]
        drifts[overshoot] = drifts[overshoot] * scale[:, None]
    pos += drifts + noise
    if remove_rigid_rotation:
        # Project back onto the constraint manifold first, then remove the rigid
        # motion that the update and the projection left in the positions.
        if apply_constraints and (
            state.common.length_constraints or state.common.coplanar_constraints
        ):
            satisfy_constraints_hybrid(
                state,
                tol=constraint_tol,
                shake_max_iter=constraint_max_iter,
                newton_max_iter=50,
            )
        pos = state.positions
        pos -= pos.mean(axis=0)
        if n_atoms >= 3:
            # The old positions are the centered body frame. The matrix H is
            # the cross covariance of the new and old positions.
            H = pos.T @ (old - old.mean(axis=0))
            U, S, Vt = np.linalg.svd(H)
            if S[0] > 0.0 and S[1] >= 1e-12 * S[0]:
                sgn = np.sign(np.linalg.det(Vt.T @ U.T))
                if sgn == 0.0:
                    sgn = 1.0
                Q = Vt.T @ np.diag([1.0, 1.0, sgn]) @ U.T
                pos[:] = pos @ Q.T
        return
    # Re-center at the origin to keep the body frame consistent with place_chain,
    # which assumes the body positions are centered on the center of mass.
    pos -= pos.mean(axis=0)
    # Project back onto the constraint manifold.
    if apply_constraints and (
        state.common.length_constraints or state.common.coplanar_constraints
    ):
        satisfy_constraints_hybrid(
            state,
            tol=constraint_tol,
            shake_max_iter=constraint_max_iter,
            newton_max_iter=50,
        )


def aggregate_chain_external_force_and_torque(
    chain_world_positions: np.ndarray,
    per_atom_forces: np.ndarray,
    com: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce per-atom external forces to a net force and torque on the chain.

    The net force is the vector sum of all per-atom forces. The net torque about
    the chain center of mass is the sum over all atoms of (r_i - com) × F_i, where
    r_i is the position of atom i, com is the center of mass, and F_i is the force
    on atom i. These two quantities drive the rigid-body translation and rotation
    of the chain in the outer Brownian-dynamics step.

    The argument chain_world_positions is the (n_atoms, 3) array of atom positions
    in the world frame, per_atom_forces is the (n_atoms, 3) array of external
    forces on each atom, and com is the (3,) chain center of mass in the world
    frame.

    The function returns the (3,) net force on the chain and the (3,) net torque
    about com.
    """
    if chain_world_positions.shape != per_atom_forces.shape:
        raise ValueError(
            f"position shape {chain_world_positions.shape} does not match "
            f"force shape {per_atom_forces.shape}"
        )
    force_net = per_atom_forces.sum(axis=0)
    arms = chain_world_positions - com
    torque_net = np.cross(arms, per_atom_forces).sum(axis=0)
    return force_net, torque_net


def chain_outer_bd_step(
    pos: np.ndarray,
    ori: Quaternion,
    chain_world_positions: np.ndarray,
    per_atom_forces: np.ndarray,
    D_trans,
    D_rot,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Quaternion]:
    """Take one Brownian-dynamics step on the chain rigid-body coordinates.

    This aggregates the per-atom external forces into a net force and torque on
    the chain center of mass, then advances (pos, ori) by one timestep dt using
    either the scalar or the tensor BD step depending on the type of D_trans and
    D_rot.

    The argument pos is the (3,) chain center-of-mass position before the step,
    and ori is the Quaternion giving the chain orientation before the step. The
    argument chain_world_positions is the (n_atoms, 3) array of world-frame atom
    positions used for the torque-arm calculation, and per_atom_forces is the
    (n_atoms, 3) array of external forces on each atom. D_trans is the
    translational diffusion coefficient, either a scalar (isotropic, in Å²/ps) or
    a (3, 3) tensor (anisotropic). D_rot is the rotational diffusion coefficient,
    either a scalar (in rad²/ps) or a (3, 3) tensor. dt is the timestep in ps, and
    rng is a numpy random Generator for the Wiener increments.

    The function returns the (3,) updated chain center-of-mass position and the
    Quaternion giving the updated chain orientation.

    Both D_trans and D_rot must be of the same kind, either both scalar or both
    tensor. Mixed types are not currently supported because the rigid-body BD step
    combines them in a single call to either bd_step_wiener or
    bd_step_wiener_tensor.
    """
    force_net, torque_net = aggregate_chain_external_force_and_torque(
        chain_world_positions,
        per_atom_forces,
        pos,
    )
    dW_t = math.sqrt(dt) * rng.standard_normal(3)
    dW_r = math.sqrt(dt) * rng.standard_normal(3)
    # Detect whether the inputs are tensors or scalars from their ndim. A scalar
    # float has no ndim attribute, so we use np.ndim, which handles both, and a
    # (3, 3) tensor has ndim == 2.
    is_tensor_t = np.ndim(D_trans) == 2
    is_tensor_r = np.ndim(D_rot) == 2
    if is_tensor_t != is_tensor_r:
        raise ValueError(
            "D_trans and D_rot must be the same kind: either both scalars "
            "or both (3, 3) tensors. Got "
            f"D_trans ndim={np.ndim(D_trans)}, D_rot ndim={np.ndim(D_rot)}."
        )
    if is_tensor_t:
        new_pos, new_ori = bd_step_wiener_tensor(
            pos,
            ori,
            force_net,
            torque_net,
            D_trans,
            D_rot,
            dt,
            dW_t,
            dW_r,
        )
    else:
        new_pos, new_ori = bd_step_wiener(
            pos,
            ori,
            force_net,
            torque_net,
            D_trans,
            D_rot,
            dt,
            dW_t,
            dW_r,
        )
    # Clamp the displacement defensively to reject the unphysical moves caused by
    # near-singular WCA forces at overlapping configurations. Normal BD steps with
    # D ≈ 0.02 Å²/ps and dt ≈ 0.2 ps displace the chain by less than 1 Å. A step
    # larger than MAX_OUTER_DISP_A signals an overlap-driven force kick, so the
    # chain stays put rather than propagate divergent dynamics. The next BD step
    # samples new noise and may escape the overlap.
    MAX_OUTER_DISP_A = 5.0
    if np.linalg.norm(new_pos - pos) > MAX_OUTER_DISP_A:
        return pos, ori
    return new_pos, new_ori


def chain_outer_bd_step_wiener(
    pos: np.ndarray,
    ori: Quaternion,
    chain_world_positions: np.ndarray,
    per_atom_forces: np.ndarray,
    D_trans,
    D_rot,
    dt: float,
    dW_t: np.ndarray,
    dW_r: np.ndarray,
) -> Tuple[np.ndarray, Quaternion]:
    """Advance the rigid-body coordinates using pre-drawn Wiener increments.

    This is the sibling of chain_outer_bd_step. It has the same physics, since it
    aggregates the per-atom forces and then advances pos and ori by dt with either
    the scalar or the tensor BD step, but it accepts the Wiener increments as
    inputs rather than drawing them internally. This is required for the
    force-change backstep subdivision, where the caller must keep the original
    Wiener increments so it can build a midpoint pair (dW_mid, dW - dW_mid) that
    preserves the Brownian bridge statistics.

    The dW arrays must already be scaled by sqrt(dt). At the simulator level this
    is the same shape NAM uses for its scalar bd_step_wiener.

    The arguments pos, ori, chain_world_positions, per_atom_forces, D_trans,
    D_rot, and dt are the same as in chain_outer_bd_step. The arguments dW_t and
    dW_r are the (3,) Wiener increments scaled by sqrt(dt), which the caller is
    responsible for supplying.
    """
    force_net, torque_net = aggregate_chain_external_force_and_torque(
        chain_world_positions,
        per_atom_forces,
        pos,
    )
    is_tensor_t = np.ndim(D_trans) == 2
    is_tensor_r = np.ndim(D_rot) == 2
    if is_tensor_t != is_tensor_r:
        raise ValueError(
            "D_trans and D_rot must be the same kind: either both scalars "
            "or both (3, 3) tensors. Got "
            f"D_trans ndim={np.ndim(D_trans)}, D_rot ndim={np.ndim(D_rot)}."
        )
    if is_tensor_t:
        new_pos, new_ori = bd_step_wiener_tensor(
            pos,
            ori,
            force_net,
            torque_net,
            D_trans,
            D_rot,
            dt,
            dW_t,
            dW_r,
        )
    else:
        new_pos, new_ori = bd_step_wiener(
            pos,
            ori,
            force_net,
            torque_net,
            D_trans,
            D_rot,
            dt,
            dW_t,
            dW_r,
        )
    # Clamp the displacement defensively to reject the unphysical moves caused by
    # near-singular WCA forces at overlapping configurations. Normal BD steps with
    # D ≈ 0.02 Å²/ps and dt ≈ 0.2 ps displace the chain by less than 1 Å. A step
    # larger than MAX_OUTER_DISP_A signals an overlap-driven force kick, so the
    # chain stays put rather than propagate divergent dynamics. The next BD step
    # samples new noise and may escape the overlap.
    MAX_OUTER_DISP_A = 5.0
    if np.linalg.norm(new_pos - pos) > MAX_OUTER_DISP_A:
        return pos, ori
    return new_pos, new_ori


def trajectory_rngs(
    seed: Optional[int], traj_idx: int
) -> Tuple[np.random.Generator, np.random.Generator]:
    """Return the main and the bridge generator of trajectory traj_idx.

    Both are spawned from SeedSequence(seed, spawn_key=(traj_idx,)), so no
    stream of one trajectory repeats a stream of another.
    """
    seq = np.random.SeedSequence(0 if seed is None else seed, spawn_key=(traj_idx,))
    main, bridge = seq.spawn(2)
    return np.random.default_rng(main), np.random.default_rng(bridge)


# The simulator of a worker process. The pool initializer sets it once, so the
# grids and the force field tables are not sent again with every trajectory.
_WORKER_SIM = None


def _chain_worker_init(sim) -> None:
    global _WORKER_SIM
    _WORKER_SIM = sim


def _run_chain_trajectory_worker(traj_idx: int):
    """Run trajectory traj_idx of the worker's simulator with its own streams."""
    rng, rng_bb = trajectory_rngs(_WORKER_SIM.params.seed, traj_idx)
    return _WORKER_SIM.run_one(rng=rng, rng_bb=rng_bb)


def _run_chain_trajectory_indexed(traj_idx: int):
    """Return (traj_idx, TrajectoryResult) for unordered pool collection."""
    return traj_idx, _run_chain_trajectory_worker(traj_idx)


def _build_chain_diagnostics(
    path_steps_buf: List[int],
    path_com_buf: List[np.ndarray],
    path_q_buf: List[List[float]],
    radial_buf: List[float],
    energy_buf: List[List[float]],
    contact_count_buf: Dict[Tuple[int, int], int],
) -> Dict[str, Any]:
    """Build the per-trajectory diagnostic kwargs for TrajectoryResult.

    This converts the running buffers populated by ChainBDSimulator.run_one()
    into numpy arrays suitable for storage on TrajectoryResult. Empty buffers map
    to None through absent dict keys, so downstream writers can detect the absence
    of data cleanly.
    """
    d: Dict[str, Any] = {}
    if path_steps_buf:
        d["path_steps"] = np.asarray(path_steps_buf, dtype=int)
        d["path_com"] = np.asarray(path_com_buf, dtype=float)
        d["path_q"] = np.asarray(path_q_buf, dtype=float)
    if radial_buf:
        d["radial_trace"] = np.asarray(radial_buf, dtype=float)
    if energy_buf:
        d["energy_steps"] = np.asarray(energy_buf, dtype=float)
    if contact_count_buf:
        d["contact_counts"] = dict(contact_count_buf)
    return d


class ChainBDSimulator:
    """Brownian-dynamics simulator for a flexible chain and a rigid target.

    Each trajectory starts with the chain at a random point on the b-sphere with
    a random orientation and propagates until the chain either satisfies a
    reaction criterion or escapes past the escape sphere. The chain undergoes
    three coupled motions, namely rigid-body translation, rigid-body rotation, and
    internal coordinate fluctuation.

    The internal coordinates advance on a smaller timestep dt_chain than the
    overall translation and rotation timestep dt, and chain_steps_per_outer
    internal steps are taken per outer step.

    External forces from the target are computed from a precomputed
    Poisson-Boltzmann potential grid (DXGrid). If no grid is provided
    (target_grid=None), the chain experiences no electrostatic force and pure
    diffusion drives the rigid-body motion.

    Born desolvation can be added on top of the electrostatic field by passing a
    separate Born potential grid through born_grid. The Born force on bead i is
    F_i = -α q_i^2 grad(g)(r_i), where q_i is the bead charge and r_i is its
    position, and it is summed into the per-atom external force alongside the
    electrostatic contribution. The desolvation prefactor α defaults to unity,
    the physically correct base for the present cavity self energy grids.

    The target is fixed at the origin by default, and the chain alone diffuses
    with the diffusion tensors D_trans and D_rot. Passing target_hydro_radius,
    the hydrodynamic radius a_T of the target in Å, switches on the two-body
    treatment. The target then rotates about the origin with the Stokes
    D_r,T and translates with the Stokes D_t,T, the relative translation uses
    D_rel = R D_t,chain Rᵀ + (D_t,T - Δ(r)) I with the two-sphere
    Rotne-Prager-Yamakawa reduction Δ(r) when hydrodynamic_interactions is on,
    and the outer propagator receives the two real diffusion groups. The
    default of None keeps the fixed target.
    """

    def __init__(
        self,
        target: Molecule,
        chain_template: ChainCommon,
        chain_init_body_positions: np.ndarray,
        params: ChainBDParameters,
        pathway_set: "PathwaySet",
        D_trans=None,
        D_rot=None,
        target_grid: "Optional[DXGrid]" = None,
        born_grid: "Optional[DXGrid]" = None,
        desolvation_alpha: float = DEFAULT_DESOLVATION_ALPHA,
        auto_diffusion: bool = False,
        outputs=None,
        target_hydro_radius: Optional[float] = None,
    ):
        if params.diffusion_update_interval < 0:
            raise ValueError(
                "diffusion_update_interval must be >= 0, got "
                f"{params.diffusion_update_interval}"
            )
        if params.diffusion_update_interval > 0 and not auto_diffusion:
            raise ValueError(
                "diffusion_update_interval > 0 requires auto_diffusion=True, "
                "because only tensors computed from the chain geometry can be "
                "rebuilt. Explicit D_trans and D_rot are fixed."
            )
        if params.inner_field and not params.remove_internal_rotation:
            raise ValueError(
                "inner_field requires remove_internal_rotation, which keeps the "
                "rigid-body part of the target forces out of the inner steps"
            )
        a_target = None
        if target_hydro_radius is not None:
            a_target = float(target_hydro_radius)
            if not (math.isfinite(a_target) and a_target > 0.0):
                raise ValueError(
                    "target_hydro_radius must be a finite number > 0, got "
                    f"{target_hydro_radius}"
                )
            if target is None:
                raise ValueError("target_hydro_radius requires a target")
            # The bead tensors and the target D use WATER_VISCOSITY and
            # ETA_WATER at T_DEFAULT, the outer propagator uses the input kT.
            # The two agree only at T_DEFAULT.
            _T = params.temperature_kT / KB_KCAL
            if abs(_T - T_DEFAULT) > 0.01:
                raise ValueError(
                    "target_hydro_radius requires the temperature "
                    f"{T_DEFAULT} K, got {_T:.3f} K. The viscosity convention "
                    "of the bead tensors is fixed at that temperature."
                )
        if chain_init_body_positions.shape != (len(chain_template.atoms), 3):
            raise ValueError(
                "chain_init_body_positions shape "
                f"{chain_init_body_positions.shape} does not match "
                f"({len(chain_template.atoms)}, 3) for the chain template"
            )
        com_init = chain_init_body_positions.mean(axis=0)
        if not np.allclose(com_init, 0.0, atol=1e-6):
            raise ValueError(
                "chain_init_body_positions should be centered at origin "
                f"(its mean is {com_init}); call positions -= positions.mean(axis=0) "
                "before constructing the simulator"
            )
        # Resolve the diffusion coefficients. Three mutually exclusive modes are
        # supported. In the first, auto_diffusion is True with no explicit D_trans
        # or D_rot, and the (3, 3) D_trans and D_rot are computed from the chain
        # geometry through chain_diffusion_tensors, giving the full anisotropic
        # Rotne-Prager-Yamakawa tensors. In the second, auto_diffusion is False
        # with explicit scalar D_trans and D_rot, the backward-compatible path
        # used by all earlier tests. In the third, auto_diffusion is False with
        # explicit tensor D_trans and D_rot, where the user supplies a
        # pre-computed anisotropic D directly.
        if auto_diffusion:
            if D_trans is not None or D_rot is not None:
                raise ValueError(
                    "auto_diffusion=True is incompatible with explicitly "
                    "supplied D_trans or D_rot. Either set auto_diffusion=True "
                    "and let the simulator compute D from chain geometry, "
                    "or pass D_trans and D_rot directly."
                )
            radii = np.array(
                [a.radius for a in chain_template.atoms],
                dtype=float,
            )
            # With rotate_body_tensors the tensors are taken about the unweighted
            # centroid, which is the origin of the centered positions. This is
            # the center of the periodic rebuilds and of the chain rotation.
            if params.rotate_body_tensors:
                D_trans_auto, D_rot_auto, _ = chain_diffusion_tensors(
                    chain_init_body_positions,
                    radii,
                    center=np.zeros(3),
                )
            else:
                D_trans_auto, D_rot_auto, _ = chain_diffusion_tensors(
                    chain_init_body_positions,
                    radii,
                )
            D_trans = D_trans_auto
            D_rot = D_rot_auto
        else:
            if D_trans is None or D_rot is None:
                raise ValueError(
                    "Either set auto_diffusion=True to compute D from chain "
                    "geometry, or supply both D_trans and D_rot explicitly. "
                    f"Got D_trans={D_trans}, D_rot={D_rot}."
                )
        self.target = target
        self.chain_template = chain_template
        self.body_positions = chain_init_body_positions.copy()
        self.params = params
        self.pathway_set = pathway_set
        self.D_trans = D_trans
        self.D_rot = D_rot
        self.target_grid = target_grid
        self.born_grid = born_grid
        self.desolvation_alpha = float(desolvation_alpha)
        self.auto_diffusion = auto_diffusion
        # Two-body hydrodynamics. These stay None for the fixed target.
        self._a_target = a_target
        self._D_t_target = None
        self._D_r_target = None
        self._a_chain = None
        self._pair_mob = None
        self._D0_pair = None
        self._warned_regularized = False
        self._warned_relative_clip = False
        self._n_relative_clipped = 0
        if a_target is not None:
            D_c0 = _isotropic_diffusion(self.D_trans)
            D_rc0 = _isotropic_diffusion(self.D_rot)
            if not (D_c0 > 0.0 and D_rc0 > 0.0):
                raise ValueError(
                    "target_hydro_radius requires D_trans and D_rot of the "
                    f"chain > 0, got tr(D_trans)/3={D_c0} and tr(D_rot)/3={D_rc0}"
                )
            self._D_t_target = stokes_translational_diffusion(a_target)
            self._D_r_target = stokes_rotational_diffusion(a_target)
            self._a_chain = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * D_c0)
            self._D0_pair = self._D_t_target + D_c0
            if params.hydrodynamic_interactions:
                self._pair_mob = MobilityTensor(
                    D_trans1=self._D_t_target,
                    D_rot1=self._D_r_target,
                    D_trans2=D_c0,
                    D_rot2=D_rc0,
                    radius1=a_target,
                    radius2=self._a_chain,
                    use_rpy=True,
                )
        # OutputConfig controls the diagnostic recording cadence (save_interval)
        # and the per-file flags. Default to a fresh OutputConfig() when the
        # caller did not supply one, which preserves backward compatibility with
        # the existing test fixtures.
        if outputs is None:
            from pystarc.pipeline.input_parser import OutputConfig

            outputs = OutputConfig()
        self.outputs = outputs
        # Cache the target atom positions for fast distance checks during the
        # contact-frequency diagnostic sampling. They are built once at
        # construction and reused across all sample steps.
        if self.target is not None and len(self.target.atoms) > 0:
            self._target_pos_cached = np.array(
                [[a.x, a.y, a.z] for a in self.target.atoms],
                dtype=float,
            )
        else:
            self._target_pos_cached = None
        self._geometry_key = None
        # Set the effective hydrodynamic radius for the force-change backstep
        # criterion in motion/do_bd_step.py:backstep_due_to_force. The criterion
        # is dt > α |6 π μ dx² / (dF · dx / a)|, in which a appears as the inverse
        # hydrodynamic radius. There are two regimes. In auto_diffusion mode, a is
        # derived from the trace of D_trans through the Stokes-Einstein relation
        # D = kBT / (6 π μ a), which is self-consistent with the rigid-body
        # diffusion tensor we computed. In the scalar or explicit-tensor mode, we
        # use the largest bead radius. This is conservative, since a smaller
        # radius makes the criterion tighter and so backsteps slightly more often
        # than strictly necessary.
        if self.auto_diffusion:
            D_iso = float(np.trace(np.asarray(self.D_trans)) / 3.0)
            self._effective_hydro_radius = 1.0 / (
                6.0 * math.pi * WATER_VISCOSITY * D_iso
            )
        else:
            self._effective_hydro_radius = max(a.radius for a in chain_template.atoms)
        self._chain_radii = np.array(
            [a.radius for a in chain_template.atoms],
            dtype=float,
        )
        # Pre-build per-trajectory reusables.
        self._chain_scratch = make_chain_scratch_molecule(chain_template)
        self._chain_charges = np.array(
            [a.charge for a in chain_template.atoms], dtype=float
        )
        # The 1-2 and 1-3 pairs, which the bond and angle terms describe, stay
        # out of the vacuum Coulomb sum of the full GB force.
        self._gb_exclude_mask = bonded_exclusions(chain_template)
        # One force evaluation builds the index arrays and the stacked COFFDROP
        # tables of the chain here, so the workers of a parallel run share them.
        compute_chain_forces(
            ChainState.from_template(chain_template, self.body_positions.copy()),
            soft_repulsion=params.use_soft_repulsion,
        )
        # RNG and accumulators.
        self.rng = np.random.default_rng(params.seed)
        # Use an independent RNG stream for the Brownian bridge sampling. Keeping
        # it separate from self.rng means bridge sampling does not advance the
        # main trajectory RNG, so bridge_on and bridge_off comparisons at the same
        # seed produce identical trajectories and only the bridge samples
        # themselves differ. The offset 0xBB is arbitrary and just needs to be
        # non-zero so that the streams differ.
        _bb_seed = (params.seed if params.seed is not None else 0) + 0xBB
        self.rng_bb = np.random.default_rng(_bb_seed)
        self.results: List[TrajectoryResult] = []
        self.n_reacted = 0
        self.n_escaped = 0
        self.n_max_steps = 0
        # Build the LMZ outer propagator for the analytical b-to-q diffusion zone.
        # Without it, trajectories that drift past 1.1 b are simply terminated as
        # ESCAPED, which biases the reaction probability roughly ten times too low
        # for typical geometries, because about b/q of the "escaped" trajectories
        # should have analytically returned to the b-sphere. This matches the
        # NAMSimulator pattern in nam_simulator.py.
        self._outer_prop = None
        try:
            from pystarc.simulation.outer_propagator import (
                OuterPropagator,
                OPGroupInfo,
            )

            # The chain bounding radius is the maximum distance from the center of
            # mass (the body origin) to any atom surface. For a single-bead chain
            # at the origin this is just the atom radius.
            atom_radii = np.array(
                [a.radius for a in chain_template.atoms],
                dtype=float,
            )
            atom_dists = np.linalg.norm(
                chain_init_body_positions,
                axis=1,
            )
            chain_br = float(np.max(atom_dists + atom_radii))
            # Resolve an effective scalar D for the chain. If D_trans is a 3x3
            # tensor (auto_diffusion), use trace/3 as the effective scalar. In
            # auto_diffusion mode the constructor arguments D_trans and D_rot are
            # None, but the real anisotropic tensors are stored on self.D_trans
            # and self.D_rot at construction. Use those so the OuterPropagator
            # gets the true rigid-body D rather than the 1e-6 placeholder, which
            # would otherwise stall the outer propagation by a factor of roughly
            # 1e4 to 1e5 and prevent trajectories from ever reaching contact or
            # escape.
            if D_trans is None:
                D_trans_eff = float(np.trace(np.asarray(self.D_trans)) / 3.0)
            elif np.ndim(D_trans) == 0:
                D_trans_eff = float(D_trans)
            else:
                D_trans_eff = float(np.trace(np.asarray(D_trans)) / 3.0)
            if D_rot is None:
                D_rot_eff = float(np.trace(np.asarray(self.D_rot)) / 3.0)
            elif np.ndim(D_rot) == 0:
                D_rot_eff = float(D_rot)
            else:
                D_rot_eff = float(np.trace(np.asarray(D_rot)) / 3.0)
            # Total ligand charge, the sum of the chain bead charges.
            chain_q = float(sum(a.charge for a in chain_template.atoms))
            if self._a_target is not None:
                # The target and the chain are the two diffusing bodies of the
                # propagator, each with its own translational and rotational D.
                g0 = OPGroupInfo(
                    q=target.total_charge(),
                    Dtrans=self._D_t_target,
                    Drot=self._D_r_target,
                )
                g1 = OPGroupInfo(
                    q=chain_q,
                    Dtrans=D_trans_eff,
                    Drot=D_rot_eff,
                )
            else:
                # The target is fixed, so the chain carries all of the relative
                # diffusion. The propagator takes two bodies and derives their
                # Stokes radii from Dtrans, so the chain D is split evenly
                # between the two groups. D_g0 + D_g1 is then the chain D, but
                # with hydrodynamic interactions the two equal radii differ
                # from the real pair, which only target_hydro_radius describes.
                half_Dtrans = 0.5 * D_trans_eff if D_trans_eff > 0 else 1e-6
                half_Drot = 0.5 * D_rot_eff if D_rot_eff > 0 else 1e-6
                g0 = OPGroupInfo(
                    q=target.total_charge() if target is not None else 0.0,
                    Dtrans=half_Dtrans,
                    Drot=half_Drot,
                )
                g1 = OPGroupInfo(
                    q=chain_q,
                    Dtrans=half_Dtrans,
                    Drot=half_Drot,
                )
            # Physical constants, the same as in NAM and in PySTARC units of Å,
            # ps, and kBT.
            kT = float(params.temperature_kT)
            viscosity = VISCOSITY
            dielectric = float(params.dielectric)
            vacuum_perm = 1.0 / (4 * math.pi * COULOMB_KCAL)
            # Debye screening length. Use the user's input.xml value, threaded
            # through params.debye_length. This was previously hardcoded to 8.0,
            # which silently overrode the <debye_length> setting.
            debye_len = float(params.debye_length)
            target_br = target.bounding_radius() if target is not None else 0.0
            max_mol_r = max(target_br, chain_br)
            self._outer_prop = OuterPropagator(
                b_radius=params.r_start,
                max_radius=max_mol_r,
                has_hi=bool(params.hydrodynamic_interactions),
                kT=kT,
                viscosity=viscosity,
                dielectric=dielectric,
                vacuum_perm=vacuum_perm,
                debye_len=debye_len,
                g0=g0,
                g1=g1,
            )
        except Exception as _op_err:
            # OuterPropagator construction can fail for several reasons. A value
            # of D_g0 + D_g1 = 0, meaning both bodies are stationary, raises
            # ZeroDivisionError. Bad geometry or negative radii raise ValueError.
            # Numerical overflow in the Romberg integration raises OverflowError.
            # Falling back to simple-escape lets the simulation proceed but biases
            # the reaction probability roughly ten times low for typical
            # geometries, so we warn to make the failure visible. It was silent
            # before, which masked real bugs.
            warnings.warn(
                f"OuterPropagator construction failed "
                f"({type(_op_err).__name__}: {_op_err}); falling back to "
                f"simple-escape behavior. P_rxn will be biased low for "
                f"typical geometries.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._outer_prop = None

    def _compute_per_atom_external_forces(
        self, world_positions: np.ndarray
    ) -> np.ndarray:
        """Compute the per-atom external forces on the chain.

        This sums up to four contributions, each included when it is configured.
        The electrostatic contribution comes from the target's PB grid when it is
        loaded. The Born desolvation contribution comes from the target's Born
        grid when it is loaded, giving a per-bead force F_i = -α q_i² grad(g)(r_i)
        that is summed on top of the electrostatic contribution. The steric
        contribution is the WCA bead-target soft repulsion when
        params.use_soft_repulsion is set. The GB self-Born contribution is the
        chain-internal generalized Born force when params.use_self_born is set.
        The path B dispatch is keyed on params.coffdrop_active: when it is False
        the full GB force is added (the diagonal term, the off-diagonal term, and
        the vacuum Coulomb interaction), and when it is True only the diagonal term
        is added.

        If none of these contributions are active, the function returns zeros.
        """
        if self.target_grid is not None:
            forces, _energy = evaluate_target_grid_force_on_chain(
                world_positions,
                self._chain_charges,
                self.target_grid,
            )
        else:
            forces = np.zeros_like(world_positions)
        if self.born_grid is not None:
            f_born, _e_born = evaluate_born_force_on_chain(
                world_positions,
                self._chain_charges,
                self.born_grid,
                alpha=self.desolvation_alpha,
            )
            forces = forces + f_born
        self._refresh_geometry()
        spheres = None
        if self.params.use_soft_repulsion:
            spheres = self._nearby_spheres(world_positions)
        if spheres is not None:
            forces = forces + chain_target_steric_forces(
                world_positions,
                self._chain_radii,
                spheres,
                eps=self.params.soft_repulsion_eps,
            )
        if self.params.use_self_born:
            obc_kwargs = dict(
                obc_alpha=self.params.gb_obc_alpha,
                obc_beta=self.params.gb_obc_beta,
                obc_gamma=self.params.gb_obc_gamma,
            )
            f_gb, _e_gb = chain_full_gb_force(
                world_positions,
                self._chain_charges,
                self._chain_radii,
                eps_in=self.params.gb_eps_in,
                eps_out=self.params.gb_eps_out,
                coffdrop_active=self.params.coffdrop_active,
                exclude_pair_mask=self._gb_exclude_mask,
                obc_kwargs=obc_kwargs,
            )
            forces = forces + f_gb
        return forces

    def _refresh_geometry(self) -> None:
        """Cache the target spheres and the reaction pair arrays.

        The cache follows the target and the pathway set, which callers may
        replace after construction. It holds the target spheres for the steric
        force and the overlap test, the distance beyond the chain reach at
        which no bead and atom interact, and per reaction the target atom positions, the chain bead indices, the
        cutoffs and the squared distances of the target atoms from the origin,
        the lever arms of the target rotation.
        """
        key = (id(self.target), id(self.pathway_set))
        if key == self._geometry_key:
            return
        self._geometry_key = key
        self._target_spheres = target_spheres(self.target)
        t_rad = self._target_spheres[1]
        self._contact_reach = 0.0
        if t_rad.size:
            self._contact_reach = 2.0 ** (1.0 / 6.0) * (
                float(t_rad.max()) + float(self._chain_radii.max())
            )
        self._rxn_arrays = []
        if self.target is not None and self.pathway_set is not None:
            for rxn in self.pathway_set.reactions:
                pairs = rxn.criteria.pairs
                atoms = [self.target.atoms[p.mol1_atom_index] for p in pairs]
                t_pos = np.array([[a.x, a.y, a.z] for a in atoms]).reshape(-1, 3)
                self._rxn_arrays.append(
                    (
                        t_pos,
                        np.array([p.mol2_atom_index for p in pairs], dtype=np.int64),
                        np.array([p.distance_cutoff for p in pairs], dtype=float),
                        np.einsum("ij,ij->i", t_pos, t_pos),
                    )
                )

    def _outer_dt(self, pair_dists) -> float:
        """Return dt_rxn while a reaction pair is closer than 1.5 times its
        cutoff, the near zone, and dt otherwise."""
        for d, arr in zip(pair_dists or [], self._rxn_arrays):
            if (d < 1.5 * arr[2]).any():
                return self.params.dt_rxn
        return self.params.dt

    def _nearby_spheres(self, world_positions: np.ndarray):
        """Return the target spheres that a bead can touch, or None.

        An atom farther from the chain centroid than the largest bead distance
        plus the WCA cutoff of the largest bead and atom interacts with no
        bead, so it is left out. The order of the atoms is kept.
        """
        t_xyz, t_rad = self._target_spheres
        if t_rad.size == 0:
            return None
        c = world_positions.mean(axis=0)
        w = world_positions - c
        reach = math.sqrt(float(np.einsum("ij,ij->i", w, w).max()))
        d = t_xyz - c
        near = np.einsum("ij,ij->i", d, d) < (reach + self._contact_reach) ** 2
        if not near.any():
            return None
        return t_xyz[near], t_rad[near]

    def _rebuild_body_tensors(self, body_positions: np.ndarray):
        """Rebuild the body frame diffusion tensors from the current positions.

        The positions are the centered body frame positions, so the rotational
        resistance is taken about the origin, the point that place_chain
        rotates about. The function returns (D_trans, D_rot). A regularized
        factorization warns once per simulator.
        """
        info: Dict[str, Any] = {}
        D_t, D_r, _ = chain_diffusion_tensors(
            body_positions,
            self._chain_radii,
            center=np.zeros(3),
            warn_regularized=False,
            info=info,
        )
        if info.get("regularized") and not self._warned_regularized:
            self._warned_regularized = True
            warnings.warn(
                "the RPY mobility matrix of the chain needed regularization "
                "during a periodic rebuild of the body tensors; the usual cause "
                "is two beads at the same coordinate",
                RuntimeWarning,
                stacklevel=2,
            )
        return D_t, D_r

    def _relative_translation(
        self, pos: np.ndarray, ori: Quaternion, D_t_body, dt: float
    ) -> Tuple[np.ndarray, "Optional[np.ndarray]"]:
        """Return the relative diffusion tensor D_rel and the drift dD/dr r_hat.

        With p the separation of the chain from the target and r = |p|,

            D_rel = R D_t_body Rᵀ + (D_t,T - Δ(r)) I,
            Δ(r)  = D_0 - D_∥(r),

        where D_0 = D_t,T + tr(D_t_body)/3 at the start of the run and D_∥(r)
        is the radial two-sphere Rotne-Prager-Yamakawa diffusion of the pair
        mobility. Δ(r) is zero without hydrodynamic interactions, and the drift
        is then None. Otherwise the drift is the central difference of D_∥ with
        the bound of radial_D_gradient. A tensor with a smallest eigenvalue
        <= 0 has its eigenvalues clipped at RELATIVE_DIFFUSION_FLOOR, and the
        clipped step is counted in self._n_relative_clipped.
        """
        I3 = np.eye(3)
        D_t_arr = np.asarray(D_t_body, dtype=float)
        if D_t_arr.ndim == 0:
            D_t_arr = float(D_t_arr) * I3
        if self._pair_mob is not None:
            delta = self._D0_pair - self._pair_mob.relative_translational_diffusion(pos)
        else:
            delta = 0.0
        D_rel = rotate_tensor_to_lab(D_t_arr, ori) + (self._D_t_target - delta) * I3
        eig, vec = np.linalg.eigh(D_rel)
        if not eig[0] > 0.0:
            # Unphysical overlap separations can make the tensor non positive
            # definite. The eigenvalues are clipped, the step is counted, and
            # one warning per trajectory is logged.
            D_rel = (vec * np.maximum(eig, RELATIVE_DIFFUSION_FLOOR)) @ vec.T
            self._n_relative_clipped += 1
            if not self._warned_relative_clip:
                self._warned_relative_clip = True
                warnings.warn(
                    "the relative diffusion tensor was not positive definite at "
                    f"r={float(np.linalg.norm(pos)):.6g} A (a_T={self._a_target:.6g} A, "
                    f"a_chain={self._a_chain:.6g} A, eigenvalues {eig}); its "
                    f"eigenvalues are clipped at {RELATIVE_DIFFUSION_FLOOR:g} A^2/ps",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self._pair_mob is not None:
            grad_D = radial_D_gradient(
                self._pair_mob, pos, float(np.trace(D_rel) / 3.0), dt
            )
        else:
            grad_D = None
        return D_rel, grad_D

    def _rigid_step(
        self,
        pos: np.ndarray,
        ori: Quaternion,
        world_pos: np.ndarray,
        per_atom_forces: np.ndarray,
        D_t_body,
        D_r_body,
        dt: float,
        dW_t: np.ndarray,
        dW_r: np.ndarray,
        dW_r1: "Optional[np.ndarray]" = None,
    ) -> Tuple[np.ndarray, Quaternion]:
        """Advance the rigid-body coordinates (pos, ori) by one (sub)step dt.

        D_t_body and D_r_body are the body frame tensors, or scalars. Without
        target_hydro_radius and without rotate_body_tensors the call is
        chain_outer_bd_step_wiener on the tensors as given. With
        rotate_body_tensors alone the same call receives the tensors rotated
        into the lab frame by the orientation at the start of the step.

        With target_hydro_radius the step is the two-body step. The force and
        torque are aggregated at the start configuration. The chain translates
        relative to the target with
        D_rel = R D_t_body Rᵀ + (D_t,T - Δ(r)) I and the drift dD/dr r_hat when
        Δ(r) is on, and it rotates with R D_r_body Rᵀ. The target rotation
        ω_T, drawn from dW_r1, turns the position and the orientation about the
        origin. A step that displaces the chain by more than 5 Å is rejected.
        """
        if self._a_target is None:
            if not self.params.rotate_body_tensors:
                return chain_outer_bd_step_wiener(
                    pos,
                    ori,
                    world_pos,
                    per_atom_forces,
                    D_t_body,
                    D_r_body,
                    dt,
                    dW_t,
                    dW_r,
                )
            return chain_outer_bd_step_wiener(
                pos,
                ori,
                world_pos,
                per_atom_forces,
                rotate_tensor_to_lab(D_t_body, ori),
                rotate_tensor_to_lab(D_r_body, ori),
                dt,
                dW_t,
                dW_r,
            )
        if dW_r1 is None:
            raise ValueError("the two-body step needs the target increment dW_r1")
        force_net, torque_net = aggregate_chain_external_force_and_torque(
            world_pos,
            per_atom_forces,
            pos,
        )
        D_r_arr = np.asarray(D_r_body, dtype=float)
        if D_r_arr.ndim == 0:
            D_r_arr = float(D_r_arr) * np.eye(3)
        D_rel, grad_D = self._relative_translation(pos, ori, D_t_body, dt)
        new_pos = ermak_mccammon_translation_tensor(
            pos, force_net, D_rel, dt, dW_t, grad_D=grad_D
        )
        new_ori = ermak_mccammon_rotation_tensor(
            ori,
            torque_net,
            rotate_tensor_to_lab(D_r_arr, ori),
            dt,
            dW_r,
        )
        omega_T = receptor_rotation_vector(
            pos, force_net, torque_net, self._D_r_target, dt, dW_r1
        )
        new_pos, new_ori = apply_frame_rotation(new_pos, new_ori, omega_T)
        # The same displacement guard as chain_outer_bd_step_wiener.
        MAX_OUTER_DISP_A = 5.0
        if np.linalg.norm(new_pos - pos) > MAX_OUTER_DISP_A:
            return pos, ori
        return new_pos, new_ori

    def _pair_distances(self, world_pos: np.ndarray) -> "list[np.ndarray]":
        """Distances of the reaction pairs, one array per reaction."""
        out = []
        for t_pos, c_idx, _cut, _l1 in self._rxn_arrays:
            d = t_pos - world_pos[c_idx]
            out.append(np.sqrt(np.einsum("ij,ij->i", d, d)))
        return out

    def _bridge_diffusion(self, body_positions, D_t_body, D_r_body) -> list:
        """Per-pair D_eff of the Brownian bridge, one array per reaction.

        D_eff = D_t + D_r L_2^2 for the chain, with L_2 the distance of the
        pair bead from the chain centroid, and with target_hydro_radius
        + D_t,T + D_r,T L_1^2 for the target, with L_1 the distance of the pair
        atom from the origin, as in SI Step 11.
        """
        D_t = _isotropic_diffusion(D_t_body)
        D_r = _isotropic_diffusion(D_r_body)
        out = []
        for _t_pos, c_idx, _cut, l1_sq in self._rxn_arrays:
            b = body_positions[c_idx]
            D = D_t + D_r * np.einsum("ij,ij->i", b, b)
            if self._a_target is not None:
                D = D + self._D_t_target + self._D_r_target * l1_sq
            out.append(D)
        return out

    def run_one(
        self,
        rng: "Optional[np.random.Generator]" = None,
        rng_bb: "Optional[np.random.Generator]" = None,
    ) -> TrajectoryResult:
        """Execute one trajectory and return its outcome.

        The function returns a TrajectoryResult with Fate.REACTED, Fate.ESCAPED
        or Fate.MAX_STEPS. The pose after the last allowed step still gets the
        reaction and escape tests, and a trajectory undecided by then counts as
        MAX_STEPS, neither reacted nor escaped. If an rng is provided, that
        generator is used for this trajectory only, and otherwise self.rng is
        used. The optional argument exists so that parallel workers can supply
        their own seeded generators.

        An outer step evaluates the target forces at its start configuration.
        The inner steps then advance the internal coordinates over the same
        time, feeling those forces in the body frame when params.inner_field is
        set, and the rigid-body coordinates move with the same forces.
        """
        params = self.params
        if rng is None:
            rng = self.rng
        if rng_bb is None:
            rng_bb = self.rng_bb
        self._refresh_geometry()

        # Per-trajectory diagnostic recording state. It is updated at the top of
        # each outer BD iteration and finalized at the return sites. The output
        # writers consume the populated fields on TrajectoryResult.
        save_interval = self.outputs.save_interval if self.outputs is not None else 0
        min_sep = float("inf")
        pos_at_min = None
        path_steps_buf: List[int] = []
        path_com_buf: List[np.ndarray] = []
        path_q_buf: List[List[float]] = []
        radial_buf: List[float] = []
        # Returns to the b-sphere between each saved separation and the one
        # before, so the milestone flux can leave the jump out.
        radial_returns_buf: List[int] = []
        n_returns = 0
        # Energy traces and contact-pair counts recorded at the recording cadence.
        energy_buf: List[List[float]] = []
        contact_count_buf: Dict[Tuple[int, int], int] = {}

        # Build a fresh ChainState for this trajectory. Every trajectory starts
        # from the same body-frame configuration but evolves independently from
        # there.
        state = ChainState.from_template(
            self.chain_template,
            self.body_positions.copy(),
        )
        # Pre-equilibrate the chain by thermalizing its internal coordinates in
        # bulk, with no external forces, before placing it on the b-sphere. With
        # params.n_equilibration_steps == 0 (the default) the loop body runs zero
        # times and no RNG state is consumed.
        for _ in range(params.n_equilibration_steps):
            chain_internal_bd_step(
                state,
                dt=params.dt_chain,
                rng=rng,
                soft_repulsion=params.use_soft_repulsion,
                soft_repulsion_eps=params.soft_repulsion_eps,
                constraint_tol=params.constraint_tol,
                constraint_max_iter=params.constraint_max_iter,
                remove_rigid_rotation=params.remove_internal_rotation,
            )
        # The body frame diffusion tensors of this trajectory. They start at the
        # tensors of the simulator and change only through the periodic rebuild,
        # so a rebuild never writes back to self (parallel workers share the
        # simulator by copy). With the interval at 0 these are self.D_trans and
        # self.D_rot for the whole trajectory.
        self._n_relative_clipped = 0
        self._warned_relative_clip = False
        D_t_body, D_r_body = self.D_trans, self.D_rot
        diff_interval = params.diffusion_update_interval
        n_rebuilds = 0
        n_rebuild_failures = 0
        # Time weighted sums of tr(D)/3 for the diagnostics.
        tr_t_cur = tr_r_cur = 0.0
        Dt_weighted = Dr_weighted = 0.0

        def _rebuild_tensors():
            nonlocal D_t_body, D_r_body, n_rebuilds, n_rebuild_failures
            nonlocal tr_t_cur, tr_r_cur
            try:
                D_t_body, D_r_body = self._rebuild_body_tensors(state.positions)
                n_rebuilds += 1
            except np.linalg.LinAlgError:
                # A collinear chain has no rotational resistance about its own
                # axis. Keep the previous tensors and count the event.
                n_rebuild_failures += 1
            tr_t_cur = _isotropic_diffusion(D_t_body)
            tr_r_cur = _isotropic_diffusion(D_r_body)

        if diff_interval > 0:
            tr_t_cur = _isotropic_diffusion(D_t_body)
            tr_r_cur = _isotropic_diffusion(D_r_body)
            if params.n_equilibration_steps > 0:
                _rebuild_tensors()

        n_hs_redraws = 0
        n_hs_rejections = 0

        def _result(fate, n_steps, name=None, **extra) -> TrajectoryResult:
            """Collect the diagnostics of the trajectory into its result."""
            d = _build_chain_diagnostics(
                path_steps_buf,
                path_com_buf,
                path_q_buf,
                radial_buf,
                energy_buf,
                contact_count_buf,
            )
            if radial_buf:
                d["radial_returns"] = np.asarray(radial_returns_buf, dtype=int)
            d["n_relative_diffusion_clipped"] = self._n_relative_clipped
            d["n_hard_sphere_redraws"] = n_hs_redraws
            d["n_hard_sphere_rejections"] = n_hs_rejections
            if diff_interval > 0:
                if t_elapsed > 0.0:
                    d["mean_D_trans"] = Dt_weighted / t_elapsed
                    d["mean_D_rot"] = Dr_weighted / t_elapsed
                else:
                    d["mean_D_trans"] = tr_t_cur
                    d["mean_D_rot"] = tr_r_cur
                d["n_diffusion_rebuilds"] = n_rebuilds
                d["n_diffusion_rebuild_failures"] = n_rebuild_failures
            if fate != Fate.REACTED and pos_at_min is not None:
                d["near_miss_pos"] = pos_at_min
                d["near_miss_dist"] = min_sep
            d.update(extra)
            return TrajectoryResult(
                fate, n_steps, t_elapsed, float(np.linalg.norm(pos)), name, **d
            )

        # Random starting center of mass and orientation on the b-sphere.
        pos, ori = initialize_bsphere(rng, params.r_start)
        # Cumulative simulated time in ps.
        t_elapsed = 0.0
        # Brownian bridge state: the pair distances, the per-pair D_eff and the
        # outer step at the start of the previous step. None skips the bridge,
        # on the first step and after a step that did not move.
        has_rxn = bool(self._rxn_arrays)
        prev_pair_dists = None
        prev_D_eff = None
        prev_dt_outer = 0.0
        for step in range(params.max_steps + 1):
            # Record the per-iteration diagnostics.
            r_curr = float(np.linalg.norm(pos))
            if r_curr < min_sep:
                min_sep = r_curr
                pos_at_min = pos.copy()
            if save_interval > 0 and step % save_interval == 0:
                path_steps_buf.append(step)
                path_com_buf.append(pos.copy())
                path_q_buf.append([ori.w, ori.x, ori.y, ori.z])
                radial_buf.append(r_curr)
                radial_returns_buf.append(n_returns)
                n_returns = 0
                # Sample the energy and the contacts.
                _world_pos = place_chain(state.positions, pos, ori)
                _e_elec = 0.0
                _e_born = 0.0
                if self.target_grid is not None:
                    _f, _e_elec = evaluate_target_grid_force_on_chain(
                        _world_pos, self._chain_charges, self.target_grid
                    )
                if self.born_grid is not None:
                    _f, _e_born = evaluate_born_force_on_chain(
                        _world_pos,
                        self._chain_charges,
                        self.born_grid,
                        alpha=self.desolvation_alpha,
                    )
                # The steric energy is left at 0 here as a documented omission,
                # because chain_target_steric_forces returns only the force.
                energy_buf.append([_e_elec + _e_born, _e_elec, _e_born, 0.0])
                if self._target_pos_cached is not None:
                    _d2 = np.sum(
                        (_world_pos[:, None, :] - self._target_pos_cached[None, :, :])
                        ** 2,
                        axis=2,
                    )
                    _close = np.where(_d2 < 36.0)
                    for _ci, _tj in zip(_close[0], _close[1]):
                        _key = (int(_tj), int(_ci))
                        contact_count_buf[_key] = contact_count_buf.get(_key, 0) + 1
            # Place the chain in the world frame and test for a reaction, with
            # the Brownian bridge over the previous step when it applies.
            world_pos = place_chain(state.positions, pos, ori)
            update_chain_scratch_positions(self._chain_scratch, world_pos)
            pair_dists = self._pair_distances(world_pos) if has_rxn else None
            if (
                params.use_brownian_bridge
                and prev_pair_dists is not None
                and pair_dists is not None
                and prev_dt_outer > 0.0
            ):
                rxn = check_reaction_with_bridge(
                    self.target,
                    self._chain_scratch,
                    self.pathway_set,
                    prev_pair_dists,
                    pair_dists,
                    prev_D_eff,
                    prev_dt_outer,
                    rng,
                    rng_bb=rng_bb,
                )
            else:
                rxn = check_chain_reaction(
                    self.target,
                    self._chain_scratch,
                    self.pathway_set,
                    rng,
                )
            if rxn is not None:
                # Interaction energy at the encounter, sampled here because
                # energy_buf is filled only every save_interval steps.
                _e_rxn = 0.0
                if self.target_grid is not None:
                    _e_rxn += evaluate_target_grid_force_on_chain(
                        world_pos, self._chain_charges, self.target_grid
                    )[1]
                if self.born_grid is not None:
                    _e_rxn += evaluate_born_force_on_chain(
                        world_pos,
                        self._chain_charges,
                        self.born_grid,
                        alpha=self.desolvation_alpha,
                    )[1]
                return _result(
                    Fate.REACTED,
                    step,
                    rxn,
                    encounter_pos=pos.copy(),
                    encounter_q=np.array([ori.w, ori.x, ori.y, ori.z]),
                    energy_at_reaction=_e_rxn,
                )
            # LMZ outer propagator. Beyond 1.1 b the analytic propagator either
            # returns the chain to the b-sphere, with probability about b/q,
            # and the trajectory continues from there, or lets it escape.
            QB_FACTOR = 1.1
            if (
                self._outer_prop is not None
                and params.use_lmz
                and float(np.linalg.norm(pos)) >= QB_FACTOR * params.r_start
            ):
                reached_b, new_pos, new_ori_arr = self._outer_prop.new_state(
                    pos,
                    ori.to_array(),
                    rng,
                )
                pos = new_pos
                ori = Quaternion(
                    w=float(new_ori_arr[0]),
                    x=float(new_ori_arr[1]),
                    y=float(new_ori_arr[2]),
                    z=float(new_ori_arr[3]),
                )
                if not reached_b:
                    # The chain has truly escaped through the q-sphere.
                    return _result(Fate.ESCAPED, step)
                # The chain jumped to the b-sphere, so the step starts there
                # and no bridge spans the jump.
                n_returns += 1
                world_pos = place_chain(state.positions, pos, ori)
                pair_dists = self._pair_distances(world_pos) if has_rxn else None
            # Fallback escape check, used when _outer_prop is None or when
            # use_lmz=False. Without LMZ it biases the reaction probability low.
            _lmz_active = self._outer_prop is not None and params.use_lmz
            if not _lmz_active and check_escape(pos, params.r_escape):
                return _result(Fate.ESCAPED, step)
            if step == params.max_steps:
                break
            # Periodic rebuild of the body frame diffusion tensors from the
            # current internal configuration. The state positions are centered
            # at the origin, the point the chain rotates about.
            if diff_interval > 0 and step > 0 and step % diff_interval == 0:
                _rebuild_tensors()
            # The inner steps cover the outer step.
            dt_outer = self._outer_dt(pair_dists)
            n_inner = max(1, round(params.chain_steps_per_outer * dt_outer / params.dt))
            # Bridge state of this step, taken at its start.
            if params.use_brownian_bridge and has_rxn:
                D_eff = self._bridge_diffusion(state.positions, D_t_body, D_r_body)
            # The target forces at the start configuration.
            per_atom_forces = self._compute_per_atom_external_forces(world_pos)
            F_net_old = per_atom_forces.sum(axis=0)
            body_old = state.positions.copy() if params.use_hard_sphere else None
            # The inner steps, internal-coordinate BD on the body frame
            # positions. With inner_field the forces turn into the body frame,
            # F_body = R^T F.
            f_body = None
            if params.inner_field:
                f_body = per_atom_forces @ ori.to_rotation_matrix()
            for _ in range(n_inner):
                chain_internal_bd_step(
                    state,
                    dt=params.dt_chain,
                    rng=rng,
                    constraint_tol=params.constraint_tol,
                    constraint_max_iter=params.constraint_max_iter,
                    soft_repulsion=params.use_soft_repulsion,
                    soft_repulsion_eps=params.soft_repulsion_eps,
                    remove_rigid_rotation=params.remove_internal_rotation,
                    external_forces=f_body,
                )
            pos_old, ori_old = pos, ori

            def _move(dW_t, dW_r, dW_r1):
                """One outer move from (pos_old, ori_old). When the force changes
                too much across it, the move is split at the Brownian bridge
                midpoint of the same increments, which keeps their sum."""
                new_pos, new_ori = self._rigid_step(
                    pos_old,
                    ori_old,
                    world_pos,
                    per_atom_forces,
                    D_t_body,
                    D_r_body,
                    dt_outer,
                    dW_t,
                    dW_r,
                    dW_r1,
                )
                if not (params.force_change_backstep and dt_outer > params.dt_rxn):
                    return new_pos, new_ori
                F_net_new = self._compute_per_atom_external_forces(
                    place_chain(state.positions, new_pos, new_ori)
                ).sum(axis=0)
                if not backstep_due_to_force(
                    F_net_new,
                    F_net_old,
                    new_pos,
                    pos_old,
                    dt_outer,
                    params.dt_rxn,
                    radius=self._effective_hydro_radius,
                ):
                    return new_pos, new_ori
                s = math.sqrt(dt_outer / 4.0)
                dW_mid_t = 0.5 * dW_t + s * rng.standard_normal(3)
                dW_mid_r = 0.5 * dW_r + s * rng.standard_normal(3)
                dW_mid_r1 = None
                if dW_r1 is not None:
                    dW_mid_r1 = 0.5 * dW_r1 + s * rng.standard_normal(3)
                hdt = dt_outer / 2.0
                mid_pos, mid_ori = self._rigid_step(
                    pos_old,
                    ori_old,
                    world_pos,
                    per_atom_forces,
                    D_t_body,
                    D_r_body,
                    hdt,
                    dW_mid_t,
                    dW_mid_r,
                    dW_mid_r1,
                )
                world_mid = place_chain(state.positions, mid_pos, mid_ori)
                return self._rigid_step(
                    mid_pos,
                    mid_ori,
                    world_mid,
                    self._compute_per_atom_external_forces(world_mid),
                    D_t_body,
                    D_r_body,
                    hdt,
                    dW_t - dW_mid_t,
                    dW_r - dW_mid_r,
                    None if dW_r1 is None else dW_r1 - dW_mid_r1,
                )

            def _draw():
                """The Wiener increments of one outer move. The target increment
                is drawn only for the two-body step."""
                dW_t = math.sqrt(dt_outer) * rng.standard_normal(3)
                dW_r = math.sqrt(dt_outer) * rng.standard_normal(3)
                dW_r1 = None
                if self._a_target is not None:
                    dW_r1 = math.sqrt(dt_outer) * rng.standard_normal(3)
                return dW_t, dW_r, dW_r1

            pos, ori = _move(*_draw())
            moved = True
            # Hard-sphere rejection on the bead and target atom pairs. An
            # overlapping move is redrawn up to MAX_HS_ATTEMPTS times. When every
            # draw overlaps, the whole step is rejected: the inner steps are
            # undone and the chain stays where it was.
            if params.use_hard_sphere:
                MAX_HS_ATTEMPTS = 3
                for attempt in range(MAX_HS_ATTEMPTS + 1):
                    world_new = place_chain(state.positions, pos, ori)
                    spheres = self._nearby_spheres(world_new)
                    if spheres is None or not _check_chain_overlap(
                        spheres, world_new, self._chain_radii
                    ):
                        break
                    if attempt == MAX_HS_ATTEMPTS:
                        state.positions[:] = body_old
                        pos, ori = pos_old, ori_old
                        n_hs_rejections += 1
                        moved = False
                        break
                    n_hs_redraws += 1
                    pos, ori = _move(*_draw())
            t_elapsed += dt_outer
            if diff_interval > 0:
                Dt_weighted += tr_t_cur * dt_outer
                Dr_weighted += tr_r_cur * dt_outer
            if params.use_brownian_bridge and has_rxn and moved:
                prev_pair_dists = pair_dists
                prev_D_eff = D_eff
                prev_dt_outer = dt_outer
            else:
                prev_pair_dists = None
        return _result(Fate.MAX_STEPS, params.max_steps)

    def run(self) -> List[TrajectoryResult]:
        """Execute n_trajectories trajectories, either serially or in parallel.

        This dispatches to a parallel multiprocessing pool when both
        params.n_threads > 1 and params.n_trajectories > 1, and otherwise it runs
        serially. Trajectory i draws from the streams of trajectory_rngs(seed, i)
        in both modes, so the results do not depend on the number of workers.
        It returns the accumulated self.results list.
        """
        n = self.params.n_trajectories
        if self.params.n_threads > 1 and n > 1:
            self._run_parallel(n)
        else:
            self._run_serial(n)
        return self.results

    def _count_fate(self, result: TrajectoryResult) -> None:
        if result.fate == Fate.REACTED:
            self.n_reacted += 1
        elif result.fate == Fate.ESCAPED:
            self.n_escaped += 1
        elif result.fate == Fate.MAX_STEPS:
            self.n_max_steps += 1

    def _record(self, result: TrajectoryResult) -> None:
        """Append one trajectory outcome and update fate counters."""
        self.results.append(result)
        self._count_fate(result)

    def _log_progress(self, done: int, total: int, t0: float) -> None:
        elapsed = time.monotonic() - t0
        pct = 100.0 * done / total if total else 0.0
        kb = float(getattr(self._outer_prop, "k_b", 0.0) or 0.0)
        p_live = self.n_reacted / done if done else 0.0
        k_live = MOLAR_RATE_CONV * kb * p_live if kb > 0.0 else 0.0
        if k_live > 0.0:
            exp = int(math.floor(math.log10(k_live)))
            man = k_live / (10.0**exp)
            print(
                f"  [T] {elapsed:6.0f}s  "
                f"done={done:,}/{total:,} ({pct:.1f}%)  "
                f"reacted={self.n_reacted:,}  escaped={self.n_escaped:,}  "
                f"P_rxn={p_live:.5f}  k_on={man:.2f}e{exp}  "
                f"max_steps={self.n_max_steps:,}",
                flush=True,
            )
        else:
            print(
                f"  [T] {elapsed:6.0f}s  "
                f"done={done:,}/{total:,} ({pct:.1f}%)  "
                f"reacted={self.n_reacted:,}  escaped={self.n_escaped:,}  "
                f"max_steps={self.n_max_steps:,}",
                flush=True,
            )

    def _run_serial(self, n: int) -> None:
        """Run n trajectories sequentially."""
        t0 = time.monotonic()
        report_every = max(1, n // 100)
        for i in range(n):
            rng, rng_bb = trajectory_rngs(self.params.seed, i)
            self._record(self.run_one(rng=rng, rng_bb=rng_bb))
            done = i + 1
            if self.params.verbose and (done % report_every == 0 or done == n):
                self._log_progress(done, n, t0)

    def _run_parallel(self, n: int) -> None:
        """Run n trajectories across a multiprocessing pool.

        Workers stream completed outcomes via imap_unordered for live counts,
        while self.results preserves deterministic trajectory index order.
        """
        n_workers = min(self.params.n_threads, n, mp.cpu_count())
        if self.params.verbose:
            print(f"  Parallel: {n_workers} workers, {n} trajectories", flush=True)
        t0 = time.monotonic()
        report_every = max(1, n // 100)
        ordered: List[Optional[TrajectoryResult]] = [None] * n
        with mp.Pool(
            n_workers, initializer=_chain_worker_init, initargs=(self,)
        ) as pool:
            for done, (idx, result) in enumerate(
                pool.imap_unordered(_run_chain_trajectory_indexed, range(n)), 1
            ):
                ordered[idx] = result
                self._count_fate(result)
                if self.params.verbose and (done % report_every == 0 or done == n):
                    self._log_progress(done, n, t0)
        self.results.extend(ordered)
