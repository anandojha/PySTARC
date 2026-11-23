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
    _coffdrop_exclusion_mask,
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
    rpy_pair_blocks,
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
    SOLUTE_DIELECTRIC,
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
    wca_core_cap_fraction: float = 0.0,
) -> np.ndarray:
    """Compute the WCA repulsion between every chain bead and every target atom.

    Returns an (n_chain, 3) array of per-bead forces in kBT/A. For each
    bead-atom pair sigma is the sum of radii and eps is uniform. When
    wca_core_cap_fraction > 0, pairs closer than r_core = f * sigma hold
    the force magnitude constant at F_WCA(r_core) to match the GPU kernel.
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
    r_act = r[ia, ka]
    sig_act = sig[ia, ka]
    core_frac = float(wca_core_cap_fraction)
    if core_frac > 0.0:
        r_core = core_frac * sig_act
        in_core = r_act < r_core
        fmag_over_r = np.empty_like(r_act)
        if np.any(in_core):
            inv_f2 = 1.0 / (core_frac * core_frac)
            inv_f6 = inv_f2 * inv_f2 * inv_f2
            f_core = (4.0 * eps / r_core[in_core]) * (
                12.0 * inv_f6 * inv_f6 - 6.0 * inv_f6
            )
            fmag_over_r[in_core] = f_core / r_act[in_core]
        out_core = ~in_core
        if np.any(out_core):
            sr6 = (sig_act[out_core] / r_act[out_core]) ** 6
            fmag_over_r[out_core] = (
                4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / (r_act[out_core] ** 2)
            )
    else:
        sr6 = (sig_act / r_act) ** 6
        fmag_over_r = 4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / (r_act**2)
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
    overlap_scale: float = 1.0,
) -> bool:
    """Return True if any chain coordinate is non-finite or overlaps a target atom.

    A bead and an atom overlap when their center distance is below
    overlap_scale * (r_bead + r_target). Ghost atoms (radius < 1e-10) are
    skipped.
    """
    if not np.all(np.isfinite(chain_world_positions)):
        return True
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
    sig = float(overlap_scale) * (chain_radii[live][:, None] + target_rad[None, :])
    return bool((r2 < sig * sig).any())


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
    # by dt_rxn / dt, so the internal and the rigid-body clocks match when
    # chain_steps_per_outer × dt_chain = dt.
    dt_chain: float = 0.01
    chain_steps_per_outer: int = 20
    # Number of internal-coordinate-only BD steps to run in bulk (with no
    # external forces) before each trajectory's main BD loop. The default of 0
    # means no equilibration and consumes no RNG state.
    n_equilibration_steps: int = 0
    constraint_tol: float = 1e-6
    constraint_max_iter: int = 200
    # Whether to subdivide an outer BD step when the external force changes too
    # much across it.
    force_change_backstep: bool = True
    # Whether to reject steps that overlap a chain bead with a target atom
    # (single-attempt Metropolis rejection preserving detailed balance).
    use_hard_sphere: bool = True
    overlap_scale: float = 0.5
    # Whether to add WCA forces with uniform ε between chain beads and target
    # atoms and between the chain bead pairs that have no bonded term and no
    # COFFDROP table. σ is the radius sum and ε is in kBT.
    use_soft_repulsion: bool = False
    soft_repulsion_eps: float = 1.0
    wca_core_cap_fraction: float = 0.7
    # Whether to apply the Brownian bridge for reaction detection.
    use_brownian_bridge: bool = True
    # Whether to add generalized Born (self-Born) electrostatics on the chain
    # inside the inner dt_chain loop.
    use_self_born: bool = False
    gb_eps_in: float = 1.0
    gb_eps_out: float = SOLVENT_DIELECTRIC
    gb_obc_alpha: float = 1.0
    gb_obc_beta: float = 0.8
    gb_obc_gamma: float = 4.85
    coffdrop_active: bool = False
    intra_debye_huckel: bool = True
    enable_born2: bool = False
    grid_float32: bool = False
    # Whether to rotate the body frame diffusion tensors into the lab frame at
    # the start of each rigid-body step, D_lab = R D_body Rᵀ, with R from
    # the orientation at the start of the (sub)step.
    rotate_body_tensors: bool = True
    # Number of outer steps between rebuilds of the body frame diffusion
    # tensors from the current internal configuration. A value of 0 never
    # rebuilds. A positive value requires auto_diffusion.
    diffusion_update_interval: int = 0
    # Whether the internal steps remove the rigid-body rotation of the chain.
    remove_internal_rotation: bool = True
    # Whether the inner steps feel the internal deformation component of target
    # forces (rigid force and torque projected out).
    inner_field: bool = False

    def __post_init__(self):
        if self.r_escape == 0.0:
            self.r_escape = self.r_start * 1.1
        if self.dt_rxn <= 0.0:
            raise ValueError(f"dt_rxn must be > 0, got {self.dt_rxn}")
        if self.dt_rxn > self.dt:
            self.dt_rxn = self.dt
        if not (0.0 < self.overlap_scale <= 2.0):
            raise ValueError(
                f"overlap_scale must be in (0, 2], got {self.overlap_scale}"
            )
        if not (0.0 <= self.wca_core_cap_fraction < 1.0):
            raise ValueError(
                f"wca_core_cap_fraction must be in [0, 1), got "
                f"{self.wca_core_cap_fraction}"
            )


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
    target: Optional[Molecule],
    chain_scratch: Molecule,
    pathway_set: Optional[PathwaySet],
    rng: Optional[np.random.Generator] = None,
) -> Optional[str]:
    """Check whether any reaction in pathway_set fires for the current chain pose.

    The chain's world-frame positions must already have been written into
    chain_scratch (via update_chain_scratch_positions) before this call.
    Returns the name of the first reaction that fires, or None.
    """
    if pathway_set is None or target is None:
        return None
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
    if pathway_set is None or target is None:
        return None
    # Fall back to the endpoint-only check when we lack matching distance data.
    if (
        not old_pair_dists_per_rxn
        or not new_pair_dists_per_rxn
        or len(old_pair_dists_per_rxn) != len(new_pair_dists_per_rxn)
    ):
        return check_chain_reaction(target, chain_scratch, pathway_set, rng)

    for rxn_idx, rxn in enumerate(pathway_set.reactions):
        n_needed = rxn.criteria.n_needed
        n_pairs = len(rxn.criteria.pairs)
        threshold = n_pairs if n_needed < 0 else n_needed
        if threshold <= 0 or threshold > n_pairs:
            continue
        D = D_eff[rxn_idx] if isinstance(D_eff, (list, tuple)) else D_eff
        old_d = np.asarray(old_pair_dists_per_rxn[rxn_idx], dtype=float)
        new_d = np.asarray(new_pair_dists_per_rxn[rxn_idx], dtype=float)
        cutoffs = np.array(
            [pair.distance_cutoff for pair in rxn.criteria.pairs],
            dtype=float,
        )
        x0 = old_d - cutoffs
        x1 = new_d - cutoffs
        D_arr = np.broadcast_to(np.asarray(D, dtype=float), x0.shape)
        Ddt = np.maximum(D_arr * dt, 1e-30)
        bb_rng = rng_bb if rng_bb is not None else rng

        if threshold == 1:
            endpoint_fired = x1 < 0
            both_above = (x0 > 0) & (x1 > 0)
            p_cross = np.zeros_like(x0)
            np.subtract(0.0, x0 * x1 / Ddt, out=p_cross, where=both_above)
            np.exp(p_cross, out=p_cross, where=both_above)
            p_cross = np.where(both_above, p_cross, 0.0)
            u = bb_rng.random(size=p_cross.shape)
            fired = bool((endpoint_fired | (u < p_cross)).any())
        else:
            if int((x1 < 0).sum()) >= threshold:
                fired = True
            else:
                k = threshold - 1
                idx0 = int(np.argpartition(x0, k)[k])
                idx1 = int(np.argpartition(x1, k)[k])
                q0 = float(x0[idx0])
                q1 = float(x1[idx1])
                if q0 > 0.0 and q1 > 0.0:
                    Ddt_q = 0.5 * (float(Ddt[idx0]) + float(Ddt[idx1]))
                    p_cross_q = math.exp(-q0 * q1 / Ddt_q)
                    fired = float(bb_rng.random()) < p_cross_q
                else:
                    fired = False

        if fired:
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


def _project_ext_rigid(pos: np.ndarray, ext_f: np.ndarray) -> np.ndarray:
    """Remove net rigid force and net rigid torque from ext_f at body coordinates pos."""
    n = pos.shape[0]
    if n == 0:
        return ext_f.copy()
    r = pos - pos.mean(axis=0)
    a = ext_f.mean(axis=0)
    tau = np.sum(np.cross(r, ext_f), axis=0)
    r2 = np.einsum("ij,ij->i", r, r)
    J = np.eye(3) * float(np.sum(r2)) - (r.T @ r)
    evals, evecs = np.linalg.eigh(J)
    emax = float(np.max(np.abs(evals)))
    if emax > 1.0e-20:
        inv_e = np.where(
            evals > 1.0e-12 * emax, 1.0 / np.where(evals > 0.0, evals, 1.0), 0.0
        )
        b_rot = evecs @ (inv_e * (evecs.T @ tau))
    else:
        b_rot = np.zeros(3, dtype=np.float64)
    return ext_f - a[None, :] - np.cross(b_rot[None, :], r)


def _intra_wca_capped_forces(
    state: ChainState,
    common: ChainCommon,
    eps: float,
    core_frac: float,
) -> np.ndarray:
    """Intra-chain WCA forces with linear core cap below core_frac * sigma."""
    n = common.n_atoms
    F = np.zeros((n, 3), dtype=float)
    if n < 2:
        return F
    has_cd = common.coffdrop_params is not None
    tables = has_cd and bool(common.pair_lookups)
    key = (
        id(common.bonds),
        len(common.bonds),
        id(common.length_constraints),
        len(common.length_constraints),
        id(common.angles),
        len(common.angles),
        id(common.torsions),
        len(common.torsions),
        has_cd,
        tables,
        len(common.pair_lookups),
        tuple(int(a.resid) for a in common.atoms) if has_cd else (),
    )
    cache = common.__dict__.get("_wca_capped_cache")
    if cache is None or cache[0] != key:
        radii = np.array([a.radius for a in common.atoms], dtype=float)
        skip = _coffdrop_exclusion_mask(common) if has_cd else bonded_exclusions(common)
        if tables:
            ij = np.array(list(common.pair_lookups), dtype=np.int64).reshape(-1, 2)
            skip[ij[:, 0], ij[:, 1]] = True
            skip[ij[:, 1], ij[:, 0]] = True
        iu, ju = np.triu_indices(n, k=1)
        live = radii >= 1e-10
        m = ~skip[iu, ju] & live[iu] & live[ju]
        iu, ju = iu[m], ju[m]
        cache = (key, iu, ju, radii[iu] + radii[ju])
        common.__dict__["_wca_capped_cache"] = cache
    _, iu, ju, sig = cache
    if iu.size == 0:
        return F
    pos = np.asarray(state.positions, dtype=float)
    ok = np.all(np.isfinite(pos), axis=1) & (np.max(np.abs(pos), axis=1) <= 1e6)
    ok_p = ok[iu] & ok[ju]
    if not np.any(ok_p):
        return F
    if not ok_p.all():
        iu, ju, sig = iu[ok_p], ju[ok_p], sig[ok_p]
    dr = pos[iu] - pos[ju]
    r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
    m = (r >= 1e-10) & (r < 2.0 ** (1.0 / 6.0) * sig)
    if not np.any(m):
        return F
    iu, ju, dr, r, s = iu[m], ju[m], dr[m], r[m], sig[m]
    r_core = core_frac * s
    in_core = r < r_core
    f_over_r = np.empty_like(r)
    if np.any(in_core):
        inv_f2 = 1.0 / (core_frac * core_frac)
        inv_f6 = inv_f2 * inv_f2 * inv_f2
        f_core = (4.0 * eps / r_core[in_core]) * (12.0 * inv_f6 * inv_f6 - 6.0 * inv_f6)
        f_over_r[in_core] = f_core / r[in_core]
    out_core = ~in_core
    if np.any(out_core):
        sr6 = (s[out_core] / r[out_core]) ** 6
        f_over_r[out_core] = (
            4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / (r[out_core] ** 2)
        )
    fvec = f_over_r[:, None] * dr
    for c in range(3):
        F[:, c] += np.bincount(iu, weights=fvec[:, c], minlength=n)
        F[:, c] -= np.bincount(ju, weights=fvec[:, c], minlength=n)
    return F


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
    debye_length: Optional[float] = None,
    dielectric: float = SOLVENT_DIELECTRIC,
    kT_scale: float = 1.0,
    use_self_born: bool = False,
    gb_params: Optional[Dict[str, Any]] = None,
    project_ext_rigid: bool = False,
    wca_core_cap_fraction: float = 0.0,
) -> None:
    """Advance chain internal coordinates by one Brownian-dynamics step.

    Applies per-atom Ermak-McCammon integration with Stokes-Einstein mobilities
    (zero mobility for ghost beads with radius < 1e-10 A), optional intra-chain
    Debye-Huckel electrostatics, optional generalized Born (GB-OBC2) forces,
    and optional rigid-mode projection of external body-frame forces.
    """
    core_frac = float(wca_core_cap_fraction)
    raw_wca = bool(soft_repulsion) and core_frac == 0.0
    compute_chain_forces(
        state,
        kT=kT,
        soft_repulsion=raw_wca,
        soft_repulsion_eps=soft_repulsion_eps,
        debye_length=debye_length,
        dielectric=dielectric,
        kT_scale=kT_scale,
    )
    if soft_repulsion and core_frac > 0.0:
        state.forces += _intra_wca_capped_forces(
            state, state.common, soft_repulsion_eps, core_frac
        )
    if use_self_born:
        gp = gb_params or {}
        excl = gp.get("exclude_pair_mask")
        if excl is None:
            excl = bonded_exclusions(state.common)
        if "kappa" in gp and gp["kappa"] is not None:
            kappa = float(gp["kappa"])
        elif (
            debye_length is not None
            and debye_length > 0.0
            and not math.isinf(debye_length)
        ):
            kappa = 0.73 / float(debye_length)
        else:
            kappa = 0.0
        charges = np.array([a.charge for a in state.common.atoms], dtype=float)
        radii_gb = np.array([a.radius for a in state.common.atoms], dtype=float)
        f_gb, _ = chain_full_gb_force(
            state.positions,
            charges,
            radii_gb,
            eps_in=float(gp.get("eps_in", 1.0)),
            eps_out=float(gp.get("eps_out", SOLVENT_DIELECTRIC)),
            coffdrop_active=bool(gp.get("coffdrop_active", False)),
            exclude_pair_mask=excl,
            obc_kwargs=gp.get("obc_kwargs"),
            kT_scale=float(gp.get("kT_scale", kT_scale)),
            kappa=kappa,
        )
        state.forces += f_gb
    if external_forces is not None:
        if project_ext_rigid:
            state.forces += _project_ext_rigid(state.positions, external_forces)
        else:
            state.forces += external_forces
    n_atoms = state.n_atoms
    pos = state.positions
    old = pos.copy() if remove_rigid_rotation else None
    forces = state.forces
    if hasattr(state.common, "hydro_radii_array"):
        radii = np.asarray(state.common.hydro_radii_array(), dtype=float)
    else:
        radii = np.array(
            [
                a.hydro_radius if a.hydro_radius > 0.0 else a.radius
                for a in state.common.atoms
            ],
            dtype=float,
        )
    live = radii >= 1e-10
    safe_radii = np.where(live, radii, 1.0)
    mobilities = np.where(live, 1.0 / (6.0 * math.pi * viscosity * safe_radii), 0.0)
    diffusivities = kT * mobilities
    drifts = diffusivities[:, None] * forces * dt
    sigmas = np.sqrt(2.0 * diffusivities * dt)
    noise = sigmas[:, None] * rng.standard_normal((n_atoms, 3))
    MAX_INNER_DRIFT_A = 5.0
    bad_drift = ~np.all(np.isfinite(drifts), axis=1)
    if bad_drift.any():
        drifts[bad_drift] = 0.0
    drift_mags = np.linalg.norm(drifts, axis=1)
    inf_mag = ~np.isfinite(drift_mags)
    if inf_mag.any():
        am = np.max(np.abs(drifts[inf_mag]), axis=1, keepdims=True)
        scaled = drifts[inf_mag] / np.where(am > 0.0, am, 1.0)
        sn = np.linalg.norm(scaled, axis=1, keepdims=True)
        drifts[inf_mag] = scaled * (MAX_INNER_DRIFT_A / np.where(sn > 0.0, sn, 1.0))
        drift_mags[inf_mag] = MAX_INNER_DRIFT_A
    overshoot = drift_mags > MAX_INNER_DRIFT_A
    if overshoot.any():
        scale = MAX_INNER_DRIFT_A / drift_mags[overshoot]
        drifts[overshoot] = drifts[overshoot] * scale[:, None]
    pos += drifts + noise
    if remove_rigid_rotation:
        if apply_constraints and (
            state.common.length_constraints or state.common.coplanar_constraints
        ):
            satisfy_constraints_hybrid(
                state,
                tol=constraint_tol,
                shake_max_iter=constraint_max_iter,
                newton_max_iter=50,
                mobilities=mobilities,
            )
        pos = state.positions
        pos -= pos.mean(axis=0)
        if n_atoms >= 3:
            H = pos.T @ (old - old.mean(axis=0))
            U, S, Vt = np.linalg.svd(H)
            if S[0] > 0.0 and S[1] >= 1e-12 * S[0]:
                sgn = np.sign(np.linalg.det(Vt.T @ U.T))
                if sgn == 0.0:
                    sgn = 1.0
                Q = Vt.T @ np.diag([1.0, 1.0, sgn]) @ U.T
                pos[:] = pos @ Q.T
        return
    pos -= pos.mean(axis=0)
    if apply_constraints and (
        state.common.length_constraints or state.common.coplanar_constraints
    ):
        satisfy_constraints_hybrid(
            state,
            tol=constraint_tol,
            shake_max_iter=constraint_max_iter,
            newton_max_iter=50,
            mobilities=mobilities,
        )
        pos = state.positions
        pos -= pos.mean(axis=0)


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
    MAX_OUTER_DISP_A = 5.0
    disp = float(np.linalg.norm(new_pos - pos))
    if not (math.isfinite(disp) and disp <= MAX_OUTER_DISP_A):
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
    *,
    return_rejected: bool = False,
):
    """Advance rigid-body coordinates with pre-drawn Wiener increments.

    When return_rejected is True, returns (new_pos, new_ori, disp_rejected)
    where disp_rejected flags moves that exceeded MAX_OUTER_DISP_A or produced
    non-finite coordinates.
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
    MAX_OUTER_DISP_A = 5.0
    disp = float(np.linalg.norm(new_pos - pos))
    rejected = not (math.isfinite(disp) and disp <= MAX_OUTER_DISP_A)
    if rejected:
        return (pos, ori, True) if return_rejected else (pos, ori)
    return (new_pos, new_ori, False) if return_rejected else (new_pos, new_ori)


def trajectory_rngs(
    seed: Optional[int], traj_idx: int
) -> Tuple[np.random.Generator, np.random.Generator]:
    """Return the main and the bridge generator of trajectory traj_idx.

    Integer seeds spawn deterministically from SeedSequence(seed,
    spawn_key=(traj_idx,)). A SeedSequence spawns child traj_idx under its
    entropy. When seed is None, fresh OS entropy is drawn.
    """
    if isinstance(seed, np.random.SeedSequence):
        seq = np.random.SeedSequence(
            seed.entropy,
            spawn_key=tuple(seed.spawn_key) + (traj_idx,),
        )
    else:
        seq = np.random.SeedSequence(seed, spawn_key=(traj_idx,))
    main, bridge = seq.spawn(2)
    return np.random.default_rng(main), np.random.default_rng(bridge)


# The simulator and per-run root seed of a worker process. The pool initializer
# sets both once, avoiding re-sending grids and force-field tables per trajectory.
_WORKER_SIM = None
_WORKER_SEED = None


def _chain_worker_init(sim, run_seed=None) -> None:
    global _WORKER_SIM, _WORKER_SEED
    _WORKER_SIM = sim
    _WORKER_SEED = run_seed if run_seed is not None else sim.params.seed


def _run_chain_trajectory_worker(traj_idx: int):
    """Run trajectory traj_idx of the worker's simulator with its own streams."""
    rng, rng_bb = trajectory_rngs(_WORKER_SEED, traj_idx)
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
        elec_grids: Optional[List] = None,
        born_grids: Optional[List] = None,
        use_far_field: bool = False,
        multipole_expansion=None,
        pdie: float = SOLUTE_DIELECTRIC,
    ):
        if params.diffusion_update_interval < 0:
            raise ValueError(
                "diffusion_update_interval must be >= 0, got "
                f"{params.diffusion_update_interval}"
            )
        if params.diffusion_update_interval > 0 and not auto_diffusion:
            raise ValueError(
                "diffusion_update_interval > 0 requires auto_diffusion=True "
                "for geometry-derived tensors. Explicit D_trans and D_rot are fixed."
            )
        if params.inner_field and not params.remove_internal_rotation:
            raise ValueError(
                "inner_field requires remove_internal_rotation to remove the "
                "rigid-body component of target forces from inner steps"
            )
        a_target = None
        _T = params.temperature_kT / KB_KCAL
        self._kT_scale = float(params.temperature_kT) / KBT_KCAL
        if target_hydro_radius is not None:
            a_target = float(target_hydro_radius)
            if not (math.isfinite(a_target) and a_target > 0.0):
                raise ValueError(
                    "target_hydro_radius must be a finite number > 0, got "
                    f"{target_hydro_radius}"
                )
            if target is None:
                raise ValueError("target_hydro_radius requires a target")
        if (target_hydro_radius is not None or auto_diffusion) and abs(
            _T - T_DEFAULT
        ) > 0.01:
            warnings.warn(
                f"Temperature {_T:.2f} K differs from {T_DEFAULT} K: "
                "water viscosity remains fixed at 25 C (0.890 mPa*s) while "
                "thermal kT and diffusion scale by T / 298.15.",
                RuntimeWarning,
                stacklevel=2,
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
        self._chain_radii = np.array(
            [a.radius for a in chain_template.atoms],
            dtype=float,
        )
        if hasattr(chain_template, "hydro_radii_array"):
            self._hydro_radii = np.asarray(
                chain_template.hydro_radii_array(), dtype=float
            )
        else:
            self._hydro_radii = self._chain_radii.copy()
        if auto_diffusion:
            if D_trans is not None or D_rot is not None:
                raise ValueError(
                    "auto_diffusion=True is incompatible with explicitly "
                    "supplied D_trans or D_rot. Either set auto_diffusion=True "
                    "and let the simulator compute D from chain geometry, "
                    "or pass D_trans and D_rot directly."
                )
            if params.rotate_body_tensors:
                D_trans_auto, D_rot_auto, _ = chain_diffusion_tensors(
                    chain_init_body_positions,
                    self._hydro_radii,
                    kT=self._kT_scale,
                    center=np.zeros(3),
                )
            else:
                D_trans_auto, D_rot_auto, _ = chain_diffusion_tensors(
                    chain_init_body_positions,
                    self._hydro_radii,
                    kT=self._kT_scale,
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
        e_grids = (
            list(elec_grids)
            if elec_grids is not None
            else ([target_grid] if target_grid is not None else [])
        )
        b_grids = (
            list(born_grids)
            if born_grids is not None
            else ([born_grid] if born_grid is not None else [])
        )
        self.elec_grids = e_grids
        self.born_grids = b_grids
        self.target_grid = self.elec_grids[0] if self.elec_grids else target_grid
        self.born_grid = self.born_grids[0] if self.born_grids else born_grid
        self.desolvation_alpha = float(desolvation_alpha)
        self.pdie = float(pdie)
        self.auto_diffusion = auto_diffusion
        self._last_disp_rejected = False

        # Explicit Euler stability check for stiff harmonic bonds and elastic network springs.
        live_hydro = self._hydro_radii[self._hydro_radii >= 1.0e-10]
        if live_hydro.size > 0:
            D_bead_max = self._kT_scale / (
                6.0 * math.pi * WATER_VISCOSITY * float(live_hydro.min())
            )
            k_springs = [float(b.k_spring) for b in chain_template.bonds] + [
                float(b.k_spring) for b in getattr(chain_template, "network", [])
            ]
            k_spring_max = max(k_springs, default=0.0)
            euler_num = D_bead_max * k_spring_max * float(params.dt_chain)
            if euler_num >= 1.5:
                warnings.warn(
                    f"Internal chain timestep dt_chain={params.dt_chain:g} ps gives "
                    f"D_bead_max * k_spring_max * dt_chain = {euler_num:.3g} >= 1.5 "
                    "(explicit Euler stability limit is 2.0); reduce dt_chain to "
                    "prevent bond oscillations.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Pre-build per-trajectory reusables, GB settings, and intra-chain DH settings.
        self._chain_scratch = make_chain_scratch_molecule(chain_template)
        self._chain_charges = np.array(
            [a.charge for a in chain_template.atoms], dtype=float
        )
        self._gb_exclude_mask = bonded_exclusions(chain_template)
        self._gb_kappa = (
            0.73 / float(params.debye_length)
            if (
                params.debye_length is not None
                and math.isfinite(float(params.debye_length))
                and float(params.debye_length) > 0.0
            )
            else 0.0
        )
        self._gb_params = dict(
            eps_in=params.gb_eps_in,
            eps_out=params.gb_eps_out,
            coffdrop_active=params.coffdrop_active,
            exclude_pair_mask=self._gb_exclude_mask,
            obc_kwargs=dict(
                obc_alpha=params.gb_obc_alpha,
                obc_beta=params.gb_obc_beta,
                obc_gamma=params.gb_obc_gamma,
            ),
            kT_scale=self._kT_scale,
            kappa=self._gb_kappa,
        )
        if params.use_self_born and params.coffdrop_active:
            warnings.warn(
                "use_self_born=True with coffdrop_active=True evaluates diagonal "
                "self-Born terms without off-diagonal GB cross terms.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._use_intra_dh = (
            bool(params.intra_debye_huckel)
            and (
                bool(params.coffdrop_active)
                or bool(getattr(chain_template, "pair_lookups", None))
            )
            and not (bool(params.use_self_born) and not bool(params.coffdrop_active))
        )
        self._intra_dh_debye = (
            float(params.debye_length)
            if (
                self._use_intra_dh
                and params.debye_length is not None
                and math.isfinite(float(params.debye_length))
                and float(params.debye_length) > 0.0
            )
            else None
        )

        # Deterministic bulk equilibration probe for equilibrium chain diffusion
        # and bounding radius (OuterPropagator and backstep radius). Fixed seed
        # and ensemble size stay independent of params.seed and n_trajectories.
        D_t_eq = self.D_trans
        D_r_eq = self.D_rot
        chain_br_eq = float(
            np.max(
                np.linalg.norm(chain_init_body_positions, axis=1) + self._chain_radii
            )
        )
        if (
            params.n_equilibration_steps > 0
            and self.auto_diffusion
            and params.diffusion_update_interval > 0
        ):
            n_eq_probe = int(params.n_equilibration_steps)
            n_probe = 8
            probe_seqs = np.random.SeedSequence(1592647681).spawn(n_probe)
            Dt_samples: List[np.ndarray] = []
            Dr_samples: List[np.ndarray] = []
            br_samples: List[float] = []
            center_arg = np.zeros(3) if params.rotate_body_tensors else None
            for seq_p in probe_seqs:
                eq_state = ChainState.from_template(
                    chain_template, self.body_positions.copy()
                )
                eq_rng = np.random.default_rng(seq_p)
                for _ in range(n_eq_probe):
                    chain_internal_bd_step(
                        eq_state,
                        dt=params.dt_chain,
                        rng=eq_rng,
                        kT=self._kT_scale,
                        soft_repulsion=params.use_soft_repulsion,
                        soft_repulsion_eps=params.soft_repulsion_eps,
                        constraint_tol=params.constraint_tol,
                        constraint_max_iter=params.constraint_max_iter,
                        remove_rigid_rotation=params.remove_internal_rotation,
                        debye_length=self._intra_dh_debye,
                        dielectric=params.dielectric,
                        kT_scale=self._kT_scale,
                        use_self_born=params.use_self_born,
                        gb_params=self._gb_params,
                        wca_core_cap_fraction=params.wca_core_cap_fraction,
                    )
                try:
                    D_t_p, D_r_p, _ = chain_diffusion_tensors(
                        eq_state.positions,
                        self._hydro_radii,
                        kT=self._kT_scale,
                        center=center_arg,
                        warn_regularized=False,
                    )
                    Dt_samples.append(D_t_p)
                    Dr_samples.append(D_r_p)
                except np.linalg.LinAlgError:
                    pass
                if np.all(np.isfinite(eq_state.positions)):
                    br_samples.append(
                        float(
                            np.max(
                                np.linalg.norm(eq_state.positions, axis=1)
                                + self._chain_radii
                            )
                        )
                    )
            if Dt_samples:
                D_t_eq = np.mean(Dt_samples, axis=0)
                D_r_eq = np.mean(Dr_samples, axis=0)
            if br_samples:
                chain_br_eq = max(br_samples)

        # Two-body hydrodynamics remain None for a fixed target.
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
            D_c0 = _isotropic_diffusion(D_t_eq)
            D_rc0 = _isotropic_diffusion(D_r_eq)
            if not (D_c0 > 0.0 and D_rc0 > 0.0):
                raise ValueError(
                    "target_hydro_radius requires D_trans and D_rot of the "
                    f"chain > 0, got tr(D_trans)/3={D_c0} and tr(D_rot)/3={D_rc0}"
                )
            self._D_t_target = stokes_translational_diffusion(a_target) * self._kT_scale
            self._D_r_target = stokes_rotational_diffusion(a_target) * self._kT_scale
            self._a_chain = self._kT_scale / (6.0 * math.pi * WATER_VISCOSITY * D_c0)
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
        if outputs is None:
            from pystarc.pipeline.input_parser import OutputConfig

            outputs = OutputConfig()
        self.outputs = outputs
        if self.target is not None and len(self.target.atoms) > 0:
            self._target_pos_cached = np.array(
                [[a.x, a.y, a.z] for a in self.target.atoms],
                dtype=float,
            )
        else:
            self._target_pos_cached = None
        self._geometry_key = None
        # Stokes hydrodynamic radius derived from translational diffusion in both modes.
        D_iso_eq = _isotropic_diffusion(D_t_eq)
        if D_iso_eq > 0.0:
            self._effective_hydro_radius = self._kT_scale / (
                6.0 * math.pi * WATER_VISCOSITY * D_iso_eq
            )
        else:
            self._effective_hydro_radius = (
                float(self._hydro_radii.max()) if self._hydro_radii.size else 1.0
            )
        self.use_far_field = bool(use_far_field)
        self.multipole_expansion = multipole_expansion
        enable_b2 = bool(
            params.enable_born2 and target is not None and self.desolvation_alpha != 0.0
        )
        self._cpu_field = None
        if len(e_grids) > 1 or len(b_grids) > 1 or self.use_far_field or enable_b2:
            from pystarc.forces.chain_target_field import ChainTargetField

            self._cpu_field = ChainTargetField(
                target=target,
                chain_charges=self._chain_charges,
                chain_radii=self._chain_radii,
                elec_grids=e_grids,
                born_grids=b_grids,
                desolvation_alpha=self.desolvation_alpha,
                use_far_field=self.use_far_field,
                target_charge=target.total_charge() if target is not None else 0.0,
                debye_length=params.debye_length,
                sdie=params.dielectric,
                kT_scale=self._kT_scale,
                multipole_expansion=multipole_expansion,
                gb_exclude_mask=self._gb_exclude_mask,
                xp=np,
                grid_float32=params.grid_float32,
                overlap_scale=params.overlap_scale,
                wca_core_cap_fraction=params.wca_core_cap_fraction,
                enable_born2=enable_b2,
                born_alpha=self.desolvation_alpha,
                pdie=self.pdie,
            )
        # Warm up cached topology arrays and stacked COFFDROP tables before parallel fork.
        compute_chain_forces(
            ChainState.from_template(chain_template, self.body_positions.copy()),
            soft_repulsion=params.use_soft_repulsion,
        )
        if params.seed is None:
            _init_seq = np.random.SeedSequence(None)
            _s_main, _s_bb = _init_seq.spawn(2)
            self.rng = np.random.default_rng(_s_main)
            self.rng_bb = np.random.default_rng(_s_bb)
        else:
            self.rng = np.random.default_rng(params.seed)
            self.rng_bb = np.random.default_rng(params.seed + 0xBB)
        self.results: List[TrajectoryResult] = []
        self.n_reacted = 0
        self.n_escaped = 0
        self.n_max_steps = 0
        self._outer_prop = None
        try:
            from pystarc.simulation.outer_propagator import (
                OuterPropagator,
                OPGroupInfo,
            )

            D_trans_eff = _isotropic_diffusion(D_t_eq)
            D_rot_eff = _isotropic_diffusion(D_r_eq)
            chain_q = float(sum(a.charge for a in chain_template.atoms))
            if self._a_target is not None:
                has_hi_op = bool(params.hydrodynamic_interactions)
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
                has_hi_op = False
                g0 = OPGroupInfo(
                    q=target.total_charge() if target is not None else 0.0,
                    Dtrans=0.0,
                    Drot=0.0,
                )
                g1 = OPGroupInfo(
                    q=chain_q,
                    Dtrans=D_trans_eff,
                    Drot=D_rot_eff,
                )
            kT = float(params.temperature_kT)
            viscosity = VISCOSITY
            dielectric = float(params.dielectric)
            vacuum_perm = 1.0 / (4 * math.pi * COULOMB_KCAL)
            debye_len = float(params.debye_length)
            target_br = target.bounding_radius() if target is not None else 0.0
            max_mol_r = max(target_br, chain_br_eq)
            self._outer_prop = OuterPropagator(
                b_radius=params.r_start,
                max_radius=max_mol_r,
                has_hi=has_hi_op,
                kT=kT,
                viscosity=viscosity,
                dielectric=dielectric,
                vacuum_perm=vacuum_perm,
                debye_len=debye_len,
                g0=g0,
                g1=g1,
            )
        except Exception as _op_err:
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
        """Compute per-atom external forces on the chain in kBT/A.

        Sums target electrostatic potential forces, Born desolvation forces
        (direction 1 and optional direction 2), and optional soft WCA steric
        repulsion against target atoms. Internal generalized Born forces run
        inside chain_internal_bd_step on the dt_chain clock.
        """
        if self._cpu_field is not None:
            wp_batch = np.asarray(world_positions, dtype=np.float64)[None, :, :]
            f_e, _ = self._cpu_field.evaluate_elec(wp_batch, want_energy=False)
            forces = f_e[0]
            if self.born_grids and self.desolvation_alpha != 0.0:
                f_b, _ = self._cpu_field.evaluate_born(wp_batch, want_energy=False)
                forces = forces + f_b[0]
            if self._cpu_field.enable_born2 and self._cpu_field.has_born2:
                f_b2, _ = self._cpu_field.evaluate_born2(wp_batch, want_energy=False)
                forces = forces + f_b2[0]
        else:
            if self.target_grid is not None:
                forces, _energy = evaluate_target_grid_force_on_chain(
                    world_positions,
                    self._chain_charges,
                    self.target_grid,
                )
            else:
                forces = np.zeros_like(world_positions)
            if self.born_grid is not None and self.desolvation_alpha != 0.0:
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
                wca_core_cap_fraction=self.params.wca_core_cap_fraction,
            )
        return forces

    def _sample_interaction_energy(
        self, world_positions: np.ndarray
    ) -> Tuple[float, float]:
        """Return (e_elec, e_born) interaction energy components in kBT."""
        e_elec = 0.0
        e_born = 0.0
        if self._cpu_field is not None:
            wp_b = np.asarray(world_positions, dtype=np.float64)[None, :, :]
            _, ee_arr = self._cpu_field.evaluate_elec(wp_b, want_energy=True)
            e_elec = float(ee_arr[0])
            if self.born_grids and self.desolvation_alpha != 0.0:
                _, eb_arr = self._cpu_field.evaluate_born(wp_b, want_energy=True)
                e_born += float(eb_arr[0])
            if self._cpu_field.enable_born2 and self._cpu_field.has_born2:
                _, eb2_arr = self._cpu_field.evaluate_born2(wp_b, want_energy=True)
                e_born += float(eb2_arr[0])
        else:
            if self.target_grid is not None:
                _, e_elec = evaluate_target_grid_force_on_chain(
                    world_positions, self._chain_charges, self.target_grid
                )
            if self.born_grid is not None and self.desolvation_alpha != 0.0:
                _, e_born = evaluate_born_force_on_chain(
                    world_positions,
                    self._chain_charges,
                    self.born_grid,
                    alpha=self.desolvation_alpha,
                )
        return e_elec, e_born

    def _refresh_geometry(self) -> None:
        """Cache target spheres and reaction pair arrays for current geometry."""
        key = (id(self.target), id(self.pathway_set))
        if key == self._geometry_key:
            return
        self._geometry_key = key
        self._target_spheres = target_spheres(self.target)
        t_rad = self._target_spheres[1]
        self._contact_reach = 0.0
        if t_rad.size:
            self._contact_reach = max(
                2.0 ** (1.0 / 6.0), float(self.params.overlap_scale)
            ) * (float(t_rad.max()) + float(self._chain_radii.max()))
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
        """Rebuild body-frame diffusion tensors from current coordinates."""
        info: Dict[str, Any] = {}
        D_t, D_r, _ = chain_diffusion_tensors(
            body_positions,
            self._hydro_radii,
            kT=self._kT_scale,
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

    def _relative_translation_tensor(
        self,
        pos: np.ndarray,
        ori: Quaternion,
        D_t_body,
        record_clip: bool = True,
    ) -> np.ndarray:
        """Compute symmetric positive-definite 3x3 two-body relative diffusion tensor.

        Forms D_rel = R D_t_body R^T + D_t_target I - 2 M_12 with the Zuk
        three-regime RPY off-diagonal block evaluated at the instantaneous
        Stokes radius of D_t_body. Eigenvalues below RELATIVE_DIFFUSION_FLOOR
        are clipped to maintain positive definiteness.
        """
        I3 = np.eye(3)
        D_t_arr = np.asarray(D_t_body, dtype=float)
        if D_t_arr.ndim == 0:
            D_t_arr = float(D_t_arr) * I3
        D_rel = rotate_tensor_to_lab(D_t_arr, ori) + self._D_t_target * I3
        if self._pair_mob is not None:
            D_c = max(float(np.trace(D_t_arr) / 3.0), 1e-12)
            a_c = self._kT_scale / (6.0 * math.pi * WATER_VISCOSITY * D_c)
            if np.all(np.isfinite(pos)):
                mtt, _, _, _ = rpy_pair_blocks(self._a_target, a_c, pos)
                D_rel = D_rel - 2.0 * (mtt * (self._kT_scale / WATER_VISCOSITY))
        D_rel = 0.5 * (D_rel + D_rel.T)
        eig, vec = np.linalg.eigh(D_rel)
        if not eig[0] > 0.0:
            D_rel = (vec * np.maximum(eig, RELATIVE_DIFFUSION_FLOOR)) @ vec.T
            if record_clip:
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
        return D_rel

    def _relative_translation(
        self, pos: np.ndarray, ori: Quaternion, D_t_body, dt: float
    ) -> Tuple[np.ndarray, "Optional[np.ndarray]"]:
        """Return (D_rel, None) for the divergence-free 3x3 Zuk RPY relative tensor."""
        D_rel = self._relative_translation_tensor(pos, ori, D_t_body, record_clip=True)
        return D_rel, None

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
        """Advance rigid-body coordinates (pos, ori) by one step dt.

        Measures MAX_OUTER_DISP_A on the translational displacement before target
        frame rotation and records rejection status in self._last_disp_rejected.
        """
        self._last_disp_rejected = False
        if self._a_target is None:
            D_t_in = (
                rotate_tensor_to_lab(D_t_body, ori)
                if self.params.rotate_body_tensors
                else D_t_body
            )
            D_r_in = (
                rotate_tensor_to_lab(D_r_body, ori)
                if self.params.rotate_body_tensors
                else D_r_body
            )
            new_pos, new_ori, rej = chain_outer_bd_step_wiener(
                pos,
                ori,
                world_pos,
                per_atom_forces,
                D_t_in,
                D_r_in,
                dt,
                dW_t,
                dW_r,
                return_rejected=True,
            )
            self._last_disp_rejected = rej
            return new_pos, new_ori
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
        MAX_OUTER_DISP_A = 5.0
        disp = float(np.linalg.norm(new_pos - pos))
        if not (math.isfinite(disp) and disp <= MAX_OUTER_DISP_A):
            self._last_disp_rejected = True
            return pos, ori
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
        if not np.all(np.isfinite(new_pos)):
            self._last_disp_rejected = True
            return pos, ori
        return new_pos, new_ori

    def _pair_distances(self, world_pos: np.ndarray) -> "list[np.ndarray]":
        """Distances of the reaction pairs, one array per reaction."""
        out = []
        for t_pos, c_idx, _cut, _l1 in self._rxn_arrays:
            d = t_pos - world_pos[c_idx]
            out.append(np.sqrt(np.einsum("ij,ij->i", d, d)))
        return out

    def _bridge_diffusion(
        self,
        body_positions,
        D_t_body,
        D_r_body,
        pos: "Optional[np.ndarray]" = None,
        ori: "Optional[Quaternion]" = None,
    ) -> list:
        """Per-pair normal diffusivity D_eff for the Brownian bridge, per reaction.

        When pos and ori are supplied, projects the lab-frame HI-reduced
        relative diffusion tensor along each pair normal n_hat, projects the
        lab-frame chain and target rotational tensors along (L x n_hat), and
        adds the internal bead diffusivity of the chain contact atom.
        """
        if pos is None or ori is None:
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

        world_pos = place_chain(body_positions, pos, ori)
        I3 = np.eye(3)
        D_t_arr = (
            float(D_t_body) * I3
            if np.ndim(D_t_body) == 0
            else np.asarray(D_t_body, dtype=float)
        )
        D_r_arr = (
            float(D_r_body) * I3
            if np.ndim(D_r_body) == 0
            else np.asarray(D_r_body, dtype=float)
        )
        if self._a_target is not None:
            D_rel = self._relative_translation_tensor(
                pos, ori, D_t_arr, record_clip=False
            )
            D_r_lab = rotate_tensor_to_lab(D_r_arr, ori)
        else:
            D_rel = (
                rotate_tensor_to_lab(D_t_arr, ori)
                if self.params.rotate_body_tensors
                else D_t_arr
            )
            D_r_lab = (
                rotate_tensor_to_lab(D_r_arr, ori)
                if self.params.rotate_body_tensors
                else D_r_arr
            )
        n_atoms = len(self._hydro_radii)
        if n_atoms <= 1:
            int_frac = 0.0
        elif n_atoms == 2:
            int_frac = 0.5
        elif self.params.remove_internal_rotation:
            int_frac = (n_atoms - 2.0) / n_atoms
        else:
            int_frac = (n_atoms - 1.0) / n_atoms
        out = []
        for t_pos, c_idx, _cut, _l1_sq in self._rxn_arrays:
            c_pts = world_pos[c_idx]
            diff = c_pts - t_pos
            d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            good = d > 1.0e-12
            n_hat = np.zeros_like(diff)
            n_hat[:, 0] = 1.0
            if np.any(good):
                n_hat[good] = diff[good] / d[good, None]
            n_D_n = np.einsum("pi,ij,pj->p", n_hat, D_rel, n_hat)
            L2_lab = c_pts - pos[None, :]
            u2 = np.cross(L2_lab, n_hat)
            u2_Dr_u2 = np.einsum("pi,ij,pj->p", u2, D_r_lab, u2)
            a_j = self._hydro_radii[c_idx]
            live_j = a_j >= 1.0e-10
            D_bead_j = np.where(
                live_j,
                self._kT_scale
                / (6.0 * math.pi * WATER_VISCOSITY * np.where(live_j, a_j, 1.0)),
                0.0,
            )
            D = (
                np.maximum(n_D_n, RELATIVE_DIFFUSION_FLOOR)
                + np.maximum(u2_Dr_u2, 0.0)
                + int_frac * D_bead_j
            )
            if self._a_target is not None and self._D_r_target > 0.0:
                u1 = np.cross(t_pos, n_hat)
                D = D + self._D_r_target * np.einsum("ij,ij->i", u1, u1)
            out.append(np.maximum(D, 1.0e-12))
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
        # Pre-equilibrate internal coordinates in bulk before starting on the b-sphere.
        for _ in range(params.n_equilibration_steps):
            chain_internal_bd_step(
                state,
                dt=params.dt_chain,
                rng=rng,
                kT=self._kT_scale,
                soft_repulsion=params.use_soft_repulsion,
                soft_repulsion_eps=params.soft_repulsion_eps,
                constraint_tol=params.constraint_tol,
                constraint_max_iter=params.constraint_max_iter,
                remove_rigid_rotation=params.remove_internal_rotation,
                debye_length=self._intra_dh_debye,
                dielectric=params.dielectric,
                kT_scale=self._kT_scale,
                use_self_born=params.use_self_born,
                gb_params=self._gb_params,
                wca_core_cap_fraction=params.wca_core_cap_fraction,
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
                _world_pos = place_chain(state.positions, pos, ori)
                _e_elec, _e_born = self._sample_interaction_energy(_world_pos)
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
            world_pos = place_chain(state.positions, pos, ori)
            update_chain_scratch_positions(self._chain_scratch, world_pos)
            pair_dists = self._pair_distances(world_pos) if has_rxn else None
            if (
                params.use_brownian_bridge
                and prev_pair_dists is not None
                and pair_dists is not None
                and prev_dt_outer > 0.0
                and self.pathway_set is not None
                and self.target is not None
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
            elif self.pathway_set is not None and self.target is not None:
                rxn = check_chain_reaction(
                    self.target,
                    self._chain_scratch,
                    self.pathway_set,
                    rng,
                )
            else:
                rxn = None
            if rxn is not None:
                _e_elec_r, _e_born_r = self._sample_interaction_energy(world_pos)
                return _result(
                    Fate.REACTED,
                    step,
                    rxn,
                    encounter_pos=pos.copy(),
                    encounter_q=np.array([ori.w, ori.x, ori.y, ori.z]),
                    energy_at_reaction=_e_elec_r + _e_born_r,
                )
            # LMZ outer propagator returns trajectories to the b-sphere or escapes them.
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
                    return _result(Fate.ESCAPED, step)
                n_returns += 1
                world_pos = place_chain(state.positions, pos, ori)
                update_chain_scratch_positions(self._chain_scratch, world_pos)
                pair_dists = self._pair_distances(world_pos) if has_rxn else None
                prev_pair_dists = None
                rxn_ret = (
                    check_chain_reaction(
                        self.target,
                        self._chain_scratch,
                        self.pathway_set,
                        rng,
                    )
                    if self.pathway_set is not None and self.target is not None
                    else None
                )
                if rxn_ret is not None:
                    _e_elec_ret, _e_born_ret = self._sample_interaction_energy(
                        world_pos
                    )
                    return _result(
                        Fate.REACTED,
                        step,
                        rxn_ret,
                        encounter_pos=pos.copy(),
                        encounter_q=np.array([ori.w, ori.x, ori.y, ori.z]),
                        energy_at_reaction=_e_elec_ret + _e_born_ret,
                    )
            _lmz_active = self._outer_prop is not None and params.use_lmz
            if not _lmz_active and check_escape(pos, params.r_escape):
                return _result(Fate.ESCAPED, step)
            if step == params.max_steps:
                break
            if diff_interval > 0 and step > 0 and step % diff_interval == 0:
                _rebuild_tensors()
            # Synchronize inner clock exactly to dt_outer.
            dt_outer = self._outer_dt(pair_dists)
            n_inner = max(1, math.ceil(dt_outer / params.dt_chain - 1e-12))
            dt_sub = dt_outer / n_inner
            if params.use_brownian_bridge and has_rxn:
                D_eff = self._bridge_diffusion(
                    state.positions, D_t_body, D_r_body, pos=pos, ori=ori
                )
            per_atom_forces = self._compute_per_atom_external_forces(world_pos)
            F_net_old = per_atom_forces.sum(axis=0)
            body_old = state.positions.copy()
            f_body = None
            if params.inner_field:
                f_body = per_atom_forces @ ori.to_rotation_matrix()
            for _ in range(n_inner):
                chain_internal_bd_step(
                    state,
                    dt=dt_sub,
                    rng=rng,
                    kT=self._kT_scale,
                    constraint_tol=params.constraint_tol,
                    constraint_max_iter=params.constraint_max_iter,
                    soft_repulsion=params.use_soft_repulsion,
                    soft_repulsion_eps=params.soft_repulsion_eps,
                    remove_rigid_rotation=params.remove_internal_rotation,
                    external_forces=f_body,
                    debye_length=self._intra_dh_debye,
                    dielectric=params.dielectric,
                    kT_scale=self._kT_scale,
                    use_self_born=params.use_self_born,
                    gb_params=self._gb_params,
                    project_ext_rigid=bool(params.inner_field),
                    wca_core_cap_fraction=params.wca_core_cap_fraction,
                )
            pos_old, ori_old = pos, ori
            mu_eff = WATER_VISCOSITY / self._kT_scale
            D_c_inst = _isotropic_diffusion(D_t_body)
            if D_c_inst > 0.0:
                inv_r_eff = 6.0 * math.pi * mu_eff * D_c_inst
            else:
                inv_r_eff = 1.0 / max(self._effective_hydro_radius, 1.0e-6)
            if self._a_target is not None:
                inv_r_eff += 1.0 / max(self._a_target, 1.0e-6)
            r_eff_backstep = 1.0 / inv_r_eff

            def _move(dW_t, dW_r, dW_r1):
                """Advance rigid pose and subdivide on displacement or force-change spikes."""
                clipped_before = self._n_relative_clipped
                self._last_disp_rejected = False
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
                disp_rej = self._last_disp_rejected
                need_subdivide = False
                if dt_outer > params.dt_rxn:
                    if disp_rej:
                        need_subdivide = True
                    elif params.force_change_backstep:
                        F_net_new = self._compute_per_atom_external_forces(
                            place_chain(state.positions, new_pos, new_ori)
                        ).sum(axis=0)
                        need_subdivide = backstep_due_to_force(
                            F_net_new,
                            F_net_old,
                            new_pos,
                            pos_old,
                            dt_outer,
                            params.dt_rxn,
                            radius=r_eff_backstep,
                            viscosity=mu_eff,
                        )
                if not need_subdivide:
                    return new_pos, new_ori, disp_rej
                # Restore clip count before subdividing so tentative clip is not double-counted.
                self._n_relative_clipped = clipped_before
                s = math.sqrt(dt_outer / 4.0)
                dW_mid_t = 0.5 * dW_t + s * rng.standard_normal(3)
                dW_mid_r = 0.5 * dW_r + s * rng.standard_normal(3)
                dW_mid_r1 = None
                if dW_r1 is not None:
                    dW_mid_r1 = 0.5 * dW_r1 + s * rng.standard_normal(3)
                hdt = dt_outer / 2.0
                self._last_disp_rejected = False
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
                if self._last_disp_rejected:
                    return pos_old, ori_old, True
                world_mid = place_chain(state.positions, mid_pos, mid_ori)
                self._last_disp_rejected = False
                end_pos, end_ori = self._rigid_step(
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
                if self._last_disp_rejected:
                    return pos_old, ori_old, True
                return end_pos, end_ori, False

            def _draw():
                """Sample Wiener increments for one outer rigid move."""
                dW_t = math.sqrt(dt_outer) * rng.standard_normal(3)
                dW_r = math.sqrt(dt_outer) * rng.standard_normal(3)
                dW_r1 = None
                if self._a_target is not None:
                    dW_r1 = math.sqrt(dt_outer) * rng.standard_normal(3)
                return dW_t, dW_r, dW_r1

            pos, ori, disp_rejected = _move(*_draw())
            moved = not disp_rejected
            if disp_rejected:
                state.positions[:] = body_old
            # Single-attempt Metropolis hard-sphere rejection preserving detailed balance.
            if params.use_hard_sphere and moved:
                world_new = place_chain(state.positions, pos, ori)
                if not np.all(np.isfinite(world_new)):
                    state.positions[:] = body_old
                    pos, ori = pos_old, ori_old
                    n_hs_rejections += 1
                    moved = False
                else:
                    spheres = self._nearby_spheres(world_new)
                    if spheres is not None and _check_chain_overlap(
                        spheres,
                        world_new,
                        self._chain_radii,
                        overlap_scale=params.overlap_scale,
                    ):
                        state.positions[:] = body_old
                        pos, ori = pos_old, ori_old
                        n_hs_rejections += 1
                        moved = False
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

        Dispatches to a parallel multiprocessing pool when both
        params.n_threads > 1 and params.n_trajectories > 1, and otherwise runs
        serially. Integer seeds spawn trajectory i deterministically from
        trajectory_rngs(seed, i) across both modes; seed=None draws one fresh
        SeedSequence per run() and spawns independent per-trajectory children.
        Returns the accumulated self.results list.
        """
        n = self.params.n_trajectories
        run_seed = (
            np.random.SeedSequence(None).entropy
            if self.params.seed is None
            else self.params.seed
        )
        if self.params.n_threads > 1 and n > 1:
            self._run_parallel(n, run_seed=run_seed)
        else:
            self._run_serial(n, run_seed=run_seed)
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
        p_raw = self.n_reacted / done if done else 0.0
        nam_omega = 0.0
        if (
            not self.params.use_lmz
            and self._outer_prop is not None
            and kb > 0.0
            and self.params.r_escape > self.params.r_start > 0.0
            and hasattr(self._outer_prop, "_relative_rate")
        ):
            rate_esc = float(self._outer_prop._relative_rate(self.params.r_escape))
            if rate_esc > 0.0:
                nam_omega = min(max(kb / rate_esc, 0.0), 1.0 - 1.0e-12)
        if nam_omega > 0.0:
            denom = 1.0 - (1.0 - p_raw) * nam_omega
            p_live = p_raw / denom if denom > 0.0 else 1.0
        else:
            p_live = p_raw
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

    def _run_serial(self, n: int, run_seed=None) -> None:
        """Run n trajectories sequentially."""
        t0 = time.monotonic()
        report_every = max(1, n // 100)
        seed = self.params.seed if run_seed is None else run_seed
        if seed is None:
            seed = np.random.SeedSequence(None).entropy
        for i in range(n):
            rng, rng_bb = trajectory_rngs(seed, i)
            self._record(self.run_one(rng=rng, rng_bb=rng_bb))
            done = i + 1
            if self.params.verbose and (done % report_every == 0 or done == n):
                self._log_progress(done, n, t0)

    def _run_parallel(self, n: int, run_seed=None) -> None:
        """Run n trajectories across a multiprocessing pool.

        Workers stream completed outcomes via imap_unordered for live counts,
        while self.results preserves deterministic trajectory index order.
        """
        n_workers = min(self.params.n_threads, n, mp.cpu_count())
        if self.params.verbose:
            print(f"  Parallel: {n_workers} workers, {n} trajectories", flush=True)
        t0 = time.monotonic()
        report_every = max(1, n // 100)
        seed = self.params.seed if run_seed is None else run_seed
        if seed is None:
            seed = np.random.SeedSequence(None).entropy
        ordered: List[Optional[TrajectoryResult]] = [None] * n
        with mp.Pool(
            n_workers, initializer=_chain_worker_init, initargs=(self, seed)
        ) as pool:
            for done, (idx, result) in enumerate(
                pool.imap_unordered(_run_chain_trajectory_indexed, range(n)), 1
            ):
                ordered[idx] = result
                self._count_fate(result)
                if self.params.verbose and (done % report_every == 0 or done == n):
                    self._log_progress(done, n, t0)
        self.results.extend(ordered)
