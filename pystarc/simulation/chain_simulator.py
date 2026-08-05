"""
PySTARC chain Brownian-dynamics simulator.

This module is a Brownian-dynamics simulator for systems made of one flexible
chain and one rigid target. It is the sister module to nam_simulator.py.

In each trajectory the chain undergoes three coupled Brownian motions: a
rigid-body translation of the chain center of mass, a rigid-body rotation of
the chain, and a fluctuation of the internal coordinates under bonded and
non-bonded forces.

The internal motion runs on a smaller timestep (dt_chain) than the overall
translation and rotation (dt) because the bonded forces are stiff, so several
internal steps are taken per outer step.

The target molecule is fixed at the origin. The chain starts on a b-sphere at
distance r_start with a random orientation and propagates until it either
reaches a reaction criterion or escapes past r_escape.
"""

from __future__ import annotations
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.motion.do_bd_step import (
    WATER_VISCOSITY,
    backstep_due_to_force,
    bd_step_wiener,
    bd_step_wiener_tensor,
)
from pystarc.simulation.coffdrop_chain import (
    ChainCommon,
    ChainForceEvaluator,
    ChainState,
    compute_chain_forces,
    satisfy_constraints_hybrid,
)
from pystarc.forces.chain_gb import chain_full_gb_force
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.hydrodynamics.rotne_prager import chain_diffusion_tensors
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Atom, Molecule
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import multiprocessing as mp
import numpy as np
import math
import warnings
from pystarc.global_defs.constants import KBT_KCAL
from pystarc.global_defs.defaults import (
    DESOLVATION_ALPHA,
    HYDRODYNAMIC_INTERACTIONS,
    SOLVENT_DIELECTRIC,
    VISCOSITY,
)


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
    it away from target atom k whenever the two centers are closer than σ.

    Ghost atoms (radius < 1e-10) on either side are skipped.

    The implementation is vectorized through numpy broadcasting, which makes it
    roughly 50 to 100 times faster than the original pure-Python double loop.
    For a 38-bead chain against a 4725-atom thrombin target this brings a single
    force evaluation down from about 30 ms to about 0.3 ms, fast enough to run
    inside the 5000-step trajectory loop in production.

    The vectorized form is mathematically equivalent to the looped version. It
    uses identical σ definitions, identical handling of r and the σ/r ratio, and
    identical force accumulation by summation over the target index.
    """
    n_chain = len(chain_radii)
    F = np.zeros((n_chain, 3))
    if target is None:
        return F
    # Build the target arrays, dropping ghost atoms (radius < 1e-10) once at the
    # top rather than branching inside the inner loop.
    target_pos_full = np.array(
        [[a.x, a.y, a.z] for a in target.atoms],
        dtype=float,
    )
    target_rad_full = np.array(
        [a.radius for a in target.atoms],
        dtype=float,
    )
    target_active = target_rad_full >= 1e-10
    if not target_active.any():
        return F
    target_pos = target_pos_full[target_active]  # (n_t_act, 3)
    target_rad = target_rad_full[target_active]  # (n_t_act,)
    # Drop ghost beads on the chain side as well.
    chain_active = chain_radii >= 1e-10
    if not chain_active.any():
        return F
    chain_pos_act = chain_world_positions[chain_active]  # (n_c_act, 3)
    chain_rad_act = chain_radii[chain_active]  # (n_c_act,)
    # Pairwise displacements and distances. The entry dr[i, k, :] is
    # chain_pos_act[i] - target_pos[k], so the force on bead i from target atom k
    # points along +dr, pushing the bead away from the atom.
    dr = chain_pos_act[:, None, :] - target_pos[None, :, :]  # (n_c, n_t, 3)
    r = np.linalg.norm(dr, axis=2)  # (n_c, n_t)
    sig = chain_rad_act[:, None] + target_rad[None, :]  # (n_c, n_t)
    # The WCA cutoff sits at r_min = 2^(1/6) σ, the Lennard-Jones minimum. Beyond
    # r_min the WCA potential and force are both zero. The +ε shift in V makes the
    # potential continuous at r_min, since V_WCA(r_min) = -ε + ε = 0.
    WCA_CUTOFF_FACTOR = 2.0 ** (1.0 / 6.0)
    sig_cutoff = WCA_CUTOFF_FACTOR * sig
    # Active pairs satisfy 0 < r < r_min. Beyond r_min the WCA force is zero.
    active = (r > 1e-10) & (r < sig_cutoff)
    if not active.any():
        return F
    # Compute the WCA force magnitude divided by r for the active pairs only. The
    # np.where calls neutralize the inactive entries so that division by r is safe
    # even where r is below 1e-10, since those positions are masked out and
    # contribute nothing to the final sum.
    r_safe = np.where(active, r, 1.0)
    sr = np.where(active, sig / r_safe, 0.0)
    sr6 = sr**6
    sr12 = sr6 * sr6
    fmag_over_r = np.where(
        active,
        4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / (r_safe * r_safe),
        0.0,
    )  # (n_c, n_t)
    # The per-bead force is the sum over k of fmag_over_r[i, k] * dr[i, k, :].
    # This is the vectorized equivalent of the inner force-accumulation loop in
    # the original implementation.
    F_act = np.einsum("ik,ikd->id", fmag_over_r, dr)  # (n_c_act, 3)

    F[chain_active] = F_act
    return F


def _check_chain_overlap(
    target,
    chain_world_positions: np.ndarray,
    chain_radii: np.ndarray,
    bonded_pairs: set,
) -> bool:
    """Return True if any chain-target or non-bonded chain-chain pair overlaps.

    This mirrors NAM's _check_hard_sphere_overlap, adapted to the chain BD case
    where the chain has several beads and the target is a separate Molecule. Two
    categories of overlap are checked. The first is chain-target overlap, where
    every chain bead is tested against every target atom and an overlap occurs
    when the center separation is smaller than the sum of the two radii. The
    second is chain-chain overlap, where every non-bonded bead pair within the
    chain is tested. Bonded neighbors are allowed to be close because the bond
    constraint sets their separation.

    Ghost atoms (radius < 1e-10) are skipped. They are zero-radius reference
    points, and although the radius-sum check would already exclude them
    geometrically, the explicit skip avoids spurious comparisons.

    Pass target as a Molecule with an .atoms attribute, or None to skip the
    chain-target overlap checks. The argument chain_world_positions is the
    (n_atoms, 3) array of world-frame bead positions, and chain_radii is the
    (n_atoms,) array of bead radii. The argument bonded_pairs is a set holding
    both (i, j) and (j, i) tuples for every bond in the chain, so that set
    membership is independent of ordering.

    The function returns True as soon as it finds the first overlap, and False if
    none is found.
    """
    n = len(chain_radii)
    # First, the chain-target overlaps.
    if target is not None:
        # Stack all target atom positions and radii once, then compute the
        # pairwise distances against the chain in a single numpy call.
        target_pos = np.array(
            [[a.x, a.y, a.z] for a in target.atoms if a.radius >= 1e-10]
        )
        target_rad = np.array([a.radius for a in target.atoms if a.radius >= 1e-10])
        chain_mask = chain_radii >= 1e-10
        if target_pos.shape[0] > 0 and chain_mask.any():
            chain_pos_active = chain_world_positions[chain_mask]
            chain_rad_active = chain_radii[chain_mask]
            # (n_target, n_chain) distance matrix
            diff = target_pos[:, None, :] - chain_pos_active[None, :, :]
            dists = np.sqrt((diff * diff).sum(axis=2))
            # (n_target, n_chain) sum-of-radii threshold
            threshold = target_rad[:, None] + chain_rad_active[None, :]
            if (dists < threshold).any():
                return True
    # Second, the intra-chain non-bonded overlaps.
    for i in range(n):
        if chain_radii[i] < 1e-10:
            continue
        for j in range(i + 1, n):
            if chain_radii[j] < 1e-10:
                continue
            if (i, j) in bonded_pairs:
                continue
            d = float(
                np.linalg.norm(chain_world_positions[i] - chain_world_positions[j])
            )
            if d < chain_radii[i] + chain_radii[j]:
                return True
    return False


def _min_reaction_distance(pathway_set, default: float = 0.0) -> float:
    """Return the smallest contact-cutoff distance across all reactions.

    This value sets the boundary at which the BD step switches from params.dt to
    the smaller params.dt_rxn. The threshold is 1.5 times this return value.

    If the pathway set is None or empty, or no reaction has a contact-cutoff
    pair, the function returns 0.0 by default. The threshold then becomes 0 and
    dt_rxn never fires anywhere, which is the right behavior when there are no
    reactions to resolve. NAM uses 5.0 as its default for the same reason, but at
    b-sphere radii where it is never reached.
    """
    if pathway_set is None:
        return default
    cutoffs = [
        pair.distance_cutoff
        for rxn in pathway_set.reactions
        for pair in rxn.criteria.pairs
    ]
    return min(cutoffs) if cutoffs else default


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
    # Chain-specific.
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
    # Whether to apply hard-sphere overlap rejection. After the outer BD step the
    # code scans for bead-target and bead-bead overlaps, and on an overlap it
    # redraws the Wiener increments and re-steps. It is on by default because it
    # is the only excluded-volume mechanism in the production chain BD path, since
    # the COFFDROP non-bonded forces are not yet wired in here. Without it beads
    # can pass through targets unphysically.
    use_hard_sphere: bool = True
    # Whether to add WCA forces with uniform ε between non-bonded chain bead pairs
    # and between chain beads and target atoms. σ is taken from the radius sums
    # and ε is in kBT. It is off by default because ε = 1 is not physical for
    # arbitrary chains, so it stays opt-in until the COFFDROP per-residue
    # parameters are wired through.
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
        n_pairs = len(rxn.criteria.pairs)
        dists = np.zeros(n_pairs, dtype=float)
        for k, pair in enumerate(rxn.criteria.pairs):
            t_atom = target.atoms[pair.mol1_atom_index]
            t_pos = np.array([t_atom.x, t_atom.y, t_atom.z])
            c_pos = chain_world_positions[pair.mol2_atom_index]
            dists[k] = float(np.linalg.norm(t_pos - c_pos))
        out.append(dists)
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
    the effective diffusion coefficient in Å²/ps for the bridge formula; use
    trace(D)/3 for anisotropic chains. dt is the timestep in ps. rng is a numpy
    random Generator for the reaction-probability gate, consumed only when
    rxn.probability < 1.0.

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

    Ddt = max(D_eff * dt, 1e-30)
    for rxn_idx, rxn in enumerate(pathway_set.reactions):
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
    """
    # Compute the internal forces from the bonded interactions, optionally adding
    # the intra-chain WCA non-bonded forces.
    compute_chain_forces(
        state,
        kT=kT,
        soft_repulsion=soft_repulsion,
        soft_repulsion_eps=soft_repulsion_eps,
    )
    # Apply the per-atom Ermak-McCammon update.
    n_atoms = state.n_atoms
    pos = state.positions
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


def _run_chain_trajectory_worker(args):
    """Run one trajectory as a top-level worker for parallel chain BD execution.

    Multiprocessing requires this function to live at module scope so that it can
    be pickled and shipped to worker processes. Each worker derives its own RNG
    from base_seed + trajectory_index, so the trajectories are independent and
    reproducible.
    """
    sim, traj_idx = args
    base_seed = sim.params.seed if sim.params.seed is not None else 0
    rng = np.random.default_rng(base_seed + traj_idx)
    # Use an independent rng_bb stream for the Brownian bridge sampling, for the
    # same reason as in ChainBDSimulator.__init__. Keeping bridge sampling off the
    # main rng makes trajectory comparisons under bridge_on and bridge_off clean.
    # The offset 0xBB is arbitrary and just needs to be non-zero.
    rng_bb = np.random.default_rng(base_seed + traj_idx + 0xBB)
    return sim.run_one(rng=rng, rng_bb=rng_bb)


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
    ):
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
        # Cache the minimum reaction-cutoff distance so we can use the smaller
        # dt_rxn timestep when the chain is close to a reaction surface. This is
        # constant across the whole run because the pathway set is fixed.
        self._rxn_min = _min_reaction_distance(pathway_set)
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
        # Cache the bonded pairs and bead radii for fast hard-sphere checks. Both
        # orderings are included in the set so that membership lookups are
        # independent of ordering.
        self._chain_radii = np.array(
            [a.radius for a in chain_template.atoms],
            dtype=float,
        )
        self._bonded_pairs = set()
        for bond in chain_template.bonds:
            i, j = bond.a.atom_idx, bond.b.atom_idx
            self._bonded_pairs.add((i, j))
            self._bonded_pairs.add((j, i))
        # Pre-build per-trajectory reusables.
        self._chain_scratch = make_chain_scratch_molecule(chain_template)
        self._chain_charges = np.array(
            [a.charge for a in chain_template.atoms], dtype=float
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
        # should have analytically returned to the b-sphere. This mirrors the
        # NAMSimulator pattern in nam_simulator.py:362-396.
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
            # The OuterPropagator was designed for two diffusing bodies. It uses
            # Stokes-Einstein to derive hydrodynamic radii from Dtrans (line 125,
            # a0 = kT / (6 π μ Dtrans)), so D = 0 would cause a division by zero.
            # As a workaround we split the chain's D evenly between g0 and g1. The
            # propagator's math depends only on D_rel = D_g0 + D_g1, which still
            # equals D_chain, and on the symmetric combination a2 = (a0² + a1²)/2,
            # so the splitting is mathematically equivalent. Drot is only used for
            # the rotational decoherence in new_state, so we split it the same way
            # for symmetry.
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
            vacuum_perm = 1.0 / (4 * math.pi * 332.0636)
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
        if self.params.use_soft_repulsion:
            forces = forces + chain_target_steric_forces(
                world_positions,
                self._chain_radii,
                self.target,
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
                exclude_pair_mask=None,
                obc_kwargs=obc_kwargs,
            )
            forces = forces + f_gb
        return forces

    def run_one(
        self,
        rng: "Optional[np.random.Generator]" = None,
        rng_bb: "Optional[np.random.Generator]" = None,
    ) -> TrajectoryResult:
        """Execute one trajectory and return its outcome.

        The function returns a TrajectoryResult with Fate.REACTED or
        Fate.ESCAPED. If the loop reaches max_steps without a reaction or escape,
        it returns MAX_STEPS with the current radius, because a step-limited
        trajectory committed to neither outcome and counting it as an escape
        would put it in the P_rxn denominator and bias the rate low. If an rng
        is provided, that
        generator is used for this trajectory only, and otherwise self.rng is
        used. The optional argument exists so that parallel workers can supply
        their own seeded generators.
        """
        params = self.params
        if rng is None:
            rng = self.rng
        if rng_bb is None:
            rng_bb = self.rng_bb

        # Per-trajectory diagnostic recording state. It is updated at the top of
        # each outer BD iteration and finalized at the four return sites: a
        # reaction, an escape through the q-sphere, the fallback escape, and the
        # max-steps exit. The output writers consume the populated fields on
        # TrajectoryResult.
        save_interval = self.outputs.save_interval if self.outputs is not None else 0
        min_sep = float("inf")
        pos_at_min = None
        q_at_min = None
        path_steps_buf: List[int] = []
        path_com_buf: List[np.ndarray] = []
        path_q_buf: List[List[float]] = []
        radial_buf: List[float] = []
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
        # times and no RNG state is consumed, so the behavior is bit-exact with
        # the pre-equilibration code path.
        for _ in range(params.n_equilibration_steps):
            chain_internal_bd_step(
                state,
                dt=params.dt_chain,
                rng=rng,
                soft_repulsion=params.use_soft_repulsion,
                soft_repulsion_eps=params.soft_repulsion_eps,
                constraint_tol=params.constraint_tol,
                constraint_max_iter=params.constraint_max_iter,
            )
        # Random starting center of mass and orientation on the b-sphere.
        pos, ori = initialize_bsphere(rng, params.r_start)
        # Cumulative simulated time in ps. It is accumulated each step because the
        # timestep is no longer constant: trajectories that spend more time in the
        # reaction zone advance more slowly in simulated time per step than those
        # in the bulk.
        t_elapsed = 0.0
        # Brownian bridge state, holding the pair distances at the previous
        # iteration's top (before the previous outer step) and the dt used in that
        # step. These are None on iteration 0, since there is no previous step to
        # bridge over.
        prev_top_pair_dists = None
        prev_dt_outer = 0.0
        # Effective D for the bridge formula, taken as trace(D_trans)/3 for an
        # isotropic-equivalent value and constant across the trajectory. This
        # handles both scalar D_trans (the legacy isotropic case) and tensor
        # D_trans (the anisotropic auto_diffusion case). It is set only when the
        # bridge is going to do real work, that is when the pathway set is
        # non-empty.
        D_eff_bb = 0.0
        _has_reactions = (
            self.pathway_set is not None and len(self.pathway_set.reactions) > 0
        )
        if params.use_brownian_bridge and _has_reactions:
            _Dt = np.asarray(self.D_trans)
            if _Dt.ndim == 0:
                # Scalar D_trans, the legacy isotropic case.
                D_eff_bb = float(_Dt)
            elif _Dt.ndim == 2 and _Dt.shape == (3, 3):
                # A 3x3 tensor, the auto_diffusion anisotropic case.
                D_eff_bb = float(np.trace(_Dt) / 3.0)
            else:
                # An unexpected shape, so we disable the bridge to avoid a silent
                # miscalculation. The bridge requires the rigid-body translational
                # diffusion scalar.
                D_eff_bb = 0.0
        for step in range(params.max_steps):
            # Record the per-iteration diagnostics.
            r_curr = float(np.linalg.norm(pos))
            if r_curr < min_sep:
                min_sep = r_curr
                pos_at_min = pos.copy()
                q_at_min = np.array([ori.w, ori.x, ori.y, ori.z])
            if save_interval > 0 and step % save_interval == 0:
                path_steps_buf.append(step)
                path_com_buf.append(pos.copy())
                path_q_buf.append([ori.w, ori.x, ori.y, ori.z])
                radial_buf.append(r_curr)
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
            # Place the chain in the world frame and update the reaction-check
            # scratch molecule.
            world_pos = place_chain(state.positions, pos, ori)
            update_chain_scratch_positions(self._chain_scratch, world_pos)
            # Compute the current pair distances when the bridge is enabled and
            # there are reactions to track. These serve as the "new" distances
            # relative to the previous iteration's outer step.
            if params.use_brownian_bridge and _has_reactions and D_eff_bb > 0:
                cur_top_pair_dists = compute_pair_distances(
                    self.target,
                    world_pos,
                    self.pathway_set,
                )
            else:
                cur_top_pair_dists = None

            # Check for a reaction. With the Brownian bridge, use the bridge-aware
            # check together with the prior pair distances, which applies once
            # step is greater than 0.
            if (
                params.use_brownian_bridge
                and prev_top_pair_dists is not None
                and cur_top_pair_dists is not None
                and prev_dt_outer > 0.0
            ):
                rxn = check_reaction_with_bridge(
                    self.target,
                    self._chain_scratch,
                    self.pathway_set,
                    prev_top_pair_dists,
                    cur_top_pair_dists,
                    D_eff_bb,
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
                _diag = _build_chain_diagnostics(
                    path_steps_buf,
                    path_com_buf,
                    path_q_buf,
                    radial_buf,
                    energy_buf,
                    contact_count_buf,
                )
                _diag["encounter_pos"] = pos.copy()
                _diag["encounter_q"] = np.array([ori.w, ori.x, ori.y, ori.z])
                # Interaction energy at the encounter, sampled here rather than
                # taken from energy_buf, because that buffer is filled only every
                # save_interval steps and a reaction rarely lands on one of them.
                # The steric term is omitted for the same reason as in the buffer,
                # namely that chain_target_steric_forces returns only the force.
                _w_rxn = place_chain(state.positions, pos, ori)
                _e_rxn = 0.0
                if self.target_grid is not None:
                    _f, _e = evaluate_target_grid_force_on_chain(
                        _w_rxn, self._chain_charges, self.target_grid
                    )
                    _e_rxn += _e
                if self.born_grid is not None:
                    _f, _e = evaluate_born_force_on_chain(
                        _w_rxn,
                        self._chain_charges,
                        self.born_grid,
                        alpha=self.desolvation_alpha,
                    )
                    _e_rxn += _e
                _diag["energy_at_reaction"] = _e_rxn
                return TrajectoryResult(
                    Fate.REACTED,
                    step,
                    t_elapsed,
                    float(np.linalg.norm(pos)),
                    rxn,
                    **_diag,
                )
            # LMZ outer propagator. When the chain has drifted far enough from the
            # target (r >= 1.1 b), use the analytical propagator to advance
            # through the b-to-q diffusion zone. The propagator does one of two
            # things. It either returns the chain to the b-sphere (reached_b is
            # True), in which case the BD trajectory continues from a fresh random
            # point on b; this is the LMZ "return" branch, with probability about
            # b/q. Or it terminates the trajectory as truly escaped (reached_b is
            # False), with probability about 1 - b/q. Without this, all
            # trajectories that drift past 1.1 b would be counted as escapes,
            # biasing the reaction probability roughly ten times too low.
            QB_FACTOR = 1.1
            r_now = float(np.linalg.norm(pos))
            if (
                self._outer_prop is not None
                and params.use_lmz
                and r_now >= QB_FACTOR * params.r_start
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
                    _r_curr = float(np.linalg.norm(pos))
                    if _r_curr < min_sep:
                        min_sep = _r_curr
                        pos_at_min = pos.copy()
                        q_at_min = np.array([ori.w, ori.x, ori.y, ori.z])
                    _diag = _build_chain_diagnostics(
                        path_steps_buf,
                        path_com_buf,
                        path_q_buf,
                        radial_buf,
                        energy_buf,
                        contact_count_buf,
                    )
                    if pos_at_min is not None:
                        _diag["near_miss_pos"] = pos_at_min
                        _diag["near_miss_dist"] = min_sep
                    return TrajectoryResult(
                        Fate.ESCAPED,
                        step,
                        t_elapsed,
                        float(np.linalg.norm(pos)),
                        **_diag,
                    )
                # Otherwise we fall through. The trajectory continues from the new
                # returned position on the b-sphere, and the next outer BD step
                # starts from there.
            # Fallback escape check. This is used when _outer_prop is None, for
            # instance when its construction failed, or when use_lmz=False is
            # explicitly requested. Without LMZ this fires whenever the chain
            # drifts past r_escape and biases the reaction probability low.
            _lmz_active = self._outer_prop is not None and params.use_lmz
            if not _lmz_active and check_escape(pos, params.r_escape):
                _r_curr = float(np.linalg.norm(pos))
                if _r_curr < min_sep:
                    min_sep = _r_curr
                    pos_at_min = pos.copy()
                    q_at_min = np.array([ori.w, ori.x, ori.y, ori.z])
                _diag = _build_chain_diagnostics(
                    path_steps_buf,
                    path_com_buf,
                    path_q_buf,
                    radial_buf,
                    energy_buf,
                    contact_count_buf,
                )
                if pos_at_min is not None:
                    _diag["near_miss_pos"] = pos_at_min
                    _diag["near_miss_dist"] = min_sep
                return TrajectoryResult(
                    Fate.ESCAPED,
                    step,
                    t_elapsed,
                    float(np.linalg.norm(pos)),
                    **_diag,
                )
            # Take the inner steps, which run internal-coordinate BD on the
            # body-frame positions. The inner dt is independent of the outer
            # adaptive scheme.
            for _ in range(params.chain_steps_per_outer):
                chain_internal_bd_step(
                    state,
                    dt=params.dt_chain,
                    rng=rng,
                    constraint_tol=params.constraint_tol,
                    constraint_max_iter=params.constraint_max_iter,
                    soft_repulsion=params.use_soft_repulsion,
                    soft_repulsion_eps=params.soft_repulsion_eps,
                )
            # Take the outer step: refresh the world positions, get the external
            # forces, and propagate the rigid-body coordinates.
            world_pos = place_chain(state.positions, pos, ori)
            per_atom_forces = self._compute_per_atom_external_forces(world_pos)
            F_net_old = per_atom_forces.sum(axis=0)
            # Two-zone adaptive timestep, using the smaller dt when close to a
            # reaction surface and the larger dt in the bulk.
            r = float(np.linalg.norm(pos))
            dt_outer = params.dt_rxn if r < 1.5 * self._rxn_min else params.dt
            # Pre-draw the Wiener increments so we can reuse them under
            # subdivision.
            dW_t = math.sqrt(dt_outer) * rng.standard_normal(3)
            dW_r = math.sqrt(dt_outer) * rng.standard_normal(3)

            pos_old, ori_old = pos, ori
            pos, ori = chain_outer_bd_step_wiener(
                pos_old,
                ori_old,
                world_pos,
                per_atom_forces,
                self.D_trans,
                self.D_rot,
                dt_outer,
                dW_t,
                dW_r,
            )
            # Force-change backstep. If the external force changes too much across
            # the step, subdivide it through the Brownian bridge midpoint
            # construction. This is skipped in the dt_rxn floor zone, because
            # subdividing below dt_rxn would step finer than the adaptive scheme
            # intends, and it is skipped when the parameter flag is off as a
            # backward-compatible opt-out.
            if params.force_change_backstep and dt_outer > params.dt_rxn:
                world_pos_trial = place_chain(state.positions, pos, ori)
                per_atom_forces_trial = self._compute_per_atom_external_forces(
                    world_pos_trial
                )
                F_net_new = per_atom_forces_trial.sum(axis=0)

                if backstep_due_to_force(
                    F_net_new,
                    F_net_old,
                    pos,
                    pos_old,
                    dt_outer,
                    params.dt_rxn,
                    radius=self._effective_hydro_radius,
                ):
                    # Brownian bridge midpoint, constructed so that dW_mid + dW_2nd
                    # = dW while preserving the half-step noise statistics.
                    s = math.sqrt(dt_outer / 4.0)
                    dW_mid_t = 0.5 * dW_t + s * rng.standard_normal(3)
                    dW_mid_r = 0.5 * dW_r + s * rng.standard_normal(3)
                    hdt = dt_outer / 2.0
                    # First half-step, taken from pos_old and ori_old with the
                    # original old-position forces.
                    pos, ori = chain_outer_bd_step_wiener(
                        pos_old,
                        ori_old,
                        world_pos,
                        per_atom_forces,
                        self.D_trans,
                        self.D_rot,
                        hdt,
                        dW_mid_t,
                        dW_mid_r,
                    )
                    # Recompute the forces at the midpoint configuration.
                    world_pos_mid = place_chain(state.positions, pos, ori)
                    per_atom_forces_mid = self._compute_per_atom_external_forces(
                        world_pos_mid
                    )
                    # Second half-step, taken from the midpoint with the midpoint
                    # forces.
                    dW_2nd_t = dW_t - dW_mid_t
                    dW_2nd_r = dW_r - dW_mid_r
                    pos, ori = chain_outer_bd_step_wiener(
                        pos,
                        ori,
                        world_pos_mid,
                        per_atom_forces_mid,
                        self.D_trans,
                        self.D_rot,
                        hdt,
                        dW_2nd_t,
                        dW_2nd_r,
                    )
            # Hard-sphere overlap rejection. If the trial position produced an
            # overlap, either between any chain bead and any target atom or between
            # any non-bonded chain bead pair, redraw the Wiener increments and
            # re-step from the same starting point with the original forces. The
            # number of attempts is capped at MAX_HS_ATTEMPTS to avoid an infinite
            # loop when the chain is wedged near the target, and if all attempts
            # overlap, the last one is accepted with a warning. The other forces,
            # namely electrostatics, soft repulsion, and constraints, typically
            # resolve transient overlaps over the next few BD steps.
            if params.use_hard_sphere:
                MAX_HS_ATTEMPTS = 3
                world_pos_check = place_chain(state.positions, pos, ori)
                _hs_attempts = 0
                while _check_chain_overlap(
                    self.target,
                    world_pos_check,
                    self._chain_radii,
                    self._bonded_pairs,
                ):
                    _hs_attempts += 1
                    if _hs_attempts > MAX_HS_ATTEMPTS:
                        warnings.warn(
                            f"hard-sphere overlap rejection exceeded "
                            f"{MAX_HS_ATTEMPTS} attempts; accepting overlapped "
                            f"step (chain may be wedged near target)",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break
                    dW_t2 = math.sqrt(dt_outer) * rng.standard_normal(3)
                    dW_r2 = math.sqrt(dt_outer) * rng.standard_normal(3)
                    pos, ori = chain_outer_bd_step_wiener(
                        pos_old,
                        ori_old,
                        world_pos,
                        per_atom_forces,
                        self.D_trans,
                        self.D_rot,
                        dt_outer,
                        dW_t2,
                        dW_r2,
                    )
                    world_pos_check = place_chain(state.positions, pos, ori)
            t_elapsed += dt_outer
            # Save the state for the next iteration's Brownian bridge check. The
            # value cur_top_pair_dists was captured before this outer step. At the
            # top of the next iteration we will compute fresh distances and use
            # those two snapshots together with prev_dt_outer = dt_outer.
            if params.use_brownian_bridge:
                prev_top_pair_dists = cur_top_pair_dists
                prev_dt_outer = dt_outer
        # max_steps was exhausted without a reaction or a formal escape, so record
        # the trajectory as ESCAPED with the current radius and accumulated time.
        _r_curr = float(np.linalg.norm(pos))
        if _r_curr < min_sep:
            min_sep = _r_curr
            pos_at_min = pos.copy()
            q_at_min = np.array([ori.w, ori.x, ori.y, ori.z])
        _diag = _build_chain_diagnostics(
            path_steps_buf,
            path_com_buf,
            path_q_buf,
            radial_buf,
            energy_buf,
            contact_count_buf,
        )
        if pos_at_min is not None:
            _diag["near_miss_pos"] = pos_at_min
            _diag["near_miss_dist"] = min_sep
        # Running out of steps is neither a reaction nor an escape. Calling it an
        # escape puts it in the P_rxn denominator and biases the rate downward.
        return TrajectoryResult(
            Fate.MAX_STEPS,
            params.max_steps,
            t_elapsed,
            float(np.linalg.norm(pos)),
            **_diag,
        )

    def run(self) -> List[TrajectoryResult]:
        """Execute n_trajectories trajectories, either serially or in parallel.

        This dispatches to a parallel multiprocessing pool when both
        params.n_threads > 1 and params.n_trajectories > 1, and otherwise it runs
        serially. It returns the accumulated self.results list.

        In serial mode all trajectories share self.rng, which is one continuous
        random stream. In parallel mode each trajectory gets its own RNG seeded
        from base_seed + trajectory_index. The results are therefore
        deterministic within each mode's own seeding scheme, but the two modes do
        not produce the same trajectories.
        """
        n = self.params.n_trajectories
        if self.params.n_threads > 1 and n > 1:
            self._run_parallel(n)
        else:
            self._run_serial(n)
        return self.results

    def _record(self, result: TrajectoryResult) -> None:
        """Append a result and update reacted/escaped counters."""
        self.results.append(result)
        if result.fate == Fate.REACTED:
            self.n_reacted += 1
        elif result.fate == Fate.ESCAPED:
            self.n_escaped += 1
        elif result.fate == Fate.MAX_STEPS:
            self.n_max_steps += 1

    def _run_serial(self, n: int) -> None:
        """Run n trajectories sequentially, sharing self.rng."""
        for i in range(n):
            if self.params.verbose:
                print(
                    f"  Trajectory {i + 1}/{n}  "
                    f"(reacted={self.n_reacted}, escaped={self.n_escaped})"
                )
            self._record(self.run_one())

    def _run_parallel(self, n: int) -> None:
        """Run n trajectories across a multiprocessing pool."""
        n_workers = min(self.params.n_threads, n, mp.cpu_count())
        if self.params.verbose:
            print(f"  Parallel: {n_workers} workers, {n} trajectories")
        args = [(self, i) for i in range(n)]
        # Use imap rather than map so that results come back as they complete,
        # which enables incremental progress reporting. imap preserves the
        # submission order, so self._record() sees the results in the same order
        # as pool.map() would deliver them.
        report_every = max(1, n // 20)  # report progress every 5 percent
        with mp.Pool(n_workers) as pool:
            for i, result in enumerate(
                pool.imap(_run_chain_trajectory_worker, args), 1
            ):
                self._record(result)
                if self.params.verbose and (i % report_every == 0 or i == n):
                    print(
                        f"  Progress: {i}/{n} trajectories " f"({100*i/n:.0f}%)",
                        flush=True,
                    )
