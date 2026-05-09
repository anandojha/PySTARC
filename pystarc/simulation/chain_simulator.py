"""
PySTARC chain BD simulator

Brownian dynamics simulator for systems with one flexible chain and one
rigid target. Sister module to nam_simulator.py.

Per trajectory the chain undergoes three coupled motions:
- rigid-body translation of the chain center of mass (Brownian)
- rigid-body rotation of the chain (Brownian)
- internal coordinate fluctuation (Brownian, under bonded + non-bonded forces)

The internal motion runs on a smaller timestep (dt_chain) than the
overall translation/rotation (dt) because bonded forces are stiff.
Multiple internal steps are taken per outer step.

Target molecule is fixed at the origin.The chain is initialized on a
b-sphere at distance r_start, with a random orientation, and propagates
until it either reaches a reaction criterion or escapes past r_escape.

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


def chain_target_steric_forces(
    chain_world_positions: np.ndarray,
    chain_radii: np.ndarray,
    target,
    eps: float = 1.0,
) -> np.ndarray:
    """WCA repulsion between every chain bead and every target atom.

    Returns (n_chain, 3) per-bead force array in kBT/A. Sigma for each
    pair is the sum of bead and atom radii; epsilon is uniform.

    WCA potential and force same as chain_intra_nonbonded_forces.
    Force on chain bead i pushes it away from target atom k when the
    centers are closer than sig.

    Ghost atoms (radius < 1e-10) on either side are skipped.

    Vectorized via numpy broadcasting (50-100x faster than the original
    pure-Python double loop). For a 38-bead chain against a 4725-atom
    thrombin target, this drops a single force evaluation from ~30 ms
    to ~0.3 ms, making this production-viable inside the 5000-step
    trajectory loop.

    The vectorized implementation is mathematically equivalent to the
    looped version: identical sigma definitions, identical r and sr
    handling, identical force accumulation via summation over the
    target index.
    """
    n_chain = len(chain_radii)
    F = np.zeros((n_chain, 3))
    if target is None:
        return F
    # Build target arrays, dropping ghost atoms (radius < 1e-10) once
    # at the top instead of branching inside the inner loop.
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
    # Pairwise displacements and distances.
    # dr[i, k, :] = chain_pos_act[i] - target_pos[k]. Force on bead i
    # from target atom k points along +dr (pushes bead AWAY from atom).
    dr = chain_pos_act[:, None, :] - target_pos[None, :, :]  # (n_c, n_t, 3)
    r = np.linalg.norm(dr, axis=2)  # (n_c, n_t)
    sig = chain_rad_act[:, None] + target_rad[None, :]  # (n_c, n_t)
    # WCA cutoff is at r_min = 2^(1/6) * sigma (the LJ minimum). Beyond
    # r_min, V_WCA = 0 and the force is zero. The +eps shift in V makes
    # the potential continuous at r_min (V_WCA(r_min) = -eps + eps = 0).
    WCA_CUTOFF_FACTOR = 2.0 ** (1.0 / 6.0)
    sig_cutoff = WCA_CUTOFF_FACTOR * sig
    # Active pairs: 0 < r < r_min. Beyond r_min, WCA force is zero.
    active = (r > 1e-10) & (r < sig_cutoff)
    if not active.any():
        return F
    # Compute the WCA force magnitude / r for active pairs only. We use
    # np.where to neutralize inactive entries so division by r is safe
    # even where r is "below 1e-10" (those positions are masked out
    # anyway and contribute nothing to the final sum).
    r_safe = np.where(active, r, 1.0)
    sr = np.where(active, sig / r_safe, 0.0)
    sr6 = sr**6
    sr12 = sr6 * sr6
    fmag_over_r = np.where(
        active,
        4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / (r_safe * r_safe),
        0.0,
    )  # (n_c, n_t)
    # Per-bead force = sum_k fmag_over_r[i, k] * dr[i, k, :].
    # This is the vectorized equivalent of the inner force-accumulation
    # loop in the original implementation.
    F_act = np.einsum("ik,ikd->id", fmag_over_r, dr)  # (n_c_act, 3)

    F[chain_active] = F_act
    return F


def _check_chain_overlap(
    target,
    chain_world_positions: np.ndarray,
    chain_radii: np.ndarray,
    bonded_pairs: set,
) -> bool:
    """Return True if any chain-target or non-bonded chain-chain pair overlap.

    Mirrors NAM's _check_hard_sphere_overlap but for the chain BD case
    where the chain has multiple beads and the target is a separate
    Molecule. Two overlap categories are checked:

    1. Chain-target: every chain bead vs every target atom. An overlap
       is when |r_bead - r_atom| < r_bead + r_atom.
    2. Chain-chain: every non-bonded bead pair within the chain.
       Bonded neighbors are allowed to be close (the bond constraint
       determines their separation).

    Ghost atoms (radius < 1e-10) are skipped: they are zero-radius
    reference points and the radius-sum check correctly excludes them
    geometrically, but the explicit skip avoids noise.

    Parameters
    ----------
    target               : Molecule with .atoms; pass None to skip
                           chain-target overlap checks.
    chain_world_positions : (n_atoms, 3) world-frame chain bead positions.
    chain_radii          : (n_atoms,) radii for each chain bead.
    bonded_pairs         : set of (i, j) and (j, i) tuples for every
                           bond in the chain. Both orderings included
                           so set membership is order-independent.

    Returns
    -------
    True on first overlap found; False if none.
    """
    n = len(chain_radii)
    # 1. Chain-target overlaps.
    if target is not None:
        # Vectorized: stack all target atom positions and radii once,
        # then pairwise-distance against chain in one numpy call.
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
    # 2. Intra-chain non-bonded overlaps.
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

    Used to set the boundary at which the BD step switches from
    params.dt to the smaller params.dt_rxn: the threshold is
    1.5 * (this return value).

    If the pathway set is None or empty, or no reaction has a
    contact-cutoff pair, returns 0.0 by default. The threshold then
    becomes 0, and dt_rxn never fires anywhere, i.e.,  which is the right
    behavior when there are no reactions to resolve. NAM uses 5.0 as
    a default for the same reason but at b-sphere radii where it is
    never reached.
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

    The first block mirrors NAMParameters: trajectory count, integration
    timestep, b-sphere radii, RNG seed, threading, etc. The second block
    is chain-specific: an inner timestep dt_chain for internal coordinate
    integration and a constraint solver tolerance.
    """

    n_trajectories: int = 1_000
    dt: float = 0.2
    dt_rxn: float = 0.05
    minimum_core_dt: float = 0.0
    minimum_core_reaction_dt: float = 0.0
    max_steps: int = 1_000_000
    r_start: float = 100.0
    r_escape: float = 0.0
    # Debye screening length in A. Default 7.858 A matches ~150 mM NaCl
    # at 298 K (consistent with PySTARCConfig.debye_length). Only used by
    # the LMZ outer propagator; chain-internal GB / COFFDROP forces do
    # not see this parameter.
    debye_length: float = 7.858
    # use_lmz: when True (default, correct), the outer propagator
    # (LMZ) handles the b->q diffusion zone analytically, returning
    # trajectories to the b-sphere with probability b/q. When False,
    # falls back to simple-escape behavior (terminate at r_escape).
    # The simple-escape behavior is biased (P_rxn ~10x too low) but
    # is what older tests were calibrated against.
    use_lmz: bool = True
    seed: Optional[int] = None
    n_threads: int = 1
    verbose: bool = False
    # Chain-specific.
    dt_chain: float = 0.05
    chain_steps_per_outer: int = 4
    # Pre-equilibration: number of internal-coordinate-only BD steps to
    # run in bulk (no external forces) before each trajectory's main BD
    # loop. Default 0 -> no equilibration (bit-exact with prior path,
    # no RNG state consumed).
    n_equilibration_steps: int = 0
    constraint_tol: float = 1e-6
    constraint_max_iter: int = 200
    # Force-change backstep: subdivide an outer BD step when the external
    # force changes too much across the step (NAM's criterion in
    # motion/do_bd_step.py:backstep_due_to_force). Default ON because
    # without it, simulations crossing steep electrostatic gradients
    # produce systematic kinetic errors. Costs an extra force evaluation
    # per outer step in the worst case.
    force_change_backstep: bool = True
    # Hard-sphere overlap rejection: after the outer BD step, scan for
    # bead-target and bead-bead overlaps. On overlap, redraw the
    # Wiener increments and re-step. Default ON because this is the
    # only excluded-volume mechanism in the production chain BD path
    # (COFFDROP non-bonded forces are not yet wired in here); without
    # it beads can pass through targets unphysically.
    use_hard_sphere: bool = True
    # Soft repulsion: if True, add WCA forces (uniform eps) between
    # non-bonded chain bead pairs and between chain beads and target
    # atoms. Uses radius sums for sigma; eps in kBT. Default OFF
    # because eps=1 is not physical for arbitrary chains; opt-in
    # until COFFDROP per-residue parameters are wired through.
    use_soft_repulsion: bool = False
    soft_repulsion_eps: float = 1.0
    # Brownian bridge for reaction detection: when True, in addition
    # to the endpoint check, also evaluate the path-crossing probability
    # for every contact pair whose pre- and post-step distances are
    # both above its cutoff. This catches reactions that occur between
    # discrete BD steps - a real correctness fix because endpoint-only
    # checks systematically under-count fast reactions.
    # Default ON because it closes a method gap. With empty PathwaySet
    # the code path is trivially skipped; existing tests unaffected.
    use_brownian_bridge: bool = True
    # GB self-Born / generalized Born electrostatics on chain. Default OFF
    # preserves bit-exact prior behavior. When True, the chain BD adds
    # full GB forces (diagonal self-Born + off-diagonal cross-term +
    # vacuum Coulomb between non-bonded chain pairs) on top of the
    # existing external forces. Path B dispatch via coffdrop_active:
    # when COFFDROP pair tables are active, only the diagonal self-Born
    # is added so the off-diagonal does not double-count COFFDROP's
    # pre-screened pair potentials. Effective radii via OBC2 set-II
    # (Onufriev, Bashford, Case 2004).
    use_self_born: bool = False
    # Interior / exterior dielectrics for GB. Defaults are vacuum
    # interior and water at 300 K exterior; tune per system if needed.
    gb_eps_in: float = 1.0
    gb_eps_out: float = 78.5
    # OBC2 set-II coefficients. Defaults are the published set-II
    # values; do not change unless calibrating an alternative
    # parameterization.
    gb_obc_alpha: float = 1.0
    gb_obc_beta: float = 0.8
    gb_obc_gamma: float = 4.85
    # Path B switch: True when COFFDROP pair tables are loaded for the
    # chain, in which case GB is restricted to diagonal-only to avoid
    # double-counting empirical pre-screened electrostatics. Default
    # False matches current production (no COFFDROP pair forces in the
    # rigid-body benchmark path).
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

    The chain stores its atoms in a body frame whose origin is the chain
    center of mass and whose orientation defines internal geometry. This
    function applies a rigid-body transformation to produce the world-
    frame positions used for force evaluation against the target and for
    reaction-criterion checks.

    Parameters
    ----------
    body_positions : (n_atoms, 3) array of body-frame atom positions.
        These should already be centered: their mean across atoms should
        be at the origin of the body frame.
    com : length-3 array, the world-frame center of mass.
    orientation : quaternion describing chain orientation in the world frame.

    Returns
    -------
    (n_atoms, 3) array of world-frame atom positions.
    """
    R = orientation.to_rotation_matrix()
    return (R @ body_positions.T).T + com


def initialize_bsphere(
    rng: np.random.Generator,
    r_start: float,
) -> Tuple[np.ndarray, Quaternion]:
    """Sample a uniformly-random starting (position, orientation) on the b-sphere.

    The position is a vector of magnitude r_start pointing in a random
    direction (uniformly distributed over the unit sphere). The
    orientation is a uniform-random unit quaternion. Together these
    define the initial state of the chain at the start of a trajectory.

    Parameters
    ----------
    rng     : numpy random Generator. Pass an explicit RNG for reproducibility.
    r_start : b-sphere radius (Angstroms).

    Returns
    -------
    pos : (3,) array, the chain center-of-mass position.
    ori : Quaternion, the chain orientation.
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
    """Per-reaction contact-pair distances for Brownian bridge tracking.

    Returns a list parallel to pathway_set.reactions; each element is a
    1D array of distances, one per ContactPair in that reaction. The
    distance for pair (i_target, j_chain) is

        |target.atoms[i_target].position - chain_world_positions[j_chain]|

    If pathway_set is None or has no reactions, returns an empty list.
    Used by check_reaction_with_bridge to capture pre-step
    distances; the same function is called post-step to compute new
    distances for the bridge evaluation.

    Parameters
    ----------
    target                : Molecule with .atoms; pass None to short-circuit.
    chain_world_positions : (n_chain, 3) world-frame positions of chain beads.
    pathway_set           : PathwaySet whose reactions we want to track.

    Returns
    -------
    List of (n_pairs_in_rxn,) distance arrays, one per reaction.
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
    """Brownian-bridge-aware reaction check.

    For each reaction, computes per-pair "fired" flags using both the
    endpoint distance and the Brownian bridge crossing probability:

        p_cross = exp(-x0 * x1 / (D_eff * dt))    when x0 > 0 and x1 > 0

    where x = pair_distance - cutoff is the signed height above the
    reaction surface. A pair is "fired" if either:
      (a) endpoint distance is below cutoff (x1 < 0), OR
      (b) bridge sample u < p_cross (path crossed during the step).

    A reaction fires when the count of fired pairs is at least
    n_needed for that reaction (matching pathway_set.check_all's
    semantics for AND/OR logic; n_needed = -1 means ALL).

    Returns the name of the first reaction that fires, or None.

    If old/new distance lists are empty (no reactions), or if
    pre- and post-step lists have inconsistent length, falls back to
    endpoint-only check via check_chain_reaction.

    Parameters
    ----------
    target, chain_scratch, pathway_set : same as check_chain_reaction.
    old_pair_dists_per_rxn  : pre-step distances per reaction (from
                              compute_chain_pair_distances).
    new_pair_dists_per_rxn  : post-step distances per reaction.
    D_eff                   : effective diffusion coefficient (A^2/ps)
                              for the bridge formula. Use trace(D)/3
                              for anisotropic chains.
    dt                      : timestep (ps).
    rng                     : numpy random Generator for the reaction
                              probability gate (consumed only when
                              rxn.probability < 1.0).
    rng_bb                  : optional independent Generator for bridge
                              samples (one uniform per pair where
                              both x0, x1 > 0). When None (default),
                              bridge samples come from rng; this
                              preserves backward-compatibility but
                              causes the main rng stream to advance
                              differently between bridge_on and
                              bridge_off runs, breaking strict
                              monotonicity comparisons. Passing an
                              independent rng_bb makes the bridge a
                              pure-additive operator on the trajectory.

    Returns
    -------
    Reaction name (str) on first firing reaction, else None.
    """
    # Fall back to endpoint-only if we don't have matching distance data.
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

        # Endpoint-fired: pair is below cutoff at the new position.
        endpoint_fired = x1 < 0
        # Bridge-fired: both x0 and x1 above zero, sample says crossed.
        both_above = (x0 > 0) & (x1 > 0)
        p_cross = np.zeros_like(x0)
        np.subtract(0.0, x0 * x1 / Ddt, out=p_cross, where=both_above)
        np.exp(p_cross, out=p_cross, where=both_above)
        # Where not both_above, p_cross is 0 (initialised to zero, no-op).
        p_cross = np.where(both_above, p_cross, 0.0)
        # Bridge sample: use rng_bb if provided (independent stream
        # makes bridge additive on the trajectory; main rng untouched
        # by bridge sampling). Falls back to rng if rng_bb is None.
        bb_rng = rng_bb if rng_bb is not None else rng
        u = bb_rng.random(size=p_cross.shape)
        bridge_fired = u < p_cross
        pair_fired = endpoint_fired | bridge_fired
        # Apply n_needed logic: -1 means ALL pairs must fire.
        n_needed = rxn.criteria.n_needed
        threshold = len(rxn.criteria.pairs) if n_needed < 0 else n_needed
        if threshold == 0:
            # Degenerate: a reaction with no pair requirement would
            # always fire. Guard against silent triggering.
            continue
        n_fired = int(pair_fired.sum())
        if n_fired >= threshold:
            # Reaction probability gating (mirror check_all semantics).
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
    """Evaluate per-atom electrostatic force and total energy on the chain
    from a precomputed Poisson-Boltzmann potential grid of the target.

    The grid is built once (typically by APBS) and represents the
    target's electrostatic potential as a 3D array of values in kBT/e.
    Each chain atom feels a force F_i = -q_i grad(phi)(r_i) and
    contributes q_i * phi(r_i) to the total interaction energy. Both are
    computed by trilinear interpolation on the grid.

    For chain atoms whose positions fall outside the grid box, the grid
    routines return zero (no electrostatic interaction), so atoms far
    from the target contribute nothing.

    Parameters
    ----------
    chain_world_positions : (n_atoms, 3) array of world-frame positions [A].
    chain_charges         : (n_atoms,) array of partial charges [e].
    grid                  : DXGrid containing the target's PB potential.

    Returns
    -------
    forces : (n_atoms, 3) array of forces in kBT/A on each chain atom.
    energy : total interaction energy in kBT.
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


DEFAULT_DESOLVATION_ALPHA = 0.07957747


def evaluate_born_force_on_chain(
    chain_world_positions: np.ndarray,
    chain_charges: np.ndarray,
    grid: DXGrid,
    alpha: float = DEFAULT_DESOLVATION_ALPHA,
) -> Tuple[np.ndarray, float]:
    """Evaluate per-atom Born desolvation force and total energy on the chain
    from a precomputed Born desolvation potential grid of the target.

    The Born grid g(r) is built once and represents the
    target's desolvation field as a 3D array. Each chain atom feels a
    self-energy contribution

        V_i  =  alpha * q_i^2 * g(r_i)
        F_i  = -alpha * q_i^2 * grad(g)(r_i)

    matching the convention used in the rigid-body engine (engine.py).
    The prefactor alpha defaults to 1/(4*pi) ~ 0.07957747; the grid is
    expected to be in raw APBS units of kBT / e^2 (per unit charge squared)
    scaled by 1/alpha. Both force and energy are computed by trilinear
    interpolation on the grid.

    For chain atoms whose positions fall outside the grid box, the grid
    routines return zero (no Born contribution), so atoms far from the
    target see no desolvation cost.

    Parameters
    ----------
    chain_world_positions : (n_atoms, 3) array of world-frame positions [A].
    chain_charges         : (n_atoms,) array of partial charges [e].
    grid                  : DXGrid containing the target's Born potential.
    alpha                 : Desolvation prefactor in kBT/(e^2 * grid_unit).
                            Defaults to DEFAULT_DESOLVATION_ALPHA. Make this
                            configurable so chain-BD Born can be compared
                            head-to-head with the rigid-body engine for the
                            same system.

    Returns
    -------
    forces : (n_atoms, 3) array of forces in kBT/A on each chain atom.
    energy : total Born self-energy in kBT.
    """
    if chain_world_positions.shape[0] != chain_charges.shape[0]:
        raise ValueError(
            f"position count {chain_world_positions.shape[0]} does not "
            f"match charge count {chain_charges.shape[0]}"
        )
    # Per-atom gradient of the Born field. Shape (n_atoms, 3).
    grads = grid.batch_gradient(chain_world_positions)
    # Per-atom interpolated Born potential. Shape (n_atoms,).
    potentials = grid.batch_interpolate(chain_world_positions)
    # F_i = -alpha * q_i^2 * grad(g)(r_i). Per-atom coefficient is -alpha*q^2.
    coeffs = -alpha * chain_charges * chain_charges  # (n_atoms,)
    forces = coeffs[:, None] * grads  # (n_atoms, 3)
    # V = alpha * sum_i q_i^2 * g(r_i).
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
    """One Brownian-dynamics step on the chain's internal coordinates.

    Implements the Ermak-McCammon update per atom with independent
    Stokes-Einstein mobilities (no hydrodynamic coupling between chain
    atoms). For each atom i with radius a_i:

        mob_i = 1 / (6 pi eta a_i)
        Delta r_i = mob_i * F_i * dt + sqrt(2 kT mob_i dt) * xi_i

    where F_i is the bonded force on atom i and xi_i is a unit Gaussian
    3-vector. After all atoms are updated, the chain is re-centered at
    the origin (so the body frame stays valid) and constraints are
    satisfied via the hybrid SHAKE/Newton solver.

    Parameters
    ----------
    state              : ChainState whose positions are advanced in place.
    dt                 : Integration timestep in ps.
    rng                : numpy random Generator for Wiener increments.
    kT                 : Thermal energy. PySTARC uses kT = 1 throughout;
                         the parameter is exposed mainly for unit tests
                         that need to vary it.
    viscosity          : Solvent viscosity in kBT.ps/A^3. Defaults to the
                         water value used elsewhere in the package.
    apply_constraints  : If True (default), call the hybrid constraint
                         solver after the integration step. Set to False
                         only for diagnostic tests of the integrator.
    constraint_tol     : Convergence tolerance for the constraint solver.
    constraint_max_iter: Hard cap on solver iterations.
    """
    # Compute internal forces from bonded interactions, optionally
    # plus intra-chain WCA non-bonded forces.
    compute_chain_forces(
        state,
        kT=kT,
        soft_repulsion=soft_repulsion,
        soft_repulsion_eps=soft_repulsion_eps,
    )
    # Per-atom Ermak-McCammon update.
    n_atoms = state.n_atoms
    pos = state.positions
    forces = state.forces
    radii = np.array([a.radius for a in state.common.atoms])
    mobilities = 1.0 / (6.0 * math.pi * viscosity * radii)
    drifts = mobilities[:, None] * forces * dt
    sigmas = np.sqrt(2.0 * kT * mobilities * dt)
    noise = sigmas[:, None] * rng.standard_normal((n_atoms, 3))
    pos += drifts + noise
    # Re-center at origin: keeps the body frame consistent with
    # place_chain (which assumes body positions are CoM-centered).
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

    The net force is just the vector sum of all per-atom forces. The net
    torque about the chain center of mass is the sum of (r_i - com) x F_i
    for all atoms. These two quantities drive the rigid-body translation
    and rotation of the chain, respectively, in the outer Brownian-
    dynamics step.

    Parameters
    ----------
    chain_world_positions : (n_atoms, 3) atom positions in the world frame.
    per_atom_forces       : (n_atoms, 3) external force on each atom.
    com                   : (3,) chain center of mass in the world frame.

    Returns
    -------
    force_net  : (3,) net force on the chain.
    torque_net : (3,) net torque about com.
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
    """One Brownian-dynamics step on the chain rigid-body coordinates.

    Aggregates per-atom external forces into a net force and torque on
    the chain center of mass, then advances (pos, ori) by one timestep
    dt using either the scalar or the tensor BD step depending on the
    type of D_trans and D_rot.

    Parameters
    ----------
    pos                   : (3,) chain center-of-mass position before the step.
    ori                   : Quaternion, chain orientation before the step.
    chain_world_positions : (n_atoms, 3) world-frame atom positions used
                            for torque-arm calculation.
    per_atom_forces       : (n_atoms, 3) external forces on each atom.
    D_trans               : translational diffusion coefficient. Either a
                            scalar (isotropic, A^2/ps) or a (3, 3) tensor
                            (anisotropic).
    D_rot                 : rotational diffusion coefficient. Either a
                            scalar (rad^2/ps) or a (3, 3) tensor.
    dt                    : timestep in ps.
    rng                   : numpy random Generator for Wiener increments.

    Returns
    -------
    new_pos : (3,) updated chain CoM position.
    new_ori : Quaternion, updated chain orientation.

    Notes
    -----
    Both D_trans and D_rot must be the same kind (both scalar or both
    tensor). Mixed types are not currently supported because the
    rigid-body BD step combines them in a single call to either
    bd_step_wiener or bd_step_wiener_tensor.
    """
    force_net, torque_net = aggregate_chain_external_force_and_torque(
        chain_world_positions,
        per_atom_forces,
        pos,
    )
    dW_t = math.sqrt(dt) * rng.standard_normal(3)
    dW_r = math.sqrt(dt) * rng.standard_normal(3)
    # Detect tensor-vs-scalar by the ndim of the input. A scalar float
    # has no ndim attribute (use np.ndim which handles both); a
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
    """Sibling of chain_outer_bd_step using pre-drawn Wiener increments.

    Same physics as chain_outer_bd_step (aggregates per-atom forces, then
    advances pos/ori by dt with either scalar or tensor BD step), but
    accepts the Wiener increments as inputs rather than drawing them
    internally. Required for force-change backstep subdivision: the
    caller must hold onto the original Wiener so it can build a midpoint
    pair (dW_mid, dW - dW_mid) that preserves Brownian bridge statistics.

    The dW arrays must already be scaled by sqrt(dt). At the simulator
    level this is the same shape NAM uses for its scalar bd_step_wiener.

    Parameters
    ----------
    pos, ori, chain_world_positions, per_atom_forces, D_trans, D_rot, dt
        Same as chain_outer_bd_step.
    dW_t, dW_r
        (3,) Wiener increments scaled by sqrt(dt). Caller's responsibility.
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
    return new_pos, new_ori


def _run_chain_trajectory_worker(args):
    """Top-level worker for parallel chain BD execution.

    Multiprocessing requires this function to be at module scope so it
    can be pickled and shipped to worker processes. Each worker derives
    its own RNG from base_seed + trajectory_index, so trajectories are
    independent and reproducible.
    """
    sim, traj_idx = args
    base_seed = sim.params.seed if sim.params.seed is not None else 0
    rng = np.random.default_rng(base_seed + traj_idx)
    # Independent rng_bb stream for Brownian bridge sampling. Same
    # rationale as ChainBDSimulator.__init__: bridge sampling stays
    # off the main rng so trajectory comparisons under bridge_on vs
    # bridge_off are clean. Offset 0xBB is arbitrary, non-zero.
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
    """Build per-trajectory diagnostic kwargs for TrajectoryResult.

    Converts the running buffers populated by ChainBDSimulator.run_one()
    into numpy arrays suitable for storage on TrajectoryResult. Empty
    buffers map to None (via absent dict keys) so downstream writers can
    detect 'no data' cleanly.
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
    """Brownian dynamics simulator for a flexible chain and a rigid target.

    Each trajectory starts with the chain at a random point on the b-sphere
    with a random orientation, and propagates until the chain either
    satisfies a reaction criterion or escapes past the escape sphere. The
    chain undergoes three coupled motions: rigid-body translation, rigid-
    body rotation, and internal coordinate fluctuation.

    Internal coordinates advance on a smaller timestep dt_chain than the
    overall translation/rotation timestep dt; chain_steps_per_outer
    internal steps are taken per outer step.

    External forces from the target are computed via a precomputed
    Poisson-Boltzmann potential grid (DXGrid). If no grid is provided
    (target_grid=None), the chain experiences no electrostatic force and
    pure diffusion drives the rigid-body motion.

    Born desolvation can be added on top of the electrostatic field by
    passing a separate Born potential grid via born_grid. The Born force
    on bead i is F_i = -alpha * q_i^2 * grad(g)(r_i), summed into the
    per-atom external force alongside the electrostatic contribution.
    The desolvation prefactor alpha defaults to 1/(4*pi) ~ 0.07957747.
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
        # Diffusion-coefficient resolution.
        # Three modes are supported, mutually exclusive:
        #   1. auto_diffusion=True with no explicit D_trans/D_rot:
        #      computes (3, 3) D_trans, D_rot from chain geometry via
        #      chain_diffusion_tensors. Full anisotropic RPY.
        #   2. auto_diffusion=False with explicit scalar D_trans, D_rot:
        #      backward-compatible path used by all earlier tests.
        #   3. auto_diffusion=False with explicit tensor D_trans, D_rot:
        #      user supplies pre-computed anisotropic D directly.
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
        # OutputConfig controls diagnostic recording cadence
        # (save_interval) and per-file flags. Default to a fresh
        # OutputConfig() if caller didn't supply one - preserves
        # backward compat with existing test fixtures.
        if outputs is None:
            from pystarc.pipeline.input_parser import OutputConfig

            outputs = OutputConfig()
        self.outputs = outputs
        # Cache target atom positions for fast distance checks during
        # diagnostic sampling (Sub-stage 2b: contact_frequency). Built
        # once at construction; reused across all sample steps.
        if self.target is not None and len(self.target.atoms) > 0:
            self._target_pos_cached = np.array(
                [[a.x, a.y, a.z] for a in self.target.atoms],
                dtype=float,
            )
        else:
            self._target_pos_cached = None
        # Cache the minimum reaction-cutoff distance so we can use the
        # smaller dt_rxn timestep when the chain is close to a reaction
        # surface. This is constant across the whole run since the
        # pathway set is fixed.
        self._rxn_min = _min_reaction_distance(pathway_set)
        # Effective hydrodynamic radius for the force-change backstep
        # criterion (motion/do_bd_step.py:backstep_due_to_force). The
        # criterion is dt > alpha * |6 pi mu * dx^2 / (dF . dx / a)|;
        # a appears as the inverse hydrodynamic radius. Two regimes:
        #
        # auto_diffusion: derive a from the trace of D_trans via the
        #   Stokes-Einstein relation D = kBT / (6 pi mu a). Self-consistent
        #   with the rigid-body diffusion tensor we computed.
        #
        # scalar / explicit-tensor: use the largest bead radius. This is
        #   conservative - a smaller radius makes the criterion tighter,
        #   so backstep slightly more often than strictly necessary.
        if self.auto_diffusion:
            D_iso = float(np.trace(np.asarray(self.D_trans)) / 3.0)
            self._effective_hydro_radius = 1.0 / (
                6.0 * math.pi * WATER_VISCOSITY * D_iso
            )
        else:
            self._effective_hydro_radius = max(a.radius for a in chain_template.atoms)
        # Cache bonded pairs and bead radii for fast hard-sphere checks.
        # Both orderings included in the set so membership lookups are
        # order-independent.
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
        # Independent RNG stream for Brownian bridge sampling. Keeping
        # this separate from self.rng means bridge sampling does not
        # advance the main trajectory RNG. So bridge_on vs bridge_off
        # comparisons at the same seed produce identical trajectories
        # (only the bridge samples themselves differ). Offset 0xBB is
        # arbitrary. Just needs to be non-zero so the streams differ.
        _bb_seed = (params.seed if params.seed is not None else 0) + 0xBB
        self.rng_bb = np.random.default_rng(_bb_seed)
        self.results: List[TrajectoryResult] = []
        self.n_reacted = 0
        self.n_escaped = 0
        # Outer propagator (LMZ) for analytical b->q diffusion zone.
        # Without this, trajectories that drift past 1.1*b are simply
        # terminated as ESCAPED, biasing P_rxn ~10x too low for typical
        # geometries (because ~b/q of "escaped" trajectories should
        # have analytically returned to the b-sphere).
        # Mirrors NAMSimulator pattern (nam_simulator.py:362-396).
        self._outer_prop = None
        try:
            from pystarc.simulation.outer_propagator import (
                OuterPropagator,
                OPGroupInfo,
            )

            # Chain bounding radius: max distance from CoM (body origin)
            # to any atom surface. For a 1-bead chain at origin, this
            # is just the atom radius.
            atom_radii = np.array(
                [a.radius for a in chain_template.atoms],
                dtype=float,
            )
            atom_dists = np.linalg.norm(
                chain_init_body_positions,
                axis=1,
            )
            chain_br = float(np.max(atom_dists + atom_radii))
            # Resolve effective scalar D for chain. If D_trans is a 3x3
            # tensor (auto_diffusion), use trace/3 as effective scalar.
            if D_trans is None:
                D_trans_eff = 0.0  # auto_diffusion case, use isotropic later
            elif np.ndim(D_trans) == 0:
                D_trans_eff = float(D_trans)
            else:
                D_trans_eff = float(np.trace(np.asarray(D_trans)) / 3.0)
            if D_rot is None:
                D_rot_eff = 0.0
            elif np.ndim(D_rot) == 0:
                D_rot_eff = float(D_rot)
            else:
                D_rot_eff = float(np.trace(np.asarray(D_rot)) / 3.0)
            # Sum of chain bead charges (total ligand charge).
            chain_q = float(sum(a.charge for a in chain_template.atoms))
            # OuterPropagator was designed for two diffusing bodies; it
            # uses Stokes-Einstein to derive hydrodynamic radii from
            # Dtrans (line 125: a0 = kT / (6*pi*mu*Dtrans)), so D=0
            # causes division by zero. Workaround: split the chain's D
            # 50/50 between g0 and g1. The propagator's math depends
            # only on D_rel = D_g0 + D_g1 (which still equals D_chain)
            # and on the symmetric combination a2 = (a0^2 + a1^2)/2,
            # so the splitting is mathematically equivalent. Drot is
            # only used for the rotational decoherence in `new_state`.
            # So, split it the same way for symmetry.
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
            # Physical constants (same as NAM, PySTARC units: A, ps, kBT).
            kT = 0.5961
            viscosity = 1.002e-3 * 1e-4 / 1e-12
            dielectric = 78.54
            vacuum_perm = 1.0 / (4 * math.pi * 332.0636)
            # Debye screening length: use the user's input.xml value
            # (threaded via params.debye_length); was hardcoded to 8.0
            # before, silently overriding <debye_length>.
            debye_len = float(params.debye_length)
            target_br = target.bounding_radius() if target is not None else 0.0
            max_mol_r = max(target_br, chain_br)
            self._outer_prop = OuterPropagator(
                b_radius=params.r_start,
                max_radius=max_mol_r,
                has_hi=True,
                kT=kT,
                viscosity=viscosity,
                dielectric=dielectric,
                vacuum_perm=vacuum_perm,
                debye_len=debye_len,
                g0=g0,
                g1=g1,
            )
        except Exception as _op_err:
            # OuterPropagator construction can fail for several reasons:
            # - D_g0 + D_g1 = 0 (both bodies stationary): ZeroDivisionError
            # - bad geometry / negative radii: ValueError
            # - numerical overflow in Romberg integration: OverflowError
            # Falling back to simple-escape lets the simulation proceed but
            # biases P_rxn ~10x low for typical geometries; warn so the
            # failure is visible (was silent before, masking real bugs).
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
        """Per-atom external forces.

        Sums up to four contributions when each is configured:
        - Electrostatic: target's PB grid if loaded.
        - Born desolvation: target's Born grid if loaded. Per-bead force
          F_i = -alpha * q_i^2 * grad(g)(r_i), summed on top of the
          electrostatic contribution.
        - Steric: WCA bead-target soft repulsion if params.use_soft_repulsion.
        - GB self-Born: chain-internal generalized Born forces if
          params.use_self_born. Path B dispatch via params.coffdrop_active:
          coffdrop_active=False -> full GB (diagonal + off-diagonal +
          vacuum Coulomb); coffdrop_active=True -> diagonal only.

        If none are active, returns zeros.
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

        Returns a TrajectoryResult with Fate.REACTED, Fate.ESCAPED, or
        the loop hits max_steps and returns ESCAPED with the
        current radius.
        If rng is provided, that generator is used for this trajectory
        only; otherwise self.rng is used. The optional argument exists
        so parallel workers can supply their own seeded generators.
        """
        params = self.params
        if rng is None:
            rng = self.rng
        if rng_bb is None:
            rng_bb = self.rng_bb

        # Per-trajectory diagnostic recording state. Updated at the
        # top of each outer BD iteration; finalized at the four
        # return sites (REACTED, ESCAPED q-sphere, ESCAPED fallback,
        # MAX_STEPS). Output writers (Stage 2b+) consume the populated
        # fields on TrajectoryResult.
        save_interval = self.outputs.save_interval if self.outputs is not None else 0
        min_sep = float("inf")
        pos_at_min = None
        q_at_min = None
        path_steps_buf: List[int] = []
        path_com_buf: List[np.ndarray] = []
        path_q_buf: List[List[float]] = []
        radial_buf: List[float] = []
        # Sub-stage 2b: energy traces and contact pair counts at recording cadence.
        energy_buf: List[List[float]] = []
        contact_count_buf: Dict[Tuple[int, int], int] = {}

        # Build a fresh ChainState for this trajectory: every trajectory
        # starts from the same body-frame configuration but evolves
        # independently from there.
        state = ChainState.from_template(
            self.chain_template,
            self.body_positions.copy(),
        )
        # Pre-equilibration: thermalize chain internal coordinates in
        # bulk (no external forces) before placing the chain on the
        # b-sphere. With params.n_equilibration_steps == 0 (default),
        # the loop body runs zero times and no RNG state is consumed,
        # so behavior is bit-exact with the pre-equilibration code path.
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
        # Random starting (CoM, orientation) on the b-sphere.
        pos, ori = initialize_bsphere(rng, params.r_start)
        # Cumulative simulated time, in ps. Accumulated each step
        # because the timestep is no longer constant: trajectories that
        # spend more time in the reaction zone advance more slowly in
        # simulated time per step than those in the bulk.
        t_elapsed = 0.0
        # Brownian bridge state: pair distances at the previous iter's
        # top (before the previous outer step), and the dt used in that
        # step. None on iter 0 (no previous step to bridge over).
        prev_top_pair_dists = None
        prev_dt_outer = 0.0
        # Effective D for the bridge formula: trace(D_trans)/3 for an
        # isotropic-equivalent. Constant across the trajectory.
        # Handles both scalar D_trans (legacy/isotropic) and tensor
        # D_trans (anisotropic auto_diffusion). Only set when bridge
        # is going to do real work (non-empty pathway_set).
        D_eff_bb = 0.0
        _has_reactions = (
            self.pathway_set is not None and len(self.pathway_set.reactions) > 0
        )
        if params.use_brownian_bridge and _has_reactions:
            _Dt = np.asarray(self.D_trans)
            if _Dt.ndim == 0:
                # scalar D_trans (legacy isotropic case)
                D_eff_bb = float(_Dt)
            elif _Dt.ndim == 2 and _Dt.shape == (3, 3):
                # 3x3 tensor (auto_diffusion / anisotropic)
                D_eff_bb = float(np.trace(_Dt) / 3.0)
            else:
                # Unexpected shape - fall back to disabling bridge to
                # avoid silent miscalculation. Bridge requires the
                # rigid-body translational diffusion scalar.
                D_eff_bb = 0.0
        for step in range(params.max_steps):
            # === Sub-stage 2a: per-iteration diagnostic recording ===
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
                # === Sub-stage 2b: energy + contact sampling ===
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
                # Steric energy: chain_target_steric_forces returns force
                # only; we leave steric=0 here as a documented omission.
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
            # === end recording ===
            # Place chain in the world frame and update the reaction-check
            # scratch molecule.
            world_pos = place_chain(state.positions, pos, ori)
            update_chain_scratch_positions(self._chain_scratch, world_pos)
            # Compute current pair distances if bridge is enabled and
            # there are reactions to track. These serve as 'new' relative
            # to the previous iteration's outer step.
            if params.use_brownian_bridge and _has_reactions and D_eff_bb > 0:
                cur_top_pair_dists = compute_pair_distances(
                    self.target,
                    world_pos,
                    self.pathway_set,
                )
            else:
                cur_top_pair_dists = None

            # Reaction check. With Brownian bridge use the bridge-aware
            # check with prior pair distances (i.e. step > 0).
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
                return TrajectoryResult(
                    Fate.REACTED,
                    step,
                    t_elapsed,
                    float(np.linalg.norm(pos)),
                    rxn,
                    **_diag,
                )
            # Outer propagator (LMZ): when the chain has drifted far
            # enough from the target (r >= 1.1*b), use the analytical
            # propagator to advance through the b->q diffusion zone.
            # The propagator either:
            #   - returns the chain to the b-sphere (reached_b=True),
            #     in which case the BD trajectory continues from a
            #     fresh random point on b. This is the LMZ "return"
            #     branch with probability ~b/q.
            #   - or terminates the trajectory as truly escaped
            #     (reached_b=False), with probability ~1-b/q.
            # Without this, all "drift past 1.1*b" trajectories were
            # counted as escapes, biasing P_rxn ~10x too low.
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
                    # Truly escaped through the q-sphere.
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
                # Otherwise: fall through. Trajectory continues from
                # the new (returned) position on the b-sphere; the
                # next outer BD step starts from there.
            # Fallback escape check (used when _outer_prop is None,
            # e.g. construction failed, or when use_lmz=False is
            # explicitly requested). Without LMZ this fires whenever
            # the chain drifts past r_escape and biases P_rxn low.
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
            # Inner steps: internal coordinate BD on body-frame positions.
            # The inner dt is independent of the outer adaptive scheme.
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
            # Outer step: refresh world positions, get external forces,
            # propagate the rigid-body coordinates.
            world_pos = place_chain(state.positions, pos, ori)
            per_atom_forces = self._compute_per_atom_external_forces(world_pos)
            F_net_old = per_atom_forces.sum(axis=0)
            # Two-zone adaptive timestep: smaller dt when close to a
            # reaction surface, larger dt in the bulk.
            r = float(np.linalg.norm(pos))
            dt_outer = params.dt_rxn if r < 1.5 * self._rxn_min else params.dt
            # Pre-draw Wiener so we can re-use it under subdivision.
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
            # Force-change backstep: if the external force changes too
            # much across the step, subdivide via the Brownian bridge
            # midpoint construction. Skipped in the dt_rxn floor zone
            # (subdividing below dt_rxn would step finer than the
            # adaptive scheme intends) and when the parameter flag
            # is off (backward-compat opt-out).
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
                    # Brownian bridge midpoint: dW_mid + dW_2nd = dW
                    # while preserving the half-step noise statistics.
                    s = math.sqrt(dt_outer / 4.0)
                    dW_mid_t = 0.5 * dW_t + s * rng.standard_normal(3)
                    dW_mid_r = 0.5 * dW_r + s * rng.standard_normal(3)
                    hdt = dt_outer / 2.0
                    # First half-step: from pos_old, ori_old, with the
                    # original (old-position) forces.
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
                    # Recompute forces at the midpoint configuration.
                    world_pos_mid = place_chain(state.positions, pos, ori)
                    per_atom_forces_mid = self._compute_per_atom_external_forces(
                        world_pos_mid
                    )
                    # Second half-step: from midpoint with midpoint forces.
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
            # Hard-sphere overlap rejection: if the trial position produced
            # an overlap (between any chain bead and any target atom, or
            # between any non-bonded chain bead pair), redraw the Wiener
            # increments and re-step from the same starting point with the
            # original forces. Capped at MAX_HS_ATTEMPTS to avoid an infinite
            # loop when the chain is wedged near the target; if all attempts
            # overlap, accept the last one with a warning. Other forces
            # (electrostatics, soft repulsion, constraints) typically resolve
            # transient overlaps over the next few BD steps.
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
            # Save state for the next iteration's Brownian bridge check.
            # cur_top_pair_dists captured 'before this outer step';
            # at next iter's top we'll compute fresh distances and use
            # those two snapshots together with prev_dt_outer = dt_outer.
            if params.use_brownian_bridge:
                prev_top_pair_dists = cur_top_pair_dists
                prev_dt_outer = dt_outer
        # max_steps exhausted without a reaction or formal escape; record
        # as ESCAPED with the current radius and accumulated time.
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
            params.max_steps,
            t_elapsed,
            float(np.linalg.norm(pos)),
            **_diag,
        )

    def run(self) -> List[TrajectoryResult]:
        """Execute n_trajectories trajectories, serial or parallel.

        Dispatches to a parallel multiprocessing pool when both
        params.n_threads > 1 and params.n_trajectories > 1; otherwise
        runs serially. Returns the accumulated self.results list.

        In serial mode, all trajectories share self.rng (one continuous
        random stream). In parallel mode, each trajectory gets its own
        RNG seeded from base_seed + trajectory_index, so results are
        deterministic across both modes within their own seeding scheme
        but the two modes do not produce the same trajectories.
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
        with mp.Pool(n_workers) as pool:
            for result in pool.map(_run_chain_trajectory_worker, args):
                self._record(result)
