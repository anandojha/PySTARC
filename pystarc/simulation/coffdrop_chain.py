"""PySTARC COFFDROP flexible chain model.

This is a Python implementation of the COFFDROP (Coarse-grained Force Field
for Disordered Proteins) flexible chain model.

COFFDROP models intrinsically disordered proteins and flexible loops as
chains of coarse-grained beads, one per residue. Each bead carries a position
in three dimensions, a diffusion coefficient set by its Stokes radius from the
COFFDROP parameter file, bonded interactions (bond lengths, bond angles, and
torsion angles), and non-bonded interactions (excluded volume and optional
electrostatics).

In the implementation, an encountering pair of systems can be two rigid
bodies, one rigid body together with one flexible chain, or two flexible
chains.

This module implements the chain kinematics, force evaluation, and Brownian
dynamics propagation for flexible chains. It is a foundation for future full
COFFDROP support.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import math
from pystarc.global_defs.constants import KBT_KCAL
from pystarc.global_defs.defaults import CHAIN_DT, DESOLVATION_ALPHA, VISCOSITY
from pystarc.global_defs.defaults import CHAIN_DT

# A bond, angle, or dihedral involves atoms that may live either on
# the chain itself or on a rigid core (a structured protein domain).
# The tagged-union representation lets a single bonded interaction span
# chain and core, which is what allows a flexible linker connecting
# two folded domains, or an IDP tail anchored to a structured region.


@dataclass(frozen=True)
class ChainAtomRef:
    """Reference to an atom on the chain itself."""

    atom_idx: int


@dataclass(frozen=True)
class CoreAtomRef:
    """Reference to an atom on a specific rigid core in a specific group."""

    group_idx: int
    core_idx: int
    atom_idx: int


# Every atom in a bonded interaction is one or the other.
AtomRef = Union[ChainAtomRef, CoreAtomRef]
# Bonded-interaction force evaluation needs to translate an AtomRef into
# an index into the chain's atom array. For a ChainAtomRef this is just
# the atom_idx field. A CoreAtomRef refers to an atom on a rigid body and
# is handled by separate machinery (chain-core coupled forces) once that
# is wired in.


def _chain_idx(ref: "AtomRef") -> int:
    """Index into the chain's atom array. Raises for core references.

    A ChainAtomRef resolves to its atom_idx field. A bare integer is already
    an index into the chain's atom array and is returned unchanged, which lets
    the same resolution path serve callers that hold either form.
    """
    if isinstance(ref, ChainAtomRef):
        return ref.atom_idx
    if isinstance(ref, (int, np.integer)):
        return int(ref)
    raise NotImplementedError(
        "CoreAtomRef in chain-only force path; chain-core coupling not yet wired"
    )


# Per-atom immutable properties: Stokes radius (sets the bead diffusion
# coefficient via Stokes-Einstein, D = kT / 6 pi eta a), partial charge,
# residue identity. These do not change during a simulation and are
# shared across all trajectory replicas. Mutable per-trajectory state
# (position, force) lives separately on the chain state.


@dataclass
class ChainAtom:
    """Topology-only atom on a chain. No position, no force."""

    radius: float
    charge: float
    resname: str = ""
    resid: int = 0


# A bond exerts a force from a potential V(r), and the distance fluctuates
# around its equilibrium. A constraint instead fixes a geometric quantity
# (distance, planarity) exactly via iterative projection at every step
# (SHAKE or RATTLE). The two need separate types because they are evaluated
# by different machinery. Forces enter the Ermak-McCammon step, while
# constraints are imposed as a post-step correction.


@dataclass
class LengthConstraint:
    """Fixed-distance constraint between two atoms."""

    a: AtomRef
    b: AtomRef
    length: float


@dataclass
class CoplanarConstraint:
    """Constrain four atoms to lie in a single plane (e.g. peptide bond)."""

    a: AtomRef
    b: AtomRef
    c: AtomRef
    d: AtomRef


# A simulation has one chain definition shared across many trajectory
# replicas. Per-atom topology (radius, charge, residue identity), bonded
# interactions, and constraints do not change between replicas, so they
# live on a single immutable object. Per-trajectory mutable state
# (positions, forces, runtime flags) lives separately on a chain state
# object that holds a reference back to its definition.


@dataclass
class ChainCommon:
    """Immutable chain definition: topology, bonded interactions, constraints."""

    name: str
    atoms: List[ChainAtom]
    bonds: List["ChainBond"] = field(default_factory=list)
    angles: List["ChainAngle"] = field(default_factory=list)
    torsions: List["ChainTorsion"] = field(default_factory=list)
    length_constraints: List[LengthConstraint] = field(default_factory=list)
    coplanar_constraints: List[CoplanarConstraint] = field(default_factory=list)
    never_frozen: bool = True
    # If set, this chain advances on a fixed timestep instead of the
    # adaptive scheme used by the global propagator.
    const_dt: Optional[float] = None
    # COFFDROP tabulated parameters. When set, bonded force functions
    # (_bond_force_state, _angle_force_state, _torsion_force_state)
    # use the tabulated potentials looked up via type_idx fields
    # instead of the harmonic / cosine-series fallback. Default None
    # preserves backward-compatible harmonic behavior.
    coffdrop_params: Optional["COFFDROPParams"] = None
    # Cache of per-pair type_idx into coffdrop_params.pair_pots.
    # Keys are (i, j) with i < j (chain atom indices). It is populated by
    # the chain construction helper for COFFDROP-aware chains and is empty
    # by default. Pairs not in this dict do not contribute a tabulated pair
    # force. They fall back to WCA when soft_repulsion is on, or are skipped
    # entirely otherwise.
    pair_lookups: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


# Per-trajectory mutable state. Positions and forces are stored as
# contiguous (n_atoms, 3) arrays for vectorized force evaluation and to
# match the layout used by the GPU batch propagator on the rigid-body side.
# estatus carries the non-bonded exclusion classification per atom. A value
# of 0 marks a normal pair, 1 marks an excluded 1-2 or 1-3 neighbour, and 2
# marks a one-four scaled pair.


@dataclass
class ChainState:
    """Mutable per-trajectory state for a chain."""

    common: ChainCommon
    positions: np.ndarray
    forces: np.ndarray
    estatus: np.ndarray
    frozen: bool = False
    has_near_interaction: bool = False

    @classmethod
    def from_template(cls, common: ChainCommon, positions: np.ndarray) -> "ChainState":
        """Build a fresh state for one trajectory from a topology and initial coords."""
        n = common.n_atoms
        positions = np.asarray(positions, dtype=float).reshape(n, 3)
        return cls(
            common=common,
            positions=positions,
            forces=np.zeros((n, 3), dtype=float),
            estatus=np.zeros(n, dtype=np.int32),
        )

    @property
    def n_atoms(self) -> int:
        return self.common.n_atoms

    def zero_forces(self) -> None:
        self.forces.fill(0.0)


# A length constraint between atoms a, b with target distance L is satisfied
# when |r_a - r_b| = L. Its violation is the signed deviation |r_a - r_b| - L.
# Positive means the bond is over-extended and negative means compressed.
#
# A coplanar constraint on atoms (a, b, c, d) requires atom a to lie in the
# plane defined by atoms b, c, d. The violation is the signed perpendicular
# distance of atom a from that plane, using the centroid of (b, c, d) as the
# in-plane reference point so that all three plane atoms enter symmetrically.
#
# The returned vector has length len(length_constraints) + len(coplanar_constraints),
# with length-constraint entries first in their stored order, then the coplanar
# entries. This canonical ordering is what the constraint solver uses when
# assembling the Jacobian and Lagrange-multiplier system.


def compute_constraint_violations(state: "ChainState") -> np.ndarray:
    """Signed violation vector for all constraints on a chain state."""
    common = state.common
    pos = state.positions
    n_len = len(common.length_constraints)
    n_cop = len(common.coplanar_constraints)
    phi = np.zeros(n_len + n_cop, dtype=float)
    for ic, c in enumerate(common.length_constraints):
        ra = pos[_chain_idx(c.a)]
        rb = pos[_chain_idx(c.b)]
        phi[ic] = float(np.linalg.norm(ra - rb)) - c.length
    for ic, c in enumerate(common.coplanar_constraints):
        ra = pos[_chain_idx(c.a)]
        rb = pos[_chain_idx(c.b)]
        rc = pos[_chain_idx(c.c)]
        rd = pos[_chain_idx(c.d)]
        centroid = (rb + rc + rd) / 3.0
        d2 = rc - rb
        d3 = rd - rb
        perp = np.cross(d2, d3)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            # Degenerate: the three plane atoms are colinear and the plane
            # is undefined. Report zero violation (the constraint cannot be
            # evaluated meaningfully); the solver will see this and either
            # accept it or report failure.
            phi[n_len + ic] = 0.0
        else:
            n_hat = perp / perp_norm
            phi[n_len + ic] = float(np.dot(ra - centroid, n_hat))
    return phi


# Iterative constraint satisfaction in the style of SHAKE.
#
# Each sweep visits every constraint and applies a local position correction
# that exactly satisfies that constraint in isolation. After visiting all
# constraints, neighboring constraints may have been perturbed, so the routine
# sweeps again. It converges when the maximum signed violation drops below tol.
#
# For a length constraint between atoms a, b with target L, the correction is
#   delta = (|r_b - r_a| - L) / 2
#   r_a += +delta * (r_b - r_a) / |r_b - r_a|
#   r_b += -delta * (r_b - r_a) / |r_b - r_a|
# After this, |r_b - r_a| equals L exactly.
#
# For a coplanar constraint on atoms (a, b, c, d), atom a must lie in the plane
# of atoms b, c, d. The plane normal n_hat is built from atoms b, c, d, and atom
# a is shifted by -phi * n_hat, where phi is the signed distance from a to
# the plane. Atoms b, c, d are not moved by this correction.
#
# This pairwise sweep is the simplest form of SHAKE. It handles loosely
# coupled constraint networks (chains, lightly cross-linked systems) but can
# converge slowly or stall on tightly coupled networks. The Newton fallback
# handles those cases.


def satisfy_constraints(
    state: "ChainState", tol: float = 1e-6, max_iter: int = 200
) -> int:
    """Iteratively project positions onto the constraint manifold.

    The positions of state are modified in place. tol is the convergence
    tolerance on the maximum signed violation, and max_iter is a hard cap on
    the number of sweeps. The function returns the number of sweeps used and
    raises RuntimeError if the solver fails to converge within max_iter sweeps.
    """
    common = state.common
    pos = state.positions
    if not common.length_constraints and not common.coplanar_constraints:
        return 0
    for it in range(max_iter):
        max_violation = 0.0
        # Length constraints: split correction symmetrically along bond axis.
        for c in common.length_constraints:
            ia = _chain_idx(c.a)
            ib = _chain_idx(c.b)
            ra = pos[ia]
            rb = pos[ib]
            d = rb - ra
            r = float(np.linalg.norm(d))
            if r < 1e-12:
                # Coincident atoms, so the direction cannot be resolved. Skip
                # this constraint and let other forces (or another sweep) move
                # the atoms apart first.
                continue
            violation = r - c.length
            if abs(violation) > max_violation:
                max_violation = abs(violation)
            d_hat = d / r
            half = 0.5 * violation
            pos[ia] = ra + half * d_hat
            pos[ib] = rb - half * d_hat
        # Coplanar constraints: shift only atom a onto the plane.
        for c in common.coplanar_constraints:
            ia = _chain_idx(c.a)
            ra = pos[ia]
            rb = pos[_chain_idx(c.b)]
            rc = pos[_chain_idx(c.c)]
            rd = pos[_chain_idx(c.d)]
            centroid = (rb + rc + rd) / 3.0
            d2 = rc - rb
            d3 = rd - rb
            perp = np.cross(d2, d3)
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm < 1e-12:
                continue
            n_hat = perp / perp_norm
            phi = float(np.dot(ra - centroid, n_hat))
            if abs(phi) > max_violation:
                max_violation = abs(phi)
            pos[ia] = ra - phi * n_hat
        if max_violation < tol:
            return it + 1
    raise RuntimeError(
        f"satisfy_constraints failed to converge in {max_iter} iterations; "
        f"final max violation = {max_violation:.3e}, tol = {tol:.3e}"
    )


# Constraint Jacobian for the Newton solver.
#
# Each row of the Jacobian is the gradient of one scalar constraint with
# respect to the flattened position vector r in R^{3*n_atoms}. Length-
# constraint gradients are analytic and simple. Coplanar-constraint
# gradients are analytic for atom a and computed by finite differences for
# the three plane atoms b, c, d. The finite-difference piece is acceptable
# because coplanar constraints are rare in production chain configurations
# and the cost is bounded, at 9 extra evaluations of one scalar per coplanar
# constraint per Newton step.


def _build_constraint_jacobian(state: "ChainState") -> np.ndarray:
    """Constraint Jacobian J of shape (n_constraints, 3 * n_atoms).

    Row i, columns 3k..3k+2 hold dc_i/dr_k for atom k.
    """
    common = state.common
    pos = state.positions
    n_atoms = state.n_atoms
    n_len = len(common.length_constraints)
    n_cop = len(common.coplanar_constraints)
    J = np.zeros((n_len + n_cop, 3 * n_atoms), dtype=float)
    # Length-constraint rows: analytic.
    for ic, c in enumerate(common.length_constraints):
        ia = _chain_idx(c.a)
        ib = _chain_idx(c.b)
        ra = pos[ia]
        rb = pos[ib]
        d = ra - rb
        r = float(np.linalg.norm(d))
        if r < 1e-12:
            continue
        d_hat = d / r
        J[ic, 3 * ia : 3 * ia + 3] = d_hat
        J[ic, 3 * ib : 3 * ib + 3] = -d_hat
    # Coplanar-constraint rows: analytic for atom a, FD for atoms b, c, d.
    for ic, c in enumerate(common.coplanar_constraints):
        row_idx = n_len + ic
        ia = _chain_idx(c.a)
        ra = pos[ia]
        rb = pos[_chain_idx(c.b)]
        rc = pos[_chain_idx(c.c)]
        rd = pos[_chain_idx(c.d)]
        centroid = (rb + rc + rd) / 3.0
        d2 = rc - rb
        d3 = rd - rb
        perp = np.cross(d2, d3)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            continue
        n_hat = perp / perp_norm
        # dc/dr_a = n_hat (the in-plane reference moves linearly with a's
        # projection along n_hat).
        J[row_idx, 3 * ia : 3 * ia + 3] = n_hat
        # FD for the three plane-defining atoms. Perturb each component of
        # r_b, r_c, r_d in turn and recompute the scalar constraint.
        eps = 1e-6
        for plane_atom_idx, plane_field in [
            (_chain_idx(c.b), "b"),
            (_chain_idx(c.c), "c"),
            (_chain_idx(c.d), "d"),
        ]:
            for k in range(3):
                pos[plane_atom_idx, k] += eps
                phi_p = _coplanar_violation(state, c)
                pos[plane_atom_idx, k] -= 2 * eps
                phi_m = _coplanar_violation(state, c)
                pos[plane_atom_idx, k] += eps
                J[row_idx, 3 * plane_atom_idx + k] = (phi_p - phi_m) / (2 * eps)

    return J


def _coplanar_violation(state: "ChainState", c: "CoplanarConstraint") -> float:
    """Signed perpendicular distance of atom a from the plane of b, c, d."""
    pos = state.positions
    ra = pos[_chain_idx(c.a)]
    rb = pos[_chain_idx(c.b)]
    rc = pos[_chain_idx(c.c)]
    rd = pos[_chain_idx(c.d)]
    centroid = (rb + rc + rd) / 3.0
    d2 = rc - rb
    d3 = rd - rb
    perp = np.cross(d2, d3)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-12:
        return 0.0
    return float(np.dot(ra - centroid, perp / perp_norm))


# Newton constraint solver.
#
# At each iteration the solver builds the constraint Jacobian J = dphi/dr (one
# row per constraint, 3 columns per atom) and the constraint residual phi. The
# step Delta_r is parameterized as Delta_r = J^T lambda for Lagrange
# multipliers lambda in R^{n_constraints}. Substituting into the linearized
# constraint equation J Delta_r = -phi gives the n_c x n_c square system
#
#   (J J^T) lambda = -phi
#
# which is small and positive-(semi)definite for non-degenerate constraint
# sets. The solver finds lambda, recovers Delta_r = J^T lambda, applies it, and
# iterates.
#
# If a step does not reduce ||phi||, the solver halves it and retries up to a
# few times. This handles mild overshoot. For severe cases (a rank-deficient
# Jacobian or a far-from-feasible start), the solver raises rather than
# silently doing the wrong thing.


def satisfy_constraints_newton(
    state: "ChainState",
    tol: float = 1e-8,
    max_iter: int = 50,
    max_damping: int = 8,
) -> int:
    """Newton-Lagrange constraint solver.

    Converges in a handful of iterations on chain configurations regardless
    of chain length, in contrast to SHAKE which scales as O(N) sweeps for
    a chain of N constrained bonds.

    Returns the number of Newton iterations used. Raises RuntimeError if
    the solver fails to converge or if the Jacobian is rank-deficient.
    """
    common = state.common
    if not common.length_constraints and not common.coplanar_constraints:
        return 0
    n_atoms = state.n_atoms
    # Initialize new_violation so the final RuntimeError still references a
    # bound name when max_iter == 0 and the loop never runs.
    new_violation = float("nan")
    for it in range(max_iter):
        phi = compute_constraint_violations(state)
        violation = float(np.max(np.abs(phi))) if phi.size else 0.0
        if violation < tol:
            return it
        J = _build_constraint_jacobian(state)
        # JJt is square (n_c x n_c), positive semi-definite. For a non-
        # degenerate constraint set it is positive definite and can
        # be solved directly. Use lstsq for graceful handling of rank
        # deficiency (e.g. redundant constraints).
        JJt = J @ J.T
        try:
            lam = np.linalg.solve(JJt, -phi)
        except np.linalg.LinAlgError:
            # Singular: fall back to least-squares (returns minimum-norm
            # solution).
            lam, *_ = np.linalg.lstsq(JJt, -phi, rcond=None)
        delta = (J.T @ lam).reshape(n_atoms, 3)
        # Damped step: try full step first, halve if violation does not
        # decrease.
        scale = 1.0
        original_pos = state.positions.copy()
        for _ in range(max_damping + 1):
            state.positions[:] = original_pos + scale * delta
            new_phi = compute_constraint_violations(state)
            new_violation = float(np.max(np.abs(new_phi))) if new_phi.size else 0.0
            if new_violation < violation:
                break
            scale *= 0.5
        else:
            # All damping attempts failed: revert and bail.
            state.positions[:] = original_pos
            raise RuntimeError(
                f"satisfy_constraints_newton: damped step failed at iter {it}; "
                f"violation = {violation:.3e}, tol = {tol:.3e}"
            )
    raise RuntimeError(
        f"satisfy_constraints_newton: did not converge in {max_iter} iterations; "
        f"final violation = {new_violation:.3e}, tol = {tol:.3e}"
    )


# Hybrid solver. It tries SHAKE first for cheap convergence, and switches to
# Newton if SHAKE stalls. SHAKE handles the common case (chains, lightly
# coupled networks) in microseconds. Newton handles the hard cases (tight
# rings, dense cross-linking) but at higher per-iteration cost. The hybrid
# gives the best of both with no user tuning required.


def satisfy_constraints_hybrid(
    state: "ChainState",
    tol: float = 1e-8,
    shake_max_iter: int = 50,
    newton_max_iter: int = 50,
) -> int:
    """SHAKE first, fall back to Newton if SHAKE does not converge.

    Returns the total iteration count across both solvers (SHAKE sweeps +
    Newton iterations).
    """
    try:
        return satisfy_constraints(state, tol=tol, max_iter=shake_max_iter)
    except RuntimeError:
        # SHAKE stalled. Try Newton.
        n = satisfy_constraints_newton(state, tol=tol, max_iter=newton_max_iter)
        return shake_max_iter + n


# Force evaluation on ChainState.
#
# compute_chain_forces fills state.forces from the bonded interactions
# (bonds, angles, torsions) defined on state.common. The kernels below are
# direct ports of the corresponding methods on ChainForceEvaluator (which
# operates on the legacy FlexibleChain type). The math is the same and only
# the data access changes. The bead position chain.beads[ref.atom_idx].pos
# becomes state.positions[_chain_idx(ref)], and where the legacy kernels
# returned a force array for the caller to sum, here state.forces is zeroed
# first and then accumulated in place.
#
# Non-bonded interactions (COFFDROP pair potentials) are not handled here.
# They are added in a separate function.


def _bond_force_state(state: "ChainState", bond: "ChainBond") -> None:
    """Accumulate harmonic-bond force contribution into state.forces."""
    ia = _chain_idx(bond.a)
    ib = _chain_idx(bond.b)
    ri = state.positions[ia]
    rj = state.positions[ib]
    # Finiteness/sanity guard: skip if either position is non-finite
    # or has extreme magnitude. Prevents NaN propagation during BD
    # when transient configurations produce extreme intermediates.
    if not (np.all(np.isfinite(ri)) and np.all(np.isfinite(rj))):
        return
    if np.max(np.abs(ri)) > 1e6 or np.max(np.abs(rj)) > 1e6:
        return
    dr = rj - ri
    r = float(np.linalg.norm(dr))
    if r < 1e-8:
        return
    r_hat = dr / r
    f_mag = -bond.k_spring * (r - bond.r0)  # kBT/A
    state.forces[ia] -= f_mag * r_hat
    state.forces[ib] += f_mag * r_hat


def _angle_force_state(state: "ChainState", angle: "ChainAngle") -> None:
    """Accumulate the angle force contribution into state.forces.

    There are two modes. The default harmonic mode uses the potential

        V(θ) = (1/2) k_angle (θ - θ0)^2

    so that dV/dθ = k_angle (θ - θ0) in radians. The tabulated COFFDROP mode
    applies when angle.type_idx >= 0 and state.common.coffdrop_params is not
    None. It looks up dV/dθ in kBT per degree from the spline at index
    type_idx and converts to radians via dV/dθ_rad = dV/dθ_deg × (180/π).

    Both modes share the same chain rule for projecting dV/dθ onto the three
    atom positions.
    """
    ia = _chain_idx(angle.a)
    ib = _chain_idx(angle.b)
    ic = _chain_idx(angle.c)
    ri = state.positions[ia]
    rj = state.positions[ib]
    rk = state.positions[ic]
    u = ri - rj
    v = rk - rj
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-8 or nv < 1e-8:
        return
    u = u / nu
    v = v / nv
    cos_t = float(np.dot(u, v))
    cos_t = max(-1.0, min(1.0, cos_t))
    theta = math.acos(cos_t)
    sin_t = math.sin(theta)
    if abs(sin_t) < 1e-8:
        return
    # Compute dV/dθ in radians, either harmonic or tabulated.
    cd_params = state.common.coffdrop_params
    if angle.type_idx >= 0 and cd_params is not None:
        # Tabulated path. The spline derivative is in kBT per degree, so
        # convert it to per radian.
        theta_deg = theta * 180.0 / math.pi
        pot = cd_params.angle_pots[angle.type_idx]
        dV_deg = pot.deriv(theta_deg)
        dV_drad = dV_deg * (180.0 / math.pi)
    else:
        # Harmonic fallback.
        d_theta = theta - angle.theta0
        dV_drad = angle.k_angle * d_theta
    coeff = dV_drad / sin_t
    fi = coeff * (v - cos_t * u) / nu
    fk = coeff * (u - cos_t * v) / nv
    state.forces[ia] += fi
    state.forces[ic] += fk
    state.forces[ib] -= fi + fk


def _torsion_force_state(state: "ChainState", tor: "ChainTorsion") -> None:
    """Accumulate the cosine-series torsion force contribution into state.forces.

    This is a direct port of ChainForceEvaluator._torsion_force and uses the
    same backward-mode automatic-differentiation formulation with explicit
    acos and asin branches.
    """
    ia = _chain_idx(tor.a)
    ib = _chain_idx(tor.b)
    ic = _chain_idx(tor.c)
    ild = _chain_idx(tor.d)
    ri = state.positions[ia]
    rj = state.positions[ib]
    rk = state.positions[ic]
    rl = state.positions[ild]
    # Numerical-safety guard: skip torsion contribution if any input
    # position is non-finite or has extreme magnitude. Cross-product
    # math overflows with very large inputs (e.g., during BD when a
    # trajectory wanders far) and contaminates state.forces with
    # NaN/Inf. Threshold of 1e6 A is well beyond any sane BD context.
    if not (
        np.all(np.isfinite(ri))
        and np.all(np.isfinite(rj))
        and np.all(np.isfinite(rk))
        and np.all(np.isfinite(rl))
    ):
        return
    if (
        np.max(np.abs(ri)) > 1e6
        or np.max(np.abs(rj)) > 1e6
        or np.max(np.abs(rk)) > 1e6
        or np.max(np.abs(rl)) > 1e6
    ):
        return
    # Delegate to the legacy kernel through a tiny ad-hoc adapter. The kernel
    # returns an (n_beads, 3) force array, whose output is then accumulated
    # onto state.forces. To avoid duplicating the entire backward-AD body
    # here, build a transient minimal FlexibleChain wrapping just these four
    # positions.
    bead_i = ChainBead(pos=ri.copy(), force=np.zeros(3), radius=1.0, charge=0.0)
    bead_j = ChainBead(pos=rj.copy(), force=np.zeros(3), radius=1.0, charge=0.0)
    bead_k = ChainBead(pos=rk.copy(), force=np.zeros(3), radius=1.0, charge=0.0)
    bead_l = ChainBead(pos=rl.copy(), force=np.zeros(3), radius=1.0, charge=0.0)
    fc = FlexibleChain(beads=[bead_i, bead_j, bead_k, bead_l])
    # Pass coffdrop_params and preserve type_idx via local_tor for
    # the legacy class to use the tabulated branch when set.
    cd_params = state.common.coffdrop_params
    local_tor_with_idx = ChainTorsion(
        a=ChainAtomRef(0),
        b=ChainAtomRef(1),
        c=ChainAtomRef(2),
        d=ChainAtomRef(3),
        phi0=tor.phi0,
        k_tor=tor.k_tor,
        n=tor.n,
        type_idx=tor.type_idx,
    )
    F = ChainForceEvaluator()._torsion_force(
        fc,
        local_tor_with_idx,
        kT=1.0,
        coffdrop_params=cd_params,
    )
    state.forces[ia] += F[0]
    state.forces[ib] += F[1]
    state.forces[ic] += F[2]
    state.forces[ild] += F[3]


def compute_target_grid_forces(
    state: "ChainState",
    dx_grids: list,
    atom_charges: np.ndarray,
) -> None:
    """Accumulate electrostatic forces from APBS DX grids into state.forces.

    Uses finest-grid-first selection per atom: each chain atom is assigned
    to the finest grid (smallest spacing) that contains its position. Atoms
    outside all grids contribute zero (negligible far-field).

    Parameters
    ----------
    state : ChainState
        Current chain state. state.forces is updated in place.
    dx_grids : list of DXGrid
        Loaded DX grids covering the target's electrostatic potential.
        Multiple grids enable coarse+fine resolution.
    atom_charges : (n_atoms,) ndarray
        Per-atom charge in elementary units e.

    Force units returned by DXGrid.batch_force_on_charges are kBT/A,
    matching the existing chain BD force units.
    """
    if not dx_grids:
        return
    # Sort by cell volume (dx*dy*dz) so the finest-spacing grid sorts first.
    # Using the diagonal product handles anisotropic spacings correctly. For
    # typical axis-aligned APBS grids with dx=dy=dz this is monotonic in any
    # single delta[i,i], so the sort order matches the simpler delta[0,0] key.
    sorted_grids = sorted(dx_grids, key=lambda g: float(np.prod(np.diag(g.delta))))
    positions = state.positions
    n_atoms = positions.shape[0]
    assigned = np.zeros(n_atoms, dtype=bool)
    for grid in sorted_grids:
        idx_list = []
        spacing = np.diag(grid.delta)
        extent = (np.array(grid.data.shape) - 1) * spacing
        for i in range(n_atoms):
            if assigned[i]:
                continue
            if abs(atom_charges[i]) < 1e-9:
                assigned[i] = True
                continue
            pt = positions[i]
            local = pt - grid.origin
            if (local >= 0).all() and (local <= extent).all():
                idx_list.append(i)
                assigned[i] = True
        if not idx_list:
            continue
        sub_pos = np.ascontiguousarray(positions[idx_list])
        sub_chg = np.ascontiguousarray(atom_charges[idx_list])
        forces = grid.batch_force_on_charges(sub_pos, sub_chg)
        for k, i in enumerate(idx_list):
            state.forces[i] += forces[k]


def compute_chain_forces(
    state: "ChainState",
    kT: float = 1.0,
    soft_repulsion: bool = False,
    soft_repulsion_eps: float = 1.0,
    target_grids: Optional[list] = None,
) -> None:
    """Fill state.forces from the bonded interactions on state.positions.

    state.forces is zeroed first, then the bond, angle, and torsion
    contributions are accumulated. If soft_repulsion is True, the function
    also adds intra-chain WCA forces between non-bonded bead pairs. This is
    not full COFFDROP, since it uses radius sums for sigma and a uniform
    epsilon.

    state is the chain ChainState holding positions, forces, and common
    topology. kT is the thermal energy unit, which is currently unused inside
    the bonded force functions and kept for API compatibility. When
    soft_repulsion is True, chain_intra_nonbonded_forces is accumulated on top
    of the bonded forces. It defaults to False because eps = 1 is not physical
    for arbitrary chains. soft_repulsion_eps is the WCA epsilon in kBT units
    and is only used when soft_repulsion is True.
    """
    state.zero_forces()
    common = state.common
    for bond in common.bonds:
        _bond_force_state(state, bond)
    for angle in common.angles:
        _angle_force_state(state, angle)
    for torsion in common.torsions:
        _torsion_force_state(state, torsion)
    # COFFDROP tabulated pair forces: applied automatically when the
    # chain has been built with both coffdrop_params and a non-empty
    # pair_lookups dict. This is the production non-bonded path for
    # COFFDROP-aware chains. Backward compatible: chains without
    # pair_lookups skip this branch.
    if common.coffdrop_params is not None and common.pair_lookups:
        F_pair = chain_intra_coffdrop_pair_forces(state, common)
        state.forces += F_pair
    # Target electrostatic forces from APBS DX grids.
    # Reads atom charges from common.atoms[i].charge (set during chain build).
    if target_grids:
        atom_charges = np.array([a.charge for a in common.atoms])
        compute_target_grid_forces(state, target_grids, atom_charges)
    if soft_repulsion:
        F_wca = chain_intra_nonbonded_forces(
            state,
            common,
            eps=soft_repulsion_eps,
        )
        state.forces += F_wca


def chain_intra_nonbonded_forces(
    state: "ChainState",
    common: "ChainCommon",
    eps: float = 1.0,
) -> np.ndarray:
    """Intra-chain WCA forces between non-bonded bead pairs.

    Returns an (n_atoms, 3) array of per-bead forces in kBT/Å units. The WCA
    potential is

        V(r) = 4 eps [(sig/r)^12 - (sig/r)^6] + eps   if r < r_min
             = 0                                       if r >= r_min

    where r_min = 2^(1/6) × sig is the Lennard-Jones minimum (the standard WCA
    convention) and sig is the sum of the two bead radii. The force on bead i
    from bead j is

        F_i = -dV/dr × r_hat_ij

    with r_hat_ij pointing from j to i, so at r < sig the force pushes bead i
    away from bead j, which is the correct repulsion. Bead j receives the
    opposite force by Newton's third law.

    Pairs (i, j) with j < i+2 are skipped, so only j >= i+2 are considered.
    This matches the legacy excluded-volume convention. It excludes both
    bonded neighbors and any 1-3 pairs that share a bonded neighbor, which for
    a non-branched chain amounts to skipping the bond and angle
    nearest-neighbor pairs. Bonded pairs that are not consecutive (for example
    ring closures, if any) are also skipped via the explicit bonded set. Ghost
    beads with radius < 1e-10 are skipped to avoid divide-by-zero forces.

    Here state is the ChainState providing positions and atom radii via
    common, common is the ChainCommon holding the chain topology (bonds,
    atoms), and eps is the WCA well depth in kBT units (default 1.0). The
    return value F is the (n_atoms, 3) per-bead force array.
    """
    n = common.n_atoms
    F = np.zeros((n, 3))
    bonded_pairs = set()
    for bond in common.bonds:
        i, j = bond.a.atom_idx, bond.b.atom_idx
        bonded_pairs.add((i, j))
        bonded_pairs.add((j, i))

    radii = np.array([a.radius for a in common.atoms], dtype=float)
    pos = state.positions

    # WCA cutoff at r_min = 2^(1/6) * sigma (the LJ minimum)
    WCA_CUTOFF_FACTOR = 2.0 ** (1.0 / 6.0)

    for i in range(n):
        if radii[i] < 1e-10:
            continue
        for j in range(i + 2, n):  # legacy convention: skip i, i+1
            if radii[j] < 1e-10:
                continue
            if (i, j) in bonded_pairs:
                continue
            dr = pos[i] - pos[j]  # vector FROM j TO i
            r = float(np.linalg.norm(dr))
            sig = radii[i] + radii[j]
            sig_cutoff = WCA_CUTOFF_FACTOR * sig
            if r < 1e-10 or r >= sig_cutoff:
                continue
            sr = sig / r
            sr6 = sr**6
            sr12 = sr6 * sr6
            # |dV/dr| = 4 eps [12 sig^12/r^13 - 6 sig^6/r^7]
            #        = (4 eps / r) * [12 sr12 - 6 sr6]
            # Force on i = -(dV/dr) * r_hat_ij where r_hat_ij = dr / r
            #            = (4 eps / r^2) * [12 sr12 - 6 sr6] * dr
            f_mag_over_r = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / (r * r)
            fvec = f_mag_over_r * dr
            F[i] += fvec
            F[j] -= fvec

    return F


def chain_intra_coffdrop_pair_forces(
    state: "ChainState",
    common: "ChainCommon",
) -> np.ndarray:
    """Intra-chain non-bonded pair forces from COFFDROP tabulated potentials.

    Returns an (n_atoms, 3) array of per-bead forces in kBT/Å units.

    For each non-bonded chain pair (i, j) with i < j and j >= i + 2 (the
    legacy excluded-volume convention, which skips bonded and 1-3 neighbors),
    if the pair has a type_idx in common.pair_lookups, the spline derivative
    dV/dr is looked up at the current pair separation r. The force on i is
    -dV/dr × (r_i - r_j) / r, and the force on j is the opposite by Newton's
    third law.

    Pairs not in pair_lookups contribute zero force here. The caller should
    fall back to WCA via chain_intra_nonbonded_forces if a soft-repulsion
    floor is desired for unmapped pairs.

    This requires common.coffdrop_params to be set. It is not checked in this
    function, since compute_chain_forces is the gate.

    Here state is the ChainState providing positions and common is the
    ChainCommon holding the pair_lookups dict and coffdrop_params. The return
    value F is the (n_atoms, 3) per-bead force array.
    """
    n = common.n_atoms
    F = np.zeros((n, 3))
    pair_lookups = common.pair_lookups
    if not pair_lookups:
        return F  # no tabulated pairs configured
    pair_pots = common.coffdrop_params.pair_pots
    pos = state.positions

    # Vectorized path: build numpy arrays once per call (cheap),
    # then do all distance math in numpy and group spline calls
    # by type_idx so each spline is invoked once per type.
    keys = list(pair_lookups.keys())
    n_pairs = len(keys)
    i_arr = np.empty(n_pairs, dtype=np.int64)
    j_arr = np.empty(n_pairs, dtype=np.int64)
    t_arr = np.empty(n_pairs, dtype=np.int64)
    for k, (i, j) in enumerate(keys):
        i_arr[k] = i
        j_arr[k] = j
        t_arr[k] = pair_lookups[(i, j)]

    # All dr vectors and distances in two numpy ops.
    dr_arr = pos[i_arr] - pos[j_arr]  # (n_pairs, 3)
    r_arr = np.linalg.norm(dr_arr, axis=1)  # (n_pairs,)

    # Group pairs by type_idx so each spline gets a vectorized
    # batch evaluation via deriv_array (handles bounds internally).
    unique_types = np.unique(t_arr)
    dV_dr_arr = np.zeros(n_pairs)
    for t in unique_types:
        mask = t_arr == t
        rs = r_arr[mask]
        dV_dr_arr[mask] = pair_pots[int(t)].deriv_array(rs)

    # Compute the force vectors. Mask out r near zero (a degenerate case) to
    # avoid divide-by-zero. deriv_array would have returned 0 there anyway
    # since x_min > 0.
    safe_r = np.where(r_arr > 1e-10, r_arr, 1.0)  # avoid div-by-zero
    f_mag_over_r = np.where(r_arr > 1e-10, -dV_dr_arr / safe_r, 0.0)
    fvecs = f_mag_over_r[:, None] * dr_arr  # (n_pairs, 3)

    # Scatter forces into F. np.add.at is the safe scatter for repeated
    # indices (atoms appearing in multiple pairs).
    np.add.at(F, i_arr, fvecs)
    np.add.at(F, j_arr, -fvecs)

    return F


# Bead data structure
@dataclass
class ChainBead:
    """One coarse-grained bead in a COFFDROP chain.

    Legacy type. Kept for compatibility with existing force evaluators
    and tests in this module. New code should use ChainAtom (topology)
    on ChainCommon and the position/force arrays on ChainState. This
    class will be removed once the last consumer is rewritten."""

    pos: np.ndarray  # (3,) current position (A)
    force: np.ndarray  # (3,) current force (kBT/A)
    radius: float  # (A) Stokes radius for diffusion
    charge: float  # (e) partial charge
    resname: str = ""  # residue name (1-letter or 3-letter)
    resid: int = 0  # residue index

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.force = np.asarray(self.force, dtype=float)


@dataclass
class ChainBond:
    """Two-body bonded interaction. Endpoints may live on the chain or
    on a rigid core. r0 is the equilibrium separation, and k_spring is the
    harmonic force constant when no tabulated potential applies. type_idx
    selects a tabulated potential when it is nonnegative, and -1 means use
    the harmonic form."""

    a: AtomRef
    b: AtomRef
    r0: float
    k_spring: float
    type_idx: int = -1


@dataclass
class ChainAngle:
    """Three-body angle interaction. b is the central atom, and the angle is
    measured at b between the vectors from b to a and from b to c. theta0 is
    the equilibrium angle in radians, and k_angle is the harmonic force
    constant when no tabulated potential applies. type_idx selects a tabulated
    potential when it is nonnegative, and -1 means use the harmonic form."""

    a: AtomRef
    b: AtomRef
    c: AtomRef
    theta0: float
    k_angle: float
    type_idx: int = -1


@dataclass
class ChainTorsion:
    """Four-body torsion (dihedral). The dihedral angle phi is measured
    around the b-c bond, from the plane (a, b, c) to the plane (b, c, d).
    When no tabulated potential applies, the energy takes the harmonic-cosine
    form V(phi) = k_tor (1 - cos(n phi - phi0)). type_idx selects a tabulated
    potential when it is nonnegative, and -1 means use the harmonic-cosine
    form."""

    a: AtomRef
    b: AtomRef
    c: AtomRef
    d: AtomRef
    phi0: float
    k_tor: float
    n: int
    type_idx: int = -1


# Chain state


@dataclass
class FlexibleChain:
    """State of a COFFDROP flexible chain.

    Legacy type. Mixes immutable topology (bonds, angles) with mutable
    per-trajectory state (bead positions and forces) in a single object.
    New code should split these into ChainCommon (topology) and
    ChainState (per-trajectory state). This class will be removed once
    the last consumer is rewritten."""

    beads: List[ChainBead]
    bonds: List[ChainBond] = field(default_factory=list)
    angles: List[ChainAngle] = field(default_factory=list)
    torsions: List[ChainTorsion] = field(default_factory=list)
    name: str = ""
    frozen: bool = False  # if True, chain doesn't move

    @property
    def n_beads(self) -> int:
        return len(self.beads)

    def positions_array(self) -> np.ndarray:
        return np.array([b.pos for b in self.beads])

    def forces_array(self) -> np.ndarray:
        return np.array([b.force for b in self.beads])

    def set_positions(self, pos: np.ndarray):
        for i, b in enumerate(self.beads):
            b.pos = pos[i].copy()

    def zero_forces(self):
        for b in self.beads:
            b.force = np.zeros(3)


# Force evaluation


class ChainForceEvaluator:
    """Evaluates all bonded and non-bonded forces on a flexible chain."""

    def compute_forces(self, chain: FlexibleChain, kT: float = KBT_KCAL) -> np.ndarray:
        """Compute all forces on the chain beads.

        Returns an (n_beads, 3) force array in kBT/Å units.
        """
        n = chain.n_beads
        F = np.zeros((n, 3))
        # Harmonic bond forces.
        for bond in chain.bonds:
            F += self._bond_force(chain, bond, kT)
        # Harmonic angle forces.
        for angle in chain.angles:
            F += self._angle_force(chain, angle, kT)
        # Periodic torsion forces.
        for tor in chain.torsions:
            F += self._torsion_force(chain, tor, kT)
        # Non-bonded excluded volume (soft sphere).
        F += self._excluded_volume_forces(chain, kT)
        return F

    def _bond_force(
        self, chain: FlexibleChain, bond: ChainBond, kT: float
    ) -> np.ndarray:
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[bond.a.atom_idx].pos
        rj = chain.beads[bond.b.atom_idx].pos
        dr = rj - ri
        r = float(np.linalg.norm(dr))
        if r < 1e-8:
            return F
        r_hat = dr / r
        f_mag = -bond.k_spring * (r - bond.r0)  # kBT/A
        F[bond.a.atom_idx] -= f_mag * r_hat
        F[bond.b.atom_idx] += f_mag * r_hat
        return F

    def _angle_force(
        self, chain: FlexibleChain, angle: ChainAngle, kT: float
    ) -> np.ndarray:
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[angle.a.atom_idx].pos
        rj = chain.beads[angle.b.atom_idx].pos
        rk = chain.beads[angle.c.atom_idx].pos
        u = ri - rj
        v = rk - rj
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-8 or nv < 1e-8:
            return F
        u /= nu
        v /= nv
        cos_t = float(np.dot(u, v))
        cos_t = max(-1.0, min(1.0, cos_t))
        theta = math.acos(cos_t)
        sin_t = math.sin(theta)
        if abs(sin_t) < 1e-8:
            return F
        d_theta = theta - angle.theta0
        # F_a = -(dV/dtheta) * (dtheta/dr_a). For V = 0.5 k (theta - theta0)^2,
        # dV/dtheta = k (theta - theta0). Gradient of theta:
        #   dtheta/dr_a = (cos_t * u_hat - v_hat) / (|u| * sin_t)
        #   dtheta/dr_c = (cos_t * v_hat - u_hat) / (|v| * sin_t)
        # so F_a = -k(theta-theta0) * (cos_t*u_hat - v_hat)/(|u|*sin_t)
        #        = (k(theta-theta0)/sin_t) * (v_hat - cos_t*u_hat) / |u|.
        coeff = angle.k_angle * d_theta / sin_t
        fi = coeff * (v - cos_t * u) / nu
        fk = coeff * (u - cos_t * v) / nv
        F[angle.a.atom_idx] += fi
        F[angle.c.atom_idx] += fk
        F[angle.b.atom_idx] -= fi + fk
        return F

    def _torsion_force(
        self,
        chain: FlexibleChain,
        tor: ChainTorsion,
        kT: float,
        coffdrop_params: "Optional[COFFDROPParams]" = None,
    ) -> np.ndarray:
        # Cosine-series torsion potential acting on four atoms (i, j, k, l),
        #   V(phi) = k_tor (1 - cos(n phi - phi0))
        # The force on each atom is F_a = -(dV/dphi) (dphi/dr_a).
        #
        # phi is the dihedral angle measured around the central bond
        # b2 = r_k - r_j. The kernel below computes phi and its position
        # gradients using backward-mode automatic differentiation by hand,
        # following the line-for-line structure of an established
        # reference implementation. There are two branches. For small
        # |cos phi| it uses phi = acos(cos phi) with the sign taken from r23.
        # For |cos phi| > 0.99 it switches to an asin formulation around an
        # auxiliary vector u3 = b2 x u2 to dodge the acos derivative
        # singularity. The four gradients sum to zero by translational
        # invariance.
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[tor.a.atom_idx].pos
        rj = chain.beads[tor.b.atom_idx].pos
        rk = chain.beads[tor.c.atom_idx].pos
        rl = chain.beads[tor.d.atom_idx].pos

        # Forward pass: bond vectors, plane normals, normalized normals.
        # b1 = r2 - r1 in the reference convention (here rj - ri).
        a13, a14, a15 = rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]
        # b2 = r3 - r2
        a16, a17, a18 = rk[0] - rj[0], rk[1] - rj[1], rk[2] - rj[2]
        # b3 = r4 - r3
        a19, a20, a21 = rl[0] - rk[0], rl[1] - rk[1], rl[2] - rk[2]

        # t1 = b1 x b2
        a22 = a14 * a18 - a15 * a17
        a23 = a15 * a16 - a13 * a18
        a24 = a13 * a17 - a14 * a16
        # t2 = b2 x b3
        a25 = a17 * a21 - a18 * a20
        a26 = a18 * a19 - a16 * a21
        a27 = a16 * a20 - a17 * a19

        a28 = math.sqrt(a22 * a22 + a23 * a23 + a24 * a24)  # |t1|
        a29 = math.sqrt(a25 * a25 + a26 * a26 + a27 * a27)  # |t2|

        if a28 < 1e-12 or a29 < 1e-12:
            return F

        # u1 = t1 / |t1|, u2 = t2 / |t2|
        a30, a31, a32 = a22 / a28, a23 / a28, a24 / a28
        a33, a34, a35 = a25 / a29, a26 / a29, a27 / a29

        # Sign of phi: sign of (u1 x u2) . b2
        c1 = a31 * a35 - a32 * a34
        c2 = a32 * a33 - a30 * a35
        c3 = a30 * a34 - a31 * a33
        sign = c1 * a16 + c2 * a17 + c3 * a18

        a36 = a30 * a33 + a31 * a34 + a32 * a35  # cos(phi) = u1 . u2

        large_cos = abs(a36) > 0.99
        if large_cos:
            # Auxiliary t3 = b2 x u2 for the asin branch.
            a37 = a17 * a35 - a18 * a34
            a38 = a18 * a33 - a16 * a35
            a39 = a16 * a34 - a17 * a33
            a40 = math.sqrt(a37 * a37 + a38 * a38 + a39 * a39)
            if a40 < 1e-12:
                return F
            # u3 = t3 / |t3|
            a41, a42, a43 = a37 / a40, a38 / a40, a39 / a40
            # cos(psi) = u3 . u1
            a44 = a41 * a30 + a42 * a31 + a43 * a32
            if a36 > 0.0:
                phi = -math.asin(a44)
            elif sign > 0.0:
                phi = math.pi + math.asin(a44)
            else:
                phi = -math.pi + math.asin(a44)
        else:
            phi = math.acos(max(-1.0, min(1.0, a36)))
            if sign <= 0.0:
                phi = -phi

        # Backward pass: seed db45 = dphi/dphi = 1, propagate.
        b45 = 1.0

        if large_cos:
            if a36 > 0.0:
                b44 = -b45 / math.sqrt(max(1.0 - a44 * a44, 1e-30))
            else:
                b44 = b45 / math.sqrt(max(1.0 - a44 * a44, 1e-30))

            b41 = b44 * a30
            b42 = b44 * a31
            b43 = b44 * a32

            b30 = b44 * a41
            b31 = b44 * a42
            b32 = b44 * a43

            b37 = b41 / a40
            b38 = b42 / a40
            b39 = b43 / a40
            b40 = -(b41 * a37 + b42 * a38 + b43 * a39) / (a40 * a40)

            b37 += b40 * a37 / a40
            b38 += b40 * a38 / a40
            b39 += b40 * a39 / a40

            # Adjoint of t3 contributing to u2 via the cross product b2 x u2.
            b33 = b38 * a18 - b39 * a17
            b34 = b39 * a16 - b37 * a18
            b35 = b37 * a17 - b38 * a16
        else:
            if sign > 0.0:
                b36 = -b45 / math.sqrt(max(1.0 - a36 * a36, 1e-30))
            else:
                b36 = b45 / math.sqrt(max(1.0 - a36 * a36, 1e-30))

            b30 = b36 * a33
            b33 = b36 * a30
            b31 = b36 * a34
            b34 = b36 * a31
            b32 = b36 * a35
            b35 = b36 * a32

        # Adjoints of u2 = t2 / |t2|.
        b25 = b33 / a29
        b26 = b34 / a29
        b27 = b35 / a29
        b29 = -(b33 * a25 + b34 * a26 + b35 * a27) / (a29 * a29)

        # Adjoints of u1 = t1 / |t1|.
        b22 = b30 / a28
        b23 = b31 / a28
        b24 = b32 / a28
        b28 = -(b30 * a22 + b31 * a23 + b32 * a24) / (a28 * a28)

        # |t1|, |t2| feed back into t1, t2.
        b25 += b29 * a25 / a29
        b26 += b29 * a26 / a29
        b27 += b29 * a27 / a29

        b22 += b28 * a22 / a28
        b23 += b28 * a23 / a28
        b24 += b28 * a24 / a28

        # Adjoints of t2 = b2 x b3 contributing to b2 and b3.
        b16 = a20 * b27 - a21 * b26
        b17 = a21 * b25 - a19 * b27
        b18 = a19 * b26 - a20 * b25
        b19 = b26 * a18 - b27 * a17
        b20 = b27 * a16 - b25 * a18
        b21 = b25 * a17 - b26 * a16

        # Adjoints of t1 = b1 x b2 contributing to b1.
        b13 = a17 * b24 - a18 * b23
        b14 = a18 * b22 - a16 * b24
        b15 = a16 * b23 - a17 * b22

        # b2 also receives contribution from t1.
        b16 += b23 * a15 - b24 * a14
        b17 += b24 * a13 - b22 * a15
        b18 += b22 * a14 - b23 * a13

        # Position gradients dphi/dr_a, with bonds expressed as differences:
        #   b1 = rj - ri  => r_i contributes -b1, r_j contributes +b1, ...
        dphi_drl = np.array([b19, b20, b21])
        dphi_drk = np.array([b16 - b19, b17 - b20, b18 - b21])
        dphi_drj = np.array([b13 - b16, b14 - b17, b15 - b18])
        dphi_dri = np.array([-b13, -b14, -b15])

        # Apply the potential through F = -(dV/dphi) dphi/dr. There are two
        # modes. The default cosine-series mode uses
        # V(phi) = k_tor (1 - cos(n phi - phi0)), so that
        # dV/dphi = k_tor n sin(n phi - phi0) in radians. The tabulated
        # COFFDROP mode applies when tor.type_idx >= 0 and coffdrop_params is
        # not None. It looks up dV/dphi in kBT per degree from the spline at
        # index type_idx and converts to radians via
        # dV/dphi_rad = dV/dphi_deg × (180/pi). The backward-AD math above
        # for dphi/dr_a is unchanged, and only the force-field-dependent
        # dV/dphi changes.
        if tor.type_idx >= 0 and coffdrop_params is not None:
            phi_deg = phi * 180.0 / math.pi
            # Wrap to [0, 360) since dihedral tables are typically
            # parameterized over that range.
            phi_deg = phi_deg % 360.0
            pot = coffdrop_params.dihedral_pots[tor.type_idx]
            dV_deg = pot.deriv(phi_deg)
            dV_dphi = dV_deg * (180.0 / math.pi)
        else:
            dV_dphi = tor.k_tor * tor.n * math.sin(tor.n * phi - tor.phi0)

        F[tor.a.atom_idx] += -dV_dphi * dphi_dri
        F[tor.b.atom_idx] += -dV_dphi * dphi_drj
        F[tor.c.atom_idx] += -dV_dphi * dphi_drk
        F[tor.d.atom_idx] += -dV_dphi * dphi_drl
        return F

    def _excluded_volume_forces(self, chain: FlexibleChain, kT: float) -> np.ndarray:
        """Soft-sphere excluded volume between non-bonded bead pairs."""
        n = chain.n_beads
        F = np.zeros((n, 3))
        bonded_pairs = {(b.a.atom_idx, b.b.atom_idx) for b in chain.bonds} | {
            (b.b.atom_idx, b.a.atom_idx) for b in chain.bonds
        }
        for i in range(n):
            for j in range(i + 2, n):  # skip bonded neighbours
                if (i, j) in bonded_pairs:
                    continue
                ri = chain.beads[i].pos
                rj = chain.beads[j].pos
                dr = (
                    ri - rj
                )  # vector from j toward i (so f_mag*dr pushes i away from j)
                r = float(np.linalg.norm(dr))
                sig = chain.beads[i].radius + chain.beads[j].radius
                if r < 1e-8 or r >= sig:
                    continue
                # WCA-style repulsion. The WCA potential is
                # V = 4 eps [(sig/r)^12 - (sig/r)^6] + eps, so dV/dr carries a
                # leading factor of 4. Legacy code missed this and produced
                # forces that were too small by a factor of 4.
                sr = sig / r
                sr12 = sr**12
                sr6 = sr**6
                eps = 1.0  # kBT units
                f_mag = 4.0 * eps * (12 * sr12 - 6 * sr6) / (r * r)
                fvec = f_mag * dr
                F[i] += fvec
                F[j] -= fvec
        return F


# BD propagation for chain


class ChainBDPropagator:
    """Brownian dynamics propagator for a flexible chain.

    Each bead moves independently with its own translational diffusion
    coefficient D_trans = kT / (6 π η a), where a is the bead radius. There is
    no hydrodynamic coupling between beads.
    """

    def __init__(self, kT: float = KBT_KCAL, viscosity: float = VISCOSITY):
        self.kT = kT
        # Solvent viscosity in kcal/mol.ps/A^3, not Pa.s. D_trans = kT/(6 pi eta a)
        # and the diffusive noise sqrt(2 D dt) both depend on it directly.
        self.eta = viscosity
        self._evaluator = ChainForceEvaluator()

    def D_trans(self, radius: float) -> float:
        """Stokes-Einstein translational diffusion (A^2/ps)."""
        return self.kT / (6.0 * math.pi * self.eta * radius)

    def step(
        self,
        chain: FlexibleChain,
        dt: float,
        rng: np.random.Generator,
        force_evaluator=None,
    ) -> FlexibleChain:
        """Advance the chain by one Brownian dynamics step of size dt in ps.

        Each bead is displaced by a deterministic drift plus a random kick,

            dpos  = mob × f × dt
            wdpos = sqrt(2 kT mob) × dW

        where mob is the bead mobility and dW is a standard Wiener increment.
        force_evaluator is an optional external object exposing a
        compute_forces(chain) method. If it is None, the internal harmonic
        ChainForceEvaluator is used.
        """
        if chain.frozen:
            return chain
        # Compute forces, using the external evaluator if one is provided.
        if force_evaluator is not None:
            forces = force_evaluator.compute_forces(chain)
        else:
            forces = self._evaluator.compute_forces(chain, self.kT)
        for i, b in enumerate(chain.beads):
            b.force = forces[i]
        # Propagate each bead
        for i, b in enumerate(chain.beads):
            mob = 1.0 / (6.0 * math.pi * self.eta * b.radius)  # A^3/(kBT*ps)
            # Deterministic drift, dpos = mob × F × dt.
            drift = mob * b.force * dt
            # Stochastic kick, wdpos = sqrt(2 kT mob) × dW.
            sigma = math.sqrt(2.0 * self.kT * mob * dt)
            noise = sigma * rng.standard_normal(3)
            b.pos += drift + noise
        return chain

    def max_time_step(self, chain: FlexibleChain) -> float:
        """Geometry-based maximum time step for the chain.

        It uses the smallest bead radius and the estimate dt ≈ R^2 / D.
        """
        if not chain.beads:
            return 0.1
        min_R = min(b.radius for b in chain.beads)
        D_max = self.D_trans(min_R)
        # The fuller 4 R^3 / D_factor estimate is simplified to R^2 / D here.
        return min_R**2 / D_max if D_max > 0 else 0.001

    def satisfy_bond_constraints(
        self, chain: FlexibleChain, tol: float = 1e-4, max_iter: int = 100
    ):
        """Bond constraint satisfaction in the style of RATTLE."""
        for _ in range(max_iter):
            max_viol = 0.0
            for bond in chain.bonds:
                ri = chain.beads[bond.a.atom_idx].pos
                rj = chain.beads[bond.b.atom_idx].pos
                dr = rj - ri
                r = float(np.linalg.norm(dr))
                if r < 1e-8:
                    continue
                viol = abs(r - bond.r0) / bond.r0
                max_viol = max(max_viol, viol)
                if viol > tol:
                    # Project back to constraint surface
                    correction = 0.5 * (r - bond.r0) * dr / r
                    chain.beads[bond.a.atom_idx].pos += correction
                    chain.beads[bond.b.atom_idx].pos -= correction
            if max_viol < tol:
                break


# Simple chain builder
def build_linear_chain(
    n_residues: int,
    bead_radius: float = 2.0,
    bead_charge: float = 0.0,
    bond_length: float = 3.8,
    start_pos: Optional[np.ndarray] = None,
) -> FlexibleChain:
    """Build a simple linear chain of n_residues beads.

    This is useful for testing. Production use should load the chain from
    COFFDROP XML instead.
    """
    if start_pos is None:
        start_pos = np.zeros(3)
    beads = []
    for i in range(n_residues):
        pos = start_pos + np.array([i * bond_length, 0.0, 0.0])
        beads.append(
            ChainBead(
                pos=pos,
                force=np.zeros(3),
                radius=bead_radius,
                charge=bead_charge,
                resname="UNK",
                resid=i,
            )
        )
    bonds = [
        ChainBond(
            a=ChainAtomRef(i), b=ChainAtomRef(i + 1), r0=bond_length, k_spring=100.0
        )
        for i in range(n_residues - 1)
    ]
    return FlexibleChain(beads=beads, bonds=bonds, name="chain")


def build_chain_from_coffdrop(
    residues: List[str], params, start_pos: Optional[np.ndarray] = None
) -> "FlexibleChain":
    """Build a FlexibleChain from a sequence of residue names.

    The equilibrium bond lengths and charges come from the COFFDROP parameter
    files. residues is a list of 3-letter residue names such as
    ['ALA', 'GLY', 'ARG'], params is a COFFDROPParams loaded from the XML
    files, and start_pos is the starting position of the first bead (default
    [0, 0, 0]). The function returns a FlexibleChain with beads, bonds, and
    charges taken from the COFFDROP data files.
    """
    if start_pos is None:
        start_pos = np.zeros(3)
    beads = []
    pos = start_pos.copy()
    for i, resname in enumerate(residues):
        # Get charge from charges.xml (CA bead is backbone, usually neutral)
        charge = params.bead_charge(resname, "CA")
        bead = ChainBead(
            pos=pos.copy(),
            force=np.zeros(3),
            radius=2.0,  # typical CA Stokes radius
            charge=charge,
            resname=resname,
            resid=i,
        )
        beads.append(bead)
        # Advance position by CA-CA backbone bond length
        ca_ca_len = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
        pos = pos + np.array([ca_ca_len, 0.0, 0.0])
    # Build bonds using equilibrium lengths from connectivity.xml
    bonds = []
    for i in range(len(residues) - 1):
        r0 = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
        bonds.append(
            ChainBond(a=ChainAtomRef(i), b=ChainAtomRef(i + 1), r0=r0, k_spring=100.0)
        )
    return FlexibleChain(beads=beads, bonds=bonds, name="-".join(residues[:3]) + "...")


def build_chain_common_from_coffdrop(
    residues: List[str],
    params,
    name: Optional[str] = None,
    k_spring: float = 100.0,
) -> "ChainCommon":
    """Build a COFFDROP-aware ChainCommon from a peptide sequence.

    This populates the type_idx fields on every angle and torsion and builds
    pair_lookups for every non-bonded chain pair. It returns a chain ready to
    use the tabulated force branches in compute_chain_forces.

    residues is a list of 3-letter residue names such as ['ALA', 'GLY',
    'ARG'], params is a COFFDROPParams loaded from the XML files, name is an
    optional chain name that defaults to a hyphen-joined prefix, and k_spring
    is the harmonic bond spring constant in kBT/Å^2. COFFDROP has no tabulated
    bond potentials, so the bonds remain harmonic.

    The returned ChainCommon has one CA bead per residue, with resname
    "RESNAME:CA" following the "RESNAME:BEADTYPE" naming convention. Its
    bonds form a linear backbone (i, i+1) with equilibrium lengths from
    connectivity.xml. Its angles are every triplet (i, i+1, i+2), with
    type_idx looked up via the forward convention residues=(XXX, R_{i+1},
    R_{i+2}) and orders=(1, 2, 3). Its torsions are every quadruplet (i, i+1,
    i+2, i+3), with type_idx via the forward convention residues=(R_i, R_{i+1},
    R_{i+2}, R_{i+3}) and orders=(1, 1, 1, 1). Its pair_lookups cover every
    non-bonded pair (i, j) with j >= i+2, with type_idx looked up by
    residues=(R_i, R_j), atoms=(CA, CA), and orders=(0, 0). Both
    coffdrop_params and pair_lookups are set.

    The angle and torsion lookup convention is a documented assumption (the
    forward convention with orders=(1, 2, 3) and (1, 1, 1, 1)). Heteropolymer
    chains may need the alternate backward convention if forces appear
    incorrect during validation. This helper is intentionally minimal, a
    CA-only backbone for testing the tabulated machinery end to end.
    Sidechain beads (CB, NG, and so on) are not added.
    """
    from pystarc.simulation.coffdrop_params import _match_pot

    if not residues:
        raise ValueError("residues list cannot be empty")
    n = len(residues)

    # Atom-type lookup for CA.
    ca_atom_idx = params.type_map["atoms"].get("CA")
    if ca_atom_idx is None:
        raise ValueError("CA bead type not found in params.type_map['atoms']")

    # Residue-type indices.
    def _ri(rname):
        return params.type_map["residues"].get(rname, 0)  # 0 = XXX wildcard

    # Build the atoms, one CA per residue. The resname follows the
    # "RESNAME:CA" naming convention.
    atoms = []
    for i, rname in enumerate(residues):
        charge = params.bead_charge(rname, "CA")
        atoms.append(
            ChainAtom(
                radius=2.0,  # typical CA Stokes radius (A)
                charge=charge,
                resname=f"{rname}:CA",
                resid=i,
            )
        )

    # Build the bonds as a linear backbone, with lengths from connectivity.xml.
    bonds = []
    for i in range(n - 1):
        r0 = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
        bonds.append(
            ChainBond(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                r0=r0,
                k_spring=k_spring,
                type_idx=-1,  # bonds are harmonic, no tabulated lookup
            )
        )

    # Build the angles, one per consecutive triplet, using the forward
    # convention residues=(XXX, R_central, R_next) and orders=(1, 2, 3).
    angles = []
    for i in range(n - 2):
        # The central atom is residue i+1 and the next is i+2.
        central = residues[i + 1]
        nxt = residues[i + 2]
        ri = (0, _ri(central), _ri(nxt))  # 0 = XXX
        ai = (ca_atom_idx, ca_atom_idx, ca_atom_idx)
        orders = (1, 2, 3)
        pot = _match_pot(params.angle_pots, ri, ai, orders)
        type_idx = pot.index if pot is not None else -1
        angles.append(
            ChainAngle(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                c=ChainAtomRef(i + 2),
                theta0=2.0,  # placeholder, ignored when type_idx >= 0
                k_angle=10.0,
                type_idx=type_idx,
            )
        )

    # Build the torsions, one per consecutive quadruplet. For a backbone
    # CA-CA-CA-CA, the diagnostic showed that all 441 such dihedrals use
    # orders=(1, 2, 3, 4) with residues=(XXX, R_{i+1}, R_{i+2}, XXX), so only
    # the two middle residues matter for the lookup.
    torsions = []
    for i in range(n - 3):
        # Only the two middle residues distinguish the lookup.
        ri = (0, _ri(residues[i + 1]), _ri(residues[i + 2]), 0)  # 0 = XXX
        ai = (ca_atom_idx,) * 4
        orders = (1, 2, 3, 4)
        pot = _match_pot(params.dihedral_pots, ri, ai, orders)
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                c=ChainAtomRef(i + 2),
                d=ChainAtomRef(i + 3),
                phi0=0.0,
                k_tor=2.0,
                n=1,  # placeholders, ignored when type_idx >= 0
                type_idx=type_idx,
            )
        )

    # Build pair_lookups for every non-bonded pair (i, j) with j >= i+3. The
    # 1-2 bonded pair (i, i+1) and the 1-3 angle pair (i, i+2) are excluded so
    # the 1-3 interaction is not double-counted with the angle term; this
    # matches the sidechain builder, which excludes 1-3 pairs explicitly. The
    # pair potentials use orders=(0, 0), the residues come from the actual chain
    # residues, and the atoms are (CA, CA).
    pair_lookups = {}
    for i in range(n):
        for j in range(i + 3, n):
            ri = (_ri(residues[i]), _ri(residues[j]))
            ai = (ca_atom_idx, ca_atom_idx)
            orders = (0, 0)
            pot = _match_pot(params.pair_pots, ri, ai, orders)
            if pot is None:
                # Try reversed (symmetric pair lookup).
                pot = _match_pot(
                    params.pair_pots, (ri[1], ri[0]), (ai[1], ai[0]), orders
                )
            if pot is not None:
                pair_lookups[(i, j)] = pot.index

    return ChainCommon(
        name=name or ("-".join(residues[:3]) + ("..." if n > 3 else "")),
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        coffdrop_params=params,
        pair_lookups=pair_lookups,
    )


def build_chain_common_with_sidechains_from_coffdrop(
    residues: List[str],
    params,
    name: Optional[str] = None,
    k_spring: float = 100.0,
    caps: Tuple[Optional[str], Optional[str]] = (None, None),
) -> "ChainCommon":
    """Build a COFFDROP-aware ChainCommon with sidechain beads.

    Each residue contributes a CA bead together with its sidechain beads (for
    example ALA has 2, ARG has 3, TRP has 4, and GLY has 1). Backbone CA-CA
    bonds connect consecutive residues, and intra-residue bonds connect the
    linear sidechain chain CA to CB to CG and onward. Equilibrium lengths come
    from connectivity.xml.

    residues is a list of 3-letter residue names, params is a COFFDROPParams,
    name is an optional chain name, and k_spring is the harmonic spring
    constant for the bonds in kBT/Å^2.

    The returned ChainCommon has its atoms laid out per residue (CA plus
    sidechain) and flattened across residues. Its bonds are the backbone CA-CA
    bonds together with the intra-residue sidechain bonds. Its angles are the
    CA-CA-CA backbone angles only, since the sidechain angles require further
    topology work as described below. Its torsions are the CA-CA-CA-CA
    backbone torsions only. Its pair_lookups cover all non-bonded pairs (i, j)
    with j > i+1, using the actual bead types for the lookup. The result is
    ready for compute_chain_forces with the sidechain pair forces enabled.

    The sidechain angle and torsion topology is not yet populated. This phase
    ships the sidechain pair forces, which are the dominant inter-residue
    interaction in COFFDROP. Sidechain angles such as CA-CB-CG would require
    more lookup-convention discovery, similar to what was done for the
    backbone CA-CA-CA case, and are left for future work.

    Bead naming follows the "RESNAME:BEADTYPE" convention, encoding both the
    residue name and the bead type in each atom's resname field.
    """
    from pystarc.simulation.coffdrop_params import _match_pot

    if not residues:
        raise ValueError("residues list cannot be empty")
    n_res = len(residues)

    # Validate the caps. The accepted combinations are ("ACE", "NME"),
    # ("ACE", None), (None, "NME"), and (None, None). Other values raise.
    n_cap, c_cap = caps
    if n_cap not in (None, "ACE"):
        raise ValueError(f"N-terminal cap must be 'ACE' or None, got {n_cap!r}")
    if c_cap not in (None, "NME"):
        raise ValueError(f"C-terminal cap must be 'NME' or None, got {c_cap!r}")

    type_map_atoms = params.type_map["atoms"]
    type_map_residues = params.type_map["residues"]

    def _ri(rname):
        return type_map_residues.get(rname, 0)

    def _ai(beadname):
        return type_map_atoms.get(beadname, -1)

    # Build the flat atom list. For each global atom index we track its
    # residue position (which residue r in 0..n_res-1) and its bead name
    # within the residue (for example "CA", "CB", or "NG").
    atoms = []
    atom_residue = []  # atom_residue[global_idx] = residue position r
    atom_beadname = []  # atom_beadname[global_idx] = bead name string
    # ca_global_idx[r] = global atom index of the CA bead in residue r.
    # This is used to wire backbone CA-CA bonds, angles, and torsions.
    ca_global_idx = []
    # For intra-residue bonds, track bead indices per residue.
    # residue_beads[r] = [global_idx_of_CA, global_idx_of_CB, ...]
    residue_beads = []

    for r, resname in enumerate(residues):
        # Get the sidechain beads for this residue, excluding caps and termini.
        rdef = params.mapping.get(resname)
        if rdef is None:
            raise ValueError(f"residue {resname} not in params.mapping")
        sidechain_names = [
            b.name
            for b in rdef.beads
            if b.btype == "" and b.location == "" and b.name != "CA"
        ]
        # Linear chain within the residue, running CA to CB and on to the last.
        bead_names_this_res = ["CA"] + sidechain_names
        beads_in_res = []
        for bname in bead_names_this_res:
            global_idx = len(atoms)
            charge = params.bead_charge(resname, bname)
            atoms.append(
                ChainAtom(
                    radius=2.0,  # uniform Stokes radius
                    charge=charge,
                    resname=f"{resname}:{bname}",
                    resid=r,
                )
            )
            atom_residue.append(r)
            atom_beadname.append(bname)
            beads_in_res.append(global_idx)
        residue_beads.append(beads_in_res)
        ca_global_idx.append(beads_in_res[0])  # CA is always first

    # Cap atoms. ACE adds a CN bead bonded to the first CA, and NME adds a CC
    # bead bonded to the last CA. Cap beads carry charge 0.0 because
    # charges.xml has no entries for CN or CC.
    cn_global_idx = -1
    cc_global_idx = -1
    if n_cap == "ACE":
        cn_global_idx = len(atoms)
        atoms.append(
            ChainAtom(
                radius=2.0,
                charge=0.0,
                resname="ACE:CN",
                resid=-1,  # caps have no residue position, so -1 marks a cap
            )
        )
        atom_residue.append(-1)
        atom_beadname.append("CN")
    if c_cap == "NME":
        cc_global_idx = len(atoms)
        atoms.append(
            ChainAtom(
                radius=2.0,
                charge=0.0,
                resname="NME:CC",
                resid=-1,
            )
        )
        atom_residue.append(-1)
        atom_beadname.append("CC")

    n_atoms = len(atoms)

    # Build the bonds.
    bonds = []
    # Backbone CA-CA bonds between consecutive residues.
    ca_ca_len = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
    for r in range(n_res - 1):
        bonds.append(
            ChainBond(
                a=ChainAtomRef(ca_global_idx[r]),
                b=ChainAtomRef(ca_global_idx[r + 1]),
                r0=ca_ca_len,
                k_spring=k_spring,
                type_idx=-1,
            )
        )
    # Cap bonds. ACE adds a CA(0)-CN bond and NME adds a CA(-1)-CC bond. The
    # equilibrium lengths from connectivity.xml are CA-CN = 3.81 and
    # CA-CC = 3.82.
    if cn_global_idx >= 0:
        ca_cn_len = params.bond_length("XXX", "CA", 1, "XXX", "CN", 1) or 3.81
        bonds.append(
            ChainBond(
                a=ChainAtomRef(cn_global_idx),
                b=ChainAtomRef(ca_global_idx[0]),
                r0=ca_cn_len,
                k_spring=k_spring,
                type_idx=-1,
            )
        )
    if cc_global_idx >= 0:
        ca_cc_len = params.bond_length("XXX", "CA", 1, "XXX", "CC", 1) or 3.82
        bonds.append(
            ChainBond(
                a=ChainAtomRef(ca_global_idx[-1]),
                b=ChainAtomRef(cc_global_idx),
                r0=ca_cc_len,
                k_spring=k_spring,
                type_idx=-1,
            )
        )
    # Intra-residue sidechain bonds, connecting bead i to bead i+1 within the
    # residue.
    for r in range(n_res):
        beads = residue_beads[r]
        if len(beads) < 2:
            continue  # GLY has only CA, so no intra-residue bonds
        resname = residues[r]
        for k in range(len(beads) - 1):
            bead_a_name = atom_beadname[beads[k]]
            bead_b_name = atom_beadname[beads[k + 1]]
            r0 = (
                params.bond_length(resname, bead_a_name, 1, resname, bead_b_name, 1)
                or 3.0
            )
            bonds.append(
                ChainBond(
                    a=ChainAtomRef(beads[k]),
                    b=ChainAtomRef(beads[k + 1]),
                    r0=r0,
                    k_spring=k_spring,
                    type_idx=-1,
                )
            )

    # Build the angles. There are three types. The first is the CA-CA-CA
    # backbone angle, using the forward convention orders=(1, 2, 3) and
    # residues=(XXX, R_central, R_next). The second is the CB-CA-CA
    # sidechain-backbone angle, the most common sidechain angle with 858
    # potentials in COFFDROP, where each residue r with a CB contributes two
    # angles, CB(r)-CA(r)-CA(r+1) and CB(r)-CA(r)-CA(r-1). The third is the
    # CA-CB-CG intra-residue angle (9 potentials, for the residues with a CG,
    # namely LEU, ILE, ASN, GLN, MET, ASP, GLU, PHE, TYR, and TRP).
    angles = []
    ca_idx = _ai("CA")
    cb_idx = _ai("CB")
    cg_idx = _ai("CG")

    # Backbone CA-CA-CA angles.
    for r in range(n_res - 2):
        ri = (0, _ri(residues[r + 1]), _ri(residues[r + 2]))
        ai = (ca_idx, ca_idx, ca_idx)
        pot = _match_pot(params.angle_pots, ri, ai, (1, 2, 3))
        type_idx = pot.index if pot is not None else -1
        angles.append(
            ChainAngle(
                a=ChainAtomRef(ca_global_idx[r]),
                b=ChainAtomRef(ca_global_idx[r + 1]),
                c=ChainAtomRef(ca_global_idx[r + 2]),
                theta0=2.0,
                k_angle=10.0,
                type_idx=type_idx,
            )
        )

    # SB (or CB) to CA to CA sidechain-backbone angles. For each residue with
    # a first sidechain bead (CB for most residues, SB for CYS), the forward
    # angle is SB(r)-CA(r)-CA(r+1) with orders=(2, 2, 3) and the backward angle
    # is SB(r)-CA(r)-CA(r-1) with orders=(3, 3, 2). The lookup uses the actual
    # atom-type index of that bead.
    for r in range(n_res):
        # The first sidechain bead is the first non-CA bead in the residue.
        sb_global = None
        sb_type_idx = None
        for bead_global in residue_beads[r]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sb_global = bead_global
                sb_type_idx = _ai(bn)
                break
        if sb_global is None:
            continue
        # Forward: SB(r)-CA(r)-CA(r+1)
        if r + 1 < n_res:
            ri = (_ri(residues[r]), _ri(residues[r]), _ri(residues[r + 1]))
            ai = (sb_type_idx, ca_idx, ca_idx)
            pot = _match_pot(params.angle_pots, ri, ai, (2, 2, 3))
            type_idx = pot.index if pot is not None else -1
            angles.append(
                ChainAngle(
                    a=ChainAtomRef(sb_global),
                    b=ChainAtomRef(ca_global_idx[r]),
                    c=ChainAtomRef(ca_global_idx[r + 1]),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=type_idx,
                )
            )
        # Backward: SB(r)-CA(r)-CA(r-1)
        if r - 1 >= 0:
            ri = (_ri(residues[r]), _ri(residues[r]), _ri(residues[r - 1]))
            ai = (sb_type_idx, ca_idx, ca_idx)
            pot = _match_pot(params.angle_pots, ri, ai, (3, 3, 2))
            type_idx = pot.index if pot is not None else -1
            angles.append(
                ChainAngle(
                    a=ChainAtomRef(sb_global),
                    b=ChainAtomRef(ca_global_idx[r]),
                    c=ChainAtomRef(ca_global_idx[r - 1]),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=type_idx,
                )
            )

    # Cap-flanking backbone angles. For ACE the angle is CN-CA(0)-CA(1) with
    # residues=(r0, r0, r1) and orders=(1, 1, 2). For NME it is
    # CA(n-2)-CA(n-1)-CC with residues=(r_{n-2}, r_{n-1}, r_{n-1}) and
    # orders=(1, 2, 2).
    cn_idx = _ai("CN")
    cc_idx = _ai("CC")
    if cn_global_idx >= 0 and n_res >= 2:
        ri = (_ri(residues[0]), _ri(residues[0]), _ri(residues[1]))
        ai = (cn_idx, ca_idx, ca_idx)
        pot = _match_pot(params.angle_pots, ri, ai, (1, 1, 2))
        type_idx = pot.index if pot is not None else -1
        angles.append(
            ChainAngle(
                a=ChainAtomRef(cn_global_idx),
                b=ChainAtomRef(ca_global_idx[0]),
                c=ChainAtomRef(ca_global_idx[1]),
                theta0=2.0,
                k_angle=10.0,
                type_idx=type_idx,
            )
        )
    if cc_global_idx >= 0 and n_res >= 2:
        ri = (
            _ri(residues[n_res - 2]),
            _ri(residues[n_res - 1]),
            _ri(residues[n_res - 1]),
        )
        ai = (ca_idx, ca_idx, cc_idx)
        pot = _match_pot(params.angle_pots, ri, ai, (1, 2, 2))
        type_idx = pot.index if pot is not None else -1
        angles.append(
            ChainAngle(
                a=ChainAtomRef(ca_global_idx[n_res - 2]),
                b=ChainAtomRef(ca_global_idx[n_res - 1]),
                c=ChainAtomRef(cc_global_idx),
                theta0=2.0,
                k_angle=10.0,
                type_idx=type_idx,
            )
        )

    # Cap-sidechain angle, on the NME side only. The angle is
    # SC1(n-1)-CA(n-1)-CC with residues=(r_{n-1}, r_{n-1}, r_{n-1}) and
    # orders=(2, 2, 2). SC1 is the first sidechain bead (CB for most residues,
    # SB for CYS). GLY has no SC1, so this angle is absent for a GLY-flanked
    # NME. On the ACE side COFFDROP has no CN-CA-SC1 angle entries, so it is
    # skipped.
    if cc_global_idx >= 0 and n_res >= 1:
        last_r = n_res - 1
        # Find first sidechain bead of last residue
        sc1_global = None
        sc1_type_idx = None
        for bead_global in residue_beads[last_r]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sc1_global = bead_global
                sc1_type_idx = _ai(bn)
                break
        if sc1_global is not None:
            ri = (_ri(residues[last_r]),) * 3
            ai = (sc1_type_idx, ca_idx, cc_idx)
            pot = _match_pot(params.angle_pots, ri, ai, (2, 2, 2))
            type_idx = pot.index if pot is not None else -1
            angles.append(
                ChainAngle(
                    a=ChainAtomRef(sc1_global),
                    b=ChainAtomRef(ca_global_idx[last_r]),
                    c=ChainAtomRef(cc_global_idx),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=type_idx,
                )
            )

    # Intra-residue sidechain angles, with two patterns. The first is
    # CA-SC1-SC2 for any residue with at least 2 sidechain beads, which covers
    # (CA, CB, CG) for LEU, ILE, MET and similar residues, (CA, CB, NG) for
    # ARG, LYS, HIS, and HIP, and (CA, CB, OG) for ASP and GLU, all with
    # orders=(3, 3, 3) and residues=(R, R, R). The second is SC1-SC2-SC3,
    # which occurs only for TRP, the one residue with 3 sidechain beads, and
    # covers (CB, CG, CD) with orders=(3, 3, 3) and residues=(R, R, R).
    for r in range(n_res):
        beads_here = residue_beads[r]
        # Pattern (a), CA-SC1-SC2.
        if len(beads_here) >= 3:
            sc1_global = beads_here[1]
            sc2_global = beads_here[2]
            sc1_type = _ai(atom_beadname[sc1_global])
            sc2_type = _ai(atom_beadname[sc2_global])
            ri = (_ri(residues[r]),) * 3
            ai = (ca_idx, sc1_type, sc2_type)
            pot = _match_pot(params.angle_pots, ri, ai, (3, 3, 3))
            type_idx = pot.index if pot is not None else -1
            angles.append(
                ChainAngle(
                    a=ChainAtomRef(ca_global_idx[r]),
                    b=ChainAtomRef(sc1_global),
                    c=ChainAtomRef(sc2_global),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=type_idx,
                )
            )
        # Pattern (b), SC1-SC2-SC3 (the TRP CB-CG-CD angle).
        if len(beads_here) >= 4:
            sc1_global = beads_here[1]
            sc2_global = beads_here[2]
            sc3_global = beads_here[3]
            sc1_type = _ai(atom_beadname[sc1_global])
            sc2_type = _ai(atom_beadname[sc2_global])
            sc3_type = _ai(atom_beadname[sc3_global])
            ri = (_ri(residues[r]),) * 3
            ai = (sc1_type, sc2_type, sc3_type)
            pot = _match_pot(params.angle_pots, ri, ai, (3, 3, 3))
            type_idx = pot.index if pot is not None else -1
            angles.append(
                ChainAngle(
                    a=ChainAtomRef(sc1_global),
                    b=ChainAtomRef(sc2_global),
                    c=ChainAtomRef(sc3_global),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=type_idx,
                )
            )

    # Build the torsions. There are three types here: the CA-CA-CA-CA backbone
    # torsions, the CA-CA-CA-CB "incoming sidechain" torsions (1875 potentials,
    # the dominant type), and the CB-CA-CA-CA "outgoing sidechain" torsions
    # (849 potentials). The 47 other dihedral types involving caps, termini,
    # CG, NG, and OG are deferred. The CB-only sidechain dihedrals capture the
    # dominant physics for standard residues.
    torsions = []

    # Backbone CA-CA-CA-CA torsions.
    for r in range(n_res - 3):
        ri = (0, _ri(residues[r + 1]), _ri(residues[r + 2]), 0)
        ai = (ca_idx,) * 4
        pot = _match_pot(params.dihedral_pots, ri, ai, (1, 2, 3, 4))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(ca_global_idx[r]),
                b=ChainAtomRef(ca_global_idx[r + 1]),
                c=ChainAtomRef(ca_global_idx[r + 2]),
                d=ChainAtomRef(ca_global_idx[r + 3]),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )

    # "Incoming sidechain" torsion CA(r)-CA(r+1)-CA(r+2)-SB(or CB)(r+2). This
    # is generalized from CB-only to any first sidechain bead so that CYS
    # works. It uses residues=(XXX, R_{r+1}, R_{r+2}, R_{r+2}) and
    # orders=(1, 3, 2, 2).
    for r in range(n_res - 2):
        sb_global = None
        sb_type_idx = None
        for bead_global in residue_beads[r + 2]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sb_global = bead_global
                sb_type_idx = _ai(bn)
                break
        if sb_global is None:
            continue
        ri = (0, _ri(residues[r + 1]), _ri(residues[r + 2]), _ri(residues[r + 2]))
        ai = (ca_idx, ca_idx, ca_idx, sb_type_idx)
        pot = _match_pot(params.dihedral_pots, ri, ai, (1, 3, 2, 2))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(ca_global_idx[r]),
                b=ChainAtomRef(ca_global_idx[r + 1]),
                c=ChainAtomRef(ca_global_idx[r + 2]),
                d=ChainAtomRef(sb_global),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )

    # "Outgoing sidechain" torsion SB(or CB)(r)-CA(r)-CA(r+1)-CA(r+2). This is
    # generalized from CB-only to any first sidechain bead so that CYS works.
    # It uses residues=(R_r, R_r, R_{r+1}, XXX) and orders=(3, 3, 2, 1).
    for r in range(n_res - 2):
        sb_global = None
        sb_type_idx = None
        for bead_global in residue_beads[r]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sb_global = bead_global
                sb_type_idx = _ai(bn)
                break
        if sb_global is None:
            continue
        ri = (_ri(residues[r]), _ri(residues[r]), _ri(residues[r + 1]), 0)
        ai = (sb_type_idx, ca_idx, ca_idx, ca_idx)
        pot = _match_pot(params.dihedral_pots, ri, ai, (3, 3, 2, 1))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(sb_global),
                b=ChainAtomRef(ca_global_idx[r]),
                c=ChainAtomRef(ca_global_idx[r + 1]),
                d=ChainAtomRef(ca_global_idx[r + 2]),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )

    # Cross-residue sidechain dihedral SB(r)-CA(r)-CA(r+1)-SB(r+1). The most
    # common form is (CB, CA, CA, CB), with 361 of 416 potentials using
    # orders=(1, 1, 2, 2) and residues=(R_r, R_r, R_{r+1}, R_{r+1}). This is
    # generalized to any first sidechain bead.
    for r in range(n_res - 1):
        # Need a first sidechain bead on both r and r+1.
        sb1_global, sb1_type = None, None
        for bead_global in residue_beads[r]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sb1_global = bead_global
                sb1_type = _ai(bn)
                break
        if sb1_global is None:
            continue
        sb2_global, sb2_type = None, None
        for bead_global in residue_beads[r + 1]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sb2_global = bead_global
                sb2_type = _ai(bn)
                break
        if sb2_global is None:
            continue
        ri = (
            _ri(residues[r]),
            _ri(residues[r]),
            _ri(residues[r + 1]),
            _ri(residues[r + 1]),
        )
        ai = (sb1_type, ca_idx, ca_idx, sb2_type)
        pot = _match_pot(params.dihedral_pots, ri, ai, (1, 1, 2, 2))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(sb1_global),
                b=ChainAtomRef(ca_global_idx[r]),
                c=ChainAtomRef(ca_global_idx[r + 1]),
                d=ChainAtomRef(sb2_global),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )

    # Sidechain-extending dihedral SC2(r)-SC1(r)-CA(r)-CA(r+/-1). Here SC1 is
    # the first sidechain bead (CB for most residues, SB for CYS) and SC2 is
    # the second sidechain bead (CG, NG, or OG depending on the residue). The
    # forward dihedral is SC2(r)-SC1(r)-CA(r)-CA(r+1) with orders=(1, 1, 1, 2)
    # and residues=(R, R, R, R_{r+1}). The backward dihedral is
    # SC2(r)-SC1(r)-CA(r)-CA(r-1) with orders=(2, 2, 2, 1) and
    # residues=(R, R, R, R_{r-1}).
    for r in range(n_res):
        # Find the first and second sidechain beads in this residue. This
        # dihedral needs at least 3 beads (CA plus 2 sidechain beads).
        beads_here = residue_beads[r]
        if len(beads_here) < 3:
            continue  # only CA, or CA plus SB1 with no SB2, so skip
        sc1_global = beads_here[1]  # first non-CA bead
        sc2_global = beads_here[2]  # second non-CA bead
        sc2_type = _ai(atom_beadname[sc2_global])
        sc1_type = _ai(atom_beadname[sc1_global])
        # Forward
        if r + 1 < n_res:
            ri = (
                _ri(residues[r]),
                _ri(residues[r]),
                _ri(residues[r]),
                _ri(residues[r + 1]),
            )
            ai = (sc2_type, sc1_type, ca_idx, ca_idx)
            pot = _match_pot(params.dihedral_pots, ri, ai, (1, 1, 1, 2))
            type_idx = pot.index if pot is not None else -1
            torsions.append(
                ChainTorsion(
                    a=ChainAtomRef(sc2_global),
                    b=ChainAtomRef(sc1_global),
                    c=ChainAtomRef(ca_global_idx[r]),
                    d=ChainAtomRef(ca_global_idx[r + 1]),
                    phi0=0.0,
                    k_tor=2.0,
                    n=1,
                    type_idx=type_idx,
                )
            )
        # Backward
        if r - 1 >= 0:
            ri = (
                _ri(residues[r]),
                _ri(residues[r]),
                _ri(residues[r]),
                _ri(residues[r - 1]),
            )
            ai = (sc2_type, sc1_type, ca_idx, ca_idx)
            pot = _match_pot(params.dihedral_pots, ri, ai, (2, 2, 2, 1))
            type_idx = pot.index if pot is not None else -1
            torsions.append(
                ChainTorsion(
                    a=ChainAtomRef(sc2_global),
                    b=ChainAtomRef(sc1_global),
                    c=ChainAtomRef(ca_global_idx[r]),
                    d=ChainAtomRef(ca_global_idx[r - 1]),
                    phi0=0.0,
                    k_tor=2.0,
                    n=1,
                    type_idx=type_idx,
                )
            )

    # Cap backbone dihedrals. For ACE the dihedral is CN-CA(0)-CA(1)-CA(2) with
    # residues=(r0, r0, r1, 0) and orders=(1, 1, 2, 3). For NME it is
    # CA(n-3)-CA(n-2)-CA(n-1)-CC with residues=(0, r_{n-2}, r_{n-1}, r_{n-1})
    # and orders=(1, 2, 3, 3).
    if cn_global_idx >= 0 and n_res >= 3:
        ri = (_ri(residues[0]), _ri(residues[0]), _ri(residues[1]), 0)
        ai = (cn_idx, ca_idx, ca_idx, ca_idx)
        pot = _match_pot(params.dihedral_pots, ri, ai, (1, 1, 2, 3))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(cn_global_idx),
                b=ChainAtomRef(ca_global_idx[0]),
                c=ChainAtomRef(ca_global_idx[1]),
                d=ChainAtomRef(ca_global_idx[2]),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )
    if cc_global_idx >= 0 and n_res >= 3:
        ri = (
            0,
            _ri(residues[n_res - 2]),
            _ri(residues[n_res - 1]),
            _ri(residues[n_res - 1]),
        )
        ai = (ca_idx, ca_idx, ca_idx, cc_idx)
        pot = _match_pot(params.dihedral_pots, ri, ai, (1, 2, 3, 3))
        type_idx = pot.index if pot is not None else -1
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(ca_global_idx[n_res - 3]),
                b=ChainAtomRef(ca_global_idx[n_res - 2]),
                c=ChainAtomRef(ca_global_idx[n_res - 1]),
                d=ChainAtomRef(cc_global_idx),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=type_idx,
            )
        )

    # Cap-sidechain dihedrals. The ACE incoming-sidechain dihedral is
    # CN(0)-CA(0)-CA(1)-SC1(1) with residues=(r0, r0, r1, r1) and
    # orders=(1, 3, 2, 2). The NME outgoing-sidechain dihedral is
    # SC1(n-2)-CA(n-2)-CA(n-1)-CC with residues=(r_{n-2}, r_{n-2}, r_{n-1},
    # r_{n-1}) and orders=(1, 1, 2, 2). SC1 is the first sidechain bead. Some
    # residues have no populated lookup and are silently skipped.
    if cn_global_idx >= 0 and n_res >= 2:
        # Find SC1 of residue 1, the residue right after the cap-flanked one.
        sc1_global, sc1_type = None, None
        for bead_global in residue_beads[1]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sc1_global = bead_global
                sc1_type = _ai(bn)
                break
        if sc1_global is not None:
            ri = (
                _ri(residues[0]),
                _ri(residues[0]),
                _ri(residues[1]),
                _ri(residues[1]),
            )
            ai = (cn_idx, ca_idx, ca_idx, sc1_type)
            pot = _match_pot(params.dihedral_pots, ri, ai, (1, 3, 2, 2))
            type_idx = pot.index if pot is not None else -1
            torsions.append(
                ChainTorsion(
                    a=ChainAtomRef(cn_global_idx),
                    b=ChainAtomRef(ca_global_idx[0]),
                    c=ChainAtomRef(ca_global_idx[1]),
                    d=ChainAtomRef(sc1_global),
                    phi0=0.0,
                    k_tor=2.0,
                    n=1,
                    type_idx=type_idx,
                )
            )
    if cc_global_idx >= 0 and n_res >= 2:
        # Find SC1 of residue n-2 (the residue 2 from the C-terminus)
        sc1_global, sc1_type = None, None
        for bead_global in residue_beads[n_res - 2]:
            bn = atom_beadname[bead_global]
            if bn != "CA":
                sc1_global = bead_global
                sc1_type = _ai(bn)
                break
        if sc1_global is not None:
            ri = (
                _ri(residues[n_res - 2]),
                _ri(residues[n_res - 2]),
                _ri(residues[n_res - 1]),
                _ri(residues[n_res - 1]),
            )
            ai = (sc1_type, ca_idx, ca_idx, cc_idx)
            pot = _match_pot(params.dihedral_pots, ri, ai, (1, 1, 2, 2))
            type_idx = pot.index if pot is not None else -1
            torsions.append(
                ChainTorsion(
                    a=ChainAtomRef(sc1_global),
                    b=ChainAtomRef(ca_global_idx[n_res - 2]),
                    c=ChainAtomRef(ca_global_idx[n_res - 1]),
                    d=ChainAtomRef(cc_global_idx),
                    phi0=0.0,
                    k_tor=2.0,
                    n=1,
                    type_idx=type_idx,
                )
            )

    # Build pair lookups for all non-bonded pairs, skipping pairs that are
    # bonded (1-2) or that are 1-3 neighbors sharing a bond.
    bonded_set = set()
    for bond in bonds:
        i, j = bond.a.atom_idx, bond.b.atom_idx
        bonded_set.add((min(i, j), max(i, j)))
    # 1-3 neighbors reached via two bonds.
    neighbors = [set() for _ in range(n_atoms)]
    for bond in bonds:
        i, j = bond.a.atom_idx, bond.b.atom_idx
        neighbors[i].add(j)
        neighbors[j].add(i)
    excluded_13 = set()
    for i in range(n_atoms):
        for j in neighbors[i]:
            for k in neighbors[j]:
                if k != i:
                    excluded_13.add((min(i, k), max(i, k)))

    # Helper that returns the residue label for an atom, handling caps. Caps
    # have atom_residue = -1, and their lookup uses the flanking residue.
    def _atom_residue_label(atom_idx):
        r = atom_residue[atom_idx]
        if r >= 0:
            return residues[r]
        # Cap atom, so find the flanking residue by looking at the bonded CA.
        if atom_idx == cn_global_idx:
            return residues[0]  # ACE flanks the first residue
        if atom_idx == cc_global_idx:
            return residues[n_res - 1]  # NME flanks the last residue
        # Fallback that should not happen, but is kept for safety.
        return residues[0]

    pair_lookups = {}
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) in bonded_set or (i, j) in excluded_13:
                continue
            res_i_name = _atom_residue_label(i)
            res_j_name = _atom_residue_label(j)
            bead_i_name = atom_beadname[i]
            bead_j_name = atom_beadname[j]
            ri = (_ri(res_i_name), _ri(res_j_name))
            ai = (_ai(bead_i_name), _ai(bead_j_name))
            orders = (0, 0)
            pot = _match_pot(params.pair_pots, ri, ai, orders)
            if pot is None:
                # Try reversed (symmetric pair lookup)
                pot = _match_pot(
                    params.pair_pots, (ri[1], ri[0]), (ai[1], ai[0]), orders
                )
            if pot is not None:
                pair_lookups[(i, j)] = pot.index

    return ChainCommon(
        name=name or ("-".join(residues[:3]) + ("..." if n_res > 3 else "") + "_sc"),
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        coffdrop_params=params,
        pair_lookups=pair_lookups,
    )


# Single-letter to 3-letter residue code map (standard 20 amino acids).
_ONE_LETTER_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def chain_from_sequence(
    sequence: str,
    coffdrop_dir: Optional[str] = None,
    name: Optional[str] = None,
    sidechains: bool = True,
    k_spring: float = 100.0,
    caps: Tuple[Optional[str], Optional[str]] = (None, None),
) -> "ChainCommon":
    """Build a COFFDROP chain from a single-letter or 3-letter sequence.

    User-facing factory: handles parameter loading and sequence conversion.

    Parameters
    ----------
    sequence : str
        Either single-letter codes ("ARWGL") or 3-letter codes
        ("ALA-ARG-TRP-GLY-LEU" or "ALA ARG TRP GLY LEU").
    coffdrop_dir : str, optional
        Path to directory containing coffdrop.xml, map.xml,
        connectivity.xml, charges.xml. If None (default), uses the
        data directory bundled with the pystarc package.
    name : str, optional
        Optional chain name, auto-generated from the sequence if not given.
    sidechains : bool
        If True, build with sidechain beads (default).
        If False, CA-only backbone.
    k_spring : float
        Harmonic spring constant for bonds (kBT/A^2).

    Returns
    -------
    ChainCommon
        Ready for compute_chain_forces.

    Examples
    --------
    >>> chain = chain_from_sequence("ARWGL")
    >>> chain.n_atoms  # ALA(2) + ARG(3) + TRP(4) + GLY(1) + LEU(3)
    13

    >>> chain = chain_from_sequence("ALA-ARG-TRP")
    >>> chain.n_atoms  # 2 + 3 + 4
    9

    >>> chain = chain_from_sequence("ARG", sidechains=False)
    >>> chain.n_atoms  # CA-only
    1
    """
    from pathlib import Path as _Path
    from pystarc.simulation.coffdrop_params import COFFDROPParams

    # Resolve coffdrop_dir: default to data bundled with the pystarc package
    if coffdrop_dir is None:
        coffdrop_dir = str(_Path(__file__).parent.parent / "coffdrop_data")
    p = _Path(coffdrop_dir)
    if not p.is_dir():
        raise FileNotFoundError(
            f"COFFDROP data directory not found: {coffdrop_dir}. "
            "Pass coffdrop_dir explicitly or check pystarc installation."
        )
    params = COFFDROPParams.load(
        ff_xml=str(p / "coffdrop.xml"),
        mapping_xml=str(p / "map.xml"),
        connectivity_xml=str(p / "connectivity.xml"),
        charges_xml=str(p / "charges.xml"),
    )

    # Parse sequence: detect format and convert to list of 3-letter codes.
    s = sequence.strip().upper()
    if "-" in s or " " in s:
        # 3-letter form: ALA-ARG or ALA ARG
        residues = [r.strip() for r in s.replace("-", " ").split() if r.strip()]
        # Validate
        for r in residues:
            if len(r) != 3:
                raise ValueError(
                    f"3-letter sequence has invalid token '{r}' "
                    f"(expected 3-letter residue code)"
                )
    else:
        # Single-letter form: ARWGL
        residues = []
        for c in s:
            if c not in _ONE_LETTER_TO_THREE:
                raise ValueError(
                    f"Unknown single-letter code '{c}' in sequence '{sequence}'. "
                    f"Use standard 20 amino acid codes (ARNDCEQGHILKMFPSTWYV)."
                )
            residues.append(_ONE_LETTER_TO_THREE[c])

    if not residues:
        raise ValueError("sequence cannot be empty")

    # Build chain
    if sidechains:
        return build_chain_common_with_sidechains_from_coffdrop(
            residues,
            params,
            name=name,
            k_spring=k_spring,
            caps=caps,
        )
    else:
        if caps != (None, None):
            raise ValueError(
                "caps are only supported with sidechains=True; "
                "the CA-only builder does not support caps."
            )
        return build_chain_common_from_coffdrop(
            residues,
            params,
            name=name,
            k_spring=k_spring,
        )


def place_relaxed_geometry(chain: "ChainCommon") -> np.ndarray:
    """Produce a sensible initial configuration for a chain.

    Backbone CAs are placed along the x-axis at 3.8 A spacing.
    Sidechain beads are placed at their bond eq length from their
    anchor, with a slight rotation per atom to avoid colinear
    projections that excite dihedral forces unfavorably.

    The result is suitable as a starting configuration for BD
    simulation or as chain_init_body_positions for ChainBDSimulator.

    Parameters
    ----------
    chain : ChainCommon
        A chain produced by chain_from_sequence or one of the
        build_chain_common_* helpers.

    Returns
    -------
    positions : np.ndarray, shape (n_atoms, 3)
        Cartesian positions in Angstroms.
    """
    import numpy as np

    n = chain.n_atoms
    positions = np.zeros((n, 3))
    # Place the CAs along x at 3.8 Å spacing.
    ca_indices = [
        i
        for i, a in enumerate(chain.atoms)
        if a.resname.endswith(":CA") or a.resname == "CA"
    ]
    if not ca_indices:
        # Fall back to atom 0 as the anchor for unusual chains.
        ca_indices = [0]
    for r, ca_i in enumerate(ca_indices):
        positions[ca_i] = [3.8 * r, 0.0, 0.0]
    # Place the sidechain beads with a varied tilt to avoid colinearity. For
    # each non-CA atom, find its bond to a previously placed atom and project
    # it at the bond equilibrium length, rotated by an angle that depends on
    # the atom index.
    placed = set(ca_indices)
    rotation_offset = 0.0
    for i in range(n):
        if i in placed:
            continue
        # Find a bond connecting i to a placed atom
        prev_i = None
        r0 = None
        for bond in chain.bonds:
            if bond.a.atom_idx == i and bond.b.atom_idx in placed:
                prev_i = bond.b.atom_idx
                r0 = bond.r0
                break
            if bond.b.atom_idx == i and bond.a.atom_idx in placed:
                prev_i = bond.a.atom_idx
                r0 = bond.r0
                break
        if prev_i is None:
            # No connection to a placed atom, so place it at an offset from
            # the origin.
            positions[i] = positions[0] + np.array([0.0, 1.0, 0.0])
            placed.add(i)
            continue
        # Tilted offset that varies with the index to break degeneracy.
        angle = rotation_offset + 0.3 * (i % 4)
        offset = np.array(
            [
                0.5 * np.sin(angle),
                r0 * np.cos(angle),
                0.5 * np.sin(angle * 1.1),
            ]
        )
        # Renormalize to the bond length.
        offset_norm = np.linalg.norm(offset)
        if offset_norm > 1e-10:
            offset = offset * (r0 / offset_norm)
        positions[i] = positions[prev_i] + offset
        placed.add(i)
        rotation_offset += 0.4
    # Center at the origin by subtracting the mean. ChainBDSimulator requires
    # body-frame positions, which are centered at the origin, and this is also
    # the natural starting frame for any rigid-body simulation.
    positions = positions - positions.mean(axis=0)
    return positions


def chain_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    coffdrop_dir: Optional[str] = None,
    name: Optional[str] = None,
    sidechains: bool = True,
    k_spring: float = 100.0,
) -> "ChainCommon":
    """Build a COFFDROP chain by extracting the sequence from a PDB file.

    Reads ATOM records from the PDB to determine the residue sequence.
    The PDB atomic coordinates are not used. The chain is built using
    COFFDROP coarse-grained beads. Use place_relaxed_geometry to
    generate starting positions afterward.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    chain_id : str, optional
        Restrict to a specific chain ID (column 22). If None and the
        PDB has multiple chains, raises ValueError to prevent silent
        misinterpretation.
    coffdrop_dir, name, sidechains, k_spring
        Forwarded to the underlying chain builder.

    Returns
    -------
    ChainCommon
        Ready for compute_chain_forces.

    Raises
    ------
    ValueError
        If the PDB has multiple chains and chain_id is not specified, or
        if a residue name in the PDB is not a standard amino acid.

    Examples
    --------
    >>> chain = chain_from_pdb("peptide.pdb")
    >>> chain.n_atoms
    13
    """
    from pathlib import Path as _Path

    pdb_file = _Path(pdb_path)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Track unique (chain, resnum) to resname pairs in order of first appearance.
    seen_residues: List[Tuple[str, int]] = []
    residue_map: Dict[Tuple[str, int], str] = {}
    chains_seen: set = set()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            # Standard PDB columns:
            #   resname:    cols 17-20 (1-indexed)
            #   chain_id:   col 22
            #   resnum:     cols 23-26
            if len(line) < 26:
                continue
            resname = line[17:20].strip()
            ch = line[21:22].strip() or "A"
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue  # skip malformed lines
            chains_seen.add(ch)
            key = (ch, resnum)
            if key not in residue_map:
                residue_map[key] = resname
                seen_residues.append(key)

    if not seen_residues:
        raise ValueError(f"No ATOM records found in {pdb_path}")

    # Multi-chain handling
    if chain_id is None:
        if len(chains_seen) > 1:
            raise ValueError(
                f"PDB has multiple chains {sorted(chains_seen)}; "
                f"specify chain_id=... to select one."
            )
        chain_id = next(iter(chains_seen))

    if chain_id not in chains_seen:
        raise ValueError(
            f"chain_id='{chain_id}' not in PDB chains {sorted(chains_seen)}"
        )

    # Extract sequence in residue-number order for the chosen chain.
    # Sort by resnum to ensure correct N-to-C order.
    chain_residues = sorted(
        [(rn, residue_map[(ch, rn)]) for (ch, rn) in seen_residues if ch == chain_id]
    )
    residues = [resname for _rn, resname in chain_residues]

    if not residues:
        raise ValueError(f"chain_id='{chain_id}' has no residues")

    # Build sequence string and delegate to chain_from_sequence
    seq_str = "-".join(residues)
    return chain_from_sequence(
        seq_str,
        coffdrop_dir=coffdrop_dir,
        name=name,
        sidechains=sidechains,
        k_spring=k_spring,
    )


def run_chain_bd_simulation(
    chain: "ChainCommon",
    initial_positions: Optional[np.ndarray] = None,
    target_pqr: Optional[str] = None,
    target_grid_dx: Optional[str] = None,
    born_grid_dx: Optional[str] = None,
    desolvation_alpha: float = DESOLVATION_ALPHA,
    reaction_pairs: Optional[list] = None,
    reaction_n_needed: int = 3,
    n_trajectories: int = 10,
    max_steps: int = 1000,
    dt: float = CHAIN_DT,
    r_start: float = 20.0,
    r_escape: float = 50.0,
    D_trans: Optional[float] = None,
    D_rot: Optional[float] = None,
    auto_diffusion: bool = False,
    use_soft_repulsion: bool = False,
    soft_repulsion_eps: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    """Run a chain BD simulation with sensible defaults.

    A target-free demo simulation: no electrostatic field, no reaction
    pathways, just chain diffusion until trajectory escapes past
    r_escape or hits max_steps. Useful for smoke-testing chain
    physics or as a starting template for real simulations.

    Parameters
    ----------
    chain : ChainCommon
        Chain template, typically from chain_from_sequence or chain_from_pdb.
    initial_positions : np.ndarray, optional
        Body-frame positions, shape (n_atoms, 3). If None, uses
        place_relaxed_geometry.
    target_pqr : str, optional
        Path to target PQR file, or None for empty target (no reaction
        criterion, free chain diffusion).
    target_grid_dx : str, optional
        Path to target electrostatic DX grid file, or None for no
        electrostatic field.
    born_grid_dx : str, optional
        Path to target Born desolvation DX grid file, or None for no
        Born contribution. The grid is expected to be in raw APBS units;
        the per-atom force is F_i = -alpha * q_i^2 * grad(g)(r_i).
    desolvation_alpha : float
        Born desolvation prefactor in kBT/(e^2 * grid_unit). Default 1.0,
        matching the rigid-body engine. The *_born.dx grids hold a cavity
        self energy, not an APBS potential. Ignored if born_grid_dx is None.
    D_trans, D_rot : float, optional
        Translational and rotational diffusion coefficients. If None
        (and auto_diffusion is False), defaults of 0.1 and 0.01 are used
        for backward compatibility with pre-RPY callers. If
        auto_diffusion=True, these MUST be None (the simulator computes
        anisotropic (3, 3) tensors from chain geometry via Rotne-Prager-
        Yamakawa).
    auto_diffusion : bool
        If True, ChainBDSimulator computes full anisotropic (3, 3) D_trans
        and D_rot tensors from the chain's bead geometry using RPY
        hydrodynamics. Default False (scalar D_trans, D_rot supplied
        explicitly or via the 0.1/0.01 defaults). Mutually exclusive
        with explicitly supplied D_trans or D_rot.
    use_soft_repulsion : bool
        If True, enable smooth WCA chain-target soft-repulsion forces
        layered on top of (or replacing) hard-sphere rejection. The
        underlying force evaluator chain_target_steric_forces is
        vectorized for production use. Default False (preserves
        backward-compatible behavior where chain-target steric handling
        is hard-sphere rejection only).
    soft_repulsion_eps : float
        WCA epsilon (in kBT) controlling the chain-target repulsion
        well depth and steering-force gradient. Conventional value for
        flexible-chain BD is 0.5 kBT (Browndye2/SDA literature). The
        function-level default is 1.0 (kept for backward compatibility);
        production callers should pass an explicit value. Ignored if
        use_soft_repulsion is False.
    n_trajectories : int
        Number of BD trajectories to run.
    max_steps : int
        Max steps per trajectory.
    dt : float
        Outer BD timestep.
    r_start, r_escape : float
        BD b-sphere and q-sphere radii (in A).
    D_trans, D_rot : float
        Translational and rotational diffusion coefficients.
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    list of TrajectoryResult
        One per trajectory. Each has fate, steps, time_ps, final_separation,
        reaction_name, energy_at_reaction.

    Examples
    --------
    >>> from pystarc.simulation.coffdrop_chain import (
    ...     chain_from_sequence, run_chain_bd_simulation,
    ... )
    >>> chain = chain_from_sequence("ARWGL")
    >>> results = run_chain_bd_simulation(chain, n_trajectories=2, max_steps=100)
    >>> all(r.steps > 0 for r in results)
    True

    Notes
    -----
    For real simulations with electrostatic targets and reaction
    criteria, use ChainBDSimulator directly with a populated Molecule
    target and PathwaySet.
    """
    from pystarc.simulation.chain_simulator import (
        ChainBDSimulator,
        ChainBDParameters,
    )
    from pystarc.structures.molecules import Molecule
    from pystarc.pathways.reaction_interface import PathwaySet

    if initial_positions is None:
        initial_positions = place_relaxed_geometry(chain)
    else:
        # Ensure the positions are centered, which the BD simulator requires.
        initial_positions = initial_positions - initial_positions.mean(axis=0)

    if target_pqr is None:
        target = Molecule(name="empty_target", atoms=[])
    else:
        from pystarc.structures.pqr_io import parse_pqr_records
        from pystarc.structures.molecules import Atom as _Atom
        from pathlib import Path as _Path

        pqr_path = _Path(target_pqr)
        if not pqr_path.exists():
            raise FileNotFoundError(f"target PQR file not found: {target_pqr}")
        records = parse_pqr_records(pqr_path)
        target_atoms = [
            _Atom(
                name=r.name,
                residue_name=r.resname,
                residue_index=r.resid,
                chain="A",
                x=r.x,
                y=r.y,
                z=r.z,
                charge=r.charge,
                radius=r.radius,
            )
            for r in records
        ]
        if not target_atoms:
            raise ValueError(f"target PQR has no atoms: {target_pqr}")
        target = Molecule(name=pqr_path.stem, atoms=target_atoms)
    # Build pathway_set from reaction_pairs if provided.
    if reaction_pairs:
        from pystarc.structures.molecules import ContactPair, ReactionCriteria
        from pystarc.pathways.reaction_interface import (
            ReactionInterface,
            PathwaySet as _PathwaySet,
        )

        pairs = [
            ContactPair(rec_idx, lig_idx, cutoff)
            for (rec_idx, lig_idx, cutoff) in reaction_pairs
        ]
        crit = ReactionCriteria(
            name="association",
            pairs=pairs,
            n_needed=reaction_n_needed,
        )
        rxn = ReactionInterface(
            name="association",
            criteria=crit,
        )
        pathway_set = _PathwaySet([rxn])
    else:
        pathway_set = PathwaySet()
    params = ChainBDParameters(
        n_trajectories=n_trajectories,
        dt=dt,
        max_steps=max_steps,
        r_start=r_start,
        r_escape=r_escape,
        seed=seed,
        verbose=verbose,
        use_soft_repulsion=use_soft_repulsion,
        soft_repulsion_eps=soft_repulsion_eps,
    )
    # Load target electrostatic grid if provided
    target_grid = None
    if target_grid_dx is not None:
        from pystarc.forces.electrostatic.grid_force import DXGrid
        from pathlib import Path as _Path

        dx_path = _Path(target_grid_dx)
        if not dx_path.exists():
            raise FileNotFoundError(f"target DX file not found: {target_grid_dx}")
        target_grid = DXGrid.from_file(dx_path)
    # Load the target Born desolvation grid if one is provided. The grid
    # encodes the target's desolvation field (a raw APBS *_born.dx grid), and
    # the simulator applies F_i = -alpha × q_i^2 × grad(g)(r_i) per chain bead.
    # alpha is configurable via desolvation_alpha and defaults to 1/(4 π) from
    # engine.py.
    born_grid = None
    if born_grid_dx is not None:
        from pystarc.forces.electrostatic.grid_force import DXGrid
        from pathlib import Path as _Path

        born_path = _Path(born_grid_dx)
        if not born_path.exists():
            raise FileNotFoundError(f"Born DX file not found: {born_grid_dx}")
        born_grid = DXGrid.from_file(born_path)
    # Resolve the diffusion coefficients. Three valid combinations are passed
    # through to ChainBDSimulator unchanged. In the first, auto_diffusion is
    # False with D_trans and D_rot both None, which applies the
    # backward-compatible scalar defaults of 0.1 and 0.01. In the second,
    # auto_diffusion is False with explicit D_trans and D_rot, which pass
    # through unchanged (a scalar or a (3, 3) tensor). In the third,
    # auto_diffusion is True with D_trans and D_rot both None, which lets
    # ChainBDSimulator compute (3, 3) RPY tensors from the chain geometry.
    # Supplying explicit D_trans or D_rot in that case would raise a
    # ValueError inside the simulator.
    if not auto_diffusion:
        if D_trans is None:
            D_trans = 0.1
        if D_rot is None:
            D_rot = 0.01
    sim = ChainBDSimulator(
        target=target,
        chain_template=chain,
        chain_init_body_positions=initial_positions,
        params=params,
        pathway_set=pathway_set,
        D_trans=D_trans,
        D_rot=D_rot,
        target_grid=target_grid,
        born_grid=born_grid,
        desolvation_alpha=desolvation_alpha,
        auto_diffusion=auto_diffusion,
    )
    return sim.run()


def _bd_worker(args):
    """Worker function for multiprocessing: runs a slice of trajectories.

    Returns a list of trajectory results from this worker's slice.
    """
    (
        chain,
        target_pqr,
        target_grid_dx,
        born_grid_dx,
        desolvation_alpha,
        reaction_pairs,
        reaction_n_needed,
        n_traj_slice,
        max_steps,
        dt,
        r_start,
        r_escape,
        D_trans,
        D_rot,
        auto_diffusion,
        use_soft_repulsion,
        soft_repulsion_eps,
        worker_seed,
    ) = args
    return run_chain_bd_simulation(
        chain=chain,
        target_pqr=target_pqr,
        target_grid_dx=target_grid_dx,
        born_grid_dx=born_grid_dx,
        desolvation_alpha=desolvation_alpha,
        reaction_pairs=reaction_pairs,
        reaction_n_needed=reaction_n_needed,
        n_trajectories=n_traj_slice,
        max_steps=max_steps,
        dt=dt,
        r_start=r_start,
        r_escape=r_escape,
        D_trans=D_trans,
        D_rot=D_rot,
        auto_diffusion=auto_diffusion,
        use_soft_repulsion=use_soft_repulsion,
        soft_repulsion_eps=soft_repulsion_eps,
        seed=worker_seed,
        verbose=False,
    )


def run_chain_bd_parallel(
    chain: "ChainCommon",
    n_trajectories: int,
    target_pqr: Optional[str] = None,
    target_grid_dx: Optional[str] = None,
    born_grid_dx: Optional[str] = None,
    desolvation_alpha: float = DESOLVATION_ALPHA,
    reaction_pairs: Optional[list] = None,
    reaction_n_needed: int = 3,
    n_workers: Optional[int] = None,
    initial_positions: Optional[np.ndarray] = None,
    max_steps: int = 1000,
    dt: float = CHAIN_DT,
    r_start: float = 20.0,
    r_escape: float = 50.0,
    D_trans: Optional[float] = None,
    D_rot: Optional[float] = None,
    auto_diffusion: bool = False,
    use_soft_repulsion: bool = False,
    soft_repulsion_eps: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
):
    """Run BD trajectories in parallel using multiprocessing.

    Splits n_trajectories evenly across n_workers processes. Each worker
    constructs an independent ChainBDSimulator with a deterministic seed
    derived from base seed + worker_id, and runs its slice of trajectories.

    Parameters
    ----------
    chain : ChainCommon
        Chain template.
    n_trajectories : int
        Total number of trajectories across all workers.
    target_pqr : str or None
        Path to target PQR file, or None for empty target.
    target_grid_dx : str or None
        Path to target electrostatic DX grid file.
    born_grid_dx : str or None
        Path to target Born desolvation DX grid file. If provided, each
        worker loads its own copy and the per-bead Born force
        F_i = -alpha * q_i^2 * grad(g)(r_i) is summed on top of the
        electrostatic contribution.
    desolvation_alpha : float
        Born prefactor in kBT/(e^2 * grid_unit). Default 1.0, matching the
        rigid-body engine. The *_born.dx grids hold a cavity self energy,
        not an APBS potential. Ignored if born_grid_dx is None.
    n_workers : int or None
        Number of parallel workers (default: cpu_count() - 2, min 1).
    seed : int
        Base seed. Worker i uses seed + i*1000000.

    Returns
    -------
    list of trajectory results, in worker order (not strictly trajectory order).
    """
    import multiprocessing as mp

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 2)
    n_workers = min(n_workers, n_trajectories)
    # Split the trajectories evenly across the workers.
    base = n_trajectories // n_workers
    extra = n_trajectories % n_workers
    slices = [base + (1 if i < extra else 0) for i in range(n_workers)]
    # Build the worker arguments.
    worker_args = []
    for i, n_slice in enumerate(slices):
        worker_seed = seed + i * 1000000
        worker_args.append(
            (
                chain,
                target_pqr,
                target_grid_dx,
                born_grid_dx,
                desolvation_alpha,
                reaction_pairs,
                reaction_n_needed,
                n_slice,
                max_steps,
                dt,
                r_start,
                r_escape,
                D_trans,
                D_rot,
                auto_diffusion,
                use_soft_repulsion,
                soft_repulsion_eps,
                worker_seed,
            )
        )
    if verbose:
        print(
            f"Running {n_trajectories} trajectories on {n_workers} workers "
            f"(slices: {slices})"
        )
    if n_workers == 1:
        # Skip the multiprocessing overhead for a single worker.
        return _bd_worker(worker_args[0])
    with mp.Pool(processes=n_workers) as pool:
        results_per_worker = pool.map(_bd_worker, worker_args)
    # Flatten the per-worker results into a single list.
    all_results = []
    for worker_results in results_per_worker:
        all_results.extend(worker_results)
    return all_results
