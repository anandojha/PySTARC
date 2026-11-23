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
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import math
from pystarc.global_defs.constants import KBT_KCAL, VACUUM_PERMITTIVITY_KBT
from pystarc.global_defs.defaults import (
    CHAIN_DT,
    DESOLVATION_ALPHA,
    GHOST_NAMES,
    SOLVENT_DIELECTRIC,
    VISCOSITY,
)

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
    hydro_radius: float = 0.0


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
    # Harmonic springs of the elastic network that holds the reference fold,
    # built by elastic_network. They act like bonds but enter no exclusion.
    network: List["ChainBond"] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def hydro_radii_array(self) -> np.ndarray:
        """Return hydrodynamic radii, falling back to steric radius when hydro_radius <= 0."""
        return np.array(
            [a.hydro_radius if a.hydro_radius > 0.0 else a.radius for a in self.atoms],
            dtype=float,
        )


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


def _coplanar_violation_and_grad(
    ra: np.ndarray, rb: np.ndarray, rc: np.ndarray, rd: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Signed plane distance of atom a from plane (b, c, d) and exact 4-atom gradients.

    Returns (phi, ga, gb, gc, gd, degenerate). When (b, c, d) are collinear
    (|(rc - rb) x (rd - rb)| < 1e-12), returns zeros with degenerate=True.
    """
    centroid = (rb + rc + rd) / 3.0
    d2 = rc - rb
    d3 = rd - rb
    perp = np.cross(d2, d3)
    perp_norm = float(np.linalg.norm(perp))
    if not math.isfinite(perp_norm) or perp_norm < 1e-12:
        z = np.zeros(3, dtype=float)
        return 0.0, z, z, z, z, True
    n_hat = perp / perp_norm
    w = ra - centroid
    phi = float(np.dot(w, n_hat))
    q = (w - phi * n_hat) / perp_norm
    g2 = np.cross(d3, q)
    g3 = np.cross(q, d2)
    third_n = n_hat / 3.0
    ga = n_hat
    gc = -third_n + g2
    gd = -third_n + g3
    gb = -third_n - g2 - g3
    return phi, ga, gb, gc, gd, False


def compute_constraint_violations(
    state: "ChainState",
    return_degenerate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """Signed violation vector for all constraints on a chain state.

    When return_degenerate is True, returns (phi, had_degenerate) where
    had_degenerate marks any coincident length pair (r < 1e-12) or collinear
    coplanar triple (|(rc - rb) x (rd - rb)| < 1e-12).
    """
    common = state.common
    pos = state.positions
    n_len = len(common.length_constraints)
    n_cop = len(common.coplanar_constraints)
    phi = np.zeros(n_len + n_cop, dtype=float)
    had_degenerate = False
    for ic, c in enumerate(common.length_constraints):
        ra = pos[_chain_idx(c.a)]
        rb = pos[_chain_idx(c.b)]
        r = float(np.linalg.norm(ra - rb))
        if not math.isfinite(r) or r < 1e-12:
            had_degenerate = True
        phi[ic] = r - c.length
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
        if not math.isfinite(perp_norm) or perp_norm < 1e-12:
            had_degenerate = True
            phi[n_len + ic] = 0.0
        else:
            n_hat = perp / perp_norm
            phi[n_len + ic] = float(np.dot(ra - centroid, n_hat))
    if return_degenerate:
        return phi, had_degenerate
    return phi


def _resolve_mobilities(n_atoms: int, mobilities: Optional[np.ndarray]) -> np.ndarray:
    if mobilities is None:
        return np.ones(n_atoms, dtype=float)
    mob = np.asarray(mobilities, dtype=float).reshape(n_atoms)
    if not np.all(np.isfinite(mob)) or np.any(mob < 0.0):
        raise ValueError("mobilities must be finite and non-negative")
    return mob


def satisfy_constraints(
    state: "ChainState",
    tol: float = 1e-6,
    max_iter: int = 200,
    mobilities: Optional[np.ndarray] = None,
) -> int:
    """Iteratively project positions onto the constraint manifold via SHAKE.

    Modifies state.positions in place. Length constraints split corrections by
    mobility weights mu_a / (mu_a + mu_b). Coplanar constraints project all
    four atoms along the analytic gradient M grad_c / (grad_c^T M grad_c),
    preserving center of mass and zero net torque under uniform mobilities.
    Refuses convergence when any constraint is degenerate.
    """
    common = state.common
    pos = state.positions
    if not common.length_constraints and not common.coplanar_constraints:
        return 0
    mob = _resolve_mobilities(state.n_atoms, mobilities)
    if max_iter > 0:
        init_phi, init_deg = compute_constraint_violations(
            state, return_degenerate=True
        )
        init_viol = float(np.max(np.abs(init_phi))) if init_phi.size else 0.0
        if not init_deg and init_viol < tol:
            return 0
    max_violation = float("nan")
    had_degenerate = False
    for it in range(max_iter):
        had_degenerate = False
        for c in common.length_constraints:
            ia = _chain_idx(c.a)
            ib = _chain_idx(c.b)
            ra = pos[ia]
            rb = pos[ib]
            d = rb - ra
            r = float(np.linalg.norm(d))
            if not math.isfinite(r) or r < 1e-12:
                had_degenerate = True
                continue
            mu_a = float(mob[ia])
            mu_b = float(mob[ib])
            mu_sum = mu_a + mu_b
            if mu_sum < 1e-30:
                had_degenerate = True
                continue
            violation = r - c.length
            d_hat = d / r
            wa = mu_a / mu_sum
            wb = mu_b / mu_sum
            pos[ia] = ra + (wa * violation) * d_hat
            pos[ib] = rb - (wb * violation) * d_hat
        for c in common.coplanar_constraints:
            ia = _chain_idx(c.a)
            ib = _chain_idx(c.b)
            ic = _chain_idx(c.c)
            ild = _chain_idx(c.d)
            phi, ga, gb, gc, gd, deg = _coplanar_violation_and_grad(
                pos[ia], pos[ib], pos[ic], pos[ild]
            )
            if deg:
                had_degenerate = True
                continue
            grads: Dict[int, np.ndarray] = {}
            for idx, gv in ((ia, ga), (ib, gb), (ic, gc), (ild, gd)):
                if idx in grads:
                    grads[idx] = grads[idx] + gv
                else:
                    grads[idx] = gv
            denom = sum(
                float(mob[idx]) * float(np.dot(gv, gv)) for idx, gv in grads.items()
            )
            if not math.isfinite(denom) or denom < 1e-30:
                had_degenerate = True
                continue
            scale = -phi / denom
            for idx, gv in grads.items():
                pos[idx] = pos[idx] + (scale * float(mob[idx])) * gv
        post_phi, post_deg = compute_constraint_violations(
            state, return_degenerate=True
        )
        max_violation = float(np.max(np.abs(post_phi))) if post_phi.size else 0.0
        if not had_degenerate and not post_deg and max_violation < tol:
            return it + 1
    reason = (
        "degenerate constraint"
        if had_degenerate
        else f"final max violation = {max_violation:.3e}"
    )
    raise RuntimeError(
        f"satisfy_constraints failed to converge in {max_iter} iterations; "
        f"{reason}, tol = {tol:.3e}"
    )


def _build_constraint_jacobian(state: "ChainState") -> np.ndarray:
    """Analytic constraint Jacobian J of shape (n_constraints, 3 * n_atoms).

    Row i, columns 3k..3k+2 hold dc_i/dr_k for atom k.
    """
    common = state.common
    pos = state.positions
    n_atoms = state.n_atoms
    n_len = len(common.length_constraints)
    n_cop = len(common.coplanar_constraints)
    J = np.zeros((n_len + n_cop, 3 * n_atoms), dtype=float)
    for ic, c in enumerate(common.length_constraints):
        ia = _chain_idx(c.a)
        ib = _chain_idx(c.b)
        d = pos[ia] - pos[ib]
        r = float(np.linalg.norm(d))
        if not math.isfinite(r) or r < 1e-12:
            continue
        d_hat = d / r
        J[ic, 3 * ia : 3 * ia + 3] += d_hat
        J[ic, 3 * ib : 3 * ib + 3] -= d_hat
    for ic, c in enumerate(common.coplanar_constraints):
        row_idx = n_len + ic
        ia = _chain_idx(c.a)
        ib = _chain_idx(c.b)
        ic_idx = _chain_idx(c.c)
        ild = _chain_idx(c.d)
        _phi, ga, gb, gc, gd, deg = _coplanar_violation_and_grad(
            pos[ia], pos[ib], pos[ic_idx], pos[ild]
        )
        if deg:
            continue
        J[row_idx, 3 * ia : 3 * ia + 3] += ga
        J[row_idx, 3 * ib : 3 * ib + 3] += gb
        J[row_idx, 3 * ic_idx : 3 * ic_idx + 3] += gc
        J[row_idx, 3 * ild : 3 * ild + 3] += gd
    return J


def _coplanar_violation(state: "ChainState", c: "CoplanarConstraint") -> float:
    """Signed perpendicular distance of atom a from the plane of b, c, d."""
    pos = state.positions
    phi, *_rest = _coplanar_violation_and_grad(
        pos[_chain_idx(c.a)],
        pos[_chain_idx(c.b)],
        pos[_chain_idx(c.c)],
        pos[_chain_idx(c.d)],
    )
    return phi


def satisfy_constraints_newton(
    state: "ChainState",
    tol: float = 1e-8,
    max_iter: int = 50,
    max_damping: int = 8,
    mobilities: Optional[np.ndarray] = None,
) -> int:
    """Newton-Lagrange constraint solver with optional mobility weighting.

    Solves (J M J^T) lambda = -phi with displacement dr = M J^T lambda.
    Falls back to least-squares and L2 norm descent checks on rank-deficient
    Jacobians, and refuses convergence on degenerate constraints.
    """
    common = state.common
    if not common.length_constraints and not common.coplanar_constraints:
        return 0
    n_atoms = state.n_atoms
    mob = _resolve_mobilities(n_atoms, mobilities)
    m_vec = np.repeat(mob, 3)
    new_violation = float("nan")
    for it in range(max_iter):
        phi, had_deg = compute_constraint_violations(state, return_degenerate=True)
        if had_deg:
            raise RuntimeError(
                f"satisfy_constraints_newton: degenerate constraint at iter {it}"
            )
        violation = float(np.max(np.abs(phi))) if phi.size else 0.0
        phi_norm = float(np.linalg.norm(phi)) if phi.size else 0.0
        if violation < tol:
            return it
        J = _build_constraint_jacobian(state)
        JM = J * m_vec[None, :]
        JMJt = JM @ J.T
        rank_deficient = False
        try:
            svals = np.linalg.svd(JMJt, compute_uv=False)
            if svals.size == 0 or svals[0] <= 0.0 or svals[-1] <= 1e-12 * svals[0]:
                rank_deficient = True
                lam, *_ = np.linalg.lstsq(JMJt, -phi, rcond=None)
            else:
                lam = np.linalg.solve(JMJt, -phi)
                if not np.all(np.isfinite(lam)):
                    rank_deficient = True
                    lam, *_ = np.linalg.lstsq(JMJt, -phi, rcond=None)
        except np.linalg.LinAlgError:
            rank_deficient = True
            lam, *_ = np.linalg.lstsq(JMJt, -phi, rcond=None)
        delta = (m_vec * (J.T @ lam)).reshape(n_atoms, 3)
        scale = 1.0
        original_pos = state.positions.copy()
        for _ in range(max_damping + 1):
            state.positions[:] = original_pos + scale * delta
            new_phi, new_deg = compute_constraint_violations(
                state, return_degenerate=True
            )
            if not new_deg and np.all(np.isfinite(new_phi)):
                new_violation = float(np.max(np.abs(new_phi))) if new_phi.size else 0.0
                new_norm = float(np.linalg.norm(new_phi)) if new_phi.size else 0.0
                if rank_deficient:
                    if new_norm < phi_norm:
                        break
                else:
                    if new_violation < violation or new_norm < phi_norm:
                        break
            scale *= 0.5
        else:
            state.positions[:] = original_pos
            raise RuntimeError(
                f"satisfy_constraints_newton: damped step failed at iter {it}; "
                f"violation = {violation:.3e}, tol = {tol:.3e}"
            )
    raise RuntimeError(
        f"satisfy_constraints_newton: did not converge in {max_iter} iterations; "
        f"final violation = {new_violation:.3e}, tol = {tol:.3e}"
    )


def satisfy_constraints_hybrid(
    state: "ChainState",
    tol: float = 1e-8,
    shake_max_iter: int = 50,
    newton_max_iter: int = 50,
    mobilities: Optional[np.ndarray] = None,
) -> int:
    """SHAKE first, rolling back any degraded state before Newton fallback."""
    common = state.common
    if not common.length_constraints and not common.coplanar_constraints:
        return 0
    pre_pos = state.positions.copy()
    pre_phi, pre_deg = compute_constraint_violations(state, return_degenerate=True)
    pre_viol = float(np.max(np.abs(pre_phi))) if pre_phi.size else 0.0
    try:
        return satisfy_constraints(
            state, tol=tol, max_iter=shake_max_iter, mobilities=mobilities
        )
    except RuntimeError:
        post_phi, post_deg = compute_constraint_violations(
            state, return_degenerate=True
        )
        post_viol = float(np.max(np.abs(post_phi))) if post_phi.size else 0.0
        if (
            not np.all(np.isfinite(state.positions))
            or (post_deg and not pre_deg)
            or post_viol > pre_viol
        ):
            state.positions[:] = pre_pos
        n = satisfy_constraints_newton(
            state, tol=tol, max_iter=newton_max_iter, mobilities=mobilities
        )
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
    """Accumulate harmonic or tabulated COFFDROP bond force into state.forces."""
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
    cd_params = state.common.coffdrop_params
    if (
        bond.type_idx >= 0
        and cd_params is not None
        and getattr(cd_params, "bond_pots", None)
    ):
        f_mag = -cd_params.bond_pots[bond.type_idx].deriv(r)  # kBT/A
    else:
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
    if not (
        np.all(np.isfinite(ri)) and np.all(np.isfinite(rj)) and np.all(np.isfinite(rk))
    ):
        return
    if np.max(np.abs(ri)) > 1e6 or np.max(np.abs(rj)) > 1e6 or np.max(np.abs(rk)) > 1e6:
        return
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
    outside all grids contribute zero (negligible far-field). Every grid but
    the coarsest is used only a margin in from its faces, 3 spacings (1 on a
    grid of at most 7 points per side), since a focused APBS grid carries
    boundary values interpolated from the coarser one and the gradient there is
    one sided. The GPU engine keeps the same margin.

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
    for g, grid in enumerate(sorted_grids):
        idx_list = []
        spacing = np.diag(grid.delta)
        extent = (np.array(grid.data.shape) - 1) * spacing
        n_min = min(grid.data.shape)
        cells = (
            0
            if g == len(sorted_grids) - 1
            else 3 if n_min > 7 else 1 if n_min > 3 else 0
        )
        margin = cells * spacing
        for i in range(n_atoms):
            if assigned[i]:
                continue
            if abs(atom_charges[i]) < 1e-9:
                assigned[i] = True
                continue
            pt = positions[i]
            local = pt - grid.origin
            if (local >= margin).all() and (local <= extent - margin).all():
                idx_list.append(i)
                assigned[i] = True
        if not idx_list:
            continue
        sub_pos = np.ascontiguousarray(positions[idx_list])
        sub_chg = np.ascontiguousarray(atom_charges[idx_list])
        forces = grid.batch_force_on_charges(sub_pos, sub_chg)
        for k, i in enumerate(idx_list):
            state.forces[i] += forces[k]


def _scatter(F: np.ndarray, idx: np.ndarray, vec: np.ndarray) -> None:
    """Add the rows of vec into F at the atom indices idx."""
    n = F.shape[0]
    for c in range(3):
        F[:, c] += np.bincount(idx, weights=vec[:, c], minlength=n)


def _spline_table(pots) -> dict:
    """Stack piecewise cubic coefficients of a list of tabulated potentials.

    Interval m of potential t holds coef[:, start[t] + m] for
    c0 s^3 + c1 s^2 + c2 s + c3 with s measured from interval start.
    """
    coefs, start, x_min, x_max, dx, n_int = [], [], [], [], [], []
    left_slope, is_periodic = [], []
    offset = 0
    for pot in pots:
        if pot._spline is not None:
            c = np.asarray(pot._spline.c, dtype=float)
        else:
            v = np.asarray(pot.values, dtype=float)
            c = np.zeros((4, max(len(v) - 1, 1)))
            if len(v) > 1:
                c[2] = np.diff(v) / pot._dx
                c[3] = v[:-1]
        coefs.append(c)
        start.append(offset)
        offset += c.shape[1]
        x_min.append(pot.x_min)
        x_max.append(pot.x_max)
        dx.append(pot._dx)
        n_int.append(c.shape[1])
        left_slope.append(float(getattr(pot, "left_slope", 0.0)))
        is_periodic.append(bool(getattr(pot, "is_periodic", False)))
    return dict(
        coef=np.concatenate(coefs, axis=1) if coefs else np.zeros((4, 0)),
        start=np.array(start, dtype=np.int64),
        x_min=np.array(x_min, dtype=float),
        x_max=np.array(x_max, dtype=float),
        dx=np.array(dx, dtype=float),
        n_int=np.array(n_int, dtype=np.int64),
        left_slope=np.array(left_slope, dtype=float),
        is_periodic=np.array(is_periodic, dtype=bool),
    )


def _table_derivative(params, kind: str, types: np.ndarray, x: np.ndarray):
    """First derivative of tabulated potentials types at x.

    kind is "pair", "angle", or "dihedral". Guards finite in-range values
    before integer indexing, includes x_min for periodic tables, and applies
    repulsive left_slope for pair tables when x <= x_min.
    """
    cache = params.__dict__.setdefault("_tables", {})
    if kind not in cache:
        cache[kind] = _spline_table(getattr(params, kind + "_pots"))
    tab = cache[kind]
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    if x.size == 0:
        return out
    x0 = tab["x_min"][types]
    x1 = tab["x_max"][types]
    per = tab["is_periodic"][types]
    finite = np.isfinite(x)
    lo_ok = np.where(per, x >= x0, x > x0)
    inside = finite & lo_ok & (x < x1)
    if np.any(inside):
        t_in = types[inside]
        x_in = x[inside]
        x0_in = x0[inside]
        h_in = tab["dx"][t_in]
        m = np.clip(
            np.floor((x_in - x0_in) / h_in).astype(np.int64),
            0,
            tab["n_int"][t_in] - 1,
        )
        s = x_in - (x0_in + h_in * m)
        c = tab["coef"][:, tab["start"][t_in] + m]
        out[inside] = (3.0 * c[0] * s + 2.0 * c[1]) * s + c[2]
    ls = tab["left_slope"][types]
    if np.any(ls != 0.0):
        left_mask = finite & (x <= x0) & (ls != 0.0)
        if np.any(left_mask):
            out[left_mask] = ls[left_mask]
    return out


def _term_arrays(common: "ChainCommon") -> dict:
    """Index and parameter arrays of the bonded terms, kept on the chain.

    ChainCommon is immutable by design, so the arrays are rebuilt only when a
    term list is replaced or changes length.
    """
    key = tuple(
        (id(x), len(x))
        for x in (common.bonds, common.angles, common.torsions, common.network)
    )
    cache = common.__dict__.get("_term_cache")
    if cache is not None and cache[0] == key:
        return cache[1]

    def ids(terms, names):
        return [
            np.array([_chain_idx(getattr(t, s)) for t in terms], dtype=np.int64)
            for s in names
        ]

    def vals(terms, name, dtype=float):
        return np.array([getattr(t, name) for t in terms], dtype=dtype)

    arrays = {}
    for name, terms in (("bond", common.bonds), ("network", common.network)):
        arrays[name] = (*ids(terms, "ab"), vals(terms, "r0"), vals(terms, "k_spring"))
    arrays["bond_type"] = vals(common.bonds, "type_idx", np.int64)
    a = common.angles
    arrays["angle"] = (
        *ids(a, "abc"),
        vals(a, "theta0"),
        vals(a, "k_angle"),
        vals(a, "type_idx", np.int64),
    )
    t = common.torsions
    arrays["torsion"] = (
        *ids(t, "abcd"),
        vals(t, "phi0"),
        vals(t, "k_tor"),
        vals(t, "n"),
        vals(t, "type_idx", np.int64),
    )
    common.__dict__["_term_cache"] = (key, arrays)
    return arrays


def _harmonic_bond_forces(F, pos, ok, terms, bond_types=None, params=None) -> None:
    """Harmonic bonds V = k/2 (r - r0)^2, or tabulated bond_pots when type_idx >= 0."""
    ia, ib, r0, k = terms
    if ia.size == 0:
        return
    d = pos[ib] - pos[ia]
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    use = ok[ia] & ok[ib] & (r >= 1e-8)
    if not np.any(use):
        return
    typ = bond_types
    if not use.all():
        ia, ib, r0, k, d, r = ia[use], ib[use], r0[use], k[use], d[use], r[use]
        if typ is not None:
            typ = typ[use]
    dV = k * (r - r0)
    if typ is not None and params is not None and getattr(params, "bond_pots", None):
        tab = typ >= 0
        if np.any(tab):
            dV[tab] = _table_derivative(params, "bond", typ[tab], r[tab])
    f = (-dV / r)[:, None] * d
    _scatter(F, ia, -f)
    _scatter(F, ib, f)


def _angle_forces(F, pos, ok, terms, params) -> None:
    """Angle forces, tabulated for type_idx >= 0 and harmonic otherwise."""
    ia, ib, ic, theta0, k, typ = terms
    if ia.size == 0:
        return
    ok_t = ok[ia] & ok[ib] & ok[ic]
    if not np.any(ok_t):
        return
    if not ok_t.all():
        ia, ib, ic = ia[ok_t], ib[ok_t], ic[ok_t]
        theta0, k, typ = theta0[ok_t], k[ok_t], typ[ok_t]
    u = pos[ia] - pos[ib]
    v = pos[ic] - pos[ib]
    nu = np.sqrt(np.einsum("ij,ij->i", u, u))
    nv = np.sqrt(np.einsum("ij,ij->i", v, v))
    use = (nu >= 1e-8) & (nv >= 1e-8)
    if not np.any(use):
        return
    if not use.all():
        ia, ib, ic = ia[use], ib[use], ic[use]
        theta0, k, typ = theta0[use], k[use], typ[use]
        u, v, nu, nv = u[use], v[use], nu[use], nv[use]
    u = u / nu[:, None]
    v = v / nv[:, None]
    cos_t = np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0)
    theta = np.arccos(cos_t)
    sin_t = np.sin(theta)
    use2 = np.abs(sin_t) >= 1e-8
    if not np.any(use2):
        return
    if not use2.all():
        ia, ib, ic = ia[use2], ib[use2], ic[use2]
        theta0, k, typ = theta0[use2], k[use2], typ[use2]
        u, v, nu, nv = u[use2], v[use2], nu[use2], nv[use2]
        cos_t, theta, sin_t = cos_t[use2], theta[use2], sin_t[use2]
    dV = k * (theta - theta0)
    if params is not None:
        tab = typ >= 0
        if np.any(tab):
            dV[tab] = _table_derivative(
                params, "angle", typ[tab], np.degrees(theta[tab])
            ) * (180.0 / math.pi)
    coeff = dV / sin_t
    fi = (coeff / nu)[:, None] * (v - cos_t[:, None] * u)
    fk = (coeff / nv)[:, None] * (u - cos_t[:, None] * v)
    _scatter(F, ia, fi)
    _scatter(F, ic, fk)
    _scatter(F, ib, -(fi + fk))


def _torsion_forces(F, pos, ok, terms, params) -> None:
    """Torsion forces, tabulated for type_idx >= 0 and V = k (1 - cos(n phi - phi0)) otherwise."""
    ia, ib, ic, il, phi0, k, n, typ = terms
    if ia.size == 0:
        return
    ok_t = ok[ia] & ok[ib] & ok[ic] & ok[il]
    if not np.any(ok_t):
        return
    if not ok_t.all():
        ia, ib, ic, il = ia[ok_t], ib[ok_t], ic[ok_t], il[ok_t]
        phi0, k, n, typ = phi0[ok_t], k[ok_t], n[ok_t], typ[ok_t]
    f = pos[ia] - pos[ib]
    g = pos[ib] - pos[ic]
    h = pos[il] - pos[ic]
    a = np.cross(f, g)
    b = np.cross(h, g)
    a2 = np.einsum("ij,ij->i", a, a)
    b2 = np.einsum("ij,ij->i", b, b)
    gn = np.sqrt(np.einsum("ij,ij->i", g, g))
    use = (a2 >= 1e-24) & (b2 >= 1e-24) & (gn >= 1e-12)
    if not np.any(use):
        return
    if not use.all():
        ia, ib, ic, il = ia[use], ib[use], ic[use], il[use]
        phi0, k, n, typ = phi0[use], k[use], n[use], typ[use]
        f, g, h = f[use], g[use], h[use]
        a, b, a2, b2, gn = a[use], b[use], a2[use], b2[use], gn[use]
    phi = np.arctan2(-gn * np.einsum("ij,ij->i", f, b), np.einsum("ij,ij->i", a, b))
    dV = k * n * np.sin(n * phi - phi0)
    if params is not None:
        tab = typ >= 0
        if np.any(tab):
            deg = np.mod(np.degrees(phi[tab]), 360.0)
            deg = np.where(deg >= 360.0, 0.0, deg)
            dV[tab] = _table_derivative(params, "dihedral", typ[tab], deg) * (
                180.0 / math.pi
            )
    fg = np.einsum("ij,ij->i", f, g) / gn
    hg = np.einsum("ij,ij->i", h, g) / gn
    grad_i = -(gn / a2)[:, None] * a
    grad_l = (gn / b2)[:, None] * b
    grad_j = ((gn + fg) / a2)[:, None] * a - (hg / b2)[:, None] * b
    grad_k = -grad_i - grad_j - grad_l
    _scatter(F, ia, -dV[:, None] * grad_i)
    _scatter(F, ib, -dV[:, None] * grad_j)
    _scatter(F, ic, -dV[:, None] * grad_k)
    _scatter(F, il, -dV[:, None] * grad_l)


def chain_intra_debye_huckel_forces(
    common: "ChainCommon",
    positions: np.ndarray,
    forces: np.ndarray,
    debye_length: float,
    dielectric: float = SOLVENT_DIELECTRIC,
    kT_scale: float = 1.0,
) -> float:
    """Accumulate intra-chain screened Coulomb forces into forces and return energy (kBT).

    Evaluates U_ij = q_i q_j l_B exp(-r / lambda_D) / r and
    F_i = q_i q_j l_B exp(-r / lambda_D) (1 + r / lambda_D) / r^2 * (r_i - r_j) / r
    for all non-excluded pairs (outside bonded_exclusions) with |q_i q_j| > 1e-12.
    """
    n = common.n_atoms
    if n < 2 or debye_length is None or debye_length <= 0.0:
        return 0.0
    charges_tuple = tuple(float(a.charge) for a in common.atoms)
    has_cd = common.coffdrop_params is not None
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
        charges_tuple,
        tuple(int(a.resid) for a in common.atoms) if has_cd else (),
    )
    cache = common.__dict__.get("_dh_cache")
    if cache is None or cache[0] != key:
        q = np.array(charges_tuple, dtype=np.float64)
        skip = _coffdrop_exclusion_mask(common) if has_cd else bonded_exclusions(common)
        iu, ju = np.triu_indices(n, k=1)
        qij = q[iu] * q[ju]
        m = (~skip[iu, ju]) & (np.abs(qij) > 1e-12)
        cache = (key, iu[m].copy(), ju[m].copy(), qij[m].copy())
        common.__dict__["_dh_cache"] = cache
    _, iu, ju, qij = cache
    if iu.size == 0:
        return 0.0
    pos = np.asarray(positions, dtype=np.float64)
    ok = np.all(np.isfinite(pos), axis=1) & (np.max(np.abs(pos), axis=1) <= 1e6)
    ok_p = ok[iu] & ok[ju]
    if not np.any(ok_p):
        return 0.0
    if not ok_p.all():
        iu, ju, qij = iu[ok_p], ju[ok_p], qij[ok_p]
    dr = pos[iu] - pos[ju]
    r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
    use = r > 1e-10
    if not np.any(use):
        return 0.0
    if not use.all():
        iu, ju, qij, dr, r = iu[use], ju[use], qij[use], dr[use], r[use]
    l_B = 1.0 / (
        4.0 * math.pi * float(dielectric) * VACUUM_PERMITTIVITY_KBT * float(kT_scale)
    )
    kappa = 0.0 if math.isinf(debye_length) else 1.0 / float(debye_length)
    kr = kappa * r
    exp_kr = np.exp(-kr)
    u_pairs = qij * l_B * exp_kr / r
    f_over_r = u_pairs * (1.0 + kr) / (r * r)
    fvec = f_over_r[:, None] * dr
    _scatter(forces, iu, fvec)
    _scatter(forces, ju, -fvec)
    return float(np.sum(u_pairs))


def compute_chain_forces(
    state: "ChainState",
    kT: float = 1.0,
    soft_repulsion: bool = False,
    soft_repulsion_eps: float = 1.0,
    target_grids: Optional[list] = None,
    debye_length: Optional[float] = None,
    dielectric: float = SOLVENT_DIELECTRIC,
    kT_scale: float = 1.0,
) -> None:
    """Fill state.forces (in kBT/A) from internal chain interactions."""
    state.zero_forces()
    common = state.common
    pos = state.positions
    F = state.forces
    ok = np.all(np.isfinite(pos), axis=1) & (np.max(np.abs(pos), axis=1) <= 1e6)
    terms = _term_arrays(common)
    params = common.coffdrop_params
    _harmonic_bond_forces(F, pos, ok, terms["bond"], terms["bond_type"], params)
    _harmonic_bond_forces(F, pos, ok, terms["network"])
    _angle_forces(F, pos, ok, terms["angle"], params)
    _torsion_forces(F, pos, ok, terms["torsion"], params)
    if params is not None and common.pair_lookups:
        F += chain_intra_coffdrop_pair_forces(state, common)
    if debye_length is not None and debye_length > 0.0:
        chain_intra_debye_huckel_forces(
            common,
            pos,
            F,
            debye_length=debye_length,
            dielectric=dielectric,
            kT_scale=kT_scale,
        )
    if target_grids:
        atom_charges = np.array([a.charge for a in common.atoms])
        compute_target_grid_forces(state, target_grids, atom_charges)
    if soft_repulsion:
        F += chain_intra_nonbonded_forces(state, common, eps=soft_repulsion_eps)


def bonded_exclusions(common: "ChainCommon") -> np.ndarray:
    """Return (n, n) mask of 1-2 bonded/constrained pairs and 1-3 angle pairs."""
    n = common.n_atoms
    bonded = np.zeros((n, n), dtype=bool)
    for bond in common.bonds:
        i, j = _chain_idx(bond.a), _chain_idx(bond.b)
        bonded[i, j] = bonded[j, i] = True
    for lc in common.length_constraints:
        i, j = _chain_idx(lc.a), _chain_idx(lc.b)
        bonded[i, j] = bonded[j, i] = True
    one_three = (bonded.astype(np.int64) @ bonded.astype(np.int64)) > 0
    excluded = bonded | one_three
    for ang in common.angles:
        ia, ic = _chain_idx(ang.a), _chain_idx(ang.c)
        excluded[ia, ic] = excluded[ic, ia] = True
    np.fill_diagonal(excluded, True)
    return excluded


def coffdrop_nonbonded_exclusions(common: "ChainCommon") -> Set[Tuple[int, int]]:
    """Return canonical (i, j) pairs (i < j) excluded from COFFDROP (0,0) and intra-DH.

    Matches BrownDye 2 aux/coffdrop_chain.cc (lines 392-444):
    - 1-2 bonds and length constraints, 1-3 bond/constraint paths, and angle pairs;
    - all 6 atom pairs from every proper or improper torsion in common.torsions;
    - all intra-residue pairs (assigning negative-resid cap beads to their bonded
      terminal residue, matching COFFDROP map.xml);
    - all pairs spanning two distinct residues connected by a bond or length
      constraint (preserving chain breaks where no connecting bond exists).
    """
    n = common.n_atoms
    excluded: Set[Tuple[int, int]] = set()
    if n < 2:
        return excluded

    base_mask = bonded_exclusions(common)
    iu, ju = np.nonzero(np.triu(base_mask, k=1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        excluded.add((i, j))

    for ang in common.angles:
        ids = (_chain_idx(ang.a), _chain_idx(ang.b), _chain_idx(ang.c))
        for p in range(3):
            for q in range(p + 1, 3):
                u, v = ids[p], ids[q]
                if u != v:
                    excluded.add((u, v) if u < v else (v, u))

    for tor in common.torsions:
        ids = (
            _chain_idx(tor.a),
            _chain_idx(tor.b),
            _chain_idx(tor.c),
            _chain_idx(tor.d),
        )
        for p in range(4):
            for q in range(p + 1, 4):
                u, v = ids[p], ids[q]
                if u != v:
                    excluded.add((u, v) if u < v else (v, u))

    eff_resid = [int(a.resid) for a in common.atoms]
    bond_edges: List[Tuple[int, int]] = []
    for bond in common.bonds:
        bond_edges.append((_chain_idx(bond.a), _chain_idx(bond.b)))
    for lc in common.length_constraints:
        bond_edges.append((_chain_idx(lc.a), _chain_idx(lc.b)))

    for i in range(n):
        if eff_resid[i] < 0:
            eff_resid[i] = -1 - i
    for ia, ib in bond_edges:
        ra, rb = common.atoms[ia].resid, common.atoms[ib].resid
        if ra < 0 <= rb:
            eff_resid[ia] = int(rb)
        elif rb < 0 <= ra:
            eff_resid[ib] = int(ra)

    res_atoms: Dict[int, List[int]] = {}
    for i, r in enumerate(eff_resid):
        res_atoms.setdefault(r, []).append(i)

    for atom_list in res_atoms.values():
        m = len(atom_list)
        for p in range(m):
            u = atom_list[p]
            for q in range(p + 1, m):
                v = atom_list[q]
                excluded.add((u, v) if u < v else (v, u))

    bonded_res_pairs: Set[Tuple[int, int]] = set()
    for ia, ib in bond_edges:
        ra, rb = eff_resid[ia], eff_resid[ib]
        if ra != rb:
            bonded_res_pairs.add((ra, rb) if ra < rb else (rb, ra))

    for r1, r2 in bonded_res_pairs:
        atoms1 = res_atoms[r1]
        atoms2 = res_atoms[r2]
        for u in atoms1:
            for v in atoms2:
                if u != v:
                    excluded.add((u, v) if u < v else (v, u))

    return excluded


def _coffdrop_exclusion_mask(common: "ChainCommon") -> np.ndarray:
    """Return (n, n) symmetric boolean mask from coffdrop_nonbonded_exclusions(common)."""
    n = common.n_atoms
    mask = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(mask, True)
    excl = coffdrop_nonbonded_exclusions(common)
    if excl:
        ij = np.array(list(excl), dtype=np.int64).reshape(-1, 2)
        mask[ij[:, 0], ij[:, 1]] = True
        mask[ij[:, 1], ij[:, 0]] = True
    return mask


def chain_intra_nonbonded_forces(
    state: "ChainState",
    common: "ChainCommon",
    eps: float = 1.0,
) -> np.ndarray:
    """Intra-chain WCA forces between non-bonded bead pairs in kBT/A."""
    n = common.n_atoms
    F = np.zeros((n, 3))
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
    cache = common.__dict__.get("_wca_cache")
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
        common.__dict__["_wca_cache"] = cache
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
    sr6 = (s / r) ** 6
    sr12 = sr6 * sr6
    fvec = (4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / (r * r))[:, None] * dr
    _scatter(F, iu, fvec)
    _scatter(F, ju, -fvec)
    return F


def chain_intra_coffdrop_pair_forces(
    state: "ChainState",
    common: "ChainCommon",
) -> np.ndarray:
    """Intra-chain non-bonded pair forces from COFFDROP tabulated potentials in kBT/A."""
    n = common.n_atoms
    F = np.zeros((n, 3))
    pair_lookups = common.pair_lookups
    if not pair_lookups:
        return F
    key = (id(pair_lookups), len(pair_lookups))
    cache = common.__dict__.get("_pair_cache")
    if cache is None or cache[0] != key:
        ijt = np.array(
            [(i, j, t) for (i, j), t in pair_lookups.items()], dtype=np.int64
        ).reshape(-1, 3)
        cache = (key, ijt[:, 0].copy(), ijt[:, 1].copy(), ijt[:, 2].copy())
        common.__dict__["_pair_cache"] = cache
    _, i_arr, j_arr, t_arr = cache
    pos = state.positions
    ok = np.all(np.isfinite(pos), axis=1) & (np.max(np.abs(pos), axis=1) <= 1e6)
    ok_p = ok[i_arr] & ok[j_arr]
    if not np.any(ok_p):
        return F
    if not ok_p.all():
        i_arr, j_arr, t_arr = i_arr[ok_p], j_arr[ok_p], t_arr[ok_p]
    dr = pos[i_arr] - pos[j_arr]
    r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
    use = r > 1e-10
    if not np.any(use):
        return F
    if not use.all():
        i_arr, j_arr, t_arr, dr, r = (
            i_arr[use],
            j_arr[use],
            t_arr[use],
            dr[use],
            r[use],
        )
    dV_dr = _table_derivative(common.coffdrop_params, "pair", t_arr, r)
    fvec = (-dV_dr / r)[:, None] * dr
    _scatter(F, i_arr, fvec)
    _scatter(F, j_arr, -fvec)
    return F


def elastic_network(
    common: "ChainCommon",
    positions: np.ndarray,
    cutoff: float,
    k_spring: float,
) -> List["ChainBond"]:
    """Springs of an elastic network that holds the fold of the reference
    positions.

    Every pair of backbone beads (bead name CA) closer than cutoff in the
    reference positions gets a harmonic spring V = k_spring/2 (r - r0)^2,
    where r0 is the reference distance, unless the pair is joined by a bond
    or shares a bonded neighbor. cutoff is in Å and k_spring in kBT/Å^2. The
    bead name is the part of ChainAtom.resname after the colon.
    """
    if not (cutoff > 0.0 and k_spring > 0.0):
        raise ValueError(
            f"elastic network needs cutoff > 0 and k_spring > 0, got {cutoff} "
            f"and {k_spring}"
        )
    pos = np.asarray(positions, dtype=float)
    backbone = np.array(
        [a.resname.split(":")[-1].strip().upper() == "CA" for a in common.atoms]
    )
    if backbone.sum() < 2:
        raise ValueError("elastic network needs at least two beads named CA")
    skip = bonded_exclusions(common)
    idx = np.flatnonzero(backbone)
    iu, ju = np.triu_indices(idx.size, k=1)
    i, j = idx[iu], idx[ju]
    r0 = np.linalg.norm(pos[i] - pos[j], axis=1)
    m = (r0 < cutoff) & ~skip[i, j]
    return [
        ChainBond(
            a=ChainAtomRef(int(a)),
            b=ChainAtomRef(int(b)),
            r0=float(d),
            k_spring=k_spring,
        )
        for a, b, d in zip(i[m], j[m], r0[m])
    ]


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
        b2_len = math.sqrt(a16 * a16 + a17 * a17 + a18 * a18)

        if a28 < 1e-12 or a29 < 1e-12 or b2_len < 1e-12:
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
            phi_deg = (phi * 180.0 / math.pi) % 360.0
            if phi_deg >= 360.0:
                phi_deg = 0.0
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
        """WCA excluded volume between bead pairs other than the 1-2 and 1-3
        pairs, whose distances the bond and angle terms hold. The repulsion
        ends at the Lennard-Jones minimum 2^(1/6) sig, where force and shifted
        energy both vanish."""
        n = chain.n_beads
        F = np.zeros((n, 3))
        bonded = np.zeros((n, n), dtype=bool)
        for b in chain.bonds:
            bonded[b.a.atom_idx, b.b.atom_idx] = bonded[b.b.atom_idx, b.a.atom_idx] = (
                True
            )
        skip = bonded | ((bonded.astype(np.int64) @ bonded.astype(np.int64)) > 0)
        for ang in chain.angles:
            skip[ang.a.atom_idx, ang.c.atom_idx] = skip[
                ang.c.atom_idx, ang.a.atom_idx
            ] = True
        wca_cut = 2.0 ** (1.0 / 6.0)
        for i in range(n):
            for j in range(i + 1, n):
                if skip[i, j]:
                    continue
                ri = chain.beads[i].pos
                rj = chain.beads[j].pos
                dr = (
                    ri - rj
                )  # vector from j toward i (so f_mag*dr pushes i away from j)
                r = float(np.linalg.norm(dr))
                sig = chain.beads[i].radius + chain.beads[j].radius
                if r < 1e-8 or r >= wca_cut * sig:
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
        # Propagate each bead: b.force is in kBT/A and D = kT * mob (A^2/ps),
        # so drift = D * F_kBT * dt = kT * mob * b.force * dt matches noise sqrt(2 D dt).
        for i, b in enumerate(chain.beads):
            mob = 1.0 / (6.0 * math.pi * self.eta * b.radius)
            drift = self.kT * mob * b.force * dt
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


_CANONICAL_RESIDUES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "HIP",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

# Amber/CHARMM protonation variants -> (canonical COFFDROP name, neutralize_charge).
_RESIDUE_VARIANT_MAP: Dict[str, Tuple[str, bool]] = {
    "HIE": ("HIS", False),
    "HID": ("HIS", False),
    "HSE": ("HIS", False),
    "HSD": ("HIS", False),
    "HSP": ("HIP", False),
    "CYX": ("CYS", False),
    "CYM": ("CYS", False),
    "ASH": ("ASP", True),
    "GLH": ("GLU", True),
    "LYN": ("LYS", True),
}

_PDB_SKIP_RESNAMES = frozenset(
    set(GHOST_NAMES)
    | {
        "HOH",
        "WAT",
        "TIP",
        "TIP3",
        "SOL",
        "NA",
        "CL",
        "K",
        "MG",
        "ZN",
        "CA",
        "FE",
        "MN",
        "CU",
        "CO",
        "NI",
        "CD",
        "HG",
        "SO4",
        "PO4",
    }
)


def _normalize_residue_name(raw: str) -> Tuple[str, bool]:
    """Normalize Amber/CHARMM protonation and terminal residue names.

    Returns (canonical_name, neutral_override). When neutral_override is True
    (ASH, GLH, LYN), the sidechain charge is forced to 0.0.
    """
    r = raw.strip().upper()
    if r in _RESIDUE_VARIANT_MAP:
        return _RESIDUE_VARIANT_MAP[r]
    if r in _CANONICAL_RESIDUES:
        return (r, False)
    if len(r) == 4 and r[0] in ("N", "C"):
        tail = r[1:]
        if tail in _RESIDUE_VARIANT_MAP:
            return _RESIDUE_VARIANT_MAP[tail]
        if tail in _CANONICAL_RESIDUES:
            return (tail, False)
    return (r, False)


def _lookup_coffdrop_pot(
    params, kind: str, ri: Tuple[int, ...], ai: Tuple[int, ...], orders: Tuple[int, ...]
):
    """Look up a tabulated COFFDROP potential using indexed dispatch when present."""
    if hasattr(params, "match_pot"):
        return params.match_pot(kind, ri, ai, orders)
    from pystarc.simulation.coffdrop_params import _match_pot

    return _match_pot(getattr(params, kind + "_pots"), ri, ai, orders)


def build_chain_common_from_coffdrop(
    residues: List[str],
    params,
    name: Optional[str] = None,
    k_spring: float = 100.0,
    chain_breaks: Optional[Set[int]] = None,
    hydro_radius: float = 0.0,
    use_bond_tables: bool = False,
) -> "ChainCommon":
    """Build a backbone CA COFFDROP-aware ChainCommon from a peptide sequence."""
    if not residues:
        raise ValueError("residues list cannot be empty")
    breaks = set(chain_breaks) if chain_breaks else set()
    norm_info = [_normalize_residue_name(r) for r in residues]
    canon_residues = [c for c, _ in norm_info]
    neutral_flags = [neut for _, neut in norm_info]
    n = len(canon_residues)

    ca_atom_idx = params.type_map["atoms"].get("CA")
    if ca_atom_idx is None:
        raise ValueError("CA bead type not found in params.type_map['atoms']")

    def _ri(rname):
        return params.type_map["residues"].get(rname, 0)

    atoms = []
    for i, (rname, neutral) in enumerate(zip(canon_residues, neutral_flags)):
        charge = 0.0 if neutral else params.bead_charge(rname, "CA")
        atoms.append(
            ChainAtom(
                radius=2.0,
                charge=charge,
                resname=f"{rname}:CA",
                resid=i,
                hydro_radius=float(hydro_radius),
            )
        )

    bonds = []
    ca_ca_len = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
    ca_bond_pot = (
        params.match_bond_pot("XXX", "CA", 1, "XXX", "CA", 2)
        if use_bond_tables and hasattr(params, "match_bond_pot")
        else None
    )
    ca_bond_type = ca_bond_pot.index if ca_bond_pot is not None else -1
    for i in range(n - 1):
        if i in breaks:
            continue
        bonds.append(
            ChainBond(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                r0=ca_ca_len,
                k_spring=k_spring,
                type_idx=ca_bond_type,
            )
        )

    angles = []
    for i in range(n - 2):
        if i in breaks or (i + 1) in breaks:
            continue
        ri = (
            _ri(canon_residues[i]),
            _ri(canon_residues[i + 1]),
            _ri(canon_residues[i + 2]),
        )
        ai = (ca_atom_idx, ca_atom_idx, ca_atom_idx)
        if hasattr(params, "match_angle_diff"):
            pot = params.match_angle_diff(ri, ai, (1, 2))
        else:
            pot = _lookup_coffdrop_pot(
                params, "angle", (ri[0], ri[1], 0), ai, (2, 3, 4)
            )
            if pot is None:
                pot = _lookup_coffdrop_pot(
                    params, "angle", (0, ri[1], ri[2]), ai, (1, 2, 3)
                )
        if pot is None:
            continue
        angles.append(
            ChainAngle(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                c=ChainAtomRef(i + 2),
                theta0=2.0,
                k_angle=10.0,
                type_idx=pot.index,
            )
        )

    torsions = []
    for i in range(n - 3):
        if i in breaks or (i + 1) in breaks or (i + 2) in breaks:
            continue
        ri = (
            _ri(canon_residues[i]),
            _ri(canon_residues[i + 1]),
            _ri(canon_residues[i + 2]),
            _ri(canon_residues[i + 3]),
        )
        ai = (ca_atom_idx,) * 4
        if hasattr(params, "match_dihedral_diff"):
            pot = params.match_dihedral_diff(ri, ai, (1, 2, 3))
        else:
            pot = _lookup_coffdrop_pot(
                params, "dihedral", (0, ri[1], ri[2], 0), ai, (1, 2, 3, 4)
            )
        if pot is None:
            continue
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                c=ChainAtomRef(i + 2),
                d=ChainAtomRef(i + 3),
                phi0=0.0,
                k_tor=2.0,
                n=1,
                type_idx=pot.index,
            )
        )

    common = ChainCommon(
        name=name or ("-".join(canon_residues[:3]) + ("..." if n > 3 else "")),
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        coffdrop_params=params,
    )
    excluded = coffdrop_nonbonded_exclusions(common)

    pair_lookups = {}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            ri = (_ri(canon_residues[i]), _ri(canon_residues[j]))
            ai = (ca_atom_idx, ca_atom_idx)
            pot = _lookup_coffdrop_pot(params, "pair", ri, ai, (0, 0))
            if pot is None:
                pot = _lookup_coffdrop_pot(
                    params, "pair", (ri[1], ri[0]), (ai[1], ai[0]), (0, 0)
                )
            if pot is not None:
                pair_lookups[(i, j)] = pot.index

    common.pair_lookups = pair_lookups
    return common


def build_chain_common_with_sidechains_from_coffdrop(
    residues: List[str],
    params,
    name: Optional[str] = None,
    k_spring: float = 100.0,
    caps: Tuple[Optional[str], Optional[str]] = (None, None),
    chain_breaks: Optional[Set[int]] = None,
    hydro_radius: float = 0.0,
    use_bond_tables: bool = False,
) -> "ChainCommon":
    """Build a COFFDROP-aware ChainCommon with sidechain beads and BD2-matched terms."""
    if not residues:
        raise ValueError("residues list cannot be empty")
    breaks = set(chain_breaks) if chain_breaks else set()
    norm_info = [_normalize_residue_name(r) for r in residues]
    canon_residues = [c for c, _ in norm_info]
    neutral_flags = [neut for _, neut in norm_info]
    n_res = len(canon_residues)

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

    atoms = []
    atom_residue = []
    atom_beadname = []
    ca_global_idx = []
    residue_beads = []

    for r, (resname, neutral) in enumerate(zip(canon_residues, neutral_flags)):
        rdef = params.mapping.get(resname)
        if rdef is None:
            raise ValueError(f"residue {resname} not in params.mapping")
        sidechain_names = [
            b.name
            for b in rdef.beads
            if b.btype == "" and b.location == "" and b.name != "CA"
        ]
        bead_names_this_res = ["CA"] + sidechain_names
        beads_in_res = []
        for bname in bead_names_this_res:
            global_idx = len(atoms)
            charge = 0.0 if neutral else params.bead_charge(resname, bname)
            atoms.append(
                ChainAtom(
                    radius=2.0,
                    charge=charge,
                    resname=f"{resname}:{bname}",
                    resid=r,
                    hydro_radius=float(hydro_radius),
                )
            )
            atom_residue.append(r)
            atom_beadname.append(bname)
            beads_in_res.append(global_idx)
        residue_beads.append(beads_in_res)
        ca_global_idx.append(beads_in_res[0])

    cn_global_idx = -1
    cc_global_idx = -1
    if n_cap == "ACE":
        cn_global_idx = len(atoms)
        atoms.append(
            ChainAtom(
                radius=2.0,
                charge=0.0,
                resname="ACE:CN",
                resid=-1,
                hydro_radius=float(hydro_radius),
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
                hydro_radius=float(hydro_radius),
            )
        )
        atom_residue.append(-1)
        atom_beadname.append("CC")

    n_atoms = len(atoms)

    # Cap beads belong to their flanking terminal residue in COFFDROP map.xml.
    res_all_atoms: List[List[int]] = [list(beads) for beads in residue_beads]
    if cn_global_idx >= 0:
        res_all_atoms[0].append(cn_global_idx)
    if cc_global_idx >= 0:
        res_all_atoms[n_res - 1].append(cc_global_idx)

    # Segment-shifted sequence index separates chain breaks by +10 so no
    # relative residue diff crosses an unbonded break.
    seq_coord_res: List[int] = [0] * n_res
    for r in range(1, n_res):
        step = 10 if (r - 1) in breaks else 1
        seq_coord_res[r] = seq_coord_res[r - 1] + step
    seq_to_res: Dict[int, int] = {seq_coord_res[r]: r for r in range(n_res)}

    def _eff_res(atom_idx: int) -> int:
        r = atom_residue[atom_idx]
        if r >= 0:
            return r
        if atom_idx == cn_global_idx:
            return 0
        if atom_idx == cc_global_idx:
            return n_res - 1
        return 0

    eff_res_arr = [_eff_res(i) for i in range(n_atoms)]
    atom_seq = [seq_coord_res[eff_res_arr[i]] for i in range(n_atoms)]
    ri_arr = [_ri(canon_residues[eff_res_arr[i]]) for i in range(n_atoms)]
    ai_arr = [_ai(atom_beadname[i]) for i in range(n_atoms)]

    def _bond_type_idx(
        res0: str, bead0: str, ord0: int, res1: str, bead1: str, ord1: int
    ) -> int:
        if not use_bond_tables or not hasattr(params, "match_bond_pot"):
            return -1
        bpot = params.match_bond_pot(res0, bead0, ord0, res1, bead1, ord1)
        return bpot.index if bpot is not None else -1

    # Bonds (equilibrium lengths read from coffdrop.xml eq_length when available).
    bonds = []
    ca_ca_len = params.bond_length("XXX", "CA", 1, "XXX", "CA", 2) or 3.8
    ca_ca_type = _bond_type_idx("XXX", "CA", 1, "XXX", "CA", 2)
    for r in range(n_res - 1):
        if r in breaks:
            continue
        bonds.append(
            ChainBond(
                a=ChainAtomRef(ca_global_idx[r]),
                b=ChainAtomRef(ca_global_idx[r + 1]),
                r0=ca_ca_len,
                k_spring=k_spring,
                type_idx=ca_ca_type,
            )
        )
    if cn_global_idx >= 0:
        ca_cn_len = params.bond_length("XXX", "CA", 1, "XXX", "CN", 1) or 3.81
        ca_cn_type = _bond_type_idx("XXX", "CA", 1, "XXX", "CN", 1)
        bonds.append(
            ChainBond(
                a=ChainAtomRef(cn_global_idx),
                b=ChainAtomRef(ca_global_idx[0]),
                r0=ca_cn_len,
                k_spring=k_spring,
                type_idx=ca_cn_type,
            )
        )
    if cc_global_idx >= 0:
        ca_cc_len = params.bond_length("XXX", "CA", 1, "XXX", "CC", 1) or 3.82
        ca_cc_type = _bond_type_idx("XXX", "CA", 1, "XXX", "CC", 1)
        bonds.append(
            ChainBond(
                a=ChainAtomRef(ca_global_idx[-1]),
                b=ChainAtomRef(cc_global_idx),
                r0=ca_cc_len,
                k_spring=k_spring,
                type_idx=ca_cc_type,
            )
        )
    for r in range(n_res):
        beads = residue_beads[r]
        if len(beads) < 2:
            continue
        resname = canon_residues[r]
        for k in range(len(beads) - 1):
            bead_a_name = atom_beadname[beads[k]]
            bead_b_name = atom_beadname[beads[k + 1]]
            r0 = (
                params.bond_length(resname, bead_a_name, 1, resname, bead_b_name, 1)
                or 3.0
            )
            b_type = _bond_type_idx(resname, bead_a_name, 1, resname, bead_b_name, 1)
            bonds.append(
                ChainBond(
                    a=ChainAtomRef(beads[k]),
                    b=ChainAtomRef(beads[k + 1]),
                    r0=r0,
                    k_spring=k_spring,
                    type_idx=b_type,
                )
            )

    # Angles: exact BD2 aux/coffdrop_chain.cc:820-896 diff and wildcard selection.
    angles = []
    angle_diffs = params.angle_diffs()
    for r0 in range(n_res):
        s0 = seq_coord_res[r0]
        rt0 = _ri(canon_residues[r0])
        atoms0 = res_all_atoms[r0]
        for d1, d2 in angle_diffs:
            r1 = seq_to_res.get(s0 + d1)
            r2 = seq_to_res.get(s0 + d2)
            if r1 is None or r2 is None:
                continue
            rt1 = _ri(canon_residues[r1])
            rt2 = _ri(canon_residues[r2])
            atoms1 = res_all_atoms[r1]
            atoms2 = res_all_atoms[r2]
            for ia0 in atoms0:
                at0 = ai_arr[ia0]
                for ia1 in atoms1:
                    if ia1 == ia0:
                        continue
                    at1 = ai_arr[ia1]
                    for ia2 in atoms2:
                        if ia2 == ia0 or ia2 == ia1:
                            continue
                        pot = params.match_angle_diff(
                            (rt0, rt1, rt2), (at0, at1, ai_arr[ia2]), (d1, d2)
                        )
                        if pot is not None:
                            angles.append(
                                ChainAngle(
                                    a=ChainAtomRef(ia0),
                                    b=ChainAtomRef(ia1),
                                    c=ChainAtomRef(ia2),
                                    theta0=2.0,
                                    k_angle=10.0,
                                    type_idx=pot.index,
                                )
                            )

    # Proper and improper dihedrals: exact BD2 aux/coffdrop_chain.cc:521-759 graph walk.
    atom_nebors: List[List[int]] = [[] for _ in range(n_atoms)]
    for bond in bonds:
        ia, ib = bond.a.atom_idx, bond.b.atom_idx
        atom_nebors[ia].append(ib)
        atom_nebors[ib].append(ia)

    torsions = []
    # Proper dihedrals along simple 4-atom bonded paths i0-i1-i2-i3.
    for i0 in range(n_atoms):
        s0, rt0, at0 = atom_seq[i0], ri_arr[i0], ai_arr[i0]
        for i1 in atom_nebors[i0]:
            s1, rt1, at1 = atom_seq[i1], ri_arr[i1], ai_arr[i1]
            d1 = s1 - s0
            for i2 in atom_nebors[i1]:
                if i2 == i0:
                    continue
                s2, rt2, at2 = atom_seq[i2], ri_arr[i2], ai_arr[i2]
                d2 = s2 - s0
                for i3 in atom_nebors[i2]:
                    if i3 == i0 or i3 == i1:
                        continue
                    pot = params.match_dihedral_diff(
                        (rt0, rt1, rt2, ri_arr[i3]),
                        (at0, at1, at2, ai_arr[i3]),
                        (d1, d2, atom_seq[i3] - s0),
                    )
                    if pot is not None:
                        torsions.append(
                            ChainTorsion(
                                a=ChainAtomRef(i0),
                                b=ChainAtomRef(i1),
                                c=ChainAtomRef(i2),
                                d=ChainAtomRef(i3),
                                phi0=0.0,
                                k_tor=2.0,
                                n=1,
                                type_idx=pot.index,
                            )
                        )

    # Improper dihedrals centered on i0 with ordered distinct neighbors (i1, i2, i3)
    # written as (i1, i2, i0, i3), matching aux/coffdrop_chain.cc:655-756.
    for i0 in range(n_atoms):
        nbrs0 = atom_nebors[i0]
        if len(nbrs0) < 3:
            continue
        s0, rt0, at0 = atom_seq[i0], ri_arr[i0], ai_arr[i0]
        for i1 in nbrs0:
            s1, rt1, at1 = atom_seq[i1], ri_arr[i1], ai_arr[i1]
            for i2 in nbrs0:
                if i2 == i1:
                    continue
                s2, rt2, at2 = atom_seq[i2], ri_arr[i2], ai_arr[i2]
                for i3 in nbrs0:
                    if i3 == i1 or i3 == i2:
                        continue
                    pot = params.match_dihedral_diff(
                        (rt1, rt2, rt0, ri_arr[i3]),
                        (at1, at2, at0, ai_arr[i3]),
                        (s2 - s1, s0 - s1, atom_seq[i3] - s1),
                    )
                    if pot is not None:
                        torsions.append(
                            ChainTorsion(
                                a=ChainAtomRef(i1),
                                b=ChainAtomRef(i2),
                                c=ChainAtomRef(i0),
                                d=ChainAtomRef(i3),
                                phi0=0.0,
                                k_tor=2.0,
                                n=1,
                                type_idx=pot.index,
                            )
                        )

    common = ChainCommon(
        name=name
        or ("-".join(canon_residues[:3]) + ("..." if n_res > 3 else "") + "_sc"),
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        coffdrop_params=params,
    )

    # Improper-bond pair tables (diff=0 intra-residue TRP CN-CD/CD-CC and diff=1
    # adjacent-residue (1,2) tables) plus general (0,0) pairs outside exclusions.
    excluded = coffdrop_nonbonded_exclusions(common)
    bonded_12 = {
        (min(b.a.atom_idx, b.b.atom_idx), max(b.a.atom_idx, b.b.atom_idx))
        for b in bonds
    }
    pair_lookups: Dict[Tuple[int, int], int] = {}

    for r0 in range(n_res):
        atoms0 = res_all_atoms[r0]
        rt0 = _ri(canon_residues[r0])
        for ia in atoms0:
            at_a = ai_arr[ia]
            for ib in atoms0:
                if ib == ia:
                    continue
                key_ij = (ia, ib) if ia < ib else (ib, ia)
                if key_ij in bonded_12:
                    continue
                pot = params.match_improper_pair((rt0, rt0), (at_a, ai_arr[ib]), 0)
                if pot is not None:
                    pair_lookups[key_ij] = pot.index
        if r0 + 1 < n_res and r0 not in breaks:
            atoms1 = res_all_atoms[r0 + 1]
            rt1 = _ri(canon_residues[r0 + 1])
            for ia in atoms0:
                at_a = ai_arr[ia]
                for ib in atoms1:
                    key_ij = (ia, ib) if ia < ib else (ib, ia)
                    if key_ij in bonded_12:
                        continue
                    pot = params.match_improper_pair((rt0, rt1), (at_a, ai_arr[ib]), 1)
                    if pot is not None:
                        pair_lookups[key_ij] = pot.index

    for i in range(n_atoms):
        ri_i = ri_arr[i]
        ai_i = ai_arr[i]
        for j in range(i + 1, n_atoms):
            if (i, j) in excluded:
                continue
            ri_j = ri_arr[j]
            ai_j = ai_arr[j]
            pot = _lookup_coffdrop_pot(
                params, "pair", (ri_i, ri_j), (ai_i, ai_j), (0, 0)
            )
            if pot is None:
                pot = _lookup_coffdrop_pot(
                    params, "pair", (ri_j, ri_i), (ai_j, ai_i), (0, 0)
                )
            if pot is not None:
                pair_lookups[(i, j)] = pot.index

    common.pair_lookups = pair_lookups
    return common


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


def _load_coffdrop_params_from_dir(coffdrop_dir: Optional[str]):
    from pathlib import Path as _Path
    from pystarc.simulation.coffdrop_params import COFFDROPParams

    if coffdrop_dir is None:
        coffdrop_dir = str(_Path(__file__).parent.parent / "coffdrop_data")
    p = _Path(coffdrop_dir)
    if not p.is_dir():
        raise FileNotFoundError(
            f"COFFDROP data directory not found: {coffdrop_dir}. "
            "Pass coffdrop_dir explicitly or check pystarc installation."
        )
    return COFFDROPParams.load(
        ff_xml=str(p / "coffdrop.xml"),
        mapping_xml=str(p / "map.xml"),
        connectivity_xml=str(p / "connectivity.xml"),
        charges_xml=str(p / "charges.xml"),
    )


def chain_from_sequence(
    sequence: str,
    coffdrop_dir: Optional[str] = None,
    name: Optional[str] = None,
    sidechains: bool = True,
    k_spring: float = 100.0,
    caps: Tuple[Optional[str], Optional[str]] = (None, None),
    hydro_radius: float = 0.0,
    use_bond_tables: bool = False,
) -> "ChainCommon":
    """Build a COFFDROP chain from a single-letter or 3-letter sequence."""
    params = _load_coffdrop_params_from_dir(coffdrop_dir)

    s = sequence.strip().upper()
    if "-" in s or " " in s:
        residues = [r.strip() for r in s.replace("-", " ").split() if r.strip()]
        for r in residues:
            canon, _ = _normalize_residue_name(r)
            if len(r) == 3:
                continue
            if len(r) == 4 and canon in _CANONICAL_RESIDUES:
                continue
            raise ValueError(
                f"3-letter sequence has invalid token '{r}' "
                "(expected 3-letter residue code)"
            )
    else:
        residues = []
        for c in s:
            if c not in _ONE_LETTER_TO_THREE:
                raise ValueError(
                    f"Unknown single-letter code '{c}' in sequence '{sequence}'. "
                    "Use standard 20 amino acid codes (ARNDCEQGHILKMFPSTWYV)."
                )
            residues.append(_ONE_LETTER_TO_THREE[c])

    if not residues:
        raise ValueError("sequence cannot be empty")

    n_cap, c_cap = caps
    if residues and residues[0] == "ACE":
        if n_cap is None:
            n_cap = "ACE"
        residues = residues[1:]
    if residues and residues[-1] == "NME":
        if c_cap is None:
            c_cap = "NME"
        residues = residues[:-1]
    if not residues:
        raise ValueError("sequence cannot be empty after stripping ACE/NME caps")
    eff_caps = (n_cap, c_cap)

    if sidechains:
        return build_chain_common_with_sidechains_from_coffdrop(
            residues,
            params,
            name=name,
            k_spring=k_spring,
            caps=eff_caps,
            hydro_radius=hydro_radius,
            use_bond_tables=use_bond_tables,
        )
    if eff_caps != (None, None):
        raise ValueError("caps require sidechains=True; backbone builder rejects caps.")
    return build_chain_common_from_coffdrop(
        residues,
        params,
        name=name,
        k_spring=k_spring,
        hydro_radius=hydro_radius,
        use_bond_tables=use_bond_tables,
    )


def place_relaxed_geometry(chain: "ChainCommon") -> np.ndarray:
    """Build an initial clash-free 3D configuration for a chain.

    Backbone CAs form a non-collinear zig-zag (110 deg bond angle, alternating
    180 deg trans dihedral) with exact equilibrium bond lengths. Caps extend
    outward along the backbone ends, and sidechains branch outward in L-chirality
    orientations with exact bond lengths. Coordinates are centered at the origin.
    """
    n = chain.n_atoms
    positions = np.zeros((n, 3), dtype=float)
    if n == 0:
        return positions

    bond_len_map: Dict[Tuple[int, int], float] = {}
    bond_nbrs: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for bond in chain.bonds:
        ia = _chain_idx(bond.a)
        ib = _chain_idx(bond.b)
        bond_len_map[(ia, ib)] = float(bond.r0)
        bond_len_map[(ib, ia)] = float(bond.r0)
        bond_nbrs[ia].append((ib, float(bond.r0)))
        bond_nbrs[ib].append((ia, float(bond.r0)))

    ca_indices = [
        i
        for i, a in enumerate(chain.atoms)
        if a.resname.endswith(":CA") or a.resname == "CA"
    ]
    if not ca_indices:
        ca_indices = [0]

    # Zig-zag with 110 deg CA-CA-CA angle (step tilt +/-35 deg from +x in xy-plane).
    tilt = math.radians(35.0)
    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    ca_rank: Dict[int, int] = {ca_i: r for r, ca_i in enumerate(ca_indices)}
    positions[ca_indices[0]] = np.array([0.0, 0.0, 0.0])
    for r in range(len(ca_indices) - 1):
        u_step = np.array([cos_t, ((-1) ** r) * sin_t, 0.0])
        r0 = bond_len_map.get((ca_indices[r], ca_indices[r + 1]), 3.8)
        positions[ca_indices[r + 1]] = positions[ca_indices[r]] + r0 * u_step

    placed: Set[int] = set(ca_indices)
    parent_of: Dict[int, int] = {}

    # Extend ACE:CN backward from CA[0] and NME:CC forward from CA[-1].
    for i, atom in enumerate(chain.atoms):
        if i in placed:
            continue
        if atom.resname == "ACE:CN":
            anchor = ca_indices[0]
            r0 = bond_len_map.get((i, anchor), 3.81)
            u_back = np.array([-cos_t, sin_t, 0.0])
            positions[i] = positions[anchor] + r0 * u_back
            placed.add(i)
            parent_of[i] = anchor
        elif atom.resname == "NME:CC":
            anchor = ca_indices[-1]
            r0 = bond_len_map.get((i, anchor), 3.82)
            r_last = len(ca_indices) - 1
            u_fwd = np.array([cos_t, ((-1) ** r_last) * sin_t, 0.0])
            positions[i] = positions[anchor] + r0 * u_fwd
            placed.add(i)
            parent_of[i] = anchor

    # Breadth-first placement of sidechain and remaining bonded atoms.
    # First sidechain bead exits CA along L-chirality outward direction;
    # distal beads continue outward with a non-collinear bend.
    remaining = [i for i in range(n) if i not in placed]
    progress = True
    while remaining and progress:
        progress = False
        next_remaining = []
        for i in remaining:
            prev_i = None
            r0 = None
            for nbr, blen in bond_nbrs[i]:
                if nbr in placed:
                    prev_i = nbr
                    r0 = blen
                    break
            if prev_i is None:
                next_remaining.append(i)
                continue
            if prev_i in ca_rank:
                r = ca_rank[prev_i]
                sy = 1.0 if (r % 2 == 1) else -1.0
                # Matches L-amino-acid improper minimum near 150 deg and alternates
                # sides between adjacent residues to prevent clashes.
                u = np.array([0.0, sy * 0.90, sy * 0.43588989435406733])
                u /= np.linalg.norm(u)
            else:
                gp = parent_of.get(prev_i)
                if gp is not None:
                    base = positions[prev_i] - positions[gp]
                    bnorm = float(np.linalg.norm(base))
                    base = base / bnorm if bnorm > 1e-10 else np.array([0.0, 1.0, 0.0])
                else:
                    base = np.array([0.0, 1.0, 0.0])
                # Slight bend away from collinearity while continuing outward.
                bend_sign = 1.0 if (i % 2 == 0) else -1.0
                perp = np.cross(base, np.array([0.0, 0.0, 1.0]))
                pnorm = float(np.linalg.norm(perp))
                if pnorm < 1e-8:
                    perp = np.array([1.0, 0.0, 0.0])
                else:
                    perp = perp / pnorm
                u = 0.85 * base + (0.5267826876426369 * bend_sign) * perp
                u /= np.linalg.norm(u)
            positions[i] = positions[prev_i] + r0 * u
            placed.add(i)
            parent_of[i] = prev_i
            progress = True
        remaining = next_remaining

    # Deterministic clash-free fallback for any unbonded orphan atoms.
    for k, i in enumerate(remaining):
        positions[i] = positions[ca_indices[0]] + np.array([0.0, 4.0 + 3.0 * k, 4.0])

    positions = positions - positions.mean(axis=0)
    return positions


def chain_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    coffdrop_dir: Optional[str] = None,
    name: Optional[str] = None,
    sidechains: bool = True,
    k_spring: float = 100.0,
    allow_breaks: bool = True,
    hydro_radius: float = 0.0,
    use_bond_tables: bool = False,
) -> "ChainCommon":
    """Build a COFFDROP chain topology from ATOM records in a PDB file.

    Preserves PDB file order, distinguishes residues by
    (chain_id, seg_idx, resnum, icode) across TER records, strips terminal
    ACE/NME caps into COFFDROP cap terms, adds disulfide covalent bonds between
    paired CYS:SB beads from SSBOND records or CYX SG-SG contacts <= 2.5 A,
    normalizes Amber/CHARMM variants, and omits backbone bonds/angles/torsions
    across TER boundaries and (when allow_breaks is True) sequence or geometry
    breaks.
    """
    from pathlib import Path as _Path

    pdb_file = _Path(pdb_path)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    seen_residues: List[Tuple[str, int, int, str]] = []
    residue_map: Dict[Tuple[str, int, int, str], str] = {}
    ca_coords: Dict[Tuple[str, int, int, str], np.ndarray] = {}
    cb_coords: Dict[Tuple[str, int, int, str], np.ndarray] = {}
    sg_coords: Dict[Tuple[str, int, int, str], np.ndarray] = {}
    ssbond_records: List[Tuple[str, int, str, str, int, str]] = []
    chains_seen: set = set()
    seg_idx = 0

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("TER") and (len(line) == 3 or line[3].isspace()):
                seg_idx += 1
                continue
            if line.startswith("SSBOND"):
                parsed_ss = None
                if len(line) >= 35:
                    try:
                        c1 = line[15:16].strip() or "A"
                        r1 = int(line[17:21].strip())
                        i1 = line[21:22].strip() if len(line) > 21 else ""
                        c2 = line[29:30].strip() or "A"
                        r2 = int(line[31:35].strip())
                        i2 = line[35:36].strip() if len(line) > 35 else ""
                        parsed_ss = (c1, r1, i1, c2, r2, i2)
                    except ValueError:
                        parsed_ss = None
                if parsed_ss is None:
                    toks = line.split()
                    if len(toks) >= 8:
                        try:
                            parsed_ss = (
                                toks[3],
                                int(toks[4]),
                                "",
                                toks[6],
                                int(toks[7]),
                                "",
                            )
                        except ValueError:
                            parsed_ss = None
                if parsed_ss is not None:
                    ssbond_records.append(parsed_ss)
                continue
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            atom_name = line[12:16].strip().upper()
            resname_raw = line[17:21].strip().upper()
            if atom_name in GHOST_NAMES or resname_raw in _PDB_SKIP_RESNAMES:
                continue
            ch = line[21:22].strip() or "A"
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            icode = line[26:27].strip() if len(line) > 26 else ""
            chains_seen.add(ch)
            key = (ch, seg_idx, resnum, icode)
            if key not in residue_map:
                residue_map[key] = resname_raw
                seen_residues.append(key)
            if len(line) >= 54 and atom_name in ("CA", "CB", "SG"):
                try:
                    coord = np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    )
                except ValueError:
                    coord = None
                if coord is not None:
                    if atom_name == "CA" and key not in ca_coords:
                        ca_coords[key] = coord
                    elif atom_name == "CB" and key not in cb_coords:
                        cb_coords[key] = coord
                    elif atom_name == "SG" and key not in sg_coords:
                        sg_coords[key] = coord

    if not seen_residues:
        raise ValueError(f"No ATOM records found in {pdb_path}")

    if chain_id is None:
        if len(chains_seen) > 1:
            raise ValueError(
                f"PDB has multiple chains {sorted(chains_seen)}; "
                "specify chain_id=... to select one."
            )
        chain_id = next(iter(chains_seen))

    if chain_id not in chains_seen:
        raise ValueError(
            f"chain_id='{chain_id}' not in PDB chains {sorted(chains_seen)}"
        )

    chain_keys = [
        (seg, rn, ic) for (ch, seg, rn, ic) in seen_residues if ch == chain_id
    ]
    if not chain_keys:
        raise ValueError(f"chain_id='{chain_id}' has no residues")

    # Group contiguous entries by TER segment in PDB file order and strip
    # terminal ACE/NME caps per segment so they never emit invalid ACE:CA/NME:CA.
    segments: List[List[Tuple[int, int, str]]] = []
    for k_item in chain_keys:
        if not segments or segments[-1][0][0] != k_item[0]:
            segments.append([k_item])
        else:
            segments[-1].append(k_item)

    core_keys: List[Tuple[int, int, str]] = []
    has_ace = False
    has_nme = False
    for s_idx, seg_items in enumerate(segments):
        start_i = 0
        end_i = len(seg_items)
        if start_i < end_i and residue_map[(chain_id, *seg_items[start_i])] == "ACE":
            if s_idx == 0:
                has_ace = True
            start_i += 1
        if start_i < end_i and residue_map[(chain_id, *seg_items[end_i - 1])] == "NME":
            if s_idx == len(segments) - 1:
                has_nme = True
            end_i -= 1
        core_keys.extend(seg_items[start_i:end_i])

    if not core_keys:
        raise ValueError(
            f"chain_id='{chain_id}' has no polymer residues after stripping caps"
        )

    core_full_keys = [(chain_id, seg, rn, ic) for seg, rn, ic in core_keys]
    residues = [residue_map[fk] for fk in core_full_keys]

    breaks: Set[int] = set()
    for r in range(len(core_keys) - 1):
        seg0, rn0, ic0 = core_keys[r]
        seg1, rn1, ic1 = core_keys[r + 1]
        if seg0 != seg1:
            breaks.add(r)
            continue
        if not allow_breaks:
            continue
        if rn1 - rn0 > 1 or rn1 < rn0:
            breaks.add(r)
            continue
        k0 = core_full_keys[r]
        k1 = core_full_keys[r + 1]
        if k0 in ca_coords and k1 in ca_coords:
            if float(np.linalg.norm(ca_coords[k1] - ca_coords[k0])) > 4.3:
                breaks.add(r)

    params = _load_coffdrop_params_from_dir(coffdrop_dir)
    if not sidechains:
        return build_chain_common_from_coffdrop(
            residues,
            params,
            name=name,
            k_spring=k_spring,
            chain_breaks=breaks,
            hydro_radius=hydro_radius,
            use_bond_tables=use_bond_tables,
        )

    caps_arg = ("ACE" if has_ace else None, "NME" if has_nme else None)
    common = build_chain_common_with_sidechains_from_coffdrop(
        residues,
        params,
        name=name,
        k_spring=k_spring,
        caps=caps_arg,
        chain_breaks=breaks,
        hydro_radius=hydro_radius,
        use_bond_tables=use_bond_tables,
    )

    # Disulfide covalent bonds between paired CYS:SB beads (SSBOND and/or CYX SG-SG <= 2.5 A).
    sb_bead_of_res: Dict[int, int] = {}
    for b_idx, atom in enumerate(common.atoms):
        if atom.resid >= 0 and atom.resname == "CYS:SB":
            sb_bead_of_res[int(atom.resid)] = b_idx

    if sb_bead_of_res:
        disulfide_res_pairs: Set[Tuple[int, int]] = set()
        rn_ic_to_indices: Dict[Tuple[int, str], List[int]] = {}
        for idx_r, (_seg_r, rn_r, ic_r) in enumerate(core_keys):
            rn_ic_to_indices.setdefault((rn_r, ic_r), []).append(idx_r)

        for c1, r1, i1, c2, r2, i2 in ssbond_records:
            if c1 != chain_id or c2 != chain_id:
                continue
            idxs1 = rn_ic_to_indices.get((r1, i1), [])
            idxs2 = rn_ic_to_indices.get((r2, i2), [])
            if not idxs1 or not idxs2:
                continue
            if (r1, i1) == (r2, i2):
                if len(idxs1) < 2:
                    continue
                ra, rb = idxs1[0], idxs1[1]
            else:
                ra, rb = idxs1[0], idxs2[0]
            if ra != rb and ra in sb_bead_of_res and rb in sb_bead_of_res:
                disulfide_res_pairs.add((ra, rb) if ra < rb else (rb, ra))

        cyx_indices = [
            idx_r
            for idx_r, fk in enumerate(core_full_keys)
            if residue_map[fk] == "CYX" and fk in sg_coords and idx_r in sb_bead_of_res
        ]
        for p in range(len(cyx_indices)):
            i1 = cyx_indices[p]
            sg1 = sg_coords[core_full_keys[i1]]
            for q in range(p + 1, len(cyx_indices)):
                i2 = cyx_indices[q]
                sg2 = sg_coords[core_full_keys[i2]]
                if float(np.linalg.norm(sg2 - sg1)) <= 2.5:
                    disulfide_res_pairs.add((i1, i2) if i1 < i2 else (i2, i1))

        if disulfide_res_pairs:
            existing_bonds = {
                (
                    min(_chain_idx(b.a), _chain_idx(b.b)),
                    max(_chain_idx(b.a), _chain_idx(b.b)),
                )
                for b in common.bonds
            }
            added_ss = False
            for r1, r2 in sorted(disulfide_res_pairs):
                sb1 = sb_bead_of_res[r1]
                sb2 = sb_bead_of_res[r2]
                pair_b = (sb1, sb2) if sb1 < sb2 else (sb2, sb1)
                if pair_b in existing_bonds:
                    continue
                fk1 = core_full_keys[r1]
                fk2 = core_full_keys[r2]
                pts1 = [
                    c for c in (cb_coords.get(fk1), sg_coords.get(fk1)) if c is not None
                ]
                pts2 = [
                    c for c in (cb_coords.get(fk2), sg_coords.get(fk2)) if c is not None
                ]
                if pts1 and pts2:
                    d_sb = float(
                        np.linalg.norm(np.mean(pts2, axis=0) - np.mean(pts1, axis=0))
                    )
                    r0_ss = d_sb if d_sb > 0.5 else 2.05
                else:
                    r0_ss = 2.05
                common.bonds.append(
                    ChainBond(
                        a=ChainAtomRef(sb1),
                        b=ChainAtomRef(sb2),
                        r0=r0_ss,
                        k_spring=k_spring,
                        type_idx=-1,
                    )
                )
                existing_bonds.add(pair_b)
                added_ss = True

            if added_ss and common.pair_lookups:
                excl_ss = coffdrop_nonbonded_exclusions(common)
                for pkey in list(common.pair_lookups.keys()):
                    lo, hi = (
                        (pkey[0], pkey[1]) if pkey[0] < pkey[1] else (pkey[1], pkey[0])
                    )
                    if (lo, hi) in excl_ss:
                        pidx = common.pair_lookups[pkey]
                        if (
                            common.coffdrop_params is None
                            or common.coffdrop_params.pair_pots[pidx].orders == (0, 0)
                        ):
                            del common.pair_lookups[pkey]

    return common
