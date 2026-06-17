"""
The Rotne-Prager-Yamakawa hydrodynamic interaction tensor.

When two spheres diffuse in a viscous fluid, each sphere's motion creates
a flow field that affects the other. This hydrodynamic interaction (HI)
modifies the effective diffusion coefficient.

For two spheres of radii a₁ and a₂ at separation r along the line of
centres, the diffusion coefficient parallel to that line is

    D_∥(r) = kBT/(6πη) × [1/a₁ + 1/a₂ - 3/r + 2ā²/r³]

where ā² = (a₁² + a₂²)/2. Physically, the term 1/a₁ + 1/a₂ is the free
diffusion of two uncoupled spheres, the term -3/r is the leading
hydrodynamic correction that slows the approach, and the term +2ā²/r³ is
the finite-size correction.

At large r the diffusion coefficient approaches the free value
D₀ = kBT/(6πη) × (1/a₁ + 1/a₂). At contact, r = a₁+a₂, it is smaller
than D₀ because the hydrodynamic interaction slows the approach by
roughly 20 to 40 percent.

This tensor follows Zuk et al., J. Fluid Mech. 741, R5 (2014), which
extends the original Rotne-Prager (1969) and Yamakawa (1970) results to
handle overlapping spheres correctly.

The hydrodynamic interaction reduces k_b for typical protein systems
because the effective diffusion near contact is slower than free
diffusion.
"""

from __future__ import annotations
from pystarc.global_defs.constants import ETA_WATER, KB_SI, T_DEFAULT, ANG_TO_M, PS_TO_S
import numpy as np
import math
import warnings


def stokes_translational_diffusion(
    radius_ang: float, eta: float = ETA_WATER, T: float = T_DEFAULT
) -> float:
    """
    Stokes-Einstein translational diffusion coefficient,

        D_t = kBT / (6πηr).

    Here r is the sphere radius in angstrom, η is the solvent viscosity,
    and T is the temperature. Returns D_t in Å²/ps.
    """
    r_m = radius_ang * ANG_TO_M
    D_m2s = KB_SI * T / (6.0 * math.pi * eta * r_m)
    return D_m2s / (ANG_TO_M**2) * PS_TO_S


def stokes_rotational_diffusion(
    radius_ang: float, eta: float = ETA_WATER, T: float = T_DEFAULT
) -> float:
    """
    Stokes rotational diffusion coefficient,

        D_r = kBT / (8πηr³).

    Here r is the sphere radius in angstrom, η is the solvent viscosity,
    and T is the temperature. Returns D_r in rad²/ps.
    """
    r_m = radius_ang * ANG_TO_M
    D_r_s = KB_SI * T / (8.0 * math.pi * eta * r_m**3)
    return D_r_s * PS_TO_S


def rpy_offdiagonal(
    r_vec: np.ndarray, a: float, b: float, D_a: float, D_b: float
) -> np.ndarray:
    """
    The off-diagonal translational mobility tensor M_12 of the
    Rotne-Prager-Yamakawa model.

    The tensor is assembled as M_12 = tt_I·I + tt_uu·r̂r̂ᵀ, where I is the
    identity and r̂ is the unit vector along the line of centres. The two
    scalar coefficients depend on the regime.

    In the far field, r > a+b, they are
    tt_I = (1 + (a²+b²)/3r²) / (8πr) and tt_uu = (1 - (a²+b²)/r²) / (8πr).
    In the partial-overlap regime, |a-b| < r <= a+b (Zuk et al. 2014,
    Eq. 11), they are tt_I = (16r³(a+b) - ((a-b)² + 3r²)²) / (192π·a·b·r³)
    and tt_uu = 3·((a-b)² - r²)² / (192π·a·b·r³). When one sphere lies
    inside the other, r <= |a-b|, the tensor reduces to the self-mobility
    of the larger sphere, tt_I = 1/(6π·max(a,b)) and tt_uu = 0.

    All terms are viscosity-scaled. PySTARC works in kBT units where
    D = kBT·mobility, so mobility = D/kBT. Returns the tensor in Å²/ps,
    consistent with the self-diffusion coefficients.
    """
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
        return np.zeros((3, 3))
    rhat = r_vec / r
    outer = np.outer(rhat, rhat)
    I3 = np.eye(3)
    PI = math.pi
    PI6 = 6.0 * PI
    PI8 = 8.0 * PI
    a2 = a * a
    b2 = b * b
    r2 = r * r
    r3 = r2 * r
    if r > a + b:
        # Far field, the two spheres do not overlap.
        den = PI8 * r
        a2ob2 = a2 + b2
        tt_I = (1.0 + a2ob2 / (3.0 * r2)) / den
        tt_uu = (1.0 - a2ob2 / r2) / den
    elif r > abs(a - b):
        # The spheres partially overlap.
        ab = a * b
        am2 = (a - b) ** 2  # the squared radius difference (a-b)²
        den = 6.0 * 32.0 * PI * ab * r3
        tt_I = (16.0 * r3 * (a + b) - (am2 + 3.0 * r2) ** 2) / den
        tt_uu = 3.0 * (am2 - r2) ** 2 / den
    else:
        # One sphere lies inside the other, so the result is the
        # self-mobility of the larger sphere.
        a_max = max(a, b)
        tt_I = 1.0 / (PI6 * a_max)
        tt_uu = 0.0
    # Convert from viscosity units to diffusion units. In Brownian
    # dynamics D = kBT·mobility, and the self-diffusion is
    # D_a = kBT/(6π·η·a), so kBT/η = D_a·6π·a. We take the geometric
    # mean of the two molecules.
    kT_over_eta = math.sqrt(D_a * PI6 * a * D_b * PI6 * b)
    M = kT_over_eta * (tt_I * I3 + tt_uu * outer)
    return M


class MobilityTensor:
    """
    The full RPY mobility tensor for a two-molecule Brownian-dynamics
    system.

    The tensor stores both the diagonal self terms and the off-diagonal
    hydrodynamic coupling terms. The effective relative diffusion
    D_rel_eff(r) depends on the current separation r between the
    molecules. When hydrodynamic interactions are enabled, the BD step
    uses D_rel_eff(r) instead of the constant D_t1 + D_t2. This slows the
    relative diffusion when the molecules are close, which is physically
    correct because nearby molecules drag solvent into the gap between
    them.

    Parameters
    ----------
    r1, r2  : hydrodynamic radii [Å]
    D_t1, D_r1 : translational and rotational diffusion for molecule 1
        [Å²/ps, rad²/ps]
    D_t2, D_r2 : translational and rotational diffusion for molecule 2
    use_rpy    : if True (the default) use the full RPY coupling, and if
        False use the diagonal approximation, which is faster but less
        accurate
    """

    def __init__(
        self,
        D_trans1: float,
        D_rot1: float,
        D_trans2: float,
        D_rot2: float,
        radius1: float = 0.0,
        radius2: float = 0.0,
        use_rpy: bool = True,
    ):
        self.D_trans1 = D_trans1
        self.D_rot1 = D_rot1
        self.D_trans2 = D_trans2
        self.D_rot2 = D_rot2
        self.radius1 = radius1
        self.radius2 = radius2
        self.use_rpy = use_rpy

    @classmethod
    def from_radii(
        cls,
        radius1: float,
        radius2: float,
        eta: float = ETA_WATER,
        T: float = T_DEFAULT,
        use_rpy: bool = True,
    ) -> "MobilityTensor":
        """Build a MobilityTensor from the two hydrodynamic radii."""
        return cls(
            D_trans1=stokes_translational_diffusion(radius1, eta, T),
            D_rot1=stokes_rotational_diffusion(radius1, eta, T),
            D_trans2=stokes_translational_diffusion(radius2, eta, T),
            D_rot2=stokes_rotational_diffusion(radius2, eta, T),
            radius1=radius1,
            radius2=radius2,
            use_rpy=use_rpy,
        )

    def relative_translational_diffusion(self, r_vec: np.ndarray = None) -> float:
        """
        The effective relative translational diffusion at separation
        r_vec.

        Without RPY coupling the diagonal approximation gives simply
        D_rel = D_t1 + D_t2. With RPY coupling, which is the default, the
        result is

            D_rel_eff(r) = D_t1 + D_t2 - (2/3)·tr(M_12(r)),

        where M_12(r) is the off-diagonal mobility tensor. The RPY
        correction reduces D_rel near contact (r ≈ a+b) and vanishes at
        large r, recovering the diagonal result.
        """
        D0 = self.D_trans1 + self.D_trans2
        if not self.use_rpy or r_vec is None:
            return D0
        if self.radius1 <= 0.0 or self.radius2 <= 0.0:
            return D0
        M12 = rpy_offdiagonal(
            r_vec, self.radius1, self.radius2, self.D_trans1, self.D_trans2
        )
        # The scalar coupling that reduces the relative diffusion,
        # D_rel = D_t1 + D_t2 - (2/3)·tr(M_12). The factor of two appears
        # because both molecules feel the coupling symmetrically.
        D_coupling = (2.0 / 3.0) * np.trace(M12)
        return max(D0 - D_coupling, 1e-12)  # clamp so the result is never negative

    def relative_rotational_diffusion(self) -> float:
        """
        The relative rotational diffusion, D_r_rel = D_r1 + D_r2. The
        RPY rotational coupling is negligible and is ignored here.
        """
        return self.D_rot1 + self.D_rot2

    def __repr__(self) -> str:
        return (
            f"MobilityTensor(Dt1={self.D_trans1:.3e}, Dr1={self.D_rot1:.3e}, "
            f"Dt2={self.D_trans2:.3e}, Dr2={self.D_rot2:.3e}, "
            f"r1={self.radius1:.1f}A, r2={self.radius2:.1f}A, RPY={self.use_rpy})"
        )


# The full RPY pair tensor, including translation, rotation, and the
# cross-coupling between them. The returned coefficients carry units of
# (length) to the power -k, where k is 1 for the translation block, 2 for
# the translation-rotation blocks, and 3 for the rotation block. Multiply
# by 1/η to recover the physical mobility, and by kT to recover the
# diffusion coefficients.


def rpy_full_components(ai, aj, r):
    """Scalar coefficients of the RPY pair mobility tensor.

    Returns the six scalars (tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps)
    that combine with the identity, outer-product, and Levi-Civita-on-u
    matrices to assemble the four 3x3 blocks of the pair tensor. See
    rpy_pair_blocks() for the assembly.

    There are three regimes set by the separation r. For r > ai + aj the
    spheres do not overlap and follow the Rotne-Prager-Yamakawa result.
    For |ai-aj| < r <= ai+aj the spheres partially overlap and follow
    Zuk et al. 2014. For r <= |ai-aj| one sphere lies fully inside the
    other.
    """
    pi = math.pi
    pi6 = 6.0 * pi
    pi8 = 8.0 * pi
    ai2 = ai * ai
    aj2 = aj * aj
    r2 = r * r
    r3 = r2 * r

    if r > ai + aj:
        # Far field, the spheres do not overlap.
        den_tt = pi8 * r
        a2or2 = (ai2 + aj2) / r2
        tt_I = (1.0 + a2or2 / 3.0) / den_tt
        tt_uu = (1.0 - a2or2) / den_tt

        den_rr = 16.0 * pi * r3
        rr_I = -1.0 / den_rr
        rr_uu = 3.0 / den_rr

        rt_eps = 1.0 / (pi8 * r2)
        tr_eps = rt_eps
        return tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps

    if r > abs(ai - aj):
        # The spheres partially overlap.
        aij = ai * aj
        am2 = (ai - aj) ** 2
        den_tt = 6.0 * 32.0 * pi * aij * r3
        tt_I = (16.0 * r3 * (ai + aj) - (am2 + 3.0 * r2) ** 2) / den_tt
        tt_uu = 3.0 * (am2 - r2) ** 2 / den_tt

        r4 = r * r3
        r6 = r2 * r4
        ai3 = ai2 * ai
        aj3 = aj2 * aj
        a4a = ai2 + 4.0 * ai * aj + aj2
        A = (
            5.0 * r6
            - 27.0 * r4 * (ai2 + aj2)
            + 32.0 * r3 * (ai3 + aj3)
            - 9.0 * r2 * (ai2 - aj2) ** 2
            - (ai - aj) ** 4 * a4a
        ) / (64.0 * r3)
        B = (3.0 * ((ai - aj) ** 2 - r2) ** 2 * (a4a - r2)) / (64.0 * r3)
        den_rr = 8.0 * pi * ai3 * aj3
        rr_I = A / den_rr
        rr_uu = B / den_rr

        # The translation-rotation cross-coupling uses an asymmetric
        # helper f(ai, aj), following the BD2 implementation.
        def _rt_term(a_force, a_torque):
            af2 = a_force * a_force
            num = (a_force - a_torque + r) ** 2 * (
                a_torque**2 + 2.0 * a_torque * (a_force + r) - 3.0 * (a_force - r) ** 2
            )
            return num / (8.0 * 16.0 * pi * af2 * a_force * a_torque * r2)

        rt_eps = _rt_term(ai, aj)
        tr_eps = _rt_term(aj, ai)
        return tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps

    # One sphere is strictly inside the other.
    a = max(ai, aj)
    tt_I = 1.0 / (pi6 * a)
    tt_uu = 0.0
    rr_I = 1.0 / (8.0 * pi * a * a * a)
    rr_uu = 0.0
    # The BD2 implementation leaves the translation-rotation coupling at
    # zero in this regime, and we follow the same choice.
    rt_eps = 0.0
    tr_eps = 0.0
    return tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps


def rpy_pair_blocks(ai, aj, r_ij):
    """Full 3x3 block decomposition of the RPY pair mobility tensor.

    Returns (mtt, mrt, mtr, mrr), each a 3x3 numpy array, such that the
    linear response of bead i's velocity and angular velocity to a force
    and torque applied at bead j is

        [v_i ]   [ mtt  mtr ] [ F_j ]
        [w_i ] = [ mrt  mrr ] [ T_j ].

    By convention the viscosity is not applied. Multiply the blocks by
    1/η to recover the physical mobility.
    """
    r = float(np.linalg.norm(r_ij))
    if r < 1e-12:
        # The two beads coincide, which is degenerate. The caller should
        # use rpy_self_blocks for this case.
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))

    u = np.asarray(r_ij, dtype=float) / r
    uu = np.outer(u, u)
    I3 = np.eye(3)

    # The Levi-Civita symbol contracted with u, eps_ijk·u_k. This is the
    # signed cross-product matrix such that eps_u @ v = u × v.
    eps_u = np.array(
        [
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ]
    )

    tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps = rpy_full_components(ai, aj, r)

    mtt = tt_I * I3 + tt_uu * uu
    mrr = rr_I * I3 + rr_uu * uu
    mrt = rt_eps * eps_u
    mtr = tr_eps * eps_u
    return mtt, mrt, mtr, mrr


def rpy_self_blocks(a):
    """Single-bead self-mobility, the diagonal i = j entry.

    Returns (mtt_self, mrr_self), each a 3x3 numpy array. The
    cross-coupling self-blocks are zero by symmetry for an isolated
    sphere.

    By convention the viscosity is not applied. The blocks are
    mtt_self = I / (6πa) and mrr_self = I / (8πa³).
    """
    pi = math.pi
    I3 = np.eye(3)
    mtt_self = I3 / (6.0 * pi * a)
    mrr_self = I3 / (8.0 * pi * a * a * a)
    return mtt_self, mrr_self


def rpy_full_mobility_matrix(positions, radii):
    """Assemble the full 6N x 6N RPY mobility matrix for N spheres.

    The mobility matrix M relates the generalized forces (force and
    torque) on each bead to the generalized velocities (linear and
    angular) of each bead, hydrodynamically coupled through the
    surrounding fluid,

        v_i = sum_j (mtt_ij F_j + mtr_ij T_j)
        w_i = sum_j (mrt_ij F_j + mrr_ij T_j).

    In the block layout bead i occupies degrees of freedom [6i, 6i+6),
    where the first three are translation and the last three are
    rotation. The 6x6 block at (i, j) decomposes as
        M[6i:6i+3, 6j:6j+3] = mtt(i,j)
        M[6i:6i+3, 6j+3:6j+6] = mtr(i,j)
        M[6i+3:6i+6, 6j:6j+3] = mrt(i,j)
        M[6i+3:6i+6, 6j+3:6j+6] = mrr(i,j).
    For i = j these come from rpy_self_blocks, where the cross-coupling
    is zero for an isolated sphere, and for i != j they come from
    rpy_pair_blocks.

    By convention the viscosity is not applied. Multiply M by 1/η for
    the physical mobility, or by kT/η for diffusion units.

    Parameters
    ----------
    positions : (N, 3) array of bead centers, in any consistent frame.
    radii     : (N,) array of bead hydrodynamic radii.

    Returns
    -------
    M : (6N, 6N) numpy array, symmetric to numerical precision by
        Onsager reciprocity.
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}")
    if radii.ndim != 1 or radii.shape[0] != positions.shape[0]:
        raise ValueError(
            f"radii must have shape (N,) matching positions[0]; "
            f"got radii.shape={radii.shape}, positions.shape={positions.shape}"
        )

    n = positions.shape[0]
    M = np.zeros((6 * n, 6 * n), dtype=float)

    for i in range(n):
        # The diagonal block is the self-mobility, with no cross-coupling.
        mtt_self, mrr_self = rpy_self_blocks(radii[i])
        M[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = mtt_self
        M[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = mrr_self
        # The mtr and mrt blocks are zero on the diagonal because an
        # isolated sphere has no self cross-coupling, so we leave them at
        # zero.

        for j in range(i + 1, n):
            # The off-diagonal block (i, j) and its transpose pair (j, i).
            r_ij = positions[j] - positions[i]
            mtt_ij, mrt_ij, mtr_ij, mrr_ij = rpy_pair_blocks(
                radii[i],
                radii[j],
                r_ij,
            )
            # The (i, j) block.
            M[6 * i : 6 * i + 3, 6 * j : 6 * j + 3] = mtt_ij
            M[6 * i : 6 * i + 3, 6 * j + 3 : 6 * j + 6] = mtr_ij
            M[6 * i + 3 : 6 * i + 6, 6 * j : 6 * j + 3] = mrt_ij
            M[6 * i + 3 : 6 * i + 6, 6 * j + 3 : 6 * j + 6] = mrr_ij
            # The (j, i) block equals the transpose of the (i, j) block by
            # Onsager reciprocity. We fill it by transposing rather than
            # re-evaluating, which guarantees exact numerical symmetry.
            M[6 * j : 6 * j + 3, 6 * i : 6 * i + 3] = mtt_ij.T
            M[6 * j : 6 * j + 3, 6 * i + 3 : 6 * i + 6] = mrt_ij.T
            M[6 * j + 3 : 6 * j + 6, 6 * i : 6 * i + 3] = mtr_ij.T
            M[6 * j + 3 : 6 * j + 6, 6 * i + 3 : 6 * i + 6] = mrr_ij.T

    return M


def _hydrodynamic_center(positions, radii):
    """Radius-weighted centroid, hc = (sum a_i·r_i) / (sum a_i).

    This is the natural reference point for a chain of beads with
    different radii. All bead radii enter linearly, so larger beads pull
    the center toward themselves.
    """
    radii = np.asarray(radii, dtype=float)
    positions = np.asarray(positions, dtype=float)
    a_sum = float(np.sum(radii))
    return (radii[:, None] * positions).sum(axis=0) / a_sum


def _translation_only_mobility(positions, radii):
    """Extract the 3N x 3N translation-translation block of the full RPY
    mobility matrix. Each 3x3 block (i, j) is mtt(a_i, a_j, r_ij).
    """
    M_full = rpy_full_mobility_matrix(positions, radii)
    n = len(positions)
    M_tt = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            M_tt[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = M_full[
                6 * i : 6 * i + 3, 6 * j : 6 * j + 3
            ]
    return M_tt



def _build_robust_solver(M):
    """Build a callable solver(v) for M @ x = v that is robust to a
    near-singular M.

    The function tries three strategies in order. First it attempts a
    plain Cholesky factorization of M, which is cheap but requires M to
    be symmetric positive definite. If that fails it tries a Cholesky
    factorization of M + eps·I with progressively larger jitter, which
    handles a near-singular M. As a last resort it uses a symmetric
    eigendecomposition with eigenvalue clipping, which handles an
    indefinite M.

    Returns
    -------
    (solver, was_regularized, info)
        solver : callable(v) returning an approximate M^{-1}·v
        was_regularized : bool, True if any fallback strategy was used
        info : str describing the strategy actually applied
    """
    n = M.shape[0]
    trace_avg = float(np.trace(M)) / max(n, 1)

    # First strategy, a plain Cholesky factorization.
    try:
        L = np.linalg.cholesky(M)
        def solver_chol(v, L=L):
            y = np.linalg.solve(L, v)
            return np.linalg.solve(L.T, y)
        return solver_chol, False, "cholesky"
    except np.linalg.LinAlgError:
        pass

    # Second strategy, a regularized Cholesky factorization with
    # progressively larger jitter on the diagonal.
    for eps_factor in (1e-12, 1e-9, 1e-6):
        eps = eps_factor * max(abs(trace_avg), 1.0)
        try:
            L_jit = np.linalg.cholesky(M + eps * np.eye(n))
            def solver_jit(v, L=L_jit):
                y = np.linalg.solve(L, v)
                return np.linalg.solve(L.T, y)
            return solver_jit, True, f"cholesky+jitter(eps={eps:.3e})"
        except np.linalg.LinAlgError:
            continue

    # Third strategy, a symmetric eigendecomposition with eigenvalue
    # clipping.
    eigvals, eigvecs = np.linalg.eigh(M)
    floor = max(float(abs(eigvals).max()), 1.0) * 1e-10
    eigvals_clipped = np.maximum(eigvals, floor)
    eigvals_inv = 1.0 / eigvals_clipped
    def solver_eig(v, Q=eigvecs, di=eigvals_inv):
        return Q @ (di * (Q.T @ v))
    info = f"eigendecomp(min_eig={float(eigvals.min()):.3e}, clipped_to={floor:.3e})"
    return solver_eig, True, info


def chain_rigid_body_resistance(positions, radii):
    """Compute the chain's rigid-body resistance matrices and
    hydrodynamic center, given the bead positions and radii.

    Returns (A, C, hc). The matrix A is the (3, 3) translational
    resistance, the force per unit velocity, acting as
    F_net = η·A @ v_chain, where v_chain is the uniform translational
    velocity of the rigid chain. The matrix C is the (3, 3) rotational
    resistance, the torque per unit angular velocity taken about the
    hydrodynamic center, acting as T_net = η·C @ omega_chain. The vector
    hc is the (3,) hydrodynamic center, the radius-weighted centroid.

    By convention the viscosity is not included. Multiply A by η for the
    physical resistance. The corresponding rigid-body translational
    diffusion is D_trans = kT·inv(η·A), and similarly for D_rot.

    The algorithm works as follows. For each Cartesian direction k it
    prescribes the rigid-body velocity field, solves M_tt·F = v for the
    per-bead forces required to maintain that motion against the fluid,
    and reads off the column of A or C from the appropriate sum or moment
    of those forces. This avoids ever forming M_tt^{-1} explicitly.

    Parameters
    ----------
    positions : (N, 3) bead centers.
    radii     : (N,)   bead hydrodynamic radii.

    Notes
    -----
    The translation-rotation cross-coupling block B is not returned. For
    chains that are well-centered at the hydrodynamic center, B is small
    or zero by symmetry. A full Stokesian-dynamics treatment would
    compute it.
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}")
    if radii.ndim != 1 or radii.shape[0] != positions.shape[0]:
        raise ValueError(
            f"radii must have shape (N,); got radii.shape={radii.shape}, "
            f"positions.shape={positions.shape}"
        )

    n = positions.shape[0]
    hc = _hydrodynamic_center(positions, radii)
    M_tt = _translation_only_mobility(positions, radii)

    # Solve M_tt @ F = v. The RPY mobility is symmetric positive definite
    # for non-overlapping beads, but coarse-grained chains derived from
    # PDB files can have near-coincident beads (for example the
    # CA fallback for disordered sidechains), which makes M_tt
    # ill-conditioned or indefinite. _build_robust_solver falls back
    # from a regularized Cholesky factorization to a symmetric
    # eigendecomposition rather than raising a LinAlgError on the user.
    solve, _was_regularized, _solver_info = _build_robust_solver(M_tt)
    if _was_regularized:
        warnings.warn(
            f"chain_rigid_body_resistance: RPY mobility matrix M_tt required "
            f"regularization ({_solver_info}); typically caused by "
            f"near-coincident beads. Check chain bead distances against "
            f"bead radii.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Build the A matrix by prescribing a uniform translation v_i = e_k
    # for every bead.
    A = np.zeros((3, 3))
    for k in range(3):
        v = np.zeros(3 * n)
        for i in range(n):
            v[3 * i + k] = 1.0
        F = solve(v)
        F_per_bead = F.reshape(n, 3)
        A[:, k] = F_per_bead.sum(axis=0)

    # Build the C matrix by prescribing a rigid-body rotation
    # omega = e_k about hc, so bead i moves with velocity
    # v_i = omega × (r_i - hc).
    C = np.zeros((3, 3))
    for k in range(3):
        omega = np.zeros(3)
        omega[k] = 1.0
        v = np.zeros(3 * n)
        for i in range(n):
            d = positions[i] - hc
            v[3 * i : 3 * i + 3] = np.cross(omega, d)
        F = solve(v)
        F_per_bead = F.reshape(n, 3)
        T_total = np.zeros(3)
        for i in range(n):
            d = positions[i] - hc
            T_total += np.cross(d, F_per_bead[i])
        C[:, k] = T_total

    return A, C, hc


def chain_diffusion_tensors(positions, radii, kT=1.0, viscosity=None):
    """Compute the chain's rigid-body diffusion tensors and hydrodynamic
    center, ready for use in a Brownian-dynamics integrator.

    Returns (D_trans, D_rot, hc). The matrix D_trans is the (3, 3)
    translational diffusion tensor in length²/time, D_rot is the (3, 3)
    rotational diffusion tensor in 1/time, and hc is the (3,)
    hydrodynamic center in length.

    The diffusion tensors are obtained by inverting the resistance
    matrices A and C from chain_rigid_body_resistance and scaling by
    kT/η,

        D_trans = (kT/η)·inv(A)
        D_rot   = (kT/η)·inv(C).

    Parameters
    ----------
    positions : (N, 3) bead centers.
    radii     : (N,)   bead hydrodynamic radii.
    kT        : thermal energy. The PySTARC convention is kT = 1.
    viscosity : solvent viscosity. If None, the package default
                WATER_VISCOSITY = 0.243 kBT·ps/Å³ from motion/do_bd_step
                is used.

    Notes
    -----
    For a chain of N >= 2 beads, both A and C are invertible by
    construction. For N = 1 the matrix A is invertible but C is the zero
    matrix (see the chain_rigid_body_resistance docstring), so a
    single-bead chain cannot use the algorithm without bead-rotation
    contributions. Callers should special-case N = 1 by using the bead's
    own rotational mobility 1/(8πηa³).
    """
    from pystarc.motion.do_bd_step import WATER_VISCOSITY

    if viscosity is None:
        viscosity = WATER_VISCOSITY

    A, C, hc = chain_rigid_body_resistance(positions, radii)

    n = len(positions)
    if n < 1:
        raise ValueError("at least one bead required")

    scale = kT / viscosity
    D_trans = scale * np.linalg.inv(A)

    if n == 1:
        # C is zero by construction in the BD2 algorithm, so for a single
        # bead we fall back to the Stokes rotational mobility, which is
        # correct.
        a = float(radii[0])
        D_rot = (kT / viscosity) * np.eye(3) / (8.0 * math.pi * a**3)
    else:
        # A singular C arises for special chain geometries. For example,
        # a perfectly collinear chain has zero rotational drag about its
        # own axis because all moment arms vanish. In that case the chain
        # is not a well-defined three-degree-of-freedom rigid rotor, so we
        # raise a clear error rather than a generic numpy LinAlgError.
        try:
            D_rot = scale * np.linalg.inv(C)
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(C)
            raise np.linalg.LinAlgError(
                "rigid-body rotational resistance C is singular for this "
                f"chain geometry (eigenvalues {eigvals}). This typically "
                "means the chain lies along a single axis (zero moment "
                "arm about that axis). Real BD trajectories rarely hit "
                "this exactly."
            ) from None

    return D_trans, D_rot, hc
