"""
Rotne-Prager-Yamakawa hydrodynamic interaction tensor
=====================================================

When two spheres diffuse in a viscous fluid, each sphere's motion
creates a flow field that affects the other.  This hydrodynamic
interaction (HI) modifies the effective diffusion coefficient.

For two spheres of radii a₁, a₂ at separation r along the line
of centres:
    D_∥(r) = kBT/(6πη) × [1/a₁ + 1/a₂ - 3/r + 2ā²/r³]
where ā² = (a₁² + a₂²)/2.

Physical interpretation:
  - 1/a₁ + 1/a₂   : free diffusion of two spheres (no coupling)
  - -3/r          : leading HI correction (slows approach)
  - +2ā²/r³       : finite-size correction

At large r: D_∥ -> D₀ = kBT/(6πη) × (1/a₁ + 1/a₂)
At contact r = a₁+a₂: D_∥ < D₀ (HI slows the approach by ~20-40%)

This tensor is from Zuk et al., J. Fluid Mech. 741, R5 (2014),
which extends the original Rotne-Prager (1969) and Yamakawa (1970)
results to handle overlapping spheres correctly.

HI reduces k_b for typical protein systems because the
effective diffusion near contact is slower than free diffusion.
"""

from __future__ import annotations
from pystarc.global_defs.constants import ETA_WATER, KB_SI, T_DEFAULT, ANG_TO_M, PS_TO_S
import numpy as np
import math


def stokes_translational_diffusion(
    radius_ang: float, eta: float = ETA_WATER, T: float = T_DEFAULT
) -> float:
    """
    Stokes-Einstein translational diffusion.  D_t = kBT / (6 π η r)
    Returns D_t in Å²/ps.
    """
    r_m = radius_ang * ANG_TO_M
    D_m2s = KB_SI * T / (6.0 * math.pi * eta * r_m)
    return D_m2s / (ANG_TO_M**2) * PS_TO_S


def stokes_rotational_diffusion(
    radius_ang: float, eta: float = ETA_WATER, T: float = T_DEFAULT
) -> float:
    """
    Stokes rotational diffusion.  D_r = kBT / (8 π η r³)
    Returns D_r in rad²/ps.
    """
    r_m = radius_ang * ANG_TO_M
    D_r_s = KB_SI * T / (8.0 * math.pi * eta * r_m**3)
    return D_r_s * PS_TO_S


def rpy_offdiagonal(
    r_vec: np.ndarray, a: float, b: float, D_a: float, D_b: float
) -> np.ndarray:
    """
    Rotne-Prager-Yamakawa off-diagonal translational mobility tensor M_12.

    Far field (r > a+b):
        tt_I  = (1 + (a²+b²)/3r²) / (8πr)
        tt_uu = (1 - (a²+b²)/r²)  / (8πr)
    Partial overlap (|a-b| < r <= a+b)  [Zuk et al. 2014 Eq. 11]:
        tt_I  = (16r³(a+b) - ((a-b)² + 3r²)²) / (192π·a·b·r³)
        tt_uu = 3·((a-b)² - r²)²             / (192π·a·b·r³)
    One inside other (r <= |a-b|):
        tt_I  = 1/(6π·max(a,b)),  tt_uu = 0
    M_12 = tt_I·I + tt_uu·r̂r̂ᵀ

    All terms are viscosity-scaled.
    PySTARC works in kBT units where D = kBT·mobility so mobility = D/kBT.
    Returns the tensor in Å²/ps units consistent with self-diffusion.
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
        # Far field
        den = PI8 * r
        a2ob2 = a2 + b2
        tt_I = (1.0 + a2ob2 / (3.0 * r2)) / den
        tt_uu = (1.0 - a2ob2 / r2) / den
    elif r > abs(a - b):
        # Partial overlap
        ab = a * b
        am2 = (a - b) ** 2  # (a-b)^2
        den = 6.0 * 32.0 * PI * ab * r3
        tt_I = (16.0 * r3 * (a + b) - (am2 + 3.0 * r2) ** 2) / den
        tt_uu = 3.0 * (am2 - r2) ** 2 / den
    else:
        # One sphere inside the other - self-mobility of larger sphere
        a_max = max(a, b)
        tt_I = 1.0 / (PI6 * a_max)
        tt_uu = 0.0
    # Scale from viscosity units to diffusion units:
    # In BD, D = kBT * mobility, and D_a = kBT/(6π·η·a).
    # So kBT/η = D_a * 6π * a.  Use geometric mean of both molecules.
    kT_over_eta = math.sqrt(D_a * PI6 * a * D_b * PI6 * b)
    M = kT_over_eta * (tt_I * I3 + tt_uu * outer)
    return M


class MobilityTensor:
    """
    Full RPY mobility tensor for a two-molecule BD system.
    Stores both diagonal (self) and off-diagonal (hydrodynamic coupling)
    terms. The effective relative diffusion D_rel_eff(r) depends on the
    current separation r between the molecules.
    When hydrodynamic_interactions = True , the BD step
    uses D_rel_eff(r) instead of the constant D_t1 + D_t2. This slows
    down the relative diffusion when the molecules are close - physically
    correct because close molecules drag solvent into the gap between them.
    Parameters
    ----------
    r1, r2  : hydrodynamic radii [Å]
    D_t1, D_r1 : translational/rotational diffusion for molecule 1 [Å²/ps, rad²/ps]
    D_t2, D_r2 : translational/rotational diffusion for molecule 2
    use_rpy    : if True (default), use full RPY coupling and if False, use diagonal approximation (faster but less accurate)
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
        """Build MobilityTensor from hydrodynamic radii."""
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
        Effective relative translational diffusion at separation r_vec.
        Without RPY (diagonal):
            D_rel = D_t1 + D_t2
        With RPY coupling (default: hydrodynamic_interactions=true):
            D_rel_eff(r) = D_t1 + D_t2 - (2/3) tr(M_12(r))
        The RPY correction reduces D_rel near contact (r ≈ a+b) and
        vanishes at large r, recovering the diagonal result.
        """
        D0 = self.D_trans1 + self.D_trans2
        if not self.use_rpy or r_vec is None:
            return D0
        if self.radius1 <= 0.0 or self.radius2 <= 0.0:
            return D0
        M12 = rpy_offdiagonal(
            r_vec, self.radius1, self.radius2, self.D_trans1, self.D_trans2
        )
        # Scalar coupling: effective D_rel = D_t1 + D_t2 - (2/3) tr(M_12)
        # Factor 2 because both molecules feel the coupling symmetrically
        D_coupling = (2.0 / 3.0) * np.trace(M12)
        return max(D0 - D_coupling, 1e-12)  # never negative

    def relative_rotational_diffusion(self) -> float:
        """D_r_rel = D_r1 + D_r2 (RPY rotational coupling is negligible)."""
        return self.D_rot1 + self.D_rot2

    def __repr__(self) -> str:
        return (
            f"MobilityTensor(Dt1={self.D_trans1:.3e}, Dr1={self.D_rot1:.3e}, "
            f"Dt2={self.D_trans2:.3e}, Dr2={self.D_rot2:.3e}, "
            f"r1={self.radius1:.1f}A, r2={self.radius2:.1f}A, RPY={self.use_rpy})"
        )


# Full RPY pair tensor (translation + rotation + cross-coupling).
# All returned coefficients carry units of (length)^(-k) where
# k depends on the block (1 for tt, 2 for tr/rt, 3 for rr).
# Multiply by 1/eta to recover physical mobility, and by kT
# to recover diffusion coefficients.


def rpy_full_components(ai, aj, r):
    """Scalar coefficients of the RPY pair mobility tensor.

    Returns six scalars (tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps)
    that combine with identity, outer-product, and Levi-Civita-on-u
    matrices to assemble the four 3x3 blocks of the pair tensor.
    See rpy_pair_blocks() for the assembly.

    Three regimes:
      r > ai + aj           : non-overlapping (Rotne-Prager-Yamakawa)
      |ai-aj| < r <= ai+aj  : partial overlap (Zuk et al. 2014)
      r <= |ai-aj|          : one sphere fully inside the other
    """
    pi = math.pi
    pi6 = 6.0 * pi
    pi8 = 8.0 * pi
    ai2 = ai * ai
    aj2 = aj * aj
    r2 = r * r
    r3 = r2 * r

    if r > ai + aj:
        # Non-overlapping (far field).
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
        # Partial overlap.
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

        # Cross-coupling: BD2 has an asymmetric f(ai,aj) helper.
        def _rt_term(a_force, a_torque):
            af2 = a_force * a_force
            num = (a_force - a_torque + r) ** 2 * (
                a_torque**2 + 2.0 * a_torque * (a_force + r) - 3.0 * (a_force - r) ** 2
            )
            return num / (8.0 * 16.0 * pi * af2 * a_force * a_torque * r2)

        rt_eps = _rt_term(ai, aj)
        tr_eps = _rt_term(aj, ai)
        return tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps

    # Fully nested: one sphere strictly inside the other.
    a = max(ai, aj)
    tt_I = 1.0 / (pi6 * a)
    tt_uu = 0.0
    rr_I = 1.0 / (8.0 * pi * a * a * a)
    rr_uu = 0.0
    # BD2's get_components() leaves rt/tr at zero in this regime
    # ("add rt components later"). Follow the same choice.
    rt_eps = 0.0
    tr_eps = 0.0
    return tt_I, tt_uu, rr_I, rr_uu, rt_eps, tr_eps


def rpy_pair_blocks(ai, aj, r_ij):
    """Full 3x3 block decomposition of the RPY pair mobility tensor.

    Returns (mtt, mrt, mtr, mrr), each a 3x3 numpy array, such that
    the linear response of bead i's velocity and angular velocity to
    a force and torque applied at bead j is:

        [v_i ]   [ mtt  mtr ] [ F_j ]
        [w_i ] = [ mrt  mrr ] [ T_j ]

    Convention: Viscosity is not applied; multiply blocks by
    1/eta to recover physical mobility.
    """
    r = float(np.linalg.norm(r_ij))
    if r < 1e-12:
        # Degenerate: same point. Caller should use rpy_self_blocks.
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))

    u = np.asarray(r_ij, dtype=float) / r
    uu = np.outer(u, u)
    I3 = np.eye(3)

    # Levi-Civita applied to u: eps_ijk u_k. The signed cross-product
    # matrix such that eps_u @ v = u x v.
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
    """Single-bead self-mobility (the diagonal i = j entry).

    Returns (mtt_self, mrr_self), each a 3x3 numpy array. The cross-
    coupling self-blocks are zero by symmetry for an isolated sphere.

    Convention: viscosity is not applied. mtt_self = I / (6 pi a),
    mrr_self = I / (8 pi a^3).
    """
    pi = math.pi
    I3 = np.eye(3)
    mtt_self = I3 / (6.0 * pi * a)
    mrr_self = I3 / (8.0 * pi * a * a * a)
    return mtt_self, mrr_self


def rpy_full_mobility_matrix(positions, radii):
    """Assemble the full 6N x 6N RPY mobility matrix for N spheres.

    The mobility matrix M relates generalized forces (force + torque)
    on each bead to generalized velocities (linear + angular) on each
    bead, hydrodynamically coupled through the surrounding fluid:

        v_i = sum_j (mtt_ij F_j + mtr_ij T_j)
        w_i = sum_j (mrt_ij F_j + mrr_ij T_j)

    Block layout: bead i occupies DOFs [6i, 6i+6), with the first
    three for translation and the last three for rotation. The 6x6
    block at (i, j) decomposes as
        M[6i:6i+3, 6j:6j+3] = mtt(i,j)
        M[6i:6i+3, 6j+3:6j+6] = mtr(i,j)
        M[6i+3:6i+6, 6j:6j+3] = mrt(i,j)
        M[6i+3:6i+6, 6j+3:6j+6] = mrr(i,j)
    where for i = j these come from rpy_self_blocks (cross-coupling
    is zero for an isolated sphere) and for i != j from rpy_pair_blocks.

    Convention: viscosity is not applied; multiply M by 1/eta for
    physical mobility, or by kT/eta for diffusion units.

    Parameters
    ----------
    positions : (N, 3) array of bead centers (in any consistent frame).
    radii     : (N,) array of bead hydrodynamic radii.

    Returns
    -------
    M : (6N, 6N) numpy array. Symmetric to numerical precision by
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
        # Diagonal block: self-mobility, no cross-coupling.
        mtt_self, mrr_self = rpy_self_blocks(radii[i])
        M[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = mtt_self
        M[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = mrr_self
        # mtr and mrt blocks are zero on the diagonal (no self-cross
        # coupling for an isolated sphere); leave them at zero.

        for j in range(i + 1, n):
            # Off-diagonal block (i, j) and its transpose-pair (j, i).
            r_ij = positions[j] - positions[i]
            mtt_ij, mrt_ij, mtr_ij, mrr_ij = rpy_pair_blocks(
                radii[i],
                radii[j],
                r_ij,
            )
            # (i, j) block.
            M[6 * i : 6 * i + 3, 6 * j : 6 * j + 3] = mtt_ij
            M[6 * i : 6 * i + 3, 6 * j + 3 : 6 * j + 6] = mtr_ij
            M[6 * i + 3 : 6 * i + 6, 6 * j : 6 * j + 3] = mrt_ij
            M[6 * i + 3 : 6 * i + 6, 6 * j + 3 : 6 * j + 6] = mrr_ij
            # (j, i) block = (i, j)^T by Onsager reciprocity. Fill
            # by transpose rather than re-evaluating to guarantee
            # exact numerical symmetry.
            M[6 * j : 6 * j + 3, 6 * i : 6 * i + 3] = mtt_ij.T
            M[6 * j : 6 * j + 3, 6 * i + 3 : 6 * i + 6] = mrt_ij.T
            M[6 * j + 3 : 6 * j + 6, 6 * i : 6 * i + 3] = mtr_ij.T
            M[6 * j + 3 : 6 * j + 6, 6 * i + 3 : 6 * i + 6] = mrr_ij.T

    return M


def _hydrodynamic_center(positions, radii):
    """Radius-weighted centroid: hc = (sum a_i r_i) / (sum a_i).

    This is the natural reference point for a heterogeneous-radius chain;
    all bead radii enter linearly, so larger beads pull the center
    toward themselves.
    """
    radii = np.asarray(radii, dtype=float)
    positions = np.asarray(positions, dtype=float)
    a_sum = float(np.sum(radii))
    return (radii[:, None] * positions).sum(axis=0) / a_sum


def _translation_only_mobility(positions, radii):
    """Extract the 3N x 3N translation-translation block of the full
    RPY mobility matrix. Each 3x3 block (i, j) is mtt(a_i, a_j, r_ij).
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


def chain_rigid_body_resistance(positions, radii):
    """Compute the chain's rigid-body resistance matrices and hydrodynamic
    center, given the bead positions and radii.

    Returns (A, C, hc):
      A  : (3, 3) translational resistance, force per unit velocity.
           A acts as: F_net = eta * A @ v_chain, where v_chain is the
           uniform translational velocity of the rigid chain.
      C  : (3, 3) rotational resistance, torque per unit angular
           velocity, taken about the hydrodynamic center.
           T_net = eta * C @ omega_chain.
      hc : (3,) hydrodynamic center (radius-weighted centroid).

    Convention: viscosity is NOT included. Multiply A by eta for
    physical resistance. The corresponding rigid-body translational
    diffusion is D_trans = kT * inv(eta * A); similarly for D_rot.

    Algorithm: For each Cartesian direction k, prescribe the rigid-body
    velocity field, solve M_tt F = v for the per-bead forces required to
    maintain that motion against the fluid, and read off the column of
    A or C from the appropriate sum or moment of those forces.
    This avoids ever forming M_tt^{-1} explicitly.

    Parameters
    ----------
    positions : (N, 3) bead centers.
    radii     : (N,)   bead hydrodynamic radii.

    Notes
    -----
    The translation-rotation cross-coupling block B is not returned.
    For chains that are well-centered at the hydrodynamic center,
    B is small or zero by symmetry; full Stokesian-dynamics
    treatment would compute it.
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

    # Cholesky factor of M_tt. M_tt is symmetric positive definite by
    # construction (it is the translation block of an RPY mobility tensor
    # with non-overlapping or properly regularized overlapping spheres).
    # If a user supplies pathological geometry (e.g. coincident beads),
    # this will raise; let numpy's error message surface.
    L = np.linalg.cholesky(M_tt)

    def solve(v):
        """Solve M_tt @ F = v using the precomputed Cholesky factor."""
        # M_tt = L L^T, so F = L^-T (L^-1 v).
        y = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, y)

    # A matrix: prescribe uniform translation v_i = e_k for every bead.
    A = np.zeros((3, 3))
    for k in range(3):
        v = np.zeros(3 * n)
        for i in range(n):
            v[3 * i + k] = 1.0
        F = solve(v)
        F_per_bead = F.reshape(n, 3)
        A[:, k] = F_per_bead.sum(axis=0)

    # C matrix: prescribe rigid-body rotation omega = e_k about hc, so
    # bead i moves with velocity v_i = omega x (r_i - hc).
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

    Returns (D_trans, D_rot, hc):
      D_trans : (3, 3) translational diffusion tensor [length^2 / time]
      D_rot   : (3, 3) rotational diffusion tensor    [1 / time]
      hc      : (3,)   hydrodynamic center            [length]

    The diffusion tensors are obtained by inverting the resistance
    matrices A and C (from chain_rigid_body_resistance) and scaling
    by kT / eta:

        D_trans = (kT / eta) * inv(A)
        D_rot   = (kT / eta) * inv(C)

    Parameters
    ----------
    positions : (N, 3) bead centers.
    radii     : (N,)   bead hydrodynamic radii.
    kT        : thermal energy. PySTARC convention is kT = 1.
    viscosity : solvent viscosity. If None, uses the package default
                WATER_VISCOSITY = 0.243 kBT.ps/A^3 from motion/do_bd_step.

    Notes
    -----
    For a chain of N >= 2 beads, both A and C are invertible by
    construction. For N = 1, A is invertible but C is the zero matrix
    (see chain_rigid_body_resistance docstring); a single-bead chain
    cannot use the algorithm without bead-rotation contributions.
    Callers should special-case N = 1 by using the bead's own
    rotational mobility 1/(8 pi eta a^3).
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
        # C is zero by construction in BD2's algorithm; fall back to
        # the single-bead Stokes rotational mobility for correctness.
        a = float(radii[0])
        D_rot = (kT / viscosity) * np.eye(3) / (8.0 * math.pi * a**3)
    else:
        # Singular C arises for special chain geometries (e.g. a
        # perfectly collinear chain has zero rotational drag about its
        # own axis since all moment arms vanish). The user constructed
        # a configuration where the chain isn't a well-defined three-
        # DOF rigid rotor; raise a clear error rather than a generic
        # numpy LinAlgError.
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
