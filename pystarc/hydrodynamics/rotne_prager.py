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
from scipy.linalg import solve_triangular
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
    PI = math.pi
    PI6 = 6.0 * PI
    PI8 = 8.0 * PI
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
        a_max = max(a, b)
        prod = D_a * PI6 * a * D_b * PI6 * b
        if a_max <= 0.0 or prod <= 0.0:
            return np.zeros((3, 3))
        kT_over_eta = math.sqrt(prod)
        return (kT_over_eta / (PI6 * a_max)) * np.eye(3)
    rhat = r_vec / r
    outer = np.outer(rhat, rhat)
    I3 = np.eye(3)
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

            D_rel_eff(r) = D_t1 + D_t2 - 2 r̂·M_12(r)·r̂,

        where M_12(r) is the off-diagonal mobility tensor. The RPY
        correction reduces D_rel near contact (r ≈ a+b) and vanishes at
        large r, recovering the diagonal result.

        The radial projection r̂·D_rel·r̂ is returned, the same scalar the k_b
        encounter integral uses. It is the component that controls the
        splitting probability between the reaction and escape surfaces. The
        isotropic average, one third of the trace, differs from it by about 8
        percent at contact. The two reductions do not coincide even at the
        b-surface: their ratio is still 0.95
        to 0.97 at ten contact radii.
        """
        D0 = self.D_trans1 + self.D_trans2
        if not self.use_rpy or r_vec is None:
            return D0
        if self.radius1 <= 0.0 or self.radius2 <= 0.0:
            return D0
        M12 = rpy_offdiagonal(
            r_vec, self.radius1, self.radius2, self.D_trans1, self.D_trans2
        )
        # The radial projection, D_rel = D_t1 + D_t2 - 2 r̂·M_12·r̂. The factor
        # of two appears because both molecules feel the coupling
        # symmetrically. Written as a projection rather than a far field
        # formula so it stays correct across all three Zuk regimes.
        _rn = float(np.linalg.norm(r_vec))
        rhat = np.asarray(r_vec, dtype=float) / max(_rn, 1e-12)
        D_coupling = 2.0 * float(rhat @ M12 @ rhat)
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


def radial_D_gradient(mobility, pos, D_here: float, dt: float) -> np.ndarray:
    """Return the divergence of the scalar relative diffusion, dD/dr r_hat.

    The scalar D(r) of a two-body step is not a divergence free tensor the way
    the full RPY pair mobility is, so the Ermak-McCammon step owes this term.
    The derivative is central differenced on the same mobility function that
    supplies D, which keeps the three Rotne-Prager-Yamakawa regimes consistent
    rather than hardcoding a far field formula that is wrong in the partial
    overlap and engulfed branches. The step is h = 1e-3 max(r, 1). The
    displacement is bounded at a tenth of the random step, 0.1 sqrt(2 D_here /
    dt), a limit that never binds on a physical configuration and exists only
    to contain the inverse quartic blow up at very small separations.

    The argument mobility is a MobilityTensor, pos is the (3,) separation, and
    D_here is the scalar diffusion at pos that sets the bound. A separation
    below 1e-8 returns zeros.
    """
    r = float(np.linalg.norm(pos))
    if r < 1.0e-8:
        return np.zeros(3)
    rhat = pos / r
    h = 1.0e-3 * max(r, 1.0)
    if r - h <= 1.0e-8:
        return np.zeros(3)
    dD_dr = (
        mobility.relative_translational_diffusion((r + h) * rhat)
        - mobility.relative_translational_diffusion((r - h) * rhat)
    ) / (2.0 * h)
    if dt > 0.0:
        cap = 0.1 * math.sqrt(2.0 * D_here / dt)
        dD_dr = max(-cap, min(cap, dD_dr))
    return dD_dr * rhat


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

    # One sphere is strictly inside the other (Zuk et al. 2014, Eq. 3.15).
    a = max(ai, aj)
    tt_I = 1.0 / (pi6 * a)
    tt_uu = 0.0
    rr_I = 1.0 / (8.0 * pi * a * a * a)
    rr_uu = 0.0
    if ai > aj:
        rt_eps = r / (8.0 * pi * ai * ai * ai)
        tr_eps = 0.0
    elif aj > ai:
        rt_eps = 0.0
        tr_eps = r / (8.0 * pi * aj * aj * aj)
    else:
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
        # Two coincident beads are one bead, so their mutual mobility is the
        # self mobility of the LARGER of them, not zero. Both limits of
        # rpy_full_components agree on this: tt_I tends to 1/(6 pi a_max) and
        # rr_I to 1/(8 pi a_max^3), with tt_uu, rr_uu and the cross terms
        # tending to zero. Returning zeros would instead charge the duplicated
        # site twice for friction, overstating the resistance and understating
        # D_trans, while leaving the assembled matrix positive definite so no
        # warning fires.
        a_max = max(ai, aj)
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        return (
            I3 / (6.0 * math.pi * a_max),
            Z3,
            Z3,
            I3 / (8.0 * math.pi * a_max * a_max * a_max),
        )

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

    # Positive finite radii are required for a positive definite RPY mobility
    # matrix.
    if not np.all(np.isfinite(radii)):
        bad = np.flatnonzero(~np.isfinite(radii))
        raise ValueError(
            "bead hydrodynamic radii must all be finite; non-finite at "
            f"indices {bad[:10].tolist()} (of {bad.size} total), values "
            f"{radii[bad[:10]].tolist()}"
        )
    if not np.all(radii > 0.0):
        bad = np.flatnonzero(~(radii > 0.0))
        raise ValueError(
            "bead hydrodynamic radii must all be strictly positive; "
            f"non-positive at indices {bad[:10].tolist()} (of {bad.size} "
            f"total), values {radii[bad[:10]].tolist()}"
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
    if a_sum <= 0.0:
        raise ValueError(
            "hydrodynamic center is undefined: the sum of bead radii must "
            f"be positive, got sum(radii)={a_sum}. This happens for empty "
            "input or all-zero radii."
        )
    return (radii[:, None] * positions).sum(axis=0) / a_sum


def _translation_only_mobility(positions, radii):
    """Assemble the 3N x 3N translation-translation block of the RPY mobility
    matrix. Each 3x3 block (i, j) is mtt(a_i, a_j, r_ij).

    The assembly is vectorized over all pairs and evaluates the same three
    regimes as rpy_full_components. The diagonal blocks are I/(6 pi a_i). A
    coincident pair, r < 1e-12, takes I/(6 pi max(a_i, a_j)) with no
    directional part, as rpy_pair_blocks does.
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if not np.all(np.isfinite(radii)):
        bad = np.flatnonzero(~np.isfinite(radii))
        raise ValueError(
            "bead hydrodynamic radii must all be finite; non-finite at "
            f"indices {bad[:10].tolist()} (of {bad.size} total), values "
            f"{radii[bad[:10]].tolist()}"
        )
    if not np.all(radii > 0.0):
        bad = np.flatnonzero(~(radii > 0.0))
        raise ValueError(
            "bead hydrodynamic radii must all be strictly positive; "
            f"non-positive at indices {bad[:10].tolist()} (of {bad.size} "
            f"total), values {radii[bad[:10]].tolist()}"
        )
    n = positions.shape[0]
    pi = math.pi
    M_tt = np.zeros((3 * n, 3 * n), dtype=float)
    blocks = M_tt.reshape(n, 3, n, 3)
    idx = np.arange(n)
    blocks[idx, :, idx, :] = np.eye(3)[None, :, :] / (6.0 * pi * radii)[:, None, None]
    if n < 2:
        return M_tt
    ii, jj = np.triu_indices(n, k=1)
    ai = radii[ii]
    aj = radii[jj]
    d = positions[jj] - positions[ii]
    r = np.linalg.norm(d, axis=1)
    coincident = r < 1e-12
    # Evaluate the formulas on a safe separation, then overwrite the coincident
    # pairs.
    r = np.where(coincident, 1.0, r)
    u = d / r[:, None]
    r2 = r * r
    r3 = r2 * r
    far = r > ai + aj
    mid = (~far) & (r > np.abs(ai - aj))
    # Far field, the spheres do not overlap.
    den_far = 8.0 * pi * r
    a2or2 = (ai * ai + aj * aj) / r2
    tt_I_far = (1.0 + a2or2 / 3.0) / den_far
    tt_uu_far = (1.0 - a2or2) / den_far
    # The spheres partially overlap.
    aij = ai * aj
    am2 = (ai - aj) ** 2
    den_mid = 6.0 * 32.0 * pi * aij * r3
    tt_I_mid = (16.0 * r3 * (ai + aj) - (am2 + 3.0 * r2) ** 2) / den_mid
    tt_uu_mid = 3.0 * (am2 - r2) ** 2 / den_mid
    # One sphere lies inside the other.
    tt_I_eng = 1.0 / (6.0 * pi * np.maximum(ai, aj))
    tt_I = np.where(far, tt_I_far, np.where(mid, tt_I_mid, tt_I_eng))
    tt_uu = np.where(far, tt_uu_far, np.where(mid, tt_uu_mid, 0.0))
    tt_I = np.where(coincident, 1.0 / (6.0 * pi * np.maximum(ai, aj)), tt_I)
    tt_uu = np.where(coincident, 0.0, tt_uu)
    u = np.where(coincident[:, None], 0.0, u)
    pair_blocks = tt_I[:, None, None] * np.eye(3)[None, :, :] + tt_uu[:, None, None] * (
        u[:, :, None] * u[:, None, :]
    )
    blocks[ii, :, jj, :] = pair_blocks
    blocks[jj, :, ii, :] = pair_blocks
    return M_tt


def _indefiniteness_report(M, trace_avg):
    """Describe how badly a mobility matrix fails to be positive definite.

    The physical scale of an RPY translational mobility is the mean self
    mobility 1/(6 pi a), which is 1.768e-02 per angstrom for a 3 A bead. An
    absolute eigenvalue threshold carries no meaning against that: -8e-03
    reads as small although it is a 49 percent violation of the matrix scale, so the
    ratio is reported alongside.
    """
    if not np.all(np.isfinite(M)):
        n_bad = int((~np.isfinite(M)).sum())
        return f"matrix contains {n_bad} non-finite entries"
    eigvals = np.linalg.eigvalsh(M)
    lam_min = float(eigvals.min())
    lam_max = float(eigvals.max())
    n_neg = int((eigvals < 0.0).sum())
    ratio = lam_min / abs(trace_avg) if trace_avg else float("nan")
    return (
        f"{n_neg} of {eigvals.size} eigenvalues are negative; "
        f"lambda_min={lam_min:.6e}, lambda_max={lam_max:.6e}, "
        f"mobility scale trace(M)/n={trace_avg:.6e}, "
        f"lambda_min/scale={ratio:.4f}"
    )


def _build_robust_solver(M):
    """Build a callable solver(v) for M @ x = v that handles a
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
        solver : callable(v) returning an approximate M^{-1}·v, where v has
                 shape (n,) or (n, k)
        was_regularized : bool, True if any fallback strategy was used
        info : str describing the strategy actually applied
    """
    n = M.shape[0]
    trace_avg = float(np.trace(M)) / max(n, 1)

    # First strategy, a plain Cholesky factorization.
    try:
        L = np.linalg.cholesky(M)

        def solver_chol(v, L=L):
            y = solve_triangular(L, v, lower=True)
            return solve_triangular(L, y, lower=True, trans="T")

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
                y = solve_triangular(L, v, lower=True)
                return solve_triangular(L, y, lower=True, trans="T")

            return solver_jit, True, f"cholesky+jitter(eps={eps:.3e})"
        except np.linalg.LinAlgError:
            continue

    # Reaching here is not ill conditioning. The largest jitter rung above is
    # eps = 1e-6 in absolute mobility units, which is 5.7e-05 of the RPY matrix
    # scale 1/(6 pi a) for a 3 A bead and about 4e9 times the Cholesky roundoff
    # threshold. Any positive semi-definite M is absorbed by the jitter, so
    # arriving here means an eigenvalue below roughly -1e-6.
    #
    # The RPY kernel is positive definite for every configuration with positive
    # finite radii, so an indefinite M is a data or assembly defect rather than
    # a numerical one.
    raise np.linalg.LinAlgError(
        "RPY mobility matrix is indefinite and cannot be factorized: "
        + _indefiniteness_report(M, trace_avg)
        + ". Diagonal jitter up to eps=1e-6 was tried first and failed, so "
        "this is not conditioning or roundoff. Check (a) bead radii that are "
        "non-positive or non-finite, and (b) beads separated by less than "
        "1e-12 A. Beads that are merely close are not the cause: a pair "
        "1e-11 A apart gives lambda_min = +1.1e-14 and factorizes."
    )


def chain_rigid_body_resistance_full(
    positions, radii, center=None, warn_regularized=True, info=None
):
    """Compute rigid-body resistance blocks (A, B, C) and hydrodynamic center hc.

    Returns (A, B, C, hc). Matrix A is translational resistance (3, 3), matrix B
    is rotation-translation coupling (3, 3) mapping uniform translation to torque
    about center, matrix C is rotational resistance (3, 3) about center, and hc
    is the radius-weighted centroid (3,). Viscosity is omitted. Multiply by eta
    for physical resistance.
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
    c = hc if center is None else np.asarray(center, dtype=float)
    M_tt = _translation_only_mobility(positions, radii)

    solve, _was_regularized, _solver_info = _build_robust_solver(M_tt)
    if info is not None:
        info["regularized"] = bool(_was_regularized)
    if _was_regularized and warn_regularized:
        warnings.warn(
            f"chain_rigid_body_resistance: RPY mobility matrix M_tt was not "
            f"strictly positive definite and needed regularization "
            f"({_solver_info}). The usual cause is a redundant degree of "
            f"freedom, for example two beads at the same coordinate. Beads "
            f"that are merely CLOSE do not trigger this: a pair 1e-11 A apart "
            f"gives lambda_min = +1.1e-14 and factorizes with plain Cholesky, "
            f"and heavy sphere overlap is handled exactly by the Zuk partial "
            f"overlap branch. Checking bead distances against bead radii will "
            f"not find the cause.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Columns 0..2 prescribe unit translation v_i = e_k. Columns 3..5 prescribe
    # rigid rotation omega = e_k about center with v_i = omega x (r_i - center).
    d = positions - c
    V = np.zeros((3 * n, 6))
    for k in range(3):
        V[k::3, k] = 1.0
        omega = np.zeros(3)
        omega[k] = 1.0
        V[:, 3 + k] = np.cross(omega, d).reshape(-1)
    F = solve(V).reshape(n, 3, 6)

    A = F[:, :, :3].sum(axis=0)
    B = np.zeros((3, 3))
    C = np.zeros((3, 3))
    for k in range(3):
        B[:, k] = np.cross(d, F[:, :, k]).sum(axis=0)
        C[:, k] = np.cross(d, F[:, :, 3 + k]).sum(axis=0)

    if info is not None:
        info["B"] = B

    return A, B, C, hc


def chain_rigid_body_resistance(
    positions, radii, center=None, warn_regularized=True, info=None
):
    """Compute rigid-body resistance matrices (A, C) and hydrodynamic center hc.

    Returns (A, C, hc). When info is a dict, info["B"] stores the (3, 3)
    rotation-translation coupling block and info["regularized"] records
    whether Cholesky regularization occurred.
    """
    A, _B, C, hc = chain_rigid_body_resistance_full(
        positions,
        radii,
        center=center,
        warn_regularized=warn_regularized,
        info=info,
    )
    return A, C, hc


def _chain_rigid_body_resistance_6n(
    positions, radii, center=None, warn_regularized=True, info=None
):
    """Compute full 6x6 rigid-body resistance blocks (A, B, C) via 6N RPY mobility."""
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
    c = hc if center is None else np.asarray(center, dtype=float)

    M_full = rpy_full_mobility_matrix(positions, radii)
    solve, _was_regularized, _solver_info = _build_robust_solver(M_full)
    if info is not None:
        info["regularized"] = bool(_was_regularized)
    if _was_regularized and warn_regularized:
        warnings.warn(
            f"chain_diffusion_tensors (method='6n'): full RPY mobility matrix "
            f"was not strictly positive definite and needed regularization "
            f"({_solver_info}).",
            RuntimeWarning,
            stacklevel=2,
        )

    d = positions - c
    K = np.zeros((6 * n, 6), dtype=float)
    for i in range(n):
        r0 = 6 * i
        di = d[i]
        K[r0 : r0 + 3, 0:3] = np.eye(3)
        K[r0, 4] = di[2]
        K[r0, 5] = -di[1]
        K[r0 + 1, 3] = -di[2]
        K[r0 + 1, 5] = di[0]
        K[r0 + 2, 3] = di[1]
        K[r0 + 2, 4] = -di[0]
        K[r0 + 3 : r0 + 6, 3:6] = np.eye(3)

    FT = solve(K)
    Xi = K.T @ FT
    Xi = 0.5 * (Xi + Xi.T)
    A = Xi[:3, :3].copy()
    B = Xi[3:, :3].copy()
    C = Xi[3:, 3:].copy()
    if info is not None:
        info["B"] = B
    return A, B, C, hc


def _diffusion_from_resistance_full(A, B, C, c, scale, info=None):
    """Invert 6x6 resistance (A, B, C) via Schur complement at center of diffusion."""
    A_sym = 0.5 * (A + A.T)
    A_inv = np.linalg.inv(A_sym)
    C_schur = C - B @ A_inv @ B.T
    C_schur = 0.5 * (C_schur + C_schur.T)
    eigvals = np.linalg.eigvalsh(C_schur)
    scale_c = abs(float(eigvals[-1]))
    if scale_c == 0.0 or float(eigvals[0]) <= 1e-12 * scale_c:
        raise np.linalg.LinAlgError(
            "rigid-body rotational resistance C is singular for this "
            f"chain geometry (eigenvalues {eigvals})."
        )
    try:
        D_rr = scale * np.linalg.inv(C_schur)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            "rigid-body rotational resistance C is singular for this "
            f"chain geometry (eigenvalues {eigvals})."
        ) from None
    D_rr = 0.5 * (D_rr + D_rr.T)
    D_tr_O = -A_inv @ B.T @ D_rr
    D_rt_O = D_tr_O.T
    D_tt_O = scale * A_inv + A_inv @ B.T @ D_rr @ B @ A_inv
    rhs = np.array(
        [
            D_rt_O[1, 2] - D_rt_O[2, 1],
            D_rt_O[2, 0] - D_rt_O[0, 2],
            D_rt_O[0, 1] - D_rt_O[1, 0],
        ],
        dtype=float,
    )
    delta = np.linalg.solve(np.trace(D_rr) * np.eye(3) - D_rr, rhs)
    U = np.array(
        [
            [0.0, -delta[2], delta[1]],
            [delta[2], 0.0, -delta[0]],
            [-delta[1], delta[0], 0.0],
        ],
        dtype=float,
    )
    D_trans = D_tt_O - U @ D_rr @ U + D_tr_O @ U - U @ D_tr_O.T
    D_trans = 0.5 * (D_trans + D_trans.T)
    if info is not None:
        info["center_of_diffusion"] = c + delta
    return D_trans, D_rr


def chain_diffusion_tensors(
    positions,
    radii,
    kT=1.0,
    viscosity=None,
    center=None,
    warn_regularized=True,
    info=None,
    include_coupling=True,
    method="3n",
):
    """Compute rigid-body diffusion tensors (D_trans, D_rot) and hydrodynamic center hc.

    With include_coupling=True (default), the full 6x6 resistance matrix (A, B, C)
    is inverted via Schur complement and shifted to the center of diffusion r_D,
    producing origin-independent D_trans and D_rot tensors. Setting method='6n'
    assembles the full 6N x 6N RPY mobility matrix including bead rotations and
    rotation-translation coupling. When info is supplied, info['center_of_diffusion']
    records r_D and info['B'] records the coupling block about center.
    """
    from pystarc.motion.do_bd_step import WATER_VISCOSITY

    if viscosity is None:
        viscosity = WATER_VISCOSITY
    if method not in ("3n", "6n"):
        raise ValueError(f"method must be '3n' or '6n'; got {method!r}")

    n = len(positions)
    if n < 1:
        raise ValueError("at least one bead required")

    scale = kT / viscosity

    if method == "6n":
        A, B, C, hc = _chain_rigid_body_resistance_6n(
            positions,
            radii,
            center=center,
            warn_regularized=warn_regularized,
            info=info,
        )
        c = hc if center is None else np.asarray(center, dtype=float)
        if include_coupling:
            D_trans, D_rot = _diffusion_from_resistance_full(
                A, B, C, c, scale, info=info
            )
        else:
            D_trans = scale * np.linalg.inv(A)
            D_rot = scale * np.linalg.inv(C)
        return D_trans, D_rot, hc

    A, B, C, hc = chain_rigid_body_resistance_full(
        positions,
        radii,
        center=center,
        warn_regularized=warn_regularized,
        info=info,
    )

    if n == 1:
        D_trans = scale * np.linalg.inv(A)
        a = float(radii[0])
        D_rot = scale * np.eye(3) / (8.0 * math.pi * a**3)
        if info is not None:
            info["center_of_diffusion"] = hc.copy()
        return D_trans, D_rot, hc

    if include_coupling:
        c = hc if center is None else np.asarray(center, dtype=float)
        D_trans, D_rot = _diffusion_from_resistance_full(A, B, C, c, scale, info=info)
        return D_trans, D_rot, hc

    D_trans = scale * np.linalg.inv(A)
    C_sym = 0.5 * (C + C.T)
    try:
        eigvals = np.linalg.eigvalsh(C_sym)
        scale_c = abs(float(eigvals[-1]))
        if scale_c == 0.0 or float(eigvals[0]) <= 1e-12 * scale_c:
            raise np.linalg.LinAlgError("singular")
        D_rot = scale * np.linalg.inv(C)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(C_sym)
        raise np.linalg.LinAlgError(
            "rigid-body rotational resistance C is singular for this "
            f"chain geometry (eigenvalues {eigvals})."
        ) from None

    return D_trans, D_rot, hc
