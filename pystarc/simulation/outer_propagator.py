"""
Outer propagator and the k_b encounter rate.

The Northrup-Allison-McCammon (NAM) method writes the association rate
constant as the product

    k_on = k_b × P_rxn

where k_b is the diffusion-limited encounter rate at a spherical b-surface
of radius b, and P_rxn is the reaction probability obtained from the
Brownian-dynamics simulation.

We compute the encounter rate from the steady-state Smoluchowski equation
with a screened Coulomb (Yukawa) interaction potential,

    k_b = 4π / ∫₀^(1/b) [exp(V(1/s)/kBT) / D_∥(1/s)] ds.

The change of variables s = 1/r turns the semi-infinite integral over
[b, ∞) into the finite interval [0, 1/b]. Here V(r) is the Yukawa monopole
potential Q₁Q₂/(4πε r) × exp(-r/λ), and D_∥(r) is the distance-dependent
parallel diffusion coefficient that includes Rotne-Prager-Yamakawa
hydrodynamic interactions. The integral is evaluated by Romberg
integration, the routine of the GPU path, to a tolerance of 10⁻⁸.

When a trajectory diffuses out beyond the escape sphere at r_esc = 2b, we
must decide whether it returns to the b-surface or escapes to infinity. The
Luty-McCammon-Zhou (LMZ) prescription gives the return probability as

    p_return = k_b(b) / k_b(r_esc).

This ratio accounts for electrostatic steering. If the molecules attract
each other, p_return is greater than 0.5, so the trajectory is more likely
to return than to escape. For neutral molecules p_return = b/r_esc = 0.5,
exactly what free diffusion predicts. Returned trajectories are placed back
on the b-surface with a uniformly random orientation, which is justified
because the outer-propagator time is much longer than the rotational
diffusion time.

The parallel diffusion coefficient D_∥(r) encodes the hydrodynamic coupling
between two spheres at separation r,

    D_∥(r) = kBT/(6πη) × (1/a₁ + 1/a₂ - 3/r + 2ā²/r³),

where ā² = (a₁² + a₂²)/2. At large r, D_∥ approaches the free-diffusion
limit D₀ = D₁ + D₂. At contact, r = a₁ + a₂, the hydrodynamic interaction
slows the approach by a factor of about 0.6 to 0.8, which reduces k_b by
the same factor.
"""

from __future__ import annotations
from pystarc.simulation.step_near_surface import step_near_absorbing_surface
from pystarc.simulation.diffusional_rotation import diffusional_rotation
from pystarc.simulation.diffusional_rotation import quat_multiply
from pystarc.simulation.diffusional_rotation import quat_rotate_vector
from pystarc.lib.numerical import romberg_integrate
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import math

# Physical constants
PI = math.pi
PI4 = 4.0 * PI
PI6 = 6.0 * PI
PI8 = 8.0 * PI
LARGE = 1.0e30


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Draw an isotropic unit vector on the unit sphere."""
    while True:
        v = rng.standard_normal(3)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            return v / nrm


@dataclass
class OPGroupInfo:
    """Per-molecule info needed by the outer propagator."""

    q: float  # Total charge in elementary units.
    Dtrans: float  # Translational diffusion coefficient in A^2/ps.
    Drot: float  # Rotational diffusion coefficient in rad^2/ps.


class OuterPropagator:
    """Analytical propagator for the outer diffusion region b <= r <= q.

    The constructor accepts b_radius in angstrom, max_radius in angstrom
    (setting q = max(20 * max_radius, 2 * b_radius)), a has_hi flag for
    Rotne-Prager-Yamakawa hydrodynamic interactions, solvent parameters
    kT, viscosity, dielectric, vacuum_perm, debye_len, and OPGroupInfo
    records g0 (receptor) and g1 (ligand).
    """

    def __init__(
        self,
        b_radius: float,
        max_radius: float,
        has_hi: bool,
        kT: float,
        viscosity: float,
        dielectric: float,
        vacuum_perm: float,
        debye_len: float,
        g0: OPGroupInfo,
        g1: OPGroupInfo,
    ):
        if b_radius <= 0.0:
            raise ValueError(f"b_radius must be strictly positive, got {b_radius}")
        if g0.Dtrans < 0.0 or g1.Dtrans < 0.0 or (g0.Dtrans + g1.Dtrans) <= 0.0:
            raise ValueError(
                f"Dtrans values must be non-negative with positive sum, "
                f"got g0.Dtrans={g0.Dtrans}, g1.Dtrans={g1.Dtrans}"
            )
        if has_hi and (g0.Dtrans <= 0.0 or g1.Dtrans <= 0.0):
            raise ValueError(
                "Hydrodynamic interactions require strictly positive Dtrans for both molecules."
            )

        self.kT = kT
        self.viscosity = viscosity
        self.debye_len = debye_len
        self.bradius = b_radius
        self.qradius = max(20.0 * max_radius, 2.0 * b_radius)
        self.has_hi = has_hi
        eps_s = dielectric * vacuum_perm
        self.V_factor = g0.q * g1.q / (PI4 * eps_s * kT)
        self.D_factor = kT / viscosity
        self.D0_free = g0.Dtrans + g1.Dtrans
        # Hydrodynamic radii from Stokes-Einstein a = kT / (6 pi eta Dt).
        # A fixed molecule (Dtrans == 0 without HI) sets a = inf with 1/a = 0.
        self.a0 = self.D_factor / (PI6 * g0.Dtrans) if g0.Dtrans > 0.0 else math.inf
        self.a1 = self.D_factor / (PI6 * g1.Dtrans) if g1.Dtrans > 0.0 else math.inf
        if math.isfinite(self.a0) and math.isfinite(self.a1):
            self.a2 = 0.5 * (self.a0**2 + self.a1**2)
        else:
            self.a2 = 0.0
        self.Drot0 = g0.Drot
        self.Drot1 = g1.Drot
        # Return probability from b-surface to outer q-surface.
        rate_b = self._relative_rate(b_radius)
        rate_q = self._relative_rate(self.qradius)
        self.return_prob = rate_b / rate_q if rate_q > 0 else 0.0
        self.k_b = rate_b
        self.b_radius = b_radius
        # Radii switching to step_near_absorbing_surface near each boundary.
        gap = self.qradius - self.bradius
        min_margin = 1e-6 * gap
        max_inner = self.bradius + 0.45 * gap
        min_outer = self.qradius - 0.45 * gap
        raw_b_cover = self._cover(is_inner=True)
        raw_q_cover = self._cover(is_inner=False)
        self.bradius_cover = min(max(raw_b_cover, self.bradius + min_margin), max_inner)
        self.qradius_cover = max(min(raw_q_cover, self.qradius - min_margin), min_outer)
        assert self.bradius < self.bradius_cover < self.qradius_cover < self.qradius

    def _D_parallel(self, r: float) -> float:
        """Translational diffusivity along the centre-to-centre axis."""
        if self.has_hi:
            ainv = 1.0 / self.a0 + 1.0 / self.a1
            dpre = self.D_factor / PI6
            return dpre * (ainv - 3.0 / r + 2.0 * self.a2 / (r**3))
        return self.D0_free

    def _D_transverse(self, r: float) -> float:
        """Translational diffusivity perpendicular to the centre-to-centre axis."""
        if self.has_hi:
            ainv = 1.0 / self.a0 + 1.0 / self.a1
            rm1 = 1.0 / r
            return self.D_factor * (
                ainv / PI6 - 2.0 * (rm1 + (2.0 / 3.0) * self.a2 * rm1**3) / PI8
            )
        return self.D0_free

    def _relative_rate(self, b: float) -> float:
        """Evaluate the encounter-rate integral at boundary radius b."""
        L = self.debye_len

        def integrand(s: float) -> float:
            if s == 0.0:
                return 1.0 / self.D0_free
            r = 1.0 / s
            v = self.V_factor * math.exp(-1.0 / (s * L)) * s
            D = self._D_parallel(r)
            return math.exp(v) / D

        result = romberg_integrate(integrand, 0.0, 1.0 / b, tol=1e-8)
        return PI4 / result if result > 0 else 0.0

    def _ts_boundary(self, rad: float, is_inner: bool) -> float:
        """Find the separation at which the force stops being linear.

        This locates the distance from the boundary at radius rad beyond
        which the radial force can no longer be treated as linear, which
        marks where the step_near_absorbing_surface propagator applies.
        """
        curve_tol = 0.05
        L = self.debye_len

        def reldiff(r: float) -> float:
            if L / 1.0 > 0.1 * LARGE:
                return 2.0 * abs(r - rad) / r - curve_tol
            else:
                return (
                    (2 * L * L + 2 * r * L + r * r) * abs(r - rad) / (r * L * (L + r))
                ) - curve_tol

        # Bracket the root before bisecting.
        if is_inner:
            rb = 2.0 * rad
            while reldiff(rb) < 0.0:
                rb *= 2.0
            rlo, rhi = rad, rb
        else:
            rb = 0.5 * rad
            while reldiff(rb) < 0.0:
                rb *= 0.5
            rlo, rhi = rb, rad
        # Refine the bracket by bisection.
        fhi = reldiff(rhi)
        while rhi - rlo > 1e-6 * rad:
            rm = 0.5 * (rhi + rlo)
            fm = reldiff(rm)
            if is_inner:
                if fm * fhi < 0.0:
                    rlo = rm
                else:
                    rhi = rm
                    fhi = fm
            else:
                flo = reldiff(rlo)
                if fm * flo < 0.0:
                    rhi = rm
                else:
                    rlo = rm
        rb = 0.5 * (rlo + rhi)
        if is_inner:
            return min(rb, (1.0 + curve_tol) * self.bradius)
        else:
            return max(rb, (1.0 - curve_tol) * self.qradius)

    def _cover(self, is_inner: bool) -> float:
        """Find the radius at which we hand off to step_near_absorbing_surface.

        The handoff radius is chosen so that the probability of crossing the
        absorbing boundary within one step falls just below a small
        threshold. Setting is_inner selects the b-surface, otherwise the
        outer q-surface.
        """
        if is_inner:
            bndy = self._ts_boundary(self.bradius, True)
            F = self._radial_force(self.bradius)
            L = bndy - self.bradius
        else:
            bndy = self._ts_boundary(self.qradius, False)
            F = -self._radial_force(self.qradius)
            L = self.qradius - bndy

        def prob(x0: float) -> float:
            if x0 <= 0:
                return 1.0
            denom = math.erf((x0 * F + 1.0) / 2.0) + 1.0
            if denom == 0:
                return 1.0
            return math.erfc((L - x0 * x0 * F - x0) / (2.0 * x0)) / denom

        thresh = 0.01
        tol = 1e-6
        lo, hi = 0.0, L
        while hi - lo > tol * L:
            mid = 0.5 * (lo + hi)
            pmid = prob(mid)
            if pmid > thresh:
                hi = mid
            else:
                lo = mid
        x0 = 0.5 * (lo + hi)
        if is_inner:
            return self.bradius + x0
        else:
            return self.qradius - x0

    def _radial_force(self, r: float) -> float:
        """Radial component of the screened-Coulomb force in units of 1/A.

        Here r is the centre-to-centre separation in angstrom.
        """
        L = self.debye_len
        rm1 = 1.0 / r
        expf = math.exp(-r / L)
        V = self.V_factor * expf * rm1
        return V * (rm1 + 1.0 / L)

    # The main outer propagator.
    def new_state(
        self,
        pos: np.ndarray,  # The (3,) initial position on the b-sphere.
        ori: np.ndarray,  # The (4,) initial quaternion of the ligand.
        rng: np.random.Generator,
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Propagate the ligand through the outer region analytically.

        This returns the tuple (reached_b, new_pos, new_ori). When reached_b
        is True the trajectory has returned to the b-sphere and the
        simulation continues. When reached_b is False the trajectory has
        escaped through the q-sphere and is finished.
        """
        pos = pos.copy().astype(float)
        t = 0.0
        reached_b = False
        reached_q = False
        returned_from_q = False

        def do_when_near_q(
            survives: bool, r: float, new_x: float, p_ret: float | None = None
        ):
            nonlocal pos, t, reached_b, reached_q, returned_from_q
            if not survives:
                prob = self.return_prob if p_ret is None else p_ret
                if rng.random() < prob:
                    pos = (self.bradius / r) * pos
                    reached_b = True
                    reached_q = False
                    returned_from_q = True
                else:
                    pos = (self.qradius / r) * pos
                    reached_q = True
                t = LARGE
            else:
                new_r = self.qradius - new_x
                if new_r <= self.bradius:
                    pos = (self.bradius / r) * pos
                    reached_b = True
                else:
                    pos = (new_r / r) * pos
                reached_q = False

        # Entry boundary checks before entering the interior propagation loop.
        r_in = float(np.linalg.norm(pos))
        if r_in == 0.0:
            pos = self.bradius * _random_unit_vector(rng)
            reached_b = True
        elif r_in <= self.bradius:
            pos = (self.bradius / r_in) * pos
            reached_b = True
        elif r_in >= self.qradius:
            if r_in > self.qradius:
                rate_r = self._relative_rate(r_in)
                p_ret = self.k_b / rate_r if rate_r > 0.0 else 0.0
                p_ret = max(0.0, min(1.0, p_ret))
            else:
                p_ret = self.return_prob
            do_when_near_q(False, r_in, r_in, p_ret=p_ret)

        while not (reached_b or reached_q):
            r = float(np.linalg.norm(pos))
            if r == 0.0:
                pos = self.bradius * _random_unit_vector(rng)
                reached_b = True
                break
            if r > self.qradius:
                pos = pos * (self.qradius / r)
                r = self.qradius
            elif r < self.bradius:
                pos = pos * (self.bradius / r)
                r = self.bradius
            if r <= self.bradius:
                pos = (self.bradius / r) * pos
                reached_b = True
            elif r >= self.qradius:
                do_when_near_q(False, r, r)
            elif r < self.bradius_cover:
                # Near the b-sphere, propagate via Lamm-Schulten with curvature drift.
                x = r - self.bradius
                Fr0 = self._radial_force(r)
                D0 = self._D_parallel(r)
                Dt = self._D_transverse(r)
                F_eff = Fr0 + 2.0 * Dt / (r * D0)
                survives, new_x, delta_t = step_near_absorbing_surface(
                    rng, x, F_eff, D0
                )
                reached_b = not survives
                t += delta_t
                new_r = self.bradius + new_x
                pos = (new_r / r) * pos
            elif r > self.qradius_cover:
                # Near the q-sphere, propagate via Lamm-Schulten with curvature drift.
                x = self.qradius - r
                Fr0 = self._radial_force(r)
                D0 = self._D_parallel(r)
                Dt = self._D_transverse(r)
                F_eff = -(Fr0 + 2.0 * Dt / (r * D0))
                survives, new_x, delta_t = step_near_absorbing_surface(
                    rng, x, F_eff, D0
                )
                t += delta_t
                do_when_near_q(survives, r, new_x)
            else:
                # Middle region analytical drift plus stochastic diffusion step.
                Fr0 = self._radial_force(r)
                L = self.debye_len
                D0 = self._D_parallel(r)
                V = self.V_factor * math.exp(-r / L) / r
                Fr1 = -V / L**2 - 2.0 * Fr0 / r
                alpha = 0.01
                if self.has_hi:
                    rm1 = 1.0 / r
                    Di = (self.D_factor / PI6) * (-3.0 / r + 2.0 * self.a2 / (r**3))
                    D1 = -3.0 * Di * rm1 - self.D_factor * rm1**2 / PI
                    D2 = -4.0 * D1 * rm1 + self.D_factor * rm1**3 / PI
                    D3 = -5.0 * D2 * rm1 - 2.0 * self.D_factor * rm1**4 / PI
                    if abs(Fr0) > 0:
                        num = D1 + D0 * Fr0
                        den = (
                            D0 * D3
                            + (D1 + 2 * Fr0 * D0) * D2
                            + Fr0 * D1**2
                            + (3 * Fr1 + Fr0**2) * D0 * D1
                            + (0.0 + Fr0 * Fr1) * D0**2
                        )
                        dtf = (
                            alpha * abs(num / den)
                            if (abs(den) > 0 and abs(num) > 0)
                            else LARGE
                        )
                    else:
                        den = D0 * D3 + D1 * D2
                        dtf = alpha * abs(D1 / den) if abs(den) > 0 else LARGE
                else:
                    if abs(Fr0) > 0:
                        dtf = alpha / abs(D0 * Fr1) if abs(Fr1) > 0 else LARGE
                    else:
                        dtf = LARGE
                dt_edge = min(self.qradius - r, r - self.bradius) ** 2 / (18.0 * D0)
                dt = min(dt_edge, dtf)
                t += dt
                Dr = D0
                unit_r = pos / r
                pos += Dr * Fr0 * unit_r * dt
                sDrdt = math.sqrt(2.0 * Dr * dt)
                if self.has_hi:
                    # Anisotropic divergence-free RPY parallel and transverse components.
                    Dt = self._D_transverse(r)
                    sDtdt = math.sqrt(2.0 * Dt * dt)
                    ur = unit_r
                    x_, y_, z_ = ur
                    rho = math.sqrt(x_**2 + y_**2)
                    if rho == 0.0:
                        ut = np.array([1.0, 0.0, 0.0])
                        up = np.array([0.0, 1.0, 0.0])
                    else:
                        ut = np.array([z_ * x_ / rho, z_ * y_ / rho, -rho])
                        up = np.array([-y_ / rho, x_ / rho, 0.0])
                    pos += (sDrdt * rng.standard_normal()) * ur
                    pos += (sDtdt * rng.standard_normal()) * ut
                    pos += (sDtdt * rng.standard_normal()) * up
                else:
                    pos += sDrdt * rng.standard_normal(3)
                r_after = float(np.linalg.norm(pos))
                if r_after == 0.0:
                    pos = self.bradius * _random_unit_vector(rng)
                    reached_b = True
                elif r_after >= self.qradius:
                    do_when_near_q(False, r_after, r_after)
                elif r_after <= self.bradius:
                    pos = pos * (self.bradius / r_after)
                    reached_b = True
        if reached_b:
            tau0 = t * self.Drot0
            tau1 = t * self.Drot1
            dq0 = diffusional_rotation(rng, tau0)  # Quaternion (w, x, y, z).
            dq1 = diffusional_rotation(rng, tau1)

            def _quat_conj(q):
                return np.array([q[0], -q[1], -q[2], -q[3]])

            dq0_inv = _quat_conj(dq0)
            ori_new = quat_multiply(dq1, ori)
            ori_new = quat_multiply(dq0_inv, ori_new)
            ori_new = ori_new / np.linalg.norm(ori_new)
            pos = quat_rotate_vector(dq0_inv, pos)
            if returned_from_q and self.Drot0 == 0.0:
                pos = self.bradius * _random_unit_vector(rng)
        else:
            ori_new = ori.copy()
        return reached_b, pos, ori_new
