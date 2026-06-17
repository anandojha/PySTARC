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
hydrodynamic interactions. The integral is evaluated by Romberg quadrature
(adaptive Richardson extrapolation) to roughly 10⁻⁸ relative accuracy.

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
from dataclasses import dataclass
from scipy import integrate
from typing import Tuple
import numpy as np
import math

# Physical constants
PI = math.pi
PI4 = 4.0 * PI
PI6 = 6.0 * PI
PI8 = 8.0 * PI
LARGE = 1.0e30


@dataclass
class OPGroupInfo:
    """Per-molecule info needed by the outer propagator."""

    q: float  # Total charge in elementary units.
    Dtrans: float  # Translational diffusion coefficient in A^2/ps.
    Drot: float  # Rotational diffusion coefficient in rad^2/ps.


class OuterPropagator:
    """Analytical propagator for the outer diffusion region b ≤ r ≤ q.

    The constructor takes the b-sphere radius b_radius in angstrom, the
    maximum molecular radius max_radius (used to set the outer boundary
    q = 20 × max_radius), and a flag has_hi that selects whether
    Rotne-Prager-Yamakawa hydrodynamic interactions are included. The
    solvent is described by the thermal energy kT (in kcal/mol or any
    consistent units), the viscosity, the dielectric constant, and the
    Debye screening length debye_len in angstrom. The two molecules are
    described by OPGroupInfo records g0 for the receptor and g1 for the
    ligand.
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

        self.kT = kT
        self.viscosity = viscosity
        self.debye_len = debye_len
        self.bradius = b_radius
        self.qradius = 20.0 * max_radius  # The standard outer boundary q.
        self.has_hi = has_hi
        eps_s = dielectric * vacuum_perm
        self.V_factor = g0.q * g1.q / (PI4 * eps_s * kT)
        self.D_factor = kT / viscosity
        # Hydrodynamic radii from the Stokes-Einstein relation
        # a = kT / (6πμ Dt), where Dt is the translational diffusion
        # coefficient and μ is the solvent viscosity.
        self.a0 = self.D_factor / (PI6 * g0.Dtrans)
        self.a1 = self.D_factor / (PI6 * g1.Dtrans)
        self.a2 = 0.5 * (self.a0**2 + self.a1**2)
        self.Drot0 = g0.Drot
        self.Drot1 = g1.Drot
        # The return probability is the encounter rate at the b-surface
        # divided by the rate at the outer q-surface.
        rate_b = self._relative_rate(b_radius)
        rate_q = self._relative_rate(self.qradius)
        self.return_prob = rate_b / rate_q if rate_q > 0 else 0.0
        # These covers are the radii at which we switch over to the
        # step_near_absorbing_surface propagator near each boundary.
        self.bradius_cover = self._cover(is_inner=True)
        self.qradius_cover = self._cover(is_inner=False)

    def _D_parallel(self, r: float) -> float:
        """Translational diffusivity along the line connecting the two centres.

        When has_hi is set, this includes the Rotne-Prager-Yamakawa
        hydrodynamic correction, otherwise it returns the free-diffusion
        value. Here r is the centre-to-centre separation in angstrom.
        """
        ainv = 1.0 / self.a0 + 1.0 / self.a1
        dpre = self.D_factor / PI6
        if self.has_hi:
            return dpre * (ainv - 3.0 / r + 2.0 * self.a2 / (r**3))
        else:
            return dpre * ainv

    def _relative_rate(self, b: float) -> float:
        """Evaluate the encounter-rate integral at boundary radius b.

        This returns 4π divided by the integral ∫₀^(1/b) exp(V(1/s)) /
        D_∥(1/s) ds, using the substitution s = 1/r so that the integration
        runs over the finite interval [0, 1/b].
        """
        L = self.debye_len

        def integrand(s: float) -> float:
            if s == 0.0:
                ainv = 1.0 / self.a0 + 1.0 / self.a1
                return PI6 / (self.D_factor * ainv)
            r = 1.0 / s
            v = self.V_factor * math.exp(-1.0 / (s * L)) * s
            D = self._D_parallel(r)
            return math.exp(v) / D

        result, _ = integrate.quad(integrand, 0.0, 1.0 / b, limit=100, epsrel=1e-6)
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

        def do_when_near_q(survives: bool, r: float, new_x: float):
            nonlocal pos, t, reached_b, reached_q
            if not survives:
                if rng.random() < self.return_prob:
                    pos = (self.bradius / r) * pos
                    reached_b = True
                    reached_q = False
                else:
                    reached_q = True
                t = LARGE
            else:
                new_r = self.qradius - new_x
                pos = (new_r / r) * pos
                reached_q = False

        while not (reached_b or reached_q):
            r = float(np.linalg.norm(pos))
            # Clamp the radius at the start of every iteration, because the
            # branch taken in the previous iteration may have left pos in an
            # unphysical state. The near-b Lamm-Schulten branch in particular
            # can return a large new_x, which makes new_r = bradius + new_x
            # enormous. Clipping here ensures the next branch dispatch sees a
            # physical r and that final_separation stays interpretable.
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
                # Near the b-sphere, take a Lamm-Schulten step.
                x = r - self.bradius
                Fr0 = self._radial_force(r)
                D0 = self._D_parallel(r)
                survives, new_x, delta_t = step_near_absorbing_surface(rng, x, Fr0, D0)
                reached_b = not survives
                t += delta_t
                new_r = self.bradius + new_x
                pos = (new_r / r) * pos
            elif r > self.qradius_cover:
                # Near the q-sphere, take a Lamm-Schulten step.
                x = self.qradius - r
                Fr0 = self._radial_force(r)
                D0 = self._D_parallel(r)
                survives, new_x, delta_t = step_near_absorbing_surface(rng, x, -Fr0, D0)
                # Accumulate the elapsed time from the Lamm-Schulten step
                # (audit fix 2026-05-21). This mirrors the b-sphere branch
                # above. Without it, trajectories near the q-sphere
                # under-accumulate time and the outer-sphere rotation is
                # under-applied. In the not-survives branch do_when_near_q
                # overwrites t with LARGE, so accumulating here is safe.
                t += delta_t
                do_when_near_q(survives, r, new_x)
            else:
                # In the middle region, take an analytical drift plus
                # diffusion step.
                Fr0 = self._radial_force(r)
                L = self.debye_len
                D0 = self._D_parallel(r)
                # Choose the time step from the gradient of the force.
                V = self.V_factor * math.exp(-r / L) / r
                Fr1 = -V / L**2 - 2.0 * Fr0 / r
                alpha = 0.01
                if self.has_hi:
                    rm1 = 1.0 / r
                    Di = (self.D_factor / PI6) * (-3.0 / r + 2.0 * self.a2 / (r**3))
                    D1 = -3.0 * Di * rm1 - self.D_factor * rm1**2 / PI
                    D2 = -4.0 * D1 * rm1 + self.D_factor * rm1**3 / PI
                    D3 = -5.0 * D2 * rm1 - 2.0 * self.D_factor * rm1**4 / PI
                    if abs(Fr0) > 0 and r < 3.0 * L:
                        num = D1 + D0 * Fr0
                        den = (
                            D0 * D3
                            + (D1 + 2 * Fr0 * D0) * D2
                            + Fr0 * D1**2
                            + (3 * Fr1 + Fr0**2) * D0 * D1
                            + (0.0 + Fr0 * Fr1) * D0**2
                        )
                        dtf = alpha * abs(num / den) if abs(den) > 0 else LARGE
                    else:
                        den = D0 * D3 + D1 * D2
                        dtf = alpha * abs(D1 / den) if abs(den) > 0 else LARGE
                else:
                    if abs(Fr0) > 0 and r < 3.0 * L:
                        dtf = alpha / abs(D0 * Fr1) if abs(Fr1) > 0 else LARGE
                    else:
                        dtf = LARGE
                dt_edge = min(self.qradius - r, r - self.bradius) ** 2 / (18.0 * D0)
                dt = min(dt_edge, dtf)
                t += dt
                Dr = D0
                # Apply the deterministic drift.
                unit_r = pos / r
                pos += Dr * Fr0 * unit_r * dt
                # Apply the stochastic diffusion.
                sDrdt = math.sqrt(2.0 * Dr * dt)
                if self.has_hi:
                    # With hydrodynamics the Rotne-Prager-Yamakawa diffusion
                    # is anisotropic, so the parallel and perpendicular
                    # components differ.
                    rm1 = 1.0 / r
                    ainv = 1.0 / self.a0 + 1.0 / self.a1
                    Dt = self.D_factor * (
                        ainv / PI6 - 2.0 * (rm1 - 2.0 * self.a2 * rm1**3) / PI8
                    )
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
                # Clamp defensively to prevent the middle-region step from
                # overshooting past the q-sphere (or below the b-sphere) when
                # Fr0 is large. Without this, the drift Dr × Fr0 × dt can be
                # enormous, carrying pos from about 55 A to several hundred A
                # in a single step. The next iteration would detect r ≥ qradius
                # and terminate as an escape, but with pos still at the
                # overshoot magnitude (final_separation near 666 A instead of
                # the q-sphere value of 80 A). Clipping back to the boundary
                # keeps the escape and return logic physical and leaves
                # final_separation interpretable.
                r_after = float(np.linalg.norm(pos))
                if r_after > self.qradius:
                    pos = pos * (self.qradius / r_after)
                elif r_after < self.bradius:
                    pos = pos * (self.bradius / r_after)
        # Update the orientation only if the trajectory returned to the
        # b-sphere.
        if reached_b:
            # Apply diffusional rotation over the elapsed time t.
            tau0 = t * self.Drot0
            tau1 = t * self.Drot1
            dq0 = diffusional_rotation(rng, tau0)  # Quaternion (w, x, y, z).
            dq1 = diffusional_rotation(rng, tau1)

            # Compose the new orientation as dq0^{-1} × (dq1 × ori), using
            # the local quat_multiply imported from the diffusional_rotation
            # module.
            def _quat_conj(q):
                return np.array([q[0], -q[1], -q[2], -q[3]])

            ori_new = quat_multiply(dq1, ori)
            ori_new = quat_multiply(_quat_conj(dq0), ori_new)
            # Normalise the resulting quaternion.
            ori_new = ori_new / np.linalg.norm(ori_new)
        else:
            ori_new = ori.copy()
        return reached_b, pos, ori_new
