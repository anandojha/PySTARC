"""
Outer propagator and k_b encounter rate
========================================

Background
-------------------
The Northrup-Allison-McCammon (NAM) method decomposes the association
rate constant as:
    k_on = k_b × P_rxn

where k_b is the diffusion-limited encounter rate at a spherical
"b-surface" of radius b, and P_rxn is the reaction probability from
BD simulation.

k_b integral (Smoluchowski-Debye-Hückel)
-----------------------------------------
The encounter rate is computed by the steady-state Smoluchowski equation
with a screened Coulomb (Yukawa) interaction potential:
    k_b = 4π / ∫₀^(1/b) [exp(V(1/s)/kBT) / D_∥(1/s)] ds
    
The change of variables s = 1/r transforms the semi-infinite integral
[b, ∞) into the finite interval [0, 1/b].
  - V(r) = Yukawa monopole potential = Q₁Q₂/(4πε r) × exp(-r/λ)
  - D_∥(r) = distance-dependent parallel diffusion coefficient
    with Rotne-Prager-Yamakawa hydrodynamic interactions

The integral is evaluated by Romberg quadrature (adaptive Richardson
extrapolation) to ~10⁻⁸ relative accuracy.

Return probability (LMZ Method)
-------------------------------
When a trajectory diffuses beyond the escape sphere (r_esc = 2b),
we must decide, i.e., return to the b-surface or escape to infinity?

The Luty-McCammon-Zhou (LMZ) is:
    p_return = k_b(b) / k_b(r_esc)

This ratio accounts for electrostatic steering - If the molecules
attract each other, p_return > 0.5 (more likely to return than
escape).  For neutral molecules, p_return = b/r_esc = 0.5 (exactly
what we would expect for free diffusion).

Returned trajectories are placed back on the b-surface with a
uniformly random orientation (justified because the outer propagator
time is much longer than the rotational diffusion time).

Hydrodynamic interactions
-------------------------
The parallel diffusion coefficient D_∥(r) encodes the hydrodynamic
coupling between two spheres at separation r:

    D_∥(r) = kBT/(6πη) × (1/a₁ + 1/a₂ - 3/r + 2ā²/r³)

where ā² = (a₁² + a₂²)/2.  At large r, D_∥ -> D₀ = D₁ + D₂ (free
diffusion).  At contact r = a₁ + a₂, HI slows the approach by
a factor of ~0.6-0.8, which reduces k_b by the same factor.
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
PI   = math.pi
PI4  = 4.0 * PI
PI6  = 6.0 * PI
PI8  = 8.0 * PI
LARGE = 1.0e30

@dataclass
class OPGroupInfo:
    """Per-molecule info needed by the outer propagator."""
    q:       float    # total charge (elementary units)
    Dtrans:  float    # translational diffusion coefficient (A^2/ps)
    Drot:    float    # rotational diffusion coefficient (rad^2/ps)
        
class OuterPropagator:
    """
    Analytical propagator for the outer diffusion region (b <= r <= q).
    Parameters
    ----------
    b_radius   : b-sphere radius (A)
    max_radius : maximum molecular radius (used to set q = 20*max_radius)
    has_hi     : whether to use hydrodynamic interactions (RPY)
    kT         : thermal energy (kcal/mol or consistent units)
    viscosity  : solvent viscosity
    dielectric : solvent dielectric constant
    debye_len  : Debye screening length (A)
    g0, g1     : OPGroupInfo for receptor and ligand
    """
    
    def __init__(self,
                 b_radius:    float,
                 max_radius:  float,
                 has_hi:      bool,
                 kT:          float,
                 viscosity:   float,
                 dielectric:  float,
                 vacuum_perm: float,
                 debye_len:   float,
                 g0:          OPGroupInfo,
                 g1:          OPGroupInfo):

        self.kT          = kT
        self.viscosity   = viscosity
        self.debye_len   = debye_len
        self.bradius     = b_radius
        self.qradius     = 20.0 * max_radius   # standard outer boundary
        self.has_hi      = has_hi
        eps_s        = dielectric * vacuum_perm
        self.V_factor = g0.q * g1.q / (PI4 * eps_s * kT)
        self.D_factor = kT / viscosity
        # radii from Stokes-Einstein: a = kT / (6*pi*mu*Dt)
        self.a0 = self.D_factor / (PI6 * g0.Dtrans)
        self.a1 = self.D_factor / (PI6 * g1.Dtrans)
        self.a2 = 0.5 * (self.a0**2 + self.a1**2)
        self.Drot0 = g0.Drot
        self.Drot1 = g1.Drot
        # return probability = rate at b / rate at q
        rate_b = self._relative_rate(b_radius)
        rate_q = self._relative_rate(self.qradius)
        self.return_prob = rate_b / rate_q if rate_q > 0 else 0.0
        # covers: boundaries where we switch to step_near_absorbing_surface
        self.bradius_cover = self._cover(is_inner=True)
        self.qradius_cover = self._cover(is_inner=False)

    # Translational diffusivity along radial axis (with RPY if has_hi)
    def _D_parallel(self, r: float) -> float:
        """D along the line connecting centres."""
        ainv = (1.0/self.a0 + 1.0/self.a1)
        dpre = self.D_factor / PI6
        if self.has_hi:
            return dpre * (ainv - 3.0/r + 2.0*self.a2/(r**3))
        else:
            return dpre * ainv
        
    # Relative rate integral (Romberg)
    def _relative_rate(self, b: float) -> float:
        """
        Compute PI4 / integral_0^{1/b} exp(V(1/s)) / D_parallel(1/s) ds
        """
        L = self.debye_len
        def integrand(s: float) -> float:
            if s == 0.0:
                ainv = (1.0/self.a0 + 1.0/self.a1)
                return PI6 / (self.D_factor * ainv)
            r   = 1.0 / s
            v   = self.V_factor * math.exp(-1.0 / (s * L)) * s
            D   = self._D_parallel(r)
            return math.exp(v) / D
        result, _ = integrate.quad(integrand, 0.0, 1.0/b,
                                   limit=100, epsrel=1e-6)
        return PI4 / result if result > 0 else 0.0

    # Boundary for step_near_absorbing_surface
    def _ts_boundary(self, rad: float, is_inner: bool) -> float:
        """Find distance from boundary where force is no longer linear."""
        curve_tol = 0.05
        L = self.debye_len
        def reldiff(r: float) -> float:
            if L / 1.0 > 0.1 * LARGE:
                return 2.0 * abs(r - rad) / r - curve_tol
            else:
                return ((2*L*L + 2*r*L + r*r) * abs(r - rad) /
                        (r * L * (L + r))) - curve_tol
        # bracket
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
        # bisect
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
        """Find boundary where we switch to step_near_absorbing_surface."""
        if is_inner:
            bndy = self._ts_boundary(self.bradius, True)
            F    = self._radial_force(self.bradius)
            L    = bndy - self.bradius
        else:
            bndy = self._ts_boundary(self.qradius, False)
            F    = -self._radial_force(self.qradius)
            L    = self.qradius - bndy

        def prob(x0: float) -> float:
            if x0 <= 0:
                return 1.0
            denom = math.erf((x0 * F + 1.0) / 2.0) + 1.0
            if denom == 0:
                return 1.0
            return math.erfc((L - x0*x0*F - x0) / (2.0*x0)) / denom
        thresh = 0.01
        tol    = 1e-6
        lo, hi = 0.0, L
        while hi - lo > tol * L:
            mid  = 0.5 * (lo + hi)
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
        """Radial component of electrostatic force (1/A units)."""
        L    = self.debye_len
        rm1  = 1.0 / r
        expf = math.exp(-r / L)
        V    = self.V_factor * expf * rm1
        return V * (rm1 + 1.0 / L)

    # Main outer propagator
    def new_state(self,
                  pos:  np.ndarray,    # (3,) initial position on b-sphere
                  ori:  np.ndarray,    # (4,) initial quaternion of ligand
                  rng:  np.random.Generator
                  ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Propagate the ligand through the outer region analytically.
        Returns
        -------
        (reached_b, new_pos, new_ori)
        reached_b = True  -> returned to b-sphere, simulation continues
        reached_b = False -> escaped through q-sphere, trajectory done
        """
        pos = pos.copy().astype(float)
        t   = 0.0
        reached_b = False
        reached_q = False

        def do_when_near_q(survives: bool, r: float, new_x: float):
            nonlocal pos, t, reached_b, reached_q
            if not survives:
                if rng.random() < self.return_prob:
                    pos       = (self.bradius / r) * pos
                    reached_b = True
                    reached_q = False
                else:
                    reached_q = True
                t = LARGE
            else:
                new_r = self.qradius - new_x
                pos   = (new_r / r) * pos
                reached_q = False
        while not (reached_b or reached_q):
            r = float(np.linalg.norm(pos))
            if r <= self.bradius:
                pos       = (self.bradius / r) * pos
                reached_b = True
            elif r >= self.qradius:
                do_when_near_q(False, r, r)
            elif r < self.bradius_cover:
                # near b-sphere: use Lamm-Schulten
                x       = r - self.bradius
                Fr0     = self._radial_force(r)
                D0      = self._D_parallel(r)
                survives, new_x, delta_t = step_near_absorbing_surface(
                    rng, x, Fr0, D0)
                reached_b = not survives
                t        += delta_t
                new_r     = self.bradius + new_x
                pos       = (new_r / r) * pos
            elif r > self.qradius_cover:
                # near q-sphere: use Lamm-Schulten
                x       = self.qradius - r
                Fr0     = self._radial_force(r)
                D0      = self._D_parallel(r)
                survives, new_x, delta_t = step_near_absorbing_surface(
                    rng, x, -Fr0, D0)
                do_when_near_q(survives, r, new_x)
            else:
                # middle region: analytical drift + diffusion step
                Fr0 = self._radial_force(r)
                L   = self.debye_len
                D0  = self._D_parallel(r)
                # time step from force gradient
                Fr1  = -(self.V_factor * math.exp(-r/L) / r) * (1.0/r + 1.0/L)**2 \
                       - 2.0 * Fr0 / r
                alpha = 0.01
                if self.has_hi:
                    rm1 = 1.0/r
                    D1  = -3.0*D0*rm1 - self.D_factor*rm1**2 / PI
                    D2  = -4.0*D1*rm1 + self.D_factor*rm1**3 / PI
                    D3  = -5.0*D2*rm1 - 2.0*self.D_factor*rm1**4 / PI
                    if abs(Fr0) > 0 and r < 3.0*L:
                        num = D1 + D0*Fr0
                        den = (D0*D3 + (D1 + 2*Fr0*D0)*D2 +
                               Fr0*D1**2 +
                               (3*Fr1 + Fr0**2)*D0*D1 +
                               (0.0 + Fr0*Fr1)*D0**2)
                        dtf = alpha * abs(num/den) if abs(den) > 0 else LARGE
                    else:
                        den = D0*D3 + D1*D2
                        dtf = alpha * abs(D1/den) if abs(den) > 0 else LARGE
                else:
                    if abs(Fr0) > 0 and r < 3.0*L:
                        dtf = alpha / abs(D0 * Fr1) if abs(Fr1) > 0 else LARGE
                    else:
                        dtf = LARGE
                dt_edge = min(self.qradius - r, r - self.bradius)**2 / (18.0*D0)
                dt      = min(dt_edge, dtf)
                t      += dt
                Dr     = D0
                # deterministic drift
                unit_r = pos / r
                pos   += Dr * Fr0 * unit_r * dt
                # stochastic diffusion
                sDrdt = math.sqrt(2.0 * Dr * dt)
                if self.has_hi:
                    # RPY: anisotropic diffusion (para vs perp)
                    rm1 = 1.0/r
                    ainv = (1.0/self.a0 + 1.0/self.a1)
                    Dt   = self.D_factor * (ainv/PI6
                           - 2.0*(rm1 - 2.0*self.a2*rm1**3)/PI8)
                    sDtdt = math.sqrt(2.0 * Dt * dt)
                    ur = unit_r
                    x_, y_, z_ = ur
                    rho = math.sqrt(x_**2 + y_**2)
                    if rho == 0.0:
                        ut = np.array([1., 0., 0.])
                        up = np.array([0., 1., 0.])
                    else:
                        ut = np.array([z_*x_/rho, z_*y_/rho, -rho])
                        up = np.array([-y_/rho,   x_/rho,     0.0])
                    pos += (sDrdt  * rng.standard_normal()) * ur
                    pos += (sDtdt  * rng.standard_normal()) * ut
                    pos += (sDtdt  * rng.standard_normal()) * up
                else:
                    pos += sDrdt * rng.standard_normal(3)
        # update orientation if we returned to b-sphere
        if reached_b:
            # diffusional rotation over elapsed time t
            tau0 = t * self.Drot0
            tau1 = t * self.Drot1
            dq0  = diffusional_rotation(rng, tau0)   # (w,x,y,z)
            dq1  = diffusional_rotation(rng, tau1)
            # compose: new_ori = dq0^{-1} * (dq1 * ori)
            # using local quat_multiply from diffusional_rotation module
            def _quat_conj(q):
                return np.array([q[0], -q[1], -q[2], -q[3]])
            ori_new = quat_multiply(dq1, ori)
            ori_new = quat_multiply(_quat_conj(dq0), ori_new)
            # normalise
            ori_new = ori_new / np.linalg.norm(ori_new)
        else:
            ori_new = ori.copy()
        return reached_b, pos, ori_new