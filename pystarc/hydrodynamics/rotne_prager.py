"""
Rotne-Prager-Yamakawa hydrodynamic interaction tensor
=====================================================

Background
-------------------
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

Impact on k_b
--------------
HI reduces k_b by 10-30% for typical protein systems because the
effective diffusion near contact is slower than free diffusion.
For the thrombin-thrombomodulin system:
  - Without HI: k_b ≈ 40 ų/ps
  - With HI:    k_b ≈ 35.7 ų/ps (11% reduction)
"""

from __future__ import annotations
from pystarc.global_defs.constants import ETA_WATER, KB_SI, T_DEFAULT, ANG_TO_M, PS_TO_S
import numpy as np
import math

def stokes_translational_diffusion(radius_ang: float,
                                   eta: float = ETA_WATER,
                                   T:   float = T_DEFAULT) -> float:
    """
    Stokes-Einstein translational diffusion.  D_t = kBT / (6 π η r)
    Returns D_t in Å²/ps.
    """
    r_m   = radius_ang * ANG_TO_M
    D_m2s = KB_SI * T / (6.0 * math.pi * eta * r_m)
    return D_m2s / (ANG_TO_M**2) * PS_TO_S

def stokes_rotational_diffusion(radius_ang: float,
                                 eta: float = ETA_WATER,
                                 T:   float = T_DEFAULT) -> float:
    """
    Stokes rotational diffusion.  D_r = kBT / (8 π η r³)
    Returns D_r in rad²/ps.
    """
    r_m   = radius_ang * ANG_TO_M
    D_r_s = KB_SI * T / (8.0 * math.pi * eta * r_m**3)
    return D_r_s * PS_TO_S

def rpy_offdiagonal(r_vec: np.ndarray,
                    a: float, b: float,
                    D_a: float, D_b: float) -> np.ndarray:
    """
    Rotne-Prager-Yamakawa off-diagonal translational mobility tensor M_12.
    Exact translation of the rotne_prager.hh get_trans_components()
    + trans_matrix() - three regimes (Zuk et al. J. Fluid Mech. 2014):
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
    rhat  = r_vec / r
    outer = np.outer(rhat, rhat)
    I3    = np.eye(3)
    PI    = math.pi
    PI6   = 6.0 * PI
    PI8   = 8.0 * PI
    a2 = a * a
    b2 = b * b
    r2 = r * r
    r3 = r2 * r
    if r > a + b:
        # Far field 
        den   = PI8 * r
        a2ob2 = a2 + b2
        tt_I  = (1.0 + a2ob2 / (3.0 * r2)) / den
        tt_uu = (1.0 - a2ob2 / r2)         / den
    elif r > abs(a - b):
        # Partial overlap
        ab   = a * b
        am2  = (a - b) ** 2          # (a-b)^2
        den  = 6.0 * 32.0 * PI * ab * r3
        tt_I  = (16.0*r3*(a + b) - (am2 + 3.0*r2)**2) / den
        tt_uu = 3.0 * (am2 - r2)**2                   / den
    else:
        # One sphere inside the other - self-mobility of larger sphere
        a_max = max(a, b)
        tt_I  = 1.0 / (PI6 * a_max)
        tt_uu = 0.0
    # Scale from viscosity units to diffusion units:
    # In BD we need D = kBT * mobility, and D_a = kBT/(6π·η·a).
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

    def __init__(self,
                 D_trans1: float, D_rot1: float,
                 D_trans2: float, D_rot2: float,
                 radius1:  float = 0.0,
                 radius2:  float = 0.0,
                 use_rpy:  bool  = True):
        self.D_trans1 = D_trans1
        self.D_rot1   = D_rot1
        self.D_trans2 = D_trans2
        self.D_rot2   = D_rot2
        self.radius1  = radius1
        self.radius2  = radius2
        self.use_rpy  = use_rpy

    @classmethod
    def from_radii(cls,
                   radius1: float,
                   radius2: float,
                   eta:     float = ETA_WATER,
                   T:       float = T_DEFAULT,
                   use_rpy: bool  = True) -> "MobilityTensor":
        """Build MobilityTensor from hydrodynamic radii."""
        return cls(
            D_trans1=stokes_translational_diffusion(radius1, eta, T),
            D_rot1  =stokes_rotational_diffusion(radius1, eta, T),
            D_trans2=stokes_translational_diffusion(radius2, eta, T),
            D_rot2  =stokes_rotational_diffusion(radius2, eta, T),
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
        M12 = rpy_offdiagonal(r_vec, self.radius1, self.radius2,
                               self.D_trans1, self.D_trans2)
        # Scalar coupling: effective D_rel = D_t1 + D_t2 - (2/3) tr(M_12)
        # Factor 2 because both molecules feel the coupling symmetrically
        D_coupling = (2.0 / 3.0) * np.trace(M12)
        return max(D0 - D_coupling, 1e-12)   # never negative

    def relative_rotational_diffusion(self) -> float:
        """D_r_rel = D_r1 + D_r2 (RPY rotational coupling is negligible)."""
        return self.D_rot1 + self.D_rot2

    def __repr__(self) -> str:
        return (f"MobilityTensor(Dt1={self.D_trans1:.3e}, Dr1={self.D_rot1:.3e}, "
                f"Dt2={self.D_trans2:.3e}, Dr2={self.D_rot2:.3e}, "
                f"r1={self.radius1:.1f}A, r2={self.radius2:.1f}A, RPY={self.use_rpy})")