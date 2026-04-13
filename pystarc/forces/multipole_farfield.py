"""
Yukawa multipole far-field expansion
=====================================

Background
-------------------
When a ligand atom is outside the APBS electrostatic grid, we need an
analytical expression for the receptor's electrostatic potential.  The
receptor is a collection of N partial charges {q_i} at positions {r_i}.
At large distances r >> molecular_size, the potential can be expanded
in multipole moments of increasing angular complexity:

  1. MONOPOLE (ℓ=0): The net charge Q = Σ q_i
     Potential:  V₀(r) = Q/(4πε r) × exp(-r/λ)
     This is the screened Coulomb (Yukawa) potential.  The exponential
     screening exp(-r/λ) arises from mobile ions in solution (Debye
     screening).  At 150 mM NaCl, λ ≈ 7.86 Å, so the potential
     effectively vanishes beyond ~25 Å.
     For charged proteins (e.g. trypsin, Q=+6e), the monopole
     dominates and provides >99% of the far-field force at r > 3a.

  2. DIPOLE (ℓ=1): The dipole moment p = Σ q_i × r_i
     Potential:  V₁(r) = (p·r̂)/(4πε r²) × (1 + r/λ) × exp(-r/λ)
     For molecules with zero net charge (e.g. β-cyclodextrin, Q≈0),
     the dipole is the leading non-zero term.  It decays as 1/r²
     (vs 1/r for monopole), so it becomes important only at
     intermediate distances.

  3. QUADRUPOLE (ℓ=2): The tensor Q_ij = ½ Σ q_k(3r_ki r_kj - r²δ_ij)
     Potential:  V₂(r) = (r̂ᵀ Q r̂)/(4πε r³) × (1 + r/λ + r²/(3λ²)) × exp(-r/λ)
     The quadrupole captures the non-spherical charge distribution.
     It decays as 1/r³ and is usually negligible for biological
     systems but included for completeness.

The total far-field potential is V = V₀ + V₁ + V₂.
Forces are F = -q_ligand × ∇V, computed by central finite difference.

Units
-----
  - Positions: Å (relative to receptor centroid)
  - Charges: elementary charges (e)
  - Potentials: kBT
  - Forces: kBT/Å
  - ε₀ = 0.000142 e²/(kBT·Å)  (vacuum permittivity in internal units)
  - ε = ε_solvent × ε₀ = 78 × 0.000142 e²/(kBT·Å)

When is the multipole accurate?
  - r > 2 × molecular_radius: excellent (< 1% error)
  - r < molecular_radius: meaningless (use APBS grid)
  - The transition zone is handled by the grid force engine, which
    uses the APBS grid when available and falls back to multipole
    with a 3-spacing safety margin.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
import math


class MultipoleExpansion:
    """Precomputed multipole moments for a molecule."""

    def __init__(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        debye_length: float,
        sdie: float = 78.0,
    ):
        """
        Parameters
        ----------
        positions : (N, 3) array
            Atom positions in Å (centered at molecular centroid).
        charges : (N,) array
            Atom partial charges in e.
        debye_length : float
            Debye screening length in Å.
        sdie : float
            Solvent dielectric constant.
        """
        self.debye = debye_length
        eps0 = 0.000142  # e²/(kBT·Å)
        self.eps = sdie * eps0
        self.four_pi_eps = 4.0 * math.pi * self.eps
        # Monopole: Q_total
        self.Q = float(np.sum(charges))
        # Dipole: p = Σ q_i × r_i
        self.dipole = np.sum(charges[:, None] * positions, axis=0)  # (3,)
        # Quadrupole: Q_ij = Σ q_i × (3 r_i r_j - r² δ_ij)
        # Traceless symmetric tensor
        r2 = np.sum(positions**2, axis=1)  # (N,)
        self.quadrupole = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                self.quadrupole[a, b] = np.sum(
                    charges
                    * (
                        3.0 * positions[:, a] * positions[:, b]
                        - (r2 if a == b else 0.0)
                    )
                )
        self.quadrupole *= 0.5  # convention: ½ Σ q(3rr - r²I)
        # Magnitudes for diagnostics
        self.dipole_mag = float(np.linalg.norm(self.dipole))
        self.quad_mag = float(np.linalg.norm(self.quadrupole))

    def potential(self, r_vec: np.ndarray) -> float:
        """
        Compute screened potential at position r_vec (Å) from centroid.
        Returns V in kBT units.
        """
        r = float(np.linalg.norm(r_vec))
        if r < 1e-10:
            return 0.0
        r_hat = r_vec / r
        lam = self.debye
        exp_r = math.exp(-r / lam)
        # Monopole
        V = self.Q / (self.four_pi_eps * r) * exp_r
        # Dipole
        if self.dipole_mag > 1e-9:
            p_dot_r = float(np.dot(self.dipole, r_hat))
            V += p_dot_r / (self.four_pi_eps * r**2) * (1.0 + r / lam) * exp_r
        # Quadrupole
        if self.quad_mag > 1e-9:
            rQr = float(r_hat @ self.quadrupole @ r_hat)
            V += (
                rQr
                / (self.four_pi_eps * r**3)
                * (1.0 + r / lam + r**2 / (3.0 * lam**2))
                * exp_r
            )
        return V

    def force(self, r_vec: np.ndarray) -> np.ndarray:
        """
        Compute screened force at position r_vec (Å) from centroid.
        Returns F = -∇V in kBT/Å units, as (3,) array.
        Computed by central difference for robustness.
        """
        h = 0.001  # Å
        F = np.zeros(3)
        for i in range(3):
            r_plus = r_vec.copy()
            r_plus[i] += h
            r_minus = r_vec.copy()
            r_minus[i] -= h
            F[i] = -(self.potential(r_plus) - self.potential(r_minus)) / (2.0 * h)
        return F

    def summary(self) -> str:
        lines = [
            f"  Multipole expansion:",
            f"    Monopole Q    = {self.Q:+.4f} e",
            f"    Dipole |p|    = {self.dipole_mag:.4f} e·Å  "
            f"p = [{self.dipole[0]:+.3f}, {self.dipole[1]:+.3f}, {self.dipole[2]:+.3f}]",
            f"    Quadrupole |Q|= {self.quad_mag:.4f} e·Å²",
            f"    Debye length  = {self.debye:.3f} Å",
        ]
        # Which terms dominate?
        if abs(self.Q) > 0.1:
            lines.append(f"    -> Monopole dominates (|Q| >> 0)")
        elif self.dipole_mag > 0.5:
            lines.append(f"    -> Dipole dominates (Q≈0, |p|={self.dipole_mag:.1f})")
        else:
            lines.append(f"    -> Quadrupole dominant or uncharged")
        return "\n".join(lines)


def compute_multipole_gpu(
    positions_gpu, charges_gpu, r_vecs_gpu, Q, dipole, quadrupole, debye, four_pi_eps
):
    """
    GPU batch multipole force computation.
    Parameters
    ----------
    r_vecs_gpu : (N_traj, 3) cupy array - centroid-to-centroid vectors
    Returns: (N_traj, 3) force array in kBT/Å
    """
    r_mag = cp.linalg.norm(r_vecs_gpu, axis=1, keepdims=True)  # (N, 1)
    r_mag = cp.maximum(r_mag, 1e-10)
    r_hat = r_vecs_gpu / r_mag  # (N, 3)
    r_s = r_mag[:, 0]  # (N,)
    lam = debye
    exp_r = cp.exp(-r_s / lam)
    # Monopole force: F = -dV/dr × r̂
    # V_0 = Q/(4πε r) exp(-r/λ)
    # F_0 = Q/(4πε) × (1/r² + 1/(rλ)) × exp(-r/λ) × r̂
    F_mono = (Q / four_pi_eps) * (1.0 / r_s**2 + 1.0 / (r_s * lam)) * exp_r
    # Dipole force (scalar projection onto r̂, then gradient)
    dipole_gpu = cp.asarray(dipole, dtype=cp.float64)
    p_dot_r = cp.sum(r_hat * dipole_gpu[None, :], axis=1)  # (N,)
    # F_dip ≈ -∇[p·r̂/(4πε r²) × (1+r/λ) × exp(-r/λ)]
    # Leading term: ~2p·r̂/(4πε r³) × (1+r/λ) × exp(-r/λ) × r̂
    dip_mag = float(cp.linalg.norm(dipole_gpu))
    if dip_mag > 1e-9:
        factor_dip = (
            p_dot_r
            / (four_pi_eps * r_s**3)
            * (2.0 + 2.0 * r_s / lam + r_s**2 / lam**2)
            * exp_r
        )
    else:
        factor_dip = cp.zeros_like(r_s)
    # Quadrupole force (leading radial term)
    Q_gpu = cp.asarray(quadrupole, dtype=cp.float64)
    quad_mag = float(cp.linalg.norm(Q_gpu))
    if quad_mag > 1e-9:
        rQr = cp.sum(r_hat * (r_hat @ Q_gpu), axis=1)  # (N,) = r̂ᵀ Q r̂
        factor_quad = (
            rQr
            / (four_pi_eps * r_s**4)
            * (3.0 + 3.0 * r_s / lam + r_s**2 / lam**2 + r_s**3 / (3.0 * lam**3))
            * exp_r
        )
    else:
        factor_quad = cp.zeros_like(r_s)
    # Total radial force × r̂
    F_total = (F_mono + factor_dip + factor_quad)[:, None] * r_hat
    return F_total
