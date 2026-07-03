"""
Yukawa multipole far-field expansion.

When a ligand atom lies outside the APBS electrostatic grid, we need an
analytical expression for the receptor's electrostatic potential. The
receptor is a collection of N partial charges {q_i} at positions {r_i}. At
large distances r >> molecular_size, the potential can be expanded in
multipole moments of increasing angular complexity.

The monopole term (ℓ=0) is the net charge Q = Σ q_i, with potential

  V₀(r) = Q/(4πε r) × exp(-r/λ).

Here Q is the total charge in e, r is the distance from the receptor
centroid in Å, ε is the solvent permittivity in internal units, and λ is
the Debye screening length in Å. This is the screened Coulomb (Yukawa)
potential. The exponential screening exp(-r/λ) arises from mobile ions in
solution (Debye screening). At 150 mM NaCl, λ ≈ 7.86 Å, so the potential
effectively vanishes beyond about 25 Å. For charged proteins (for example
trypsin, Q = +6e) the monopole dominates and supplies more than 99% of the
far-field force at r > 3a.

The dipole term (ℓ=1) uses the dipole moment p = Σ q_i × r_i, with potential

  V₁(r) = (p·r̂)/(4πε r²) × (1 + r/λ) × exp(-r/λ).

Here p is the dipole moment vector and r̂ is the unit vector along r. For
molecules with zero net charge (for example β-cyclodextrin, Q ≈ 0) the
dipole is the leading non-zero term. It decays as 1/r² rather than 1/r, so
it becomes important only at intermediate distances.

The quadrupole term (ℓ=2) uses the tensor Q_ij = ½ Σ q_k(3 r_ki r_kj - r² δ_ij),
with potential

  V₂(r) = (r̂ᵀ Q r̂)/(4πε r³) × (1 + r/λ + r²/(3λ²)) × exp(-r/λ).

Here Q is the traceless symmetric quadrupole tensor. The quadrupole
captures the non-spherical charge distribution. It decays as 1/r³ and is
usually negligible for biological systems but is included for completeness.

The total far-field potential is V = V₀ + V₁ + V₂. Forces are
F = -q_ligand × ∇V, computed by central finite difference.

The internal units are as follows. Positions are in Å relative to the
receptor centroid, charges in elementary charges (e), potentials in kBT,
and forces in kBT/Å. The vacuum permittivity is ε₀ = 0.000142 e²/(kBT·Å)
in these internal units, and the solvent permittivity is
ε = ε_solvent × ε₀ = 78 × 0.000142 e²/(kBT·Å).

The multipole expansion is accurate to better than 1% for r greater than
twice the molecular radius, and is meaningless for r below the molecular
radius (where the APBS grid must be used instead). The transition zone is
handled by the grid force engine, which uses the APBS grid when available
and falls back to the multipole with a three-spacing safety margin.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
import math

from pystarc.global_defs.constants import VACUUM_PERMITTIVITY_KBT


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
        Build the monopole, dipole, and quadrupole moments of the molecule.

        The argument positions is an (N, 3) array of atom positions in Å,
        centered at the molecular centroid. The argument charges is an (N,)
        array of atom partial charges in e. The argument debye_length is the
        Debye screening length in Å, and sdie is the solvent dielectric
        constant.
        """
        self.debye = debye_length
        eps0 = VACUUM_PERMITTIVITY_KBT  # vacuum permittivity in e²/(kBT·Å)
        self.eps = sdie * eps0
        self.four_pi_eps = 4.0 * math.pi * self.eps
        # Monopole: the total charge Q.
        self.Q = float(np.sum(charges))
        # Dipole moment p = Σ q_i × r_i.
        self.dipole = np.sum(charges[:, None] * positions, axis=0)  # shape (3,)
        # Quadrupole tensor Q_ij = Σ q_i (3 r_i r_j - r² δ_ij), a traceless
        # symmetric tensor.
        r2 = np.sum(positions**2, axis=1)  # squared distances, shape (N,)
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
        self.quadrupole *= 0.5  # apply the convention ½ Σ q(3rr - r²I)
        # Screened-kernel isotropic (trace) term. For the Yukawa kernel
        # ∇²G = G/λ² is nonzero (unlike Coulomb, where ∇²(1/r) = 0), so the
        # primitive second-moment trace tr(M) = Σ q_i |r_i|² contributes an
        # isotropic term that acts as an effective monopole of charge
        # tr(M)/(6λ²). Keeping only the traceless quadrupole is correct for the
        # Coulomb kernel but drops this term for a screened kernel; it is the
        # leading far-field error for net-neutral quadrupolar hosts such as
        # β-cyclodextrin.
        self.trace_moment = float(np.sum(charges * r2))  # tr(M) = Σ q_i |r_i|²
        # Moment magnitudes, kept for diagnostics.
        self.dipole_mag = float(np.linalg.norm(self.dipole))
        self.quad_mag = float(np.linalg.norm(self.quadrupole))

    def potential(self, r_vec: np.ndarray) -> float:
        """
        Compute the screened potential at position r_vec (in Å, measured from
        the centroid) and return V in kBT units.
        """
        r = float(np.linalg.norm(r_vec))
        if r < 1e-10:
            return 0.0
        r_hat = r_vec / r
        lam = self.debye
        exp_r = math.exp(-r / lam)
        # Monopole contribution.
        V = self.Q / (self.four_pi_eps * r) * exp_r
        # Screened isotropic trace term: acts as an effective monopole of charge
        # tr(M)/(6λ²). It is spherically symmetric, so it has the same 1/r
        # screened form as the monopole and carries no angular dependence.
        V += (self.trace_moment / (6.0 * lam**2)) / (self.four_pi_eps * r) * exp_r
        # Dipole contribution.
        if self.dipole_mag > 1e-9:
            p_dot_r = float(np.dot(self.dipole, r_hat))
            V += p_dot_r / (self.four_pi_eps * r**2) * (1.0 + r / lam) * exp_r
        # Quadrupole contribution.
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
        Compute the screened force at position r_vec (in Å, measured from the
        centroid) and return F = -∇V in kBT/Å units as a (3,) array. The
        gradient is evaluated by central difference for robustness.
        """
        h = 0.001  # finite-difference step in Å
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
        # Report which multipole term dominates.
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
    Evaluate the multipole force for a batch of trajectories on the GPU.

    The argument r_vecs_gpu is an (N_traj, 3) cupy array of
    centroid-to-centroid vectors. The function returns an (N_traj, 3) force
    array in kBT/Å.
    """
    r_mag = cp.linalg.norm(r_vecs_gpu, axis=1, keepdims=True)  # shape (N, 1)
    r_mag = cp.maximum(r_mag, 1e-10)
    r_hat = r_vecs_gpu / r_mag  # unit vectors, shape (N, 3)
    r_s = r_mag[:, 0]  # distances, shape (N,)
    lam = debye
    exp_r = cp.exp(-r_s / lam)
    # Monopole force F = -dV/dr × r̂. Differentiating V₀ = Q/(4πε r) exp(-r/λ)
    # gives F₀ = Q/(4πε) × (1/r² + 1/(rλ)) × exp(-r/λ) × r̂.
    F_mono = (Q / four_pi_eps) * (1.0 / r_s**2 + 1.0 / (r_s * lam)) * exp_r
    # Dipole force: project the dipole onto r̂, then take the radial gradient.
    dipole_gpu = cp.asarray(dipole, dtype=cp.float64)
    p_dot_r = cp.sum(r_hat * dipole_gpu[None, :], axis=1)  # shape (N,)
    # The dipole force is F_dip ≈ -∇[p·r̂/(4πε r²) × (1+r/λ) × exp(-r/λ)], whose
    # leading radial term is approximately
    # 2 p·r̂/(4πε r³) × (1+r/λ) × exp(-r/λ) × r̂.
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
    # Quadrupole force, keeping the leading radial term.
    Q_gpu = cp.asarray(quadrupole, dtype=cp.float64)
    quad_mag = float(cp.linalg.norm(Q_gpu))
    if quad_mag > 1e-9:
        rQr = cp.sum(r_hat * (r_hat @ Q_gpu), axis=1)  # the scalar r̂ᵀ Q r̂, shape (N,)
        factor_quad = (
            rQr
            / (four_pi_eps * r_s**4)
            * (3.0 + 3.0 * r_s / lam + r_s**2 / lam**2 + r_s**3 / (3.0 * lam**3))
            * exp_r
        )
    else:
        factor_quad = cp.zeros_like(r_s)
    # Combine the radial magnitudes and project back onto r̂.
    F_total = (F_mono + factor_dip + factor_quad)[:, None] * r_hat
    return F_total
