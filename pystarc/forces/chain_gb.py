"""Generalized Born (GB-OBC2) self-Born desolvation for chain BD.

Implements GB-OBC2 (Onufriev, Bashford, Case 2004) for chain BD. The
polarization free energy of a chain in implicit solvent decomposes as

    Delta G_pol = U_self + U_offdiag

    U_self    = -1/2 * cf * sum_i  q_i^2 / R_eff,i
    U_offdiag = -    cf * sum_{i<j} q_i q_j / f_GB(r_ij, R_eff,i, R_eff,j)

with the dielectric and Coulomb factor

    cf = (1/eps_in - 1/eps_out) * k_e * e^2 / kBT  [units: A]

and Still 1990 generalized distance

    f_GB(r, R_i, R_j) = sqrt(r^2 + R_i R_j exp(-r^2 / (4 R_i R_j))).

R_eff,i is computed via OBC2:

    R_eff,i^-1 = rho_tilde,i^-1 - rho_i^-1 * tanh(alpha * psi - beta * psi^2 + gamma * psi^3)

where rho_tilde,i = rho_i - offset, psi_i = rho_tilde,i * I_i, and I_i is
the HCT pairwise descreening integral over surrounding chain atoms.
Atoms that are solvent-exposed get R_eff approx rho_tilde; buried atoms
get R_eff that grows toward the cluster size.

Path B dispatch (handled by chain_full_gb_force in stage 1B):
  - When the chain has no COFFDROP empirical pair tables active, the
    chain has no other electrostatic treatment, so we apply full GB:
    vacuum Coulomb baseline + diagonal + off-diagonal.
  - When COFFDROP pair tables are active, those tables already encode
    pre-screened effective electrostatics. To avoid double-counting we
    apply only the diagonal self-energy term.

This module ships the forward energy functions only. Force functions
(analytical gradients with finite-difference verification) are added in
stage 1B so that R_eff can be sanity-checked before forces are committed.

References:
  Onufriev, A.; Bashford, D.; Case, D. A. Proteins 2004, 55, 383.
  Hawkins, G. D.; Cramer, C. J.; Truhlar, D. G. JPC 1996, 100, 19824.
  Still, W. C.; Tempczyk, A.; Hawley, R. C.; Hendrickson, T. JACS 1990,
    112, 6127.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

# OBC2 parameters (Onufriev 2004, set II)
DEFAULT_OBC_ALPHA = 1.0
DEFAULT_OBC_BETA = 0.8
DEFAULT_OBC_GAMMA = 4.85
DEFAULT_OBC_OFFSET = 0.09  # offset between vdW and intrinsic radius (A)
DEFAULT_HCT_SCALE = 0.85  # uniform Hawkins-Cramer-Truhlar scaling

# Coulomb constant: k_e * e^2 / kBT in units of A (at T = 300.15 K).
# Derivation:
#   k_e * e^2 (SI) = 8.9875e9 * (1.602176634e-19)^2 = 2.307e-28 J*m
#                  = 2.307e-18 J*A
#   kBT (T = 300.15 K) = 1.380649e-23 * 300.15 = 4.144e-21 J
#   ratio = 556.86 A * kBT / e^2
COULOMB_K_KBT_A = 556.86


def _hct_integrand(r, rho_tilde_i, rho_S_j):
    """HCT pairwise descreening integrand for atom i from atom j.

    Vector-safe. Inputs may be arrays broadcast-compatible with each
    other. Returns the integrand value at the given r.

    Three cases:
      case_engulf  : rho_tilde_i >= r + rho_S_j
        atom j is fully inside atom i's volume; no descreening; 0.
      case_overlap : rho_tilde_i >= r - rho_S_j (and not engulfed)
        j overlaps i's surface; L = rho_tilde_i, U = r + rho_S_j.
      case_outside : r > rho_tilde_i + rho_S_j
        j is fully outside i's volume; L = r - rho_S_j, U = r + rho_S_j.
    """
    case_engulf = rho_tilde_i >= r + rho_S_j
    case_overlap = (rho_tilde_i >= r - rho_S_j) & ~case_engulf

    L = np.where(case_engulf, 1.0, np.where(case_overlap, rho_tilde_i, r - rho_S_j))
    U = np.where(case_engulf, 1.0, r + rho_S_j)

    L = np.maximum(L, 1e-10)
    U = np.maximum(U, 1e-10)
    r_safe = np.maximum(r, 1e-10)

    inv_L = 1.0 / L
    inv_U = 1.0 / U
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U

    integrand = 0.5 * (
        inv_L
        - inv_U
        + (r_safe / 4.0) * (inv_U2 - inv_L2)
        + (1.0 / (2.0 * r_safe)) * np.log(L / U)
        + (rho_S_j * rho_S_j / (4.0 * r_safe)) * (inv_L2 - inv_U2)
    )

    integrand = np.where(case_engulf, 0.0, integrand)
    return integrand


def obc_effective_radii(
    positions,
    intrinsic_radii,
    obc_alpha=DEFAULT_OBC_ALPHA,
    obc_beta=DEFAULT_OBC_BETA,
    obc_gamma=DEFAULT_OBC_GAMMA,
    obc_offset=DEFAULT_OBC_OFFSET,
    hct_scale=DEFAULT_HCT_SCALE,
):
    """Compute OBC2 effective Born radii for chain atoms.

    R_eff,i^-1 = rho_tilde,i^-1 - rho_i^-1 * tanh(alpha*psi - beta*psi^2 + gamma*psi^3)

    with rho_tilde,i = rho_i - offset, psi_i = rho_tilde,i * I_i, and
    I_i = sum_{j != i} HCT_integrand(r_ij, rho_tilde,i, rho_S,j) with
    rho_S,j = hct_scale * rho_j.

    Parameters
    ----------
    positions       : (n_atoms, 3) array of atom positions [A]
    intrinsic_radii : (n_atoms,) array of intrinsic Born radii rho_i [A]
                      (van der Waals radii before offset subtraction)
    obc_alpha, obc_beta, obc_gamma : OBC2 scaling parameters
                                     Onufriev 2004 set II defaults: 1.0, 0.8, 4.85
    obc_offset      : offset between vdW and intrinsic radius [A]
    hct_scale       : Hawkins-Cramer-Truhlar scaling factor S [unitless]

    Returns
    -------
    R_eff : (n_atoms,) array of effective Born radii [A]
    """
    positions = np.asarray(positions, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)
    n = positions.shape[0]
    if intrinsic_radii.shape != (n,):
        raise ValueError(
            f"intrinsic_radii shape {intrinsic_radii.shape} must be ({n},) "
            f"to match positions shape {positions.shape}"
        )

    rho_tilde = intrinsic_radii - obc_offset
    if np.any(rho_tilde <= 0):
        raise ValueError(
            "rho_tilde = rho - offset must be positive for all atoms; "
            "received an intrinsic radius below the offset"
        )
    rho_S = hct_scale * intrinsic_radii

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    rho_tilde_i = rho_tilde[:, None]
    rho_S_j = rho_S[None, :]
    integrand = _hct_integrand(r, rho_tilde_i, rho_S_j)
    np.fill_diagonal(integrand, 0.0)

    I = np.sum(integrand, axis=1)
    psi = rho_tilde * I
    arg = obc_alpha * psi - obc_beta * psi * psi + obc_gamma * psi * psi * psi
    tanh_arg = np.tanh(arg)

    inv_R_eff = 1.0 / rho_tilde - tanh_arg / intrinsic_radii
    inv_R_eff = np.maximum(inv_R_eff, 1e-10)
    R_eff = 1.0 / inv_R_eff
    return R_eff


def gb_self_born_energy(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=78.5,
    R_eff=None,
    obc_kwargs=None,
):
    """Diagonal GB self-Born energy.

    U_self = -1/2 * cf * sum_i q_i^2 / R_eff,i

    cf = (1/eps_in - 1/eps_out) * COULOMB_K_KBT_A.

    Returns
    -------
    energy : scalar in kBT
    """
    charges = np.asarray(charges, dtype=float)
    if R_eff is None:
        kw = obc_kwargs or {}
        R_eff = obc_effective_radii(positions, intrinsic_radii, **kw)
    cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A
    energy = -0.5 * cf * float(np.sum(charges * charges / R_eff))
    return energy


def gb_offdiagonal_energy(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=78.5,
    R_eff=None,
    obc_kwargs=None,
):
    """Off-diagonal GB cross-term energy.

    U_offdiag = -cf * sum_{i<j} q_i q_j / f_GB(r_ij, R_eff,i, R_eff,j)

    f_GB = sqrt(r^2 + R_i R_j * exp(-r^2 / (4 R_i R_j)))   (Still 1990)
    cf   = (1/eps_in - 1/eps_out) * COULOMB_K_KBT_A.

    Returns
    -------
    energy : scalar in kBT
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]
    if R_eff is None:
        kw = obc_kwargs or {}
        R_eff = obc_effective_radii(positions, intrinsic_radii, **kw)

    cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    R_i = R_eff[:, None]
    R_j = R_eff[None, :]
    A = R_i * R_j
    A_safe = np.maximum(A, 1e-10)
    D = -r * r / (4.0 * A_safe)
    f_GB = np.sqrt(r * r + A * np.exp(D))
    f_GB_safe = np.maximum(f_GB, 1e-10)

    qq = charges[:, None] * charges[None, :]
    pair = qq / f_GB_safe

    upper = np.triu_indices(n, k=1)
    energy = -cf * float(np.sum(pair[upper]))
    return energy


def gb_vacuum_coulomb_energy(
    positions,
    charges,
    eps_in=1.0,
    exclude_pair_mask=None,
):
    """Vacuum Coulomb baseline between non-bonded chain pairs.

    U_Coulomb = (COULOMB_K_KBT_A / eps_in) * sum_{i<j, not excluded} q_i q_j / r_ij

    Pairs flagged in exclude_pair_mask (typically bonded 1-2, 1-3, 1-4)
    are skipped.

    Parameters
    ----------
    positions          : (n_atoms, 3) [A]
    charges            : (n_atoms,)   [e]
    eps_in             : interior dielectric (typically 1.0 for chain in vacuum-like solute)
    exclude_pair_mask  : (n_atoms, n_atoms) boolean; True at (i, j) means skip that pair.
                         Only the upper triangle is read.

    Returns
    -------
    energy : scalar in kBT
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r_safe = np.maximum(r, 1e-10)

    qq = charges[:, None] * charges[None, :]
    pair = qq / (eps_in * r_safe)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if exclude_pair_mask is not None:
        upper_mask = upper_mask & ~np.asarray(exclude_pair_mask, dtype=bool)

    energy = COULOMB_K_KBT_A * float(np.sum(pair[upper_mask]))
    return energy


# =============================================================================
#  Forces (Stage 1B): vacuum Coulomb and diagonal self-Born.
#
#  Diagonal self-Born force chain rule:
#
#      U_self = -1/2 cf sum_i q_i^2 / R_eff,i
#
#      dE/dR_eff,i = +1/2 cf q_i^2 / R_eff,i^2
#      dR_eff,i/dr_k = D_i * dI_i/dr_k       (OBC chain rule)
#
#  with
#      D_i = R_eff,i^2 * (1 - tanh^2(arg_i))/rho_i
#                       * (alpha - 2 beta psi_i + 3 gamma psi_i^2)
#                       * rho_tilde,i
#
#  Define eta_i = (dE/dR_eff,i) * D_i = 1/2 cf q_i^2 / R_eff,i^2 * D_i.
#  Then dE/dr_k = sum_i eta_i * dI_i/dr_k, where
#
#      dI_i/dr_k = sum_{j != i} K[i, j] * d(r_ij)/dr_k
#      K[i, j]   := d(integrand_ij)/dr_ij   (depends on rho_tilde,i and rho_S,j)
#
#  d(r_ij)/dr_k is the unit vector from j toward i when k=i, the unit vector
#  from i toward j when k=j, and zero otherwise.
#
#  Force F_k = -dE/dr_k assembles two pieces (both with K[k, k] = 0):
#
#      Term 1 (k acts as the subject):   -eta_k * sum_j K[k, j] * unit[k, j]
#      Term 2 (k acts as a screener):    +sum_i eta_i * K[i, k] * unit[i, k]
#
#  Both terms are computed via einsum over precomputed K and unit-vector tensors.
#  Translation invariance check: sum_k F_k = -sum_{i,j} eta_i K[i,j] unit[i,j]
#  + sum_{i,j} eta_i K[i,j] unit[i,j] = 0, as required.
# =============================================================================


def _hct_integrand_deriv(r, rho_tilde_i, rho_S_j):
    """Derivative of HCT integrand w.r.t. r at fixed rho_tilde_i, rho_S_j.

    Returns d(integrand)/dr (vector-safe, broadcast-compatible).

    Three branches matching _hct_integrand:
      case_engulf  : derivative is 0
      case_overlap : L = rho_tilde_i (constant in r), U = r + rho_S_j (dL/dr=0, dU/dr=1)
      case_outside : L = r - rho_S_j, U = r + rho_S_j (dL/dr=1, dU/dr=1)
    """
    case_engulf = rho_tilde_i >= r + rho_S_j
    case_overlap = (rho_tilde_i >= r - rho_S_j) & ~case_engulf

    L = np.where(case_engulf, 1.0, np.where(case_overlap, rho_tilde_i, r - rho_S_j))
    U = np.where(case_engulf, 1.0, r + rho_S_j)
    L = np.maximum(L, 1e-10)
    U = np.maximum(U, 1e-10)
    r_safe = np.maximum(r, 1e-10)

    dL_dr = np.where(case_overlap, 0.0, 1.0)  # 0 in overlap, 1 in outside
    dU_dr = np.ones_like(r)  # always 1 in non-engulf

    inv_L = 1.0 / L
    inv_U = 1.0 / U
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U
    inv_L3 = inv_L2 * inv_L
    inv_U3 = inv_U2 * inv_U
    inv_r2 = 1.0 / (r_safe * r_safe)

    # T1 = 1/L: dT1/dr = -1/L^2 * dL/dr
    dT1 = -inv_L2 * dL_dr
    # T2 = -1/U: dT2/dr = +1/U^2 * dU/dr
    dT2 = inv_U2 * dU_dr
    # T3 = (r/4)(1/U^2 - 1/L^2):
    #   dT3/dr = (1/4)(1/U^2 - 1/L^2) + (r/4)(-2/U^3 * dU/dr + 2/L^3 * dL/dr)
    dT3 = 0.25 * (inv_U2 - inv_L2) + (r_safe / 4.0) * (
        -2.0 * inv_U3 * dU_dr + 2.0 * inv_L3 * dL_dr
    )
    # T4 = (1/(2r)) * ln(L/U):
    #   dT4/dr = -ln(L/U)/(2 r^2) + (1/(2r))(dL/dr / L - dU/dr / U)
    log_LU = np.log(L / U)
    dT4 = -log_LU * 0.5 * inv_r2 + (1.0 / (2.0 * r_safe)) * (
        dL_dr * inv_L - dU_dr * inv_U
    )
    # T5 = (rho_S^2 / (4r))(1/L^2 - 1/U^2):
    #   dT5/dr = -rho_S^2/(4 r^2)*(1/L^2 - 1/U^2)
    #          + (rho_S^2/(4r))(-2/L^3 * dL/dr + 2/U^3 * dU/dr)
    rho_S_sq = rho_S_j * rho_S_j
    dT5 = -rho_S_sq * 0.25 * inv_r2 * (inv_L2 - inv_U2) + (
        rho_S_sq / (4.0 * r_safe)
    ) * (-2.0 * inv_L3 * dL_dr + 2.0 * inv_U3 * dU_dr)

    integrand_deriv = 0.5 * (dT1 + dT2 + dT3 + dT4 + dT5)
    integrand_deriv = np.where(case_engulf, 0.0, integrand_deriv)
    return integrand_deriv


def _obc_chain_rule_data(
    positions,
    intrinsic_radii,
    obc_alpha=DEFAULT_OBC_ALPHA,
    obc_beta=DEFAULT_OBC_BETA,
    obc_gamma=DEFAULT_OBC_GAMMA,
    obc_offset=DEFAULT_OBC_OFFSET,
    hct_scale=DEFAULT_HCT_SCALE,
):
    """Precompute everything needed for OBC force chain rule.

    Returns dict with keys:
      R_eff         : (n,)       effective Born radii
      D             : (n,)       D_i = dR_eff_i / dI_i  (OBC chain rule scalar)
      K             : (n, n)     K[i, j] = d(integrand_ij) / dr_ij; diagonal zero
      r             : (n, n)     pairwise distances
      unit          : (n, n, 3)  unit[i, j, :] = (r_i - r_j) / r_ij; diagonal zero
      rho_tilde     : (n,)
    """
    positions = np.asarray(positions, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)
    n = positions.shape[0]

    rho_tilde = intrinsic_radii - obc_offset
    if np.any(rho_tilde <= 0):
        raise ValueError("rho_tilde must be positive for all atoms")
    rho_S = hct_scale * intrinsic_radii

    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    r = np.linalg.norm(diff, axis=2)  # (n, n)
    r_safe = np.where(np.eye(n, dtype=bool), 1.0, np.maximum(r, 1e-10))
    unit = np.zeros_like(diff)
    off_diag = ~np.eye(n, dtype=bool)
    unit[off_diag] = diff[off_diag] / r_safe[off_diag, None]

    rho_tilde_i = rho_tilde[:, None]
    rho_S_j = rho_S[None, :]
    integrand = _hct_integrand(r, rho_tilde_i, rho_S_j)
    K = _hct_integrand_deriv(r, rho_tilde_i, rho_S_j)
    np.fill_diagonal(integrand, 0.0)
    np.fill_diagonal(K, 0.0)

    I = np.sum(integrand, axis=1)
    psi = rho_tilde * I
    arg = obc_alpha * psi - obc_beta * psi * psi + obc_gamma * psi * psi * psi
    tanh_arg = np.tanh(arg)
    sech2 = 1.0 - tanh_arg * tanh_arg

    inv_R_eff = 1.0 / rho_tilde - tanh_arg / intrinsic_radii
    inv_R_eff = np.maximum(inv_R_eff, 1e-10)
    R_eff = 1.0 / inv_R_eff

    # D_i = dR_eff_i / dI_i
    #     = R_eff_i^2 * (1 - tanh^2(arg)) / rho_i * (alpha - 2 beta psi + 3 gamma psi^2) * rho_tilde
    darg_dpsi = obc_alpha - 2.0 * obc_beta * psi + 3.0 * obc_gamma * psi * psi
    D = R_eff * R_eff * sech2 / intrinsic_radii * darg_dpsi * rho_tilde

    return {
        "R_eff": R_eff,
        "D": D,
        "K": K,
        "r": r,
        "unit": unit,
        "rho_tilde": rho_tilde,
    }


def chain_vacuum_coulomb_force(
    positions,
    charges,
    eps_in=1.0,
    exclude_pair_mask=None,
):
    """Vacuum Coulomb forces and energy between non-bonded chain pairs.

    F_k = (COULOMB_K / eps_in) * q_k * sum_{m != k, m allowed} q_m * (r_k - r_m) / r_km^3
    U   = (COULOMB_K / eps_in) * sum_{i<j, allowed} q_i q_j / r_ij

    Returns
    -------
    forces : (n, 3) array of forces in kBT/A on each chain atom
    energy : scalar in kBT
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]

    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    r = np.linalg.norm(diff, axis=2)
    r_safe = np.where(np.eye(n, dtype=bool), 1.0, np.maximum(r, 1e-10))
    inv_r = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / r_safe)
    inv_r3 = inv_r * inv_r * inv_r

    qq = charges[:, None] * charges[None, :]

    # Allowed-pair mask (symmetric). Default: all off-diagonal allowed.
    allowed = ~np.eye(n, dtype=bool)
    if exclude_pair_mask is not None:
        excl = np.asarray(exclude_pair_mask, dtype=bool)
        excl_sym = excl | excl.T
        allowed = allowed & ~excl_sym

    K_factor = COULOMB_K_KBT_A / eps_in
    pair_e = qq * inv_r * allowed
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1) & allowed
    energy = K_factor * float(np.sum(qq[upper_mask] * inv_r[upper_mask]))

    # F_k vec: K_factor * q_k * sum_{m allowed} q_m (r_k - r_m)/r_km^3
    #        = K_factor * q_k * sum_m q_m * inv_r3[k, m] * (r_k - r_m)
    weight = qq * inv_r3 * allowed  # (n, n)
    forces = K_factor * np.einsum("km,kmd->kd", weight, diff)
    return forces, energy


def chain_self_born_diagonal_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=78.5,
    obc_kwargs=None,
):
    """Diagonal GB self-Born forces and energy.

    F_k = -dE_self/dr_k = -sum_i eta_i * dI_i/dr_k, with
      eta_i           = 0.5 * cf * q_i^2 / R_eff_i^2 * D_i
      dI_k/dr_k       = sum_j K[k, j] * unit[k, j, :]   (k as subject atom)
      dI_i/dr_k (i!=k) = -K[i, k] * unit[i, k, :]        (k as screener for i)

    Substituting and using K[k, k] = 0 to extend sums over all atoms:

      F_k = -eta_k * sum_j K[k, j] * unit[k, j, :]
            + sum_i eta_i * K[i, k] * unit[i, k, :]

    Returns
    -------
    forces : (n, 3) [kBT/A]
    energy : scalar [kBT]
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)

    kw = obc_kwargs or {}
    data = _obc_chain_rule_data(positions, intrinsic_radii, **kw)
    R_eff = data["R_eff"]
    D = data["D"]
    K = data["K"]
    unit = data["unit"]

    cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A
    energy = -0.5 * cf * float(np.sum(charges * charges / R_eff))

    # eta_i = 0.5 * cf * q_i^2 / R_eff_i^2 * D_i
    eta = 0.5 * cf * (charges * charges) / (R_eff * R_eff) * D  # (n,)

    # Term 1: F1[k, :] = -eta_k * sum_j K[k, j] * unit[k, j, :]
    T1 = -eta[:, None] * np.einsum("kj,kjd->kd", K, unit)
    # Term 2: F2[k, :] = +sum_i eta_i * K[i, k] * unit[i, k, :]
    T2 = np.einsum("i,ik,ikd->kd", eta, K, unit)
    forces = T1 + T2
    return forces, energy


def _finite_difference_force(positions, energy_fn, h=1e-5):
    """Central-difference force approximation: F = -dE/dr.

    energy_fn must accept a position array of shape (n, 3) and return a scalar.
    Returns array of shape (n, 3). Used as ground-truth for verifying analytical
    force implementations.
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    F = np.zeros((n, 3))
    for i in range(n):
        for d in range(3):
            pp = positions.copy()
            pm = positions.copy()
            pp[i, d] += h
            pm[i, d] -= h
            E_p = energy_fn(pp)
            E_m = energy_fn(pm)
            F[i, d] = -(E_p - E_m) / (2.0 * h)
    return F


# =============================================================================
#  Stage 1B-beta: off-diagonal GB cross-term force + top-level dispatcher.
#
#  Off-diagonal force has two contributions:
#    Direct (r_ij in f_GB):
#      F_k^direct = -cf q_k sum_{m != k} q_m / f_GB,km^2 * df_GB/dr_km * unit[k, m]
#    Indirect (R_eff,m for all m, propagated via OBC chain rule):
#      eta_m^offdiag = D_m * cf * q_m * sum_{j != m} q_j / f_GB,mj^2 * df_GB,mj/dR_m
#      F_k^indirect  = -eta_k^offdiag * sum_j K[k, j] * unit[k, j]
#                     + sum_i eta_i^offdiag * K[i, k] * unit[i, k]
#
#  The indirect piece reuses the OBC chain-rule structure of the diagonal force
#  with eta^offdiag in place of eta^diag.
# =============================================================================


def chain_offdiagonal_gb_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=78.5,
    obc_kwargs=None,
):
    """Off-diagonal GB cross-term forces and energy.

    Parameters
    ----------
    positions       : (n, 3) [A]
    charges         : (n,)   [e]
    intrinsic_radii : (n,)   [A]
    eps_in, eps_out : interior / exterior dielectrics
    obc_kwargs      : optional OBC2 parameter overrides

    Returns
    -------
    forces : (n, 3) [kBT/A]
    energy : scalar [kBT]
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)
    n = positions.shape[0]

    kw = obc_kwargs or {}
    data = _obc_chain_rule_data(positions, intrinsic_radii, **kw)
    R_eff = data["R_eff"]
    D_chain = data["D"]
    K = data["K"]
    r = data["r"]
    unit = data["unit"]

    cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A
    eye_mask = np.eye(n, dtype=bool)

    # f_GB and its r and R_i derivatives over all (i, j)
    R_i_mat = R_eff[:, None]
    R_j_mat = R_eff[None, :]
    A = R_i_mat * R_j_mat
    A_safe = np.where(eye_mask, 1.0, np.maximum(A, 1e-10))
    r_safe = np.where(eye_mask, 1.0, np.maximum(r, 1e-10))
    D_grid = -r_safe * r_safe / (4.0 * A_safe)
    expD = np.exp(D_grid)
    f_GB_sq = r_safe * r_safe + A_safe * expD
    f_GB = np.sqrt(f_GB_sq)
    inv_f_GB = np.where(eye_mask, 0.0, 1.0 / f_GB)
    inv_f_GB2 = inv_f_GB * inv_f_GB

    qq = charges[:, None] * charges[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -cf * float(np.sum(qq[upper] * inv_f_GB[upper]))

    # Direct: df_GB/dr = r/f_GB * (1 - expD/4); zero on diagonal
    df_GB_dr = r_safe * (1.0 - 0.25 * expD) * inv_f_GB
    df_GB_dr[eye_mask] = 0.0
    weight_direct = qq * inv_f_GB2 * df_GB_dr
    weight_direct[eye_mask] = 0.0
    F_direct = -cf * np.einsum("km,kmd->kd", weight_direct, unit)

    # Indirect via R_eff (per-atom chain rule):
    # df_GB,mj/dR_m = R_j * expD * (1 - D) / (2 f_GB)  (treating first R-arg as variable)
    df_GB_dR_first = R_j_mat * expD * (1.0 - D_grid) * inv_f_GB * 0.5
    df_GB_dR_first[eye_mask] = 0.0
    # eta_m^offdiag = D_m * cf * q_m * sum_{j != m} q_j / f_GB,mj^2 * df_GB,mj/dR_m
    inner = inv_f_GB2 * df_GB_dR_first  # (n, n)
    sum_per_m = np.sum(inner * charges[None, :], axis=1)  # (n,)
    eta_off = D_chain * cf * charges * sum_per_m  # (n,)

    # Apply OBC chain rule with eta_off (same structure as diagonal force)
    T1 = -eta_off[:, None] * np.einsum("kj,kjd->kd", K, unit)
    T2 = np.einsum("i,ik,ikd->kd", eta_off, K, unit)
    F_indirect = T1 + T2

    forces = F_direct + F_indirect
    return forces, energy


def chain_full_gb_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=78.5,
    coffdrop_active=False,
    exclude_pair_mask=None,
    obc_kwargs=None,
):
    """Top-level GB force composer with Path B dispatch.

    Path B logic:
      coffdrop_active=False -> full GB: diagonal + off-diagonal + vacuum Coulomb.
      coffdrop_active=True  -> diagonal only (avoids double-counting against
                               COFFDROP's pre-screened pair potentials).

    Parameters
    ----------
    positions, charges, intrinsic_radii : standard chain arrays
    eps_in, eps_out                     : dielectrics
    coffdrop_active                     : if True, restrict to diagonal-only
    exclude_pair_mask                   : (n, n) bool, pairs to skip in vacuum Coulomb
                                          (typically bonded 1-2, 1-3, 1-4 exclusions)
    obc_kwargs                          : OBC2 parameter overrides

    Returns
    -------
    forces   : (n, 3) [kBT/A]
    energies : dict with keys 'self', 'offdiag', 'coulomb', 'total' [kBT]
               (offdiag and coulomb are 0 when coffdrop_active=True)
    """
    F_diag, E_self = chain_self_born_diagonal_force(
        positions, charges, intrinsic_radii, eps_in, eps_out, obc_kwargs
    )
    forces = F_diag
    energies = {"self": E_self, "offdiag": 0.0, "coulomb": 0.0}

    if not coffdrop_active:
        F_off, E_off = chain_offdiagonal_gb_force(
            positions, charges, intrinsic_radii, eps_in, eps_out, obc_kwargs
        )
        F_C, E_C = chain_vacuum_coulomb_force(
            positions, charges, eps_in, exclude_pair_mask
        )
        forces = forces + F_off + F_C
        energies["offdiag"] = E_off
        energies["coulomb"] = E_C

    energies["total"] = energies["self"] + energies["offdiag"] + energies["coulomb"]
    return forces, energies
