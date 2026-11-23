"""Generalized Born (GB-OBC2) self-Born desolvation for chain Brownian dynamics.

This module implements the GB-OBC2 model of Onufriev, Bashford, and Case (2004)
for chain Brownian dynamics. The polarization free energy of a chain in implicit
solvent splits into a diagonal self-energy and an off-diagonal cross term,

    Delta G_pol = U_self + U_offdiag

where the two pieces are

    U_self    = -1/2 cf sum_i q_i^2 / R_eff,i
    U_offdiag = -cf sum_{i<j} q_i q_j / f_GB(r_ij, R_eff,i, R_eff,j).

Here q_i is the partial charge on atom i in units of the elementary charge,
R_eff,i is its effective Born radius in angstrom, and r_ij is the interatomic
distance. The dielectric prefactor combines the interior and exterior dielectric
constants with the Coulomb constant,

    cf = (1/eps_in - 1/eps_out) k_e e^2 / kBT,

which carries units of angstrom. The generalized distance f_GB is the smooth
interpolation of Still and coworkers (1990),

    f_GB(r, R_i, R_j) = sqrt(r^2 + R_i R_j exp(-r^2 / (4 R_i R_j))).

Each effective radius R_eff,i follows from the OBC2 rescaling,

    1/R_eff,i = 1/rho_tilde,i - (1/rho_i) tanh(alpha psi - beta psi^2 + gamma psi^3),

where rho_tilde,i = rho_i - offset is the intrinsic radius after the offset is
removed, psi_i = rho_tilde,i I_i, and I_i is the Hawkins-Cramer-Truhlar pairwise
descreening integral summed over the surrounding chain atoms. Solvent-exposed
atoms end up with R_eff close to rho_tilde, while buried atoms acquire a larger
R_eff that grows toward the size of the surrounding cluster.

The top-level composer chain_full_gb_force selects between two treatments. When
the chain carries no active COFFDROP empirical pair tables, the chain has no
other electrostatic treatment, so the full GB model is applied, namely the vacuum
Coulomb baseline together with the diagonal and off-diagonal terms. When COFFDROP
pair tables are active, those tables already encode pre-screened effective
electrostatics, so only the diagonal self-energy term is applied to avoid
double-counting.

Both the forward energy functions and their analytical force counterparts are
provided. The force routines are verified against central finite differences so
that R_eff can be sanity-checked before the forces are relied upon.

References. Onufriev, A.; Bashford, D.; Case, D. A. Proteins 2004, 55, 383.
Hawkins, G. D.; Cramer, C. J.; Truhlar, D. G. JPC 1996, 100, 19824.
Still, W. C.; Tempczyk, A.; Hawley, R. C.; Hendrickson, T. JACS 1990, 112, 6127.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

# OBC2 parameters (Onufriev 2004, set II)
DEFAULT_OBC_ALPHA = 1.0
DEFAULT_OBC_BETA = 0.8
DEFAULT_OBC_GAMMA = 4.85
DEFAULT_OBC_OFFSET = (
    0.09  # Offset between the van der Waals and intrinsic radius, in angstrom.
)
DEFAULT_HCT_SCALE = 0.85  # Uniform Hawkins-Cramer-Truhlar scaling factor.

# Coulomb constant k_e e^2 / kBT expressed in angstrom at T = 298.15 K. The value is
# recomputed from first principles as
#
#     COULOMB_K_KBT_A = k_e e^2 * 1e10 / (kB T)
#
# with k_e = 8.9875517873681764e9 N m^2 / C^2 (CODATA, 1/(4 pi eps0)),
# e = 1.602176634e-19 C, kB = 1.380649e-23 J/K, and T = 298.15 K (T_DEFAULT, the
# value used by the rest of the engine). The factor 1e10 converts the meters in
# k_e e^2 to angstrom. Numerically k_e e^2 = 2.30708e-18 J*A and
# kB T = 4.11640e-21 J, so the ratio is 560.46 A in units of kBT per squared
# elementary charge. This constant carries only the Coulomb prefactor and temperature;
# the water dielectric enters separately through the cf prefactor below and is taken
# as the single consistent value WATER_DIELECTRIC = 78.5.
COULOMB_K_KBT_A = 560.46

# Single consistent water (exterior solvent) dielectric used throughout this module
# as the default eps_out. Defining it once avoids the value drifting between nearby
# literals such as 78.5 and 78.54 across the various energy and force routines.
WATER_DIELECTRIC = 78.5


def _hct_integrand(r, rho_tilde_i, rho_S_j):
    """Hawkins-Cramer-Truhlar pairwise descreening integrand for atom i due to atom j.

    Supports broadcast-compatible NumPy arrays and scalars at separation r.
    When rho_tilde_i >= r + rho_S_j, atom j sits completely inside atom i and
    the integrand vanishes. When r + rho_tilde_i < rho_S_j, atom i lies inside
    the scaled volume of atom j and contributes an additional full-shell term
    (1/rho_tilde_i - 1/L), evaluated by a fourth-order Taylor expansion for
    r < 2e-3 * rho_S_j to avoid O(1/r) cancellation.
    """
    r = np.asarray(r, dtype=float)
    rho_tilde_i = np.asarray(rho_tilde_i, dtype=float)
    rho_S_j = np.asarray(rho_S_j, dtype=float)

    case_engulf = rho_tilde_i >= r + rho_S_j
    case_inside = (~case_engulf) & (r + rho_tilde_i < rho_S_j)
    case_tiny = case_inside & (r < 2.0e-3 * rho_S_j)

    L = np.where(case_engulf, 1.0, np.maximum(rho_tilde_i, np.abs(r - rho_S_j)))
    U = np.where(case_engulf, 1.0, r + rho_S_j)

    L = np.maximum(L, 1e-10)
    U = np.maximum(U, 1e-10)
    r_safe = np.maximum(r, 1e-10)
    a_safe = np.maximum(rho_tilde_i, 1e-10)

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
    integrand = integrand + np.where(case_inside, 1.0 / a_safe - inv_L, 0.0)

    s_safe = np.where(case_tiny, np.maximum(rho_S_j, 1e-10), 1.0)
    inv_s = 1.0 / s_safe
    r2_s2 = (r * inv_s) ** 2
    H_tiny = 1.0 / a_safe - inv_s * (1.0 + r2_s2 * ((2.0 / 3.0) + (3.0 / 5.0) * r2_s2))
    integrand = np.where(case_tiny, H_tiny, integrand)
    integrand = np.where(case_engulf, 0.0, integrand)
    return integrand


def _validate_obc_radii(intrinsic_radii: np.ndarray, obc_offset: float):
    """Split live atoms from zero-radius ghost beads and validate rho_tilde > 0."""
    live = intrinsic_radii != 0.0
    if np.any(live & (intrinsic_radii <= obc_offset)):
        raise ValueError(
            "rho_tilde = rho - offset must be positive for all atoms; "
            "received an intrinsic radius below the offset"
        )
    rho_safe = np.where(live, intrinsic_radii, 1.0)
    rho_tilde_safe = np.where(live, intrinsic_radii - obc_offset, 1.0)
    return live, rho_safe, rho_tilde_safe


def obc_effective_radii(
    positions,
    intrinsic_radii,
    obc_alpha=DEFAULT_OBC_ALPHA,
    obc_beta=DEFAULT_OBC_BETA,
    obc_gamma=DEFAULT_OBC_GAMMA,
    obc_offset=DEFAULT_OBC_OFFSET,
    hct_scale=DEFAULT_HCT_SCALE,
):
    """Compute the OBC2 effective Born radii for the chain atoms.

    Each live atom obeys the OBC2 rescaling

        1/R_eff,i = 1/rho_tilde,i - (1/rho_i) tanh(alpha psi - beta psi^2 + gamma psi^3),

    with rho_tilde,i = rho_i - offset, psi_i = rho_tilde,i I_i, and screening
    radius rho_S,j = hct_scale * rho_tilde,j. Zero-radius ghost beads receive
    zero screening and return R_eff = 0.0.
    """
    positions = np.asarray(positions, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)
    n = positions.shape[0]
    if intrinsic_radii.shape != (n,):
        raise ValueError(
            f"intrinsic_radii shape {intrinsic_radii.shape} must be ({n},) "
            f"to match positions shape {positions.shape}"
        )

    live, rho_safe, rho_tilde = _validate_obc_radii(intrinsic_radii, obc_offset)
    rho_S = np.where(live, hct_scale * rho_tilde, 0.0)

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    rho_tilde_i = rho_tilde[:, None]
    rho_S_j = rho_S[None, :]
    integrand = _hct_integrand(r, rho_tilde_i, rho_S_j)
    pair_live = live[:, None] & live[None, :] & (~np.eye(n, dtype=bool))
    integrand = np.where(pair_live, integrand, 0.0)

    I = np.sum(integrand, axis=1)
    psi = rho_tilde * I
    arg = obc_alpha * psi - obc_beta * psi * psi + obc_gamma * psi * psi * psi
    tanh_arg = np.tanh(arg)

    raw_inv_R = 1.0 / rho_tilde - tanh_arg / rho_safe
    inv_R_eff = np.maximum(raw_inv_R, 1e-10)
    R_eff = np.where(live, 1.0 / inv_R_eff, 0.0)
    return R_eff


def gb_self_born_energy(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    R_eff=None,
    obc_kwargs=None,
    kT_scale=1.0,
    kappa=0.0,
):
    """Diagonal GB self-Born energy in kBT with optional temperature scaling and salt screening."""
    charges = np.asarray(charges, dtype=float)
    if R_eff is None:
        kw = obc_kwargs or {}
        R_eff = obc_effective_radii(positions, intrinsic_radii, **kw)
    else:
        R_eff = np.asarray(R_eff, dtype=float)
    live = (
        (R_eff > 0.0)
        if intrinsic_radii is None
        else (np.asarray(intrinsic_radii, dtype=float) != 0.0)
    )
    R_safe = np.where(live, R_eff, 1.0)
    q_live = np.where(live, charges, 0.0)
    k_coul = COULOMB_K_KBT_A / float(kT_scale)
    if kappa > 0.0:
        diel = 1.0 / eps_in - np.exp(-float(kappa) * R_safe) / eps_out
    else:
        diel = 1.0 / eps_in - 1.0 / eps_out
    energy = -0.5 * k_coul * float(np.sum(diel * q_live * q_live / R_safe))
    return energy


def gb_offdiagonal_energy(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    R_eff=None,
    obc_kwargs=None,
    kT_scale=1.0,
    kappa=0.0,
):
    """Off-diagonal GB cross-term energy in kBT with optional temperature scaling and salt screening."""
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]
    if R_eff is None:
        kw = obc_kwargs or {}
        R_eff = obc_effective_radii(positions, intrinsic_radii, **kw)
    else:
        R_eff = np.asarray(R_eff, dtype=float)
    live = (
        (R_eff > 0.0)
        if intrinsic_radii is None
        else (np.asarray(intrinsic_radii, dtype=float) != 0.0)
    )
    R_safe = np.where(live, R_eff, 1.0)
    q_live = np.where(live, charges, 0.0)
    k_coul = COULOMB_K_KBT_A / float(kT_scale)

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    R_i = R_safe[:, None]
    R_j = R_safe[None, :]
    A = R_i * R_j
    A_safe = np.maximum(A, 1e-10)
    D = -r * r / (4.0 * A_safe)
    f_GB = np.sqrt(r * r + A * np.exp(D))
    f_GB_safe = np.maximum(f_GB, 1e-10)

    qq = q_live[:, None] * q_live[None, :]
    if kappa > 0.0:
        diel = 1.0 / eps_in - np.exp(-float(kappa) * f_GB_safe) / eps_out
    else:
        diel = 1.0 / eps_in - 1.0 / eps_out
    pair = diel * qq / f_GB_safe

    upper = np.triu_indices(n, k=1)
    energy = -k_coul * float(np.sum(pair[upper]))
    return energy


def gb_vacuum_coulomb_energy(
    positions,
    charges,
    eps_in=1.0,
    exclude_pair_mask=None,
    kT_scale=1.0,
    intrinsic_radii=None,
    live_mask=None,
):
    """Vacuum Coulomb baseline between non-bonded chain pairs in kBT.

    Symmetrizes exclude_pair_mask across the diagonal, scales the Coulomb
    prefactor by 1 / kT_scale, and masks out ghost atoms (rho == 0) when
    intrinsic_radii or live_mask is passed.
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r_safe = np.maximum(r, 1e-10)

    if live_mask is not None:
        live = np.asarray(live_mask, dtype=bool)
    elif intrinsic_radii is not None:
        live = np.asarray(intrinsic_radii, dtype=float) != 0.0
    else:
        live = None

    q_eff = np.where(live, charges, 0.0) if live is not None else charges
    qq = q_eff[:, None] * q_eff[None, :]
    pair = qq / (eps_in * r_safe)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if live is not None:
        upper_mask = upper_mask & live[:, None] & live[None, :]
    if exclude_pair_mask is not None:
        excl = np.asarray(exclude_pair_mask, dtype=bool)
        upper_mask = upper_mask & ~(excl | excl.T)

    energy = (COULOMB_K_KBT_A / float(kT_scale)) * float(np.sum(pair[upper_mask]))
    return energy


def _hct_integrand_deriv(r, rho_tilde_i, rho_S_j):
    """Derivative d(integrand)/dr of the HCT descreening kernel at fixed radii.

    Matches every branch of _hct_integrand, including the full-shell derivative
    (-1/L^2) for r + rho_tilde_i < rho_S_j and the exact Taylor derivative for
    r < 2e-3 * rho_S_j.
    """
    r = np.asarray(r, dtype=float)
    rho_tilde_i = np.asarray(rho_tilde_i, dtype=float)
    rho_S_j = np.asarray(rho_S_j, dtype=float)

    case_engulf = rho_tilde_i >= r + rho_S_j
    case_inside = (~case_engulf) & (r + rho_tilde_i < rho_S_j)
    case_tiny = case_inside & (r < 2.0e-3 * rho_S_j)

    abs_r_minus_S = np.abs(r - rho_S_j)
    rho_tilde_wins = rho_tilde_i >= abs_r_minus_S

    L = np.where(case_engulf, 1.0, np.maximum(rho_tilde_i, abs_r_minus_S))
    U = np.where(case_engulf, 1.0, r + rho_S_j)
    L = np.maximum(L, 1e-10)
    U = np.maximum(U, 1e-10)
    r_safe = np.maximum(r, 1e-10)

    dL_dr = np.where(rho_tilde_wins, 0.0, np.sign(r - rho_S_j))
    dU_dr = np.ones_like(r)

    inv_L = 1.0 / L
    inv_U = 1.0 / U
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U
    inv_L3 = inv_L2 * inv_L
    inv_U3 = inv_U2 * inv_U
    inv_r2 = 1.0 / (r_safe * r_safe)

    dT1 = -inv_L2 * dL_dr
    dT2 = inv_U2 * dU_dr
    dT3 = 0.25 * (inv_U2 - inv_L2) + (r_safe / 4.0) * (
        -2.0 * inv_U3 * dU_dr + 2.0 * inv_L3 * dL_dr
    )
    log_LU = np.log(L / U)
    dT4 = -log_LU * 0.5 * inv_r2 + (1.0 / (2.0 * r_safe)) * (
        dL_dr * inv_L - dU_dr * inv_U
    )
    rho_S_sq = rho_S_j * rho_S_j
    dT5 = -rho_S_sq * 0.25 * inv_r2 * (inv_L2 - inv_U2) + (
        rho_S_sq / (4.0 * r_safe)
    ) * (-2.0 * inv_L3 * dL_dr + 2.0 * inv_U3 * dU_dr)

    integrand_deriv = 0.5 * (dT1 + dT2 + dT3 + dT4 + dT5)
    integrand_deriv = integrand_deriv + np.where(case_inside, -inv_L2, 0.0)

    s_safe = np.where(case_tiny, np.maximum(rho_S_j, 1e-10), 1.0)
    inv_s = 1.0 / s_safe
    inv_s2 = inv_s * inv_s
    dH_tiny = -(r * inv_s * inv_s2) * ((4.0 / 3.0) + (12.0 / 5.0) * (r * r) * inv_s2)
    integrand_deriv = np.where(case_tiny, dH_tiny, integrand_deriv)
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
    """Precompute effective Born radii, OBC chain-rule scalars, and HCT derivatives."""
    positions = np.asarray(positions, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)
    n = positions.shape[0]

    live, rho_safe, rho_tilde = _validate_obc_radii(intrinsic_radii, obc_offset)
    rho_S = np.where(live, hct_scale * rho_tilde, 0.0)

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    eye = np.eye(n, dtype=bool)
    r_safe = np.where(eye, 1.0, np.maximum(r, 1e-10))
    unit = np.zeros_like(diff)
    off_diag = ~eye
    unit[off_diag] = diff[off_diag] / r_safe[off_diag, None]

    rho_tilde_i = rho_tilde[:, None]
    rho_S_j = rho_S[None, :]
    pair_live = live[:, None] & live[None, :] & off_diag
    integrand = np.where(pair_live, _hct_integrand(r, rho_tilde_i, rho_S_j), 0.0)
    K = np.where(pair_live, _hct_integrand_deriv(r, rho_tilde_i, rho_S_j), 0.0)

    I = np.sum(integrand, axis=1)
    psi = rho_tilde * I
    arg = obc_alpha * psi - obc_beta * psi * psi + obc_gamma * psi * psi * psi
    tanh_arg = np.tanh(arg)
    sech2 = 1.0 - tanh_arg * tanh_arg

    raw_inv_R = 1.0 / rho_tilde - tanh_arg / rho_safe
    clamped = raw_inv_R <= 1e-10
    inv_R_eff = np.maximum(raw_inv_R, 1e-10)
    R_eff = np.where(live, 1.0 / inv_R_eff, 0.0)

    darg_dpsi = obc_alpha - 2.0 * obc_beta * psi + 3.0 * obc_gamma * psi * psi
    D = np.where(
        live & (~clamped),
        R_eff * R_eff * sech2 / rho_safe * darg_dpsi * rho_tilde,
        0.0,
    )

    return {
        "R_eff": R_eff,
        "D": D,
        "K": K,
        "r": r,
        "unit": unit,
        "rho_tilde": rho_tilde,
        "live": live,
    }


def chain_vacuum_coulomb_force(
    positions,
    charges,
    eps_in=1.0,
    exclude_pair_mask=None,
    kT_scale=1.0,
    intrinsic_radii=None,
    live_mask=None,
):
    """Vacuum Coulomb forces (kBT/A) and energy (kBT) between non-bonded chain pairs."""
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = positions.shape[0]

    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    eye = np.eye(n, dtype=bool)
    r_safe = np.where(eye, 1.0, np.maximum(r, 1e-10))
    inv_r = np.where(eye, 0.0, 1.0 / r_safe)
    inv_r3 = inv_r * inv_r * inv_r

    if live_mask is not None:
        live = np.asarray(live_mask, dtype=bool)
    elif intrinsic_radii is not None:
        live = np.asarray(intrinsic_radii, dtype=float) != 0.0
    else:
        live = None

    q_eff = np.where(live, charges, 0.0) if live is not None else charges
    qq = q_eff[:, None] * q_eff[None, :]

    allowed = ~eye
    if live is not None:
        allowed = allowed & live[:, None] & live[None, :]
    if exclude_pair_mask is not None:
        excl = np.asarray(exclude_pair_mask, dtype=bool)
        excl_sym = excl | excl.T
        allowed = allowed & ~excl_sym

    K_factor = COULOMB_K_KBT_A / (float(kT_scale) * eps_in)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1) & allowed
    energy = K_factor * float(np.sum(qq[upper_mask] * inv_r[upper_mask]))

    weight = qq * inv_r3 * allowed
    forces = K_factor * np.einsum("km,kmd->kd", weight, diff)
    return forces, energy


def chain_self_born_diagonal_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    obc_kwargs=None,
    kT_scale=1.0,
    kappa=0.0,
):
    """Diagonal GB self-Born forces (kBT/A) and energy (kBT)."""
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    intrinsic_radii = np.asarray(intrinsic_radii, dtype=float)

    kw = obc_kwargs or {}
    data = _obc_chain_rule_data(positions, intrinsic_radii, **kw)
    R_eff = data["R_eff"]
    D = data["D"]
    K = data["K"]
    unit = data["unit"]
    live = data["live"]

    R_safe = np.where(live, R_eff, 1.0)
    q_live = np.where(live, charges, 0.0)
    q2 = q_live * q_live
    k_coul = COULOMB_K_KBT_A / float(kT_scale)

    if kappa > 0.0:
        kap = float(kappa)
        exp_kR = np.exp(-kap * R_safe)
        diel_e = 1.0 / eps_in - exp_kR / eps_out
        diel_f = 1.0 / eps_in - (1.0 + kap * R_safe) * exp_kR / eps_out
    else:
        diel_e = 1.0 / eps_in - 1.0 / eps_out
        diel_f = diel_e

    energy = -0.5 * k_coul * float(np.sum(diel_e * q2 / R_safe))
    eta = 0.5 * k_coul * diel_f * q2 / (R_safe * R_safe) * D

    T1 = -eta[:, None] * np.einsum("kj,kjd->kd", K, unit)
    T2 = np.einsum("i,ik,ikd->kd", eta, K, unit)
    forces = T1 + T2
    return forces, energy


def _finite_difference_force(positions, energy_fn, h=1e-5):
    """Central-difference approximation of the force F = -dE/dr."""
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


def chain_offdiagonal_gb_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    obc_kwargs=None,
    kT_scale=1.0,
    kappa=0.0,
):
    """Off-diagonal GB cross-term forces (kBT/A) and energy (kBT)."""
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
    live = data["live"]

    R_safe = np.where(live, R_eff, 1.0)
    q_live = np.where(live, charges, 0.0)
    k_coul = COULOMB_K_KBT_A / float(kT_scale)
    eye_mask = np.eye(n, dtype=bool)
    pair_mask = (~eye_mask) & live[:, None] & live[None, :]

    R_i_mat = R_safe[:, None]
    R_j_mat = R_safe[None, :]
    A = R_i_mat * R_j_mat
    A_safe = np.where(pair_mask, np.maximum(A, 1e-10), 1.0)
    r_safe = np.where(pair_mask, np.maximum(r, 1e-10), 1.0)
    D_grid = -r_safe * r_safe / (4.0 * A_safe)
    expD = np.exp(D_grid)
    f_GB_sq = r_safe * r_safe + A_safe * expD
    f_GB = np.sqrt(f_GB_sq)
    inv_f_GB = np.where(pair_mask, 1.0 / f_GB, 0.0)
    inv_f_GB2 = inv_f_GB * inv_f_GB

    if kappa > 0.0:
        kap = float(kappa)
        exp_kf = np.exp(-kap * f_GB)
        diel_e = 1.0 / eps_in - exp_kf / eps_out
        diel_f = 1.0 / eps_in - (1.0 + kap * f_GB) * exp_kf / eps_out
    else:
        diel_e = 1.0 / eps_in - 1.0 / eps_out
        diel_f = diel_e

    qq = q_live[:, None] * q_live[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1) & pair_mask
    if kappa > 0.0:
        energy = -k_coul * float(np.sum(diel_e[upper] * qq[upper] * inv_f_GB[upper]))
    else:
        energy = -k_coul * diel_e * float(np.sum(qq[upper] * inv_f_GB[upper]))

    # Explicit-r term: -k_coul * sum_{m != k} q_k q_m diel_f(f_km) / f_km^2 * df/dr * unit[k, m].
    df_GB_dr = np.where(pair_mask, r_safe * (1.0 - 0.25 * expD) * inv_f_GB, 0.0)
    weight_direct = diel_f * qq * inv_f_GB2 * df_GB_dr
    F_direct = -k_coul * np.einsum("km,kmd->kd", weight_direct, unit)

    # Indirect term through R_eff via the OBC chain rule.
    df_GB_dR_first = np.where(
        pair_mask, R_j_mat * expD * (1.0 - D_grid) * inv_f_GB * 0.5, 0.0
    )
    inner = diel_f * inv_f_GB2 * df_GB_dR_first
    sum_per_m = np.sum(inner * q_live[None, :], axis=1)
    eta_off = D_chain * k_coul * q_live * sum_per_m

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
    eps_out=WATER_DIELECTRIC,
    coffdrop_active=False,
    exclude_pair_mask=None,
    obc_kwargs=None,
    kT_scale=1.0,
    kappa=0.0,
):
    """Top-level GB force composer selecting diagonal self-Born or full GB + vacuum Coulomb."""
    F_diag, E_self = chain_self_born_diagonal_force(
        positions,
        charges,
        intrinsic_radii,
        eps_in,
        eps_out,
        obc_kwargs,
        kT_scale=kT_scale,
        kappa=kappa,
    )
    forces = F_diag
    energies = {"self": E_self, "offdiag": 0.0, "coulomb": 0.0}

    if not coffdrop_active:
        F_off, E_off = chain_offdiagonal_gb_force(
            positions,
            charges,
            intrinsic_radii,
            eps_in,
            eps_out,
            obc_kwargs,
            kT_scale=kT_scale,
            kappa=kappa,
        )
        F_C, E_C = chain_vacuum_coulomb_force(
            positions,
            charges,
            eps_in,
            exclude_pair_mask,
            kT_scale=kT_scale,
            intrinsic_radii=intrinsic_radii,
        )
        forces = forces + F_off + F_C
        energies["offdiag"] = E_off
        energies["coulomb"] = E_C

    energies["total"] = energies["self"] + energies["offdiag"] + energies["coulomb"]
    return forces, energies
