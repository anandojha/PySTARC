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
DEFAULT_OBC_OFFSET = 0.09  # Offset between the van der Waals and intrinsic radius, in angstrom.
DEFAULT_HCT_SCALE = 0.85  # Uniform Hawkins-Cramer-Truhlar scaling factor.

# Coulomb constant k_e e^2 / kBT expressed in angstrom at T = 300.15 K. The value is
# recomputed from first principles as
#
#     COULOMB_K_KBT_A = k_e e^2 * 1e10 / (kB T)
#
# with k_e = 8.9875517873681764e9 N m^2 / C^2 (CODATA, 1/(4 pi eps0)),
# e = 1.602176634e-19 C, kB = 1.380649e-23 J/K, and T = 300.15 K. The factor 1e10
# converts the meters in k_e e^2 to angstrom. Numerically k_e e^2 = 2.30708e-18 J*A
# and kB T = 4.14402e-21 J, so the ratio is 556.72 A in units of kBT per squared
# elementary charge. This constant carries only the Coulomb prefactor and temperature;
# the water dielectric enters separately through the cf prefactor below and is taken
# as the single consistent value WATER_DIELECTRIC = 78.5.
COULOMB_K_KBT_A = 556.72

# Single consistent water (exterior solvent) dielectric used throughout this module
# as the default eps_out. Defining it once avoids the value drifting between nearby
# literals such as 78.5 and 78.54 across the various energy and force routines.
WATER_DIELECTRIC = 78.5


def _hct_integrand(r, rho_tilde_i, rho_S_j):
    """Hawkins-Cramer-Truhlar pairwise descreening integrand for atom i due to atom j.

    The routine is vector-safe. The inputs may be arrays as long as they are
    broadcast-compatible with each other, and the integrand is returned at the
    given separation r.

    The geometry falls into two regimes. When rho_tilde_i is at least r + rho_S_j,
    atom j sits entirely inside atom i's volume, so it cannot descreen and the
    integrand is zero. Otherwise the integration runs from a lower limit
    L = max(rho_tilde_i, abs(r - rho_S_j)) to an upper limit U = r + rho_S_j. The
    lower limit equals rho_tilde_i when atom j overlaps the surface of atom i, equals
    r - rho_S_j when atom j lies fully outside atom i, and equals rho_S_j - r when
    atom i sits inside the volume of the larger atom j.
    """
    case_engulf = rho_tilde_i >= r + rho_S_j

    L = np.where(case_engulf, 1.0, np.maximum(rho_tilde_i, np.abs(r - rho_S_j)))
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
    """Compute the OBC2 effective Born radii for the chain atoms.

    Each effective radius follows from the OBC2 rescaling,

        1/R_eff,i = 1/rho_tilde,i - (1/rho_i) tanh(alpha psi - beta psi^2 + gamma psi^3),

    where rho_tilde,i = rho_i - offset and psi_i = rho_tilde,i I_i. The descreening
    integral is I_i = sum_{j != i} HCT_integrand(r_ij, rho_tilde,i, rho_S,j) with the
    scaled radius rho_S,j = hct_scale rho_j.

    The argument positions is the (n_atoms, 3) array of atom positions in angstrom.
    The argument intrinsic_radii is the (n_atoms,) array of intrinsic Born radii rho_i
    in angstrom, that is the van der Waals radii before the offset is subtracted. The
    parameters obc_alpha, obc_beta, and obc_gamma are the OBC2 scaling coefficients,
    whose Onufriev 2004 set II defaults are 1.0, 0.8, and 4.85. The argument obc_offset
    is the offset between the van der Waals and intrinsic radius in angstrom, and
    hct_scale is the dimensionless Hawkins-Cramer-Truhlar scaling factor S.

    The return value R_eff is the (n_atoms,) array of effective Born radii in angstrom.
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
    eps_out=WATER_DIELECTRIC,
    R_eff=None,
    obc_kwargs=None,
):
    """Diagonal GB self-Born energy.

    The self-energy is U_self = -1/2 cf sum_i q_i^2 / R_eff,i, where the dielectric
    prefactor is cf = (1/eps_in - 1/eps_out) COULOMB_K_KBT_A.

    The return value is a scalar energy in units of kBT.
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
    eps_out=WATER_DIELECTRIC,
    R_eff=None,
    obc_kwargs=None,
):
    """Off-diagonal GB cross-term energy.

    The cross term is U_offdiag = -cf sum_{i<j} q_i q_j / f_GB(r_ij, R_eff,i, R_eff,j),
    where the generalized distance of Still and coworkers (1990) is
    f_GB = sqrt(r^2 + R_i R_j exp(-r^2 / (4 R_i R_j))) and the dielectric prefactor is
    cf = (1/eps_in - 1/eps_out) COULOMB_K_KBT_A.

    The return value is a scalar energy in units of kBT.
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

    The baseline is U_Coulomb = (COULOMB_K_KBT_A / eps_in) sum q_i q_j / r_ij, where
    the sum runs over pairs i < j that are not excluded. Pairs flagged in
    exclude_pair_mask, typically the bonded 1-2, 1-3, and 1-4 neighbors, are skipped.

    The argument positions is the (n_atoms, 3) array of positions in angstrom and
    charges is the (n_atoms,) array of partial charges in units of the elementary
    charge. The argument eps_in is the interior dielectric, which is typically 1.0 for
    a chain treated as a vacuum-like solute. The argument exclude_pair_mask is an
    (n_atoms, n_atoms) boolean array where a True entry at (i, j) means that pair is
    skipped, and only its upper triangle is read.

    The return value is a scalar energy in units of kBT.
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


# Forces: vacuum Coulomb and diagonal self-Born.
#
# The diagonal self-Born force comes from differentiating U_self = -1/2 cf sum_i
# q_i^2 / R_eff,i through the OBC chain rule. The dependence on the effective radius
# gives dE/dR_eff,i = +1/2 cf q_i^2 / R_eff,i^2, and the effective radius depends on
# the descreening integral through dR_eff,i/dr_k = D_i dI_i/dr_k. The OBC scalar D_i
# is R_eff,i^2 (1 - tanh^2(arg_i)) / rho_i times (alpha - 2 beta psi_i + 3 gamma psi_i^2)
# times rho_tilde,i.
#
# It is convenient to define eta_i = (dE/dR_eff,i) D_i = 1/2 cf q_i^2 / R_eff,i^2 D_i,
# so that dE/dr_k = sum_i eta_i dI_i/dr_k. The integral gradient is
# dI_i/dr_k = sum_{j != i} K[i, j] d(r_ij)/dr_k, where K[i, j] = d(integrand_ij)/dr_ij
# depends on rho_tilde,i and rho_S,j. The distance gradient d(r_ij)/dr_k is the unit
# vector from j toward i when k = i, the unit vector from i toward j when k = j, and
# zero otherwise.
#
# The force F_k = -dE/dr_k therefore assembles two pieces, both relying on K[k, k] = 0.
# In the first piece atom k acts as the subject, contributing -eta_k sum_j K[k, j]
# unit[k, j]. In the second piece atom k acts as a screener for the other atoms,
# contributing +sum_i eta_i K[i, k] unit[i, k]. Both pieces are evaluated with einsum
# over the precomputed K and unit-vector tensors. As a translation-invariance check,
# sum_k F_k cancels term by term and gives zero, as required.


def _hct_integrand_deriv(r, rho_tilde_i, rho_S_j):
    """Derivative of the HCT integrand with respect to r at fixed rho_tilde_i, rho_S_j.

    The routine returns d(integrand)/dr and is vector-safe and broadcast-compatible.

    The branches mirror those of _hct_integrand. In the engulfed regime the integrand
    is constant, so the derivative is zero. Otherwise the lower limit is
    L = max(rho_tilde_i, abs(r - rho_S_j)) and the upper limit is U = r + rho_S_j. The
    upper limit always varies with r, so dU/dr = 1. The lower limit is constant in r
    when rho_tilde_i is the larger of the two, giving dL/dr = 0, and otherwise tracks
    abs(r - rho_S_j), giving dL/dr = sign(r - rho_S_j).
    """
    case_engulf = rho_tilde_i >= r + rho_S_j
    abs_r_minus_S = np.abs(r - rho_S_j)
    rho_tilde_wins = rho_tilde_i >= abs_r_minus_S

    L = np.where(case_engulf, 1.0, np.maximum(rho_tilde_i, abs_r_minus_S))
    U = np.where(case_engulf, 1.0, r + rho_S_j)
    L = np.maximum(L, 1e-10)
    U = np.maximum(U, 1e-10)
    r_safe = np.maximum(r, 1e-10)

    # dL/dr is 0 when L is fixed at rho_tilde_i, and is sign(r - rho_S_j) when
    # L tracks abs(r - rho_S_j).
    dL_dr = np.where(rho_tilde_wins, 0.0, np.sign(r - rho_S_j))
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
    """Precompute everything needed for the OBC force chain rule.

    The return value is a dictionary. The entry R_eff holds the (n,) effective Born
    radii. The entry D holds the (n,) OBC chain-rule scalars D_i = dR_eff_i / dI_i.
    The entry K holds the (n, n) integrand derivatives K[i, j] = d(integrand_ij) / dr_ij
    with the diagonal set to zero. The entry r holds the (n, n) pairwise distances. The
    entry unit holds the (n, n, 3) unit vectors unit[i, j, :] = (r_i - r_j) / r_ij with
    the diagonal set to zero. The entry rho_tilde holds the (n,) offset-corrected radii.
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

    The force on atom k is F_k = (COULOMB_K / eps_in) q_k sum q_m (r_k - r_m) / r_km^3,
    summed over allowed partners m other than k, and the energy is
    U = (COULOMB_K / eps_in) sum q_i q_j / r_ij over allowed pairs i < j.

    The return value forces is the (n, 3) array of forces in kBT per angstrom on each
    chain atom, and energy is a scalar in units of kBT.
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

    # Build the symmetric mask of allowed pairs. By default every off-diagonal pair
    # is allowed.
    allowed = ~np.eye(n, dtype=bool)
    if exclude_pair_mask is not None:
        excl = np.asarray(exclude_pair_mask, dtype=bool)
        excl_sym = excl | excl.T
        allowed = allowed & ~excl_sym

    K_factor = COULOMB_K_KBT_A / eps_in
    pair_e = qq * inv_r * allowed
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1) & allowed
    energy = K_factor * float(np.sum(qq[upper_mask] * inv_r[upper_mask]))

    # The force vector on atom k is K_factor q_k sum q_m (r_k - r_m) / r_km^3 over the
    # allowed partners m, which equals K_factor q_k sum_m q_m inv_r3[k, m] (r_k - r_m).
    weight = qq * inv_r3 * allowed  # (n, n)
    forces = K_factor * np.einsum("km,kmd->kd", weight, diff)
    return forces, energy


def chain_self_born_diagonal_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    obc_kwargs=None,
):
    """Diagonal GB self-Born forces and energy.

    The force is F_k = -dE_self/dr_k = -sum_i eta_i dI_i/dr_k, where the per-atom weight
    is eta_i = 0.5 cf q_i^2 / R_eff_i^2 D_i. When atom k is itself the subject, the
    integral gradient is dI_k/dr_k = sum_j K[k, j] unit[k, j, :], and when k instead
    screens another atom i, the gradient is dI_i/dr_k = -K[i, k] unit[i, k, :] for i not
    equal to k. Substituting these and using K[k, k] = 0 to extend the sums over all
    atoms gives

        F_k = -eta_k sum_j K[k, j] unit[k, j, :] + sum_i eta_i K[i, k] unit[i, k, :].

    The return value forces is the (n, 3) array in kBT per angstrom and energy is a
    scalar in kBT.
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

    # The per-atom weight is eta_i = 0.5 cf q_i^2 / R_eff_i^2 D_i.
    eta = 0.5 * cf * (charges * charges) / (R_eff * R_eff) * D  # (n,)

    # First term, atom k acting as the subject: -eta_k sum_j K[k, j] unit[k, j, :].
    T1 = -eta[:, None] * np.einsum("kj,kjd->kd", K, unit)
    # Second term, atom k acting as a screener: +sum_i eta_i K[i, k] unit[i, k, :].
    T2 = np.einsum("i,ik,ikd->kd", eta, K, unit)
    forces = T1 + T2
    return forces, energy


def _finite_difference_force(positions, energy_fn, h=1e-5):
    """Central-difference approximation of the force F = -dE/dr.

    The callable energy_fn must accept a position array of shape (n, 3) and return a
    scalar energy. The routine returns an array of shape (n, 3) and serves as the
    ground truth for verifying the analytical force implementations.
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


# Off-diagonal GB cross-term force and top-level dispatcher.
#
# The off-diagonal force has two contributions. The direct contribution comes from
# the explicit r_ij inside f_GB and is
# F_k^direct = -cf q_k sum_{m != k} q_m / f_GB,km^2 df_GB/dr_km unit[k, m]. The indirect
# contribution comes from the dependence of every R_eff,m on the geometry, propagated
# through the OBC chain rule. Writing
# eta_m^offdiag = D_m cf q_m sum_{j != m} q_j / f_GB,mj^2 df_GB,mj/dR_m, the indirect
# force is F_k^indirect = -eta_k^offdiag sum_j K[k, j] unit[k, j]
# + sum_i eta_i^offdiag K[i, k] unit[i, k]. The indirect piece reuses the OBC chain-rule
# structure of the diagonal force with eta^offdiag in place of eta^diag.


def chain_offdiagonal_gb_force(
    positions,
    charges,
    intrinsic_radii,
    eps_in=1.0,
    eps_out=WATER_DIELECTRIC,
    obc_kwargs=None,
):
    """Off-diagonal GB cross-term forces and energy.

    The argument positions is the (n, 3) array of positions in angstrom, charges is the
    (n,) array of partial charges in units of the elementary charge, and intrinsic_radii
    is the (n,) array of intrinsic Born radii in angstrom. The arguments eps_in and
    eps_out are the interior and exterior dielectric constants, and obc_kwargs holds any
    optional OBC2 parameter overrides.

    The return value forces is the (n, 3) array in kBT per angstrom and energy is a
    scalar in kBT.
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

    # Evaluate f_GB along with its r and R_i derivatives over all pairs (i, j).
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

    # Direct piece. The distance derivative is df_GB/dr = (r/f_GB)(1 - expD/4) and is
    # zeroed on the diagonal.
    df_GB_dr = r_safe * (1.0 - 0.25 * expD) * inv_f_GB
    df_GB_dr[eye_mask] = 0.0
    weight_direct = qq * inv_f_GB2 * df_GB_dr
    weight_direct[eye_mask] = 0.0
    F_direct = -cf * np.einsum("km,kmd->kd", weight_direct, unit)

    # Indirect piece through R_eff, applied with the per-atom chain rule. Treating the
    # first radius argument as the variable, df_GB,mj/dR_m = R_j expD (1 - D) / (2 f_GB).
    df_GB_dR_first = R_j_mat * expD * (1.0 - D_grid) * inv_f_GB * 0.5
    df_GB_dR_first[eye_mask] = 0.0
    # The per-atom weight is eta_m^offdiag = D_m cf q_m sum_{j != m} q_j / f_GB,mj^2 df_GB,mj/dR_m.
    inner = inv_f_GB2 * df_GB_dR_first  # (n, n)
    sum_per_m = np.sum(inner * charges[None, :], axis=1)  # (n,)
    eta_off = D_chain * cf * charges * sum_per_m  # (n,)

    # Apply the OBC chain rule with eta_off, using the same structure as the diagonal force.
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
):
    """Top-level GB force composer that selects between the two electrostatic treatments.

    When coffdrop_active is False the full GB model is assembled, namely the diagonal
    self-energy together with the off-diagonal cross term and the vacuum Coulomb
    baseline. When coffdrop_active is True only the diagonal term is kept, which avoids
    double-counting against COFFDROP's pre-screened pair potentials.

    The arguments positions, charges, and intrinsic_radii are the standard chain arrays,
    and eps_in and eps_out are the interior and exterior dielectric constants. The flag
    coffdrop_active restricts the result to the diagonal term when True. The argument
    exclude_pair_mask is an (n, n) boolean array of pairs to skip in the vacuum Coulomb
    sum, typically the bonded 1-2, 1-3, and 1-4 exclusions, and obc_kwargs holds any
    OBC2 parameter overrides.

    The return value forces is the (n, 3) array in kBT per angstrom. The return value
    energies is a dictionary with the keys 'self', 'offdiag', 'coulomb', and 'total' in
    kBT, where the off-diagonal and Coulomb entries are zero when coffdrop_active is True.
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
