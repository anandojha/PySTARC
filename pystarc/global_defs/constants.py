"""
Physical constants and unit definitions for PySTARC.

All internal calculations use reduced units in which the thermal energy is the
natural energy scale. Length is measured in Ångströms (Å), where 1 Å = 10⁻¹⁰ m.
Time is measured in picoseconds (ps), where 1 ps = 10⁻¹² s. Energy is measured in
units of kBT at 298.15 K, where 1 kBT ≈ 2.479 kJ/mol ≈ 0.593 kcal/mol. Charge is
measured in elementary charges (e), where 1 e = 1.602 × 10⁻¹⁹ C.

In this unit system the Boltzmann factor exp(-E/kBT) reduces to simply exp(-E)
once E is expressed in kBT units. All electrostatic potentials, forces, and
energies are therefore expressed in kBT or kBT/Å.

These units suit the problem well. Brownian dynamics operates at the thermal
energy scale, so working in kBT removes repeated division by kBT. Ångströms match
the coordinates in PQR files and APBS grid outputs. Picoseconds are the natural
timescale for the diffusive motion of proteins, whose diffusion coefficients are
typically D ≈ 0.01 to 1 Å²/ps.

To convert a computed rate to SI units, use

    k_on [M⁻¹s⁻¹] = 6.022 × 10⁸ × k_b [ų/ps] × P_rxn

where the prefactor 6.022 × 10⁸ is N_A × (10⁻¹⁰)³ / (10⁻¹² × 10⁻³). Here k_b is
the rate in internal units of ų/ps and P_rxn is the reaction probability.
"""

import math

# Temperature
T_DEFAULT: float = 298.15  # K
# Boltzmann constant
KB_SI: float = 1.380649e-23  # J / K
KB_KCAL: float = 1.987204e-3  # kcal / (mol·K)
# kBT at default temperature (in kcal/mol)
KBT_KCAL: float = KB_KCAL * T_DEFAULT  # ~ 0.5922 kcal/mol
# Elementary charge
E_CHARGE: float = 1.602176634e-19  # C
# Dielectric permittivity of vacuum
EPS0_SI: float = 8.8541878128e-12  # C² / (N·m²)
# Vacuum permittivity in PySTARC's internal kBT units, that is ε₀ expressed in
# e²/(kBT·Å). This equals 1/(4π × COULOMB_K), where COULOMB_K ≈ 560.86 kBT·Å·e⁻²
# is Coulomb's constant in kBT units. It is used for in-vacuum and Debye-Huckel
# electrostatics throughout the codebase, including the NAM simulator, the GPU
# batch, and the multipole far-field.
VACUUM_PERMITTIVITY_KBT: float = 0.000142  # e²/(kBT·Å)
# Bjerrum length in water at 298 K, in Å. It is defined by
#     l_B = e² / (4π ε₀ ε_r k_B T)
# with ε_r = 78.0, and equals 1/(4π × EPS_WATER × VACUUM_PERMITTIVITY_KBT).
BJERRUM_LENGTH: float = 7.1846760153  # Å
# Relative permittivity of water, matching the Bjerrum length above and the
# solvent dielectric (sdie = 78.0) used in the production electrostatics.
EPS_WATER: float = 78.0
# Avogadro's number
AVOGADRO: float = 6.02214076e23
# Conversion factors
ANG_TO_M: float = 1.0e-10  # Å to m
PS_TO_S: float = 1.0e-12  # ps to s
KCAL_TO_J: float = 4184.0  # kcal/mol to J/mol (divide by AVOGADRO for a single molecule)
KCAL_PER_MOL_TO_KBT: float = 1.0 / KBT_KCAL  # kcal/mol to kBT
# Ion properties (default NaCl)
DEFAULT_IONIC_STRENGTH: float = 0.15  # mol/L
DEFAULT_DEBYE_LENGTH: float = 7.9  # Å at 150 mM NaCl, 298 K
# Reference viscosity for diffusion coefficients. The diffusion coefficient
# D = kBT / (6 π η r) is computed at runtime, and this is the viscosity η of water.
ETA_WATER: float = 1.002e-3  # Pa·s, water at 20°C
# Pi
PI: float = math.pi
TWO_PI: float = 2.0 * math.pi
FOUR_PI: float = 4.0 * math.pi
__all__ = [
    "T_DEFAULT",
    "KB_SI",
    "KB_KCAL",
    "KBT_KCAL",
    "E_CHARGE",
    "EPS0_SI",
    "VACUUM_PERMITTIVITY_KBT",
    "BJERRUM_LENGTH",
    "EPS_WATER",
    "AVOGADRO",
    "ANG_TO_M",
    "PS_TO_S",
    "KCAL_TO_J",
    "KCAL_PER_MOL_TO_KBT",
    "DEFAULT_IONIC_STRENGTH",
    "DEFAULT_DEBYE_LENGTH",
    "ETA_WATER",
    "PI",
    "TWO_PI",
    "FOUR_PI",
]
