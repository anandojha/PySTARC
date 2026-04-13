"""
Physical constants and unit definitions for PySTARC.

Unit system
-----------
All internal calculations use "reduced" units where thermal energy
is the natural energy scale.
  Length  : Ångströms (Å)     - 1 Å = 10⁻¹⁰ m
  Time    : picoseconds (ps)  - 1 ps = 10⁻¹² s
  Energy  : kBT at 298.15 K   - 1 kBT ≈ 2.479 kJ/mol ≈ 0.593 kcal/mol
  Charge  : elementary charge (e) - 1 e = 1.602 × 10⁻¹⁹ C

In this unit system, the Boltzmann factor exp(-E/kBT) becomes
simply exp(-E) when E is already in kBT units.  All electrostatic
potentials, forces, and energies are expressed in kBT or kBT/Å.

Why these units?
  - Brownian dynamics operates at the thermal energy scale, so
    expressing everything in kBT eliminates repeated division.
  - Ångströms match PQR file coordinates and APBS grid outputs.
  - Picoseconds are the natural timescale for diffusive motion of
    proteins (D ≈ 0.01-1 Å²/ps for typical biomolecules).

Conversion to SI:
  k_on [M⁻¹s⁻¹] = 6.022 × 10⁸ × k_b [ų/ps] × P_rxn
  where the prefactor = N_A × (10⁻¹⁰)³ / (10⁻¹² × 10⁻³)
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
# Bjerrum length in water at 298 K (Å)
# l_B = e² / (4π ε₀ ε_r k_B T)   ε_r = 78.54
BJERRUM_LENGTH: float = 7.1846760153  # Å  - exact from vacuum_permittivity=0.000142
# = 1/(4π × sdie × ε₀_reduced) = 1/(4π × 78 × 0.000142)
# Dielectric constant of water
EPS_WATER: float = 78.54
# Avogadro's number
AVOGADRO: float = 6.02214076e23
# Conversion factors
ANG_TO_M: float = 1.0e-10  # Å -> m
PS_TO_S: float = 1.0e-12  # ps -> s
KCAL_TO_J: float = 4184.0  # kcal/mol -> J/mol  (per molecule: /AVOGADRO)
KCAL_PER_MOL_TO_KBT: float = 1.0 / KBT_KCAL  # kcal/mol -> kBT
# Ion properties (default NaCl)
DEFAULT_IONIC_STRENGTH: float = 0.15  # mol/L
DEFAULT_DEBYE_LENGTH: float = 7.9  # Å  at 150 mM NaCl, 298 K
# Diffusion coefficient reference
# D = kBT / (6 π η r)  - computed at runtime; this is η of water
ETA_WATER: float = 1.002e-3  # Pa·s  - water at 20°C
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
