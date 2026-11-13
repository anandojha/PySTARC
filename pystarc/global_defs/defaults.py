"""
Default values for every configurable quantity that affects a computed rate.

This module is the single place a physics default may be defined. Every
parser, dataclass and force engine takes its value from here.

INPUT_DEFAULTS is keyed by the literal tag name in pystarc_input.xml, so the two
readers of that file, pipeline.input_parser.parse and
pipeline.prepare_bd_surface.parse_config, resolve an omitted tag identically.

Several entries are sentinels rather than physical values. A floor of 0.0 means
no floor is imposed, and a radius of 0.0 means the value is derived from another
quantity at setup. Those are meaningful settings, not missing ones, and they are
marked below. A sentinel and a physical value must never share a name.
"""

from typing import Any, Dict

from pystarc.global_defs.constants import ETA_WATER

# ---------------------------------------------------------------------------
# Solvent conditions
# ---------------------------------------------------------------------------

# Bulk water at 298.15 K. The dielectric enters the screened Coulomb forces and
# the Boltzmann integrand that normalises k_b, so the two must agree.
TEMPERATURE: float = 298.15  # K
SOLVENT_DIELECTRIC: float = 78.0  # relative permittivity of water
SOLUTE_DIELECTRIC: float = 4.0  # relative permittivity of protein interior

# Debye screening length for about 150 mM monovalent salt at 298.15 K. The
# electrostatic force carries the factor exp(-r/lambda_D), and the desolvation
# grid carries (1 + kappa r)^2 exp(-2 kappa r) with kappa = 1/lambda_D. A rigid
# input without <debye_length> takes it from ion_concentration, sdie and
# temperature instead (input_parser.debye_from_ions).
DEBYE_LENGTH: float = 7.858  # Å

# Probe radius used to roll the solvent excluded surface in APBS.
SOLVENT_PROBE_RADIUS: float = 1.5  # Å

# Conversion from SI viscosity to the internal energy-time-volume units.
# 1 Pa.s is 1 J.s/m^3, and 1 J is N_A/4184 kcal/mol, 1 s is 1e12 ps, and
# 1 m^3 is 1e30 A^3, so 1 Pa.s = (6.02214076e23 / 4184) * 1e12 / 1e30
#                              = 143.93262 kcal/mol.ps/A^3.
PA_S_TO_INTERNAL: float = 143.93262

# Water viscosity in those units, 0.890 mPa.s. motion/do_bd_step.py carries the
# BrownDye 2 value WATER_VISCOSITY = 0.243 kBT.ps/A^3, and 0.243 * 0.592545
# kcal/mol per kBT = 0.14399, 1.124 times this one.
#
# The viscosity is that of water at 25 C at every run temperature:
# <temperature> scales kBT only, and <relative_viscosity> must be 1.
VISCOSITY: float = ETA_WATER * PA_S_TO_INTERNAL

# ---------------------------------------------------------------------------
# Desolvation
# ---------------------------------------------------------------------------

# Multiplier on the Born cavity self energy read from the *_born.dx grids.
# Those grids hold the Kirkwood n = 1 image energy with the rigorous
# normalisation (1/2) C D already folded in, so a charge sees exactly
# dG = alpha q^2 a^3 / r^4, so the physically correct base is 1.0.
DESOLVATION_ALPHA: float = 1.0

# ---------------------------------------------------------------------------
# Geometry of the b and q surfaces
# ---------------------------------------------------------------------------

# Radius of the b surface on which trajectories are started. The rate is
# constructed so that k_on does not depend on this choice.
BD_MILESTONE_RADIUS: float = 30.0  # Å

# GHO-GHO reaction distance of the automatic criterion. Zero is a sentinel for
# no such criterion: auto_detect_reactions refuses the GHO path unless
# 0 < bd_milestone_radius_inner < bd_milestone_radius.
BD_MILESTONE_RADIUS_INNER: float = 0.0  # Å, sentinel

# The escape sphere sits at this multiple of the b surface radius. A trajectory
# reaching it is either recycled or counted as escaped.
R_ESCAPE_FACTOR: float = 2.0

# The outer propagator takes over at this multiple of the b surface radius. This
# is the LMZ handover, a distinct quantity from the escape sphere above.
QB_FACTOR: float = 1.1

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

# Fixed outer step of chain BD; rigid runs ignore it. DT_RXN is the step the CPU
# model takes inside the reaction shell.
DT: float = 0.2  # ps
DT_RXN: float = 0.05  # ps

# A flexible chain integrates a stiff bonded potential alongside the diffusive
# motion, so it needs a step an order of magnitude shorter than a rigid body.
CHAIN_DT: float = 0.01  # ps

# Floors of the adaptive time step, zero for none. minimum_core_dt bounds every
# step; minimum_core_reaction_dt bounds only the reaction refinement of
# state-machine runs. They are floors, not step sizes like DT and DT_RXN.
MINIMUM_CORE_DT: float = 0.0  # ps, sentinel
MINIMUM_CORE_REACTION_DT: float = 0.0  # ps, sentinel

# ---------------------------------------------------------------------------
# Hydrodynamics
# ---------------------------------------------------------------------------

# Whether the Rotne-Prager-Yamakawa correction is applied to the relative
# mobility. Displacing solvent from between two approaching surfaces slows the
# approach. In the outer propagator it lowers k_on by roughly 10 to 35 percent.
# In the inner region it acts only through the divergence drift: without that
# term the same scalar multiplies drift and noise, D(r) factors out of the
# generator, and the splitting probability is unchanged by hydrodynamics.
HYDRODYNAMIC_INTERACTIONS: bool = True
SAVE_INTERVAL: int = 100  # steps between saved path points, 0 saves endpoints only

# ---------------------------------------------------------------------------
# The registry consulted by both readers of pystarc_input.xml
# ---------------------------------------------------------------------------

INPUT_DEFAULTS: Dict[str, Any] = {
    "temperature": TEMPERATURE,
    "sdie": SOLVENT_DIELECTRIC,
    "pdie": SOLUTE_DIELECTRIC,
    "debye_length": DEBYE_LENGTH,
    "srad": SOLVENT_PROBE_RADIUS,
    "desolvation_alpha": DESOLVATION_ALPHA,
    "bd_milestone_radius": BD_MILESTONE_RADIUS,
    "bd_milestone_radius_inner": BD_MILESTONE_RADIUS_INNER,
    "dt": DT,
    "minimum_core_dt": MINIMUM_CORE_DT,
    "minimum_core_reaction_dt": MINIMUM_CORE_REACTION_DT,
    "hydrodynamic_interactions": HYDRODYNAMIC_INTERACTIONS,
}

# ---------------------------------------------------------------------------
# Conventions of the external simulation engine
# ---------------------------------------------------------------------------
#
# prepare_bd_surface does not configure PySTARC. It writes the input deck for
# the external simulation engine, whose conventions differ from PySTARC in
# four places.
#
# The screening length is a sentinel there: zero means derive it from the ion
# concentration rather than take it as given, which is why it must not inherit
# DEBYE_LENGTH above. The two timestep floors are the reference defaults and
# are not sentinels on that side. The reaction sphere is a physical radius
# there, whereas PySTARC treats zero as "fall back to the b surface".
REFERENCE_DEFAULTS: Dict[str, Any] = {
    "debye_length": 0.0,  # sentinel: derive from ion concentration
    "bd_milestone_radius_inner": 12.0,  # Å
    "minimum_core_dt": 0.2,  # ps
    "minimum_core_reaction_dt": 0.05,  # ps
}

# Tags prepare_bd_surface resolves from its own conventions above. Every other
# tag it reads comes from INPUT_DEFAULTS, so the two readers agree except where
# the external engine genuinely differs.
REFERENCE_LOCAL_TAGS = frozenset(REFERENCE_DEFAULTS)

# Names that must carry the same default wherever they appear, including in
# dataclass fields, function signatures and getattr fallbacks. The regression
# test in tests/test_config_defaults.py enforces this by walking the syntax tree
# of the whole package.
PHYSICS_DEFAULT_NAMES = frozenset(INPUT_DEFAULTS) | {
    "temperature_kT",
    "dielectric",
    "dielectric_in",
    "dielectric_out",
    "viscosity",
    "r_start",
    "r_escape",
    "dt_rxn",
}
