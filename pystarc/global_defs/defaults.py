"""
Default values for every configurable quantity that affects a computed rate.

A default that is written down in more than one place eventually disagrees with
itself. When that happens the association rate depends on which entry point the
user happened to call rather than on the physics, and nothing in the output says
so. This module is the single place a physics default may be defined. Every
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
# grid carries (1 + kappa r)^2 exp(-2 kappa r) with kappa = 1/lambda_D.
DEBYE_LENGTH: float = 7.858  # Å

# Probe radius used to roll the solvent excluded surface in APBS.
SOLVENT_PROBE_RADIUS: float = 1.5  # Å

# Conversion from SI viscosity to the internal energy-time-volume units.
# 1 Pa.s is 1 J.s/m^3, and 1 J is N_A/4184 kcal/mol, 1 s is 1e12 ps, and
# 1 m^3 is 1e30 A^3, so 1 Pa.s = (6.02214076e23 / 4184) * 1e12 / 1e30
#                              = 143.93262 kcal/mol.ps/A^3.
PA_S_TO_INTERNAL: float = 143.93262

# Water viscosity in those units. Cross-check: motion/do_bd_step.py carries
# WATER_VISCOSITY = 0.243 kBT.ps/A^3 in a different convention, and
# 0.243 * 0.592545 kcal/mol per kBT = 0.14399, agreeing to 0.2 percent.
#
# The viscosity is taken at T_DEFAULT, so the thermal energy and the solvent
# friction in D_t = kBT/(6 pi eta r) refer to the same temperature.
VISCOSITY: float = ETA_WATER * PA_S_TO_INTERNAL

# ---------------------------------------------------------------------------
# Desolvation
# ---------------------------------------------------------------------------

# Multiplier on the Born cavity self energy read from the *_born.dx grids.
# Those grids hold the Kirkwood n = 1 image energy with the rigorous
# normalisation (1/2) C D already folded in, so a charge sees exactly
# dG = alpha q^2 a^3 / r^4 and the physically correct base is 1.0. The value
# 1/(4 pi) = 0.07957747 belongs to the retired convention in which the grid held
# an APBS potential rather than a cavity self energy, and is 12.566x too weak
# against the present grids.
DESOLVATION_ALPHA: float = 1.0

# ---------------------------------------------------------------------------
# Geometry of the b and q surfaces
# ---------------------------------------------------------------------------

# Radius of the b surface on which trajectories are started. The rate is
# constructed so that k_on does not depend on this choice.
BD_MILESTONE_RADIUS: float = 30.0  # Å

# Radius of the reaction sphere. Zero is a sentinel meaning the reaction cutoff
# falls back to the b surface radius, which makes the two spheres coincide and
# reduces k_on to the bare encounter rate. PySTARCConfig.validate rejects that
# degenerate case rather than reporting a rate that ignores the binding site.
BD_MILESTONE_RADIUS_INNER: float = 0.0  # Å, sentinel

# The escape sphere sits at this multiple of the b surface radius. A trajectory
# reaching it is either recycled or counted as escaped.
R_ESCAPE_FACTOR: float = 2.0

# The outer propagator takes over at this multiple of the b surface radius. This
# is the LMZ handover, a different quantity from the escape sphere above, and
# the two were previously conflated onto one field.
QB_FACTOR: float = 1.1

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

# Outer region timestep, and the smaller step taken inside the reaction shell
# where the force varies fastest.
DT: float = 0.2  # ps
DT_RXN: float = 0.05  # ps

# A flexible chain integrates a stiff bonded potential alongside the diffusive
# motion, so it needs a step an order of magnitude shorter than a rigid body.
CHAIN_DT: float = 0.01  # ps

# Lower bounds on the adaptive timestep. Zero means no bound is imposed and the
# step is free to follow the local force gradient. These are floors and are a
# separate quantity from DT and DT_RXN above, which are step sizes.
MINIMUM_CORE_DT: float = 0.0  # ps, sentinel
MINIMUM_CORE_REACTION_DT: float = 0.0  # ps, sentinel

# ---------------------------------------------------------------------------
# Hydrodynamics
# ---------------------------------------------------------------------------

# Whether the Rotne-Prager-Yamakawa correction is applied to the relative
# mobility in the outer propagator. Squeezing solvent out from between two
# approaching surfaces slows the approach, so enabling it can only lower k_on,
# by roughly 10 to 35 percent for a protein and a small ligand.
#
# Default on. Displacing solvent from between two approaching surfaces is a real
# effect, so switching it off is an approximation rather than a neutral choice.
# Note the size of the effect differs by region. In the outer propagator it is a
# quadrature and lowers k_on by roughly 10 to 35 percent. In the inner region it
# acts only through the divergence drift: without that term the same scalar
# multiplies drift and noise, D(r) factors out of the generator, and the
# splitting probability is unchanged by hydrodynamics altogether
# and should not be what an unspecified run silently receives. The reference
# implementation documents the same default (browndye2.tex:481). No existing
# result changes, because every shipped input deck sets the flag explicitly.
HYDRODYNAMIC_INTERACTIONS: bool = True

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
# Conventions of the external reference binary
# ---------------------------------------------------------------------------
#
# prepare_bd_surface does not configure PySTARC. It writes the input deck for
# the reference nam_simulation binary, whose conventions differ from ours in
# four places. Those differences are deliberate and are recorded here so that
# they read as choices rather than as drift.
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
# the reference binary genuinely differs.
REFERENCE_LOCAL_TAGS = frozenset(REFERENCE_DEFAULTS)

# Names that must carry the same default wherever they appear, including in
# dataclass fields, function signatures and getattr fallbacks. The regression
# test in tests/test_config_defaults.py enforces this by walking the syntax tree
# of the whole package, so a future divergence fails the suite rather than
# quietly changing a rate.
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
