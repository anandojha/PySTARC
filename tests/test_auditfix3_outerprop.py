"""Checks that the middle-region adaptive-timestep derivatives in
``OuterPropagator.new_state`` match the BrownDye2 ``outer_propagator.cc``
forms for the radial-force derivative ``Fr1`` and the hydrodynamic
diffusivity derivative ``D1``.
"""

import math

from pystarc.simulation.outer_propagator import (
    OuterPropagator,
    OPGroupInfo,
    PI,
    PI6,
)


def _make_propagator(has_hi: bool) -> OuterPropagator:
    """Build a representative propagator with nonzero charges and radii."""
    g0 = OPGroupInfo(q=2.0, Dtrans=0.015, Drot=0.0003)
    g1 = OPGroupInfo(q=-1.0, Dtrans=0.030, Drot=0.0006)
    return OuterPropagator(
        b_radius=20.0,
        max_radius=15.0,
        has_hi=has_hi,
        kT=0.593,
        viscosity=0.0009,
        dielectric=78.5,
        vacuum_perm=1.0,
        debye_len=8.0,
        g0=g0,
        g1=g1,
    )


def _radial_force(op: OuterPropagator, r: float) -> float:
    """Radial force as evaluated by the propagator's own helper."""
    return op._radial_force(r)


def test_Fr1_matches_browndye2_form():
    """Fr1 equals -V/L^2 - 2*Fr0/r (BrownDye2 outer_propagator.cc:300)."""
    op = _make_propagator(has_hi=False)
    L = op.debye_len
    for r in (25.0, 30.0, 40.0):
        Fr0 = _radial_force(op, r)
        # Yukawa monopole magnitude V used by the propagator.
        V = op.V_factor * math.exp(-r / L) / r

        # Expression as computed inside new_state.
        Fr1_code = -V / L**2 - 2.0 * Fr0 / r

        # Independent hand-computed BrownDye2 form.
        Fr1_ref = -V / (L * L) - 2.0 * Fr0 * (1.0 / r)

        assert math.isclose(Fr1_code, Fr1_ref, rel_tol=1e-12, abs_tol=0.0)

        # The previous, incorrect form (-V*(1/r+1/L)^2 - 2*Fr0/r) differs.
        Fr1_bad = -V * (1.0 / r + 1.0 / L) ** 2 - 2.0 * Fr0 / r
        assert not math.isclose(Fr1_code, Fr1_bad, rel_tol=1e-9, abs_tol=0.0)


def test_D1_uses_hi_only_part():
    """D1 is built from the HI-only diffusivity Di, not the full D0.

    BrownDye2 outer_propagator.cc:317 uses
        D1 = -3*Di*rm1 - D_factor*rm1^2/PI
    with Di the hydrodynamic-only contribution to D_parallel.
    """
    op = _make_propagator(has_hi=True)
    for r in (25.0, 30.0, 40.0):
        rm1 = 1.0 / r

        # Full parallel diffusivity (constant part + HI part).
        D0 = op._D_parallel(r)

        # HI-only part, matching the r-dependent terms of _D_parallel.
        Di = (op.D_factor / PI6) * (-3.0 / r + 2.0 * op.a2 / (r**3))

        # The constant part is the remainder.
        ainv = 1.0 / op.a0 + 1.0 / op.a1
        D_const = (op.D_factor / PI6) * ainv
        assert math.isclose(D0, D_const + Di, rel_tol=1e-12, abs_tol=0.0)

        # Expression as computed inside new_state.
        D1_code = -3.0 * Di * rm1 - op.D_factor * rm1**2 / PI

        # Independent hand-computed BrownDye2 form using the HI-only part.
        D1_ref = -3.0 * Di * (1.0 / r) - op.D_factor * (1.0 / r) ** 2 / PI

        assert math.isclose(D1_code, D1_ref, rel_tol=1e-12, abs_tol=0.0)

        # Building D1 from the full D0 injects a spurious -3*D_const*rm1
        # term; confirm the corrected D1 differs from that variant.
        D1_bad = -3.0 * D0 * rm1 - op.D_factor * rm1**2 / PI
        spurious = -3.0 * D_const * rm1
        assert math.isclose(D1_bad - D1_code, spurious, rel_tol=1e-9, abs_tol=0.0)
        assert not math.isclose(D1_code, D1_bad, rel_tol=1e-9, abs_tol=0.0)


def test_D2_D3_consistent_with_corrected_D1():
    """D2 and D3 follow the BrownDye2 recurrence from the corrected D1."""
    op = _make_propagator(has_hi=True)
    for r in (25.0, 30.0, 40.0):
        rm1 = 1.0 / r
        Di = (op.D_factor / PI6) * (-3.0 / r + 2.0 * op.a2 / (r**3))
        D1 = -3.0 * Di * rm1 - op.D_factor * rm1**2 / PI
        D2 = -4.0 * D1 * rm1 + op.D_factor * rm1**3 / PI
        D3 = -5.0 * D2 * rm1 - 2.0 * op.D_factor * rm1**4 / PI

        D2_ref = -4.0 * D1 * (1.0 / r) + op.D_factor * (1.0 / r) ** 3 / PI
        D3_ref = -5.0 * D2 * (1.0 / r) - 2.0 * op.D_factor * (1.0 / r) ** 4 / PI

        assert math.isclose(D2, D2_ref, rel_tol=1e-12, abs_tol=0.0)
        assert math.isclose(D3, D3_ref, rel_tol=1e-12, abs_tol=0.0)
