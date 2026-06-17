"""Regression tests for the HCT pairwise descreening lower integration limit.

These tests target the engulfed-atom regime in which a small atom i sits inside a
larger close neighbor j (separation r < rho_S_j). The Hawkins-Cramer-Truhlar lower
integration limit is L = max(rho_tilde_i, abs(r - rho_S_j)), and the absolute value
is what keeps the limit physical when r < rho_S_j.
"""

import numpy as np

from pystarc.forces.chain_gb import _hct_integrand, _hct_integrand_deriv


def _hct_closed_form(L, U, r, rho_S_j):
    """Closed-form HCT integrand for explicit lower and upper limits L and U."""
    return 0.5 * (
        1.0 / L
        - 1.0 / U
        + (r / 4.0) * (1.0 / U**2 - 1.0 / L**2)
        + (1.0 / (2.0 * r)) * np.log(L / U)
        + (rho_S_j * rho_S_j / (4.0 * r)) * (1.0 / L**2 - 1.0 / U**2)
    )


def test_engulfed_atom_integrand_uses_absolute_value():
    """For r < ρ_S_j the HCT integrand lower limit is abs(r - ρ_S_j) rather than ρ̃_i."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0
    # The atom is engulfed by the larger neighbor: rho_S_j - r = 1.0 exceeds
    # rho_tilde_i = 0.8, so the canonical lower limit is abs(r - rho_S_j) = 1.0.
    assert r < rho_S_j
    assert (rho_S_j - r) > rho_tilde_i

    L_canonical = max(rho_tilde_i, abs(r - rho_S_j))
    U = r + rho_S_j
    reference = _hct_closed_form(L_canonical, U, r, rho_S_j)

    got = float(_hct_integrand(r, rho_tilde_i, rho_S_j))
    assert np.isclose(got, reference, rtol=0, atol=1e-12)


def test_engulfed_atom_integrand_smaller_than_old_expression():
    """The abs(r - ρ_S_j) lower limit yields a smaller integrand than the old ρ̃_i form, removing the descreening overcount."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0
    U = r + rho_S_j

    corrected = float(_hct_integrand(r, rho_tilde_i, rho_S_j))

    # The expression without the absolute value would route this geometry to the
    # surface-overlap branch and integrate from L = rho_tilde_i, which lies below the
    # physical limit abs(r - rho_S_j) and overcounts the descreening.
    old_expression = _hct_closed_form(rho_tilde_i, U, r, rho_S_j)

    assert old_expression > corrected
    assert not np.isclose(old_expression, corrected)


def test_engulfed_atom_derivative_matches_hand_reference():
    """In the engulfed regime the analytic HCT integrand derivative gives dL/dr = -1, matching finite differences."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0

    analytic = float(_hct_integrand_deriv(r, rho_tilde_i, rho_S_j))

    h = 1e-6
    fd = (
        float(_hct_integrand(r + h, rho_tilde_i, rho_S_j))
        - float(_hct_integrand(r - h, rho_tilde_i, rho_S_j))
    ) / (2.0 * h)

    assert np.isclose(analytic, fd, rtol=0, atol=1e-6)


def test_standard_outside_regime_unchanged():
    """For r > ρ_S_j the absolute value is a no-op, so integrand and derivative match the canonical reference."""
    for r, rho_tilde_i, rho_S_j in [(5.0, 1.5, 1.2), (3.0, 2.0, 1.0), (4.0, 1.0, 1.8)]:
        assert r > rho_S_j  # abs(r - rho_S_j) == r - rho_S_j here

        L_canonical = max(rho_tilde_i, r - rho_S_j)
        U = r + rho_S_j
        reference = _hct_closed_form(L_canonical, U, r, rho_S_j)
        got = float(_hct_integrand(r, rho_tilde_i, rho_S_j))
        assert np.isclose(got, reference, rtol=0, atol=1e-12)

        analytic = float(_hct_integrand_deriv(r, rho_tilde_i, rho_S_j))
        h = 1e-6
        fd = (
            float(_hct_integrand(r + h, rho_tilde_i, rho_S_j))
            - float(_hct_integrand(r - h, rho_tilde_i, rho_S_j))
        ) / (2.0 * h)
        assert np.isclose(analytic, fd, rtol=0, atol=1e-5)
