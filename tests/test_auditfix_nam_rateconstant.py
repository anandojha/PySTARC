"""Regression tests for SimulationResult.rate_constant.

These tests check that a result carrying a nonzero stored LMZ rate self.k_db
returns the LMZ-based association rate constant, consistent with the internal
_k_from_P helper that summary() uses for its confidence interval, rather than
falling back to the Smoluchowski expression.
"""

import math

from pystarc.simulation.nam_simulator import SimulationResult, _k_from_P


def _make_result(k_db):
    return SimulationResult(
        n_trajectories=1000,
        n_reacted=10,
        n_escaped=990,
        n_max_steps=0,
        reaction_counts={"rxn": 10},
        r_start=100.0,
        r_escape=110.0,
        dt=0.2,
        k_db=k_db,
    )


def test_rate_constant_uses_stored_k_db_for_lmz():
    """A nonzero stored k_db yields the Luty-McCammon-Zhou rate rather than the Smoluchowski fallback."""
    k_db = 5.0  # Å³/ps from the outer propagator.
    res = _make_result(k_db)
    D_rel = 0.05  # Å²/ps.

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    expected_lmz = CONV_A3ps * k_db * P

    k = res.rate_constant(D_rel)

    assert math.isclose(k, expected_lmz, rel_tol=1e-9)


def test_point_estimate_matches_k_from_P():
    """The rate_constant point estimate equals _k_from_P evaluated at the same reaction probability."""
    res = _make_result(5.0)
    D_rel = 0.05

    k = res.rate_constant(D_rel)
    k_ref = _k_from_P(res, res.reaction_probability, D_rel)

    assert math.isclose(k, k_ref, rel_tol=1e-12)


def test_lmz_differs_from_smoluchowski_fallback():
    """With a stored k_db, the rate_constant result differs from the Smoluchowski fallback expression."""
    res = _make_result(5.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    smoluchowski = CONV_A3ps * k_D * P / (1.0 - P * (1.0 - beta))

    k = res.rate_constant(D_rel)

    assert not math.isclose(k, smoluchowski, rel_tol=1e-6)


def test_zero_stored_k_db_falls_back_to_smoluchowski():
    """When the stored k_db is 0.0, rate_constant uses the Smoluchowski expression."""
    res = _make_result(0.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    expected = CONV_A3ps * k_D * P / (1.0 - P * (1.0 - beta))

    k = res.rate_constant(D_rel)

    assert math.isclose(k, expected, rel_tol=1e-9)


def test_explicit_k_db_argument_overrides_stored():
    """An explicit positive k_db argument takes precedence over the stored self.k_db."""
    res = _make_result(5.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    arg_k_db = 7.5
    expected = CONV_A3ps * arg_k_db * P

    k = res.rate_constant(D_rel, k_db=arg_k_db)

    assert math.isclose(k, expected, rel_tol=1e-9)
