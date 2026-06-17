"""
Focused robustness tests for GPUBatchResult helper methods.

These exercise pure-Python rate-constant and confidence-interval helpers that
do not require a GPU, APBS, or any external binary. They cover two low-severity
robustness fixes:

1. The Smoluchowski-fallback rate denominator (1 - P*(1 - beta)) is guarded
   against the near-zero degenerate case (P near 1, beta near 0) in both
   rate_constant and rate_constant_ci.
2. reaction_probability_ci emits a warning when there are no completed
   trajectories (n == 0), while returning the unchanged interval [0, 1].
"""

import math
import warnings

import pytest

from pystarc.simulation.gpu_batch_simulator import GPUBatchResult


def _make_result(n_reacted, n_escaped, r_start, r_escape):
    """Build a GPUBatchResult with only the fields these helpers read."""
    return GPUBatchResult(
        n_trajectories=n_reacted + n_escaped,
        n_reacted=n_reacted,
        n_escaped=n_escaped,
        n_max_steps=0,
        reaction_counts={},
        r_start=r_start,
        r_escape=r_escape,
        dt=1.0,
        elapsed_sec=1.0,
        steps_per_sec=1.0,
    )


def test_rate_constant_guard_raises_on_degenerate_denominator():
    # P_rxn == 1 (all completed trajectories reacted) and a tiny
    # r_start/r_escape ratio drive beta toward zero, so the denominator
    # 1 - P*(1 - beta) approaches zero.
    res = _make_result(n_reacted=100, n_escaped=0, r_start=1.0, r_escape=1.0e30)
    assert res.reaction_probability == 1.0
    with pytest.raises(ValueError, match="denominator"):
        res.rate_constant(D_rel=1.0, k_b=0.0)


def test_rate_constant_ci_guard_raises_on_degenerate_denominator():
    # The CI path uses the upper bound of the probability interval. With every
    # completed trajectory reacted the upper CI bound is 1, so the same
    # degenerate denominator appears inside the closure.
    res = _make_result(n_reacted=100, n_escaped=0, r_start=1.0, r_escape=1.0e30)
    with pytest.raises(ValueError, match="denominator"):
        res.rate_constant_ci(D_rel=1.0, k_b=0.0)


def test_rate_constant_healthy_path_unchanged():
    # A non-degenerate denominator must give exactly the original formula
    # CONV * k_D * P / (1 - P*(1 - beta)) with no guard interference.
    res = _make_result(n_reacted=30, n_escaped=70, r_start=10.0, r_escape=50.0)
    D_rel = 0.25
    P = res.reaction_probability
    CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    expected = CONV * k_D * P / (1.0 - P * (1.0 - beta))
    assert res.rate_constant(D_rel=D_rel, k_b=0.0) == expected


def test_rate_constant_steering_path_unchanged():
    # With k_b > 0 the steering branch is used and the denominator guard does
    # not apply at all; result is CONV * k_b * P.
    res = _make_result(n_reacted=100, n_escaped=0, r_start=1.0, r_escape=1.0e30)
    k_b = 2.5
    P = res.reaction_probability
    CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    expected = CONV * k_b * P
    # Should not raise even though the Smoluchowski denominator would be zero.
    assert res.rate_constant(D_rel=1.0, k_b=k_b) == expected


def test_reaction_probability_ci_zero_completed_warns_and_returns_unit_interval():
    res = _make_result(n_reacted=0, n_escaped=0, r_start=10.0, r_escape=50.0)
    assert res.n_completed == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci = res.reaction_probability_ci()
    assert ci == (0.0, 1.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_reaction_probability_ci_nonzero_does_not_warn():
    # For n > 0 the returned interval and behaviour must be unchanged: no
    # warning is emitted and the bounds lie within [0, 1].
    res = _make_result(n_reacted=40, n_escaped=60, r_start=10.0, r_escape=50.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lo, hi = res.reaction_probability_ci()
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert 0.0 <= lo <= hi <= 1.0
