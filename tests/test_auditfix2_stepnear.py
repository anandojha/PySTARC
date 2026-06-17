"""
Regression tests for step_near_absorbing_surface.

These tests confirm that the healthy sampling path is numerically unchanged and
that an over-constrained, degenerate configuration makes the rejection-sampling
exhaustion visible through a warning instead of silently returning a biased,
fixed value.
"""

import math
import warnings

import numpy as np
import pytest

from pystarc.simulation.step_near_surface import step_near_absorbing_surface


class _StubRNG:
    """Deterministic stand-in for numpy's Generator exposing only random().

    The first draw selects between survival and absorption; every later draw
    returns a fixed value so that the rejection-sampling acceptance test never
    succeeds, exercising the attempt-cap path.
    """

    def __init__(self, first: float, rest: float):
        self._first = first
        self._rest = rest
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self._first if self.calls == 1 else self._rest


def _psurv(x0: float, F: float) -> float:
    b = -F
    tau = x0 * x0
    st2 = 2.0 * math.sqrt(tau)
    bt = b * tau
    erfmt = math.erf((x0 - bt) / st2)
    erfpt = math.erf((x0 + bt) / st2)
    p = 0.5 * (math.exp(b * x0) * (erfpt - 1.0) + erfmt + 1.0)
    return max(0.0, min(1.0, p))


def test_normal_path_is_deterministic_and_unchanged():
    """Fixed seeds reproduce the established zero-force absorbing-surface step outputs exactly."""
    reference = {
        0: [
            (False, 0.0, 6.7446678441),
            (True, 12.6390963248, 25.0),
            (False, 0.0, 4.3913905151),
            (False, 0.0, 7.4927972634),
            (True, 9.3712826847, 25.0),
        ],
        1: [
            (True, 17.5759728873, 25.0),
            (False, 0.0, 7.7957863003),
            (False, 0.0, 10.2299784092),
            (True, 11.2759206018, 25.0),
            (True, 12.0073043847, 25.0),
        ],
        7: [
            (False, 0.0, 5.6301797498),
            (False, 0.0, 7.5758106705),
            (True, 6.3935978972, 25.0),
            (False, 0.0, 5.3827174559),
            (False, 0.0, 1.098550199),
        ],
    }
    for seed, expected in reference.items():
        rng = np.random.default_rng(seed)
        for exp_survives, exp_x, exp_t in expected:
            survives, new_x, time = step_near_absorbing_surface(rng, 5.0, 0.0, 1.0)
            assert survives == exp_survives
            assert new_x == pytest.approx(exp_x, abs=1e-8)
            assert time == pytest.approx(exp_t, abs=1e-8)


def test_normal_path_nonzero_force_unchanged():
    """A nonzero force reproduces its established absorbing-surface step outputs exactly."""
    expected = [
        (False, 0.0, 3.1381561308),
        (False, 0.0, 0.576511347),
        (True, 11.2645266541, 4.5),
        (False, 0.0, 1.9953638947),
        (True, 6.7743307886, 4.5),
    ]
    rng = np.random.default_rng(42)
    for exp_survives, exp_x, exp_t in expected:
        survives, new_x, time = step_near_absorbing_surface(rng, 3.0, 0.2, 2.0)
        assert survives == exp_survives
        assert new_x == pytest.approx(exp_x, abs=1e-8)
        assert time == pytest.approx(exp_t, abs=1e-8)


def test_normal_path_emits_no_warning():
    """Healthy sampling over many absorbing-surface steps never trips the rejection-sampling warning."""
    rng = np.random.default_rng(2024)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2000):
            step_near_absorbing_surface(rng, 4.0, 0.1, 1.5)


def test_survival_fraction_matches_probability():
    """The empirical survival fraction matches the analytic survival probability P_surv(x0, F)."""
    x0, F, D = 4.0, 0.1, 1.5
    expected = _psurv(x0, F)
    rng = np.random.default_rng(2024)
    n = 20000
    survived = sum(
        1 for _ in range(n) if step_near_absorbing_surface(rng, x0, F, D)[0]
    )
    assert survived / n == pytest.approx(expected, abs=0.02)


def test_survival_exhaustion_warns_and_returns_valid_position():
    """A degenerate survival draw warns about non-convergence and returns a finite, valid no-flux position rather than the deterministic fallback."""
    rng = _StubRNG(first=0.0, rest=0.999999)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survives, new_x, time = step_near_absorbing_surface(rng, 2.0, -5.0, 1.0)
    messages = [str(w.message) for w in caught]
    assert any("rejection sampling did not converge" in m for m in messages)
    assert survives is True
    assert new_x >= 0.0
    assert math.isfinite(new_x)
    assert math.isfinite(time)
    # The returned position must not be the fixed deterministic fallback; it is
    # the last valid no-flux proposal draw.
    assert new_x != pytest.approx(max(2.0, 0.001), abs=1e-12)


def test_absorption_exhaustion_warns():
    """A degenerate absorption draw warns and returns survival False with the position pinned at the absorbing surface 0."""
    rng = _StubRNG(first=0.9999, rest=0.999999)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survives, new_x, time = step_near_absorbing_surface(rng, 2.0, -5.0, 1.0)
    messages = [str(w.message) for w in caught]
    assert any("rejection" in m for m in messages)
    assert survives is False
    assert new_x == 0.0
    assert math.isfinite(time)
