"""
Tests for the low-severity dead-code removal in analyse_convergence.

The unreachable N==0 branch inside the Wilson confidence interval block was
removed. These tests confirm that the N==0 case still early-returns the same
dictionary as before and that a healthy-path call still produces a valid
Wilson interval.
"""

import math

from pystarc.analysis.convergence import analyse_convergence


def test_n_zero_early_returns():
    """With no completed trajectories the function returns the early-return dict
    and never reaches the Wilson interval code."""
    result = analyse_convergence(n_reacted=0, n_escaped=0, k_b=1.0)
    assert result == {"converged": False, "reason": "no completed trajectories"}


def test_healthy_path_wilson_interval_unchanged():
    """A normal call with reactions and escapes still yields a Wilson interval
    in [0, 1] with lo <= hi, confirming the surviving branch computes correctly."""
    result = analyse_convergence(n_reacted=40, n_escaped=60, k_b=2.0)
    assert result["N"] == 100
    assert math.isclose(result["P_rxn"], 0.4)
    lo, hi = result["wilson_CI_P"]
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < result["P_rxn"] < hi
