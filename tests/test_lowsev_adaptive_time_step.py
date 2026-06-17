"""
Tests for the low-severity dead-code cleanup in pystarc.motion.adaptive_time_step.

The size constraint in max_time_step previously carried a redundant
"if D_rel > 0 else _LARGE" ternary. The function already returns early when
D_rel <= 0, so that branch was unreachable. These tests confirm the early
return still guards non-physical inputs and that the healthy-path size
constraint is computed as 4 r_min^2 / D_rel.
"""

import math

from pystarc.motion.adaptive_time_step import max_time_step, _LARGE


def test_non_physical_inputs_take_safe_default():
    # D_rel <= 0 must short-circuit to the safe default before the size term.
    assert max_time_step(10.0, 0.0, 1.0, 5.0, 6.0) == 0.2
    assert max_time_step(10.0, -1.0, 1.0, 5.0, 6.0) == 0.2
    # r <= 0 also short-circuits.
    assert max_time_step(0.0, 1.0, 1.0, 5.0, 6.0) == 0.2


def test_size_constraint_is_the_minimum_when_it_dominates():
    # Choose values so the size constraint is the smallest of the three and
    # confirm it equals 4 r_min^2 / D_rel with no _LARGE fallback involved.
    r = 100.0
    D_rel = 1.0
    D_rot = 1.0e30  # makes dt_rot tiny only if large; here keep rotational term huge
    r_hydro1, r_hydro2 = 1.0, 2.0
    # With these numbers dt_pair = 0.01/2 * 100^2 / 1 = 50, dt_rot = pi^2/1e30 ~ 0,
    # so pick D_rot small instead to isolate the size term.
    D_rot = 1.0e-30
    dt = max_time_step(r, D_rel, D_rot, r_hydro1, r_hydro2)
    expected_size = 4.0 * min(r_hydro1, r_hydro2) ** 2 / D_rel
    dt_pair = (0.1 ** 2 / 2.0) * r ** 2 / D_rel
    dt_rot = math.pi ** 2 / D_rot
    assert dt == min(dt_pair, dt_rot, expected_size)
    assert dt == expected_size
    assert dt != _LARGE


def test_healthy_path_matches_closed_form():
    # General healthy case: result is the min of the three closed-form terms.
    r, D_rel, D_rot, r_h1, r_h2 = 25.0, 0.3, 0.05, 8.0, 12.0
    dt = max_time_step(r, D_rel, D_rot, r_h1, r_h2)
    dt_pair = (0.1 ** 2 / 2.0) * r ** 2 / D_rel
    dt_rot = math.pi ** 2 / D_rot
    dt_size = 4.0 * min(r_h1, r_h2) ** 2 / D_rel
    assert dt == min(dt_pair, dt_rot, dt_size)
