"""
Focused test for the cosmetic simplification of the verbose progress guard in
NAMSimulator._run_serial. The original condition was

    if self.params.verbose and i % 1 == 0:

The term i % 1 == 0 is always True, so the guard reduces to self.params.verbose.
These tests confirm the simplified guard preserves the original behavior on both
the verbose and non-verbose paths, namely that exactly one progress line is
printed per trajectory when verbose is True, and no lines are printed when
verbose is False. The NAMSimulator is built without its heavy constructor by
using object.__new__ and stubbing run_one and _record, so the test needs no GPU,
APBS, or external binaries.
"""

import io
from contextlib import redirect_stdout

from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters


def _make_sim(verbose: bool, n: int):
    sim = object.__new__(NAMSimulator)
    sim.params = NAMParameters(n_trajectories=n, verbose=verbose)
    sim.n_reacted = 0
    sim.n_escaped = 0
    recorded = []
    sim.run_one = lambda: "traj"
    sim._record = lambda result: recorded.append(result)
    return sim, recorded


def test_verbose_prints_one_line_per_trajectory():
    n = 5
    sim, recorded = _make_sim(verbose=True, n=n)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sim._run_serial(n)
    lines = [ln for ln in buf.getvalue().splitlines() if "Trajectory" in ln]
    assert len(lines) == n
    # Every trajectory must be recorded regardless of the guard.
    assert len(recorded) == n


def test_non_verbose_prints_nothing():
    n = 4
    sim, recorded = _make_sim(verbose=False, n=n)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sim._run_serial(n)
    assert buf.getvalue() == ""
    assert len(recorded) == n
