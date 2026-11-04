"""
Trajectories still running at the end of a run.

PySTARC can leave the unfinished trajectories out of P_rxn, as in the published
runs, or count them as not reacted, as BrownDye2 does, and it reports the range
of P_rxn and k_on between counting them all as not reacted and all as reacted.
The stall rule that stops a run early can be switched off. These tests cover the
result object, the pooled shard counts, the stall test and the configuration.
"""

from __future__ import annotations

import pytest

from pystarc.multi_GPU.combine_data import _p_rxn_bounds, _p_rxn_denominator
from pystarc.pipeline.input_parser import PySTARCConfig, parse
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult, _stall_detected

CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3


def _result(unfinished_in_p_rxn=False, n_react=10, n_escape=980, n_max=10):
    return GPUBatchResult(
        n_trajectories=n_react + n_escape + n_max,
        n_reacted=n_react,
        n_escaped=n_escape,
        n_max_steps=n_max,
        reaction_counts={"stage_0": n_react},
        r_start=10.0,
        r_escape=20.0,
        dt=0.2,
        elapsed_sec=5.0,
        steps_per_sec=100000,
        unfinished_in_p_rxn=unfinished_in_p_rxn,
    )


class TestUnfinishedInResult:
    def test_left_out_by_default(self):
        """By default the unfinished trajectories are left out of P_rxn."""
        r = _result()
        assert r.n_counted == 990
        assert r.reaction_probability == pytest.approx(10 / 990)

    def test_counted_as_not_reacted(self):
        """With the option set they count as not reacted."""
        r = _result(unfinished_in_p_rxn=True)
        assert r.n_counted == 1000
        assert r.reaction_probability == pytest.approx(10 / 1000)

    def test_ci_uses_the_same_denominator(self):
        """Counting 10 unfinished as not reacted gives the CI of 10 more escapes."""
        counted = _result(unfinished_in_p_rxn=True).reaction_probability_ci()
        escaped = _result(n_escape=990, n_max=0).reaction_probability_ci()
        assert counted == pytest.approx(escaped)

    def test_bounds(self):
        """The bounds count every unfinished trajectory as not reacted, then as reacted."""
        for flag in (False, True):
            lo, hi = _result(unfinished_in_p_rxn=flag).reaction_probability_bounds()
            assert lo == pytest.approx(10 / 1000)
            assert hi == pytest.approx(20 / 1000)
        assert _result(n_react=0, n_escape=0, n_max=0).reaction_probability_bounds() == (
            0.0,
            1.0,
        )

    def test_rate_bounds(self):
        """With k_b the k_on bounds are CONV × k_b × the P_rxn bounds."""
        lo, hi = _result().rate_constant_bounds(D_rel=0.1, k_b=30.0)
        assert lo == pytest.approx(CONV * 30.0 * 0.01)
        assert hi == pytest.approx(CONV * 30.0 * 0.02)

    def test_summary_names_the_choice(self):
        """The summary says how the unfinished trajectories were treated and gives the bound."""
        assert "(excluded)" in _result().summary(D_rel=0.1, k_b=30.0)
        s = _result(unfinished_in_p_rxn=True).summary(D_rel=0.1, k_b=30.0)
        assert "(counted as not reacted)" in s
        assert "k_on bound" in s


class TestPooledShards:
    def test_denominator(self):
        """The pooled denominator adds the unfinished trajectories only when asked."""
        assert _p_rxn_denominator(10, 980, 10, False) == 990
        assert _p_rxn_denominator(10, 980, 10, True) == 1000

    def test_bounds(self):
        """The pooled bounds match the single-run bounds."""
        assert _p_rxn_bounds(10, 980, 10) == pytest.approx((0.01, 0.02))
        assert _p_rxn_bounds(0, 0, 0) == (0.0, 1.0)


class TestStallDetected:
    def test_stops_when_nearly_done_and_stuck(self):
        """Over 95% done and fewer than 2 finished since the last check stops the run."""
        assert _stall_detected(1986, 1985, 2000)

    def test_continues_while_progressing(self):
        assert not _stall_detected(1990, 1986, 2000)

    def test_continues_below_95_percent(self):
        assert not _stall_detected(1900, 1900, 2000)


class TestConfig:
    def test_defaults_keep_the_published_behaviour(self):
        """By default the stall rule is on and the unfinished are left out."""
        cfg = PySTARCConfig()
        assert cfg.stall_check is True
        assert cfg.unfinished_in_p_rxn is False

    def test_xml(self, tmp_path):
        """Both settings are read from the input XML."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <stall_check>false</stall_check>
  <unfinished_in_p_rxn>true</unfinished_in_p_rxn>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.stall_check is False
        assert cfg.unfinished_in_p_rxn is True
