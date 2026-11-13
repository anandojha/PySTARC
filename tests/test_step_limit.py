"""
The step limit as the single end of a trajectory other than reaction or escape.

Every trajectory runs until it reacts, escapes or reaches max_steps. A
trajectory at the limit counts as not reacted, as in BrownDye2, and the result
reports the range of P_rxn and k_on between counting every such trajectory as
not reacted and as reacted. There is no stall rule and no switch. These tests
cover the GPU result, the CPU result, the flexible-chain rate block, the pooled
shard counts, the convergence analysis and the input file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pystarc.analysis.convergence import analyse_convergence
from pystarc.molsystem.system_state import Fate
from pystarc.multi_GPU.combine_data import _p_rxn_bounds
from pystarc.pipeline.chain_output_writer import _rate_block
from pystarc.pipeline.input_parser import PySTARCConfig, parse
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult
from pystarc.simulation.nam_simulator import SimulationResult

CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3


def _gpu_result(n_react=10, n_escape=980, n_max=10):
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
    )


def _cpu_result(n_react=10, n_escape=980, n_max=10):
    return SimulationResult(
        n_trajectories=n_react + n_escape + n_max,
        n_reacted=n_react,
        n_escaped=n_escape,
        n_max_steps=n_max,
        reaction_counts={"stage_0": n_react},
        r_start=10.0,
        r_escape=20.0,
        dt=0.2,
        k_db=30.0,
    )


class TestGPUResult:
    def test_counted_as_not_reacted(self):
        """A trajectory at the step limit is in the denominator and not reacted."""
        r = _gpu_result()
        assert r.n_run == 1000
        assert r.n_completed == 990
        assert r.reaction_probability == pytest.approx(10 / 1000)

    def test_no_switch(self):
        """The result carries no setting for the treatment of unfinished trajectories."""
        assert not hasattr(GPUBatchResult, "unfinished_in_p_rxn")
        assert not hasattr(GPUBatchResult, "n_counted")

    def test_ci_matches_escapes(self):
        """Ten trajectories at the limit give the CI of ten more escapes."""
        at_limit = _gpu_result().reaction_probability_ci()
        escaped = _gpu_result(n_escape=990, n_max=0).reaction_probability_ci()
        assert at_limit == pytest.approx(escaped)

    def test_bounds(self):
        """The bound runs from the reported value to every max-steps trajectory reacted."""
        lo, hi = _gpu_result().reaction_probability_bounds()
        assert lo == pytest.approx(10 / 1000)
        assert hi == pytest.approx(20 / 1000)
        assert _gpu_result(0, 0, 0).reaction_probability_bounds() == (0.0, 1.0)

    def test_rate_bounds(self):
        """With k_b the k_on bounds are CONV × k_b × the P_rxn bounds."""
        lo, hi = _gpu_result().rate_constant_bounds(D_rel=0.1, k_b=30.0)
        assert lo == pytest.approx(CONV * 30.0 * 0.01)
        assert hi == pytest.approx(CONV * 30.0 * 0.02)

    def test_all_at_limit_is_zero_without_warning(self):
        """A run in which every trajectory reached the limit has P_rxn = 0 and a defined CI."""
        r = _gpu_result(0, 0, 50)
        assert r.reaction_probability == 0.0
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lo, hi = r.reaction_probability_ci()
        assert lo == pytest.approx(0.0, abs=1e-12) and hi < 0.1

    def test_summary(self):
        """The summary says the max-steps trajectories count as not reacted and gives the bound."""
        s = _gpu_result().summary(D_rel=0.1, k_b=30.0)
        assert "Max-steps    : 10  (counted as not reacted)" in s
        assert "k_on bound" in s
        assert "excluded" not in s


class TestCPUResult:
    def test_counted_as_not_reacted(self):
        r = _cpu_result()
        assert r.n_run == 1000
        assert r.reaction_probability == pytest.approx(10 / 1000)

    def test_ci_matches_escapes(self):
        at_limit = _cpu_result().reaction_probability_ci()
        escaped = _cpu_result(n_escape=990, n_max=0).reaction_probability_ci()
        assert at_limit == pytest.approx(escaped)

    def test_bounds_and_summary(self):
        r = _cpu_result()
        assert r.reaction_probability_bounds() == pytest.approx((0.01, 0.02))
        lo, hi = r.rate_constant_bounds(D_rel=0.1)
        assert lo == pytest.approx(CONV * 30.0 * 0.01)
        assert hi == pytest.approx(CONV * 30.0 * 0.02)
        s = r.summary(D_rel=0.1)
        assert "(counted as not reacted)" in s
        assert "k_on bound" in s


class TestChainRateBlock:
    def test_counted_as_not_reacted(self):
        """The flexible-chain rate block uses the same denominator and bound."""
        sim = SimpleNamespace(
            _outer_prop=SimpleNamespace(k_b=30.0, b_radius=10.0, qradius=20.0, return_prob=0.5),
            params=SimpleNamespace(use_lmz=False),
        )
        results = (
            [SimpleNamespace(fate=Fate.REACTED)] * 10
            + [SimpleNamespace(fate=Fate.ESCAPED)] * 980
            + [SimpleNamespace(fate=Fate.MAX_STEPS)] * 10
        )
        out = _rate_block(sim, results)
        assert out["n_censored"] == 10
        assert out["P_rxn"] == pytest.approx(10 / 1000)
        assert out["P_rxn_bound_low"] == pytest.approx(10 / 1000)
        assert out["P_rxn_bound_high"] == pytest.approx(20 / 1000)
        assert out["k_on_bound_high"] > out["k_on"] == out["k_on_bound_low"]


class TestPooledShards:
    def test_bounds(self):
        """The pooled bounds match the single-run bounds."""
        assert _p_rxn_bounds(10, 980, 10) == pytest.approx((0.01, 0.02))
        assert _p_rxn_bounds(0, 0, 0) == (0.0, 1.0)


class TestConvergence:
    def test_counts_max_steps(self):
        """The convergence analysis counts the max-steps trajectories as not reacted."""
        r = analyse_convergence(n_reacted=10, n_escaped=980, k_b=30.0, n_max_steps=10)
        assert r["N"] == 1000
        assert r["n_max_steps"] == 10
        assert r["P_rxn"] == pytest.approx(0.01)

    def test_default_is_zero(self):
        r = analyse_convergence(n_reacted=10, n_escaped=990, k_b=30.0)
        assert r["N"] == 1000 and r["n_max_steps"] == 0


class TestConfig:
    def test_no_switches(self):
        """The configuration has only the step limit."""
        cfg = PySTARCConfig()
        assert cfg.max_steps == 1000000
        assert not hasattr(cfg, "stall_check")
        assert not hasattr(cfg, "unfinished_in_p_rxn")

    def test_old_tags_warn_and_are_ignored(self, tmp_path):
        """The tags of the earlier stall rule and switch warn and change nothing."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <soft_repulsion>true</soft_repulsion>
  <max_steps>50000</max_steps>
  <stall_check>false</stall_check>
  <unfinished_in_p_rxn>true</unfinished_in_p_rxn>
</pystarc>"""
        )
        with pytest.warns(UserWarning, match="no longer used") as rec:
            cfg = parse(xml)
        assert len(rec) == 2
        assert cfg.max_steps == 50000
        assert not hasattr(cfg, "stall_check")
