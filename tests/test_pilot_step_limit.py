"""
The pilot check that chooses max_steps before a production run.

The recommended limit is the smallest multiple of the granularity at which the
fraction of trajectories still running, S(max_steps), is at most tol × P_rxn,
so that the trajectories at the limit cannot move k_on by more than tol.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pystarc.analysis.step_limit import (
    format_report,
    main,
    recommend_max_steps,
    survival_curve,
)

MEAN = 2000.0  # Mean trajectory length of the synthetic pilot, in steps.


def _exponential_pilot(n=20000, mean=MEAN, p_react=0.05, seed=1, cap=None):
    """Lengths drawn from an exponential, outcomes reacted with probability p_react.
    With cap, lengths above it are censored at the cap and reported as max_steps."""
    rng = np.random.default_rng(seed)
    lengths = np.maximum(1, rng.exponential(mean, n).astype(int))
    outcomes = np.where(rng.random(n) < p_react, "reacted", "escaped").astype(object)
    if cap is not None:
        over = lengths > cap
        lengths = np.where(over, cap, lengths)
        outcomes[over] = "max_steps"
    return lengths.tolist(), outcomes.tolist()


def _exact_cap(p, tol=0.01, mean=MEAN, g=10_000):
    """The smallest multiple of g with exp(-cap/mean) <= tol p."""
    n_star = mean * math.log(1.0 / (tol * p))
    return int(math.ceil(n_star / g) * g)


class TestSurvivalCurve:
    def test_values(self):
        lengths, S = survival_curve([5, 1, 3, 3, 9])
        assert lengths.tolist() == [1, 3, 5, 9]
        assert S.tolist() == pytest.approx([0.8, 0.4, 0.2, 0.0])

    def test_empty(self):
        lengths, S = survival_curve([])
        assert lengths.size == 0 and S.size == 0


class TestRecommendation:
    def test_exponential_pilot(self):
        """With a fully observed exponential tail the limit is the analytic one."""
        n_steps, outcomes = _exponential_pilot()
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.05)
        assert rec["max_steps"] == _exact_cap(0.05)
        assert rec["meets_tolerance"]
        assert rec["survival_at_max_steps"] <= rec["target_survival"]
        assert rec["max_steps"] % 10_000 == 0

    def test_p_rxn_from_pilot(self):
        """Without p_rxn the pilot's own reacted fraction is used."""
        n_steps, outcomes = _exponential_pilot(p_react=0.05)
        rec = recommend_max_steps(n_steps, outcomes)
        assert rec["p_rxn_source"] == "pilot"
        assert rec["p_rxn"] == pytest.approx(outcomes.count("reacted") / len(outcomes))

    def test_censored_pilot_extrapolates(self):
        """Lengths censored at a short pilot limit are extrapolated with the tail fit."""
        n_steps, outcomes = _exponential_pilot(cap=12_000)
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.05)
        assert rec["n_at_limit"] > 0 and rec["pilot_cap"] == 12_000
        assert rec["method"] == "extrapolated"
        assert rec["tail_fit"]["tau_steps"] == pytest.approx(MEAN, rel=0.3)
        assert rec["max_steps"] == _exact_cap(0.05)
        assert rec["meets_tolerance"]

    def test_below_resolution_uses_fit_and_warns(self):
        """A small pilot cannot observe the target survival; the fit is used and said so."""
        n_steps, outcomes = _exponential_pilot(n=1000, seed=3)
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.01)
        assert rec["target_survival"] < rec["resolution"]
        assert rec["method"] == "extrapolated"
        assert any("resolution" in w for w in rec["warnings"])
        assert rec["max_steps"] == pytest.approx(_exact_cap(0.01), abs=10_000)

    def test_fast_pilot_gives_the_minimum(self):
        """When every trajectory finishes early the limit is one granularity."""
        n_steps = [10, 20, 30, 400, 50] * 100
        outcomes = ["reacted"] + ["escaped"] * 499
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.1)
        assert rec["max_steps"] == 10_000
        assert rec["meets_tolerance"]

    def test_trapped_tail(self):
        """Trajectories at the pilot limit with no finish in the last 20% are reported as stuck."""
        n_steps, outcomes = _exponential_pilot(n=2000, cap=100_000)
        n_steps = n_steps + [100_000] * 20
        outcomes = outcomes + ["max_steps"] * 20
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.01, pilot_cap=100_000)
        assert rec["tail_trapped"]
        assert not rec["meets_tolerance"]
        assert rec["max_steps"] == rec["ceiling"]
        assert rec["method"] == "trapped"
        assert any("stuck" in w for w in rec["warnings"])

    def test_no_reaction_needs_p_rxn(self):
        n_steps = [100] * 50
        outcomes = ["escaped"] * 50
        with pytest.raises(ValueError, match="no reaction"):
            recommend_max_steps(n_steps, outcomes)
        rec = recommend_max_steps(n_steps, outcomes, p_rxn=0.01)
        assert rec["max_steps"] == 10_000

    def test_ceiling_and_granularity(self):
        n_steps, outcomes = _exponential_pilot(n=2000, mean=50_000, seed=5)
        rec = recommend_max_steps(
            n_steps, outcomes, p_rxn=1e-3, granularity=5_000, ceiling=200_000
        )
        assert rec["max_steps"] == 200_000
        assert not rec["meets_tolerance"]
        rec2 = recommend_max_steps(n_steps, outcomes, p_rxn=0.5, granularity=5_000)
        assert rec2["max_steps"] % 5_000 == 0 and rec2["meets_tolerance"]

    def test_outcome_codes(self):
        """Integer outcome codes are accepted."""
        n_steps = [100, 200, 300, 400]
        rec = recommend_max_steps(n_steps, [1, 2, 2, 2])
        assert rec["n_reacted"] == 1 and rec["n_escaped"] == 3

    def test_bad_input(self):
        with pytest.raises(ValueError, match="unknown outcome"):
            recommend_max_steps([1, 2], ["reacted", "lost"])
        with pytest.raises(ValueError):
            recommend_max_steps([], [])
        with pytest.raises(ValueError):
            recommend_max_steps([1], ["reacted"], tol=0)


class TestCLI:
    def test_report_and_exit_code(self, tmp_path, capsys):
        n_steps, outcomes = _exponential_pilot(n=5000, seed=7)
        csv_path = tmp_path / "trajectories.csv"
        with open(csv_path, "w") as f:
            f.write("traj_id,outcome,n_steps\n")
            for i, (o, s) in enumerate(zip(outcomes, n_steps)):
                f.write(f"{i},{o},{s}\n")
        out_json = tmp_path / "step_limit.json"
        code = main([str(csv_path), "--p-rxn", "0.05", "--json", str(out_json)])
        text = capsys.readouterr().out
        assert code == 0
        assert "max_steps" in text and "Survival S(n)" in text
        assert out_json.exists()

    def test_missing_columns(self, tmp_path, capsys):
        p = tmp_path / "t.csv"
        p.write_text("a,b\n1,2\n")
        assert main([str(p)]) == 2
        assert "expected the columns" in capsys.readouterr().out

    def test_format_report_mentions_warnings(self):
        n_steps = [100] * 50
        rec = recommend_max_steps(n_steps, ["escaped"] * 50, p_rxn=0.01)
        rec["warnings"].append("a warning")
        assert "Warning: a warning" in format_report(rec, n_steps)
