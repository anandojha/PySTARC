"""Figures from the analysis files of a finished run.

Every figure reads only the small summary files of a run directory, so a
synthetic directory with those files exercises the whole module. One case has
reactions, one has none, and one lacks optional files, which must be skipped
with a note rather than fail.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from pystarc.analysis.plots import FIGURES, RunFigures, plot_run, read_csv_columns, wilson
from pystarc.cli.main import cli


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def make_run(tmp: Path, n_react: int = 40, n_total: int = 400, with_optional: bool = True) -> Path:
    run = tmp / "bd_sims"
    run.mkdir()
    rng = np.random.default_rng(1)
    r_start, r_escape = 40.0, 80.0
    k_b = 25.0
    p = n_react / n_total
    json.dump(
        {"n_trajectories": n_total, "n_reacted": n_react, "P_rxn": p, "k_b": k_b,
         "k_on": k_b * 6.02214076e8 * p, "r_start": r_start, "r_escape": r_escape},
        open(run / "results.json", "w"),
    )
    reacted = np.zeros(n_total, dtype=bool)
    reacted[rng.choice(n_total, size=n_react, replace=False)] = True
    rows = []
    for i in range(n_total):
        rows.append([i, "reacted" if reacted[i] else "escaped", int(rng.integers(50, 5000)), 1000.0])
    _write_csv(run / "trajectories.csv", ["traj_id", "outcome", "n_steps", "total_time_ps"], rows)
    r = np.arange(0.5, 100, 1.0)
    dens = np.exp(-((r - 25) / 6) ** 2) * 1e-5 + 1e-7 * (r < r_escape)
    _write_csv(run / "radial_density.csv", ["r_center", "r_low", "r_high", "count", "density"],
               [[rc, rc - 0.5, rc + 0.5, int(d * 1e9), d] for rc, d in zip(r, dens)])
    react_ids = np.nonzero(reacted)[0]
    _write_csv(run / "fpt_distribution.csv", ["traj_id", "first_passage_time_ps", "min_distance"],
               [[int(i), float(rng.exponential(20000)), 2.0] for i in react_ids])
    (run / "receptor.pqr").write_text(
        "ATOM      1  OG1 THR A 196      1.0  2.0  3.0 -0.6 1.5\n"
        "ATOM      2  NE2 HIS A  91      1.0  2.0  3.0 -0.3 1.55\n"
        "ATOM      3  O   SER A 192      1.0  2.0  3.0 -0.5 1.5\n"
    )
    if with_optional:
        q = rng.normal(size=(n_react, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        _write_csv(run / "encounters.csv", ["traj_id", "step", "x", "y", "z", "q0", "q1", "q2", "q3", "n_pairs_satisfied"],
                   [[int(i), 10, *rng.normal(size=3), *q[k], 2] for k, i in enumerate(react_ids)])
    return run


def test_wilson_matches_known_values():
    p, lo, hi = wilson(np.array([0, 5, 50]), np.array([100, 100, 100]))
    assert p[0] == 0 and lo[0] == 0 and 0.03 < hi[0] < 0.04
    assert abs(p[1] - 0.05) < 1e-12 and 0.02 < lo[1] < 0.03 and 0.11 < hi[1] < 0.12
    assert 0.40 < lo[2] < 0.41 and 0.59 < hi[2] < 0.60


def test_read_csv_columns_numeric_and_text(tmp_path):
    p = tmp_path / "t.csv"
    _write_csv(p, ["a", "b"], [[1, "x"], [2, "y"]])
    cols = read_csv_columns(p)
    assert cols["a"].dtype.kind == "f" and list(cols["b"]) == ["x", "y"]


def test_every_figure_written_with_reactions(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "figs"
    status = plot_run(run, out, dpi=60)
    assert set(status) == set(FIGURES)
    assert all(v == "ok" for v in status.values()), status
    for name in FIGURES:
        assert (out / f"fig_{name}.png").stat().st_size > 0
    assert not (out / "overview.png").exists()


def test_run_without_reactions_still_plots(tmp_path):
    run = make_run(tmp_path, n_react=0, with_optional=False)
    status = plot_run(run, tmp_path / "figs", dpi=60)
    assert status["rate"] == "ok"
    assert status["radial"] == "ok"
    assert status["first_passage"] == "note: no reactions"
    assert status["steps"] == "note: no reactions, escaped only"
    assert status["first_hitting"].startswith("skipped: no reactions")
    assert (tmp_path / "figs" / "fig_rate.png").exists()
    assert (tmp_path / "figs" / "fig_first_passage.png").exists()
    assert not (tmp_path / "figs" / "fig_first_hitting.png").exists()


def test_missing_optional_file_is_skipped(tmp_path):
    run = make_run(tmp_path)
    (run / "receptor.pqr").unlink()
    status = plot_run(run, tmp_path / "figs", dpi=60)
    assert status["first_hitting"] == "skipped: receptor.pqr missing"
    assert not (tmp_path / "figs" / "fig_first_hitting.png").exists()
    assert status["rate"] == "ok"


def test_rate_scale_falls_back_to_kon_over_prxn(tmp_path):
    run = make_run(tmp_path)
    res = json.load(open(run / "results.json"))
    del res["k_b"]
    json.dump(res, open(run / "results.json", "w"))
    rf = RunFigures(run)
    assert rf._rate_scale() == pytest.approx(res["k_on"] / res["P_rxn"])


def test_cli_plot_command(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "cli_figs"
    result = CliRunner().invoke(cli, ["plot", str(run), "--out", str(out), "--dpi", "60", "--only", "rate,first_passage"])
    assert result.exit_code == 0, result.output
    assert (out / "fig_rate.png").exists() and (out / "fig_first_passage.png").exists()
    assert not (out / "fig_radial.png").exists()
    bad = CliRunner().invoke(cli, ["plot", str(run), "--only", "nope"])
    assert bad.exit_code != 0


# ---- the axis number rule shared by every figure ---------------------------------------------
from pystarc.analysis.plots import _ends_even, even_ticks, log_powers  # noqa: E402


@pytest.mark.parametrize("top, expected", [
    (5.07, [0, 2, 4, 6]),
    (8.16, [0, 2, 4, 6, 8, 10]),
    (50.0, [0, 10, 20, 30, 40, 50]),
    (0.53, [0.0, 0.2, 0.4, 0.6]),
    (0.0143, [0.0, 0.004, 0.008, 0.012, 0.016]),
])
def test_even_ticks_end_on_first_even_number_at_or_above_data(top, expected):
    ticks = even_ticks(top)
    assert np.allclose(ticks, expected)
    assert 4 <= len(ticks) <= 6 and ticks[-1] >= top


def test_even_ticks_rule_holds_over_many_magnitudes():
    for top in np.geomspace(1e-4, 1e7, 1500):
        ticks = even_ticks(top)
        step = ticks[1] - ticks[0]
        assert 4 <= len(ticks) <= 6, top
        assert ticks[0] == 0 and ticks[-1] >= top * (1 - 1e-9), top
        assert _ends_even(ticks[-1], step), top
        assert ticks[-1] - step < top, top


@pytest.mark.parametrize("lo, hi, expected", [
    (1, 5e6, [0, 2, 4, 6, 8]),
    (1, 1e5, [0, 2, 4, 6]),
    (1, 1e6, [0, 2, 4, 6]),
    (3e-5, 2e2, [-6, -4, -2, 0, 2, 4]),
    (0.05, 3.0, [-2, -1, 0, 1, 2]),
    (1e-16, 1.0, [-16, -12, -8, -4, 0]),
    (1, 100, None),
])
def test_log_powers_start_and_end_on_even_powers(lo, hi, expected):
    assert log_powers(lo, hi) == expected


def test_log_powers_rule_holds_over_many_ranges():
    for lo in (1e-12, 1e-7, 0.03, 1.0, 7.0):
        for hi in np.geomspace(lo * 1.5, lo * 1e14, 300):
            powers = log_powers(lo, hi)
            if powers is None:
                continue
            assert 4 <= len(powers) <= 6
            assert powers[0] % 2 == 0 and powers[-1] % 2 == 0
            assert 10.0 ** powers[0] <= lo * (1 + 1e-9) and 10.0 ** powers[-1] >= hi * (1 - 1e-9)


def test_rate_axis_starts_at_one_trajectory(tmp_path):
    import matplotlib.pyplot as plt
    run = make_run(tmp_path, n_react=40, n_total=2000)
    fig, ax = plt.subplots()
    RunFigures(run).draw_rate(ax)
    majors = list(ax.xaxis.get_major_locator()())
    lo, hi = ax.get_xlim()
    scale = ax.get_xscale()
    plt.close(fig)
    assert scale == "log" and majors[0] == 1.0 and majors[-1] == 1e4
    assert lo < 1.0 and 1e4 < hi < 1.2e4


def test_radial_profile_stops_at_escape_radius(tmp_path):
    import matplotlib.pyplot as plt
    run = make_run(tmp_path)
    rf = RunFigures(run)
    r = read_csv_columns(run / "radial_density.csv")["r_center"]
    rf.r_escape = float(np.median(r))
    fig, ax = plt.subplots()
    assert rf.draw_radial(ax) is None
    x_drawn = ax.lines[0].get_xdata()
    plt.close(fig)
    assert x_drawn.max() <= rf.r_escape


def test_two_dimensional_figures_share_one_page_size(tmp_path):
    import matplotlib.image as mpimg
    run = make_run(tmp_path)
    plot_run(run, tmp_path / "figs", dpi=60, which=("rate", "radial", "first_passage", "steps"))
    shapes = {mpimg.imread(f).shape for f in (tmp_path / "figs").glob("fig_*.png")}
    assert len(list((tmp_path / "figs").glob("fig_*.png"))) == 4
    assert len(shapes) == 1


def test_fine_bins_round_width_and_count():
    from pystarc.analysis.plots import fine_bins
    rng = np.random.default_rng(1)
    for n, scale in ((266, 30.0), (50, 3.0), (5000, 200.0)):
        x = rng.exponential(scale, n)
        e = fine_bins(x)
        w = e[1] - e[0]
        assert e[0] == 0.0 and e[-1] >= x.max()
        assert np.allclose(np.diff(e), w)
        mant = w / 10.0 ** np.floor(np.log10(w))
        assert any(np.isclose(mant, m) for m in (1.0, 2.0, 2.5, 4.0, 5.0))
        assert 15 <= len(e) - 1 <= 75


def test_steps_shows_each_outcome_as_its_own_density(tmp_path):
    import matplotlib.pyplot as plt
    run = make_run(tmp_path)
    fig, ax = plt.subplots()
    assert RunFigures(run).draw_steps(ax) is None
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    scale = ax.get_xscale()
    areas = []
    for bars in ax.containers:
        lo = np.array([b.get_x() for b in bars])
        hi = lo + np.array([b.get_width() for b in bars])
        areas.append(float(np.sum(np.array([b.get_height() for b in bars]) * np.log10(hi / lo))))
    plt.close(fig)
    assert labels == ["Escaped trajectories", "Reactive trajectories"]
    assert scale == "log"
    assert np.allclose(areas, 1.0)
