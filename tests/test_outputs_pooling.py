"""The BD time file, results.json, the multi-GPU split and the pooled outputs."""

import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pystarc.multi_GPU import combine_data, multi_GPU_runs
from pystarc.pipeline.input_parser import OutputConfig
from pystarc.pipeline.output_writer import write_all
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult

PKG = Path(__file__).resolve().parent.parent / "pystarc"


def _rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _outputs(**on):
    off = {k: False for k in OutputConfig.__dataclass_fields__ if k != "save_interval"}
    return OutputConfig(**{**off, **on})


def test_bd_time_file_has_the_returns_and_results_json_the_tolerance(tmp_path):
    res = GPUBatchResult(n_trajectories=4, n_reacted=2, n_escaped=2, n_max_steps=0,
                         reaction_counts={}, r_start=10.0, r_escape=20.0, dt=0.2,
                         elapsed_sec=1.0, steps_per_sec=1.0)
    sim = {"outcome": np.array([1, 2, 1, 2]), "total_time_ps": np.array([12.5, 3.0, 40.25, 1.0]),
           "n_returns": np.array([0, 1, 3, 0]), "min_dist": np.array([1.5, 9.0, 2.25, 8.0])}
    write_all(tmp_path, res, sim, _outputs(results_json=True, bd_time_to_reaction=True), 3.0, 0.05,
              confidence=0.9, convergence_tol=0.02)
    text = (tmp_path / "bd_time_to_reaction.csv").read_text()
    assert text == "traj_id,bd_time_ps,n_returns,min_distance\n0,12.5000,0,1.500000\n2,40.2500,3,2.250000\n"
    rj = json.loads((tmp_path / "results.json").read_text())
    assert rj["convergence_tol"] == 0.02 and rj["confidence_level"] == 0.9
    assert not (tmp_path / "fpt_distribution.csv").exists()


def test_first_passage_wording_is_left_to_chain_files():
    hits = [p.name for p in PKG.rglob("*.py") if "chain" not in p.name
            and re.search(r"first[ -]passage", p.read_text(), re.I)]
    assert hits == []


def _shard(bd, name, n_traj, n_reacted, ids=(), conf=0.95, tol=0.05, rxns=None):
    d = bd / name
    d.mkdir(parents=True)
    rj = {"n_reacted": n_reacted, "n_escaped": n_traj - n_reacted, "n_max_steps": 0,
          "n_trajectories": n_traj, "k_b": 2.0, "D_rel": 0.05, "r_start": 10.0, "r_escape": 20.0,
          "confidence_level": conf, "convergence_tol": tol}
    if rxns:
        rj["completed_reactions"] = rxns
    (d / "results.json").write_text(json.dumps(rj))
    if ids:
        with open(d / "pose_clusters.csv", "w") as f:
            f.write("traj_id,cluster,angle_rad,q0,q1,q2,q3\n")
            f.writelines(f"{i},0,0.1,1,0,0,0\n" for i in ids)
        with open(d / "bd_time_to_reaction.csv", "w") as f:
            f.write("traj_id,bd_time_ps,n_returns,min_distance\n")
            f.writelines(f"{i},1.0,0,2.0\n" for i in ids)


def _combine(bd, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["combine_data.py", "--bd-sims", str(bd)])
    return combine_data.main()


def test_pooled_pose_ids_follow_the_shard_offsets(tmp_path, monkeypatch):
    bd = tmp_path / "bd_sims"
    _shard(bd, "bd_1", 5, 2, ids=(0, 3))
    _shard(bd, "bd_2", 5, 2, ids=(0, 2))
    assert _combine(bd, monkeypatch) == 0
    assert [r["traj_id"] for r in _rows(bd / "pose_clusters.csv")] == ["0", "3", "5", "7"]
    assert [r["traj_id"] for r in _rows(bd / "bd_time_to_reaction.csv")] == ["0", "3", "5", "7"]
    rj = json.loads((bd / "results.json").read_text())
    assert rj["log10_k_on"] == pytest.approx(math.log10(rj["k_on"]), rel=1e-15)


def test_pooled_run_without_reactions_has_no_log10(tmp_path, monkeypatch):
    bd = tmp_path / "bd_sims"
    _shard(bd, "bd_1", 5, 0)
    _shard(bd, "bd_2", 5, 0)
    assert _combine(bd, monkeypatch) == 0
    assert "log10_k_on" not in json.loads((bd / "results.json").read_text())


def test_pooled_level_and_tolerance_come_from_the_shards(tmp_path, monkeypatch, capsys):
    bd = tmp_path / "bd_sims"
    _shard(bd, "bd_1", 400, 40, conf=0.9, tol=0.2)
    _shard(bd, "bd_2", 400, 40, conf=0.9, tol=0.2)
    assert _combine(bd, monkeypatch) == 0
    out = capsys.readouterr().out
    rj = json.loads((bd / "results.json").read_text())
    conv = json.loads((bd / "convergence.json").read_text())
    z = statistics.NormalDist().inv_cdf(0.95)
    P, n = 0.1, 800
    c = (P + z * z / (2 * n)) / (1 + z * z / n)
    assert rj["P_rxn_low"] == pytest.approx(c - z * math.sqrt(P * (1 - P) / n + z * z / (4 * n * n)) / (1 + z * z / n))
    assert "Wilson 90% CI" in out and conv["converged"] and conv["tol"] == 0.2
    assert rj["confidence_level"] == 0.9 and conv["confidence_level"] == 0.9


def test_reactions_pool_by_name_and_states(tmp_path, monkeypatch):
    bd = tmp_path / "bd_sims"
    a = {"name": "r", "n_fired": 2, "state_before": "x", "state_after": "y"}
    b = {"name": "r", "n_fired": 3, "state_before": "y", "state_after": "z"}
    _shard(bd, "bd_1", 5, 1, rxns=[a, b])
    _shard(bd, "bd_2", 5, 1, rxns=[a])
    assert _combine(bd, monkeypatch) == 0
    pooled = json.loads((bd / "results.json").read_text())["completed_reactions"]
    assert [(r["state_before"], r["n_fired"]) for r in pooled] == [("x", 4), ("y", 3)]


def test_combiner_pools_the_split_and_stops_on_a_wrong_shard(tmp_path, monkeypatch):
    bd = tmp_path / "bd_sims"
    for i, n in ((1, 3), (2, 2), (3, 9)):
        _shard(bd, f"bd_{i}", n, 1)
    shards = [{"dir": "bd_1", "seed": 1, "n_trajectories": 3}, {"dir": "bd_2", "seed": 2, "n_trajectories": 2}]
    (bd / "split.json").write_text(json.dumps({"n_splits": 2, "n_trajectories": 5, "shards": shards}))
    assert _combine(bd, monkeypatch) == 0
    assert json.loads((bd / "results.json").read_text())["n_trajectories"] == 5
    shards[1]["n_trajectories"] = 4
    (bd / "split.json").write_text(json.dumps({"n_splits": 2, "n_trajectories": 7, "shards": shards}))
    assert _combine(bd, monkeypatch) == 1
    assert _combine(tmp_path / "missing", monkeypatch) == 1


def _split(tmp_path, monkeypatch, n_splits, n_traj=10, extra=""):
    (tmp_path / "input.xml").write_text(
        f"<pystarc><n_trajectories>{n_traj}</n_trajectories><seed>5</seed><pdb>x.pdb</pdb>"
        f"<rxns_xml>rxns.xml</rxns_xml><work_dir>out</work_dir>{extra}</pystarc>")
    calls = []

    def grid_pass(cmd, cwd):
        calls.append(cmd)
        out = Path(cwd) / "out"
        out.mkdir(exist_ok=True)
        for f in ("receptor0.dx", "receptor_grids.key", "receptor.pqr", "r.cache", "results.json"):
            (out / f).write_text(f)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(multi_GPU_runs.subprocess, "run", grid_pass)
    monkeypatch.setattr(sys, "argv", ["multi_GPU_runs.py", str(tmp_path / "input.xml"), "--n-splits", str(n_splits)])
    return multi_GPU_runs.main(), calls


def test_split_keeps_every_trajectory_and_writes_split_json(tmp_path, monkeypatch):
    ret, calls = _split(tmp_path, monkeypatch, 4)
    out = tmp_path / "out"
    split = json.loads((out / "split.json").read_text())
    assert ret == 0 and len(calls) == 1
    assert [s["n_trajectories"] for s in split["shards"]] == [3, 3, 2, 2]
    assert [s["seed"] for s in split["shards"]] == [5 + 11111111 * i for i in (1, 2, 3, 4)]
    assert not (out / "results.json").exists()
    shard = out / "bd_3"
    assert sorted(f for f in os.listdir(shard) if os.path.islink(shard / f)) == ["r.cache", "receptor0.dx", "receptor_grids.key"]
    xml = (shard / "input.xml").read_text()
    assert f"<pdb>{tmp_path / 'x.pdb'}</pdb>" in xml and "<n_trajectories>2</n_trajectories>" in xml


@pytest.mark.parametrize("n_splits", [0, 11])
def test_split_count_must_lie_between_1_and_n(tmp_path, monkeypatch, n_splits):
    with pytest.raises(SystemExit):
        _split(tmp_path, monkeypatch, n_splits)


def test_split_refuses_a_folder_with_an_earlier_run(tmp_path, monkeypatch):
    (tmp_path / "out" / "bd_2").mkdir(parents=True)
    (tmp_path / "out" / "bd_2" / "results.json").write_text("{}")
    ret, calls = _split(tmp_path, monkeypatch, 2)
    assert ret == 1 and calls == []


def test_convergence_report_carries_its_level(capsys):
    from pystarc.analysis.convergence import analyse_convergence, print_convergence

    res = analyse_convergence(n_reacted=50, n_escaped=950, k_b=2.0, tol=0.025, confidence=0.9)
    print_convergence(res)
    out = capsys.readouterr().out
    assert res["confidence_level"] == 0.9 and "Wilson 90% CI" in out and "2.5% threshold" in out
