"""Tests for the cross-shard consistency check in combine_data.

These exercise _warn_run_mismatch, which warns when shards report different
values for the physical parameters (k_b, D_rel, r_start, r_escape) that are
pooled assuming an identical system. The function does not require a GPU, APBS,
or any external binary.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from pystarc.multi_GPU.combine_data import _warn_run_mismatch


def _base_run():
    return {"k_b": 1.5, "D_rel": 0.3, "r_start": 10.0, "r_escape": 50.0}


def test_consistent_runs_emit_no_warning(capsys):
    runs = [_base_run(), _base_run(), _base_run()]
    _warn_run_mismatch(runs)
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_single_run_emits_no_warning(capsys):
    _warn_run_mismatch([_base_run()])
    out = capsys.readouterr().out
    assert out == ""


def test_kb_mismatch_warns(capsys):
    r2 = _base_run()
    r2["k_b"] = 2.0
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "k_b" in out


def test_geometry_mismatch_warns(capsys):
    r2 = _base_run()
    r2["r_escape"] = 60.0
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "r_escape" in out


def test_tiny_float_noise_does_not_warn(capsys):
    r2 = _base_run()
    r2["k_b"] = 1.5 + 1e-13
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_missing_value_in_later_shard_is_skipped(capsys):
    r2 = _base_run()
    del r2["D_rel"]
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "D_rel" not in out
