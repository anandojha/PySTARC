"""Regression tests for the chain BD pipeline orchestrator.

These tests cover three behaviors of pystarc.pipeline.chain_pipeline:

1. A default chain config (auto_diffusion off, D_trans=0, D_rot=0) resolves to
   the documented scalar diffusion defaults (0.1 Å²/ps and 0.01 rad²/ps) and
   constructs the simulator without error.
2. run_chain forwards the parsed OutputConfig to write_chain_results so the
   user's <outputs> flags are honored.
3. _ensure_chain_apbs_grids raises a clear error when born_grid_dx is set but
   target_grid_dx is left empty, rather than letting a downstream file open
   fail with FileNotFoundError.
"""

import types

import numpy as np
import pytest

from pystarc.pipeline import chain_pipeline
from pystarc.pipeline.input_parser import (
    ChainConfig,
    OutputConfig,
    PySTARCConfig,
)


class _FakeAtom:
    def __init__(self, radius=1.5):
        self.radius = radius


class _FakeChain:
    def __init__(self, n_atoms=3):
        self.name = "fake_chain"
        self.atoms = [_FakeAtom() for _ in range(n_atoms)]
        self.bonds = []
        self.angles = []
        self.torsions = []


def _make_config(tmp_path, chain_overrides=None, output_overrides=None):
    """Build a PySTARCConfig with a chain block for pipeline testing."""
    chain_kwargs = dict(
        chain_json=str(tmp_path / "chain.json"),
        reaction_pairs_json=str(tmp_path / "reaction_pairs.json"),
    )
    if chain_overrides:
        chain_kwargs.update(chain_overrides)
    chain = ChainConfig(**chain_kwargs)

    outputs = OutputConfig()
    if output_overrides:
        for key, value in output_overrides.items():
            setattr(outputs, key, value)

    cfg = PySTARCConfig(
        work_dir=tmp_path / "work",
        chain=chain,
        outputs=outputs,
    )
    return cfg


def _patch_pipeline_seams(monkeypatch, captured):
    """Replace the heavy run_chain dependencies with light stand-ins.

    The captured dict records the keyword arguments passed to the
    ChainBDSimulator constructor and to write_chain_results so the tests can
    assert on what run_chain forwarded.
    """
    chain = _FakeChain(n_atoms=3)
    body_positions = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )

    monkeypatch.setattr(
        chain_pipeline,
        "load_chain_from_json",
        lambda path: (chain, body_positions),
    )
    monkeypatch.setattr(
        chain_pipeline,
        "parse_pqr",
        lambda path: types.SimpleNamespace(name="target"),
    )
    monkeypatch.setattr(
        chain_pipeline,
        "_load_reaction_pairs_json",
        lambda path: [(0, 0, 4.0), (1, 1, 4.0), (2, 2, 4.0)],
    )
    monkeypatch.setattr(
        chain_pipeline,
        "_ensure_chain_apbs_grids",
        lambda config: None,
    )

    class _FakeSimulator:
        def __init__(self, **kwargs):
            captured["sim_kwargs"] = kwargs

        def run(self):
            return []

    monkeypatch.setattr(chain_pipeline, "ChainBDSimulator", _FakeSimulator)

    def _fake_write(work_dir, sim, results, wall_time_sec=0.0, outputs=None):
        captured["write_outputs"] = outputs
        return []

    monkeypatch.setattr(chain_pipeline, "write_chain_results", _fake_write)


def test_default_config_resolves_diffusion_defaults(tmp_path, monkeypatch):
    """A default chain config with D=0 forwards the scalar defaults D_trans=0.1 and D_rot=0.01 with auto_diffusion off."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(tmp_path)
    # The default chain config leaves auto_diffusion off with zero diffusion.
    assert cfg.chain.auto_diffusion is False
    assert cfg.chain.D_trans == 0.0
    assert cfg.chain.D_rot == 0.0

    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs.get("auto_diffusion") is not True
    assert sim_kwargs["D_trans"] == 0.1
    assert sim_kwargs["D_rot"] == 0.01


def test_explicit_diffusion_is_preserved(tmp_path, monkeypatch):
    """Explicit non-zero D_trans and D_rot pass through to the simulator unchanged."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(
        tmp_path,
        chain_overrides=dict(D_trans=0.25, D_rot=0.05),
    )
    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs["D_trans"] == 0.25
    assert sim_kwargs["D_rot"] == 0.05


def test_auto_diffusion_does_not_set_scalar_d(tmp_path, monkeypatch):
    """With auto_diffusion enabled, no scalar D_trans or D_rot is forwarded to the simulator."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(tmp_path, chain_overrides=dict(auto_diffusion=True))
    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs["auto_diffusion"] is True
    assert "D_trans" not in sim_kwargs
    assert "D_rot" not in sim_kwargs


def test_run_chain_forwards_outputs(tmp_path, monkeypatch):
    """The parsed OutputConfig is forwarded to write_chain_results so user output flags are honored."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(
        tmp_path,
        output_overrides=dict(encounters_csv=False, full_paths=False),
    )
    chain_pipeline.run_chain(cfg)

    forwarded = captured["write_outputs"]
    assert forwarded is cfg.outputs
    assert forwarded.encounters_csv is False
    assert forwarded.full_paths is False


def test_born_grid_without_target_grid_raises(tmp_path):
    """Setting born_grid_dx without target_grid_dx raises a ValueError naming target_grid_dx."""
    cfg = _make_config(
        tmp_path,
        chain_overrides=dict(
            target_grid_dx="",
            born_grid_dx=str(tmp_path / "missing_born.dx"),
        ),
    )
    with pytest.raises(ValueError, match="target_grid_dx"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_no_grids_is_a_noop(tmp_path):
    """With neither grid path set, APBS grid generation is skipped and returns None."""
    cfg = _make_config(tmp_path)
    assert cfg.chain.target_grid_dx == ""
    assert cfg.chain.born_grid_dx == ""
    # Should return without raising and without attempting any APBS work.
    assert chain_pipeline._ensure_chain_apbs_grids(cfg) is None
