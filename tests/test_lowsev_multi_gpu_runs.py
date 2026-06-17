"""Focused test for the bd_sims directory guard in multi_GPU_runs.main().

The guard added on the low-severity branch ensures that when grid generation
reports success (subprocess return code 0) but the expected bd_sims/ directory
was not actually produced, main() reports a clear error and returns instead of
crashing with an opaque FileNotFoundError from the subsequent os.listdir call.

The test stubs subprocess.run so it returns a zero return code without creating
bd_sims/. No GPU, APBS, or external binary is exercised.
"""

import importlib.util
import os

import pytest

MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pystarc",
    "multi_GPU",
    "multi_GPU_runs.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_GPU_runs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_xml(path):
    with open(path, "w") as fh:
        fh.write(
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            "<input>\n"
            "  <n_trajectories>4</n_trajectories>\n"
            "  <seed>1</seed>\n"
            "</input>\n"
        )


class _FakeReturn:
    def __init__(self, returncode):
        self.returncode = returncode


def test_missing_bd_sims_after_grid_gen_reports_clear_error(tmp_path, monkeypatch, capsys):
    module = _load_module()

    xml_path = tmp_path / "input.xml"
    _write_minimal_xml(str(xml_path))

    # bd_sims/ deliberately does not exist, so main() enters the grid-generation
    # branch. The stub makes that step look successful without producing bd_sims/.
    def fake_run(*args, **kwargs):
        return _FakeReturn(0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["multi_GPU_runs.py", str(xml_path)])

    bd_sims = tmp_path / "bd_sims"
    assert not bd_sims.exists()

    # Must return cleanly rather than raising FileNotFoundError from os.listdir.
    result = module.main()
    assert result is None

    out = capsys.readouterr().out
    assert "did not create" in out
    # The guard must short-circuit before any per-split directory is created.
    assert not bd_sims.exists()


def test_grid_gen_failure_returns_without_listdir(tmp_path, monkeypatch, capsys):
    module = _load_module()

    xml_path = tmp_path / "input.xml"
    _write_minimal_xml(str(xml_path))

    def fake_run(*args, **kwargs):
        return _FakeReturn(1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["multi_GPU_runs.py", str(xml_path)])

    result = module.main()
    assert result is None
    out = capsys.readouterr().out
    assert "grid generation failed" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
