"""Tests for the post-run output check in make_combined_pqr.

The ambpdb step in pystarc.pipeline.make_pqr.make_combined_pqr writes its PQR
through shell redirection, which has no built in success check. The fix adds a
guard that the combined PQR file exists and holds at least one ATOM or HETATM
record, raising a clear RuntimeError otherwise so that a silent ambpdb failure
is reported at its source rather than as a confusing downstream error.

These tests replace the external _check_tool and _run helpers with stand ins so
they run without ambpdb, cpptraj, a GPU, or any external binary. The stand in
for _run writes a chosen content into the expected output file to mimic the
three observable outcomes: missing file, empty or atomless file, and valid file.
"""

import re
from pathlib import Path

import pytest

import pystarc.pipeline.make_pqr as make_pqr


def _install_stubs(monkeypatch, pqr_content):
    """Patch _check_tool and _run so make_combined_pqr runs with no binaries.

    The _run stand in inspects the command string. For the ambpdb command it
    writes pqr_content (or nothing when pqr_content is None) to complex.pqr in
    the working directory, reproducing what the real shell redirection would do.
    The cpptraj command is treated as a no op that creates the expected rst file.
    """
    monkeypatch.setattr(make_pqr, "_check_tool", lambda name: None)

    def fake_run(cmd, cwd, step):
        cwd = Path(cwd)
        if step == "cpptraj":
            # cpptraj would normally write complex.rst from the pdb.
            (cwd / "complex.rst").write_text("dummy restart\n")
        elif step == "ambpdb":
            if pqr_content is not None:
                # Mimic the shell redirection target ambpdb -pqr > complex.pqr.
                (cwd / "complex.pqr").write_text(pqr_content)
            # When pqr_content is None we leave the file absent on purpose.
        return None

    monkeypatch.setattr(make_pqr, "_run", fake_run)


def test_missing_output_raises_clear_error(tmp_path, monkeypatch):
    """A missing combined PQR file raises a clear RuntimeError naming ambpdb."""
    _install_stubs(monkeypatch, pqr_content=None)
    with pytest.raises(RuntimeError, match=r"ambpdb.*no output"):
        make_pqr.make_combined_pqr(
            prmtop_path=tmp_path / "complex.prmtop",
            complex_pdb=tmp_path / "complex.pdb",
            work_dir=tmp_path,
        )


def test_atomless_output_raises_clear_error(tmp_path, monkeypatch):
    """A combined PQR file with no ATOM or HETATM records raises a clear error."""
    # A header only file with no atom records, the kind a failed run might leave.
    _install_stubs(monkeypatch, pqr_content="REMARK   1 nothing useful here\nEND\n")
    with pytest.raises(RuntimeError, match=r"no ATOM or HETATM"):
        make_pqr.make_combined_pqr(
            prmtop_path=tmp_path / "complex.prmtop",
            complex_pdb=tmp_path / "complex.pdb",
            work_dir=tmp_path,
        )


def test_valid_output_passes_and_cleans_intermediates(tmp_path, monkeypatch):
    """A PQR holding atom records returns its path and removes intermediate files.

    This is the healthy path. The guard must not raise, the returned path must be
    complex.pqr in the working directory, the atom content must be untouched, and
    the cpptraj input and inpcrd intermediates must be cleaned up as before.
    """
    valid = (
        "ATOM      1  N   ALA     1      11.104   6.134  -6.504  0.1000 1.5500\n"
        "HETATM    2  C1  LIG     2      12.000   7.000  -5.000 -0.2000 1.7000\n"
        "END\n"
    )
    _install_stubs(monkeypatch, pqr_content=valid)
    out = make_pqr.make_combined_pqr(
        prmtop_path=tmp_path / "complex.prmtop",
        complex_pdb=tmp_path / "complex.pdb",
        work_dir=tmp_path,
    )
    assert out == tmp_path / "complex.pqr"
    assert out.read_text() == valid
    # Intermediates created or consumed by the function must be gone.
    assert not (tmp_path / "get_inpcrd.cpptraj").exists()
    assert not (tmp_path / "complex.inpcrd").exists()


def test_first_atom_only_is_enough(tmp_path, monkeypatch):
    """The guard accepts a file whose first atom record is a HETATM line."""
    content = "HETATM    1  C1  LIG     1       0.000   0.000   0.000 0.0000 1.7000\n"
    _install_stubs(monkeypatch, pqr_content=content)
    out = make_pqr.make_combined_pqr(
        prmtop_path=tmp_path / "complex.prmtop",
        complex_pdb=tmp_path / "complex.pdb",
        work_dir=tmp_path,
    )
    assert out.read_text() == content
