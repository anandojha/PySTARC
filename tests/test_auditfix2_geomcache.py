"""Regression tests for the hydrodynamic-radius cache key in analyse_molecule.

These tests check that the cache file written next to a PQR encodes the Monte
Carlo sample count and a digest of the atom coordinates and radii. Two analyses
that differ only in n_mc must write to distinct cache files, and editing the
structure must also produce a distinct cache file, so that a re-run never reuses
a stale radius that does not match the requested precision or structure.
"""

import glob

from pathlib import Path

from pystarc.pipeline import geometry
from pystarc.pipeline.geometry import analyse_molecule


_PQR_BODY = """\
ATOM      1  N   ALA     1       0.000   0.000   0.000  0.10 1.50
ATOM      2  C   ALA     1       3.000   0.000   0.000 -0.10 1.70
ATOM      3  O   ALA     1       0.000   3.000   0.000 -0.20 1.40
ATOM      4  C   ALA     1       0.000   0.000   3.000  0.20 1.70
"""


def _write_pqr(path: Path, body: str) -> None:
    path.write_text(body)


def _cache_files(pqr_path: Path):
    return sorted(glob.glob(str(pqr_path) + ".r_hydro_*.cache"))


def test_different_n_mc_use_different_cache_files(tmp_path):
    """Two analyses differing only in n_mc write to distinct cache files."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)

    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=5000)

    caches = _cache_files(pqr)
    assert len(caches) == 2, caches
    assert "_n2000_" in "".join(caches)
    assert "_n5000_" in "".join(caches)


def test_changed_n_mc_does_not_reuse_stale_value(tmp_path, monkeypatch):
    """The second call with a new n_mc recomputes rather than reading the cache."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)

    calls = {"n": 0}
    real = geometry.mc_hydrodynamic_radius

    def counting_mc(coords, radii, spacing, n_mc):
        calls["n"] += 1
        return real(coords, radii, spacing=spacing, n_mc=n_mc)

    monkeypatch.setattr(geometry, "mc_hydrodynamic_radius", counting_mc)

    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    assert calls["n"] == 1
    # A second call with the same n_mc reuses the cache, so no new computation.
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    assert calls["n"] == 1
    # A call with a different n_mc must recompute instead of reusing the cache.
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=5000)
    assert calls["n"] == 2


def test_edited_structure_uses_different_cache_file(tmp_path):
    """Editing the atom coordinates yields a distinct cache file."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    first = _cache_files(pqr)
    assert len(first) == 1

    moved = _PQR_BODY.replace(
        "ATOM      2  C   ALA     1       3.000   0.000   0.000 -0.10 1.70",
        "ATOM      2  C   ALA     1       4.000   0.000   0.000 -0.10 1.70",
    )
    _write_pqr(pqr, moved)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)

    caches = _cache_files(pqr)
    assert len(caches) == 2, caches
