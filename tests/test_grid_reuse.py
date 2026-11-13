"""APBS grids are reused only when their key matches, the APBS input keeps the
input values, and PQR coordinates at or below -100 Å stay apart."""

import os

import pytest

import pystarc.pipeline.run_apbs as ra
from pystarc.pipeline.prepare_bd_surface import PQRAtom, _col8
from pystarc.structures.pqr_io import parse_pqr_records

SETTINGS = dict(ion_conc=0.15, debye_length=7.858, dielectric_in=4.0, dielectric_out=78.0,
                srad=1.5, temp=298.15, dime=129)


class Built(Exception):
    pass


def _grids(work, mol="receptor"):
    names = [f"{mol}{i}{s}.dx" for s in ("", "_born") for i in (0, 1)]
    for n in names:
        (work / n).write_text(n)
    return [work / n for n in names]


def _key(pqr, **kw):
    s = dict(SETTINGS, **kw)
    return ra._grid_key(pqr, dict(
        ion_conc=s["ion_conc"], debye_length=s["debye_length"], pdie=s["dielectric_in"],
        sdie=s["dielectric_out"], srad=s["srad"], temp=s["temp"], dime=s["dime"], coarse_dime=0,
        fine_dime=0, cglen=0.0, fglen=0.0, ion_radius_pos=0.95, ion_radius_neg=1.81))


@pytest.fixture
def work(tmp_path, monkeypatch):
    def refuse(name):
        raise Built(name)

    monkeypatch.setattr(ra, "_check_tool", refuse)
    (tmp_path / "receptor.pqr").write_text("ATOM      1  C   AAA A   1       0.000   0.000   0.000  0.1000 1.5000\n")
    return tmp_path


def test_matching_key_reuses_the_grids_in_build_order(work):
    files = _grids(work)
    (work / "receptor_grids.key").write_text(_key(work / "receptor.pqr") + "\n")
    assert ra.run_apbs(work / "receptor.pqr", "receptor", work, **SETTINGS) == files


@pytest.mark.parametrize("change", [dict(ion_conc=0.1), dict(debye_length=7.0), dict(dielectric_in=2.0),
                                    dict(dielectric_out=80.0), dict(srad=1.4), dict(temp=310.0),
                                    dict(dime=257)])
def test_any_changed_input_rebuilds(work, change):
    _grids(work)
    (work / "receptor_grids.key").write_text(_key(work / "receptor.pqr") + "\n")
    with pytest.raises(Built):
        ra.run_apbs(work / "receptor.pqr", "receptor", work, **dict(SETTINGS, **change))
    assert not any((work / f"receptor{i}{s}.dx").exists() for i in (0, 1) for s in ("", "_born"))


def test_changed_pqr_or_missing_key_rebuilds(work):
    _grids(work)
    with pytest.raises(Built):
        ra.run_apbs(work / "receptor.pqr", "receptor", work, **SETTINGS)
    _grids(work)
    (work / "receptor_grids.key").write_text(_key(work / "receptor.pqr") + "\n")
    (work / "receptor.pqr").write_text("ATOM      1  C   AAA A   1       1.000   0.000   0.000  0.1000 1.5000\n")
    with pytest.raises(Built):
        ra.run_apbs(work / "receptor.pqr", "receptor", work, **SETTINGS)


def test_rebuild_unlinks_shard_links_without_writing_through(work):
    parent = work / "parent"
    parent.mkdir()
    files = _grids(parent)
    shard = work / "bd_1"
    shard.mkdir()
    for f in files:
        os.symlink(f, shard / f.name)
    (shard / "receptor.pqr").write_text("ATOM      1  C   AAA A   1       5.000   0.000   0.000  0.1000 1.5000\n")
    with pytest.raises(Built):
        ra.run_apbs(shard / "receptor.pqr", "receptor", shard, **SETTINGS)
    assert all(f.read_text() == f.name for f in files)
    assert not any(os.path.lexists(shard / f.name) for f in files)


def test_key_depends_on_grid_format(work, monkeypatch):
    k = _key(work / "receptor.pqr")
    monkeypatch.setattr(ra, "GRID_FORMAT", ra.GRID_FORMAT + 1)
    assert _key(work / "receptor.pqr") != k


def test_run_apbs_both_forwards_the_ligand_dime(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ra, "run_apbs", lambda pqr, mol, work, *a: seen.append((mol, a[6])) or [])
    ra.run_apbs_both(tmp_path / "r.pqr", tmp_path / "l.pqr", tmp_path, dime=257, ligand_dime=161)
    assert seen == [("receptor", 257), ("ligand", 161)]


@pytest.mark.parametrize("pdie,conc,sdie", [(2.2, 0.15, 78.0), (4.0, 0.0001, 78.36), (4.0, 0.15, 78.5),
                                            (1.0, 0.05, 80.0), (20.0, 1.0, 78.0), (4.0, 0.15, 78.0)])
def test_apbs_input_keeps_the_input_values(tmp_path, pdie, conc, sdie):
    pqr = tmp_path / "m.pqr"
    pqr.write_text("ATOM      1  C   AAA A   1       0.000   0.000   0.000  0.1000 1.5000\n")
    params = {"dime": [65, 65, 65], "glen": [64.0] * 3, "gcent": [0.0] * 3, "bcfl": "sdh"}
    text = ra._write_apbs_input(pqr, "out", params, None, tmp_path, "t.in", False, conc, pdie, sdie,
                                1.5, 298.15).read_text()
    vals = {w[0]: w[1:] for w in (line.split() for line in text.splitlines()) if w}
    assert float(vals["pdie"][0]) == pdie and float(vals["sdie"][0]) == sdie
    assert float(vals["ion"][3]) == conc


@pytest.mark.parametrize("y,z", [(-100.0, -100.123), (-150.5, -999.999), (-123.456, 12.0), (-12.5, -99.9)])
def test_pqr_coordinates_stay_apart_and_read_back(tmp_path, y, z):
    a = PQRAtom(7, "CA", "ALA", 3, -150.123, y, z, -0.25, 1.9)
    line = a.to_pqr_line()
    assert len(line.split()) == 10
    p = tmp_path / "a.pqr"
    p.write_text(line)
    r = parse_pqr_records(p)[0]
    assert (r.x, r.y, r.z) == pytest.approx((-150.123, y, z), abs=0.005)


def test_lines_above_minus_100_keep_the_pdb_columns():
    a = PQRAtom(1, "N", "GLY", 1, 12.345, -99.999, 45.6, 0.1, 1.5)
    assert a.to_pqr_line()[30:54] == f"{12.345:8.3f}{-99.999:8.3f}{45.6:8.3f}"
    assert _col8(-100.5) == " -100.50" and _col8(-1000.25)[0] == " "
