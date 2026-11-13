"""Range checks of PySTARCConfig.validate and the Debye length of the salt."""

import math

import pytest

from pystarc.pipeline.input_parser import PySTARCConfig, debye_from_ions, parse


def _write(tmp_path, **tags):
    t = {"receptor_pqr": "r.pqr", "ligand_pqr": "l.pqr", "rxns_xml": "rxns.xml",
         "soft_repulsion": "true", "work_dir": str(tmp_path / "bd"), **tags}
    p = tmp_path / "input.xml"
    p.write_text("<pystarc>" + "".join(f"<{k}>{v}</{k}>" for k, v in t.items() if v is not None)
                 + "</pystarc>")
    return p


def _cfg(tmp_path, **kw):
    return PySTARCConfig(receptor_pqr="r.pqr", ligand_pqr="l.pqr", work_dir=tmp_path / "bd", **kw)


@pytest.mark.parametrize("lam", [0, 0.0, -7.858, float("nan")])
def test_debye_length_must_be_positive(tmp_path, lam):
    with pytest.raises(ValueError, match="debye_length and sdie must be positive"):
        _cfg(tmp_path, debye_length=lam).validate()


def test_sdie_must_be_positive_and_inf_means_no_salt(tmp_path):
    with pytest.raises(ValueError, match="debye_length and sdie"):
        _cfg(tmp_path, sdie=0.0).validate()
    _cfg(tmp_path, debye_length=math.inf, ion_concentration=0.0).validate()
    assert parse(_write(tmp_path, debye_length="7.858", ion_concentration="0.15")).debye_length == 7.858


def test_debye_length_follows_the_salt_when_omitted(tmp_path):
    assert debye_from_ions(0.15, 78.0, 298.15) == pytest.approx(7.8295, abs=1e-4)
    assert debye_from_ions(0.1, 78.0, 298.15) == pytest.approx(9.5891, abs=1e-4)
    cfg = parse(_write(tmp_path, ion_concentration="0.15", sdie="78"))
    assert cfg.debye_length == debye_from_ions(0.15, 78.0, 298.15)
    assert parse(_write(tmp_path, ion_concentration="0")).debye_length == math.inf


@pytest.mark.parametrize("lam,conc", [("7.0", "0.15"), ("7.858", "0"), ("inf", "0.15"), ("9.0", "0.15")])
def test_debye_length_that_disagrees_with_the_salt_raises(tmp_path, lam, conc):
    with pytest.raises(ValueError, match="<debye_length>.*<ion_concentration>"):
        parse(_write(tmp_path, debye_length=lam, ion_concentration=conc))


def test_chain_input_keeps_the_default_debye_length(tmp_path):
    p = tmp_path / "input.xml"
    p.write_text(f"<pystarc><receptor_pqr>r.pqr</receptor_pqr><work_dir>{tmp_path / 'bd'}</work_dir>"
                 "<chain><chain_json>c.json</chain_json></chain></pystarc>")
    assert parse(p).debye_length == 7.858


@pytest.mark.parametrize("r_esc,ok", [(0.0, True), (-1.0, False), (30.0, False), (33.0, False),
                                      (33.1, True), (float("nan"), False)])
def test_r_escape_must_exceed_qb_factor_b(tmp_path, r_esc, ok):
    cfg = _cfg(tmp_path, bd_milestone_radius=30.0, r_escape=r_esc)
    if ok:
        cfg.validate()
    else:
        with pytest.raises(ValueError, match="r_escape must be 0"):
            cfg.validate()


def test_compute_geometry_refuses_r_escape_inside_the_trigger(tmp_path):
    from pystarc.pipeline.geometry import compute_geometry

    line = "ATOM      1  C   AAA A   1    {:8.3f}   0.000   0.000  0.0000 1.5000\n"
    for name in ("r.pqr", "l.pqr"):
        (tmp_path / name).write_text(line.format(0.0) + line.format(2.0))
    with pytest.raises(ValueError, match="r_escape must be 0"):
        compute_geometry(tmp_path / "r.pqr", tmp_path / "l.pqr", bd_milestone_radius=20.0,
                         r_hydro_rec=2.0, r_hydro_lig=2.0, r_escape=21.0)


@pytest.mark.parametrize("key,value", [("confidence_interval", 0.0), ("confidence_interval", 1.0),
                                       ("confidence_interval", 95.0), ("confidence_interval", float("nan")),
                                       ("convergence_tol", 0.0), ("convergence_tol", 1.0)])
def test_level_and_tolerance_lie_between_0_and_1(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"{key} must lie between 0 and 1"):
        _cfg(tmp_path, **{key: value}).validate()


def test_force_batch_alpha_and_save_interval_ranges(tmp_path):
    _cfg(tmp_path, gpu_force_batch=0).validate()
    _cfg(tmp_path, gpu_force_batch=1000).validate()
    with pytest.raises(ValueError, match="gpu_force_batch must be >= 0"):
        _cfg(tmp_path, gpu_force_batch=-1).validate()
    with pytest.raises(ValueError, match="desolvation_alpha must be >= 0"):
        _cfg(tmp_path, desolvation_alpha=-0.1).validate()
    cfg = _cfg(tmp_path)
    cfg.outputs.save_interval = -1
    with pytest.raises(ValueError, match="save_interval must be >= 0"):
        cfg.validate()
