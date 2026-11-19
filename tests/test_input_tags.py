"""Strict reading of pystarc_input.xml: flags, unknown and retired tags, and the
excluded-volume, time-step and viscosity keys."""

import math
import re
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pystarc.pipeline.input_parser import parse, parse_flag

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _write(tmp_path, body="", **tags):
    t = {"receptor_pqr": "r.pqr", "ligand_pqr": "l.pqr", "rxns_xml": "rxns.xml",
         "soft_repulsion": "true", "work_dir": str(tmp_path / "bd"), **tags}
    p = tmp_path / "input.xml"
    p.write_text("<pystarc>" + "".join(f"<{k}>{v}</{k}>" for k, v in t.items() if v is not None)
                 + body + "</pystarc>")
    return p


def _quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return parse(path)


@pytest.mark.parametrize("text,value", [("TRUE", True), ("Yes", True), ("1", True),
                                        ("false", False), ("NO", False), (" 0 ", False)])
def test_flags_accept_true_yes_1_false_no_0(tmp_path, text, value):
    assert _quiet(_write(tmp_path, gpu=text)).gpu is value


@pytest.mark.parametrize("text", ["ture", "on", "2", "y"])
def test_other_flag_values_raise_naming_the_tag(tmp_path, text):
    with pytest.raises(ValueError, match="<hydrodynamic_interactions> must be true or false"):
        parse(_write(tmp_path, hydrodynamic_interactions=text))


def test_number_that_does_not_parse_names_its_tag(tmp_path):
    with pytest.raises(ValueError, match="<seed> must be int"):
        parse(_write(tmp_path, seed="12a"))


def test_blank_element_counts_as_omitted(tmp_path):
    cfg = _quiet(_write(tmp_path, seed="  ", gpu=" "))
    assert cfg.seed == 1523 and cfg.gpu is True


def test_unknown_tag_raises_with_a_close_match(tmp_path):
    with pytest.raises(ValueError, match="<n_trajectorys>.*did you mean <n_trajectories>"):
        parse(_write(tmp_path, n_trajectorys="10"))
    assert not (tmp_path / "bd").exists()


@pytest.mark.parametrize("tag,home", [("save_interval", "<outputs>"), ("full_paths", "<outputs>"),
                                      ("dt_chain", "<chain>"), ("chain_json", "<chain>")])
def test_block_field_at_the_root_names_its_block(tmp_path, tag, home):
    with pytest.raises(ValueError, match=f"<{tag}>.*belongs inside {home}"):
        parse(_write(tmp_path, **{tag: "1"}))


def test_unknown_tags_inside_outputs_and_chain_raise(tmp_path):
    with pytest.raises(ValueError, match="<paths> in <outputs>"):
        parse(_write(tmp_path, "<outputs><paths>true</paths></outputs>"))
    with pytest.raises(ValueError, match="<seed> in <outputs>: it belongs at the top level"):
        parse(_write(tmp_path, "<outputs><seed>3</seed></outputs>"))
    with pytest.raises(ValueError, match="<dt_chian> in <chain>"):
        parse(_write(tmp_path, "<chain><chain_json>c.json</chain_json><dt_chian>0.1</dt_chian></chain>"))


def test_receptor_resname_passes_silently(tmp_path):
    _quiet(_write(tmp_path, receptor_resname="REC"))


def test_tags_of_prepare_bd_surface_warn_once(tmp_path):
    with pytest.warns(UserWarning, match="<parm7>, <ion_type> are read by prepare_bd_surface only"):
        parse(_write(tmp_path, parm7="x.parm7", ion_type="NaCl"))


@pytest.mark.parametrize("tag", ["stall_check", "unfinished_in_p_rxn", "apbs_grid_spacing",
                                 "apbs_coarse_spacing"])
def test_retired_tags_warn(tmp_path, tag):
    with pytest.warns(UserWarning, match=f"<{tag}> is no longer used"):
        parse(_write(tmp_path, **{tag: "1"}))


def test_p_commit_is_retired_and_fpt_distribution_is_an_alias(tmp_path):
    with pytest.warns(UserWarning, match="<p_commit> is no longer used"):
        cfg = parse(_write(tmp_path, "<outputs><p_commit>true</p_commit>"
                                     "<fpt_distribution>false</fpt_distribution></outputs>"))
    assert cfg.outputs.bd_time_to_reaction is False and cfg.outputs.fpt_distribution is False
    cfg = _quiet(_write(tmp_path, "<outputs><bd_time_to_reaction>false</bd_time_to_reaction></outputs>"))
    assert cfg.outputs.bd_time_to_reaction is False


def test_prepare_bd_surface_flags_are_strict(tmp_path):
    from pystarc.pipeline.prepare_bd_surface import parse_config

    with pytest.raises(ValueError, match="<inject_gho> must be true or false"):
        parse_config(_write(tmp_path, inject_gho="ture"))
    assert parse_flag("x", "Yes") is True


def test_rigid_input_must_name_the_excluded_volume(tmp_path):
    with pytest.raises(ValueError, match="soft_repulsion.*overlap_check"):
        parse(_write(tmp_path, soft_repulsion=None))
    assert _quiet(_write(tmp_path, soft_repulsion=None, overlap_check="true")).overlap_check
    chain = "<chain><chain_json>c.json</chain_json></chain>"
    _quiet(_write(tmp_path, chain, soft_repulsion=None, ligand_pqr=None, rxns_xml=None))


@pytest.mark.parametrize("cfg_path", sorted(EXAMPLES.glob("**/config.xml")),
                         ids=lambda p: p.parent.name)
def test_every_example_config_follows_the_universal_template(cfg_path):
    """Every example config.xml has the same 9 sections and tag order, with no setup.py left."""
    from pystarc.pipeline import build_inputs as bi

    assert not (cfg_path.parent / "setup.py").exists()
    root = ET.parse(cfg_path).getroot()
    assert root.tag == "config"
    assert [sec.tag for sec in root] == [name for name, _ in bi.SECTIONS]
    for sec, (_, expected_tags) in zip(root, bi.SECTIONS):
        assert [el.tag for el in sec] == list(expected_tags)


@pytest.mark.parametrize("cfg_path", [p for p in sorted(EXAMPLES.glob("**/config.xml"))
                                      if p.parent.name != "barnase_barstar"],
                         ids=lambda p: p.parent.name)
def test_input_written_from_an_example_config_parses_without_warning(tmp_path, cfg_path):
    """write_input of build_inputs produces an input.xml that parses without warning."""
    from pystarc.pipeline import build_inputs as bi

    raw = bi.read_config(cfg_path)
    raw["work_dir"] = str(tmp_path)
    raw["bd_work_dir"] = str(tmp_path / "bd")
    bi.write_input(str(tmp_path / "input.xml"), raw)
    cfg = _quiet(tmp_path / "input.xml")
    assert cfg.dt == float(raw["dt"]) and cfg.gpu_force_batch == 0


def test_relative_viscosity_must_be_one_and_temperature_warns(tmp_path):
    _quiet(_write(tmp_path, relative_viscosity="1.0"))
    with pytest.raises(ValueError, match="water at 25 °C"):
        parse(_write(tmp_path, relative_viscosity="2"))
    with pytest.warns(UserWarning, match="scales kBT only"):
        parse(_write(tmp_path, temperature="310"))


def test_ligand_dime_is_read_and_checked(tmp_path):
    assert _quiet(_write(tmp_path, apbs_ligand_dime="257")).apbs_ligand_dime == 257
    with pytest.raises(ValueError, match="apbs_ligand_dime"):
        parse(_write(tmp_path, apbs_ligand_dime="128"))


@pytest.mark.parametrize("sub,want_pairs", [
    ("two_charged_spheres", 1),
    ("trypsin_benzamidine", 6),
    ("thrombin_thrombomodulin", 6),
    ("barnase_barstar", 568),
])
def test_example_configs_with_local_files_build_and_parse_cleanly(tmp_path, sub, want_pairs):
    import json

    if sub == "barnase_barstar" and shutil.which("tleap") is None:
        pytest.skip("tleap not installed")
    dst = tmp_path / sub
    shutil.copytree(EXAMPLES / sub, dst)
    cfg = _quiet(dst / "config.xml")
    rec = json.loads((dst / "bd_sims" / "build_record.json").read_text())
    assert len(rec["pairs"]) == want_pairs
    assert math.isfinite(cfg.debye_length)


def test_config_strict_tag_and_section_validation(tmp_path):
    from pystarc.pipeline import build_inputs as bi

    p = tmp_path / "bad_sec.xml"
    p.write_text("<config><unknown_sec/></config>")
    with pytest.raises(ValueError, match="Unknown section <unknown_sec>"):
        parse(p)
    p2 = tmp_path / "misplaced.xml"
    p2.write_text("<config><outputs><seed>1</seed></outputs></config>")
    with pytest.raises(ValueError, match="<seed> in <outputs>: place it inside <simulation>"):
        parse(p2)
    p3 = tmp_path / "typo.xml"
    p3.write_text("<config><structure><recpetor>receptor.pdb</recpetor></structure></config>")
    with pytest.raises(ValueError, match="Unknown tag <recpetor>.*did you mean <receptor>"):
        parse(p3)
    blank = tmp_path / "template.xml"
    blank.write_text(bi.config_xml({}, full_template=True))
    assert bi.read_config(blank) == {"criterion": {}, "chain_cfg": {}}


def test_run_pystarc_template_and_setup_cli(tmp_path):
    import subprocess
    import sys

    runner = EXAMPLES.parent / "run_pystarc.py"
    tpl = tmp_path / "config.xml"
    r1 = subprocess.run([sys.executable, str(runner), "--template", str(tpl)], capture_output=True, text=True)
    assert r1.returncode == 0 and tpl.exists()
    dst = tmp_path / "tcs"
    shutil.copytree(EXAMPLES / "two_charged_spheres", dst)
    r2 = subprocess.run([sys.executable, str(runner), str(dst / "config.xml"), "--setup"], capture_output=True, text=True)
    assert r2.returncode == 0 and (dst / "bd_sims" / "rxns.xml").exists()
