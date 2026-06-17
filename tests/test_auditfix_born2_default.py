"""Regression test for the enable_born2_torque parser default.

The BORN2 reciprocal torque (the ligand Born term acting on the receptor
atoms) contributes about 3% of the total torque. When the input XML omits
the <enable_born2_torque> tag, the parser should adopt the same default as
the PySTARCConfig dataclass, which leaves the term enabled.
"""

from pathlib import Path

from pystarc.pipeline.input_parser import PySTARCConfig, parse


def _write_input_without_tag(tmp_path: Path) -> Path:
    """Write a minimal input XML that omits <enable_born2_torque>."""
    work_dir = tmp_path / "bd_sims"
    xml = f"""<?xml version="1.0" ?>
<pystarc_input>
    <receptor_pqr>receptor.pqr</receptor_pqr>
    <ligand_pqr>ligand.pqr</ligand_pqr>
    <ligand_resname>BEN</ligand_resname>
    <ligand_charge>1</ligand_charge>
    <work_dir>{work_dir}</work_dir>
    <n_trajectories>10000</n_trajectories>
    <bd_milestone_radius>30.0</bd_milestone_radius>
    <ghost_atoms>auto</ghost_atoms>
</pystarc_input>
"""
    xml_path = tmp_path / "pystarc_input.xml"
    xml_path.write_text(xml)
    return xml_path


def test_born2_default_true_when_tag_absent(tmp_path):
    """Parsing input without the tag enables the BORN2 reciprocal torque."""
    xml_path = _write_input_without_tag(tmp_path)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is True


def test_born2_parser_default_matches_dataclass(tmp_path):
    """The parser default agrees with the PySTARCConfig dataclass default."""
    xml_path = _write_input_without_tag(tmp_path)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is PySTARCConfig.enable_born2_torque


def test_born2_explicit_false_is_respected(tmp_path):
    """An explicit <enable_born2_torque>false</enable_born2_torque> disables it."""
    work_dir = tmp_path / "bd_sims"
    xml = f"""<?xml version="1.0" ?>
<pystarc_input>
    <receptor_pqr>receptor.pqr</receptor_pqr>
    <ligand_pqr>ligand.pqr</ligand_pqr>
    <ligand_resname>BEN</ligand_resname>
    <ligand_charge>1</ligand_charge>
    <work_dir>{work_dir}</work_dir>
    <n_trajectories>10000</n_trajectories>
    <bd_milestone_radius>30.0</bd_milestone_radius>
    <ghost_atoms>auto</ghost_atoms>
    <enable_born2_torque>false</enable_born2_torque>
</pystarc_input>
"""
    xml_path = tmp_path / "pystarc_input.xml"
    xml_path.write_text(xml)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is False
