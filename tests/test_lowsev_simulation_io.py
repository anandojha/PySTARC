"""
Low-severity robustness test for getf in pystarc.xml_io.simulation_io.

getf is the inner float reader used by parse_simulation_xml. Previously it did
not guard the literal string "None", so a value of "None" (which is what
str(None) produces when a None default is written by write_simulation_xml and
read back) reached float("None") and raised ValueError. geti already guarded
this case. These tests exercise the guard through parse_simulation_xml without
needing a GPU, APBS, or any external binary.
"""

import textwrap

from pystarc.xml_io.simulation_io import parse_simulation_xml


def _write_xml(tmp_path, body: str):
    p = tmp_path / "sim.xml"
    p.write_text("<simulation>\n" + body + "\n</simulation>\n")
    return p


def test_getf_handles_literal_none(tmp_path):
    # A float-valued tag carrying the literal string "None" must fall back to
    # the default rather than raising ValueError from float("None").
    p = _write_xml(tmp_path, "  <dt>None</dt>\n  <r_start>None</r_start>")
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.2
    assert result["r_start"] == 100.0


def test_getf_healthy_numeric_value_unchanged(tmp_path):
    # A normal numeric value must still parse to that exact float.
    p = _write_xml(tmp_path, "  <dt>0.05</dt>\n  <r_start>250.0</r_start>")
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.05
    assert result["r_start"] == 250.0


def test_getf_missing_tag_uses_default(tmp_path):
    # A missing float tag must still use the supplied default.
    p = _write_xml(tmp_path, "  <n_trajectories>5</n_trajectories>")
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.2
    assert result["r_escape"] == 0.0
