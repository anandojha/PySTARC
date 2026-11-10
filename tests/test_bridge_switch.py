"""
The input key brownian_bridge that switches the Brownian bridge on or off.

The bridge is on by default. With brownian_bridge false a reaction counts only
when the criterion holds at the end of a step. The key reaches the CPU engine
through NAMParameters.use_brownian_bridge, which the GPU engine reads too.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pystarc.pipeline.input_parser import parse
from pystarc.simulation.nam_simulator import NAMParameters

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "two_charged_spheres"


def _input_with(tmp_path, extra: str) -> Path:
    dst = tmp_path / "tcs"
    shutil.copytree(EXAMPLE, dst)
    path = dst / "input_bridge.xml"
    base = (dst / "input.xml").read_text()
    path.write_text(base.replace("</pystarc>", extra + "</pystarc>"))
    return path


def test_bridge_is_on_by_default(tmp_path):
    assert parse(_input_with(tmp_path, "")).brownian_bridge is True
    assert NAMParameters().use_brownian_bridge is True


def test_bridge_switch_reads_false_and_true(tmp_path):
    off = _input_with(tmp_path, "  <brownian_bridge>false</brownian_bridge>\n")
    assert parse(off).brownian_bridge is False
    on = off.with_name("input_bridge_on.xml")
    on.write_text(off.read_text().replace(">false</brownian_bridge>", ">true</brownian_bridge>"))
    assert parse(on).brownian_bridge is True
