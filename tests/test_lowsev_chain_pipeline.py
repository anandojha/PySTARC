"""Validation test for the chain BD APBS grid naming guard.

_ensure_chain_apbs_grids derives the APBS mol_name by stripping trailing digits
from the target_grid_dx stem, and run_apbs only ever writes files named
{mol_name}0.dx, {mol_name}1.dx, {mol_name}0_born.dx, and {mol_name}1_born.dx.
A target_grid_dx whose filename lacks the trailing level digit (for example
"target.dx" instead of "target1.dx") therefore names a file APBS never produces:
the APBS run would succeed yet leave the requested path absent, and the later
DXGrid.from_file(cc.target_grid_dx) would fail with a bare FileNotFoundError that
hides the real cause. These tests check that the function raises a clear
ValueError up front for a non-conforming name, and that a conforming name passes
the naming guard. They exercise only the pure-Python validation branch and do
not invoke APBS or any external binary.
"""

import pytest

from pystarc.pipeline import chain_pipeline
from pystarc.pipeline.input_parser import ChainConfig, PySTARCConfig


def _make_config(tmp_path, target_grid_dx, receptor_pqr=""):
    chain = ChainConfig(
        chain_json=str(tmp_path / "chain.json"),
        reaction_pairs_json=str(tmp_path / "reaction_pairs.json"),
        target_grid_dx=target_grid_dx,
    )
    return PySTARCConfig(
        work_dir=tmp_path / "work",
        chain=chain,
        receptor_pqr=receptor_pqr,
    )


def test_target_grid_without_trailing_digit_raises(tmp_path):
    """A missing target_grid_dx whose name lacks the trailing level digit raises a clear ValueError before any APBS work."""
    # The file does not exist on disk, so the function takes the generation
    # branch. The stem "target" strips to mol_name "target", and APBS would
    # write "target1.dx" rather than the requested "target.dx".
    bad_path = tmp_path / "grids" / "target.dx"
    cfg = _make_config(tmp_path, target_grid_dx=str(bad_path))

    with pytest.raises(ValueError, match=r"target1\.dx"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_conforming_target_grid_passes_naming_guard(tmp_path):
    """A conforming '{mol_name}1.dx' name passes the naming guard and proceeds to the receptor_pqr check, not the naming error."""
    # "target1.dx" strips to mol_name "target", and APBS writes "target1.dx",
    # so the naming guard must not fire. Pointing receptor_pqr at a path that
    # does not exist makes the function fail at the receptor_pqr existence
    # check, raising FileNotFoundError rather than the naming ValueError. This
    # confirms the naming guard passed without ever invoking APBS.
    good_path = tmp_path / "grids" / "target1.dx"
    missing_pqr = str(tmp_path / "does_not_exist.pqr")
    cfg = _make_config(tmp_path, target_grid_dx=str(good_path), receptor_pqr=missing_pqr)

    with pytest.raises(FileNotFoundError, match="receptor_pqr"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_coarse_electrostatic_grid_name_passes_naming_guard(tmp_path):
    """The coarse electrostatic name '{mol_name}0.dx' is among the names APBS produces and passes the naming guard."""
    good_path = tmp_path / "grids" / "target0.dx"
    missing_pqr = str(tmp_path / "does_not_exist.pqr")
    cfg = _make_config(tmp_path, target_grid_dx=str(good_path), receptor_pqr=missing_pqr)

    # mol_name strips to "target"; "target0.dx" is in the produced set, so the
    # naming guard passes and execution reaches the receptor_pqr check.
    with pytest.raises(FileNotFoundError, match="receptor_pqr"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)
