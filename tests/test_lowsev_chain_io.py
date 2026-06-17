"""Low-severity robustness test for pdb_to_bead_positions cap-bead handling.

Cap beads (ACE:CN, NME:CC) carry resid=-1 because they have no PDB residue of
their own. Feeding a capped chain into pdb_to_bead_positions previously let the
resid=-1 sentinel leak into the unique-residue count, which produced a
confusing residue-count mismatch error or an out-of-range sequence position.
The function now rejects capped chains up front with a clear error. These tests
exercise that error and confirm the healthy no-cap path is not affected by the
new guard.
"""

import numpy as np
import pytest

from pystarc.simulation.coffdrop_chain import ChainAtom, ChainCommon
from pystarc.structures.chain_io import pdb_to_bead_positions


def _write_pdb(tmp_path, lines):
    path = tmp_path / "test.pdb"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _single_ala_pdb(tmp_path):
    """One ALA residue with N, CA, C, O, CB heavy atoms on chain A."""
    lines = [
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C",
        "ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C",
        "ATOM      4  O   ALA A   1       2.500   3.500   4.500  1.00  0.00           O",
        "ATOM      5  CB  ALA A   1       3.000   4.000   5.000  1.00  0.00           C",
    ]
    return _write_pdb(tmp_path, lines)


def test_capped_chain_raises_clear_error(tmp_path):
    """A chain carrying a cap bead (resid=-1) is rejected with a clear caps-unsupported error."""
    pdb = _single_ala_pdb(tmp_path)
    # One ordinary residue bead plus an ACE cap bead carrying resid=-1.
    common = ChainCommon(
        name="capped",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:CA", resid=0),
            ChainAtom(radius=2.0, charge=0.0, resname="ACE:CN", resid=-1),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        pdb_to_bead_positions(common, pdb, chain_id="A")
    message = str(excinfo.value).lower()
    assert "cap" in message
    assert "resid < 0" in message


def test_nme_cap_bead_also_rejected(tmp_path):
    """A C-terminal NME cap bead is detected by the resid<0 guard, not by atom name."""
    pdb = _single_ala_pdb(tmp_path)
    common = ChainCommon(
        name="capped",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:CA", resid=0),
            ChainAtom(radius=2.0, charge=0.0, resname="NME:CC", resid=-1),
        ],
    )
    with pytest.raises(ValueError, match="cap"):
        pdb_to_bead_positions(common, pdb, chain_id="A")


def test_no_cap_chain_does_not_trip_cap_guard(tmp_path):
    """A chain with only non-negative resids never raises the caps-unsupported error.

    A bead atom name intentionally absent from the COFFDROP ALA map drives the
    function past the cap guard. The resulting error must therefore be the
    bead-lookup KeyError, not the caps-unsupported ValueError, which confirms
    the new guard does not fire on a healthy no-cap chain.
    """
    pdb = _single_ala_pdb(tmp_path)
    common = ChainCommon(
        name="nocaps",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:QQ", resid=0),
        ],
    )
    with pytest.raises(KeyError) as excinfo:
        pdb_to_bead_positions(common, pdb, chain_id="A")
    assert "cap" not in str(excinfo.value).lower()
