"""Tests for PDB residue parsing with insertion codes in chain_io."""

import numpy as np

from pystarc.structures.chain_io import _parse_pdb_chain_for_beads


def _write_pdb(tmp_path, lines):
    path = tmp_path / "test.pdb"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_insertion_code_residues_are_distinct(tmp_path):
    """Residues sharing a sequence number but differing by insertion code are parsed as two separate residues with their own atoms."""
    # Columns are laid out per the fixed-width PDB ATOM record. The insertion
    # code sits in column 27 (index 26).
    lines = [
        "ATOM      1  N   ALA A 100       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A 100       1.500   2.500   3.500  1.00  0.00           C",
        "ATOM      3  N   GLY A 100A      4.000   5.000   6.000  1.00  0.00           N",
        "ATOM      4  CA  GLY A 100A      4.500   5.500   6.500  1.00  0.00           C",
    ]
    pdb = _write_pdb(tmp_path, lines)
    residues = _parse_pdb_chain_for_beads(pdb, chain_id="A")

    assert len(residues) == 2
    assert residues[0]["resname"] == "ALA"
    assert residues[0]["resid"] == 100
    assert residues[1]["resname"] == "GLY"
    assert residues[1]["resid"] == 100
    # The two residues carry their own atoms rather than merging into one.
    assert set(residues[0]["atoms"]) == {"N", "CA"}
    assert set(residues[1]["atoms"]) == {"N", "CA"}
    assert np.allclose(residues[0]["atoms"]["CA"], [1.5, 2.5, 3.5])
    assert np.allclose(residues[1]["atoms"]["CA"], [4.5, 5.5, 6.5])


def test_no_insertion_code_groups_by_resid(tmp_path):
    """Without insertion codes the parser groups atoms by sequence number, yielding two residues with resids 1 and 2."""
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  N   ALA A   2       2.000   0.000   0.000  1.00  0.00           N",
        "ATOM      4  CA  ALA A   2       3.000   0.000   0.000  1.00  0.00           C",
    ]
    pdb = _write_pdb(tmp_path, lines)
    residues = _parse_pdb_chain_for_beads(pdb, chain_id="A")

    assert len(residues) == 2
    assert [r["resid"] for r in residues] == [1, 2]
    assert set(residues[0]["atoms"]) == {"N", "CA"}
    assert set(residues[1]["atoms"]) == {"N", "CA"}
