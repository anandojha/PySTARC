"""Validation tests for splitting a combined PQR into receptor and ligand atoms.

These tests exercise split_receptor_ligand, the helper that partitions the
combined PQR atom list by residue name. A residue name that matches no atoms
must raise a clear ValueError that names the offending residue name and lists
the residue names actually present.
"""

import pytest

from pystarc.pipeline.prepare_bd_surface import PQRAtom, split_receptor_ligand


def _make_atoms():
    """Build a small combined atom list with receptor (MGO) and ligand (APN) atoms."""
    return [
        PQRAtom(1, "C1", "MGO", 1, 0.0, 0.0, 0.0, -0.2, 1.7),
        PQRAtom(2, "C2", "MGO", 1, 1.0, 0.0, 0.0, 0.1, 1.7),
        PQRAtom(3, "O1", "MGO", 1, 0.0, 1.0, 0.0, -0.4, 1.5),
        PQRAtom(4, "N1", "APN", 2, 5.0, 5.0, 5.0, 0.3, 1.6),
        PQRAtom(5, "C3", "APN", 2, 6.0, 5.0, 5.0, 0.0, 1.7),
    ]


def test_matching_resnames_split_correctly():
    """Atoms split by matching residue names yield the receptor and ligand atom sets with correct sizes and names."""
    atoms = _make_atoms()
    rec, lig = split_receptor_ligand(atoms, "MGO", "APN")
    assert len(rec) == 3
    assert len(lig) == 2
    assert all(a.resname == "MGO" for a in rec)
    assert all(a.resname == "APN" for a in lig)


def test_nonmatching_receptor_resname_raises_named_valueerror():
    """A receptor resname matching no atoms raises a ValueError naming receptor_resname, the bad value, and the residue names present."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "XXX", "APN")
    msg = str(excinfo.value)
    # The error names the receptor residue name that matched nothing.
    assert "receptor_resname" in msg
    assert "XXX" in msg
    # The error lists residue names actually present in the PQR.
    assert "MGO" in msg
    assert "APN" in msg


def test_nonmatching_ligand_resname_raises_named_valueerror():
    """A ligand resname matching no atoms raises a ValueError naming ligand_resname, the bad value, and a present residue name."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "MGO", "ZZZ")
    msg = str(excinfo.value)
    assert "ligand_resname" in msg
    assert "ZZZ" in msg
    assert "MGO" in msg


def test_both_nonmatching_resnames_named_in_valueerror():
    """When both resnames match nothing, the ValueError message names both bad values."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "AAA", "BBB")
    msg = str(excinfo.value)
    assert "AAA" in msg
    assert "BBB" in msg
