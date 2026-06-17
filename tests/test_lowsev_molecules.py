"""Low-severity robustness tests for pystarc.structures.molecules.

These tests cover the input-validation guard added to
ReactionCriteria.is_satisfied so that an out-of-range contact-pair atom index
raises a clear, descriptive error instead of a bare IndexError raised deep
inside a trajectory loop. The healthy path with valid indices is also exercised
to confirm its behavior is unchanged.
"""

import pytest

from pystarc.structures.molecules import Atom, ContactPair, Molecule, ReactionCriteria


def _two_atom_mol(name, x):
    """Build a small molecule with two atoms positioned along the x axis."""
    return Molecule(name=name, atoms=[Atom(x=x), Atom(x=x + 1.0)])


def test_out_of_range_mol1_index_raises_clear_error():
    """An mol1 contact index past the atom count names the index and atom count."""
    mol1 = _two_atom_mol("ligand", 0.0)
    mol2 = _two_atom_mol("receptor", 0.0)
    # mol1 has 2 atoms, so index 5 is out of range.
    criteria = ReactionCriteria(name="assoc", pairs=[ContactPair(5, 0, 5.0)])
    with pytest.raises(IndexError) as excinfo:
        criteria.is_satisfied(mol1, mol2)
    msg = str(excinfo.value)
    assert "5" in msg
    assert "2 atoms" in msg
    assert "ligand" in msg
    assert "assoc" in msg


def test_out_of_range_mol2_index_raises_clear_error():
    """An mol2 contact index past the atom count names the index and atom count."""
    mol1 = _two_atom_mol("ligand", 0.0)
    mol2 = _two_atom_mol("receptor", 0.0)
    criteria = ReactionCriteria(name="assoc", pairs=[ContactPair(0, 7, 5.0)])
    with pytest.raises(IndexError) as excinfo:
        criteria.is_satisfied(mol1, mol2)
    msg = str(excinfo.value)
    assert "7" in msg
    assert "receptor" in msg


def test_valid_indices_unchanged_true():
    """Healthy path with valid in-range indices still fires when within cutoff."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 0.0)  # atom0 of each coincides, distance 0
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(0, 0, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is True


def test_valid_indices_unchanged_false():
    """Healthy path with valid in-range indices returns False beyond the cutoff."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 100.0)  # far apart, beyond cutoff
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(0, 0, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is False


def test_negative_wraparound_index_still_works():
    """Negative-wraparound indices that were valid before remain valid (no behavior change)."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 0.0)
    # -1 refers to the last atom; atom1 of each is at x=1, distance 0, within cutoff.
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(-1, -1, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is True
