"""
Tests for the low-severity robustness guards in make_default_reaction.

These cover the degenerate cases where one or both molecules are empty, where a
non-positive pair count is requested, and where a molecule is too small to supply
the requested number of contact pairs. The healthy path for normally sized
molecules is verified to be unchanged.
"""

import numpy as np
import pytest

from pystarc.structures.molecules import Atom, Molecule
from pystarc.pathways.reaction_interface import make_default_reaction


def _line_molecule(name, n_atoms, x0=0.0):
    """Build a molecule of n_atoms atoms spaced one angstrom apart along x."""
    mol = Molecule(name=name)
    mol.atoms = [Atom(x=float(i) + x0) for i in range(n_atoms)]
    return mol


def test_empty_first_molecule_raises():
    """An empty mol1 must not yield a zero-pair always-firing reaction."""
    mol1 = Molecule(name="empty")
    mol2 = _line_molecule("m2", 5, x0=20.0)
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2)


def test_empty_second_molecule_raises():
    """An empty mol2 must not yield a zero-pair always-firing reaction."""
    mol1 = _line_molecule("m1", 5)
    mol2 = Molecule(name="empty")
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2)


def test_both_empty_molecules_raise():
    """Two empty molecules must raise rather than build an always-firing reaction."""
    with pytest.raises(ValueError):
        make_default_reaction(Molecule(name="a"), Molecule(name="b"))


@pytest.mark.parametrize("bad_n", [0, -1, -3])
def test_non_positive_n_pairs_raises(bad_n):
    """A non-positive n_pairs would give empty criteria and must raise."""
    mol1 = _line_molecule("m1", 5)
    mol2 = _line_molecule("m2", 5, x0=20.0)
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2, n_pairs=bad_n)


def test_small_molecule_clamps_pair_count():
    """When a molecule is smaller than n_pairs, the pair count is clamped to the
    smaller atom count and the reaction stays non-degenerate."""
    mol1 = _line_molecule("m1", 5)
    mol2 = _line_molecule("m2", 1, x0=20.0)
    rxn = make_default_reaction(mol1, mol2, n_pairs=3)
    # Only one atom is available on mol2, so exactly one contact pair can form.
    assert len(rxn.criteria.pairs) == 1
    # The criterion must require at least one contact, so it is not always satisfied.
    assert len(rxn.criteria.pairs) >= 1


def test_single_atom_molecules_produce_one_pair():
    """Two single-atom molecules give one valid contact pair, not a degenerate
    empty reaction."""
    mol1 = _line_molecule("m1", 1)
    mol2 = _line_molecule("m2", 1, x0=20.0)
    rxn = make_default_reaction(mol1, mol2, n_pairs=3)
    assert len(rxn.criteria.pairs) == 1


def test_normal_molecules_unchanged():
    """For molecules large enough to supply n_pairs atoms on both sides, the
    requested number of pairs is produced exactly as before the guard was added."""
    mol1 = _line_molecule("m1", 10)
    mol2 = _line_molecule("m2", 10, x0=20.0)
    for n in (1, 2, 3, 4, 5):
        rxn = make_default_reaction(mol1, mol2, n_pairs=n)
        assert len(rxn.criteria.pairs) == n
        # Pairs index the closest atoms on each side, which are well defined.
        for pair in rxn.criteria.pairs:
            assert 0 <= pair.mol1_atom_index < len(mol1.atoms)
            assert 0 <= pair.mol2_atom_index < len(mol2.atoms)
