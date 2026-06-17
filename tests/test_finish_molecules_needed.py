"""Tests for the effective-threshold-zero behavior of ReactionCriteria.is_satisfied.

These tests check that a criterion whose effective threshold is zero, meaning
n_needed is set to 0 or n_needed is left negative while the pair list is empty,
reports as not satisfied, matching BrownDye2. They also confirm that the normal
n_needed >= 1 path is unchanged, so the rigid-body benchmark behavior is
preserved.
"""

from pystarc.structures.molecules import (
    Atom,
    Molecule,
    ContactPair,
    ReactionCriteria,
)


def _two_molecules(separation: float) -> tuple[Molecule, Molecule]:
    """Build two single-atom molecules placed separation angstrom apart on x."""
    mol1 = Molecule(name="mol1", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0)])
    mol2 = Molecule(name="mol2", atoms=[Atom(index=0, x=separation, y=0.0, z=0.0)])
    return mol1, mol2


def test_zero_needed_explicit_returns_false():
    """A criterion with n_needed == 0 can never fire, even with close contacts."""
    mol1, mol2 = _two_molecules(separation=1.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="zero_needed", pairs=pairs, n_needed=0)
    assert criterion.is_satisfied(mol1, mol2) is False


def test_negative_needed_empty_pairs_returns_false():
    """A criterion with no pairs and the default n_needed == -1 cannot fire."""
    mol1, mol2 = _two_molecules(separation=1.0)
    criterion = ReactionCriteria(name="empty", pairs=[], n_needed=-1)
    assert criterion.is_satisfied(mol1, mol2) is False


def test_n_needed_one_satisfied_unchanged():
    """With n_needed == 1 a single close contact still satisfies the criterion."""
    mol1, mol2 = _two_molecules(separation=1.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="one_needed", pairs=pairs, n_needed=1)
    assert criterion.is_satisfied(mol1, mol2) is True


def test_n_needed_one_not_satisfied_unchanged():
    """With n_needed == 1 a contact beyond the cutoff still fails the criterion."""
    mol1, mol2 = _two_molecules(separation=10.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="one_needed", pairs=pairs, n_needed=1)
    assert criterion.is_satisfied(mol1, mol2) is False
