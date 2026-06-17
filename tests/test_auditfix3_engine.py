"""Regression tests for the CPU unified force engine fixes."""
import numpy as np

from pystarc.forces.engine import PySTARCEngine, _group_centroid
from pystarc.forces.lj import LJForceEngine, LJParams, LJAtomType
from pystarc.global_defs.constants import KCAL_PER_MOL_TO_KBT
from pystarc.structures.molecules import Molecule, Atom


def _mol(coords, charge=0.0, radius=1.8):
    m = Molecule(name="m")
    m.atoms = [Atom(x=c[0], y=c[1], z=c[2], charge=charge, radius=radius) for c in coords]
    return m


def test_group_centroid_uses_only_charged_atoms():
    """_group_centroid averages only charged atoms and returns the origin when no atom is charged."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    charges = np.array([1.0, -1.0, 1e-12])
    # The third atom is uncharged and must be excluded, leaving the mean of the
    # first two at (1, 0, 0).
    assert np.allclose(_group_centroid(positions, charges), [1.0, 0.0, 0.0])
    # With no charged atoms the reference is the origin.
    assert np.allclose(_group_centroid(positions, np.zeros(3)), [0.0, 0.0, 0.0])


def test_lj_type_id_fallback_and_kbt_conversion():
    # One Lennard-Jones type shared by every atom; the engine must map all atoms
    # to type index 0 rather than to per-atom indices (which would index past a
    # single-type table).
    """A single shared Lennard-Jones type maps all atoms to type index 0 without an IndexError, and the engine converts the LJ contribution from kcal/mol to kBT."""
    ljp = LJParams(atom_types=[LJAtomType(name="X", epsilon=0.2, sigma=3.0)])
    engine = PySTARCEngine(lj_params=ljp)  # no electrostatic or Born grids

    # Three atoms per molecule (more atoms than Lennard-Jones types).
    mol1 = _mol([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mol2 = _mol([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]])

    # Must not raise an IndexError despite atoms outnumbering the single type.
    force, torque, energy = engine(mol1, mol2)
    assert np.all(np.isfinite(force)) and np.isfinite(energy)

    # With no grids the engine output is the Lennard-Jones contribution alone,
    # which must be converted from kcal/mol to kBT before accumulation.
    raw = LJForceEngine(ljp)
    pos1 = mol1.positions_array()
    pos2 = mol2.positions_array()
    _, f2_raw, e_raw = raw.compute(pos1, pos2, [0, 0, 0], [0, 0, 0])
    assert np.linalg.norm(f2_raw) > 0.0  # the configuration is within range
    assert np.allclose(force, f2_raw * KCAL_PER_MOL_TO_KBT)
    assert np.isclose(energy, e_raw * KCAL_PER_MOL_TO_KBT)
