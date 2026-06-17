"""
Tests for the empty-reactions warning in WESimulator._make_bins.

When the PathwaySet carries no reactions, the bin edges collapse to the narrow
interval [0.9 * r_start, r_start] and provide no resolution in the reaction
zone. The simulator should warn in that case while leaving the bins unchanged.
"""

import types
import warnings

import numpy as np

from pystarc.simulation.we_simulator import WESimulator, WEParameters
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Molecule, Atom


def _make_mol(name: str) -> Molecule:
    mol = Molecule(name=name)
    mol.atoms = [
        Atom(index=0, x=0.0, y=0.0, z=0.0, charge=1.0, radius=1.5),
        Atom(index=1, x=2.0, y=0.0, z=0.0, charge=-1.0, radius=1.5),
    ]
    return mol


def _make_simulator(pathway_set: PathwaySet) -> WESimulator:
    mol1 = _make_mol("rec")
    mol2 = _make_mol("lig")
    mobility = MobilityTensor.from_radii(20.0, 20.0)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=1,
        r_start=40.0,
        r_escape=80.0,
        seed=1,
    )
    return WESimulator(mol1, mol2, mobility, pathway_set, params)


def test_empty_reactions_emits_warning():
    """Constructing a WESimulator with an empty PathwaySet warns about the
    collapsed reaction-zone resolution."""
    pathway_set = PathwaySet()  # no reactions
    assert pathway_set.reactions == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = _make_simulator(pathway_set)
    messages = [str(w.message) for w in caught]
    assert any("no reactions" in m for m in messages), messages
    # The bins must still be the unchanged [0.9 * r_start, r_start] interval.
    bins = sim._bins
    assert len(bins) == 6  # n_bins + 1 edges
    assert np.isclose(bins[0], max(40.0 * 0.9, 1.0))
    assert np.isclose(bins[-1], 40.0)


def test_nonempty_reactions_no_warning():
    """When the PathwaySet carries a reaction with a contact cutoff, no
    empty-reactions warning is emitted and the bins reach the cutoff zone."""
    # A lightweight duck-typed reaction exposing the attributes the simulator
    # reads, namely criteria.pairs with a distance_cutoff on each pair.
    pair = types.SimpleNamespace(distance_cutoff=5.0)
    criteria = types.SimpleNamespace(pairs=[pair])
    rxn = types.SimpleNamespace(criteria=criteria)
    pathway_set = PathwaySet()
    pathway_set.reactions = [rxn]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = _make_simulator(pathway_set)
    messages = [str(w.message) for w in caught]
    assert not any("no reactions" in m for m in messages), messages
    # The lower bin edge follows the reaction cutoff (0.9 * 5.0), not r_start.
    assert np.isclose(sim._bins[0], max(5.0 * 0.9, 1.0))
