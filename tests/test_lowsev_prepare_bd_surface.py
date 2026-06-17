"""Low-severity robustness tests for prepare_bd_surface.

These exercise input validation in compute_grid_params, which previously
produced a NaN grid centre and an empty-sequence max() ValueError with no
clear message when handed an atom list that contains no real atoms.
"""

import pytest

from pystarc.pipeline.prepare_bd_surface import PQRAtom, compute_grid_params


def _gho_atom(serial: int = 1) -> PQRAtom:
    """Return a GHO ghost atom at the origin with zero charge and radius."""
    return PQRAtom(
        serial=serial,
        name="GHO",
        resname="GHO",
        resid=serial,
        x=0.0,
        y=0.0,
        z=0.0,
        charge=0.0,
        radius=0.0,
    )


def _real_atom() -> PQRAtom:
    """Return a single ordinary atom away from the origin."""
    return PQRAtom(
        serial=1,
        name="C1",
        resname="LIG",
        resid=1,
        x=1.0,
        y=2.0,
        z=3.0,
        charge=0.1,
        radius=1.7,
    )


def test_compute_grid_params_empty_atom_list_raises():
    """An empty atom list raises a clear ValueError rather than max() on empty."""
    with pytest.raises(ValueError) as excinfo:
        compute_grid_params([])
    assert "no real atoms" in str(excinfo.value)


def test_compute_grid_params_only_gho_raises():
    """An atom list of only GHO ghost atoms raises the same clear ValueError."""
    with pytest.raises(ValueError) as excinfo:
        compute_grid_params([_gho_atom(1), _gho_atom(2)])
    assert "no real atoms" in str(excinfo.value)


def test_compute_grid_params_healthy_path_still_works():
    """A list with at least one real atom still returns the expected grid blocks."""
    grids = compute_grid_params([_real_atom(), _gho_atom(2)])
    assert len(grids) == 3
    for g in grids:
        assert set(g.keys()) == {"spacing", "dime", "glen", "gcent"}
        assert len(g["dime"]) == 3
        assert len(g["glen"]) == 3
        assert len(g["gcent"]) == 3
    # The grid centre is the heavy-atom centroid, ignoring the GHO atom.
    assert grids[0]["gcent"] == [1.0, 2.0, 3.0]
