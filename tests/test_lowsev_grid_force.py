"""
Low-severity robustness tests for pystarc.forces.electrostatic.grid_force.DXGrid.

These exercise the input-validation guards added for non-physical grid
spacings, non-orthogonal grids, and malformed DX files. They do not require a
GPU, APBS, or any external binary.
"""

import numpy as np
import pytest

from pystarc.forces.electrostatic.grid_force import DXGrid


def _orthogonal_delta(hx=1.0, hy=1.0, hz=1.0):
    return np.array([[hx, 0.0, 0.0], [0.0, hy, 0.0], [0.0, 0.0, hz]])


def test_valid_orthogonal_grid_sets_inv_dx():
    """A valid orthogonal grid still builds and computes the unchanged _inv_dx."""
    delta = _orthogonal_delta(0.5, 2.0, 4.0)
    data = np.zeros((3, 3, 3))
    grid = DXGrid(origin=np.zeros(3), delta=delta, data=data)
    assert np.allclose(grid._inv_dx, 1.0 / np.array([0.5, 2.0, 4.0]))


def test_zero_spacing_raises():
    delta = _orthogonal_delta(1.0, 0.0, 1.0)
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="axis 1"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_negative_spacing_raises():
    delta = _orthogonal_delta(1.0, 1.0, -1.0)
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="axis 2"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_non_orthogonal_grid_raises():
    delta = _orthogonal_delta(1.0, 1.0, 1.0)
    delta[0, 1] = 0.3  # large off-diagonal entry
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="not orthogonal"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_tiny_off_diagonal_is_accepted():
    """Numerical noise far below the diagonal scale must not trip the guard."""
    delta = _orthogonal_delta(2.0, 2.0, 2.0)
    delta[1, 0] = 1e-12  # negligible relative to spacing of 2.0
    data = np.zeros((3, 3, 3))
    grid = DXGrid(origin=np.zeros(3), delta=delta, data=data)
    assert np.allclose(grid._inv_dx, 0.5)


def _write_dx(tmp_path, nx, ny, nz, n_values):
    lines = [
        f"object 1 class gridpositions counts {nx} {ny} {nz}",
        "origin 0.0 0.0 0.0",
        "delta 1.0 0.0 0.0",
        "delta 0.0 1.0 0.0",
        "delta 0.0 0.0 1.0",
        "object 2 class gridconnections counts {0} {1} {2}".format(nx, ny, nz),
        f"object 3 class array type double rank 0 items {n_values} data follows",
    ]
    values = [str(float(i)) for i in range(n_values)]
    # Six values per line, mimicking the OpenDX layout.
    for start in range(0, n_values, 6):
        lines.append(" ".join(values[start : start + 6]))
    lines.append('attribute "dep" string "positions"')
    path = tmp_path / "grid.dx"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_from_file_value_count_mismatch_raises(tmp_path):
    # Declare a 3x3x3 grid (27 values) but only supply 20.
    path = _write_dx(tmp_path, 3, 3, 3, n_values=20)
    with pytest.raises(ValueError) as excinfo:
        DXGrid.from_file(path)
    msg = str(excinfo.value)
    assert str(path) in msg
    assert "27" in msg  # expected count
    assert "20" in msg  # actual count


def test_from_file_well_formed_roundtrip(tmp_path):
    nx, ny, nz = 2, 3, 4
    path = _write_dx(tmp_path, nx, ny, nz, n_values=nx * ny * nz)
    grid = DXGrid.from_file(path)
    assert grid.data.shape == (nx, ny, nz)
    # Values were written as 0..N-1 in C order.
    expected = np.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz)
    assert np.array_equal(grid.data, expected)
