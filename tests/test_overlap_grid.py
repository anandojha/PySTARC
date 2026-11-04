"""
Grid-filtered overlap check.

OverlapGrid must reject exactly the poses that the all-pairs test rejects. These
tests compare the two on random poses around a synthetic receptor for several
grid spacings and overlap fractions, check the lower bound on g that the
filter relies on, and pin down the strict threshold and the region off the
grid. They run with NumPy standing in for CuPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystarc.simulation.overlap_grid import OverlapGrid

RADII = np.array([0.0, 1.2, 1.3, 1.5, 1.55, 1.7, 1.8])


def _all_pairs(rec_pos, rec_radii, lig_pos, lig_radii, f):
    """The all-pairs test as written in the GPU simulator."""
    diff = lig_pos[:, :, None, :] - rec_pos[None, None, :, :]
    dist_sq = (diff * diff).sum(axis=-1)
    thresh = f * (lig_radii[:, None] + rec_radii[None, :])
    return (dist_sq < (thresh * thresh)[None]).any(axis=(1, 2))


def _random_rotations(rng, n):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(n, 3, 3)


@pytest.fixture(scope="module")
def system():
    """A receptor of 300 atoms in a 12 Å ball with a ghost at the origin, and a
    ligand of 12 atoms with a ghost at its centre."""
    rng = np.random.default_rng(11)
    u = rng.standard_normal((299, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rec_pos = np.vstack([u * 12.0 * rng.random((299, 1)) ** (1 / 3), np.zeros((1, 3))])
    rec_radii = np.append(rng.choice(RADII[1:], 299), 0.0)
    lig0 = rng.normal(0.0, 1.5, (11, 3))
    lig0 = np.vstack([lig0 - lig0.mean(axis=0), np.zeros((1, 3))])
    lig_radii = np.append(rng.choice(RADII[1:], 11), 0.0)
    n = 4000
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    centres = d * rng.uniform(6.0, 18.0, (n, 1))
    lig_pos = np.einsum("nij,kj->nki", _random_rotations(rng, n), lig0) + centres[:, None]
    return rec_pos, rec_radii, lig_pos, lig_radii


class TestOverlapGrid:
    @pytest.mark.parametrize("spacing", [0.25, 0.5, 1.0])
    @pytest.mark.parametrize("fraction", [0.5, 0.75])
    def test_matches_all_pairs(self, system, spacing, fraction):
        """The grid-filtered test rejects exactly the poses the all-pairs test rejects."""
        rec_pos, rec_radii, lig_pos, lig_radii = system
        grid = OverlapGrid(rec_pos, rec_radii, fraction, lig_radii.max(), spacing)
        ref = _all_pairs(rec_pos, rec_radii, lig_pos, lig_radii, fraction)
        assert 0 < ref.sum() < len(ref)
        np.testing.assert_array_equal(grid.overlaps(lig_pos, lig_radii), ref)

    def test_matches_all_pairs_in_small_chunks(self, system):
        """Chunking over poses does not change the decisions."""
        rec_pos, rec_radii, lig_pos, lig_radii = system
        grid = OverlapGrid(rec_pos, rec_radii, 0.5, lig_radii.max(), 0.5)
        ref = _all_pairs(rec_pos, rec_radii, lig_pos, lig_radii, 0.5)
        np.testing.assert_array_equal(grid.overlaps(lig_pos, lig_radii, budget=50), ref)

    def test_grid_bounds_g_from_below(self, system):
        """g at a point is never below g at its nearest grid point minus the slack."""
        rec_pos, rec_radii, _, _ = system
        grid = OverlapGrid(rec_pos, rec_radii, 0.5, 1.8, 0.7)
        rng = np.random.default_rng(5)
        hi = grid.origin + grid.spacing * (grid.shape - 1)
        y = rng.uniform(grid.origin, hi, (20000, 3))
        diff = y[:, None, :] - rec_pos[None, :, :]
        g = (np.sqrt((diff * diff).sum(axis=-1)) - 0.5 * rec_radii[None, :]).min(axis=1)
        ijk = np.rint((y - grid.origin) / grid.spacing).astype(np.int64)
        g_n = grid.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        assert np.all(g >= g_n - grid.slack)

    def test_threshold_is_strict(self):
        """A pair exactly at f (r_i + r_j) is clear and one just inside overlaps."""
        rec_pos = np.zeros((1, 3))
        rec_radii = np.array([1.5])
        grid = OverlapGrid(rec_pos, rec_radii, 0.5, 1.5, 0.5)
        lig = np.array([[[1.5, 0.0, 0.0]], [[1.4999, 0.0, 0.0]], [[0.0, -1.5, 0.0]]])
        np.testing.assert_array_equal(
            grid.overlaps(lig, np.array([1.5])), [False, True, False]
        )

    def test_atoms_off_the_grid_are_clear(self, system):
        """Ligand atoms beyond the grid cannot overlap and are cleared."""
        rec_pos, rec_radii, lig_pos, lig_radii = system
        grid = OverlapGrid(rec_pos, rec_radii, 0.5, lig_radii.max(), 0.5)
        far = lig_pos[:100] + np.array([100.0, 0.0, 0.0])
        assert not grid.overlaps(far, lig_radii).any()
