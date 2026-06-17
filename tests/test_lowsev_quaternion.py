"""Tests for low-severity robustness fixes in pystarc.transforms.quaternion."""

import numpy as np

from pystarc.transforms.quaternion import Quaternion, RigidTransform


def test_apply_accepts_python_list_single_point():
    """A single point given as a Python list should not raise AttributeError.

    The identity transform leaves the point unchanged, and the result should
    be a 1D array of shape (3,) matching a (3,) input.
    """
    t = RigidTransform.identity()
    result = t.apply([1.0, 2.0, 3.0])
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_apply_accepts_python_list_of_points():
    """A list of points should be treated as a (N, 3) array and return (N, 3)."""
    t = RigidTransform.identity()
    result = t.apply([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_apply_list_matches_ndarray_healthy_path():
    """List input must give the same result as the equivalent ndarray input."""
    rot = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    t = RigidTransform(rot, np.array([0.5, -1.0, 2.0]))
    pts_list = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    pts_arr = np.array(pts_list, dtype=float)
    np.testing.assert_allclose(t.apply(pts_list), t.apply(pts_arr))
