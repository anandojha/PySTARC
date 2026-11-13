"""
Born direction 2 in the GPU force engine: the reach of born2_kernel.

The kernel reads the ligand Born grids at the receptor atoms in the ligand
frame, each atom on the finest grid whose usable box (one spacing in from the
faces) holds it. It skips a pose whose centre is farther than the reach from
every receptor atom, and an atom beyond the reach. The reach is the smaller of
the largest distance to a box corner and the farthest nonzero node plus 2.5
spacings, so a skip never drops a nonzero force. The reference below evaluates
every atom with NumPy and no skip. The tests need a GPU.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pystarc.forces.electrostatic.grid_force import DXGrid

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1).sum()
except Exception:
    pytest.skip("no usable CUDA device", allow_module_level=True)

ALPHA = 0.7
A_FINE = np.array([0.3, -0.2, 0.1])
A_COARSE = np.array([0.04, 0.07, -0.05])
REC_POS = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-0.8, 0.3, 0.6]])
REC_Q = np.array([0.5, -1.0, 0.25])


def _grid(spacing, field, n=33):
    origin = np.full(3, -0.5 * (n - 1) * spacing)
    axis = origin[0] + np.arange(n) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return DXGrid(origin, np.eye(3) * spacing, field(x, y, z))


def _linear(a):
    return lambda x, y, z: a[0] * x + a[1] * y + a[2] * z


def _interp(g, x):
    """Trilinear interpolation of the DXGrid g at the points x, zero outside."""
    sp = np.diag(g.delta)
    dims = np.array(g.data.shape)
    s = (x - g.origin) / sp
    i0 = np.floor(s).astype(np.int64)
    frac = s - i0
    inside = np.all((i0 >= 0) & (i0 < dims - 1), axis=1)
    i0[~inside] = 0
    v = np.zeros(len(x))
    for c in itertools.product((0, 1), repeat=3):
        w = np.prod(np.where(np.array(c) == 1, frac, 1.0 - frac), axis=1)
        idx = i0 + np.array(c)
        v += g.data[idx[:, 0], idx[:, 1], idx[:, 2]] * w
    return np.where(inside, v, 0.0)


def _engine(grids, rec_pos=REC_POS, rec_q=REC_Q):
    import pystarc.forces.gpu_batch_engine as gbe

    eng = object.__new__(gbe.GPUBatchForceEngine)
    eng.alpha = ALPHA
    eng._lig_born_grids_gpu = eng._upload_grids(grids, margin=1.0)
    eng._rec_pos_gpu = cp.asarray(rec_pos, dtype=cp.float64)
    eng._rec_charges_gpu = cp.asarray(rec_q, dtype=cp.float64)
    eng._init_born_reverse()
    return eng


def _reference(grids, R, cen, rec_pos=REC_POS, rec_q=REC_Q):
    """Force and torque on the ligand with every receptor atom evaluated."""
    grids = sorted(grids, key=lambda g: g.delta[0, 0])
    F = np.zeros((len(cen), 3))
    T = np.zeros((len(cen), 3))
    for n in range(len(cen)):
        y = (rec_pos - cen[n]) @ R[n]
        f = np.zeros_like(y)
        done = np.zeros(len(y), dtype=bool)
        for g in grids:
            sp = np.diag(g.delta)
            lo = g.origin + sp
            hi = g.origin + (np.array(g.data.shape) - 2) * sp
            use = ~done & np.all((y > lo) & (y < hi), axis=1)
            for k in range(3):
                h = np.zeros(3)
                h[k] = 0.5 * sp[k]
                d = (_interp(g, y[use] + h) - _interp(g, y[use] - h)) / sp[k]
                f[use, k] = -ALPHA * rec_q[use] ** 2 * d
            done |= use
        F[n] = -R[n] @ f.sum(axis=0)
        T[n] = -R[n] @ np.cross(y, f).sum(axis=0)
    return F, T


def _poses(rng, radii):
    q = rng.normal(size=(len(radii), 4))
    w, x, y, z = (q / np.linalg.norm(q, axis=1)[:, None]).T
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)
    d = rng.normal(size=(len(radii), 3))
    return R, np.asarray(radii)[:, None] * d / np.linalg.norm(d, axis=1)[:, None]


def _run(eng, R, cen):
    F, T = eng._eval_born_reverse(cp.asarray(R), cp.asarray(cen), len(cen))
    return cp.asnumpy(F), cp.asnumpy(T)


def test_born_grids_take_one_spacing_and_apbs_grids_three():
    import pystarc.forces.gpu_batch_engine as gbe

    eng = object.__new__(gbe.GPUBatchForceEngine)
    g = _grid(0.5, _linear(A_COARSE))
    assert np.allclose(eng._upload_grids([g], margin=1.0)[0]["lo"], g.origin + 0.5)
    assert np.allclose(eng._upload_grids([g])[0]["lo"], g.origin + 1.5)


def test_force_matches_every_atom_reference_at_all_distances():
    """Fine and coarse grids, poses from inside to beyond every box."""
    grids = [_grid(0.5, _linear(A_COARSE)), _grid(0.25, _linear(A_FINE))]
    eng = _engine(grids)
    R, cen = _poses(np.random.default_rng(11), np.linspace(0.5, 14.0, 200))
    F_ref, T_ref = _reference(grids, R, cen)
    F, T = _run(eng, R, cen)
    assert np.any(np.linalg.norm(F_ref, axis=1) == 0.0)
    assert np.any(np.linalg.norm(F_ref, axis=1) > 1e-3)
    assert np.allclose(F, F_ref, rtol=1e-12, atol=1e-15)
    assert np.allclose(T, T_ref, rtol=1e-12, atol=1e-15)


def test_atom_near_a_corner_of_the_coarsest_box_counts():
    """|c| beyond the box half width of 7.5 A and within the corner distance."""
    grids = [_grid(0.5, _linear(A_COARSE)), _grid(0.25, _linear(A_FINE))]
    eng = _engine(grids, rec_pos=np.zeros((1, 3)), rec_q=np.array([0.8]))
    corners = 7.3 * np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    R, _ = _poses(np.random.default_rng(12), np.ones(3))
    cen = -np.einsum("nij,nj->ni", R, corners)
    F, T = _run(eng, R, cen)
    f_rec = -ALPHA * 0.8**2 * A_COARSE
    for n in range(3):
        assert np.allclose(F[n], -R[n] @ f_rec, rtol=1e-12, atol=1e-15)
        assert np.allclose(T[n], -R[n] @ np.cross(corners[n], f_rec), rtol=1e-12, atol=1e-15)


def test_a_finer_grid_that_is_the_larger_box_counts():
    grids = [_grid(0.5, _linear(A_COARSE), n=21), _grid(0.25, _linear(A_FINE), n=65)]
    eng = _engine(grids, rec_pos=np.zeros((1, 3)), rec_q=np.array([0.8]))
    y = 7.0 * np.ones((1, 3))
    R, _ = _poses(np.random.default_rng(15), np.ones(1))
    F, T = _run(eng, R, -np.einsum("nij,nj->ni", R, y))
    f_rec = -ALPHA * 0.8**2 * A_FINE
    assert np.allclose(F[0], -R[0] @ f_rec, rtol=1e-12, atol=1e-15)
    assert np.allclose(T[0], -R[0] @ np.cross(y[0], f_rec), rtol=1e-12, atol=1e-15)


def test_reach_stops_at_the_nonzero_nodes_and_drops_nothing():
    """A field that is zero beyond 4 A gives a reach below the corner reach,
    and poses on both sides of it match the every-atom reference."""

    def cut(x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        return np.where(r < 4.0, (4.0 - r) ** 2 * (1.0 + 0.1 * x - 0.2 * z), 0.0)

    grids = [_grid(0.5, cut), _grid(0.25, cut)]
    eng = _engine(grids)
    far = max(
        np.linalg.norm(np.argwhere(g.data != 0) * g.delta[0, 0] + g.origin, axis=1).max()
        + 2.5 * g.delta[0, 0]
        for g in grids
    )
    assert eng._b2_reach == pytest.approx(1e-6 + far)
    assert eng._b2_reach < 7.5 * np.sqrt(3)
    R, cen = _poses(np.random.default_rng(16), np.linspace(2.0, 9.0, 300))
    F_ref, T_ref = _reference(grids, R, cen)
    F, T = _run(eng, R, cen)
    assert np.allclose(F, F_ref, rtol=1e-11, atol=1e-14)
    assert np.allclose(T, T_ref, rtol=1e-11, atol=1e-14)


def test_poses_beyond_the_reach_get_exact_zeros():
    grids = [_grid(0.5, _linear(A_COARSE)), _grid(0.25, _linear(A_FINE))]
    eng = _engine(grids)
    R, cen = _poses(np.random.default_rng(13), np.full(6, eng._b2_reach + 1.3))
    F, T = _run(eng, R, cen)
    assert np.all(F == 0.0) and np.all(T == 0.0)
