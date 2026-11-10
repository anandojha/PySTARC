"""
The chunk skip of the reverse Born force in the GPU force engine.

The reverse Born force reads the ligand Born grids at the receptor atoms in the
ligand frame. _eval_born_reverse skips a chunk of poses when no receptor atom
can fall inside any ligand Born grid. The grids are sorted finest first. A skip
test against the first grid alone drops the whole force of a chunk of far poses
while receptor atoms still sit inside a coarser grid. A skip test against a box
half width also misses atoms near a box corner. The reach must be the farthest
usable box corner over all ligand Born grids.

The engine runs here with NumPy standing in for CuPy and with a NumPy copy of
the CUDA kernel. Each grid holds a linear field. The trilinear interpolation
and the central difference of a linear field are exact. The reference force is
therefore known in closed form.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pystarc.forces.electrostatic.grid_force import DXGrid

ALPHA = 0.7
# Two ligand Born grids of 33 points per axis around the ligand center. The
# usable box sits 3 spacings in from each edge. The fine box half width is
# 4 - 0.75 = 3.25 Å. The coarse box half width is 8 - 1.5 = 6.5 Å.
FINE_SPACING, FINE_HALF = 0.25, 3.25
COARSE_SPACING, COARSE_HALF = 0.5, 6.5
A_FINE = np.array([0.3, -0.2, 0.1])  # field gradient on the fine grid
A_COARSE = np.array([0.04, 0.07, -0.05])  # field gradient on the coarse grid
# Receptor atoms near the receptor center. The largest distance is 1.2247 Å.
REC_POS = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-0.8, 0.3, 0.6]])
REC_Q = np.array([0.5, -1.0, 0.25])


def _linear_grid(spacing: float, gradient: np.ndarray, n: int = 33) -> DXGrid:
    origin = np.full(3, -0.5 * (n - 1) * spacing)
    axis = np.arange(n) * spacing
    x, y, z = np.meshgrid(*(origin[i] + axis for i in range(3)), indexing="ij")
    data = gradient[0] * x + gradient[1] * y + gradient[2] * z
    return DXGrid(origin, np.eye(3) * spacing, data)


def _batch_force_kernel(blocks, threads, args):
    """NumPy copy of batch_force_kernel over every thread at once."""
    pos, chg, data, origin, inv_sp, sp, alpha, is_born = args[:8]
    nx, ny, nz, _n_traj, _n_atoms, out_f, out_e = args[8:]
    dims = np.array([int(nx), int(ny), int(nz)])
    grid = data.reshape(tuple(dims))
    p = pos.reshape(-1, 3)
    q = chg.reshape(-1)

    def interp(x):
        s = (x - origin) * inv_sp
        i0 = np.floor(s).astype(np.int64)
        frac = s - i0
        inside = np.all((i0 >= 0) & (i0 < dims - 1), axis=1)
        i0[~inside] = 0
        value = np.zeros(len(x))
        for corner in itertools.product((0, 1), repeat=3):
            c = np.array(corner)
            weight = np.prod(np.where(c == 1, frac, 1.0 - frac), axis=1)
            idx = i0 + c
            value += grid[idx[:, 0], idx[:, 1], idx[:, 2]] * weight
        return np.where(inside, value, 0.0)

    grad = np.zeros_like(p)
    for k in range(3):
        step = np.zeros(3)
        step[k] = 0.5 * sp[k]
        grad[:, k] = (interp(p + step) - interp(p - step)) / sp[k]
    coeff = -alpha * q * q if is_born else -q
    e_coeff = alpha * q * q if is_born else q
    live = np.abs(q) >= 1e-9
    out_f[:] = np.where(live[:, None], coeff[:, None] * grad, 0.0).ravel()
    out_e[:] = np.where(live, e_coeff * interp(p), 0.0)


@pytest.fixture
def gbe_numpy(monkeypatch):
    """The GPU force engine module with NumPy standing in for CuPy."""
    import pystarc.forces.gpu_batch_engine as gbe

    monkeypatch.setattr(gbe, "cp", np, raising=False)
    return gbe


def _engine(gbe, rec_pos=REC_POS, rec_q=REC_Q, grids=None):
    eng = object.__new__(gbe.GPUBatchForceEngine)
    eng.alpha = ALPHA
    eng._kernel = _batch_force_kernel
    eng._has_yukawa = False
    eng._call_count = 0
    # The pipeline passes the coarse grid first. _upload_grids sorts finest first.
    if grids is None:
        grids = [
            _linear_grid(COARSE_SPACING, A_COARSE),
            _linear_grid(FINE_SPACING, A_FINE),
        ]
    eng._lig_born_grids_gpu = eng._upload_grids(grids)
    eng._rec_pos_gpu = np.asarray(rec_pos, dtype=np.float64)
    eng._rec_charges_gpu = np.asarray(rec_q, dtype=np.float64)
    return eng


def _count_batches(eng):
    """Record every _eval_batch call of the engine."""
    calls = []
    inner = eng._eval_batch

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return inner(*args, **kwargs)

    eng._eval_batch = spy
    return calls


def _rotations(rng, n):
    """Uniform random rotation matrices from unit quaternions (w, x, y, z)."""
    q = rng.normal(size=(n, 4))
    w, x, y, z = (q / np.linalg.norm(q, axis=1)[:, None]).T
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R.transpose(2, 0, 1)


def _poses(rng, n, radius):
    d = rng.normal(size=(n, 3))
    cen = radius * d / np.linalg.norm(d, axis=1)[:, None]
    return _rotations(rng, n), cen


def _reference(R, cen, rec_pos=REC_POS, rec_q=REC_Q):
    """Reverse Born force and torque on the ligand for each pose."""
    F = np.zeros((len(cen), 3))
    T = np.zeros((len(cen), 3))
    for n in range(len(cen)):
        f_lig = np.zeros(3)
        t_lig = np.zeros(3)
        for x, q in zip(rec_pos, rec_q):
            y = R[n].T @ (x - cen[n])
            if np.all(np.abs(y) < FINE_HALF):
                f = -ALPHA * q * q * A_FINE
            elif np.all(np.abs(y) < COARSE_HALF):
                f = -ALPHA * q * q * A_COARSE
            else:
                continue
            f_lig += f
            t_lig += np.cross(y, f)
        F[n] = -R[n] @ f_lig
        T[n] = -R[n] @ t_lig
    return F, T


def test_far_chunk_keeps_the_coarse_grid_force(gbe_numpy):
    """A chunk of far poses keeps the force of receptor atoms inside the coarse grid."""
    eng = _engine(gbe_numpy)
    rng = np.random.default_rng(11)
    R, cen = _poses(rng, 8, 5.0)
    # Every pose has min |c| - max |x_rec| = 3.775 Å beyond the fine box half
    # width of 3.25 Å. Every receptor atom lies within 6.23 Å of the ligand
    # center and so inside the coarse box.
    F_ref, T_ref = _reference(R, cen)
    assert np.all(np.linalg.norm(F_ref, axis=1) > 1e-3)
    F, T = eng._eval_born_reverse(R, cen, len(cen))
    assert np.allclose(F, F_ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(T, T_ref, rtol=1e-12, atol=1e-14)
    # The same poses next to one near pose give the same forces. The near pose
    # has every receptor atom inside the fine box.
    R_near, cen_near = _poses(rng, 1, 1.0)
    R_mix = np.concatenate([R, R_near])
    cen_mix = np.concatenate([cen, cen_near])
    F_mix, T_mix = eng._eval_born_reverse(R_mix, cen_mix, len(cen_mix))
    assert np.allclose(F_mix[:-1], F, rtol=1e-13, atol=0.0)
    assert np.allclose(T_mix[:-1], T, rtol=1e-13, atol=0.0)
    F_mix_ref, T_mix_ref = _reference(R_mix, cen_mix)
    assert np.allclose(F_mix, F_mix_ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(T_mix, T_mix_ref, rtol=1e-12, atol=1e-14)


def test_atom_near_a_corner_of_the_coarsest_grid_counts(gbe_numpy):
    """A receptor atom near a corner of the coarsest box counts beyond the box half width."""
    eng = _engine(gbe_numpy, rec_pos=np.zeros((1, 3)), rec_q=np.array([0.8]))
    corners = 6.4 * np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    R = _rotations(np.random.default_rng(12), 3)
    # The receptor atom at the origin sits at y = R^T (0 - c) = corner. Then
    # |c| = 11.09 Å lies beyond the coarse box half width of 6.5 Å and within
    # the coarse box corner distance of 11.26 Å.
    cen = -np.einsum("nij,nj->ni", R, corners)
    F, T = eng._eval_born_reverse(R, cen, 3)
    f_rec = -ALPHA * 0.8**2 * A_COARSE
    for n in range(3):
        t_rec = np.cross(corners[n], f_rec)
        assert np.allclose(F[n], -R[n] @ f_rec, rtol=1e-12, atol=1e-14)
        assert np.allclose(T[n], -R[n] @ t_rec, rtol=1e-12, atol=1e-14)


def test_chunk_beyond_every_grid_is_still_skipped(gbe_numpy):
    """A chunk whose poses cannot reach any ligand Born grid is skipped and gets zero."""
    eng = _engine(gbe_numpy)
    calls = _count_batches(eng)
    R, cen = _poses(np.random.default_rng(13), 6, 13.0)
    # min |c| - max |x_rec| = 11.775 Å exceeds the coarse box corner distance
    # of 6.5 sqrt(3) = 11.258 Å.
    F, T = eng._eval_born_reverse(R, cen, len(cen))
    assert calls == []
    assert np.all(F == 0.0) and np.all(T == 0.0)
    assert np.all(_reference(R, cen)[0] == 0.0)


def test_evaluated_chunk_without_atoms_in_a_box_adds_exact_zeros(gbe_numpy):
    """A chunk within reach with every atom outside every box keeps the force bit for bit."""
    eng = _engine(gbe_numpy)
    calls = _count_batches(eng)
    R = np.tile(np.eye(3), (3, 1, 1))
    cen = 9.0 * np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    # min |c| - max |x_rec| = 7.775 Å lies within the reach of 11.258 Å. The
    # chunk is evaluated. Each atom has one coordinate of at least 8 Å and
    # sits outside both boxes.
    F, T = eng._eval_born_reverse(R, cen, 3)
    assert len(calls) == 1
    assert np.all(F == 0.0) and np.all(T == 0.0)
    base = np.random.default_rng(14).normal(size=(3, 3))
    base[0, 1] = 0.0
    assert np.array_equal((base + F).view(np.int64), base.view(np.int64))
    assert np.array_equal((base + T).view(np.int64), base.view(np.int64))


def test_reach_covers_a_finer_grid_that_is_the_larger_box(gbe_numpy):
    """The reach takes every grid, so a finer grid that is the larger box still counts."""
    # The fine grid of 0.25 Å and 65 points has a usable half width of
    # 8 - 0.75 = 7.25 Å. The coarse grid of 0.5 Å and 21 points has 5 - 1.5 = 3.5 Å.
    grids = [_linear_grid(0.5, A_COARSE, n=21), _linear_grid(0.25, A_FINE, n=65)]
    eng = _engine(gbe_numpy, rec_pos=np.zeros((1, 3)), rec_q=np.array([0.8]), grids=grids)
    y = 7.0 * np.array([[1.0, 1.0, 1.0]])
    R = _rotations(np.random.default_rng(15), 1)
    # The receptor atom at the origin sits at y = 7 (1, 1, 1) in the ligand
    # frame with |c| = 12.12 Å. It lies inside the fine box and outside the
    # coarse box.
    cen = -np.einsum("nij,nj->ni", R, y)
    F, T = eng._eval_born_reverse(R, cen, 1)
    f_rec = -ALPHA * 0.8**2 * A_FINE
    assert np.allclose(F[0], -R[0] @ f_rec, rtol=1e-12, atol=1e-14)
    assert np.allclose(T[0], -R[0] @ np.cross(y[0], f_rec), rtol=1e-12, atol=1e-14)
