"""WCA of the GPU engine against the sum over pairs of real atoms. The tests need
a GPU."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1).sum()
except Exception:
    pytest.skip("no usable CUDA device", allow_module_level=True)

from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine

EPS = 0.17


def _engine(rec, rec_r, lig_r):
    return GPUBatchForceEngine([], [], alpha=0.0, rec_positions=rec, rec_radii=rec_r,
                               lig_radii=lig_r, use_lj=True)


def _pairs(rec, rec_r, lig, lig_r, cen):
    """Force and torque about cen of 4 eps ((s/r)^12 - (s/r)^6) below 2^(1/6) s."""
    F, T = np.zeros((len(lig), 3)), np.zeros((len(lig), 3))
    for n in range(len(lig)):
        for j in np.nonzero(lig_r > 0)[0]:
            d = lig[n, j] - rec[rec_r > 0]
            r2 = (d * d).sum(axis=1)
            s = lig_r[j] + rec_r[rec_r > 0]
            sr6 = (s * s / r2) ** 3
            f = np.where(np.sqrt(r2) < 2 ** (1 / 6) * s, 4 * EPS * (12 * sr6 * sr6 - 6 * sr6) / r2, 0.0)
            fj = (f[:, None] * d).sum(axis=0)
            F[n] += fj
            T[n] += np.cross(lig[n, j] - cen[n], fj)
    return F, T


def test_wca_equals_the_pair_sum_over_real_atoms():
    g = np.random.default_rng(21)
    rec = g.uniform(-6.0, 6.0, (40, 3))
    rec_r = g.uniform(1.2, 1.9, 40)
    rec_r[[0, 17]] = 0.0
    lig0 = np.vstack([g.normal(size=(5, 3)), np.zeros((1, 3))])
    lig_r = np.append(g.uniform(1.2, 1.9, 5), 0.0)
    q = g.normal(size=(100, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]).transpose(2, 0, 1)
    u = g.normal(size=(100, 3))
    cen = u / np.linalg.norm(u, axis=1, keepdims=True) * g.uniform(6.0, 12.0, (100, 1))
    lig = np.einsum("nij,kj->nki", R, lig0) + cen[:, None]
    F, T = _engine(rec, rec_r, lig_r)._wca_forces_gpu(cp.asarray(lig), cp.asarray(cen))
    F_ref, T_ref = _pairs(rec, rec_r, lig, lig_r, cen)
    hit = np.linalg.norm(F_ref, axis=1) > 0
    assert hit.sum() > 20
    assert np.allclose(cp.asnumpy(F), F_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(cp.asnumpy(T), T_ref, rtol=1e-10, atol=1e-12)


def test_ghost_atoms_take_no_part_and_radii_are_needed():
    th = 2.0 * np.pi * np.arange(7) / 7.0
    rec = np.vstack([np.stack([4.5 * np.cos(th), 4.5 * np.sin(th), np.zeros(7)], axis=1), np.zeros((1, 3))])
    eng = _engine(rec, np.array([1.7] * 7 + [0.0]), np.array([1.7, 0.0]))
    lig = cp.asarray([[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    F, T = eng._wca_forces_gpu(lig, lig[:, 1])
    assert not F.any() and not T.any()
    with pytest.raises(ValueError, match="radius sets"):
        GPUBatchForceEngine([], [], alpha=0.0, rec_positions=rec, use_lj=True)
