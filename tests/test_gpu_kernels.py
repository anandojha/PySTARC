"""The CUDA kernels of the GPU loop against NumPy and CuPy references. The tests
need a GPU."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1).sum()
except Exception:
    pytest.skip("no usable CUDA device", allow_module_level=True)

import pystarc.simulation.gpu_batch_simulator as gbs
from pystarc.simulation.gpu_random import GPURandom

K = gbs._step_kernels()


def _dt(r, force=True, V0=-3.0, debye=7.8, D_t=0.03, D_r1=1e-5):
    """Pair, force and edge rules of the time step, by the CuPy chain."""
    r_b, r_esc = 20.0, 40.0
    dt = cp.empty(len(r))
    outer = cp.empty(len(r), dtype=cp.bool_)
    inv = 1.0 / debye if force else 0.0
    K["dt"](r, 1.1 * r_b, 0.005, D_t, D_r1, force, V0, debye, inv, debye**2,
            3.0 * debye, 0.01, r_b, r_esc, 18.0 * D_t, dt, outer)
    pair = 0.005 * r * r / (D_t + D_r1 * r * r)
    f = pair
    if force:
        V = V0 * cp.exp(-r / debye) / r
        DF1 = cp.abs(D_t * (-V / debye**2 - 2.0 * V * (1.0 / r + 1.0 / debye) / r))
        f = cp.where((DF1 > 1e-15) & (r < 3.0 * debye), 0.01 / DF1, pair)
    e = cp.minimum(cp.maximum(r - r_b, 1e-3), cp.maximum(r_esc - r, 1e-3))
    ref = cp.where(r > 1.1 * r_b, cp.minimum(cp.minimum(f, e**2 / (18.0 * D_t)), pair),
                   cp.minimum(f, pair))
    return dt, ref


def test_time_step_matches_the_cupy_chain():
    r = cp.asarray(np.random.default_rng(1).uniform(5.0, 45.0, 5000))
    dt, ref = _dt(r)
    assert cp.allclose(dt, ref, rtol=1e-13, atol=0.0)


def test_time_step_without_force_runs_at_debye_zero():
    r = cp.asarray(np.random.default_rng(2).uniform(5.0, 45.0, 100))
    dt, ref = _dt(r, force=False, V0=0.0, debye=0.0)
    assert bool(cp.all(cp.isfinite(dt))) and cp.allclose(dt, ref, rtol=1e-13)


def test_nan_pose_keeps_nan_time_step_through_the_floor():
    r = cp.asarray([25.0, np.nan])
    dt, ref = _dt(r)
    assert bool(cp.isnan(dt[1])) and bool(cp.isnan(ref[1]))
    out = cp.empty(2)
    frc = cp.asarray([[0.1, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    K["surface"](r, dt, frc, cp.zeros((2, 3)), cp.full(2, 2.0), cp.zeros(2), True,
                 False, 3.0, 30.0, 0.03, 1e-5, 0.04, 0.01, 0.0, 0.2, out)
    assert bool(cp.isfinite(out[0])) and bool(cp.isnan(out[1]))


def test_fire_counts_pairs_that_ended_inside():
    u = cp.asarray([[0.5, 0.5], [0.9, 0.1], [0.0, 0.0]])
    pc = cp.asarray([[0.6, 0.0], [0.0, 0.2], [1.0, 1.0]])
    d1 = cp.asarray([[5.0, 3.0], [5.0, 5.0], [3.0, 3.0]])
    cut = cp.asarray([4.0, 4.0])
    rev = cp.asarray([False, False, True])
    fired = cp.empty((3, 2), dtype=cp.bool_)
    total, inside = cp.empty(3, dtype=cp.int64), cp.empty(3, dtype=cp.int64)
    hit = cp.empty(3, dtype=cp.bool_)
    K["fire"](u, pc, d1, cut, rev, True, 2, 2, fired, total, inside, hit)
    assert total.tolist() == [2, 1, 0] and inside.tolist() == [1, 0, 0]
    assert hit.tolist() == [True, False, False]


def test_random_numbers_do_not_depend_on_the_other_trajectories():
    rng = GPURandom(7)
    n_steps = cp.arange(10, dtype=cp.int64) * 3
    alone = rng.normal(cp.asarray([4]), n_steps, 3, rng.NOISE_T)
    batch = rng.normal(cp.arange(10), n_steps, 3, rng.NOISE_T)
    assert bool(cp.all(alone[0] == batch[4]))
    assert not bool(cp.all(batch[4] == rng.normal(cp.asarray([4]), n_steps + 1, 3, rng.NOISE_T)[0]))


def test_move_matches_the_reference_chain():
    sim = object.__new__(gbs.GPUBatchSimulator)
    g = np.random.default_rng(3)
    n = 400
    pos = cp.asarray(g.normal(size=(n, 3)) * 20.0)
    q = cp.asarray(g.normal(size=(n, 4)))
    q /= cp.linalg.norm(q, axis=1, keepdims=True)
    f, tq = cp.asarray(g.normal(size=(n, 3))), cp.asarray(g.normal(size=(n, 3)))
    nt, nr, nr1 = (cp.asarray(g.normal(size=(n, 3))) for _ in range(3))
    dt, D = cp.asarray(g.uniform(0.1, 2.0, n)), cp.asarray(g.uniform(0.01, 0.05, n))
    Dr1, Dr2 = 2e-5, 3e-4
    idx = cp.arange(n, dtype=cp.int64)
    cen = pos.copy()
    p_k, q_k = pos.copy(), q.copy()
    drift, sig = cp.empty((n, 3)), cp.empty(n)
    K["move"](idx, dt, D, cen, f, tq, f, nt, nr, nr1, False, True, Dr1, Dr2,
              2 * Dr1, 2 * Dr2, p_k, q_k, drift, sig)
    p_r = pos + D[:, None] * dt[:, None] * f + cp.sqrt(2 * D * dt)[:, None] * nt
    w = Dr2 * dt[:, None] * tq + cp.sqrt(2 * Dr2 * dt)[:, None] * nr
    q_r = sim._apply_rotation_gpu(q, w)
    p_r, q_r = sim._receptor_rotation_gpu(p_r, q_r, cen, f, tq, dt, nr1, Dr1)
    assert cp.allclose(p_k, p_r, rtol=1e-12, atol=1e-12)
    assert cp.allclose(q_k, q_r, rtol=1e-12, atol=1e-12)


def test_book_time_maps_add_dt_at_the_start_pose():
    g = np.random.default_rng(4)
    n, n_pairs = 3000, 2
    c0 = g.normal(size=(n, 3)) * g.uniform(5.0, 40.0, (n, 1))
    r0 = np.linalg.norm(c0, axis=1)
    pos = c0 + g.normal(size=(n, 3))
    dt = g.uniform(0.05, 50.0, n)
    gho = c0[:, None, :] + g.normal(size=(n, n_pairs, 3))
    rec = g.normal(size=(n_pairs, 3)) * 5.0
    cut = np.array([30.0, 35.0])
    rad_e, trans_e, ms_r = np.linspace(0, 48, 201), np.linspace(0, 48, 51), np.linspace(20, 40, 11)
    u64 = lambda *s: cp.zeros(s, dtype=cp.uint64)
    out = dict(rad=u64(200), ang=u64(36, 72), ms_out=u64(11), ms_in=u64(11),
               trans=u64(50, 50), contact=u64(n_pairs), cnt=u64(2), rad_t=u64(200),
               ang_t=u64(36, 72), con_t=u64(n_pairs))
    a = cp.asarray
    gbs.cp.RawKernel(gbs._BOOK_KERNEL_CODE, "book_kernel")(
        ((n + 255) // 256,), (256,),
        (a(np.arange(n)), a(np.linalg.norm(pos, axis=1)), a(dt), np.int32(n), a(pos),
         a(np.tile([1.0, 0, 0, 0], (n, 1))), cp.zeros(n, dtype=cp.int32), np.int64(0),
         cp.zeros(0, dtype=cp.int64), np.int32(0), a(gho), a(rec), a(cut), np.int32(n_pairs),
         a(rad_e), np.int32(201), a(trans_e), np.int32(51), a(ms_r), np.int32(11),
         np.int32(36), np.int32(72), np.float64(gbs._ANGLE_EDGE), cp.zeros(n, dtype=cp.int64),
         cp.full(n, 1e30), cp.zeros(n, dtype=cp.int64), cp.zeros((n, 3)), cp.zeros((n, 4)),
         cp.zeros(n), a(r0), out["rad"], out["ang"], out["ms_out"], out["ms_in"],
         out["trans"], out["contact"], cp.empty((n, 4)), out["cnt"], a(r0), a(c0),
         np.float64(gbs._TICKS), out["rad_t"], out["ang_t"], out["con_t"]),
    )
    w = np.rint(dt * gbs._TICKS).astype(np.uint64)
    rad_t = np.zeros(200, dtype=np.uint64)
    np.add.at(rad_t, np.clip(np.searchsorted(rad_e, r0) - 1, 0, 199), w)
    assert np.array_equal(cp.asnumpy(out["rad_t"]), rad_t)
    th = np.clip((np.arccos(np.clip(c0[:, 2] / r0, -1, 1)) / np.pi * 36).astype(int), 0, 35)
    ph = np.clip(((np.arctan2(c0[:, 1], c0[:, 0]) + np.pi) / (2 * np.pi) * 72).astype(int), 0, 71)
    ang_t = np.zeros((36, 72), dtype=np.uint64)
    np.add.at(ang_t, (th, ph), w)
    assert np.array_equal(cp.asnumpy(out["ang_t"]), ang_t)
    inside = np.linalg.norm(gho - rec[None], axis=2) < cut
    assert np.array_equal(cp.asnumpy(out["con_t"]), (inside * w[:, None]).sum(axis=0))
    assert np.array_equal(cp.asnumpy(out["contact"]), inside.sum(axis=0))
