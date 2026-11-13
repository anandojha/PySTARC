"""
Closest accessible approach of the ligand center to a reference point under the
soft excluded-volume wall of pystarc.forces.soft_repulsion.

A rigid pose of the ligand is accessible when its wall energy against the fixed
receptor is at most e_max (2 kT). The closest approach is the smallest distance
of the ligand centroid from the reference point among the accessible poses that
three searches find. A coarse scan draws n_coarse poses with the center uniform
in a ball of radius ball around the reference point and a uniform orientation.
A dense scan draws n_dense poses in a ball 1.5 A wider than the coarse result.
When the scans find an approach below descent_below and the ligand has a bound
pose, n_starts poses near the bound pose (center displaced by a normal of
0.5 A, orientation turned by up to 20 degrees) follow the wall force and torque
in steps of 0.05 A and 0.02 rad until their energy is at most e_max. The
defaults and the seed reproduce the closest approaches of the criterion table of
2026-11-08. The scans run on the GPU when CuPy is present.
"""

from __future__ import annotations

import math

import numpy as np

from pystarc.forces.soft_repulsion import SoftRepulsion

try:
    import cupy as cp
except Exception:  # pragma: no cover - CuPy is optional
    cp = None

SEED = 20261108
BATCH = 20_000


def _rotations(q, xp):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def scan(wall, body, ref, n, ball, rng, xp, e_max=2.0):
    """Smallest center distance from ref among the accessible poses of n random
    poses in a ball of radius ball, and the number of accessible poses."""
    best, n_ok = math.inf, 0
    for s0 in range(0, n, BATCH):
        b = min(BATCH, n - s0)
        v = rng.normal(size=(b, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        r = ball * rng.random(b) ** (1.0 / 3.0)
        c = ref + v * r[:, None]
        q = rng.normal(size=(b, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        L = np.einsum("nij,aj->nai", _rotations(q, np), body) + c[:, None, :]
        _, _, e, _ = wall(xp.asarray(L), xp.asarray(c))
        ok = (e.get() if xp is not np else e) <= e_max
        n_ok += int(ok.sum())
        if ok.any():
            best = min(best, float(r[ok].min()))
    return (None if not math.isfinite(best) else best), n_ok


def _qmul(a, b, xp):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return xp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _axis_angle(v, xp):
    ang = xp.linalg.norm(v, axis=1)
    ax = v / xp.maximum(ang, 1e-12)[:, None]
    return xp.concatenate([xp.cos(ang / 2)[:, None], ax * xp.sin(ang / 2)[:, None]], axis=1)


def descend(wall, bound, ref, n_starts, n_steps, xp, e_max=2.0, seed=SEED):
    """Descent from the bound pose bound (N_lig, 3). Returns the smallest center
    distance from ref at the first accessible pose of each start, and the number
    of starts that reached e_max."""
    c0 = bound.mean(axis=0)
    body = xp.asarray(bound - c0)
    rng = np.random.default_rng(seed)
    c = xp.asarray(c0 + rng.normal(scale=0.5, size=(n_starts, 3)))
    c[0] = xp.asarray(c0)
    ax = rng.normal(size=(n_starts, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ang = rng.uniform(0.0, np.radians(20.0), n_starts)
    ang[0] = 0.0
    q = _axis_angle(xp.asarray(ax * ang[:, None]), xp)
    ref_x = xp.asarray(ref)
    found = xp.full(n_starts, xp.nan)
    for _ in range(n_steps):
        L = xp.einsum("nij,aj->nai", _rotations(q, xp), body) + c[:, None, :]
        F, T, E, _ = wall(L, c)
        new = (E <= e_max) & xp.isnan(found)
        found = xp.where(new, xp.linalg.norm(c - ref_x, axis=1), found)
        live = xp.isnan(found)
        if not bool(live.any()):
            break
        fn = xp.linalg.norm(F, axis=1)
        tn = xp.linalg.norm(T, axis=1)
        c = xp.where(live[:, None], c + 0.05 * F / xp.maximum(fn, 1e-12)[:, None], c)
        q = xp.where(live[:, None], _qmul(_axis_angle(0.02 * T / xp.maximum(tn, 1e-12)[:, None], xp), q, xp), q)
        q = q / xp.linalg.norm(q, axis=1)[:, None]
    f = found.get() if xp is not np else found
    ok = ~np.isnan(f)
    return (float(f[ok].min()) if ok.any() else None), int(ok.sum())


def closest_approach(rec_xyz, rec_radii, lig_xyz, lig_radii, ref, bound=True, ball=16.0, n_coarse=2_000_000,
                     n_dense=8_000_000, widen=1.5, descent_below=8.0, n_starts=2048, n_steps=4000,
                     e_max=2.0, seed=SEED):
    """Closest accessible approach of the ligand centroid to ref (in angstroms).

    The atoms are the real atoms without ghost atoms. lig_xyz is the bound pose
    when bound is True; otherwise only its shape enters and the descent is
    skipped. Returns the coarse, dense and descent results, the closest approach
    rounded to 0.01 A, and the settings."""
    xp = cp if cp is not None else np
    rec_xyz, lig_xyz = np.asarray(rec_xyz, float), np.asarray(lig_xyz, float)
    ref = np.asarray(ref, float)
    wall = SoftRepulsion(rec_xyz, np.asarray(rec_radii, float), np.asarray(lig_radii, float), xp=xp)
    body = lig_xyz - lig_xyz.mean(axis=0)
    rng = np.random.default_rng(seed)
    coarse, n0 = scan(wall, body, ref, n_coarse, ball, rng, xp, e_max)
    dense, n1 = (None, 0) if coarse is None else scan(wall, body, ref, n_dense, coarse + widen, rng, xp, e_max)
    scans = [x for x in (coarse, dense) if x is not None]
    descent, n2 = None, 0
    if bound and scans and min(scans) < descent_below:
        descent, n2 = descend(wall, lig_xyz, ref, n_starts, n_steps, xp, e_max, seed)
    found = [x for x in (coarse, dense, descent) if x is not None]
    return dict(closest=None if not found else round(min(found), 2), coarse=coarse, dense=dense, descent=descent,
                accessible_coarse=n0, accessible_dense=n1, descent_reached=n2, ball=ball, n_coarse=n_coarse,
                n_dense=n_dense, widen=widen, descent_below=descent_below, n_starts=n_starts, n_steps=n_steps,
                e_max=e_max, seed=seed, device="gpu" if xp is not np else "cpu")
