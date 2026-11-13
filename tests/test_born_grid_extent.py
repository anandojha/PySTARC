"""The Born grids reach the 15 Å desolvation cutoff of SI Step 6, and Born
direction 2 of the GPU engine is the gradient of the SI sum. The direction 2
tests need a GPU."""

import numpy as np
import pytest

from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pipeline import run_apbs
from pystarc.pipeline.desolvation_grid import CUTOFF, DEN_FLOOR, coulomb_kbt, dielectric_factor

LAM, ALPHA, Q = 7.86, 0.7, -0.8
# A ring of six atoms, two side atoms, a hydrogen of radius 0 and the ghost.
XYZ = np.array([[1.4, 0.0, 0.0], [0.7, 1.21, 0.0], [-0.7, 1.21, 0.0], [-1.4, 0.0, 0.0],
                [-0.7, -1.21, 0.0], [0.7, -1.21, 0.0], [2.9, 0.0, 0.3], [-2.6, 0.4, -0.2],
                [3.9, 0.9, 0.0], [0.0, 0.0, 0.0]])
RAD = np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.55, 1.2, 0.0, 0.0])
NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "N7", "O8", "HO", "GHO"]
REAL = RAD > 0
# The six faces of the box and four other directions, ligand centre to receptor atom.
_u = np.vstack([np.eye(3), -np.eye(3), np.random.default_rng(17).normal(size=(4, 3))])
DIRS = _u / np.linalg.norm(_u, axis=1, keepdims=True)


def _pqr(path):
    with open(path, "w") as f:
        for i, ((x, y, z), r, name) in enumerate(zip(XYZ, RAD, NAMES)):
            f.write(f"ATOM  {i + 1:5d} {name:<4} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f" {0.1:7.4f} {r:6.4f}\n")
    return path


def _si(y):
    """phi_B' of SI Step 6 at the points y, in kBT per e^2."""
    a = RAD[REAL]
    d = np.linalg.norm(y[:, None, :] - XYZ[None, REAL], axis=2)
    r2 = np.maximum(d * d, np.maximum(a * a, DEN_FLOOR))
    t = a**3 * (1.0 + d / LAM) ** 2 * np.exp(-2.0 * d / LAM) / (r2 * r2)
    return 0.5 * coulomb_kbt() * dielectric_factor() * np.where(d < CUTOFF, t, 0.0).sum(axis=1)


@pytest.mark.parametrize("dime", [65, 129])
def test_coarse_born_grid_is_the_smallest_that_holds_the_cutoff(tmp_path, dime):
    pqr = _pqr(tmp_path / "lig.pqr")
    coarse, _ = run_apbs._compute_grid_params(pqr, 1.4, LAM, dime, 0.0, 0.0)
    wide = run_apbs._born_reach(coarse, pqr)
    sp, n = coarse["spacing"], wide["dime"][0]
    usable = lambda n: 0.5 * sp * (n - 1) - sp
    need = np.abs(XYZ[REAL] - coarse["gcent"]).max() + CUTOFF
    assert usable(n) >= need > usable(n - 2)
    assert n > dime and (n - dime) % 2 == 0 and wide["dime"] == [n] * 3
    assert wide["glen"] == [sp * (n - 1)] * 3 and wide["gcent"] == coarse["gcent"]
    assert run_apbs._born_reach(wide, pqr) == wide


@pytest.fixture(scope="module")
def born2(tmp_path_factory):
    """The direction 2 force on a receptor charge Q at the origin, for ligand
    centres cen, from the Born grids run_apbs writes for the ligand."""
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1).sum()
    except Exception:
        pytest.skip("no usable CUDA device")
    from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine

    w = tmp_path_factory.mktemp("born")
    pqr = _pqr(w / "lig.pqr")
    coarse, fine = run_apbs._compute_grid_params(pqr, 1.4, LAM, 129, 0.0, 0.0)
    coarse = run_apbs._born_reach(coarse, pqr)
    dx = run_apbs._write_desolvation_grids(pqr, "lig", w, coarse, fine, 4.0, 78.0, 298.15, LAM)
    eng = GPUBatchForceEngine([], [], alpha=ALPHA, debye_length=LAM,
                              lig_born_grids=[DXGrid.from_file(p) for p in dx],
                              rec_positions=np.zeros((1, 3)), rec_charges=np.array([Q]))

    def force(cen):
        n = len(cen)
        f, _, _ = eng(cp.asarray(XYZ[None] + cen[:, None]), cp.zeros(len(XYZ)),
                      cp.asarray(np.tile(np.eye(3), (n, 1, 1))), cp.asarray(cen))
        return cp.asnumpy(f)

    return force


def test_direction_2_is_the_gradient_of_the_si_sum_from_3_to_20_A(born2):
    cen = -(np.arange(3.0, 20.0, 0.05)[None, :, None] * DIRS[:, None, :]).reshape(-1, 3)
    F = born2(cen)
    h = 1e-5
    ref = ALPHA * Q**2 * np.stack([(_si(-cen + h * e) - _si(-cen - h * e)) / (2 * h)
                                   for e in np.eye(3)], axis=1)
    d = np.linalg.norm(cen[:, None, :] + XYZ[None, REAL], axis=2)
    smooth = np.all((np.abs(d - CUTOFF) > 1.0) & (d > RAD[REAL] + 1.0), axis=1)
    far = np.all(d > CUTOFF + 1.0, axis=1)
    near = smooth & ~far
    assert near.sum() > 1500 and far.sum() > 400
    err = np.linalg.norm(F - ref, axis=1)
    assert np.all(err[near] <= 0.02 * np.linalg.norm(ref[near], axis=1))
    assert np.all(F[far] == 0.0)


@pytest.mark.parametrize("s0", [8.0, 13.0])
def test_direction_2_work_from_22_A_is_the_si_energy(born2, s0):
    """Along the x axis, where the box of the molecule ends at 13.6 Å."""
    s = np.linspace(s0, 22.0, int(round((22.0 - s0) / 0.005)) + 1)
    f = -born2(-s[:, None] * np.eye(3)[0])[:, 0]
    work = 0.5 * ((f[1:] + f[:-1]) * np.diff(s)).sum()
    U = ALPHA * Q**2 * _si(np.array([[s0, 0.0, 0.0]]))[0]
    assert work == pytest.approx(U, rel=0.01)
