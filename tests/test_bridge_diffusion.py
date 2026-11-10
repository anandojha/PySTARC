"""
The diffusion coefficient of the Brownian bridge in the GPU step.

The bridge gives each reaction pair the chance exp(-x0 x1 / (D dt)) of having
touched its cutoff between two steps. D must be the diffusion coefficient the
step moved the pair with. With hydrodynamic interactions the step uses D_par at
the separation at the start of the step and not the far field D_trans. A bridge
with D_trans counts crossings the move does not make and raises P_rxn.

The two rotations of the step move a pair distance only through the part of
each lever arm across the pair. The rotation part of D is
D_r1 |g_rec × s|² + D_r2 |R g_lig × s|² at the start of the step. The lever arm
form D_r L² is its upper bound. A one step Monte Carlo of the coded update
measures the variance of the pair distance and checks both forms. With zero
lever arms the bridge is unchanged bit for bit.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator
from pystarc.transforms.quaternion import random_quaternion

D_TRANS = 0.25
# Rotation part of four trajectories and three pairs.
D_ROT = np.array(
    [[0.0, 0.03, 0.11], [0.0, 0.01, 0.02], [0.0, 0.0, 0.07], [0.0, 0.05, 0.0]]
)
D_PAR = np.array([0.06, 0.11, 0.2, 0.25])  # D_par of four trajectories
DT = np.array([0.2, 0.05, 0.3, 1.0])


def _sim(use_hi: bool) -> SimpleNamespace:
    return SimpleNamespace(_use_hi=use_hi)


def _bridge_D_dt(sim: SimpleNamespace, D_rot=D_ROT) -> np.ndarray:
    return GPUBatchSimulator._bridge_D_dt(sim, D_PAR, DT, D_TRANS, D_rot)


class TestBridgeDiffusion:
    def test_hi_takes_the_step_D_of_each_trajectory(self):
        out = _bridge_D_dt(_sim(use_hi=True))
        assert out.shape == (4, 3)
        for i in range(4):
            for k in range(3):
                assert out[i, k] == (D_PAR[i] + D_ROT[i, k]) * DT[i]

    def test_hi_never_takes_the_far_field_D(self):
        out = _bridge_D_dt(_sim(use_hi=True))
        far = (D_TRANS + D_ROT) * DT[:, None]
        assert np.all(out[:3] < far[:3])
        assert np.array_equal(out[3], far[3])  # D_par equals D_trans far away

    def test_without_hi_takes_D_trans(self):
        out = _bridge_D_dt(_sim(use_hi=False))
        assert np.array_equal(out, (D_TRANS + D_ROT) * DT[:, None])

    def test_without_pair_atoms(self):
        out_hi = _bridge_D_dt(_sim(use_hi=True), None)
        assert np.array_equal(out_hi, D_PAR[:, None] * DT[:, None])
        out_free = _bridge_D_dt(_sim(use_hi=False), None)
        assert np.array_equal(out_free, D_TRANS * DT[:, None])

    def test_zero_rotation_part_is_unchanged_bit_for_bit(self):
        """Zero lever arms give the bridge D of the lever arm form exactly."""
        zero = np.zeros((4, 3))
        before_hi = (D_PAR[:, None] + np.zeros(3)[None, :]) * DT[:, None]
        assert np.array_equal(_bridge_D_dt(_sim(use_hi=True), zero), before_hi)
        before_free = (D_TRANS + np.zeros(3))[None, :] * DT[:, None]
        assert np.array_equal(_bridge_D_dt(_sim(use_hi=False), zero), before_free)


@pytest.fixture
def gbs_numpy(monkeypatch):
    """The GPU simulator module with NumPy standing in for CuPy."""
    import pystarc.simulation.gpu_batch_simulator as gbs

    monkeypatch.setattr(gbs, "cp", np, raising=False)
    return gbs


def _rot_sim(gbs, rec_gho: np.ndarray):
    sim = object.__new__(gbs.GPUBatchSimulator)
    sim._rec_gho_pos = np.asarray(rec_gho, dtype=float)
    return sim


def _pose(theta_deg: float, L_rec: float, L_lig: float, r0: float):
    """Receptor pair atom, ligand pair atom and ligand centroid.

    The pair runs along +x with length r0. Both lever arms stand at theta_deg
    to the pair. The receptor lever arm tilts towards +y and the ligand lever
    arm towards +z.
    """
    th = math.radians(theta_deg)
    g_rec = L_rec * np.array([math.cos(th), math.sin(th), 0.0])
    lever = L_lig * np.array([-math.cos(th), 0.0, math.sin(th)])
    lig = g_rec + np.array([r0, 0.0, 0.0])
    return g_rec, lig, lig - lever


def _one_step(gbs, theta_deg, D_t, D_r1, D_r2, L_rec, L_lig, r0, dt, n, seed):
    """Variance rate of a pair distance over one coded step and the bridge D.

    n copies of one start pose take one step of the coded update with zero
    force and torque. The translation is the line of the step loop. The two
    turns are the methods of the step. Returns the measured variance rate, its
    standard error and the bridge D of the start pose.
    """
    rng = np.random.default_rng(seed)
    g_rec, lig0, cen0 = _pose(theta_deg, L_rec, L_lig, r0)
    sim = _rot_sim(gbs, g_rec[None, :])
    q0 = random_quaternion(rng).to_array()
    R0 = sim._quats_to_rotmats(q0[None, :])[0]
    g_lig = R0.T @ (lig0 - cen0)  # ligand pair atom in the body frame
    lig_start = R0 @ g_lig + cen0
    d0 = float(np.linalg.norm(lig_start - g_rec))
    pos = np.tile(cen0, (n, 1))
    q = np.tile(q0, (n, 1))
    dt_arr = np.full(n, dt)
    zero = np.zeros((n, 3))
    pos = pos + np.sqrt(2.0 * D_t * dt_arr)[:, None] * rng.standard_normal((n, 3))
    omega = np.sqrt(2.0 * D_r2 * dt_arr)[:, None] * rng.standard_normal((n, 3))
    q = sim._apply_rotation_gpu(q, omega)
    noise_r1 = rng.standard_normal((n, 3))
    pos, q = sim._receptor_rotation_gpu(
        pos, q, np.tile(cen0, (n, 1)), zero, zero, dt_arr, noise_r1, D_r1
    )
    lig1 = np.einsum("nij,j->ni", sim._quats_to_rotmats(q), g_lig) + pos
    dev = np.linalg.norm(lig1 - g_rec, axis=1) - d0
    dev = dev - dev.mean()
    m2 = float(np.mean(dev**2))
    m4 = float(np.mean(dev**4))
    meas = m2 / (2.0 * dt)
    se = math.sqrt((m4 - m2 * m2) / n) / (2.0 * dt)
    D_rot = sim._bridge_D_rot(
        lig_start[None, None, :], cen0[None, :], np.array([[d0]]), D_r1, D_r2
    )
    return meas, se, D_t + float(D_rot[0, 0])


class TestBridgeRotation:
    D_R1, D_R2, L_REC, L_LIG, R0 = 2.5e-5, 5.0e-5, 18.0, 13.0, 5.0
    FULL = D_R1 * L_REC**2 + D_R2 * L_LIG**2

    def _D_rot(self, gbs, theta_deg: float) -> float:
        g_rec, lig, cen = _pose(theta_deg, self.L_REC, self.L_LIG, self.R0)
        d = np.array([[np.linalg.norm(lig - g_rec)]])
        sim = _rot_sim(gbs, g_rec[None, :])
        out = sim._bridge_D_rot(
            lig[None, None, :], cen[None, :], d, self.D_R1, self.D_R2
        )
        return float(out[0, 0])

    def test_lever_arms_along_the_pair_add_nothing(self, gbs_numpy):
        assert self._D_rot(gbs_numpy, 0.0) == 0.0

    def test_lever_arms_across_the_pair_give_D_r_L2(self, gbs_numpy):
        assert self._D_rot(gbs_numpy, 90.0) == pytest.approx(self.FULL, rel=1e-12)

    def test_sin_squared_of_the_angle_to_the_pair(self, gbs_numpy):
        for theta in (30.0, 60.0):
            expected = self.FULL * math.sin(math.radians(theta)) ** 2
            assert self._D_rot(gbs_numpy, theta) == pytest.approx(expected, rel=1e-12)

    def test_random_poses_match_the_projection(self, gbs_numpy):
        """|g × s|² equals |g|² - (g · s)² and never exceeds |g|²."""
        rng = np.random.default_rng(21)
        n, k, D_r1, D_r2 = 50, 4, 1e-4, 3e-3
        rec = rng.standard_normal((k, 3)) * 10.0
        cen = rng.standard_normal((n, 3)) * 20.0
        lig = cen[:, None, :] + rng.standard_normal((n, k, 3)) * 3.0
        d = np.linalg.norm(lig - rec[None], axis=2)
        out = _rot_sim(gbs_numpy, rec)._bridge_D_rot(lig, cen, d, D_r1, D_r2)
        s = (lig - rec[None]) / d[:, :, None]
        lev = lig - cen[:, None, :]
        rec_n = np.broadcast_to(rec, lig.shape)
        ref = D_r1 * (
            np.sum(rec_n**2, axis=2) - np.sum(rec_n * s, axis=2) ** 2
        ) + D_r2 * (np.sum(lev**2, axis=2) - np.sum(lev * s, axis=2) ** 2)
        assert out.shape == (n, k)
        assert np.allclose(out, ref, rtol=1e-9, atol=0.0)
        bound = D_r1 * np.sum(rec_n**2, axis=2) + D_r2 * np.sum(lev**2, axis=2)
        assert np.all(out <= bound * (1.0 + 1e-12))

    def test_zero_lever_arms_give_exact_zero(self, gbs_numpy):
        """Both pair atoms sit at the rotation centers as for two spheres."""
        rng = np.random.default_rng(22)
        n = 40
        q = np.array([random_quaternion(rng).to_array() for _ in range(n)])
        sim = _rot_sim(gbs_numpy, np.zeros((1, 3)))
        R = sim._quats_to_rotmats(q)
        cen = rng.standard_normal((n, 3)) * 10.0
        # The ligand pair atoms as the step loop forms them.
        lig = np.einsum("nij,kj->nki", R, np.zeros((1, 3))) + cen[:, None, :]
        d = np.linalg.norm(lig - sim._rec_gho_pos[None], axis=2)
        out = sim._bridge_D_rot(lig, cen, d, 0.023, 0.023)
        assert np.array_equal(out, np.zeros((n, 1)))

    def test_one_step_monte_carlo_of_the_coded_update(self, gbs_numpy):
        """The measured rate matches the bridge D at 0, 30, 60 and 90 degrees.

        The lever arm form is above the measured rate by more than 4 standard
        errors at 0, 30 and 60 degrees and matches it at 90 degrees.
        """
        D_t, dt, n = 0.03, 0.02, 200_000
        old = D_t + self.FULL
        for i, theta in enumerate((0.0, 30.0, 60.0, 90.0)):
            meas, se, new = _one_step(
                gbs_numpy,
                theta,
                D_t,
                self.D_R1,
                self.D_R2,
                self.L_REC,
                self.L_LIG,
                self.R0,
                dt,
                n,
                seed=31 + i,
            )
            assert abs(meas - new) < 4.0 * se
            if theta < 90.0:
                assert old - meas > 4.0 * se
            else:
                assert abs(old - meas) < 4.0 * se
