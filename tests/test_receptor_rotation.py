"""
Receptor rotation in the rigid-body BD step.

The simulation frame is fixed to the receptor, so the receptor's own rotational
diffusion must appear as a rigid rotation of the relative configuration about
the frame origin: the ligand's separation vector and its orientation turn
together. These tests pin down that construction on the CPU step, on the
batched GPU step (run with NumPy standing in for CuPy), on the GPU return snap
and on the CPU outer propagator, and check the time-step bound that covers the
new tangential motion.
"""

from __future__ import annotations

import math
import os
import types
import warnings

import numpy as np
import pytest

from pystarc.motion.adaptive_time_step import max_time_step
from pystarc.motion.do_bd_step import (
    apply_frame_rotation,
    bd_step_wiener,
    bd_step_wiener_two_body,
    receptor_rotation_step,
    receptor_rotation_vector,
)
from pystarc.simulation.diffusional_rotation import (
    quat_multiply,
    quat_of_rotvec,
    quat_rotate_vector,
)
from pystarc.transforms.quaternion import Quaternion, random_quaternion


def _rel_angle(q_new: Quaternion, q_old: Quaternion) -> float:
    """Rotation angle taking q_old to q_new."""
    dq = (q_new * q_old.conjugate()).normalized()
    return 2.0 * math.acos(min(1.0, abs(dq.w)))


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


class TestReceptorRotationStep:
    def test_two_body_reduces_to_one_body_when_receptor_fixed(self):
        """With D_rot1 = 0 the two-body step is bd_step_wiener exactly."""
        rng = np.random.default_rng(1)
        pos = np.array([12.0, -3.0, 5.0])
        ori = random_quaternion(rng)
        F = rng.standard_normal(3)
        T = rng.standard_normal(3)
        dt = 0.3
        dW_t = math.sqrt(dt) * rng.standard_normal(3)
        dW_r = math.sqrt(dt) * rng.standard_normal(3)
        dW_r1 = math.sqrt(dt) * rng.standard_normal(3)
        p1, o1 = bd_step_wiener(pos, ori, F, T, 0.05, 0.002, dt, dW_t, dW_r)
        p2, o2 = bd_step_wiener_two_body(
            pos, ori, F, T, 0.05, 0.0, 0.002, dt, dW_t, dW_r1, dW_r
        )
        assert np.array_equal(p1, p2)
        assert np.array_equal(o1.to_array(), o2.to_array())

    def test_separation_is_preserved(self):
        """The receptor rotation turns the separation vector but keeps its length."""
        rng = np.random.default_rng(2)
        for _ in range(50):
            pos = rng.standard_normal(3) * 20.0
            ori = random_quaternion(rng)
            F = rng.standard_normal(3)
            T = rng.standard_normal(3)
            dW = rng.standard_normal(3)
            p, o = receptor_rotation_step(pos, ori, F, T, 0.01, 0.5, dW)
            assert abs(np.linalg.norm(p) - np.linalg.norm(pos)) < 1e-10
            assert abs(o.norm() - 1.0) < 1e-12

    def test_rotation_acts_on_position_and_orientation_together(self):
        """The same quaternion rotates the separation and the orientation."""
        rng = np.random.default_rng(3)
        pos = np.array([10.0, 0.0, 0.0])
        ori = random_quaternion(rng)
        dW = rng.standard_normal(3)
        p, o = receptor_rotation_step(pos, ori, np.zeros(3), np.zeros(3), 0.01, 0.5, dW)
        dq = (o * ori.conjugate()).normalized()
        assert np.allclose(dq.rotate_vector(pos), p)

    def test_torque_drift_has_newton_third_law_sign(self):
        """A tangential force on the ligand turns the receptor the other way.

        Receptor at the origin, ligand at +x, force on the ligand along +y. The
        reaction force on the receptor makes it rotate about -z; in the
        receptor frame the ligand therefore moves toward +y, the direction it
        is being pushed, through the angle D_rot1 dt |r x F|.
        """
        pos = np.array([10.0, 0.0, 0.0])
        F = np.array([0.0, 1.0, 0.0])
        D_r1, dt = 0.01, 1.0
        p, _ = receptor_rotation_step(
            pos, Quaternion.identity(), F, np.zeros(3), D_r1, dt, np.zeros(3)
        )
        angle = D_r1 * dt * 10.0  # |r x F| = 10
        assert np.allclose(p, [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0])

    def test_ligand_torque_also_turns_receptor(self):
        """A torque on the ligand alone drives the receptor by -tau (Newton III)."""
        pos = np.array([0.0, 0.0, 10.0])
        T = np.array([0.0, 0.0, 1.0])  # about z, so pos (along z) is unchanged
        D_r1, dt = 0.02, 1.0
        p, o = receptor_rotation_step(
            pos, Quaternion.identity(), np.zeros(3), T, D_r1, dt, np.zeros(3)
        )
        assert np.allclose(p, pos)
        # Receptor turns by -D_r1 dt T; the relative orientation by +D_r1 dt T.
        dq = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), D_r1 * dt)
        assert np.allclose(o.to_array(), dq.to_array())


class TestTwoBodyPositionPath:
    """The full two-body step moves the position with D_rot1 > 0."""

    def test_force_driven_orbit_through_two_body_step(self):
        """Same Newton-III check as above, through bd_step_wiener_two_body."""
        pos = np.array([10.0, 0.0, 0.0])
        F = np.array([0.0, 1.0, 0.0])
        D_r1, dt = 0.01, 1.0
        z = np.zeros(3)
        p, o = bd_step_wiener_two_body(
            pos, Quaternion.identity(), F, z, 0.0, D_r1, 0.0, dt, z, z, z
        )
        angle = D_r1 * dt * 10.0
        assert np.allclose(p, [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0])
        expected = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
        assert np.allclose(o.to_array(), expected.to_array())

    def test_rotation_applies_to_post_translation_position(self):
        """The receptor rotation turns the translated position, lever arm pre-step."""
        pos = np.array([10.0, 0.0, 0.0])
        D_t, dt = 1.0, 1.0
        # Translation along +y and a receptor turn about +z do not commute, so
        # rotating the pre-step position instead of the translated one fails.
        dW_t = np.array([0.0, 2.0, 0.0])
        dW_r1 = np.array([0.0, 0.0, 0.3])
        z = np.zeros(3)
        p, _ = bd_step_wiener_two_body(
            pos, Quaternion.identity(), z, z, D_t, 0.2, 0.0, dt, dW_t, dW_r1, z
        )
        translated = pos + math.sqrt(2.0 * D_t) * dW_t
        omega1 = math.sqrt(2.0 * 0.2) * dW_r1
        rot = quat_of_rotvec(-omega1)
        assert np.allclose(p, quat_rotate_vector(rot, translated))

    def test_tangential_msd_is_4_Dr1_t(self):
        """With only receptor rotation the centroid's angular MSD is 4 D_rot1 t."""
        rng = np.random.default_rng(4)
        D_r1, dt, n = 0.5, 0.01, 20000
        pos = np.array([0.0, 0.0, 25.0])
        z = np.zeros(3)
        s2 = 0.0
        for _ in range(n):
            dW_r1 = math.sqrt(dt) * rng.standard_normal(3)
            p, _ = bd_step_wiener_two_body(
                pos, Quaternion.identity(), z, z, 0.0, D_r1, 0.0, dt, z, dW_r1, z
            )
            assert abs(np.linalg.norm(p) - 25.0) < 1e-10
            cos_t = float(p @ pos) / (np.linalg.norm(p) * 25.0)
            s2 += math.acos(max(-1.0, min(1.0, cos_t))) ** 2
        msd = s2 / n
        expected = 4.0 * D_r1 * dt
        assert abs(msd - expected) < 0.05 * expected

    def test_relative_orientation_variance_is_2_D1_plus_D2_dt(self):
        """The relative orientation diffuses with D_rot1 + D_rot2."""
        rng = np.random.default_rng(5)
        D1, D2, dt, n = 0.3, 0.7, 0.01, 20000
        pos = np.array([0.0, 0.0, 25.0])
        z = np.zeros(3)
        s2 = 0.0
        for _ in range(n):
            ori = random_quaternion(rng)
            dW_r1 = math.sqrt(dt) * rng.standard_normal(3)
            dW_r2 = math.sqrt(dt) * rng.standard_normal(3)
            _, o = bd_step_wiener_two_body(
                pos, ori, z, z, 0.0, D1, D2, dt, z, dW_r1, dW_r2
            )
            s2 += _rel_angle(o, ori) ** 2
        msd = s2 / n
        expected = 3.0 * 2.0 * (D1 + D2) * dt
        assert abs(msd - expected) < 0.05 * expected


class TestQuatRotateVector:
    def test_matches_quaternion_class(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            q = random_quaternion(rng)
            v = rng.standard_normal(3)
            assert np.allclose(quat_rotate_vector(q.to_array(), v), q.rotate_vector(v))

    def test_rotvec_matches_rodrigues(self):
        omega = np.array([0.1, -0.2, 0.3])
        q = quat_of_rotvec(omega)
        v = np.array([1.0, 2.0, 3.0])
        angle = np.linalg.norm(omega)
        k = omega / angle
        rodrigues = (
            v * math.cos(angle)
            + np.cross(k, v) * math.sin(angle)
            + k * (k @ v) * (1.0 - math.cos(angle))
        )
        assert np.allclose(quat_rotate_vector(q, v), rodrigues)


@pytest.fixture
def gbs_numpy(monkeypatch):
    """The GPU simulator module with NumPy standing in for CuPy."""
    import pystarc.simulation.gpu_batch_simulator as gbs

    monkeypatch.setattr(gbs, "cp", np, raising=False)
    return gbs


class TestGPUReceptorRotation:
    def test_rotate_vectors_matches_quaternion_rotation(self, gbs_numpy):
        rng = np.random.default_rng(7)
        omega = rng.standard_normal((64, 3)) * 0.3
        omega[0] = 0.0  # the tiny-angle branch
        v = rng.standard_normal((64, 3))
        out = gbs_numpy.GPUBatchSimulator._rotate_vectors_gpu(None, v, omega)
        for i in range(64):
            assert np.allclose(out[i], quat_rotate_vector(quat_of_rotvec(omega[i]), v[i]))

    def test_rotate_vectors_consistent_with_apply_rotation(self, gbs_numpy):
        """Rotating a vector and rotating a quaternion by the same omega agree."""
        rng = np.random.default_rng(8)
        omega = rng.standard_normal((16, 3)) * 0.5
        v = rng.standard_normal((16, 3))
        ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (16, 1))
        sim = gbs_numpy.GPUBatchSimulator
        q = sim._apply_rotation_gpu(None, ident, omega)
        out = sim._rotate_vectors_gpu(None, v, omega)
        for i in range(16):
            assert np.allclose(out[i], quat_rotate_vector(q[i], v[i]))

    def test_batched_step_matches_cpu_step(self, gbs_numpy):
        """The GPU receptor block equals the CPU functions row by row."""
        rng = np.random.default_rng(9)
        n = 32
        sim = object.__new__(gbs_numpy.GPUBatchSimulator)
        lever = rng.standard_normal((n, 3)) * 30.0
        pos = lever + rng.standard_normal((n, 3))  # post-translation positions
        q = np.array([random_quaternion(rng).to_array() for _ in range(n)])
        F = rng.standard_normal((n, 3))
        T = rng.standard_normal((n, 3))
        dt_arr = rng.uniform(0.1, 2.0, n)
        noise = rng.standard_normal((n, 3))
        D_r1 = 0.003
        p_gpu, q_gpu = sim._receptor_rotation_gpu(
            pos, q, lever, F, T, dt_arr, noise, D_r1
        )
        for i in range(n):
            dW = math.sqrt(dt_arr[i]) * noise[i]
            omega1 = receptor_rotation_vector(lever[i], F[i], T[i], D_r1, dt_arr[i], dW)
            p_cpu, q_cpu = apply_frame_rotation(pos[i], Quaternion(*q[i]), omega1)
            assert np.allclose(p_gpu[i], p_cpu)
            assert np.allclose(q_gpu[i], q_cpu.to_array())

    def test_batched_step_force_sign(self, gbs_numpy):
        """A +y force on a +x ligand moves it toward +y on the GPU path too."""
        sim = object.__new__(gbs_numpy.GPUBatchSimulator)
        lever = np.array([[10.0, 0.0, 0.0]])
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        F = np.array([[0.0, 1.0, 0.0]])
        p, _ = sim._receptor_rotation_gpu(
            lever.copy(), q, lever, F, np.zeros((1, 3)), np.array([1.0]),
            np.zeros((1, 3)), 0.01,
        )
        assert p[0, 1] > 0.0
        assert abs(np.linalg.norm(p[0]) - 10.0) < 1e-12

    def test_return_snap_rotates_reentry_point(self, gbs_numpy, monkeypatch):
        """The return snap turns pos and q by the receptor's dq0^-1."""
        dq0 = quat_of_rotvec(np.array([0.2, -0.4, 0.7]))
        monkeypatch.setattr(
            gbs_numpy, "diffusional_rotation", lambda rng, tau: dq0.copy()
        )
        rng = np.random.default_rng(10)
        q = np.array([random_quaternion(rng).to_array() for _ in range(5)])
        pos = rng.standard_normal((5, 3)) * 40.0
        q_new, pos_new = gbs_numpy._return_rotation_host(q, pos, 0.0, 0.5, rng)
        for i in range(5):
            assert np.allclose(pos_new[i], quat_rotate_vector(_conj(dq0), pos[i]))
            assert np.allclose(q_new[i], quat_multiply(_conj(dq0), q[i]))
            assert abs(np.linalg.norm(pos_new[i]) - np.linalg.norm(pos[i])) < 1e-9

    def test_return_snap_ligand_only_leaves_position(self, gbs_numpy, monkeypatch):
        """With tau_rec = 0 the re-entry point is not moved."""
        dq1 = quat_of_rotvec(np.array([0.5, 0.1, 0.0]))
        monkeypatch.setattr(
            gbs_numpy, "diffusional_rotation", lambda rng, tau: dq1.copy()
        )
        rng = np.random.default_rng(11)
        q = np.array([random_quaternion(rng).to_array() for _ in range(3)])
        pos = rng.standard_normal((3, 3)) * 40.0
        q_new, pos_new = gbs_numpy._return_rotation_host(q, pos, 0.5, 0.0, rng)
        assert np.array_equal(pos_new, pos)
        for i in range(3):
            assert np.allclose(q_new[i], quat_multiply(dq1, q[i]))

    def test_split_uses_mobility(self, gbs_numpy):
        from pystarc.hydrodynamics.rotne_prager import MobilityTensor

        mob = MobilityTensor(0.01, 0.002, 0.02, 0.03, 20.0, 5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert gbs_numpy._split_rotational_diffusion(mob, 0.032) == (0.002, 0.03)

    def test_split_fallback_warns(self, gbs_numpy):
        with pytest.warns(RuntimeWarning, match="D_rot1/D_rot2"):
            d1, d2 = gbs_numpy._split_rotational_diffusion(types.SimpleNamespace(), 0.04)
        assert (d1, d2) == (0.0, 0.04)


class TestOuterPropagatorReturn:
    def test_return_turns_reentry_point_with_receptor(self, monkeypatch):
        """On return the position is turned by the receptor's dq0^-1, exactly.

        Two runs from the same seed differ only in the receptor rotation the
        fake sampler returns (identity, then dq0). The fake consumes no random
        numbers, so the radial path is identical and the returned positions
        must differ by exactly dq0^-1.
        """
        from pystarc.simulation.outer_propagator import OuterPropagator, OPGroupInfo
        import pystarc.simulation.outer_propagator as opmod

        op = OuterPropagator(
            b_radius=30.0,
            max_radius=2.0,
            has_hi=False,
            kT=0.593,
            viscosity=0.000243,
            dielectric=78.0,
            vacuum_perm=1.0 / (4 * math.pi * 332.0636),
            debye_len=7.86,
            g0=OPGroupInfo(q=0.0, Dtrans=0.01, Drot=0.05),
            g1=OPGroupInfo(q=0.0, Dtrans=0.01, Drot=0.0),
        )
        dq0 = quat_of_rotvec(np.array([0.0, 0.3, 0.7]))
        ident = np.array([1.0, 0.0, 0.0, 0.0])
        pos0 = np.array([33.0, 0.0, 0.0])
        ori0 = random_quaternion(np.random.default_rng(12)).to_array()

        def run(rec_rot):
            monkeypatch.setattr(
                opmod,
                "diffusional_rotation",
                lambda rng, tau: rec_rot.copy() if tau > 0 else ident.copy(),
            )
            out = []
            rng = np.random.default_rng(13)
            for _ in range(40):
                out.append(op.new_state(pos0.copy(), ori0.copy(), rng))
            return out

        plain = run(ident)
        turned = run(dq0)
        n_ret = 0
        for (r1, p1, o1), (r2, p2, o2) in zip(plain, turned):
            assert r1 == r2
            if r1:
                n_ret += 1
                assert np.allclose(p2, quat_rotate_vector(_conj(dq0), p1))
                assert np.allclose(o2, quat_multiply(_conj(dq0), o1))
                assert abs(np.linalg.norm(p2) - 30.0) < 1e-6
        assert n_ret > 0


class TestTimeStepBound:
    def test_orbital_term_tightens_pair_bound(self):
        r, D_t, D_r1 = 40.0, 0.02, 1e-4
        free = max_time_step(r, D_t, 1e-9, 1e3, 1e3)
        bound = max_time_step(r, D_t, 1e-9, 1e3, 1e3, D_rot_frame=D_r1)
        assert abs(free - 0.005 * r * r / D_t) < 1e-9
        assert abs(bound - 0.005 * r * r / (D_t + D_r1 * r * r)) < 1e-9
        assert bound < free

    def test_zero_orbital_term_is_unchanged(self):
        assert max_time_step(25.0, 0.03, 0.001, 20.0, 5.0) == max_time_step(
            25.0, 0.03, 0.001, 20.0, 5.0, D_rot_frame=0.0
        )


@pytest.mark.skipif(
    not os.environ.get("PYSTARC_SLOW_TESTS"),
    reason="about 3 s per trajectory in pure Python; set PYSTARC_SLOW_TESTS=1 to run",
)
class TestIsotropicRateInvariance:
    """For a central (isotropic) criterion the receptor's rotation cannot change
    the reaction probability, since it preserves the separation."""

    @staticmethod
    def _run(D_rot1: float, n: int, seed: int) -> float:
        from pystarc.structures.molecules import Molecule, Atom
        from pystarc.hydrodynamics.rotne_prager import MobilityTensor
        from pystarc.pathways.reaction_interface import (
            ContactPair,
            ReactionCriteria,
            ReactionInterface,
            PathwaySet,
        )
        from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters

        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom(x=0, y=0, z=0, charge=0.0, radius=2.0)]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom(x=0, y=0, z=0, charge=0.0, radius=2.0)]
        mob = MobilityTensor(
            D_trans1=0.01,
            D_rot1=D_rot1,
            D_trans2=0.01,
            D_rot2=0.001,
            radius1=20.0,
            radius2=20.0,
            use_rpy=False,
        )
        pair = ContactPair(0, 0, 15.0)
        ps = PathwaySet([ReactionInterface("rxn", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(n_trajectories=n, r_start=30.0, seed=seed, verbose=False)
        sim = NAMSimulator(mol1, mol2, mob, ps, params)
        res = sim.run()
        return res.n_reacted / max(1, res.n_reacted + res.n_escaped)

    def test_reaction_probability_unchanged(self):
        n = 150
        p_fixed = self._run(0.0, n, 11)
        p_turn = self._run(0.05, n, 12)
        p = 0.5 * (p_fixed + p_turn)
        sigma = math.sqrt(2.0 * p * (1.0 - p) / n)
        assert abs(p_fixed - p_turn) < 4.0 * sigma + 1e-9
