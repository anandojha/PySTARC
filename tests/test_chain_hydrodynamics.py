"""
Tests of the chain hydrodynamics.

The tests cover the vectorized rigid-body tensors, the rotation of body frame
tensors into the lab frame, the periodic rebuild of the tensors, the removal of
rigid rotation in the internal steps, the two-body treatment with a target of
given hydrodynamic radius, and the Romberg integrator of the outer propagator.
Every system is synthetic. The accept numbers are those of the specification.

Run with:  pytest tests/test_chain_hydrodynamics.py -v
"""

from __future__ import annotations
from pystarc.simulation.coffdrop_chain import (
    ChainAngle,
    ChainAtom,
    ChainAtomRef,
    ChainBond,
    ChainCommon,
    ChainState,
    LengthConstraint,
)
from pystarc.simulation import chain_simulator
from pystarc.simulation.chain_simulator import (
    ChainBDParameters,
    ChainBDSimulator,
    chain_internal_bd_step,
    chain_outer_bd_step_wiener,
    place_chain,
    rotate_tensor_to_lab,
)
from pystarc.hydrodynamics.rotne_prager import (
    MobilityTensor,
    _translation_only_mobility,
    chain_diffusion_tensors,
    chain_rigid_body_resistance,
    radial_D_gradient,
    rpy_full_mobility_matrix,
    stokes_rotational_diffusion,
    stokes_translational_diffusion,
)
from pystarc.simulation.outer_propagator import OPGroupInfo, OuterPropagator
from pystarc.simulation.nam_simulator import NAMSimulator
from pystarc.motion.do_bd_step import (
    WATER_VISCOSITY,
    ermak_mccammon_translation_tensor,
)
from pystarc.pipeline import chain_pipeline
from pystarc.pipeline.input_parser import (
    ChainConfig,
    OutputConfig,
    PySTARCConfig,
    parse,
)
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.molsystem.system_state import Fate
from pystarc.structures.molecules import Atom, Molecule
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from scipy import integrate
from scipy.spatial.transform import Rotation
from scipy.stats import ks_2samp
import numpy as np
import types
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_coil(n, rng, step=3.8):
    """A random walk of n beads with fixed step length, centered at the centroid."""
    v = rng.standard_normal((n, 3))
    v *= step / np.linalg.norm(v, axis=1)[:, None]
    pos = np.cumsum(v, axis=0)
    return pos - pos.mean(axis=0)


def _random_spd(rng, scale=1.0):
    A = rng.standard_normal((3, 3))
    return scale * (A @ A.T + 0.5 * np.eye(3))


def _free_common(n, radius=2.0):
    atoms = [
        ChainAtom(radius=radius, charge=0.0, resname="X", resid=i) for i in range(n)
    ]
    return ChainCommon(name="free", atoms=atoms)


def _bonded_common(n, radius=2.0, k_spring=10.0, constraints=False):
    atoms = [
        ChainAtom(radius=radius, charge=0.0, resname="X", resid=i) for i in range(n)
    ]
    bonds = []
    cons = []
    for i in range(n - 1):
        if constraints:
            cons.append(LengthConstraint(ChainAtomRef(i), ChainAtomRef(i + 1), 3.8))
        else:
            bonds.append(ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), 3.8, k_spring))
    return ChainCommon(name="bonded", atoms=atoms, bonds=bonds, length_constraints=cons)


def _pair_distances(pos):
    d = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(d, axis=2)


def _fit_rotation(old, new):
    """The unweighted best rotation R with R @ new ~ old, scipy's Kabsch."""
    rot, _ = Rotation.align_vectors(old, new)
    return rot


def _single_atom_target(radius=2.0):
    return Molecule(atoms=[Atom(x=0.0, y=0.0, z=0.0, radius=radius)])


def _make_sim(
    n_atoms=6,
    *,
    a_target=None,
    hi=False,
    auto=True,
    D_trans=None,
    D_rot=None,
    interval=0,
    rotate=False,
    remove=False,
    r_start=60.0,
    max_steps=200,
    seed=7,
    radius=2.0,
    equil=0,
    save_interval=0,
    use_lmz=False,
    constraints=False,
    seed_positions=3,
    target=None,
    extra_params=None,
):
    """A small bonded chain against a single atom target, no reactions."""
    rng = np.random.default_rng(seed_positions)
    common = _bonded_common(n_atoms, radius=radius, constraints=constraints)
    positions = _random_coil(n_atoms, rng)
    kwargs = dict(
        n_trajectories=1,
        max_steps=max_steps,
        r_start=r_start,
        r_escape=1.2 * r_start,
        dt=0.2,
        dt_chain=0.05,
        chain_steps_per_outer=2,
        seed=seed,
        use_lmz=use_lmz,
        use_hard_sphere=False,
        force_change_backstep=False,
        hydrodynamic_interactions=hi,
        rotate_body_tensors=rotate,
        remove_internal_rotation=remove,
        diffusion_update_interval=interval,
        n_equilibration_steps=equil,
    )
    if extra_params:
        kwargs.update(extra_params)
    params = ChainBDParameters(**kwargs)
    sim_kwargs = dict(
        target=target if target is not None else _single_atom_target(),
        chain_template=common,
        chain_init_body_positions=positions,
        params=params,
        pathway_set=PathwaySet(reactions=[]),
        outputs=OutputConfig(save_interval=save_interval),
        target_hydro_radius=a_target,
    )
    if auto:
        sim_kwargs["auto_diffusion"] = True
    else:
        sim_kwargs["D_trans"] = D_trans
        sim_kwargs["D_rot"] = D_rot
    return ChainBDSimulator(**sim_kwargs)


def _legacy_rigid_step(
    self, pos, ori, world_pos, forces, D_t, D_r, dt, dW_t, dW_r, dW_r1=None
):
    """The rigid step as it was before the hydrodynamics change."""
    return chain_outer_bd_step_wiener(
        pos, ori, world_pos, forces, self.D_trans, self.D_rot, dt, dW_t, dW_r
    )


# A frozen copy of the earlier implementation of the rigid-body tensors.


def _frozen_translation_only_mobility(positions, radii):
    M_full = rpy_full_mobility_matrix(positions, radii)
    n = len(positions)
    M_tt = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            M_tt[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = M_full[
                6 * i : 6 * i + 3, 6 * j : 6 * j + 3
            ]
    return M_tt


def _frozen_chain_diffusion_tensors(
    positions, radii, kT=1.0, viscosity=WATER_VISCOSITY
):
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = positions.shape[0]
    hc = (radii[:, None] * positions).sum(axis=0) / radii.sum()
    M_tt = _frozen_translation_only_mobility(positions, radii)
    L = np.linalg.cholesky(M_tt)

    def solve(v):
        y = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, y)

    A = np.zeros((3, 3))
    for k in range(3):
        v = np.zeros(3 * n)
        for i in range(n):
            v[3 * i + k] = 1.0
        A[:, k] = solve(v).reshape(n, 3).sum(axis=0)
    C = np.zeros((3, 3))
    for k in range(3):
        omega = np.zeros(3)
        omega[k] = 1.0
        v = np.zeros(3 * n)
        for i in range(n):
            v[3 * i : 3 * i + 3] = np.cross(omega, positions[i] - hc)
        F = solve(v).reshape(n, 3)
        T_total = np.zeros(3)
        for i in range(n):
            T_total += np.cross(positions[i] - hc, F[i])
        C[:, k] = T_total
    scale = kT / viscosity
    return scale * np.linalg.inv(A), scale * np.linalg.inv(C), hc


def _assert_close_to_largest(a, b, rtol):
    """Entrywise closeness with the tolerance relative to the largest entry."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# H1. Fast rigid-body tensors
# ---------------------------------------------------------------------------


def _chain_with_all_regimes(rng, n=40):
    """A coil with overlapping beads, an engulfed bead, and a coincident pair."""
    pos = np.cumsum(rng.standard_normal((n, 3)) * 3.0, axis=0)
    radii = rng.uniform(1.5, 6.0, n)
    # An engulfed bead, small and inside a large bead.
    radii[9] = 6.0
    radii[10] = 1.0
    pos[10] = pos[9] + np.array([0.5, 0.0, 0.0])
    # A coincident pair.
    pos[5] = pos[4]
    return pos, radii


def test_T1_vectorized_block_matches_rpy_full_mobility_matrix():
    """The vectorized tt block equals the tt blocks of rpy_full_mobility_matrix to 1e-15."""
    rng = np.random.default_rng(11)
    pos, radii = _chain_with_all_regimes(rng)
    # The set holds all three regimes.
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    ai = radii[:, None]
    aj = radii[None, :]
    off = ~np.eye(len(radii), dtype=bool)
    assert np.any((d < ai + aj) & (d > np.abs(ai - aj)) & off)
    assert np.any((d <= np.abs(ai - aj)) & (d > 1e-12) & off)
    assert np.any((d < 1e-12) & off)
    ref = _frozen_translation_only_mobility(pos, radii)
    new = _translation_only_mobility(pos, radii)
    assert new.shape == ref.shape
    assert np.max(np.abs(new - ref)) <= 1e-15


def test_T1_block_rejects_bad_radii():
    pos = np.zeros((2, 3))
    pos[1, 0] = 5.0
    with pytest.raises(ValueError, match="strictly positive"):
        _translation_only_mobility(pos, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        _translation_only_mobility(pos, np.array([1.0, np.nan]))


def test_T2_chain_diffusion_tensors_matches_frozen_implementation():
    """Uncoupled chain_diffusion_tensors matches the earlier implementation to 1e-12; coupled Schur differs cleanly."""
    rng = np.random.default_rng(12)
    pos, radii = _chain_with_all_regimes(rng)
    pos[5] += np.array([1.0, 0.0, 0.0])  # without the coincident pair
    D_t, D_r, hc = chain_diffusion_tensors(pos, radii, include_coupling=False)
    D_t0, D_r0, hc0 = _frozen_chain_diffusion_tensors(pos, radii)
    _assert_close_to_largest(D_t, D_t0, 1e-12)
    _assert_close_to_largest(D_r, D_r0, 1e-12)
    np.testing.assert_allclose(hc, hc0, rtol=1e-12)
    # Equal radii as well.
    pos2 = _random_coil(30, rng)
    r2 = np.full(30, 3.0)
    D_t, D_r, _ = chain_diffusion_tensors(pos2, r2, include_coupling=False)
    D_t0, D_r0, _ = _frozen_chain_diffusion_tensors(pos2, r2)
    _assert_close_to_largest(D_t, D_t0, 1e-12)
    _assert_close_to_largest(D_r, D_r0, 1e-12)


def test_T2_regularized_matrix_warns_only_on_request(monkeypatch):
    """A regularized factorization warns by default, stays silent when asked, and sets info."""
    import warnings
    from pystarc.hydrodynamics import rotne_prager

    original = rotne_prager._build_robust_solver

    def regularized(M):
        solver, _, _ = original(M)
        return solver, True, "stub"

    monkeypatch.setattr(rotne_prager, "_build_robust_solver", regularized)
    pos = _random_coil(5, np.random.default_rng(2))
    radii = np.full(5, 2.0)
    with pytest.warns(RuntimeWarning, match="regularization"):
        chain_rigid_body_resistance(pos, radii)
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chain_rigid_body_resistance(pos, radii, warn_regularized=False, info=info)
    assert info["regularized"] is True


def test_T3_center_argument():
    """center = hc equals the default, and for equal radii the centroid equals it too."""
    rng = np.random.default_rng(13)
    pos = _random_coil(25, rng) + np.array([3.0, -2.0, 1.0])
    radii = rng.uniform(1.5, 4.0, 25)
    D_t, D_r, hc = chain_diffusion_tensors(pos, radii)
    D_t_c, D_r_c, _ = chain_diffusion_tensors(pos, radii, center=hc)
    _assert_close_to_largest(D_t_c, D_t, 1e-12)
    _assert_close_to_largest(D_r_c, D_r, 1e-12)
    eq = np.full(25, 3.0)
    D_t, D_r, hc = chain_diffusion_tensors(pos, eq)
    D_t_c, D_r_c, _ = chain_diffusion_tensors(pos, eq, center=pos.mean(axis=0))
    np.testing.assert_allclose(hc, pos.mean(axis=0), atol=1e-12)
    _assert_close_to_largest(D_t_c, D_t, 1e-12)
    _assert_close_to_largest(D_r_c, D_r, 1e-12)


def test_T3_center_changes_rotational_tensor():
    """Schur-coupled D_trans and D_rot are origin-independent; uncoupled D_rot depends on center."""
    rng = np.random.default_rng(14)
    pos = _random_coil(20, rng)
    radii = rng.uniform(1.5, 4.0, 20)
    D_t, D_r, _ = chain_diffusion_tensors(pos, radii)
    D_t2, D_r2, _ = chain_diffusion_tensors(pos, radii, center=pos.mean(axis=0) + 5.0)
    _assert_close_to_largest(D_t2, D_t, 1e-12)
    _assert_close_to_largest(D_r2, D_r, 1e-12)
    D_tu, D_ru, _ = chain_diffusion_tensors(pos, radii, include_coupling=False)
    D_tu2, D_ru2, _ = chain_diffusion_tensors(
        pos, radii, center=pos.mean(axis=0) + 5.0, include_coupling=False
    )
    _assert_close_to_largest(D_tu2, D_tu, 1e-12)
    assert np.max(np.abs(D_ru2 - D_ru)) > 1e-3 * np.max(np.abs(D_ru))


def test_T6_rotation_equivariance_of_body_tensors():
    """Rotating the body positions by Q rotates both tensors as Q D Qᵀ."""
    rng = np.random.default_rng(15)
    pos = _random_coil(30, rng)
    radii = rng.uniform(1.5, 4.0, 30)
    Q = Rotation.random(random_state=5).as_matrix()
    D_t, D_r, _ = chain_diffusion_tensors(pos, radii, center=np.zeros(3))
    D_t2, D_r2, _ = chain_diffusion_tensors(pos @ Q.T, radii, center=np.zeros(3))
    _assert_close_to_largest(D_t2, Q @ D_t @ Q.T, 1e-12)
    _assert_close_to_largest(D_r2, Q @ D_r @ Q.T, 1e-12)
    assert np.max(np.abs(D_t2 - Q @ D_t @ Q.T)) <= 1e-12 * np.max(np.abs(D_t))
    assert np.max(np.abs(D_r2 - Q @ D_r @ Q.T)) <= 1e-12 * np.max(np.abs(D_r))


# ---------------------------------------------------------------------------
# H2. Body frame tensors in the lab frame
# ---------------------------------------------------------------------------


def test_T4_rotate_tensor_to_lab():
    """rotate_tensor_to_lab gives R D Rᵀ for 100 random orientations and SPD tensors."""
    rng = np.random.default_rng(16)
    for _ in range(100):
        q = random_quaternion(rng)
        D = _random_spd(rng, 0.01)
        D_lab = rotate_tensor_to_lab(D, q)
        R = q.to_rotation_matrix()
        assert np.max(np.abs(D_lab - R @ D @ R.T)) <= 1e-12
        # An independent check through the quaternion action on vectors.
        v = rng.standard_normal(3)
        expected = q.rotate_vector(D @ q.conjugate().rotate_vector(v))
        assert np.max(np.abs(D_lab @ v - expected)) <= 1e-12
        assert np.max(np.abs(D_lab - D_lab.T)) <= 1e-15
        np.testing.assert_allclose(
            np.linalg.eigvalsh(D_lab), np.linalg.eigvalsh(D), rtol=1e-10
        )
    D = _random_spd(rng)
    assert np.max(np.abs(rotate_tensor_to_lab(D, Quaternion.identity()) - D)) <= 1e-12
    assert rotate_tensor_to_lab(0.25, random_quaternion(rng)) == 0.25


def test_T5_covariance_of_rigid_step_is_the_rotated_tensor():
    """The lab displacement covariance of _rigid_step divided by 2 dt is R D_body Rᵀ."""
    sim = _make_sim(n_atoms=6, rotate=True)
    ori = Quaternion.from_axis_angle(np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0), 0.9)
    D_body = np.diag([0.01, 0.02, 0.04])
    D_r_body = np.diag([1e-4, 2e-4, 3e-4])
    pos = np.array([30.0, 0.0, 0.0])
    world = place_chain(sim.body_positions, pos, ori)
    forces = np.zeros_like(world)
    dt = 0.2
    rng = np.random.default_rng(17)
    n = 200_000
    X = np.empty((n, 3))
    for k in range(n):
        dW_t = math.sqrt(dt) * rng.standard_normal(3)
        dW_r = math.sqrt(dt) * rng.standard_normal(3)
        new_pos, _ = sim._rigid_step(
            pos, ori, world, forces, D_body, D_r_body, dt, dW_t, dW_r
        )
        X[k] = new_pos - pos
    cov = X.T @ X / n / (2.0 * dt)
    R = ori.to_rotation_matrix()
    expected = R @ D_body @ R.T
    assert np.max(np.abs(cov - expected)) <= 0.02 * np.max(np.abs(expected))
    # Without the rotation the same data would not match the body tensor.
    assert np.max(np.abs(expected - D_body)) > 0.1 * np.max(np.abs(expected))


def test_rigid_step_without_rotation_flag_hands_tensors_unrotated():
    """With the default flags _rigid_step is chain_outer_bd_step_wiener on the given tensors."""
    sim = _make_sim(n_atoms=6)
    rng = np.random.default_rng(18)
    ori = random_quaternion(rng)
    pos = np.array([30.0, 5.0, 0.0])
    world = place_chain(sim.body_positions, pos, ori)
    forces = rng.standard_normal(world.shape) * 0.01
    D_t = _random_spd(rng, 0.01)
    D_r = _random_spd(rng, 1e-4)
    dW_t = rng.standard_normal(3)
    dW_r = rng.standard_normal(3)
    a = sim._rigid_step(pos, ori, world, forces, D_t, D_r, 0.2, dW_t, dW_r)
    b = chain_outer_bd_step_wiener(pos, ori, world, forces, D_t, D_r, 0.2, dW_t, dW_r)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1].to_array(), b[1].to_array())
    # With the flag on, the tensors are rotated first.
    sim_r = _make_sim(n_atoms=6, rotate=True)
    c = sim_r._rigid_step(pos, ori, world, forces, D_t, D_r, 0.2, dW_t, dW_r)
    d = chain_outer_bd_step_wiener(
        pos,
        ori,
        world,
        forces,
        rotate_tensor_to_lab(D_t, ori),
        rotate_tensor_to_lab(D_r, ori),
        0.2,
        dW_t,
        dW_r,
    )
    assert np.array_equal(c[0], d[0])
    assert np.array_equal(c[1].to_array(), d[1].to_array())


# ---------------------------------------------------------------------------
# H4. Removal of rigid rotation in the internal steps
# ---------------------------------------------------------------------------


def _state(n, constraints, seed=21, radius=2.0):
    rng = np.random.default_rng(seed)
    common = _bonded_common(n, radius=radius, constraints=constraints)
    return ChainState.from_template(common, _random_coil(n, rng))


@pytest.mark.parametrize("constraints", [False, True])
def test_T7_internal_step_keeps_centroid_at_origin(constraints):
    state = _state(8, constraints)
    rng = np.random.default_rng(22)
    for _ in range(20):
        chain_internal_bd_step(state, 0.05, rng, remove_rigid_rotation=True)
        assert np.max(np.abs(state.positions.mean(axis=0))) < 1e-12


@pytest.mark.parametrize("constraints", [False, True])
def test_T8_internal_step_leaves_no_rotation(constraints):
    state = _state(8, constraints)
    rng = np.random.default_rng(23)
    for _ in range(20):
        old = state.positions.copy()
        chain_internal_bd_step(state, 0.05, rng, remove_rigid_rotation=True)
        Q = _fit_rotation(old, state.positions).as_matrix()
        assert np.linalg.norm(Q - np.eye(3)) < 1e-12


@pytest.mark.parametrize("constraints", [False, True])
def test_T9_pair_distances_equal_with_and_without_projection(constraints):
    s0 = _state(8, constraints)
    s1 = _state(8, constraints)
    r0 = np.random.default_rng(24)
    r1 = np.random.default_rng(24)
    for _ in range(10):
        chain_internal_bd_step(s0, 0.05, r0, remove_rigid_rotation=False)
        chain_internal_bd_step(s1, 0.05, r1, remove_rigid_rotation=True)
        np.testing.assert_allclose(
            _pair_distances(s1.positions), _pair_distances(s0.positions), atol=1e-12
        )
        # Keep the two copies on one trajectory by handing over the positions.
        s1.positions[:] = s0.positions


def _mu_rr_free_draining(pos, radius):
    """The 3x3 rotational block of the free draining 6x6 rigid mobility."""
    zeta = 6.0 * math.pi * WATER_VISCOSITY * radius
    Z = np.zeros((6, 6))
    for d in pos:
        X = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
        Z[:3, :3] += zeta * np.eye(3)
        Z[:3, 3:] += -zeta * X
        Z[3:, :3] += zeta * X
        Z[3:, 3:] += -zeta * X @ X
    return np.linalg.inv(Z)[3:, 3:]


def test_T10_rotational_leak_of_the_internal_step():
    """With the projection every step rotates by < 1e-12, without it the leak follows free draining theory."""
    n_beads, n_steps, dt, radius = 40, 2000, 0.05, 2.0
    for flag in (True, False):
        rng_pos = np.random.default_rng(25)
        common = _free_common(n_beads, radius)
        state = ChainState.from_template(common, _random_coil(n_beads, rng_pos))
        rng = np.random.default_rng(26)
        angles = np.empty(n_steps)
        theory = np.empty(n_steps)
        for k in range(n_steps):
            old = state.positions.copy()
            theory[k] = 2.0 * np.trace(_mu_rr_free_draining(old, radius)) * dt
            chain_internal_bd_step(
                state, dt, rng, apply_constraints=False, remove_rigid_rotation=flag
            )
            angles[k] = _fit_rotation(old, state.positions).magnitude()
        if flag:
            assert np.max(angles) < 1e-12
        else:
            ratio = np.mean(angles**2) / np.mean(theory)
            assert abs(ratio - 1.0) < 0.10


def test_T11_equilibrium_angle_distribution_is_unchanged():
    """The distribution of a harmonic angle is the same with and without the projection."""
    common = ChainCommon(
        name="angle",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(3)
        ],
        bonds=[
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 10.0),
            ChainBond(ChainAtomRef(1), ChainAtomRef(2), 3.8, 10.0),
        ],
        angles=[
            ChainAngle(ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), 2.0, 4.0)
        ],
    )
    th0 = 2.0
    start = np.array(
        [
            [3.8 * math.sin(th0 / 2), -3.8 * math.cos(th0 / 2), 0.0],
            [0.0, 0.0, 0.0],
            [-3.8 * math.sin(th0 / 2), -3.8 * math.cos(th0 / 2), 0.0],
        ]
    )
    start -= start.mean(axis=0)
    n_steps, stride = 100_000, 100
    samples = {}
    for flag in (False, True):
        state = ChainState.from_template(common, start.copy())
        rng = np.random.default_rng(27)
        out = []
        for k in range(n_steps):
            chain_internal_bd_step(state, 0.05, rng, remove_rigid_rotation=flag)
            if k % stride == 0:
                a = state.positions[0] - state.positions[1]
                c = state.positions[2] - state.positions[1]
                out.append(
                    math.acos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))
                )
        samples[flag] = np.array(out)
    n = len(samples[False])
    crit = 1.628 * math.sqrt(2.0 / n)
    assert ks_2samp(samples[False], samples[True]).statistic < crit
    # Both follow the Boltzmann weight with the sine Jacobian of the angle, which
    # moves the mean to about theta0 + cot(theta0) / k.
    expected_mean = th0 + 1.0 / math.tan(th0) / 4.0
    assert abs(np.mean(samples[True]) - expected_mean) < 0.05
    assert abs(np.mean(samples[False]) - expected_mean) < 0.05


def test_T12_small_and_collinear_chains():
    """One and two atoms, and a collinear chain, step without error and match the plain step."""
    cases = {
        1: np.zeros((1, 3)),
        2: np.array([[-1.9, 0, 0], [1.9, 0, 0]]),
        4: np.array([[-5.7, 0, 0], [-1.9, 0, 0], [1.9, 0, 0], [5.7, 0, 0]]),
    }
    for n, pos in cases.items():
        common = _bonded_common(n)
        sa = ChainState.from_template(common, pos.copy())
        sb = ChainState.from_template(common, pos.copy())
        ra = np.random.default_rng(28)
        rb = np.random.default_rng(28)
        chain_internal_bd_step(sa, 0.05, ra, remove_rigid_rotation=False)
        chain_internal_bd_step(sb, 0.05, rb, remove_rigid_rotation=True)
        assert np.max(np.abs(sa.positions - sb.positions)) < 1e-12
        assert np.max(np.abs(sb.positions.mean(axis=0))) < 1e-12


# ---------------------------------------------------------------------------
# H3. Periodic recomputation of the body tensors
# ---------------------------------------------------------------------------


def test_T13_rebuild_count_and_diagnostics(monkeypatch):
    n_steps = 23
    sim = _make_sim(
        n_atoms=10, interval=5, rotate=True, remove=True, max_steps=n_steps, seed=31
    )
    calls = []
    original = chain_simulator.chain_diffusion_tensors

    def counting(*args, **kwargs):
        calls.append(kwargs.get("center"))
        return original(*args, **kwargs)

    monkeypatch.setattr(chain_simulator, "chain_diffusion_tensors", counting)
    D_start = (sim.D_trans.copy(), sim.D_rot.copy())
    result = sim.run_one()
    assert result.fate == Fate.MAX_STEPS
    assert len(calls) == (n_steps - 1) // 5
    assert all(np.array_equal(c, np.zeros(3)) for c in calls)
    assert result.n_diffusion_rebuilds == (n_steps - 1) // 5
    assert result.n_diffusion_rebuild_failures == 0
    assert result.mean_D_trans > 0.0 and result.mean_D_rot > 0.0
    # The rebuild never writes back to the simulator.
    assert np.array_equal(sim.D_trans, D_start[0])
    assert np.array_equal(sim.D_rot, D_start[1])


def test_T13_rebuild_after_equilibration(monkeypatch):
    sim = _make_sim(n_atoms=8, interval=5, max_steps=11, equil=3, seed=32)
    calls = []
    original = chain_simulator.chain_diffusion_tensors
    monkeypatch.setattr(
        chain_simulator,
        "chain_diffusion_tensors",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    sim.run_one()
    assert len(calls) == 1 + (11 - 1) // 5


def test_T13_interval_zero_is_the_earlier_trajectory(monkeypatch):
    """With the defaults the trajectory and the random stream equal those of the earlier step."""
    runs = {}
    for label in ("new", "legacy"):
        sim = _make_sim(n_atoms=8, max_steps=40, seed=33, save_interval=1)
        if label == "legacy":
            monkeypatch.setattr(ChainBDSimulator, "_rigid_step", _legacy_rigid_step)
        calls = []
        original = chain_simulator.chain_diffusion_tensors
        monkeypatch.setattr(
            chain_simulator,
            "chain_diffusion_tensors",
            lambda *a, **k: (calls.append(1), original(*a, **k))[1],
        )
        rng = np.random.default_rng(34)
        res = sim.run_one(rng=rng)
        assert calls == []
        runs[label] = (res, rng.bit_generator.state)
        monkeypatch.undo()
    new, legacy = runs["new"], runs["legacy"]
    assert np.array_equal(new[0].path_com, legacy[0].path_com)
    assert np.array_equal(new[0].path_q, legacy[0].path_q)
    assert new[0].steps == legacy[0].steps
    assert new[0].mean_D_trans is None
    assert new[1] == legacy[1]


def test_T14_interval_needs_auto_diffusion():
    with pytest.raises(ValueError, match="auto_diffusion"):
        _make_sim(n_atoms=4, auto=False, D_trans=0.05, D_rot=0.01, interval=5)
    with pytest.raises(ValueError, match=">= 0"):
        _make_sim(n_atoms=4, interval=-1)


def test_rebuild_failure_keeps_previous_tensors(monkeypatch):
    sim = _make_sim(n_atoms=6, interval=2, max_steps=9, seed=35)

    def failing(*a, **k):
        raise np.linalg.LinAlgError("collinear")

    monkeypatch.setattr(chain_simulator, "chain_diffusion_tensors", failing)
    result = sim.run_one()
    assert result.n_diffusion_rebuilds == 0
    assert result.n_diffusion_rebuild_failures == (9 - 1) // 2
    assert result.mean_D_trans == pytest.approx(np.trace(sim.D_trans) / 3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# H5. Target diffusion and the chain-target hydrodynamics
# ---------------------------------------------------------------------------


def _q0_sim(a_target, hi, r_start=60.0):
    return _make_sim(n_atoms=6, a_target=a_target, hi=hi, r_start=r_start)


def test_T15_kb_analytic_without_charges_and_without_hi():
    """k_b = 4 pi (D_T + tr(D_c)/3) b to rtol 1e-9."""
    a_T = 14.0
    sim = _q0_sim(a_T, hi=False)
    assert sim._outer_prop is not None
    D_T = stokes_translational_diffusion(a_T)
    D_c = np.trace(sim.D_trans) / 3.0
    expected = 4.0 * math.pi * (D_T + D_c) * 60.0
    assert sim._outer_prop.k_b == pytest.approx(expected, rel=1e-9)
    # The groups carry the real diffusion coefficients.
    assert sim._outer_prop.a0 == pytest.approx(a_T, rel=1e-6)
    assert sim._a_chain == pytest.approx(
        1.0 / (6.0 * math.pi * WATER_VISCOSITY * D_c), rel=1e-14
    )


def test_T16_kb_against_the_half_split():
    """With q = 0 and no HI the new k_b over the earlier one is (D_T + D_c)/D_c."""
    a_T = 14.0
    new = _q0_sim(a_T, hi=False)
    old = _q0_sim(None, hi=False)
    D_T = stokes_translational_diffusion(a_T)
    D_c = np.trace(new.D_trans) / 3.0
    ratio = new._outer_prop.k_b / old._outer_prop.k_b
    assert ratio == pytest.approx((D_T + D_c) / D_c, rel=1e-9)


def test_T16_hi_lowers_kb():
    on = _q0_sim(14.0, hi=True)
    off = _q0_sim(14.0, hi=False)
    assert on._outer_prop.k_b < off._outer_prop.k_b


def test_T17_target_diffusion_is_the_stokes_value():
    a_T = 11.7
    sim = _q0_sim(a_T, hi=True)
    mob = MobilityTensor.from_radii(a_T, 5.0)
    assert sim._D_t_target == pytest.approx(mob.D_trans1, rel=1e-14)
    assert sim._D_r_target == pytest.approx(mob.D_rot1, rel=1e-14)
    assert sim._a_target == a_T
    assert sim._pair_mob.radius1 == a_T
    assert _q0_sim(a_T, hi=False)._pair_mob is None


def test_T18_relative_msd_of_free_two_body_diffusion():
    """MSD/(6 t) of the relative position equals D_T + tr(D_c)/3 within 3 percent."""
    sim = _q0_sim(12.0, hi=False)
    sim._D_r_target = 0.0
    D_T = sim._D_t_target
    D_body = np.diag([0.5, 1.0, 2.0]) * 0.012
    D_r_body = np.eye(3) * 0.01
    dt, block, n_blocks, n_traj = 0.2, 4, 200, 100
    rng = np.random.default_rng(41)
    msd = []
    for _ in range(n_traj):
        pos = 100.0 * np.array([1.0, 0.0, 0.0])
        ori = random_quaternion(rng)
        forces = np.zeros((6, 3))
        for b in range(n_blocks):
            start = pos.copy()
            for _ in range(block):
                world = place_chain(sim.body_positions, pos, ori)
                dW_t = math.sqrt(dt) * rng.standard_normal(3)
                dW_r = math.sqrt(dt) * rng.standard_normal(3)
                dW_r1 = math.sqrt(dt) * rng.standard_normal(3)
                pos, ori = sim._rigid_step(
                    pos, ori, world, forces, D_body, D_r_body, dt, dW_t, dW_r, dW_r1
                )
            msd.append(float(np.sum((pos - start) ** 2)))
    estimate = np.mean(msd) / (6.0 * block * dt)
    expected = D_T + np.trace(D_body) / 3.0
    assert estimate == pytest.approx(expected, rel=0.03)


def test_T19_orbital_diffusion_of_the_separation_direction():
    """The direction of the separation diffuses with 4 D_r,T t and its length stays fixed."""
    sim = _q0_sim(12.0, hi=False)
    D_rT = 2.0e-3
    sim._D_r_target = D_rT
    sim._D_t_target = 1e-12
    D_body = 1e-12 * np.eye(3)
    D_r_body = 1e-12 * np.eye(3)
    dt, block, n_blocks = 0.2, 10, 5000
    r0 = 20.0
    rng = np.random.default_rng(42)
    pos = np.array([r0, 0.0, 0.0])
    ori = Quaternion.identity()
    forces = np.zeros((6, 3))
    zero = np.zeros(3)
    drift = 0.0
    msd = []
    for _ in range(n_blocks):
        start = pos / np.linalg.norm(pos)
        for _ in range(block):
            world = place_chain(sim.body_positions, pos, ori)
            dW_r1 = math.sqrt(dt) * rng.standard_normal(3)
            pos, ori = sim._rigid_step(
                pos, ori, world, forces, D_body, D_r_body, dt, zero, zero, dW_r1
            )
            drift = max(drift, abs(np.linalg.norm(pos) - r0))
        msd.append(float(np.sum((pos / np.linalg.norm(pos) - start) ** 2)))
    assert drift < 1e-12
    t = block * dt
    assert np.mean(msd) == pytest.approx(4.0 * D_rT * t, rel=0.05)


def _two_sphere_radial_reference(a_T, a_c, D_T, D_c, r):
    """The radial two-sphere diffusion from the three Zuk regimes."""
    kT_eta = math.sqrt(D_T * 6 * math.pi * a_T * D_c * 6 * math.pi * a_c)
    a, b = a_T, a_c
    if r > a + b:
        c = (1.0 - (a * a + b * b) / (3.0 * r * r)) / (4.0 * math.pi * r)
    elif r > abs(a - b):
        am2 = (a - b) ** 2
        c = (
            16.0 * r**3 * (a + b) + 2.0 * am2**2 - 12.0 * am2 * r**2 - 6.0 * r**4
        ) / (192.0 * math.pi * a * b * r**3)
    else:
        c = 1.0 / (6.0 * math.pi * max(a, b))
    return D_T + D_c - 2.0 * kT_eta * c


def test_T20_relative_tensor_reproduces_the_two_sphere_diffusion():
    a_T = 10.0
    # An isotropic body tensor of a chain of radius 6.
    a_c = 6.0
    D_c = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * a_c)
    sim2 = _make_sim(
        n_atoms=6,
        a_target=a_T,
        hi=True,
        auto=False,
        D_trans=D_c,
        D_rot=1e-4,
    )
    assert sim2._a_chain == pytest.approx(a_c, rel=1e-12)
    ori = random_quaternion(np.random.default_rng(43))
    D_T = sim2._D_t_target
    for r in (40.0, 10.0, 2.5):  # far, overlap, engulfed
        for direction in (
            np.array([1.0, 0, 0]),
            np.array([1.0, 2.0, -1.0]) / math.sqrt(6),
        ):
            p = r * direction
            D_rel, _ = sim2._relative_translation(p, ori, D_c * np.eye(3), 0.2)
            radial = float(direction @ D_rel @ direction)
            ref = _two_sphere_radial_reference(a_T, a_c, D_T, D_c, r)
            assert radial == pytest.approx(ref, rel=1e-7)
            assert np.all(np.linalg.eigvalsh(D_rel) > 0.0)
    # The far field agrees with the outer propagator parallel diffusion.
    p = 40.0 * np.array([1.0, 0, 0])
    D_rel, _ = sim2._relative_translation(p, ori, D_c * np.eye(3), 0.2)
    assert D_rel[0, 0] == pytest.approx(sim2._outer_prop._D_parallel(40.0), rel=1e-6)


def test_T20_non_positive_relative_tensor_is_clipped_and_counted():
    sim = _make_sim(n_atoms=6, a_target=10.0, hi=False)
    sim._D_t_target = -1.0
    ori = Quaternion.identity()
    with pytest.warns(RuntimeWarning, match="positive definite"):
        D_rel, _ = sim._relative_translation(
            np.array([50.0, 0, 0]), ori, 0.01 * np.eye(3), 0.2
        )
    eig = np.linalg.eigvalsh(D_rel)
    assert np.all(eig >= chain_simulator.RELATIVE_DIFFUSION_FLOOR * (1 - 1e-9))
    assert eig[0] == pytest.approx(chain_simulator.RELATIVE_DIFFUSION_FLOOR, rel=1e-6)
    assert sim._n_relative_clipped == 1
    # One warning per trajectory, and every clipped step is counted.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim._relative_translation(np.array([50.0, 0, 0]), ori, 0.01 * np.eye(3), 0.2)
    assert sim._n_relative_clipped == 2


def test_T20_run_one_reports_the_clipped_step_count():
    sim = _make_sim(n_atoms=6, a_target=10.0, hi=False, max_steps=5)
    sim._D_t_target = -1.0
    with pytest.warns(RuntimeWarning, match="positive definite"):
        result = sim.run_one()
    assert result.n_relative_diffusion_clipped > 0
    clean = _make_sim(n_atoms=6, a_target=10.0, hi=False, max_steps=5)
    assert clean.run_one().n_relative_diffusion_clipped == 0


def test_rotate_body_tensors_start_tensors_use_the_centroid(monkeypatch):
    calls = []
    original = chain_simulator.chain_diffusion_tensors

    def recording(*args, **kwargs):
        calls.append(kwargs.get("center", "default"))
        return original(*args, **kwargs)

    monkeypatch.setattr(chain_simulator, "chain_diffusion_tensors", recording)
    _make_sim(n_atoms=6, rotate=True)
    assert len(calls) == 1 and np.array_equal(calls[0], np.zeros(3))
    calls.clear()
    _make_sim(n_atoms=6, rotate=False)
    assert calls == ["default"]


def test_T21_drift_equals_the_nam_gradient():
    """Full 3x3 Zuk RPY off-diagonal tensor M_12 is divergence-free across all 3 regimes."""
    sim = _make_sim(n_atoms=6, a_target=10.0, hi=True)
    ori = Quaternion.identity()
    eps = 1.0e-5
    for p in (
        np.array([30.0, 0.0, 0.0]),
        np.array([8.0, 6.0, 4.0]),
        np.array([0.0, 0.0, 3.0]),
        np.array([50.0, -20.0, 10.0]),
    ):
        D_rel, grad = sim._relative_translation(p, ori, sim.D_trans, 0.2)
        assert grad is None
        # Numerical divergence of D_rel(p) vanishes to finite-difference precision.
        div = np.zeros(3)
        for j in range(3):
            dp = np.zeros(3)
            dp[j] = eps
            Dp, _ = sim._relative_translation(p + dp, ori, sim.D_trans, 0.2)
            Dm, _ = sim._relative_translation(p - dp, ori, sim.D_trans, 0.2)
            div += (Dp[:, j] - Dm[:, j]) / (2.0 * eps)
        assert np.linalg.norm(div) < 1.0e-8
    # Without HI there is no drift.
    off = _make_sim(n_atoms=6, a_target=10.0, hi=False)
    assert (
        off._relative_translation(np.array([30.0, 0, 0]), ori, off.D_trans, 0.2)[1]
        is None
    )


def test_translation_tensor_grad_term():
    p = np.array([10.0, 0.0, 0.0])
    D = np.diag([0.02, 0.03, 0.04])
    dW = np.array([0.1, -0.2, 0.3])
    base = ermak_mccammon_translation_tensor(p, np.zeros(3), D, 0.2, dW)
    g = np.array([0.001, 0.0, 0.0])
    withg = ermak_mccammon_translation_tensor(p, np.zeros(3), D, 0.2, dW, grad_D=g)
    np.testing.assert_allclose(withg - base, g * 0.2, atol=1e-15)
    again = ermak_mccammon_translation_tensor(p, np.zeros(3), D, 0.2, dW, grad_D=None)
    assert np.array_equal(base, again)


def test_T22_fixed_target_draws_no_target_increment():
    """Without a target radius the random stream is that of the earlier step."""
    streams = {}
    for label, a_T in (("fixed", None), ("two_body", 12.0)):
        sim = _make_sim(n_atoms=6, a_target=a_T, max_steps=15, seed=44, save_interval=1)
        rng = np.random.default_rng(45)
        sim.run_one(rng=rng)
        streams[label] = rng.bit_generator.state
    sim = _make_sim(n_atoms=6, max_steps=15, seed=44, save_interval=1)
    ChainBDSimulator._rigid_step, saved = (
        _legacy_rigid_step,
        ChainBDSimulator._rigid_step,
    )
    try:
        rng = np.random.default_rng(45)
        sim.run_one(rng=rng)
        legacy_state = rng.bit_generator.state
    finally:
        ChainBDSimulator._rigid_step = saved
    assert streams["fixed"] == legacy_state
    assert streams["two_body"] != legacy_state


def test_two_body_run_one_completes_and_logs_tensors():
    sim = _make_sim(
        n_atoms=6,
        a_target=12.0,
        hi=True,
        interval=4,
        rotate=True,
        remove=True,
        max_steps=30,
        seed=46,
        r_start=40.0,
    )
    result = sim.run_one()
    assert result.fate in (Fate.MAX_STEPS, Fate.ESCAPED, Fate.REACTED)
    assert result.n_diffusion_rebuilds == (30 - 1) // 4 or result.steps < 30


def test_two_body_backstep_and_hard_sphere_call_sites():
    """The backstep halves and the hard-sphere redraw also take dW_r1."""
    seen = []
    original = ChainBDSimulator._rigid_step

    def spy(self, *args, **kwargs):
        seen.append(args[-1])
        return original(self, *args, **kwargs)

    sim = _make_sim(
        n_atoms=6,
        a_target=12.0,
        hi=True,
        max_steps=12,
        seed=47,
        r_start=30.0,
        extra_params=dict(force_change_backstep=True, use_hard_sphere=True),
    )
    ChainBDSimulator._rigid_step = spy
    try:
        sim.run_one()
    finally:
        ChainBDSimulator._rigid_step = original
    assert len(seen) >= 12
    assert all(s is not None and np.shape(s) == (3,) for s in seen)


def test_target_radius_validation():
    for bad in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="target_hydro_radius"):
            _make_sim(n_atoms=4, a_target=bad)
    with pytest.raises(ValueError, match="requires a target"):
        common = _bonded_common(4)
        ChainBDSimulator(
            target=None,
            chain_template=common,
            chain_init_body_positions=_random_coil(4, np.random.default_rng(1)),
            params=ChainBDParameters(),
            pathway_set=PathwaySet(reactions=[]),
            auto_diffusion=True,
            target_hydro_radius=10.0,
        )
    with pytest.warns(RuntimeWarning, match="viscosity"):
        sim_hot = _make_sim(
            n_atoms=4, a_target=10.0, extra_params=dict(temperature_kT=0.7)
        )
    sim_ref = _make_sim(n_atoms=4, a_target=10.0)
    kT_scale = sim_hot._kT_scale
    assert kT_scale == pytest.approx(0.7 / sim_ref.params.temperature_kT, rel=1e-12)
    assert sim_hot._D_t_target == pytest.approx(
        sim_ref._D_t_target * kT_scale, rel=1e-12
    )
    assert sim_hot._D_r_target == pytest.approx(
        sim_ref._D_r_target * kT_scale, rel=1e-12
    )


def test_bridge_effective_diffusion_includes_the_target(monkeypatch):
    """The bridge D_eff of a two-body run carries D_t,T and the lever terms
    D_r L_2^2 of the chain and D_r,T L_1^2 of the target (SI Step 11)."""
    from pystarc.structures.molecules import ContactPair, ReactionCriteria
    from pystarc.pathways.reaction_interface import ReactionInterface

    crit = ReactionCriteria(name="r", pairs=[ContactPair(0, 0, 3.0)], n_needed=1)
    paths = PathwaySet([ReactionInterface(name="r", criteria=crit)])
    sim = _make_sim(n_atoms=6, a_target=12.0, max_steps=3, r_start=40.0)
    sim.pathway_set = paths
    seen = []
    original = chain_simulator.check_reaction_with_bridge

    def spy(target, scratch, pset, old, new, D_eff, dt, rng, rng_bb=None):
        seen.append(D_eff)
        return original(target, scratch, pset, old, new, D_eff, dt, rng, rng_bb=rng_bb)

    monkeypatch.setattr(chain_simulator, "check_reaction_with_bridge", spy)
    sim.run_one()
    assert seen
    t = sim.target.atoms[0]
    # Legacy 3-argument _bridge_diffusion still matches isotropic formula.
    expected_iso = (
        np.trace(sim.D_trans) / 3.0
        + np.trace(sim.D_rot)
        / 3.0
        * float(sim.body_positions[0] @ sim.body_positions[0])
        + sim._D_t_target
        + sim._D_r_target * (t.x**2 + t.y**2 + t.z**2)
    )
    iso_d = sim._bridge_diffusion(sim.body_positions, sim.D_trans, sim.D_rot)
    assert iso_d[0] == pytest.approx([expected_iso], rel=1e-14)
    # Full-pose bridge diffusion includes target diffusivity, HI slowing, and internal bead motion.
    d_seen = float(np.asarray(seen[0][0]).ravel()[0])
    assert d_seen > sim._D_t_target

    pose_pos = np.array([40.0, 0.0, 0.0])
    pose_ori = Quaternion.identity()

    sim_hi = _make_sim(n_atoms=6, a_target=12.0, hi=True, max_steps=1, r_start=40.0)
    sim_hi.pathway_set = paths
    sim_hi.run_one()
    d_hi = float(
        sim_hi._bridge_diffusion(
            sim_hi.body_positions, sim_hi.D_trans, sim_hi.D_rot, pose_pos, pose_ori
        )[0][0]
    )

    sim_no_hi = _make_sim(n_atoms=6, a_target=12.0, hi=False, max_steps=1, r_start=40.0)
    sim_no_hi.pathway_set = paths
    sim_no_hi.run_one()
    d_no_hi = float(
        sim_no_hi._bridge_diffusion(
            sim_no_hi.body_positions,
            sim_no_hi.D_trans,
            sim_no_hi.D_rot,
            pose_pos,
            pose_ori,
        )[0][0]
    )

    sim_fixed = _make_sim(n_atoms=6, a_target=None, max_steps=1, r_start=40.0)
    sim_fixed.pathway_set = paths
    sim_fixed.run_one()
    d_fixed = float(
        sim_fixed._bridge_diffusion(
            sim_fixed.body_positions,
            sim_fixed.D_trans,
            sim_fixed.D_rot,
            pose_pos,
            pose_ori,
        )[0][0]
    )

    assert (d_no_hi - d_fixed) == pytest.approx(sim_no_hi._D_t_target, rel=1e-12)
    assert d_hi < d_no_hi


# ---------------------------------------------------------------------------
# H6. Romberg integrator of the outer propagator
# ---------------------------------------------------------------------------


def _quad_reference(prop, b):
    L = prop.debye_len

    def integrand(s):
        if s == 0.0:
            ainv = 1.0 / prop.a0 + 1.0 / prop.a1
            return 6.0 * math.pi / (prop.D_factor * ainv)
        r = 1.0 / s
        v = prop.V_factor * math.exp(-1.0 / (s * L)) * s
        return math.exp(v) / prop._D_parallel(r)

    result, _ = integrate.quad(
        integrand, 0.0, 1.0 / b, limit=400, epsabs=0.0, epsrel=1e-13
    )
    return 4.0 * math.pi / result


@pytest.mark.parametrize(
    "q0,q1,debye,hi",
    [
        (0.0, 0.0, 7.858, False),
        (4.0, -3.0, 7.858, True),
        (-6.0, -5.0, 10.0, True),
        (8.0, 2.0, 15.0, False),
        (3.0, -1.0, 5.0, True),
    ],
)
def test_T23_romberg_matches_quadrature(q0, q1, debye, hi):
    kT = 0.5922
    prop = OuterPropagator(
        b_radius=60.0,
        max_radius=30.0,
        has_hi=hi,
        kT=kT,
        viscosity=kT * WATER_VISCOSITY,
        dielectric=78.0,
        vacuum_perm=1.0 / (4 * math.pi * 332.0636),
        debye_len=debye,
        g0=OPGroupInfo(q=q0, Dtrans=0.0187, Drot=5e-5),
        g1=OPGroupInfo(q=q1, Dtrans=0.0165, Drot=5e-5),
    )
    for b in (60.0, prop.qradius):
        ref = _quad_reference(prop, b)
        assert prop._relative_rate(b) == pytest.approx(ref, rel=1e-9)
    assert prop.k_b == pytest.approx(_quad_reference(prop, 60.0), rel=1e-9)


# ---------------------------------------------------------------------------
# Configuration, pipeline and writer
# ---------------------------------------------------------------------------


def _pipeline_config(tmp_path, chain_overrides=None):
    kwargs = dict(
        chain_json=str(tmp_path / "chain.json"),
        reaction_pairs_json=str(tmp_path / "reaction_pairs.json"),
    )
    if chain_overrides:
        kwargs.update(chain_overrides)
    return PySTARCConfig(
        work_dir=tmp_path / "work",
        receptor_pqr=str(tmp_path / "target.pqr"),
        chain=ChainConfig(**kwargs),
        outputs=OutputConfig(),
    )


def _patch_pipeline(monkeypatch, captured, target_xyz=((0.0, 0.0, 0.0),)):
    class _Chain:
        name = "c"
        atoms = [types.SimpleNamespace(radius=2.0) for _ in range(3)]
        bonds = angles = torsions = []
        coffdrop_params = None
        pair_lookups = {}

    body = np.array([[-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0]])
    monkeypatch.setattr(
        chain_pipeline, "load_chain_from_json", lambda p: (_Chain(), body)
    )
    atoms = [Atom(x=x, y=y, z=z, radius=2.0) for (x, y, z) in target_xyz]
    monkeypatch.setattr(chain_pipeline, "parse_pqr", lambda p: Molecule(atoms=atoms))
    monkeypatch.setattr(
        chain_pipeline,
        "_load_reaction_pairs_json",
        lambda p: [(0, 0, 4.0), (0, 1, 4.0), (0, 2, 4.0)],
    )
    monkeypatch.setattr(chain_pipeline, "_ensure_chain_apbs_grids", lambda c: None)

    class _Sim:
        def __init__(self, **kwargs):
            captured["sim_kwargs"] = kwargs
            captured["params"] = kwargs["params"]

        def run(self):
            return []

    monkeypatch.setattr(chain_pipeline, "ChainBDSimulator", _Sim)
    monkeypatch.setattr(chain_pipeline, "write_chain_results", lambda **k: [])
    monkeypatch.setattr(
        chain_pipeline,
        "analyse_molecule",
        lambda path, **k: types.SimpleNamespace(hydrodynamic_r=13.5),
    )


def test_T24_pipeline_switches_the_new_behavior_on(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured)
    cfg = _pipeline_config(
        tmp_path, dict(auto_diffusion=True, diffusion_update_interval=7)
    )
    chain_pipeline.run_chain(cfg)
    params = captured["params"]
    assert params.rotate_body_tensors is True
    assert params.remove_internal_rotation is True
    assert params.diffusion_update_interval == 7
    assert captured["sim_kwargs"]["target_hydro_radius"] == 13.5


def test_T24_pipeline_input_radius_overrides_the_computed_one(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured)
    cfg = _pipeline_config(tmp_path, dict(auto_diffusion=True))
    cfg.r_hydro_rec = 15.25
    chain_pipeline.run_chain(cfg)
    assert captured["sim_kwargs"]["target_hydro_radius"] == 15.25


def test_T24_pipeline_explicit_diffusion_has_no_rebuild(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(monkeypatch, captured)
    cfg = _pipeline_config(tmp_path, dict(D_trans=0.05, D_rot=0.001))
    chain_pipeline.run_chain(cfg)
    assert captured["params"].diffusion_update_interval == 0
    assert captured["sim_kwargs"]["D_trans"] == 0.05


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(D_trans=0.05),
        dict(D_rot=0.001),
    ],
)
def test_T24_pipeline_placeholders_are_a_hard_error(tmp_path, monkeypatch, overrides):
    _patch_pipeline(monkeypatch, {})
    cfg = _pipeline_config(tmp_path, overrides)
    with pytest.raises(ValueError, match="D_trans.*D_rot"):
        chain_pipeline.run_chain(cfg)


def test_T24_pipeline_off_center_target_is_an_error(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, {}, target_xyz=((1.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    cfg = _pipeline_config(tmp_path, dict(auto_diffusion=True))
    with pytest.raises(ValueError, match="center"):
        chain_pipeline.run_chain(cfg)


def test_T24_pipeline_target_centroid_ignores_ghost_atoms(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(
        monkeypatch, captured, target_xyz=((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    atoms = [
        Atom(x=-2.0, y=0.0, z=0.0, radius=2.0, name="CA"),
        Atom(x=2.0, y=0.0, z=0.0, radius=2.0, name="CB"),
        Atom(x=30.0, y=0.0, z=0.0, radius=0.0, name="GHO"),
        Atom(x=-40.0, y=5.0, z=0.0, radius=0.0, name="PKT"),
    ]
    monkeypatch.setattr(chain_pipeline, "parse_pqr", lambda p: Molecule(atoms=atoms))
    cfg = _pipeline_config(tmp_path, dict(auto_diffusion=True))
    chain_pipeline.run_chain(cfg)
    assert "sim_kwargs" in captured


def test_T24_pipeline_centered_target_within_tolerance(tmp_path, monkeypatch):
    captured = {}
    _patch_pipeline(
        monkeypatch, captured, target_xyz=((-2.0, 0.0, 0.0), (2.004, 0.0, 0.0))
    )
    cfg = _pipeline_config(tmp_path, dict(auto_diffusion=True))
    chain_pipeline.run_chain(cfg)
    assert "sim_kwargs" in captured


def test_T24_pipeline_non_positive_radius_is_an_error(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch, {})
    monkeypatch.setattr(
        chain_pipeline,
        "analyse_molecule",
        lambda path, **k: types.SimpleNamespace(hydrodynamic_r=0.0),
    )
    cfg = _pipeline_config(tmp_path, dict(auto_diffusion=True))
    with pytest.raises(ValueError, match="hydrodynamic radius"):
        chain_pipeline.run_chain(cfg)


def test_diffusion_update_interval_config_key(tmp_path):
    xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>t.pqr</receptor_pqr>
  <chain>
    <chain_json>c.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
    {key}
  </chain>
</pystarc>"""
    path = tmp_path / "input.xml"
    path.write_text(xml.format(key=""))
    assert parse(path).chain.diffusion_update_interval == 10
    path.write_text(
        xml.format(key="<diffusion_update_interval>25</diffusion_update_interval>")
    )
    assert parse(path).chain.diffusion_update_interval == 25
    path.write_text(
        xml.format(key="<diffusion_update_interval>-1</diffusion_update_interval>")
    )
    with pytest.raises(ValueError, match="diffusion_update_interval"):
        parse(path)


def test_results_json_logs_tensor_means(tmp_path):
    """results.json holds the start and the pooled mean of the trace values."""
    import json
    from pystarc.pipeline.chain_output_writer import write_results_json

    sim = _make_sim(
        n_atoms=6, interval=3, rotate=True, remove=True, max_steps=10, seed=48
    )
    sim.run()
    path = write_results_json(tmp_path, sim, sim.results, 1.0)
    data = json.loads(path.read_text())
    block = data["diffusion"]
    assert block["D_trans_start"] == pytest.approx(np.trace(sim.D_trans) / 3.0)
    assert block["D_rot_start"] == pytest.approx(np.trace(sim.D_rot) / 3.0)
    assert block["D_trans_mean"] > 0.0
    assert block["D_rot_mean"] > 0.0
    assert block["update_interval"] == 3
    assert block["n_rebuilds"] == sum(r.n_diffusion_rebuilds for r in sim.results)
