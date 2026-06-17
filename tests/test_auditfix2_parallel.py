"""
Regression tests for the NumPy vectorised batch runner in
pystarc.simulation.parallel.

These cover two behaviours of _run_numpy_batch: that a non-zero force function
enters the Brownian-dynamics step as the Ermak-McCammon drift, and that
trajectories which exhaust max_steps report the full step count and the
matching elapsed time, consistent with the serial runner.
"""

import math
import numpy as np

from pystarc.simulation.parallel import _run_numpy_batch
from pystarc.simulation.nam_simulator import NAMParameters, zero_force
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Molecule, Atom
from pystarc.molsystem.system_state import Fate


def _make_molecules():
    """Build a tiny receptor and a single-atom ligand for the batch runner."""
    mol1 = Molecule(name="receptor", atoms=[Atom(index=0, name="A", x=0.0, y=0.0, z=0.0)])
    mol2 = Molecule(name="ligand", atoms=[Atom(index=0, name="B", x=0.0, y=0.0, z=0.0)])
    return mol1, mol2


def _make_mobility():
    """A simple isotropic mobility tensor with no RPY coupling."""
    return MobilityTensor(
        D_trans1=0.01,
        D_rot1=0.001,
        D_trans2=0.01,
        D_rot2=0.001,
        radius1=1.0,
        radius2=1.0,
        use_rpy=False,
    )


def _empty_pathways():
    """A pathway set with no reactions, so trajectories only escape or time out."""
    return PathwaySet(reactions=[])


def test_max_steps_reports_full_step_count_and_time():
    """Trajectories that run out the clock report steps=max_steps and the
    matching time, mirroring the serial MAX_STEPS fate."""
    mol1, mol2 = _make_molecules()
    mob = _make_mobility()
    ps = _empty_pathways()
    # A large escape radius and small diffusion keep every trajectory inside the
    # escape sphere for the whole run, so all of them hit MAX_STEPS.
    params = NAMParameters(
        n_trajectories=4,
        dt=0.2,
        max_steps=5,
        r_start=10.0,
        r_escape=1.0e6,
        seed=123,
    )
    results = _run_numpy_batch(mol1, mol2, mob, ps, params, zero_force, [], False)

    assert len(results) == params.n_trajectories
    for r in results:
        assert r.fate == Fate.MAX_STEPS
        assert r.steps == params.max_steps
        assert math.isclose(r.time_ps, params.max_steps * params.dt, rel_tol=1e-12)
    # The aggregate step total must reflect the full work done, not zero.
    total_steps = sum(r.steps for r in results)
    assert total_steps == params.n_trajectories * params.max_steps


def _replicate_single_step_positions(mol2, mob, params, force_vec):
    """Reproduce the batch runner's first-step random draws and return the
    per-trajectory final positions for a constant translational force.

    The batch runner draws, from a generator seeded with params.seed and in this
    order: the starting directions, one uniform triple per trajectory consumed
    by the random orientation, then the translational noise. With max_steps=1
    and no reactions or escapes every trajectory takes exactly one step, so the
    final position is start + D_trans * F * dt + sqrt(2 * D_trans * dt) * W.
    """
    N = params.n_trajectories
    D_t = mob.relative_translational_diffusion()
    dt = params.dt
    sigma_t = math.sqrt(2.0 * D_t * dt)
    rng = np.random.default_rng(params.seed)
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pos = v * params.r_start
    # The orientation draws consume three uniforms per trajectory.
    for _ in range(N):
        rng.uniform(0, 1, 3)
    drift = D_t * np.asarray(force_vec) * dt
    noise = sigma_t * rng.standard_normal((N, 3))
    return pos + drift + noise


def test_force_enters_as_ermak_mccammon_drift():
    """A constant force biases the single-step displacement by exactly
    D_trans * F * dt, matching the Ermak-McCammon drift used by the serial
    runner and the BrownDye2 reference (dpos = mobility * force * dt)."""
    mol1, mol2 = _make_molecules()
    mob = _make_mobility()
    ps = _empty_pathways()
    params = NAMParameters(
        n_trajectories=64,
        dt=0.2,
        max_steps=1,
        r_start=10.0,
        r_escape=1.0e6,
        seed=7,
    )

    # A constant force, with zero torque and zero energy. The signature matches
    # zero_force and the StandardForceEngine __call__.
    F = np.array([5.0, -2.0, 1.0])

    def const_force(m1, m2):
        return F.copy(), np.zeros(3), 0.0

    res_force = _run_numpy_batch(mol1, mol2, mob, ps, params, const_force, [], False)

    # The expected separations follow analytically from the reproduced draws.
    expected_pos = _replicate_single_step_positions(mol2, mob, params, F)
    expected_sep = np.linalg.norm(expected_pos, axis=1)
    got_sep = np.array([r.final_separation for r in res_force])
    assert np.allclose(got_sep, expected_sep, rtol=1e-10, atol=1e-10)

    # A zero force reproduces the same draws without the drift, so the forced
    # run must differ from the zero-force run, confirming the drift is applied.
    res_zero = _run_numpy_batch(mol1, mol2, mob, ps, params, zero_force, [], False)
    zero_sep = np.array([r.final_separation for r in res_zero])
    expected_zero_pos = _replicate_single_step_positions(
        mol2, mob, params, np.zeros(3)
    )
    expected_zero_sep = np.linalg.norm(expected_zero_pos, axis=1)
    assert np.allclose(zero_sep, expected_zero_sep, rtol=1e-10, atol=1e-10)
    assert not np.allclose(got_sep, zero_sep)


def test_zero_force_path_is_unchanged_by_drift_term():
    """With zero_force the drift term vanishes and the batch result is identical
    to a run that supplies an explicit all-zero force function."""
    mol1, mol2 = _make_molecules()
    mob = _make_mobility()
    ps = _empty_pathways()
    params = NAMParameters(
        n_trajectories=16,
        dt=0.2,
        max_steps=3,
        r_start=10.0,
        r_escape=1.0e6,
        seed=99,
    )

    def explicit_zero(m1, m2):
        return np.zeros(3), np.zeros(3), 0.0

    res_sentinel = _run_numpy_batch(mol1, mol2, mob, ps, params, zero_force, [], False)
    res_explicit = _run_numpy_batch(mol1, mol2, mob, ps, params, explicit_zero, [], False)

    for a, b in zip(res_sentinel, res_explicit):
        assert a.fate == b.fate
        assert a.steps == b.steps
        assert math.isclose(a.final_separation, b.final_separation, rel_tol=1e-12)
