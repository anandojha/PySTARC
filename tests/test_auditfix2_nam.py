"""
Regression tests for the NAM simulator's outer-propagator fallback signalling
and for the reported trajectory time in picoseconds.

These tests cover two behaviours of pystarc.simulation.nam_simulator:

1. When the outer propagator cannot be constructed, the simulator falls back to
   the simple escape check and emits a warning that carries the originating
   exception message, so the downgrade is observable rather than silent.

2. The TrajectoryResult.time_ps reported by run_one equals the running total of
   the adaptive time steps actually applied along the trajectory, including the
   two half steps taken on a force backstep, rather than the step count times
   the nominal normal step.
"""

import warnings

import numpy as np

import pystarc.simulation.nam_simulator as nsim
import pystarc.simulation.outer_propagator as op
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.molsystem.system_state import Fate
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.simulation.nam_simulator import NAMParameters, NAMSimulator
from pystarc.structures.molecules import Atom, Molecule


def _make_mol(charge: float) -> Molecule:
    mol = Molecule(name="m")
    mol.atoms = [Atom(index=0, x=0.0, y=0.0, z=0.0, charge=charge, radius=1.5)]
    return mol


def _make_inputs():
    mol1 = _make_mol(1.0)
    mol2 = _make_mol(-1.0)
    mobility = MobilityTensor.from_radii(5.0, 5.0)
    pathway_set = PathwaySet()  # No reactions, so trajectories escape.
    params = NAMParameters(
        n_trajectories=1,
        r_start=50.0,
        r_escape=60.0,
        max_steps=5000,
        seed=7,
        use_brownian_bridge=False,
        use_hard_sphere=False,
    )
    return mol1, mol2, mobility, pathway_set, params


def test_outer_propagator_failure_warns_and_falls_back():
    """A failed outer-propagator setup emits one RuntimeWarning carrying the failure message and leaves the outer propagator disabled."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    original = op.OuterPropagator

    def _raise(*args, **kwargs):
        raise ValueError("synthetic outer propagator failure")

    op.OuterPropagator = _raise
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)
    finally:
        op.OuterPropagator = original

    assert sim._outer_prop is None
    runtime_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 1
    message = str(runtime_warnings[0].message)
    assert "synthetic outer propagator failure" in message


def test_successful_outer_propagator_setup_is_silent():
    """A successful outer-propagator setup enables the propagator and emits no RuntimeWarning."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)

    assert sim._outer_prop is not None
    runtime_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 0


def test_time_ps_sums_actual_adaptive_steps():
    """result.time_ps equals the sum of the adaptive steps actually applied, not steps times the nominal dt."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()
    sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)
    # Use the simple escape fallback so the adaptive controller drives every step.
    sim._outer_prop = None

    applied_dts = []
    original_get_dt = nsim.AdaptiveTimeStep.get_dt

    def _recording_get_dt(self, *args, **kwargs):
        dt = original_get_dt(self, *args, **kwargs)
        applied_dts.append(dt)
        return dt

    nsim.AdaptiveTimeStep.get_dt = _recording_get_dt
    try:
        result = sim.run_one()
    finally:
        nsim.AdaptiveTimeStep.get_dt = original_get_dt

    assert result.fate == Fate.ESCAPED
    assert applied_dts  # The controller was queried at least once.
    # The reported time is the running total of the applied adaptive steps.
    assert result.time_ps == sum(applied_dts)
    # The adaptive steps differ from the nominal normal step, so the reported
    # time is not the step count times params.dt.
    assert abs(result.time_ps - result.steps * params.dt) > 1.0
    assert result.time_ps > 0.0


def test_time_ps_accumulates_backstep_half_steps():
    """A force backstep contributes its full step time as two half steps so time_ps still equals the sum of the chosen steps."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    def _strong_varying_force(m1, m2):
        atom = m2.atoms[0]
        r_vec = np.array([atom.x, atom.y, atom.z])
        d = float(np.linalg.norm(r_vec))
        force = -5000.0 * r_vec / (d**3 + 1e-6)
        return force, np.zeros(3), 0.0

    sim = NAMSimulator(
        mol1, mol2, mobility, pathway_set, params, force_fn=_strong_varying_force
    )
    sim._outer_prop = None

    applied_dts = []
    original_get_dt = nsim.AdaptiveTimeStep.get_dt

    def _recording_get_dt(self, *args, **kwargs):
        dt = original_get_dt(self, *args, **kwargs)
        applied_dts.append(dt)
        return dt

    backstep_count = {"n": 0}
    original_backstep = nsim.backstep_due_to_force

    def _counting_backstep(*args, **kwargs):
        fired = original_backstep(*args, **kwargs)
        if fired:
            backstep_count["n"] += 1
        return fired

    nsim.AdaptiveTimeStep.get_dt = _recording_get_dt
    nsim.backstep_due_to_force = _counting_backstep
    try:
        result = sim.run_one()
    finally:
        nsim.AdaptiveTimeStep.get_dt = original_get_dt
        nsim.backstep_due_to_force = original_backstep

    # At least one backstep must fire for this test to exercise the half-step path.
    assert backstep_count["n"] > 0
    # Each backstep advances its two half steps, which sum to the full step, so
    # the reported time still equals the sum of the steps the controller chose.
    assert result.time_ps == sum(applied_dts)
