"""
Regression tests for the elapsed-time accounting of the weighted ensemble
Brownian-dynamics simulator.

The flux into the reacted and escaped states is the accumulated probability
weight divided by the elapsed simulation time, so the denominator must reflect
the time genuinely simulated. Each weighted ensemble trajectory maintains its
own clock, advanced by the real per-step time. The ensemble clock therefore
advances by the genuine elapsed time of each iteration, which reduces to
steps_per_iteration multiplied by the time step only when every trajectory
takes the full number of steps at the fixed time step. When trajectories stop
early at a reacted or escaped outcome, or take the smaller adaptive time step
near a reaction boundary, the elapsed time is correspondingly shorter.
"""

import math

import numpy as np

from pystarc.structures.molecules import (
    Molecule,
    Atom,
    ReactionCriteria,
    ContactPair,
)
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.simulation.we_simulator import WESimulator, WEParameters


def _make_molecules(lig_x=50.0, charge=0.0):
    mol1 = Molecule(name="rec")
    mol1.atoms.append(
        Atom(
            index=0,
            name="A",
            residue_name="X",
            residue_index=1,
            chain="A",
            x=0.0,
            y=0.0,
            z=0.0,
            charge=charge,
            radius=2.0,
        )
    )
    mol2 = Molecule(name="lig")
    mol2.atoms.append(
        Atom(
            index=0,
            name="B",
            residue_name="Y",
            residue_index=1,
            chain="A",
            x=lig_x,
            y=0.0,
            z=0.0,
            charge=-charge,
            radius=2.0,
        )
    )
    return mol1, mol2


def _make_pathways(cutoff=10.0):
    criteria = ReactionCriteria(
        name="r", pairs=[ContactPair(0, 0, cutoff)], n_needed=1
    )
    rxn = ReactionInterface(name="rxn", criteria=criteria)
    return PathwaySet(reactions=[rxn])


def test_full_steps_match_steps_times_dt():
    """When every trajectory takes all steps, total_time_ps equals n_iterations·steps_per_iteration·dt."""
    mol1, mol2 = _make_molecules(lig_x=50.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=10.0)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=3,
        r_start=50.0,
        steps_per_iteration=4,
        dt=0.2,
        adaptive_dt=True,
        seed=42,
    )
    sim = WESimulator(mol1, mol2, mob, ps, params)
    sim.run()
    expected = params.n_iterations * params.steps_per_iteration * params.dt
    assert math.isclose(sim.total_time_ps, expected, abs_tol=1e-9), (
        f"total_time_ps={sim.total_time_ps}, expected={expected}"
    )


def test_adaptive_small_step_near_boundary_shortens_time():
    """Near a reaction boundary the adaptive integrator uses dt_rxn, so total_time_ps equals the small-step product."""
    mol1, mol2 = _make_molecules(lig_x=12.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=10.0)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=3,
        r_start=12.0,
        steps_per_iteration=4,
        dt=0.2,
        dt_rxn=0.05,
        adaptive_dt=True,
        seed=42,
    )
    fixed = params.n_iterations * params.steps_per_iteration * params.dt
    rxn_scaled = params.n_iterations * params.steps_per_iteration * params.dt_rxn
    sim = WESimulator(mol1, mol2, mob, ps, params)
    sim.run()
    # The start sits inside 1.5 * cutoff = 15 A, so every adaptive step uses
    # dt_rxn and the elapsed time equals the small-step product, well below the
    # large-step product.
    assert sim.total_time_ps < fixed - 1e-9
    assert math.isclose(sim.total_time_ps, rxn_scaled, abs_tol=1e-9), (
        f"total_time_ps={sim.total_time_ps}, expected={rxn_scaled}"
    )


def test_immediate_escape_records_no_elapsed_time():
    """Trajectories that escape immediately accumulate escaped weight while total_time_ps stays at zero."""
    mol1, mol2 = _make_molecules(lig_x=50.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=10.0)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=3,
        r_start=50.0,
        r_escape=40.0,  # below r_start, so every trajectory escapes at step 0
        steps_per_iteration=4,
        dt=0.2,
        adaptive_dt=False,
        seed=42,
    )
    sim = WESimulator(mol1, mol2, mob, ps, params)
    sim.run()
    assert sim.weight_escaped > 0.0
    assert math.isclose(sim.total_time_ps, 0.0, abs_tol=1e-12), (
        f"total_time_ps={sim.total_time_ps}, expected 0.0"
    )


def test_elapsed_time_never_exceeds_fixed_step_product():
    """The accumulated total_time_ps stays positive but never exceeds the fixed-step product across a mixed run."""
    mol1, mol2 = _make_molecules(lig_x=20.0, charge=2.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=10.0)
    params = WEParameters(
        n_per_bin=3,
        n_bins=6,
        n_iterations=5,
        r_start=20.0,
        steps_per_iteration=6,
        dt=0.2,
        dt_rxn=0.05,
        adaptive_dt=True,
        seed=7,
    )
    upper = params.n_iterations * params.steps_per_iteration * params.dt
    sim = WESimulator(mol1, mol2, mob, ps, params)
    result = sim.run()
    assert sim.total_time_ps <= upper + 1e-9
    assert sim.total_time_ps > 0.0


def test_flux_matches_weight_over_elapsed_time():
    """The reported reaction and escape fluxes equal the accumulated weights divided by total_time_ps."""
    mol1, mol2 = _make_molecules(lig_x=20.0, charge=2.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=10.0)
    params = WEParameters(
        n_per_bin=3,
        n_bins=6,
        n_iterations=5,
        r_start=20.0,
        steps_per_iteration=6,
        dt=0.2,
        dt_rxn=0.05,
        adaptive_dt=True,
        seed=7,
    )
    sim = WESimulator(mol1, mol2, mob, ps, params)
    result = sim.run()
    assert sim.total_time_ps > 0.0
    assert math.isclose(
        result.flux_reaction,
        result.weight_reacted / sim.total_time_ps,
        rel_tol=1e-9,
        abs_tol=1e-15,
    )
    assert math.isclose(
        result.flux_escape,
        result.weight_escaped / sim.total_time_ps,
        rel_tol=1e-9,
        abs_tol=1e-15,
    )
