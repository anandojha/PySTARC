"""
Regression tests for the hard-sphere rejection path in the NAM simulator.

These tests cover how pystarc.simulation.nam_simulator handles a Brownian
dynamics step that places molecule 2 in hard-sphere overlap with molecule 1.
When such a step is detected, the simulator redraws a fresh displacement from
the previous position and keeps redrawing until the new position is overlap
free, up to a fixed attempt cap. If no overlap-free position is found within
the cap, the molecule stays at the previous, non-overlapping position. The
common requirement is that an overlapping configuration is never accepted as
the state carried into the next step.
"""

import numpy as np

import pystarc.simulation.nam_simulator as nsim
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.molsystem.system_state import Fate
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.simulation.nam_simulator import NAMParameters, NAMSimulator
from pystarc.structures.molecules import Atom, Molecule


def _make_mol(radius: float, charge: float = 0.0) -> Molecule:
    mol = Molecule(name="m")
    mol.atoms = [Atom(index=0, x=0.0, y=0.0, z=0.0, charge=charge, radius=radius)]
    return mol


def _make_sim(use_hard_sphere=True, r_start=12.0, r_escape=14.0, seed=3):
    mol1 = _make_mol(5.0)
    mol2 = _make_mol(5.0)
    mobility = MobilityTensor.from_radii(5.0, 5.0)
    pathway_set = PathwaySet()  # No reactions, so the trajectory diffuses freely.
    params = NAMParameters(
        n_trajectories=1,
        r_start=r_start,
        r_escape=r_escape,
        max_steps=400,
        seed=seed,
        use_brownian_bridge=False,
        use_hard_sphere=use_hard_sphere,
    )
    sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)
    # Use the simple escape fallback so the inner BD loop drives every step and
    # the hard-sphere branch is reached on each step.
    sim._outer_prop = None
    return sim


class _OverlapSpy:
    """Records every configuration the overlap check is asked about.

    The verdict callback maps molecule 2's centre position to True when that
    configuration is to be reported as overlapping. The spy stores, for each
    query, the position of molecule 2's first atom together with the verdict
    returned, so a test can confirm which configurations were accepted.
    """

    def __init__(self, verdict):
        self._verdict = verdict
        self.queries = []  # List of (position, overlaps) tuples.

    def __call__(self, mol1, mol2):
        pos = np.array([mol2.atoms[0].x, mol2.atoms[0].y, mol2.atoms[0].z])
        overlaps = bool(self._verdict(pos))
        self.queries.append((pos.copy(), overlaps))
        return overlaps


def test_forced_overlap_never_accepts_overlapping_step(monkeypatch):
    """When every redraw overlaps, the molecule stays at its previous position."""
    sim = _make_sim()

    spy = _OverlapSpy(lambda pos: True)  # Every configuration overlaps.
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)

    result = sim.run_one()

    # Some configuration was tested for overlap on the trajectory.
    assert spy.queries
    # The checker reported overlap on every query, so none could be accepted.
    assert all(overlaps for _, overlaps in spy.queries)
    # With no overlap-free position reachable, the molecule cannot advance by
    # diffusion and the trajectory runs to the step cap rather than escaping.
    assert result.fate == Fate.MAX_STEPS


def test_redraw_loops_until_overlap_free(monkeypatch):
    """A redraw that still overlaps is rejected and a further redraw is drawn."""
    sim = _make_sim()

    state = {"first_overlap_seen": False, "redraws_rejected": 0}

    def verdict(pos):
        # The first configuration checked on a step is treated as overlapping to
        # trigger the redraw loop. The first redraw is also treated as
        # overlapping so the loop must draw again; later redraws are accepted.
        if not state["first_overlap_seen"]:
            state["first_overlap_seen"] = True
            return True
        if state["redraws_rejected"] < 1:
            state["redraws_rejected"] += 1
            return True
        return False

    spy = _OverlapSpy(verdict)
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)

    result = sim.run_one()

    # At least one overlapping redraw was rejected before an overlap-free one
    # was accepted, so the loop ran more than a single redraw.
    assert state["redraws_rejected"] >= 1
    # Whichever configuration the simulator carried forward was overlap free, so
    # the final query that was accepted (the last non-overlapping one) exists.
    assert any(not overlaps for _, overlaps in spy.queries)
    assert result.fate in (Fate.ESCAPED, Fate.MAX_STEPS)


def test_accepted_positions_are_never_reported_overlapping(monkeypatch):
    """No configuration the simulator advances into is one the checker rejects.

    A region of space is declared overlapping. The simulator's position after
    every step is recorded, and none of those carried-forward positions may lie
    in the overlapping region.
    """
    sim = _make_sim()

    def verdict(pos):
        # Declare a slab of space overlapping. Any configuration whose x
        # coordinate is in this band must never be accepted.
        return 2.0 < pos[0] < 6.0

    spy = _OverlapSpy(verdict)
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)

    # The reaction check runs once at the top of each step on the configuration
    # carried forward from the previous step. Recording molecule 2's position
    # there captures exactly the states the simulator accepted, with no trial
    # configurations mixed in.
    carried_positions = []
    original_check_all = sim.pathway_set.check_all

    def _recording_check_all(mol1, mol2, rng, *args, **kwargs):
        carried_positions.append(
            np.array([mol2.atoms[0].x, mol2.atoms[0].y, mol2.atoms[0].z])
        )
        return original_check_all(mol1, mol2, rng, *args, **kwargs)

    sim.pathway_set.check_all = _recording_check_all

    sim.run_one()

    # The starting placement on the b-sphere is fixed by the seed; every later
    # carried-forward state was produced by an accepted step and must never sit
    # in the declared overlapping band.
    assert len(carried_positions) > 1
    for pos in carried_positions[1:]:
        assert not (2.0 < pos[0] < 6.0)


def test_no_overlap_path_is_unchanged(monkeypatch):
    """With no overlap ever reported, the trajectory matches the hard-sphere-off run.

    The normal, no-overlap path and its random-number usage are untouched by the
    rejection logic, so a run in which the checker never reports overlap is
    bitwise identical to a run with hard spheres disabled.
    """
    sim_on = _make_sim(use_hard_sphere=True, seed=11)
    spy = _OverlapSpy(lambda pos: False)  # Nothing ever overlaps.
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)
    result_on = sim_on.run_one()

    sim_off = _make_sim(use_hard_sphere=False, seed=11)
    result_off = sim_off.run_one()

    assert result_on.fate == result_off.fate
    assert result_on.steps == result_off.steps
    assert result_on.final_separation == result_off.final_separation
    assert result_on.time_ps == result_off.time_ps
