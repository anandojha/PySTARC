"""
The Brownian bridge after a force backstep in the CPU engine.

The bridge gives each reaction pair the chance exp(-x0 x1 / (D dt)) of having
touched its cutoff between two steps. The bridged move must have variance
2 D dt per direction. A force backstep retakes a step of dt as two half steps
of hdt. The first half step moves with D at the start of the step and the
second with D at the midpoint. The bridge gets D dt summed over both halves. A
move redrawn after a hard sphere overlap is one full step with D at the start.
A step without a backstep gives the bridge the very D and dt of the move as
before.
"""

from __future__ import annotations

import numpy as np
import pytest

import pystarc.simulation.nam_simulator as nsim
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.molsystem.system_state import Fate
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.simulation.nam_simulator import NAMParameters, NAMSimulator
from pystarc.structures.molecules import Atom, ContactPair, Molecule, ReactionCriteria

# The wrappers below always call the original functions.
_STEP = nsim.bd_step_wiener_two_body
_BRIDGE = nsim.check_reaction_with_bridge


def _sim(use_bb: bool = True, use_hard_sphere: bool = False) -> NAMSimulator:
    """Two spheres with hydrodynamic coupling and one reaction pair far away."""
    mol1 = Molecule(name="m1", atoms=[Atom(index=0, radius=1.0)])
    mol2 = Molecule(name="m2", atoms=[Atom(index=0, radius=1.0)])
    pair = ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=2.0)
    rxn = ReactionInterface(name="contact", criteria=ReactionCriteria(pairs=[pair]))
    params = NAMParameters(
        n_trajectories=1,
        max_steps=3,
        r_start=20.0,
        r_escape=1000.0,
        seed=11,
        use_brownian_bridge=use_bb,
        use_hard_sphere=use_hard_sphere,
    )
    # The RPY coupling makes D depend on the separation.
    mobility = MobilityTensor.from_radii(5.0, 5.0)
    sim = NAMSimulator(mol1, mol2, mobility, PathwaySet([rxn]), params)
    # The simple escape check keeps all three steps in the BD loop.
    sim._outer_prop = None
    return sim


def _run(monkeypatch, sim, backsteps, overlaps=()):
    """Run one trajectory. The force backstep test fires on the listed calls and
    the hard sphere test finds an overlap on the listed calls. Return the result,
    every move as (D, dt, start position) and every bridge check as (D_eff, dt)."""
    moves, bridges = [], []
    calls = {"backstep": 0, "overlap": 0}

    def _step(position, orientation, force, torque, D_trans, D_rot1, D_rot2, dt, *rest):
        moves.append((D_trans, dt, tuple(position)))
        return _STEP(
            position, orientation, force, torque, D_trans, D_rot1, D_rot2, dt, *rest
        )

    def _backstep(*args, **kwargs):
        calls["backstep"] += 1
        return calls["backstep"] - 1 in backsteps

    def _overlap(mol1, mol2):
        calls["overlap"] += 1
        return calls["overlap"] - 1 in overlaps

    def _bridge(target, scratch, pathway_set, old, new, D_eff, dt, rng, rng_bb=None):
        bridges.append((D_eff, dt))
        return _BRIDGE(
            target, scratch, pathway_set, old, new, D_eff, dt, rng, rng_bb=rng_bb
        )

    monkeypatch.setattr(nsim, "bd_step_wiener_two_body", _step)
    monkeypatch.setattr(nsim, "backstep_due_to_force", _backstep)
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", _overlap)
    monkeypatch.setattr(nsim, "check_reaction_with_bridge", _bridge)
    return sim.run_one(), moves, bridges


class TestBridgeAfterBackstep:
    def test_bridge_gets_D_dt_of_both_half_steps(self, monkeypatch):
        sim = _sim()
        result, moves, bridges = _run(monkeypatch, sim, backsteps={0})
        assert result.fate == Fate.MAX_STEPS
        # Step 0 is a trial move and two half steps. Steps 1 and 2 move once.
        assert len(moves) == 5
        (D_a, dt0, start), (D_1, h_1, start_1), (D_b, h_2, mid) = moves[:3]
        assert start_1 == start and D_1 == D_a
        assert h_1 == h_2 == dt0 / 2.0
        # D_a is D at the start of the step and D_b is D at the midpoint.
        D_rel = sim.mobility.relative_translational_diffusion
        assert D_a == D_rel(np.array(start)) and D_b == D_rel(np.array(mid))
        assert abs(D_b - D_a) > 1.0e-6 * D_a
        # The bridge runs at the top of steps 1 and 2.
        assert len(bridges) == 2
        D_eff, dt_bb = bridges[0]
        assert dt_bb == dt0
        assert D_eff * dt_bb == pytest.approx(D_a * h_1 + D_b * h_2, rel=1.0e-12)
        # Before the fix the bridge got D_a hdt. The move has about twice that.
        assert D_eff * dt_bb > 1.9 * D_a * h_1
        # The controller still grows the next step from hdt.
        assert moves[3][1] <= 1.1 * h_1 * (1.0 + 1.0e-12)
        # A step without a backstep gives the bridge its own D and dt bit for bit.
        assert bridges[1] == moves[3][:2]

    def test_redraw_after_backstep_gets_one_full_step(self, monkeypatch):
        sim = _sim(use_hard_sphere=True)
        _, moves, bridges = _run(monkeypatch, sim, backsteps={0}, overlaps={0})
        # Step 0 is a trial move, two half steps and one redraw of the full step.
        assert len(moves) == 6
        (D_a, dt0, start), _, _, (D_r, dt_r, start_r) = moves[:4]
        assert start_r == start and D_r == D_a and dt_r == dt0
        assert bridges[0] == (D_r, dt_r)

    def test_bridge_on_and_off_take_the_same_path_through_backsteps(self, monkeypatch):
        # The bridge draws from rng_bb only. The main rng gives the same path.
        paths = [
            _run(monkeypatch, _sim(use_bb=use_bb), backsteps={0, 1, 2})[1]
            for use_bb in (True, False)
        ]
        assert len(paths[0]) == 9
        assert paths[0] == paths[1]
