"""
Tests for the weighted ensemble bridge crossing test, the hard-sphere
rejection, the Rotne-Prager-Yamakawa position dependence, and the innermost-bin
clamp added to WESimulator.

These exercise the WE code path only. They build the smallest possible
single-atom receptor and ligand, a mobility tensor, and a one-pair reaction, and
they call WESimulator methods directly. No GPU, APBS, or external binary is
required.
"""

import math
import numpy as np

from pystarc.simulation.we_simulator import (
    WESimulator,
    WEParameters,
    WETrajectory,
)
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.structures.molecules import (
    Molecule,
    Atom,
    ReactionCriteria,
    ContactPair,
)
from pystarc.transforms.quaternion import Quaternion


def _make_molecules(lig_x, radius=2.0):
    """Build a single-atom receptor at the origin and a single-atom ligand on
    the x-axis at lig_x."""
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
            charge=0.0,
            radius=radius,
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
            charge=0.0,
            radius=radius,
        )
    )
    return mol1, mol2


def _make_pathways(cutoff=10.0):
    criteria = ReactionCriteria(name="r", pairs=[ContactPair(0, 0, cutoff)], n_needed=1)
    rxn = ReactionInterface(name="rxn", criteria=criteria)
    return PathwaySet(reactions=[rxn])


class _ZeroRNG:
    """A bridge RNG stub whose uniform draws are always 0.0, so the bridge
    sample u = 0.0 is below any positive crossing probability and the bridge
    fires whenever the path-crossing probability is positive."""

    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size)


class _OneRNG:
    """A bridge RNG stub whose uniform draws are always nearly 1.0, so the
    bridge sample never falls below the crossing probability and the bridge does
    not fire."""

    def random(self, size=None):
        if size is None:
            return 1.0 - 1e-12
        return np.full(size, 1.0 - 1e-12)


def _build_sim(lig_x, cutoff, r_start, use_brownian_bridge=True, use_rpy=True):
    mol1, mol2 = _make_molecules(lig_x)
    mob = MobilityTensor.from_radii(10.0, 5.0, use_rpy=use_rpy)
    ps = _make_pathways(cutoff=cutoff)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=1,
        r_start=r_start,
        steps_per_iteration=1,
        dt=0.2,
        adaptive_dt=False,
        use_brownian_bridge=use_brownian_bridge,
        seed=42,
    )
    return WESimulator(mol1, mol2, mob, ps, params)


def test_bin_of_clamps_below_inner_edge_to_bin_zero():
    """A separation below the innermost bin edge r_lo returns bin 0 rather than
    -1, so the walker is placed in the innermost bin instead of being pinned to
    its previous bin."""
    sim = _build_sim(lig_x=50.0, cutoff=10.0, r_start=50.0)
    r_lo = float(sim._bins[0])
    r_hi = float(sim._bins[-1])
    # Just below the innermost edge.
    assert sim._bin_of(r_lo - 0.5) == 0
    # Far below the innermost edge.
    assert sim._bin_of(0.0) == 0
    # A separation strictly inside the bin range lands in a valid bin.
    mid = 0.5 * (r_lo + r_hi)
    assert 0 <= sim._bin_of(mid) < sim.params.n_bins
    # At or beyond the outermost edge lies the escape region and returns -1.
    assert sim._bin_of(r_hi + 1.0) == -1


def test_bridge_fires_when_endpoints_stay_above_cutoff():
    """When both the previous and current pair distances stay just above the
    cutoff the endpoint-only test does not fire, but the Brownian bridge fires
    when the bridge sample falls below the crossing probability."""
    cutoff = 10.0
    # Place the ligand so the single contact pair distance is just above the
    # cutoff at the current position, x1 = 0.01.
    lig_x = cutoff + 0.01
    sim = _build_sim(lig_x=lig_x, cutoff=cutoff, r_start=80.0, use_brownian_bridge=True)
    # Force the bridge sample to fire by replacing the bridge RNG.
    sim.rng_bb = _ZeroRNG()
    # Build a trajectory whose previous-step state puts the pair just above the
    # cutoff as well, x0 = 0.01, with a finite previous dt and diffusion so the
    # crossing probability is positive.
    traj = WETrajectory(
        position=np.array([lig_x, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=1.0,
        bin_idx=0,
        prev_pair_dists=[np.array([cutoff + 0.01])],
        prev_dt=0.2,
        prev_D_eff=1.0,
    )
    _, outcome = sim._step_traj(traj)
    assert outcome == "reacted"


def test_bridge_does_not_fire_when_disabled():
    """With the bridge disabled the same configuration, where both endpoints
    stay above the cutoff, does not register a reaction."""
    cutoff = 10.0
    lig_x = cutoff + 0.01
    sim = _build_sim(
        lig_x=lig_x, cutoff=cutoff, r_start=80.0, use_brownian_bridge=False
    )
    traj = WETrajectory(
        position=np.array([lig_x, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=1.0,
        bin_idx=0,
        prev_pair_dists=[np.array([cutoff + 0.01])],
        prev_dt=0.2,
        prev_D_eff=1.0,
    )
    _, outcome = sim._step_traj(traj)
    assert outcome == "ongoing"


def test_bridge_does_not_fire_when_sample_above_probability():
    """With the bridge enabled but the bridge sample at nearly 1.0, the path is
    not counted as crossing, so the endpoint-only outcome of no reaction
    stands."""
    cutoff = 10.0
    lig_x = cutoff + 0.01
    sim = _build_sim(lig_x=lig_x, cutoff=cutoff, r_start=80.0, use_brownian_bridge=True)
    sim.rng_bb = _OneRNG()
    traj = WETrajectory(
        position=np.array([lig_x, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=1.0,
        bin_idx=0,
        prev_pair_dists=[np.array([cutoff + 0.01])],
        prev_dt=0.2,
        prev_D_eff=1.0,
    )
    _, outcome = sim._step_traj(traj)
    assert outcome == "ongoing"


def test_endpoint_below_cutoff_still_fires_with_bridge_off():
    """An endpoint that sits below the cutoff fires through the ordinary
    endpoint test even with no prior bridge state, confirming the bridge does
    not suppress ordinary detections."""
    cutoff = 10.0
    # Place the ligand inside the cutoff so the endpoint test fires.
    lig_x = cutoff - 0.5
    sim = _build_sim(lig_x=lig_x, cutoff=cutoff, r_start=80.0, use_brownian_bridge=True)
    traj = WETrajectory(
        position=np.array([lig_x, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=1.0,
        bin_idx=0,
    )
    _, outcome = sim._step_traj(traj)
    assert outcome == "reacted"


def test_step_uses_position_dependent_rpy_diffusion():
    """The step evaluates the relative translational diffusion at the current
    pair separation, so the Rotne-Prager-Yamakawa position dependence is
    applied. The recorded call argument is the pair separation vector, and the
    position-dependent value differs from the bare D0 near contact."""
    cutoff = 10.0
    # A small separation, where the RPY correction is appreciable.
    lig_x = 16.0
    sim = _build_sim(lig_x=lig_x, cutoff=cutoff, r_start=80.0)
    recorded = []
    real = sim.mobility.relative_translational_diffusion

    def _recording(r_vec=None):
        recorded.append(r_vec)
        return real(r_vec)

    sim.mobility.relative_translational_diffusion = _recording
    traj = WETrajectory(
        position=np.array([lig_x, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=1.0,
        bin_idx=0,
    )
    sim._step_traj(traj)
    # The diffusion was evaluated with the pair separation vector, not with the
    # default None.
    assert len(recorded) >= 1
    first = recorded[0]
    assert first is not None
    assert np.allclose(np.asarray(first), traj.position)
    # The position-dependent RPY value is smaller than the bare D0 near contact.
    d_pos = real(traj.position)
    d0 = real(None)
    assert d_pos < d0


def test_hard_sphere_rejection_keeps_ligand_when_all_redraws_overlap():
    """When every redrawn displacement overlaps the receptor, the ligand stays
    at its previous non-overlapping position rather than being placed inside the
    receptor."""
    # Large radii so any small displacement overlaps. The receptor and ligand
    # radii sum to 4.0; start the ligand far enough out to be free.
    cutoff = 1.0  # tiny cutoff so the reaction never fires during the step
    mol1, mol2 = _make_molecules(lig_x=4.0, radius=2.0)
    mob = MobilityTensor.from_radii(10.0, 5.0)
    ps = _make_pathways(cutoff=cutoff)
    params = WEParameters(
        n_per_bin=2,
        n_bins=5,
        n_iterations=1,
        r_start=80.0,
        steps_per_iteration=1,
        dt=0.2,
        adaptive_dt=False,
        use_hard_sphere=True,
        use_brownian_bridge=False,
        seed=1,
    )
    sim = WESimulator(mol1, mol2, mob, ps, params)
    # Start the ligand exactly at the overlap boundary so it is not overlapping
    # yet, but any inward draw overlaps. Force overlap on every redraw by
    # patching the overlap check to always report overlap.
    import pystarc.simulation.we_simulator as we_mod

    original = we_mod._check_hard_sphere_overlap
    we_mod._check_hard_sphere_overlap = lambda m1, m2: True
    try:
        start = np.array([5.0, 0.0, 0.0])
        traj = WETrajectory(
            position=start.copy(),
            orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
            weight=1.0,
            bin_idx=0,
        )
        new_traj, outcome = sim._step_traj(traj)
    finally:
        we_mod._check_hard_sphere_overlap = original
    assert outcome == "ongoing"
    # The ligand was held at its previous position because no overlap-free
    # redraw was found.
    assert np.allclose(new_traj.position, start)
