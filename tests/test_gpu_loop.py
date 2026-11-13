"""The GPU loop on a synthetic system with no force: the last allowed step, the
checkpoint guard, the encounter rows and the overlap map. The tests need a
GPU."""

from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.zeros(1).sum()
except Exception:
    pytest.skip("no usable CUDA device", allow_module_level=True)

import pystarc.simulation.gpu_batch_simulator as gbs
from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.simulation.nam_simulator import NAMParameters
from pystarc.structures.molecules import Atom, ContactPair, Molecule, ReactionCriteria

KEYS = ("outcome", "n_steps", "total_time_ps", "bb_triggered", "encounter_traj",
        "encounter_step", "encounter_pos", "encounter_bridge_only")


def _mol(xyz, radii):
    return Molecule(atoms=[Atom(index=i, name="GHO" if r == 0 else "C", x=p[0], y=p[1], z=p[2],
                                radius=r) for i, (p, r) in enumerate(zip(xyz, radii))])


def _run(work, n=400, max_steps=40, seed=5, rxns=((12.0, None, None),), b=15.0, r_esc=16.6,
         bridge=True, ckpt=0, key="k", rec=None, lig=None, overlap=False):
    """One pair, receptor atom 0 and ligand atom 0, per reaction (cutoff, before, after)."""
    rec = rec or _mol([[0.0, 0.0, 0.0]], [2.0])
    lig = lig or _mol([[0.0, 0.0, 0.0]], [1.5])
    rx = [ReactionInterface(name=f"r{i}", criteria=ReactionCriteria(
        name=f"r{i}", pairs=[ContactPair(0, 0, c)], n_needed=1, state_before=sb, state_after=sa),
        state_before=sb, state_after=sa) for i, (c, sb, sa) in enumerate(rxns)]
    p = NAMParameters(n_trajectories=n, max_steps=max_steps, r_start=b, r_escape=r_esc, seed=seed,
                      hydrodynamic_interactions=False, use_brownian_bridge=bridge,
                      minimum_core_reaction_dt=0.05)
    p.checkpoint_interval, p._run_key, p._work_dir, p._overlap_check = ckpt, key, str(work), overlap
    eng = GPUBatchForceEngine([], [], alpha=0.0, debye_length=7.86)
    sim = gbs.GPUBatchSimulator(rec, lig, MobilityTensor.from_radii(10.0, 2.0),
                                PathwaySet(rx, first_state=rxns[0][1]), p, eng)
    return sim.run().sim_data


def _same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in KEYS)


def test_max_steps_1_books_the_escapes_after_the_first_move(tmp_path):
    one = _run(tmp_path, n=5000, max_steps=1)
    two = _run(tmp_path, n=5000, max_steps=2)
    esc = (two["outcome"] == 2) & (two["n_steps"] == 1)
    assert esc.sum() > 20 and np.array_equal(one["outcome"] == 2, esc)


def test_runs_of_m_and_m_plus_1_steps_agree_on_every_trajectory_ended_by_m(tmp_path):
    m = 25
    a, b = _run(tmp_path, n=1000, max_steps=m), _run(tmp_path, n=1000, max_steps=m + 1)
    end = (b["outcome"] != 3) & (b["n_steps"] <= m)
    assert (b["outcome"][end] == 1).sum() > 20 and (b["outcome"][end] == 2).sum() > 20
    assert np.array_equal(a["outcome"][end], b["outcome"][end])
    assert np.array_equal(a["n_steps"][end], b["n_steps"][end])
    assert np.all(a["outcome"][~end] == 3) and np.all(a["n_steps"][~end] == m)


class _Stop(BaseException):
    pass


def _stopped(monkeypatch, work, **kw):
    """A run ended right after its first checkpoint lands."""
    replace = Path.replace

    def stop(self, target):
        replace(self, target)
        raise _Stop

    monkeypatch.setattr(Path, "replace", stop)
    with pytest.raises(_Stop):
        _run(work, ckpt=50, **kw)
    monkeypatch.undo()
    assert (work / "checkpoint.npz").exists()


def test_a_resumed_run_equals_the_run_that_was_not_stopped(tmp_path, monkeypatch):
    whole = _run(tmp_path / "a", ckpt=50)
    _stopped(monkeypatch, tmp_path / "b")
    assert _same(_run(tmp_path / "b", ckpt=50), whole)
    assert not (tmp_path / "b" / "checkpoint.npz").exists()


@pytest.mark.parametrize("change,word", [(dict(n=401), "run_N"), (dict(seed=6), "run_seed"),
                                         (dict(max_steps=41), "run_max_steps"),
                                         (dict(key="other"), "run_key")])
def test_a_checkpoint_of_another_run_raises(tmp_path, monkeypatch, change, word):
    _stopped(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match=f"written by another run.*{word}"):
        _run(tmp_path, ckpt=50, **change)


def test_a_checkpoint_of_another_state_machine_raises(tmp_path, monkeypatch):
    sm = ((14.6, "a", "b"), (14.0, "b", "c"))
    _stopped(monkeypatch, tmp_path, max_steps=200, rxns=sm)
    with pytest.raises(RuntimeError, match="sm_rxn_fire_counts shape"):
        _run(tmp_path, ckpt=50, max_steps=200, rxns=sm + ((13.0, "c", "d"),))


def test_checkpoints_off_leave_a_checkpoint_alone(tmp_path, monkeypatch):
    _stopped(monkeypatch, tmp_path / "b")
    with pytest.warns(RuntimeWarning, match="left untouched"):
        out = _run(tmp_path / "b")
    assert _same(out, _run(tmp_path / "a")) and (tmp_path / "b" / "checkpoint.npz").exists()


def test_flattened_encounter_rows_are_the_reactions_at_the_end_of_their_step(tmp_path):
    s = _run(tmp_path, max_steps=200)
    t = s["encounter_traj"]
    assert np.array_equal(np.sort(t), np.nonzero(s["outcome"] == 1)[0])
    assert np.array_equal(s["encounter_step"], s["n_steps"][t] - 1)
    crossed = np.linalg.norm(s["encounter_pos"], axis=1) >= 12.0
    assert 0 < crossed.sum() < len(t)
    assert np.array_equal(s["encounter_bridge_only"], crossed.astype(np.int64))
    assert np.array_equal(s["bb_triggered"][t], s["encounter_bridge_only"])
    assert s["bb_triggered"].sum() == crossed.sum()


def test_state_machine_reaction_at_the_start_pose_gets_its_row(tmp_path):
    s = _run(tmp_path, n=50, rxns=((16.0, "a", "b"),), bridge=False)
    assert np.array_equal(np.sort(s["encounter_traj"]), np.arange(50))
    assert np.all(s["encounter_step"] == 0) and np.all(s["n_steps"] == 0)
    assert np.array_equal(s["encounter_pos"], s["start_pos"][s["encounter_traj"]])
    assert not s["encounter_bridge_only"].any() and not s["bb_triggered"].any()


def test_state_machine_rows_are_the_trajectories_that_ended(tmp_path):
    s = _run(tmp_path, n=300, max_steps=300, rxns=((14.6, "a", "b"), (14.0, "b", "c")))
    reacted = np.nonzero(s["outcome"] == 1)[0]
    assert len(reacted) > 50 and np.array_equal(np.sort(s["encounter_traj"]), reacted)
    t = s["encounter_traj"]
    crossed = np.linalg.norm(s["encounter_pos"], axis=1) >= 14.0
    assert np.array_equal(s["encounter_bridge_only"], crossed.astype(np.int64))
    assert np.array_equal(s["bb_triggered"][t], s["encounter_bridge_only"])


def test_overlap_map_and_test_hold_real_atoms_only(tmp_path, monkeypatch):
    """A ring of seven 1.7 Å atoms with a ghost at its centre."""
    th = 2.0 * np.pi * np.arange(7) / 7.0
    ring = np.stack([4.5 * np.cos(th), 4.5 * np.sin(th), np.zeros(7)], axis=1)
    rec = _mol(np.vstack([ring, np.zeros((1, 3))]), [1.7] * 7 + [0.0])
    lig = _mol([[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.7, 1.5, 0.0])
    built, radii = [], []

    class Spy(gbs.OverlapGrid):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            built.append(self)

        def overlaps(self, lig_pos, lig_radii, *a, **k):
            radii.append(cp.asnumpy(lig_radii))
            return super().overlaps(lig_pos, lig_radii, *a, **k)

    monkeypatch.setattr(gbs, "OverlapGrid", Spy)
    _run(tmp_path, n=50, max_steps=1, rec=rec, lig=lig, overlap=True)
    g = built[0]
    assert np.array_equal(cp.asnumpy(g.rec_radii), [1.7] * 7)
    assert np.array_equal(radii[0], [1.7, 1.5])
    one = lambda x: bool(g.overlaps(cp.asarray([[x]]), cp.asarray([1.7]))[0])
    assert not one([0.4, 0.0, 0.0]) and one(ring[2] * (1.0 - 1.0 / 4.5))
    rng = np.random.default_rng(10)
    lig0 = np.array([[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]])
    qn = rng.normal(size=(4000, 4))
    w, x, y, z = (qn / np.linalg.norm(qn, axis=1, keepdims=True)).T
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]).transpose(2, 0, 1)
    pos = np.einsum("nij,kj->nki", R, lig0) + rng.uniform(-5.5, 5.5, (4000, 1, 3))
    d = np.linalg.norm(pos[:, :, None] - ring[None, None], axis=3)
    ref = (d < 0.5 * (np.array([1.7, 1.5])[:, None] + 1.7)).any(axis=(1, 2))
    got = cp.asnumpy(g.overlaps(cp.asarray(pos), cp.asarray([1.7, 1.5])))
    assert ref.sum() > 200 and (~ref).sum() > 200 and np.array_equal(got, ref)
