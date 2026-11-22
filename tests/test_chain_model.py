"""Tests of the flexible chain model: COFFDROP tables in the chain JSON,
vectorized chain forces, exclusions and the elastic network, the inner clock
and the target field of the inner steps, the hard-sphere rejection, the last
step, and the random streams of the trajectories."""

from __future__ import annotations
from pystarc.simulation import chain_simulator
from pystarc.simulation import coffdrop_chain as cc
from pystarc.simulation.chain_simulator import (
    ChainBDParameters,
    ChainBDSimulator,
    _check_chain_overlap,
    trajectory_rngs,
)
from pystarc.structures.chain_io import load_chain_from_json, save_chain_to_json
from pystarc.pipeline.input_parser import ChainConfig, PySTARCConfig
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.structures.molecules import Atom, ContactPair, Molecule, ReactionCriteria
from pystarc.molsystem.system_state import Fate
import numpy as np
import json
import pytest


@pytest.fixture(scope="module")
def tabled_chain():
    """A COFFDROP chain with side chains and its tables, and coil positions."""
    chain = cc.chain_from_sequence("KAEWRDGLYSTV")
    rng = np.random.default_rng(5)
    pos = np.cumsum(rng.normal(scale=2.2, size=(chain.n_atoms, 3)), axis=0)
    return chain, pos - pos.mean(axis=0)


def _reference_forces(state):
    """Forces of the per-term kernels and of the scalar table derivative."""
    state.zero_forces()
    common = state.common
    for b in common.bonds + common.network:
        cc._bond_force_state(state, b)
    for a in common.angles:
        cc._angle_force_state(state, a)
    for t in common.torsions:
        cc._torsion_force_state(state, t)
    F = state.forces.copy()
    if common.coffdrop_params is not None:
        P = state.positions
        for (i, j), t in common.pair_lookups.items():
            d = P[i] - P[j]
            r = np.linalg.norm(d)
            f = -common.coffdrop_params.pair_pots[t].deriv(r) * d / r
            F[i] += f
            F[j] -= f
    return F


def test_vectorized_forces_match_the_term_kernels(tabled_chain):
    chain, pos = tabled_chain
    assert all(a.type_idx >= 0 for a in chain.angles)
    assert chain.pair_lookups
    state = cc.ChainState.from_template(chain, pos.copy())
    cc.compute_chain_forces(state)
    F = state.forces.copy()
    F_ref = _reference_forces(state)
    assert np.abs(F - F_ref).max() <= 1e-11 * max(1.0, np.abs(F_ref).max())


def test_vectorized_forces_match_without_tables(tabled_chain):
    chain, pos = tabled_chain
    plain = cc.ChainCommon(
        name="p",
        atoms=chain.atoms,
        bonds=chain.bonds,
        angles=[cc.ChainAngle(a.a, a.b, a.c, 1.9, 8.0) for a in chain.angles],
        torsions=[
            cc.ChainTorsion(t.a, t.b, t.c, t.d, 0.4, 1.5, 3) for t in chain.torsions
        ],
    )
    state = cc.ChainState.from_template(plain, pos.copy())
    cc.compute_chain_forces(state)
    F = state.forces.copy()
    F_ref = _reference_forces(state)
    assert np.abs(F - F_ref).max() <= 1e-11 * max(1.0, np.abs(F_ref).max())


def test_chain_json_keeps_the_coffdrop_tables(tabled_chain, tmp_path):
    chain, pos = tabled_chain
    path = save_chain_to_json(chain, pos, tmp_path / "chain.json")
    loaded, body = load_chain_from_json(path)
    assert loaded.coffdrop_params is not None
    assert loaded.pair_lookups == chain.pair_lookups
    assert [a.type_idx for a in loaded.angles] == [a.type_idx for a in chain.angles]
    assert [t.type_idx for t in loaded.torsions] == [t.type_idx for t in chain.torsions]
    s0 = cc.ChainState.from_template(chain, pos.copy())
    s1 = cc.ChainState.from_template(loaded, body.copy())
    cc.compute_chain_forces(s0)
    cc.compute_chain_forces(s1)
    np.testing.assert_allclose(s1.forces, s0.forces, rtol=1e-12, atol=1e-12)


def test_chain_json_refuses_changed_coffdrop_files(tabled_chain, tmp_path):
    chain, pos = tabled_chain
    path = save_chain_to_json(chain, pos, tmp_path / "chain.json")
    data = json.loads(path.read_text())
    data["coffdrop"]["files"]["ff_xml"]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256"):
        load_chain_from_json(path)


def test_chain_json_without_tables_loads_harmonic_terms(tabled_chain, tmp_path):
    chain, pos = tabled_chain
    path = save_chain_to_json(chain, pos, tmp_path / "chain.json")
    data = json.loads(path.read_text())
    del data["coffdrop"]
    path.write_text(json.dumps(data))
    loaded, _ = load_chain_from_json(path)
    assert loaded.coffdrop_params is None and not loaded.pair_lookups


def _three_beads(sep13):
    """Beads 0-1-2 joined by two bonds, beads 0 and 2 sep13 apart."""
    atoms = [cc.ChainAtom(radius=2.0, charge=0.0, resname="ALA:CA") for _ in range(3)]
    bonds = [
        cc.ChainBond(cc.ChainAtomRef(0), cc.ChainAtomRef(1), 3.8, 10.0),
        cc.ChainBond(cc.ChainAtomRef(1), cc.ChainAtomRef(2), 3.8, 10.0),
    ]
    common = cc.ChainCommon(name="t", atoms=atoms, bonds=bonds)
    h = np.sqrt(3.8**2 - (sep13 / 2) ** 2)
    pos = np.array([[-sep13 / 2, 0, 0], [0, h, 0], [sep13 / 2, 0, 0]])
    return common, pos - pos.mean(axis=0)


def test_wca_leaves_out_the_one_three_pair():
    common, pos = _three_beads(3.5)
    state = cc.ChainState.from_template(common, pos)
    F = cc.chain_intra_nonbonded_forces(state, common, eps=1.0)
    assert np.all(F == 0.0)


def test_wca_leaves_out_the_pairs_with_a_table(tabled_chain):
    chain, pos = tabled_chain
    state = cc.ChainState.from_template(chain, pos.copy())
    assert np.all(cc.chain_intra_nonbonded_forces(state, chain) == 0.0)


def test_elastic_network_holds_the_reference(tabled_chain):
    chain, pos = tabled_chain
    springs = cc.elastic_network(chain, pos, 9.0, 2.0)
    ca = {i for i, a in enumerate(chain.atoms) if a.resname.endswith(":CA")}
    skip = cc.bonded_exclusions(chain)
    assert springs
    for s in springs:
        i, j = s.a.atom_idx, s.b.atom_idx
        assert i in ca and j in ca and not skip[i, j]
        assert s.r0 == pytest.approx(np.linalg.norm(pos[i] - pos[j])) and s.r0 < 9.0
    net = cc.ChainCommon(name="n", atoms=chain.atoms, network=springs)
    state = cc.ChainState.from_template(net, pos.copy())
    cc.compute_chain_forces(state)
    assert np.abs(state.forces).max() < 1e-9
    i = springs[0].a.atom_idx
    state.positions[i] += np.array([0.3, 0.0, 0.0])
    cc.compute_chain_forces(state)
    assert state.forces[i, 0] < 0.0


def _small_sim(target_radius=1.0, r_start=20.0, **extra):
    common, pos = _three_beads(6.0)
    target = Molecule(
        name="t",
        atoms=[Atom(index=0, name="X", x=0.0, y=0.0, z=0.0, radius=target_radius)],
    )
    crit = ReactionCriteria(name="r", pairs=[ContactPair(0, 1, 3.0)], n_needed=1)
    kwargs = dict(
        n_trajectories=1,
        dt=0.2,
        dt_rxn=0.05,
        dt_chain=0.05,
        chain_steps_per_outer=4,
        max_steps=20,
        r_start=r_start,
        r_escape=1e6,
        seed=3,
        use_lmz=False,
        remove_internal_rotation=True,
    )
    kwargs.update(extra)
    return ChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos,
        params=ChainBDParameters(**kwargs),
        pathway_set=PathwaySet([ReactionInterface(name="r", criteria=crit)]),
        D_trans=0.02,
        D_rot=0.001,
    )


def test_inner_steps_cover_the_outer_step(monkeypatch):
    calls = []
    real = chain_simulator.chain_internal_bd_step

    def spy(state, **kw):
        calls.append(kw["dt"])
        return real(state, **kw)

    monkeypatch.setattr(chain_simulator, "chain_internal_bd_step", spy)
    sim = _small_sim(max_steps=5)
    sim.run_one()
    assert len(calls) == 5 * 4
    calls.clear()
    monkeypatch.setattr(sim, "_outer_dt", lambda d: sim.params.dt_rxn)
    res = sim.run_one()
    assert len(calls) == 5 * 1 and res.time_ps == pytest.approx(5 * 0.05)


def test_inner_steps_feel_the_target_field_in_the_body_frame(monkeypatch):
    seen = []
    real = chain_simulator.chain_internal_bd_step

    def spy(state, **kw):
        seen.append(kw["external_forces"])
        return real(state, **kw)

    monkeypatch.setattr(chain_simulator, "chain_internal_bd_step", spy)
    sim = _small_sim(max_steps=1, inner_field=True)
    field = np.array([[0.3, -0.2, 0.1], [0.0, 0.5, 0.0], [-0.1, 0.0, 0.2]])
    monkeypatch.setattr(sim, "_compute_per_atom_external_forces", lambda w: field)
    # The first draws of the stream place the chain, so a copy of the stream
    # gives the start orientation.
    _, ori = chain_simulator.initialize_bsphere(trajectory_rngs(3, 0)[0], 20.0)
    sim.run_one(*trajectory_rngs(3, 0))
    R = ori.to_rotation_matrix()
    np.testing.assert_allclose(seen[0], field @ R, atol=1e-12)
    np.testing.assert_allclose(R @ seen[0][0], field[0], atol=1e-12)


def test_inner_field_needs_the_rotation_removal():
    with pytest.raises(ValueError, match="inner_field"):
        _small_sim(inner_field=True, remove_internal_rotation=False)


def test_parser_checks_the_inner_clock(tmp_path):
    chain = ChainConfig(chain_json="c.json", dt_chain=0.025, chain_steps_per_outer=4)
    cfg = PySTARCConfig(receptor_pqr="t.pqr", chain=chain, dt=0.2)
    with pytest.raises(ValueError, match="chain_steps_per_outer"):
        cfg.validate()


def test_overlap_test_covers_only_the_target():
    target = Molecule(name="t", atoms=[Atom(x=0.0, y=0.0, z=0.0, radius=2.0)])
    radii = np.array([1.5, 1.5])
    close_pair = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    assert not _check_chain_overlap(target, close_pair, radii)
    assert _check_chain_overlap(target, close_pair - [7.0, 0.0, 0.0], radii)
    assert not _check_chain_overlap(None, close_pair, radii)


def test_hard_sphere_rejects_every_step_inside_the_target():
    sim = _small_sim(target_radius=40.0, max_steps=6, use_hard_sphere=True)
    res = sim.run_one(*trajectory_rngs(1, 0))
    assert res.fate == Fate.MAX_STEPS
    assert res.n_hard_sphere_rejections == 6
    assert res.n_hard_sphere_redraws == 6 * 3
    assert res.final_separation == pytest.approx(20.0)


def test_hard_sphere_accepts_steps_away_from_the_target():
    sim = _small_sim(max_steps=6, use_hard_sphere=True)
    res = sim.run_one(*trajectory_rngs(1, 0))
    assert res.n_hard_sphere_rejections == 0 and res.n_hard_sphere_redraws == 0


def test_the_last_pose_is_tested():
    sim = _small_sim(max_steps=0, r_start=1.0)
    res = sim.run_one(*trajectory_rngs(1, 0))
    assert res.fate == Fate.REACTED and res.steps == 0


def test_trajectory_streams_are_distinct():
    firsts = set()
    for seed in (1523, 1524):
        for i in range(400):
            main, bridge = trajectory_rngs(seed, i)
            firsts.add(main.random())
            firsts.add(bridge.random())
    assert len(firsts) == 1600


def test_serial_and_parallel_runs_agree():
    a = _small_sim(n_trajectories=4, max_steps=30, use_hard_sphere=True)
    b = _small_sim(n_trajectories=4, max_steps=30, use_hard_sphere=True, n_threads=2)
    ra, rb = a.run(), b.run()
    assert [(r.fate, r.steps, r.time_ps) for r in ra] == [
        (r.fate, r.steps, r.time_ps) for r in rb
    ]


def test_nearby_target_atoms_give_the_same_steric_force():
    rng = np.random.default_rng(2)
    atoms = [
        Atom(x=x, y=y, z=z, radius=r)
        for (x, y, z), r in zip(
            rng.uniform(-15, 15, (400, 3)), rng.uniform(1.0, 2.0, 400)
        )
    ]
    sim = _small_sim()
    sim.target = Molecule(name="t", atoms=atoms)
    sim._refresh_geometry()
    full = chain_simulator.target_spheres(sim.target)
    for shift in (8.0, 14.0, 40.0):
        world = sim.body_positions + np.array([shift, 1.0, -2.0])
        near = sim._nearby_spheres(world)
        F_full = chain_simulator.chain_target_steric_forces(
            world, sim._chain_radii, full
        )
        if near is None:
            assert np.all(F_full == 0.0)
            continue
        assert len(near[1]) < len(full[1])
        F_near = chain_simulator.chain_target_steric_forces(
            world, sim._chain_radii, near
        )
        np.testing.assert_array_equal(F_near, F_full)
        assert _check_chain_overlap(
            near, world, sim._chain_radii
        ) == _check_chain_overlap(full, world, sim._chain_radii)


def test_chain_live_progress_logs_counts_and_rate(capsys):
    sim = _small_sim(n_trajectories=2, max_steps=0, r_start=1.0, verbose=True)
    sim._outer_prop = type("_Prop", (), {"k_b": 10.0})()
    sim.run()
    out = capsys.readouterr().out
    assert "[T]" in out
    assert "done=2/2 (100.0%)" in out
    assert "reacted=2" in out
    assert "escaped=0" in out
    assert "P_rxn=1.00000" in out
    assert "k_on=6.02e9" in out
    assert "max_steps=0" in out
