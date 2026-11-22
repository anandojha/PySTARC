"""Regression tests of the input preparation, the forces, the time step, the
outputs and the figures. The GPU energy tests need a GPU."""

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

import pystarc.pipeline.make_pqr as make_pqr
import pystarc.simulation.chain_simulator as chain_simulator
import pystarc.simulation.coffdrop_chain as cc
import pystarc.simulation.nam_simulator as nsim
from pystarc.analysis.plots import AVOGADRO_CONV, RunFigures, plot_run
from pystarc.forces.chain_gb import chain_full_gb_force
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.forces.engine import PySTARCEngine
from pystarc.forces.lj import (
    HydrophobicParams,
    LJAtomType,
    LJForceEngine,
    LJParams,
    hydrophobic_sasa_force,
    lj_pair_force,
)
from pystarc.forces.multipole_farfield import MultipoleExpansion, compute_multipole_gpu
from pystarc.global_defs.constants import (
    AVOGADRO,
    COULOMB_KCAL,
    KB_KCAL,
    KCAL_PER_MOL_TO_KBT,
    MOLAR_RATE_CONV,
    VACUUM_PERMITTIVITY_KBT,
)
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.motion.adaptive_time_step import AdaptiveTimeStep
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.pathways.reaction_interface import (
    ContactPair,
    PathwaySet,
    ReactionCriteria,
    ReactionInterface,
)
from pystarc.pipeline.build_inputs import read_pqr as build_read_pqr
from pystarc.pipeline.chain_output_writer import write_milestone_flux_csv
from pystarc.pipeline.desolvation_grid import read_pqr_geometry
from pystarc.pipeline.extract import extract
from pystarc.pipeline.input_parser import OutputConfig, debye_from_ions
from pystarc.pipeline.output_writer import write_cpu_results
from pystarc.pipeline.pipeline import _checklist
from pystarc.pipeline.prepare_bd_surface import BDSurfaceConfig
from pystarc.pipeline.run_apbs import _run as apbs_run
from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator
from pystarc.structures.molecules import Atom, Molecule
from pystarc.structures.pqr_io import parse_pqr_records
from pystarc.transforms.quaternion import Quaternion
from pystarc.xml_io.simulation_io import parse_simulation_xml, write_simulation_xml

REPO = Path(__file__).resolve().parents[1]


def _gpu():
    try:
        import cupy

        cupy.zeros(1).sum()
        return cupy
    except Exception:
        return None


CP = _gpu()
needs_gpu = pytest.mark.skipif(CP is None, reason="needs a GPU")


# Input preparation


@pytest.mark.skipif(os.name == "nt", reason="stub binaries are shell scripts")
def test_ambpdb_output_reaches_the_combined_pqr(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ambpdb").write_text(
        '#!/bin/sh\nfor a in "$@"; do [ "$a" = ">" ] && exit 3; done\n'
        'echo "ATOM      1  N   ALA     1       1.000   2.000   3.000  0.1000 1.5500"\n'
        "echo END\n"
    )
    (bin_dir / "cpptraj").write_text("#!/bin/sh\necho dummy > complex.rst\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    out = make_pqr.make_combined_pqr(
        tmp_path / "c.prmtop", tmp_path / "c.pdb", tmp_path
    )
    assert out == tmp_path / "complex.pqr"
    assert out.read_text().startswith("ATOM      1  N   ALA     1")


_COMPLEX_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2 Na+  Na+     2      10.000   0.000   0.000  1.00  0.00          Na\n"
    "ATOM      3 Cl-  Cl-     3      20.000   0.000   0.000  1.00  0.00          Cl\n"
    "HETATM    4  C1  LIG     4       5.000   0.000   0.000  1.00  0.00           C\n"
)


def test_amber_counter_ions_stay_out_of_the_receptor(tmp_path, monkeypatch):
    pdb = tmp_path / "c.pdb"
    pdb.write_text(_COMPLEX_PDB)
    rec, lig = extract(pdb, "LIG", tmp_path / "x")
    assert [l[17:20] for l in rec.read_text().splitlines() if l.startswith("ATOM")] == [
        "ALA"
    ]
    pqr = tmp_path / "complex.pqr"
    pqr.write_text(
        "ATOM      1  CA  ALA     1       0.000   0.000   0.000  0.1000 1.9080\n"
        "ATOM      2 Na+  Na+     2      10.000   0.000   0.000  1.0000 1.8680\n"
        "ATOM      3 Cl-  Cl-     3      20.000   0.000   0.000 -1.0000 2.4700\n"
        "ATOM      4  C1  LIG     4       5.000   0.000   0.000 -0.1000 1.9080\n"
    )
    rec_pqr, _ = make_pqr.split_pqr(pqr, "LIG", tmp_path)
    assert [
        l[17:20] for l in rec_pqr.read_text().splitlines() if l.startswith("ATOM")
    ] == ["ALA"]
    monkeypatch.setattr(make_pqr, "_check_tool", lambda name: None)
    monkeypatch.setattr(make_pqr, "_run", lambda *a, **k: None)
    make_pqr.build_complex(
        pdb,
        tmp_path / "l.mol2",
        tmp_path / "l.frcmod",
        tmp_path / "l.lib",
        "LIG",
        tmp_path,
    )
    kept = [
        l[17:20] for l in (tmp_path / "complex_nowater.pdb").read_text().splitlines()
    ]
    assert kept == ["ALA", "LIG"]


# Forces


def test_multipole_force_is_minus_the_gradient_of_the_potential():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(2, 8))
        mp = MultipoleExpansion(
            rng.normal(0, 4, (n, 3)),
            rng.normal(0, 1, n),
            debye_length=rng.uniform(4, 15),
        )
        r = rng.normal(size=3)
        r *= rng.uniform(12, 40) / np.linalg.norm(r)
        h = 1e-5
        ref = -np.array(
            [
                (mp.potential(r + e) - mp.potential(r - e)) / (2 * h)
                for e in np.eye(3) * h
            ]
        )
        f = compute_multipole_gpu(
            None,
            None,
            r[None],
            mp.Q,
            mp.dipole,
            mp.quadrupole,
            mp.debye,
            mp.four_pi_eps,
            mp.trace_moment,
        )[0]
        assert np.linalg.norm(f - ref) <= 1e-6 * np.linalg.norm(ref)


def test_dipole_normal_to_the_separation_still_pulls():
    mp = MultipoleExpansion(
        np.array([[2.0, 0, 0], [-2.0, 0, 0]]), np.array([1.0, -1.0]), debye_length=8.0
    )
    f = compute_multipole_gpu(
        None,
        None,
        np.array([[0.0, 15.0, 0.0]]),
        mp.Q,
        mp.dipole,
        mp.quadrupole,
        mp.debye,
        mp.four_pi_eps,
    )[0]
    assert np.allclose(f, mp.force(np.array([0.0, 15.0, 0.0])), rtol=1e-6, atol=1e-12)
    assert f[0] < 0.0


def _chain(n, charge=1.0, radius=1.5):
    atoms = [
        cc.ChainAtom(radius=radius, charge=charge, resname="ALA:CA") for _ in range(n)
    ]
    bonds = [
        cc.ChainBond(cc.ChainAtomRef(i), cc.ChainAtomRef(i + 1), 3.8, 10.0)
        for i in range(n - 1)
    ]
    return cc.ChainCommon(name="t", atoms=atoms, bonds=bonds)


def _chain_sim(common, pos, outputs=None, **extra):
    target = Molecule(
        name="t", atoms=[Atom(index=0, name="X", x=0.0, y=0.0, z=0.0, radius=1.0)]
    )
    crit = ReactionCriteria(name="r", pairs=[ContactPair(0, 1, 3.0)], n_needed=1)
    kw = dict(
        n_trajectories=1,
        dt=0.2,
        dt_rxn=0.05,
        dt_chain=0.05,
        chain_steps_per_outer=4,
        max_steps=20,
        r_start=20.0,
        r_escape=1e6,
        seed=3,
        use_lmz=False,
        remove_internal_rotation=True,
    )
    kw.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return chain_simulator.ChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=pos - pos.mean(axis=0),
            params=chain_simulator.ChainBDParameters(**kw),
            pathway_set=PathwaySet([ReactionInterface(name="r", criteria=crit)]),
            D_trans=0.02,
            D_rot=0.001,
            outputs=outputs,
        )


def test_chain_gb_leaves_bonded_pairs_out_of_vacuum_coulomb(monkeypatch):
    common = _chain(3)
    pos = np.array([[0.0, 0, 0], [3.8, 0, 0], [6.0, 3.1, 0]])
    sim = _chain_sim(common, pos, use_self_born=True, coffdrop_active=False)
    seen = {}
    real = chain_simulator.chain_full_gb_force

    def spy(*a, **k):
        seen["mask"] = k["exclude_pair_mask"]
        return real(*a, **k)

    monkeypatch.setattr(chain_simulator, "chain_full_gb_force", spy)
    world = pos + np.array([30.0, 0.0, 0.0])
    f = sim._compute_per_atom_external_forces(world)
    assert np.array_equal(seen["mask"], cc.bonded_exclusions(common))
    p = sim.params
    kw = dict(
        eps_in=p.gb_eps_in,
        eps_out=p.gb_eps_out,
        coffdrop_active=False,
        obc_kwargs=dict(
            obc_alpha=p.gb_obc_alpha, obc_beta=p.gb_obc_beta, obc_gamma=p.gb_obc_gamma
        ),
    )
    q, rad = np.ones(3), np.full(3, 1.5)
    f_mask, _ = chain_full_gb_force(
        world, q, rad, exclude_pair_mask=cc.bonded_exclusions(common), **kw
    )
    f_open, _ = chain_full_gb_force(world, q, rad, **kw)
    assert np.allclose(f, f_mask) and not np.allclose(f, f_open)


def test_hydrophobic_energy_is_the_integral_of_its_force():
    hp = HydrophobicParams()
    u = np.array([1.0, 0.0, 0.0])
    args = (1.5, 0.9, 20.0, 12.0, hp)
    E = lambda r: hydrophobic_sasa_force(r, u, *args)[1]
    kinks = [k - R for R in (1.5, 0.9) for k in (hp.a, hp.b)]
    h = 1e-5
    for r in np.linspace(0.5, 3.5, 61):
        if min(abs(r - k) for k in kinks) < 1e-3:
            continue
        f = hydrophobic_sasa_force(r, u, *args)[0][0]
        assert f == pytest.approx((E(r + h) - E(r - h)) / (2 * h), abs=1e-8)
    for k in kinks:
        assert E(k - 1e-9) == pytest.approx(E(k + 1e-9), abs=1e-7)
    assert E(4.0) == 0.0 and E(0.5) < E(2.5) < 0.0


def test_lj_torque_on_the_ligand_is_the_sum_of_the_pair_torques():
    eng = PySTARCEngine(lj_params=LJParams([LJAtomType("C", 1.0, 2.0)]))
    m1 = Molecule(name="r", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0)])
    m2 = Molecule(
        name="l",
        atoms=[
            Atom(index=0, x=3.5, y=1.0, z=0.0, charge=1.0),
            Atom(index=1, x=3.5, y=-5.0, z=0.0, charge=-1.0),
        ],
    )
    force, torque, _ = eng(m1, m2)
    c2 = m2.positions_array().mean(axis=0)
    F, T = np.zeros(3), np.zeros(3)
    for p in m2.positions_array():
        f_on_1, _ = lj_pair_force(np.zeros(3), p, 1.0, 4.0)
        F -= f_on_1
        T += np.cross(p - c2, -f_on_1)
    assert np.allclose(force, F * KCAL_PER_MOL_TO_KBT)
    assert np.allclose(torque, T * KCAL_PER_MOL_TO_KBT) and abs(torque[2]) > 1.0


def test_cpu_born2_transforms_receptor_into_ligand_frame():
    n = 33
    origin = np.full(3, -0.5 * (n - 1) * 0.5)
    ax = origin[0] + np.arange(n) * 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    a, c = np.array([0.3, -0.2, 0.1]), 2.0
    grid = DXGrid(origin, np.eye(3) * 0.5, a[0] * x + a[1] * y + a[2] * z + c)
    rec_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-0.8, 0.3, 0.6]])
    rec_q = np.array([0.5, -1.0, 0.25])
    mol1 = Molecule(
        name="r",
        atoms=[
            Atom(index=i, x=float(p[0]), y=float(p[1]), z=float(p[2]), charge=float(q))
            for i, (p, q) in enumerate(zip(rec_xyz, rec_q))
        ],
    )
    th = 0.7
    R = np.array(
        [
            [math.cos(th), -math.sin(th), 0.0],
            [math.sin(th), math.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cen = np.array([1.2, -0.8, 0.4])
    mol2 = Molecule(
        name="l",
        atoms=[
            Atom(index=0, x=float(cen[0]), y=float(cen[1]), z=float(cen[2]), charge=0.0)
        ],
    )
    mol2._rotation_matrix = R
    eng = PySTARCEngine(born_mol2=[grid], desolvation_alpha=0.7)
    eng.backend = "numpy"
    force, torque, energy = eng(mol1, mol2)
    yl = (rec_xyz - cen) @ R
    f_rec_lig = -0.7 * (rec_q[:, None] ** 2) * a[None, :]
    f_ref = -(R @ f_rec_lig.sum(axis=0))
    t_ref = -(R @ np.cross(yl, f_rec_lig).sum(axis=0))
    e_ref = float(0.7 * np.sum(rec_q**2 * (yl @ a + c)))
    assert np.allclose(force, f_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(torque, t_ref, rtol=1e-10, atol=1e-12)
    assert energy == pytest.approx(e_ref, rel=1e-10)


def test_lj_engine_matches_the_pair_functions():
    rng = np.random.default_rng(3)
    p1, p2 = rng.uniform(-4, 4, (6, 3)), rng.uniform(-4, 4, (5, 3)) + [3.0, 0, 0]
    t1, t2 = rng.integers(0, 3, 6), rng.integers(0, 3, 5)
    rad1, rad2 = rng.uniform(1.0, 2.0, 6), rng.uniform(1.0, 2.0, 5)
    s1, s2 = rng.uniform(0, 30, 6), rng.uniform(0, 30, 5)
    hp = HydrophobicParams(a=2.0, b=6.0)
    for wca in (False, True):
        lj = LJParams(
            [
                LJAtomType("a", 0.2, 1.6),
                LJAtomType("b", 0.1, 1.9),
                LJAtomType("c", 0.3, 1.2),
            ],
            use_wca=wca,
        )
        f1, f2, t, e = LJForceEngine(lj, hp).compute_with_torque(
            p1, p2, t1, t2, rad1, rad2, s1, s2, ref2=[1.0, 2.0, 0.5]
        )
        F1, T, E = np.zeros(3), np.zeros(3), 0.0
        for i in range(6):
            for j in range(5):
                d = p2[j] - p1[i]
                r = np.linalg.norm(d)
                fa, ea = lj_pair_force(
                    p1[i],
                    p2[j],
                    math.sqrt(lj.epsilon(t1[i]) * lj.epsilon(t2[j])),
                    lj.sigma(t1[i]) + lj.sigma(t2[j]),
                    use_wca=wca,
                )
                fh, eh = hydrophobic_sasa_force(
                    r, d / r, rad1[i], rad2[j], s1[i], s2[j], hp
                )
                F1 += fa + fh
                T += np.cross(p2[j] - [1.0, 2.0, 0.5], -(fa + fh))
                E += ea + eh
        assert (
            np.allclose(f1, F1)
            and np.allclose(f2, -F1)
            and np.allclose(t, T)
            and e == pytest.approx(E)
        )


# Time step and the CPU simulator


def test_reaction_step_follows_the_closest_reaction_pair():
    kw = dict(
        D_rel=0.02,
        D_rot=0.0005,
        r_hydro1=20.0,
        r_hydro2=15.0,
        dt_min=0.05,
        dt_rxn_min=0.0125,
    )
    near = AdaptiveTimeStep().get_dt(
        r=35.5, rxn_distances=[5.0], min_pair_dist=6.0, **kw
    )
    far = AdaptiveTimeStep().get_dt(
        r=35.5, rxn_distances=[5.0], min_pair_dist=20.0, **kw
    )
    assert near == pytest.approx(0.0125) and far > 1.0
    assert AdaptiveTimeStep().get_dt(
        r=35.5, rxn_distances=[5.0], **kw
    ) == pytest.approx(0.0125)
    assert AdaptiveTimeStep().get_dt(r=60.0, rxn_distances=[5.0], **kw) > 1.0


def _nam_sim(force_fn=None, reactions=True, **extra):
    mol1 = Molecule(name="r", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0, radius=2.0)])
    mol2 = Molecule(name="l", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0, radius=2.0)])
    crit = ReactionCriteria(name="r", pairs=[ContactPair(0, 0, 1.0)], n_needed=1)
    ps = (
        PathwaySet([ReactionInterface(name="r", criteria=crit)])
        if reactions
        else PathwaySet()
    )
    kw = dict(n_trajectories=1, r_start=12.0, r_escape=40.0, max_steps=20, seed=5)
    kw.update(extra)
    args = (
        mol1,
        mol2,
        MobilityTensor.from_radii(2.0, 2.0),
        ps,
        nsim.NAMParameters(**kw),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            nsim.NAMSimulator(*args, force_fn) if force_fn else nsim.NAMSimulator(*args)
        )


def _spy_bridge(monkeypatch):
    calls = []
    real = nsim.check_reaction_with_bridge

    def spy(*a, **k):
        calls.append(a[-2])
        return real(*a, **k)

    monkeypatch.setattr(nsim, "check_reaction_with_bridge", spy)
    return calls


def test_rejected_step_leaves_no_bridge(monkeypatch):
    sim = _nam_sim()
    calls = _spy_bridge(monkeypatch)
    monkeypatch.setattr(
        nsim, "_check_hard_sphere_overlap", lambda m1, m2, scale=1.0: True
    )
    assert sim.run_one().fate == Fate.MAX_STEPS
    assert calls == []


def test_return_to_b_starts_the_step_afresh(monkeypatch):
    out = lambda m1, m2: (
        5e3 * m2.atoms[0].position / np.linalg.norm(m2.atoms[0].position),
        np.zeros(3),
        0.0,
    )
    sim = _nam_sim(force_fn=out, use_hard_sphere=False)
    assert sim._outer_prop is not None
    returns = []

    def new_state(pos, ori, rng):
        returns.append(1)
        if len(returns) == 1:
            return True, 12.0 * pos / np.linalg.norm(pos), ori
        return False, pos, ori

    monkeypatch.setattr(sim._outer_prop, "new_state", new_state)
    resets = []
    real_reset = sim._dt_ctrl.reset
    monkeypatch.setattr(sim._dt_ctrl, "reset", lambda: (resets.append(1), real_reset()))
    calls = _spy_bridge(monkeypatch)
    assert sim.run_one().fate == Fate.ESCAPED
    assert len(returns) == 2 and len(calls) == 2 and len(resets) == 2


@pytest.mark.parametrize(
    "core, core_rxn, want", [(0.37, 0.09, (0.37, 0.09)), (0.0, 0.0, (0.05, 0.0125))]
)
def test_core_floors_reach_the_time_step(monkeypatch, core, core_rxn, want):
    sim = _nam_sim(
        reactions=False,
        minimum_core_dt=core,
        minimum_core_reaction_dt=core_rxn,
        max_steps=3,
    )
    sim._outer_prop = None
    seen = []
    real = AdaptiveTimeStep.get_dt

    def spy(self, *a, **k):
        seen.append((k["dt_min"], k["dt_rxn_min"]))
        return real(self, *a, **k)

    monkeypatch.setattr(AdaptiveTimeStep, "get_dt", spy)
    sim.run_one()
    assert seen and all(s == pytest.approx(want) for s in seen)


def test_cpu_overlap_uses_half_the_radius_sum():
    m1 = Molecule(name="a", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0, radius=2.0)])
    m2 = Molecule(name="b", atoms=[Atom(index=0, x=3.0, y=0.0, z=0.0, radius=2.0)])
    assert nsim.NAMParameters().overlap_scale == 0.5
    assert nsim._check_hard_sphere_overlap(m1, m2)
    assert not nsim._check_hard_sphere_overlap(m1, m2, 0.5)
    m2.atoms[0].x = 1.9
    assert nsim._check_hard_sphere_overlap(m1, m2, 0.5)


# CPU outputs and the run log


def test_cpu_run_writes_results_and_trajectories(tmp_path):
    res = nsim.SimulationResult(
        n_trajectories=4,
        n_reacted=1,
        n_escaped=2,
        n_max_steps=1,
        reaction_counts={"r": 1},
        r_start=20.0,
        r_escape=40.0,
        dt=0.2,
        k_db=3.0,
    )
    trajs = [
        TrajectoryResult(Fate.REACTED, 10, 5.0, 8.0, "r"),
        TrajectoryResult(Fate.ESCAPED, 20, 9.0, 44.0),
        TrajectoryResult(Fate.ESCAPED, 30, 12.0, 44.0),
        TrajectoryResult(Fate.MAX_STEPS, 40, 15.0, 30.0),
    ]
    write_cpu_results(
        tmp_path,
        res,
        trajs,
        OutputConfig(),
        D_rel=0.05,
        wall_time_sec=2.0,
        total_steps=100,
    )
    data = json.loads((tmp_path / "results.json").read_text())
    assert (
        data["P_rxn"] == 0.25 and data["k_b"] == 3.0 and data["P_rxn_bound_high"] == 0.5
    )
    assert (
        data["k_on"] == pytest.approx(MOLAR_RATE_CONV * 3.0 * 0.25)
        and data["steps_per_sec"] == 50.0
    )
    rows = (tmp_path / "trajectories.csv").read_text().splitlines()
    assert rows[0].startswith("traj_id,outcome,n_steps,total_time_ps")
    assert rows[1].startswith("0,reacted,10,5.0000") and rows[4].startswith(
        "3,max_steps,40"
    )
    figs = RunFigures(tmp_path)
    assert figs.k_b == 3.0 and len(figs._trajectories()["outcome"]) == 4


def test_cpu_checklist_reports_the_overlap_and_wall_that_ran():
    cfg = SimpleNamespace(
        desolvation_alpha=0.0,
        r_hydro_rec=0.0,
        r_hydro_lig=0.0,
        temperature=298.15,
        minimum_core_dt=0.0,
    )
    geom = SimpleNamespace(
        receptor=SimpleNamespace(hydrodynamic_r=20.0),
        ligand=SimpleNamespace(hydrodynamic_r=3.0),
    )
    eng = SimpleNamespace(_born1=[], _born2=[], _lj_engine=None)
    text = lambda hs: "\n".join(
        _checklist(
            cfg,
            SimpleNamespace(
                hydrodynamic_interactions=False,
                use_hard_sphere=hs,
                use_brownian_bridge=True,
            ),
            geom,
            0.0,
            None,
            eng,
            None,
        )
    )
    assert "Overlap check (overlapping moves are rejected)" in text(True)
    assert "Overlap check disabled" in text(False) and "LJ forces disabled" in text(
        False
    )


# Chain forces and outputs


def test_fine_grid_edge_hands_over_to_the_coarse_grid():
    def grid(spacing, n, slope):
        origin = np.full(3, -0.5 * (n - 1) * spacing)
        ax = origin[0] + np.arange(n) * spacing
        x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
        return DXGrid(origin, np.eye(3) * spacing, slope * x)

    grids = [grid(0.5, 17, 1.0), grid(1.0, 33, 3.0)]
    state = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), forces=np.zeros((2, 3))
    )
    cc.compute_target_grid_forces(state, grids, np.array([1.0, 1.0]))
    assert state.forces[0, 0] == pytest.approx(-1.0) and state.forces[
        1, 0
    ] == pytest.approx(-3.0)


def _beads(points, radius=2.0):
    return [
        cc.ChainBead(
            pos=np.array(p, float), force=np.zeros(3), radius=radius, charge=0.0
        )
        for p in points
    ]


def test_excluded_volume_reaches_neighbours_and_ends_at_the_wca_minimum():
    ev = cc.ChainForceEvaluator()._excluded_volume_forces
    F = ev(
        cc.FlexibleChain(beads=_beads([[0, 0, 0], [2.5, 0, 0], [0, 30, 0]])), kT=0.592
    )
    assert np.linalg.norm(F[1]) > 100.0
    f = lambda y: np.linalg.norm(
        ev(cc.FlexibleChain(beads=_beads([[0, 0, 0], [0, y, 0]])), kT=0.592)[1]
    )
    assert f(3.999) == pytest.approx(f(4.001), rel=0.05) and f(4.001) > 1.0
    assert f(2 ** (1 / 6) * 4.0 + 1e-6) == 0.0
    bonds = [
        cc.ChainBond(cc.ChainAtomRef(0), cc.ChainAtomRef(1), 3.8, 10.0),
        cc.ChainBond(cc.ChainAtomRef(1), cc.ChainAtomRef(2), 3.8, 10.0),
    ]
    bent = cc.FlexibleChain(
        beads=_beads([[0, 0, 0], [3.0, 2.0, 0], [3.0, 0, 0]]), bonds=bonds
    )
    assert not ev(bent, kT=0.592).any()


def test_milestone_flux_leaves_out_the_return_to_b(tmp_path):
    tr = TrajectoryResult(Fate.ESCAPED, 3, 1.0, 50.0)
    tr.radial_trace = np.array([40.0, 43.0, 41.0])
    tr.radial_returns = np.array([0, 0, 1])
    rows = lambda b: (
        write_milestone_flux_csv(tmp_path, [tr], n_shells=2, b_radius=b)
        .read_text()
        .splitlines()[1:]
    )
    assert rows(40.0) == ["41.0000,2,0", "42.0000,1,0"]
    assert rows(None) == ["41.0000,1,0", "42.0000,1,1"]


def test_chain_run_records_returns_with_the_radial_trace(monkeypatch):
    common = _chain(3, charge=0.0, radius=2.0)
    pos = np.array([[-3.0, 0, 0], [0, np.sqrt(3.8**2 - 9.0), 0], [3.0, 0, 0]])
    sim = _chain_sim(
        common,
        pos,
        outputs=OutputConfig(save_interval=1),
        use_lmz=True,
        r_escape=40.0,
        max_steps=6,
    )
    calls = []

    def new_state(p, ori, rng):
        calls.append(1)
        return (
            (True, p / np.linalg.norm(p) * 20.0, ori)
            if len(calls) == 1
            else (False, p, ori)
        )

    sim._outer_prop = SimpleNamespace(new_state=new_state)
    monkeypatch.setattr(
        chain_simulator,
        "initialize_bsphere",
        lambda rng, r: (np.array([1.2 * r, 0.0, 0.0]), Quaternion(1.0, 0.0, 0.0, 0.0)),
    )
    res = sim.run_one(rng=np.random.default_rng(3), rng_bb=np.random.default_rng(4))
    assert len(res.radial_returns) == len(res.radial_trace)
    assert (
        list(res.radial_returns[:2]) == [0, 1]
        and res.radial_returns.sum() == len(calls) == 1
    )


# Rates, pooling and constants


def test_k_b_carries_the_screened_trace_of_the_receptor():
    debye, sdie, b, D0, trace = 7.828, 78.0, 10.0, 0.03, 150.0
    s = SimpleNamespace(
        params=SimpleNamespace(
            r_start=b, r_escape=20.0, dielectric=sdie, _kT_scale=1.0
        ),
        _qb_factor=1.1,
        mol1=SimpleNamespace(total_charge=lambda: 0.0),
        mol2=SimpleNamespace(total_charge=lambda: 1.0),
        engine=SimpleNamespace(_debye=debye, _multipole=object(), _mp_trace=trace),
        mob=SimpleNamespace(
            relative_translational_diffusion=lambda: D0, radius1=5.0, radius2=5.0
        ),
        _use_hi=False,
    )
    GPUBatchSimulator._compute_return_prob(s)
    fpe = 4 * math.pi * sdie * VACUUM_PERMITTIVITY_KBT
    U = lambda r: trace / (6 * debye**2) * math.exp(-r / debye) / (fpe * r)
    f = lambda x: 1.0 / D0 if x == 0.0 else math.exp(U(1.0 / x)) / D0
    kb = 4 * math.pi / quad(f, 0.0, 1.0 / b, epsabs=0, epsrel=1e-12)[0]
    assert not s._kb_pure_diffusion and s._k_b == pytest.approx(kb, rel=1e-7)
    assert s._k_b < 4 * math.pi * D0 * b


def test_shard_pooling_counts_step_capped_trajectories(tmp_path):
    shutil.copy(REPO / "examples" / "barnase_barstar" / "combine_shards.py", tmp_path)
    for i in range(2):
        d = tmp_path / "shards" / f"shard_{i}" / "bd_sims"
        d.mkdir(parents=True)
        (d / "results.json").write_text(
            json.dumps(
                {
                    "rate": {
                        "n_reacted": 10,
                        "n_escaped": 60,
                        "n_censored": 30,
                        "k_b": 1000.0,
                        "b_radius": 60.0,
                    },
                    "summary": {"wall_time_sec": 3600},
                    "params": {"r_start": 60.0},
                }
            )
        )
    out = subprocess.run(
        [sys.executable, str(tmp_path / "combine_shards.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (
        "P_rxn            0.10000" in out
        and "P_rxn bound      0.10000 to 0.40000" in out
    )


def test_one_set_of_physical_constants():
    assert (
        MOLAR_RATE_CONV == pytest.approx(AVOGADRO * 1e-15, rel=1e-15)
        and AVOGADRO_CONV == MOLAR_RATE_CONV
    )
    assert COULOMB_KCAL == pytest.approx(332.0637, abs=1e-4)
    assert KB_KCAL == pytest.approx(1.987204259e-3, rel=1e-9)
    res = nsim.SimulationResult(
        n_trajectories=10,
        n_reacted=2,
        n_escaped=8,
        n_max_steps=0,
        reaction_counts={},
        r_start=20.0,
        r_escape=40.0,
        dt=0.2,
        k_db=3.0,
    )
    assert res.rate_constant(0.05) == pytest.approx(
        MOLAR_RATE_CONV * 3.0 * 0.2, rel=1e-15
    )


def test_debye_length_follows_the_solvent_dielectric():
    cfg = BDSurfaceConfig()
    cfg.debye_length, cfg.ion_concentration, cfg.temperature = 0.0, 0.15, 298.15
    lam = {}
    for sdie in (78.0, 80.0):
        cfg.sdie = sdie
        lam[sdie] = cfg.compute_debye_length()
        assert lam[sdie] == debye_from_ions(0.15, sdie, 298.15)
    assert lam[80.0] / lam[78.0] == pytest.approx(math.sqrt(80.0 / 78.0), rel=1e-12)


# Files


def test_simulation_xml_round_trip_keeps_paths_and_seed_zero(tmp_path):
    p = tmp_path / "in.xml"
    p.write_text(
        "<simulation><seed>0</seed><molecule1_pqr>rec.pqr</molecule1_pqr>"
        "<molecule2_pqr>lig.pqr</molecule2_pqr></simulation>"
    )
    write_simulation_xml(parse_simulation_xml(p), tmp_path / "out.xml")
    cfg = parse_simulation_xml(tmp_path / "out.xml")
    assert (cfg["seed"], cfg["mol1_pqr"], cfg["mol2_pqr"]) == (0, "rec.pqr", "lig.pqr")
    write_simulation_xml({"seed": None, "n_trajectories": 5}, tmp_path / "none.xml")
    assert parse_simulation_xml(tmp_path / "none.xml")["seed"] is None


_PQR_VARIANTS = (
    "ATOM      1  N   ALA     1      11.104   6.134  -6.504 -0.3000 1.8240\n"
    "ATOM      2  CA  ALA A   1      12.560   6.000  -6.900  0.0337 1.9080\n"
    "ATOM      3 CB   THR     6     12.5400 -8.8180 -10.1790 0.3654 1.7000\n"
    "ATOM      4  OG1 THR     6    -112.540-108.818 -10.179 -0.6761 1.7210 O\n"
)


def test_pqr_readers_take_chain_free_spacing_and_touching_fields(tmp_path):
    p = tmp_path / "m.pqr"
    p.write_text(_PQR_VARIANTS)
    xyz, rad = read_pqr_geometry(p)
    assert np.allclose(rad, [1.824, 1.908, 1.7, 1.721])
    assert np.allclose(xyz[:, 2], [-6.504, -6.9, -10.179, -10.179]) and np.allclose(
        xyz[3, :2], [-112.54, -108.818]
    )
    assert [r.charge for r in parse_pqr_records(p)] == [-0.3, 0.0337, 0.3654, -0.6761]
    atoms = build_read_pqr(p)
    assert [a["q"] for a in atoms] == [-0.3, 0.0337, 0.3654, -0.6761] and atoms[3][
        "el"
    ] == "O"


def test_apbs_command_runs_without_a_shell(tmp_path):
    apbs_run(
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'w').write('x')",
            "a b.txt",
        ],
        tmp_path,
        "t",
    )
    assert (tmp_path / "a b.txt").read_text() == "x"
    apbs_run(f"{shlex.quote(sys.executable)} -c pass ; touch hacked", tmp_path, "t")
    assert not (tmp_path / "hacked").exists()


# Figures


def _receptor(d):
    (d / "receptor.pqr").write_text(
        "ATOM      1  CA  ALA     1       0.000   0.000   0.000  0.1000 1.9000\nEND\n"
    )


@pytest.mark.parametrize(
    "rows", ["0,10,0,0,1,0,0,0\n", "0,10,0,0,1,0,0,0\n1,-10,0,0,1,0,0,0\n"]
)
def test_first_hitting_draws_one_hit_and_hits_averaging_to_zero(tmp_path, rows):
    import matplotlib.pyplot as plt

    _receptor(tmp_path)
    (tmp_path / "encounters.csv").write_text("traj_id,x,y,z,q0,q1,q2,q3\n" + rows)
    fig, ax = plt.subplots()
    assert RunFigures(tmp_path).draw_first_hitting(ax) is None
    plt.close(fig)


def test_figures_read_a_chain_run(tmp_path):
    _receptor(tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "summary": {},
                "rate": {"k_b": 100.0, "P_rxn": 0.5, "k_on": 3e10, "b_radius": 40.0},
                "params": {"r_start": 40.0, "r_escape": 80.0},
            }
        )
    )
    (tmp_path / "trajectories.csv").write_text(
        "traj_id,fate,steps,time_ps,final_separation,reaction_name,energy_at_reaction\n"
        "0,REACTED,120,50.0,10.0,r,0.0\n1,ESCAPED,300,80.0,90.0,,0.0\n"
        "2,REACTED,200,70.0,11.0,r,0.0\n3,ESCAPED,250,60.0,90.0,,0.0\n"
    )
    (tmp_path / "encounters.csv").write_text(
        "traj_id,step,time_ps,x,y,z,q_w,q_x,q_y,q_z,reaction_name\n"
        "0,120,50.0,10,1,0,1,0,0,0,r\n2,200,70.0,9,-1,1,1,0,0,0,r\n"
    )
    figs = RunFigures(tmp_path)
    assert (figs.k_b, figs.r_start, figs.r_escape) == (100.0, 40.0, 80.0)
    status = plot_run(
        tmp_path,
        tmp_path / "fig",
        dpi=40,
        which=("rate", "first_hitting", "bd_time", "steps"),
    )
    assert set(status.values()) == {"ok"}


# GPU energies


@needs_gpu
def test_gpu_wca_energy_is_the_pair_sum():
    from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine

    g = np.random.default_rng(21)
    rec, rec_r = g.uniform(-6.0, 6.0, (40, 3)), g.uniform(1.2, 1.9, 40)
    lig_r = g.uniform(1.2, 1.9, 4)
    cen = g.normal(size=(30, 3))
    cen *= g.uniform(5.0, 10.0, (30, 1)) / np.linalg.norm(cen, axis=1, keepdims=True)
    lig = cen[:, None] + g.normal(size=(30, 4, 3))
    eng = GPUBatchForceEngine(
        [],
        [],
        alpha=0.0,
        rec_positions=rec,
        rec_radii=rec_r,
        lig_radii=lig_r,
        use_lj=True,
    )
    _, _, E = eng._wca_forces_gpu(CP.asarray(lig), CP.asarray(cen))
    d = lig[:, :, None, :] - rec[None, None]
    r = np.linalg.norm(d, axis=3)
    s = lig_r[None, :, None] + rec_r[None, None, :]
    sr6 = (s / r) ** 6
    e = np.where(r < 2 ** (1 / 6) * s, 4 * 0.17 * (sr6 * sr6 - sr6) + 0.17, 0.0).sum(
        axis=(1, 2)
    )
    assert (e > 0).sum() > 5 and np.allclose(CP.asnumpy(E), e, rtol=1e-10, atol=1e-12)


@needs_gpu
def test_gpu_born2_energy_and_call_total():
    import pystarc.forces.gpu_batch_engine as gbe

    n = 33
    origin = np.full(3, -0.5 * (n - 1) * 0.5)
    ax = origin[0] + np.arange(n) * 0.5
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    a, c = np.array([0.3, -0.2, 0.1]), 2.0
    grid = DXGrid(origin, np.eye(3) * 0.5, a[0] * x + a[1] * y + a[2] * z + c)
    rec = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [-0.8, 0.3, 0.6]])
    q = np.array([0.5, -1.0, 0.25])
    th = 0.7
    R = np.array(
        [
            [
                [math.cos(th), -math.sin(th), 0.0],
                [math.sin(th), math.cos(th), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    cen = np.array([[0.3, -0.2, 0.1]])
    yl = (rec - cen[0]) @ R[0]
    e_ref = 0.7 * np.sum(q**2 * (yl @ a + c))
    eng = gbe.GPUBatchForceEngine(
        [],
        [],
        alpha=0.7,
        lig_born_grids=[grid],
        rec_positions=rec,
        rec_charges=q,
        rec_radii=np.full(3, 1.5),
        lig_radii=np.array([1.6]),
        use_lj=True,
    )
    _, _, e2 = eng._eval_born_reverse(CP.asarray(R), CP.asarray(cen), 1)
    assert float(e2[0]) == pytest.approx(e_ref, rel=1e-10)
    lig = CP.asarray(cen[:, None, :])
    _, _, e = eng(
        lig, CP.asarray([0.0]), R_matrices=CP.asarray(R), centroids=CP.asarray(cen)
    )
    _, _, e_wca = eng._wca_forces_gpu(lig, CP.asarray(cen))
    assert (
        float(e[0]) == pytest.approx(e_ref + float(e_wca[0]), rel=1e-10)
        and float(e_wca[0]) > 0
    )
