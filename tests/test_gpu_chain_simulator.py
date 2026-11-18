"""
Parity, invariance, and ensemble rate tests for GPUChainBDSimulator, batched
RPY chain diffusion tensors, rigid-body two-body integration, and outer
propagation.
"""

from __future__ import annotations

from pystarc.simulation.gpu_chain_simulator import (
    GPUChainBDSimulator,
    _ChainStepRNG,
    chain_diffusion_tensors_batch,
    forces_to_body_batch,
    outer_propagate_batch,
    place_chain_batch,
)
from pystarc.simulation.coffdrop_chain import (
    ChainAtom,
    ChainCommon,
    chain_from_sequence,
    elastic_network,
    place_relaxed_geometry,
)
from pystarc.simulation.chain_simulator import (
    RELATIVE_DIFFUSION_FLOOR,
    ChainBDParameters,
    ChainBDSimulator,
    place_chain,
)
from pystarc.structures.molecules import (
    Atom,
    ContactPair,
    Molecule,
    ReactionCriteria,
)
from pystarc.pathways.reaction_interface import PathwaySet, ReactionInterface
from pystarc.simulation.outer_propagator import OPGroupInfo, OuterPropagator
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.hydrodynamics.rotne_prager import chain_diffusion_tensors
from pystarc.pipeline.chain_output_writer import write_chain_results
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.pipeline.input_parser import OutputConfig
from pystarc.global_defs.defaults import VISCOSITY
from pystarc.global_defs.constants import KBT_KCAL
from pystarc.molsystem.system_state import Fate
import numpy as np
import warnings
import pytest
import math

try:
    import cupy as cp

    cp.arange(2).sum()
    _BACKENDS = [np, cp]
except Exception:
    cp = None
    _BACKENDS = [np]


def _to_np(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _make_target(seed: int = 11, charge_bias: float = -0.35) -> Molecule:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-6.0, 6.0, size=(24, 3))
    coords -= coords.mean(axis=0)
    coords[0] = np.array([5.5, 0.0, 0.0])
    radii = rng.uniform(1.5, 2.1, size=24)
    charges = rng.uniform(-0.4, 0.4, size=24) + charge_bias
    atoms = [
        Atom(
            index=i,
            name="CA",
            residue_name="ASP" if charges[i] < 0 else "LYS",
            residue_index=i + 1,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
            charge=float(charges[i]),
            radius=float(radii[i]),
        )
        for i in range(24)
    ]
    return Molecule(name="synthetic_receptor", atoms=atoms)


def _make_grid(origin, spacing: float, n: int, scale: float, seed: int) -> DXGrid:
    rng = np.random.default_rng(seed)
    xs = origin[0] + spacing * np.arange(n)
    ys = origin[1] + spacing * np.arange(n)
    zs = origin[2] + spacing * np.arange(n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z + 16.0)
    data = scale * np.exp(-r / 7.858) / r + 0.01 * rng.standard_normal((n, n, n))
    return DXGrid(np.asarray(origin, dtype=float), np.diag([spacing] * 3), data)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_chain_diffusion_tensors_batch_matches_cpu(xp):
    rng = np.random.default_rng(19)
    # Single-bead Stokes rotational fallback.
    pos1 = np.array([[[1.5, -2.0, 0.5], [0.0, 0.0, 0.0]]]).reshape(2, 1, 3)
    rad1 = np.array([2.4])
    Dt1, Dr1, hc1, reg1, fail1 = chain_diffusion_tensors_batch(pos1, rad1, xp=xp)
    assert not reg1.any() and not fail1.any()
    for b in range(2):
        dt_ref, dr_ref, hc_ref = chain_diffusion_tensors(pos1[b], rad1)
        np.testing.assert_allclose(_to_np(Dt1)[b], dt_ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_to_np(Dr1)[b], dr_ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_to_np(hc1)[b], hc_ref, rtol=1e-12, atol=1e-14)

    # Multi-bead batch including far, partial-overlap, engulfed, coincident, and collinear cases.
    n = 16
    batch = np.cumsum(rng.normal(0.0, 3.2, size=(6, n, 3)), axis=1)
    batch -= batch.mean(axis=1, keepdims=True)
    radii = rng.uniform(1.6, 4.5, size=n)
    radii[2] = 5.0
    radii[3] = 1.1
    batch[1, 3] = batch[1, 2] + np.array([0.4, 0.0, 0.0])
    # Slice 3 has two coincident equal-radius beads -> regularized Cholesky.
    radii[5] = radii[4]
    batch[3, 5] = batch[3, 4].copy()
    # Slice 4 is strictly collinear along a rotated 3D axis -> singular C.
    u_dir = np.array([1.0, -2.0, 0.5]) / math.sqrt(5.25)
    t_line = np.linspace(-15.0, 15.0, n)[:, None]
    batch[4] = t_line * u_dir[None, :]

    for center_arg in (None, np.zeros(3)):
        Dt_b, Dr_b, hc_b, reg_m, fail_m = chain_diffusion_tensors_batch(
            xp.asarray(batch), radii, center=center_arg, xp=xp, chunk_size=3
        )
        Dt_np = _to_np(Dt_b)
        Dr_np = _to_np(Dr_b)
        hc_np = _to_np(hc_b)
        assert bool(reg_m[3]) is True
        assert bool(fail_m[4]) is True
        for b in (0, 1, 2, 3, 5):
            assert bool(fail_m[b]) is False
            info = {}
            dt_ref, dr_ref, hc_ref = chain_diffusion_tensors(
                batch[b],
                radii,
                center=center_arg,
                warn_regularized=False,
                info=info,
            )
            assert bool(reg_m[b]) is bool(info["regularized"])
            rtol = 1e-8 if info["regularized"] else 1e-11
            np.testing.assert_allclose(
                Dt_np[b], dt_ref, rtol=rtol, atol=rtol * np.max(np.abs(dt_ref))
            )
            np.testing.assert_allclose(
                Dr_np[b], dr_ref, rtol=rtol, atol=rtol * np.max(np.abs(dr_ref))
            )
            np.testing.assert_allclose(hc_np[b], hc_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_place_chain_and_forces_to_body_batch_match_cpu(xp):
    rng = np.random.default_rng(23)
    B, n = 20, 14
    body = rng.normal(0.0, 4.0, size=(B, n, 3))
    body -= body.mean(axis=1, keepdims=True)
    coms = rng.normal(0.0, 25.0, size=(B, 3))
    quats = [random_quaternion(rng) for _ in range(B)]
    q_arr = np.array([q.to_array() for q in quats], dtype=np.float64)
    w_forces = rng.normal(0.0, 1.5, size=(B, n, 3))

    wp_got = _to_np(
        place_chain_batch(xp.asarray(body), xp.asarray(coms), xp.asarray(q_arr), xp=xp)
    )
    bf_got = _to_np(
        forces_to_body_batch(xp.asarray(w_forces), xp.asarray(q_arr), xp=xp)
    )
    for b in range(B):
        wp_ref = place_chain(body[b], coms[b], quats[b])
        bf_ref = w_forces[b] @ quats[b].to_rotation_matrix()
        np.testing.assert_allclose(wp_got[b], wp_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(bf_got[b], bf_ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_rigid_step_batch_matches_cpu_all_modes(xp):
    rng = np.random.default_rng(37)
    target = _make_target(seed=5)
    common = chain_from_sequence("ARKDEQG", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    n = common.n_atoms
    B = 8

    quats = [random_quaternion(rng) for _ in range(B)]
    q_arr = np.array([q.to_array() for q in quats])
    # Separations spanning far, partial-overlap, and engulfed regimes.
    radii_sep = [42.0, 25.0, 14.0, 9.5, 4.5, 2.2, 30.0, 18.0]
    dirs = rng.normal(size=(B, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    coms = dirs * np.array(radii_sep)[:, None]
    world = np.array([place_chain(pos0, coms[b], quats[b]) for b in range(B)])
    forces = rng.normal(0.0, 0.4, size=(B, n, 3))
    # Force an overshoot (> 5 A) on slice 7 to exercise the outer displacement cap.
    forces[7] = 500.0
    dt_arr = np.full(B, 0.2, dtype=np.float64)
    dW_t = math.sqrt(0.2) * rng.normal(size=(B, 3))
    dW_r = math.sqrt(0.2) * rng.normal(size=(B, 3))
    dW_r1 = math.sqrt(0.2) * rng.normal(size=(B, 3))

    configs = [
        dict(auto_diffusion=False, D_trans=0.025, D_rot=0.0015, rotate=False, a_T=None, hi=False),
        dict(auto_diffusion=True, D_trans=None, D_rot=None, rotate=False, a_T=None, hi=False),
        dict(auto_diffusion=True, D_trans=None, D_rot=None, rotate=True, a_T=None, hi=False),
        dict(auto_diffusion=True, D_trans=None, D_rot=None, rotate=True, a_T=11.5, hi=False),
        dict(auto_diffusion=True, D_trans=None, D_rot=None, rotate=True, a_T=11.5, hi=True),
    ]

    for cfg in configs:
        params = ChainBDParameters(
            rotate_body_tensors=cfg["rotate"],
            hydrodynamic_interactions=cfg["hi"],
            use_lmz=False,
        )
        cpu_sim = ChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=pos0,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            D_trans=cfg["D_trans"],
            D_rot=cfg["D_rot"],
            auto_diffusion=cfg["auto_diffusion"],
            target_hydro_radius=cfg["a_T"],
        )
        gpu_sim = GPUChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=pos0,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            D_trans=cfg["D_trans"],
            D_rot=cfg["D_rot"],
            auto_diffusion=cfg["auto_diffusion"],
            target_hydro_radius=cfg["a_T"],
            xp=xp,
        )
        Dt_batch = np.broadcast_to(gpu_sim._D_t_init_3x3[None, :, :], (B, 3, 3)).copy()
        Dr_batch = np.broadcast_to(gpu_sim._D_r_init_3x3[None, :, :], (B, 3, 3)).copy()
        np_got, no_got, fnet_got, clip_got = gpu_sim._rigid_step_batch(
            xp.asarray(coms),
            xp.asarray(q_arr),
            xp.asarray(world),
            xp.asarray(forces),
            xp.asarray(Dt_batch),
            xp.asarray(Dr_batch),
            xp.asarray(dt_arr),
            xp.asarray(dW_t),
            xp.asarray(dW_r),
            xp.asarray(dW_r1) if cfg["a_T"] is not None else None,
        )
        np_got = _to_np(np_got)
        no_got = _to_np(no_got)
        fnet_got = _to_np(fnet_got)
        assert not bool(_to_np(clip_got).any())
        np.testing.assert_allclose(fnet_got, forces.sum(axis=1), rtol=1e-12, atol=1e-12)

        for b in range(B):
            p_ref, q_ref = cpu_sim._rigid_step(
                coms[b],
                quats[b],
                world[b],
                forces[b],
                cpu_sim.D_trans,
                cpu_sim.D_rot,
                0.2,
                dW_t[b],
                dW_r[b],
                dW_r1[b] if cfg["a_T"] is not None else None,
            )
            np.testing.assert_allclose(np_got[b], p_ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                no_got[b], q_ref.to_array(), rtol=1e-12, atol=1e-12
            )
        # Slice 7 exceeded 5 A and stayed at its initial pose.
        np.testing.assert_array_equal(np_got[7], coms[7])
        np.testing.assert_array_equal(no_got[7], q_arr[7])

    # Non-positive-definite D_rel triggers eigenvalue floor clipping and one warning.
    params_clip = ChainBDParameters(hydrodynamic_interactions=False, use_lmz=False)
    sim_clip = GPUChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params_clip,
        pathway_set=PathwaySet(reactions=[]),
        auto_diffusion=True,
        target_hydro_radius=10.0,
        xp=xp,
    )
    sim_clip._D_t_target = -1.0
    Dt_b = np.broadcast_to(sim_clip._D_t_init_3x3[None, :, :], (2, 3, 3)).copy()
    Dr_b = np.broadcast_to(sim_clip._D_r_init_3x3[None, :, :], (2, 3, 3)).copy()
    with pytest.warns(RuntimeWarning, match="positive definite"):
        _, _, _, clip_m = sim_clip._rigid_step_batch(
            xp.asarray(coms[:2]),
            xp.asarray(q_arr[:2]),
            xp.asarray(world[:2]),
            xp.zeros((2, n, 3), dtype=xp.float64),
            xp.asarray(Dt_b),
            xp.asarray(Dr_b),
            xp.full(2, 0.2, dtype=xp.float64),
            xp.zeros((2, 3), dtype=xp.float64),
            xp.zeros((2, 3), dtype=xp.float64),
            xp.zeros((2, 3), dtype=xp.float64),
        )
    assert np.all(_to_np(clip_m))
    assert RELATIVE_DIFFUSION_FLOOR > 0.0


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_outer_propagate_batch_return_prob_and_rotations(xp):
    for has_hi, q0, q1 in [(False, 0.0, 0.0), (True, -4.0, 3.0)]:
        op = OuterPropagator(
            b_radius=50.0,
            max_radius=12.0,
            has_hi=has_hi,
            kT=KBT_KCAL,
            viscosity=VISCOSITY,
            dielectric=78.0,
            vacuum_perm=1.0 / (4.0 * math.pi * 332.0636),
            debye_len=7.858,
            g0=OPGroupInfo(q=q0, Dtrans=0.012, Drot=4.0e-4),
            g1=OPGroupInfo(q=q1, Dtrans=0.020, Drot=9.0e-4),
        )
        n_samp = 2500
        rng = np.random.default_rng(91)
        dirs = rng.normal(size=(n_samp, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # Start on qradius so each trajectory immediately tests return_prob, and also
        # test interior excursions from 1.1 * b_radius.
        pos_q = dirs * op.qradius
        ori_q = np.array([random_quaternion(rng).to_array() for _ in range(n_samp)])
        helper = _ChainStepRNG(seed=1234, xp=xp)
        tids = xp.arange(n_samp, dtype=xp.int64)
        stps = xp.zeros(n_samp, dtype=xp.int64)
        rb_q, p_out_q, q_out_q = outer_propagate_batch(
            op, xp.asarray(pos_q), xp.asarray(ori_q), tids, stps, helper
        )
        rb_q_np = _to_np(rb_q)
        p_out_q_np = _to_np(p_out_q)
        q_out_q_np = _to_np(q_out_q)
        emp_prob = float(rb_q_np.mean())
        se = math.sqrt(op.return_prob * (1.0 - op.return_prob) / n_samp)
        assert abs(emp_prob - op.return_prob) < 4.0 * se
        ret_norms = np.linalg.norm(p_out_q_np[rb_q_np], axis=1)
        np.testing.assert_allclose(ret_norms, op.bradius, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.linalg.norm(q_out_q_np, axis=1), 1.0, rtol=1e-12, atol=1e-12
        )

        # Interior excursion from 1.1 * b_radius covering Lamm-Schulten and middle steps.
        n_mid = 300
        pos_m = dirs[:n_mid] * (1.1 * op.bradius)
        rb_m, p_out_m, q_out_m = outer_propagate_batch(
            op,
            xp.asarray(pos_m),
            xp.asarray(ori_q[:n_mid]),
            xp.arange(n_mid, dtype=xp.int64),
            xp.ones(n_mid, dtype=xp.int64),
            helper,
        )
        rb_m_np = _to_np(rb_m)
        p_out_m_np = _to_np(p_out_m)
        q_out_m_np = _to_np(q_out_m)
        assert rb_m_np.any() and (~rb_m_np).any()
        np.testing.assert_allclose(
            np.linalg.norm(p_out_m_np[rb_m_np], axis=1),
            op.bradius,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.linalg.norm(q_out_m_np, axis=1), 1.0, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_batch_size_invariance_and_all_features(xp, tmp_path):
    target = _make_target(seed=29, charge_bias=-0.4)
    common = chain_from_sequence("AKRDEQ", sidechains=True, caps=("ACE", "NME"))
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    common.network = elastic_network(common, pos0, cutoff=8.5, k_spring=2.0)

    elec_grid = _make_grid([-25.0, -25.0, -25.0], 2.5, 21, scale=-3.0, seed=41)
    born_grid = _make_grid([-20.0, -20.0, -20.0], 2.5, 17, scale=0.4, seed=42)
    rxn = ReactionInterface(
        name="contact_rxn",
        criteria=ReactionCriteria(
            name="c0",
            pairs=[ContactPair(0, 0, 9.0), ContactPair(1, 2, 10.0)],
            n_needed=1,
        ),
    )
    pset = PathwaySet(reactions=[rxn])

    params = ChainBDParameters(
        n_trajectories=14,
        dt=0.2,
        dt_rxn=0.05,
        dt_chain=0.025,
        chain_steps_per_outer=2,
        max_steps=18,
        r_start=22.0,
        r_escape=32.0,
        use_lmz=True,
        seed=777,
        n_equilibration_steps=2,
        force_change_backstep=True,
        use_hard_sphere=True,
        use_soft_repulsion=True,
        soft_repulsion_eps=0.8,
        use_brownian_bridge=True,
        use_self_born=True,
        coffdrop_active=True,
        rotate_body_tensors=True,
        diffusion_update_interval=4,
        remove_internal_rotation=True,
        inner_field=True,
    )
    outputs = OutputConfig(save_interval=3)

    runs = {}
    for bsz in (3, 19):
        wdir = tmp_path / f"run_{xp.__name__}_{bsz}"
        sim = GPUChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=pos0,
            params=params,
            pathway_set=pset,
            target_grid=elec_grid,
            born_grid=born_grid,
            auto_diffusion=True,
            outputs=outputs,
            target_hydro_radius=9.5,
            batch_size=bsz,
            checkpoint_interval=5,
            work_dir=wdir,
            xp=xp,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runs[bsz] = (sim, sim.run())
        assert not (wdir / "chain_checkpoint.npz").exists()
        assert not (wdir / "chain_checkpoint.tmp.npz").exists()

    _, res_small = runs[3]
    sim_large, res_large = runs[19]
    assert len(res_small) == 14 and len(res_large) == 14
    for r_a, r_b in zip(res_small, res_large):
        assert r_a.fate == r_b.fate
        assert r_a.steps == r_b.steps
        assert r_a.time_ps == pytest.approx(r_b.time_ps, rel=1e-14, abs=1e-14)
        assert r_a.final_separation == pytest.approx(
            r_b.final_separation, rel=1e-11, abs=1e-11
        )
        assert r_a.reaction_name == r_b.reaction_name
        assert r_a.energy_at_reaction == pytest.approx(
            r_b.energy_at_reaction, rel=1e-11, abs=1e-11
        )
        assert r_a.n_diffusion_rebuilds == r_b.n_diffusion_rebuilds
        assert r_a.n_hard_sphere_redraws == r_b.n_hard_sphere_redraws
        assert r_a.n_hard_sphere_rejections == r_b.n_hard_sphere_rejections
        if r_a.path_com is not None:
            if xp is np:
                np.testing.assert_array_equal(r_a.path_com, r_b.path_com)
                np.testing.assert_array_equal(r_a.path_q, r_b.path_q)
                np.testing.assert_array_equal(r_a.energy_steps, r_b.energy_steps)
            else:
                np.testing.assert_allclose(
                    r_a.path_com, r_b.path_com, rtol=1e-11, atol=1e-11
                )
                np.testing.assert_allclose(
                    r_a.path_q, r_b.path_q, rtol=1e-11, atol=1e-11
                )
                np.testing.assert_allclose(
                    r_a.energy_steps, r_b.energy_steps, rtol=1e-11, atol=1e-11
                )

    out_dir = tmp_path / f"written_{xp.__name__}"
    written = dict(
        write_chain_results(
            out_dir, sim_large, res_large, wall_time_sec=0.5, outputs=outputs
        )
    )
    assert "results.json" in written
    assert "trajectories.csv" in written
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "trajectories.csv").is_file()


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_cpu_vs_gpu_ensemble_rate_parity(xp):
    target = Molecule(
        name="reactive_sphere",
        atoms=[Atom(index=0, name="CA", residue_name="ASP", residue_index=1, x=0.0, y=0.0, z=0.0, charge=-2.0, radius=3.0)],
    )
    common = ChainCommon(
        name="dimer_chain",
        atoms=[
            ChainAtom(radius=2.0, charge=1.0, resname="LYS", resid=1),
            ChainAtom(radius=2.0, charge=1.0, resname="LYS", resid=2),
            ChainAtom(radius=2.0, charge=0.0, resname="ALA", resid=3),
        ],
    )
    from pystarc.simulation.coffdrop_chain import ChainAtomRef, ChainBond

    common.bonds = [
        ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 12.0),
        ChainBond(ChainAtomRef(1), ChainAtomRef(2), 3.8, 12.0),
    ]
    pos0 = np.array([[-3.0, -1.0, 0.0], [0.0, 2.0, 0.0], [3.0, -1.0, 0.0]], dtype=float)
    pos0 -= pos0.mean(axis=0)

    elec_grid = _make_grid([-24.0, -24.0, -24.0], 3.0, 17, scale=-6.0, seed=55)
    rxn = ReactionInterface(
        name="bind",
        criteria=ReactionCriteria(
            name="c",
            pairs=[ContactPair(0, 0, 9.0), ContactPair(0, 1, 9.0)],
            n_needed=1,
        ),
    )
    pset = PathwaySet(reactions=[rxn])

    n_traj = 240
    params = ChainBDParameters(
        n_trajectories=n_traj,
        dt=0.25,
        dt_rxn=0.05,
        dt_chain=0.05,
        chain_steps_per_outer=2,
        max_steps=120,
        r_start=12.0,
        r_escape=18.0,
        use_lmz=False,
        seed=2026,
        force_change_backstep=True,
        use_hard_sphere=True,
        use_brownian_bridge=True,
        rotate_body_tensors=True,
        remove_internal_rotation=True,
    )

    cpu_sim = ChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params,
        pathway_set=pset,
        target_grid=elec_grid,
        auto_diffusion=True,
        outputs=OutputConfig(save_interval=0),
        target_hydro_radius=6.0,
    )
    gpu_sim = GPUChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params,
        pathway_set=pset,
        target_grid=elec_grid,
        auto_diffusion=True,
        outputs=OutputConfig(save_interval=0),
        target_hydro_radius=6.0,
        xp=xp,
    )
    assert gpu_sim._outer_prop.k_b == pytest.approx(cpu_sim._outer_prop.k_b, rel=1e-12)
    assert gpu_sim._outer_prop.return_prob == pytest.approx(
        cpu_sim._outer_prop.return_prob, rel=1e-12
    )

    cpu_res = cpu_sim.run()
    gpu_res = gpu_sim.run()
    p_cpu = sum(r.fate == Fate.REACTED for r in cpu_res) / n_traj
    p_gpu = sum(r.fate == Fate.REACTED for r in gpu_res) / n_traj
    assert 0.05 < p_cpu < 0.95
    p_pool = 0.5 * (p_cpu + p_gpu)
    se = math.sqrt(max(2.0 * p_pool * (1.0 - p_pool) / n_traj, 1.0e-12))
    assert abs(p_gpu - p_cpu) < 3.5 * se


@pytest.mark.parametrize("xp", _BACKENDS)
def test_gpu_chain_checkpoint_resume_matches_uninterrupted(tmp_path, xp):
    """Resuming GPUChainBDSimulator from an out-of-order chain_checkpoint.npz reproduces the uninterrupted run."""
    common = chain_from_sequence("AKRD", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    target = Molecule(
        name="t",
        atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0, charge=-1.0, radius=2.0)],
    )
    rxn = ReactionInterface(
        name="hit",
        criteria=ReactionCriteria(
            name="hit",
            pairs=[ContactPair(0, 0, 7.5)],
            n_needed=1,
        ),
    )
    params = ChainBDParameters(
        n_trajectories=10,
        dt=0.25,
        dt_rxn=0.05,
        dt_chain=0.05,
        chain_steps_per_outer=2,
        max_steps=40,
        r_start=10.0,
        r_escape=16.0,
        use_lmz=False,
        seed=77,
    )
    sim_full = GPUChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params,
        pathway_set=PathwaySet([rxn]),
        auto_diffusion=True,
        outputs=OutputConfig(save_interval=0),
        batch_size=4,
        xp=xp,
    )
    full_res = sim_full.run()
    assert len(full_res) == 10

    # Seed a checkpoint containing an out-of-order subset of finished trajectories.
    ckpt_dir = tmp_path / "ckpt_run"
    ckpt_dir.mkdir()
    done_ids = [1, 4, 7, 9]
    partial = [full_res[i] if i in done_ids else None for i in range(10)]
    sim_full.n_reacted = sum(full_res[i].fate == Fate.REACTED for i in done_ids)
    sim_full.n_escaped = sum(full_res[i].fate == Fate.ESCAPED for i in done_ids)
    sim_full.n_max_steps = sum(full_res[i].fate == Fate.MAX_STEPS for i in done_ids)
    sim_full._write_checkpoint(ckpt_dir / "chain_checkpoint.npz", partial, len(done_ids))
    assert (ckpt_dir / "chain_checkpoint.npz").is_file()

    sim_resumed = GPUChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params,
        pathway_set=PathwaySet([rxn]),
        auto_diffusion=True,
        outputs=OutputConfig(save_interval=0),
        batch_size=3,
        checkpoint_interval=2,
        work_dir=ckpt_dir,
        xp=xp,
    )
    resumed_res = sim_resumed.run()
    assert not (ckpt_dir / "chain_checkpoint.npz").exists()
    assert len(resumed_res) == 10
    for r_ref, r_got in zip(full_res, resumed_res):
        assert r_got.fate == r_ref.fate
        assert r_got.steps == r_ref.steps
        assert r_got.time_ps == pytest.approx(r_ref.time_ps, rel=1e-12, abs=1e-12)
        assert r_got.final_separation == pytest.approx(
            r_ref.final_separation, rel=1e-12, abs=1e-12
        )
        assert r_got.reaction_name == r_ref.reaction_name


@pytest.mark.parametrize("xp", _BACKENDS)
def test_gpu_chain_lmz_return_records_radial_returns(monkeypatch, xp):
    """GPUChainBDSimulator records radial_returns across LMZ returns matching ChainBDSimulator."""
    import pystarc.simulation.gpu_chain_simulator as gcs

    common = chain_from_sequence("AAA", sidechains=False)
    pos0 = place_relaxed_geometry(common)
    pos0 -= pos0.mean(axis=0)
    target = Molecule(
        name="t",
        atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0, charge=0.0, radius=1.5)],
    )
    params = ChainBDParameters(
        n_trajectories=1,
        dt=0.5,
        dt_rxn=0.1,
        dt_chain=0.05,
        chain_steps_per_outer=1,
        max_steps=4,
        r_start=20.0,
        r_escape=40.0,
        use_lmz=True,
        seed=11,
    )
    sim = GPUChainBDSimulator(
        target=target,
        chain_template=common,
        chain_init_body_positions=pos0,
        params=params,
        pathway_set=PathwaySet([]),
        D_trans=50.0,
        D_rot=0.01,
        outputs=OutputConfig(save_interval=1),
        xp=xp,
    )
    calls = []

    def fake_outer(outer_prop, pos_sub, ori_sub, tids, stps, rng_h):
        calls.append(1)
        b_len = len(tids)
        if len(calls) == 1:
            nrm = xp.maximum(xp.sqrt(xp.sum(pos_sub * pos_sub, axis=1, keepdims=True)), 1e-12)
            return xp.ones(b_len, dtype=bool), (pos_sub / nrm) * 20.0, ori_sub
        return xp.zeros(b_len, dtype=bool), pos_sub, ori_sub

    monkeypatch.setattr(gcs, "outer_propagate_batch", fake_outer)
    orig_spawn = sim._spawn_slots

    def spawn_at_outer(tids, rng_h):
        slot = orig_spawn(tids, rng_h)
        slot["pos"] = xp.asarray([[24.0, 0.0, 0.0]] * len(tids), dtype=xp.float64)
        return slot

    monkeypatch.setattr(sim, "_spawn_slots", spawn_at_outer)
    res = sim.run()
    assert len(res) == 1
    assert res[0].radial_returns is not None
    assert len(res[0].radial_returns) == len(res[0].radial_trace)
    assert int(res[0].radial_returns.sum()) == 1

