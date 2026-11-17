"""
Deterministic parity tests comparing the GPU flexible-chain force kernels,
generalized Born solver, fused internal integrator, Kabsch rotation removal,
and target grid and cell-list evaluators against the CPU reference routines.
"""

from __future__ import annotations

from pystarc.simulation.chain_simulator import (
    _check_chain_overlap,
    chain_internal_bd_step,
    chain_target_steric_forces,
    evaluate_born_force_on_chain,
    evaluate_target_grid_force_on_chain,
)
from pystarc.simulation.coffdrop_chain import (
    ChainAngle,
    ChainAtom,
    ChainAtomRef,
    ChainBond,
    ChainCommon,
    ChainState,
    ChainTorsion,
    CoplanarConstraint,
    LengthConstraint,
    bonded_exclusions,
    chain_from_sequence,
    compute_chain_forces,
    compute_constraint_violations,
    elastic_network,
    place_relaxed_geometry,
)
from pystarc.forces.gpu_chain_forces import (
    ChainInternalPlan,
    GPUChainTargetField,
    chain_full_gb_force_batch,
)
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.structures.molecules import Atom, Molecule
from pystarc.forces.chain_gb import chain_full_gb_force
import numpy as np
import pytest

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


def _make_synthetic_target(seed: int = 7) -> Molecule:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-8.0, 8.0, size=(48, 3))
    coords -= coords.mean(axis=0)
    charges = rng.uniform(-0.8, 0.8, size=48)
    radii = rng.uniform(1.3, 2.1, size=48)
    radii[0] = 0.0
    atoms = [
        Atom(
            index=i,
            name="GHO" if i == 0 else "CA",
            residue_name="ALA",
            residue_index=i + 1,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
            charge=0.0 if i == 0 else float(charges[i]),
            radius=float(radii[i]),
        )
        for i in range(48)
    ]
    return Molecule(name="synthetic_target", atoms=atoms)


def _make_synthetic_dx(origin, spacing: float, n: int, seed: int) -> DXGrid:
    rng = np.random.default_rng(seed)
    xs = origin[0] + spacing * np.arange(n)
    ys = origin[1] + spacing * np.arange(n)
    zs = origin[2] + spacing * np.arange(n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z + 9.0)
    data = (2.5 * np.exp(-r / 7.8) / r) + 0.02 * rng.standard_normal((n, n, n))
    return DXGrid(np.asarray(origin, dtype=float), np.diag([spacing] * 3), data)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_coffdrop_internal_forces_match_cpu(xp):
    common = chain_from_sequence(
        "ARKDEQGHILFPSTWYV", sidechains=True, caps=("ACE", "NME")
    )
    pos0 = place_relaxed_geometry(common)
    common.network = elastic_network(common, pos0, cutoff=9.0, k_spring=2.5)
    rng = np.random.default_rng(101)
    batch = pos0[None, :, :] + rng.normal(0.0, 0.35, size=(24, common.n_atoms, 3))
    batch -= batch.mean(axis=1, keepdims=True)

    for soft_rep in (False, True):
        plan = ChainInternalPlan(
            common, dt_chain=0.025, soft_repulsion=soft_rep, soft_repulsion_eps=1.3, xp=xp
        )
        f_got = _to_np(plan.evaluate_forces_batch(xp.asarray(batch)))
        st = ChainState.from_template(common, pos0.copy())
        for b in range(batch.shape[0]):
            st.positions[:] = batch[b]
            compute_chain_forces(
                st, soft_repulsion=soft_rep, soft_repulsion_eps=1.3
            )
            np.testing.assert_allclose(f_got[b], st.forces, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_harmonic_and_cosine_fallback_forces_match_cpu(xp):
    n = 10
    rng = np.random.default_rng(202)
    atoms = [
        ChainAtom(
            radius=0.0 if i == n - 1 else float(1.5 + 0.2 * (i % 3)),
            charge=float((-1) ** i * 0.4),
            resname="ALA:CA",
            resid=i,
        )
        for i in range(n)
    ]
    bonds = [
        ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), r0=3.8, k_spring=85.0)
        for i in range(n - 1)
    ]
    angles = [
        ChainAngle(
            ChainAtomRef(i), ChainAtomRef(i + 1), ChainAtomRef(i + 2),
            theta0=1.95, k_angle=14.0, type_idx=-1,
        )
        for i in range(n - 2)
    ]
    torsions = [
        ChainTorsion(
            ChainAtomRef(i), ChainAtomRef(i + 1), ChainAtomRef(i + 2), ChainAtomRef(i + 3),
            phi0=0.4 * i, k_tor=2.5, n=(i % 3) + 1, type_idx=-1,
        )
        for i in range(n - 3)
    ]
    common = ChainCommon(
        name="harmonic_chain", atoms=atoms, bonds=bonds, angles=angles, torsions=torsions
    )
    batch = rng.uniform(-3.0, 3.0, size=(16, n, 3))
    # Include a near-planar quadruplet to exercise the |cos(phi)| -> 1 regime.
    batch[0, :4] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [3.8, 1.0, 1e-5]])
    batch -= batch.mean(axis=1, keepdims=True)

    plan = ChainInternalPlan(
        common, dt_chain=0.05, soft_repulsion=True, soft_repulsion_eps=1.7, xp=xp
    )
    f_got = _to_np(plan.evaluate_forces_batch(xp.asarray(batch)))
    st = ChainState.from_template(common, batch[0].copy())
    for b in range(batch.shape[0]):
        st.positions[:] = batch[b]
        compute_chain_forces(st, soft_repulsion=True, soft_repulsion_eps=1.7)
        np.testing.assert_allclose(f_got[b], st.forces, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("coffdrop_active", [True, False])
def test_gb_obc2_batch_matches_cpu(xp, coffdrop_active):
    common = chain_from_sequence("RKDEQHIL", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    rng = np.random.default_rng(303)
    batch = pos0[None, :, :] + rng.normal(0.0, 0.4, size=(9, common.n_atoms, 3))
    charges = np.array([a.charge for a in common.atoms], dtype=float)
    radii = np.array([a.radius for a in common.atoms], dtype=float)
    excl = bonded_exclusions(common)
    obc_kw = {"obc_alpha": 1.0, "obc_beta": 0.8, "obc_gamma": 4.85}

    f_batch, e_batch = chain_full_gb_force_batch(
        xp.asarray(batch),
        xp.asarray(charges),
        xp.asarray(radii),
        eps_in=1.5,
        eps_out=78.0,
        coffdrop_active=coffdrop_active,
        exclude_pair_mask=excl,
        obc_kwargs=obc_kw,
        xp=xp,
        chunk_size=4,
    )
    f_got = _to_np(f_batch)
    e_got = _to_np(e_batch)
    for b in range(batch.shape[0]):
        f_ref, e_ref = chain_full_gb_force(
            batch[b],
            charges,
            radii,
            eps_in=1.5,
            eps_out=78.0,
            coffdrop_active=coffdrop_active,
            exclude_pair_mask=excl,
            obc_kwargs=obc_kw,
        )
        np.testing.assert_allclose(f_got[b], f_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(e_got[b], e_ref["total"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("remove_rot", [True, False])
@pytest.mark.parametrize("force_global", [False, True])
def test_fused_inner_steps_and_kabsch_match_cpu(xp, remove_rot, force_global):
    common = chain_from_sequence("ARNDCEQGHILK", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    common.network = elastic_network(common, pos0, cutoff=8.5, k_spring=2.0)
    rng = np.random.default_rng(404)
    B = 8
    n_in = 4
    batch = pos0[None, :, :] + rng.normal(0.0, 0.25, size=(B, common.n_atoms, 3))
    batch -= batch.mean(axis=1, keepdims=True)
    ext_f = rng.normal(0.0, 1.5, size=(B, common.n_atoms, 3))
    # Trigger the 5.0 A inner drift clamp on trajectory 0, bead 0.
    ext_f[0, 0] = np.array([2.5e4, -1.5e4, 1.0e4])
    normals = rng.standard_normal((B, n_in, common.n_atoms, 3))

    plan = ChainInternalPlan(
        common, dt_chain=0.025, soft_repulsion=True, soft_repulsion_eps=1.0, xp=xp
    )
    pos_dev = xp.asarray(batch.copy())
    plan.step_internal_batch(
        pos_dev,
        xp.asarray(ext_f),
        xp.arange(B, dtype=xp.int64),
        xp.zeros(B, dtype=xp.int64),
        xp.full(B, n_in, dtype=xp.int32),
        rng_key=np.array([11, 22], dtype=np.uint32),
        stream=5,
        remove_rigid_rotation=remove_rot,
        supplied_normals=xp.asarray(normals),
        force_global_kernel=force_global,
    )
    got = _to_np(pos_dev)

    class _SeqRNG:
        def __init__(self, z_seq):
            self._z = z_seq
            self._i = 0

        def standard_normal(self, shape):
            out = self._z[self._i]
            self._i += 1
            return out.reshape(shape)

    for b in range(B):
        st = ChainState.from_template(common, batch[b].copy())
        seq_rng = _SeqRNG(normals[b])
        for _ in range(n_in):
            chain_internal_bd_step(
                st,
                dt=0.025,
                rng=seq_rng,
                soft_repulsion=True,
                soft_repulsion_eps=1.0,
                remove_rigid_rotation=remove_rot,
                external_forces=ext_f[b],
            )
        np.testing.assert_allclose(got[b], st.positions, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_inner_step_constraints_and_philox_slot_invariance(xp):
    atoms = [ChainAtom(radius=1.8, charge=0.0, resname="ALA:CA", resid=i) for i in range(6)]
    common = ChainCommon(
        name="constrained",
        atoms=atoms,
        bonds=[ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), r0=3.8, k_spring=40.0) for i in range(5)],
        length_constraints=[
            LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), length=3.8),
            LengthConstraint(ChainAtomRef(2), ChainAtomRef(3), length=3.8),
        ],
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(4), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    pos0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [5.8, 3.2, 0.0],
            [9.6, 3.2, 0.0],
            [6.5, 1.5, 0.0],
            [11.0, 5.0, 1.2],
        ],
        dtype=float,
    )
    pos0 -= pos0.mean(axis=0)
    rng = np.random.default_rng(505)
    B = 6
    normals = rng.standard_normal((B, 2, 6, 3))
    plan = ChainInternalPlan(common, dt_chain=0.02, xp=xp)
    pos_dev = xp.asarray(np.tile(pos0, (B, 1, 1)))
    plan.step_internal_batch(
        pos_dev,
        None,
        xp.arange(B, dtype=xp.int64),
        xp.zeros(B, dtype=xp.int64),
        xp.full(B, 2, dtype=xp.int32),
        rng_key=np.array([7, 9], dtype=np.uint32),
        stream=3,
        apply_constraints=True,
        constraint_tol=1e-7,
        remove_rigid_rotation=True,
        supplied_normals=xp.asarray(normals),
    )
    got = _to_np(pos_dev)
    for b in range(B):
        st = ChainState.from_template(common, got[b].copy())
        viol = compute_constraint_violations(st)
        assert np.max(np.abs(viol)) < 1e-6

    if xp is not np:
        # Verify Philox counter invariance under slot reordering and subsetting.
        key = np.random.SeedSequence(999).generate_state(2)
        p_full = xp.asarray(np.tile(pos0, (12, 1, 1)))
        ids_full = xp.arange(100, 112, dtype=xp.int64)
        steps_full = xp.arange(12, dtype=xp.int64) * 4
        plan.step_internal_batch(
            p_full, None, ids_full, steps_full,
            xp.full(12, 3, dtype=xp.int32), key, stream=2, remove_rigid_rotation=True,
        )
        sub_order = np.array([9, 2, 11, 0, 5], dtype=np.int64)
        p_sub = xp.asarray(np.tile(pos0, (len(sub_order), 1, 1)))
        plan.step_internal_batch(
            p_sub, None, xp.asarray(100 + sub_order), xp.asarray(sub_order * 4),
            xp.full(len(sub_order), 3, dtype=xp.int32), key, stream=2, remove_rigid_rotation=True,
        )
        np.testing.assert_allclose(_to_np(p_sub), _to_np(p_full)[sub_order], rtol=0, atol=0)


@pytest.mark.parametrize("xp", _BACKENDS, ids=lambda m: m.__name__)
def test_target_field_grids_wca_and_overlap_match_cpu(xp):
    target = _make_synthetic_target(seed=17)
    t_pos = np.array([[a.x, a.y, a.z] for a in target.atoms], dtype=float)
    t_chg = np.array([a.charge for a in target.atoms], dtype=float)

    elec_fine = _make_synthetic_dx([-15.0, -15.0, -15.0], 1.5, 21, seed=21)
    elec_coarse = _make_synthetic_dx([-30.0, -30.0, -30.0], 3.0, 21, seed=22)
    born_grid = _make_synthetic_dx([-15.0, -15.0, -15.0], 1.5, 21, seed=23)

    common = chain_from_sequence("RKDEAG", sidechains=True)
    pos0 = place_relaxed_geometry(common)
    charges = np.array([a.charge for a in common.atoms], dtype=float)
    radii = np.array([a.radius for a in common.atoms], dtype=float)
    radii[-1] = 0.0  # ghost chain bead

    rng = np.random.default_rng(606)
    B = 32
    shifts = rng.uniform(-18.0, 18.0, size=(B, 1, 3))
    world_pos = pos0[None, :, :] + shifts
    # Place one bead right near a grid face to exercise clamped central differences.
    world_pos[0, 0] = np.array([-14.8, 0.2, -0.3])

    field_single = GPUChainTargetField(
        target=target,
        chain_charges=charges,
        chain_radii=radii,
        elec_grids=[elec_fine],
        born_grids=[born_grid],
        desolvation_alpha=0.85,
        xp=xp,
    )
    f_ext, e_el, e_bo = field_single.evaluate_external_forces(
        xp.asarray(world_pos),
        soft_repulsion=True,
        soft_repulsion_eps=1.4,
        want_energy=True,
    )
    f_got = _to_np(f_ext)
    e_el_got = _to_np(e_el)
    e_bo_got = _to_np(e_bo)
    ov_got = _to_np(field_single.check_overlap(xp.asarray(world_pos)))

    for b in range(B):
        fe_ref, ee_ref = evaluate_target_grid_force_on_chain(world_pos[b], charges, elec_fine)
        fb_ref, eb_ref = evaluate_born_force_on_chain(world_pos[b], charges, born_grid, alpha=0.85)
        fw_ref = chain_target_steric_forces(world_pos[b], radii, target, eps=1.4)
        np.testing.assert_allclose(f_got[b], fe_ref + fb_ref + fw_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(e_el_got[b], ee_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(e_bo_got[b], eb_ref, rtol=1e-12, atol=1e-12)
        assert bool(ov_got[b]) is _check_chain_overlap(target, world_pos[b], radii)

    # Multi-grid + multipole Yukawa far-field consistency between CPU and GPU.
    mp = MultipoleExpansion(t_pos, t_chg, debye_length=7.858, sdie=78.0)
    far_pos = pos0[None, :, :] + rng.uniform(-45.0, 45.0, size=(16, 1, 3))
    field_cpu = GPUChainTargetField(
        target=target,
        chain_charges=charges,
        chain_radii=radii,
        elec_grids=[elec_fine, elec_coarse],
        born_grids=[born_grid],
        use_far_field=True,
        target_charge=target.total_charge(),
        debye_length=7.858,
        sdie=78.0,
        multipole_expansion=mp,
        xp=np,
    )
    field_xp = GPUChainTargetField(
        target=target,
        chain_charges=charges,
        chain_radii=radii,
        elec_grids=[elec_fine, elec_coarse],
        born_grids=[born_grid],
        use_far_field=True,
        target_charge=target.total_charge(),
        debye_length=7.858,
        sdie=78.0,
        multipole_expansion=mp,
        xp=xp,
    )
    f_c, ee_c, eb_c = field_cpu.evaluate_external_forces(far_pos, want_energy=True)
    f_x, ee_x, eb_x = field_xp.evaluate_external_forces(xp.asarray(far_pos), want_energy=True)
    np.testing.assert_allclose(_to_np(f_x), f_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_to_np(ee_x), ee_c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_to_np(eb_x), eb_c, rtol=1e-12, atol=1e-12)
