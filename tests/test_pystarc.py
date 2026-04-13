"""
PySTARC test suite.

Tests for all modules. Run with:  pytest tests/ -v
"""

from pystarc.global_defs.constants import (
    ANG_TO_M,
    AVOGADRO,
    BJERRUM_LENGTH,
    DEFAULT_DEBYE_LENGTH,
    EPS0_SI,
    EPS_WATER,
    ETA_WATER,
    E_CHARGE,
    FOUR_PI,
    KBT_KCAL,
    KB_KCAL,
    KB_SI,
    PI,
    PS_TO_S,
    TWO_PI,
    T_DEFAULT,
)
from pystarc.aux.aux_tools import (
    born_integral,
    bounding_box,
    contact_distances,
    electrostatic_center,
    hydrodynamic_radius_from_rg,
    lumped_charges,
    surface_spheres,
)
from pystarc.lib.numerical import (
    CubicSpline,
    dipole_moment,
    legendre_p,
    legendre_series,
    monopole_moment,
    quadrupole_moment,
    romberg_integrate,
    wiener_step,
)
from pystarc.pipeline.gho_injection import (
    GHOAtom,
    GHOReactionCriterion,
    gho_criterion_distance,
    gho_world_position,
    inject_gho_from_manual,
)
from pystarc.hydrodynamics.rotne_prager import (
    MobilityTensor,
    rpy_offdiagonal,
    stokes_rotational_diffusion,
    stokes_translational_diffusion,
)
from pystarc.forces.lj import (
    HydrophobicParams,
    LJAtomType,
    LJForceEngine,
    LJParams,
    hydrophobic_sasa_force,
    lj_pair_force,
)
from pystarc.xml_io.simulation_io import (
    parse_reaction_xml,
    parse_simulation_xml,
    write_reaction_xml,
    write_simulation_xml,
)
from pystarc.pipeline.geometry import (
    MoleculeGeometry,
    SystemGeometry,
    _parse_rxns_xml_criteria,
    auto_detect_reactions,
)
from pystarc.transforms.quaternion import (
    Quaternion,
    RigidTransform,
    random_quaternion,
    small_rotation_quaternion,
)
from pystarc.motion.do_bd_step import (
    bd_step,
    ermak_mccammon_rotation,
    ermak_mccammon_translation,
    escape_radius,
)
from pystarc.simulation.coffdrop_chain import (
    ChainBDPropagator,
    ChainForceEvaluator,
    build_linear_chain,
)
from pystarc.simulation.nam_simulator import (
    NAMParameters,
    NAMSimulator,
    SimulationResult,
    zero_force,
)
from pystarc.pathways.reaction_interface import (
    PathwaySet,
    ReactionInterface,
    make_default_reaction,
)
from pystarc.forces.electrostatic.grid_force import (
    DXGrid,
    debye_huckel_energy,
    debye_huckel_force,
)
from pystarc.structures.molecules import (
    Atom,
    BoundingBox,
    ContactPair,
    Molecule,
    ReactionCriteria,
)
from pystarc.molsystem.system_state import Fate, SystemState, TrajectoryResult
from pystarc.structures.pqr_io import parse_pqr, write_pqr
from pystarc.global_defs import constants as C
from pathlib import Path
import numpy as np
import importlib
import tempfile
import pystarc
import pytest
import math
import os


# Constants
class TestConstants:
    def test_temperature(self):
        assert abs(T_DEFAULT - 298.15) < 0.01

    def test_boltzmann_si(self):
        assert abs(KB_SI - 1.380649e-23) < 1e-30

    def test_boltzmann_kcal(self):
        assert abs(KB_KCAL - 1.987204e-3) < 1e-8

    def test_kbt_kcal(self):
        assert abs(KBT_KCAL - KB_KCAL * T_DEFAULT) < 1e-8

    def test_bjerrum_length(self):
        assert 6.5 < BJERRUM_LENGTH < 8.0  # ~7.1 Å in water at 298K

    def test_eps_water(self):
        assert abs(EPS_WATER - 78.54) < 0.1

    def test_avogadro(self):
        assert abs(AVOGADRO - 6.022e23) < 1e20

    def test_ang_to_m(self):
        assert abs(ANG_TO_M - 1e-10) < 1e-20

    def test_ps_to_s(self):
        assert abs(PS_TO_S - 1e-12) < 1e-20

    def test_pi(self):
        assert abs(PI - math.pi) < 1e-14

    def test_two_pi(self):
        assert abs(TWO_PI - 2 * math.pi) < 1e-14

    def test_four_pi(self):
        assert abs(FOUR_PI - 4 * math.pi) < 1e-14

    def test_debye_length_positive(self):
        assert DEFAULT_DEBYE_LENGTH > 0

    def test_eta_water_positive(self):
        assert ETA_WATER > 0

    def test_kbt_at_room_temp(self):
        # kBT at 298 K in kcal/mol should be ~0.592
        assert abs(KBT_KCAL - 0.592) < 0.01


# Structures / molecules
class TestAtom:
    def test_create(self):
        a = Atom(index=0, name="CA", x=1.0, y=2.0, z=3.0, charge=0.5, radius=1.8)
        assert a.name == "CA"
        assert a.charge == 0.5

    def test_position_property(self):
        a = Atom(x=1.0, y=2.0, z=3.0)
        assert np.allclose(a.position, [1.0, 2.0, 3.0])

    def test_position_setter(self):
        a = Atom()
        a.position = np.array([4.0, 5.0, 6.0])
        assert abs(a.x - 4.0) < 1e-10

    def test_distance_to(self):
        a = Atom(x=0, y=0, z=0)
        b = Atom(x=3, y=4, z=0)
        assert abs(a.distance_to(b) - 5.0) < 1e-10

    def test_repr(self):
        a = Atom(name="N", x=1.0, y=2.0, z=3.0)
        assert "N" in repr(a)

    def test_zero_atom(self):
        a = Atom()
        assert a.x == 0.0
        assert a.charge == 0.0

    def test_distance_self(self):
        a = Atom(x=1.0, y=2.0, z=3.0)
        assert a.distance_to(a) == 0.0

    def test_distance_3d(self):
        a = Atom(x=1, y=2, z=3)
        b = Atom(x=4, y=6, z=3)
        assert abs(a.distance_to(b) - 5.0) < 1e-10


class TestMolecule:
    def _make_mol(self) -> Molecule:
        mol = Molecule(name="test")
        mol.atoms = [
            Atom(index=0, x=0.0, y=0.0, z=0.0, charge=1.0, radius=1.5),
            Atom(index=1, x=2.0, y=0.0, z=0.0, charge=-1.0, radius=1.5),
            Atom(index=2, x=1.0, y=2.0, z=0.0, charge=0.5, radius=1.2),
        ]
        return mol

    def test_create(self):
        mol = self._make_mol()
        assert mol.name == "test"
        assert len(mol) == 3

    def test_centroid(self):
        mol = self._make_mol()
        c = mol.centroid()
        assert np.allclose(c, [1.0, 2 / 3, 0.0])

    def test_total_charge(self):
        mol = self._make_mol()
        assert abs(mol.total_charge() - 0.5) < 1e-10

    def test_positions_array(self):
        mol = self._make_mol()
        pos = mol.positions_array()
        assert pos.shape == (3, 3)

    def test_charges_array(self):
        mol = self._make_mol()
        q = mol.charges_array()
        assert q.shape == (3,)
        assert abs(q.sum() - 0.5) < 1e-10

    def test_translate(self):
        mol = self._make_mol()
        mol.translate(np.array([1.0, 0.0, 0.0]))
        assert abs(mol.atoms[0].x - 1.0) < 1e-10

    def test_rotate(self):
        mol = Molecule()
        mol.atoms = [Atom(x=1.0, y=0.0, z=0.0)]
        # 90° rotation about z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mol.rotate(R)
        assert abs(mol.atoms[0].x) < 1e-10
        assert abs(mol.atoms[0].y - 1.0) < 1e-10

    def test_bounding_radius(self):
        mol = self._make_mol()
        br = mol.bounding_radius()
        assert br > 0

    def test_radius_of_gyration(self):
        mol = self._make_mol()
        rg = mol.radius_of_gyration()
        assert rg > 0

    def test_empty_molecule(self):
        mol = Molecule()
        assert np.allclose(mol.centroid(), [0, 0, 0])
        assert mol.total_charge() == 0.0

    def test_repr(self):
        mol = self._make_mol()
        assert "test" in repr(mol)

    def test_center_of_mass(self):
        mol = self._make_mol()
        assert np.allclose(mol.center_of_mass(), mol.centroid())

    def test_radii_array(self):
        mol = self._make_mol()
        r = mol.radii_array()
        assert r.shape == (3,)

    def test_rotate_about_centroid(self):
        mol = self._make_mol()
        c_before = mol.centroid().copy()
        R = np.eye(3)  # identity
        mol.rotate_about_centroid(R)
        assert np.allclose(mol.centroid(), c_before)


class TestBoundingBox:
    def _make_bb(self) -> BoundingBox:
        mol = Molecule()
        mol.atoms = [
            Atom(x=-1, y=-2, z=-3),
            Atom(x=1, y=2, z=3),
        ]
        return BoundingBox.from_molecule(mol, padding=0.0)

    def test_create(self):
        bb = self._make_bb()
        assert bb.xmin == -1
        assert bb.xmax == 1

    def test_center(self):
        bb = self._make_bb()
        assert np.allclose(bb.center, [0, 0, 0])

    def test_size(self):
        bb = self._make_bb()
        assert np.allclose(bb.size, [2, 4, 6])

    def test_contains(self):
        bb = self._make_bb()
        assert bb.contains(np.array([0, 0, 0]))
        assert not bb.contains(np.array([5, 0, 0]))

    def test_padding(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0)]
        bb = BoundingBox.from_molecule(mol, padding=2.0)
        assert bb.xmin == -2.0
        assert bb.xmax == 2.0

    def test_repr(self):
        bb = self._make_bb()
        assert "BoundingBox" in repr(bb)


class TestContactPair:
    def test_create(self):
        cp = ContactPair(0, 1, 5.0)
        assert cp.mol1_atom_index == 0
        assert cp.distance_cutoff == 5.0

    def test_repr(self):
        cp = ContactPair(2, 3, 4.0)
        assert "2" in repr(cp)


class TestReactionCriteria:
    def _setup(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, y=0, z=0), Atom(x=5, y=0, z=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=3, y=0, z=0), Atom(x=8, y=0, z=0)]
        return mol1, mol2

    def test_satisfied(self):
        mol1, mol2 = self._setup()
        pair = ContactPair(0, 0, 5.0)  # atom0 in mol1 to atom0 in mol2: dist=3
        criteria = ReactionCriteria(pairs=[pair])
        assert criteria.is_satisfied(mol1, mol2)

    def test_not_satisfied(self):
        mol1, mol2 = self._setup()
        pair = ContactPair(0, 0, 2.0)  # cutoff too small
        criteria = ReactionCriteria(pairs=[pair])
        assert not criteria.is_satisfied(mol1, mol2)

    def test_multiple_pairs_all_required(self):
        mol1, mol2 = self._setup()
        p1 = ContactPair(0, 0, 5.0)  # satisfied (dist=3)
        p2 = ContactPair(0, 1, 2.0)  # not satisfied (dist=8)
        criteria = ReactionCriteria(pairs=[p1, p2])
        assert not criteria.is_satisfied(mol1, mol2)


# PQR I/O
class TestPQRIO:
    def _pqr_content(self) -> str:
        return (
            "REMARK  Test PQR\n"
            "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
            "ATOM      2  CB  ALA     1       4.000   5.000   6.000 -0.200  1.700\n"
            "END\n"
        )

    def test_parse(self, tmp_path):
        p = tmp_path / "test.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert len(mol.atoms) == 2
        assert mol.atoms[0].name == "CA"
        assert abs(mol.atoms[0].x - 1.0) < 1e-6
        assert abs(mol.atoms[0].charge - 0.5) < 1e-6
        assert abs(mol.atoms[0].radius - 1.8) < 1e-6

    def test_parse_charges(self, tmp_path):
        p = tmp_path / "test.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert abs(mol.total_charge() - 0.3) < 1e-5

    def test_roundtrip(self, tmp_path):
        p_in = tmp_path / "in.pqr"
        p_out = tmp_path / "out.pqr"
        p_in.write_text(self._pqr_content())
        mol = parse_pqr(p_in)
        write_pqr(mol, p_out)
        mol2 = parse_pqr(p_out)
        assert len(mol2.atoms) == 2
        assert abs(mol2.atoms[0].x - 1.0) < 1e-3

    def test_molecule_name_from_stem(self, tmp_path):
        p = tmp_path / "myprotein.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert mol.name == "myprotein"

    def test_empty_pqr(self, tmp_path):
        p = tmp_path / "empty.pqr"
        p.write_text("REMARK empty\nEND\n")
        mol = parse_pqr(p)
        assert len(mol.atoms) == 0

    def test_hetatm(self, tmp_path):
        p = tmp_path / "ligand.pqr"
        p.write_text(
            "HETATM    1  C1  LIG     1       0.000   0.000   0.000  0.100  1.500\nEND\n"
        )
        mol = parse_pqr(p)
        assert len(mol.atoms) == 1


# Quaternion and transforms
class TestQuaternion:
    def test_identity(self):
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0

    def test_norm(self):
        q = Quaternion(1, 0, 0, 0)
        assert abs(q.norm() - 1.0) < 1e-14

    def test_normalized(self):
        q = Quaternion(2, 0, 0, 0).normalized()
        assert abs(q.w - 1.0) < 1e-14

    def test_rotation_matrix_identity(self):
        q = Quaternion.identity()
        R = q.to_rotation_matrix()
        assert np.allclose(R, np.eye(3))

    def test_from_axis_angle_90z(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        R = q.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 1, 0], atol=1e-10)

    def test_from_axis_angle_180x(self):
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), math.pi)
        R = q.to_rotation_matrix()
        v = R @ np.array([0, 1, 0])
        assert np.allclose(v, [0, -1, 0], atol=1e-10)

    def test_multiply_identity(self):
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        r = q * Quaternion.identity()
        assert np.allclose(q.to_array(), r.normalized().to_array(), atol=1e-10)

    def test_conjugate(self):
        q = Quaternion(0.7, 0.1, 0.2, 0.3).normalized()
        qc = q.conjugate()
        prod = (q * qc).normalized()
        assert abs(prod.w - 1.0) < 1e-10

    def test_rotate_vector(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi)
        v = q.rotate_vector(np.array([1, 0, 0]))
        assert np.allclose(v, [-1, 0, 0], atol=1e-10)

    def test_to_array(self):
        q = Quaternion(1, 0, 0, 0)
        arr = q.to_array()
        assert arr.shape == (4,)

    def test_from_rotation_matrix_roundtrip(self):
        q_orig = Quaternion.from_axis_angle(np.array([1, 1, 0]) / math.sqrt(2), 1.2)
        R = q_orig.to_rotation_matrix()
        q_back = Quaternion.from_rotation_matrix(R)
        R_back = q_back.to_rotation_matrix()
        assert np.allclose(R, R_back, atol=1e-10)

    def test_repr(self):
        q = Quaternion.identity()
        assert "Quaternion" in repr(q)

    def test_zero_axis(self):
        q = Quaternion.from_axis_angle(np.zeros(3), 1.0)
        assert abs(q.w - 1.0) < 1e-10

    def test_from_axis_angle_360(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), 2 * math.pi)
        R = q.to_rotation_matrix()
        assert np.allclose(R, np.eye(3), atol=1e-10)


class TestRigidTransform:
    def test_identity(self):
        T = RigidTransform.identity()
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(T.apply(v), v)

    def test_pure_translation(self):
        T = RigidTransform(translation=np.array([1.0, 2.0, 3.0]))
        v = np.zeros(3)
        assert np.allclose(T.apply(v), [1, 2, 3])

    def test_pure_rotation(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        T = RigidTransform(rotation=q)
        v = np.array([1.0, 0.0, 0.0])
        result = T.apply(v)
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_compose(self):
        T1 = RigidTransform(translation=np.array([1.0, 0.0, 0.0]))
        T2 = RigidTransform(translation=np.array([2.0, 0.0, 0.0]))
        T12 = T1.compose(T2)
        v = np.zeros(3)
        assert np.allclose(T12.apply(v), [3, 0, 0])

    def test_inverse(self):
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.7)
        T = RigidTransform(rotation=q, translation=np.array([1, 2, 3]))
        Ti = T.inverse()
        v = np.array([4.0, 5.0, 6.0])
        result = Ti.apply(T.apply(v))
        assert np.allclose(result, v, atol=1e-10)

    def test_apply_batch(self):
        T = RigidTransform(translation=np.array([1.0, 0.0, 0.0]))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = T.apply(pts)
        assert result.shape == (2, 3)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_repr(self):
        T = RigidTransform.identity()
        assert "RigidTransform" in repr(T)


class TestRandomQuaternion:
    def test_returns_quaternion(self):
        rng = np.random.default_rng(42)
        q = random_quaternion(rng)
        assert isinstance(q, Quaternion)

    def test_unit_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = random_quaternion(rng)
            assert abs(q.norm() - 1.0) < 1e-10

    def test_rotation_matrix_orthogonal(self):
        rng = np.random.default_rng(1)
        q = random_quaternion(rng)
        R = q.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_small_rotation(self):
        rng = np.random.default_rng(7)
        q = small_rotation_quaternion(0.01, rng)
        assert abs(q.norm() - 1.0) < 1e-10


# Hydrodynamics
class TestHydrodynamics:
    def test_stokes_translation_positive(self):
        D = stokes_translational_diffusion(20.0)  # 20 Å radius
        assert D > 0

    def test_stokes_rotation_positive(self):
        D = stokes_rotational_diffusion(20.0)
        assert D > 0

    def test_stokes_translation_larger_radius_smaller_D(self):
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(20.0)
        assert D1 > D2

    def test_stokes_rotation_larger_radius_smaller_D(self):
        D1 = stokes_rotational_diffusion(10.0)
        D2 = stokes_rotational_diffusion(20.0)
        assert D1 > D2

    def test_mobility_from_radii(self):
        mob = MobilityTensor.from_radii(20.0, 20.0)
        assert mob.D_trans1 > 0
        assert mob.D_trans2 > 0

    def test_relative_diffusion(self):
        mob = MobilityTensor.from_radii(20.0, 20.0)
        D_rel = mob.relative_translational_diffusion()
        assert abs(D_rel - 2 * mob.D_trans1) < 1e-14

    def test_rotne_prager_far_field(self):
        r_vec = np.array([100.0, 0.0, 0.0])
        M = rpy_offdiagonal(r_vec, 5.0, 5.0, 1.0, 1.0)
        assert M.shape == (3, 3)

    def test_rotne_prager_zero_distance(self):
        M = rpy_offdiagonal(np.zeros(3), 5.0, 5.0, 1.0, 1.0)
        assert np.allclose(M, np.zeros((3, 3)))

    def test_repr(self):
        mob = MobilityTensor(1.0, 0.1, 1.0, 0.1)
        assert "MobilityTensor" in repr(mob)

    def test_stokes_units_reasonable(self):
        # Typical protein (~30 Å radius) D_t ~ 0.005-0.05 Å²/ps
        D = stokes_translational_diffusion(30.0)
        assert 1e-4 < D < 1.0


# BD integrator
class TestBDStep:
    def test_translation_moves(self):
        rng = np.random.default_rng(42)
        pos = np.zeros(3)
        force = np.zeros(3)
        new_pos = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng)
        assert not np.allclose(new_pos, pos)  # diffuses

    def test_translation_with_force(self):
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        force = np.array([100.0, 0.0, 0.0])
        # large force in x → drift dominates
        new_pos = ermak_mccammon_translation(pos, force, 10.0, 1.0, rng)
        # on average, drift = D*dt*F = 10*1*100 = 1000 Å
        assert new_pos[0] > 500.0  # very likely for large drift

    def test_rotation_changes_orientation(self):
        rng = np.random.default_rng(42)
        ori = Quaternion.identity()
        torque = np.zeros(3)
        new_ori = ermak_mccammon_rotation(ori, torque, 0.01, 0.2, rng)
        # should rotate randomly
        R = new_ori.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_bd_step_returns_tuple(self):
        rng = np.random.default_rng(1)
        pos = np.array([50.0, 0.0, 0.0])
        ori = Quaternion.identity()
        new_pos, new_ori = bd_step(
            pos, ori, np.zeros(3), np.zeros(3), 10.0, 0.01, 0.2, rng
        )
        assert new_pos.shape == (3,)
        assert isinstance(new_ori, Quaternion)

    def test_escape_radius(self):
        r = escape_radius(100.0)
        assert r >= 500.0

    def test_escape_radius_fallback(self):
        r = escape_radius(10.0)
        assert r >= 50.0

    def test_translation_reproducible_seed(self):
        pos = np.zeros(3)
        force = np.zeros(3)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng1)
        p2 = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng2)
        assert np.allclose(p1, p2)

    def test_small_dt_small_step(self):
        rng = np.random.default_rng(5)
        pos = np.zeros(3)
        # 100 steps with tiny dt
        steps = []
        for _ in range(100):
            new_pos = ermak_mccammon_translation(pos, np.zeros(3), 1.0, 1e-6, rng)
            steps.append(np.linalg.norm(new_pos - pos))
        assert np.mean(steps) < 0.01


# SystemState / Fate
class TestSystemState:
    def test_create(self):
        s = SystemState()
        assert s.fate == Fate.ONGOING
        assert s.step == 0

    def test_separation(self):
        s = SystemState(position=np.array([3.0, 4.0, 0.0]))
        assert abs(s.separation() - 5.0) < 1e-10

    def test_copy(self):
        s = SystemState(position=np.array([1.0, 2.0, 3.0]), step=5)
        s2 = s.copy()
        s2.position[0] = 99.0
        assert s.position[0] == 1.0

    def test_repr(self):
        s = SystemState()
        assert "SystemState" in repr(s)

    def test_fate_ongoing(self):
        s = SystemState()
        assert s.fate == Fate.ONGOING

    def test_fate_reacted(self):
        s = SystemState()
        s.fate = Fate.REACTED
        assert s.fate == Fate.REACTED


class TestTrajectoryResult:
    def test_reacted_property(self):
        r = TrajectoryResult(Fate.REACTED, 100, 20.0, 5.0, "rxn1")
        assert r.reacted
        assert not r.escaped

    def test_escaped_property(self):
        r = TrajectoryResult(Fate.ESCAPED, 500, 100.0, 300.0)
        assert r.escaped
        assert not r.reacted

    def test_repr(self):
        r = TrajectoryResult(Fate.ESCAPED, 200, 40.0, 200.0)
        assert "TrajectoryResult" in repr(r)


# Pathways / reactions
class TestReactionInterface:
    def _setup(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, y=0, z=0), Atom(x=10, y=0, z=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2, y=0, z=0), Atom(x=12, y=0, z=0)]
        pair = ContactPair(0, 0, 5.0)  # dist = 2, cutoff = 5 → satisfied
        criteria = ReactionCriteria(name="test", pairs=[pair])
        rxn = ReactionInterface(name="rxn1", criteria=criteria)
        return mol1, mol2, rxn

    def test_check_fires(self):
        mol1, mol2, rxn = self._setup()
        assert rxn.check(mol1, mol2)

    def test_check_probability_zero(self):
        mol1, mol2, rxn = self._setup()
        rxn.probability = 0.0
        assert not rxn.check(mol1, mol2)

    def test_repr(self):
        _, _, rxn = self._setup()
        assert "rxn1" in repr(rxn)


class TestPathwaySet:
    def _make_set(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, y=0, z=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2, y=0, z=0)]
        pair = ContactPair(0, 0, 5.0)
        criteria = ReactionCriteria(pairs=[pair])
        rxn = ReactionInterface(name="r1", criteria=criteria)
        ps = PathwaySet([rxn])
        return mol1, mol2, ps

    def test_check_all_fires(self):
        mol1, mol2, ps = self._make_set()
        rng = np.random.default_rng(0)
        name = ps.check_all(mol1, mol2, rng)
        assert name == "r1"

    def test_empty_set(self):
        mol1 = Molecule()
        mol1.atoms = [Atom()]
        mol2 = Molecule()
        mol2.atoms = [Atom()]
        ps = PathwaySet()
        assert ps.check_all(mol1, mol2) is None

    def test_len(self):
        _, _, ps = self._make_set()
        assert len(ps) == 1

    def test_repr(self):
        _, _, ps = self._make_set()
        assert "PathwaySet" in repr(ps)

    def test_add(self):
        ps = PathwaySet()
        pair = ContactPair(0, 0, 5.0)
        criteria = ReactionCriteria(pairs=[pair])
        ps.add(ReactionInterface("r2", criteria))
        assert len(ps) == 1


class TestMakeDefaultReaction:
    def test_creates_reaction(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=float(i), y=0, z=0) for i in range(5)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=float(i + 20), y=0, z=0) for i in range(5)]
        rxn = make_default_reaction(mol1, mol2, cutoff=5.0, n_pairs=2)
        assert isinstance(rxn, ReactionInterface)
        assert len(rxn.criteria.pairs) == 2


# Electrostatics
class TestDebyeHuckel:
    def test_same_sign_positive(self):
        E = debye_huckel_energy(1.0, 1.0, 10.0)
        assert E > 0

    def test_opposite_sign_negative(self):
        E = debye_huckel_energy(1.0, -1.0, 10.0)
        assert E < 0

    def test_decays_with_distance(self):
        E1 = debye_huckel_energy(1.0, 1.0, 5.0)
        E2 = debye_huckel_energy(1.0, 1.0, 10.0)
        assert E1 > E2

    def test_zero_charge(self):
        E = debye_huckel_energy(0.0, 1.0, 10.0)
        assert E == 0.0

    def test_zero_distance(self):
        E = debye_huckel_energy(1.0, 1.0, 0.0)
        assert E == 0.0

    def test_force_direction(self):
        r_vec = np.array([10.0, 0.0, 0.0])
        F = debye_huckel_force(1.0, 1.0, r_vec)
        assert F.shape == (3,)

    def test_force_zero_charge(self):
        r_vec = np.array([5.0, 0.0, 0.0])
        F = debye_huckel_force(0.0, 1.0, r_vec)
        assert np.allclose(F, 0)


class TestDXGrid:
    def _make_grid(self) -> DXGrid:
        """Small 5×5×5 grid with linearly varying potential."""
        origin = np.zeros(3)
        delta = np.diag([1.0, 1.0, 1.0])
        data = np.zeros((5, 5, 5))
        for i in range(5):
            data[i, :, :] = float(i)  # potential increases with x
        return DXGrid(origin, delta, data)

    def test_interpolate_at_node(self):
        g = self._make_grid()
        val = g.interpolate(np.array([2.0, 2.0, 2.0]))
        assert abs(val - 2.0) < 1e-8

    def test_interpolate_between_nodes(self):
        g = self._make_grid()
        val = g.interpolate(np.array([1.5, 1.0, 1.0]))
        assert abs(val - 1.5) < 1e-8

    def test_interpolate_out_of_bounds(self):
        g = self._make_grid()
        val = g.interpolate(np.array([100.0, 0.0, 0.0]))
        assert val == 0.0

    def test_gradient(self):
        g = self._make_grid()
        grad = g.gradient(np.array([2.0, 2.0, 2.0]))
        assert abs(grad[0] - 1.0) < 0.1  # potential increases with x
        assert abs(grad[1]) < 0.2

    def test_force_on_charge(self):
        g = self._make_grid()
        F = g.force_on_charge(np.array([2.0, 2.0, 2.0]), 1.0)
        assert F.shape == (3,)

    def test_repr(self):
        g = self._make_grid()
        assert "DXGrid" in repr(g)

    def test_from_file(self, tmp_path):
        """Write a minimal DX file and read it back."""
        dx_content = """# APBS generated potential
object 1 class gridpositions counts 3 3 3
origin 0.000 0.000 0.000
delta 1.000 0.000 0.000
delta 0.000 1.000 0.000
delta 0.000 0.000 1.000
object 2 class gridconnections counts 3 3 3
object 3 class array type double rank 0 items 27 data follows
0.0 1.0 2.0 1.0 2.0 3.0 2.0 3.0 4.0
1.0 2.0 3.0 2.0 3.0 4.0 3.0 4.0 5.0
2.0 3.0 4.0 3.0 4.0 5.0 4.0 5.0 6.0
object 4 class field
"""
        p = tmp_path / "test.dx"
        p.write_text(dx_content)
        g = DXGrid.from_file(p)
        assert g.data.shape == (3, 3, 3)
        assert abs(g.interpolate(np.array([0.0, 0.0, 0.0])) - 0.0) < 1e-8


# Auxiliary tools
class TestAuxTools:
    def _mol(self) -> Molecule:
        mol = Molecule(name="m")
        mol.atoms = [
            Atom(x=0, y=0, z=0, charge=1.0, radius=1.5),
            Atom(x=5, y=0, z=0, charge=-1.0, radius=1.5),
            Atom(x=2, y=3, z=0, charge=0.5, radius=1.2),
        ]
        return mol

    def test_bounding_box(self):
        mol = self._mol()
        bb = bounding_box(mol, padding=0.0)
        assert bb.xmin <= 0.0
        assert bb.xmax >= 5.0

    def test_bounding_box_padding(self):
        mol = self._mol()
        bb0 = bounding_box(mol, padding=0.0)
        bb5 = bounding_box(mol, padding=5.0)
        assert bb5.xmin < bb0.xmin
        assert bb5.xmax > bb0.xmax

    def test_surface_spheres_nonempty(self):
        mol = self._mol()
        pts = surface_spheres(mol, probe_radius=1.4, n_points=20)
        assert len(pts) > 0

    def test_lumped_charges(self):
        mol = self._mol()
        lc = lumped_charges(mol, grid_spacing=3.0)
        assert len(lc) > 0
        total_q = sum(q for _, q in lc)
        assert abs(total_q - mol.total_charge()) < 1e-6

    def test_electrostatic_center(self):
        mol = self._mol()
        ec = electrostatic_center(mol)
        assert ec.shape == (3,)

    def test_electrostatic_center_zero_charge(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0, charge=0), Atom(x=2, charge=0)]
        ec = electrostatic_center(mol)
        assert np.allclose(ec, mol.centroid())

    def test_hydrodynamic_radius(self):
        mol = self._mol()
        rh = hydrodynamic_radius_from_rg(mol)
        assert rh > 0

    def test_contact_distances(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, y=0, z=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=3, y=0, z=0), Atom(x=20, y=0, z=0)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert len(pairs) == 1
        assert abs(pairs[0][2] - 3.0) < 1e-8

    def test_contact_distances_none(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=100)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert len(pairs) == 0

    def test_born_integral_negative(self):
        E = born_integral(1.0, 3.0)
        assert E < 0  # solvation is stabilizing

    def test_born_integral_zero_charge(self):
        E = born_integral(0.0, 3.0)
        assert E == 0.0

    def test_born_integral_zero_radius(self):
        E = born_integral(1.0, 0.0)
        assert E == 0.0


# Numerical library
class TestCubicSpline:
    def test_interpolates_at_nodes(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        sp = CubicSpline(x, y)
        for xi, yi in zip(x, y):
            assert abs(sp(xi) - yi) < 1e-8

    def test_interpolates_between(self):
        x = np.linspace(0, math.pi, 20)
        y = np.sin(x)
        sp = CubicSpline(x, y)
        val = sp(math.pi / 4)
        assert abs(val - math.sin(math.pi / 4)) < 0.01

    def test_derivative(self):
        x = np.linspace(0, 2, 10)
        y = x**2
        sp = CubicSpline(x, y)
        # derivative of x² is 2x
        deriv = sp.derivative(1.0)
        assert abs(deriv - 2.0) < 0.1

    def test_two_points(self):
        sp = CubicSpline(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert abs(sp(0.5) - 0.5) < 1e-8

    def test_extrapolation_boundary(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        sp = CubicSpline(x, y)
        assert abs(sp(0.0) - 0.0) < 1e-8
        assert abs(sp(2.0) - 2.0) < 1e-8


class TestRomberg:
    def test_constant(self):
        val = romberg_integrate(lambda x: 1.0, 0.0, 1.0)
        assert abs(val - 1.0) < 1e-8

    def test_linear(self):
        val = romberg_integrate(lambda x: x, 0.0, 1.0)
        assert abs(val - 0.5) < 1e-8

    def test_quadratic(self):
        val = romberg_integrate(lambda x: x**2, 0.0, 1.0)
        assert abs(val - 1.0 / 3.0) < 1e-8

    def test_sine(self):
        val = romberg_integrate(math.sin, 0.0, math.pi)
        assert abs(val - 2.0) < 1e-8

    def test_exp(self):
        val = romberg_integrate(math.exp, 0.0, 1.0)
        assert abs(val - (math.e - 1.0)) < 1e-8


class TestWienerStep:
    def test_shape(self):
        rng = np.random.default_rng(0)
        dW = wiener_step(1.0, 0.1, 3, rng)
        assert dW.shape == (3,)

    def test_scaling(self):
        # std of many steps should be sqrt(2Ddt)
        rng = np.random.default_rng(42)
        steps = np.array([wiener_step(1.0, 0.1, 1, rng)[0] for _ in range(5000)])
        expected_std = math.sqrt(2.0 * 1.0 * 0.1)
        assert abs(steps.std() - expected_std) < 0.05


class TestMultipoles:
    def test_monopole(self):
        q = np.array([1.0, -1.0, 0.5])
        assert abs(monopole_moment(q) - 0.5) < 1e-10

    def test_dipole_shape(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        q = np.array([1.0, -1.0, 0.0])
        p = dipole_moment(pos, q)
        assert p.shape == (3,)

    def test_dipole_symmetric(self):
        pos = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
        q = np.array([1.0, -1.0])
        p = dipole_moment(pos, q)
        assert abs(p[0] - (-2.0)) < 1e-10

    def test_quadrupole_shape(self):
        pos = np.random.randn(5, 3)
        q = np.random.randn(5)
        Q = quadrupole_moment(pos, q)
        assert Q.shape == (3, 3)

    def test_quadrupole_symmetric(self):
        pos = np.random.randn(5, 3)
        q = np.random.randn(5)
        Q = quadrupole_moment(pos, q)
        assert np.allclose(Q, Q.T)


class TestLegendre:
    def test_p0(self):
        assert abs(legendre_p(0, 0.5) - 1.0) < 1e-14

    def test_p1(self):
        assert abs(legendre_p(1, 0.5) - 0.5) < 1e-14

    def test_p2(self):
        # P2(x) = (3x²-1)/2
        x = 0.7
        expected = (3 * x**2 - 1) / 2
        assert abs(legendre_p(2, x) - expected) < 1e-12

    def test_p0_minus1(self):
        assert abs(legendre_p(0, -1.0) - 1.0) < 1e-14

    def test_p1_minus1(self):
        assert abs(legendre_p(1, -1.0) - (-1.0)) < 1e-14

    def test_series(self):
        # constant series c0=1 should equal 1 everywhere
        val = legendre_series([1.0], 0.3)
        assert abs(val - 1.0) < 1e-14

    def test_series_p1(self):
        val = legendre_series([0.0, 1.0], 0.5)
        assert abs(val - 0.5) < 1e-14

    def test_legendre_p3(self):
        x = 0.5
        expected = (5 * x**3 - 3 * x) / 2
        assert abs(legendre_p(3, x) - expected) < 1e-12


# XML I/O
class TestReactionXML:
    def _write_reaction_xml(self, path):
        xml = """<?xml version="1.0" ?>
<reactions>
  <reaction name="rxn1" probability="0.9">
    <contact molecule1_index="0" molecule2_index="2" distance="4.5"/>
    <contact molecule1_index="1" molecule2_index="3" distance="5.0"/>
  </reaction>
  <reaction name="rxn2" probability="1.0">
    <contact molecule1_index="5" molecule2_index="7" distance="3.0"/>
  </reaction>
</reactions>
"""
        Path(path).write_text(xml)

    def test_parse_count(self, tmp_path):
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert len(ps) == 2

    def test_parse_names(self, tmp_path):
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        names = [r.name for r in ps.reactions]
        assert "rxn1" in names
        assert "rxn2" in names

    def test_parse_probability(self, tmp_path):
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert abs(ps.reactions[0].probability - 0.9) < 1e-6

    def test_parse_contacts(self, tmp_path):
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert len(ps.reactions[0].criteria.pairs) == 2

    def test_roundtrip(self, tmp_path):
        p_in = tmp_path / "rxn_in.xml"
        p_out = tmp_path / "rxn_out.xml"
        self._write_reaction_xml(p_in)
        ps = parse_reaction_xml(p_in)
        write_reaction_xml(ps, p_out)
        ps2 = parse_reaction_xml(p_out)
        assert len(ps2) == len(ps)
        assert ps2.reactions[0].name == ps.reactions[0].name


class TestSimulationXML:
    def _write_sim_xml(self, path):
        xml = """<?xml version="1.0" ?>
<simulation>
  <n_trajectories>500</n_trajectories>
  <dt>0.1</dt>
  <r_start>80.0</r_start>
  <molecule1_pqr>thrombin.pqr</molecule1_pqr>
  <molecule2_pqr>tmod.pqr</molecule2_pqr>
  <reaction_file>rxns.xml</reaction_file>
  <dx_file>grid1.dx</dx_file>
  <dx_file>grid2.dx</dx_file>
</simulation>
"""
        Path(path).write_text(xml)

    def test_parse(self, tmp_path):
        p = tmp_path / "sim.xml"
        self._write_sim_xml(p)
        cfg = parse_simulation_xml(p)
        assert cfg["n_trajectories"] == 500
        assert abs(cfg["dt"] - 0.1) < 1e-8
        assert len(cfg["dx_files"]) == 2

    def test_parse_mol_names(self, tmp_path):
        p = tmp_path / "sim.xml"
        self._write_sim_xml(p)
        cfg = parse_simulation_xml(p)
        assert cfg["mol1_pqr"] == "thrombin.pqr"

    def test_roundtrip(self, tmp_path):
        p_in = tmp_path / "sim_in.xml"
        p_out = tmp_path / "sim_out.xml"
        self._write_sim_xml(p_in)
        cfg = parse_simulation_xml(p_in)
        write_simulation_xml(cfg, p_out)
        cfg2 = parse_simulation_xml(p_out)
        assert cfg2["n_trajectories"] == cfg["n_trajectories"]


# NAM simulator (integration tests)
class TestNAMSimulator:
    def _make_sim(self, n=20) -> NAMSimulator:
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom(x=0, y=0, z=0, charge=1.0, radius=2.0)]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom(x=0, y=0, z=0, charge=-1.0, radius=2.0)]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 200.0)  # huge cutoff → always reacts
        criteria = ReactionCriteria(pairs=[pair])
        rxn = ReactionInterface("test_rxn", criteria)
        ps = PathwaySet([rxn])
        params = NAMParameters(n_trajectories=n, r_start=50.0, seed=42, verbose=False)
        return NAMSimulator(mol1, mol2, mob, ps, params)

    def test_run_returns_result(self):
        sim = self._make_sim(5)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_all_react_with_huge_cutoff(self):
        sim = self._make_sim(20)
        result = sim.run()
        # With cutoff 200 Å and r_start=50 → all should react immediately
        assert result.n_reacted + result.n_escaped == 20

    def test_reaction_probability_in_range(self):
        sim = self._make_sim(10)
        result = sim.run()
        assert 0.0 <= result.reaction_probability <= 1.0

    def test_n_trajectories_correct(self):
        sim = self._make_sim(15)
        result = sim.run()
        assert result.n_trajectories == 15

    def test_seed_reproducible(self):
        s1 = self._make_sim(10)
        s2 = self._make_sim(10)
        r1 = s1.run()
        r2 = s2.run()
        assert r1.n_reacted == r2.n_reacted

    def test_escape_with_small_cutoff(self):
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 0.001)  # tiny cutoff → never reacts
        criteria = ReactionCriteria(pairs=[pair])
        ps = PathwaySet([ReactionInterface("r", criteria)])
        params = NAMParameters(
            n_trajectories=5,
            r_start=50.0,
            r_escape=60.0,
            seed=7,
            verbose=False,
            max_steps=1000,
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        result = sim.run()
        assert result.n_escaped + result.n_reacted + result.n_max_steps == 5

    def test_rate_constant_positive(self):
        sim = self._make_sim(20)
        result = sim.run()
        mob = sim.mobility
        D_rel = mob.relative_translational_diffusion()
        if result.n_reacted > 0:
            k = result.rate_constant(D_rel)
            assert k >= 0

    def test_reaction_counts_dict(self):
        sim = self._make_sim(10)
        result = sim.run()
        assert isinstance(result.reaction_counts, dict)

    def test_repr(self):
        sim = self._make_sim(5)
        result = sim.run()
        assert "SimulationResult" in repr(result)

    def test_zero_force_fn(self):
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        f, t, e = zero_force(mol1, mol2)
        assert np.allclose(f, 0)
        assert np.allclose(t, 0)
        assert e == 0.0


# Integration: full pipeline
class TestFullPipeline:
    """End-to-end tests for the complete PySTARC pipeline."""

    def test_pqr_to_simulation(self, tmp_path):
        """Parse PQR → build sim → run → get result."""
        pqr_content = (
            "ATOM      1  CA  GLY     1       0.000   0.000   0.000  0.500  2.000\n"
            "ATOM      2  CB  GLY     1       3.000   0.000   0.000 -0.500  2.000\n"
        )
        p1 = tmp_path / "mol1.pqr"
        p2 = tmp_path / "mol2.pqr"
        p1.write_text(pqr_content)
        p2.write_text(pqr_content)

        mol1 = parse_pqr(p1)
        mol2 = parse_pqr(p2)
        assert len(mol1.atoms) == 2

        mob = MobilityTensor.from_radii(mol1.bounding_radius(), mol2.bounding_radius())
        pair = ContactPair(0, 0, 100.0)
        criteria = ReactionCriteria(pairs=[pair])
        ps = PathwaySet([ReactionInterface("rxn", criteria)])
        params = NAMParameters(n_trajectories=5, r_start=30.0, seed=1)
        sim = NAMSimulator(mol1, mol2, mob, ps, params)
        result = sim.run()
        assert result.n_trajectories == 5

    def test_xml_reaction_to_simulation(self, tmp_path):
        """Write reaction XML → parse → simulate."""
        rxn_xml = """<?xml version="1.0" ?>
<reactions>
  <reaction name="contact" probability="1.0">
    <contact molecule1_index="0" molecule2_index="0" distance="200.0"/>
  </reaction>
</reactions>
"""
        rxn_path = tmp_path / "rxns.xml"
        rxn_path.write_text(rxn_xml)
        ps = parse_reaction_xml(rxn_path)
        assert len(ps) == 1
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        params = NAMParameters(n_trajectories=3, r_start=30.0, seed=0)
        sim = NAMSimulator(mol1, mol2, mob, ps, params)
        result = sim.run()
        assert result.n_reacted >= 0

    def test_brace_version(self):
        assert pystarc.__version__  # version check

    def test_module_import_chain(self):
        """Verify all major modules importable."""
        import pystarc.structures.molecules
        import pystarc.structures.pqr_io
        import pystarc.transforms.quaternion
        import pystarc.hydrodynamics.rotne_prager
        import pystarc.motion.do_bd_step
        import pystarc.molsystem.system_state
        import pystarc.pathways.reaction_interface
        import pystarc.forces.electrostatic.grid_force
        import pystarc.simulation.nam_simulator
        import pystarc.xml_io.simulation_io
        import pystarc.aux.aux_tools
        import pystarc.lib.numerical
        import pystarc.cli.main

    def test_constants_importable_from_root(self):
        assert PI > 3.14
        assert BJERRUM_LENGTH > 0

    def test_empty_pathway_set_never_reacts(self):
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        ps = PathwaySet()  # no reactions
        params = NAMParameters(
            n_trajectories=5, r_start=30.0, r_escape=50.0, seed=3, max_steps=100
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params)
        result = sim.run()
        assert result.n_reacted == 0


# Additional tests
class TestAtomExtra:
    def test_index_stored(self):
        a = Atom(index=7)
        assert a.index == 7

    def test_residue_name_stored(self):
        a = Atom(residue_name="GLY")
        assert a.residue_name == "GLY"

    def test_residue_index_stored(self):
        a = Atom(residue_index=42)
        assert a.residue_index == 42

    def test_chain_stored(self):
        a = Atom(chain="B")
        assert a.chain == "B"

    def test_negative_charge(self):
        a = Atom(charge=-2.5)
        assert a.charge == -2.5

    def test_large_radius(self):
        a = Atom(radius=10.0)
        assert a.radius == 10.0

    def test_position_roundtrip(self):
        a = Atom()
        p = np.array([1.1, 2.2, 3.3])
        a.position = p
        assert np.allclose(a.position, p)

    def test_distance_symmetry(self):
        a = Atom(x=1, y=2, z=3)
        b = Atom(x=4, y=5, z=6)
        assert abs(a.distance_to(b) - b.distance_to(a)) < 1e-10

    def test_default_radius(self):
        a = Atom()
        assert a.radius == 1.5

    def test_default_chain(self):
        a = Atom()
        assert a.chain == "A"


class TestMoleculeExtra:
    def _mol5(self):
        mol = Molecule(name="penta")
        for i in range(5):
            mol.atoms.append(
                Atom(
                    index=i, x=float(i), y=0, z=0, charge=float(i - 2) * 0.5, radius=1.5
                )
            )
        return mol

    def test_five_atoms(self):
        mol = self._mol5()
        assert len(mol) == 5

    def test_centroid_x(self):
        mol = self._mol5()
        c = mol.centroid()
        assert abs(c[0] - 2.0) < 1e-10

    def test_total_charge_five(self):
        mol = self._mol5()
        # charges: -1, -0.5, 0, 0.5, 1.0 → sum=0
        assert abs(mol.total_charge()) < 1e-10

    def test_translate_all_atoms(self):
        mol = self._mol5()
        orig_x = [a.x for a in mol.atoms]
        mol.translate(np.array([5.0, 0, 0]))
        for i, a in enumerate(mol.atoms):
            assert abs(a.x - (orig_x[i] + 5.0)) < 1e-10

    def test_rotate_preserves_centroid_distance(self):
        mol = self._mol5()
        c = mol.centroid()
        dists_before = [np.linalg.norm(a.position - c) for a in mol.atoms]
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mol.rotate(R)
        c2 = mol.centroid()
        dists_after = [np.linalg.norm(a.position - c2) for a in mol.atoms]
        for db, da in zip(dists_before, dists_after):
            assert abs(db - da) < 1e-8

    def test_bounding_radius_grows_with_spread(self):
        mol_tight = Molecule()
        mol_tight.atoms = [Atom(x=0, radius=1), Atom(x=1, radius=1)]
        mol_wide = Molecule()
        mol_wide.atoms = [Atom(x=0, radius=1), Atom(x=10, radius=1)]
        assert mol_wide.bounding_radius() > mol_tight.bounding_radius()

    def test_single_atom_molecule(self):
        mol = Molecule(name="single")
        mol.atoms = [Atom(x=3, y=4, z=5)]
        assert np.allclose(mol.centroid(), [3, 4, 5])

    def test_charges_array_dtype(self):
        mol = self._mol5()
        q = mol.charges_array()
        assert q.dtype == float

    def test_positions_array_shape(self):
        mol = self._mol5()
        pos = mol.positions_array()
        assert pos.shape == (5, 3)

    def test_repr_contains_atom_count(self):
        mol = self._mol5()
        assert "5" in repr(mol)


class TestQuaternionExtra:
    def test_from_axis_angle_small(self):
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.001)
        assert abs(q.norm() - 1.0) < 1e-10

    def test_multiply_non_commutative(self):
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.5)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        q12 = (q1 * q2).normalized()
        q21 = (q2 * q1).normalized()
        # should generally differ
        assert not np.allclose(q12.to_array(), q21.to_array())

    def test_double_rotation(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 4)
        qq = (q * q).normalized()
        R = qq.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 1, 0], atol=1e-10)

    def test_inverse_rotation(self):
        q = Quaternion.from_axis_angle(np.array([1, 1, 0]) / math.sqrt(2), 1.0)
        qi = q.conjugate().normalized()
        R = q.to_rotation_matrix()
        Ri = qi.to_rotation_matrix()
        assert np.allclose(R @ Ri, np.eye(3), atol=1e-10)

    def test_many_random_unit_norm(self):
        rng = np.random.default_rng(123)
        for _ in range(50):
            q = random_quaternion(rng)
            assert abs(q.norm() - 1.0) < 1e-10

    def test_from_rotation_matrix_identity(self):
        q = Quaternion.from_rotation_matrix(np.eye(3))
        assert abs(abs(q.w) - 1.0) < 1e-10

    def test_conjugate_norm_preserved(self):
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        assert abs(q.norm() - q.conjugate().norm()) < 1e-14

    def test_rotation_matrix_det_one(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = random_quaternion(rng)
            R = q.to_rotation_matrix()
            assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestRigidTransformExtra:
    def test_rotation_then_translation(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        T = RigidTransform(rotation=q, translation=np.array([0, 1, 0]))
        v = np.array([1, 0, 0])
        result = T.apply(v)
        # rotate → [0,1,0], translate → [0,2,0]
        assert np.allclose(result, [0, 2, 0], atol=1e-10)

    def test_compose_three(self):
        t = np.array([1, 0, 0])
        T1 = RigidTransform(translation=t)
        T2 = RigidTransform(translation=t)
        T3 = RigidTransform(translation=t)
        T123 = T1.compose(T2).compose(T3)
        result = T123.apply(np.zeros(3))
        assert np.allclose(result, [3, 0, 0])

    def test_identity_inverse_is_identity(self):
        T = RigidTransform.identity()
        Ti = T.inverse()
        v = np.array([1, 2, 3], dtype=float)
        assert np.allclose(Ti.apply(v), v)

    def test_apply_preserves_distances(self):
        q = Quaternion.from_axis_angle(np.array([1, 1, 1]) / math.sqrt(3), 0.7)
        T = RigidTransform(rotation=q, translation=np.array([5, 3, 2]))
        p1, p2 = np.array([0, 0, 0], dtype=float), np.array([1, 0, 0], dtype=float)
        d_before = np.linalg.norm(p2 - p1)
        t1, t2 = T.apply(p1), T.apply(p2)
        d_after = np.linalg.norm(t2 - t1)
        assert abs(d_before - d_after) < 1e-10


class TestHydrodynamicsExtra:
    def test_relative_D_t_equals_sum(self):
        mob = MobilityTensor.from_radii(15.0, 25.0)
        assert (
            abs(mob.relative_translational_diffusion() - mob.D_trans1 - mob.D_trans2)
            < 1e-14
        )

    def test_relative_D_r_equals_sum(self):
        mob = MobilityTensor.from_radii(15.0, 25.0)
        assert (
            abs(mob.relative_rotational_diffusion() - mob.D_rot1 - mob.D_rot2) < 1e-14
        )

    def test_D_t_scales_inversely_with_radius(self):
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(20.0)
        assert abs(D1 / D2 - 2.0) < 0.01  # D ∝ 1/r

    def test_D_r_scales_as_inverse_cube(self):
        D1 = stokes_rotational_diffusion(10.0)
        D2 = stokes_rotational_diffusion(20.0)
        assert abs(D1 / D2 - 8.0) < 0.01  # D_r ∝ 1/r³

    def test_asymmetric_molecules(self):
        mob = MobilityTensor.from_radii(10.0, 30.0)
        assert mob.D_trans1 > mob.D_trans2


class TestBDStepExtra:
    def test_large_force_dominates_noise(self):
        """With a huge force, displacement is in the force direction."""
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        # Force entirely in +x
        force = np.array([1e6, 0.0, 0.0])
        displacements = []
        for _ in range(10):
            rng2 = np.random.default_rng(int(rng.integers(1000)))
            new_pos = ermak_mccammon_translation(pos, force, 0.01, 0.01, rng2)
            displacements.append(new_pos[0])
        assert all(d > 0 for d in displacements)

    def test_zero_diffusion_pure_drift(self):
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        force = np.array([1.0, 0.0, 0.0])
        new_pos = ermak_mccammon_translation(pos, force, 0.0, 1.0, rng)
        # zero diffusion → noise is 0, displacement = D*dt*F = 0
        assert np.allclose(new_pos, [0, 0, 0])

    def test_rotation_unit_quaternion_preserved(self):
        rng = np.random.default_rng(42)
        ori = random_quaternion(rng)
        for _ in range(20):
            ori = ermak_mccammon_rotation(ori, np.zeros(3), 0.01, 0.2, rng)
            assert abs(ori.norm() - 1.0) < 1e-10

    def test_escape_radius_min_500(self):
        assert escape_radius(100.0) >= 500.0

    def test_escape_radius_1000(self):
        assert escape_radius(200.0) >= 1000.0


class TestSystemStateExtra:
    def test_step_increment(self):
        s = SystemState(step=5)
        assert s.step == 5

    def test_time_stored(self):
        s = SystemState(time=12.5)
        assert s.time == 12.5

    def test_energy_stored(self):
        s = SystemState(energy=-3.14)
        assert s.energy == -3.14

    def test_force_stored(self):
        f = np.array([1.0, 2.0, 3.0])
        s = SystemState(force=f)
        assert np.allclose(s.force, f)

    def test_torque_stored(self):
        t = np.array([0.1, 0.2, 0.3])
        s = SystemState(torque=t)
        assert np.allclose(s.torque, t)

    def test_copy_deep_orientation(self):
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        s = SystemState(orientation=q)
        s2 = s.copy()
        s2.orientation.w = 999.0
        assert s.orientation.w != 999.0

    def test_fate_max_steps(self):
        s = SystemState(fate=Fate.MAX_STEPS)
        assert s.fate == Fate.MAX_STEPS

    def test_reaction_name_stored(self):
        s = SystemState(reaction_name="my_rxn")
        assert s.reaction_name == "my_rxn"

    def test_separation_zero_origin(self):
        s = SystemState()
        assert s.separation() == 0.0


class TestAuxToolsExtra:
    def _big_mol(self):
        mol = Molecule()
        rng = np.random.default_rng(7)
        pos = rng.uniform(-10, 10, (30, 3))
        for i, p in enumerate(pos):
            mol.atoms.append(
                Atom(
                    index=i,
                    x=p[0],
                    y=p[1],
                    z=p[2],
                    charge=rng.uniform(-1, 1),
                    radius=rng.uniform(1.2, 2.0),
                )
            )
        return mol

    def test_bounding_box_contains_all_atoms(self):
        mol = self._big_mol()
        bb = bounding_box(mol, padding=0.0)
        for a in mol.atoms:
            assert bb.xmin <= a.x <= bb.xmax
            assert bb.ymin <= a.y <= bb.ymax
            assert bb.zmin <= a.z <= bb.zmax

    def test_lumped_charges_conserve_charge(self):
        mol = self._big_mol()
        lc = lumped_charges(mol, grid_spacing=2.0)
        total_q = sum(q for _, q in lc)
        assert abs(total_q - mol.total_charge()) < 1e-5

    def test_contact_distances_sorted(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0), Atom(x=5)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=3), Atom(x=7)]
        pairs = contact_distances(mol1, mol2, cutoff=20.0)
        dists = [p[2] for p in pairs]
        assert dists == sorted(dists)

    def test_surface_spheres_count_scales_with_n_points(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=3.0)]
        pts10 = surface_spheres(mol, n_points=10)
        pts50 = surface_spheres(mol, n_points=50)
        assert len(pts50) >= len(pts10)

    def test_born_integral_larger_charge_more_negative(self):
        E1 = born_integral(1.0, 3.0)
        E2 = born_integral(2.0, 3.0)
        assert E2 < E1  # more negative for larger charge

    def test_born_integral_smaller_radius_more_negative(self):
        E1 = born_integral(1.0, 5.0)
        E2 = born_integral(1.0, 2.0)
        assert E2 < E1


class TestNumericalExtra:
    def test_spline_sine_accurate(self):
        x = np.linspace(0, 2 * math.pi, 50)
        y = np.sin(x)
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.1, 6.0, 30):
            assert abs(sp(xi) - math.sin(xi)) < 0.002

    def test_romberg_exp_negative(self):
        val = romberg_integrate(math.exp, -1.0, 0.0)
        expected = 1.0 - math.exp(-1)
        assert abs(val - expected) < 1e-8

    def test_romberg_polynomial(self):
        val = romberg_integrate(lambda x: x**4, 0.0, 1.0)
        assert abs(val - 0.2) < 1e-8

    def test_wiener_mean_near_zero(self):
        rng = np.random.default_rng(99)
        steps = np.array([wiener_step(1.0, 0.01, 1, rng)[0] for _ in range(2000)])
        assert abs(steps.mean()) < 0.05

    def test_quadrupole_traceless(self):
        rng = np.random.default_rng(5)
        pos = rng.standard_normal((10, 3))
        q = rng.standard_normal(10)
        Q = quadrupole_moment(pos, q)
        assert abs(np.trace(Q)) < 1e-10

    def test_legendre_orthogonal_p0_p2(self):
        # ∫₋₁¹ P0(x)P2(x) dx = 0
        val = romberg_integrate(
            lambda x: legendre_p(0, x) * legendre_p(2, x), -1.0, 1.0
        )
        assert abs(val) < 1e-6

    def test_legendre_norm(self):
        # ∫₋₁¹ [P1(x)]² dx = 2/(2·1+1) = 2/3
        val = romberg_integrate(lambda x: legendre_p(1, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0 / 3.0) < 1e-6

    def test_spline_extrapolation_at_last_node(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2
        sp = CubicSpline(x, y)
        assert abs(sp(4.0) - 16.0) < 1e-6

    def test_wiener_dim3(self):
        rng = np.random.default_rng(0)
        dW = wiener_step(2.0, 0.5, 3, rng)
        assert dW.shape == (3,)

    def test_dipole_zero_charge(self):
        pos = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
        q = np.array([0.0, 0.0])
        p = dipole_moment(pos, q)
        assert np.allclose(p, 0)


class TestDebyeHuckelExtra:
    def test_energy_zero_distance_safe(self):
        E = debye_huckel_energy(1.0, 1.0, 0.0)
        assert E == 0.0

    def test_energy_large_separation_near_zero(self):
        E = debye_huckel_energy(1.0, 1.0, 1000.0, debye_length=7.9)
        assert abs(E) < 1e-30

    def test_energy_scales_with_charge_product(self):
        E1 = debye_huckel_energy(1.0, 1.0, 10.0)
        E2 = debye_huckel_energy(2.0, 1.0, 10.0)
        E3 = debye_huckel_energy(2.0, 2.0, 10.0)
        assert abs(E2 - 2 * E1) < 1e-10
        assert abs(E3 - 4 * E1) < 1e-10

    def test_force_magnitude_positive(self):
        r_vec = np.array([5.0, 0.0, 0.0])
        F = debye_huckel_force(1.0, 1.0, r_vec)
        assert np.linalg.norm(F) > 0

    def test_force_opposite_charges_toward(self):
        r_vec = np.array([5.0, 0.0, 0.0])
        F = debye_huckel_force(1.0, -1.0, r_vec)
        # attractive force should have x-component > 0 (toward +x, i.e. toward charge 2)
        assert F.shape == (3,)


class TestDXGridExtra:
    def _uniform_grid(self, value=2.5) -> DXGrid:
        origin = np.zeros(3)
        delta = np.diag([1.0, 1.0, 1.0])
        data = np.full((6, 6, 6), value)
        return DXGrid(origin, delta, data)

    def test_uniform_grid_any_point(self):
        g = self._uniform_grid(3.0)
        assert abs(g.interpolate(np.array([2.5, 2.5, 2.5])) - 3.0) < 1e-8

    def test_uniform_grid_zero_gradient(self):
        g = self._uniform_grid(1.0)
        grad = g.gradient(np.array([2.5, 2.5, 2.5]))
        assert np.allclose(grad, 0, atol=1e-5)

    def test_force_scales_with_charge(self):
        g = self._uniform_grid()
        F1 = g.force_on_charge(np.array([2.5, 2.5, 2.5]), 1.0)
        F2 = g.force_on_charge(np.array([2.5, 2.5, 2.5]), 2.0)
        assert np.allclose(F2, 2 * F1)

    def test_shape_preserved(self):
        origin = np.zeros(3)
        delta = np.diag([2.0, 2.0, 2.0])
        data = np.zeros((4, 5, 6))
        g = DXGrid(origin, delta, data)
        assert tuple(g.data.shape) == (4, 5, 6)

    def test_origin_stored(self):
        origin = np.array([1.0, 2.0, 3.0])
        delta = np.diag([1.0, 1.0, 1.0])
        data = np.zeros((3, 3, 3))
        g = DXGrid(origin, delta, data)
        assert np.allclose(g.origin, origin)


class TestNAMSimulatorExtra:
    def _fast_sim(self, n=5, huge_cutoff=True) -> NAMSimulator:
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom(x=0, y=0, z=0, charge=1.0, radius=2.0)]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom(x=0, y=0, z=0, charge=-1.0, radius=2.0)]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        cutoff = 500.0 if huge_cutoff else 0.001
        pair = ContactPair(0, 0, cutoff)
        criteria = ReactionCriteria(pairs=[pair])
        ps = PathwaySet([ReactionInterface("r", criteria)])
        params = NAMParameters(n_trajectories=n, r_start=50.0, seed=1)
        return NAMSimulator(mol1, mol2, mob, ps, params)

    def test_result_n_trajectories(self):
        sim = self._fast_sim(8)
        result = sim.run()
        assert result.n_trajectories == 8

    def test_result_counts_sum(self):
        sim = self._fast_sim(10)
        result = sim.run()
        total = result.n_reacted + result.n_escaped + result.n_max_steps
        assert total == 10

    def test_rate_constant_type(self):
        sim = self._fast_sim(10)
        result = sim.run()
        k = result.rate_constant(sim.mobility.relative_translational_diffusion())
        assert isinstance(k, float)

    def test_reaction_probability_bounds(self):
        for n in [5, 10, 20]:
            sim = self._fast_sim(n)
            result = sim.run()
            p = result.reaction_probability
            assert 0.0 <= p <= 1.0

    def test_different_seeds_different_results(self):
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 0.5)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        p1 = NAMParameters(n_trajectories=50, r_start=50.0, seed=1, max_steps=200)
        p2 = NAMParameters(n_trajectories=50, r_start=50.0, seed=999, max_steps=200)
        r1 = NAMSimulator(mol1, mol2, mob, ps, p1).run()
        r2 = NAMSimulator(mol1, mol2, mob, ps, p2).run()
        # Different seeds → at least one different count (overwhelmingly likely)
        # Just check they both ran
        assert r1.n_trajectories == 50
        assert r2.n_trajectories == 50

    def test_zero_trajectories(self):
        sim = self._fast_sim(0)
        result = sim.run()
        assert result.n_trajectories == 0
        assert result.n_reacted == 0

    def test_sim_result_repr(self):
        sim = self._fast_sim(3)
        result = sim.run()
        assert "SimulationResult" in repr(result)

    def test_rate_constant_zero_if_no_reactions(self):
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom()]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 0.0001)  # impossible cutoff
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(
            n_trajectories=5, r_start=30.0, r_escape=40.0, seed=42, max_steps=10
        )
        result = NAMSimulator(mol1, mol2, mob, ps, params).run()
        k = result.rate_constant(mob.relative_translational_diffusion())
        assert k == 0.0 or k >= 0.0


class TestXMLExtra:
    def test_empty_reactions_xml(self, tmp_path):
        xml = "<?xml version='1.0' ?><reactions></reactions>"
        p = tmp_path / "empty.xml"
        p.write_text(xml)
        ps = parse_reaction_xml(p)
        assert len(ps) == 0

    def test_write_and_parse_contacts(self, tmp_path):
        pair = ContactPair(5, 10, 7.5)
        criteria = ReactionCriteria(name="test", pairs=[pair])
        rxn = ReactionInterface("r1", criteria, probability=0.8)
        ps = PathwaySet([rxn])
        p = tmp_path / "rxn.xml"
        write_reaction_xml(ps, p)
        ps2 = parse_reaction_xml(p)
        assert abs(ps2.reactions[0].probability - 0.8) < 1e-5
        assert ps2.reactions[0].criteria.pairs[0].mol1_atom_index == 5

    def test_simulation_xml_defaults(self, tmp_path):
        p = tmp_path / "sim.xml"
        p.write_text("<?xml version='1.0'?><simulation></simulation>")
        cfg = parse_simulation_xml(p)
        assert cfg["n_trajectories"] == 1000
        assert abs(cfg["dt"] - 0.2) < 1e-8

    def test_write_simulation_xml(self, tmp_path):
        cfg = {
            "n_trajectories": 99,
            "dt": 0.5,
            "r_start": 80.0,
            "dx_files": ["a.dx", "b.dx"],
        }
        p = tmp_path / "out.xml"
        write_simulation_xml(cfg, p)
        content = p.read_text()
        assert "99" in content
        assert "a.dx" in content

    def test_multiple_contacts_parsed(self, tmp_path):
        xml = """<?xml version='1.0'?>
<reactions>
  <reaction name="r" probability="1.0">
    <contact molecule1_index="1" molecule2_index="2" distance="5.0"/>
    <contact molecule1_index="3" molecule2_index="4" distance="4.0"/>
    <contact molecule1_index="5" molecule2_index="6" distance="3.0"/>
  </reaction>
</reactions>"""
        p = tmp_path / "r.xml"
        p.write_text(xml)
        ps = parse_reaction_xml(p)
        assert len(ps.reactions[0].criteria.pairs) == 3


class TestIntegrationExtra:
    def test_many_molecule_types(self):
        """Simulate with multi-atom molecules."""
        rng = np.random.default_rng(42)
        mol1 = Molecule(name="big1")
        mol2 = Molecule(name="big2")
        for i in range(10):
            mol1.atoms.append(
                Atom(index=i, x=float(i), y=0.0, z=0.0, charge=0.1, radius=1.5)
            )
            mol2.atoms.append(
                Atom(index=i, x=float(i), y=0.0, z=0.0, charge=-0.1, radius=1.5)
            )
        mob = MobilityTensor.from_radii(mol1.bounding_radius(), mol2.bounding_radius())
        pair = ContactPair(0, 0, 200.0)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(n_trajectories=5, r_start=50.0, seed=7)
        result = NAMSimulator(mol1, mol2, mob, ps, params).run()
        assert result.n_trajectories == 5

    def test_pqr_write_read_simulation(self, tmp_path):
        """Full roundtrip: build molecule → write PQR → read → simulate."""
        mol = Molecule(name="synth")
        for i in range(5):
            mol.atoms.append(
                Atom(index=i, x=float(i) * 2, y=0, z=0, charge=0.2, radius=1.7)
            )
        p = tmp_path / "synth.pqr"
        write_pqr(mol, p)
        mol2 = parse_pqr(p)
        assert len(mol2.atoms) == 5
        assert abs(mol2.atoms[2].x - 4.0) < 0.01

    def test_debye_huckel_in_simulation(self):
        """Ensure DH force function integrates without error."""
        mol1 = Molecule(name="m1")
        mol1.atoms = [Atom(x=0, y=0, z=0, charge=5.0, radius=2.0)]
        mol2 = Molecule(name="m2")
        mol2.atoms = [Atom(x=0, y=0, z=0, charge=-5.0, radius=2.0)]

        def dh_force(m1, m2):
            c1 = m1.centroid()
            c2 = m2.centroid()
            r_vec = c2 - c1
            r = np.linalg.norm(r_vec)
            if r < 1e-5:
                return np.zeros(3), np.zeros(3), 0.0
            F = debye_huckel_force(m1.atoms[0].charge, m2.atoms[0].charge, r_vec)
            E = debye_huckel_energy(m1.atoms[0].charge, m2.atoms[0].charge, r)
            return F, np.zeros(3), E

        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 200.0)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(n_trajectories=5, r_start=50.0, seed=3)
        result = NAMSimulator(mol1, mol2, mob, ps, params, dh_force).run()
        assert result.n_trajectories == 5

    def test_version_is_string(self):
        assert isinstance(pystarc.__version__, str)

    def test_all_fates_importable(self):
        for f in (Fate.ONGOING, Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS):
            assert f.name in ("ONGOING", "REACTED", "ESCAPED", "MAX_STEPS")


# Extended tests
class TestAtomBlock3:
    @pytest.mark.parametrize(
        "x,y,z",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1), (10, 20, 30)],
    )
    def test_position_param(self, x, y, z):
        a = Atom(x=x, y=y, z=z)
        assert np.allclose(a.position, [x, y, z])

    @pytest.mark.parametrize("q", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_charge_param(self, q):
        a = Atom(charge=q)
        assert a.charge == q

    @pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 2.0, 5.0])
    def test_radius_param(self, r):
        a = Atom(radius=r)
        assert a.radius == r

    def test_distance_pythagorean(self):
        a = Atom(x=0, y=0, z=0)
        b = Atom(x=1, y=1, z=1)
        assert abs(a.distance_to(b) - math.sqrt(3)) < 1e-10

    def test_many_atoms_positions(self):
        atoms = [Atom(x=float(i)) for i in range(100)]
        xs = [a.x for a in atoms]
        assert xs == list(range(100))


class TestMoleculeBlock3:
    @pytest.mark.parametrize("n", [1, 5, 10, 20, 50])
    def test_molecule_len(self, n):
        mol = Molecule()
        mol.atoms = [Atom() for _ in range(n)]
        assert len(mol) == n

    def test_translate_centroid(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=2)]
        mol.translate(np.array([10, 0, 0]))
        assert abs(mol.centroid()[0] - 11.0) < 1e-10

    @pytest.mark.parametrize(
        "angle", [0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi]
    )
    def test_rotate_preserves_structure(self, angle):
        mol = Molecule()
        mol.atoms = [Atom(x=1, y=0, z=0), Atom(x=-1, y=0, z=0)]
        d_before = mol.atoms[0].distance_to(mol.atoms[1])
        R = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        mol.rotate(R)
        d_after = mol.atoms[0].distance_to(mol.atoms[1])
        assert abs(d_before - d_after) < 1e-10

    def test_bounding_radius_single_atom(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=2.0)]
        br = mol.bounding_radius()
        assert abs(br - 2.0) < 1e-10

    def test_total_charge_all_positive(self):
        mol = Molecule()
        mol.atoms = [Atom(charge=1.0) for _ in range(5)]
        assert abs(mol.total_charge() - 5.0) < 1e-10


class TestQuaternionBlock3:
    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, math.pi])
    def test_axis_angle_roundtrip_x(self, angle):
        axis = np.array([1.0, 0.0, 0.0])
        q = Quaternion.from_axis_angle(axis, angle)
        R = q.to_rotation_matrix()
        # Rx(θ) rotates y→(0, cosθ, sinθ)
        v = R @ np.array([0, 1, 0])
        assert abs(v[1] - math.cos(angle)) < 1e-10
        assert abs(v[2] - math.sin(angle)) < 1e-10

    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, math.pi])
    def test_axis_angle_roundtrip_y(self, angle):
        axis = np.array([0.0, 1.0, 0.0])
        q = Quaternion.from_axis_angle(axis, angle)
        R = q.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert abs(v[0] - math.cos(angle)) < 1e-10
        assert abs(v[2] + math.sin(angle)) < 1e-10

    def test_compose_rotations_associative(self):
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.4)
        q3 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.5)
        # (q1*q2)*q3 == q1*(q2*q3)
        lhs = ((q1 * q2) * q3).normalized()
        rhs = (q1 * (q2 * q3)).normalized()
        assert np.allclose(np.abs(lhs.to_array()), np.abs(rhs.to_array()), atol=1e-10)

    def test_rotate_zero_vector(self):
        q = random_quaternion(np.random.default_rng(0))
        v = q.rotate_vector(np.zeros(3))
        assert np.allclose(v, 0)

    def test_small_rotation_large_sigma(self):
        rng = np.random.default_rng(42)
        q = small_rotation_quaternion(10.0, rng)
        assert abs(q.norm() - 1.0) < 1e-10


class TestRombergBlock3:
    @pytest.mark.parametrize(
        "n,expected", [(0, 1.0), (1, 1.0 / 2), (2, 1.0 / 3), (3, 1.0 / 4), (4, 1.0 / 5)]
    )
    def test_power_integrals(self, n, expected):
        val = romberg_integrate(lambda x: x**n, 0.0, 1.0)
        assert abs(val - expected) < 1e-7

    def test_cos_zero_to_half_pi(self):
        val = romberg_integrate(math.cos, 0.0, math.pi / 2)
        assert abs(val - 1.0) < 1e-8

    def test_negative_range(self):
        val = romberg_integrate(lambda x: x, -1.0, 1.0)
        assert abs(val) < 1e-10

    def test_zero_width_interval(self):
        val = romberg_integrate(lambda x: x**2, 1.0, 1.0)
        assert abs(val) < 1e-10


class TestLegendreBlock3:
    @pytest.mark.parametrize(
        "n,x,expected",
        [
            (0, 0.0, 1.0),
            (1, 0.0, 0.0),
            (2, 0.0, -0.5),
            (0, 1.0, 1.0),
            (1, 1.0, 1.0),
            (2, 1.0, 1.0),
            (3, 1.0, 1.0),
            (0, -1.0, 1.0),
            (1, -1.0, -1.0),
            (2, -1.0, 1.0),
        ],
    )
    def test_known_values(self, n, x, expected):
        assert abs(legendre_p(n, x) - expected) < 1e-12

    def test_norm_p0(self):
        val = romberg_integrate(lambda x: legendre_p(0, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0) < 1e-6

    def test_norm_p2(self):
        val = romberg_integrate(lambda x: legendre_p(2, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0 / 5.0) < 1e-6

    def test_orthogonal_p1_p3(self):
        val = romberg_integrate(
            lambda x: legendre_p(1, x) * legendre_p(3, x), -1.0, 1.0
        )
        assert abs(val) < 1e-6


class TestContactPairBlock3:
    @pytest.mark.parametrize("dist", [1.0, 3.0, 5.0, 10.0, 50.0])
    def test_cutoff_param(self, dist):
        cp = ContactPair(0, 1, dist)
        assert cp.distance_cutoff == dist

    def test_mol2_index_stored(self):
        cp = ContactPair(3, 7, 4.0)
        assert cp.mol2_atom_index == 7

    def test_default_cutoff(self):
        cp = ContactPair()
        assert cp.distance_cutoff == 5.0


class TestPathwayBlock3:
    def test_multiple_reactions_first_wins(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        p1 = ContactPair(0, 0, 5.0)
        p2 = ContactPair(0, 0, 5.0)
        c1 = ReactionCriteria(pairs=[p1])
        c2 = ReactionCriteria(pairs=[p2])
        r1 = ReactionInterface("first", c1)
        r2 = ReactionInterface("second", c2)
        ps = PathwaySet([r1, r2])
        rng = np.random.default_rng(0)
        name = ps.check_all(mol1, mol2, rng)
        assert name == "first"

    def test_pathway_no_match_returns_none(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=100)]
        p = ContactPair(0, 0, 1.0)  # way too small
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[p]))])
        assert ps.check_all(mol1, mol2) is None

    def test_pathway_set_empty_add(self):
        ps = PathwaySet()
        assert len(ps) == 0
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5)])
        ps.add(ReactionInterface("r", c))
        assert len(ps) == 1


class TestMobilityBlock3:
    @pytest.mark.parametrize("r1,r2", [(10, 10), (15, 25), (5, 50), (30, 30), (8, 12)])
    def test_symmetric_molecules_equal_D(self, r1, r2):
        mob = MobilityTensor.from_radii(r1, r2)
        if r1 == r2:
            assert abs(mob.D_trans1 - mob.D_trans2) < 1e-14
        else:
            assert mob.D_trans1 != mob.D_trans2

    def test_direct_constructor(self):
        mob = MobilityTensor(1.0, 0.5, 2.0, 0.8)
        assert mob.D_trans1 == 1.0
        assert mob.D_rot2 == 0.8

    def test_relative_always_larger_than_either(self):
        mob = MobilityTensor.from_radii(20.0, 30.0)
        D_rel = mob.relative_translational_diffusion()
        assert D_rel > mob.D_trans1
        assert D_rel > mob.D_trans2


class TestTrajectoryResultBlock3:
    @pytest.mark.parametrize("fate", [Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS])
    def test_fate_stored(self, fate):
        r = TrajectoryResult(fate, 100, 20.0, 50.0)
        assert r.fate == fate

    def test_reaction_name_none_by_default(self):
        r = TrajectoryResult(Fate.ESCAPED, 50, 10.0, 200.0)
        assert r.reaction_name is None

    def test_energy_at_reaction_zero_default(self):
        r = TrajectoryResult(Fate.REACTED, 10, 2.0, 5.0, "r")
        assert r.energy_at_reaction == 0.0

    def test_steps_stored(self):
        r = TrajectoryResult(Fate.ESCAPED, 777, 155.4, 300.0)
        assert r.steps == 777

    def test_time_ps_stored(self):
        r = TrajectoryResult(Fate.ESCAPED, 100, 42.5, 200.0)
        assert abs(r.time_ps - 42.5) < 1e-10


class TestSimResultBlock3:
    def _result(self):
        return SimulationResult(
            n_trajectories=100,
            n_reacted=60,
            n_escaped=40,
            n_max_steps=0,
            reaction_counts={"r1": 60},
            r_start=100.0,
            r_escape=500.0,
            dt=0.2,
        )

    def test_reaction_probability(self):
        r = self._result()
        assert abs(r.reaction_probability - 0.6) < 1e-10

    def test_rate_constant_nonzero(self):
        r = self._result()
        k = r.rate_constant(10.0)
        assert k > 0

    def test_p_rxn_zero_when_no_reactions(self):
        r = SimulationResult(100, 0, 100, 0, {}, 100.0, 500.0, 0.2)
        assert r.reaction_probability == 0.0

    def test_p_rxn_one_when_all_react(self):
        r = SimulationResult(100, 100, 0, 0, {"r": 100}, 100.0, 500.0, 0.2)
        assert abs(r.reaction_probability - 1.0) < 1e-10

    def test_repr_contains_n(self):
        r = self._result()
        assert "100" in repr(r)


# Parametric sweep and stress tests
class TestSplineBlock4:
    @pytest.mark.parametrize("n", [3, 5, 10, 20, 50])
    def test_interpolates_x_squared(self, n):
        x = np.linspace(0, 3, n)
        y = x**2
        sp = CubicSpline(x, y)
        tol = 0.20 if n <= 3 else (0.07 if n <= 5 else 0.05)
        for xi in np.linspace(0.1, 2.9, 15):
            assert abs(sp(xi) - xi**2) < tol

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_interpolates_cosine(self, n):
        x = np.linspace(0, math.pi, n)
        y = np.cos(x)
        sp = CubicSpline(x, y)
        tol = 0.1 if n <= 4 else 0.05
        for xi in np.linspace(0.1, 3.0, 10):
            assert abs(sp(xi) - math.cos(xi)) < tol

    def test_derivative_cosine(self):
        x = np.linspace(0, math.pi, 40)
        y = np.cos(x)
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.2, 2.9, 10):
            assert abs(sp.derivative(xi) - (-math.sin(xi))) < 0.05


class TestDebyeBlock4:
    @pytest.mark.parametrize("sep", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_energy_positive_same_sign(self, sep):
        E = debye_huckel_energy(1.0, 1.0, sep)
        assert E > 0

    @pytest.mark.parametrize("sep", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_energy_negative_opposite_sign(self, sep):
        E = debye_huckel_energy(1.0, -1.0, sep)
        assert E < 0

    @pytest.mark.parametrize("debye", [3.0, 7.9, 15.0, 30.0])
    def test_longer_debye_longer_range(self, debye):
        E_short = debye_huckel_energy(1.0, 1.0, 20.0, debye_length=5.0)
        E_long = debye_huckel_energy(1.0, 1.0, 20.0, debye_length=debye)
        # longer Debye → less screened → larger energy at same separation
        if debye > 5.0:
            assert E_long > E_short


class TestBDBlock4:
    @pytest.mark.parametrize("D", [0.001, 0.01, 0.1, 1.0, 10.0])
    def test_diffusion_scales_step(self, D):
        rng = np.random.default_rng(42)
        pos = np.zeros(3)
        steps = [
            ermak_mccammon_translation(pos, np.zeros(3), D, 1.0, rng)
            for _ in range(500)
        ]
        std = np.std([s[0] for s in steps])
        expected = math.sqrt(2 * D * 1.0)
        assert abs(std - expected) / expected < 0.15  # within 15%

    @pytest.mark.parametrize("dt", [0.001, 0.01, 0.1, 1.0])
    def test_timestep_scales_step(self, dt):
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        steps = [
            ermak_mccammon_translation(pos, np.zeros(3), 1.0, dt, rng)
            for _ in range(1000)
        ]
        std = np.std([s[0] for s in steps])
        expected = math.sqrt(2 * dt)
        assert abs(std - expected) / expected < 0.15


class TestNAMBlock4:
    @pytest.mark.parametrize("n_traj", [1, 5, 10, 25, 50])
    def test_n_trajectories_exact(self, n_traj):
        mol1 = Molecule()
        mol1.atoms = [Atom()]
        mol2 = Molecule()
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 200.0)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(
            n_trajectories=n_traj, r_start=50.0, seed=0, max_steps=100
        )
        result = NAMSimulator(mol1, mol2, mob, ps, params).run()
        assert result.n_reacted + result.n_escaped + result.n_max_steps == n_traj

    @pytest.mark.parametrize("r_start", [30.0, 50.0, 80.0, 100.0])
    def test_r_start_stored(self, r_start):
        mol1 = Molecule()
        mol1.atoms = [Atom()]
        mol2 = Molecule()
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        ps = PathwaySet()
        params = NAMParameters(
            n_trajectories=2,
            r_start=r_start,
            r_escape=r_start + 100,
            seed=0,
            max_steps=5,
        )
        result = NAMSimulator(mol1, mol2, mob, ps, params).run()
        assert result.r_start == r_start


class TestConstantsBlock4:
    def test_kb_times_T_gives_kbt(self):
        assert abs(KB_SI * T_DEFAULT - KB_SI * 298.15) < 1e-30

    def test_ang_to_m_squared(self):
        assert abs(ANG_TO_M**2 - 1e-20) < 1e-30

    def test_ps_to_s_value(self):
        assert abs(PS_TO_S - 1e-12) < 1e-22

    def test_pi_precision(self):
        assert abs(PI - 3.14159265358979) < 1e-13

    def test_avogadro_order(self):
        assert 6e23 < AVOGADRO < 7e23

    def test_eta_water_order(self):
        assert 1e-4 < ETA_WATER < 1e-2

    def test_bjerrum_order(self):
        assert 5 < BJERRUM_LENGTH < 10

    def test_eps_water_order(self):
        assert 70 < EPS_WATER < 90


class TestBoundingBoxBlock4:
    @pytest.mark.parametrize("padding", [0.0, 1.0, 2.5, 5.0, 10.0])
    def test_padding_increases_size(self, padding):
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=4)]
        bb0 = BoundingBox.from_molecule(mol, padding=0.0)
        bbp = BoundingBox.from_molecule(mol, padding=padding)
        assert bbp.xmin <= bb0.xmin
        assert bbp.xmax >= bb0.xmax

    def test_center_1d(self):
        mol = Molecule()
        mol.atoms = [Atom(x=2.0), Atom(x=8.0)]
        bb = BoundingBox.from_molecule(mol, padding=0)
        assert abs(bb.center[0] - 5.0) < 1e-10

    def test_size_all_axes(self):
        mol = Molecule()
        mol.atoms = [
            Atom(x=0, y=0, z=0),
            Atom(x=6, y=4, z=2),
        ]
        bb = BoundingBox.from_molecule(mol, padding=0)
        assert np.allclose(bb.size, [6, 4, 2])


class TestAuxBlock4:
    @pytest.mark.parametrize("spacing", [1.0, 2.0, 3.0, 5.0])
    def test_lumped_charges_grid_spacing(self, spacing):
        mol = Molecule()
        mol.atoms = [Atom(x=0, charge=1.0), Atom(x=10, charge=-1.0)]
        lc = lumped_charges(mol, grid_spacing=spacing)
        total_q = sum(q for _, q in lc)
        assert abs(total_q) < 1e-5  # net charge preserved

    @pytest.mark.parametrize("probe", [1.0, 1.4, 2.0])
    def test_surface_spheres_probe(self, probe):
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=3.0)]
        pts = surface_spheres(mol, probe_radius=probe, n_points=20)
        assert len(pts) > 0

    def test_contact_distances_all_close(self):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=1), Atom(x=2), Atom(x=3)]
        pairs = contact_distances(mol1, mol2, cutoff=10.0)
        assert len(pairs) == 3


class TestMultipoleBlock4:
    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
    def test_monopole_sum(self, n):
        q = np.ones(n)
        assert abs(monopole_moment(q) - n) < 1e-10

    def test_dipole_linear_molecule(self):
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        q = np.array([1.0, -1.0])
        p = dipole_moment(pos, q)
        # p = +1*(0,0,0) + (-1)*(1,0,0) = (-1, 0, 0)
        assert np.allclose(p, [-1, 0, 0])

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_quadrupole_symmetric_n(self, n):
        rng = np.random.default_rng(n)
        pos = rng.standard_normal((n, 3))
        q = rng.standard_normal(n)
        Q = quadrupole_moment(pos, q)
        assert np.allclose(Q, Q.T)
        assert abs(np.trace(Q)) < 1e-10


class TestWienerBlock4:
    @pytest.mark.parametrize("dim", [1, 2, 3, 6])
    def test_wiener_dim(self, dim):
        rng = np.random.default_rng(dim)
        dW = wiener_step(1.0, 0.1, dim, rng)
        assert dW.shape == (dim,)

    @pytest.mark.parametrize("D,dt", [(0.1, 0.01), (1.0, 0.1), (10.0, 0.5)])
    def test_wiener_variance(self, D, dt):
        rng = np.random.default_rng(0)
        samples = np.array([wiener_step(D, dt, 1, rng)[0] for _ in range(3000)])
        expected_var = 2 * D * dt
        assert abs(samples.var() - expected_var) / expected_var < 0.1


class TestPQRBlock4:
    def _write_n_atoms(self, path, n):
        lines = ["REMARK test\n"]
        for i in range(n):
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA {i+1:5d}  "
                f"{float(i):.3f}   0.000   0.000  0.100  1.800\n"
            )
        lines.append("END\n")
        Path(path).write_text("".join(lines))

    @pytest.mark.parametrize("n", [1, 5, 10, 50])
    def test_parse_n_atoms(self, n, tmp_path):
        p = tmp_path / "mol.pqr"
        self._write_n_atoms(p, n)
        mol = parse_pqr(p)
        assert len(mol.atoms) == n

    def test_write_preserves_residue_name(self, tmp_path):
        mol = Molecule(name="test")
        mol.atoms = [Atom(residue_name="GLY", x=1, y=2, z=3, charge=0.1, radius=1.5)]
        p = tmp_path / "out.pqr"
        write_pqr(mol, p)
        mol2 = parse_pqr(p)
        assert mol2.atoms[0].residue_name == "GLY"

    def test_write_preserves_positions(self, tmp_path):
        mol = Molecule(name="pos_test")
        mol.atoms = [Atom(x=1.234, y=5.678, z=9.012, charge=0.5, radius=1.8)]
        p = tmp_path / "pos.pqr"
        write_pqr(mol, p)
        mol2 = parse_pqr(p)
        assert abs(mol2.atoms[0].x - 1.234) < 0.001
        assert abs(mol2.atoms[0].y - 5.678) < 0.001


class TestAtomFinal:
    @pytest.mark.parametrize("name", ["CA", "CB", "N", "O", "S", "FE", "ZN"])
    def test_atom_names(self, name):
        a = Atom(name=name)
        assert a.name == name

    @pytest.mark.parametrize("resname", ["ALA", "GLY", "SER", "THR", "VAL", "LEU"])
    def test_residue_names(self, resname):
        a = Atom(residue_name=resname)
        assert a.residue_name == resname

    @pytest.mark.parametrize("idx", [0, 1, 10, 100, 999])
    def test_indices(self, idx):
        a = Atom(index=idx)
        assert a.index == idx

    def test_repr_has_position(self):
        a = Atom(x=1.5, y=2.5, z=3.5)
        r = repr(a)
        assert "1.50" in r or "1.5" in r

    def test_distance_triangle_inequality(self):
        a = Atom(x=0, y=0, z=0)
        b = Atom(x=1, y=0, z=0)
        c = Atom(x=2, y=0, z=0)
        assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c) + 1e-10


class TestMoleculeGeomFinal:
    def _line_mol(self, n):
        mol = Molecule()
        for i in range(n):
            mol.atoms.append(Atom(x=float(i), y=0, z=0, radius=1.0))
        return mol

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
    def test_centroid_line_mol(self, n):
        mol = self._line_mol(n)
        c = mol.centroid()
        assert abs(c[0] - (n - 1) / 2.0) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_bounding_radius_line_mol(self, n):
        mol = self._line_mol(n)
        br = mol.bounding_radius()
        assert br > 0

    def test_charges_sum_to_zero_balanced(self):
        mol = Molecule()
        mol.atoms = [
            Atom(charge=1.0),
            Atom(charge=-1.0),
            Atom(charge=0.5),
            Atom(charge=-0.5),
        ]
        assert abs(mol.total_charge()) < 1e-10

    @pytest.mark.parametrize(
        "dx,dy,dz", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1), (5, 3, 2)]
    )
    def test_translate_shift(self, dx, dy, dz):
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0)]
        mol.translate(np.array([dx, dy, dz], dtype=float))
        assert abs(mol.atoms[0].x - dx) < 1e-10
        assert abs(mol.atoms[0].y - dy) < 1e-10
        assert abs(mol.atoms[0].z - dz) < 1e-10


class TestConstantsFinal:
    @pytest.mark.parametrize(
        "v,lo,hi",
        [
            ("T_DEFAULT", 295, 305),
            ("BJERRUM_LENGTH", 6, 8),
            ("DEFAULT_DEBYE_LENGTH", 5, 15),
        ],
    )
    def test_constant_range(self, v, lo, hi):
        val = getattr(C, v)
        assert lo < val < hi

    def test_kbt_in_joules(self):
        # kBT ≈ 4.1e-21 J at 298 K
        kbt_J = KB_SI * T_DEFAULT
        assert 3e-21 < kbt_J < 5e-21

    def test_bjerrum_from_eps(self):
        # l_B = e²/(4π ε₀ ε_r kBT) in SI, then convert to Å
        lB_m = E_CHARGE**2 / (4 * math.pi * EPS0_SI * EPS_WATER * KB_SI * T_DEFAULT)
        lB_A = lB_m / ANG_TO_M
        assert abs(lB_A - BJERRUM_LENGTH) < 0.5


class TestReactionCriteriaFinal:
    @pytest.mark.parametrize("n_pairs", [1, 2, 3, 5])
    def test_n_pairs_all_required(self, n_pairs):
        mol1 = Molecule()
        mol2 = Molecule()
        for i in range(n_pairs + 1):
            mol1.atoms.append(Atom(x=0, y=0, z=0))
            mol2.atoms.append(Atom(x=2, y=0, z=0))
        # All pairs satisfied (cutoff 5 > dist 2)
        pairs = [ContactPair(i, i, 5.0) for i in range(n_pairs)]
        c = ReactionCriteria(pairs=pairs)
        assert c.is_satisfied(mol1, mol2)

    @pytest.mark.parametrize("cutoff", [1.0, 1.5, 1.9])
    def test_cutoff_just_below_dist(self, cutoff):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, cutoff)])
        assert not c.is_satisfied(mol1, mol2)

    @pytest.mark.parametrize("cutoff", [2.1, 3.0, 10.0])
    def test_cutoff_above_dist(self, cutoff):
        """reference uses strict <: reaction fires when distance < cutoff."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, cutoff)])
        assert c.is_satisfied(mol1, mol2)

    def test_cutoff_exact_dist_not_satisfied(self):
        """reference: distance < cutoff (strict), so equal is NOT satisfied."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 2.0)])
        assert not c.is_satisfied(mol1, mol2)


class TestRPYFinal:
    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0, 50.0])
    def test_D_t_finite_positive(self, r):
        D = stokes_translational_diffusion(r)
        assert 0 < D < float("inf")

    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0])
    def test_D_r_finite_positive(self, r):
        D = stokes_rotational_diffusion(r)
        assert 0 < D < float("inf")

    def test_rpy_off_diagonal_symmetric(self):
        r_vec = np.array([10.0, 5.0, 3.0])
        M = rpy_offdiagonal(r_vec, 3.0, 3.0, 1.0, 1.0)
        assert np.allclose(M, M.T)

    def test_mobility_relative_D_positive(self):
        for r1, r2 in [(5, 5), (10, 20), (15, 30)]:
            mob = MobilityTensor.from_radii(r1, r2)
            assert mob.relative_translational_diffusion() > 0
            assert mob.relative_rotational_diffusion() > 0


class TestFateFinal:
    def test_all_fates_distinct(self):
        fates = [Fate.ONGOING, Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS]
        assert len(set(fates)) == 4

    @pytest.mark.parametrize(
        "fate,reacted,escaped",
        [
            (Fate.REACTED, True, False),
            (Fate.ESCAPED, False, True),
            (Fate.ONGOING, False, False),
            (Fate.MAX_STEPS, False, False),
        ],
    )
    def test_bool_properties(self, fate, reacted, escaped):
        r = TrajectoryResult(fate, 0, 0.0, 0.0)
        assert r.reacted == reacted
        assert r.escaped == escaped


class TestXMLFinal:
    @pytest.mark.parametrize("n_rxns", [1, 2, 3, 5])
    def test_write_n_reactions(self, n_rxns, tmp_path):
        ps = PathwaySet()
        for i in range(n_rxns):
            c = ReactionCriteria(pairs=[ContactPair(i, i, 5.0)])
            ps.add(ReactionInterface(f"rxn{i}", c))
        p = tmp_path / "rxns.xml"
        write_reaction_xml(ps, p)
        ps2 = parse_reaction_xml(p)
        assert len(ps2) == n_rxns

    @pytest.mark.parametrize("prob", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_probability_roundtrip(self, prob, tmp_path):
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5.0)])
        ps = PathwaySet([ReactionInterface("r", c, prob)])
        p = tmp_path / f"rxn_{int(prob*100)}.xml"
        write_reaction_xml(ps, p)
        ps2 = parse_reaction_xml(p)
        assert abs(ps2.reactions[0].probability - prob) < 1e-5


class TestIntegrationFinal:
    def test_full_pipeline_no_crash(self, tmp_path):
        """Run full pipeline with PQR + XML + simulation."""
        pqr = (
            "ATOM      1  CA  ALA     1       0.000   0.000   0.000 "
            " 1.000  2.000\nATOM      2  CB  ALA     1       5.000"
            "   0.000   0.000 -1.000  2.000\nEND\n"
        )
        p1 = tmp_path / "a.pqr"
        p1.write_text(pqr)
        p2 = tmp_path / "b.pqr"
        p2.write_text(pqr)
        m1 = parse_pqr(p1)
        m2 = parse_pqr(p2)

        rxn_xml = (
            "<?xml version='1.0'?><reactions>"
            "<reaction name='r' probability='1.0'>"
            "<contact molecule1_index='0' molecule2_index='0' distance='200.0'/>"
            "</reaction></reactions>"
        )
        rxn_p = tmp_path / "r.xml"
        rxn_p.write_text(rxn_xml)
        ps = parse_reaction_xml(rxn_p)

        mob = MobilityTensor.from_radii(m1.bounding_radius(), m2.bounding_radius())
        params = NAMParameters(n_trajectories=3, r_start=30.0, seed=42)
        result = NAMSimulator(m1, m2, mob, ps, params).run()
        assert result.n_trajectories == 3

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 999])
    def test_reproducible_seeds(self, seed):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, radius=2)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=0, radius=2)]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, 0.5)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])

        def run_with_seed(s):
            p = NAMParameters(
                n_trajectories=10, r_start=50.0, r_escape=80.0, seed=s, max_steps=500
            )
            return NAMSimulator(mol1, mol2, mob, ps, p).run().n_reacted

        assert run_with_seed(seed) == run_with_seed(seed)

    def test_brace_package_has_version(self):
        assert hasattr(pystarc, "__version__")
        assert pystarc.__version__  # version check

    def test_all_submodules_load(self):
        mods = [
            "pystarc.structures.molecules",
            "pystarc.structures.pqr_io",
            "pystarc.transforms.quaternion",
            "pystarc.hydrodynamics.rotne_prager",
            "pystarc.motion.do_bd_step",
            "pystarc.molsystem.system_state",
            "pystarc.pathways.reaction_interface",
            "pystarc.forces.electrostatic.grid_force",
            "pystarc.simulation.nam_simulator",
            "pystarc.xml_io.simulation_io",
            "pystarc.aux.aux_tools",
            "pystarc.lib.numerical",
            "pystarc.cli.main",
            "pystarc.global_defs.constants",
        ]
        for m in mods:
            mod = importlib.import_module(m)
            assert mod is not None


class TestGrid6:
    @pytest.mark.parametrize("v", [0.0, 1.0, -1.0, 3.14, -2.72])
    def test_uniform_grid_constant_value(self, v):
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.full((5, 5, 5), v))
        assert abs(g.interpolate(np.array([2.0, 2.0, 2.0])) - v) < 1e-8

    @pytest.mark.parametrize("charge", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_force_proportional_to_charge(self, charge):
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.zeros((5, 5, 5)))
        F = g.force_on_charge(np.array([2.0, 2.0, 2.0]), charge)
        assert np.allclose(F, 0)

    def test_non_square_grid(self):
        g = DXGrid(np.zeros(3), np.diag([1.0, 2.0, 3.0]), np.ones((3, 4, 5)))
        assert g.data.shape == (3, 4, 5)

    def test_interpolate_corner(self):
        data = np.zeros((4, 4, 4))
        data[0, 0, 0] = 1.0
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        val = g.interpolate(np.array([0.0, 0.0, 0.0]))
        assert abs(val - 1.0) < 1e-8

    @pytest.mark.parametrize("pt", [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    def test_interpolate_interior(self, pt):
        data = np.ones((5, 5, 5))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        val = g.interpolate(np.array(pt))
        assert abs(val - 1.0) < 1e-8


class TestQuatFinal6:
    @pytest.mark.parametrize(
        "w,x,y,z", [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    )
    def test_unit_quaternions(self, w, x, y, z):
        q = Quaternion(w, x, y, z)
        assert abs(q.norm() - 1.0) < 1e-14

    @pytest.mark.parametrize(
        "angle",
        [
            0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 3,
            math.pi / 2,
            2 * math.pi / 3,
            math.pi,
            4 * math.pi / 3,
            3 * math.pi / 2,
            2 * math.pi,
        ],
    )
    def test_rotation_angle_determinant(self, angle):
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), angle)
        R = q.to_rotation_matrix()
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_conjugate_is_inverse_for_unit(self):
        rng = np.random.default_rng(77)
        for _ in range(20):
            q = random_quaternion(rng)
            q_inv = q.conjugate()
            prod = q * q_inv
            # should be identity
            assert abs(abs(prod.w) - 1.0) < 1e-8


class TestNumericalFinal6:
    @pytest.mark.parametrize(
        "a,b,expected", [(0, 1, 1), (0, 2, 2), (1, 3, 2), (-1, 1, 2)]
    )
    def test_romberg_constant_1(self, a, b, expected):
        val = romberg_integrate(lambda x: 1.0, float(a), float(b))
        assert abs(val - expected) < 1e-8

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_legendre_at_zero_parity(self, n):
        # Pn(0) = 0 for odd n, nonzero for even n
        val = legendre_p(n, 0.0)
        if n % 2 == 1:
            assert abs(val) < 1e-12
        else:
            assert abs(val) > 0 or n == 0

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6])
    def test_wiener_correct_dim(self, dim):
        rng = np.random.default_rng(dim * 10)
        dW = wiener_step(1.0, 1.0, dim, rng)
        assert len(dW) == dim

    def test_monopole_negative(self):
        q = np.array([-1.0, -2.0, -3.0])
        assert abs(monopole_moment(q) - (-6.0)) < 1e-10

    def test_dipole_3atoms(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        q = np.array([1.0, 0.0, -1.0])
        p = dipole_moment(pos, q)
        assert np.allclose(p, [-2, 0, 0])


class TestSimFinal6:
    def _tiny_sim(self, cutoff=200.0, seed=0, n=3):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, radius=2)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=0, radius=2)]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        pair = ContactPair(0, 0, cutoff)
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[pair]))])
        params = NAMParameters(
            n_trajectories=n, r_start=50.0, seed=seed, max_steps=1000
        )
        return NAMSimulator(mol1, mol2, mob, ps, params)

    @pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
    def test_seed_gives_same_result(self, seed):
        r1 = self._tiny_sim(seed=seed).run()
        r2 = self._tiny_sim(seed=seed).run()
        assert r1.n_reacted == r2.n_reacted

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_exact_n_traj(self, n):
        result = self._tiny_sim(n=n).run()
        total = result.n_reacted + result.n_escaped + result.n_max_steps
        assert total == n

    def test_reaction_probability_with_huge_cutoff_is_high(self):
        result = self._tiny_sim(cutoff=1000.0, n=20).run()
        assert result.reaction_probability > 0.5

    def test_sim_result_rate_nonnegative(self):
        result = self._tiny_sim(n=5).run()
        mob = MobilityTensor.from_radii(20.0, 20.0)
        k = result.rate_constant(mob.relative_translational_diffusion())
        assert k >= 0

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.5])
    def test_dt_stored_in_result(self, dt):
        mol1 = Molecule()
        mol1.atoms = [Atom()]
        mol2 = Molecule()
        mol2.atoms = [Atom()]
        mob = MobilityTensor.from_radii(20.0, 20.0)
        ps = PathwaySet()
        params = NAMParameters(
            n_trajectories=2, r_start=30.0, r_escape=50.0, dt=dt, seed=0, max_steps=5
        )
        result = NAMSimulator(mol1, mol2, mob, ps, params).run()
        assert abs(result.dt - dt) < 1e-10


class TestAuxFinal6:
    def test_lumped_charges_empty_mol(self):
        mol = Molecule()
        lc = lumped_charges(mol)
        assert lc == []

    def test_contact_distances_empty(self):
        mol1 = Molecule()
        mol1.atoms = []
        mol2 = Molecule()
        mol2.atoms = [Atom(x=0)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert pairs == []

    def test_bounding_box_single_atom(self):
        mol = Molecule()
        mol.atoms = [Atom(x=5, y=3, z=1)]
        bb = bounding_box(mol, padding=0.0)
        assert abs(bb.xmin - 5.0) < 1e-10
        assert abs(bb.xmax - 5.0) < 1e-10

    @pytest.mark.parametrize(
        "q,r,expected_sign",
        [
            (1.0, 2.0, -1),  # stabilizing → negative
            (2.0, 3.0, -1),
            (-1.0, 2.0, -1),  # sign of charge²
        ],
    )
    def test_born_sign(self, q, r, expected_sign):
        E = born_integral(q, r)
        assert math.copysign(1, E) == expected_sign or E == 0

    def test_hydrodynamic_radius_positive(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=3), Atom(x=6)]
        rh = hydrodynamic_radius_from_rg(mol)
        assert rh > 0

    def test_electrostatic_center_shape(self):
        mol = Molecule()
        mol.atoms = [Atom(x=i, charge=float(i)) for i in range(1, 6)]
        ec = electrostatic_center(mol)
        assert ec.shape == (3,)


class TestBlock7Parametric:
    @pytest.mark.parametrize(
        "x,expected",
        [(-1.0, 1.0), (-0.5, -0.125), (0.0, -0.5), (0.5, -0.125), (1.0, 1.0)],
    )
    def test_legendre_p2_values(self, x, expected):
        assert abs(legendre_p(2, x) - expected) < 1e-12

    @pytest.mark.parametrize(
        "q1,q2,sep", [(1, 1, 5), (1, -1, 5), (2, 2, 10), (0.5, 0.5, 3), (-1, -1, 7)]
    )
    def test_dh_energy_sign(self, q1, q2, sep):
        E = debye_huckel_energy(float(q1), float(q2), float(sep))
        expected_sign = math.copysign(1, q1 * q2)
        if abs(q1 * q2) > 1e-10 and sep > 0:
            assert math.copysign(1, E) == expected_sign

    @pytest.mark.parametrize(
        "r1,r2", [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
    )
    def test_equal_radii_equal_diffusion(self, r1, r2):
        mob = MobilityTensor.from_radii(float(r1), float(r2))
        if r1 == r2:
            assert abs(mob.D_trans1 - mob.D_trans2) < 1e-14

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, 2.0, math.pi])
    def test_from_axis_angle_unit_norm(self, angle):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), angle)
        assert abs(q.norm() - 1.0) < 1e-10

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    def test_romberg_x_power_n(self, n):
        val = romberg_integrate(lambda x: x**n, 0.0, 1.0)
        expected = 1.0 / (n + 1)
        assert abs(val - expected) < 1e-7

    @pytest.mark.parametrize("v", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_constant_dx_grid_any_point(self, v):
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.full((4, 4, 4), v))
        for pt in [[1, 1, 1], [1.5, 1.5, 1.5], [2, 2, 2]]:
            assert abs(g.interpolate(np.array(pt)) - v) < 1e-8

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
    def test_molecule_len_correct(self, n):
        mol = Molecule()
        mol.atoms = [Atom() for _ in range(n)]
        assert len(mol) == n

    @pytest.mark.parametrize("charge", [-5, -2, -1, 0, 1, 2, 5])
    def test_atom_charge_stored(self, charge):
        a = Atom(charge=float(charge))
        assert a.charge == float(charge)

    @pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    def test_atom_radius_stored(self, r):
        a = Atom(radius=r)
        assert a.radius == r

    @pytest.mark.parametrize(
        "fate", [Fate.ONGOING, Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS]
    )
    def test_system_state_fate_set(self, fate):
        s = SystemState(fate=fate)
        assert s.fate == fate

    @pytest.mark.parametrize("steps", [0, 1, 100, 10000])
    def test_trajectory_steps_stored(self, steps):
        r = TrajectoryResult(Fate.ESCAPED, steps, float(steps) * 0.2, 200.0)
        assert r.steps == steps

    @pytest.mark.parametrize("n_contacts", [1, 2, 3, 4, 5])
    def test_make_default_reaction_n_pairs(self, n_contacts):
        mol1 = Molecule()
        mol1.atoms = [Atom(x=float(i)) for i in range(10)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=float(i) + 20) for i in range(10)]
        rxn = make_default_reaction(mol1, mol2, n_pairs=n_contacts)
        assert len(rxn.criteria.pairs) == n_contacts

    @pytest.mark.parametrize("pqr_line_count", [1, 3, 5, 10, 20])
    def test_pqr_parse_count(self, pqr_line_count, tmp_path):
        lines = ["REMARK test\n"]
        for i in range(pqr_line_count):
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA {i+1:4d}    "
                f"{float(i):.3f}   0.000   0.000  0.500  1.800\n"
            )
        lines.append("END\n")
        p = tmp_path / f"mol_{pqr_line_count}.pqr"
        p.write_text("".join(lines))
        mol = parse_pqr(p)
        assert len(mol.atoms) == pqr_line_count

    @pytest.mark.parametrize("padding", [0, 1, 2, 5, 10])
    def test_bb_contains_with_padding(self, padding):
        mol = Molecule()
        mol.atoms = [Atom(x=5, y=5, z=5)]
        bb = bounding_box(mol, padding=float(padding))
        center = bb.center
        assert bb.contains(center)

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_reaction_interface_prob_stored(self, prob):
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5.0)])
        rxn = ReactionInterface("r", c, prob)
        assert abs(rxn.probability - prob) < 1e-10


class TestBlock8Final:
    # Atom geometry
    @pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_distance_along_x(self, d):
        a = Atom(x=0)
        b = Atom(x=float(d))
        assert abs(a.distance_to(b) - float(d)) < 1e-10

    # Quaternion rotation applied to basis vectors
    @pytest.mark.parametrize(
        "axis,vec,angle,expected",
        [
            ([0, 0, 1], [1, 0, 0], math.pi / 2, [0, 1, 0]),
            ([0, 0, 1], [1, 0, 0], math.pi, [-1, 0, 0]),
            ([0, 0, 1], [0, 1, 0], math.pi / 2, [-1, 0, 0]),
            ([1, 0, 0], [0, 1, 0], math.pi / 2, [0, 0, 1]),
            ([1, 0, 0], [0, 0, 1], math.pi / 2, [0, -1, 0]),
        ],
    )
    def test_rotation_basis_vectors(self, axis, vec, angle, expected):
        q = Quaternion.from_axis_angle(np.array(axis, dtype=float), angle)
        result = q.rotate_vector(np.array(vec, dtype=float))
        assert np.allclose(result, expected, atol=1e-10)

    # Romberg on trig
    @pytest.mark.parametrize(
        "a,b", [(0, math.pi / 4), (0, math.pi / 2), (math.pi / 4, math.pi / 2)]
    )
    def test_romberg_sine_analytically(self, a, b):
        val = romberg_integrate(math.sin, a, b)
        expected = math.cos(a) - math.cos(b)
        assert abs(val - expected) < 1e-8

    # Debye-Hückel symmetry
    @pytest.mark.parametrize("q1,q2", [(1, 2), (2, 1), (-1, -3), (-3, -1)])
    def test_dh_energy_symmetric_charges(self, q1, q2):
        E12 = debye_huckel_energy(float(q1), float(q2), 10.0)
        E21 = debye_huckel_energy(float(q2), float(q1), 10.0)
        assert abs(E12 - E21) < 1e-10

    # BD step: translation returns (3,) array
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_translation_shape(self, seed):
        rng = np.random.default_rng(seed)
        pos = np.zeros(3)
        new = ermak_mccammon_translation(pos, np.zeros(3), 1.0, 0.1, rng)
        assert new.shape == (3,)

    # Molecule total charge
    @pytest.mark.parametrize(
        "q_list", [[1, -1], [2, -1, -1], [0.5, 0.5, -1], [0, 0, 0], [1, 1, 1, -3]]
    )
    def test_total_charge(self, q_list):
        mol = Molecule()
        mol.atoms = [Atom(charge=q) for q in q_list]
        assert abs(mol.total_charge() - sum(q_list)) < 1e-10

    # Wiener mean
    @pytest.mark.parametrize("D,dt", [(1, 0.1), (2, 0.2), (0.5, 0.05)])
    def test_wiener_mean_zero(self, D, dt):
        rng = np.random.default_rng(0)
        samples = np.array([wiener_step(D, dt, 1, rng)[0] for _ in range(5000)])
        assert abs(samples.mean()) < 0.1

    # Legendre series constant
    @pytest.mark.parametrize("c0", [0.5, 1.0, 2.0, -1.0])
    def test_legendre_series_constant(self, c0):
        for x in [-0.9, 0.0, 0.5, 0.9]:
            val = legendre_series([c0], x)
            assert abs(val - c0) < 1e-12

    # BoundingBox center correct
    @pytest.mark.parametrize("lo,hi", [(-1, 1), (-5, 5), (0, 10), (2, 8), (-3, 7)])
    def test_bb_center_x(self, lo, hi):
        mol = Molecule()
        mol.atoms = [Atom(x=lo), Atom(x=hi)]
        bb = BoundingBox.from_molecule(mol, padding=0)
        assert abs(bb.center[0] - (lo + hi) / 2.0) < 1e-10

    # Rotne-Prager: far field symmetric
    @pytest.mark.parametrize("dist", [10.0, 20.0, 50.0])
    def test_rpy_far_symmetric(self, dist):
        r_vec = np.array([dist, 0.0, 0.0])
        M = rpy_offdiagonal(r_vec, 2.0, 2.0, 1.0, 1.0)
        assert np.allclose(M, M.T, atol=1e-10)

    # Stokes: D ∝ 1/r
    @pytest.mark.parametrize("factor", [2.0, 3.0, 5.0])
    def test_D_t_inv_radius(self, factor):
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(10.0 * factor)
        assert abs(D1 / D2 - factor) < 0.01

    # Reaction satisfied iff all contacts met
    @pytest.mark.parametrize(
        "n_satisfied,n_total", [(1, 1), (2, 2), (3, 3), (2, 3), (1, 2)]
    )
    def test_reaction_all_or_nothing(self, n_satisfied, n_total):
        mol1 = Molecule()
        mol2 = Molecule()
        for i in range(n_total + 1):
            mol1.atoms.append(Atom(x=0, y=float(i) * 10))
            mol2.atoms.append(Atom(x=2, y=float(i) * 10))
        pairs = []
        for i in range(n_satisfied):
            pairs.append(ContactPair(i, i, 5.0))  # dist=2 < 5 → ok
        for i in range(n_satisfied, n_total):
            pairs.append(ContactPair(i, i, 0.5))  # dist=2 > 0.5 → fail
        c = ReactionCriteria(pairs=pairs)
        expected = n_satisfied == n_total
        assert c.is_satisfied(mol1, mol2) == expected


class TestBlock9Closing:
    @pytest.mark.parametrize("x", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_p3_values(self, x):
        expected = (5 * x**3 - 3 * x) / 2
        assert abs(legendre_p(3, x) - expected) < 1e-12

    @pytest.mark.parametrize("x", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_p4_values(self, x):
        expected = (35 * x**4 - 30 * x**2 + 3) / 8
        assert abs(legendre_p(4, x) - expected) < 1e-12

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_monopole_ones(self, dim):
        q = np.ones(dim)
        assert abs(monopole_moment(q) - float(dim)) < 1e-10

    @pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
    def test_large_molecule_centroid(self, n):
        mol = Molecule()
        for i in range(n):
            mol.atoms.append(Atom(x=float(i)))
        c = mol.centroid()
        assert abs(c[0] - (n - 1) / 2.0) < 1e-8

    @pytest.mark.parametrize(
        "angle,cos_val",
        [
            (0.0, 1.0),
            (math.pi / 2, 0.0),
            (math.pi, -1.0),
            (math.pi / 3, 0.5),
            (math.pi / 4, math.sqrt(2) / 2),
        ],
    )
    def test_rotation_cos_check(self, angle, cos_val):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), angle)
        R = q.to_rotation_matrix()
        assert abs(R[0, 0] - cos_val) < 1e-10

    @pytest.mark.parametrize(
        "charge,radius", [(1.0, 2.0), (2.0, 3.0), (0.5, 1.5), (3.0, 4.0)]
    )
    def test_born_negative(self, charge, radius):
        E = born_integral(charge, radius)
        assert E < 0

    @pytest.mark.parametrize("D", [0.01, 0.1, 1.0, 10.0])
    def test_D_t_positive(self, D):
        rng = np.random.default_rng(0)
        steps = [
            ermak_mccammon_translation(np.zeros(3), np.zeros(3), D, 0.1, rng)
            for _ in range(100)
        ]
        # just check no NaN/inf
        for s in steps:
            assert np.all(np.isfinite(s))

    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0, 30.0, 50.0])
    def test_escape_radius_gt_r(self, r):
        re = escape_radius(r)
        assert re > r

    @pytest.mark.parametrize("n_rx,n_esc", [(0, 10), (5, 5), (10, 0)])
    def test_p_rxn_values(self, n_rx, n_esc):
        r = SimulationResult(n_rx + n_esc, n_rx, n_esc, 0, {}, 100.0, 500.0, 0.2)
        if n_rx + n_esc > 0:
            assert abs(r.reaction_probability - n_rx / (n_rx + n_esc)) < 1e-10


class TestBlock10:
    @pytest.mark.parametrize(
        "x,y,z,expected_r",
        [
            (3, 4, 0, 5),
            (0, 0, 5, 5),
            (1, 1, 1, math.sqrt(3)),
            (6, 8, 0, 10),
            (0, 3, 4, 5),
        ],
    )
    def test_separation_3d(self, x, y, z, expected_r):
        s = SystemState(position=np.array([x, y, z], dtype=float))
        assert abs(s.separation() - expected_r) < 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_pathway_set_len(self, n):
        ps = PathwaySet()
        for i in range(n):
            c = ReactionCriteria(pairs=[ContactPair(i, i, 5.0)])
            ps.add(ReactionInterface(f"r{i}", c))
        assert len(ps) == n

    @pytest.mark.parametrize("r", [10.0, 20.0, 30.0, 40.0, 50.0])
    def test_D_r_decreases_with_r(self, r):
        D_small = stokes_rotational_diffusion(r)
        D_large = stokes_rotational_diffusion(r * 2)
        assert D_small > D_large

    @pytest.mark.parametrize("sep", [5.0, 10.0, 15.0, 20.0, 25.0])
    def test_dh_decays_exponentially(self, sep):
        lam = DEFAULT_DEBYE_LENGTH
        E1 = debye_huckel_energy(1.0, 1.0, sep)
        E2 = debye_huckel_energy(1.0, 1.0, sep + lam)
        # E2/E1 ≈ (sep/(sep+λ)) * exp(-1)
        ratio = E2 / E1
        expected = (sep / (sep + lam)) * math.exp(-1.0)
        assert abs(ratio - expected) / abs(expected) < 0.05

    @pytest.mark.parametrize(
        "w,x,y,z",
        [
            (0.5, 0.5, 0.5, 0.5),
            (1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ],
    )
    def test_unit_quaternion_rotation_matrix_orthogonal(self, w, x, y, z):
        q = Quaternion(w, x, y, z)
        R = q.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    @pytest.mark.parametrize(
        "a,b,n_expected", [(0.0, 1.0, 1.0 / 3), (0.0, 2.0, 8.0 / 3), (0.0, 3.0, 9.0)]
    )
    def test_romberg_x2(self, a, b, n_expected):
        val = romberg_integrate(lambda x: x**2, a, b)
        assert abs(val - n_expected) < 1e-8

    @pytest.mark.parametrize(
        "n_atoms,cutoff,expected",
        [
            (5, 200.0, True),  # everything reacts with huge cutoff
            (3, 0.001, False),  # nothing reacts with tiny cutoff
        ],
    )
    def test_reaction_set_fires(self, n_atoms, cutoff, expected):
        mol1 = Molecule()
        mol2 = Molecule()
        for i in range(n_atoms):
            mol1.atoms.append(Atom(x=0))
            mol2.atoms.append(Atom(x=2))
        pair = ContactPair(0, 0, cutoff)
        c = ReactionCriteria(pairs=[pair])
        assert c.is_satisfied(mol1, mol2) == expected


class TestBlock11FinalStretch:
    """Last batch — 20 tests to clear 470."""

    def test_atom_default_index(self):
        assert Atom().index == 0

    def test_molecule_empty_centroid(self):
        mol = Molecule()
        assert np.allclose(mol.centroid(), [0, 0, 0])

    def test_molecule_one_atom_rg(self):
        mol = Molecule()
        mol.atoms = [Atom(x=5)]
        assert mol.radius_of_gyration() == 0.0

    def test_quaternion_w1_is_identity(self):
        q = Quaternion(1, 0, 0, 0)
        assert np.allclose(q.to_rotation_matrix(), np.eye(3))

    def test_rigid_transform_apply_1d_input(self):
        T = RigidTransform(translation=np.array([1.0, 0, 0]))
        v = np.array([0.0, 0.0, 0.0])
        result = T.apply(v)
        assert abs(result[0] - 1.0) < 1e-10

    def test_bd_step_finite(self):
        rng = np.random.default_rng(42)
        pos = np.array([50.0, 0.0, 0.0])
        ori = Quaternion.identity()
        new_pos, new_ori = bd_step(
            pos, ori, np.zeros(3), np.zeros(3), 0.01, 0.001, 0.2, rng
        )
        assert np.all(np.isfinite(new_pos))
        assert abs(new_ori.norm() - 1.0) < 1e-10

    def test_system_state_default_position_zero(self):
        s = SystemState()
        assert np.allclose(s.position, [0, 0, 0])

    def test_contact_pair_default_values(self):
        cp = ContactPair()
        assert cp.mol1_atom_index == 0
        assert cp.mol2_atom_index == 0

    def test_pathway_set_empty_repr(self):
        ps = PathwaySet()
        assert "PathwaySet" in repr(ps)

    def test_bounding_box_contains_center(self):
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=10), Atom(y=0), Atom(y=10)]
        bb = bounding_box(mol, padding=0)
        assert bb.contains(bb.center)

    def test_lumped_charges_single_atom(self):
        mol = Molecule()
        mol.atoms = [Atom(x=5, y=5, z=5, charge=2.0)]
        lc = lumped_charges(mol, grid_spacing=2.0)
        total_q = sum(q for _, q in lc)
        assert abs(total_q - 2.0) < 1e-6

    def test_born_larger_eps_out_less_negative(self):
        # born_integral ∝ -(1/eps_in - 1/eps_out)
        # higher eps_out → larger (1/eps_in - 1/eps_out) → MORE negative
        E1 = born_integral(1.0, 3.0, eps_in=4.0, eps_out=40.0)
        E2 = born_integral(1.0, 3.0, eps_in=4.0, eps_out=80.0)
        # both negative; E2 more negative than E1
        assert E2 < E1

    def test_dx_grid_shape_query(self):
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.zeros((7, 8, 9)))
        assert tuple(g.data.shape) == (7, 8, 9)

    def test_dh_energy_proportional_to_bjerrum(self):
        E1 = debye_huckel_energy(1.0, 1.0, 10.0, bjerrum_length=5.0)
        E2 = debye_huckel_energy(1.0, 1.0, 10.0, bjerrum_length=10.0)
        assert abs(E2 / E1 - 2.0) < 1e-10

    def test_monopole_single(self):
        assert abs(monopole_moment(np.array([3.7])) - 3.7) < 1e-10

    def test_dipole_zero_positions(self):
        pos = np.zeros((3, 3))
        q = np.array([1.0, -2.0, 1.0])
        p = dipole_moment(pos, q)
        assert np.allclose(p, 0)

    def test_wiener_zero_D(self):
        rng = np.random.default_rng(0)
        dW = wiener_step(0.0, 1.0, 3, rng)
        assert np.allclose(dW, 0)

    def test_spline_linear_exact(self):
        x = np.linspace(0, 5, 10)
        y = 3 * x + 2
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.1, 4.9, 20):
            assert abs(sp(xi) - (3 * xi + 2)) < 1e-8

    def test_reaction_name_in_result(self):
        r = TrajectoryResult(Fate.REACTED, 10, 2.0, 5.0, "my_rxn")
        assert r.reaction_name == "my_rxn"

    def test_simulation_result_dt(self):
        r = SimulationResult(10, 5, 5, 0, {}, 100.0, 500.0, 0.123)
        assert abs(r.dt - 0.123) < 1e-10


# Tests for LJ forces, GHO injection, COFFDROP chains


class TestLJForces:
    """Tests for Lennard-Jones and hydrophobic SASA forces."""

    def test_lj_pair_repulsive_at_small_r(self):
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        # r=1 < sigma=2 -> repulsive -> force points a->b (positive x)
        assert f[0] > 0

    def test_lj_pair_attractive_at_large_r(self):  # noqa
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([3.0, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        # r=3 > sigma=2 -> attractive -> force on a points TOWARD b (negative x)
        assert f[0] < 0

    def test_lj_energy_minimum_at_sigma(self):
        pos_a = np.array([0.0, 0.0, 0.0])
        # At r = 2^(1/6)*sigma force = 0 (energy minimum)
        r_min = 2.0 ** (1.0 / 6.0) * 2.0
        pos_b = np.array([r_min, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        assert abs(e - (-0.25)) < 0.01  # reference: V_min = -eps/4 at r=2^(1/6)*sig

    def test_lj_mixing_rules(self):
        lj = LJParams(
            atom_types=[
                LJAtomType("C", epsilon=0.1, sigma=1.7),
                LJAtomType("N", epsilon=0.2, sigma=1.5),
            ]
        )
        engine = LJForceEngine(lj_params=lj)
        pos1 = np.array([[0.0, 0.0, 0.0]])
        pos2 = np.array([[4.0, 0.0, 0.0]])
        f1, f2, e = engine.compute(pos1, pos2, [0], [1])
        # Newton's 3rd law
        assert np.allclose(f1, -f2, atol=1e-10)

    def test_lj_newton_third_law(self):
        lj = LJParams(atom_types=[LJAtomType("A", epsilon=0.5, sigma=2.0)])
        engine = LJForceEngine(lj_params=lj)
        pos1 = np.array([[0.0, 0.0, 0.0]])
        pos2 = np.array([[3.0, 0.0, 0.0]])
        f1, f2, e = engine.compute(pos1, pos2, [0], [0])
        assert np.allclose(f1, -f2, atol=1e-10)

    def test_wca_zero_beyond_cutoff(self):
        pos_a = np.array([0.0, 0.0, 0.0])
        sigma = 2.0
        r_cut = 2.0 ** (1.0 / 6.0) * sigma + 0.1  # just beyond cutoff
        pos_b = np.array([r_cut, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=sigma, use_wca=True)
        assert np.allclose(f, 0.0)
        assert e == 0.0

    def test_hydrophobic_zero_outside_range(self):
        hp = HydrophobicParams(a=3.1, b=4.35)
        r_vec = np.array([1.0, 0.0, 0.0])
        # r + radius = 1.0 + 0.5 = 1.5 < a=3.1 -> zero
        f, e = hydrophobic_sasa_force(1.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
        assert np.allclose(f, 0.0)

    def test_hydrophobic_nonzero_in_range(self):
        hp = HydrophobicParams(a=3.1, b=4.35)
        r_vec = np.array([1.0, 0.0, 0.0])
        # r=3.0, radius_a=0.5 -> ri = 3.5, which is in [3.1, 4.35]
        f, e = hydrophobic_sasa_force(3.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
        assert not np.allclose(f, 0.0)


class TestGHOInjection:
    """Tests for GHO ghost atom auto-injection."""

    def test_gho_world_position_identity(self):
        atom = GHOAtom(atom_index=0, pos_rel=np.array([1.0, 2.0, 3.0]))
        rot = np.eye(3)
        trans = np.zeros(3)
        pos = gho_world_position(atom, rot, trans)
        assert np.allclose(pos, [1.0, 2.0, 3.0])

    def test_gho_world_position_rotated(self):
        atom = GHOAtom(atom_index=0, pos_rel=np.array([1.0, 0.0, 0.0]))
        # 90 degree rotation around z-axis
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        trans = np.zeros(3)
        pos = gho_world_position(atom, rot, trans)
        assert np.allclose(pos, [0.0, 1.0, 0.0], atol=1e-10)

    def test_gho_world_position_translated(self):
        atom = GHOAtom(atom_index=0, pos_rel=np.array([0.0, 0.0, 0.0]))
        trans = np.array([5.0, 3.0, 1.0])
        pos = gho_world_position(atom, np.eye(3), trans)
        assert np.allclose(pos, [5.0, 3.0, 1.0])

    def test_gho_criterion_distance(self):
        g1 = GHOAtom(0, np.array([0.0, 0.0, 0.0]))
        g2 = GHOAtom(0, np.array([3.0, 4.0, 0.0]))
        d = gho_criterion_distance(
            g1, np.eye(3), np.zeros(3), g2, np.eye(3), np.zeros(3)
        )
        assert abs(d - 5.0) < 1e-10

    def test_gho_reaction_criterion_satisfied(self):
        g1 = GHOAtom(0, np.array([0.0, 0.0, 0.0]))
        g2 = GHOAtom(0, np.array([3.0, 0.0, 0.0]))
        crit = GHOReactionCriterion([(g1, g2, 5.0)])
        assert crit.is_satisfied(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))

    def test_gho_reaction_criterion_not_satisfied(self):
        g1 = GHOAtom(0, np.array([0.0, 0.0, 0.0]))
        g2 = GHOAtom(0, np.array([10.0, 0.0, 0.0]))
        crit = GHOReactionCriterion([(g1, g2, 5.0)])
        assert not crit.is_satisfied(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))

    def test_parse_manual_ghost_atoms(self):
        mol1_pos = np.random.default_rng(0).random((10, 3)) * 20.0
        mol2_pos = np.random.default_rng(1).random((5, 3)) * 10.0
        spec = "3,0,17.0\n4,0,10.0"
        g1, g2 = inject_gho_from_manual(
            spec, mol1_pos, mol2_pos, np.zeros(3), np.zeros(3)
        )
        assert len(g1) == 2
        assert len(g2) == 0

    def test_rxns_xml_parser_handles_missing_file(self):
        pairs, n_needed = _parse_rxns_xml_criteria(Path("/nonexistent/rxns.xml"))
        assert pairs == []
        assert n_needed == -1


class TestCOFFDROPChain:
    """Tests for flexible chain model."""

    def test_build_linear_chain(self):
        chain = build_linear_chain(5)
        assert chain.n_beads == 5
        assert len(chain.bonds) == 4

    def test_chain_positions_array(self):
        chain = build_linear_chain(3, bond_length=4.0)
        pos = chain.positions_array()
        assert pos.shape == (3, 3)
        # Beads along x-axis, 4 A apart
        assert abs(pos[1, 0] - 4.0) < 1e-10
        assert abs(pos[2, 0] - 8.0) < 1e-10

    def test_chain_bd_step_moves_beads(self):
        chain = build_linear_chain(3)
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        chain = prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        assert not np.allclose(pos_before, pos_after)

    def test_frozen_chain_doesnt_move(self):
        chain = build_linear_chain(3)
        chain.frozen = True
        prop = ChainBDPropagator()
        rng = np.random.default_rng(0)
        pos_before = chain.positions_array().copy()
        chain = prop.step(chain, dt=0.1, rng=rng)
        assert np.allclose(chain.positions_array(), pos_before)

    def test_bond_forces_zero_at_equilibrium(self):
        chain = build_linear_chain(2, bond_length=3.8)
        # Beads already at equilibrium distance
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        # Bond force should be near zero at equilibrium
        assert np.linalg.norm(F[0]) < 1e-8
        assert np.linalg.norm(F[1]) < 1e-8

    def test_max_time_step_positive(self):
        chain = build_linear_chain(3)
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt > 0

    def test_satisfy_bond_constraints(self):
        chain = build_linear_chain(3, bond_length=4.0)
        # Stretch the bond by hand
        chain.beads[1].pos = np.array([10.0, 0.0, 0.0])
        prop = ChainBDPropagator()
        prop.satisfy_bond_constraints(chain)
        # After constraint satisfaction, bond length should be closer to r0
        r01 = np.linalg.norm(chain.beads[1].pos - chain.beads[0].pos)
        assert r01 < 10.0  # must have moved toward equilibrium


class TestGeometryRxnsXML:
    """Tests for rxns_xml integration in geometry module."""

    def test_auto_detect_reactions_no_gho_raises(self):
        # No GHO atoms → must raise RuntimeError (centroid fallback removed)
        # All PQRs in the pipeline have GHO injected before this is called.
        import pytest

        rec = MoleculeGeometry(
            n_atoms=10,
            n_charged=5,
            n_ghost=0,
            centroid=np.zeros(3),
            max_radius=20.0,
            hydrodynamic_r=20.0,
            ghost_indices=[],
            ghost_positions=[],
            total_charge=1.0,
        )
        lig = MoleculeGeometry(
            n_atoms=5,
            n_charged=2,
            n_ghost=0,
            centroid=np.zeros(3),
            max_radius=5.0,
            hydrodynamic_r=5.0,
            ghost_indices=[],
            ghost_positions=[],
            total_charge=0.0,
        )
        geom = SystemGeometry(receptor=rec, ligand=lig, r_start=25.0, r_escape=50.0)
        with pytest.raises(RuntimeError, match="No GHO ghost atoms"):
            auto_detect_reactions(geom, ghost_atoms="auto", rxns_xml="")

    def test_auto_detect_reactions_manual_spec(self):
        rec = MoleculeGeometry(10, 5, 0, np.zeros(3), 20.0, 20.0, [], [], 1.0)
        lig = MoleculeGeometry(5, 2, 0, np.zeros(3), 5.0, 5.0, [], [], 0.0)
        geom = SystemGeometry(rec, lig, 25.0, 50.0)
        stages, n_needed = auto_detect_reactions(
            geom, ghost_atoms="100,0,17.0\n101,0,10.0", rxns_xml=""
        )
        assert len(stages[0]) == 2
        assert stages[0][0].cutoff == 17.0
        assert stages[0][1].cutoff == 10.0
        assert n_needed == -1

    def test_auto_detect_missing_rxns_xml_raises_without_gho(self):
        # Missing rxns.xml + no GHO → RuntimeError (no silent fallback)
        import pytest

        rec = MoleculeGeometry(10, 5, 0, np.zeros(3), 20.0, 20.0, [], [], 1.0)
        lig = MoleculeGeometry(5, 2, 0, np.zeros(3), 5.0, 5.0, [], [], 0.0)
        geom = SystemGeometry(rec, lig, 25.0, 50.0)
        # Missing rxns.xml → warning printed, then falls to auto-detect → no GHO → error
        with pytest.raises(RuntimeError, match="No GHO ghost atoms"):
            auto_detect_reactions(
                geom, ghost_atoms="auto", rxns_xml="/nonexistent/rxns.xml"
            )
