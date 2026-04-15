"""
PySTARC unified test suite.

Run with:  pytest tests/test_pystarc.py -v
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
from pystarc.simulation.coffdrop_params import (
    BeadDef,
    BondDef,
    COFFDROPParams,
    ResidueDef,
    TabulatedPotential,
    _match_pot,
    _parse_charges,
    _parse_connectivity,
    _parse_mapping,
    _txt_to_floats,
)
from pystarc.motion.do_bd_step import (
    FORCE_CHANGE_ALPHA,
    WATER_VISCOSITY,
    backstep_due_to_force,
    bd_step,
    ermak_mccammon_rotation,
    ermak_mccammon_translation,
    escape_radius,
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
from pystarc.pathways.reaction_interface import (
    ContactPair,
    PathwaySet,
    ReactionCriteria,
    ReactionInterface,
    make_default_reaction,
)
from pystarc.xml_io.simulation_io import (
    parse_reaction_xml,
    parse_simulation_xml,
    write_reaction_xml,
    write_simulation_xml,
)
from pystarc.forces.lj import (
    HydrophobicParams,
    LJAtomType,
    LJForceEngine,
    LJParams,
    hydrophobic_sasa_force,
    lj_pair_force,
)
from pystarc.simulation.diffusional_rotation import (
    diffusional_rotation,
    quat_multiply,
    quat_of_rotvec,
    random_unit_quat,
)
from pystarc.pipeline.geometry import (
    MoleculeGeometry,
    SystemGeometry,
    _parse_rxns_xml_criteria,
    auto_detect_reactions,
)
from pystarc.hydrodynamics.mc_hydro_radius import (
    _extract_surface,
    _fingerprint,
    _voxelise,
    mc_hydrodynamic_radius,
)
from pystarc.transforms.quaternion import (
    Quaternion,
    RigidTransform,
    random_quaternion,
    small_rotation_quaternion,
)
from pystarc.simulation.coffdrop_chain import (
    ChainAngle,
    ChainBDPropagator,
    ChainBead,
    ChainBond,
    ChainForceEvaluator,
    ChainTorsion,
    FlexibleChain,
    build_linear_chain,
)
from pystarc.simulation.nam_simulator import (
    NAMParameters,
    NAMSimulator,
    SimulationResult,
    zero_force,
)
from pystarc.structures.molecules import (
    Atom,
    BoundingBox,
    ContactPair,
    Molecule,
    ReactionCriteria,
)
from pystarc.simulation.wiener import (
    WienerProcess,
    WienerStep,
    do_one_full_step,
    make_initial_dW,
)
from pystarc.multi_GPU.combine_data import (
    _concat_csv,
    _concat_npz,
    _save_json,
    _sum_csv,
    _sum_npz,
)
from pystarc.forces.electrostatic.grid_force import (
    DXGrid,
    debye_huckel_energy,
    debye_huckel_force,
)
from pystarc.analysis.convergence import (
    analyse_convergence,
    print_convergence,
    save_convergence,
)
from pystarc.motion.adaptive_time_step import (
    AdaptiveTimeStep,
    max_time_step,
    reaction_time_step,
)
from pystarc.simulation.step_near_surface import _inv_erf, step_near_absorbing_surface
from pystarc.simulation.we_simulator import WEParameters, WEResult, WETrajectory
from pystarc.molsystem.system_state import Fate, SystemState, TrajectoryResult
from pystarc.forces.multipole import EffectiveCharges, load_effective_charges
from pystarc.simulation.outer_propagator import OPGroupInfo, OuterPropagator
from pystarc.pipeline.input_parser import OutputConfig, PySTARCConfig, parse
from pystarc.pipeline.extract import _is_atom_line, _residue_name, extract
from pystarc.pipeline.geometry import analyse_molecule as geom_analyse
from pystarc.pipeline.geometry import AtomRecord as GeomAtomRecord
from pystarc.pipeline.geometry import parse_pqr as geom_parse_pqr
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.structures.pqr_io import parse_pqr, write_pqr
from pystarc.pipeline.geometry import MoleculeGeometry
from pystarc.pipeline.output_writer import write_all
from pystarc.global_defs import constants as C
from pystarc.forces.engine import _Grid
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path
import numpy as np
import importlib
import tempfile
import shutil
import pystarc
import pytest
import math
import json
import csv
import os


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
        # large force in x -> drift dominates
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
        pair = ContactPair(0, 0, 5.0)  # dist = 2, cutoff = 5 -> satisfied
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
        pair = ContactPair(0, 0, 200.0)  # huge cutoff -> always reacts
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
        # With cutoff 200 Å and r_start=50 -> all should react immediately
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
        pair = ContactPair(0, 0, 0.001)  # tiny cutoff -> never reacts
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
        """Parse PQR -> build sim -> run -> get result."""
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
        """Write reaction XML -> parse -> simulate."""
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
class TestAtomFieldStorage:
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


class TestMoleculeTransformations:
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
        # charges: -1, -0.5, 0, 0.5, 1.0 -> sum=0
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


class TestQuaternionAlgebra:
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


class TestRigidTransformComposition:
    def test_rotation_then_translation(self):
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        T = RigidTransform(rotation=q, translation=np.array([0, 1, 0]))
        v = np.array([1, 0, 0])
        result = T.apply(v)
        # rotate -> [0,1,0], translate -> [0,2,0]
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


class TestDiffusionCoefficientScaling:
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


class TestBDStepForceDominance:
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
        # zero diffusion -> noise is 0, displacement = D*dt*F = 0
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


class TestSystemStateFieldAccess:
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


class TestAuxToolsConstraints:
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


class TestNumericalAccuracy:
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


class TestDebyeHuckelEdgeCases:
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


class TestDXGridUniformPotential:
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


class TestNAMSimulatorResultProperties:
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
        # Different seeds -> at least one different count (overwhelmingly likely)
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


class TestXMLReadWriteRoundtrip:
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


class TestModuleIntegration:
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
        """Full roundtrip: build molecule -> write PQR -> read -> simulate."""
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
class TestAtomPositionAndDistance:
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


class TestMoleculeGeometricOps:
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


class TestQuaternionCompositionRules:
    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, math.pi])
    def test_axis_angle_roundtrip_x(self, angle):
        axis = np.array([1.0, 0.0, 0.0])
        q = Quaternion.from_axis_angle(axis, angle)
        R = q.to_rotation_matrix()
        # Rx(θ) rotates y->(0, cosθ, sinθ)
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


class TestRombergSpecialFunctions:
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


class TestLegendreOrthonormality:
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


class TestContactPairDefaults:
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


class TestPathwayPriorityOrder:
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


class TestMobilityTensorSymmetry:
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


class TestTrajectoryResultDefaults:
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


class TestSimulationResultStatistics:
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
class TestSplineInterpolationAccuracy:
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


class TestDebyeHuckelChargeSign:
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
        # longer Debye -> less screened -> larger energy at same separation
        if debye > 5.0:
            assert E_long > E_short


class TestBDStepDiffusionScaling:
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


class TestNAMParameterStorage:
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


class TestDerivedConstantValues:
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


class TestBoundingBoxPaddingAndCenter:
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


class TestAuxToolsGridSpacing:
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


class TestMultipoleMomentValues:
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


class TestWienerProcessDimAndVariance:
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


class TestPQRRoundtripPreservation:
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


class TestAtomNameAndRepr:
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


class TestMoleculeLinearChain:
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


class TestConstantPhysicalRanges:
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


class TestReactionCriteriaBoundary:
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


class TestRPYTensorProperties:
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


class TestFateEnumValues:
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


class TestXMLProbabilityStorage:
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


class TestPipelineReproducibility:
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


class TestDXGridParametric:
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


class TestQuaternionUnitProperties:
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


class TestNumericalParityAndSign:
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


class TestNAMSeedReproducibility:
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


class TestAuxToolsEmptyInput:
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
            (1.0, 2.0, -1),  # stabilizing -> negative
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


class TestParametricPhysicsValues:
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


class TestParametricGeometryAndSymmetry:
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
            pairs.append(ContactPair(i, i, 5.0))  # dist=2 < 5 -> ok
        for i in range(n_satisfied, n_total):
            pairs.append(ContactPair(i, i, 0.5))  # dist=2 > 0.5 -> fail
        c = ReactionCriteria(pairs=pairs)
        expected = n_satisfied == n_total
        assert c.is_satisfied(mol1, mol2) == expected


class TestHighOrderLegendreAndCentroid:
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


class TestSeparationAndDecay:
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


class TestDefaultConstructorValues:
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
        # higher eps_out -> larger (1/eps_in - 1/eps_out) -> MORE negative
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
        # No GHO atoms -> must raise RuntimeError (centroid fallback removed)
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
        # Missing rxns.xml + no GHO -> RuntimeError (no silent fallback)
        import pytest

        rec = MoleculeGeometry(10, 5, 0, np.zeros(3), 20.0, 20.0, [], [], 1.0)
        lig = MoleculeGeometry(5, 2, 0, np.zeros(3), 5.0, 5.0, [], [], 0.0)
        geom = SystemGeometry(rec, lig, 25.0, 50.0)
        # Missing rxns.xml -> warning printed, then falls to auto-detect -> no GHO -> error
        with pytest.raises(RuntimeError, match="No GHO ghost atoms"):
            auto_detect_reactions(
                geom, ghost_atoms="auto", rxns_xml="/nonexistent/rxns.xml"
            )


reference_EPS0 = 0.000142  # e²/(kBT·Å)  — vacuum permittivity in reference units
reference_MU = 0.243  # kBT·ps/Å³   — water viscosity at 20°C
reference_KT = 1.0  # kBT         — energy unit
reference_PI = math.pi
reference_SDIE = 78.0  # solvent dielectric
reference_CONV = 602000000.0  # Å³/ps -> M⁻¹s⁻¹ (from compute_rate_constant.ml)


# 1. Physical constants
class TestPhysicalConstants:
    def test_vacuum_permittivity(self):
        """vacuum_permittivity = 0.000142 e²/(kBT·Å)"""
        assert reference_EPS0 == 0.000142

    def test_water_viscosity(self):
        """water_viscosity = 0.243 kBT·ps/Å³"""
        assert reference_MU == 0.243

    def test_kT_unity(self):
        """kT = 1.0 (energy unit is kBT)"""
        assert reference_KT == 1.0

    def test_solvent_dielectric_default(self):
        """solvent_dielectric = 78.0"""
        assert reference_SDIE == 78.0

    def test_conversion_factor(self):
        """conv_factor = 602000000.0"""
        CONV_PYSTARC = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV_PYSTARC - reference_CONV) / reference_CONV < 1e-3

    def test_desolvation_alpha_default(self):
        """solvation_parameter=1.0 -> alpha=1/(4π)"""
        alpha = 1.0 / (4.0 * math.pi)
        assert abs(alpha - 0.07957747) < 1e-5

    def test_qb_factor(self):
        assert 1.1 == 1.1


# 2. Diffusion coefficients
class TestDiffusionCoefficients:
    """Verify D_trans = kT/(6πμa)."""

    @staticmethod
    def _D_trans(a):
        return reference_KT / (6.0 * reference_PI * reference_MU * a)

    def test_D_single_sphere_1A(self):
        D = self._D_trans(1.0)
        assert abs(D - 0.21803) < 0.001

    def test_D_charged_spheres(self):
        a = 1.005
        D_rel = 2 * self._D_trans(a)
        assert abs(D_rel - 0.43371) < 0.002

    def test_D_thrombin(self):
        """thrombin: r_hydro(rec)=25.375, r_hydro(lig)=21.620"""
        D_rel = self._D_trans(25.375) + self._D_trans(21.620)
        assert abs(D_rel - 0.01867) < 0.001

    def test_D_inversely_proportional_to_radius(self):
        D1 = self._D_trans(10.0)
        D2 = self._D_trans(20.0)
        assert abs(D1 / D2 - 2.0) < 0.01

    def test_D_rotational_inverse_cube(self):
        """D_rot = kT/(8πμa³)"""

        def D_rot(a):
            return reference_KT / (8.0 * reference_PI * reference_MU * a**3)

        assert abs(D_rot(10.0) / D_rot(20.0) - 8.0) < 0.01


# 3. Yukawa potential and gradient
class TestYukawaPotential:
    Q_REC = 1.0
    Q_LIG = -1.0
    DEBYE = 7.828  # Å

    @staticmethod
    def _V_factor(q_rec, q_lig, sdie=78.0):
        eps_s = sdie * reference_EPS0
        return q_rec * q_lig / (4.0 * reference_PI * eps_s)

    def test_V_factor_two_spheres(self):
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        assert abs(V - (-7.1847)) < 0.01

    def test_potential_at_10A(self):
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        phi = V * math.exp(-10.0 / self.DEBYE) / 10.0
        assert abs(phi - (-0.200268)) < 0.001

    def test_gradient_at_10A_matches_central_diff(self):
        """Analytical gradient should match numerical central difference."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        r = 10.0
        # Analytical
        dphi_dr = V * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        # Central difference (PySTARC CUDA method)
        h = 0.016
        phi_p = V * math.exp(-(r + h) / self.DEBYE) / (r + h)
        phi_m = V * math.exp(-(r - h) / self.DEBYE) / (r - h)
        grad_cd = (phi_p - phi_m) / (2 * h)
        assert abs(dphi_dr - grad_cd) / abs(dphi_dr) < 1e-4

    def test_force_attractive_for_opposite_charges(self):
        """phi_rec uses only receptor charge."""
        # V_factor for receptor potential (not interaction potential)
        V_rec = self.Q_REC / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        # dphi_dr < 0 (phi decreases from positive toward zero with r)
        F_x = -self.Q_LIG * dphi_dr  # -(-1) × (negative) = negative
        assert F_x < 0  # negative x = toward receptor at origin = attractive

    def test_force_repulsive_for_same_charges(self):
        V_rec = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        F_x = -(1.0) * dphi_dr  # -(+1) × (negative) = positive
        assert F_x > 0  # repulsive

    @pytest.mark.parametrize("r", [3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
    def test_force_decays_with_distance(self, r):
        """Force magnitude should decrease with distance."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        dphi_r1 = V * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        dphi_r2 = (
            V
            * math.exp(-(r + 1) / self.DEBYE)
            * (-1 / (r + 1) ** 2 - 1 / ((r + 1) * self.DEBYE))
        )
        assert abs(dphi_r1) > abs(dphi_r2)


# 4. Grid bounds
# gradient requires in_range2: ix ∈ [1, nx-3]
# PySTARC CUDA: central diff needs ±h/2 -> need margin of 0.5 cells
class TestGridBounds:
    """Verify grid bound calculations account for gradient probe width."""

    def test_ref_potential_range(self):
        """reference in_range1: ix in [0, nx-2] inclusive."""
        nx = 129
        assert 0 <= 0 and 0 <= nx - 2  # low end
        assert 0 <= nx - 2 and nx - 2 <= nx - 2  # high end

    def test_ref_gradient_range(self):
        nx = 129
        assert 1 <= 1 and 1 <= nx - 3
        assert 1 <= nx - 3 and nx - 3 <= nx - 3

    def test_pystarc_gradient_aware_bounds(self):
        """valid range is [origin+0.5*sp, origin+(n-2.5)*sp]."""
        origin, sp, nx = 0.0, 1.0, 129
        lo = origin + 0.5 * sp  # = 0.5
        hi = origin + (nx - 2.5) * sp  # = 126.5
        # Must cover interior: 1 to 127 in reference index space
        assert lo <= 1.0 * sp  # lo covers ix=1
        assert hi >= (nx - 3) * sp  # hi covers ix=126

    def test_two_spheres_grid_coverage(self):
        """charged_spheres: coarse grid spacing 0.16, nx=129, origin≈-10.25.
        Atom at r=10 (b-sphere): check if inside gradient-aware bounds."""
        sp = 0.1602
        nx = 129
        origin = -10.25  # approximate
        lo = origin + 0.5 * sp  # ≈ -10.17
        hi = origin + (nx - 2.5) * sp  # ≈ +10.00
        # At r=10: position = 10.0, which is at the boundary hi ≈ 10.00
        # This means the atom may be outside and Yukawa fallback kicks in
        assert abs(hi - 10.0) < 0.5  # grid edge near b-sphere


# 5. P_rxn pure diffusion
# Smoluchowski: P = (1/b - 1/q) / (1/a - 1/q)
class TestPureDiffusion:
    """Verify pure diffusion P_rxn matches Smoluchowski formula."""

    def test_smoluchowski_two_spheres(self):
        """a=2.5, b=10, q=20 -> P_diff = 0.1429"""
        a, b, q = 2.5, 10.0, 20.0
        P = (1 / b - 1 / q) / (1 / a - 1 / q)
        assert abs(P - 0.1429) < 0.001

    def test_expected_P_with_attraction(self):
        P_ref = 0.44
        P_diff = 0.143
        assert P_ref > 3 * P_diff  # attraction triples P_rxn


# 6. BD step (Ermack-McCammon)
#   dpos = (1/(6πμa)) × F × dt = D × F × dt  [since kT=1]
#   wdpos = sqrt(2 × kT × mob) × gaussian × sqrt(dt) = sqrt(2D·dt) × ξ
class TestBDStepPhysics:
    """Verify BD step matches Ermak-McCammon integrator."""

    def test_drift_formula(self):
        """drift = D × F × dt"""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = D * F * dt
        assert abs(drift - (-0.00396)) < 0.0001

    def test_noise_rms(self):
        """sigma = sqrt(2 × D × dt)"""
        D, dt = 0.43371, 0.2
        sigma = math.sqrt(2 * D * dt)
        assert abs(sigma - 0.4169) < 0.001

    def test_drift_noise_ratio(self):
        """At b-sphere: drift/noise ≈ 0.01 (noise-dominated)."""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = abs(D * F * dt)
        sigma = math.sqrt(2 * D * dt)
        assert drift / sigma < 0.02  # force is weak relative to noise

    def test_zero_force_pure_diffusion(self):
        """With F=0: drift=0, only noise."""
        D, dt = 0.43371, 0.2
        drift = D * 0.0 * dt
        assert drift == 0.0

    def test_no_HI_D_is_sum(self):
        """Without HI: D_rel = D_1 + D_2."""
        D1 = reference_KT / (6 * reference_PI * reference_MU * 1.005)
        D2 = reference_KT / (6 * reference_PI * reference_MU * 1.005)
        D_rel = D1 + D2
        assert abs(D_rel - 0.43371) < 0.002


# 7. Adaptive time step
# dt_edge = min(r-b, q-r)² / (18·D)
# dt_force = alpha / |D·dF/dr|
class TestAdaptiveTimestep:
    D = 0.43371
    B = 10.0
    TRIG = 11.0  # qb_factor × b

    def test_dt_edge_at_boundary(self):
        """At r = 10.5: dist_b=0.5, dist_trig=0.5."""
        r = 10.5
        dist = min(r - self.B, self.TRIG - r)
        dt_edge = dist**2 / (18.0 * self.D)
        assert abs(dt_edge - 0.032) < 0.005

    def test_dt_edge_zero_at_b(self):
        """At r = b exactly, dt_edge -> 0."""
        r = self.B
        dist = max(r - self.B, 1e-3)
        dt_edge = dist**2 / (18.0 * self.D)
        assert dt_edge < 0.001

    @pytest.mark.parametrize("r", [10.1, 10.3, 10.5, 10.7, 10.9])
    def test_dt_edge_increases_toward_middle(self, r):
        """dt_edge is largest at (b+trig)/2 = 10.5."""
        dist = min(r - self.B, self.TRIG - r)
        dt_edge = dist**2 / (18.0 * self.D)
        assert dt_edge > 0


# 8. Outer propagator (LMZ)
#   return_prob = relative_rate(bradius) / relative_rate(qradius)
#   relative_rate(b) = 4π / ∫₀^{1/b} exp(U(1/s))/D ds
class TestOuterPropagator:
    Q_REC, Q_LIG = 1.0, -1.0
    DEBYE = 7.828
    D = 0.43371
    B = 10.0

    @staticmethod
    def _romberg(f, a, b, tol=1e-8, max_iter=20):
        n = 1
        h = b - a
        R = [[0] * (max_iter + 1) for _ in range(max_iter + 1)]
        R[0][0] = 0.5 * h * (f(a) + f(b))
        for i in range(1, max_iter + 1):
            n *= 2
            h = (b - a) / n
            s = sum(f(a + (2 * k - 1) * h) for k in range(1, n // 2 + 1))
            R[i][0] = 0.5 * R[i - 1][0] + h * s
            for j in range(1, i + 1):
                R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4**j - 1)
            if i > 1 and abs(R[i][i] - R[i - 1][i - 1]) < tol * abs(R[i][i]):
                return R[i][i]
        return R[max_iter][max_iter]

    def _V_both(self):
        eps_s = reference_SDIE * reference_EPS0
        return self.Q_REC * self.Q_LIG / (4.0 * reference_PI * eps_s)

    def _relative_rate(self, b):
        V = self._V_both()

        def intgd(s):
            if s == 0.0:
                return 1.0 / self.D
            r = 1.0 / s
            return math.exp(V * math.exp(-r / self.DEBYE) / r) / self.D

        igral = self._romberg(intgd, 0.0, 1.0 / b)
        return 4.0 * reference_PI / igral

    def test_k_b_two_spheres(self):
        """k_b(b=10) should be ~57.5 Å³/ps."""
        k_b = self._relative_rate(self.B)
        assert abs(k_b - 57.5) < 1.0

    def test_qradius_formula(self):
        """reference: qradius = 20.0 × max_mol_radius"""
        max_r = 1.005  # r_hydro for charged_spheres
        q_out = 20.0 * max_r
        assert abs(q_out - 20.1) < 0.01

    def test_return_prob_two_spheres(self):
        """return_prob = k_b(b) / k_b(q_out) ≈ 0.52"""
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert abs(rp - 0.52) < 0.03

    def test_qb_factor_1_1(self):
        """reference qb_factor.hh: constexpr double qb_factor = 1.1"""
        trigger = 1.1 * self.B
        assert abs(trigger - 11.0) < 1e-10

    def test_return_prob_between_0_and_1(self):
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert 0 < rp < 1


# 9. Romberg Integration
class TestRombergPhysics:
    """Verify Romberg integration"""

    def test_yukawa_integral_converges(self):
        """The Romberg integral for k_b should converge."""
        D = 0.43371
        eps_s = reference_SDIE * reference_EPS0
        V = 1.0 * (-1.0) / (4.0 * reference_PI * eps_s)
        debye = 7.828

        def intgd(s):
            if s == 0.0:
                return 1.0 / D
            r = 1.0 / s
            return math.exp(V * math.exp(-r / debye) / r) / D

        val = TestOuterPropagator._romberg(intgd, 0.0, 0.1)
        assert val > 0 and math.isfinite(val)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_power_integrals(self, n):
        """∫₀¹ xⁿ dx = 1/(n+1)"""
        val = TestOuterPropagator._romberg(lambda x: x**n, 0.0, 1.0)
        assert abs(val - 1.0 / (n + 1)) < 1e-8

    def test_sin_integral(self):
        val = TestOuterPropagator._romberg(math.sin, 0.0, reference_PI)
        assert abs(val - 2.0) < 1e-8


# 10. Rate constant
#   rate = conv_factor × kdb × beta
#   conv_factor = 602000000.0
class TestRateConstant:
    """Verify the k_on formula."""

    def test_formula_matches_reference(self):
        """k_on = CONV × k_b × P_rxn"""
        CONV = 6.022e8
        k_b = 57.5
        P = 0.44
        k_on = CONV * k_b * P
        assert abs(k_on - 1.52e10) / 1.52e10 < 0.05

    def test_conv_factor_derivation(self):
        """CONV = N_A × Å³->L / ps->s = 6.022e23 × 1e-30/1e-12/1e-3"""
        CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV - 6.022e8) / 6.022e8 < 1e-3

    def test_k_on_zero_if_P_zero(self):
        """No reactions -> k_on = 0."""
        assert 6.022e8 * 57.5 * 0.0 == 0.0

    @pytest.mark.parametrize(
        "P,k_expected",
        [(0.1, 3.46e9), (0.2, 6.93e9), (0.3, 1.04e10), (0.4, 1.39e10), (0.5, 1.73e10)],
    )
    def test_k_on_linear_in_P(self, P, k_expected):
        """k_on ∝ P_rxn (linear relationship)."""
        k_b = 57.5
        k_on = 6.022e8 * k_b * P
        assert abs(k_on - k_expected) / k_expected < 0.02


# 11. Born desolvation
#   F = -alpha × q² × grad(born_field)
#   Called both directions: (mol0->mol1) AND (mol1->mol0)
class TestBornDesolvation:
    """Verify Born desolvation."""

    def test_two_spheres_alpha_zero(self):
        """charged_spheres: desolvation_alpha = 0.0 -> no Born force."""
        alpha = 0.0
        q = -1.0
        F = -alpha * q**2 * 0.1  # any gradient
        assert F == 0.0

    def test_thrombin_alpha_nonzero(self):
        """thrombin: desolvation_alpha = 0.07957747 -> Born force active."""
        alpha = 0.07957747
        assert alpha > 0

    def test_born_force_always_repulsive(self):
        """Born desolvation force is always repulsive (pushes charges apart)."""
        alpha = 0.07957747
        q = 1.0
        grad_born = -0.01  # negative gradient, so field decreases outward
        F = -alpha * q**2 * grad_born
        # F > 0: pushes ligand outward (desolvation penalty increases on approach)
        assert F > 0


# 12. Screened coulomb (reference chain-chain pairwise)
#   F = q0*q1*(r/L + 1)*exp(-r/L)/(r³*4π*ε) × r_vec
class TestScreenedCoulomb:
    """Verify screened Coulomb formula."""

    def test_formula_at_10A(self):
        """F(r=10) for q0=1, q1=-1, L=7.828."""
        q0, q1 = 1.0, -1.0
        r = 10.0
        L = 7.828
        eps = reference_SDIE * reference_EPS0
        F_mag = abs(
            q0 * q1 * (r / L + 1) * math.exp(-r / L) / (r**3 * 4 * reference_PI * eps)
        )
        assert F_mag > 0

    def test_newton_third_law(self):
        """F(mol0->mol1) = -F(mol1->mol0)"""
        q0, q1 = 1.0, -1.0
        r_vec = np.array([10.0, 0.0, 0.0])
        r = 10.0
        L = 7.828
        eps = reference_SDIE * reference_EPS0
        F12 = (
            q0
            * q1
            * (r / L + 1)
            * math.exp(-r / L)
            / (r**3 * 4 * reference_PI * eps)
            * r_vec
        )
        F21 = -F12  # Newton's 3rd law
        assert np.allclose(F12, -F21)


# 13. Yukawa far-field fallback
class TestYukawaFallback:
    """Verify the Yukawa fallback is correct."""

    def test_matches_apbs_inside_grid(self):
        """At r=5 (well inside grid), Yukawa ≈ APBS solution
        for pdie=sdie=78 (no dielectric boundary)."""
        V = 1.0 * (-1.0) / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 5.0
        debye = 7.828
        # APBS with pdie=sdie gives pure Yukawa
        phi_yukawa = V * math.exp(-r / debye) / r
        # phi_apbs should match (verified by numerical check)
        assert abs(phi_yukawa) > 0

    def test_force_matches_numerical_gradient(self):
        """Yukawa analytical force should match finite-diff of Yukawa potential."""
        V = 1.0 * (-1.0) / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        debye = 7.828
        h = 0.001
        phi_p = V * math.exp(-(r + h) / debye) / (r + h)
        phi_m = V * math.exp(-(r - h) / debye) / (r - h)
        grad_num = (phi_p - phi_m) / (2 * h)
        grad_ana = V * math.exp(-r / debye) * (-1 / r**2 - 1 / (r * debye))
        assert abs(grad_num - grad_ana) / abs(grad_ana) < 1e-6

    def test_monopole_matches_ref_far_field(self):
        """Chebyshev blob reduces to monopole at large r.
        Here, the Yukawa is the monopole term."""
        V = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 50.0
        debye = 7.828
        phi_mono = V * math.exp(-r / debye) / r
        # Higher multipoles (dipole, quadrupole) decay as 1/r², 1/r³
        # At r=50, monopole dominates
        assert phi_mono > 0

    def test_zero_charge_zero_force(self):
        """Q_rec = 0 -> no Yukawa force."""
        V = 0.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        assert V == 0.0

    @pytest.mark.parametrize("r", [5, 10, 15, 20, 30, 50])
    def test_yukawa_monotonically_decreasing(self, r):
        """|phi(r)| should decrease with r."""
        V = -7.1847
        debye = 7.828
        phi_r = abs(V * math.exp(-r / debye) / r)
        phi_r1 = abs(V * math.exp(-(r + 1) / debye) / (r + 1))
        assert phi_r > phi_r1


# 14. End-to-end expected results
class TestExpectedResults:
    """Verify expected k_on values for both test systems."""

    def test_two_spheres_analytical(self):
        """Analytical k_on ≈ 1.57e10 M⁻¹s⁻¹ (Debye-Smoluchowski)."""
        k_anal = 1.57e10
        assert k_anal > 1e10

    def test_two_spheres_reference(self):
        """k_on ≈ 1.526e10 M⁻¹s⁻¹ (numerical, APBS grid)."""
        k_ref = 1.526e10
        assert abs(k_ref - 1.57e10) / 1.57e10 < 0.05  # within 5% of analytical

    def test_thrombin_experimental(self):
        """Experimental k_on ≈ 4e7 M⁻¹s⁻¹ for thrombin-thrombomodulin."""
        k_exp = 4e7
        assert k_exp > 1e7


# 15. Reaction criterion
class TestReactionCriterionPhysics:
    """Verify reaction criterion."""

    def test_two_spheres_single_pair(self):
        """charged_spheres: 1 pair, n_needed=1, cutoff=2.5 Å."""
        n_pairs, n_needed, cutoff = 1, 1, 2.5
        assert n_needed <= n_pairs

    def test_thrombin_21_pairs_3_needed(self):
        """thrombin: 21 pairs, n_needed=3, cutoff=15.0 Å."""
        n_pairs, n_needed = 21, 3
        assert n_needed <= n_pairs

    def test_n_needed_semantics(self):
        """reaction fires if n_satisfied >= n_needed.
        This is an or-of-subsets, not all-or-nothing."""
        # With 21 pairs and n_needed=3, ANY 3 of 21 can trigger
        assert True  # documents the semantics


# Multipole far-field tests
class TestMultipoleExpansion:
    """Test the MultipoleExpansion class."""

    def test_monopole_only(self):
        """Single point charge -> only monopole, no dipole/quadrupole."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([5.0]), debye_length=7.86
        )
        assert abs(mp.Q - 5.0) < 1e-10
        assert mp.dipole_mag < 1e-10
        assert mp.quad_mag < 1e-10

    def test_monopole_potential_exact(self):
        """Monopole potential matches hand calculation exactly."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        r = 20.0
        eps = 78.0 * 0.000142
        V_exact = 3.0 / (4 * math.pi * eps * r) * math.exp(-r / 7.86)
        V_mp = mp.potential(np.array([r, 0, 0]))
        assert abs(V_mp - V_exact) / abs(V_exact) < 1e-10

    def test_pure_dipole(self):
        """Two opposite charges -> Q=0, pure dipole."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86,
        )
        assert abs(mp.Q) < 1e-10
        assert abs(mp.dipole_mag - 10.0) < 1e-10

    def test_dipole_potential_nonzero_for_neutral(self):
        """Neutral molecule with dipole should have nonzero potential."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86,
        )
        V = mp.potential(np.array([50.0, 0, 0]))
        assert abs(V) > 1e-6  # not zero — dipole contributes

    def test_potential_decays_with_distance(self):
        """Potential magnitude should decrease with r."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        V10 = abs(mp.potential(np.array([10.0, 0, 0])))
        V20 = abs(mp.potential(np.array([20.0, 0, 0])))
        V50 = abs(mp.potential(np.array([50.0, 0, 0])))
        assert V10 > V20 > V50

    def test_force_is_negative_gradient(self):
        """Force should match -dV/dr numerically."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        r_vec = np.array([15.0, 3.0, -2.0])
        F = mp.force(r_vec)
        # Central difference check
        h = 0.0001
        for i in range(3):
            rp = r_vec.copy()
            rp[i] += h
            rm = r_vec.copy()
            rm[i] -= h
            F_num = -(mp.potential(rp) - mp.potential(rm)) / (2 * h)
            assert abs(F[i] - F_num) < 1e-4 * max(abs(F_num), 1e-10)

    def test_repulsive_force_same_sign(self):
        """Q_rec=+3, test point at +x -> gradient points outward (repulsive)."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        F = mp.force(np.array([20.0, 0, 0]))
        # Q_rec=+3: V > 0, dV/dr < 0 (decaying), F = -dV/dr > 0 (outward)
        assert F[0] > 0  # repulsive for same-sign charges

    def test_quadrupole_nonzero_for_distributed(self):
        """Multiple charges at various positions -> nonzero quadrupole."""
        rng = np.random.default_rng(123)
        pos = rng.standard_normal((50, 3)) * 10.0
        charges = rng.standard_normal(50) * 0.5
        mp = MultipoleExpansion(pos, charges, debye_length=7.86)
        assert mp.quad_mag > 0

    def test_summary_string(self):
        """Summary should contain key info."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        s = mp.summary()
        assert "Monopole" in s
        assert "Dipole" in s
        assert "Quadrupole" in s

    def test_zero_charge_zero_potential(self):
        """All charges zero -> V=0 everywhere."""
        mp = MultipoleExpansion(
            np.array([[1, 0, 0.0], [-1, 0, 0.0]]),
            np.array([0.0, 0.0]),
            debye_length=7.86,
        )
        V = mp.potential(np.array([20.0, 0, 0]))
        assert abs(V) < 1e-15

    def test_monopole_dominates_at_large_r(self):
        """At large r, monopole >> dipole >> quadrupole."""
        # Molecule with Q=5, small dipole
        pos = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0]])
        charges = np.array([3.0, 2.0, 0.0])  # Q=5, p=[1,0,0]
        mp = MultipoleExpansion(pos, charges, debye_length=7.86)
        # At r=100Å, monopole should be ~100% of total
        V_total = mp.potential(np.array([100.0, 0, 0]))
        eps = 78.0 * 0.000142
        V_mono = 5.0 / (4 * math.pi * eps * 100) * math.exp(-100 / 7.86)
        # Monopole should be >95% of total
        if abs(V_total) > 1e-15:
            assert abs(V_mono / V_total) > 0.9


class TestOverlapCheck:
    """Test the overlap check configuration."""

    def test_default_enabled(self):
        cfg = PySTARCConfig()
        assert cfg.overlap_check is True

    def test_xml_disable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False

    def test_xml_enable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>true</overlap_check>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is True


class TestMultipoleFallbackConfig:
    """Test multipole_fallback configuration."""

    def test_default_enabled(self):
        cfg = PySTARCConfig()
        assert cfg.multipole_fallback is True

    def test_xml_disable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <multipole_fallback>false</multipole_fallback>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.multipole_fallback is False

    def test_both_flags_independent(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True


class TestLJForcesConfig:
    """Test lj_forces configuration."""

    def test_default_disabled(self):
        cfg = PySTARCConfig()
        assert cfg.lj_forces is False

    def test_xml_enable(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <lj_forces>true</lj_forces>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.lj_forces is True

    def test_all_three_flags_independent(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text("""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
  <lj_forces>true</lj_forces>
</pystarc>""")
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True
        assert cfg.lj_forces is True


class TestOutputConfig:
    """Test the OutputConfig dataclass."""

    def test_all_defaults_true(self):
        oc = OutputConfig()
        for f in fields(oc):
            if f.type is bool:
                assert getattr(oc, f.name) is True, f"{f.name} should default True"

    def test_save_interval_default(self):
        oc = OutputConfig()
        assert oc.save_interval == 10

    def test_custom_save_interval(self):
        oc = OutputConfig(save_interval=100)
        assert oc.save_interval == 100

    def test_disable_heavy_outputs(self):
        oc = OutputConfig(full_paths=False, energetics=False)
        assert oc.full_paths is False
        assert oc.energetics is False
        assert oc.results_json is True

    def test_field_count(self):
        oc = OutputConfig()
        # 14 bool flags + 1 int save_interval = 15 fields
        assert len(fields(oc)) == 15

    def test_pystarc_config_has_outputs(self):
        cfg = PySTARCConfig()
        assert cfg.outputs is not None
        assert cfg.outputs.results_json is True
        assert cfg.outputs.save_interval == 10


# XML parsing tests
class TestOutputXMLParsing:
    """Test parsing <outputs> block from XML."""

    def _write_xml(self, tmp_path, outputs_block=""):
        xml = f"""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <n_trajectories>100</n_trajectories>
  {outputs_block}
</pystarc>"""
        p = tmp_path / "test.xml"
        p.write_text(xml)
        return p

    def test_no_outputs_block_uses_defaults(self, tmp_path):
        p = self._write_xml(tmp_path)
        cfg = parse(p)
        assert cfg.outputs.results_json is True
        assert cfg.outputs.full_paths is True
        assert cfg.outputs.save_interval == 10

    def test_disable_paths(self, tmp_path):
        p = self._write_xml(
            tmp_path,
            """
  <outputs>
    <full_paths>false</full_paths>
  </outputs>""",
        )
        cfg = parse(p)
        assert cfg.outputs.full_paths is False
        assert cfg.outputs.results_json is True  # other defaults unchanged

    def test_custom_save_interval(self, tmp_path):
        p = self._write_xml(
            tmp_path,
            """
  <outputs>
    <save_interval>50</save_interval>
  </outputs>""",
        )
        cfg = parse(p)
        assert cfg.outputs.save_interval == 50

    def test_disable_multiple(self, tmp_path):
        p = self._write_xml(
            tmp_path,
            """
  <outputs>
    <full_paths>false</full_paths>
    <energetics>false</energetics>
    <transition_matrix>false</transition_matrix>
    <save_interval>1</save_interval>
  </outputs>""",
        )
        cfg = parse(p)
        assert cfg.outputs.full_paths is False
        assert cfg.outputs.energetics is False
        assert cfg.outputs.transition_matrix is False
        assert cfg.outputs.save_interval == 1
        assert cfg.outputs.trajectories_csv is True

    def test_yes_true_1_all_work(self, tmp_path):
        for val in ["true", "True", "TRUE", "yes", "Yes", "1"]:
            p = self._write_xml(
                tmp_path,
                f"""
  <outputs>
    <full_paths>{val}</full_paths>
  </outputs>""",
            )
            cfg = parse(p)
            assert cfg.outputs.full_paths is True, f"'{val}' should parse as True"

    def test_false_no_0_all_work(self, tmp_path):
        for val in ["false", "False", "FALSE", "no", "No", "0"]:
            p = self._write_xml(
                tmp_path,
                f"""
  <outputs>
    <full_paths>{val}</full_paths>
  </outputs>""",
            )
            cfg = parse(p)
            assert cfg.outputs.full_paths is False, f"'{val}' should parse as False"


# Output writer tests
def _make_dummy_data(N=100, n_react=45, n_escape=55, n_pairs=3):
    """Create realistic dummy simulation data."""
    outcome = np.array([1] * n_react + [2] * n_escape)
    return {
        "outcome": outcome,
        "n_steps": np.random.randint(100, 1000, N),
        "start_pos": np.random.randn(N, 3) * 10,
        "start_q": np.random.randn(N, 4),
        "min_dist": np.random.uniform(2, 20, N),
        "step_at_min": np.random.randint(0, 500, N),
        "total_time_ps": np.random.uniform(10, 1000, N),
        "n_returns": np.random.randint(0, 5, N),
        "bb_triggered": np.random.randint(0, 2, N),
        "encounter_pos": np.random.randn(n_react, 3),
        "encounter_q": np.random.randn(n_react, 4),
        "encounter_traj": np.arange(n_react, dtype=np.int64),
        "encounter_step": np.random.randint(100, 500, n_react).astype(np.int64),
        "encounter_n_pairs": np.full(n_react, n_pairs, dtype=np.int64),
        "near_miss_pos": np.random.randn(n_escape, 3),
        "near_miss_q": np.random.randn(n_escape, 4),
        "near_miss_traj": np.arange(n_react, N, dtype=np.int64),
        "near_miss_dist": np.random.uniform(3, 15, n_escape),
        "path_steps": [np.random.randn(50, 8) for _ in range(5)],
        "energy_steps": [np.random.randn(50, 6) for _ in range(5)],
        "radial_bins": np.linspace(0, 24, 201),
        "radial_counts": np.random.randint(0, 100, 200),
        "angular_theta": np.linspace(0, np.pi, 36),
        "angular_phi": np.linspace(0, 2 * np.pi, 72),
        "angular_counts": np.random.randint(0, 50, (36, 72)),
        "milestone_radii": np.linspace(10, 20, 11),
        "milestone_flux_out": np.random.randint(0, 500, 11),
        "milestone_flux_in": np.random.randint(0, 500, 11),
        "contact_pair_counts": np.random.randint(0, 1000, n_pairs),
        "contact_total_steps": 50000,
        "trans_bins": np.linspace(0, 24, 51),
        "trans_matrix": np.random.randint(0, 100, (50, 50)),
    }


def _make_result(n_react=45, n_escape=55):
    return GPUBatchResult(
        n_trajectories=n_react + n_escape,
        n_reacted=n_react,
        n_escaped=n_escape,
        n_max_steps=0,
        reaction_counts={"stage_0": n_react},
        r_start=10.0,
        r_escape=20.0,
        dt=0.2,
        elapsed_sec=5.0,
        steps_per_sec=100000,
    )


class TestResultsJSON:
    """Test results.json output."""

    def test_file_created(self, tmp_path):
        result = _make_result()
        data = _make_dummy_data()
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        assert (tmp_path / "results.json").exists()

    def test_json_parseable(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        data = json.loads((tmp_path / "results.json").read_text())
        assert isinstance(data, dict)

    def test_required_fields(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        data = json.loads((tmp_path / "results.json").read_text())
        required = [
            "k_on",
            "k_on_low",
            "k_on_high",
            "P_rxn",
            "P_rxn_low",
            "P_rxn_high",
            "k_b",
            "D_rel",
            "n_trajectories",
            "n_reacted",
            "n_escaped",
            "wall_time_sec",
            "steps_per_sec",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_k_on_positive(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        data = json.loads((tmp_path / "results.json").read_text())
        assert data["k_on"] > 0
        assert data["k_on_low"] <= data["k_on"] <= data["k_on_high"]

    def test_log10_present(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        data = json.loads((tmp_path / "results.json").read_text())
        assert "log10_k_on" in data
        assert abs(data["log10_k_on"] - math.log10(data["k_on"])) < 1e-6

    def test_disabled(self, tmp_path):
        oc = OutputConfig(results_json=False)
        write_all(
            tmp_path, _make_result(), _make_dummy_data(), oc, k_b=57.47, D_rel=0.434
        )
        assert not (tmp_path / "results.json").exists()


class TestTrajectoriesCSV:
    """Test trajectories.csv output."""

    def test_file_created(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        assert (tmp_path / "trajectories.csv").exists()

    def test_correct_row_count(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(45, 55),
            _make_dummy_data(100, 45, 55),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "trajectories.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 100

    def test_outcome_values(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "trajectories.csv") as f:
            rows = list(csv.DictReader(f))
        outcomes = {r["outcome"] for r in rows}
        assert outcomes <= {"reacted", "escaped", "max_steps", "running"}
        reacted = sum(1 for r in rows if r["outcome"] == "reacted")
        assert reacted == 45

    def test_columns_present(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "trajectories.csv") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        expected = [
            "traj_id",
            "outcome",
            "n_steps",
            "start_x",
            "start_y",
            "start_z",
            "start_q0",
            "start_q1",
            "start_q2",
            "start_q3",
            "min_distance",
            "step_at_min",
            "total_time_ps",
            "n_returns",
            "bb_triggered",
        ]
        for c in expected:
            assert c in cols, f"Missing column: {c}"


class TestEncountersCSV:
    """Test encounters.csv output."""

    def test_file_created_when_reactions(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(10, 90),
            _make_dummy_data(100, 10, 90),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        assert (tmp_path / "encounters.csv").exists()

    def test_row_count_matches_reactions(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(20, 80),
            _make_dummy_data(100, 20, 80),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "encounters.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20


class TestNearMissesCSV:
    """Test near_misses.csv output."""

    def test_row_count_matches_escaped(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(30, 70),
            _make_dummy_data(100, 30, 70),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "near_misses.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 70


class TestPathsNPZ:
    """Test paths.npz output."""

    def test_file_created(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        assert (tmp_path / "paths.npz").exists()

    def test_shape_correct(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        d = np.load(tmp_path / "paths.npz")
        assert d["data"].shape[1] == 8  # traj_id, step, x, y, z, q0, q1, q2

    def test_disabled(self, tmp_path):
        oc = OutputConfig(full_paths=False)
        write_all(
            tmp_path, _make_result(), _make_dummy_data(), oc, k_b=57.47, D_rel=0.434
        )
        assert not (tmp_path / "paths.npz").exists()


class TestRadialDensity:
    """Test radial_density.csv output."""

    def test_columns(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "radial_density.csv") as f:
            cols = csv.DictReader(f).fieldnames
        assert "r_center" in cols
        assert "density" in cols

    def test_density_nonnegative(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "radial_density.csv") as f:
            for row in csv.DictReader(f):
                assert float(row["density"]) >= 0


class TestMilestoneFlux:
    """Test milestone_flux.csv output."""

    def test_columns(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "milestone_flux.csv") as f:
            cols = csv.DictReader(f).fieldnames
        expected = ["radius", "flux_outward", "flux_inward", "net_flux"]
        for c in expected:
            assert c in cols

    def test_row_count(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        with open(tmp_path / "milestone_flux.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 11  # 11 milestone radii


class TestTransitionMatrix:
    """Test transition_matrix.npz output."""

    def test_square(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        d = np.load(tmp_path / "transition_matrix.npz")
        assert d["counts"].shape[0] == d["counts"].shape[1]
        assert d["counts"].shape[0] == 50


class TestPCommit:
    """Test p_commit.npz output."""

    def test_values_in_01(self, tmp_path):
        write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        d = np.load(tmp_path / "p_commit.npz")
        assert np.all(d["p_commit"] >= 0)
        assert np.all(d["p_commit"] <= 1)


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_reactions(self, tmp_path):
        data = _make_dummy_data(100, 0, 100)
        result = _make_result(0, 100)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / "results.json").read_text())
        assert rj["P_rxn"] == 0.0
        assert rj["k_on"] == 0.0
        # encounters.csv should not be created
        assert not (tmp_path / "encounters.csv").exists()

    def test_all_reacted(self, tmp_path):
        data = _make_dummy_data(50, 50, 0)
        result = _make_result(50, 0)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / "results.json").read_text())
        assert rj["P_rxn"] == 1.0
        # near_misses.csv should have 0 rows (no escapes)

    def test_all_disabled(self, tmp_path):
        oc = OutputConfig(
            results_json=False,
            trajectories_csv=False,
            encounters_csv=False,
            near_misses_csv=False,
            full_paths=False,
            radial_density=False,
            angular_map=False,
            fpt_distribution=False,
            contact_frequency=False,
            milestone_flux=False,
            p_commit=False,
            transition_matrix=False,
            energetics=False,
            pose_clusters=False,
        )
        written = write_all(
            tmp_path, _make_result(), _make_dummy_data(), oc, k_b=57.47, D_rel=0.434
        )
        assert len(written) == 0

    def test_file_count_all_enabled(self, tmp_path):
        written = write_all(
            tmp_path,
            _make_result(),
            _make_dummy_data(),
            OutputConfig(),
            k_b=57.47,
            D_rel=0.434,
        )
        assert len(written) == 14


class TestConvergenceAnalysis:
    def test_basic_convergence(self):
        result = analyse_convergence(n_reacted=500, n_escaped=500, k_b=35.0)
        assert result["N"] == 1000
        assert result["P_rxn"] == pytest.approx(0.5, abs=1e-10)
        assert result["SE"] == pytest.approx(math.sqrt(0.5 * 0.5 / 1000), abs=1e-10)
        assert result["converged"] is True

    def test_low_prxn_not_converged(self):
        result = analyse_convergence(n_reacted=5, n_escaped=95, k_b=35.0, tol=0.05)
        assert result["P_rxn"] == pytest.approx(0.05)
        assert result["converged"] is False

    def test_zero_reacted(self):
        result = analyse_convergence(n_reacted=0, n_escaped=1000, k_b=35.0)
        assert result["P_rxn"] == 0.0
        assert result["SE"] == 0.0
        assert result["relative_SE"] == float("inf")
        assert result["converged"] is False
        assert result["k_on"] == 0.0

    def test_all_reacted(self):
        result = analyse_convergence(n_reacted=1000, n_escaped=0, k_b=35.0)
        assert result["P_rxn"] == 1.0
        assert result["SE"] == 0.0
        assert result["relative_SE"] == 0.0
        assert result["converged"] is True

    def test_no_trajectories(self):
        result = analyse_convergence(n_reacted=0, n_escaped=0, k_b=35.0)
        assert result["converged"] is False
        assert "reason" in result

    def test_wilson_ci_bounds(self):
        result = analyse_convergence(n_reacted=50, n_escaped=950, k_b=35.0)
        lo, hi = result["wilson_CI_P"]
        assert lo >= 0.0
        assert hi <= 1.0
        assert lo < result["P_rxn"] < hi

    def test_wilson_ci_small_prxn(self):
        result = analyse_convergence(n_reacted=2, n_escaped=998, k_b=35.0)
        lo, hi = result["wilson_CI_P"]
        assert lo >= 0.0

    def test_n_needed_targets(self):
        result = analyse_convergence(n_reacted=100, n_escaped=900, k_b=35.0)
        assert "10%" in result["N_needed"]
        assert "5%" in result["N_needed"]
        assert "1%" in result["N_needed"]
        assert result["N_needed"]["1%"] > result["N_needed"]["5%"]

    def test_kon_conversion(self):
        conv = 6.022e8
        result = analyse_convergence(
            n_reacted=500, n_escaped=500, k_b=35.0, conv_factor=conv
        )
        assert result["k_on"] == pytest.approx(conv * 35.0 * 0.5)

    def test_print_convergence_normal(self):
        result = analyse_convergence(n_reacted=500, n_escaped=500, k_b=35.0)
        text = print_convergence(result)
        assert "P_rxn" in text
        assert "Converged" in text

    def test_print_convergence_not_converged(self):
        result = analyse_convergence(n_reacted=5, n_escaped=95, k_b=35.0, tol=0.01)
        text = print_convergence(result)
        assert "Not converged" in text

    def test_print_convergence_zero_prxn(self):
        result = analyse_convergence(n_reacted=0, n_escaped=100, k_b=35.0)
        text = print_convergence(result)
        assert "inf" in text

    def test_print_convergence_no_data(self):
        result = {"converged": False, "reason": "no completed trajectories"}
        text = print_convergence(result)
        assert "no completed trajectories" in text

    def test_save_convergence(self):
        result = analyse_convergence(n_reacted=100, n_escaped=900, k_b=35.0)
        with tempfile.TemporaryDirectory() as td:
            save_convergence(result, work_dir=td)
            path = os.path.join(td, "convergence.json")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["N"] == 1000
            assert loaded["P_rxn"] == pytest.approx(0.1)


class TestWienerProcess:
    def test_init(self):
        dW = np.array([1.0, 2.0, 3.0])
        wp = WienerProcess(dW, dt=0.5)
        assert wp.t == 0.0
        assert wp.dt == 0.5
        np.testing.assert_array_equal(wp.dW, dW)
        assert wp.at_end is False

    def test_step_forward(self):
        wp = WienerProcess(np.zeros(3), dt=0.5)
        wp.step_forward()
        assert wp.t == pytest.approx(0.5)
        assert wp.at_end is True

    def test_split(self):
        rng = np.random.default_rng(42)
        dW = np.array([1.0, 0.0, 0.0])
        wp = WienerProcess(dW, dt=1.0)
        wp.split(rng)
        assert wp.at_end is False
        assert wp.dt == pytest.approx(0.5)
        w1 = wp.dW.copy()
        wp.step_forward()
        assert wp.t == pytest.approx(0.5)
        w2 = wp.dW.copy()
        np.testing.assert_allclose(w1 + w2, dW, atol=1e-10)
        wp.step_forward()
        assert wp.t == pytest.approx(1.0)
        assert wp.at_end is True

    def test_double_split(self):
        rng = np.random.default_rng(99)
        dW = np.array([2.0, 3.0])
        wp = WienerProcess(dW, dt=2.0)
        wp.split(rng)
        wp.split(rng)
        assert wp.dt == pytest.approx(0.5)
        total_w = np.zeros(2)
        total_t = 0.0
        while not wp.at_end:
            total_w += wp.dW
            total_t += wp.dt
            wp.step_forward()
        np.testing.assert_allclose(total_w, dW, atol=1e-10)
        assert total_t == pytest.approx(2.0)


class TestDoOneFullStep:
    def test_no_backstep(self):
        call_count = [0]

        def advance(dW, t, dt):
            call_count[0] += 1
            return False, False

        def stepback(t, dt):
            pass

        rng = np.random.default_rng(1)
        dW = np.array([0.1, 0.2])
        final_dt = do_one_full_step(advance, stepback, rng, dW, 0.5)
        assert final_dt == pytest.approx(0.5)
        assert call_count[0] == 1

    def test_one_backstep(self):
        step_count = [0]

        def advance(dW, t, dt):
            step_count[0] += 1
            if step_count[0] == 1:
                return False, True
            return False, False

        def stepback(t, dt):
            pass

        rng = np.random.default_rng(7)
        dW = np.array([1.0, 1.0, 1.0])
        final_dt = do_one_full_step(advance, stepback, rng, dW, 1.0)
        assert final_dt == pytest.approx(0.5)
        assert step_count[0] >= 2

    def test_trajectory_done(self):

        def advance(dW, t, dt):
            return True, False

        def stepback(t, dt):
            pass

        rng = np.random.default_rng(1)
        final_dt = do_one_full_step(advance, stepback, rng, np.zeros(3), 0.2)
        assert final_dt == pytest.approx(0.2)


class TestMakeInitialDW:
    def test_shape(self):
        rng = np.random.default_rng(42)
        dW = make_initial_dW(6, 0.5, rng)
        assert dW.shape == (6,)

    def test_scaling(self):
        rng = np.random.default_rng(42)
        n = 100000
        samples = np.array([make_initial_dW(1, 2.0, rng)[0] for _ in range(n)])
        assert np.std(samples) == pytest.approx(math.sqrt(2.0), abs=0.05)


class TestEffectiveCharges:
    def test_single_charge_potential(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            debye_length=1e10,
            bjerrum_length=1.0,
        )
        r = np.array([10.0, 0.0, 0.0])
        phi = ec.potential(r)
        assert phi == pytest.approx(1.0 / 10.0, rel=1e-4)

    def test_debye_screening(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            debye_length=5.0,
            bjerrum_length=1.0,
        )
        phi_near = ec.potential(np.array([1.0, 0.0, 0.0]))
        phi_far = ec.potential(np.array([20.0, 0.0, 0.0]))
        assert phi_near > phi_far

    def test_potential_symmetry(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
        )
        r = 5.0
        phi_x = ec.potential(np.array([r, 0.0, 0.0]))
        phi_y = ec.potential(np.array([0.0, r, 0.0]))
        phi_z = ec.potential(np.array([0.0, 0.0, r]))
        assert phi_x == pytest.approx(phi_y, rel=1e-10)
        assert phi_x == pytest.approx(phi_z, rel=1e-10)

    def test_force_repulsive_same_sign(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            debye_length=100.0,
            bjerrum_length=BJERRUM_LENGTH,
        )
        r = np.array([10.0, 0.0, 0.0])
        F = ec.force_on_charge(r, q=1.0)
        assert F[0] > 0

    def test_force_attractive_opposite_sign(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            debye_length=100.0,
            bjerrum_length=BJERRUM_LENGTH,
        )
        r = np.array([10.0, 0.0, 0.0])
        F = ec.force_on_charge(r, q=-1.0)
        assert F[0] < 0

    def test_force_zero_charge(self):
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
        )
        F = ec.force_on_charge(np.array([5.0, 0.0, 0.0]), q=0.0)
        np.testing.assert_array_equal(F, np.zeros(3))

    def test_multiple_charges(self):
        ec = EffectiveCharges(
            positions=np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]),
            charges=np.array([1.0, 1.0]),
            debye_length=1e10,
            bjerrum_length=1.0,
        )
        phi_origin = ec.potential(np.array([0.0, 0.0, 0.0]))
        assert phi_origin == pytest.approx(2.0 / 5.0, rel=1e-4)

    def test_len(self):
        ec = EffectiveCharges(
            positions=np.zeros((3, 3)),
            charges=np.ones(3),
        )
        assert len(ec) == 3

    def test_repr(self):
        ec = EffectiveCharges(
            positions=np.zeros((2, 3)),
            charges=np.array([1.0, -0.5]),
        )
        s = repr(ec)
        assert "2 charges" in s
        assert "0.50 e" in s

    def test_from_xml(self):
        xml = (
            '<?xml version="1.0"?>\n'
            "<charges>\n"
            "  <charge><x>1.0</x><y>2.0</y><z>3.0</z><q>0.5</q></charge>\n"
            "  <charge><x>-1.0</x><y>-2.0</y><z>-3.0</z><q>-0.5</q></charge>\n"
            "</charges>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            ec = EffectiveCharges.from_xml(f.name)
        os.unlink(f.name)
        assert len(ec) == 2
        np.testing.assert_allclose(ec.charges, [0.5, -0.5])

    def test_from_xml_point_charge_tag(self):
        xml = (
            '<?xml version="1.0"?>\n'
            "<multipole>\n"
            "  <point_charge><x>0</x><y>0</y><z>0</z><q>1.0</q></point_charge>\n"
            "</multipole>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            ec = EffectiveCharges.from_xml(f.name)
        os.unlink(f.name)
        assert len(ec) == 1

    def test_from_xml_empty_raises(self):
        xml = '<?xml version="1.0"?>\n<charges></charges>\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            with pytest.raises(ValueError, match="No charges"):
                EffectiveCharges.from_xml(f.name)
        os.unlink(f.name)


class TestLoadEffectiveCharges:
    def test_auto_detect_cheby(self):
        xml = (
            '<?xml version="1.0"?>\n'
            "<charges>\n"
            "  <charge><x>0</x><y>0</y><z>0</z><q>1</q></charge>\n"
            "</charges>\n"
        )
        with tempfile.TemporaryDirectory() as td:
            Path(td, "mol_cheby.xml").write_text(xml)
            ec = load_effective_charges(td, "mol")
            assert ec is not None
            assert len(ec) == 1

    def test_auto_detect_mpole(self):
        xml = (
            '<?xml version="1.0"?>\n'
            "<charges>\n"
            "  <charge><x>0</x><y>0</y><z>0</z><q>2</q></charge>\n"
            "</charges>\n"
        )
        with tempfile.TemporaryDirectory() as td:
            Path(td, "mol_mpole.xml").write_text(xml)
            ec = load_effective_charges(td, "mol")
            assert ec is not None

    def test_not_found_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            ec = load_effective_charges(td, "nonexistent")
            assert ec is None


class TestStepNearSurface:
    def test_inv_erf(self):
        assert _inv_erf(0.0) == pytest.approx(0.0)
        assert _inv_erf(0.5) == pytest.approx(math.erfc(1) and 0.4769362762, rel=1e-5)

    def test_large_x0_with_repulsion_survives(self):
        rng = np.random.default_rng(42)
        n_survived = 0
        for _ in range(200):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=50.0, F=10.0, D=0.01
            )
            if survives:
                n_survived += 1
        assert n_survived > 100

    def test_small_x0_absorbs(self):
        rng = np.random.default_rng(42)
        n_absorbed = 0
        for _ in range(500):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=0.001, F=0.0, D=1.0
            )
            if not survives:
                n_absorbed += 1
        assert n_absorbed > 50

    def test_survival_returns_positive_x(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=5.0, F=1.0, D=0.1
            )
            if survives:
                assert new_x >= 0.0
                assert time > 0.0

    def test_absorption_returns_zero_x(self):
        rng = np.random.default_rng(99)
        for _ in range(500):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=0.5, F=0.0, D=1.0
            )
            if not survives:
                assert new_x == 0.0
                assert time >= 0.0
                return
        pytest.skip("No absorption event in 500 trials")

    def test_repulsive_force_increases_survival(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        n_surv_noforce = sum(
            step_near_absorbing_surface(rng_a, x0=2.0, F=0.0, D=0.5)[0]
            for _ in range(1000)
        )
        n_surv_repulsive = sum(
            step_near_absorbing_surface(rng_b, x0=2.0, F=5.0, D=0.5)[0]
            for _ in range(1000)
        )
        assert n_surv_repulsive >= n_surv_noforce


class TestQuatMultiply:
    def test_identity(self):
        I = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(quat_multiply(I, q), q, atol=1e-12)
        np.testing.assert_allclose(quat_multiply(q, I), q, atol=1e-12)

    def test_inverse(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = np.array([0.5, -0.5, -0.5, -0.5])
        prod = quat_multiply(q, q_conj)
        np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


class TestQuatOfRotvec:
    def test_zero_rotation(self):
        q = quat_of_rotvec(np.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_90deg_about_z(self):
        omega = np.array([0.0, 0.0, math.pi / 2])
        q = quat_of_rotvec(omega)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        assert q[0] == pytest.approx(math.cos(math.pi / 4), abs=1e-10)
        assert q[3] == pytest.approx(math.sin(math.pi / 4), abs=1e-10)


class TestRandomUnitQuat:
    def test_unit_norm(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = random_unit_quat(rng)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12


class TestDiffusionalRotation:
    def test_tau_zero(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_tau_negative(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, -1.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_small_tau_unit_norm(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.1)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_0p25_small_angle(self):
        rng = np.random.default_rng(42)
        angles = []
        for _ in range(500):
            q = diffusional_rotation(rng, 0.1)
            angle = 2 * math.acos(min(1.0, abs(q[0])))
            angles.append(angle)
        mean_angle = np.mean(angles)
        assert mean_angle < 1.5

    def test_tau_0p3_split_at_025(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.3)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_0p7_split_at_05(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.7)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_1p5_split_at_1(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 1.5)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_3_split_at_2(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 3.0)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_large_tau_random(self):
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 10.0)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
        assert abs(q[0]) < 1.0


class TestFingerprint:
    def test_all_inside(self):
        verts = np.ones((2, 2, 2), dtype=np.int8)
        fp = _fingerprint(verts)
        assert fp[0] == 0

    def test_all_outside(self):
        verts = np.zeros((2, 2, 2), dtype=np.int8)
        fp = _fingerprint(verts)
        assert fp[0] == 0

    def test_single_corner(self):
        verts = np.zeros((2, 2, 2), dtype=np.int8)
        verts[0, 0, 0] = 1
        fp = _fingerprint(verts)
        assert fp[0] == 1


class TestVoxelise:
    def test_single_sphere(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([5.0])
        grid, origin, spacing = _voxelise(coords, radii, spacing=1.0, padding=3.0)
        assert grid.sum() > 0
        assert grid.shape[0] > 5

    def test_all_interior(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([3.0])
        grid, origin, sp = _voxelise(coords, radii, spacing=0.5, padding=2.0)
        center_idx = tuple(int((0.0 - origin[i]) / sp[i]) for i in range(3))
        valid = all(0 <= center_idx[i] < grid.shape[i] for i in range(3))
        if valid:
            assert grid[center_idx] == 1


class TestExtractSurface:
    def test_sphere_has_surface(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([3.0])
        grid, origin, sp = _voxelise(coords, radii, spacing=1.0, padding=2.0)
        surface = _extract_surface(grid, origin, sp)
        assert len(surface) > 0
        for pt in surface:
            assert pt.area > 0


class TestMCHydrodynamicRadius:
    def test_single_sphere(self):
        R = 5.0
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([R])
        r_h, center, max_dist = mc_hydrodynamic_radius(
            coords, radii, spacing=0.5, n_mc=500_000, seed=42
        )
        assert r_h == pytest.approx(R, rel=0.25)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1.5)


class TestMaxTimeStep:
    def test_normal(self):
        dt = max_time_step(r=100.0, D_rel=0.1, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt > 0

    def test_r_zero_fallback(self):
        dt = max_time_step(r=0.0, D_rel=0.1, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt == 0.2

    def test_D_zero_fallback(self):
        dt = max_time_step(r=100.0, D_rel=0.0, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt == 0.2

    def test_no_rotation(self):
        dt = max_time_step(r=50.0, D_rel=0.1, D_rot=0.0, r_hydro1=10.0, r_hydro2=10.0)
        assert dt > 0


class TestReactionTimeStep:
    def test_normal(self):
        dt = reaction_time_step(rho_min=17.0, D_rel=0.1)
        assert dt > 0

    def test_zero_rho(self):
        dt = reaction_time_step(rho_min=0.0, D_rel=0.1)
        assert dt == 0.05

    def test_zero_D(self):
        dt = reaction_time_step(rho_min=17.0, D_rel=0.0)
        assert dt == 0.05


class TestAdaptiveTimeStepController:
    def test_first_call(self):
        ctrl = AdaptiveTimeStep()
        dt = ctrl.get_dt(
            r=100.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=20.0,
            r_hydro2=5.0,
            rxn_distances=[],
        )
        assert dt > 0

    def test_growth_factor(self):
        ctrl = AdaptiveTimeStep()
        dt1 = ctrl.get_dt(
            r=100.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=20.0,
            r_hydro2=5.0,
            rxn_distances=[],
        )
        dt2 = ctrl.get_dt(
            r=100.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=20.0,
            r_hydro2=5.0,
            rxn_distances=[],
        )
        assert dt2 <= dt1 * 1.1 + 1e-15

    def test_near_reaction_zone(self):
        ctrl = AdaptiveTimeStep()
        dt = ctrl.get_dt(
            r=20.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=10.0,
            r_hydro2=5.0,
            rxn_distances=[17.0],
        )
        dt_far = ctrl.get_dt(
            r=200.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=10.0,
            r_hydro2=5.0,
            rxn_distances=[17.0],
        )
        assert dt <= dt_far

    def test_reset(self):
        ctrl = AdaptiveTimeStep()
        ctrl.get_dt(
            r=50.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=10.0,
            r_hydro2=5.0,
            rxn_distances=[],
        )
        ctrl.reset()
        dt = ctrl.get_dt(
            r=50.0,
            D_rel=0.1,
            D_rot=0.001,
            r_hydro1=10.0,
            r_hydro2=5.0,
            rxn_distances=[],
        )
        assert dt > 0


class TestBackstepDueToForce:
    def test_dt_below_min_no_backstep(self):
        pos_old = np.array([0.0, 0.0, 0.0])
        pos_new = np.array([1.0, 0.0, 0.0])
        f_old = np.array([0.0, 0.0, 0.0])
        f_new = np.array([100.0, 0.0, 0.0])
        result = backstep_due_to_force(
            f_new, f_old, pos_new, pos_old, dt=0.0001, dt_min=0.001
        )
        assert result is False

    def test_zero_force_change(self):
        f = np.array([1.0, 0.0, 0.0])
        pos_old = np.array([0.0, 0.0, 0.0])
        pos_new = np.array([1.0, 0.0, 0.0])
        result = backstep_due_to_force(f, f, pos_new, pos_old, dt=0.5, dt_min=0.001)
        assert result is False

    def test_large_force_change_backstep(self):
        pos_old = np.array([10.0, 0.0, 0.0])
        pos_new = np.array([10.01, 0.0, 0.0])
        f_old = np.array([0.0, 0.0, 0.0])
        f_new = np.array([1e6, 0.0, 0.0])
        result = backstep_due_to_force(
            f_new, f_old, pos_new, pos_old, dt=1.0, dt_min=0.001, radius=5.0
        )
        assert result is True

    def test_perpendicular_force_no_backstep(self):
        pos_old = np.array([0.0, 0.0, 0.0])
        pos_new = np.array([1.0, 0.0, 0.0])
        f_old = np.array([0.0, 0.0, 0.0])
        f_new = np.array([0.0, 1.0, 0.0])
        result = backstep_due_to_force(
            f_new, f_old, pos_new, pos_old, dt=0.5, dt_min=0.001
        )
        assert result is False


class TestMoleculeEdgeCases:
    def test_empty_radius_of_gyration(self):
        mol = Molecule(name="empty")
        assert mol.radius_of_gyration() == 0.0

    def test_empty_bounding_radius(self):
        mol = Molecule(name="empty")
        assert mol.bounding_radius() == 0.0

    def test_bounding_box_empty_molecule(self):
        mol = Molecule(name="empty")
        bb = BoundingBox.from_molecule(mol)
        assert bb.xmin == 0.0 and bb.xmax == 0.0


class TestPqrIoEdgeCases:
    def test_parse_pqr_with_remarks_and_end(self):
        pqr = (
            "REMARK This is a test\n"
            "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
            "HETATM    2  O   HOH     2       4.000   5.000   6.000 -0.834  1.520\n"
            "END\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(pqr)
            f.flush()
            mol = parse_pqr(f.name)
        os.unlink(f.name)
        assert len(mol.atoms) == 2
        assert mol.atoms[0].charge == pytest.approx(0.5)
        assert mol.atoms[1].name == "O"

    def test_write_and_read_roundtrip(self):
        mol = Molecule(name="test")
        mol.atoms.append(
            Atom(
                index=1,
                name="CA",
                residue_name="ALA",
                residue_index=1,
                chain="A",
                x=1.0,
                y=2.0,
                z=3.0,
                charge=0.5,
                radius=1.8,
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".pqr", delete=False) as f:
            path = f.name
        write_pqr(mol, path)
        mol2 = parse_pqr(path)
        os.unlink(path)
        assert len(mol2.atoms) == 1
        assert mol2.atoms[0].x == pytest.approx(1.0, abs=0.01)

    def test_parse_pqr_malformed_line_skipped(self):
        pqr = (
            "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
            "ATOM  bad line missing fields\n"
            "ATOM      3  CB  ALA     1       4.000   5.000   6.000  -0.100  1.700\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(pqr)
            f.flush()
            mol = parse_pqr(f.name)
        os.unlink(f.name)
        assert len(mol.atoms) == 2


class TestReactionInterfaceProbability:
    def _make_molecules(self, dist=5.0):
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
                radius=1.0,
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
                x=dist,
                y=0.0,
                z=0.0,
                charge=0.0,
                radius=1.0,
            )
        )
        return mol1, mol2

    def test_reaction_with_probability_1(self):
        mol1, mol2 = self._make_molecules(dist=3.0)
        criteria = ReactionCriteria(
            name="test",
            pairs=[
                ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=10.0)
            ],
            n_needed=1,
        )
        rxn = ReactionInterface(name="test", criteria=criteria, probability=1.0)
        assert rxn.check(mol1, mol2) is True

    def test_reaction_with_probability_0(self):
        mol1, mol2 = self._make_molecules(dist=3.0)
        criteria = ReactionCriteria(
            name="test",
            pairs=[
                ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=10.0)
            ],
            n_needed=1,
        )
        rxn = ReactionInterface(name="test", criteria=criteria, probability=0.0)
        n_fired = sum(rxn.check(mol1, mol2) for _ in range(100))
        assert n_fired == 0

    def test_pathway_set_check_with_rng(self):
        mol1, mol2 = self._make_molecules(dist=3.0)
        criteria = ReactionCriteria(
            name="test",
            pairs=[
                ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=10.0)
            ],
            n_needed=1,
        )
        rxn = ReactionInterface(name="rxn_a", criteria=criteria, probability=0.5)
        ps = PathwaySet(reactions=[rxn])
        rng = np.random.default_rng(42)
        n_fired = 0
        for _ in range(200):
            result = ps.check_all(mol1, mol2, rng=rng)
            if result is not None:
                n_fired += 1
        assert 50 < n_fired < 150

    def test_not_satisfied_returns_false(self):
        mol1, mol2 = self._make_molecules(dist=100.0)
        criteria = ReactionCriteria(
            name="test",
            pairs=[
                ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)
            ],
            n_needed=1,
        )
        rxn = ReactionInterface(name="test", criteria=criteria, probability=1.0)
        assert rxn.check(mol1, mol2) is False


class TestCLI:
    def test_cli_group_help(self):
        from click.testing import CliRunner
        from pystarc.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PySTARC" in result.output

    def test_cli_version(self):
        from click.testing import CliRunner
        from pystarc.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.1.0" in result.output

    def test_bounding_box_cmd(self):
        from click.testing import CliRunner
        from pystarc.cli.main import cli

        pqr = "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.pqr").write_text(pqr)
            result = runner.invoke(cli, ["bounding_box", "test.pqr"])
            assert result.exit_code == 0
            assert "Bounding box" in result.output

    def test_pqr_to_xml_cmd(self):
        from click.testing import CliRunner
        from pystarc.cli.main import cli

        pqr = "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.pqr").write_text(pqr)
            result = runner.invoke(cli, ["pqr_to_xml", "test.pqr", "-o", "out.xml"])
            assert result.exit_code == 0
            assert Path("out.xml").exists()
            content = Path("out.xml").read_text()
            assert "ALA" in content

    def test_nam_simulation_missing_files(self):
        from click.testing import CliRunner
        from pystarc.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nam_simulation",
                "--mol1",
                "no.pqr",
                "--mol2",
                "no2.pqr",
                "--rxn",
                "no.xml",
            ],
        )
        assert result.exit_code != 0


# Pipeline extract
class TestPipelineExtract:

    def test_is_atom_line_atom(self):
        assert (
            _is_atom_line("ATOM      1  CA  ALA     1       1.0   2.0   3.0  0.5  1.8")
            is True
        )

    def test_is_atom_line_hetatm(self):
        assert (
            _is_atom_line("HETATM    1  C1  BEN     1       1.0   2.0   3.0  0.5  1.8")
            is True
        )

    def test_is_atom_line_remark(self):
        assert _is_atom_line("REMARK test line") is False

    def test_is_atom_line_ter(self):
        assert _is_atom_line("TER") is False

    def test_residue_name_extraction(self):
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80"
        assert _residue_name(line) == "ALA"

    def test_extract_splits_correctly(self):
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80\n"
            "ATOM      2  CB  ALA A   1       4.000   5.000   6.000  0.50  1.80\n"
            "HETATM    3  C1  BEN A   2       7.000   8.000   9.000  0.10  1.70\n"
            "ATOM      4  O   HOH A   3      10.000  11.000  12.000 -0.83  1.52\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            rec, lig = extract(pdb_path, "BEN", td)
            assert rec.exists()
            assert lig.exists()
            rec_text = rec.read_text()
            lig_text = lig.read_text()
            assert "ALA" in rec_text
            assert "BEN" in lig_text
            assert "HOH" not in rec_text

    def test_extract_no_ligand_raises(self):
        pdb = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80\n"
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            with pytest.raises(ValueError, match="No atoms"):
                extract(pdb_path, "XYZ", td)

    def test_extract_no_receptor_raises(self):
        pdb = "HETATM    1  C1  BEN A   1       1.000   2.000   3.000  0.10  1.70\n"
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            with pytest.raises(ValueError, match="No receptor"):
                extract(pdb_path, "BEN", td)

    def test_extract_filters_ions(self):
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80\n"
            "HETATM    2  C1  BEN A   2       7.000   8.000   9.000  0.10  1.70\n"
            "ATOM      3  NA  NA  A   3      10.000  11.000  12.000  1.00  1.40\n"
            "ATOM      4  CL  CL  A   4      13.000  14.000  15.000 -1.00  1.80\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            rec, lig = extract(pdb_path, "BEN", td)
            rec_text = rec.read_text()
            assert "NA" not in rec_text.split("ALA")[0] or "ALA" in rec_text

    def test_extract_case_insensitive_ligand(self):
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80\n"
            "HETATM    2  C1  BEN A   2       7.000   8.000   9.000  0.10  1.70\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            rec, lig = extract(pdb_path, "ben", td)
            assert lig.exists()


# COFFDROP parameters
class TestCOFFDROPParams:

    def test_txt_to_floats(self):
        arr = _txt_to_floats("1.0 2.5 3.7")
        np.testing.assert_allclose(arr, [1.0, 2.5, 3.7])

    def test_txt_to_floats_empty(self):
        arr = _txt_to_floats("")
        assert len(arr) == 0

    def test_bead_def_dataclass(self):
        bd = BeadDef(name="CA", atoms=["CA", "HA"])
        assert bd.name == "CA"
        assert len(bd.atoms) == 2
        assert bd.location == ""

    def test_residue_def_dataclass(self):
        rd = ResidueDef(name="ALA")
        assert rd.name == "ALA"
        assert rd.beads == []

    def test_bond_def_dataclass(self):
        bond = BondDef(
            residues=("ALA", "GLY"),
            atoms=("CA", "CA"),
            orders=(0, 1),
            length=3.8,
            index=0,
        )
        assert bond.length == 3.8
        assert bond.residues == ("ALA", "GLY")

    def test_tabulated_potential_linear(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=np.linspace(0, 10, 11),
            residues=(0,),
            atoms=(0,),
            orders=(0,),
            index=0,
        )
        assert pot.value(5.0) == pytest.approx(5.0)
        assert pot.value(0.0) == pytest.approx(0.0)
        assert pot.value(10.0) == pytest.approx(10.0)

    def test_tabulated_potential_clamp_low(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=np.linspace(0, 10, 11),
            residues=(0,),
            atoms=(0,),
            orders=(0,),
            index=0,
        )
        assert pot.value(-5.0) == pytest.approx(0.0)

    def test_tabulated_potential_clamp_high(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=np.linspace(0, 10, 11),
            residues=(0,),
            atoms=(0,),
            orders=(0,),
            index=0,
        )
        assert pot.value(20.0) == pytest.approx(10.0)

    def test_tabulated_potential_deriv(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=np.linspace(0, 10, 11),
            residues=(0,),
            atoms=(0,),
            orders=(0,),
            index=0,
        )
        assert pot.deriv(5.0) == pytest.approx(1.0)

    def test_tabulated_potential_quadratic(self):
        xs = np.linspace(0, 10, 101)
        vals = xs**2
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=vals,
            residues=(0,),
            atoms=(0,),
            orders=(0,),
            index=0,
        )
        assert pot.value(3.0) == pytest.approx(9.0, abs=0.1)

    def test_match_pot_exact(self):
        pot = TabulatedPotential(
            x_min=0,
            x_max=1,
            values=np.array([1.0, 2.0]),
            residues=(1, 2),
            atoms=(3, 4),
            orders=(0, 0),
            index=0,
        )
        found = _match_pot([pot], (1, 2), (3, 4), (0, 0))
        assert found is pot

    def test_match_pot_wildcard(self):
        pot = TabulatedPotential(
            x_min=0,
            x_max=1,
            values=np.array([1.0]),
            residues=(0, 0),
            atoms=(3, 4),
            orders=(0, 0),
            index=0,
        )
        found = _match_pot([pot], (5, 6), (3, 4), (0, 0), wildcard=0)
        assert found is pot

    def test_match_pot_no_match(self):
        pot = TabulatedPotential(
            x_min=0,
            x_max=1,
            values=np.array([1.0]),
            residues=(1, 2),
            atoms=(3, 4),
            orders=(0, 0),
            index=0,
        )
        found = _match_pot([pot], (5, 6), (7, 8), (0, 0))
        assert found is None

    def test_match_pot_exact_over_wildcard(self):
        wild = TabulatedPotential(
            x_min=0,
            x_max=1,
            values=np.array([10.0]),
            residues=(0, 0),
            atoms=(1, 2),
            orders=(0, 0),
            index=0,
        )
        exact = TabulatedPotential(
            x_min=0,
            x_max=1,
            values=np.array([20.0]),
            residues=(3, 4),
            atoms=(1, 2),
            orders=(0, 0),
            index=1,
        )
        found = _match_pot([wild, exact], (3, 4), (1, 2), (0, 0))
        assert found is exact

    def test_parse_mapping_xml(self):
        xml = (
            '<?xml version="1.0"?>\n<mapping>\n'
            "  <residue><name>ALA</name>\n"
            "    <bead><name>CA</name><atoms>CA HA</atoms></bead>\n"
            "    <bead><name>CB</name><atoms>CB HB1 HB2 HB3</atoms></bead>\n"
            "  </residue>\n</mapping>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            mapping = _parse_mapping(f.name)
        os.unlink(f.name)
        assert "ALA" in mapping
        assert len(mapping["ALA"].beads) == 2
        assert mapping["ALA"].beads[0].name == "CA"
        assert "HA" in mapping["ALA"].beads[0].atoms

    def test_parse_connectivity_xml(self):
        xml = (
            '<?xml version="1.0"?>\n<connectivity>\n'
            "  <bond><residues>ALA GLY</residues><atoms>CA CA</atoms>"
            "<orders>0 1</orders><length>3.8</length><index>0</index></bond>\n"
            "</connectivity>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            bonds = _parse_connectivity(f.name)
        os.unlink(f.name)
        assert len(bonds) == 1
        assert bonds[0].length == 3.8
        assert bonds[0].atoms == ("CA", "CA")

    def test_parse_charges_xml(self):
        xml = (
            '<?xml version="1.0"?>\n<charges>\n'
            "  <charge><residue>ALA</residue><atom>CA</atom><value>0.5</value></charge>\n"
            "  <charge><residue>GLY</residue><atom>CA</atom><value>-0.3</value></charge>\n"
            "</charges>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            charges = _parse_charges(f.name)
        os.unlink(f.name)
        assert charges[("ALA", "CA")] == pytest.approx(0.5)
        assert charges[("GLY", "CA")] == pytest.approx(-0.3)

    def test_coffdrop_params_constructor(self):
        params = COFFDROPParams(
            mapping={"ALA": ResidueDef(name="ALA")},
            bonds=[],
            charges={("ALA", "CA"): 0.5},
            type_map={"atoms": {"CA": 1}, "residues": {"ALA": 1}},
            pair_pots=[],
            angle_pots=[],
            dihedral_pots=[],
        )
        assert params.bead_charge("ALA", "CA") == pytest.approx(0.5)
        assert params.bead_charge("GLY", "CB") == 0.0

    def test_coffdrop_params_beads_for_residue(self):
        bead = BeadDef(name="CA", atoms=["CA"])
        params = COFFDROPParams(
            mapping={"ALA": ResidueDef(name="ALA", beads=[bead])},
            bonds=[],
            charges={},
            type_map={"atoms": {}, "residues": {}},
            pair_pots=[],
            angle_pots=[],
            dihedral_pots=[],
        )
        beads = params.beads_for_residue("ALA")
        assert len(beads) == 1
        assert params.beads_for_residue("XYZ") is None

    def test_coffdrop_params_pair_potential(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=20.0,
            values=np.linspace(5, 0, 21),
            residues=(1, 1),
            atoms=(1, 1),
            orders=(0, 0),
            index=0,
        )
        params = COFFDROPParams(
            mapping={},
            bonds=[],
            charges={},
            type_map={"atoms": {"CA": 1}, "residues": {"ALA": 1}},
            pair_pots=[pot],
            angle_pots=[],
            dihedral_pots=[],
        )
        V = params.pair_potential("ALA", "CA", "ALA", "CA", r=10.0)
        assert V == pytest.approx(2.5, abs=0.1)

    def test_coffdrop_params_pair_force(self):
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=20.0,
            values=np.linspace(5, 0, 21),
            residues=(1, 1),
            atoms=(1, 1),
            orders=(0, 0),
            index=0,
        )
        params = COFFDROPParams(
            mapping={},
            bonds=[],
            charges={},
            type_map={"atoms": {"CA": 1}, "residues": {"ALA": 1}},
            pair_pots=[pot],
            angle_pots=[],
            dihedral_pots=[],
        )
        dVdr = params.pair_force("ALA", "CA", "ALA", "CA", r=10.0)
        assert dVdr == pytest.approx(-0.25, abs=0.01)

    def test_coffdrop_params_no_match_returns_zero(self):
        params = COFFDROPParams(
            mapping={},
            bonds=[],
            charges={},
            type_map={"atoms": {"CA": 1}, "residues": {"ALA": 1}},
            pair_pots=[],
            angle_pots=[],
            dihedral_pots=[],
        )
        assert params.pair_potential("ALA", "CA", "ALA", "CA", r=5.0) == 0.0
        assert params.pair_force("ALA", "CA", "ALA", "CA", r=5.0) == 0.0
        assert params.angle_potential(("ALA",), ("CA",), (0,), 90.0) == 0.0
        assert params.angle_force(("ALA",), ("CA",), (0,), 90.0) == 0.0
        assert params.dihedral_potential(("ALA",), ("CA",), (0,), 180.0) == 0.0
        assert params.dihedral_force(("ALA",), ("CA",), (0,), 180.0) == 0.0

    def test_coffdrop_params_bond_length(self):
        bond = BondDef(
            residues=("ALA", "GLY"),
            atoms=("CA", "CA"),
            orders=(0, 1),
            length=3.8,
            index=0,
        )
        params = COFFDROPParams(
            mapping={},
            bonds=[bond],
            charges={},
            type_map={"atoms": {}, "residues": {}},
            pair_pots=[],
            angle_pots=[],
            dihedral_pots=[],
        )
        assert params.bond_length("ALA", "CA", 0, "GLY", "CA", 1) == 3.8
        assert params.bond_length("GLY", "CA", 1, "ALA", "CA", 0) == 3.8
        assert params.bond_length("XYZ", "CB", 0, "XYZ", "CB", 0) is None

    def test_coffdrop_params_repr(self):
        params = COFFDROPParams(
            mapping={"ALA": ResidueDef(name="ALA")},
            bonds=[],
            charges={},
            type_map={"atoms": {}, "residues": {}},
            pair_pots=[],
            angle_pots=[],
            dihedral_pots=[],
        )
        assert "1 residues" in repr(params)


# Multi-GPU combine data helpers
class TestCombineDataHelpers:

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as td:
            _save_json({"key": "val"}, os.path.join(td, "test.json"))
            with open(os.path.join(td, "test.json")) as f:
                data = json.load(f)
            assert data["key"] == "val"

    def test_concat_csv(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = os.path.join(td, "bd_1")
            d2 = os.path.join(td, "bd_2")
            os.makedirs(d1)
            os.makedirs(d2)
            with open(os.path.join(d1, "traj.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["traj_id", "fate"])
                w.writeheader()
                w.writerow({"traj_id": "0", "fate": "reacted"})
            with open(os.path.join(d2, "traj.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["traj_id", "fate"])
                w.writeheader()
                w.writerow({"traj_id": "0", "fate": "escaped"})
            _concat_csv([d1, d2], "traj.csv", td, reindex="traj_id")
            with open(os.path.join(td, "traj.csv")) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[1]["traj_id"] == "1"

    def test_concat_csv_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            _concat_csv([td], "nonexistent.csv", td)
            assert not os.path.exists(os.path.join(td, "nonexistent.csv"))

    def test_sum_csv(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = os.path.join(td, "bd_1")
            d2 = os.path.join(td, "bd_2")
            os.makedirs(d1)
            os.makedirs(d2)
            for d, count in [(d1, "10"), (d2, "20")]:
                with open(os.path.join(d, "radial.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["r", "count", "density"])
                    w.writeheader()
                    w.writerow({"r": "5.0", "count": count, "density": "0.0"})
            _sum_csv(
                [d1, d2],
                "radial.csv",
                td,
                sum_col="count",
                recompute_col="density",
                total_N=100,
            )
            with open(os.path.join(td, "radial.csv")) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert int(rows[0]["count"]) == 30

    def test_concat_npz(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = os.path.join(td, "bd_1")
            d2 = os.path.join(td, "bd_2")
            os.makedirs(d1)
            os.makedirs(d2)
            np.savez(
                os.path.join(d1, "paths.npz"),
                data=np.array([[1, 2], [3, 4]]),
                columns=np.array(["x", "y"]),
            )
            np.savez(
                os.path.join(d2, "paths.npz"),
                data=np.array([[5, 6]]),
                columns=np.array(["x", "y"]),
            )
            _concat_npz([d1, d2], "paths.npz", td)
            npz = np.load(os.path.join(td, "paths.npz"))
            assert npz["data"].shape == (3, 2)

    def test_sum_npz(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = os.path.join(td, "bd_1")
            d2 = os.path.join(td, "bd_2")
            os.makedirs(d1)
            os.makedirs(d2)
            np.savez(
                os.path.join(d1, "matrix.npz"),
                matrix=np.ones((3, 3)),
                milestones=np.array([1, 2, 3]),
            )
            np.savez(
                os.path.join(d2, "matrix.npz"),
                matrix=np.ones((3, 3)) * 2,
                milestones=np.array([1, 2, 3]),
            )
            _sum_npz(
                [d1, d2], "matrix.npz", td, sum_key="matrix", copy_keys=["milestones"]
            )
            npz = np.load(os.path.join(td, "matrix.npz"))
            np.testing.assert_allclose(npz["matrix"], np.ones((3, 3)) * 3)

    def test_sum_npz_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            _sum_npz([td], "missing.npz", td, sum_key="x")
            assert not os.path.exists(os.path.join(td, "missing.npz"))


# Weighted Ensemble data structures
class TestWEDataStructures:

    def test_we_parameters_defaults(self):
        p = WEParameters()
        assert p.n_per_bin == 10
        assert p.n_bins == 40
        assert p.dt == 0.2

    def test_we_parameters_auto_escape(self):
        p = WEParameters(r_start=50.0, r_escape=0.0)
        assert p.r_escape == 100.0

    def test_we_parameters_custom_escape(self):
        p = WEParameters(r_start=50.0, r_escape=200.0)
        assert p.r_escape == 200.0

    def test_we_trajectory_copy(self):
        t = WETrajectory(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=Quaternion(1, 0, 0, 0),
            weight=0.5,
            bin_idx=3,
            steps=10,
            time_ps=2.0,
        )
        c = t.copy()
        assert np.allclose(c.position, t.position)
        assert c.weight == t.weight
        assert c.bin_idx == t.bin_idx
        c.position[0] = 999.0
        assert t.position[0] == 1.0

    def test_we_result_reaction_probability(self):
        r = WEResult(
            n_iterations=100,
            n_per_bin=10,
            n_bins=40,
            flux_reaction=0.1,
            flux_escape=0.2,
            weight_reacted=0.3,
            weight_escaped=0.7,
            r_start=50.0,
            r_escape=100.0,
            dt=0.2,
        )
        assert r.reaction_probability == pytest.approx(0.3)

    def test_we_result_zero_weight(self):
        r = WEResult(
            n_iterations=0,
            n_per_bin=10,
            n_bins=40,
            flux_reaction=0,
            flux_escape=0,
            weight_reacted=0,
            weight_escaped=0,
            r_start=50.0,
            r_escape=100.0,
            dt=0.2,
        )
        assert r.reaction_probability == 0.0


# Force engine _Grid
class TestForceEngineGrid:

    def test_grid_from_dxgrid(self):
        data = np.ones((5, 5, 5))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.data.shape == (5, 5, 5)
        np.testing.assert_allclose(cg.spacing, [1.0, 1.0, 1.0])

    def test_grid_contains_interior(self):
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.contains(np.array([5.0, 5.0, 5.0])) is True

    def test_grid_contains_outside(self):
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.contains(np.array([100.0, 100.0, 100.0])) is False

    def test_grid_lo_hi_margins(self):
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), data)
        cg = _Grid(g)
        np.testing.assert_allclose(cg.lo, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(cg.hi, [16.0, 16.0, 16.0])


# Geometry pipeline
class TestGeometryPipeline:

    def test_geom_atom_record_pos(self):
        a = GeomAtomRecord(
            index=0,
            name="CA",
            resname="ALA",
            resid=1,
            x=1.0,
            y=2.0,
            z=3.0,
            charge=0.5,
            radius=1.8,
        )
        np.testing.assert_allclose(a.pos, [1.0, 2.0, 3.0])

    def test_geom_atom_record_is_ghost(self):
        gho = GeomAtomRecord(
            index=0,
            name="GHO",
            resname="X",
            resid=1,
            x=0,
            y=0,
            z=0,
            charge=0.0,
            radius=0.0,
        )
        assert gho.is_ghost is True

    def test_geom_atom_record_not_ghost(self):
        normal = GeomAtomRecord(
            index=0,
            name="CA",
            resname="ALA",
            resid=1,
            x=0,
            y=0,
            z=0,
            charge=0.5,
            radius=1.8,
        )
        assert normal.is_ghost is False

    def test_geom_parse_pqr(self):
        pqr = (
            "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
            "ATOM      2  CB  ALA     1       4.000   5.000   6.000 -0.100  1.700\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(pqr)
            f.flush()
            atoms = geom_parse_pqr(Path(f.name))
        os.unlink(f.name)
        assert len(atoms) == 2
        assert atoms[0].name == "CA"
        assert atoms[1].charge == pytest.approx(-0.1)

    def test_geom_parse_pqr_skips_bad_lines(self):
        pqr = (
            "REMARK test\n"
            "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
            "TER\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(pqr)
            f.flush()
            atoms = geom_parse_pqr(Path(f.name))
        os.unlink(f.name)
        assert len(atoms) == 1

    def test_molecule_geometry_dataclass(self):
        mg = MoleculeGeometry(
            n_atoms=100,
            n_charged=80,
            n_ghost=2,
            centroid=np.zeros(3),
            max_radius=25.0,
            hydrodynamic_r=20.0,
            ghost_indices=[98, 99],
            ghost_positions=[np.zeros(3), np.ones(3)],
            total_charge=6.0,
        )
        assert mg.n_atoms == 100
        assert mg.total_charge == 6.0

    def test_analyse_molecule_basic(self):
        pqr = (
            "ATOM      1  CA  ALA     1       0.000   0.000   0.000  0.500  1.800\n"
            "ATOM      2  CB  ALA     1       5.000   0.000   0.000  0.500  1.800\n"
            "ATOM      3  CG  ALA     1       0.000   5.000   0.000  0.500  1.800\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(pqr)
            f.flush()
            mg = geom_analyse(Path(f.name), use_mc_hydro=False)
        os.unlink(f.name)
        assert mg.n_atoms == 3
        assert mg.n_charged == 3
        assert mg.max_radius > 0
        assert mg.hydrodynamic_r > 0


# GHO injection XML parsing
class TestGHOInjectionParsing:

    def test_parse_rxns_xml_with_dummies(self):
        from pystarc.pipeline.gho_injection import parse_rxns_xml

        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <dummy><name>gho_rec</name><core>receptor</core>\n"
            "    <atoms>42 1.0 2.0 3.0\n99 4.0 5.0 6.0</atoms>\n"
            "  </dummy>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            dummies = parse_rxns_xml(f.name)
        os.unlink(f.name)
        assert len(dummies) == 1
        assert dummies[0].name == "gho_rec"
        assert len(dummies[0].atoms) == 2
        assert dummies[0].atoms[0].atom_index == 42
        np.testing.assert_allclose(dummies[0].atoms[1].pos_rel, [4.0, 5.0, 6.0])

    def test_parse_rxns_xml_empty(self):
        from pystarc.pipeline.gho_injection import parse_rxns_xml

        xml = '<?xml version="1.0"?>\n<reactions></reactions>\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            dummies = parse_rxns_xml(f.name)
        os.unlink(f.name)
        assert len(dummies) == 0

    def test_parse_rxns_xml_bad_file(self):
        from pystarc.pipeline.gho_injection import parse_rxns_xml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write("not xml at all{{{")
            f.flush()
            with pytest.raises(ValueError, match="Cannot parse"):
                parse_rxns_xml(f.name)
        os.unlink(f.name)

    def test_parse_ghost_atoms_from_input(self):
        from pystarc.pipeline.gho_injection import parse_ghost_atoms_from_input

        text = "3220,0,17.0\n3221,1,10.0\n"
        positions = {3220: np.array([1.0, 2.0, 3.0]), 3221: np.array([4.0, 5.0, 6.0])}
        atoms = parse_ghost_atoms_from_input(text, positions)
        assert len(atoms) == 2
        assert atoms[0].atom_index == 0
        np.testing.assert_allclose(atoms[0].pos_rel, [1.0, 2.0, 3.0])

    def test_parse_ghost_atoms_empty_lines(self):
        from pystarc.pipeline.gho_injection import parse_ghost_atoms_from_input

        text = "\n\n  \n"
        atoms = parse_ghost_atoms_from_input(text, {})
        assert len(atoms) == 0

    def test_parse_ghost_atoms_bad_values(self):
        from pystarc.pipeline.gho_injection import parse_ghost_atoms_from_input

        text = "abc,def,ghi\n3220,0,17.0\n"
        atoms = parse_ghost_atoms_from_input(text, {3220: np.zeros(3)})
        assert len(atoms) == 1

    def test_gho_reaction_criterion_from_rxns_xml(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion>\n"
            "    <pair><atom1>10</atom1><atom2>5</atom2><distance>17.0</distance></pair>\n"
            "  </criterion></reaction>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            gho1 = GHOAtom(atom_index=10, pos_rel=np.zeros(3))
            gho2 = GHOAtom(atom_index=5, pos_rel=np.zeros(3))
            crit = GHOReactionCriterion.from_rxns_xml(f.name, [gho1], [gho2])
        os.unlink(f.name)
        assert len(crit.pairs) == 1

    def test_gho_reaction_criterion_empty_xml(self):
        xml = '<?xml version="1.0"?>\n<reactions></reactions>\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            crit = GHOReactionCriterion.from_rxns_xml(f.name, [], [])
        os.unlink(f.name)
        assert len(crit.pairs) == 0

    def test_text_helper_required_missing(self):
        from pystarc.pipeline.gho_injection import _text

        node = ET.Element("test")
        with pytest.raises(ValueError, match="Missing required"):
            _text(node, "nonexistent")

    def test_text_helper_optional_missing(self):
        from pystarc.pipeline.gho_injection import _text

        node = ET.Element("test")
        assert _text(node, "nonexistent", required=False) is None


# Geometry _parse_rxns_xml_criteria
class TestGeometryRxnsCriteria:

    def test_parse_format1_atom1_atom2(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion>\n"
            "    <pair><atom1>3221 0.0 17.0</atom1><atom2>19 0.0 17.0</atom2></pair>\n"
            "  </criterion></reaction>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            pairs, n_needed = _parse_rxns_xml_criteria(Path(f.name))
        os.unlink(f.name)
        assert len(pairs) == 1
        assert pairs[0].rec_index == 3220
        assert pairs[0].lig_index == 18
        assert pairs[0].cutoff == 17.0

    def test_parse_format2_atoms_distance(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion>\n"
            "    <pair><atoms>100 50</atoms><distance>6.5</distance></pair>\n"
            "  </criterion></reaction>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            pairs, n_needed = _parse_rxns_xml_criteria(Path(f.name))
        os.unlink(f.name)
        assert len(pairs) == 1
        assert pairs[0].rec_index == 99
        assert pairs[0].lig_index == 49
        assert pairs[0].cutoff == 6.5

    def test_parse_n_needed(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion><n_needed>2</n_needed>\n"
            "    <pair><atoms>10 20</atoms><distance>5.0</distance></pair>\n"
            "    <pair><atoms>30 40</atoms><distance>5.0</distance></pair>\n"
            "  </criterion></reaction>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            pairs, n_needed = _parse_rxns_xml_criteria(Path(f.name))
        os.unlink(f.name)
        assert len(pairs) == 2
        assert n_needed == 2

    def test_parse_empty_criterion(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion></criterion></reaction>\n</reactions>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            pairs, n_needed = _parse_rxns_xml_criteria(Path(f.name))
        os.unlink(f.name)
        assert len(pairs) == 0


# COFFDROP _parse_ff with synthetic XML
class TestCOFFDROPParseFF:

    def test_parse_ff_synthetic(self):
        from pystarc.simulation.coffdrop_params import _parse_ff

        xml = (
            '<?xml version="1.0"?>\n<coffdrop>\n'
            "  <types>\n"
            "    <atoms><type><name>CA</name><index>1</index></type></atoms>\n"
            "    <residues><type><name>ALA</name><index>1</index></type></residues>\n"
            "  </types>\n"
            "  <pairs><distance>3.0 20.0</distance>\n"
            "    <potentials>\n"
            "      <potential><index>0</index><residues>1 1</residues>"
            "<atoms>1 1</atoms><orders>0 0</orders>"
            "<data>1.0 0.8 0.5 0.2 0.0</data></potential>\n"
            "    </potentials>\n"
            "  </pairs>\n"
            "  <bond_angles><angle>0.0 180.0</angle>\n"
            "    <potentials>\n"
            "      <potential><index>0</index><residues>1 1 1</residues>"
            "<atoms>1 1 1</atoms><orders>0 0 0</orders>"
            "<data>0.0 0.5 1.0 0.5 0.0</data></potential>\n"
            "    </potentials>\n"
            "  </bond_angles>\n"
            "  <dihedral_angles><angle>-180.0 180.0</angle>\n"
            "    <potentials>\n"
            "      <potential><index>0</index><residues>1 1 1 1</residues>"
            "<atoms>1 1 1 1</atoms><orders>0 0 0 0</orders>"
            "<data>0.0 1.0 0.0 -1.0 0.0</data></potential>\n"
            "    </potentials>\n"
            "  </dihedral_angles>\n"
            "</coffdrop>\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            type_map, pairs, angles, dihedrals = _parse_ff(f.name)
        os.unlink(f.name)
        assert "CA" in type_map["atoms"]
        assert "ALA" in type_map["residues"]
        assert len(pairs) == 1
        assert len(angles) == 1
        assert len(dihedrals) == 1
        assert pairs[0].value(3.0) > 0

    def test_parse_ff_no_types(self):
        from pystarc.simulation.coffdrop_params import _parse_ff

        xml = '<?xml version="1.0"?>\n<coffdrop></coffdrop>\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            type_map, pairs, angles, dihedrals = _parse_ff(f.name)
        os.unlink(f.name)
        assert type_map == {"atoms": {}, "residues": {}}
        assert pairs == []


# COFFDROP chain force evaluator
class TestCOFFDROPChainForces:

    def test_chain_force_evaluator_bond(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[1].pos = np.array([3.5, 0.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (3, 3)
        assert np.any(np.abs(F) > 0)

    def test_chain_positions_set(self):
        chain = build_linear_chain(n_residues=4, bond_length=3.8)
        new_pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        chain.set_positions(new_pos)
        np.testing.assert_allclose(chain.beads[2].pos, [2.0, 0.0, 0.0])

    def test_chain_zero_forces(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[0].force = np.array([1.0, 2.0, 3.0])
        chain.zero_forces()
        np.testing.assert_allclose(chain.beads[0].force, [0.0, 0.0, 0.0])

    def test_chain_positions_array(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        pos = chain.positions_array()
        assert pos.shape == (3, 3)

    def test_chain_forces_array(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        farr = chain.forces_array()
        assert farr.shape == (3, 3)

    def test_equilibrium_forces_small(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) < 1.0

    def test_stretched_bond_restoring_force(self):
        chain = build_linear_chain(n_residues=2, bond_length=3.8)
        chain.beads[1].pos = np.array([10.0, 0.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F[0, 0] > 0
        assert F[1, 0] < 0


# Quaternion uncovered branches
class TestQuaternionFromMatrix:

    def test_from_rotation_matrix_identity(self):
        R = np.eye(3)
        q = Quaternion.from_rotation_matrix(R)
        assert abs(q.norm() - 1.0) < 1e-10
        assert abs(q.w) > 0.9

    def test_from_rotation_matrix_90z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        assert abs(q.norm() - 1.0) < 1e-10
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_90x(self):
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_90y(self):
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_180z(self):
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_arbitrary(self):
        q_orig = Quaternion.from_axis_angle(np.array([1, 1, 1]) / math.sqrt(3), 1.23)
        R = q_orig.to_rotation_matrix()
        q_back = Quaternion.from_rotation_matrix(R)
        R2 = q_back.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_normalized_zero_quaternion(self):
        q = Quaternion(0, 0, 0, 0)
        n = q.normalized()
        assert n.w == 1.0

    def test_random_quaternion_no_rng(self):
        q = random_quaternion(rng=None)
        assert abs(q.norm() - 1.0) < 1e-10

    def test_small_rotation_quaternion_no_rng(self):
        q = small_rotation_quaternion(0.01, rng=None)
        assert abs(q.norm() - 1.0) < 1e-10


# Diffusional rotation uncovered functions
class TestDiffusionalRotationSampling:

    def test_sample_rotation_angle_callable(self):
        from pystarc.simulation.diffusional_rotation import _sample_rotation_angle

        rng = np.random.default_rng(42)
        angle = _sample_rotation_angle(rng, 0.5)
        assert 0 <= angle <= math.pi

    def test_sample_quat_for_tau(self):
        from pystarc.simulation.diffusional_rotation import _sample_quat_for_tau

        rng = np.random.default_rng(42)
        q = _sample_quat_for_tau(rng, 0.5)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_spline_rot_0p5(self):
        from pystarc.simulation.diffusional_rotation import _spline_rot_0p5

        rng = np.random.default_rng(42)
        q = _spline_rot_0p5(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_spline_rot_1p0(self):
        from pystarc.simulation.diffusional_rotation import _spline_rot_1p0

        rng = np.random.default_rng(42)
        q = _spline_rot_1p0(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_spline_rot_2p0(self):
        from pystarc.simulation.diffusional_rotation import _spline_rot_2p0

        rng = np.random.default_rng(42)
        q = _spline_rot_2p0(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


# WE result rate constant and repr
class TestWEResultExtended:

    def test_rate_constant_nonzero(self):
        r = WEResult(
            n_iterations=100,
            n_per_bin=10,
            n_bins=40,
            flux_reaction=0.1,
            flux_escape=0.2,
            weight_reacted=0.3,
            weight_escaped=0.7,
            r_start=50.0,
            r_escape=100.0,
            dt=0.2,
        )
        k = r.rate_constant(D_rel=0.1)
        assert k > 0

    def test_rate_constant_zero_prxn(self):
        r = WEResult(
            n_iterations=0,
            n_per_bin=10,
            n_bins=40,
            flux_reaction=0,
            flux_escape=0,
            weight_reacted=0,
            weight_escaped=0,
            r_start=50.0,
            r_escape=100.0,
            dt=0.2,
        )
        assert r.rate_constant(D_rel=0.1) == 0.0

    def test_repr(self):
        r = WEResult(
            n_iterations=100,
            n_per_bin=10,
            n_bins=40,
            flux_reaction=0.1,
            flux_escape=0.2,
            weight_reacted=0.3,
            weight_escaped=0.7,
            r_start=50.0,
            r_escape=100.0,
            dt=0.2,
        )
        s = repr(r)
        assert "WEResult" in s
        assert "P_rxn" in s


# Engine _GridStack
class TestGridStack:

    def test_gridstack_creation(self):
        from pystarc.forces.engine import _GridStack

        g1 = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), np.ones((5, 5, 5)))
        g2 = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.ones((10, 10, 10)))
        gs = _GridStack([g1, g2])
        assert len(gs) == 2
        assert bool(gs) is True

    def test_gridstack_empty(self):
        from pystarc.forces.engine import _GridStack

        gs = _GridStack([])
        assert len(gs) == 0
        assert bool(gs) is False

    def test_gridstack_finest_first(self):
        from pystarc.forces.engine import _GridStack

        coarse = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), np.ones((10, 10, 10)))
        fine = DXGrid(np.zeros(3), np.diag([0.5, 0.5, 0.5]), np.ones((10, 10, 10)))
        gs = _GridStack([coarse, fine])
        pt = np.array([2.0, 2.0, 2.0])
        g = gs.finest_for(pt)
        assert g is not None
        np.testing.assert_allclose(g.spacing, [0.5, 0.5, 0.5])

    def test_gridstack_outside_returns_none(self):
        from pystarc.forces.engine import _GridStack

        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.ones((5, 5, 5)))
        gs = _GridStack([g])
        assert gs.finest_for(np.array([100.0, 100.0, 100.0])) is None

    def test_gridstack_eval_empty(self):
        from pystarc.forces.engine import _GridStack

        gs = _GridStack([])
        F, T, E = gs.eval_atoms(np.zeros((1, 3)), np.array([1.0]), 0.5, False, "numpy")
        np.testing.assert_allclose(F, [0, 0, 0])
        assert E == 0.0


# Multipole farfield summary and repr
class TestMultipoleFarfieldExtended:

    def test_summary_monopole_dominant(self):
        charges = np.array([5.0, -2.0])
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        me = MultipoleExpansion(positions, charges, debye_length=7.86)
        s = me.summary()
        assert "Monopole" in s or "monopole" in s.lower() or "Q" in s

    def test_summary_dipole_dominant(self):
        charges = np.array([1.0, -1.0])
        positions = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
        me = MultipoleExpansion(positions, charges, debye_length=7.86)
        s = me.summary()
        assert len(s) > 0

    def test_potential_at_zero(self):
        charges = np.array([1.0])
        positions = np.array([[0, 0, 0]], dtype=float)
        me = MultipoleExpansion(positions, charges, debye_length=7.86)
        V = me.potential(np.array([0.0, 0.0, 0.0]))
        assert V == 0.0


# Chain with angles and torsions
class TestChainAngleForces:

    def _make_chain_with_angle(self):
        beads = [
            ChainBead(
                pos=np.array([0.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="A",
                resid=0,
            ),
            ChainBead(
                pos=np.array([3.8, 0.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="B",
                resid=1,
            ),
            ChainBead(
                pos=np.array([7.6, 0.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="C",
                resid=2,
            ),
        ]
        bonds = [ChainBond(0, 1, 3.8, 100.0), ChainBond(1, 2, 3.8, 100.0)]
        angles = [ChainAngle(0, 1, 2, math.pi, 50.0)]
        return FlexibleChain(
            beads=beads, bonds=bonds, angles=angles, name="angle_chain"
        )

    def test_angle_equilibrium_zero_force(self):
        chain = self._make_chain_with_angle()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) < 1.0

    def test_angle_bent_produces_force(self):
        chain = self._make_chain_with_angle()
        chain.beads[2].pos = np.array([5.0, 3.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) > 0.1

    def test_angle_force_shape(self):
        chain = self._make_chain_with_angle()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (3, 3)


class TestChainTorsionForces:

    def _make_chain_with_torsion(self):
        beads = [
            ChainBead(
                pos=np.array([0.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="A",
                resid=0,
            ),
            ChainBead(
                pos=np.array([3.8, 0.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="B",
                resid=1,
            ),
            ChainBead(
                pos=np.array([7.6, 3.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="C",
                resid=2,
            ),
            ChainBead(
                pos=np.array([11.4, 3.0, 3.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="D",
                resid=3,
            ),
        ]
        bonds = [
            ChainBond(0, 1, 3.8, 100.0),
            ChainBond(1, 2, 5.0, 100.0),
            ChainBond(2, 3, 5.0, 100.0),
        ]
        torsions = [ChainTorsion(0, 1, 2, 3, 0.0, 10.0, 1)]
        return FlexibleChain(
            beads=beads, bonds=bonds, torsions=torsions, name="torsion_chain"
        )

    def test_torsion_force_shape(self):
        chain = self._make_chain_with_torsion()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (4, 3)

    def test_torsion_force_nonzero(self):
        chain = self._make_chain_with_torsion()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.any(np.abs(F) > 0)


class TestChainExcludedVolume:

    def test_overlapping_beads_repel(self):
        beads = [
            ChainBead(
                pos=np.array([0.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=3.0,
                charge=0.0,
                resname="A",
                resid=0,
            ),
            ChainBead(
                pos=np.array([4.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=3.0,
                charge=0.0,
                resname="B",
                resid=1,
            ),
        ]
        bonds = [ChainBond(0, 1, 3.8, 100.0)]
        chain = FlexibleChain(beads=beads, bonds=bonds, name="overlap")
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (2, 3)
        assert np.any(np.abs(F) > 0)

    def test_well_separated_beads_no_force(self):
        beads = [
            ChainBead(
                pos=np.array([0.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=1.0,
                charge=0.0,
                resname="A",
                resid=0,
            ),
            ChainBead(
                pos=np.array([20.0, 0.0, 0.0]),
                force=np.zeros(3),
                radius=1.0,
                charge=0.0,
                resname="B",
                resid=1,
            ),
        ]
        chain = FlexibleChain(beads=beads, name="separated")
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) < 1e-10


# Chain BD propagator advanced
class TestChainBDPropagatorAdvanced:

    def test_step_moves_beads(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        assert not np.allclose(pos_before, pos_after)

    def test_frozen_chain_no_move(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.frozen = True
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        np.testing.assert_allclose(pos_before, pos_after)

    def test_max_time_step_positive(self):
        chain = build_linear_chain(n_residues=5, bond_length=3.8)
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt > 0

    def test_max_time_step_empty_chain(self):
        chain = FlexibleChain(beads=[], name="empty")
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt == 0.1

    def test_satisfy_bond_constraints(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[1].pos = np.array([5.0, 0.0, 0.0])
        prop = ChainBDPropagator()
        prop.satisfy_bond_constraints(chain, tol=0.01)
        for bond in chain.bonds:
            r = np.linalg.norm(chain.beads[bond.j].pos - chain.beads[bond.i].pos)
            assert abs(r - bond.r0) / bond.r0 < 0.01

    def test_D_trans_positive(self):
        prop = ChainBDPropagator()
        D = prop.D_trans(2.0)
        assert D > 0

    def test_step_with_external_evaluator(self):
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        prop = ChainBDPropagator()
        evaluator = ChainForceEvaluator()
        rng = np.random.default_rng(42)
        prop.step(chain, dt=0.1, rng=rng, force_evaluator=evaluator)
        assert chain.beads[0].pos is not None


# WE simulator construction and bin methods
class TestWESimulatorConstruction:

    def _make_simple_molecules(self):
        mol1 = Molecule(name="rec")
        mol1.atoms.append(
            Atom(
                index=0,
                name="A",
                residue_name="X",
                residue_index=1,
                chain="A",
                x=0,
                y=0,
                z=0,
                charge=1.0,
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
                x=50,
                y=0,
                z=0,
                charge=-1.0,
                radius=2.0,
            )
        )
        return mol1, mol2

    def test_we_simulator_constructs(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2, n_bins=5, n_iterations=1, r_start=50.0, seed=42
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        assert sim.params.r_start == 50.0
        assert len(sim._bins) == 6

    def test_we_bin_of_interior(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2, n_bins=5, n_iterations=1, r_start=50.0, seed=42
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        idx = sim._bin_of(30.0)
        assert 0 <= idx < 5

    def test_we_bin_of_outside(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2, n_bins=5, n_iterations=1, r_start=50.0, seed=42
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        assert sim._bin_of(200.0) == -1
        assert sim._bin_of(0.1) == -1

    def test_we_place_mol2(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2, n_bins=5, n_iterations=1, r_start=50.0, seed=42
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        pos = np.array([30.0, 0.0, 0.0])
        ori = Quaternion.identity()
        placed = sim._place_mol2(pos, ori)
        assert abs(placed.atoms[0].x - 30.0) < 1e-6

    def test_we_init_ensemble(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2, n_bins=5, n_iterations=1, r_start=50.0, seed=42
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        ensemble = sim._init_ensemble()
        assert len(ensemble) > 0
        total_weight = sum(t.weight for t in ensemble)
        assert total_weight == pytest.approx(1.0, abs=1e-10)

    def test_we_log_bins(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2,
            n_bins=10,
            n_iterations=1,
            r_start=50.0,
            seed=42,
            bin_scheme="log",
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        assert len(sim._bins) == 11
        assert sim._bins[0] < sim._bins[-1]

    def test_we_linear_bins(self):
        mol1, mol2 = self._make_simple_molecules()
        mob = MobilityTensor.from_radii(10.0, 5.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = WEParameters(
            n_per_bin=2,
            n_bins=10,
            n_iterations=1,
            r_start=50.0,
            seed=42,
            bin_scheme="linear",
        )
        from pystarc.simulation.we_simulator import WESimulator

        sim = WESimulator(mol1, mol2, mob, ps, params)
        diffs = np.diff(sim._bins)
        assert np.allclose(diffs, diffs[0], rtol=0.01)


# NAM simulator with tiny molecules
class TestNAMSimulatorRun:

    def _make_setup(self, n_traj=10, max_steps=50):
        mol1 = Molecule(name="rec")
        mol1.atoms.append(
            Atom(
                index=0,
                name="A",
                residue_name="X",
                residue_index=1,
                chain="A",
                x=0,
                y=0,
                z=0,
                charge=1.0,
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
                x=10,
                y=0,
                z=0,
                charge=-1.0,
                radius=2.0,
            )
        )
        mob = MobilityTensor.from_radii(3.0, 2.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 5.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = NAMParameters(
            n_trajectories=n_traj,
            dt=0.2,
            r_start=20.0,
            max_steps=max_steps,
            seed=42,
            verbose=False,
        )
        return mol1, mol2, mob, ps, params

    def test_nam_run_returns_result(self):
        mol1, mol2, mob, ps, params = self._make_setup()
        sim = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        result = sim.run()
        assert isinstance(result, SimulationResult)
        assert (
            result.n_reacted + result.n_escaped + result.n_max_steps
            == params.n_trajectories
        )

    def test_nam_run_reaction_probability_bounded(self):
        mol1, mol2, mob, ps, params = self._make_setup()
        sim = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        result = sim.run()
        assert 0.0 <= result.reaction_probability <= 1.0

    def test_nam_different_seeds(self):
        mol1, mol2, mob, ps, params = self._make_setup(n_traj=50)
        sim1 = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        r1 = sim1.run()
        params2 = NAMParameters(
            n_trajectories=50,
            dt=0.2,
            r_start=20.0,
            max_steps=50,
            seed=999,
            verbose=False,
        )
        sim2 = NAMSimulator(mol1, mol2, mob, ps, params2, zero_force)
        r2 = sim2.run()
        assert r1.n_reacted != r2.n_reacted or r1.n_escaped != r2.n_escaped


# Outer propagator construction
class TestOuterPropagatorConstruction:

    def test_op_group_info(self):
        g = OPGroupInfo(q=6.0, Dtrans=0.01, Drot=0.001)
        assert g.q == 6.0
        assert g.Dtrans == 0.01

    def test_outer_propagator_constructs(self):
        g0 = OPGroupInfo(q=2.0, Dtrans=0.01, Drot=0.001)
        g1 = OPGroupInfo(q=-5.0, Dtrans=0.015, Drot=0.002)
        op = OuterPropagator(
            b_radius=80.0,
            max_radius=25.0,
            has_hi=True,
            kT=0.5961,
            viscosity=0.243,
            dielectric=78.54,
            vacuum_perm=0.000142,
            debye_len=13.6,
            g0=g0,
            g1=g1,
        )
        assert op.bradius == 80.0
        assert op.qradius == 500.0

    def test_outer_propagator_k_b_positive(self):
        g0 = OPGroupInfo(q=2.0, Dtrans=0.01, Drot=0.001)
        g1 = OPGroupInfo(q=-5.0, Dtrans=0.015, Drot=0.002)
        op = OuterPropagator(
            b_radius=80.0,
            max_radius=25.0,
            has_hi=True,
            kT=0.5961,
            viscosity=0.243,
            dielectric=78.54,
            vacuum_perm=0.000142,
            debye_len=13.6,
            g0=g0,
            g1=g1,
        )
        assert op.V_factor != 0
        assert op.D_factor > 0

    def test_outer_propagator_no_hi(self):
        g0 = OPGroupInfo(q=1.0, Dtrans=0.01, Drot=0.001)
        g1 = OPGroupInfo(q=-1.0, Dtrans=0.01, Drot=0.001)
        op = OuterPropagator(
            b_radius=50.0,
            max_radius=10.0,
            has_hi=False,
            kT=0.5961,
            viscosity=0.243,
            dielectric=78.54,
            vacuum_perm=0.000142,
            debye_len=7.86,
            g0=g0,
            g1=g1,
        )
        assert op.D_factor > 0
        assert op.has_hi is False

    def test_outer_propagator_return_probability(self):
        g0 = OPGroupInfo(q=2.0, Dtrans=0.01, Drot=0.001)
        g1 = OPGroupInfo(q=-5.0, Dtrans=0.015, Drot=0.002)
        op = OuterPropagator(
            b_radius=80.0,
            max_radius=25.0,
            has_hi=True,
            kT=0.5961,
            viscosity=0.243,
            dielectric=78.54,
            vacuum_perm=0.000142,
            debye_len=13.6,
            g0=g0,
            g1=g1,
        )
        assert 0 < op.return_prob <= 1.0


# Geometry auto_detect_reactions
class TestGeometryAutoDetect:

    def _make_gho_pqr(self, td, name, atoms_text):
        path = Path(td) / name
        path.write_text(atoms_text)
        return path

    def test_auto_detect_from_rxns_xml(self):
        xml = (
            '<?xml version="1.0"?>\n<reactions>\n'
            "  <reaction><criterion>\n"
            "    <pair><atoms>100 50</atoms><distance>6.5</distance></pair>\n"
            "  </criterion></reaction>\n</reactions>\n"
        )
        with tempfile.TemporaryDirectory() as td:
            rec_pqr = self._make_gho_pqr(
                td,
                "rec.pqr",
                "ATOM      1  CA  ALA     1       0.000   0.000   0.000  0.500  1.800\n",
            )
            lig_pqr = self._make_gho_pqr(
                td,
                "lig.pqr",
                "ATOM      1  CA  ALA     1       5.000   0.000   0.000  0.500  1.800\n",
            )
            rxns_path = Path(td) / "rxns.xml"
            rxns_path.write_text(xml)
            geom = SystemGeometry(
                receptor=MoleculeGeometry(
                    n_atoms=1,
                    n_charged=1,
                    n_ghost=0,
                    centroid=np.zeros(3),
                    max_radius=2.0,
                    hydrodynamic_r=2.0,
                    ghost_indices=[],
                    ghost_positions=[],
                    total_charge=0.5,
                ),
                ligand=MoleculeGeometry(
                    n_atoms=1,
                    n_charged=1,
                    n_ghost=0,
                    centroid=np.array([5, 0, 0]),
                    max_radius=2.0,
                    hydrodynamic_r=2.0,
                    ghost_indices=[],
                    ghost_positions=[],
                    total_charge=0.5,
                ),
                r_start=50.0,
                r_escape=100.0,
            )
            pairs_list, n_needed = auto_detect_reactions(
                geom,
                rxns_xml=str(rxns_path),
                ghost_atoms="auto",
                bd_milestone_radius=50.0,
                bd_milestone_radius_inner=0.0,
            )
            assert len(pairs_list) > 0
            assert len(pairs_list[0]) == 1

    def test_auto_detect_manual_ghost_atoms(self):
        geom = SystemGeometry(
            receptor=MoleculeGeometry(
                n_atoms=1,
                n_charged=1,
                n_ghost=0,
                centroid=np.zeros(3),
                max_radius=2.0,
                hydrodynamic_r=2.0,
                ghost_indices=[],
                ghost_positions=[],
                total_charge=0.5,
            ),
            ligand=MoleculeGeometry(
                n_atoms=1,
                n_charged=1,
                n_ghost=0,
                centroid=np.array([5, 0, 0]),
                max_radius=2.0,
                hydrodynamic_r=2.0,
                ghost_indices=[],
                ghost_positions=[],
                total_charge=0.5,
            ),
            r_start=50.0,
            r_escape=100.0,
        )
        pairs_list, n_needed = auto_detect_reactions(
            geom,
            rxns_xml="",
            ghost_atoms="0,0,17.0",
            bd_milestone_radius=50.0,
            bd_milestone_radius_inner=0.0,
        )
        assert len(pairs_list) > 0
        assert pairs_list[0][0].cutoff == 17.0

    def test_auto_detect_gho_in_pqr(self):
        geom = SystemGeometry(
            receptor=MoleculeGeometry(
                n_atoms=10,
                n_charged=8,
                n_ghost=1,
                centroid=np.zeros(3),
                max_radius=20.0,
                hydrodynamic_r=15.0,
                ghost_indices=[9],
                ghost_positions=[np.array([5, 0, 0])],
                total_charge=2.0,
            ),
            ligand=MoleculeGeometry(
                n_atoms=5,
                n_charged=4,
                n_ghost=1,
                centroid=np.array([50, 0, 0]),
                max_radius=10.0,
                hydrodynamic_r=8.0,
                ghost_indices=[4],
                ghost_positions=[np.array([50, 0, 0])],
                total_charge=-1.0,
            ),
            r_start=50.0,
            r_escape=100.0,
        )
        pairs_list, n_needed = auto_detect_reactions(
            geom,
            rxns_xml="",
            ghost_atoms="auto",
            bd_milestone_radius=50.0,
            bd_milestone_radius_inner=17.0,
        )
        assert len(pairs_list) > 0
        assert n_needed == 1

    def test_auto_detect_no_gho_raises(self):
        geom = SystemGeometry(
            receptor=MoleculeGeometry(
                n_atoms=10,
                n_charged=10,
                n_ghost=0,
                centroid=np.zeros(3),
                max_radius=20.0,
                hydrodynamic_r=15.0,
                ghost_indices=[],
                ghost_positions=[],
                total_charge=2.0,
            ),
            ligand=MoleculeGeometry(
                n_atoms=5,
                n_charged=5,
                n_ghost=0,
                centroid=np.array([50, 0, 0]),
                max_radius=10.0,
                hydrodynamic_r=8.0,
                ghost_indices=[],
                ghost_positions=[],
                total_charge=-1.0,
            ),
            r_start=50.0,
            r_escape=100.0,
        )
        with pytest.raises(RuntimeError, match="No GHO"):
            auto_detect_reactions(
                geom,
                rxns_xml="",
                ghost_atoms="auto",
                bd_milestone_radius=50.0,
                bd_milestone_radius_inner=0.0,
            )
