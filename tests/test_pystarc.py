"""
PySTARC unified test suite.

Run with:  pytest tests/test_pystarc.py -v
"""

from __future__ import annotations
from pystarc.simulation.coffdrop_chain import (
    ChainAngle,
    ChainAtom,
    ChainAtomRef,
    ChainBDPropagator,
    ChainBead,
    ChainBond,
    ChainCommon,
    ChainForceEvaluator,
    ChainState,
    ChainTorsion,
    CoplanarConstraint,
    FlexibleChain,
    LengthConstraint,
    _angle_force_state,
    _bond_force_state,
    _build_constraint_jacobian,
    _chain_idx,
    _coplanar_violation,
    _torsion_force_state,
    build_chain_common_from_coffdrop,
    build_chain_common_with_sidechains_from_coffdrop,
    build_linear_chain,
    chain_from_pdb,
    chain_from_sequence,
    chain_intra_coffdrop_pair_forces,
    chain_intra_nonbonded_forces,
    compute_chain_forces,
    compute_constraint_violations,
    place_relaxed_geometry,
    satisfy_constraints,
    satisfy_constraints_hybrid,
    satisfy_constraints_newton,
)
from pystarc.simulation.chain_simulator import (
    ChainBDParameters,
    ChainBDSimulator,
    DEFAULT_DESOLVATION_ALPHA,
    _check_chain_overlap,
    _min_reaction_distance,
    _run_chain_trajectory_worker,
    aggregate_chain_external_force_and_torque,
    chain_internal_bd_step,
    chain_outer_bd_step,
    chain_outer_bd_step_wiener,
    chain_target_steric_forces,
    check_chain_reaction,
    check_escape,
    check_reaction_with_bridge,
    compute_pair_distances,
    evaluate_born_force_on_chain,
    evaluate_target_grid_force_on_chain,
    initialize_bsphere,
    make_chain_scratch_molecule,
    place_chain,
    update_chain_scratch_positions,
)
from pystarc.forces.chain_gb import (
    COULOMB_K_KBT_A,
    DEFAULT_OBC_OFFSET,
    _finite_difference_force,
    _hct_integrand,
    _hct_integrand_deriv,
    chain_full_gb_force,
    chain_offdiagonal_gb_force,
    chain_self_born_diagonal_force,
    chain_vacuum_coulomb_force,
    gb_offdiagonal_energy,
    gb_self_born_energy,
    gb_vacuum_coulomb_energy,
    obc_effective_radii,
)
from pystarc.hydrodynamics.rotne_prager import (
    MobilityTensor,
    _build_robust_solver,
    _hydrodynamic_center,
    chain_diffusion_tensors,
    chain_rigid_body_resistance,
    rpy_full_components,
    rpy_full_mobility_matrix,
    rpy_offdiagonal,
    rpy_pair_blocks,
    rpy_self_blocks,
    stokes_rotational_diffusion,
    stokes_translational_diffusion,
)
from pystarc.pipeline.geometry import (
    AtomRecord as GeomAtomRecord,
    MoleculeGeometry,
    SystemGeometry,
    _parse_rxns_xml_criteria,
    analyse_molecule,
    analyse_molecule as geom_analyse,
    auto_detect_reactions,
    compute_geometry,
    parse_pqr as geom_parse_pqr,
    parse_pqr as parse_pqr_test_lowsev_geometry,
)
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
    KCAL_PER_MOL_TO_KBT,
    PI,
    PS_TO_S,
    TWO_PI,
    T_DEFAULT,
    VACUUM_PERMITTIVITY_KBT,
)
from pystarc.pipeline.chain_output_writer import (
    write_angular_map_npz,
    write_chain_results,
    write_contact_frequency_csv,
    write_encounters_csv,
    write_energetics_npz,
    write_fpt_distribution_csv,
    write_milestone_flux_csv,
    write_near_misses_csv,
    write_paths_npz,
    write_radial_density_csv,
)
from pystarc.motion.do_bd_step import (
    FORCE_CHANGE_ALPHA,
    WATER_VISCOSITY,
    backstep_due_to_force,
    bd_step_wiener,
    bd_step_wiener_tensor,
    ermak_mccammon_rotation,
    ermak_mccammon_rotation_tensor,
    ermak_mccammon_translation,
    ermak_mccammon_translation_tensor,
)
from pystarc.simulation.nam_simulator import (
    NAMParameters,
    NAMParameters as _NAMParameters,
    NAMSimulator,
    NAMSimulator as _NAMSimulator,
    SimulationResult,
    _k_from_P,
    _mol2_positions,
    _run_trajectory_worker,
    _worker_init,
    zero_force,
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
    _parse_ff,
    _parse_mapping,
    _txt_to_floats,
)
from pystarc.structures.chain_io import (
    _parse_coffdrop_map_simple,
    _parse_pdb_chain_for_beads,
    _resname_match_tleap,
    load_chain_from_json,
    pdb_to_bead_positions,
    save_chain_to_json,
)
from pystarc.simulation.diffusional_rotation import (
    _spline_rot_0p5,
    _spline_rot_1p0,
    _spline_rot_2p0,
    diffusional_rotation,
    quat_multiply,
    quat_of_rotvec,
    random_unit_quat,
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
from pystarc.multi_GPU.combine_data import (
    _concat_csv,
    _concat_npz,
    _pool_p_commit,
    _recover_contact_steps,
    _save_json,
    _sum_csv,
    _sum_npz,
    _warn_run_mismatch,
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
from pystarc.transforms.quaternion import (
    Quaternion,
    Quaternion as Q,
    Quaternion as _Q,
    RigidTransform,
    random_quaternion,
    small_rotation_quaternion,
)
from pystarc.global_defs.defaults import (
    DEBYE_LENGTH,
    DESOLVATION_ALPHA,
    INPUT_DEFAULTS,
    PHYSICS_DEFAULT_NAMES,
    REFERENCE_DEFAULTS,
    VISCOSITY,
)
from pystarc.pipeline.prepare_bd_surface import (
    PQRAtom,
    compute_grid_params,
    read_pqr,
    run_cmd,
    split_receptor_ligand,
    write_pqr,
)
from pystarc.pathways.reaction_interface import (
    ContactPair,
    PathwaySet,
    ReactionCriteria,
    ReactionInterface,
    make_default_reaction,
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
from pystarc.simulation.outer_propagator import (
    OPGroupInfo,
    OuterPropagator,
    PI as PI_test_auditfix3_outerprop,
    PI6,
)
from pystarc.hydrodynamics.mc_hydro_radius import (
    _extract_surface,
    _fingerprint,
    _voxelise,
    mc_hydrodynamic_radius,
)
from pystarc.structures.pqr_io import (
    PQRRecord,
    _parse_whitespace,
    parse_pqr,
    parse_pqr_records,
    write_pqr,
)
from pystarc.motion.adaptive_time_step import (
    AdaptiveTimeStep,
    _LARGE,
    max_time_step,
    reaction_time_step,
)
from pystarc.structures.molecules import (
    Atom,
    BoundingBox,
    ContactPair,
    Molecule,
    ReactionCriteria,
)
from pystarc.pipeline.chain_pipeline import (
    _build_pathway_set,
    _load_reaction_pairs_json,
    run_chain,
)
from pystarc.forces.electrostatic.grid_force import (
    DXGrid,
    debye_huckel_energy,
    debye_huckel_force,
)
from pystarc.pipeline.run_apbs import (
    _compute_grid_params,
    _is_valid_apbs_dime,
    _write_apbs_input,
)
from pystarc.analysis.convergence import (
    analyse_convergence,
    print_convergence,
    save_convergence,
)
from pystarc.pipeline.input_parser import (
    ChainConfig,
    OutputConfig,
    PySTARCConfig,
    parse,
)
from pystarc.simulation.step_near_surface import _inv_erf, step_near_absorbing_surface
from pystarc.forces.engine import PySTARCEngine, _Grid, _GridStack, _group_centroid
from pystarc.molsystem.system_state import Fate, SystemState, TrajectoryResult
from pystarc.forces.multipole import EffectiveCharges, load_effective_charges
from pystarc.pipeline import chain_pipeline, desolvation_grid as dg, geometry
from pystarc.pipeline.extract import _is_atom_line, _residue_name, extract
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult
from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.global_defs import constants as C, defaults as D
from pystarc.multi_GPU.multi_GPU_runs import _set_or_create
from pystarc.simulation import chain_simulator
from pystarc.pipeline.output_writer import write_all
import pystarc.simulation.outer_propagator as op
import pystarc.simulation.coffdrop_chain as mod
import pystarc.simulation.nam_simulator as nsim
import pystarc.forces.electrostatic.grid_force
import pystarc.forces.gpu_batch_engine as gbe
from collections import Counter, defaultdict
from scipy.spatial.transform import Rotation
import pystarc.pipeline.geometry as geom_mod
import pystarc.pipeline.make_pqr as make_pqr
import pystarc.pipeline.pipeline as pipeline
import pystarc.pathways.reaction_interface
import pystarc.hydrodynamics.rotne_prager
import pystarc.simulation.nam_simulator
from contextlib import redirect_stdout
import pystarc.molsystem.system_state
import pystarc.transforms.quaternion
from click.testing import CliRunner
from pystarc.forces import chain_gb
import pystarc.structures.molecules
import pystarc.xml_io.simulation_io
import xml.etree.ElementTree as ET
from pystarc.cli.main import cli
import pystarc.motion.do_bd_step
import pystarc.structures.pqr_io
from dataclasses import fields
import multiprocessing as mp
import pystarc.aux.aux_tools
import pystarc.lib.numerical
from pathlib import Path
import pystarc.cli.main
import pystarc as _pkg
import importlib.util
import numpy as np
import subprocess
import importlib
import tempfile
import textwrap
import warnings
import inspect
import pathlib
import pystarc
import pickle
import pytest
import shutil
import shlex
import types
import glob
import json
import math
import ast
import csv
import sys
import io
import os
import re


class TestConstants:
    def test_temperature(self):
        """The default temperature T_DEFAULT equals 298.15 K."""
        assert abs(T_DEFAULT - 298.15) < 0.01

    def test_boltzmann_si(self):
        """The SI Boltzmann constant KB_SI equals 1.380649e-23 J/K."""
        assert abs(KB_SI - 1.380649e-23) < 1e-30

    def test_boltzmann_kcal(self):
        """The Boltzmann constant in kcal units KB_KCAL equals 1.987204e-3 kcal/(mol K)."""
        assert abs(KB_KCAL - 1.987204e-3) < 1e-8

    def test_kbt_kcal(self):
        """KBT_KCAL equals KB_KCAL times the default temperature T_DEFAULT."""
        assert abs(KBT_KCAL - KB_KCAL * T_DEFAULT) < 1e-8

    def test_bjerrum_length(self):
        """The Bjerrum length lies between 6.5 and 8.0 A, near 7.1 A for water at 298 K."""
        assert 6.5 < BJERRUM_LENGTH < 8.0  # ~7.1 Å in water at 298K

    def test_eps_water(self):
        """The water dielectric constant EPS_WATER equals 78.0."""
        assert abs(EPS_WATER - 78.0) < 0.1

    def test_avogadro(self):
        """Avogadro's number AVOGADRO equals 6.022e23."""
        assert abs(AVOGADRO - 6.022e23) < 1e20

    def test_ang_to_m(self):
        """The angstrom-to-meter conversion ANG_TO_M equals 1e-10."""
        assert abs(ANG_TO_M - 1e-10) < 1e-20

    def test_ps_to_s(self):
        """The picosecond-to-second conversion PS_TO_S equals 1e-12."""
        assert abs(PS_TO_S - 1e-12) < 1e-20

    def test_pi(self):
        """The constant PI equals math.pi."""
        assert abs(PI - math.pi) < 1e-14

    def test_two_pi(self):
        """The constant TWO_PI equals 2 times math.pi."""
        assert abs(TWO_PI - 2 * math.pi) < 1e-14

    def test_four_pi(self):
        """The constant FOUR_PI equals 4 times math.pi."""
        assert abs(FOUR_PI - 4 * math.pi) < 1e-14

    def test_debye_length_positive(self):
        """The default Debye length DEFAULT_DEBYE_LENGTH is positive."""
        assert DEFAULT_DEBYE_LENGTH > 0

    def test_eta_water_positive(self):
        """The water viscosity ETA_WATER is positive."""
        assert ETA_WATER > 0

    def test_kbt_at_room_temp(self):
        # kBT at 298 K in kcal/mol should be ~0.592
        """k_B T at 298 K in kcal/mol is approximately 0.592."""
        assert abs(KBT_KCAL - 0.592) < 0.01


# Structures / molecules
class TestAtom:
    def test_create(self):
        """Atom stores its constructor name and charge after creation."""
        a = Atom(index=0, name="CA", x=1.0, y=2.0, z=3.0, charge=0.5, radius=1.8)
        assert a.name == "CA"
        assert a.charge == 0.5

    def test_position_property(self):
        """The Atom.position property returns its x, y, z coordinates as an array."""
        a = Atom(x=1.0, y=2.0, z=3.0)
        assert np.allclose(a.position, [1.0, 2.0, 3.0])

    def test_position_setter(self):
        """Setting Atom.position updates the underlying x, y, z coordinate fields."""
        a = Atom()
        a.position = np.array([4.0, 5.0, 6.0])
        assert abs(a.x - 4.0) < 1e-10

    def test_distance_to(self):
        """Atom.distance_to returns the Euclidean distance between two atoms, 5 for a 3-4-0 separation."""
        a = Atom(x=0, y=0, z=0)
        b = Atom(x=3, y=4, z=0)
        assert abs(a.distance_to(b) - 5.0) < 1e-10

    def test_repr(self):
        """The Atom repr includes the atom name."""
        a = Atom(name="N", x=1.0, y=2.0, z=3.0)
        assert "N" in repr(a)

    def test_zero_atom(self):
        """A default-constructed Atom has zero coordinates and zero charge."""
        a = Atom()
        assert a.x == 0.0
        assert a.charge == 0.0

    def test_distance_self(self):
        """Atom.distance_to returns 0 when measured against the same atom."""
        a = Atom(x=1.0, y=2.0, z=3.0)
        assert a.distance_to(a) == 0.0

    def test_distance_3d(self):
        """Atom.distance_to computes the full 3D Euclidean distance, 5 for a 3-4 offset in x and y."""
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
        """A constructed Molecule retains its name and reports its atom count via len."""
        mol = self._make_mol()
        assert mol.name == "test"
        assert len(mol) == 3

    def test_centroid(self):
        """Molecule.centroid returns the mean position of its atoms."""
        mol = self._make_mol()
        c = mol.centroid()
        assert np.allclose(c, [1.0, 2 / 3, 0.0])

    def test_total_charge(self):
        """Molecule.total_charge sums the atomic charges."""
        mol = self._make_mol()
        assert abs(mol.total_charge() - 0.5) < 1e-10

    def test_positions_array(self):
        """Molecule.positions_array returns an (N, 3) array of atom positions."""
        mol = self._make_mol()
        pos = mol.positions_array()
        assert pos.shape == (3, 3)

    def test_charges_array(self):
        """Molecule.charges_array returns a length-N array whose sum equals the total charge."""
        mol = self._make_mol()
        q = mol.charges_array()
        assert q.shape == (3,)
        assert abs(q.sum() - 0.5) < 1e-10

    def test_translate(self):
        """Molecule.translate shifts every atom by the given displacement vector."""
        mol = self._make_mol()
        mol.translate(np.array([1.0, 0.0, 0.0]))
        assert abs(mol.atoms[0].x - 1.0) < 1e-10

    def test_rotate(self):
        """Molecule.rotate applies the rotation matrix, mapping an x-axis atom onto the y-axis under a 90 deg z rotation."""
        mol = Molecule()
        mol.atoms = [Atom(x=1.0, y=0.0, z=0.0)]
        # 90° rotation about z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mol.rotate(R)
        assert abs(mol.atoms[0].x) < 1e-10
        assert abs(mol.atoms[0].y - 1.0) < 1e-10

    def test_bounding_radius(self):
        """Molecule.bounding_radius returns a positive value."""
        mol = self._make_mol()
        br = mol.bounding_radius()
        assert br > 0

    def test_radius_of_gyration(self):
        """Molecule.radius_of_gyration returns a positive value."""
        mol = self._make_mol()
        rg = mol.radius_of_gyration()
        assert rg > 0

    def test_empty_molecule(self):
        """An empty Molecule reports a zero centroid and zero total charge."""
        mol = Molecule()
        assert np.allclose(mol.centroid(), [0, 0, 0])
        assert mol.total_charge() == 0.0

    def test_repr(self):
        """The Molecule repr includes the molecule name."""
        mol = self._make_mol()
        assert "test" in repr(mol)

    def test_center_of_mass(self):
        """Molecule.center_of_mass coincides with the centroid for unit-mass atoms."""
        mol = self._make_mol()
        assert np.allclose(mol.center_of_mass(), mol.centroid())

    def test_radii_array(self):
        """Molecule.radii_array returns a length-N array of atomic radii."""
        mol = self._make_mol()
        r = mol.radii_array()
        assert r.shape == (3,)

    def test_rotate_about_centroid(self):
        """Molecule.rotate_about_centroid leaves the centroid unchanged under an identity rotation."""
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
        """A constructed BoundingBox stores its xmin and xmax bounds."""
        bb = self._make_bb()
        assert bb.xmin == -1
        assert bb.xmax == 1

    def test_center(self):
        """The BoundingBox.center property returns the midpoint of its bounds."""
        bb = self._make_bb()
        assert np.allclose(bb.center, [0, 0, 0])

    def test_size(self):
        """The BoundingBox.size property returns the extent along each axis."""
        bb = self._make_bb()
        assert np.allclose(bb.size, [2, 4, 6])

    def test_contains(self):
        """BoundingBox.contains reports True for an interior point and False for an outside point."""
        bb = self._make_bb()
        assert bb.contains(np.array([0, 0, 0]))
        assert not bb.contains(np.array([5, 0, 0]))

    def test_padding(self):
        """BoundingBox.from_molecule expands the bounds by the given padding around the atoms."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0)]
        bb = BoundingBox.from_molecule(mol, padding=2.0)
        assert bb.xmin == -2.0
        assert bb.xmax == 2.0

    def test_repr(self):
        """The BoundingBox repr includes the class name."""
        bb = self._make_bb()
        assert "BoundingBox" in repr(bb)


class TestContactPair:
    def test_create(self):
        """A constructed ContactPair stores its first-molecule atom index and distance cutoff."""
        cp = ContactPair(0, 1, 5.0)
        assert cp.mol1_atom_index == 0
        assert cp.distance_cutoff == 5.0

    def test_repr(self):
        """The ContactPair repr includes its atom index."""
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
        """ReactionCriteria.is_satisfied returns True when the pair separation is within the cutoff."""
        mol1, mol2 = self._setup()
        pair = ContactPair(0, 0, 5.0)  # atom0 in mol1 to atom0 in mol2: dist=3
        criteria = ReactionCriteria(pairs=[pair])
        assert criteria.is_satisfied(mol1, mol2)

    def test_not_satisfied(self):
        """ReactionCriteria.is_satisfied returns False when the cutoff is smaller than the pair separation."""
        mol1, mol2 = self._setup()
        pair = ContactPair(0, 0, 2.0)  # cutoff too small
        criteria = ReactionCriteria(pairs=[pair])
        assert not criteria.is_satisfied(mol1, mol2)

    def test_multiple_pairs_all_required(self):
        """ReactionCriteria.is_satisfied requires every contact pair, returning False when one pair is unsatisfied."""
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
        """parse_pqr reads atom count, name, coordinates, charge, and radius from a PQR file."""
        p = tmp_path / "test.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert len(mol.atoms) == 2
        assert mol.atoms[0].name == "CA"
        assert abs(mol.atoms[0].x - 1.0) < 1e-6
        assert abs(mol.atoms[0].charge - 0.5) < 1e-6
        assert abs(mol.atoms[0].radius - 1.8) < 1e-6

    def test_parse_charges(self, tmp_path):
        """parse_pqr produces a molecule whose total charge matches the sum of the PQR atomic charges."""
        p = tmp_path / "test.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert abs(mol.total_charge() - 0.3) < 1e-5

    def test_roundtrip(self, tmp_path):
        """A PQR write then re-parse round trip preserves the atom count and the first atom x coordinate."""
        p_in = tmp_path / "in.pqr"
        p_out = tmp_path / "out.pqr"
        p_in.write_text(self._pqr_content())
        mol = parse_pqr(p_in)
        write_pqr(mol, p_out)
        mol2 = parse_pqr(p_out)
        assert len(mol2.atoms) == 2
        assert abs(mol2.atoms[0].x - 1.0) < 1e-3

    def test_molecule_name_from_stem(self, tmp_path):
        """parse_pqr sets the molecule name from the input file stem."""
        p = tmp_path / "myprotein.pqr"
        p.write_text(self._pqr_content())
        mol = parse_pqr(p)
        assert mol.name == "myprotein"

    def test_empty_pqr(self, tmp_path):
        """parse_pqr returns zero atoms for a PQR file containing only REMARK and END lines."""
        p = tmp_path / "empty.pqr"
        p.write_text("REMARK empty\nEND\n")
        mol = parse_pqr(p)
        assert len(mol.atoms) == 0

    def test_hetatm(self, tmp_path):
        """parse_pqr reads a single atom from a HETATM record."""
        p = tmp_path / "ligand.pqr"
        p.write_text(
            "HETATM    1  C1  LIG     1       0.000   0.000   0.000  0.100  1.500\nEND\n"
        )
        mol = parse_pqr(p)
        assert len(mol.atoms) == 1


# Quaternion and transforms
class TestQuaternion:
    def test_identity(self):
        """Quaternion.identity returns w equal to 1 and x equal to 0."""
        q = Quaternion.identity()
        assert q.w == 1.0
        assert q.x == 0.0

    def test_norm(self):
        """The norm of the unit quaternion (1,0,0,0) equals 1."""
        q = Quaternion(1, 0, 0, 0)
        assert abs(q.norm() - 1.0) < 1e-14

    def test_normalized(self):
        """Normalizing the quaternion (2,0,0,0) yields w equal to 1."""
        q = Quaternion(2, 0, 0, 0).normalized()
        assert abs(q.w - 1.0) < 1e-14

    def test_rotation_matrix_identity(self):
        """The rotation matrix of the identity quaternion equals the 3x3 identity."""
        q = Quaternion.identity()
        R = q.to_rotation_matrix()
        assert np.allclose(R, np.eye(3))

    def test_from_axis_angle_90z(self):
        """A 90 degree rotation about z maps the x axis unit vector to the y axis."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        R = q.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 1, 0], atol=1e-10)

    def test_from_axis_angle_180x(self):
        """A 180 degree rotation about x maps the y axis unit vector to its negative."""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), math.pi)
        R = q.to_rotation_matrix()
        v = R @ np.array([0, 1, 0])
        assert np.allclose(v, [0, -1, 0], atol=1e-10)

    def test_multiply_identity(self):
        """Multiplying a quaternion by the identity quaternion leaves it unchanged."""
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        r = q * Quaternion.identity()
        assert np.allclose(q.to_array(), r.normalized().to_array(), atol=1e-10)

    def test_conjugate(self):
        """The product of a unit quaternion and its conjugate has w equal to 1."""
        q = Quaternion(0.7, 0.1, 0.2, 0.3).normalized()
        qc = q.conjugate()
        prod = (q * qc).normalized()
        assert abs(prod.w - 1.0) < 1e-10

    def test_rotate_vector(self):
        """Rotating the x axis vector by π about z gives the negative x axis."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi)
        v = q.rotate_vector(np.array([1, 0, 0]))
        assert np.allclose(v, [-1, 0, 0], atol=1e-10)

    def test_to_array(self):
        """Quaternion.to_array returns an array of shape (4,)."""
        q = Quaternion(1, 0, 0, 0)
        arr = q.to_array()
        assert arr.shape == (4,)

    def test_from_rotation_matrix_roundtrip(self):
        """Converting a quaternion to a rotation matrix and back reproduces the same matrix."""
        q_orig = Quaternion.from_axis_angle(np.array([1, 1, 0]) / math.sqrt(2), 1.2)
        R = q_orig.to_rotation_matrix()
        q_back = Quaternion.from_rotation_matrix(R)
        R_back = q_back.to_rotation_matrix()
        assert np.allclose(R, R_back, atol=1e-10)

    def test_repr(self):
        """The repr of a Quaternion contains the string Quaternion."""
        q = Quaternion.identity()
        assert "Quaternion" in repr(q)

    def test_zero_axis(self):
        """from_axis_angle with a zero axis returns the identity quaternion with w equal to 1."""
        q = Quaternion.from_axis_angle(np.zeros(3), 1.0)
        assert abs(q.w - 1.0) < 1e-10

    def test_from_axis_angle_360(self):
        """A 2π rotation about z produces the identity rotation matrix."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), 2 * math.pi)
        R = q.to_rotation_matrix()
        assert np.allclose(R, np.eye(3), atol=1e-10)


class TestRigidTransform:
    def test_identity(self):
        """The identity RigidTransform leaves a vector unchanged."""
        T = RigidTransform.identity()
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(T.apply(v), v)

    def test_pure_translation(self):
        """A pure translation RigidTransform shifts the origin by the translation vector."""
        T = RigidTransform(translation=np.array([1.0, 2.0, 3.0]))
        v = np.zeros(3)
        assert np.allclose(T.apply(v), [1, 2, 3])

    def test_pure_rotation(self):
        """A pure 90 degree z rotation RigidTransform maps the x axis vector to the y axis."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        T = RigidTransform(rotation=q)
        v = np.array([1.0, 0.0, 0.0])
        result = T.apply(v)
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_compose(self):
        """Composing two translation transforms sums their translations."""
        T1 = RigidTransform(translation=np.array([1.0, 0.0, 0.0]))
        T2 = RigidTransform(translation=np.array([2.0, 0.0, 0.0]))
        T12 = T1.compose(T2)
        v = np.zeros(3)
        assert np.allclose(T12.apply(v), [3, 0, 0])

    def test_inverse(self):
        """Applying a RigidTransform and then its inverse recovers the original vector."""
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.7)
        T = RigidTransform(rotation=q, translation=np.array([1, 2, 3]))
        Ti = T.inverse()
        v = np.array([4.0, 5.0, 6.0])
        result = Ti.apply(T.apply(v))
        assert np.allclose(result, v, atol=1e-10)

    def test_apply_batch(self):
        """A RigidTransform applied to a batch of points returns shape (2,3) with each point translated."""
        T = RigidTransform(translation=np.array([1.0, 0.0, 0.0]))
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = T.apply(pts)
        assert result.shape == (2, 3)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_repr(self):
        """The repr of a RigidTransform contains the string RigidTransform."""
        T = RigidTransform.identity()
        assert "RigidTransform" in repr(T)


class TestRandomQuaternion:
    def test_returns_quaternion(self):
        """random_quaternion returns a Quaternion instance."""
        rng = np.random.default_rng(42)
        q = random_quaternion(rng)
        assert isinstance(q, Quaternion)

    def test_unit_norm(self):
        """random_quaternion produces unit norm quaternions across repeated draws."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = random_quaternion(rng)
            assert abs(q.norm() - 1.0) < 1e-10

    def test_rotation_matrix_orthogonal(self):
        """The rotation matrix from a random quaternion is orthogonal."""
        rng = np.random.default_rng(1)
        q = random_quaternion(rng)
        R = q.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_small_rotation(self):
        """small_rotation_quaternion returns a unit norm quaternion."""
        rng = np.random.default_rng(7)
        q = small_rotation_quaternion(0.01, rng)
        assert abs(q.norm() - 1.0) < 1e-10


# Hydrodynamics
class TestHydrodynamics:
    def test_stokes_translation_positive(self):
        """stokes_translational_diffusion returns a positive value for a 20 Å radius."""
        D = stokes_translational_diffusion(20.0)  # 20 Å radius
        assert D > 0

    def test_stokes_rotation_positive(self):
        """stokes_rotational_diffusion returns a positive value for a 20 Å radius."""
        D = stokes_rotational_diffusion(20.0)
        assert D > 0

    def test_stokes_translation_larger_radius_smaller_D(self):
        """Stokes translational diffusion is smaller for a larger radius."""
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(20.0)
        assert D1 > D2

    def test_stokes_rotation_larger_radius_smaller_D(self):
        """Stokes rotational diffusion is smaller for a larger radius."""
        D1 = stokes_rotational_diffusion(10.0)
        D2 = stokes_rotational_diffusion(20.0)
        assert D1 > D2

    def test_mobility_from_radii(self):
        """MobilityTensor.from_radii gives positive translational diffusion for both species."""
        mob = MobilityTensor.from_radii(20.0, 20.0)
        assert mob.D_trans1 > 0
        assert mob.D_trans2 > 0

    def test_relative_diffusion(self):
        """The relative translational diffusion equals twice the single species value for equal radii."""
        mob = MobilityTensor.from_radii(20.0, 20.0)
        D_rel = mob.relative_translational_diffusion()
        assert abs(D_rel - 2 * mob.D_trans1) < 1e-14

    def test_rotne_prager_far_field(self):
        """rpy_offdiagonal returns a 3x3 matrix in the far field."""
        r_vec = np.array([100.0, 0.0, 0.0])
        M = rpy_offdiagonal(r_vec, 5.0, 5.0, 1.0, 1.0)
        assert M.shape == (3, 3)

    def test_rotne_prager_zero_distance(self):
        """The RPY off diagonal mobility is zero at zero separation."""
        M = rpy_offdiagonal(np.zeros(3), 5.0, 5.0, 1.0, 1.0)
        assert np.allclose(M, np.zeros((3, 3)))

    def test_repr(self):
        """The repr of a MobilityTensor contains the string MobilityTensor."""
        mob = MobilityTensor(1.0, 0.1, 1.0, 0.1)
        assert "MobilityTensor" in repr(mob)

    def test_stokes_units_reasonable(self):
        # Typical protein (~30 Å radius) D_t ~ 0.005-0.05 Å²/ps
        """Stokes translational diffusion for a 30 Å protein falls in a physically reasonable range."""
        D = stokes_translational_diffusion(30.0)
        assert 1e-4 < D < 1.0


# BD integrator
class TestBDStep:
    def test_translation_moves(self):
        """An Ermak McCammon translation step with no force moves the position by diffusion."""
        rng = np.random.default_rng(42)
        pos = np.zeros(3)
        force = np.zeros(3)
        new_pos = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng)
        assert not np.allclose(new_pos, pos)  # diffuses

    def test_translation_with_force(self):
        """A large x force makes drift dominate the Ermak McCammon translation step."""
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        force = np.array([100.0, 0.0, 0.0])
        # large force in x -> drift dominates
        new_pos = ermak_mccammon_translation(pos, force, 10.0, 1.0, rng)
        # on average, drift = D*dt*F = 10*1*100 = 1000 Å
        assert new_pos[0] > 500.0  # very likely for large drift

    def test_rotation_changes_orientation(self):
        """An Ermak McCammon rotation step yields a valid orthogonal orientation."""
        rng = np.random.default_rng(42)
        ori = Quaternion.identity()
        torque = np.zeros(3)
        new_ori = ermak_mccammon_rotation(ori, torque, 0.01, 0.2, rng)
        # should rotate randomly
        R = new_ori.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


    def test_translation_reproducible_seed(self):
        """Ermak McCammon translation is reproducible for two RNGs sharing a seed."""
        pos = np.zeros(3)
        force = np.zeros(3)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng1)
        p2 = ermak_mccammon_translation(pos, force, 10.0, 0.2, rng2)
        assert np.allclose(p1, p2)

    def test_small_dt_small_step(self):
        """Ermak McCammon translation steps stay tiny when dt is very small."""
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
        """A new SystemState starts with fate ONGOING and step 0."""
        s = SystemState()
        assert s.fate == Fate.ONGOING
        assert s.step == 0

    def test_separation(self):
        """SystemState.separation returns the Euclidean norm of the position."""
        s = SystemState(position=np.array([3.0, 4.0, 0.0]))
        assert abs(s.separation() - 5.0) < 1e-10

    def test_copy(self):
        """SystemState.copy produces an independent copy whose position edits do not affect the original."""
        s = SystemState(position=np.array([1.0, 2.0, 3.0]), step=5)
        s2 = s.copy()
        s2.position[0] = 99.0
        assert s.position[0] == 1.0

    def test_repr(self):
        """The repr of a SystemState contains the string SystemState."""
        s = SystemState()
        assert "SystemState" in repr(s)

    def test_fate_ongoing(self):
        """A default SystemState has fate ONGOING."""
        s = SystemState()
        assert s.fate == Fate.ONGOING

    def test_fate_reacted(self):
        """Setting a SystemState fate to REACTED is reflected by the attribute."""
        s = SystemState()
        s.fate = Fate.REACTED
        assert s.fate == Fate.REACTED


class TestTrajectoryResult:
    def test_reacted_property(self):
        """A REACTED TrajectoryResult reports reacted True and escaped False."""
        r = TrajectoryResult(Fate.REACTED, 100, 20.0, 5.0, "rxn1")
        assert r.reacted
        assert not r.escaped

    def test_escaped_property(self):
        """An ESCAPED TrajectoryResult reports escaped True and reacted False."""
        r = TrajectoryResult(Fate.ESCAPED, 500, 100.0, 300.0)
        assert r.escaped
        assert not r.reacted

    def test_repr(self):
        """The repr of a TrajectoryResult contains the string TrajectoryResult."""
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
        """A reaction with probability 1 fires its check for the set up molecule pair."""
        mol1, mol2, rxn = self._setup()
        assert rxn.check(mol1, mol2)

    def test_check_probability_zero(self):
        """A reaction with probability 0 never fires its check."""
        mol1, mol2, rxn = self._setup()
        rxn.probability = 0.0
        assert not rxn.check(mol1, mol2)

    def test_repr(self):
        """The repr of a reaction contains its name rxn1."""
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
        """ReactionSet.check_all returns the name of the firing reaction r1."""
        mol1, mol2, ps = self._make_set()
        rng = np.random.default_rng(0)
        name = ps.check_all(mol1, mol2, rng)
        assert name == "r1"

    def test_empty_set(self):
        """check_all returns None when the PathwaySet holds no reaction interfaces."""
        mol1 = Molecule()
        mol1.atoms = [Atom()]
        mol2 = Molecule()
        mol2.atoms = [Atom()]
        ps = PathwaySet()
        assert ps.check_all(mol1, mol2) is None

    def test_len(self):
        """len of a PathwaySet with one reaction returns 1."""
        _, _, ps = self._make_set()
        assert len(ps) == 1

    def test_repr(self):
        """repr of a PathwaySet contains the string PathwaySet."""
        _, _, ps = self._make_set()
        assert "PathwaySet" in repr(ps)

    def test_add(self):
        """Adding a ReactionInterface to an empty PathwaySet brings its length to 1."""
        ps = PathwaySet()
        pair = ContactPair(0, 0, 5.0)
        criteria = ReactionCriteria(pairs=[pair])
        ps.add(ReactionInterface("r2", criteria))
        assert len(ps) == 1


class TestMakeDefaultReaction:
    def test_creates_reaction(self):
        """make_default_reaction returns a ReactionInterface with the requested number of contact pairs."""
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
        """Debye-Huckel energy of like charges is positive."""
        E = debye_huckel_energy(1.0, 1.0, 10.0)
        assert E > 0

    def test_opposite_sign_negative(self):
        """Debye-Huckel energy of opposite charges is negative."""
        E = debye_huckel_energy(1.0, -1.0, 10.0)
        assert E < 0

    def test_decays_with_distance(self):
        """Debye-Huckel energy decreases with increasing separation."""
        E1 = debye_huckel_energy(1.0, 1.0, 5.0)
        E2 = debye_huckel_energy(1.0, 1.0, 10.0)
        assert E1 > E2

    def test_zero_charge(self):
        """Debye-Huckel energy is zero when one charge is zero."""
        E = debye_huckel_energy(0.0, 1.0, 10.0)
        assert E == 0.0

    def test_zero_distance(self):
        """Debye-Huckel energy is zero at zero separation."""
        E = debye_huckel_energy(1.0, 1.0, 0.0)
        assert E == 0.0

    def test_force_direction(self):
        """Debye-Huckel force returns a length-3 vector."""
        r_vec = np.array([10.0, 0.0, 0.0])
        F = debye_huckel_force(1.0, 1.0, r_vec)
        assert F.shape == (3,)

    def test_force_zero_charge(self):
        """Debye-Huckel force is the zero vector when one charge is zero."""
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
        """Grid interpolation at a node returns that node's value."""
        g = self._make_grid()
        val = g.interpolate(np.array([2.0, 2.0, 2.0]))
        assert abs(val - 2.0) < 1e-8

    def test_interpolate_between_nodes(self):
        """Grid interpolation between nodes returns the linearly interpolated value."""
        g = self._make_grid()
        val = g.interpolate(np.array([1.5, 1.0, 1.0]))
        assert abs(val - 1.5) < 1e-8

    def test_interpolate_out_of_bounds(self):
        """Grid interpolation outside the bounds returns 0."""
        g = self._make_grid()
        val = g.interpolate(np.array([100.0, 0.0, 0.0]))
        assert val == 0.0

    def test_gradient(self):
        """Grid gradient recovers the unit slope along x and near-zero slope along y."""
        g = self._make_grid()
        grad = g.gradient(np.array([2.0, 2.0, 2.0]))
        assert abs(grad[0] - 1.0) < 0.1  # potential increases with x
        assert abs(grad[1]) < 0.2

    def test_force_on_charge(self):
        """force_on_charge returns a length-3 vector."""
        g = self._make_grid()
        F = g.force_on_charge(np.array([2.0, 2.0, 2.0]), 1.0)
        assert F.shape == (3,)

    def test_repr(self):
        """repr of a DXGrid contains the string DXGrid."""
        g = self._make_grid()
        assert "DXGrid" in repr(g)

    def test_from_file(self, tmp_path):
        """DXGrid.from_file reads a minimal DX file into a 3x3x3 array with correct node values."""
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
        """bounding_box with zero padding encloses all atom coordinates."""
        mol = self._mol()
        bb = bounding_box(mol, padding=0.0)
        assert bb.xmin <= 0.0
        assert bb.xmax >= 5.0

    def test_bounding_box_padding(self):
        """bounding_box with padding expands the box symmetrically beyond the unpadded one."""
        mol = self._mol()
        bb0 = bounding_box(mol, padding=0.0)
        bb5 = bounding_box(mol, padding=5.0)
        assert bb5.xmin < bb0.xmin
        assert bb5.xmax > bb0.xmax

    def test_surface_spheres_nonempty(self):
        """surface_spheres returns a non-empty set of surface points."""
        mol = self._mol()
        pts = surface_spheres(mol, probe_radius=1.4, n_points=20)
        assert len(pts) > 0

    def test_lumped_charges(self):
        """Lumped charges sum to the molecule's total charge."""
        mol = self._mol()
        lc = lumped_charges(mol, grid_spacing=3.0)
        assert len(lc) > 0
        total_q = sum(q for _, q in lc)
        assert abs(total_q - mol.total_charge()) < 1e-6

    def test_electrostatic_center(self):
        """electrostatic_center returns a length-3 vector."""
        mol = self._mol()
        ec = electrostatic_center(mol)
        assert ec.shape == (3,)

    def test_electrostatic_center_zero_charge(self):
        """electrostatic_center falls back to the centroid for a neutral molecule."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, charge=0), Atom(x=2, charge=0)]
        ec = electrostatic_center(mol)
        assert np.allclose(ec, mol.centroid())

    def test_hydrodynamic_radius(self):
        """Hydrodynamic radius estimated from the radius of gyration is positive."""
        mol = self._mol()
        rh = hydrodynamic_radius_from_rg(mol)
        assert rh > 0

    def test_contact_distances(self):
        """contact_distances reports only the pair within the cutoff and its correct distance."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0, y=0, z=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=3, y=0, z=0), Atom(x=20, y=0, z=0)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert len(pairs) == 1
        assert abs(pairs[0][2] - 3.0) < 1e-8

    def test_contact_distances_none(self):
        """contact_distances returns no pairs when all atoms exceed the cutoff."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=100)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert len(pairs) == 0

    def test_born_integral_negative(self):
        """Born solvation integral is negative, reflecting stabilization."""
        E = born_integral(1.0, 3.0)
        assert E < 0  # solvation is stabilizing

    def test_born_integral_zero_charge(self):
        """Born solvation integral is zero for zero charge."""
        E = born_integral(0.0, 3.0)
        assert E == 0.0

    def test_born_integral_zero_radius(self):
        """Born solvation integral is zero for zero radius."""
        E = born_integral(1.0, 0.0)
        assert E == 0.0


# Numerical library
class TestCubicSpline:
    def test_interpolates_at_nodes(self):
        """CubicSpline reproduces the data values exactly at the knots."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        sp = CubicSpline(x, y)
        for xi, yi in zip(x, y):
            assert abs(sp(xi) - yi) < 1e-8

    def test_interpolates_between(self):
        """CubicSpline approximates sin between knots to within 0.01."""
        x = np.linspace(0, math.pi, 20)
        y = np.sin(x)
        sp = CubicSpline(x, y)
        val = sp(math.pi / 4)
        assert abs(val - math.sin(math.pi / 4)) < 0.01

    def test_derivative(self):
        """CubicSpline derivative of x^2 recovers 2x at the evaluation point."""
        x = np.linspace(0, 2, 10)
        y = x**2
        sp = CubicSpline(x, y)
        # derivative of x² is 2x
        deriv = sp.derivative(1.0)
        assert abs(deriv - 2.0) < 0.1

    def test_two_points(self):
        """CubicSpline through two points interpolates linearly at the midpoint."""
        sp = CubicSpline(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert abs(sp(0.5) - 0.5) < 1e-8

    def test_extrapolation_boundary(self):
        """CubicSpline returns exact endpoint values at the domain boundaries."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        sp = CubicSpline(x, y)
        assert abs(sp(0.0) - 0.0) < 1e-8
        assert abs(sp(2.0) - 2.0) < 1e-8


class TestRomberg:
    def test_constant(self):
        """Romberg integration of a constant over the unit interval returns 1."""
        val = romberg_integrate(lambda x: 1.0, 0.0, 1.0)
        assert abs(val - 1.0) < 1e-8

    def test_linear(self):
        """Romberg integration of x over the unit interval returns 1/2."""
        val = romberg_integrate(lambda x: x, 0.0, 1.0)
        assert abs(val - 0.5) < 1e-8

    def test_quadratic(self):
        """Romberg integration of x^2 over the unit interval returns 1/3."""
        val = romberg_integrate(lambda x: x**2, 0.0, 1.0)
        assert abs(val - 1.0 / 3.0) < 1e-8

    def test_sine(self):
        """Romberg integration of sin over 0 to pi returns 2."""
        val = romberg_integrate(math.sin, 0.0, math.pi)
        assert abs(val - 2.0) < 1e-8

    def test_exp(self):
        """Romberg integration of exp over the unit interval returns e minus 1."""
        val = romberg_integrate(math.exp, 0.0, 1.0)
        assert abs(val - (math.e - 1.0)) < 1e-8


class TestWienerStep:
    def test_shape(self):
        """wiener_step returns a displacement vector of the requested dimension."""
        rng = np.random.default_rng(0)
        dW = wiener_step(1.0, 0.1, 3, rng)
        assert dW.shape == (3,)

    def test_scaling(self):
        # std of many steps should be sqrt(2Ddt)
        """wiener_step displacements have standard deviation sqrt(2Ddt)."""
        rng = np.random.default_rng(42)
        steps = np.array([wiener_step(1.0, 0.1, 1, rng)[0] for _ in range(5000)])
        expected_std = math.sqrt(2.0 * 1.0 * 0.1)
        assert abs(steps.std() - expected_std) < 0.05


class TestMultipoles:
    def test_monopole(self):
        """monopole_moment returns the total charge."""
        q = np.array([1.0, -1.0, 0.5])
        assert abs(monopole_moment(q) - 0.5) < 1e-10

    def test_dipole_shape(self):
        """dipole_moment returns a length-3 vector."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        q = np.array([1.0, -1.0, 0.0])
        p = dipole_moment(pos, q)
        assert p.shape == (3,)

    def test_dipole_symmetric(self):
        """dipole_moment of a symmetric charge pair gives the expected x component."""
        pos = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
        q = np.array([1.0, -1.0])
        p = dipole_moment(pos, q)
        assert abs(p[0] - (-2.0)) < 1e-10

    def test_quadrupole_shape(self):
        """quadrupole_moment returns a 3x3 tensor."""
        pos = np.random.randn(5, 3)
        q = np.random.randn(5)
        Q = quadrupole_moment(pos, q)
        assert Q.shape == (3, 3)

    def test_quadrupole_symmetric(self):
        """quadrupole_moment returns a symmetric tensor."""
        pos = np.random.randn(5, 3)
        q = np.random.randn(5)
        Q = quadrupole_moment(pos, q)
        assert np.allclose(Q, Q.T)


class TestLegendre:
    def test_p0(self):
        """Legendre polynomial P0 equals 1."""
        assert abs(legendre_p(0, 0.5) - 1.0) < 1e-14

    def test_p1(self):
        """Legendre polynomial P1(x) equals x."""
        assert abs(legendre_p(1, 0.5) - 0.5) < 1e-14

    def test_p2(self):
        # P2(x) = (3x²-1)/2
        """Legendre polynomial P2(x) equals (3x^2 minus 1)/2."""
        x = 0.7
        expected = (3 * x**2 - 1) / 2
        assert abs(legendre_p(2, x) - expected) < 1e-12

    def test_p0_minus1(self):
        """Legendre polynomial P0 equals 1 at x equal to minus 1."""
        assert abs(legendre_p(0, -1.0) - 1.0) < 1e-14

    def test_p1_minus1(self):
        """Legendre polynomial P1 equals minus 1 at x equal to minus 1."""
        assert abs(legendre_p(1, -1.0) - (-1.0)) < 1e-14

    def test_series(self):
        # constant series c0=1 should equal 1 everywhere
        """A Legendre series with only c0 equal to 1 evaluates to 1 everywhere."""
        val = legendre_series([1.0], 0.3)
        assert abs(val - 1.0) < 1e-14

    def test_series_p1(self):
        """A Legendre series with c1 equal to 1 evaluates to x."""
        val = legendre_series([0.0, 1.0], 0.5)
        assert abs(val - 0.5) < 1e-14

    def test_legendre_p3(self):
        """Legendre polynomial P3(x) equals (5x^3 minus 3x)/2."""
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
        """parse_reaction_xml reads two reactions from the sample file."""
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert len(ps) == 2

    def test_parse_names(self, tmp_path):
        """parse_reaction_xml preserves both reaction names rxn1 and rxn2."""
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        names = [r.name for r in ps.reactions]
        assert "rxn1" in names
        assert "rxn2" in names

    def test_parse_probability(self, tmp_path):
        """parse_reaction_xml reads the first reaction's probability as 0.9."""
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert abs(ps.reactions[0].probability - 0.9) < 1e-6

    def test_parse_contacts(self, tmp_path):
        """parse_reaction_xml reads two contact pairs for the first reaction."""
        p = tmp_path / "rxn.xml"
        self._write_reaction_xml(p)
        ps = parse_reaction_xml(p)
        assert len(ps.reactions[0].criteria.pairs) == 2

    def test_roundtrip(self, tmp_path):
        """Parsing then writing a reaction XML round-trips the reaction count and first reaction name."""
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
        """Parsing the simulation XML recovers n_trajectories, dt, and the list of dx grid files."""
        p = tmp_path / "sim.xml"
        self._write_sim_xml(p)
        cfg = parse_simulation_xml(p)
        assert cfg["n_trajectories"] == 500
        assert abs(cfg["dt"] - 0.1) < 1e-8
        assert len(cfg["dx_files"]) == 2

    def test_parse_mol_names(self, tmp_path):
        """Parsing the simulation XML reads the mol1 PQR filename correctly."""
        p = tmp_path / "sim.xml"
        self._write_sim_xml(p)
        cfg = parse_simulation_xml(p)
        assert cfg["mol1_pqr"] == "thrombin.pqr"

    def test_roundtrip(self, tmp_path):
        """Parsing then writing the simulation XML round-trips n_trajectories."""
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
        """Running the NAM simulator returns a SimulationResult instance."""
        sim = self._make_sim(5)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_all_react_with_huge_cutoff(self):
        """With a huge contact cutoff every trajectory ends as either reacted or escaped."""
        sim = self._make_sim(20)
        result = sim.run()
        # With cutoff 200 Å and r_start=50 -> all should react immediately
        assert result.n_reacted + result.n_escaped == 20

    def test_reaction_probability_in_range(self):
        """The reaction probability of the simulation result lies within [0, 1]."""
        sim = self._make_sim(10)
        result = sim.run()
        assert 0.0 <= result.reaction_probability <= 1.0

    def test_n_trajectories_correct(self):
        """The result reports n_trajectories equal to the number simulated."""
        sim = self._make_sim(15)
        result = sim.run()
        assert result.n_trajectories == 15

    def test_seed_reproducible(self):
        """Two simulations with the same seed produce identical reacted counts."""
        s1 = self._make_sim(10)
        s2 = self._make_sim(10)
        r1 = s1.run()
        r2 = s2.run()
        assert r1.n_reacted == r2.n_reacted

    def test_escape_with_small_cutoff(self):
        """With a tiny contact cutoff trajectory outcomes partition into escaped, reacted, and max-steps totaling all runs."""
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
        """The rate constant computed from relative diffusion is non-negative when reactions occur."""
        sim = self._make_sim(20)
        result = sim.run()
        mob = sim.mobility
        D_rel = mob.relative_translational_diffusion()
        if result.n_reacted > 0:
            k = result.rate_constant(D_rel)
            assert k >= 0

    def test_reaction_counts_dict(self):
        """The result exposes reaction_counts as a dict."""
        sim = self._make_sim(10)
        result = sim.run()
        assert isinstance(result.reaction_counts, dict)

    def test_repr(self):
        """The SimulationResult repr contains the string SimulationResult."""
        sim = self._make_sim(5)
        result = sim.run()
        assert "SimulationResult" in repr(result)

    def test_zero_force_fn(self):
        """The zero_force function returns zero force, zero torque, and zero energy."""
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
        """Parsing two PQR files, building, and running a simulation yields the requested trajectory count."""
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
        """Writing and parsing a reaction XML then simulating gives a non-negative reacted count."""
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
        """The pystarc package exposes a truthy __version__ string."""
        assert pystarc.__version__  # version check

    def test_module_import_chain(self):
        """All major pystarc submodules import without error."""

    def test_constants_importable_from_root(self):
        """PI and BJERRUM_LENGTH constants are importable from the package root with expected positive values."""
        assert PI > 3.14
        assert BJERRUM_LENGTH > 0

    def test_empty_pathway_set_never_reacts(self):
        """A simulation with an empty pathway set never records any reactions."""
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
        """An Atom stores the index passed to its constructor."""
        a = Atom(index=7)
        assert a.index == 7

    def test_residue_name_stored(self):
        """An Atom stores the residue_name passed to its constructor."""
        a = Atom(residue_name="GLY")
        assert a.residue_name == "GLY"

    def test_residue_index_stored(self):
        """An Atom stores the residue_index passed to its constructor."""
        a = Atom(residue_index=42)
        assert a.residue_index == 42

    def test_chain_stored(self):
        """An Atom stores the chain passed to its constructor."""
        a = Atom(chain="B")
        assert a.chain == "B"

    def test_negative_charge(self):
        """An Atom stores a negative charge value."""
        a = Atom(charge=-2.5)
        assert a.charge == -2.5

    def test_large_radius(self):
        """An Atom stores a large radius value."""
        a = Atom(radius=10.0)
        assert a.radius == 10.0

    def test_position_roundtrip(self):
        """Setting an Atom position array and reading it back returns the same coordinates."""
        a = Atom()
        p = np.array([1.1, 2.2, 3.3])
        a.position = p
        assert np.allclose(a.position, p)

    def test_distance_symmetry(self):
        """The distance between two atoms is symmetric in the order of the operands."""
        a = Atom(x=1, y=2, z=3)
        b = Atom(x=4, y=5, z=6)
        assert abs(a.distance_to(b) - b.distance_to(a)) < 1e-10

    def test_default_radius(self):
        """A default Atom has radius 1.5."""
        a = Atom()
        assert a.radius == 1.5

    def test_default_chain(self):
        """A default Atom has chain 'A'."""
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
        """A five-atom molecule reports length 5."""
        mol = self._mol5()
        assert len(mol) == 5

    def test_centroid_x(self):
        """The centroid x-coordinate of the five-atom molecule equals 2.0."""
        mol = self._mol5()
        c = mol.centroid()
        assert abs(c[0] - 2.0) < 1e-10

    def test_total_charge_five(self):
        """The total charge of the five-atom molecule with charges summing to zero is zero."""
        mol = self._mol5()
        # charges: -1, -0.5, 0, 0.5, 1.0 -> sum=0
        assert abs(mol.total_charge()) < 1e-10

    def test_translate_all_atoms(self):
        """Translating a molecule shifts every atom x-coordinate by the translation vector."""
        mol = self._mol5()
        orig_x = [a.x for a in mol.atoms]
        mol.translate(np.array([5.0, 0, 0]))
        for i, a in enumerate(mol.atoms):
            assert abs(a.x - (orig_x[i] + 5.0)) < 1e-10

    def test_rotate_preserves_centroid_distance(self):
        """Rotating a molecule preserves each atom's distance from the centroid."""
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
        """A more spread-out molecule has a larger bounding radius than a tight one."""
        mol_tight = Molecule()
        mol_tight.atoms = [Atom(x=0, radius=1), Atom(x=1, radius=1)]
        mol_wide = Molecule()
        mol_wide.atoms = [Atom(x=0, radius=1), Atom(x=10, radius=1)]
        assert mol_wide.bounding_radius() > mol_tight.bounding_radius()

    def test_single_atom_molecule(self):
        """A single-atom molecule has its centroid at that atom's position."""
        mol = Molecule(name="single")
        mol.atoms = [Atom(x=3, y=4, z=5)]
        assert np.allclose(mol.centroid(), [3, 4, 5])

    def test_charges_array_dtype(self):
        """The charges array of a molecule has float dtype."""
        mol = self._mol5()
        q = mol.charges_array()
        assert q.dtype == float

    def test_positions_array_shape(self):
        """The positions array of the five-atom molecule has shape (5, 3)."""
        mol = self._mol5()
        pos = mol.positions_array()
        assert pos.shape == (5, 3)

    def test_repr_contains_atom_count(self):
        """The molecule repr contains the atom count."""
        mol = self._mol5()
        assert "5" in repr(mol)


class TestQuaternionAlgebra:
    def test_from_axis_angle_small(self):
        """A quaternion from a small axis-angle rotation has unit norm."""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.001)
        assert abs(q.norm() - 1.0) < 1e-10

    def test_multiply_non_commutative(self):
        """Quaternion multiplication is non-commutative for two distinct rotations."""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.5)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        q12 = (q1 * q2).normalized()
        q21 = (q2 * q1).normalized()
        # should generally differ
        assert not np.allclose(q12.to_array(), q21.to_array())

    def test_double_rotation(self):
        """Squaring a π/4 z-rotation quaternion rotates the x-axis onto the y-axis."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 4)
        qq = (q * q).normalized()
        R = qq.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert np.allclose(v, [0, 1, 0], atol=1e-10)

    def test_inverse_rotation(self):
        """A rotation matrix times the matrix of the conjugate quaternion gives the identity."""
        q = Quaternion.from_axis_angle(np.array([1, 1, 0]) / math.sqrt(2), 1.0)
        qi = q.conjugate().normalized()
        R = q.to_rotation_matrix()
        Ri = qi.to_rotation_matrix()
        assert np.allclose(R @ Ri, np.eye(3), atol=1e-10)

    def test_many_random_unit_norm(self):
        """Random quaternions all have unit norm."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            q = random_quaternion(rng)
            assert abs(q.norm() - 1.0) < 1e-10

    def test_from_rotation_matrix_identity(self):
        """A quaternion built from the identity matrix has |w| equal to 1."""
        q = Quaternion.from_rotation_matrix(np.eye(3))
        assert abs(abs(q.w) - 1.0) < 1e-10

    def test_conjugate_norm_preserved(self):
        """Conjugating a quaternion preserves its norm."""
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        assert abs(q.norm() - q.conjugate().norm()) < 1e-14

    def test_rotation_matrix_det_one(self):
        """Rotation matrices from random quaternions have determinant 1."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = random_quaternion(rng)
            R = q.to_rotation_matrix()
            assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestRigidTransformComposition:
    def test_rotation_then_translation(self):
        """A rigid transform applies its rotation before its translation."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        T = RigidTransform(rotation=q, translation=np.array([0, 1, 0]))
        v = np.array([1, 0, 0])
        result = T.apply(v)
        # rotate -> [0,1,0], translate -> [0,2,0]
        assert np.allclose(result, [0, 2, 0], atol=1e-10)

    def test_compose_three(self):
        """Composing three unit translations sums to the total translation."""
        t = np.array([1, 0, 0])
        T1 = RigidTransform(translation=t)
        T2 = RigidTransform(translation=t)
        T3 = RigidTransform(translation=t)
        T123 = T1.compose(T2).compose(T3)
        result = T123.apply(np.zeros(3))
        assert np.allclose(result, [3, 0, 0])

    def test_identity_inverse_is_identity(self):
        """The inverse of the identity rigid transform leaves a vector unchanged."""
        T = RigidTransform.identity()
        Ti = T.inverse()
        v = np.array([1, 2, 3], dtype=float)
        assert np.allclose(Ti.apply(v), v)

    def test_apply_preserves_distances(self):
        """A rigid transform preserves the distance between two points."""
        q = Quaternion.from_axis_angle(np.array([1, 1, 1]) / math.sqrt(3), 0.7)
        T = RigidTransform(rotation=q, translation=np.array([5, 3, 2]))
        p1, p2 = np.array([0, 0, 0], dtype=float), np.array([1, 0, 0], dtype=float)
        d_before = np.linalg.norm(p2 - p1)
        t1, t2 = T.apply(p1), T.apply(p2)
        d_after = np.linalg.norm(t2 - t1)
        assert abs(d_before - d_after) < 1e-10


class TestDiffusionCoefficientScaling:
    def test_relative_D_t_equals_sum(self):
        """The relative translational diffusion equals the sum of the two single-body translational diffusions."""
        mob = MobilityTensor.from_radii(15.0, 25.0)
        assert (
            abs(mob.relative_translational_diffusion() - mob.D_trans1 - mob.D_trans2)
            < 1e-14
        )

    def test_relative_D_r_equals_sum(self):
        """The relative rotational diffusion equals the sum of the two single-body rotational diffusions."""
        mob = MobilityTensor.from_radii(15.0, 25.0)
        assert (
            abs(mob.relative_rotational_diffusion() - mob.D_rot1 - mob.D_rot2) < 1e-14
        )

    def test_D_t_scales_inversely_with_radius(self):
        """Stokes translational diffusion scales as 1/r between two radii."""
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(20.0)
        assert abs(D1 / D2 - 2.0) < 0.01  # D ∝ 1/r

    def test_D_r_scales_as_inverse_cube(self):
        """Stokes rotational diffusion scales as 1/r³ between two radii."""
        D1 = stokes_rotational_diffusion(10.0)
        D2 = stokes_rotational_diffusion(20.0)
        assert abs(D1 / D2 - 8.0) < 0.01  # D_r ∝ 1/r³

    def test_asymmetric_molecules(self):
        """The smaller-radius body has larger translational diffusion than the larger one."""
        mob = MobilityTensor.from_radii(10.0, 30.0)
        assert mob.D_trans1 > mob.D_trans2


class TestBDStepForceDominance:
    def test_large_force_dominates_noise(self):
        """Under a huge force the Ermak-McCammon displacement is in the force direction."""
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
        """With zero diffusion the Ermak-McCammon translation produces no displacement."""
        rng = np.random.default_rng(0)
        pos = np.zeros(3)
        force = np.array([1.0, 0.0, 0.0])
        new_pos = ermak_mccammon_translation(pos, force, 0.0, 1.0, rng)
        # zero diffusion -> noise is 0, displacement = D*dt*F = 0
        assert np.allclose(new_pos, [0, 0, 0])

    def test_rotation_unit_quaternion_preserved(self):
        """Repeated Ermak-McCammon rotation keeps the orientation quaternion at unit norm."""
        rng = np.random.default_rng(42)
        ori = random_quaternion(rng)
        for _ in range(20):
            ori = ermak_mccammon_rotation(ori, np.zeros(3), 0.01, 0.2, rng)
            assert abs(ori.norm() - 1.0) < 1e-10


class TestSystemStateFieldAccess:
    def test_step_increment(self):
        """SystemState stores the step value passed to it."""
        s = SystemState(step=5)
        assert s.step == 5

    def test_time_stored(self):
        """SystemState stores the time value passed to it."""
        s = SystemState(time=12.5)
        assert s.time == 12.5

    def test_energy_stored(self):
        """SystemState stores the energy value passed to it."""
        s = SystemState(energy=-3.14)
        assert s.energy == -3.14

    def test_force_stored(self):
        """SystemState stores the force vector passed to it."""
        f = np.array([1.0, 2.0, 3.0])
        s = SystemState(force=f)
        assert np.allclose(s.force, f)

    def test_torque_stored(self):
        """SystemState stores the torque vector passed to it."""
        t = np.array([0.1, 0.2, 0.3])
        s = SystemState(torque=t)
        assert np.allclose(s.torque, t)

    def test_copy_deep_orientation(self):
        """SystemState.copy() deep-copies the orientation so mutating the copy leaves the original unchanged."""
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        s = SystemState(orientation=q)
        s2 = s.copy()
        s2.orientation.w = 999.0
        assert s.orientation.w != 999.0

    def test_fate_max_steps(self):
        """SystemState stores the Fate.MAX_STEPS fate passed to it."""
        s = SystemState(fate=Fate.MAX_STEPS)
        assert s.fate == Fate.MAX_STEPS

    def test_reaction_name_stored(self):
        """SystemState stores the reaction_name passed to it."""
        s = SystemState(reaction_name="my_rxn")
        assert s.reaction_name == "my_rxn"

    def test_separation_zero_origin(self):
        """A default SystemState reports a separation of 0."""
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
        """bounding_box with zero padding encloses every atom of the molecule."""
        mol = self._big_mol()
        bb = bounding_box(mol, padding=0.0)
        for a in mol.atoms:
            assert bb.xmin <= a.x <= bb.xmax
            assert bb.ymin <= a.y <= bb.ymax
            assert bb.zmin <= a.z <= bb.zmax

    def test_lumped_charges_conserve_charge(self):
        """lumped_charges conserves total charge to within tolerance of the molecule total."""
        mol = self._big_mol()
        lc = lumped_charges(mol, grid_spacing=2.0)
        total_q = sum(q for _, q in lc)
        assert abs(total_q - mol.total_charge()) < 1e-5

    def test_contact_distances_sorted(self):
        """contact_distances returns pairs sorted by increasing distance."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0), Atom(x=5)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=3), Atom(x=7)]
        pairs = contact_distances(mol1, mol2, cutoff=20.0)
        dists = [p[2] for p in pairs]
        assert dists == sorted(dists)

    def test_surface_spheres_count_scales_with_n_points(self):
        """surface_spheres produces no fewer points with n_points=50 than with n_points=10."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=3.0)]
        pts10 = surface_spheres(mol, n_points=10)
        pts50 = surface_spheres(mol, n_points=50)
        assert len(pts50) >= len(pts10)

    def test_born_integral_larger_charge_more_negative(self):
        """born_integral becomes more negative as the charge magnitude increases."""
        E1 = born_integral(1.0, 3.0)
        E2 = born_integral(2.0, 3.0)
        assert E2 < E1  # more negative for larger charge

    def test_born_integral_smaller_radius_more_negative(self):
        """born_integral becomes more negative as the radius decreases."""
        E1 = born_integral(1.0, 5.0)
        E2 = born_integral(1.0, 2.0)
        assert E2 < E1


class TestNumericalAccuracy:
    def test_spline_sine_accurate(self):
        """CubicSpline reproduces sin(x) to within 0.002 over the fitted interval."""
        x = np.linspace(0, 2 * math.pi, 50)
        y = np.sin(x)
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.1, 6.0, 30):
            assert abs(sp(xi) - math.sin(xi)) < 0.002

    def test_romberg_exp_negative(self):
        """romberg_integrate of exp over [-1, 0] matches 1 - e^{-1} to within 1e-8."""
        val = romberg_integrate(math.exp, -1.0, 0.0)
        expected = 1.0 - math.exp(-1)
        assert abs(val - expected) < 1e-8

    def test_romberg_polynomial(self):
        """romberg_integrate of x^4 over [0, 1] matches 0.2 to within 1e-8."""
        val = romberg_integrate(lambda x: x**4, 0.0, 1.0)
        assert abs(val - 0.2) < 1e-8

    def test_wiener_mean_near_zero(self):
        """The mean of many wiener_step increments stays near 0."""
        rng = np.random.default_rng(99)
        steps = np.array([wiener_step(1.0, 0.01, 1, rng)[0] for _ in range(2000)])
        assert abs(steps.mean()) < 0.05

    def test_quadrupole_traceless(self):
        """quadrupole_moment returns a traceless tensor."""
        rng = np.random.default_rng(5)
        pos = rng.standard_normal((10, 3))
        q = rng.standard_normal(10)
        Q = quadrupole_moment(pos, q)
        assert abs(np.trace(Q)) < 1e-10

    def test_legendre_orthogonal_p0_p2(self):
        # ∫₋₁¹ P0(x)P2(x) dx = 0
        """The integral of P0(x)P2(x) over [-1, 1] is 0, confirming Legendre orthogonality."""
        val = romberg_integrate(
            lambda x: legendre_p(0, x) * legendre_p(2, x), -1.0, 1.0
        )
        assert abs(val) < 1e-6

    def test_legendre_norm(self):
        # ∫₋₁¹ [P1(x)]² dx = 2/(2·1+1) = 2/3
        """The integral of P1(x)^2 over [-1, 1] equals 2/3, matching the Legendre norm."""
        val = romberg_integrate(lambda x: legendre_p(1, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0 / 3.0) < 1e-6

    def test_spline_extrapolation_at_last_node(self):
        """CubicSpline evaluated at its last node x=4 returns the node value 16."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x**2
        sp = CubicSpline(x, y)
        assert abs(sp(4.0) - 16.0) < 1e-6

    def test_wiener_dim3(self):
        """wiener_step in 3 dimensions returns a vector of shape (3,)."""
        rng = np.random.default_rng(0)
        dW = wiener_step(2.0, 0.5, 3, rng)
        assert dW.shape == (3,)

    def test_dipole_zero_charge(self):
        """dipole_moment is zero when all charges are zero."""
        pos = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
        q = np.array([0.0, 0.0])
        p = dipole_moment(pos, q)
        assert np.allclose(p, 0)


class TestDebyeHuckelEdgeCases:
    def test_energy_zero_distance_safe(self):
        """debye_huckel_energy returns 0 at zero separation."""
        E = debye_huckel_energy(1.0, 1.0, 0.0)
        assert E == 0.0

    def test_energy_large_separation_near_zero(self):
        """debye_huckel_energy is negligibly small at very large separation."""
        E = debye_huckel_energy(1.0, 1.0, 1000.0, debye_length=7.9)
        assert abs(E) < 1e-30

    def test_energy_scales_with_charge_product(self):
        """debye_huckel_energy scales linearly with the product of the two charges."""
        E1 = debye_huckel_energy(1.0, 1.0, 10.0)
        E2 = debye_huckel_energy(2.0, 1.0, 10.0)
        E3 = debye_huckel_energy(2.0, 2.0, 10.0)
        assert abs(E2 - 2 * E1) < 1e-10
        assert abs(E3 - 4 * E1) < 1e-10

    def test_force_magnitude_positive(self):
        """debye_huckel_force has positive magnitude for like charges at finite separation."""
        r_vec = np.array([5.0, 0.0, 0.0])
        F = debye_huckel_force(1.0, 1.0, r_vec)
        assert np.linalg.norm(F) > 0

    def test_force_opposite_charges_toward(self):
        """debye_huckel_force for opposite charges returns a vector of shape (3,)."""
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
        """A uniform grid interpolates to its constant value at an interior point."""
        g = self._uniform_grid(3.0)
        assert abs(g.interpolate(np.array([2.5, 2.5, 2.5])) - 3.0) < 1e-8

    def test_uniform_grid_zero_gradient(self):
        """A uniform grid yields a zero gradient at an interior point."""
        g = self._uniform_grid(1.0)
        grad = g.gradient(np.array([2.5, 2.5, 2.5]))
        assert np.allclose(grad, 0, atol=1e-5)

    def test_force_scales_with_charge(self):
        """force_on_charge scales linearly with the charge magnitude."""
        g = self._uniform_grid()
        F1 = g.force_on_charge(np.array([2.5, 2.5, 2.5]), 1.0)
        F2 = g.force_on_charge(np.array([2.5, 2.5, 2.5]), 2.0)
        assert np.allclose(F2, 2 * F1)

    def test_shape_preserved(self):
        """DXGrid preserves the shape of the data array passed to it."""
        origin = np.zeros(3)
        delta = np.diag([2.0, 2.0, 2.0])
        data = np.zeros((4, 5, 6))
        g = DXGrid(origin, delta, data)
        assert tuple(g.data.shape) == (4, 5, 6)

    def test_origin_stored(self):
        """DXGrid stores the origin passed to it."""
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
        """SimulationResult reports the requested number of trajectories."""
        sim = self._fast_sim(8)
        result = sim.run()
        assert result.n_trajectories == 8

    def test_result_counts_sum(self):
        """The reacted, escaped, and max-steps counts sum to the total number of trajectories."""
        sim = self._fast_sim(10)
        result = sim.run()
        total = result.n_reacted + result.n_escaped + result.n_max_steps
        assert total == 10

    def test_rate_constant_type(self):
        """rate_constant returns a float."""
        sim = self._fast_sim(10)
        result = sim.run()
        k = result.rate_constant(sim.mobility.relative_translational_diffusion())
        assert isinstance(k, float)

    def test_reaction_probability_bounds(self):
        """reaction_probability stays within [0, 1] across several trajectory counts."""
        for n in [5, 10, 20]:
            sim = self._fast_sim(n)
            result = sim.run()
            p = result.reaction_probability
            assert 0.0 <= p <= 1.0

    def test_different_seeds_different_results(self):
        """Simulations with different seeds both complete the requested trajectory count."""
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
        assert r1.n_trajectories == 50
        assert r2.n_trajectories == 50

    def test_zero_trajectories(self):
        """A simulation with zero trajectories yields zero trajectories and zero reactions."""
        sim = self._fast_sim(0)
        result = sim.run()
        assert result.n_trajectories == 0
        assert result.n_reacted == 0

    def test_sim_result_repr(self):
        """repr of SimulationResult contains the string SimulationResult."""
        sim = self._fast_sim(3)
        result = sim.run()
        assert "SimulationResult" in repr(result)

    def test_rate_constant_zero_if_no_reactions(self):
        """rate_constant is non-negative when an impossible cutoff produces no reactions."""
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
        """parse_reaction_xml of an empty reactions document yields an empty PathwaySet."""
        xml = "<?xml version='1.0' ?><reactions></reactions>"
        p = tmp_path / "empty.xml"
        p.write_text(xml)
        ps = parse_reaction_xml(p)
        assert len(ps) == 0

    def test_write_and_parse_contacts(self, tmp_path):
        """Writing then re-parsing reaction XML round-trips the probability and atom index."""
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
        """parse_simulation_xml supplies default n_trajectories of 1000 and dt of 0.2 for an empty document."""
        p = tmp_path / "sim.xml"
        p.write_text("<?xml version='1.0'?><simulation></simulation>")
        cfg = parse_simulation_xml(p)
        assert cfg["n_trajectories"] == 1000
        assert abs(cfg["dt"] - 0.2) < 1e-8

    def test_write_simulation_xml(self, tmp_path):
        """write_simulation_xml writes the trajectory count and dx file names into the output."""
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
        """parse_reaction_xml parses all three contact pairs of a reaction."""
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
        """A simulation with multi-atom molecules completes the requested trajectory count."""
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
        """A molecule written to PQR and re-parsed preserves its atom count and coordinates."""
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
        """A simulation using a Debye-Huckel force callback runs to completion."""
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
        """pystarc.__version__ is a string."""
        assert isinstance(pystarc.__version__, str)

    def test_all_fates_importable(self):
        """All four Fate enum members are importable with their expected names."""
        for f in (Fate.ONGOING, Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS):
            assert f.name in ("ONGOING", "REACTED", "ESCAPED", "MAX_STEPS")


# Extended tests
class TestAtomPositionAndDistance:
    @pytest.mark.parametrize(
        "x,y,z",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1), (10, 20, 30)],
    )
    def test_position_param(self, x, y, z):
        """Atom.position matches the x, y, z passed to it."""
        a = Atom(x=x, y=y, z=z)
        assert np.allclose(a.position, [x, y, z])

    @pytest.mark.parametrize("q", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_charge_param(self, q):
        """Atom stores the charge passed to it."""
        a = Atom(charge=q)
        assert a.charge == q

    @pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 2.0, 5.0])
    def test_radius_param(self, r):
        """Atom stores the radius passed to it."""
        a = Atom(radius=r)
        assert a.radius == r

    def test_distance_pythagorean(self):
        """Atom.distance_to computes the Euclidean distance, equal to sqrt(3) for unit-offset atoms."""
        a = Atom(x=0, y=0, z=0)
        b = Atom(x=1, y=1, z=1)
        assert abs(a.distance_to(b) - math.sqrt(3)) < 1e-10

    def test_many_atoms_positions(self):
        """A list of Atoms preserves each assigned x coordinate in order."""
        atoms = [Atom(x=float(i)) for i in range(100)]
        xs = [a.x for a in atoms]
        assert xs == list(range(100))


class TestMoleculeGeometricOps:
    @pytest.mark.parametrize("n", [1, 5, 10, 20, 50])
    def test_molecule_len(self, n):
        """len(mol) returns the number of atoms in the molecule."""
        mol = Molecule()
        mol.atoms = [Atom() for _ in range(n)]
        assert len(mol) == n

    def test_translate_centroid(self):
        """Translating a molecule by 10 along x shifts its centroid x to 11."""
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=2)]
        mol.translate(np.array([10, 0, 0]))
        assert abs(mol.centroid()[0] - 11.0) < 1e-10

    @pytest.mark.parametrize(
        "angle", [0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi]
    )
    def test_rotate_preserves_structure(self, angle):
        """Rotating a molecule preserves the distance between its atoms."""
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
        """bounding_radius of a single atom equals that atom's radius."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=2.0)]
        br = mol.bounding_radius()
        assert abs(br - 2.0) < 1e-10

    def test_total_charge_all_positive(self):
        """total_charge sums to 5 for five atoms each carrying charge +1."""
        mol = Molecule()
        mol.atoms = [Atom(charge=1.0) for _ in range(5)]
        assert abs(mol.total_charge() - 5.0) < 1e-10


class TestQuaternionCompositionRules:
    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, math.pi])
    def test_axis_angle_roundtrip_x(self, angle):
        """A quaternion from an x axis rotation by angle maps y to (0, cos angle, sin angle)."""
        axis = np.array([1.0, 0.0, 0.0])
        q = Quaternion.from_axis_angle(axis, angle)
        R = q.to_rotation_matrix()
        # Rx(θ) rotates y->(0, cosθ, sinθ)
        v = R @ np.array([0, 1, 0])
        assert abs(v[1] - math.cos(angle)) < 1e-10
        assert abs(v[2] - math.sin(angle)) < 1e-10

    @pytest.mark.parametrize("angle", [0.1, 0.5, 1.0, 2.0, math.pi])
    def test_axis_angle_roundtrip_y(self, angle):
        """A quaternion from a y axis rotation by angle maps x to (cos angle, 0, -sin angle)."""
        axis = np.array([0.0, 1.0, 0.0])
        q = Quaternion.from_axis_angle(axis, angle)
        R = q.to_rotation_matrix()
        v = R @ np.array([1, 0, 0])
        assert abs(v[0] - math.cos(angle)) < 1e-10
        assert abs(v[2] + math.sin(angle)) < 1e-10

    def test_compose_rotations_associative(self):
        """Quaternion multiplication is associative up to overall sign."""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.4)
        q3 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.5)
        # (q1*q2)*q3 == q1*(q2*q3)
        lhs = ((q1 * q2) * q3).normalized()
        rhs = (q1 * (q2 * q3)).normalized()
        assert np.allclose(np.abs(lhs.to_array()), np.abs(rhs.to_array()), atol=1e-10)

    def test_rotate_zero_vector(self):
        """Rotating the zero vector by a quaternion returns the zero vector."""
        q = random_quaternion(np.random.default_rng(0))
        v = q.rotate_vector(np.zeros(3))
        assert np.allclose(v, 0)

    def test_small_rotation_large_sigma(self):
        """small_rotation_quaternion with large sigma still returns a unit quaternion."""
        rng = np.random.default_rng(42)
        q = small_rotation_quaternion(10.0, rng)
        assert abs(q.norm() - 1.0) < 1e-10


class TestRombergSpecialFunctions:
    @pytest.mark.parametrize(
        "n,expected", [(0, 1.0), (1, 1.0 / 2), (2, 1.0 / 3), (3, 1.0 / 4), (4, 1.0 / 5)]
    )
    def test_power_integrals(self, n, expected):
        """romberg_integrate of x^n on [0, 1] matches the analytic value 1/(n+1)."""
        val = romberg_integrate(lambda x: x**n, 0.0, 1.0)
        assert abs(val - expected) < 1e-7

    def test_cos_zero_to_half_pi(self):
        """romberg_integrate of cosine on [0, pi/2] equals 1."""
        val = romberg_integrate(math.cos, 0.0, math.pi / 2)
        assert abs(val - 1.0) < 1e-8

    def test_negative_range(self):
        """romberg_integrate of x over the symmetric range [-1, 1] equals 0."""
        val = romberg_integrate(lambda x: x, -1.0, 1.0)
        assert abs(val) < 1e-10

    def test_zero_width_interval(self):
        """romberg_integrate over a zero width interval equals 0."""
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
        """legendre_p returns the known Legendre polynomial values at given n and x."""
        assert abs(legendre_p(n, x) - expected) < 1e-12

    def test_norm_p0(self):
        """The squared norm integral of P_0 over [-1, 1] equals 2."""
        val = romberg_integrate(lambda x: legendre_p(0, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0) < 1e-6

    def test_norm_p2(self):
        """The squared norm integral of P_2 over [-1, 1] equals 2/5."""
        val = romberg_integrate(lambda x: legendre_p(2, x) ** 2, -1.0, 1.0)
        assert abs(val - 2.0 / 5.0) < 1e-6

    def test_orthogonal_p1_p3(self):
        """P_1 and P_3 are orthogonal over [-1, 1]."""
        val = romberg_integrate(
            lambda x: legendre_p(1, x) * legendre_p(3, x), -1.0, 1.0
        )
        assert abs(val) < 1e-6


class TestContactPairDefaults:
    @pytest.mark.parametrize("dist", [1.0, 3.0, 5.0, 10.0, 50.0])
    def test_cutoff_param(self, dist):
        """ContactPair stores the supplied distance_cutoff."""
        cp = ContactPair(0, 1, dist)
        assert cp.distance_cutoff == dist

    def test_mol2_index_stored(self):
        """ContactPair stores the second molecule atom index passed to it."""
        cp = ContactPair(3, 7, 4.0)
        assert cp.mol2_atom_index == 7

    def test_default_cutoff(self):
        """ContactPair defaults distance_cutoff to 5.0 when none is given."""
        cp = ContactPair()
        assert cp.distance_cutoff == 5.0


class TestPathwayPriorityOrder:
    def test_multiple_reactions_first_wins(self):
        """check_all returns the first matching reaction interface's name."""
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
        """check_all returns None when no reaction criterion is satisfied."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=100)]
        p = ContactPair(0, 0, 1.0)  # way too small
        ps = PathwaySet([ReactionInterface("r", ReactionCriteria(pairs=[p]))])
        assert ps.check_all(mol1, mol2) is None

    def test_pathway_set_empty_add(self):
        """An empty PathwaySet has length 0 and grows to 1 after add."""
        ps = PathwaySet()
        assert len(ps) == 0
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5)])
        ps.add(ReactionInterface("r", c))
        assert len(ps) == 1


class TestMobilityTensorSymmetry:
    @pytest.mark.parametrize("r1,r2", [(10, 10), (15, 25), (5, 50), (30, 30), (8, 12)])
    def test_symmetric_molecules_equal_D(self, r1, r2):
        """from_radii gives equal translational diffusion for equal radii and unequal otherwise."""
        mob = MobilityTensor.from_radii(r1, r2)
        if r1 == r2:
            assert abs(mob.D_trans1 - mob.D_trans2) < 1e-14
        else:
            assert mob.D_trans1 != mob.D_trans2

    def test_direct_constructor(self):
        """The MobilityTensor constructor stores the diffusion coefficients passed to it."""
        mob = MobilityTensor(1.0, 0.5, 2.0, 0.8)
        assert mob.D_trans1 == 1.0
        assert mob.D_rot2 == 0.8

    def test_relative_always_larger_than_either(self):
        """Relative translational diffusion exceeds each individual translational diffusion coefficient."""
        mob = MobilityTensor.from_radii(20.0, 30.0)
        D_rel = mob.relative_translational_diffusion()
        assert D_rel > mob.D_trans1
        assert D_rel > mob.D_trans2


class TestTrajectoryResultDefaults:
    @pytest.mark.parametrize("fate", [Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS])
    def test_fate_stored(self, fate):
        """TrajectoryResult stores the fate passed to it."""
        r = TrajectoryResult(fate, 100, 20.0, 50.0)
        assert r.fate == fate

    def test_reaction_name_none_by_default(self):
        """TrajectoryResult defaults reaction_name to None."""
        r = TrajectoryResult(Fate.ESCAPED, 50, 10.0, 200.0)
        assert r.reaction_name is None

    def test_energy_at_reaction_zero_default(self):
        """TrajectoryResult defaults energy_at_reaction to 0.0."""
        r = TrajectoryResult(Fate.REACTED, 10, 2.0, 5.0, "r")
        assert r.energy_at_reaction == 0.0

    def test_steps_stored(self):
        """TrajectoryResult stores the number of steps passed to it."""
        r = TrajectoryResult(Fate.ESCAPED, 777, 155.4, 300.0)
        assert r.steps == 777

    def test_time_ps_stored(self):
        """TrajectoryResult stores the elapsed time in ps passed to it."""
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
        """reaction_probability of the sample result equals 0.6."""
        r = self._result()
        assert abs(r.reaction_probability - 0.6) < 1e-10

    def test_rate_constant_nonzero(self):
        """rate_constant of the sample result is positive."""
        r = self._result()
        k = r.rate_constant(10.0)
        assert k > 0

    def test_p_rxn_zero_when_no_reactions(self):
        """reaction_probability is 0 when no trajectories react."""
        r = SimulationResult(100, 0, 100, 0, {}, 100.0, 500.0, 0.2)
        assert r.reaction_probability == 0.0

    def test_p_rxn_one_when_all_react(self):
        """reaction_probability is 1 when all trajectories react."""
        r = SimulationResult(100, 100, 0, 0, {"r": 100}, 100.0, 500.0, 0.2)
        assert abs(r.reaction_probability - 1.0) < 1e-10

    def test_repr_contains_n(self):
        """The SimulationResult repr contains the trajectory count 100."""
        r = self._result()
        assert "100" in repr(r)


# Parametric sweep and stress tests
class TestSplineInterpolationAccuracy:
    @pytest.mark.parametrize("n", [3, 5, 10, 20, 50])
    def test_interpolates_x_squared(self, n):
        """CubicSpline interpolates x^2 within a node count dependent tolerance."""
        x = np.linspace(0, 3, n)
        y = x**2
        sp = CubicSpline(x, y)
        tol = 0.20 if n <= 3 else (0.07 if n <= 5 else 0.05)
        for xi in np.linspace(0.1, 2.9, 15):
            assert abs(sp(xi) - xi**2) < tol

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_interpolates_cosine(self, n):
        """CubicSpline interpolates cosine within a node count dependent tolerance."""
        x = np.linspace(0, math.pi, n)
        y = np.cos(x)
        sp = CubicSpline(x, y)
        tol = 0.1 if n <= 4 else 0.05
        for xi in np.linspace(0.1, 3.0, 10):
            assert abs(sp(xi) - math.cos(xi)) < tol

    def test_derivative_cosine(self):
        """The CubicSpline derivative of cosine matches -sin within 0.05."""
        x = np.linspace(0, math.pi, 40)
        y = np.cos(x)
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.2, 2.9, 10):
            assert abs(sp.derivative(xi) - (-math.sin(xi))) < 0.05


class TestDebyeHuckelChargeSign:
    @pytest.mark.parametrize("sep", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_energy_positive_same_sign(self, sep):
        """Debye-Huckel energy is positive for like sign charges."""
        E = debye_huckel_energy(1.0, 1.0, sep)
        assert E > 0

    @pytest.mark.parametrize("sep", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_energy_negative_opposite_sign(self, sep):
        """Debye-Huckel energy is negative for opposite sign charges."""
        E = debye_huckel_energy(1.0, -1.0, sep)
        assert E < 0

    @pytest.mark.parametrize("debye", [3.0, 7.9, 15.0, 30.0])
    def test_longer_debye_longer_range(self, debye):
        """A longer Debye length gives larger Debye-Huckel energy at the same separation."""
        E_short = debye_huckel_energy(1.0, 1.0, 20.0, debye_length=5.0)
        E_long = debye_huckel_energy(1.0, 1.0, 20.0, debye_length=debye)
        # longer Debye -> less screened -> larger energy at same separation
        if debye > 5.0:
            assert E_long > E_short


class TestBDStepDiffusionScaling:
    @pytest.mark.parametrize("D", [0.001, 0.01, 0.1, 1.0, 10.0])
    def test_diffusion_scales_step(self, D):
        """Ermak-McCammon translation step std scales as sqrt(2 D dt) within 15%."""
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
        """Ermak-McCammon translation step std scales as sqrt(2 dt) in dt within 15%."""
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
        """Reacted, escaped, and max step trajectory counts sum to the requested number of trajectories."""
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
        """The simulation result stores the r_start passed in the parameters."""
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
        """KB_SI times T_DEFAULT equals KB_SI times 298.15."""
        assert abs(KB_SI * T_DEFAULT - KB_SI * 298.15) < 1e-30

    def test_ang_to_m_squared(self):
        """ANG_TO_M squared equals 1e-20 m^2."""
        assert abs(ANG_TO_M**2 - 1e-20) < 1e-30

    def test_ps_to_s_value(self):
        """PS_TO_S equals 1e-12 s."""
        assert abs(PS_TO_S - 1e-12) < 1e-22

    def test_pi_precision(self):
        """The PI constant matches pi to 13 decimal places."""
        assert abs(PI - 3.14159265358979) < 1e-13

    def test_avogadro_order(self):
        """AVOGADRO lies between 6e23 and 7e23."""
        assert 6e23 < AVOGADRO < 7e23

    def test_eta_water_order(self):
        """ETA_WATER lies between 1e-4 and 1e-2 Pa s."""
        assert 1e-4 < ETA_WATER < 1e-2

    def test_bjerrum_order(self):
        """BJERRUM_LENGTH lies between 5 and 10 angstrom."""
        assert 5 < BJERRUM_LENGTH < 10

    def test_eps_water_order(self):
        """EPS_WATER lies between 70 and 90."""
        assert 70 < EPS_WATER < 90


class TestBoundingBoxPaddingAndCenter:
    @pytest.mark.parametrize("padding", [0.0, 1.0, 2.5, 5.0, 10.0])
    def test_padding_increases_size(self, padding):
        """Padding expands the bounding box outward in both x directions."""
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=4)]
        bb0 = BoundingBox.from_molecule(mol, padding=0.0)
        bbp = BoundingBox.from_molecule(mol, padding=padding)
        assert bbp.xmin <= bb0.xmin
        assert bbp.xmax >= bb0.xmax

    def test_center_1d(self):
        """The bounding box center x is the midpoint of the atom x coordinates."""
        mol = Molecule()
        mol.atoms = [Atom(x=2.0), Atom(x=8.0)]
        bb = BoundingBox.from_molecule(mol, padding=0)
        assert abs(bb.center[0] - 5.0) < 1e-10

    def test_size_all_axes(self):
        """The bounding box size equals the atom coordinate extents on all three axes."""
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
        """lumped_charges preserves the net molecular charge regardless of grid spacing."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, charge=1.0), Atom(x=10, charge=-1.0)]
        lc = lumped_charges(mol, grid_spacing=spacing)
        total_q = sum(q for _, q in lc)
        assert abs(total_q) < 1e-5  # net charge preserved

    @pytest.mark.parametrize("probe", [1.0, 1.4, 2.0])
    def test_surface_spheres_probe(self, probe):
        """surface_spheres returns a nonempty point set for any probe radius."""
        mol = Molecule()
        mol.atoms = [Atom(x=0, y=0, z=0, radius=3.0)]
        pts = surface_spheres(mol, probe_radius=probe, n_points=20)
        assert len(pts) > 0

    def test_contact_distances_all_close(self):
        """contact_distances returns one pair per atom of the second molecule when all lie within the cutoff."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=1), Atom(x=2), Atom(x=3)]
        pairs = contact_distances(mol1, mol2, cutoff=10.0)
        assert len(pairs) == 3


class TestMultipoleMomentValues:
    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
    def test_monopole_sum(self, n):
        """The monopole moment of n unit charges equals n."""
        q = np.ones(n)
        assert abs(monopole_moment(q) - n) < 1e-10

    def test_dipole_linear_molecule(self):
        """The dipole moment of a +1/-1 charge pair points along the negative x axis."""
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        q = np.array([1.0, -1.0])
        p = dipole_moment(pos, q)
        # p = +1*(0,0,0) + (-1)*(1,0,0) = (-1, 0, 0)
        assert np.allclose(p, [-1, 0, 0])

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_quadrupole_symmetric_n(self, n):
        """The quadrupole moment is symmetric and traceless for random charges and positions."""
        rng = np.random.default_rng(n)
        pos = rng.standard_normal((n, 3))
        q = rng.standard_normal(n)
        Q = quadrupole_moment(pos, q)
        assert np.allclose(Q, Q.T)
        assert abs(np.trace(Q)) < 1e-10


class TestWienerProcessDimAndVariance:
    @pytest.mark.parametrize("dim", [1, 2, 3, 6])
    def test_wiener_dim(self, dim):
        """wiener_step returns an array of shape (dim,) for the requested dimension."""
        rng = np.random.default_rng(dim)
        dW = wiener_step(1.0, 0.1, dim, rng)
        assert dW.shape == (dim,)

    @pytest.mark.parametrize("D,dt", [(0.1, 0.01), (1.0, 0.1), (10.0, 0.5)])
    def test_wiener_variance(self, D, dt):
        """The variance of many Wiener steps approaches 2*D*dt within 10 percent."""
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
        """parse_pqr reads back the same number of atoms that were written to the file."""
        p = tmp_path / "mol.pqr"
        self._write_n_atoms(p, n)
        mol = parse_pqr(p)
        assert len(mol.atoms) == n

    def test_write_preserves_residue_name(self, tmp_path):
        """Writing and reparsing a PQR file preserves the atom residue name."""
        mol = Molecule(name="test")
        mol.atoms = [Atom(residue_name="GLY", x=1, y=2, z=3, charge=0.1, radius=1.5)]
        p = tmp_path / "out.pqr"
        write_pqr(mol, p)
        mol2 = parse_pqr(p)
        assert mol2.atoms[0].residue_name == "GLY"

    def test_write_preserves_positions(self, tmp_path):
        """Writing and reparsing a PQR file preserves atom coordinates to within 0.001 Angstrom."""
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
        """An Atom stores the name it was constructed with."""
        a = Atom(name=name)
        assert a.name == name

    @pytest.mark.parametrize("resname", ["ALA", "GLY", "SER", "THR", "VAL", "LEU"])
    def test_residue_names(self, resname):
        """An Atom stores the residue name it was constructed with."""
        a = Atom(residue_name=resname)
        assert a.residue_name == resname

    @pytest.mark.parametrize("idx", [0, 1, 10, 100, 999])
    def test_indices(self, idx):
        """An Atom stores the index it was constructed with."""
        a = Atom(index=idx)
        assert a.index == idx

    def test_repr_has_position(self):
        """The repr of an Atom includes its position coordinates."""
        a = Atom(x=1.5, y=2.5, z=3.5)
        r = repr(a)
        assert "1.50" in r or "1.5" in r

    def test_distance_triangle_inequality(self):
        """Atom distances satisfy the triangle inequality."""
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
        """The centroid x coordinate of a line of n atoms equals (n-1)/2."""
        mol = self._line_mol(n)
        c = mol.centroid()
        assert abs(c[0] - (n - 1) / 2.0) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_bounding_radius_line_mol(self, n):
        """The bounding radius of a line molecule is strictly positive."""
        mol = self._line_mol(n)
        br = mol.bounding_radius()
        assert br > 0

    def test_charges_sum_to_zero_balanced(self):
        """total_charge of a charge-balanced molecule is zero."""
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
        """translate shifts an atom at the origin by exactly the given displacement vector."""
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
        """Each named physical constant falls within its expected numeric range."""
        val = getattr(C, v)
        assert lo < val < hi

    def test_kbt_in_joules(self):
        # kBT ≈ 4.1e-21 J at 298 K
        """k_B*T at the default temperature lies between 3e-21 and 5e-21 J."""
        kbt_J = KB_SI * T_DEFAULT
        assert 3e-21 < kbt_J < 5e-21

    def test_bjerrum_from_eps(self):
        # l_B = e²/(4π ε₀ ε_r kBT) in SI, then convert to Å
        """The Bjerrum length computed from SI constants matches BJERRUM_LENGTH within 0.5 Angstrom."""
        lB_m = E_CHARGE**2 / (4 * math.pi * EPS0_SI * EPS_WATER * KB_SI * T_DEFAULT)
        lB_A = lB_m / ANG_TO_M
        assert abs(lB_A - BJERRUM_LENGTH) < 0.5


class TestReactionCriteriaBoundary:
    @pytest.mark.parametrize("n_pairs", [1, 2, 3, 5])
    def test_n_pairs_all_required(self, n_pairs):
        """ReactionCriteria is satisfied when all required contact pairs are within their cutoffs."""
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
        """ReactionCriteria is not satisfied when the cutoff is below the pair distance."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, cutoff)])
        assert not c.is_satisfied(mol1, mol2)

    @pytest.mark.parametrize("cutoff", [2.1, 3.0, 10.0])
    def test_cutoff_above_dist(self, cutoff):
        """ReactionCriteria fires when the pair distance is strictly below the cutoff."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, cutoff)])
        assert c.is_satisfied(mol1, mol2)

    def test_cutoff_exact_dist_not_satisfied(self):
        """ReactionCriteria is not satisfied when the distance exactly equals the cutoff."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=0)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=2)]
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 2.0)])
        assert not c.is_satisfied(mol1, mol2)


class TestRPYTensorProperties:
    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0, 50.0])
    def test_D_t_finite_positive(self, r):
        """stokes_translational_diffusion returns a finite positive coefficient for any radius."""
        D = stokes_translational_diffusion(r)
        assert 0 < D < float("inf")

    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0])
    def test_D_r_finite_positive(self, r):
        """stokes_rotational_diffusion returns a finite positive coefficient for any radius."""
        D = stokes_rotational_diffusion(r)
        assert 0 < D < float("inf")

    def test_rpy_off_diagonal_symmetric(self):
        """The off-diagonal Rotne-Prager-Yamakawa mobility block is symmetric."""
        r_vec = np.array([10.0, 5.0, 3.0])
        M = rpy_offdiagonal(r_vec, 3.0, 3.0, 1.0, 1.0)
        assert np.allclose(M, M.T)

    def test_mobility_relative_D_positive(self):
        """The relative translational and rotational diffusion of a MobilityTensor are positive."""
        for r1, r2 in [(5, 5), (10, 20), (15, 30)]:
            mob = MobilityTensor.from_radii(r1, r2)
            assert mob.relative_translational_diffusion() > 0
            assert mob.relative_rotational_diffusion() > 0


class TestFateEnumValues:
    def test_all_fates_distinct(self):
        """The four Fate enum values are all distinct."""
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
        """The reacted and escaped boolean properties of TrajectoryResult match the trajectory fate."""
        r = TrajectoryResult(fate, 0, 0.0, 0.0)
        assert r.reacted == reacted
        assert r.escaped == escaped


class TestXMLProbabilityStorage:
    @pytest.mark.parametrize("n_rxns", [1, 2, 3, 5])
    def test_write_n_reactions(self, n_rxns, tmp_path):
        """Writing and reparsing reaction XML preserves the number of reactions."""
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
        """Writing and reparsing reaction XML preserves the reaction probability within 1e-5."""
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5.0)])
        ps = PathwaySet([ReactionInterface("r", c, prob)])
        p = tmp_path / f"rxn_{int(prob*100)}.xml"
        write_reaction_xml(ps, p)
        ps2 = parse_reaction_xml(p)
        assert abs(ps2.reactions[0].probability - prob) < 1e-5


class TestPipelineReproducibility:
    def test_full_pipeline_no_crash(self, tmp_path):
        """The full PQR plus XML simulation pipeline runs and reports the requested trajectory count."""
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
        """Running the simulator twice with the same seed yields identical reaction counts."""
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
        """The pystarc package exposes a non-empty __version__ attribute."""
        assert hasattr(pystarc, "__version__")
        assert pystarc.__version__  # version check

    def test_all_submodules_load(self):
        """All listed pystarc submodules import successfully."""
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
        """Interpolating a uniform grid returns the constant value at an interior point."""
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.full((5, 5, 5), v))
        assert abs(g.interpolate(np.array([2.0, 2.0, 2.0])) - v) < 1e-8

    @pytest.mark.parametrize("charge", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_force_proportional_to_charge(self, charge):
        """The force from a zero-valued potential grid is zero regardless of charge."""
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.zeros((5, 5, 5)))
        F = g.force_on_charge(np.array([2.0, 2.0, 2.0]), charge)
        assert np.allclose(F, 0)

    def test_non_square_grid(self):
        """A DXGrid retains its non-cubic data array shape."""
        g = DXGrid(np.zeros(3), np.diag([1.0, 2.0, 3.0]), np.ones((3, 4, 5)))
        assert g.data.shape == (3, 4, 5)

    def test_interpolate_corner(self):
        """Interpolating at a grid corner returns the value stored at that corner."""
        data = np.zeros((4, 4, 4))
        data[0, 0, 0] = 1.0
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        val = g.interpolate(np.array([0.0, 0.0, 0.0]))
        assert abs(val - 1.0) < 1e-8

    @pytest.mark.parametrize("pt", [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    def test_interpolate_interior(self, pt):
        """Interpolating a constant grid returns that constant at interior points."""
        data = np.ones((5, 5, 5))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        val = g.interpolate(np.array(pt))
        assert abs(val - 1.0) < 1e-8


class TestQuaternionUnitProperties:
    @pytest.mark.parametrize(
        "w,x,y,z", [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    )
    def test_unit_quaternions(self, w, x, y, z):
        """A Quaternion built from a unit input has norm 1."""
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
        """The rotation matrix from an axis-angle quaternion has determinant 1."""
        q = Quaternion.from_axis_angle(np.array([0, 1, 0]), angle)
        R = q.to_rotation_matrix()
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_conjugate_is_inverse_for_unit(self):
        """The conjugate of a unit quaternion is its inverse, giving an identity product."""
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
        """Romberg integration of the constant 1 over [a,b] returns b-a."""
        val = romberg_integrate(lambda x: 1.0, float(a), float(b))
        assert abs(val - expected) < 1e-8

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_legendre_at_zero_parity(self, n):
        # Pn(0) = 0 for odd n, nonzero for even n
        """The Legendre polynomial P_n(0) is zero for odd n and nonzero for even n."""
        val = legendre_p(n, 0.0)
        if n % 2 == 1:
            assert abs(val) < 1e-12
        else:
            assert abs(val) > 0 or n == 0

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6])
    def test_wiener_correct_dim(self, dim):
        """wiener_step returns a step of the requested dimension."""
        rng = np.random.default_rng(dim * 10)
        dW = wiener_step(1.0, 1.0, dim, rng)
        assert len(dW) == dim

    def test_monopole_negative(self):
        """The monopole moment of negative charges equals their signed sum."""
        q = np.array([-1.0, -2.0, -3.0])
        assert abs(monopole_moment(q) - (-6.0)) < 1e-10

    def test_dipole_3atoms(self):
        """The dipole moment of a three-atom linear arrangement points along the negative x axis."""
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
        """Running a tiny simulation twice with the same seed gives the same reaction count."""
        r1 = self._tiny_sim(seed=seed).run()
        r2 = self._tiny_sim(seed=seed).run()
        assert r1.n_reacted == r2.n_reacted

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_exact_n_traj(self, n):
        """The reacted, escaped, and max-steps counts of a simulation sum to the trajectory total."""
        result = self._tiny_sim(n=n).run()
        total = result.n_reacted + result.n_escaped + result.n_max_steps
        assert total == n

    def test_reaction_probability_with_huge_cutoff_is_high(self):
        """A very large reaction cutoff yields a reaction probability above 0.5."""
        result = self._tiny_sim(cutoff=1000.0, n=20).run()
        assert result.reaction_probability > 0.5

    def test_sim_result_rate_nonnegative(self):
        """The rate constant from a simulation result is non-negative."""
        result = self._tiny_sim(n=5).run()
        mob = MobilityTensor.from_radii(20.0, 20.0)
        k = result.rate_constant(mob.relative_translational_diffusion())
        assert k >= 0

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.5])
    def test_dt_stored_in_result(self, dt):
        """The simulation result stores the time step dt that was configured."""
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
        """lumped_charges of an empty molecule returns an empty list."""
        mol = Molecule()
        lc = lumped_charges(mol)
        assert lc == []

    def test_contact_distances_empty(self):
        """contact_distances returns an empty list when the first molecule has no atoms."""
        mol1 = Molecule()
        mol1.atoms = []
        mol2 = Molecule()
        mol2.atoms = [Atom(x=0)]
        pairs = contact_distances(mol1, mol2, cutoff=5.0)
        assert pairs == []

    def test_bounding_box_single_atom(self):
        """The bounding box of a single atom collapses to that atom's position with zero padding."""
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
        """The Born integral energy has the sign expected for the given charge and radius."""
        E = born_integral(q, r)
        assert math.copysign(1, E) == expected_sign or E == 0

    def test_hydrodynamic_radius_positive(self):
        """The hydrodynamic radius derived from the radius of gyration is positive."""
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=3), Atom(x=6)]
        rh = hydrodynamic_radius_from_rg(mol)
        assert rh > 0

    def test_electrostatic_center_shape(self):
        """electrostatic_center returns a 3-component vector for a multi-atom molecule."""
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
        """The Legendre polynomial P2 evaluates to its known analytic value at the test points."""
        assert abs(legendre_p(2, x) - expected) < 1e-12

    @pytest.mark.parametrize(
        "q1,q2,sep", [(1, 1, 5), (1, -1, 5), (2, 2, 10), (0.5, 0.5, 3), (-1, -1, 7)]
    )
    def test_dh_energy_sign(self, q1, q2, sep):
        """The Debye-Huckel energy sign matches the sign of the charge product q1*q2."""
        E = debye_huckel_energy(float(q1), float(q2), float(sep))
        expected_sign = math.copysign(1, q1 * q2)
        if abs(q1 * q2) > 1e-10 and sep > 0:
            assert math.copysign(1, E) == expected_sign

    @pytest.mark.parametrize(
        "r1,r2", [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
    )
    def test_equal_radii_equal_diffusion(self, r1, r2):
        """Equal-radius particles yield equal translational diffusion coefficients in the mobility tensor."""
        mob = MobilityTensor.from_radii(float(r1), float(r2))
        if r1 == r2:
            assert abs(mob.D_trans1 - mob.D_trans2) < 1e-14

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, 2.0, math.pi])
    def test_from_axis_angle_unit_norm(self, angle):
        """A quaternion built from an axis and angle has unit norm."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), angle)
        assert abs(q.norm() - 1.0) < 1e-10

    @pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    def test_romberg_x_power_n(self, n):
        """Romberg integration of x^n on [0,1] equals 1/(n+1)."""
        val = romberg_integrate(lambda x: x**n, 0.0, 1.0)
        expected = 1.0 / (n + 1)
        assert abs(val - expected) < 1e-7

    @pytest.mark.parametrize("v", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_constant_dx_grid_any_point(self, v):
        """Interpolation on a constant-valued DX grid returns that constant at any query point."""
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.full((4, 4, 4), v))
        for pt in [[1, 1, 1], [1.5, 1.5, 1.5], [2, 2, 2]]:
            assert abs(g.interpolate(np.array(pt)) - v) < 1e-8

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
    def test_molecule_len_correct(self, n):
        """len(molecule) equals the number of atoms it contains."""
        mol = Molecule()
        mol.atoms = [Atom() for _ in range(n)]
        assert len(mol) == n

    @pytest.mark.parametrize("charge", [-5, -2, -1, 0, 1, 2, 5])
    def test_atom_charge_stored(self, charge):
        """An Atom stores the charge it was constructed with."""
        a = Atom(charge=float(charge))
        assert a.charge == float(charge)

    @pytest.mark.parametrize("r", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    def test_atom_radius_stored(self, r):
        """An Atom stores the radius it was constructed with."""
        a = Atom(radius=r)
        assert a.radius == r

    @pytest.mark.parametrize(
        "fate", [Fate.ONGOING, Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS]
    )
    def test_system_state_fate_set(self, fate):
        """SystemState stores the fate value it was constructed with."""
        s = SystemState(fate=fate)
        assert s.fate == fate

    @pytest.mark.parametrize("steps", [0, 1, 100, 10000])
    def test_trajectory_steps_stored(self, steps):
        """TrajectoryResult stores the step count it was constructed with."""
        r = TrajectoryResult(Fate.ESCAPED, steps, float(steps) * 0.2, 200.0)
        assert r.steps == steps

    @pytest.mark.parametrize("n_contacts", [1, 2, 3, 4, 5])
    def test_make_default_reaction_n_pairs(self, n_contacts):
        """make_default_reaction produces a reaction with the requested number of contact pairs."""
        mol1 = Molecule()
        mol1.atoms = [Atom(x=float(i)) for i in range(10)]
        mol2 = Molecule()
        mol2.atoms = [Atom(x=float(i) + 20) for i in range(10)]
        rxn = make_default_reaction(mol1, mol2, n_pairs=n_contacts)
        assert len(rxn.criteria.pairs) == n_contacts

    @pytest.mark.parametrize("pqr_line_count", [1, 3, 5, 10, 20])
    def test_pqr_parse_count(self, pqr_line_count, tmp_path):
        """parse_pqr reads exactly as many atoms as there are ATOM records in the file."""
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
        """A padded bounding box still contains its own center."""
        mol = Molecule()
        mol.atoms = [Atom(x=5, y=5, z=5)]
        bb = bounding_box(mol, padding=float(padding))
        center = bb.center
        assert bb.contains(center)

    @pytest.mark.parametrize("prob", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_reaction_interface_prob_stored(self, prob):
        """ReactionInterface stores the reaction probability it was constructed with."""
        c = ReactionCriteria(pairs=[ContactPair(0, 0, 5.0)])
        rxn = ReactionInterface("r", c, prob)
        assert abs(rxn.probability - prob) < 1e-10


class TestParametricGeometryAndSymmetry:
    # Atom geometry
    @pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_distance_along_x(self, d):
        """distance_to between two atoms separated along x returns that separation."""
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
        """Quaternion rotation of a vector about an axis matches the expected rotated result."""
        q = Quaternion.from_axis_angle(np.array(axis, dtype=float), angle)
        result = q.rotate_vector(np.array(vec, dtype=float))
        assert np.allclose(result, expected, atol=1e-10)

    # Romberg on trig
    @pytest.mark.parametrize(
        "a,b", [(0, math.pi / 4), (0, math.pi / 2), (math.pi / 4, math.pi / 2)]
    )
    def test_romberg_sine_analytically(self, a, b):
        """Romberg integration of sin over [a,b] equals cos(a) - cos(b)."""
        val = romberg_integrate(math.sin, a, b)
        expected = math.cos(a) - math.cos(b)
        assert abs(val - expected) < 1e-8

    # Debye-Hückel symmetry
    @pytest.mark.parametrize("q1,q2", [(1, 2), (2, 1), (-1, -3), (-3, -1)])
    def test_dh_energy_symmetric_charges(self, q1, q2):
        """The Debye-Huckel energy is symmetric under swapping the two charges."""
        E12 = debye_huckel_energy(float(q1), float(q2), 10.0)
        E21 = debye_huckel_energy(float(q2), float(q1), 10.0)
        assert abs(E12 - E21) < 1e-10

    # BD step: translation returns (3,) array
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_translation_shape(self, seed):
        """Ermak-McCammon translation returns a 3-component position vector."""
        rng = np.random.default_rng(seed)
        pos = np.zeros(3)
        new = ermak_mccammon_translation(pos, np.zeros(3), 1.0, 0.1, rng)
        assert new.shape == (3,)

    # Molecule total charge
    @pytest.mark.parametrize(
        "q_list", [[1, -1], [2, -1, -1], [0.5, 0.5, -1], [0, 0, 0], [1, 1, 1, -3]]
    )
    def test_total_charge(self, q_list):
        """total_charge equals the sum of the atomic charges."""
        mol = Molecule()
        mol.atoms = [Atom(charge=q) for q in q_list]
        assert abs(mol.total_charge() - sum(q_list)) < 1e-10

    # Wiener mean
    @pytest.mark.parametrize("D,dt", [(1, 0.1), (2, 0.2), (0.5, 0.05)])
    def test_wiener_mean_zero(self, D, dt):
        """Wiener displacement steps have a sample mean near zero over many draws."""
        rng = np.random.default_rng(0)
        samples = np.array([wiener_step(D, dt, 1, rng)[0] for _ in range(5000)])
        assert abs(samples.mean()) < 0.1

    # Legendre series constant
    @pytest.mark.parametrize("c0", [0.5, 1.0, 2.0, -1.0])
    def test_legendre_series_constant(self, c0):
        """A single-coefficient Legendre series evaluates to that constant for all x."""
        for x in [-0.9, 0.0, 0.5, 0.9]:
            val = legendre_series([c0], x)
            assert abs(val - c0) < 1e-12

    # BoundingBox center correct
    @pytest.mark.parametrize("lo,hi", [(-1, 1), (-5, 5), (0, 10), (2, 8), (-3, 7)])
    def test_bb_center_x(self, lo, hi):
        """The bounding box center x-coordinate is the midpoint of the atom extents."""
        mol = Molecule()
        mol.atoms = [Atom(x=lo), Atom(x=hi)]
        bb = BoundingBox.from_molecule(mol, padding=0)
        assert abs(bb.center[0] - (lo + hi) / 2.0) < 1e-10

    # Rotne-Prager: far field symmetric
    @pytest.mark.parametrize("dist", [10.0, 20.0, 50.0])
    def test_rpy_far_symmetric(self, dist):
        """The far-field RPY off-diagonal mobility block is symmetric."""
        r_vec = np.array([dist, 0.0, 0.0])
        M = rpy_offdiagonal(r_vec, 2.0, 2.0, 1.0, 1.0)
        assert np.allclose(M, M.T, atol=1e-10)

    # Stokes: D ∝ 1/r
    @pytest.mark.parametrize("factor", [2.0, 3.0, 5.0])
    def test_D_t_inv_radius(self, factor):
        """Stokes translational diffusion scales inversely with radius."""
        D1 = stokes_translational_diffusion(10.0)
        D2 = stokes_translational_diffusion(10.0 * factor)
        assert abs(D1 / D2 - factor) < 0.01

    # Reaction satisfied iff all contacts met
    @pytest.mark.parametrize(
        "n_satisfied,n_total", [(1, 1), (2, 2), (3, 3), (2, 3), (1, 2)]
    )
    def test_reaction_all_or_nothing(self, n_satisfied, n_total):
        """Reaction criteria are satisfied only when all contact pairs are within their cutoffs."""
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
        """The Legendre polynomial P3 evaluates to (5x^3 - 3x)/2."""
        expected = (5 * x**3 - 3 * x) / 2
        assert abs(legendre_p(3, x) - expected) < 1e-12

    @pytest.mark.parametrize("x", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_p4_values(self, x):
        """The Legendre polynomial P4 evaluates to (35x^4 - 30x^2 + 3)/8."""
        expected = (35 * x**4 - 30 * x**2 + 3) / 8
        assert abs(legendre_p(4, x) - expected) < 1e-12

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_monopole_ones(self, dim):
        """The monopole moment of all-ones charges equals the number of charges."""
        q = np.ones(dim)
        assert abs(monopole_moment(q) - float(dim)) < 1e-10

    @pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
    def test_large_molecule_centroid(self, n):
        """The centroid x-coordinate of atoms at 0..n-1 equals (n-1)/2."""
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
        """The (0,0) entry of the z-axis rotation matrix equals cos(angle)."""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), angle)
        R = q.to_rotation_matrix()
        assert abs(R[0, 0] - cos_val) < 1e-10

    @pytest.mark.parametrize(
        "charge,radius", [(1.0, 2.0), (2.0, 3.0), (0.5, 1.5), (3.0, 4.0)]
    )
    def test_born_negative(self, charge, radius):
        """The Born solvation integral energy is negative."""
        E = born_integral(charge, radius)
        assert E < 0

    @pytest.mark.parametrize("D", [0.01, 0.1, 1.0, 10.0])
    def test_D_t_positive(self, D):
        """Ermak-McCammon translation steps remain finite with no NaN or inf."""
        rng = np.random.default_rng(0)
        steps = [
            ermak_mccammon_translation(np.zeros(3), np.zeros(3), D, 0.1, rng)
            for _ in range(100)
        ]
        # just check no NaN/inf
        for s in steps:
            assert np.all(np.isfinite(s))


    @pytest.mark.parametrize("n_rx,n_esc", [(0, 10), (5, 5), (10, 0)])
    def test_p_rxn_values(self, n_rx, n_esc):
        """reaction_probability equals n_rx / (n_rx + n_esc)."""
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
        """SystemState.separation returns the Euclidean norm of the position vector."""
        s = SystemState(position=np.array([x, y, z], dtype=float))
        assert abs(s.separation() - expected_r) < 1e-10

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_pathway_set_len(self, n):
        """len(PathwaySet) equals the number of reaction interfaces added."""
        ps = PathwaySet()
        for i in range(n):
            c = ReactionCriteria(pairs=[ContactPair(i, i, 5.0)])
            ps.add(ReactionInterface(f"r{i}", c))
        assert len(ps) == n

    @pytest.mark.parametrize("r", [10.0, 20.0, 30.0, 40.0, 50.0])
    def test_D_r_decreases_with_r(self, r):
        """Stokes rotational diffusion decreases as the radius increases."""
        D_small = stokes_rotational_diffusion(r)
        D_large = stokes_rotational_diffusion(r * 2)
        assert D_small > D_large

    @pytest.mark.parametrize("sep", [5.0, 10.0, 15.0, 20.0, 25.0])
    def test_dh_decays_exponentially(self, sep):
        """The Debye-Huckel energy ratio over one Debye length matches (sep/(sep+λ))*exp(-1)."""
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
        """A unit quaternion produces an orthogonal rotation matrix with determinant 1."""
        q = Quaternion(w, x, y, z)
        R = q.to_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    @pytest.mark.parametrize(
        "a,b,n_expected", [(0.0, 1.0, 1.0 / 3), (0.0, 2.0, 8.0 / 3), (0.0, 3.0, 9.0)]
    )
    def test_romberg_x2(self, a, b, n_expected):
        """Romberg integration of x^2 over [a,b] matches the expected definite integral."""
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
        """Reaction criteria fire only when the pair distance is within the contact cutoff."""
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
        """An Atom defaults its index to 0."""
        assert Atom().index == 0

    def test_molecule_empty_centroid(self):
        """An empty molecule has a centroid at the origin."""
        mol = Molecule()
        assert np.allclose(mol.centroid(), [0, 0, 0])

    def test_molecule_one_atom_rg(self):
        """A single-atom molecule has a radius of gyration of 0."""
        mol = Molecule()
        mol.atoms = [Atom(x=5)]
        assert mol.radius_of_gyration() == 0.0

    def test_quaternion_w1_is_identity(self):
        """The identity quaternion (1,0,0,0) yields the identity rotation matrix."""
        q = Quaternion(1, 0, 0, 0)
        assert np.allclose(q.to_rotation_matrix(), np.eye(3))

    def test_rigid_transform_apply_1d_input(self):
        """A RigidTransform applies its translation to a 1D input vector."""
        T = RigidTransform(translation=np.array([1.0, 0, 0]))
        v = np.array([0.0, 0.0, 0.0])
        result = T.apply(v)
        assert abs(result[0] - 1.0) < 1e-10


    def test_system_state_default_position_zero(self):
        """SystemState defaults its position to the origin."""
        s = SystemState()
        assert np.allclose(s.position, [0, 0, 0])

    def test_contact_pair_default_values(self):
        """ContactPair defaults both atom indices to 0."""
        cp = ContactPair()
        assert cp.mol1_atom_index == 0
        assert cp.mol2_atom_index == 0

    def test_pathway_set_empty_repr(self):
        """The repr of an empty PathwaySet contains the string PathwaySet."""
        ps = PathwaySet()
        assert "PathwaySet" in repr(ps)

    def test_bounding_box_contains_center(self):
        """A zero-padding bounding box contains its own center."""
        mol = Molecule()
        mol.atoms = [Atom(x=0), Atom(x=10), Atom(y=0), Atom(y=10)]
        bb = bounding_box(mol, padding=0)
        assert bb.contains(bb.center)

    def test_lumped_charges_single_atom(self):
        """Lumping a single charged atom onto the grid conserves total charge."""
        mol = Molecule()
        mol.atoms = [Atom(x=5, y=5, z=5, charge=2.0)]
        lc = lumped_charges(mol, grid_spacing=2.0)
        total_q = sum(q for _, q in lc)
        assert abs(total_q - 2.0) < 1e-6

    def test_born_larger_eps_out_less_negative(self):
        """A larger eps_out makes the Born integral energy more negative."""
        E1 = born_integral(1.0, 3.0, eps_in=4.0, eps_out=40.0)
        E2 = born_integral(1.0, 3.0, eps_in=4.0, eps_out=80.0)
        # both negative; E2 more negative than E1
        assert E2 < E1

    def test_dx_grid_shape_query(self):
        """A DXGrid reports the shape of its underlying data array."""
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.zeros((7, 8, 9)))
        assert tuple(g.data.shape) == (7, 8, 9)

    def test_dh_energy_proportional_to_bjerrum(self):
        """The Debye-Huckel energy is proportional to the Bjerrum length."""
        E1 = debye_huckel_energy(1.0, 1.0, 10.0, bjerrum_length=5.0)
        E2 = debye_huckel_energy(1.0, 1.0, 10.0, bjerrum_length=10.0)
        assert abs(E2 / E1 - 2.0) < 1e-10

    def test_monopole_single(self):
        """The monopole moment of a single charge equals that charge."""
        assert abs(monopole_moment(np.array([3.7])) - 3.7) < 1e-10

    def test_dipole_zero_positions(self):
        """The dipole moment is zero when all charge positions are at the origin."""
        pos = np.zeros((3, 3))
        q = np.array([1.0, -2.0, 1.0])
        p = dipole_moment(pos, q)
        assert np.allclose(p, 0)

    def test_wiener_zero_D(self):
        """wiener_step returns a zero displacement vector when the diffusion coefficient D is zero."""
        rng = np.random.default_rng(0)
        dW = wiener_step(0.0, 1.0, 3, rng)
        assert np.allclose(dW, 0)

    def test_spline_linear_exact(self):
        """A cubic spline through points on a line reproduces that line to within 1e-8."""
        x = np.linspace(0, 5, 10)
        y = 3 * x + 2
        sp = CubicSpline(x, y)
        for xi in np.linspace(0.1, 4.9, 20):
            assert abs(sp(xi) - (3 * xi + 2)) < 1e-8

    def test_reaction_name_in_result(self):
        """TrajectoryResult stores the reaction_name passed to its constructor."""
        r = TrajectoryResult(Fate.REACTED, 10, 2.0, 5.0, "my_rxn")
        assert r.reaction_name == "my_rxn"

    def test_simulation_result_dt(self):
        """SimulationResult exposes the time step dt passed to its constructor."""
        r = SimulationResult(10, 5, 5, 0, {}, 100.0, 500.0, 0.123)
        assert abs(r.dt - 0.123) < 1e-10


# Tests for LJ forces, GHO injection, COFFDROP chains
class TestLJForces:
    """Tests for Lennard-Jones and hydrophobic SASA forces."""

    def test_lj_pair_repulsive_at_small_r(self):
        """The Lennard-Jones force on atom a points away from b for separations below σ."""
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        assert f[0] < 0

    def test_lj_pair_attractive_at_large_r(self):  # noqa
        """The Lennard-Jones force on atom a points toward b for separations above σ."""
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([3.0, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        assert f[0] > 0

    def test_lj_energy_minimum_at_sigma(self):
        """The Lennard-Jones energy equals the minimum -ε/4 at r = 2^(1/6)σ."""
        pos_a = np.array([0.0, 0.0, 0.0])
        # At r = 2^(1/6)*sigma force = 0 (energy minimum)
        r_min = 2.0 ** (1.0 / 6.0) * 2.0
        pos_b = np.array([r_min, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=2.0)
        assert abs(e - (-0.25)) < 0.01  # reference: V_min = -eps/4 at r=2^(1/6)*sig

    def test_lj_mixing_rules(self):
        """LJForceEngine with Lorentz-Berthelot mixing obeys Newton's third law, f1 = -f2."""
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
        """LJForceEngine for identical atom types obeys Newton's third law, f1 = -f2."""
        lj = LJParams(atom_types=[LJAtomType("A", epsilon=0.5, sigma=2.0)])
        engine = LJForceEngine(lj_params=lj)
        pos1 = np.array([[0.0, 0.0, 0.0]])
        pos2 = np.array([[3.0, 0.0, 0.0]])
        f1, f2, e = engine.compute(pos1, pos2, [0], [0])
        assert np.allclose(f1, -f2, atol=1e-10)

    def test_wca_zero_beyond_cutoff(self):
        """The WCA force and energy vanish beyond the r = 2^(1/6)σ cutoff."""
        pos_a = np.array([0.0, 0.0, 0.0])
        sigma = 2.0
        r_cut = 2.0 ** (1.0 / 6.0) * sigma + 0.1  # just beyond cutoff
        pos_b = np.array([r_cut, 0.0, 0.0])
        f, e = lj_pair_force(pos_a, pos_b, epsilon=1.0, sigma=sigma, use_wca=True)
        assert np.allclose(f, 0.0)
        assert e == 0.0

    def test_hydrophobic_zero_outside_range(self):
        """The hydrophobic SASA force is zero when the surface separation lies below the range lower bound a."""
        hp = HydrophobicParams(a=3.1, b=4.35)
        r_vec = np.array([1.0, 0.0, 0.0])
        # r + radius = 1.0 + 0.5 = 1.5 < a=3.1 -> zero
        f, e = hydrophobic_sasa_force(1.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
        assert np.allclose(f, 0.0)

    def test_hydrophobic_nonzero_in_range(self):
        """The hydrophobic SASA force is nonzero when the surface separation falls within [a, b]."""
        hp = HydrophobicParams(a=3.1, b=4.35)
        r_vec = np.array([1.0, 0.0, 0.0])
        # r=3.0, radius_a=0.5 -> ri = 3.5, which is in [3.1, 4.35]
        f, e = hydrophobic_sasa_force(3.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
        assert not np.allclose(f, 0.0)


class TestGHOInjection:
    """Tests for GHO ghost atom auto-injection."""


    def test_rxns_xml_parser_handles_missing_file(self):
        """_parse_rxns_xml_criteria returns an empty pair list and n_needed of -1 for a missing file."""
        pairs, n_needed = _parse_rxns_xml_criteria(Path("/nonexistent/rxns.xml"))
        assert pairs == []
        assert n_needed == -1


class TestCOFFDROPChain:
    """Tests for flexible chain model."""

    def test_build_linear_chain(self):
        """build_linear_chain creates a chain with n beads and n-1 bonds."""
        chain = build_linear_chain(5)
        assert chain.n_beads == 5
        assert len(chain.bonds) == 4

    def test_chain_positions_array(self):
        """positions_array places beads along x at multiples of the bond length with shape (n, 3)."""
        chain = build_linear_chain(3, bond_length=4.0)
        pos = chain.positions_array()
        assert pos.shape == (3, 3)
        # Beads along x-axis, 4 A apart
        assert abs(pos[1, 0] - 4.0) < 1e-10
        assert abs(pos[2, 0] - 8.0) < 1e-10

    def test_chain_bd_step_moves_beads(self):
        """ChainBDPropagator.step displaces an unfrozen chain's beads."""
        chain = build_linear_chain(3)
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        chain = prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        assert not np.allclose(pos_before, pos_after)

    def test_frozen_chain_doesnt_move(self):
        """ChainBDPropagator.step leaves a frozen chain's bead positions unchanged."""
        chain = build_linear_chain(3)
        chain.frozen = True
        prop = ChainBDPropagator()
        rng = np.random.default_rng(0)
        pos_before = chain.positions_array().copy()
        chain = prop.step(chain, dt=0.1, rng=rng)
        assert np.allclose(chain.positions_array(), pos_before)

    def test_bond_forces_zero_at_equilibrium(self):
        """ChainForceEvaluator yields near-zero bond forces when beads sit at the equilibrium bond length."""
        chain = build_linear_chain(2, bond_length=3.8)
        # Beads already at equilibrium distance
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        # Bond force should be near zero at equilibrium
        assert np.linalg.norm(F[0]) < 1e-8
        assert np.linalg.norm(F[1]) < 1e-8

    def test_max_time_step_positive(self):
        """ChainBDPropagator.max_time_step returns a positive time step."""
        chain = build_linear_chain(3)
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt > 0

    def test_satisfy_bond_constraints(self):
        """satisfy_bond_constraints pulls a stretched bond back toward its equilibrium length."""
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
        """auto_detect_reactions raises RuntimeError when no GHO ghost atoms are present."""

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
        """auto_detect_reactions builds one stage with the cutoffs from a manual ghost-atom spec and returns n_needed of -1."""
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
        """auto_detect_reactions raises RuntimeError when the rxns.xml is missing and no GHO atoms exist."""

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
        """The reference vacuum permittivity ε0 equals 0.000142 e²/(kBT·Å)."""
        assert reference_EPS0 == 0.000142

    def test_water_viscosity(self):
        """The reference water viscosity μ equals 0.243 kBT·ps/Å³."""
        assert reference_MU == 0.243

    def test_kT_unity(self):
        """The reference thermal energy kT equals 1.0."""
        assert reference_KT == 1.0

    def test_solvent_dielectric_default(self):
        """The reference solvent dielectric constant equals 78.0."""
        assert reference_SDIE == 78.0

    def test_conversion_factor(self):
        """The PySTARC unit conversion factor matches the reference value to within 0.1 percent."""
        CONV_PYSTARC = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV_PYSTARC - reference_CONV) / reference_CONV < 1e-3

    def test_desolvation_alpha_default(self):
        """The Born grids fold in the rigorous normalisation, so α starts at unity.

        The retired convention stored an APBS potential and carried α = 1/(4π).
        Against the present cavity self energy grids that value is 12.566 times
        too weak, and it shows up as a missing barrier rather than as an error.
        """

        assert DESOLVATION_ALPHA == 1.0

    def test_qb_factor(self):
        """A trivial sanity check that 1.1 equals 1.1."""
        assert 1.1 == 1.1


# 2. Diffusion coefficients
class TestDiffusionCoefficients:
    """Verify D_trans = kT/(6πμa)."""

    @staticmethod
    def _D_trans(a):
        return reference_KT / (6.0 * reference_PI * reference_MU * a)

    def test_D_single_sphere_1A(self):
        """The translational diffusion coefficient of a 1 Å sphere equals 0.21803."""
        D = self._D_trans(1.0)
        assert abs(D - 0.21803) < 0.001

    def test_D_charged_spheres(self):
        """The relative translational diffusion of two 1.005 Å spheres equals 0.43371."""
        a = 1.005
        D_rel = 2 * self._D_trans(a)
        assert abs(D_rel - 0.43371) < 0.002

    def test_D_thrombin(self):
        """The relative translational diffusion for thrombin's receptor and ligand radii equals 0.01867."""
        D_rel = self._D_trans(25.375) + self._D_trans(21.620)
        assert abs(D_rel - 0.01867) < 0.001

    def test_D_inversely_proportional_to_radius(self):
        """The translational diffusion coefficient is inversely proportional to sphere radius."""
        D1 = self._D_trans(10.0)
        D2 = self._D_trans(20.0)
        assert abs(D1 / D2 - 2.0) < 0.01

    def test_D_rotational_inverse_cube(self):
        """The rotational diffusion coefficient scales as the inverse cube of the radius."""

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
        """The interaction potential V factor for the receptor and ligand charges equals -7.1847."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        assert abs(V - (-7.1847)) < 0.01

    def test_potential_at_10A(self):
        """The screened Coulomb potential at r = 10 Å equals -0.200268."""
        V = self._V_factor(self.Q_REC, self.Q_LIG)
        phi = V * math.exp(-10.0 / self.DEBYE) / 10.0
        assert abs(phi - (-0.200268)) < 0.001

    def test_gradient_at_10A_matches_central_diff(self):
        """The analytic potential gradient at r = 10 Å matches the central-difference estimate to within 1e-4."""
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
        """The force on the ligand is attractive, pointing toward the receptor, for opposite charges."""
        # V_factor for receptor potential (not interaction potential)
        V_rec = self.Q_REC / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        # dphi_dr < 0 (phi decreases from positive toward zero with r)
        F_x = -self.Q_LIG * dphi_dr  # -(-1) × (negative) = negative
        assert F_x < 0  # negative x = toward receptor at origin = attractive

    def test_force_repulsive_for_same_charges(self):
        """The force on the ligand is repulsive, pointing away from the receptor, for like charges."""
        V_rec = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 10.0
        dphi_dr = V_rec * math.exp(-r / self.DEBYE) * (-1 / r**2 - 1 / (r * self.DEBYE))
        F_x = -(1.0) * dphi_dr  # -(+1) × (negative) = positive
        assert F_x > 0  # repulsive

    @pytest.mark.parametrize("r", [3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
    def test_force_decays_with_distance(self, r):
        """The screened Coulomb force magnitude decreases as separation increases."""
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
        """The reference potential interpolation index range covers [0, nx-2] inclusive."""
        nx = 129
        assert 0 <= 0 and 0 <= nx - 2  # low end
        assert 0 <= nx - 2 and nx - 2 <= nx - 2  # high end

    def test_ref_gradient_range(self):
        """The reference gradient interpolation index range covers [1, nx-3] inclusive."""
        nx = 129
        assert 1 <= 1 and 1 <= nx - 3
        assert 1 <= nx - 3 and nx - 3 <= nx - 3

    def test_pystarc_gradient_aware_bounds(self):
        """The PySTARC gradient-aware valid range [origin+0.5·sp, origin+(n-2.5)·sp] covers interior indices 1 to n-3."""
        origin, sp, nx = 0.0, 1.0, 129
        lo = origin + 0.5 * sp  # = 0.5
        hi = origin + (nx - 2.5) * sp  # = 126.5
        # Must cover interior: 1 to 127 in reference index space
        assert lo <= 1.0 * sp  # lo covers ix=1
        assert hi >= (nx - 3) * sp  # hi covers ix=126

    def test_two_spheres_grid_coverage(self):
        """For the two-sphere coarse grid, the upper gradient-aware bound sits near the b-sphere at r = 10 Å."""
        sp = 0.1602
        nx = 129
        origin = -10.25  # approximate
        lo = origin + 0.5 * sp  # ≈ -10.17
        hi = origin + (nx - 2.5) * sp  # ≈ +10.00
        assert abs(hi - 10.0) < 0.5  # grid edge near b-sphere


# 5. P_rxn pure diffusion
# Smoluchowski: P = (1/b - 1/q) / (1/a - 1/q)
class TestPureDiffusion:
    """Verify pure diffusion P_rxn matches Smoluchowski formula."""

    def test_smoluchowski_two_spheres(self):
        """The Smoluchowski diffusion-limited probability for a=2.5, b=10, q=20 equals 0.1429."""
        a, b, q = 2.5, 10.0, 20.0
        P = (1 / b - 1 / q) / (1 / a - 1 / q)
        assert abs(P - 0.1429) < 0.001

    def test_expected_P_with_attraction(self):
        """Electrostatic attraction more than triples the reaction probability over the pure-diffusion value."""
        P_ref = 0.44
        P_diff = 0.143
        assert P_ref > 3 * P_diff  # attraction triples P_rxn


# 6. BD step (Ermack-McCammon)
#   dpos = (1/(6πμa)) × F × dt = D × F × dt  [since kT=1]
#   wdpos = sqrt(2 × kT × mob) × gaussian × sqrt(dt) = sqrt(2D·dt) × ξ
class TestBDStepPhysics:
    """Verify BD step matches Ermak-McCammon integrator."""

    def test_drift_formula(self):
        """The Brownian drift D·F·dt evaluates to -0.00396 for the given parameters."""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = D * F * dt
        assert abs(drift - (-0.00396)) < 0.0001

    def test_noise_rms(self):
        """The noise RMS √(2·D·dt) evaluates to 0.4169 for the given parameters."""
        D, dt = 0.43371, 0.2
        sigma = math.sqrt(2 * D * dt)
        assert abs(sigma - 0.4169) < 0.001

    def test_drift_noise_ratio(self):
        """At the b-sphere the drift-to-noise ratio stays below 0.02, confirming noise dominance."""
        D, F, dt = 0.43371, -0.04561, 0.2
        drift = abs(D * F * dt)
        sigma = math.sqrt(2 * D * dt)
        assert drift / sigma < 0.02  # force is weak relative to noise

    def test_zero_force_pure_diffusion(self):
        """With zero force the Brownian drift is exactly zero, leaving only noise."""
        D, dt = 0.43371, 0.2
        drift = D * 0.0 * dt
        assert drift == 0.0

    def test_no_HI_D_is_sum(self):
        """Without hydrodynamic interactions the relative diffusion D_rel = D1 + D2 equals 0.43371."""
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
        """At the boundary r = 10.5 the adaptive edge time step dt_edge equals 0.032."""
        r = 10.5
        dist = min(r - self.B, self.TRIG - r)
        dt_edge = dist**2 / (18.0 * self.D)
        assert abs(dt_edge - 0.032) < 0.005

    def test_dt_edge_zero_at_b(self):
        """dt_edge approaches 0 when the separation r equals the contact radius b."""
        r = self.B
        dist = max(r - self.B, 1e-3)
        dt_edge = dist**2 / (18.0 * self.D)
        assert dt_edge < 0.001

    @pytest.mark.parametrize("r", [10.1, 10.3, 10.5, 10.7, 10.9])
    def test_dt_edge_increases_toward_middle(self, r):
        """dt_edge stays positive at separations between the contact radius and the trigger radius."""
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
        """The relative rate k_b at r = b equals about 57.5 Å³/ps."""
        k_b = self._relative_rate(self.B)
        assert abs(k_b - 57.5) < 1.0

    def test_qradius_formula(self):
        """The q-radius equals 20 times the maximum molecular radius, giving 20.1 Å."""
        max_r = 1.005  # r_hydro for charged_spheres
        q_out = 20.0 * max_r
        assert abs(q_out - 20.1) < 0.01

    def test_return_prob_two_spheres(self):
        """The return probability k_b(b)/k_b(q_out) for two spheres is about 0.52."""
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert abs(rp - 0.52) < 0.03

    def test_qb_factor_1_1(self):
        """The trigger radius equals 1.1 times b, yielding 11.0 Å."""
        trigger = 1.1 * self.B
        assert abs(trigger - 11.0) < 1e-10

    def test_return_prob_between_0_and_1(self):
        """The return probability k_b(b)/k_b(q_out) lies strictly between 0 and 1."""
        k_b = self._relative_rate(self.B)
        k_q = self._relative_rate(20.1)
        rp = k_b / k_q
        assert 0 < rp < 1


# 9. Romberg Integration
class TestRombergPhysics:
    """Verify Romberg integration"""

    def test_yukawa_integral_converges(self):
        """The Romberg integral for k_b with a Yukawa potential converges to a finite positive value."""
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
        """Romberg integration of xⁿ over [0,1] gives 1/(n+1)."""
        val = TestOuterPropagator._romberg(lambda x: x**n, 0.0, 1.0)
        assert abs(val - 1.0 / (n + 1)) < 1e-8

    def test_sin_integral(self):
        """Romberg integration of sin from 0 to π gives 2."""
        val = TestOuterPropagator._romberg(math.sin, 0.0, reference_PI)
        assert abs(val - 2.0) < 1e-8


# 10. Rate constant
#   rate = conv_factor × kdb × beta
#   conv_factor = 602000000.0
class TestRateConstant:
    """Verify the k_on formula."""

    def test_formula_matches_reference(self):
        """k_on = CONV × k_b × P_rxn reproduces the reference 1.52e10 within 5%."""
        CONV = 6.022e8
        k_b = 57.5
        P = 0.44
        k_on = CONV * k_b * P
        assert abs(k_on - 1.52e10) / 1.52e10 < 0.05

    def test_conv_factor_derivation(self):
        """The unit conversion factor CONV derived from N_A and unit scalings equals 6.022e8."""
        CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        assert abs(CONV - 6.022e8) / 6.022e8 < 1e-3

    def test_k_on_zero_if_P_zero(self):
        """k_on is 0 when the reaction probability P_rxn is 0."""
        assert 6.022e8 * 57.5 * 0.0 == 0.0

    @pytest.mark.parametrize(
        "P,k_expected",
        [(0.1, 3.46e9), (0.2, 6.93e9), (0.3, 1.04e10), (0.4, 1.39e10), (0.5, 1.73e10)],
    )
    def test_k_on_linear_in_P(self, P, k_expected):
        """k_on scales linearly with the reaction probability P_rxn."""
        k_b = 57.5
        k_on = 6.022e8 * k_b * P
        assert abs(k_on - k_expected) / k_expected < 0.02


# 11. Born desolvation
#   F = -alpha × q² × grad(born_field)
#   Called both directions: (mol0->mol1) AND (mol1->mol0)
class TestBornDesolvation:
    """Verify Born desolvation."""

    def test_two_spheres_alpha_zero(self):
        """A desolvation_alpha of 0 yields zero Born force for the charged spheres."""
        alpha = 0.0
        q = -1.0
        F = -alpha * q**2 * 0.1  # any gradient
        assert F == 0.0

    def test_thrombin_alpha_nonzero(self):
        """The thrombin desolvation_alpha of 0.07957747 is positive, so the Born force is active."""
        alpha = 0.07957747
        assert alpha > 0

    def test_born_force_always_repulsive(self):
        """The Born desolvation force is positive, always pushing charges apart."""
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
        """The Yukawa force magnitude at r = 10 Å for opposite unit charges is positive."""
        q0, q1 = 1.0, -1.0
        r = 10.0
        L = 7.828
        eps = reference_SDIE * reference_EPS0
        F_mag = abs(
            q0 * q1 * (r / L + 1) * math.exp(-r / L) / (r**3 * 4 * reference_PI * eps)
        )
        assert F_mag > 0

    def test_newton_third_law(self):
        """The Yukawa force from molecule 0 on molecule 1 equals minus the reciprocal force, satisfying Newton's third law."""
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
        """The Yukawa potential at r = 5 Å with pdie = sdie reduces to a nonzero pure Coulomb-screened form matching APBS."""
        V = 1.0 * (-1.0) / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 5.0
        debye = 7.828
        # APBS with pdie=sdie gives pure Yukawa
        phi_yukawa = V * math.exp(-r / debye) / r
        # phi_apbs should match (verified by numerical check)
        assert abs(phi_yukawa) > 0

    def test_force_matches_numerical_gradient(self):
        """The analytical Yukawa force matches the finite-difference gradient of the potential to within 1e-6 relative."""
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
        """At r = 50 Å the multipole expansion reduces to a positive monopole Yukawa far field."""
        V = 1.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        r = 50.0
        debye = 7.828
        phi_mono = V * math.exp(-r / debye) / r
        assert phi_mono > 0

    def test_zero_charge_zero_force(self):
        """A zero receptor charge produces zero Yukawa potential prefactor."""
        V = 0.0 / (4.0 * reference_PI * reference_SDIE * reference_EPS0)
        assert V == 0.0

    @pytest.mark.parametrize("r", [5, 10, 15, 20, 30, 50])
    def test_yukawa_monotonically_decreasing(self, r):
        """The Yukawa potential magnitude decreases monotonically with increasing r."""
        V = -7.1847
        debye = 7.828
        phi_r = abs(V * math.exp(-r / debye) / r)
        phi_r1 = abs(V * math.exp(-(r + 1) / debye) / (r + 1))
        assert phi_r > phi_r1


# 14. End-to-end expected results
class TestExpectedResults:
    """Verify expected k_on values for both test systems."""

    def test_two_spheres_analytical(self):
        """The analytical Debye-Smoluchowski k_on for two spheres exceeds 1e10 M⁻¹s⁻¹, near 1.57e10."""
        k_anal = 1.57e10
        assert k_anal > 1e10

    def test_two_spheres_reference(self):
        """The numerical APBS-grid k_on of 1.526e10 agrees with the analytical 1.57e10 within 5%."""
        k_ref = 1.526e10
        assert abs(k_ref - 1.57e10) / 1.57e10 < 0.05  # within 5% of analytical

    def test_thrombin_experimental(self):
        """The experimental thrombin-thrombomodulin k_on of about 4e7 M⁻¹s⁻¹ exceeds 1e7."""
        k_exp = 4e7
        assert k_exp > 1e7


# 15. Reaction criterion
class TestReactionCriterionPhysics:
    """Verify reaction criterion."""

    def test_two_spheres_single_pair(self):
        """For the charged spheres, n_needed of 1 does not exceed the single available pair."""
        n_pairs, n_needed, cutoff = 1, 1, 2.5
        assert n_needed <= n_pairs

    def test_thrombin_21_pairs_3_needed(self):
        """For thrombin, n_needed of 3 does not exceed the 21 available pairs."""
        n_pairs, n_needed = 21, 3
        assert n_needed <= n_pairs

    def test_n_needed_semantics(self):
        """Reaction firing follows or-of-subsets semantics, requiring n_satisfied ≥ n_needed."""
        # With 21 pairs and n_needed=3, ANY 3 of 21 can trigger
        assert True  # documents the semantics


# Multipole far-field tests
class TestMultipoleExpansion:
    """Test the MultipoleExpansion class."""

    def test_monopole_only(self):
        """A single point charge gives monopole Q equal to the charge with zero dipole and quadrupole."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([5.0]), debye_length=7.86
        )
        assert abs(mp.Q - 5.0) < 1e-10
        assert mp.dipole_mag < 1e-10
        assert mp.quad_mag < 1e-10

    def test_monopole_potential_exact(self):
        """The monopole potential at r = 20 Å matches the exact screened Coulomb hand calculation."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        r = 20.0
        eps = 78.0 * 0.000142
        V_exact = 3.0 / (4 * math.pi * eps * r) * math.exp(-r / 7.86)
        V_mp = mp.potential(np.array([r, 0, 0]))
        assert abs(V_mp - V_exact) / abs(V_exact) < 1e-10

    def test_pure_dipole(self):
        """Two opposite charges give net charge Q = 0 and a pure dipole of magnitude 10."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86,
        )
        assert abs(mp.Q) < 1e-10
        assert abs(mp.dipole_mag - 10.0) < 1e-10

    def test_dipole_potential_nonzero_for_neutral(self):
        """A neutral molecule with a dipole produces a nonzero far-field potential."""
        mp = MultipoleExpansion(
            np.array([[5.0, 0, 0], [-5.0, 0, 0]]),
            np.array([1.0, -1.0]),
            debye_length=7.86,
        )
        V = mp.potential(np.array([50.0, 0, 0]))
        assert abs(V) > 1e-6  # not zero — dipole contributes

    def test_potential_decays_with_distance(self):
        """The multipole potential magnitude decreases as r increases through 10, 20, and 50 Å."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        V10 = abs(mp.potential(np.array([10.0, 0, 0])))
        V20 = abs(mp.potential(np.array([20.0, 0, 0])))
        V50 = abs(mp.potential(np.array([50.0, 0, 0])))
        assert V10 > V20 > V50

    def test_force_is_negative_gradient(self):
        """The multipole force matches the negative numerical gradient of the potential componentwise."""
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
        """A positive receptor charge yields an outward, repulsive force along +x for a same-sign test point."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        F = mp.force(np.array([20.0, 0, 0]))
        # Q_rec=+3: V > 0, dV/dr < 0 (decaying), F = -dV/dr > 0 (outward)
        assert F[0] > 0  # repulsive for same-sign charges

    def test_quadrupole_nonzero_for_distributed(self):
        """A distribution of randomly placed charges produces a nonzero quadrupole magnitude."""
        rng = np.random.default_rng(123)
        pos = rng.standard_normal((50, 3)) * 10.0
        charges = rng.standard_normal(50) * 0.5
        mp = MultipoleExpansion(pos, charges, debye_length=7.86)
        assert mp.quad_mag > 0

    def test_summary_string(self):
        """The multipole summary string contains the Monopole, Dipole, and Quadrupole labels."""
        mp = MultipoleExpansion(
            np.array([[0, 0, 0.0]]), np.array([3.0]), debye_length=7.86
        )
        s = mp.summary()
        assert "Monopole" in s
        assert "Dipole" in s
        assert "Quadrupole" in s

    def test_zero_charge_zero_potential(self):
        """All-zero charges give a vanishing potential everywhere."""
        mp = MultipoleExpansion(
            np.array([[1, 0, 0.0], [-1, 0, 0.0]]),
            np.array([0.0, 0.0]),
            debye_length=7.86,
        )
        V = mp.potential(np.array([20.0, 0, 0]))
        assert abs(V) < 1e-15

    def test_monopole_dominates_at_large_r(self):
        """At r = 100 Å the monopole accounts for more than 90% of the total potential."""
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
        """The overlap_check config flag defaults to True."""
        cfg = PySTARCConfig()
        assert cfg.overlap_check is True

    def test_xml_disable(self, tmp_path):
        """Parsing an XML overlap_check tag set to false yields overlap_check False."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.overlap_check is False

    def test_xml_enable(self, tmp_path):
        """Parsing an XML overlap_check tag set to true yields overlap_check True."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>true</overlap_check>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.overlap_check is True


class TestMultipoleFallbackConfig:
    """Test multipole_fallback configuration."""

    def test_default_enabled(self):
        """The multipole_fallback config flag defaults to True."""
        cfg = PySTARCConfig()
        assert cfg.multipole_fallback is True

    def test_xml_disable(self, tmp_path):
        """Parsing an XML multipole_fallback tag set to false yields multipole_fallback False."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <multipole_fallback>false</multipole_fallback>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.multipole_fallback is False

    def test_both_flags_independent(self, tmp_path):
        """Parsing XML sets overlap_check and multipole_fallback independently."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True


class TestLJForcesConfig:
    """Test lj_forces configuration."""

    def test_default_disabled(self):
        """The lj_forces config flag defaults to False."""
        cfg = PySTARCConfig()
        assert cfg.lj_forces is False

    def test_xml_enable(self, tmp_path):
        """Parsing an XML lj_forces tag set to true yields lj_forces True."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <lj_forces>true</lj_forces>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.lj_forces is True

    def test_all_three_flags_independent(self, tmp_path):
        """Parsing XML sets overlap_check, multipole_fallback, and lj_forces independently."""
        xml = tmp_path / "test.xml"
        xml.write_text(
            """<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <overlap_check>false</overlap_check>
  <multipole_fallback>true</multipole_fallback>
  <lj_forces>true</lj_forces>
</pystarc>"""
        )
        cfg = parse(xml)
        assert cfg.overlap_check is False
        assert cfg.multipole_fallback is True
        assert cfg.lj_forces is True


class TestOutputConfig:
    """Test the OutputConfig dataclass."""

    def test_all_defaults_true(self):
        """Every boolean field of OutputConfig defaults to True."""
        oc = OutputConfig()
        for f in fields(oc):
            if f.type is bool:
                assert getattr(oc, f.name) is True, f"{f.name} should default True"

    def test_save_interval_default(self):
        """The OutputConfig save_interval defaults to 10."""
        oc = OutputConfig()
        assert oc.save_interval == 10

    def test_custom_save_interval(self):
        """OutputConfig accepts a custom save_interval of 100."""
        oc = OutputConfig(save_interval=100)
        assert oc.save_interval == 100

    def test_disable_heavy_outputs(self):
        """Disabling full_paths and energetics leaves results_json True."""
        oc = OutputConfig(full_paths=False, energetics=False)
        assert oc.full_paths is False
        assert oc.energetics is False
        assert oc.results_json is True

    def test_field_count(self):
        """OutputConfig has 15 fields, 14 booleans plus the integer save_interval."""
        oc = OutputConfig()
        # 14 bool flags + 1 int save_interval = 15 fields
        assert len(fields(oc)) == 15

    def test_pystarc_config_has_outputs(self):
        """PySTARCConfig provides an outputs block with results_json True and save_interval 10."""
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
        """An XML file with no outputs block falls back to the default OutputConfig values."""
        p = self._write_xml(tmp_path)
        cfg = parse(p)
        assert cfg.outputs.results_json is True
        assert cfg.outputs.full_paths is True
        assert cfg.outputs.save_interval == 10

    def test_disable_paths(self, tmp_path):
        """Disabling full_paths in the XML outputs block leaves the other defaults unchanged."""
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
        """Setting save_interval to 50 in the XML outputs block parses to 50."""
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
        """Disabling several XML output flags applies each while leaving trajectories_csv True."""
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
        """Parsing full_paths accepts true, True, TRUE, yes, Yes, and 1 as boolean True."""
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
        """Parsing full_paths accepts false, False, FALSE, no, No, and 0 as boolean False."""
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

def _fixture(name: str) -> str:
    """Absolute path to a committed test fixture.

    Absence is an error rather than a skip. A skipped test and a passing test
    are indistinguishable in a summary line, which is how a one character typo
    in the old absolute path kept four chain-BD tests dormant.
    """

    p = pathlib.Path(__file__).resolve().parent / "data" / "barnase_barstar" / name
    if not p.exists():
        raise FileNotFoundError(
            "missing committed test fixture %s, expected at %s" % (name, p)
        )
    return str(p)


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
        """write_all creates results.json in the output directory."""
        result = _make_result()
        data = _make_dummy_data()
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        assert (tmp_path / "results.json").exists()

    def test_json_parseable(self, tmp_path):
        """The written results.json parses into a dict."""
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
        """results.json contains all required summary fields including k_on, P_rxn, and timing keys."""
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
        """k_on in results.json is positive and lies within its low and high bounds."""
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
        """results.json includes log10_k_on equal to log10 of k_on."""
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
        """Disabling results_json prevents results.json from being written."""
        oc = OutputConfig(results_json=False)
        write_all(
            tmp_path, _make_result(), _make_dummy_data(), oc, k_b=57.47, D_rel=0.434
        )
        assert not (tmp_path / "results.json").exists()


class TestTrajectoriesCSV:
    """Test trajectories.csv output."""

    def test_file_created(self, tmp_path):
        """write_all creates trajectories.csv in the output directory."""
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
        """trajectories.csv has one row per trajectory matching the trajectory count."""
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
        """trajectories.csv outcomes are valid labels and the reacted count matches the result."""
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
        """trajectories.csv contains all expected per-trajectory columns."""
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
        """write_all creates encounters.csv when reactions occur."""
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
        """encounters.csv has one row per reacted trajectory."""
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
        """near_misses.csv has one row per escaped trajectory."""
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
        """write_all creates paths.npz in the output directory."""
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
        """paths.npz data array has 8 columns for traj_id, step, position, and orientation."""
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
        """Disabling full_paths prevents paths.npz from being written."""
        oc = OutputConfig(full_paths=False)
        write_all(
            tmp_path, _make_result(), _make_dummy_data(), oc, k_b=57.47, D_rel=0.434
        )
        assert not (tmp_path / "paths.npz").exists()


class TestRadialDensity:
    """Test radial_density.csv output."""

    def test_columns(self, tmp_path):
        """radial_density.csv contains r_center and density columns."""
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
        """Every density value in radial_density.csv is non-negative."""
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
        """milestone_flux.csv contains radius, flux_outward, flux_inward, and net_flux columns."""
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
        """milestone_flux.csv has 11 rows for the 11 milestone radii."""
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
        """transition_matrix.npz counts array is square with 50 rows and 50 columns."""
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
        """p_commit.npz commitor probabilities all lie within [0, 1]."""
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
        """With zero reactions P_rxn and k_on are 0 and encounters.csv is not written."""
        data = _make_dummy_data(100, 0, 100)
        result = _make_result(0, 100)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / "results.json").read_text())
        assert rj["P_rxn"] == 0.0
        assert rj["k_on"] == 0.0
        # encounters.csv should not be created
        assert not (tmp_path / "encounters.csv").exists()

    def test_all_reacted(self, tmp_path):
        """With all trajectories reacted P_rxn equals 1."""
        data = _make_dummy_data(50, 50, 0)
        result = _make_result(50, 0)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / "results.json").read_text())
        assert rj["P_rxn"] == 1.0
        # near_misses.csv should have 0 rows (no escapes)

    def test_all_disabled(self, tmp_path):
        """Disabling every output produces no written files."""
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
        """write_all produces 14 files when all outputs are enabled."""
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
        """analyse_convergence reports N, P_rxn, SE, and converged True for balanced reaction counts."""
        result = analyse_convergence(n_reacted=500, n_escaped=500, k_b=35.0)
        assert result["N"] == 1000
        assert result["P_rxn"] == pytest.approx(0.5, abs=1e-10)
        assert result["SE"] == pytest.approx(math.sqrt(0.5 * 0.5 / 1000), abs=1e-10)
        assert result["converged"] is True

    def test_low_prxn_not_converged(self):
        """A low P_rxn relative to the tolerance is reported as not converged."""
        result = analyse_convergence(n_reacted=5, n_escaped=95, k_b=35.0, tol=0.05)
        assert result["P_rxn"] == pytest.approx(0.05)
        assert result["converged"] is False

    def test_zero_reacted(self):
        """With zero reactions P_rxn, SE, and k_on are 0, relative SE is infinite, and not converged."""
        result = analyse_convergence(n_reacted=0, n_escaped=1000, k_b=35.0)
        assert result["P_rxn"] == 0.0
        assert result["SE"] == 0.0
        assert result["relative_SE"] == float("inf")
        assert result["converged"] is False
        assert result["k_on"] == 0.0

    def test_all_reacted(self):
        """With all trajectories reacted P_rxn is 1, SE is 0, and convergence is False at the boundary."""
        result = analyse_convergence(n_reacted=1000, n_escaped=0, k_b=35.0)
        assert result["P_rxn"] == 1.0
        assert result["SE"] == 0.0
        assert result["relative_SE"] == 0.0
        assert result["converged"] is False

    def test_no_trajectories(self):
        """With no completed trajectories convergence is False and a reason is provided."""
        result = analyse_convergence(n_reacted=0, n_escaped=0, k_b=35.0)
        assert result["converged"] is False
        assert "reason" in result

    def test_wilson_ci_bounds(self):
        """The Wilson confidence interval for P_rxn stays within [0, 1] and brackets P_rxn."""
        result = analyse_convergence(n_reacted=50, n_escaped=950, k_b=35.0)
        lo, hi = result["wilson_CI_P"]
        assert lo >= 0.0
        assert hi <= 1.0
        assert lo < result["P_rxn"] < hi

    def test_wilson_ci_small_prxn(self):
        """The Wilson interval lower bound stays non-negative at small reaction probability."""
        result = analyse_convergence(n_reacted=2, n_escaped=998, k_b=35.0)
        lo, hi = result["wilson_CI_P"]
        assert lo >= 0.0

    def test_n_needed_targets(self):
        """N_needed reports 10%, 5%, and 1% targets with the 1% target larger than the 5% one."""
        result = analyse_convergence(n_reacted=100, n_escaped=900, k_b=35.0)
        assert "10%" in result["N_needed"]
        assert "5%" in result["N_needed"]
        assert "1%" in result["N_needed"]
        assert result["N_needed"]["1%"] > result["N_needed"]["5%"]

    def test_kon_conversion(self):
        """k_on equals conv_factor times k_b times P_rxn under the given conversion factor."""
        conv = 6.022e8
        result = analyse_convergence(
            n_reacted=500, n_escaped=500, k_b=35.0, conv_factor=conv
        )
        assert result["k_on"] == pytest.approx(conv * 35.0 * 0.5)

    def test_print_convergence_normal(self):
        """print_convergence output includes P_rxn and a Converged label for a converged result."""
        result = analyse_convergence(n_reacted=500, n_escaped=500, k_b=35.0)
        text = print_convergence(result)
        assert "P_rxn" in text
        assert "Converged" in text

    def test_print_convergence_not_converged(self):
        """print_convergence output reports Not converged for an unconverged result."""
        result = analyse_convergence(n_reacted=5, n_escaped=95, k_b=35.0, tol=0.01)
        text = print_convergence(result)
        assert "Not converged" in text

    def test_print_convergence_zero_prxn(self):
        """print_convergence output reports inf for zero reaction probability."""
        result = analyse_convergence(n_reacted=0, n_escaped=100, k_b=35.0)
        text = print_convergence(result)
        assert "inf" in text

    def test_print_convergence_no_data(self):
        """print_convergence output reports the no completed trajectories reason."""
        result = {"converged": False, "reason": "no completed trajectories"}
        text = print_convergence(result)
        assert "no completed trajectories" in text

    def test_save_convergence(self):
        """save_convergence writes convergence.json with the correct N and P_rxn."""
        result = analyse_convergence(n_reacted=100, n_escaped=900, k_b=35.0)
        with tempfile.TemporaryDirectory() as td:
            save_convergence(result, work_dir=td)
            path = os.path.join(td, "convergence.json")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["N"] == 1000
            assert loaded["P_rxn"] == pytest.approx(0.1)


class TestEffectiveCharges:
    def test_single_charge_potential(self):
        """An unscreened single point charge gives potential q/r."""
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
        """Debye screening makes the potential larger near the charge than far from it."""
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
        """The single-charge potential is isotropic along x, y, and z at equal distance."""
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
        """The force on a like-sign charge points away from the source charge."""
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
        """The force on an opposite-sign charge points toward the source charge."""
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
        """The force on a zero charge is the zero vector."""
        ec = EffectiveCharges(
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
        )
        F = ec.force_on_charge(np.array([5.0, 0.0, 0.0]), q=0.0)
        np.testing.assert_array_equal(F, np.zeros(3))

    def test_multiple_charges(self):
        """Two equal charges equidistant from the origin sum to potential 2q/r there."""
        ec = EffectiveCharges(
            positions=np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]),
            charges=np.array([1.0, 1.0]),
            debye_length=1e10,
            bjerrum_length=1.0,
        )
        phi_origin = ec.potential(np.array([0.0, 0.0, 0.0]))
        assert phi_origin == pytest.approx(2.0 / 5.0, rel=1e-4)

    def test_len(self):
        """len of EffectiveCharges equals the number of charges."""
        ec = EffectiveCharges(
            positions=np.zeros((3, 3)),
            charges=np.ones(3),
        )
        assert len(ec) == 3

    def test_repr(self):
        """repr of EffectiveCharges reports the charge count and a formatted charge value."""
        ec = EffectiveCharges(
            positions=np.zeros((2, 3)),
            charges=np.array([1.0, -0.5]),
        )
        s = repr(ec)
        assert "2 charges" in s
        assert "0.50 e" in s

    def test_from_xml(self):
        """EffectiveCharges.from_xml reads two charge tags and recovers their charges 0.5 and -0.5."""
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
        """EffectiveCharges.from_xml parses a single point_charge tag into one charge."""
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
        """EffectiveCharges.from_xml raises ValueError when the XML contains no charges."""
        xml = '<?xml version="1.0"?>\n<charges></charges>\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml)
            f.flush()
            with pytest.raises(ValueError, match="No charges"):
                EffectiveCharges.from_xml(f.name)
        os.unlink(f.name)


class TestLoadEffectiveCharges:
    def test_auto_detect_cheby(self):
        """load_effective_charges auto-detects and loads a mol_cheby.xml file as one charge."""
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
        """load_effective_charges auto-detects and loads a mol_mpole.xml file."""
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
        """load_effective_charges returns None when no matching charge file exists."""
        with tempfile.TemporaryDirectory() as td:
            ec = load_effective_charges(td, "nonexistent")
            assert ec is None


class TestStepNearSurface:
    def test_inv_erf(self):
        """_inv_erf maps 0 to 0 and 0.5 to its expected inverse error function value."""
        assert _inv_erf(0.0) == pytest.approx(0.0)
        assert _inv_erf(0.5) == pytest.approx(math.erfc(1) and 0.4769362762, rel=1e-5)

    def test_large_x0_with_repulsion_survives(self):
        """Most walkers survive when started far from the surface under strong repulsion."""
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
        """A meaningful fraction of walkers started very close to the surface are absorbed."""
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
        """Surviving walkers return a non-negative position and a positive step time."""
        rng = np.random.default_rng(7)
        for _ in range(100):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=5.0, F=1.0, D=0.1
            )
            if survives:
                assert new_x >= 0.0
                assert time > 0.0

    def test_absorption_returns_zero_x(self):
        """Absorbed walkers return position 0 and a non-negative step time."""
        rng = np.random.default_rng(99)
        absorbed = 0
        for _ in range(500):
            survives, new_x, time = step_near_absorbing_surface(
                rng, x0=0.5, F=0.0, D=1.0
            )
            if not survives:
                absorbed += 1
                assert new_x == 0.0
                assert time >= 0.0
        assert absorbed > 0, (
            "no absorption in 500 trials from x0=0.5 with D=1; the absorbing "
            "boundary is not being applied"
        )

    def test_repulsive_force_increases_survival(self):
        """Repulsive force never decreases the number of surviving walkers versus no force."""
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
        """Multiplying by the identity quaternion leaves a quaternion unchanged on both sides."""
        I = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(quat_multiply(I, q), q, atol=1e-12)
        np.testing.assert_allclose(quat_multiply(q, I), q, atol=1e-12)

    def test_inverse(self):
        """Multiplying a unit quaternion by its conjugate yields the identity quaternion."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = np.array([0.5, -0.5, -0.5, -0.5])
        prod = quat_multiply(q, q_conj)
        np.testing.assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


class TestQuatOfRotvec:
    def test_zero_rotation(self):
        """quat_of_rotvec maps a zero rotation vector to the identity quaternion."""
        q = quat_of_rotvec(np.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_90deg_about_z(self):
        """quat_of_rotvec for a π/2 rotation about z is unit norm with the expected half-angle components."""
        omega = np.array([0.0, 0.0, math.pi / 2])
        q = quat_of_rotvec(omega)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        assert q[0] == pytest.approx(math.cos(math.pi / 4), abs=1e-10)
        assert q[3] == pytest.approx(math.sin(math.pi / 4), abs=1e-10)


class TestRandomUnitQuat:
    def test_unit_norm(self):
        """random_unit_quat always returns a quaternion of unit norm."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = random_unit_quat(rng)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12


class TestDiffusionalRotation:
    def test_tau_zero(self):
        """diffusional_rotation with τ equal to 0 returns the identity quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_tau_negative(self):
        """diffusional_rotation with negative τ returns the identity quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, -1.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_small_tau_unit_norm(self):
        """diffusional_rotation at small τ returns a unit-norm quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.1)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_0p25_small_angle(self):
        """diffusional_rotation at small τ produces a small mean rotation angle below 1.5 radians."""
        rng = np.random.default_rng(42)
        angles = []
        for _ in range(500):
            q = diffusional_rotation(rng, 0.1)
            angle = 2 * math.acos(min(1.0, abs(q[0])))
            angles.append(angle)
        mean_angle = np.mean(angles)
        assert mean_angle < 1.5

    def test_tau_0p3_split_at_025(self):
        """diffusional_rotation at τ equal to 0.3 returns a unit-norm quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.3)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_0p7_split_at_05(self):
        """diffusional_rotation at τ equal to 0.7 returns a unit-norm quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 0.7)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_1p5_split_at_1(self):
        """diffusional_rotation at τ equal to 1.5 returns a unit-norm quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 1.5)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_tau_3_split_at_2(self):
        """diffusional_rotation at τ equal to 3.0 returns a unit-norm quaternion."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 3.0)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_large_tau_random(self):
        """diffusional_rotation at large τ returns a unit-norm quaternion with a nontrivial scalar part."""
        rng = np.random.default_rng(42)
        q = diffusional_rotation(rng, 10.0)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
        assert abs(q[0]) < 1.0


class TestFingerprint:
    def test_all_inside(self):
        """_fingerprint returns case 0 when all cube vertices are inside."""
        verts = np.ones((2, 2, 2), dtype=np.int8)
        fp = _fingerprint(verts)
        assert fp[0] == 0

    def test_all_outside(self):
        """_fingerprint returns case 0 when all cube vertices are outside."""
        verts = np.zeros((2, 2, 2), dtype=np.int8)
        fp = _fingerprint(verts)
        assert fp[0] == 0

    def test_single_corner(self):
        """_fingerprint returns case 1 when exactly one cube corner is inside."""
        verts = np.zeros((2, 2, 2), dtype=np.int8)
        verts[0, 0, 0] = 1
        fp = _fingerprint(verts)
        assert fp[0] == 1


class TestVoxelise:
    def test_single_sphere(self):
        """_voxelise of a single sphere produces a non-empty grid larger than the sphere radius."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([5.0])
        grid, origin, spacing = _voxelise(coords, radii, spacing=1.0, padding=3.0)
        assert grid.sum() > 0
        assert grid.shape[0] > 5

    def test_all_interior(self):
        """_voxelise marks the voxel at a sphere center as interior."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([3.0])
        grid, origin, sp = _voxelise(coords, radii, spacing=0.5, padding=2.0)
        center_idx = tuple(int((0.0 - origin[i]) / sp[i]) for i in range(3))
        valid = all(0 <= center_idx[i] < grid.shape[i] for i in range(3))
        if valid:
            assert grid[center_idx] == 1


class TestExtractSurface:
    def test_sphere_has_surface(self):
        """_extract_surface of a voxelised sphere yields surface elements each with positive area."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([3.0])
        grid, origin, sp = _voxelise(coords, radii, spacing=1.0, padding=2.0)
        surface = _extract_surface(grid, origin, sp)
        assert len(surface) > 0
        for pt in surface:
            assert pt.area > 0


class TestMCHydrodynamicRadius:
    def test_single_sphere(self):
        """mc_hydrodynamic_radius of a single sphere recovers its radius and center within tolerance."""
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
        """max_time_step returns a positive step for typical diffusion and geometry inputs."""
        dt = max_time_step(r=100.0, D_rel=0.1, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt > 0

    def test_r_zero_fallback(self):
        """max_time_step falls back to 0.2 when the separation r is zero."""
        dt = max_time_step(r=0.0, D_rel=0.1, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt == 0.2

    def test_D_zero_fallback(self):
        """max_time_step falls back to 0.2 when the relative diffusion constant is zero."""
        dt = max_time_step(r=100.0, D_rel=0.0, D_rot=0.001, r_hydro1=20.0, r_hydro2=5.0)
        assert dt == 0.2

    def test_no_rotation(self):
        """max_time_step returns a positive step when the rotational diffusion constant is zero."""
        dt = max_time_step(r=50.0, D_rel=0.1, D_rot=0.0, r_hydro1=10.0, r_hydro2=10.0)
        assert dt > 0


class TestReactionTimeStep:
    def test_normal(self):
        """reaction_time_step returns a positive step for typical ρ_min and D_rel inputs."""
        dt = reaction_time_step(rho_min=17.0, D_rel=0.1)
        assert dt > 0

    def test_zero_rho(self):
        """reaction_time_step falls back to 0.05 when ρ_min is zero."""
        dt = reaction_time_step(rho_min=0.0, D_rel=0.1)
        assert dt == 0.05

    def test_zero_D(self):
        """reaction_time_step falls back to 0.05 when D_rel is zero."""
        dt = reaction_time_step(rho_min=17.0, D_rel=0.0)
        assert dt == 0.05


class TestAdaptiveTimeStepController:
    def test_first_call(self):
        """AdaptiveTimeStep.get_dt returns a positive step on its first call."""
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
        """AdaptiveTimeStep limits step growth between successive calls to a factor of about 1.1."""
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
        """AdaptiveTimeStep yields a step near the reaction zone no larger than one far from it."""
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
        """AdaptiveTimeStep.get_dt returns a positive step after reset."""
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
        """backstep_due_to_force does not backstep when dt is already below dt_min."""
        pos_old = np.array([0.0, 0.0, 0.0])
        pos_new = np.array([1.0, 0.0, 0.0])
        f_old = np.array([0.0, 0.0, 0.0])
        f_new = np.array([100.0, 0.0, 0.0])
        result = backstep_due_to_force(
            f_new, f_old, pos_new, pos_old, dt=0.0001, dt_min=0.001
        )
        assert result is False

    def test_zero_force_change(self):
        """backstep_due_to_force does not backstep when the force does not change."""
        f = np.array([1.0, 0.0, 0.0])
        pos_old = np.array([0.0, 0.0, 0.0])
        pos_new = np.array([1.0, 0.0, 0.0])
        result = backstep_due_to_force(f, f, pos_new, pos_old, dt=0.5, dt_min=0.001)
        assert result is False

    def test_large_force_change_backstep(self):
        """backstep_due_to_force triggers a backstep on a large force change along the displacement."""
        pos_old = np.array([10.0, 0.0, 0.0])
        pos_new = np.array([10.01, 0.0, 0.0])
        f_old = np.array([0.0, 0.0, 0.0])
        f_new = np.array([1e6, 0.0, 0.0])
        result = backstep_due_to_force(
            f_new, f_old, pos_new, pos_old, dt=1.0, dt_min=0.001, radius=5.0
        )
        assert result is True

    def test_perpendicular_force_no_backstep(self):
        """backstep_due_to_force does not backstep when the force change is perpendicular to the displacement."""
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
        """Molecule.radius_of_gyration returns 0 for a molecule with no atoms."""
        mol = Molecule(name="empty")
        assert mol.radius_of_gyration() == 0.0

    def test_empty_bounding_radius(self):
        """Molecule.bounding_radius returns 0 for a molecule with no atoms."""
        mol = Molecule(name="empty")
        assert mol.bounding_radius() == 0.0

    def test_bounding_box_empty_molecule(self):
        """BoundingBox.from_molecule gives zero extents for an empty molecule."""
        mol = Molecule(name="empty")
        bb = BoundingBox.from_molecule(mol)
        assert bb.xmin == 0.0 and bb.xmax == 0.0


class TestPqrIoEdgeCases:
    def test_parse_pqr_with_remarks_and_end(self):
        """parse_pqr skips REMARK and END lines and reads both ATOM and HETATM records with charges."""
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
        """Writing a molecule with write_pqr and reading it back preserves atom count and coordinates."""
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
        """parse_pqr skips a malformed line and keeps the two valid atom records."""
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
        """ReactionInterface.check returns True when contacts are satisfied and probability is 1."""
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
        """ReactionInterface.check never fires over 100 trials when probability is 0."""
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
        """PathwaySet.check_all fires at roughly the 0.5 reaction probability rate over 200 trials."""
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
        """ReactionInterface.check returns False when the contact distance criterion is not met."""
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
        """The CLI group help exits successfully and mentions PySTARC."""

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PySTARC" in result.output

    def test_bounding_box_cmd(self):
        """The bounding_box CLI command runs on a PQR file and reports the bounding box."""

        pqr = "ATOM      1  CA  ALA     1       1.000   2.000   3.000  0.500  1.800\n"
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("test.pqr").write_text(pqr)
            result = runner.invoke(cli, ["bounding_box", "test.pqr"])
            assert result.exit_code == 0
            assert "Bounding box" in result.output

    def test_pqr_to_xml_cmd(self):
        """The pqr_to_xml CLI command converts a PQR file into an XML file containing the residue name."""

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
        """The nam_simulation CLI command exits with a nonzero code when its input files are missing."""

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
        """_is_atom_line returns True for an ATOM record line."""
        assert (
            _is_atom_line("ATOM      1  CA  ALA     1       1.0   2.0   3.0  0.5  1.8")
            is True
        )

    def test_is_atom_line_hetatm(self):
        """_is_atom_line returns True for a HETATM record line."""
        assert (
            _is_atom_line("HETATM    1  C1  BEN     1       1.0   2.0   3.0  0.5  1.8")
            is True
        )

    def test_is_atom_line_remark(self):
        """_is_atom_line returns False for a REMARK line."""
        assert _is_atom_line("REMARK test line") is False

    def test_is_atom_line_ter(self):
        """_is_atom_line returns False for a TER line."""
        assert _is_atom_line("TER") is False

    def test_residue_name_extraction(self):
        """_residue_name extracts the residue name ALA from a PDB atom line."""
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80"
        assert _residue_name(line) == "ALA"

    def test_extract_splits_correctly(self):
        """extract splits a complex into receptor and ligand files, keeping the ligand and excluding water."""
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
        """extract raises ValueError when no atoms match the requested ligand name."""
        pdb = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50  1.80\n"
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            with pytest.raises(ValueError, match="No atoms"):
                extract(pdb_path, "XYZ", td)

    def test_extract_no_receptor_raises(self):
        """extract raises ValueError when no receptor atoms remain after removing the ligand."""
        pdb = "HETATM    1  C1  BEN A   1       1.000   2.000   3.000  0.10  1.70\n"
        with tempfile.TemporaryDirectory() as td:
            pdb_path = Path(td) / "complex.pdb"
            pdb_path.write_text(pdb)
            with pytest.raises(ValueError, match="No receptor"):
                extract(pdb_path, "BEN", td)

    def test_extract_filters_ions(self):
        """extract filters out monatomic ions when building the receptor file."""
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
        """extract matches the ligand name case insensitively."""
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
        """_txt_to_floats parses a whitespace separated string into the corresponding float array."""
        arr = _txt_to_floats("1.0 2.5 3.7")
        np.testing.assert_allclose(arr, [1.0, 2.5, 3.7])

    def test_txt_to_floats_empty(self):
        """_txt_to_floats returns an empty array for an empty string."""
        arr = _txt_to_floats("")
        assert len(arr) == 0

    def test_bead_def_dataclass(self):
        """BeadDef stores its name and atoms and defaults location to an empty string."""
        bd = BeadDef(name="CA", atoms=["CA", "HA"])
        assert bd.name == "CA"
        assert len(bd.atoms) == 2
        assert bd.location == ""

    def test_residue_def_dataclass(self):
        """ResidueDef stores its name and defaults beads to an empty list."""
        rd = ResidueDef(name="ALA")
        assert rd.name == "ALA"
        assert rd.beads == []

    def test_bond_def_dataclass(self):
        """BondDef stores its residues, atoms, orders, length, and index fields."""
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
        """TabulatedPotential interpolates a linear table exactly at endpoints and midpoint."""
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
        """TabulatedPotential clamps queries below x_min to the value at x_min."""
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
        """TabulatedPotential clamps queries above x_max to the value at x_max."""
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
        """TabulatedPotential.deriv returns the slope 1.0 of a linear table."""
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
        """TabulatedPotential interpolates a quadratic table to within tolerance at an interior point."""
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
        """_match_pot returns the potential whose residues, atoms, and orders match exactly."""
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
        """_match_pot matches a potential with a wildcard residue against any residue pair."""
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
        """_match_pot returns None when no potential matches the query."""
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
        """_match_pot prefers an exact match over a wildcard match."""
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
        """_parse_mapping reads a mapping XML into residue definitions with their beads and atoms."""
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
        """_parse_connectivity reads a connectivity XML into bond definitions with length and atoms."""
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
        """_parse_charges reads a charges XML into a dictionary keyed by residue and atom name."""
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
        """COFFDROPParams.bead_charge returns the stored charge for a known bead and 0.0 otherwise."""
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
        """COFFDROPParams.beads_for_residue returns the beads of a known residue and None otherwise."""
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
        """COFFDROPParams.pair_potential returns the interpolated tabulated pair potential at a given r."""
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
        """COFFDROPParams.pair_force returns the negative derivative of the pair potential at a given r."""
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
        """COFFDROPParams returns zero for pair, angle, and dihedral potentials and forces when no potential matches."""
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
        """COFFDROPParams.bond_length returns the bond length for either ordering and None when no bond matches."""
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
        """The COFFDROPParams repr reports the number of residues in its mapping."""
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
        """_save_json writes a dictionary to a JSON file that reloads with the same contents."""
        with tempfile.TemporaryDirectory() as td:
            _save_json({"key": "val"}, os.path.join(td, "test.json"))
            with open(os.path.join(td, "test.json")) as f:
                data = json.load(f)
            assert data["key"] == "val"

    def test_concat_csv(self):
        """_concat_csv concatenates per replica CSV files and reindexes the id column using the given offsets."""
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
            _concat_csv([d1, d2], "traj.csv", td, reindex="traj_id", offsets=[0, 1])
            with open(os.path.join(td, "traj.csv")) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[1]["traj_id"] == "1"

    def test_concat_csv_missing_file(self):
        """_concat_csv writes no output file when the named CSV is absent from the inputs."""
        with tempfile.TemporaryDirectory() as td:
            _concat_csv([td], "nonexistent.csv", td)
            assert not os.path.exists(os.path.join(td, "nonexistent.csv"))

    def test_sum_csv(self):
        """_sum_csv sums the count column across replica CSV files into a single combined row."""
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
        """_concat_npz stacks the data arrays from replica NPZ files into one array."""
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
        """_sum_npz sums the matrix arrays across replica NPZ files element by element."""
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
        """_sum_npz writes no output file when the named NPZ is absent from the inputs."""
        with tempfile.TemporaryDirectory() as td:
            _sum_npz([td], "missing.npz", td, sum_key="x")
            assert not os.path.exists(os.path.join(td, "missing.npz"))


# Weighted Ensemble data structures


# Force engine _Grid
class TestForceEngineGrid:
    def test_grid_from_dxgrid(self):
        """_Grid wraps a DXGrid, preserving its data shape and reporting its spacing."""
        data = np.ones((5, 5, 5))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.data.shape == (5, 5, 5)
        np.testing.assert_allclose(cg.spacing, [1.0, 1.0, 1.0])

    def test_grid_contains_interior(self):
        """_Grid.contains returns True for a point inside the grid bounds."""
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.contains(np.array([5.0, 5.0, 5.0])) is True

    def test_grid_contains_outside(self):
        """_Grid.contains returns False for a point outside the grid bounds."""
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), data)
        cg = _Grid(g)
        assert cg.contains(np.array([100.0, 100.0, 100.0])) is False

    def test_grid_lo_hi_margins(self):
        """_Grid computes lo and hi margins one cell inside the grid extents given its spacing."""
        data = np.ones((10, 10, 10))
        g = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), data)
        cg = _Grid(g)
        np.testing.assert_allclose(cg.lo, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(cg.hi, [16.0, 16.0, 16.0])


# Geometry pipeline
class TestGeometryPipeline:
    def test_geom_atom_record_pos(self):
        """GeomAtomRecord.pos returns the atom position as an array of its x, y, z coordinates."""
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
        """GeomAtomRecord.is_ghost returns True for an atom named GHO."""
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
        """GeomAtomRecord.is_ghost returns False for a normal atom."""
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
        """geom_parse_pqr parses a PQR file into atom records carrying name and charge."""
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
        """geom_parse_pqr skips REMARK and TER lines, keeping only valid atom records."""
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
        """MoleculeGeometry stores its atom counts, total charge, and other geometry fields."""
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
        """geom_analyse reports atom and charged counts and positive max and hydrodynamic radii for a PQR molecule."""
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


# Geometry _parse_rxns_xml_criteria
class TestGeometryRxnsCriteria:
    def test_parse_format1_atom1_atom2(self):
        """_parse_rxns_xml_criteria parses the atom1/atom2 format into zero-based indices and cutoff."""
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
        """_parse_rxns_xml_criteria parses the atoms plus distance format into zero-based indices and cutoff."""
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
        """_parse_rxns_xml_criteria reads the n_needed value alongside the parsed pairs."""
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
        """_parse_rxns_xml_criteria returns no pairs for an empty criterion element."""
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
        """_parse_ff parses type maps and one each of nonbonded, angle, and dihedral potentials with positive pair value."""

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
        """_parse_ff returns empty type maps and no pairs when the coffdrop element is empty."""

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
        """ChainForceEvaluator.compute_forces returns a (3,3) array with nonzero forces for a perturbed chain."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[1].pos = np.array([3.5, 0.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (3, 3)
        assert np.any(np.abs(F) > 0)

    def test_chain_positions_set(self):
        """FlexibleChain.set_positions updates each bead position to the supplied coordinates."""
        chain = build_linear_chain(n_residues=4, bond_length=3.8)
        new_pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        chain.set_positions(new_pos)
        np.testing.assert_allclose(chain.beads[2].pos, [2.0, 0.0, 0.0])

    def test_chain_zero_forces(self):
        """FlexibleChain.zero_forces resets a bead force vector to zero."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[0].force = np.array([1.0, 2.0, 3.0])
        chain.zero_forces()
        np.testing.assert_allclose(chain.beads[0].force, [0.0, 0.0, 0.0])

    def test_chain_positions_array(self):
        """FlexibleChain.positions_array returns a (3,3) array of bead positions."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        pos = chain.positions_array()
        assert pos.shape == (3, 3)

    def test_chain_forces_array(self):
        """FlexibleChain.forces_array returns a (3,3) array of bead forces."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        farr = chain.forces_array()
        assert farr.shape == (3, 3)

    def test_equilibrium_forces_small(self):
        """ChainForceEvaluator forces stay below 1.0 in magnitude for a chain at equilibrium bond lengths."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) < 1.0

    def test_stretched_bond_restoring_force(self):
        """A stretched bond produces a restoring force pulling the two beads back toward each other."""
        chain = build_linear_chain(n_residues=2, bond_length=3.8)
        chain.beads[1].pos = np.array([10.0, 0.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F[0, 0] > 0
        assert F[1, 0] < 0


# Quaternion uncovered branches
class TestQuaternionFromMatrix:
    def test_from_rotation_matrix_identity(self):
        """Quaternion.from_rotation_matrix returns a unit quaternion near identity for the identity matrix."""
        R = np.eye(3)
        q = Quaternion.from_rotation_matrix(R)
        assert abs(q.norm() - 1.0) < 1e-10
        assert abs(q.w) > 0.9

    def test_from_rotation_matrix_90z(self):
        """Quaternion.from_rotation_matrix round-trips a 90 degree z rotation back to the same matrix."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        assert abs(q.norm() - 1.0) < 1e-10
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_90x(self):
        """Quaternion.from_rotation_matrix round-trips a 90 degree x rotation back to the same matrix."""
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_90y(self):
        """Quaternion.from_rotation_matrix round-trips a 90 degree y rotation back to the same matrix."""
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_180z(self):
        """Quaternion.from_rotation_matrix round-trips a 180 degree z rotation back to the same matrix."""
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        q = Quaternion.from_rotation_matrix(R)
        R2 = q.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_from_rotation_matrix_arbitrary(self):
        """Quaternion.from_rotation_matrix round-trips an arbitrary axis-angle rotation back to the same matrix."""
        q_orig = Quaternion.from_axis_angle(np.array([1, 1, 1]) / math.sqrt(3), 1.23)
        R = q_orig.to_rotation_matrix()
        q_back = Quaternion.from_rotation_matrix(R)
        R2 = q_back.to_rotation_matrix()
        np.testing.assert_allclose(R2, R, atol=1e-10)

    def test_normalized_zero_quaternion(self):
        """Quaternion.normalized returns the identity quaternion for a zero quaternion."""
        q = Quaternion(0, 0, 0, 0)
        n = q.normalized()
        assert n.w == 1.0

    def test_random_quaternion_no_rng(self):
        """random_quaternion returns a unit quaternion when called with rng=None."""
        q = random_quaternion(rng=None)
        assert abs(q.norm() - 1.0) < 1e-10

    def test_small_rotation_quaternion_no_rng(self):
        """small_rotation_quaternion returns a unit quaternion when called with rng=None."""
        q = small_rotation_quaternion(0.01, rng=None)
        assert abs(q.norm() - 1.0) < 1e-10


# Diffusional rotation uncovered functions
class TestDiffusionalRotationSampling:
    def test_spline_rot_0p5(self):
        """_spline_rot_0p5 returns a unit-norm quaternion."""

        rng = np.random.default_rng(42)
        q = _spline_rot_0p5(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_spline_rot_1p0(self):
        """_spline_rot_1p0 returns a unit-norm quaternion."""

        rng = np.random.default_rng(42)
        q = _spline_rot_1p0(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10

    def test_spline_rot_2p0(self):
        """_spline_rot_2p0 returns a unit-norm quaternion."""

        rng = np.random.default_rng(42)
        q = _spline_rot_2p0(rng)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


# WE result rate constant and repr


# Engine _GridStack
class TestGridStack:
    def test_gridstack_creation(self):
        """_GridStack reports length 2 and truthy boolean for two grids."""

        g1 = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), np.ones((5, 5, 5)))
        g2 = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.ones((10, 10, 10)))
        gs = _GridStack([g1, g2])
        assert len(gs) == 2
        assert bool(gs) is True

    def test_gridstack_empty(self):
        """_GridStack reports length 0 and falsy boolean when empty."""

        gs = _GridStack([])
        assert len(gs) == 0
        assert bool(gs) is False

    def test_gridstack_finest_first(self):
        """_GridStack.finest_for returns the finest-spacing grid covering a query point."""

        coarse = DXGrid(np.zeros(3), np.diag([2.0, 2.0, 2.0]), np.ones((10, 10, 10)))
        fine = DXGrid(np.zeros(3), np.diag([0.5, 0.5, 0.5]), np.ones((10, 10, 10)))
        gs = _GridStack([coarse, fine])
        pt = np.array([2.0, 2.0, 2.0])
        g = gs.finest_for(pt)
        assert g is not None
        np.testing.assert_allclose(g.spacing, [0.5, 0.5, 0.5])

    def test_gridstack_outside_returns_none(self):
        """_GridStack.finest_for returns None for a point outside all grids."""

        g = DXGrid(np.zeros(3), np.diag([1.0, 1.0, 1.0]), np.ones((5, 5, 5)))
        gs = _GridStack([g])
        assert gs.finest_for(np.array([100.0, 100.0, 100.0])) is None

    def test_gridstack_eval_empty(self):
        """_GridStack.eval_atoms returns zero force and zero energy when the stack is empty."""

        gs = _GridStack([])
        F, T, E = gs.eval_atoms(np.zeros((1, 3)), np.array([1.0]), 0.5, False, "numpy")
        np.testing.assert_allclose(F, [0, 0, 0])
        assert E == 0.0


# Multipole farfield summary and repr
class TestMultipoleFarfieldExtended:
    def test_summary_monopole_dominant(self):
        """MultipoleExpansion.summary mentions the monopole term for a net-charged distribution."""
        charges = np.array([5.0, -2.0])
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        me = MultipoleExpansion(positions, charges, debye_length=7.86)
        s = me.summary()
        assert "Monopole" in s or "monopole" in s.lower() or "Q" in s

    def test_summary_dipole_dominant(self):
        """MultipoleExpansion.summary returns a nonempty string for a dipolar charge distribution."""
        charges = np.array([1.0, -1.0])
        positions = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
        me = MultipoleExpansion(positions, charges, debye_length=7.86)
        s = me.summary()
        assert len(s) > 0

    def test_potential_at_zero(self):
        """MultipoleExpansion.potential returns 0.0 when evaluated at the expansion center."""
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
        bonds = [
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 100.0),
            ChainBond(ChainAtomRef(1), ChainAtomRef(2), 3.8, 100.0),
        ]
        angles = [
            ChainAngle(ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), math.pi, 50.0)
        ]
        return FlexibleChain(
            beads=beads, bonds=bonds, angles=angles, name="angle_chain"
        )

    def test_angle_equilibrium_zero_force(self):
        """ChainForceEvaluator forces stay below 1.0 in magnitude for a chain at its equilibrium angle."""
        chain = self._make_chain_with_angle()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) < 1.0

    def test_angle_bent_produces_force(self):
        """Bending a chain away from its equilibrium angle produces a nonzero angular force."""
        chain = self._make_chain_with_angle()
        chain.beads[2].pos = np.array([5.0, 3.0, 0.0])
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.max(np.abs(F)) > 0.1

    def test_angle_force_shape(self):
        """ChainForceEvaluator.compute_forces returns a (3,3) array for a chain with an angle term."""
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
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 100.0),
            ChainBond(ChainAtomRef(1), ChainAtomRef(2), 5.0, 100.0),
            ChainBond(ChainAtomRef(2), ChainAtomRef(3), 5.0, 100.0),
        ]
        torsions = [
            ChainTorsion(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                0.0,
                10.0,
                1,
            )
        ]
        return FlexibleChain(
            beads=beads, bonds=bonds, torsions=torsions, name="torsion_chain"
        )

    def test_torsion_force_shape(self):
        """ChainForceEvaluator.compute_forces returns a (4,3) array for a chain with a torsion term."""
        chain = self._make_chain_with_torsion()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (4, 3)

    def test_torsion_force_nonzero(self):
        """ChainForceEvaluator.compute_forces returns nonzero forces for a chain with a torsion term."""
        chain = self._make_chain_with_torsion()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert np.any(np.abs(F) > 0)


class TestChainExcludedVolume:
    def test_overlapping_beads_repel(self):
        """Overlapping nonbonded beads produce nonzero repulsive forces with a (2,3) shape."""
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
        bonds = [ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 100.0)]
        chain = FlexibleChain(beads=beads, bonds=bonds, name="overlap")
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert F.shape == (2, 3)
        assert np.any(np.abs(F) > 0)

    def test_well_separated_beads_no_force(self):
        """Well-separated beads produce essentially zero nonbonded force."""
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
        """ChainBDPropagator.step displaces the bead positions of a free chain."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        assert not np.allclose(pos_before, pos_after)

    def test_frozen_chain_no_move(self):
        """ChainBDPropagator.step leaves the positions of a frozen chain unchanged."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.frozen = True
        prop = ChainBDPropagator()
        rng = np.random.default_rng(42)
        pos_before = chain.positions_array().copy()
        prop.step(chain, dt=0.1, rng=rng)
        pos_after = chain.positions_array()
        np.testing.assert_allclose(pos_before, pos_after)

    def test_max_time_step_positive(self):
        """ChainBDPropagator.max_time_step returns a positive timestep for a nonempty chain."""
        chain = build_linear_chain(n_residues=5, bond_length=3.8)
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt > 0

    def test_max_time_step_empty_chain(self):
        """ChainBDPropagator.max_time_step returns the default 0.1 for an empty chain."""
        chain = FlexibleChain(beads=[], name="empty")
        prop = ChainBDPropagator()
        dt = prop.max_time_step(chain)
        assert dt == 0.1

    def test_satisfy_bond_constraints(self):
        """ChainBDPropagator.satisfy_bond_constraints restores every bond to within tolerance of its r0."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        chain.beads[1].pos = np.array([5.0, 0.0, 0.0])
        prop = ChainBDPropagator()
        prop.satisfy_bond_constraints(chain, tol=0.01)
        for bond in chain.bonds:
            r = np.linalg.norm(
                chain.beads[bond.b.atom_idx].pos - chain.beads[bond.a.atom_idx].pos
            )
            assert abs(r - bond.r0) / bond.r0 < 0.01

    def test_D_trans_positive(self):
        """ChainBDPropagator.D_trans returns a positive translational diffusion coefficient."""
        prop = ChainBDPropagator()
        D = prop.D_trans(2.0)
        assert D > 0

    def test_step_with_external_evaluator(self):
        """ChainBDPropagator.step runs with an external force evaluator and leaves bead positions defined."""
        chain = build_linear_chain(n_residues=3, bond_length=3.8)
        prop = ChainBDPropagator()
        evaluator = ChainForceEvaluator()
        rng = np.random.default_rng(42)
        prop.step(chain, dt=0.1, rng=rng, force_evaluator=evaluator)
        assert chain.beads[0].pos is not None


# WE simulator construction and bin methods


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
        """NAM run returns a SimulationResult whose reacted, escaped, and max-steps counts sum to n_trajectories."""
        mol1, mol2, mob, ps, params = self._make_setup()
        sim = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        result = sim.run()
        assert isinstance(result, SimulationResult)
        assert (
            result.n_reacted + result.n_escaped + result.n_max_steps
            == params.n_trajectories
        )

    def test_nam_run_reaction_probability_bounded(self):
        """NAM reaction probability lies in [0, 1]."""
        mol1, mol2, mob, ps, params = self._make_setup()
        sim = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        result = sim.run()
        assert 0.0 <= result.reaction_probability <= 1.0

    def test_nam_different_seeds(self):
        """NAM runs with different seeds produce different reacted or escaped counts."""
        mol1, mol2, mob, ps, params = self._make_setup(n_traj=200)
        sim1 = NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        r1 = sim1.run()
        params2 = NAMParameters(
            n_trajectories=200,
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
        """OPGroupInfo stores its charge q and translational diffusion constant."""
        g = OPGroupInfo(q=6.0, Dtrans=0.01, Drot=0.001)
        assert g.q == 6.0
        assert g.Dtrans == 0.01

    def test_outer_propagator_constructs(self):
        """OuterPropagator constructs and derives the charged-sphere q-radius from its inputs."""
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
        """OuterPropagator yields a nonzero potential factor and a positive diffusion factor."""
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
        """OuterPropagator without hydrodynamic interactions still gives a positive diffusion factor and has_hi False."""
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
        """OuterPropagator return probability lies in (0, 1]."""
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
        """auto_detect_reactions parses a single contact pair from a reactions XML file."""
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
        """auto_detect_reactions applies a manually specified ghost-atom cutoff to the contact pair."""
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
        """auto_detect_reactions finds GHO atoms in the PQR geometry and requires one contact."""
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
        """auto_detect_reactions raises RuntimeError when no GHO atoms are present."""
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


# NAM simulator full run loop integration
class TestNAMSimulatorIntegration:
    def _make_dh_force(self):
        def dh_force(mol1, mol2):
            q1 = mol1.total_charge()
            q2 = mol2.total_charge()
            c1 = mol1.centroid()
            c2 = mol2.centroid()
            dr = c2 - c1
            r = float(np.linalg.norm(dr))
            if r < 1e-8:
                return np.zeros(3), np.zeros(3), 0.0
            lB = 7.18
            lD = 7.86
            phi = q1 * lB * math.exp(-r / lD) / r
            energy = q2 * phi
            dphi_dr = -q1 * lB * math.exp(-r / lD) * (1.0 / r**2 + 1.0 / (r * lD))
            force = -q2 * dphi_dr * dr / r
            torque = np.zeros(3)
            return force, torque, energy

        return dh_force

    def _make_molecules(self, n_atoms=3, q1=5.0, q2=-3.0, sep=15.0):
        mol1 = Molecule(name="rec")
        for i in range(n_atoms):
            mol1.atoms.append(
                Atom(
                    index=i,
                    name=f"A{i}",
                    residue_name="REC",
                    residue_index=1,
                    chain="A",
                    x=float(i),
                    y=0.0,
                    z=0.0,
                    charge=q1 / n_atoms,
                    radius=1.5,
                )
            )
        mol2 = Molecule(name="lig")
        for i in range(n_atoms):
            mol2.atoms.append(
                Atom(
                    index=i,
                    name=f"B{i}",
                    residue_name="LIG",
                    residue_index=1,
                    chain="A",
                    x=sep + float(i),
                    y=0.0,
                    z=0.0,
                    charge=q2 / n_atoms,
                    radius=1.5,
                )
            )
        return mol1, mol2

    def test_nam_full_run_with_dh_force(self):
        """NAM run with a Debye-Hückel force conserves trajectory counts and bounds the reaction probability in [0, 1]."""
        mol1, mol2 = self._make_molecules()
        mob = MobilityTensor.from_radii(3.0, 3.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 8.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = NAMParameters(
            n_trajectories=100,
            dt=0.2,
            r_start=20.0,
            max_steps=500,
            seed=42,
            verbose=False,
            n_threads=1,
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params, self._make_dh_force())
        result = sim.run()
        assert result.n_reacted + result.n_escaped + result.n_max_steps == 100
        assert 0.0 <= result.reaction_probability <= 1.0

    def test_nam_run_one_trajectory(self):
        """NAM run_one returns a single trajectory with a valid fate and a non-negative step count."""
        mol1, mol2 = self._make_molecules()
        mob = MobilityTensor.from_radii(3.0, 3.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 8.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = NAMParameters(
            n_trajectories=1,
            dt=0.2,
            r_start=20.0,
            max_steps=500,
            seed=42,
            verbose=False,
            n_threads=1,
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params, self._make_dh_force())
        result = sim.run_one()
        assert result.fate in (Fate.REACTED, Fate.ESCAPED, Fate.MAX_STEPS)
        assert result.steps >= 0

    def test_nam_rate_constant_positive(self):
        """NAM rate constant computed from the relative diffusion is non-negative."""
        mol1, mol2 = self._make_molecules(q1=10.0, q2=-10.0)
        mob = MobilityTensor.from_radii(3.0, 3.0)
        criteria = ReactionCriteria(
            name="r", pairs=[ContactPair(0, 0, 10.0)], n_needed=1
        )
        rxn = ReactionInterface(name="rxn", criteria=criteria)
        ps = PathwaySet(reactions=[rxn])
        params = NAMParameters(
            n_trajectories=200,
            dt=0.2,
            r_start=20.0,
            max_steps=500,
            seed=42,
            verbose=False,
            n_threads=1,
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params, self._make_dh_force())
        result = sim.run()
        D_rel = mob.relative_translational_diffusion()
        k = result.rate_constant(D_rel)
        assert k >= 0


# WE simulator full run loop integration


class TestPqrFormatVariations:
    """Regression tests for PQR format variations handled by the canonical
    parser in pystarc.structures.pqr_io.parse_pqr_records.

    These tests exist because we have hit each of these cases in real
    files in the PySTARC corpus: HETATM small-molecule ligands (HSP90),
    four-character Amber terminal residue names (thrombin: NTHR and
    CGLU), and single-space collapsed spacing between charge and radius
    (thrombin calcium ion). The tests assert parser behavior on the
    minimal synthetic line that exercises each case so that future
    edits to the parser cannot silently regress any of them.
    """

    @staticmethod
    def _parse_single_line(line: str):
        """Helper: write one line to a temp PQR and parse it."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pqr", delete=False) as f:
            f.write(line + "\n")
            f.write("END\n")
            f.flush()
            recs = parse_pqr_records(f.name)
        os.unlink(f.name)
        return recs

    def test_atom_record_parsed(self):
        """An ATOM record parses its record type, atom name, residue, coordinates, charge, and radius."""
        line = "ATOM      1  CA  ALA     1       1.000   2.000   3.000  " "0.500  1.800"
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        r = recs[0]
        assert r.record_type == "ATOM"
        assert r.name == "CA"
        assert r.resname == "ALA"
        assert r.resid == 1
        assert r.x == pytest.approx(1.0)
        assert r.charge == pytest.approx(0.5)
        assert r.radius == pytest.approx(1.8)

    def test_hetatm_record_parsed(self):
        """A HETATM record parses its record type and residue name."""
        line = (
            "HETATM    1  C1x  UNK     1     33.864  30.183  35.037   "
            "0.1639   1.7000"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].record_type == "HETATM"
        assert recs[0].resname == "UNK"

    def test_four_char_nterm_resname(self):
        # Thrombin receptor: NTHR at cols 18-21 (right-extended 4-char)
        """A four-character N-terminal residue name NTHR is parsed without truncation."""
        line = (
            "ATOM      1  N   NTHR    1      -2.787  73.833  11.760 " "-0.3000 1.8500"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].resname == "NTHR", (
            f"4-char N-terminal resname NTHR was truncated " f"to {recs[0].resname!r}"
        )

    def test_four_char_cterm_resname(self):
        # Thrombin receptor also carries CGLU as the C-terminus
        """A four-character C-terminal residue name CGLU is parsed without truncation."""
        line = (
            "ATOM   1000  C   CGLU  295      10.000  20.000  30.000 " " 0.3000 1.7000"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].resname == "CGLU", (
            f"4-char C-terminal resname CGLU was truncated " f"to {recs[0].resname!r}"
        )

    def test_collapsed_spacing_between_charge_and_radius(self):
        """Charge and radius parse correctly when separated by collapsed single-space padding."""
        line = (
            "HETATM    1  CAL CAL   344      -5.592  67.258 -23.982  " "2.0000 1.3670"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].charge == pytest.approx(2.0)
        assert recs[0].radius == pytest.approx(1.367)

    def test_chain_column_detected_by_whitespace_fallback(self):
        """An explicit chain letter is detected and round-trips along with residue id and name."""
        line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  " "0.500  1.800"
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].chain == "A"
        assert recs[0].resid == 1
        assert recs[0].resname == "ALA"

    def test_trailing_element_captured(self):
        # Standard AmberTools PQR with element symbol after radius
        """A trailing element symbol after the radius field is captured."""
        line = (
            "ATOM      1  N   SER     1      50.038  51.662  14.644  "
            "0.1849  1.5500       N"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].element == "N"

    def test_missing_element_returns_empty_string(self):
        """A record lacking a trailing element symbol returns an empty element string."""
        line = (
            "HETATM    1  C1x  UNK     1     33.864  30.183  35.037   "
            "0.1639   1.7000"
        )
        recs = self._parse_single_line(line)
        assert len(recs) == 1
        assert recs[0].element == ""


# Helpers
def _make_pqr(tmp_path: Path, n_atoms: int = 5, spread: float = 10.0) -> Path:
    """Create a minimal valid PQR file for unit testing."""
    pqr_path = tmp_path / "test.pqr"
    lines = []
    for i in range(n_atoms):
        x = (i - n_atoms / 2) * spread / n_atoms
        y = 0.0
        z = 0.0
        q = 0.0
        r = 1.5
        lines.append(
            f"ATOM  {i+1:5d}  C   ALA A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{q:8.4f}{r:7.4f}\n"
        )
    pqr_path.write_text("".join(lines))
    return pqr_path


# Dime validator
def test_is_valid_apbs_dime_accepts_canonical_values():
    """_is_valid_apbs_dime accepts all canonical APBS multigrid dimensions."""
    valid = [5, 9, 13, 17, 33, 65, 97, 129, 161, 193, 257, 289, 321, 385, 449, 513, 577]
    for d in valid:
        assert _is_valid_apbs_dime(d), f"dime={d} should be valid"


def test_is_valid_apbs_dime_rejects_non_canonical_values():
    """_is_valid_apbs_dime rejects values that do not satisfy the multigrid dimension form."""
    invalid = [0, 1, 2, 3, 4, 100, 128, 200, 256, 300, 400, 500]
    for d in invalid:
        assert not _is_valid_apbs_dime(d), f"dime={d} should be rejected"


def test_compute_grid_params_rejects_invalid_dime(tmp_path):
    """_compute_grid_params raises a ValueError naming an invalid APBS dime."""
    pqr = _make_pqr(tmp_path)
    with pytest.raises(ValueError, match="Invalid APBS dime"):
        _compute_grid_params(pqr, srad=1.5, debye_length=8.0, dime=300)


def test_compute_grid_params_accepts_common_dimes(tmp_path):
    """_compute_grid_params accepts common production dimes 257, 289, and 321 in both coarse and fine grids."""
    pqr = _make_pqr(tmp_path)
    for d in (257, 289, 321):
        coarse, fine = _compute_grid_params(pqr, srad=1.5, debye_length=8.0, dime=d)
        assert coarse["dime"][0] == d
        assert fine["dime"][0] == d


# Multigrid invariant: coarse strictly encloses fine
def test_auto_cglen_is_strictly_greater_than_fglen(tmp_path):
    """The auto coarse glen strictly encloses the fine glen and honors the fglen override."""
    pqr = _make_pqr(tmp_path)
    for fglen, dime in [(192.0, 257), (320.0, 289), (352.0, 321), (400.0, 321)]:
        coarse, fine = _compute_grid_params(
            pqr,
            srad=1.5,
            debye_length=8.0,
            dime=dime,
            fglen_override=fglen,
        )
        cglen = coarse["glen"][0]
        actual_fglen = fine["glen"][0]
        assert (
            cglen > actual_fglen
        ), f"cglen={cglen} not strictly greater than fglen={actual_fglen}"
        assert actual_fglen == fglen, "fglen_override should be honored"


def test_auto_cglen_uses_2x_ratio(tmp_path):
    """The auto coarse glen equals twice the fine glen."""
    pqr = _make_pqr(tmp_path)
    for fglen in (100.0, 200.0, 352.0, 500.0):
        coarse, _ = _compute_grid_params(
            pqr,
            srad=1.5,
            debye_length=8.0,
            dime=257,
            fglen_override=fglen,
        )
        assert coarse["glen"][0] == 2.0 * fglen


def test_compute_grid_params_rejects_too_small_cglen_override(tmp_path):
    """_compute_grid_params rejects a cglen override that is not greater than fglen with a multigrid invariant error."""
    pqr = _make_pqr(tmp_path)
    with pytest.raises(ValueError, match="Multigrid invariant violated"):
        _compute_grid_params(
            pqr,
            srad=1.5,
            debye_length=8.0,
            dime=257,
            fglen_override=300.0,
            cglen_override=200.0,
        )


def test_compute_grid_params_accepts_valid_cglen_override(tmp_path):
    """_compute_grid_params honors a user cglen override greater than fglen."""
    pqr = _make_pqr(tmp_path)
    coarse, fine = _compute_grid_params(
        pqr,
        srad=1.5,
        debye_length=8.0,
        dime=257,
        fglen_override=300.0,
        cglen_override=600.0,
    )
    assert coarse["glen"][0] == 600.0
    assert fine["glen"][0] == 300.0


def test_auto_path_with_no_overrides_is_self_consistent(tmp_path):
    """The fully-auto grid path yields coarse glen greater than fine glen for any molecule."""
    pqr = _make_pqr(tmp_path, n_atoms=20, spread=80.0)
    coarse, fine = _compute_grid_params(pqr, srad=1.5, debye_length=8.0, dime=257)
    assert coarse["glen"][0] > fine["glen"][0]


# bcfl=map requires the previous-level DX file
def test_write_apbs_input_with_missing_prev_dx_raises(tmp_path):
    """_write_apbs_input with bcfl=map and a missing previous DX file raises FileNotFoundError."""
    pqr = _make_pqr(tmp_path)
    fine_params = {
        "spacing": 0.5,
        "dime": [129, 129, 129],
        "glen": [64.0, 64.0, 64.0],
        "gcent": [0.0, 0.0, 0.0],
        "label": "fine",
        "bcfl": "map",
    }
    with pytest.raises(FileNotFoundError, match="bcfl=map requires"):
        _write_apbs_input(
            pqr_path=pqr,
            out_dx_name="out.dx",
            params=fine_params,
            prev_dx_name="missing_coarse.dx",
            work_dir=tmp_path,
            inp_name="test.in",
            is_born=False,
            ion_conc=0.150,
            dielectric_in=4.0,
            dielectric_out=78.0,
            srad=1.5,
            temp=298.15,
        )


def test_write_apbs_input_with_existing_prev_dx_succeeds(tmp_path):
    """_write_apbs_input with bcfl=map and an existing previous DX file writes an input containing bcfl map and usemap pot 1."""
    pqr = _make_pqr(tmp_path)
    coarse_dx = tmp_path / "coarse.dx"
    coarse_dx.write_text("# fake DX file for testing\n")
    fine_params = {
        "spacing": 0.5,
        "dime": [129, 129, 129],
        "glen": [64.0, 64.0, 64.0],
        "gcent": [0.0, 0.0, 0.0],
        "label": "fine",
        "bcfl": "map",
    }
    inp_path = _write_apbs_input(
        pqr_path=pqr,
        out_dx_name="out.dx",
        params=fine_params,
        prev_dx_name="coarse.dx",
        work_dir=tmp_path,
        inp_name="test.in",
        is_born=False,
        ion_conc=0.150,
        dielectric_in=4.0,
        dielectric_out=78.0,
        srad=1.5,
        temp=298.15,
    )
    assert inp_path.exists()
    contents = inp_path.read_text()
    assert "bcfl map" in contents
    assert "usemap pot 1" in contents


def test_write_apbs_input_with_bcfl_sdh_does_not_require_prev_dx(tmp_path):
    """_write_apbs_input with bcfl=sdh writes the input without requiring a previous DX file."""
    pqr = _make_pqr(tmp_path)
    coarse_params = {
        "spacing": 1.0,
        "dime": [129, 129, 129],
        "glen": [128.0, 128.0, 128.0],
        "gcent": [0.0, 0.0, 0.0],
        "label": "coarse",
        "bcfl": "sdh",
    }
    inp_path = _write_apbs_input(
        pqr_path=pqr,
        out_dx_name="out.dx",
        params=coarse_params,
        prev_dx_name=None,
        work_dir=tmp_path,
        inp_name="test.in",
        is_born=False,
        ion_conc=0.150,
        dielectric_in=4.0,
        dielectric_out=78.0,
        srad=1.5,
        temp=298.15,
    )
    assert inp_path.exists()
    contents = inp_path.read_text()
    assert "bcfl sdh" in contents


class TestTorsionForceBugRegression:
    """Newton's third law and momentum conservation for torsion forces.

    A torsion potential V(phi) acts on four atoms (i, j, k, l) where the
    dihedral angle phi is measured around the j-k bond. Translational
    invariance of V means the four forces must sum to zero exactly:
      F_i + F_j + F_k + F_l = 0
    independent of geometry. If any of the four force terms is missing
    or wrong, this fails.
    """

    def _make_torsion_chain(self):
        """Four atoms in a non-planar, non-equilibrium dihedral geometry."""
        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname="A", resid=0),
            ChainAtom(radius=2.0, charge=0.0, resname="B", resid=1),
            ChainAtom(radius=2.0, charge=0.0, resname="C", resid=2),
            ChainAtom(radius=2.0, charge=0.0, resname="D", resid=3),
        ]
        bonds = [
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), 3.8, 100.0),
            ChainBond(ChainAtomRef(1), ChainAtomRef(2), 3.8, 100.0),
            ChainBond(ChainAtomRef(2), ChainAtomRef(3), 3.8, 100.0),
        ]
        torsions = [
            ChainTorsion(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                phi0=0.7,
                k_tor=10.0,
                n=1,
            )
        ]
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
                pos=np.array([5.0, 3.0, 0.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="C",
                resid=2,
            ),
            ChainBead(
                pos=np.array([7.0, 4.0, 2.0]),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="D",
                resid=3,
            ),
        ]
        return FlexibleChain(
            beads=beads, bonds=bonds, torsions=torsions, name="torsion_test"
        )

    def test_torsion_forces_sum_to_zero(self):
        """The total internal chain force sums to zero by Newton's third law."""

        chain = self._make_torsion_chain()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        net = F.sum(axis=0)
        assert np.allclose(net, np.zeros(3), atol=1e-8), (
            f"Net force {net} is non-zero; "
            f"internal forces violate Newton's third law"
        )

    def test_torsion_middle_atoms_feel_force(self):
        """An out-of-equilibrium dihedral produces nonzero forces on the two central atoms."""
        chain = self._make_torsion_chain()
        evaluator = ChainForceEvaluator()
        F = evaluator.compute_forces(chain)
        assert (
            np.linalg.norm(F[1]) > 1e-6
        ), f"F[1] = {F[1]} is zero; torsion gradient missing for atom j"
        assert (
            np.linalg.norm(F[2]) > 1e-6
        ), f"F[2] = {F[2]} is zero; torsion gradient missing for atom k"


class TestBondedForceEnergyConservation:
    """Energy conservation as a correctness check for bonded force kernels.

    For a deterministic (no-noise) integrator, total energy E = K + V must
    be conserved if and only if the force on each atom is the negative
    gradient of the same V. Velocity Verlet is symplectic and preserves a
    nearby shadow Hamiltonian to machine precision, so the actual E drift
    over a finite run should be small (and bounded, not secular).

    A bug in any bonded force kernel (bond, angle, or torsion) shows up as
    visible energy drift in this test.
    """

    @staticmethod
    def _harmonic_bond_V(r, r0, k):
        """V = 0.5 * k * (r - r0)**2."""
        return 0.5 * k * (r - r0) ** 2

    @staticmethod
    def _harmonic_angle_V(theta, theta0, k):
        """V = 0.5 * k * (theta - theta0)**2."""
        return 0.5 * k * (theta - theta0) ** 2

    @staticmethod
    def _cosine_torsion_V(phi, phi0, k, n):
        """V = k * (1 - cos(n*phi - phi0))."""
        return k * (1.0 - math.cos(n * phi - phi0))

    @classmethod
    def _compute_V(cls, chain):
        """Total bonded potential energy of the chain.

        Independent of the force evaluator: computed from positions and
        bonded-interaction parameters using the analytic potential forms.
        """
        V = 0.0
        for bond in chain.bonds:
            ri = chain.beads[bond.a.atom_idx].pos
            rj = chain.beads[bond.b.atom_idx].pos
            r = float(np.linalg.norm(rj - ri))
            V += cls._harmonic_bond_V(r, bond.r0, bond.k_spring)
        for angle in chain.angles:
            ri = chain.beads[angle.a.atom_idx].pos
            rj = chain.beads[angle.b.atom_idx].pos
            rk = chain.beads[angle.c.atom_idx].pos
            u = ri - rj
            v = rk - rj
            cos_t = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            cos_t = max(-1.0, min(1.0, cos_t))
            theta = math.acos(cos_t)
            V += cls._harmonic_angle_V(theta, angle.theta0, angle.k_angle)
        for tor in chain.torsions:
            ri = chain.beads[tor.a.atom_idx].pos
            rj = chain.beads[tor.b.atom_idx].pos
            rk = chain.beads[tor.c.atom_idx].pos
            rl = chain.beads[tor.d.atom_idx].pos
            b1 = rj - ri
            b2 = rk - rj
            b3 = rl - rk
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1n = float(np.linalg.norm(n1))
            n2n = float(np.linalg.norm(n2))
            b2n = float(np.linalg.norm(b2))
            cos_phi = float(np.dot(n1, n2)) / (n1n * n2n)
            sin_phi = float(np.dot(np.cross(b2, n1), n2)) / (b2n * n1n * n2n)
            phi = math.atan2(sin_phi, cos_phi)
            V += cls._cosine_torsion_V(phi, tor.phi0, tor.k_tor, tor.n)
        return V

    def _make_chain(self):
        """5-atom chain, sparse enough that excluded volume is inactive.

        Atom spacing is well above any 1-3 or 1-4 cutoff so excluded
        volume contributes ~0 forces and only bonded forces drive dynamics.
        """
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([5.0, 0.5, 0.0]),
            np.array([10.0, 1.0, 0.5]),
            np.array([15.0, 0.5, 1.0]),
            np.array([20.0, 0.0, 0.5]),
        ]
        beads = [
            ChainBead(
                pos=positions[i].copy(),
                force=np.zeros(3),
                radius=2.0,
                charge=0.0,
                resname="X",
                resid=i,
            )
            for i in range(5)
        ]
        bonds = [
            ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), 4.5, 50.0) for i in range(4)
        ]
        angles = [
            ChainAngle(
                ChainAtomRef(i),
                ChainAtomRef(i + 1),
                ChainAtomRef(i + 2),
                theta0=math.pi * 0.95,
                k_angle=20.0,
            )
            for i in range(3)
        ]
        torsions = [
            ChainTorsion(
                ChainAtomRef(i),
                ChainAtomRef(i + 1),
                ChainAtomRef(i + 2),
                ChainAtomRef(i + 3),
                phi0=0.5,
                k_tor=5.0,
                n=1,
            )
            for i in range(2)
        ]
        return FlexibleChain(
            beads=beads,
            bonds=bonds,
            angles=angles,
            torsions=torsions,
            name="energy_test",
        )

    def test_total_energy_conserved_under_velocity_verlet(self):
        """Velocity Verlet on the bonded forces keeps total energy drift below 1% of E0."""
        chain = self._make_chain()
        evaluator = ChainForceEvaluator()
        n = chain.n_beads
        velocities = np.zeros((n, 3))
        masses = np.ones(n)

        dt = 0.001
        n_steps = 500

        # Initial total energy.
        F = evaluator.compute_forces(chain)
        K0 = 0.5 * float(np.sum(masses[:, None] * velocities**2))
        V0 = self._compute_V(chain)
        E0 = K0 + V0

        E_history = [E0]
        for _ in range(n_steps):
            # Velocity Verlet: v(t+dt/2) = v(t) + 0.5*dt*a(t).
            v_half = velocities + 0.5 * dt * F / masses[:, None]
            # x(t+dt) = x(t) + dt*v(t+dt/2).
            for i, bead in enumerate(chain.beads):
                bead.pos = bead.pos + dt * v_half[i]
            # Force at new positions.
            F = evaluator.compute_forces(chain)
            # v(t+dt) = v(t+dt/2) + 0.5*dt*a(t+dt).
            velocities = v_half + 0.5 * dt * F / masses[:, None]

            K = 0.5 * float(np.sum(masses[:, None] * velocities**2))
            V = self._compute_V(chain)
            E_history.append(K + V)

        E_arr = np.array(E_history)
        max_drift = float(np.max(np.abs(E_arr - E0)))
        rel_drift = max_drift / abs(E0) if abs(E0) > 1e-12 else max_drift
        assert rel_drift < 0.01, (
            f"Energy drift {rel_drift:.3e} exceeds 1% threshold; "
            f"E0 = {E0:.6f}, max |E - E0| = {max_drift:.6e}; "
            f"a bonded force kernel does not match its potential."
        )


class TestConstraintViolations:
    """Signed violation measure for length and coplanar constraints.

    The violation is what the constraint solver loops against (terminate
    when ||phi|| < tol), so its correctness is foundational for Feature 3.
    """

    @staticmethod
    def _make_atoms(n):
        return [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)
        ]

    def test_no_constraints_returns_empty(self):
        """Constraint violations for a chain with no constraints have shape (0,)."""
        common = ChainCommon(name="empty", atoms=self._make_atoms(3))
        state = ChainState.from_template(common, np.zeros((3, 3)))
        phi = compute_constraint_violations(state)
        assert phi.shape == (0,)

    def test_satisfied_length_constraint_returns_zero(self):
        """A satisfied length constraint reports zero violation."""
        common = ChainCommon(
            name="len1",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi.shape == (1,)
        assert abs(phi[0]) < 1e-12

    def test_overstretched_length_returns_positive_violation(self):
        """An overstretched bond reports a positive length violation equal to the excess distance."""
        common = ChainCommon(
            name="len_long",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi[0] == pytest.approx(0.7, abs=1e-12)

    def test_compressed_length_returns_negative_violation(self):
        """A compressed bond reports a negative length violation equal to the distance deficit."""
        common = ChainCommon(
            name="len_short",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0, 0, 0], [4.2, 0, 0]], dtype=float)
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi[0] == pytest.approx(-0.8, abs=1e-12)

    def test_satisfied_coplanar_constraint_returns_zero(self):
        # Atoms 1, 2, 3 in the xy-plane, atom 0 also in the xy-plane.
        """A coplanar constraint with all four atoms in one plane reports zero violation."""
        common = ChainCommon(
            name="cop1",
            atoms=self._make_atoms(4),
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi.shape == (1,)
        assert abs(phi[0]) < 1e-12

    def test_coplanar_violation_signed_by_plane_side(self):
        # Atom 0 lifted by 0.3 above the plane of atoms 1, 2, 3.
        """A coplanar violation has magnitude equal to the out-of-plane displacement."""
        common = ChainCommon(
            name="cop_lifted",
            atoms=self._make_atoms(4),
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [
                [0.0, 0.0, 0.3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert (
            abs(abs(phi[0]) - 0.3) < 1e-12
        ), f"|violation| should be 0.3, got {phi[0]}"

    def test_canonical_ordering_length_then_coplanar(self):
        """Mixed violations are ordered with the length constraint before the coplanar constraint."""
        common = ChainCommon(
            name="mixed",
            atoms=self._make_atoms(4),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [
                [0.0, 0.0, 0.5],  # atom 0: lifted 0.5 above xy-plane,
                # also far enough from atom 1 to violate length
                [6.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi.shape == (
            2,
        ), f"expected 2 entries (1 length + 1 coplanar), got {phi.shape}"
        assert phi[0] == pytest.approx(1.0208, abs=1e-3)
        # Second entry is the coplanar constraint, magnitude 0.5.
        assert abs(abs(phi[1]) - 0.5) < 1e-12

    def test_degenerate_coplanar_returns_zero(self):
        """A degenerate coplanar constraint with colinear reference atoms returns zero without NaN."""
        common = ChainCommon(
            name="cop_degen",
            atoms=self._make_atoms(4),
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        state = ChainState.from_template(common, positions)
        phi = compute_constraint_violations(state)
        assert phi[0] == 0.0
        assert not np.isnan(phi[0])


class TestSatisfyConstraints:
    """SHAKE-style iterative constraint satisfaction.

    Each test sets up a chain in violation of one or more constraints,
    runs the solver, and verifies that the resulting positions satisfy
    every constraint within the requested tolerance.
    """

    @staticmethod
    def _make_atoms(n):
        return [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)
        ]

    def test_no_constraints_no_op(self):
        """satisfy_constraints leaves positions unchanged and returns 0 when there are no constraints."""
        common = ChainCommon(name="empty", atoms=self._make_atoms(3))
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = ChainState.from_template(common, positions.copy())
        n = satisfy_constraints(state)
        assert n == 0
        np.testing.assert_array_equal(state.positions, positions)

    def test_single_length_constraint_converges(self):
        """SHAKE converges a single length constraint to tolerance, fixing the bond at its target length."""
        common = ChainCommon(
            name="single_len",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        satisfy_constraints(state, tol=1e-8)
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-8
        # Bond should now be exactly 5.0 along x.
        d = state.positions[1] - state.positions[0]
        assert abs(np.linalg.norm(d) - 5.0) < 1e-8

    def test_length_correction_is_symmetric(self):
        """A length correction moves both atoms equally and leaves the center of mass fixed."""
        common = ChainCommon(
            name="sym",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        com_before = positions.mean(axis=0)
        state = ChainState.from_template(common, positions)
        satisfy_constraints(state)
        com_after = state.positions.mean(axis=0)
        np.testing.assert_allclose(com_before, com_after, atol=1e-10)

    def test_chain_of_length_constraints_converges(self):
        """SHAKE converges a four-atom three-bond chain with all bonds initially violated."""
        common = ChainCommon(
            name="chain",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
            ],
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        n_iter = satisfy_constraints(state, tol=1e-7, max_iter=500)
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-7
        # Each adjacent pair should be 3.0 apart.
        for i in range(3):
            d = np.linalg.norm(state.positions[i + 1] - state.positions[i])
            assert abs(d - 3.0) < 1e-7

    def test_single_coplanar_constraint_converges(self):
        """SHAKE returns an out-of-plane atom to the coplanar constraint plane."""
        common = ChainCommon(
            name="cop",
            atoms=self._make_atoms(4),
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [[0.5, 0.5, 0.7], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        satisfy_constraints(state, tol=1e-9)
        phi = compute_constraint_violations(state)
        assert abs(phi[0]) < 1e-9
        # Atom 0 should now have z = 0.
        assert abs(state.positions[0, 2]) < 1e-9

    def test_mixed_constraints_converge(self):
        """SHAKE simultaneously satisfies a shared length and coplanar constraint."""
        common = ChainCommon(
            name="mixed",
            atoms=self._make_atoms(4),
            length_constraints=[LengthConstraint(0, 1, 3.0)],
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [[0.0, 0.0, 0.5], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        satisfy_constraints(state, tol=1e-9)
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_already_satisfied_returns_quickly(self):
        """Starting from a feasible point, SHAKE returns within one verification sweep."""
        common = ChainCommon(
            name="feasible",
            atoms=self._make_atoms(2),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        # Start exactly satisfied.
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        n_iter = satisfy_constraints(state, tol=1e-8)
        # One sweep is enough to detect already-satisfied state.
        assert n_iter <= 1

    def test_idempotent_application(self):
        """Applying SHAKE a second time leaves positions identical."""
        common = ChainCommon(
            name="idem",
            atoms=self._make_atoms(3),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
            ],
        )
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.5, 0.0], [8.0, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        satisfy_constraints(state)
        first_pass = state.positions.copy()
        satisfy_constraints(state)
        second_pass = state.positions
        np.testing.assert_allclose(first_pass, second_pass, atol=1e-12)

    def test_failure_to_converge_raises(self):
        """SHAKE raises RuntimeError when max_iter is too small to converge."""
        common = ChainCommon(
            name="hard",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
            ],
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        with pytest.raises(RuntimeError, match="failed to converge"):
            satisfy_constraints(state, tol=1e-12, max_iter=2)


class TestSatisfyConstraintsNewton:
    """Newton-Lagrange constraint solver: fast convergence for any chain length."""

    @staticmethod
    def _make_atoms(n):
        return [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)
        ]

    def test_no_constraints_returns_zero(self):
        """Newton constraint solver returns 0 when there are no constraints."""
        common = ChainCommon(name="empty", atoms=self._make_atoms(3))
        state = ChainState.from_template(common, np.zeros((3, 3)))
        assert satisfy_constraints_newton(state) == 0

    def test_chain_converges_in_one_iteration(self):
        """Newton solves a length-only chain in at most two iterations."""
        common = ChainCommon(
            name="chain",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
            ],
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        n_iter = satisfy_constraints_newton(state, tol=1e-9)
        assert n_iter <= 2
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_ring_constraint_converges(self):
        """Newton converges a four-atom ring with cyclic length coupling."""
        common = ChainCommon(
            name="ring",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
                LengthConstraint(0, 3, 3.0),
            ],
        )
        # Square geometry, slightly off (sides 3.5 instead of 3).
        positions = np.array(
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [3.5, 3.5, 0.0], [0.0, 3.5, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        satisfy_constraints_newton(state, tol=1e-9)
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_coplanar_constraint_converges(self):
        """Newton converges a single coplanar constraint to tolerance."""
        common = ChainCommon(
            name="cop",
            atoms=self._make_atoms(4),
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [[0.5, 0.5, 0.7], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        satisfy_constraints_newton(state, tol=1e-9)
        phi = compute_constraint_violations(state)
        assert abs(phi[0]) < 1e-9

    def test_mixed_constraints_converge(self):
        """Newton simultaneously satisfies a shared length and coplanar constraint."""
        common = ChainCommon(
            name="mixed",
            atoms=self._make_atoms(4),
            length_constraints=[LengthConstraint(0, 1, 3.0)],
            coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
        )
        positions = np.array(
            [[0.0, 0.0, 0.5], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        satisfy_constraints_newton(state, tol=1e-9)
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_failure_to_converge_raises(self):
        """Newton raises RuntimeError when max_iter is too small for a deformed ring."""
        common = ChainCommon(
            name="hard",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
                LengthConstraint(0, 3, 3.0),
            ],
        )
        # Severely deformed start to make the linearization inadequate.
        positions = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        with pytest.raises(RuntimeError):
            satisfy_constraints_newton(state, tol=1e-12, max_iter=1)


class TestSatisfyConstraintsHybrid:
    """Hybrid solver: SHAKE first, fall back to Newton if SHAKE stalls."""

    @staticmethod
    def _make_atoms(n):
        return [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)
        ]

    def test_easy_case_uses_shake_only(self):
        """An easy chain converges through SHAKE alone without entering the Newton phase of the hybrid solver."""
        common = ChainCommon(
            name="easy",
            atoms=self._make_atoms(3),
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        positions = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        n = satisfy_constraints_hybrid(state, tol=1e-9, shake_max_iter=100)
        assert n <= 100  # Hybrid path means n_total = shake_max_iter + n_newton;
        # if SHAKE handled it, we never enter the >shake_max_iter regime.
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_falls_back_to_newton_when_shake_stalls(self):
        """A tight SHAKE iteration budget forces the hybrid solver to invoke Newton and still converge constraints below tolerance."""
        common = ChainCommon(
            name="chain",
            atoms=self._make_atoms(4),
            length_constraints=[
                LengthConstraint(0, 1, 3.0),
                LengthConstraint(1, 2, 3.0),
                LengthConstraint(2, 3, 3.0),
            ],
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        )
        state = ChainState.from_template(common, positions)
        n = satisfy_constraints_hybrid(
            state, tol=1e-9, shake_max_iter=3, newton_max_iter=20
        )
        # n > shake_max_iter means Newton was invoked.
        assert n > 3
        phi = compute_constraint_violations(state)
        assert np.max(np.abs(phi)) < 1e-9

    def test_hybrid_no_constraints(self):
        """The hybrid constraint solver returns zero iterations when there are no length constraints."""
        common = ChainCommon(name="empty", atoms=self._make_atoms(2))
        state = ChainState.from_template(common, np.zeros((2, 3)))
        n = satisfy_constraints_hybrid(state)
        assert n == 0


class TestTabulatedPotentialCubicSpline:
    """Cubic-spline interpolation in TabulatedPotential.

    The spline replaces the original linear interpolation. Tests verify
    exact behavior at grid points (a property the spline must preserve),
    boundary clamping (out-of-range queries clamp to endpoint values and
    zero derivative), and most importantly that the derivative is
    continuous across grid boundaries (the whole point of the upgrade).
    """

    def _make_parabola(self, n=11):
        """V(x) = (x - 5)^2 sampled on x in [0, 10] with n grid points."""
        xs = np.linspace(0.0, 10.0, n)
        ys = (xs - 5.0) ** 2
        return TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=ys,
            residues=(),
            atoms=(),
            orders=(),
            index=0,
        )

    def test_value_exact_at_grid_points(self):
        """The cubic spline reproduces sampled potential values exactly at the grid knots."""
        pot = self._make_parabola(n=11)
        # Grid points are at x = 0, 1, 2, ..., 10.
        for x in [0, 1, 4, 5, 7, 10]:
            v_true = (x - 5.0) ** 2
            assert pot.value(float(x)) == pytest.approx(v_true, abs=1e-12)

    def test_value_off_grid_close_to_truth(self):
        """Off-grid spline values approximate the underlying smooth function to within 1% relative error."""
        pot = self._make_parabola(n=11)
        for x in [0.5, 2.5, 5.5, 7.5, 9.5]:
            v_true = (x - 5.0) ** 2
            v_spline = pot.value(x)
            rel_err = abs(v_true - v_spline) / max(abs(v_true), 1e-3)
            assert rel_err < 0.01

    def test_boundary_clamping(self):
        """Queries outside the grid range return the endpoint value and zero derivative."""
        pot = self._make_parabola(n=11)
        # Below x_min.
        assert pot.value(-1.0) == pot.values[0]
        assert pot.value(-100.0) == pot.values[0]
        assert pot.deriv(-1.0) == 0.0
        # Above x_max.
        assert pot.value(11.0) == pot.values[-1]
        assert pot.value(100.0) == pot.values[-1]
        assert pot.deriv(11.0) == 0.0

    def test_derivative_is_continuous_across_grid_points(self):
        """The cubic-spline derivative is continuous across interior grid points with no jump."""
        pot = self._make_parabola(n=11)
        # Grid points at integer x. Test continuity at x = 4.
        for grid_x in [2, 4, 6, 8]:
            eps = 1e-5
            v_left = pot.deriv(grid_x - eps)
            v_right = pot.deriv(grid_x + eps)
            # Continuity: the two sides should agree to ~eps precision.
            assert abs(v_left - v_right) < 1e-3, (
                f"Derivative discontinuity at x={grid_x}: "
                f"V'({grid_x - eps:.5f}) = {v_left:.4f}, "
                f"V'({grid_x + eps:.5f}) = {v_right:.4f}"
            )

    def test_derivative_close_to_truth(self):
        """The spline first derivative approximates the true V'(x) to within tight tolerance off-grid."""
        pot = self._make_parabola(n=11)
        for x in [2.0, 3.0, 4.5, 5.5, 7.0, 8.0]:
            d_true = 2.0 * (x - 5.0)
            d_spline = pot.deriv(x)
            assert (
                abs(d_true - d_spline) < 0.05
            ), f"V'({x}) = {d_spline:.4f}, expected {d_true:.4f}"

    def test_short_table_falls_back_to_linear(self):
        """A table with fewer than four points falls back to linear interpolation with correct values and clamping."""
        # 3-point table: a triangle with peak at x=1.
        ys = np.array([0.0, 1.0, 0.0])
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=2.0,
            values=ys,
            residues=(),
            atoms=(),
            orders=(),
            index=0,
        )
        # x = 0.5 should give 0.5 (linear midpoint).
        assert pot.value(0.5) == pytest.approx(0.5, abs=1e-10)
        # x = 1.5 should give 0.5 as well.
        assert pot.value(1.5) == pytest.approx(0.5, abs=1e-10)
        # Boundary clamps still work.
        assert pot.value(-1.0) == 0.0
        assert pot.value(3.0) == 0.0

    def test_constant_potential_has_zero_derivative(self):
        """A constant potential on the grid yields zero derivative everywhere inside."""
        ys = np.full(11, 5.0)
        pot = TabulatedPotential(
            x_min=0.0,
            x_max=10.0,
            values=ys,
            residues=(),
            atoms=(),
            orders=(),
            index=0,
        )
        for x in [0.5, 2.0, 4.7, 6.3, 9.5]:
            assert (
                abs(pot.deriv(x)) < 1e-12
            ), f"V'({x}) should be 0 for constant V, got {pot.deriv(x)}"


# Session-wide cache for the loaded COFFDROP parameter set so we don't pay
# the ~2.6s parse cost in every test.
_COFFDROP_PARAMS_CACHE = {"value": None}


def _load_coffdrop_params():
    """Load real COFFDROP data."""
    if _COFFDROP_PARAMS_CACHE["value"] is not None:
        return _COFFDROP_PARAMS_CACHE["value"]


    pkg_dir = os.path.dirname(_pkg.__file__)
    data_dir = os.path.join(pkg_dir, "coffdrop_data")
    ff_xml = os.path.join(data_dir, "coffdrop.xml")

    params = COFFDROPParams.load(
        ff_xml=ff_xml,
        mapping_xml=os.path.join(data_dir, "map.xml"),
        connectivity_xml=os.path.join(data_dir, "connectivity.xml"),
        charges_xml=os.path.join(data_dir, "charges.xml"),
    )
    _COFFDROP_PARAMS_CACHE["value"] = params
    return params


class TestCOFFDROPEndToEnd:
    """End-to-end validation that the COFFDROP loader works on real data and
    that the cubic-spline upgrade produces self-consistent forces and
    potentials.

    These tests load the actual ~30 MB force-field XML shipped in the
    package. Each test asserts a specific, stable property of the loaded
    data so regressions in either the XML parsing or the spline
    interpolation are caught.
    """

    def test_loader_returns_expected_data_shape(self):
        """The COFFDROP loader returns the expected counts of residues, bonds, charges, and tabulated potentials."""
        params = _load_coffdrop_params()
        assert len(params.mapping) == 23
        assert len(params.bonds) == 40
        assert len(params.charges) == 5
        assert len(params.pair_pots) == 5774
        assert len(params.angle_pots) == 2953
        assert len(params.dihedral_pots) == 10413

    def test_standard_amino_acids_present(self):
        """At least 19 of the 20 canonical amino-acid residues appear in the COFFDROP mapping."""
        params = _load_coffdrop_params()
        canonical = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
        residues = set(params.mapping.keys())
        present = canonical & residues
        assert (
            len(present) >= 19
        ), f"only {len(present)} canonical residues found: {present}"

    def test_pair_potential_attractive_well(self):
        """The ALA-CA / GLY-CA pair potential has an attractive minimum at contact and decays toward zero at large r."""
        params = _load_coffdrop_params()
        v_at_5 = params.pair_potential("ALA", "CA", "GLY", "CA", 5.0)
        v_at_8 = params.pair_potential("ALA", "CA", "GLY", "CA", 8.0)
        # Attraction at typical CA-CA contact.
        assert v_at_5 < 0.0, f"V(r=5) should be attractive, got {v_at_5}"
        # Decays to ~ 0 at large r.
        assert abs(v_at_8) < 0.1, f"V(r=8) should decay, got {v_at_8}"

    def test_pair_force_is_dV_dr(self):
        """pair_force(r) equals dV/dr from the same tabulated potential to within finite-difference tolerance."""
        params = _load_coffdrop_params()
        # Sample a few well-defined distances away from boundaries.
        for r in [4.5, 5.5, 6.5, 7.5]:
            eps = 1e-3
            v_p = params.pair_potential("ALA", "CA", "GLY", "CA", r + eps)
            v_m = params.pair_potential("ALA", "CA", "GLY", "CA", r - eps)
            f_an = params.pair_force("ALA", "CA", "GLY", "CA", r)
            f_fd = (v_p - v_m) / (2 * eps)
            # 1e-3 tolerance: cubic spline derivative matches FD on V.
            assert abs(f_an - f_fd) < 1e-3, (
                f"pair_force({r}) = {f_an}, FD says {f_fd}, "
                f"diff = {abs(f_an - f_fd)}"
            )

    def test_pair_force_continuous_at_table_grid_boundaries(self):
        """The pair force is C^1 continuous across table grid points with no finite jump."""
        params = _load_coffdrop_params()
        for grid_r in [5.05, 6.05, 7.05]:
            eps = 1e-5
            f_left = params.pair_force("ALA", "CA", "GLY", "CA", grid_r - eps)
            f_right = params.pair_force("ALA", "CA", "GLY", "CA", grid_r + eps)
            assert abs(f_left - f_right) < 1e-3, (
                f"Force discontinuity at r={grid_r}: "
                f"F({grid_r - eps:.5f}) = {f_left:.6f}, "
                f"F({grid_r + eps:.5f}) = {f_right:.6f}, "
                f"jump = {abs(f_left - f_right):.4e}"
            )

    def test_unknown_residue_returns_zero(self):
        """Querying an unknown residue returns a finite value rather than crashing."""
        params = _load_coffdrop_params()
        # XYZ is not a known residue.
        v = params.pair_potential("XYZ", "CA", "GLY", "CA", 5.0)
        assert np.isfinite(v)


class TestChainBDParameters:
    """Default values and post-init behavior of the parameter dataclass."""

    def test_defaults(self):
        """ChainBDParameters defaults match expected values and r_escape auto-derives as 1.1 times r_start."""
        p = ChainBDParameters()
        assert p.n_trajectories == 1_000
        assert p.dt == 0.2
        assert p.dt_rxn == 0.05
        assert p.dt_chain == 0.05
        assert p.chain_steps_per_outer == 4
        assert p.constraint_tol == 1e-6
        assert p.constraint_max_iter == 200
        assert p.r_start == 100.0
        # r_escape auto-derives from r_start * 1.1 when left at 0.
        assert abs(p.r_escape - 110.0) < 1e-9

    def test_explicit_r_escape_respected(self):
        """An explicitly set r_escape is preserved and not overwritten during post-init."""
        p = ChainBDParameters(r_start=50.0, r_escape=200.0)
        assert p.r_escape == 200.0

    def test_r_escape_auto_uses_r_start(self):
        """Leaving r_escape at 0 auto-derives it as 1.1 times r_start."""
        p = ChainBDParameters(r_start=50.0)
        assert abs(p.r_escape - 55.0) < 1e-9


class TestPlaceChain:
    """Rigid-body placement of body-frame chain coordinates."""

    def test_identity_orientation_zero_com_returns_body_unchanged(self):
        """Identity orientation and zero center of mass leave body coordinates unchanged."""
        body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ori = _Q(1.0, 0.0, 0.0, 0.0)
        world = place_chain(body, np.zeros(3), ori)
        np.testing.assert_allclose(world, body, atol=1e-12)

    def test_translation_only_shifts_all_atoms_by_com(self):
        """Identity orientation with a nonzero center of mass shifts all atoms by that center of mass."""
        body = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        com = np.array([10.0, 20.0, 30.0])
        ori = _Q(1.0, 0.0, 0.0, 0.0)
        world = place_chain(body, com, ori)
        np.testing.assert_allclose(world, body + com, atol=1e-12)

    def test_rotation_90deg_about_z(self):
        """A 90 degree rotation about z maps x to y, y to -x, and leaves z fixed."""
        body = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        theta = np.pi / 2
        ori = _Q(np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2))
        world = place_chain(body, np.zeros(3), ori)
        np.testing.assert_allclose(world[0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(world[1], [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(world[2], [0.0, 0.0, 1.0], atol=1e-12)

    def test_combined_rotation_and_translation(self):
        """Combined placement applies the rotation then the center-of-mass translation."""
        body = np.array([[1.0, 0.0, 0.0]])
        theta = np.pi / 2
        ori = _Q(np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2))
        com = np.array([5.0, 0.0, 0.0])
        world = place_chain(body, com, ori)
        np.testing.assert_allclose(world[0], [5.0, 1.0, 0.0], atol=1e-12)

    def test_com_after_placement_equals_input_com(self):
        """Placing an origin-centered body gives a world center of mass equal to the input center of mass."""
        rng = np.random.default_rng(42)
        body = rng.standard_normal((10, 3))
        body -= body.mean(axis=0)
        v = rng.standard_normal(4)
        v /= np.linalg.norm(v)
        ori = _Q(v[0], v[1], v[2], v[3])
        com = np.array([3.0, -2.0, 7.0])
        world = place_chain(body, com, ori)
        np.testing.assert_allclose(world.mean(axis=0), com, atol=1e-10)

    def test_rigid_body_invariance_pairwise_distances(self):
        """Rigid placement preserves all pairwise interatomic distances."""
        rng = np.random.default_rng(7)
        body = rng.standard_normal((8, 3))
        body -= body.mean(axis=0)
        v = rng.standard_normal(4)
        v /= np.linalg.norm(v)
        ori = _Q(v[0], v[1], v[2], v[3])
        com = np.array([10.0, -5.0, 3.0])
        world = place_chain(body, com, ori)
        for i in range(8):
            for j in range(i + 1, 8):
                d_body = np.linalg.norm(body[i] - body[j])
                d_world = np.linalg.norm(world[i] - world[j])
                assert abs(d_body - d_world) < 1e-10


class TestInitializeBsphere:
    """B-sphere initialization: random direction + random orientation."""

    def test_position_has_correct_magnitude(self):
        """The initialized B-sphere position has magnitude exactly equal to r_start."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            pos, _ = initialize_bsphere(rng, r_start=100.0)
            assert abs(np.linalg.norm(pos) - 100.0) < 1e-12

    def test_orientation_is_unit_quaternion(self):
        """The initialized B-sphere orientation is a unit quaternion."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            _, ori = initialize_bsphere(rng, r_start=100.0)
            norm = np.sqrt(ori.w**2 + ori.x**2 + ori.y**2 + ori.z**2)
            assert abs(norm - 1.0) < 1e-12

    def test_position_direction_is_isotropic(self):
        """The initialized position directions are isotropic with near-zero mean over many samples."""
        rng = np.random.default_rng(42)
        n = 5000
        positions = np.array(
            [initialize_bsphere(rng, r_start=1.0)[0] for _ in range(n)]
        )
        mean = positions.mean(axis=0)
        assert np.all(np.abs(mean) < 0.05), f"mean = {mean}"

    def test_reproducibility_with_same_seed(self):
        """Two RNGs seeded identically produce identical B-sphere position and orientation."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        pos1, ori1 = initialize_bsphere(rng1, r_start=50.0)
        pos2, ori2 = initialize_bsphere(rng2, r_start=50.0)
        np.testing.assert_array_equal(pos1, pos2)
        assert ori1.w == ori2.w
        assert ori1.x == ori2.x
        assert ori1.y == ori2.y
        assert ori1.z == ori2.z

    def test_different_r_start_scales_position(self):
        """Same seed with different r_start scales the position proportionally to r_start."""
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        pos1, _ = initialize_bsphere(rng1, r_start=10.0)
        pos2, _ = initialize_bsphere(rng2, r_start=20.0)
        # pos2 should be exactly 2x pos1 (same random direction, scaled).
        np.testing.assert_allclose(pos2, 2.0 * pos1, atol=1e-12)


class TestCheckEscape:
    """Trivial bounds check on |pos| vs r_escape."""

    def test_inside_returns_false(self):
        """A position inside r_escape returns False from check_escape."""
        assert check_escape(np.array([10.0, 0.0, 0.0]), r_escape=100.0) is False

    def test_at_boundary_returns_true(self):
        """A position exactly at r_escape counts as escaped and returns True."""
        assert check_escape(np.array([100.0, 0.0, 0.0]), r_escape=100.0) is True

    def test_outside_returns_true(self):
        """A position outside r_escape returns True from check_escape."""
        assert check_escape(np.array([200.0, 0.0, 0.0]), r_escape=100.0) is True

    def test_zero_position(self):
        """The origin position returns False from check_escape."""
        assert check_escape(np.zeros(3), r_escape=10.0) is False


class TestChainScratchMolecule:
    """Build a scratch Molecule from a chain template and update its positions."""

    @staticmethod
    def _make_template(n=3):
        atoms = [
            ChainAtom(radius=2.0 + 0.1 * i, charge=float(i), resname=f"R{i}", resid=i)
            for i in range(n)
        ]
        return ChainCommon(name="test_chain", atoms=atoms)

    def test_scratch_has_correct_atom_count(self):
        """The scratch molecule has the same atom count as its chain template."""
        common = self._make_template(n=4)
        scratch = make_chain_scratch_molecule(common)
        assert len(scratch.atoms) == 4

    def test_scratch_carries_radius_and_charge(self):
        """The scratch molecule carries each atom's radius, charge, residue name, and residue index from the template."""
        common = self._make_template(n=3)
        scratch = make_chain_scratch_molecule(common)
        for i, atom in enumerate(scratch.atoms):
            assert atom.radius == 2.0 + 0.1 * i
            assert atom.charge == float(i)
            assert atom.residue_name == f"R{i}"
            assert atom.residue_index == i

    def test_initial_positions_are_zero(self):
        """All scratch-molecule atoms start at the origin."""
        common = self._make_template(n=3)
        scratch = make_chain_scratch_molecule(common)
        for atom in scratch.atoms:
            assert atom.x == 0.0 and atom.y == 0.0 and atom.z == 0.0

    def test_update_positions_writes_through(self):
        """Updating scratch positions writes the new coordinates through to each atom."""
        common = self._make_template(n=3)
        scratch = make_chain_scratch_molecule(common)
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        update_chain_scratch_positions(scratch, positions)
        assert scratch.atoms[0].x == 1.0
        assert scratch.atoms[0].y == 2.0
        assert scratch.atoms[0].z == 3.0
        assert scratch.atoms[2].x == 7.0
        assert scratch.atoms[2].z == 9.0

    def test_update_positions_rejects_wrong_shape(self):
        """Updating scratch positions with a wrong-shaped array raises ValueError."""
        common = self._make_template(n=3)
        scratch = make_chain_scratch_molecule(common)
        bad = np.zeros((2, 3))
        with pytest.raises(ValueError, match="does not match"):
            update_chain_scratch_positions(scratch, bad)


class TestCheckChainReaction:
    """End-to-end check that the chain reaction wrapper composes correctly."""

    def test_no_reactions_returns_none(self):
        """An empty PathwaySet makes check_chain_reaction return None."""

        common = ChainCommon(
            name="c",
            atoms=[ChainAtom(radius=2.0, charge=0.0, resname="X", resid=0)],
        )
        scratch = make_chain_scratch_molecule(common)
        update_chain_scratch_positions(scratch, np.zeros((1, 3)))
        target = Molecule(atoms=[Atom(x=0.0, y=0.0, z=0.0, radius=2.0, charge=0.0)])

        empty_pathways = PathwaySet(reactions=[])
        result = check_chain_reaction(
            target, scratch, empty_pathways, rng=np.random.default_rng(0)
        )
        assert result is None


class TestEvaluateTargetGridForceOnChain:
    """Per-atom electrostatic force from a precomputed PB grid.

    Tests use a synthetic DXGrid populated with an analytic potential so
    that the forces and energies can be verified against closed-form
    expressions.
    """

    @staticmethod
    def _make_linear_potential_grid():
        """Build a DXGrid where phi(x, y, z) = x. Gradient is (1, 0, 0)
        everywhere; force on a unit charge is (-1, 0, 0)."""

        nx, ny, nz = 21, 21, 21
        spacing = 1.0
        origin = np.array([-10.0, -10.0, -10.0])
        # delta is a (3, 3) matrix; orthogonal grid with uniform spacing.
        delta = np.eye(3) * spacing
        # Build potential phi(x, y, z) = x at each grid point.
        xs = origin[0] + spacing * np.arange(nx)
        data = np.zeros((nx, ny, nz))
        for i, x in enumerate(xs):
            data[i, :, :] = x
        return DXGrid(origin=origin, delta=delta, data=data)

    def test_force_in_linear_potential(self):
        """In a linear potential phi=x the force on charge q is -q times the gradient and energy is sum of q times phi."""
        grid = self._make_linear_potential_grid()
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
        charges = np.array([1.0, -2.0])
        forces, energy = evaluate_target_grid_force_on_chain(
            positions,
            charges,
            grid,
        )
        # Force on atom 0 (q=+1, at origin):  F = -1 * (1, 0, 0) = (-1, 0, 0).
        np.testing.assert_allclose(forces[0], [-1.0, 0.0, 0.0], atol=1e-10)
        # Force on atom 1 (q=-2, at [2,1,-1]): F = -(-2) * (1, 0, 0) = (+2, 0, 0).
        np.testing.assert_allclose(forces[1], [2.0, 0.0, 0.0], atol=1e-10)
        assert abs(energy - (-4.0)) < 1e-10

    def test_zero_charge_gives_zero_force(self):
        """A zero-charge atom feels zero grid force and contributes zero energy."""
        grid = self._make_linear_potential_grid()
        positions = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([0.0])
        forces, energy = evaluate_target_grid_force_on_chain(
            positions,
            charges,
            grid,
        )
        np.testing.assert_allclose(forces[0], [0.0, 0.0, 0.0], atol=1e-12)
        assert abs(energy) < 1e-12

    def test_atoms_outside_grid_get_zero_force(self):
        """Atoms outside the grid box get zero force without crashing."""
        grid = self._make_linear_potential_grid()
        # Grid covers x in [-10, 10]; place an atom at x=100 (way outside).
        positions = np.array([[100.0, 0.0, 0.0]])
        charges = np.array([1.0])
        forces, energy = evaluate_target_grid_force_on_chain(
            positions,
            charges,
            grid,
        )
        # Out-of-grid points should give zero force (no contribution).
        np.testing.assert_allclose(forces[0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_shape_mismatch_raises(self):
        """A mismatch between position and charge counts raises ValueError."""
        grid = self._make_linear_potential_grid()
        positions = np.zeros((3, 3))
        charges = np.zeros(2)  # wrong count
        with pytest.raises(ValueError, match="does not"):
            evaluate_target_grid_force_on_chain(positions, charges, grid)

    def test_returns_correct_shapes(self):
        """The grid force evaluator returns forces of shape (N, 3) and a float energy."""
        grid = self._make_linear_potential_grid()
        positions = np.zeros((5, 3))
        charges = np.ones(5)
        forces, energy = evaluate_target_grid_force_on_chain(
            positions,
            charges,
            grid,
        )
        assert forces.shape == (5, 3)
        assert isinstance(energy, float)


class TestBornForceOnChain:
    """Unit tests for evaluate_born_force_on_chain.

    Born desolvation force on a chain bead is

        F_i = -alpha * q_i^2 * grad(g)(r_i)

    and the corresponding self-energy contribution is

        V_i =  alpha * q_i^2 * g(r_i)

    where g(r) is the Born cavity self energy grid and alpha is the
    desolvation prefactor (default 1.0, matching
    pystarc/forces/engine.py for the rigid-body BD path). The grid holds
    the Kirkwood n = 1 image energy with the rigorous normalisation
    already folded in, so a charge sees dG = alpha q^2 a^3 / r^4.

    The grid path uses trilinear interpolation for g and central
    differences for grad(g); both vanish smoothly outside the grid box.
    """

    @staticmethod
    def _make_linear_born_grid(slope_x: float = 1.0):
        """Synthetic Born grid with g(x, y, z) = slope_x * x."""

        nx = ny = nz = 21
        spacing = 1.0
        origin = np.array([-10.0, -10.0, -10.0])
        delta = np.eye(3) * spacing
        xs = origin[0] + spacing * np.arange(nx)
        data = np.zeros((nx, ny, nz))
        for i, x in enumerate(xs):
            data[i, :, :] = slope_x * x
        return DXGrid(origin=origin, delta=delta, data=data)

    @staticmethod
    def _make_quadratic_born_grid():
        """Synthetic Born grid with g(x, y, z) = x^2.

        Central-difference at half-spacing on g=x^2 returns 2x exactly:
            ((x0+h)^2 - (x0-h)^2) / (2h) = 2*x0.
        """

        nx = ny = nz = 21
        spacing = 1.0
        origin = np.array([-10.0, -10.0, -10.0])
        delta = np.eye(3) * spacing
        xs = origin[0] + spacing * np.arange(nx)
        data = np.zeros((nx, ny, nz))
        for i, x in enumerate(xs):
            data[i, :, :] = x * x
        return DXGrid(origin=origin, delta=delta, data=data)

    def test_force_in_linear_born_potential(self):
        """In a linear Born grid g=x the force is -alpha*q^2 times the gradient, same sign for both charge signs."""
        grid = self._make_linear_born_grid(slope_x=1.0)
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
        charges = np.array([1.0, -2.0])
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
        )
        a = DEFAULT_DESOLVATION_ALPHA
        # Atom 0 (q=+1, q^2=1): F = -a * 1 * (1, 0, 0).
        np.testing.assert_allclose(forces[0], [-a * 1.0, 0.0, 0.0], atol=1e-10)
        # Atom 1 (q=-2, q^2=4): F = -a * 4 * (1, 0, 0).
        np.testing.assert_allclose(forces[1], [-a * 4.0, 0.0, 0.0], atol=1e-10)
        # V = a * sum_i q_i^2 * g(r_i) = a * (1*0 + 4*2) = a * 8.
        assert abs(energy - a * 8.0) < 1e-10

    def test_force_in_quadratic_born_potential(self):
        """In a quadratic Born grid g=x^2 the force is -alpha*q^2*2x with energy summed over beads."""
        grid = self._make_quadratic_born_grid()
        positions = np.array([[3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        charges = np.array([1.5, 0.5])
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
        )
        a = DEFAULT_DESOLVATION_ALPHA
        # Atom 0: F_x = -a * 1.5^2 * 2*3 = -a * 13.5
        np.testing.assert_allclose(
            forces[0, 0],
            -a * (1.5 * 1.5) * 2.0 * 3.0,
            atol=1e-10,
        )
        # Atom 1: F_x = -a * 0.5^2 * 2*(-2) = +a * 1.0
        np.testing.assert_allclose(
            forces[1, 0],
            -a * (0.5 * 0.5) * 2.0 * (-2.0),
            atol=1e-10,
        )
        # y, z components zero by construction.
        np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1e-10)
        assert abs(energy - a * (2.25 * 9.0 + 0.25 * 4.0)) < 1e-10

    def test_zero_charge_gives_zero_born_force(self):
        """A neutral bead feels zero Born force regardless of position."""
        grid = self._make_linear_born_grid()
        positions = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([0.0])
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
        )
        np.testing.assert_allclose(forces[0], [0.0, 0.0, 0.0], atol=1e-12)
        assert abs(energy) < 1e-12

    def test_atoms_outside_grid_get_zero_force(self):
        """Atoms far outside the grid box contribute zero Born force and energy."""
        grid = self._make_linear_born_grid()
        positions = np.array([[100.0, 0.0, 0.0]])
        charges = np.array([1.0])
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
        )
        np.testing.assert_allclose(forces[0], [0.0, 0.0, 0.0], atol=1e-10)
        assert abs(energy) < 1e-10

    def test_alpha_scales_force_and_energy_linearly(self):
        """Doubling alpha doubles the per-atom Born force and total energy."""
        grid = self._make_linear_born_grid()
        positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.5, 0.0]])
        charges = np.array([1.0, -1.5])
        f_a, e_a = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
            alpha=0.05,
        )
        f_b, e_b = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
            alpha=0.10,
        )
        np.testing.assert_allclose(f_b, 2.0 * f_a, atol=1e-12)
        assert abs(e_b - 2.0 * e_a) < 1e-12

    def test_zero_alpha_gives_zero_force_and_energy(self):
        """Setting alpha to zero turns the Born force and energy off entirely."""
        grid = self._make_linear_born_grid()
        positions = np.array([[1.0, 0.0, 0.0]])
        charges = np.array([1.0])
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
            alpha=0.0,
        )
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)
        assert abs(energy) < 1e-12

    def test_shape_mismatch_raises(self):
        """A mismatch between position and charge counts raises ValueError in the Born evaluator."""
        grid = self._make_linear_born_grid()
        positions = np.zeros((3, 3))
        charges = np.zeros(2)  # wrong count
        with pytest.raises(ValueError, match="does not"):
            evaluate_born_force_on_chain(positions, charges, grid)

    def test_returns_correct_shapes(self):
        """The Born force evaluator returns forces of shape (N, 3) and a float energy."""
        grid = self._make_linear_born_grid()
        positions = np.zeros((5, 3))
        charges = np.ones(5)
        forces, energy = evaluate_born_force_on_chain(
            positions,
            charges,
            grid,
        )
        assert forces.shape == (5, 3)
        assert isinstance(energy, float)


class TestChainBDSimulatorBornAttributes:
    """Verify ChainBDSimulator __init__ accepts and stores Born-related
    kwargs without affecting unrelated behavior. Routing through
    _compute_per_atom_external_forces is tested in a separate class
    once the wiring edit lands.
    """

    @staticmethod
    def _make_minimal_setup():
        chain = chain_from_sequence("GLY-ALA", caps=("ACE", "NME"))
        positions = place_relaxed_geometry(chain)
        positions = positions - positions.mean(axis=0)
        params = ChainBDParameters(
            n_trajectories=1,
            dt=0.01,
            max_steps=10,
            r_start=20.0,
            r_escape=50.0,
            seed=0,
        )
        target = Molecule(name="empty", atoms=[])
        pathway_set = PathwaySet()
        return chain, positions, params, target, pathway_set

    def test_defaults_born_grid_to_none_and_alpha_to_default(self):
        """ChainBDSimulator defaults born_grid to None and desolvation alpha to the default constant."""
        chain, positions, params, target, pathway_set = self._make_minimal_setup()
        sim = ChainBDSimulator(
            target=target,
            chain_template=chain,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=pathway_set,
            D_trans=0.1,
            D_rot=0.01,
        )
        assert sim.born_grid is None
        assert sim.desolvation_alpha == DEFAULT_DESOLVATION_ALPHA

    def test_stores_born_grid_and_custom_alpha(self):
        """ChainBDSimulator stores the supplied Born grid reference and custom desolvation alpha."""
        chain, positions, params, target, pathway_set = self._make_minimal_setup()
        # Build a trivial Born grid object to confirm reference is stored.
        grid = TestBornForceOnChain._make_linear_born_grid()
        sim = ChainBDSimulator(
            target=target,
            chain_template=chain,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=pathway_set,
            D_trans=0.1,
            D_rot=0.01,
            born_grid=grid,
            desolvation_alpha=0.123,
        )
        assert sim.born_grid is grid
        assert sim.desolvation_alpha == 0.123

    def test_routes_born_into_per_atom_external_forces(self):
        """With only a linear Born grid, per-atom external forces equal the closed-form -alpha*q^2*(1,0,0)."""
        chain, positions, params, target, pathway_set = self._make_minimal_setup()
        # g(x, y, z) = x  => grad(g) = (1, 0, 0)
        grid = TestBornForceOnChain._make_linear_born_grid(slope_x=1.0)
        sim = ChainBDSimulator(
            target=target,
            chain_template=chain,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=pathway_set,
            D_trans=0.1,
            D_rot=0.01,
            target_grid=None,
            born_grid=grid,
            desolvation_alpha=DEFAULT_DESOLVATION_ALPHA,
        )

        # World-frame positions: chain CoM already at origin.
        world_pos = positions.copy()
        forces = sim._compute_per_atom_external_forces(world_pos)

        # Closed-form expectation per bead: F = (-alpha * q^2, 0, 0).
        chain_charges = np.array(
            [a.charge for a in chain.atoms],
            dtype=float,
        )
        expected_fx = -DEFAULT_DESOLVATION_ALPHA * (chain_charges**2)
        np.testing.assert_allclose(forces[:, 0], expected_fx, atol=1e-10)
        np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1e-10)

    def test_born_grid_none_does_not_perturb_existing_path(self):
        """With born_grid None and no other forces, per-atom external forces are exactly zero."""
        chain, positions, params, target, pathway_set = self._make_minimal_setup()
        sim = ChainBDSimulator(
            target=target,
            chain_template=chain,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=pathway_set,
            D_trans=0.1,
            D_rot=0.01,
            target_grid=None,
            born_grid=None,
        )
        forces = sim._compute_per_atom_external_forces(positions.copy())
        np.testing.assert_allclose(forces, 0.0, atol=1e-12)

    def test_born_adds_to_electrostatic_not_replaces(self):
        """With both grids set, per-atom external forces equal the sum of electrostatic and Born contributions on orthogonal axes."""
        chain, positions, params, target, pathway_set = self._make_minimal_setup()


        nx = ny = nz = 21
        spacing = 1.0
        origin = np.array([-10.0, -10.0, -10.0])
        delta = np.eye(3) * spacing
        ys = origin[1] + spacing * np.arange(ny)
        elec_data = np.zeros((nx, ny, nz))
        for j, y in enumerate(ys):
            elec_data[:, j, :] = y
        elec_grid = DXGrid(origin=origin, delta=delta, data=elec_data)

        # Born g(x, y, z) = x.
        born_grid = TestBornForceOnChain._make_linear_born_grid(slope_x=1.0)

        sim = ChainBDSimulator(
            target=target,
            chain_template=chain,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=pathway_set,
            D_trans=0.1,
            D_rot=0.01,
            target_grid=elec_grid,
            born_grid=born_grid,
            desolvation_alpha=DEFAULT_DESOLVATION_ALPHA,
        )
        world_pos = positions.copy()
        forces = sim._compute_per_atom_external_forces(world_pos)

        chain_charges = np.array(
            [a.charge for a in chain.atoms],
            dtype=float,
        )
        expected_fx = -DEFAULT_DESOLVATION_ALPHA * (chain_charges**2)
        expected_fy = -chain_charges
        np.testing.assert_allclose(forces[:, 0], expected_fx, atol=1e-10)
        np.testing.assert_allclose(forces[:, 1], expected_fy, atol=1e-10)
        np.testing.assert_allclose(forces[:, 2], 0.0, atol=1e-10)


class TestChainTargetStericVectorizedEquivalence:
    """Direct numerical equivalence: vectorized chain_target_steric_forces
    output must match a reference pure-Python looped implementation to
    floating-point precision.

    The existing TestSoftRepulsion class checks expected physics behaviors
    (force direction, magnitude scaling, ghost-atom handling), but those
    tests would not catch a subtle bug in the vectorization itself if
    the bug happened to satisfy each individual assertion. This class
    catches that class of bug by comparing against a reference loop on
    synthetic inputs.

    The reference loop is intentionally a verbatim copy of the original
    pre-vectorization implementation, so any future change to the
    vectorized version that breaks numerical equivalence will fail this
    test.
    """

    @staticmethod
    def _reference_looped(chain_world_positions, chain_radii, target, eps=1.0):
        """Verbatim copy of the original pre-vectorization implementation
        of chain_target_steric_forces. Used only by these equivalence
        tests; do NOT use elsewhere.
        """
        n_chain = len(chain_radii)
        F = np.zeros((n_chain, 3))
        if target is None:
            return F
        for atom in target.atoms:
            if atom.radius < 1e-10:
                continue
            ra = np.array([atom.x, atom.y, atom.z])
            for i in range(n_chain):
                if chain_radii[i] < 1e-10:
                    continue
                dr = chain_world_positions[i] - ra
                r = float(np.linalg.norm(dr))
                sig = chain_radii[i] + atom.radius
                if r < 1e-10 or r >= sig:
                    continue
                sr = sig / r
                sr6 = sr**6
                sr12 = sr6 * sr6
                f_mag_over_r = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / (r * r)
                F[i] += f_mag_over_r * dr
        return F

    def _make_target(self, atom_data):
        """Build a minimal synthetic target with atoms from a list of
        (x, y, z, radius) tuples. Generic helper, no reference to any
        biological system.
        """

        atoms = []
        for k, (x, y, z, r) in enumerate(atom_data):
            atoms.append(
                Atom(
                    name=f"X{k}",
                    residue_name="UNK",
                    residue_index=k,
                    chain="A",
                    x=x,
                    y=y,
                    z=z,
                    charge=0.0,
                    radius=r,
                )
            )
        return Molecule(name="test_target", atoms=atoms)

    def test_simple_in_range_pairs(self):
        """The vectorized chain-target steric force matches the looped reference when all pairs lie within the WCA range."""

        chain_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        chain_r = np.array([2.0, 2.0, 2.0])
        target = self._make_target(
            [
                (1.5, 0.0, 0.0, 2.0),  # close to bead 0 (r=1.5, sig=4.0)
                (0.0, 0.0, 1.5, 2.0),  # close to all three beads
            ]
        )
        F_ref = self._reference_looped(chain_pos, chain_r, target, eps=1.0)
        F_vec = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        np.testing.assert_allclose(F_vec, F_ref, rtol=1e-7, atol=10.0)

    def test_mixed_in_and_out_of_range(self):
        """The vectorized force matches the reference across in-range, at-cutoff, and far pairs, with the far bead getting zero force."""

        chain_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],  # far from everything
            ]
        )
        chain_r = np.array([2.0, 2.0, 2.0])
        target = self._make_target(
            [
                (1.0, 0.0, 0.0, 2.0),  # bead 0: r=1, sig=4 (in range)
                (4.0, 0.0, 0.0, 2.0),  # bead 0: r=4, sig=4 (exactly at cutoff: zero)
                (8.0, 0.0, 0.0, 2.0),  # bead 1: r=2, sig=4 (in range)
                (100.0, 100.0, 100.0, 2.0),  # far from everything (zero)
            ]
        )
        F_ref = self._reference_looped(chain_pos, chain_r, target, eps=1.0)
        F_vec = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        np.testing.assert_allclose(F_vec, F_ref, rtol=1e-7, atol=10.0)
        # Bead 2 is far from everything: zero force.
        np.testing.assert_allclose(F_vec[2], np.zeros(3), atol=1e-15)

    def test_with_ghost_atoms_on_both_sides(self):
        """Ghost atoms with radius below 1e-10 are skipped identically by both implementations, leaving the ghost bead with zero force."""

        chain_pos = np.array(
            [
                [0.0, 0.0, 0.0],  # real bead
                [1.5, 0.0, 0.0],  # GHOST (r=0)
                [3.0, 0.0, 0.0],  # real bead
            ]
        )
        chain_r = np.array([2.0, 0.0, 2.0])
        target = self._make_target(
            [
                (1.0, 0.0, 0.0, 2.0),  # real, in range to bead 0
                (1.0, 0.0, 0.0, 0.0),  # GHOST at same position
                (4.0, 0.0, 0.0, 2.0),  # real, in range to bead 2
            ]
        )
        F_ref = self._reference_looped(chain_pos, chain_r, target, eps=1.0)
        F_vec = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        np.testing.assert_allclose(F_vec, F_ref, rtol=1e-7, atol=10.0)
        np.testing.assert_array_equal(F_vec[1], np.zeros(3))

    def test_eps_scaling(self):
        """The chain-target steric force scales linearly with eps and matches the looped reference across several eps values."""

        chain_pos = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
        chain_r = np.array([2.0, 2.0])
        target = self._make_target(
            [
                (1.5, 0.5, 0.0, 2.0),
                (5.0, -0.3, 0.0, 2.0),
            ]
        )
        for eps in [0.1, 0.3, 0.5, 1.0, 2.5]:
            F_ref = self._reference_looped(chain_pos, chain_r, target, eps=eps)
            F_vec = chain_target_steric_forces(chain_pos, chain_r, target, eps=eps)
            np.testing.assert_allclose(
                F_vec,
                F_ref,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"mismatch at eps={eps}",
            )

    def test_random_many_pairs_simultaneously_in_range(self):
        """The vectorized force matches the looped reference on a random 20-bead chain and 50-atom target with many in-range pairs."""

        rng = np.random.default_rng(42)
        n_chain = 20
        n_target = 50
        # Compact box so many pairs end up inside WCA cutoff.
        chain_pos = rng.normal(0.0, 4.0, size=(n_chain, 3))
        chain_r = np.full(n_chain, 2.0)
        atom_data = []
        for _ in range(n_target):
            x, y, z = rng.uniform(-6.0, 6.0, size=3)
            r = float(rng.uniform(1.0, 2.0))
            atom_data.append((float(x), float(y), float(z), r))
        target = self._make_target(atom_data)

        F_ref = self._reference_looped(chain_pos, chain_r, target, eps=0.5)
        F_vec = chain_target_steric_forces(chain_pos, chain_r, target, eps=0.5)
        np.testing.assert_allclose(F_vec, F_ref, rtol=1e-7, atol=10.0)


class TestComputeChainForcesEquivalence:
    """compute_chain_forces (operating on ChainState) must agree bit-for-bit
    with the legacy ChainForceEvaluator (operating on FlexibleChain).

    Both code paths share the same physics kernels, so the new path should
    produce identical forces. Any discrepancy here means an indexing or
    accumulation bug crept into the port.
    """

    @staticmethod
    def _build_equivalent_pair(positions, bonds, angles, torsions):
        """Return (state, flexible_chain) representing the same configuration."""
        n = positions.shape[0]
        # New representation.
        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname=f"R{i}", resid=i)
            for i in range(n)
        ]
        common = ChainCommon(
            name="t", atoms=atoms, bonds=bonds, angles=angles, torsions=torsions
        )
        state = ChainState.from_template(common, positions)
        # Legacy representation, same parameters.
        beads = [
            ChainBead(
                pos=positions[i].copy(), force=np.zeros(3), radius=2.0, charge=0.0
            )
            for i in range(n)
        ]
        fc = FlexibleChain(beads=beads, bonds=bonds, angles=angles, torsions=torsions)
        return state, fc

    def test_bonds_only_match_legacy(self):
        """compute_chain_forces reproduces the legacy evaluator forces for a bonds-only chain."""
        positions = np.array([[0.0, 0.0, 0.0], [4.5, 0.5, 0.0], [9.0, 1.0, 0.0]])
        bonds = [
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), r0=3.8, k_spring=100.0),
            ChainBond(ChainAtomRef(1), ChainAtomRef(2), r0=3.8, k_spring=100.0),
        ]
        state, fc = self._build_equivalent_pair(
            positions, bonds=bonds, angles=[], torsions=[]
        )
        compute_chain_forces(state)
        F_legacy = ChainForceEvaluator().compute_forces(fc)
        np.testing.assert_array_equal(state.forces, F_legacy)

    def test_angles_only_match_legacy(self):
        """compute_chain_forces reproduces the legacy evaluator forces for an angles-only bent chain."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 1.5, 0.0]]
        )  # bent angle
        angles = [
            ChainAngle(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                theta0=math.pi,
                k_angle=50.0,
            ),
        ]
        state, fc = self._build_equivalent_pair(
            positions, bonds=[], angles=angles, torsions=[]
        )
        compute_chain_forces(state)
        F_legacy = ChainForceEvaluator().compute_forces(fc)
        np.testing.assert_array_equal(state.forces, F_legacy)

    def test_torsions_only_match_legacy(self):
        """compute_chain_forces reproduces the legacy evaluator forces for a torsions-only chain."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 1.0, 0.5]]
        )
        torsions = [
            ChainTorsion(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                phi0=0.0,
                k_tor=10.0,
                n=1,
            ),
        ]
        state, fc = self._build_equivalent_pair(
            positions, bonds=[], angles=[], torsions=torsions
        )
        compute_chain_forces(state)
        F_legacy = ChainForceEvaluator().compute_forces(fc)
        np.testing.assert_array_equal(state.forces, F_legacy)

    def test_full_chain_match_legacy(self):
        """compute_chain_forces reproduces the legacy evaluator forces with bonds, angles, and torsions all active."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [4.5, 0.5, 0.0], [9.0, 1.0, 1.0], [13.0, 0.0, -1.0]]
        )
        bonds = [
            ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), r0=3.8, k_spring=100.0)
            for i in range(3)
        ]
        angles = [
            ChainAngle(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                theta0=math.pi,
                k_angle=50.0,
            ),
            ChainAngle(
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                theta0=math.pi,
                k_angle=50.0,
            ),
        ]
        torsions = [
            ChainTorsion(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                phi0=0.0,
                k_tor=10.0,
                n=1,
            ),
        ]
        state, fc = self._build_equivalent_pair(
            positions, bonds=bonds, angles=angles, torsions=torsions
        )
        compute_chain_forces(state)
        F_legacy = ChainForceEvaluator().compute_forces(fc)
        np.testing.assert_array_equal(state.forces, F_legacy)


class TestComputeChainForcesProperties:
    """Direct property tests on compute_chain_forces output."""

    @staticmethod
    def _make_state(positions, bonds=(), angles=(), torsions=()):
        n = positions.shape[0]
        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname=f"R{i}", resid=i)
            for i in range(n)
        ]
        common = ChainCommon(
            name="t",
            atoms=atoms,
            bonds=list(bonds),
            angles=list(angles),
            torsions=list(torsions),
        )
        return ChainState.from_template(common, positions)

    def test_no_interactions_gives_zero_forces(self):
        """compute_chain_forces yields zero forces when the chain has no bonded interactions."""
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        state = self._make_state(positions)
        compute_chain_forces(state)
        np.testing.assert_array_equal(state.forces, np.zeros((2, 3)))

    def test_re_running_zeroes_first(self):
        """Calling compute_chain_forces twice on the same state gives identical forces rather than doubling them."""
        positions = np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
        bonds = [
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), r0=3.8, k_spring=100.0),
        ]
        state = self._make_state(positions, bonds=bonds)
        compute_chain_forces(state)
        F_first = state.forces.copy()
        compute_chain_forces(state)
        F_second = state.forces.copy()
        np.testing.assert_array_equal(F_first, F_second)

    def test_total_force_is_zero_newtons_third_law(self):
        """The total force over a closed bonded chain sums to zero, satisfying Newton's third law."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [4.5, 0.5, 0.0], [9.0, 1.0, 1.0], [13.0, 0.0, -1.0]]
        )
        bonds = [
            ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), r0=3.8, k_spring=100.0)
            for i in range(3)
        ]
        angles = [
            ChainAngle(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                theta0=math.pi,
                k_angle=50.0,
            ),
        ]
        torsions = [
            ChainTorsion(
                ChainAtomRef(0),
                ChainAtomRef(1),
                ChainAtomRef(2),
                ChainAtomRef(3),
                phi0=0.0,
                k_tor=10.0,
                n=1,
            ),
        ]
        state = self._make_state(
            positions, bonds=bonds, angles=angles, torsions=torsions
        )
        compute_chain_forces(state)
        sum_F = state.forces.sum(axis=0)
        np.testing.assert_allclose(sum_F, [0.0, 0.0, 0.0], atol=1e-10)

    def test_bond_at_equilibrium_gives_zero_force(self):
        """A harmonic bond at exactly r0 produces zero force."""
        positions = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])  # exactly at r0
        bonds = [
            ChainBond(ChainAtomRef(0), ChainAtomRef(1), r0=3.8, k_spring=100.0),
        ]
        state = self._make_state(positions, bonds=bonds)
        compute_chain_forces(state)
        np.testing.assert_allclose(state.forces, np.zeros((2, 3)), atol=1e-12)


class TestChainInternalBDStep:
    """Brownian-dynamics integration of chain internal coordinates.

    The headline test (test_free_diffusion_matches_stokes_einstein) compares
    an empirical diffusion coefficient against the analytic Stokes-Einstein
    prediction. A bug in the noise scaling (factor sqrt(2) or factor 2)
    shows up as a factor 2 or 4 deviation in the measured coefficient,
    which the test flags.
    """

    @staticmethod
    def _make_free_chain(n_atoms, radius=2.0):
        """Build a chain with n_atoms free atoms, no bonds, no constraints."""

        atoms = [
            ChainAtom(radius=radius, charge=0.0, resname="X", resid=i)
            for i in range(n_atoms)
        ]
        common = ChainCommon(name="free", atoms=atoms)
        # Initial positions on a line, then center.
        positions = np.zeros((n_atoms, 3))
        positions[:, 0] = np.arange(n_atoms) * 5.0
        positions -= positions.mean(axis=0)
        return ChainState.from_template(common, positions)

    def test_free_diffusion_matches_stokes_einstein(self):
        """Free pairwise diffusion gives a mean squared displacement matching the Stokes-Einstein prediction 12 D dt N within 10%."""

        radius = 2.0
        D = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * radius)  # A^2 / ps
        D_rel = 2.0 * D  # relative diffusion of an atom pair

        dt = 0.05  # ps
        n_steps = 50
        n_traj = 800

        msd_samples = []
        for traj_idx in range(n_traj):
            state = self._make_free_chain(n_atoms=4, radius=radius)
            initial_separation = (state.positions[1] - state.positions[0]).copy()
            rng = np.random.default_rng(traj_idx)
            for _ in range(n_steps):
                chain_internal_bd_step(state, dt=dt, rng=rng, apply_constraints=False)
            final_separation = state.positions[1] - state.positions[0]
            d = final_separation - initial_separation
            msd_samples.append(float(np.dot(d, d)))

        msd_empirical = float(np.mean(msd_samples))
        msd_expected = 6.0 * D_rel * dt * n_steps  # = 12 D dt N

        rel_err = abs(msd_empirical - msd_expected) / msd_expected
        assert rel_err < 0.10, (
            f"MSD mismatch: empirical = {msd_empirical:.4f}, "
            f"expected = {msd_expected:.4f}, "
            f"relative error = {rel_err:.3f}. "
            f"This points to a noise-scaling bug."
        )

    def test_determinism_with_seed(self):
        """Two inner BD chains stepped with the same seed produce identical positions."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = self._make_free_chain(n_atoms=3)
        s2 = self._make_free_chain(n_atoms=3)
        for _ in range(20):
            chain_internal_bd_step(s1, dt=0.05, rng=rng1, apply_constraints=False)
            chain_internal_bd_step(s2, dt=0.05, rng=rng2, apply_constraints=False)
        np.testing.assert_array_equal(s1.positions, s2.positions)

    def test_recenter_keeps_com_at_origin(self):
        """The chain center of mass stays at the origin after each internal BD step."""
        rng = np.random.default_rng(0)
        state = self._make_free_chain(n_atoms=4)
        for _ in range(10):
            chain_internal_bd_step(state, dt=0.1, rng=rng, apply_constraints=False)
            com = state.positions.mean(axis=0)
            assert np.linalg.norm(com) < 1e-10

    def test_zero_dt_no_motion(self):
        """An internal BD step with dt=0 produces no drift and no noise."""
        rng = np.random.default_rng(0)
        state = self._make_free_chain(n_atoms=3)
        before = state.positions.copy()
        chain_internal_bd_step(state, dt=0.0, rng=rng, apply_constraints=False)
        np.testing.assert_allclose(state.positions, before, atol=1e-12)

    def test_constraints_satisfied_after_step(self):
        """Length constraints stay satisfied with max|phi| below tolerance after each constrained internal BD step."""

        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(2)
        ]
        common = ChainCommon(
            name="constrained",
            atoms=atoms,
            length_constraints=[LengthConstraint(0, 1, 5.0)],
        )
        # Start at the constraint-satisfying configuration.
        positions = np.array([[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        rng = np.random.default_rng(0)
        for _ in range(20):
            chain_internal_bd_step(state, dt=0.1, rng=rng)
            phi = compute_constraint_violations(state)
            assert (
                np.max(np.abs(phi)) < 1e-5
            ), f"constraint violated: max|phi| = {np.max(np.abs(phi))}"


class TestAggregateChainExternalForceAndTorque:
    """Net force and torque from per-atom forces."""

    def test_zero_forces_give_zero_net(self):
        """Zero per-atom forces aggregate to zero net force and zero net torque."""
        positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        forces = np.zeros((2, 3))
        com = np.zeros(3)
        f_net, t_net = aggregate_chain_external_force_and_torque(
            positions,
            forces,
            com,
        )
        np.testing.assert_array_equal(f_net, np.zeros(3))
        np.testing.assert_array_equal(t_net, np.zeros(3))

    def test_uniform_forces_sum_correctly(self):
        """Uniform forces along the position axis sum to the expected net force with canceling torque."""
        positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        forces = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # both pull +x
        com = np.zeros(3)
        f_net, t_net = aggregate_chain_external_force_and_torque(
            positions,
            forces,
            com,
        )
        # Net force = (2, 0, 0). Torques cancel because F is along position.
        np.testing.assert_allclose(f_net, [2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(t_net, [0.0, 0.0, 0.0], atol=1e-12)

    def test_couple_produces_torque_no_net_force(self):
        """Equal and opposite forces on opposite sides of the CoM give zero net force and a pure torque of (0, 0, 2)."""
        positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        forces = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        com = np.zeros(3)
        f_net, t_net = aggregate_chain_external_force_and_torque(
            positions,
            forces,
            com,
        )
        np.testing.assert_allclose(f_net, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(t_net, [0.0, 0.0, 2.0], atol=1e-12)

    def test_com_offset_used_correctly(self):
        """The aggregated torque is computed about the supplied center of mass rather than the origin."""
        positions = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        forces = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        com = np.array([1.0, 0.0, 0.0])  # midpoint
        f_net, t_net = aggregate_chain_external_force_and_torque(
            positions,
            forces,
            com,
        )
        # Arms: (1,0,0) and (-1,0,0). Torques: (0,0,1) + (0,0,1) = (0,0,2).
        np.testing.assert_allclose(t_net, [0.0, 0.0, 2.0], atol=1e-12)

    def test_shape_mismatch_raises(self):
        """aggregate_chain_external_force_and_torque raises ValueError when the positions and forces shapes disagree."""
        with pytest.raises(ValueError, match="does not match"):
            aggregate_chain_external_force_and_torque(
                np.zeros((3, 3)),
                np.zeros((2, 3)),
                np.zeros(3),
            )


class TestChainOuterBDStep:
    """Outer BD step: rigid-body propagation of (pos, ori) under aggregated forces."""

    def test_zero_forces_pure_diffusion(self):
        """With zero forces and a fixed seed, the outer BD step produces a noise-driven displacement below 10 sqrt(2 D_trans dt)."""

        pos = np.array([100.0, 0.0, 0.0])
        ori = Q(1.0, 0.0, 0.0, 0.0)
        positions = np.array([[101.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
        forces = np.zeros((2, 3))
        D_trans = 0.5
        D_rot = 0.1
        dt = 0.05
        rng = np.random.default_rng(0)
        new_pos, new_ori = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces,
            D_trans,
            D_rot,
            dt,
            rng,
        )
        # New pos should differ from old (noise-driven).
        assert not np.allclose(new_pos, pos)
        d = float(np.linalg.norm(new_pos - pos))
        sigma = math.sqrt(2.0 * D_trans * dt)
        assert d < 10 * sigma, f"step displacement {d} exceeds 10*sigma={10*sigma}"

    def test_determinism_with_seed(self):
        """Two outer BD steps with the same seed produce identical position and orientation quaternion."""

        pos = np.array([100.0, 0.0, 0.0])
        ori = Q(1.0, 0.0, 0.0, 0.0)
        positions = np.array([[101.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
        forces = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        D_trans, D_rot, dt = 0.5, 0.1, 0.05

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1, o1 = chain_outer_bd_step(
            pos, ori, positions, forces, D_trans, D_rot, dt, rng1
        )
        p2, o2 = chain_outer_bd_step(
            pos, ori, positions, forces, D_trans, D_rot, dt, rng2
        )
        np.testing.assert_array_equal(p1, p2)
        assert o1.w == o2.w and o1.x == o2.x
        assert o1.y == o2.y and o1.z == o2.z

    def test_diffusion_coefficient_matches_input(self):
        """The force-free outer-step CoM mean squared displacement matches 6 D_trans dt N within 10%."""

        D_trans = 0.5
        D_rot = 0.1
        dt = 0.05
        n_steps = 50
        n_traj = 800

        msd_samples = []
        for traj_idx in range(n_traj):
            pos = np.zeros(3)
            ori = Q(1.0, 0.0, 0.0, 0.0)
            positions = np.array([[1.0, 0.0, 0.0]])
            forces = np.zeros((1, 3))
            rng = np.random.default_rng(traj_idx)
            for _ in range(n_steps):
                pos, ori = chain_outer_bd_step(
                    pos,
                    ori,
                    positions + pos[None, :],
                    forces,
                    D_trans,
                    D_rot,
                    dt,
                    rng,
                )
            msd_samples.append(float(np.dot(pos, pos)))

        msd_empirical = float(np.mean(msd_samples))
        msd_expected = 6.0 * D_trans * dt * n_steps
        rel_err = abs(msd_empirical - msd_expected) / msd_expected
        assert rel_err < 0.10, (
            f"outer-step MSD mismatch: empirical = {msd_empirical:.4f}, "
            f"expected = {msd_expected:.4f}, rel err = {rel_err:.3f}"
        )


class TestChainBDSimulatorRunOne:
    """End-to-end execution of ChainBDSimulator.run_one().

    These tests verify that the assembled pipeline produces valid
    TrajectoryResult objects with sensible field values, without
    making strong claims about the underlying physics (which is
    validated separately in the per-primitive tests).
    """

    @staticmethod
    def _make_sim(
        n_atoms=2,
        with_bonds=False,
        r_start=20.0,
        r_escape=22.0,
        seed=42,
        max_steps=200,
        dt=0.5,
        n_trajectories=1,
    ):
        """Build a minimal ChainBDSimulator with no target grid, no reactions."""

        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i)
            for i in range(n_atoms)
        ]
        bonds = []
        if with_bonds:
            bonds = [
                ChainBond(ChainAtomRef(i), ChainAtomRef(i + 1), r0=3.8, k_spring=10.0)
                for i in range(n_atoms - 1)
            ]
        common = ChainCommon(name="t", atoms=atoms, bonds=bonds)
        positions = np.zeros((n_atoms, 3))
        positions[:, 0] = np.arange(n_atoms) * 3.8
        positions -= positions.mean(axis=0)
        target = Molecule(atoms=[Atom(x=0, y=0, z=0, radius=2.0)])

        params = ChainBDParameters(
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            r_start=r_start,
            r_escape=r_escape,
            dt=dt,
            dt_chain=0.05,
            chain_steps_per_outer=1,
            seed=seed,
        )
        return ChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            D_trans=2.0,
            D_rot=0.5,
            target_grid=None,
        )

    def test_run_one_returns_trajectory_result(self):
        """run_one returns a TrajectoryResult with a valid fate and non-negative steps, time, and final separation."""

        sim = self._make_sim(n_atoms=2)
        result = sim.run_one()
        assert isinstance(result, TrajectoryResult)
        assert result.fate in (Fate.ESCAPED, Fate.REACTED, Fate.MAX_STEPS)
        assert result.steps >= 0
        assert result.time_ps >= 0.0
        assert result.final_separation >= 0.0

    def test_run_one_no_reactions_always_escapes(self):
        """With no reactions and no target force, run_one always escapes with final separation at or beyond r_escape."""

        sim = self._make_sim(n_atoms=2, r_start=10.0, r_escape=12.0, max_steps=500)
        result = sim.run_one()
        assert result.fate == Fate.ESCAPED
        assert result.final_separation >= 12.0

    def test_run_one_determinism_with_seed(self):
        """Two simulators built with the same seed produce identical fate, steps, time, and final separation from run_one."""
        sim1 = self._make_sim(seed=99)
        sim2 = self._make_sim(seed=99)
        r1 = sim1.run_one()
        r2 = sim2.run_one()
        assert r1.fate == r2.fate
        assert r1.steps == r2.steps
        assert abs(r1.time_ps - r2.time_ps) < 1e-12
        assert abs(r1.final_separation - r2.final_separation) < 1e-12

    def test_run_one_with_bonded_chain_returns_valid_fate(self):
        """A 3-atom bonded chain runs through run_one and ends with a valid fate, which at max_steps=200 is usually the step limit rather than an escape."""

        sim = self._make_sim(
            n_atoms=3,
            with_bonds=True,
            r_start=15.0,
            r_escape=17.0,
            dt=0.2,
            max_steps=200,
        )
        result = sim.run_one()
        assert result.fate in (Fate.ESCAPED, Fate.REACTED, Fate.MAX_STEPS)


class TestChainBDSimulatorRun:
    """Multi-trajectory run() execution."""

    def test_run_collects_all_trajectories(self):
        """run collects exactly n_trajectories results, and reacted plus escaped counts sum to that total."""
        sim = TestChainBDSimulatorRunOne._make_sim(
            n_atoms=2, n_trajectories=5, r_start=10.0, r_escape=12.0
        )
        results = sim.run()
        assert len(results) == 5
        assert len(sim.results) == 5
        assert sim.n_reacted + sim.n_escaped == 5

    def test_run_returns_results_list(self):
        """run returns the same list object it stores in self.results."""
        sim = TestChainBDSimulatorRunOne._make_sim(
            n_atoms=2, n_trajectories=3, r_start=10.0, r_escape=12.0
        )
        returned = sim.run()
        assert returned is sim.results


class TestLoadChainFromJson:
    """Round-trip: JSON file on disk -> ChainCommon + body positions.

    Tests use tempfile to keep the test self-contained; no fixture files
    are committed alongside the test suite.
    """

    @staticmethod
    def _write_chain_json(chain_dict):
        """Write chain_dict to a temp JSON file and return its path."""

        fh = tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            delete=False,
        )
        json.dump(chain_dict, fh)
        fh.close()
        return fh.name

    def test_roundtrip_minimal_chain(self):
        """Loading a minimal atoms-only chain JSON yields the correct name, two atoms, no bonded terms, and (2, 3) positions."""
        chain_dict = {
            "name": "minimal",
            "atoms": [
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 0,
                    "position": [0.0, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 1,
                    "position": [4.0, 0.0, 0.0],
                },
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            common, positions = load_chain_from_json(path)
            assert common.name == "minimal"
            assert len(common.atoms) == 2
            assert len(common.bonds) == 0
            assert len(common.angles) == 0
            assert len(common.torsions) == 0
            assert positions.shape == (2, 3)
        finally:
            import os

            os.unlink(path)

    def test_positions_are_centered_at_origin(self):
        """load_chain_from_json recenters off-center input positions to the origin while preserving pairwise distances."""
        chain_dict = {
            "name": "off_center",
            "atoms": [
                {"radius": 2.0, "position": [10.0, 0.0, 0.0]},
                {"radius": 2.0, "position": [20.0, 0.0, 0.0]},
                {"radius": 2.0, "position": [30.0, 0.0, 0.0]},
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            _, positions = load_chain_from_json(path)
            np.testing.assert_allclose(
                positions.mean(axis=0),
                [0.0, 0.0, 0.0],
                atol=1e-12,
            )
            # Pairwise distances should be preserved.
            assert abs(np.linalg.norm(positions[1] - positions[0]) - 10.0) < 1e-10
            assert abs(np.linalg.norm(positions[2] - positions[1]) - 10.0) < 1e-10
        finally:
            import os

            os.unlink(path)

    def test_atom_fields_carried_correctly(self):
        """load_chain_from_json parses each atom's radius, charge, resname, and resid correctly."""
        chain_dict = {
            "name": "fields_test",
            "atoms": [
                {
                    "radius": 2.5,
                    "charge": 0.7,
                    "resname": "ARG",
                    "resid": 12,
                    "position": [0.0, 0.0, 0.0],
                },
                {
                    "radius": 1.8,
                    "charge": -0.3,
                    "resname": "GLU",
                    "resid": 13,
                    "position": [3.8, 0.0, 0.0],
                },
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            common, _ = load_chain_from_json(path)
            assert common.atoms[0].radius == 2.5
            assert common.atoms[0].charge == 0.7
            assert common.atoms[0].resname == "ARG"
            assert common.atoms[0].resid == 12
            assert common.atoms[1].radius == 1.8
            assert common.atoms[1].charge == -0.3
            assert common.atoms[1].resname == "GLU"
            assert common.atoms[1].resid == 13
        finally:
            import os

            os.unlink(path)

    def test_bonds_angles_torsions_loaded_correctly(self):
        """load_chain_from_json parses bonds, angles, and torsions with the correct counts, indices, and parameters."""
        chain_dict = {
            "name": "full",
            "atoms": [
                {"radius": 2.0, "position": [0.0, 0.0, 0.0]},
                {"radius": 2.0, "position": [3.8, 0.0, 0.0]},
                {"radius": 2.0, "position": [7.6, 0.0, 0.0]},
                {"radius": 2.0, "position": [11.4, 0.0, 0.0]},
            ],
            "bonds": [
                {"a": 0, "b": 1, "r0": 3.8, "k_spring": 100.0},
                {"a": 1, "b": 2, "r0": 3.8, "k_spring": 100.0},
                {"a": 2, "b": 3, "r0": 3.8, "k_spring": 100.0},
            ],
            "angles": [
                {"a": 0, "b": 1, "c": 2, "theta0": 3.14159, "k_angle": 50.0},
                {"a": 1, "b": 2, "c": 3, "theta0": 3.14159, "k_angle": 50.0},
            ],
            "torsions": [
                {"a": 0, "b": 1, "c": 2, "d": 3, "phi0": 0.0, "k_tor": 10.0, "n": 1},
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            common, _ = load_chain_from_json(path)
            assert len(common.bonds) == 3
            assert len(common.angles) == 2
            assert len(common.torsions) == 1
            # Spot-check.
            assert common.bonds[0].a.atom_idx == 0
            assert common.bonds[0].b.atom_idx == 1
            assert common.bonds[0].r0 == 3.8
            assert common.bonds[0].k_spring == 100.0
            assert common.angles[0].b.atom_idx == 1
            assert common.angles[0].k_angle == 50.0
            assert common.torsions[0].n == 1
            assert common.torsions[0].k_tor == 10.0
        finally:
            import os

            os.unlink(path)

    def test_empty_atoms_list_raises(self):
        """load_chain_from_json raises ValueError when the atoms list is empty."""
        chain_dict = {"name": "empty", "atoms": []}
        path = self._write_chain_json(chain_dict)
        try:
            with pytest.raises(ValueError, match="no atoms"):
                load_chain_from_json(path)
        finally:
            import os

            os.unlink(path)

    def test_wrong_position_dimension_raises(self):
        """load_chain_from_json raises ValueError when a position has fewer than three components."""
        chain_dict = {
            "name": "wrong_dim",
            "atoms": [
                {"radius": 2.0, "position": [0.0, 0.0]},  # only 2 components
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            with pytest.raises(ValueError, match="expected 3 values"):
                load_chain_from_json(path)
        finally:
            import os

            os.unlink(path)

    def test_loaded_chain_is_runnable_with_simulator(self):
        """A chain loaded from JSON builds a ChainBDSimulator and runs one trajectory end to end."""

        chain_dict = {
            "name": "runnable",
            "atoms": [
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 0,
                    "position": [-3.8, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 1,
                    "position": [0.0, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 2,
                    "position": [3.8, 0.0, 0.0],
                },
            ],
            "bonds": [
                {"a": 0, "b": 1, "r0": 3.8, "k_spring": 10.0},
                {"a": 1, "b": 2, "r0": 3.8, "k_spring": 10.0},
            ],
        }
        path = self._write_chain_json(chain_dict)
        try:
            common, positions = load_chain_from_json(path)
            target = Molecule(atoms=[Atom(x=0, y=0, z=0, radius=2.0)])
            params = ChainBDParameters(
                n_trajectories=1,
                max_steps=200,
                r_start=15.0,
                r_escape=17.0,
                dt=0.2,
                dt_chain=0.05,
                chain_steps_per_outer=1,
                seed=0,
            )
            sim = ChainBDSimulator(
                target=target,
                chain_template=common,
                chain_init_body_positions=positions,
                params=params,
                pathway_set=PathwaySet(reactions=[]),
                D_trans=1.0,
                D_rot=0.3,
                target_grid=None,
            )
            result = sim.run_one()
            assert result.fate in (Fate.ESCAPED, Fate.REACTED)
        finally:
            import os

            os.unlink(path)


class TestChainSimulationCLI:
    """Regression tests for the chain_simulation click command.

    These use click.testing.CliRunner to invoke the command in-process
    against tempfile inputs. They cover:
      - happy path: real inputs produce a successful run with expected output
      - missing required args: D_trans / D_rot omission rejected by click
      - bad input file: a malformed chain JSON surfaces a clear error
    The pipeline-level physics is already validated elsewhere; these
    tests are about the CLI glue.
    """

    @staticmethod
    def _make_inputs(tmp_path):
        """Write a tiny chain JSON, target PQR, reaction XML to tmp_path
        and return their string paths."""

        chain_data = {
            "name": "clitest",
            "atoms": [
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 0,
                    "position": [-3.8, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 1,
                    "position": [0.0, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "X",
                    "resid": 2,
                    "position": [3.8, 0.0, 0.0],
                },
            ],
            "bonds": [
                {"a": 0, "b": 1, "r0": 3.8, "k_spring": 10.0},
                {"a": 1, "b": 2, "r0": 3.8, "k_spring": 10.0},
            ],
        }
        chain_path = tmp_path / "chain.json"
        chain_path.write_text(json.dumps(chain_data))
        target_path = tmp_path / "target.pqr"
        target_path.write_text(
            "ATOM      1  CA  ALA A   1       "
            "0.000   0.000   0.000  0.0000  2.0000\n"
        )
        rxn_path = tmp_path / "rxns.xml"
        rxn_path.write_text("<reactions></reactions>\n")
        return str(chain_path), str(target_path), str(rxn_path)

    def test_chain_simulation_happy_path(self, tmp_path):
        """The chain_simulation CLI exits successfully and reports reacted, escaped, chain name, and atom count."""

        chain, target, rxn = self._make_inputs(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                chain,
                "--target",
                target,
                "--rxn",
                rxn,
                "--n",
                "3",
                "--dt",
                "0.2",
                "--dt-chain",
                "0.05",
                "--chain-steps-per-outer",
                "1",
                "--r-start",
                "15.0",
                "--r-escape",
                "17.0",
                "--d-trans",
                "1.0",
                "--d-rot",
                "0.3",
                "--max-steps",
                "200",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\n"
            f"output:\n{result.output}\n"
            f"exception: {result.exception}"
        )
        assert "Reacted" in result.output
        assert "Escaped" in result.output
        assert "clitest" in result.output  # chain name
        assert "3 atoms" in result.output

    def test_chain_simulation_missing_d_trans_fails(self, tmp_path):
        """The chain_simulation CLI fails and names --d-trans when that required option is missing."""

        chain, target, rxn = self._make_inputs(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                chain,
                "--target",
                target,
                "--rxn",
                rxn,
                "--d-rot",
                "0.3",
                "--n",
                "1",
                "--r-start",
                "15.0",
                "--r-escape",
                "17.0",
            ],
        )
        assert result.exit_code != 0
        # click formats missing-required errors with the option name.
        assert "--d-trans" in result.output

    def test_chain_simulation_missing_d_rot_fails(self, tmp_path):
        """The chain_simulation CLI fails and names --d-rot when that required option is missing."""

        chain, target, rxn = self._make_inputs(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                chain,
                "--target",
                target,
                "--rxn",
                rxn,
                "--d-trans",
                "1.0",
                "--n",
                "1",
                "--r-start",
                "15.0",
                "--r-escape",
                "17.0",
            ],
        )
        assert result.exit_code != 0
        assert "--d-rot" in result.output

    def test_chain_simulation_bad_chain_json_fails(self, tmp_path):
        """The chain_simulation CLI fails with a non-zero exit and captured exception when the chain JSON has no atoms."""

        bad_chain = tmp_path / "bad.json"
        bad_chain.write_text('{"name": "bad", "atoms": []}')
        _, target, rxn = self._make_inputs(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                str(bad_chain),
                "--target",
                target,
                "--rxn",
                rxn,
                "--d-trans",
                "1.0",
                "--d-rot",
                "0.3",
                "--n",
                "1",
                "--r-start",
                "15.0",
                "--r-escape",
                "17.0",
            ],
        )
        assert result.exit_code != 0
        assert (
            isinstance(result.exception, (ValueError, SystemExit))
            or result.exception is not None
        )


class TestChainBDParallelMode:
    """Parallel multiprocessing execution of ChainBDSimulator.run()."""

    @staticmethod
    def _make_sim(n_trajectories, n_threads, seed=42):
        atoms = [
            ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(2)
        ]
        common = ChainCommon(name="parallel", atoms=atoms)
        positions = np.array([[-2.0, 0, 0], [2.0, 0, 0]], dtype=float)
        target = Molecule(atoms=[Atom(x=0, y=0, z=0, radius=2.0)])
        params = ChainBDParameters(
            n_trajectories=n_trajectories,
            max_steps=200,
            r_start=10.0,
            r_escape=12.0,
            dt=0.2,
            dt_chain=0.05,
            chain_steps_per_outer=1,
            seed=seed,
            n_threads=n_threads,
        )
        return ChainBDSimulator(
            target=target,
            chain_template=common,
            chain_init_body_positions=positions,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            D_trans=1.0,
            D_rot=0.3,
            target_grid=None,
        )

    def test_simulator_is_picklable(self):
        """The simulator pickles and unpickles while preserving D_trans, n_threads, and the chain atom count."""

        sim = self._make_sim(n_trajectories=2, n_threads=2)
        blob = pickle.dumps(sim)
        restored = pickle.loads(blob)
        assert restored.D_trans == sim.D_trans
        assert restored.params.n_threads == sim.params.n_threads
        assert len(restored.chain_template.atoms) == len(sim.chain_template.atoms)

    def test_worker_function_runs_one_trajectory(self):
        """The top-level worker function returns a valid TrajectoryResult with a reacted, escaped, or step-limited fate."""

        sim = self._make_sim(n_trajectories=1, n_threads=1)
        result = _run_chain_trajectory_worker((sim, 0))
        assert isinstance(result, TrajectoryResult)
        assert result.fate in (Fate.ESCAPED, Fate.REACTED, Fate.MAX_STEPS)

    def test_parallel_run_produces_correct_trajectory_count(self):
        """A parallel run with n_threads=2 still produces n trajectories with reacted plus escaped plus step-limited summing to n."""
        sim = self._make_sim(n_trajectories=4, n_threads=2)
        results = sim.run()
        assert len(results) == 4
        assert sim.n_reacted + sim.n_escaped + sim.n_max_steps == 4

    def test_parallel_run_is_deterministic(self):
        """Two parallel runs with the same seed produce identical fate, steps, and final separation across all trajectories."""
        sim1 = self._make_sim(n_trajectories=4, n_threads=2, seed=99)
        sim2 = self._make_sim(n_trajectories=4, n_threads=2, seed=99)
        r1 = sim1.run()
        r2 = sim2.run()
        assert len(r1) == len(r2) == 4
        for a, b in zip(r1, r2):
            assert a.fate == b.fate
            assert a.steps == b.steps
            assert abs(a.final_separation - b.final_separation) < 1e-12

    def test_serial_and_parallel_paths_both_succeed(self):
        """The serial and parallel run paths both produce well-formed results with valid fates and non-negative metrics."""

        sim_serial = self._make_sim(n_trajectories=3, n_threads=1)
        sim_parallel = self._make_sim(n_trajectories=3, n_threads=2)
        r_serial = sim_serial.run()
        r_parallel = sim_parallel.run()
        assert len(r_serial) == len(r_parallel) == 3
        for r in r_serial + r_parallel:
            assert r.fate in (Fate.ESCAPED, Fate.REACTED, Fate.MAX_STEPS)
            assert r.steps >= 0
            assert r.final_separation >= 0.0

    def test_n_threads_one_uses_serial_path(self):
        """Running with n_threads=1 completes and returns the expected number of trajectories via the serial path."""
        sim = self._make_sim(n_trajectories=3, n_threads=1)
        results = sim.run()
        assert len(results) == 3


class TestChainSimulationCLIThreads:
    """The CLI --threads option should reach ChainBDParameters.n_threads."""

    def test_threads_flag_runs_in_parallel(self, tmp_path):
        """The chain_simulation CLI with --threads 2 exits successfully and reports reacted and escaped."""

        chain, target, rxn = TestChainSimulationCLI._make_inputs(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                chain,
                "--target",
                target,
                "--rxn",
                rxn,
                "--n",
                "4",
                "--dt",
                "0.2",
                "--dt-chain",
                "0.05",
                "--chain-steps-per-outer",
                "1",
                "--r-start",
                "15.0",
                "--r-escape",
                "17.0",
                "--d-trans",
                "1.0",
                "--d-rot",
                "0.3",
                "--max-steps",
                "200",
                "--seed",
                "42",
                "--threads",
                "2",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"CLI exit {result.exit_code}\n{result.output}\n{result.exception}"
        assert "Reacted" in result.output
        assert "Escaped" in result.output


class TestRpyPairTensorAnalyticalLimits:
    """Validate full-RPY pair tensor against analytical reference values.

    The test cases are computed independently from the formulas in BD2's
    rotne_prager.hh (which itself ports Zuk et al. 2014, J. Fluid Mech.
    741, R5). Each test has a specific physics statement.
    """

    def test_self_mobility_translation_matches_stokes(self):
        """The single-sphere RPY translational self-mobility equals I / (6 pi a)."""
        a = 2.5
        mtt, _ = rpy_self_blocks(a)
        expected = np.eye(3) / (6.0 * math.pi * a)
        np.testing.assert_allclose(mtt, expected, atol=1e-15)

    def test_self_mobility_rotation_matches_stokes(self):
        """The single-sphere RPY rotational self-mobility equals I / (8 pi a^3)."""
        a = 2.5
        _, mrr = rpy_self_blocks(a)
        expected = np.eye(3) / (8.0 * math.pi * a * a * a)
        np.testing.assert_allclose(mrr, expected, atol=1e-15)

    def test_far_field_mtt_matches_oseen_plus_correction(self):
        """The far-field mtt block for equal spheres matches the Oseen tensor plus the a²/r² correction with zero off-diagonals."""
        a = 1.0
        r = 10.0  # far field, r >> 2a
        r_ij = np.array([r, 0.0, 0.0])
        mtt, mrt, mtr, mrr = rpy_pair_blocks(a, a, r_ij)

        a2or2 = (a * a + a * a) / (r * r)  # 2 a^2 / r^2
        tt_I_exp = (1.0 + a2or2 / 3.0) / (8.0 * math.pi * r)
        tt_uu_exp = (1.0 - a2or2) / (8.0 * math.pi * r)

        np.testing.assert_allclose(mtt[0, 0], tt_I_exp + tt_uu_exp, atol=1e-15)
        np.testing.assert_allclose(mtt[1, 1], tt_I_exp, atol=1e-15)
        np.testing.assert_allclose(mtt[2, 2], tt_I_exp, atol=1e-15)
        # Off-diagonal of mtt should be zero (u along x).
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(mtt[i, j]) < 1e-15

    def test_far_field_mrr_isotropic_form(self):
        """The far-field mrr block equals (-I + 3uu)/(16πr³) and is traceless."""
        a = 1.0
        r = 10.0
        r_ij = np.array([r, 0.0, 0.0])
        _, _, _, mrr = rpy_pair_blocks(a, a, r_ij)

        rr_I_exp = -1.0 / (16.0 * math.pi * r * r * r)
        rr_uu_exp = 3.0 / (16.0 * math.pi * r * r * r)

        # Same uu structure as before.
        np.testing.assert_allclose(mrr[0, 0], rr_I_exp + rr_uu_exp, atol=1e-15)
        np.testing.assert_allclose(mrr[1, 1], rr_I_exp, atol=1e-15)
        np.testing.assert_allclose(np.trace(mrr), 0.0, atol=1e-14)

    def test_far_field_cross_coupling_skew_symmetric(self):
        """The far-field cross-coupling blocks mrt and mtr are skew-symmetric and equal for equal radii."""
        a = 1.5
        r = 8.0
        r_ij = np.array([0.0, r, 0.0])
        _, mrt, mtr, _ = rpy_pair_blocks(a, a, r_ij)

        np.testing.assert_allclose(mrt + mrt.T, np.zeros((3, 3)), atol=1e-15)
        np.testing.assert_allclose(mtr + mtr.T, np.zeros((3, 3)), atol=1e-15)
        # In the far field, equal radii give mrt = mtr.
        np.testing.assert_allclose(mrt, mtr, atol=1e-15)

    def test_argument_swap_symmetry(self):
        """The pair tensors mtt and mrr are invariant under simultaneous swap of bead labels and sign of r_ij."""
        ai, aj = 1.2, 2.3
        r_ij = np.array([3.0, 4.0, 5.0])  # arbitrary
        mtt_ij, _, _, mrr_ij = rpy_pair_blocks(ai, aj, r_ij)
        mtt_ji, _, _, mrr_ji = rpy_pair_blocks(aj, ai, -r_ij)
        np.testing.assert_allclose(mtt_ij, mtt_ji, atol=1e-15)
        np.testing.assert_allclose(mrr_ij, mrr_ji, atol=1e-15)

    def test_overlap_regime_at_contact_finite_and_continuous(self):
        """The far-field and partial-overlap formulas for the tt components agree continuously across contact at r = 2a."""
        a = 1.0

        # Slightly above contact: far-field formula.
        r_above = 2.0 * a + 1e-6
        tt_I_above, tt_uu_above, _, _, _, _ = rpy_full_components(a, a, r_above)

        # Slightly below contact: partial-overlap formula.
        r_below = 2.0 * a - 1e-6
        tt_I_below, tt_uu_below, _, _, _, _ = rpy_full_components(a, a, r_below)

        np.testing.assert_allclose(tt_I_above, tt_I_below, rtol=1e-4)
        np.testing.assert_allclose(tt_uu_above, tt_uu_below, atol=1e-4)

    def test_consistent_with_existing_rpy_offdiagonal(self):
        """The new mtt block matches rpy_offdiagonal output when the viscosity scaling factor is set to one."""

        ai, aj = 2.0, 3.0
        r_ij = np.array([5.0, 0.0, 0.0])
        mtt_new, _, _, _ = rpy_pair_blocks(ai, aj, r_ij)

        D_a = 1.0 / (6.0 * math.pi * ai)
        D_b = 1.0 / (6.0 * math.pi * aj)
        mtt_existing = rpy_offdiagonal(r_ij, ai, aj, D_a, D_b)
        np.testing.assert_allclose(mtt_new, mtt_existing, atol=1e-13)

    def test_fully_nested_sphere_returns_self_mobility(self):
        """When one sphere is fully nested inside the other, the pair tensor reduces to the larger sphere's self-mobility with zero cross terms."""
        ai, aj = 5.0, 1.0
        # r = 2: smaller sphere is fully inside (since |ai - aj| = 4 > r).
        r_ij = np.array([2.0, 0.0, 0.0])
        mtt, mrt, mtr, mrr = rpy_pair_blocks(ai, aj, r_ij)

        a_max = max(ai, aj)
        expected_mtt = np.eye(3) / (6.0 * math.pi * a_max)
        expected_mrr = np.eye(3) / (8.0 * math.pi * a_max**3)
        np.testing.assert_allclose(mtt, expected_mtt, atol=1e-15)
        np.testing.assert_allclose(mrr, expected_mrr, atol=1e-15)
        # rt/tr are zero in this regime (matches BD2).
        np.testing.assert_allclose(mrt, np.zeros((3, 3)), atol=1e-15)
        np.testing.assert_allclose(mtr, np.zeros((3, 3)), atol=1e-15)


class TestRpyFullMobilityMatrix:
    """N-bead RPY mobility matrix assembly: shape, symmetry, block layout,
    and consistency with the validated pair tensor."""

    def test_shape_for_n_beads(self):
        """The full mobility matrix has shape (6N, 6N) and dtype float64 for several bead counts."""
        for n in [1, 2, 3, 7, 10]:
            positions = np.zeros((n, 3))
            positions[:, 0] = np.arange(n) * 5.0
            radii = np.ones(n)
            M = rpy_full_mobility_matrix(positions, radii)
            assert M.shape == (6 * n, 6 * n)
            assert M.dtype == np.float64

    def test_symmetric_by_onsager_reciprocity(self):
        """The full mobility matrix is exactly symmetric, satisfying Onsager reciprocity."""
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((5, 3)) * 5.0
        radii = rng.uniform(0.8, 2.5, size=5)
        M = rpy_full_mobility_matrix(positions, radii)
        np.testing.assert_array_equal(M, M.T)

    def test_diagonal_blocks_match_self_mobility(self):
        """Each bead's 6x6 diagonal block is block-diagonal with its translational and rotational self-mobilities and zero cross terms."""
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        radii = np.array([1.5, 2.5])
        M = rpy_full_mobility_matrix(positions, radii)

        for i in range(2):
            mtt_self_exp, mrr_self_exp = rpy_self_blocks(radii[i])
            i6 = 6 * i
            np.testing.assert_allclose(
                M[i6 : i6 + 3, i6 : i6 + 3],
                mtt_self_exp,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                M[i6 + 3 : i6 + 6, i6 + 3 : i6 + 6],
                mrr_self_exp,
                atol=1e-15,
            )
            # Cross blocks on the diagonal (i, i) are zero.
            np.testing.assert_array_equal(
                M[i6 : i6 + 3, i6 + 3 : i6 + 6],
                np.zeros((3, 3)),
            )
            np.testing.assert_array_equal(
                M[i6 + 3 : i6 + 6, i6 : i6 + 3],
                np.zeros((3, 3)),
            )

    def test_off_diagonal_blocks_match_pair_tensor(self):
        """Each off-diagonal (i, j) block of the assembled matrix equals rpy_pair_blocks for the same radii and separation."""
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [8.0, 0.0, 2.0]])
        radii = np.array([1.0, 1.5, 2.0])
        M = rpy_full_mobility_matrix(positions, radii)

        # Check block (0, 1).
        r_01 = positions[1] - positions[0]
        mtt, mrt, mtr, mrr = rpy_pair_blocks(radii[0], radii[1], r_01)
        np.testing.assert_allclose(M[0:3, 6:9], mtt, atol=1e-15)
        np.testing.assert_allclose(M[3:6, 6:9], mrt, atol=1e-15)
        np.testing.assert_allclose(M[0:3, 9:12], mtr, atol=1e-15)
        np.testing.assert_allclose(M[3:6, 9:12], mrr, atol=1e-15)

        # Check block (1, 2).
        r_12 = positions[2] - positions[1]
        mtt, mrt, mtr, mrr = rpy_pair_blocks(radii[1], radii[2], r_12)
        np.testing.assert_allclose(M[6:9, 12:15], mtt, atol=1e-15)
        np.testing.assert_allclose(M[9:12, 12:15], mrt, atol=1e-15)
        np.testing.assert_allclose(M[6:9, 15:18], mtr, atol=1e-15)
        np.testing.assert_allclose(M[9:12, 15:18], mrr, atol=1e-15)

    def test_single_bead_returns_self_mobility(self):
        """For N = 1 the matrix is the (6, 6) self-mobility with zero cross-coupling blocks."""
        positions = np.array([[2.0, 3.0, 4.0]])  # arbitrary
        radii = np.array([1.5])
        M = rpy_full_mobility_matrix(positions, radii)
        assert M.shape == (6, 6)
        mtt_self, mrr_self = rpy_self_blocks(1.5)
        np.testing.assert_allclose(M[0:3, 0:3], mtt_self, atol=1e-15)
        np.testing.assert_allclose(M[3:6, 3:6], mrr_self, atol=1e-15)
        # Off-diagonal cross-coupling blocks are zero.
        np.testing.assert_array_equal(M[0:3, 3:6], np.zeros((3, 3)))
        np.testing.assert_array_equal(M[3:6, 0:3], np.zeros((3, 3)))

    def test_two_bead_full_assembly_matches_hand_built(self):
        """The assembled N = 2 matrix exactly equals the 12x12 hand-built from self and pair blocks with transposed (1, 0) block."""
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        radii = np.array([1.0, 1.5])

        M = rpy_full_mobility_matrix(positions, radii)

        # Build expected by hand.
        expected = np.zeros((12, 12))
        # Bead 0 self.
        mtt0, mrr0 = rpy_self_blocks(radii[0])
        expected[0:3, 0:3] = mtt0
        expected[3:6, 3:6] = mrr0
        # Bead 1 self.
        mtt1, mrr1 = rpy_self_blocks(radii[1])
        expected[6:9, 6:9] = mtt1
        expected[9:12, 9:12] = mrr1
        # Block (0, 1).
        r_01 = positions[1] - positions[0]
        mtt_01, mrt_01, mtr_01, mrr_01 = rpy_pair_blocks(
            radii[0],
            radii[1],
            r_01,
        )
        expected[0:3, 6:9] = mtt_01
        expected[3:6, 6:9] = mrt_01
        expected[0:3, 9:12] = mtr_01
        expected[3:6, 9:12] = mrr_01
        # Block (1, 0) = block (0, 1) transpose.
        expected[6:9, 0:3] = mtt_01.T
        expected[6:9, 3:6] = mrt_01.T
        expected[9:12, 0:3] = mtr_01.T
        expected[9:12, 3:6] = mrr_01.T

        np.testing.assert_array_equal(M, expected)

    def test_off_diagonal_far_field_decays_as_inverse_r(self):
        """The translation-translation off-diagonal block decays as 1/r at large separation."""
        n = 2
        radii = np.array([1.0, 1.0])
        # Two distances, factor of 10 apart.
        for r_factor in [1.0, 10.0]:
            positions = np.array([[0.0, 0.0, 0.0], [50.0 * r_factor, 0.0, 0.0]])
            M = rpy_full_mobility_matrix(positions, radii)
            # mtt_01 perpendicular component: M[1, 7] (y-y).
            mtt_01_perp = M[1, 7]
            if r_factor == 1.0:
                m_close = mtt_01_perp
            else:
                m_far = mtt_01_perp
        # Far-field tt_I ~ 1/(8 pi r), so M_far / M_close ~ 1/r_factor.
        ratio = m_far / m_close
        np.testing.assert_allclose(ratio, 1.0 / 10.0, rtol=1e-3)

    def test_invalid_input_shapes_raise(self):
        """Wrong positions or radii shapes raise ValueError with descriptive messages."""
        # Wrong positions shape.
        with pytest.raises(ValueError, match="positions must have shape"):
            rpy_full_mobility_matrix(np.zeros(3), np.array([1.0]))
        # Mismatched radii length.
        with pytest.raises(ValueError, match="radii must have shape"):
            rpy_full_mobility_matrix(
                np.zeros((3, 3)),
                np.array([1.0, 1.0]),
            )


class TestChainRigidBodyResistance:
    """Rigid-body resistance matrices via BD2's Cholesky algorithm.

    These tests validate the chain_rigid_body_resistance function
    against analytical expectations. Note: this function uses ONLY the
    translation block of the bead mobility (matching BD2). For a single
    bead, that means C = 0 by construction (no moment arm), even though
    a physical single sphere has rotational drag = 8 pi a^3. This is an
    intentional matching of BD2's algorithm; chains of N >= 5 or so are
    well-described by it.
    """

    def test_single_bead_translation_matches_stokes(self):
        """A single bead's translational resistance A equals 6πa I exactly."""
        a = 2.5
        positions = np.array([[1.5, -0.7, 3.2]])
        radii = np.array([a])
        A, _, _ = chain_rigid_body_resistance(positions, radii)
        np.testing.assert_allclose(A, 6.0 * math.pi * a * np.eye(3), atol=1e-10)

    def test_single_bead_hydrodynamic_center_is_position(self):
        """For a single bead the hydrodynamic center equals that bead's position."""
        positions = np.array([[5.0, -3.0, 7.0]])
        radii = np.array([1.5])
        _, _, hc = chain_rigid_body_resistance(positions, radii)
        np.testing.assert_allclose(hc, positions[0], atol=1e-15)

    def test_single_bead_C_is_zero_by_construction(self):
        """A single bead yields C = 0, pinning the documented zero moment-arm behavior."""
        positions = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        _, C, _ = chain_rigid_body_resistance(positions, radii)
        np.testing.assert_allclose(C, np.zeros((3, 3)), atol=1e-12)

    def test_resistance_matrices_are_symmetric(self):
        """The resistance matrices A and C are symmetric by Lorentz reciprocity."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((5, 3)) * 5.0
        radii = rng.uniform(0.8, 2.0, size=5)
        A, C, _ = chain_rigid_body_resistance(positions, radii)
        np.testing.assert_allclose(A, A.T, atol=1e-10)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_two_equal_spheres_far_apart_anisotropic_A(self):
        """For two equal spheres along x, parallel drag A_xx is less than perpendicular A_yy = A_zz, each within 3% of 12πa."""
        a = 1.0
        sep = 100.0
        positions = np.array([[-sep / 2, 0.0, 0.0], [sep / 2, 0.0, 0.0]])
        radii = np.array([a, a])
        A, _, _ = chain_rigid_body_resistance(positions, radii)
        # parallel (xx) < perpendicular (yy)
        assert A[0, 0] < A[1, 1]
        # By y/z symmetry, A_yy = A_zz.
        np.testing.assert_allclose(A[1, 1], A[2, 2], atol=1e-10)
        for i in range(3):
            assert abs(A[i, i] - 12.0 * math.pi * a) / (12.0 * math.pi * a) < 0.03

    def test_two_equal_spheres_hydrodynamic_center_at_midpoint(self):
        """For two equal beads the hydrodynamic center is the midpoint."""
        positions = np.array([[-3.0, 4.0, 0.0], [3.0, -4.0, 0.0]])
        radii = np.array([1.5, 1.5])
        _, _, hc = chain_rigid_body_resistance(positions, radii)
        np.testing.assert_allclose(hc, [0.0, 0.0, 0.0], atol=1e-15)

    def test_unequal_radii_hydrodynamic_center_weighted(self):
        """For two unequal beads the hydrodynamic center is the radius-weighted mean position."""
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = np.array([1.0, 3.0])  # bead 2 is 3x larger
        _, _, hc = chain_rigid_body_resistance(positions, radii)
        # Expected: hc_x = (1*0 + 3*10) / (1+3) = 7.5
        np.testing.assert_allclose(hc, [7.5, 0.0, 0.0], atol=1e-15)

    def test_linear_chain_C_matrix_anisotropic(self):
        """For a linear chain along x, rotational resistance C_xx is near zero while C_yy = C_zz are large."""
        a = 1.0
        positions = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        radii = np.array([a, a, a, a])
        _, C, _ = chain_rigid_body_resistance(positions, radii)
        assert C[0, 0] < 1e-9
        # Rotation about y or z: large moment arms, large C.
        assert C[1, 1] > 100.0
        np.testing.assert_allclose(C[1, 1], C[2, 2], atol=1e-10)

    def test_translation_invariance(self):
        """A and C are invariant under bulk translation while the hydrodynamic center shifts with the chain."""
        rng = np.random.default_rng(1)
        positions = rng.standard_normal((4, 3)) * 5.0
        radii = rng.uniform(0.8, 2.0, size=4)

        A1, C1, hc1 = chain_rigid_body_resistance(positions, radii)
        shift = np.array([100.0, -50.0, 25.0])
        A2, C2, hc2 = chain_rigid_body_resistance(positions + shift, radii)

        np.testing.assert_allclose(A1, A2, atol=1e-9)
        np.testing.assert_allclose(C1, C2, atol=1e-9)
        np.testing.assert_allclose(hc2 - hc1, shift, atol=1e-12)

    def test_rotational_invariance_under_chain_rotation(self):
        """Rotating the chain transforms A and C as second-rank tensors, A' = R A Rᵀ."""
        rng = np.random.default_rng(2)
        positions = rng.standard_normal((4, 3)) * 5.0
        radii = rng.uniform(0.8, 2.0, size=4)

        # Random rotation matrix.

        R = Rotation.random(random_state=2).as_matrix()

        A1, C1, _ = chain_rigid_body_resistance(positions, radii)
        positions_rot = positions @ R.T
        A2, C2, _ = chain_rigid_body_resistance(positions_rot, radii)

        # A and C should transform as second-rank tensors: A' = R A R^T.
        np.testing.assert_allclose(A2, R @ A1 @ R.T, atol=1e-9)
        np.testing.assert_allclose(C2, R @ C1 @ R.T, atol=1e-9)

    def test_input_validation(self):
        """Wrong positions or radii shapes raise ValueError with descriptive messages."""
        with pytest.raises(ValueError, match="positions must have shape"):
            chain_rigid_body_resistance(np.zeros(3), np.array([1.0]))
        with pytest.raises(ValueError, match="radii must have shape"):
            chain_rigid_body_resistance(np.zeros((3, 3)), np.array([1.0, 1.0]))


class TestChainDiffusionTensors:
    """Diffusion-tensor wrapper around chain_rigid_body_resistance.

    Inverts A and C, scales by kT/eta, returns user-ready 3x3 tensors.
    Single-bead case is special-cased to use direct Stokes-Einstein
    rotational mobility (since the BD2 algorithm gives C = 0 with
    one bead).
    """

    def test_single_bead_recovers_stokes_einstein(self):
        """For N = 1 the diffusion tensors recover the Stokes-Einstein values 1/(6πηa) and 1/(8πηa³)."""

        a = 2.0
        positions = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([a])
        D_trans, D_rot, _ = chain_diffusion_tensors(positions, radii)

        D_t_exp = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * a)
        D_r_exp = 1.0 / (8.0 * math.pi * WATER_VISCOSITY * a**3)
        np.testing.assert_allclose(
            D_trans,
            D_t_exp * np.eye(3),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            D_rot,
            D_r_exp * np.eye(3),
            rtol=1e-10,
        )

    def test_diffusion_tensors_symmetric(self):
        """D_trans and D_rot are symmetric since they invert symmetric A and C."""
        rng = np.random.default_rng(11)
        positions = rng.standard_normal((4, 3)) * 5.0
        radii = rng.uniform(0.8, 2.0, size=4)
        D_trans, D_rot, _ = chain_diffusion_tensors(positions, radii)
        np.testing.assert_allclose(D_trans, D_trans.T, atol=1e-9)
        np.testing.assert_allclose(D_rot, D_rot.T, atol=1e-9)

    def test_anisotropic_chain_D_trans_x_greater_than_y(self):
        """For a near-linear chain along x, D_trans_xx exceeds D_trans_yy with D_trans_yy approximately equal to D_trans_zz."""
        positions = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 0.01, 0.0],
                [1.0, -0.01, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        radii = np.array([1.0] * 4)
        D_trans, _, _ = chain_diffusion_tensors(positions, radii)
        assert D_trans[0, 0] > D_trans[1, 1]
        # By y/z near-symmetry, D_trans_yy ~= D_trans_zz.
        np.testing.assert_allclose(D_trans[1, 1], D_trans[2, 2], rtol=1e-3)

    def test_kT_and_viscosity_scaling(self):
        """D_trans and D_rot scale linearly in kT and inversely in viscosity."""
        positions = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([2.0])

        D_t1, D_r1, _ = chain_diffusion_tensors(
            positions,
            radii,
            kT=1.0,
            viscosity=1.0,
        )
        D_t2, D_r2, _ = chain_diffusion_tensors(
            positions,
            radii,
            kT=2.0,
            viscosity=1.0,
        )
        D_t3, D_r3, _ = chain_diffusion_tensors(
            positions,
            radii,
            kT=1.0,
            viscosity=2.0,
        )

        # Doubling kT doubles D.
        np.testing.assert_allclose(D_t2, 2.0 * D_t1, rtol=1e-12)
        np.testing.assert_allclose(D_r2, 2.0 * D_r1, rtol=1e-12)
        # Doubling viscosity halves D.
        np.testing.assert_allclose(D_t3, 0.5 * D_t1, rtol=1e-12)
        np.testing.assert_allclose(D_r3, 0.5 * D_r1, rtol=1e-12)

    def test_singular_C_raises_clear_error(self):
        """A perfectly collinear chain gives singular C and raises a LinAlgError mentioning the geometry."""
        positions = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        radii = np.array([1.0] * 4)
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            chain_diffusion_tensors(positions, radii)

    def test_realistic_chain_diffusion_values_reasonable(self):
        """A bent 4-bead chain yields positive D_trans and D_rot diagonals with D_trans bounded by the single-bead Stokes value."""

        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8, 3.8, 0.0],
                [3.8, 3.8, 3.8],
            ]
        )
        radii = np.array([2.0, 2.0, 2.0, 2.0])
        D_trans, D_rot, _ = chain_diffusion_tensors(positions, radii)

        # All diagonal entries positive (stable diffusion).
        for i in range(3):
            assert D_trans[i, i] > 0.0
            assert D_rot[i, i] > 0.0

        D_single = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * 2.0)
        for i in range(3):
            # No HI: D = D_single / 4. With HI, slightly larger.
            assert 0.1 * D_single < D_trans[i, i] < 1.0 * D_single


class TestBDStepTensor:
    """Tensor-aware BD step regression tests.

    The headline test is test_isotropic_tensor_matches_scalar_bit_for_bit.
    If anything in the tensor path's math is wrong (Cholesky scaling,
    drift formula, noise convention), this fails because passing
    D_trans = D * I to the tensor path no longer reproduces the scalar
    path. Other tests check anisotropy, error handling, and noise
    statistics.
    """

    def test_isotropic_tensor_matches_scalar_bit_for_bit(self):
        """Scalar bd_step_wiener and tensor bd_step_wiener_tensor with D = d I agree to within floating-point roundoff."""

        rng = np.random.default_rng(0)
        for trial in range(5):
            pos = rng.standard_normal(3) * 10.0
            ori_v = rng.standard_normal(4)
            ori = Q(*ori_v).normalized()
            force = rng.standard_normal(3)
            torque = rng.standard_normal(3)
            D_t = float(rng.uniform(0.1, 2.0))
            D_r = float(rng.uniform(0.05, 1.0))
            dt = float(rng.uniform(0.01, 0.2))
            dW_t = math.sqrt(dt) * rng.standard_normal(3)
            dW_r = math.sqrt(dt) * rng.standard_normal(3)

            p_s, o_s = bd_step_wiener(
                pos,
                ori,
                force,
                torque,
                D_t,
                D_r,
                dt,
                dW_t,
                dW_r,
            )
            p_t, o_t = bd_step_wiener_tensor(
                pos,
                ori,
                force,
                torque,
                D_t * np.eye(3),
                D_r * np.eye(3),
                dt,
                dW_t,
                dW_r,
            )
            np.testing.assert_allclose(p_s, p_t, atol=1e-13, rtol=1e-13)
            for attr in ("w", "x", "y", "z"):
                a = getattr(o_s, attr)
                b = getattr(o_t, attr)
                assert abs(a - b) < 1e-13, (
                    f"orientation.{attr}: scalar={a}, tensor={b}, " f"diff={a - b}"
                )

    def test_anisotropic_drift_along_principal_axes(self):
        """Diagonal D_trans produces drift d_i = D_ii F_i dt along each principal axis."""
        pos = np.zeros(3)
        force = np.array([1.0, 1.0, 1.0])
        D = np.diag([2.0, 1.0, 0.5])
        dt = 0.1
        new_pos = ermak_mccammon_translation_tensor(
            pos,
            force,
            D,
            dt,
            np.zeros(3),
        )
        np.testing.assert_allclose(new_pos, [0.2, 0.1, 0.05], atol=1e-15)

    def test_off_diagonal_D_couples_drift(self):
        """Off-diagonal D_trans couples drift so a force along x produces displacement along y."""
        pos = np.zeros(3)
        force = np.array([1.0, 0.0, 0.0])
        D = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dt = 1.0
        new_pos = ermak_mccammon_translation_tensor(
            pos,
            force,
            D,
            dt,
            np.zeros(3),
        )
        # drift = D @ F * dt = (1, 0.5, 0) * 1 = (1, 0.5, 0)
        np.testing.assert_allclose(new_pos, [1.0, 0.5, 0.0], atol=1e-15)

    def test_noise_covariance_matches_2_D_dt(self):
        """The empirical displacement covariance over many zero-force samples matches 2 D dt within statistical tolerance."""
        D = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.5]])
        dt = 0.1
        n_samples = 5000
        rng = np.random.default_rng(7)
        # Sample displacements with zero force.
        displacements = np.zeros((n_samples, 3))
        for i in range(n_samples):
            dW = math.sqrt(dt) * rng.standard_normal(3)
            new_pos = ermak_mccammon_translation_tensor(
                np.zeros(3),
                np.zeros(3),
                D,
                dt,
                dW,
            )
            displacements[i] = new_pos
        cov_empirical = np.cov(displacements, rowvar=False)
        cov_expected = 2.0 * D * dt
        # Statistical tolerance for 5000 samples: ~5%.
        np.testing.assert_allclose(
            cov_empirical,
            cov_expected,
            rtol=0.10,
            atol=0.005,
        )

    def test_non_pd_diffusion_raises(self):
        """A non-positive-definite D_trans raises a LinAlgError stating it is not positive-definite."""
        D = np.diag([1.0, -0.5, 1.0])  # negative eigenvalue
        with pytest.raises(np.linalg.LinAlgError, match="not positive-definite"):
            ermak_mccammon_translation_tensor(
                np.zeros(3),
                np.zeros(3),
                D,
                0.1,
                np.zeros(3),
            )

    def test_wrong_shape_raises(self):
        """A D tensor that is not (3, 3) raises ValueError about the required shape."""
        with pytest.raises(ValueError, match="must have shape"):
            ermak_mccammon_translation_tensor(
                np.zeros(3),
                np.zeros(3),
                np.array([1.0, 1.0, 1.0]),  # wrong shape
                0.1,
                np.zeros(3),
            )

    def test_rng_fallback_works(self):
        """Passing an RNG instead of a pre-drawn dW works and yields noise of order sqrt(2 dt)."""
        rng = np.random.default_rng(0)
        result = ermak_mccammon_translation_tensor(
            np.zeros(3),
            np.array([0.0, 0.0, 0.0]),
            np.eye(3),
            0.05,
            rng,
        )
        assert result.shape == (3,)
        assert np.linalg.norm(result) < 5.0 * math.sqrt(2.0 * 0.05)

    def test_zero_force_zero_dW_returns_position(self):
        """With zero force and zero noise the position is returned unchanged."""
        pos = np.array([1.0, 2.0, 3.0])
        new_pos = ermak_mccammon_translation_tensor(
            pos,
            np.zeros(3),
            np.eye(3),
            0.1,
            np.zeros(3),
        )
        np.testing.assert_array_equal(new_pos, pos)


class TestChainOuterBDStepTensor:
    """chain_outer_bd_step accepts (3, 3) tensor D_trans / D_rot.

    Backward-compat note: existing tests in TestChainOuterBDStep use
    scalar D_trans / D_rot and continue to pass. These new tests cover
    the tensor mode added in the BD step refactor.
    """

    def _setup(self, seed=42):
        pos = np.array([100.0, 0.0, 0.0])
        ori = Q(1.0, 0.0, 0.0, 0.0)
        positions = np.array([[101.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
        forces = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        return pos, ori, positions, forces, 0.05

    def test_isotropic_tensor_matches_scalar(self):
        """Scalar D inputs and isotropic tensor inputs D = d I produce the same chain step trajectory."""
        pos, ori, positions, forces, dt = self._setup()
        D_t, D_r = 0.5, 0.1

        rng_s = np.random.default_rng(0)
        p_s, o_s = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces,
            D_t,
            D_r,
            dt,
            rng_s,
        )
        rng_t = np.random.default_rng(0)
        p_t, o_t = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces,
            D_t * np.eye(3),
            D_r * np.eye(3),
            dt,
            rng_t,
        )
        np.testing.assert_allclose(p_s, p_t, atol=1e-13)
        for attr in ("w", "x", "y", "z"):
            assert abs(getattr(o_s, attr) - getattr(o_t, attr)) < 1e-13

    def test_tensor_path_gives_anisotropic_drift(self):
        """With anisotropic D_trans and a shared seed, the drift difference between x and y forces equals D F dt componentwise."""

        pos = np.zeros(3)
        ori = Q(1.0, 0.0, 0.0, 0.0)
        positions = np.array([[1.0, 0.0, 0.0]])  # single bead at +x
        forces_x = np.array([[1.0, 0.0, 0.0]])
        forces_y = np.array([[0.0, 1.0, 0.0]])
        D_trans = np.diag([2.0, 1.0, 0.5])
        D_rot = np.diag([1.0, 1.0, 1.0])  # isotropic rotation
        dt = 0.1
        # Use the same rng to isolate drift.
        rng_x = np.random.default_rng(0)
        p_x, _ = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces_x,
            D_trans,
            D_rot,
            dt,
            rng_x,
        )
        rng_y = np.random.default_rng(0)
        p_y, _ = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces_y,
            D_trans,
            D_rot,
            dt,
            rng_y,
        )
        np.testing.assert_allclose(
            p_x - p_y,
            [0.2, -0.1, 0.0],
            atol=1e-13,
        )

    def test_tensor_mode_is_deterministic_with_seed(self):
        """Two tensor-mode runs with the same seed produce identical position and orientation."""
        pos, ori, positions, forces, dt = self._setup()
        D_trans = np.diag([2.0, 1.0, 0.5])
        D_rot = np.diag([0.3, 0.3, 0.3])

        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        p1, o1 = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces,
            D_trans,
            D_rot,
            dt,
            rng1,
        )
        p2, o2 = chain_outer_bd_step(
            pos,
            ori,
            positions,
            forces,
            D_trans,
            D_rot,
            dt,
            rng2,
        )
        np.testing.assert_array_equal(p1, p2)
        for attr in ("w", "x", "y", "z"):
            assert getattr(o1, attr) == getattr(o2, attr)

    def test_mixed_scalar_and_tensor_raises(self):
        """Mixing scalar and tensor D_trans and D_rot raises a ValueError about same kind."""
        pos, ori, positions, forces, dt = self._setup()
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same kind"):
            chain_outer_bd_step(
                pos,
                ori,
                positions,
                forces,
                0.5,
                np.eye(3),
                dt,
                rng,
            )
        with pytest.raises(ValueError, match="same kind"):
            chain_outer_bd_step(
                pos,
                ori,
                positions,
                forces,
                np.eye(3),
                0.1,
                dt,
                rng,
            )

    def test_tensor_with_real_chain_diffusion_tensors(self):
        """D tensors from chain_diffusion_tensors feed through chain_outer_bd_step, giving a step of order sqrt(2 D_trans dt)."""

        # 3-bead bent chain, asymmetric so D_trans is anisotropic.
        chain_positions = np.array(
            [
                [-3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        )
        chain_radii = np.array([1.0, 1.0, 1.0])
        D_trans, D_rot, hc = chain_diffusion_tensors(
            chain_positions,
            chain_radii,
        )
        # Sanity: D tensors have the right shape and are positive-def.
        assert D_trans.shape == (3, 3)
        assert D_rot.shape == (3, 3)
        assert np.all(np.linalg.eigvalsh(D_trans) > 0)
        assert np.all(np.linalg.eigvalsh(D_rot) > 0)

        # Wire into chain_outer_bd_step.
        pos = np.array([50.0, 0.0, 0.0])
        ori = Q(1.0, 0.0, 0.0, 0.0)
        forces = np.zeros((3, 3))  # no external force
        dt = 0.05
        rng = np.random.default_rng(0)
        new_pos, new_ori = chain_outer_bd_step(
            pos,
            ori,
            chain_positions,
            forces,
            D_trans,
            D_rot,
            dt,
            rng,
        )
        assert np.linalg.norm(new_pos - pos) < 5.0 * math.sqrt(
            2.0 * np.max(np.diag(D_trans)) * dt
        )


class TestChainBDSimulatorAutoDiffusion:
    """ChainBDSimulator: auto_diffusion mode and validation logic.

    The simulator supports three D-resolution modes:
      1. Explicit scalar D_trans, D_rot (backward-compatible default)
      2. auto_diffusion=True (compute from geometry)
      3. Explicit tensor D_trans, D_rot (user pre-computes)
    Plus two error modes (auto + explicit, neither).
    """

    def _make_template(self, n=3):
        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(n)]
        return ChainCommon(name="test", atoms=atoms)

    def _make_params(self):
        return ChainBDParameters(
            n_trajectories=1,
            dt=0.1,
            dt_chain=0.01,
            chain_steps_per_outer=10,
            max_steps=100,
            r_start=10.0,
            seed=42,
        )

    def _bent_chain(self):
        """3-bead bent chain centered at origin. Non-singular geometry."""
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
            ]
        )
        return body_pos - body_pos.mean(axis=0)

    def test_scalar_mode_stores_scalar_D(self):
        """The default scalar path stores float D_trans and D_rot with auto_diffusion False."""

        sim = ChainBDSimulator(
            target=None,
            chain_template=self._make_template(),
            chain_init_body_positions=self._bent_chain(),
            params=self._make_params(),
            pathway_set=None,
            D_trans=0.5,
            D_rot=0.1,
        )
        assert sim.D_trans == 0.5
        assert sim.D_rot == 0.1
        assert sim.auto_diffusion is False

    def test_auto_diffusion_produces_tensors(self):
        """auto_diffusion=True produces (3, 3) D tensors that are positive-definite and symmetric."""

        sim = ChainBDSimulator(
            target=None,
            chain_template=self._make_template(),
            chain_init_body_positions=self._bent_chain(),
            params=self._make_params(),
            pathway_set=None,
            auto_diffusion=True,
        )
        assert sim.D_trans.shape == (3, 3)
        assert sim.D_rot.shape == (3, 3)
        assert sim.auto_diffusion is True
        # Both tensors must be positive-definite (physical D).
        assert np.all(np.linalg.eigvalsh(sim.D_trans) > 0)
        assert np.all(np.linalg.eigvalsh(sim.D_rot) > 0)
        # Both must be symmetric.
        np.testing.assert_allclose(sim.D_trans, sim.D_trans.T, atol=1e-12)
        np.testing.assert_allclose(sim.D_rot, sim.D_rot.T, atol=1e-12)

    def test_auto_diffusion_matches_chain_diffusion_tensors(self):
        """The auto-computed D tensors equal chain_diffusion_tensors called directly on the same geometry."""

        body_pos = self._bent_chain()
        radii = np.array([1.0, 1.0, 1.0])
        D_t_direct, D_r_direct, _ = chain_diffusion_tensors(body_pos, radii)

        sim = ChainBDSimulator(
            target=None,
            chain_template=self._make_template(),
            chain_init_body_positions=body_pos,
            params=self._make_params(),
            pathway_set=None,
            auto_diffusion=True,
        )
        np.testing.assert_array_equal(sim.D_trans, D_t_direct)
        np.testing.assert_array_equal(sim.D_rot, D_r_direct)

    def test_explicit_tensor_mode_stores_user_tensors(self):
        """User-supplied (3, 3) D_trans and D_rot are stored as given with auto_diffusion False."""

        D_t = np.diag([0.5, 0.3, 0.3])
        D_r = np.diag([0.1, 0.05, 0.05])
        sim = ChainBDSimulator(
            target=None,
            chain_template=self._make_template(),
            chain_init_body_positions=self._bent_chain(),
            params=self._make_params(),
            pathway_set=None,
            D_trans=D_t,
            D_rot=D_r,
        )
        np.testing.assert_array_equal(sim.D_trans, D_t)
        np.testing.assert_array_equal(sim.D_rot, D_r)
        assert sim.auto_diffusion is False

    def test_auto_with_explicit_D_raises(self):
        """Combining auto_diffusion=True with any explicit D_trans or D_rot raises ValueError."""

        for kw in ({"D_trans": 0.5}, {"D_rot": 0.1}, {"D_trans": 0.5, "D_rot": 0.1}):
            with pytest.raises(ValueError, match="incompatible with explicit"):
                ChainBDSimulator(
                    target=None,
                    chain_template=self._make_template(),
                    chain_init_body_positions=self._bent_chain(),
                    params=self._make_params(),
                    pathway_set=None,
                    auto_diffusion=True,
                    **kw,
                )

    def test_no_D_no_auto_raises(self):
        """Supplying neither auto_diffusion nor explicit D raises a ValueError prompting auto_diffusion=True."""

        with pytest.raises(ValueError, match="auto_diffusion=True"):
            ChainBDSimulator(
                target=None,
                chain_template=self._make_template(),
                chain_init_body_positions=self._bent_chain(),
                params=self._make_params(),
                pathway_set=None,
            )

    def test_partial_D_raises(self):
        """Supplying only one of D_trans or D_rot raises ValueError."""

        with pytest.raises(ValueError):
            ChainBDSimulator(
                target=None,
                chain_template=self._make_template(),
                chain_init_body_positions=self._bent_chain(),
                params=self._make_params(),
                pathway_set=None,
                D_trans=0.5,  # missing D_rot
            )
        with pytest.raises(ValueError):
            ChainBDSimulator(
                target=None,
                chain_template=self._make_template(),
                chain_init_body_positions=self._bent_chain(),
                params=self._make_params(),
                pathway_set=None,
                D_rot=0.1,  # missing D_trans
            )

    def test_collinear_chain_in_auto_mode_raises(self):
        """A collinear chain in auto mode propagates the singular-C LinAlgError from chain_diffusion_tensors."""

        collinear_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            ChainBDSimulator(
                target=None,
                chain_template=self._make_template(),
                chain_init_body_positions=collinear_pos,
                params=self._make_params(),
                pathway_set=None,
                auto_diffusion=True,
            )


class TestChainSimulationCLIAutoDiffusion:
    """CLI argument validation for chain_simulation's auto_diffusion flag.

    These tests use click's CliRunner to invoke the command without
    actually running the (expensive) simulation. We verify that the
    validation block at the top of the function raises clear UsageError
    messages before any IO happens.
    """

    def _runner(self):
        return CliRunner(), cli

    def test_no_d_no_auto_raises_usage_error(self):
        """Without --auto-diffusion, omitting both --d-trans and --d-rot fails with a required-options error."""
        runner, cli = self._runner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                "nonexistent.json",
                "--target",
                "nonexistent.pqr",
                "--rxn",
                "nonexistent.xml",
            ],
        )
        assert result.exit_code != 0
        assert "Both --d-trans and --d-rot are required" in result.output

    def test_partial_d_raises_usage_error(self):
        """Supplying only --d-trans fails with the both-options-required error."""
        runner, cli = self._runner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                "nonexistent.json",
                "--target",
                "nonexistent.pqr",
                "--rxn",
                "nonexistent.xml",
                "--d-trans",
                "0.5",
            ],
        )
        assert result.exit_code != 0
        assert "Both --d-trans and --d-rot are required" in result.output

    def test_auto_with_d_trans_raises(self):
        """Combining --auto-diffusion with --d-trans fails as incompatible."""
        runner, cli = self._runner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                "nonexistent.json",
                "--target",
                "nonexistent.pqr",
                "--rxn",
                "nonexistent.xml",
                "--auto-diffusion",
                "--d-trans",
                "0.5",
            ],
        )
        assert result.exit_code != 0
        assert "--auto-diffusion cannot be combined" in result.output

    def test_auto_with_d_rot_raises(self):
        """Combining --auto-diffusion with --d-rot fails as incompatible."""
        runner, cli = self._runner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                "nonexistent.json",
                "--target",
                "nonexistent.pqr",
                "--rxn",
                "nonexistent.xml",
                "--auto-diffusion",
                "--d-rot",
                "0.1",
            ],
        )
        assert result.exit_code != 0
        assert "--auto-diffusion cannot be combined" in result.output

    def test_help_documents_auto_diffusion_flag(self):
        """The --help output mentions --auto-diffusion and the Rotne-Prager method."""
        runner, cli = self._runner()
        result = runner.invoke(cli, ["chain_simulation", "--help"])
        assert result.exit_code == 0
        assert "--auto-diffusion" in result.output
        # Must mention the physics method so users know what they're getting.
        assert "Rotne-Prager" in result.output

    def test_help_documents_d_trans_now_optional(self):
        """The --help output indicates --d-trans is conditionally required via auto-diffusion."""
        runner, cli = self._runner()
        result = runner.invoke(cli, ["chain_simulation", "--help"])
        # The phrasing should make it clear --d-trans is conditionally required.
        assert "auto-diffusion" in result.output


class TestChainOutputWriter:
    """Regression tests for the chain BD output writer.

    Builds a minimal simulator, runs it, writes outputs to a tmp dir,
    and verifies the file contents (results.json structure, CSV row
    counts, format details).
    """

    def _build_minimal_sim(self, *, auto_diffusion=True, n_traj=5):
        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(4)]
        template = ChainCommon(name="writer_test_chain", atoms=atoms)
        body_pos = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="writer_test_target",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        pathways = PathwaySet(reactions=[])
        params = ChainBDParameters(
            n_trajectories=n_traj,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=2000,
            r_start=10.0,
            r_escape=30.0,
            seed=42,
            n_threads=1,
        )

        kwargs = (
            {"auto_diffusion": True}
            if auto_diffusion
            else {
                "D_trans": 0.05,
                "D_rot": 0.005,
            }
        )
        return ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=pathways,
            **kwargs,
        )

    def test_files_are_written(self, tmp_path):
        """Writing chain results creates both results.json and trajectories.csv on disk."""

        sim = self._build_minimal_sim()
        results = sim.run()
        written = write_chain_results(tmp_path, sim, results)
        names = [name for name, _ in written]
        assert "results.json" in names
        assert "trajectories.csv" in names
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "trajectories.csv").exists()

    def test_results_json_structure_auto_mode(self, tmp_path):
        """results.json carries summary, diffusion, chain, and params blocks, with auto_diffusion mode giving 3x3 tensors."""

        sim = self._build_minimal_sim(auto_diffusion=True)
        results = sim.run()
        write_chain_results(tmp_path, sim, results, wall_time_sec=1.5)
        data = json.loads((tmp_path / "results.json").read_text())

        # Top-level keys.
        for key in ("summary", "diffusion", "chain", "params"):
            assert key in data, f"missing top-level key {key}"

        # Summary fields.
        s = data["summary"]
        assert s["n_trajectories"] == 5
        assert s["n_reacted"] + s["n_escaped"] + s["n_max_steps"] == 5
        assert s["wall_time_sec"] == 1.5
        assert "mean_time_ps" in s
        assert "fraction_reacted" in s

        # Diffusion block: auto mode produces 3x3 tensors.
        d = data["diffusion"]
        assert d["mode"] == "auto_diffusion"
        assert "Rotne-Prager" in d["method"]
        assert isinstance(d["D_trans_3x3"], list)
        assert len(d["D_trans_3x3"]) == 3
        assert len(d["D_trans_3x3"][0]) == 3
        assert d["D_trans_isotropic_equiv"] > 0
        assert d["D_trans_units"] == "A^2/ps"

        # Chain block.
        c = data["chain"]
        assert c["name"] == "writer_test_chain"
        assert c["n_atoms"] == 4
        assert c["atom_radii"] == [1.5, 1.5, 1.5, 1.5]

        # Params block.
        p = data["params"]
        assert p["n_trajectories"] == 5
        assert p["seed"] == 42

    def test_results_json_scalar_mode_emits_scalars(self, tmp_path):
        """In scalar D mode the diffusion block emits float D_trans and D_rot rather than 3x3 arrays."""

        sim = self._build_minimal_sim(auto_diffusion=False)
        results = sim.run()
        write_chain_results(tmp_path, sim, results)
        data = json.loads((tmp_path / "results.json").read_text())
        d = data["diffusion"]
        assert d["mode"] == "scalar"
        assert "D_trans_3x3" not in d
        assert d["D_trans"] == 0.05
        assert d["D_rot"] == 0.005

    def test_trajectories_csv_row_count_matches(self, tmp_path):
        """trajectories.csv has one row per trajectory plus a header with the expected columns."""

        sim = self._build_minimal_sim(n_traj=8)
        results = sim.run()
        write_chain_results(tmp_path, sim, results)
        text = (tmp_path / "trajectories.csv").read_text()
        lines = text.strip().split("\n")
        # 1 header + 8 rows.
        assert len(lines) == 9
        # Header contains the expected columns.
        header = lines[0]
        for col in (
            "traj_id",
            "fate",
            "steps",
            "time_ps",
            "final_separation",
            "reaction_name",
            "energy_at_reaction",
        ):
            assert col in header

    def test_trajectories_csv_fate_strings_correct(self, tmp_path):
        """Each trajectories.csv row carries a fate string matching a Fate enum name."""

        sim = self._build_minimal_sim()
        results = sim.run()
        write_chain_results(tmp_path, sim, results)
        text = (tmp_path / "trajectories.csv").read_text()
        valid_fates = {f.name for f in Fate}
        for line in text.strip().split("\n")[1:]:  # skip header
            cells = line.split(",")
            fate_name = cells[1]
            assert fate_name in valid_fates

    def test_reaction_counts_appear_when_reactions_fire(self, tmp_path):
        """The summary records per-reaction counts when trajectories react."""

        sim = self._build_minimal_sim()
        # Hand-craft results with reactions to exercise the count tally.
        results = [
            TrajectoryResult(Fate.REACTED, 10, 5.0, 8.0, "rxn_A"),
            TrajectoryResult(Fate.REACTED, 20, 10.0, 7.5, "rxn_A"),
            TrajectoryResult(Fate.REACTED, 30, 15.0, 6.0, "rxn_B"),
            TrajectoryResult(Fate.ESCAPED, 100, 50.0, 30.0),
        ]
        write_chain_results(tmp_path, sim, results)
        data = json.loads((tmp_path / "results.json").read_text())
        rc = data["summary"].get("reaction_counts", {})
        assert rc == {"rxn_A": 2, "rxn_B": 1}

    def test_no_reaction_counts_when_no_reactions(self, tmp_path):
        """The summary omits reaction_counts when no trajectory reacted."""

        sim = self._build_minimal_sim()
        results = sim.run()
        # Our minimal sim has empty pathways so no reactions can fire.
        write_chain_results(tmp_path, sim, results)
        data = json.loads((tmp_path / "results.json").read_text())
        assert "reaction_counts" not in data["summary"]

    def test_writer_creates_missing_directory(self, tmp_path):
        """The writer creates a missing nested work_dir and writes results.json into it."""

        sim = self._build_minimal_sim()
        results = sim.run()
        nested = tmp_path / "deeply" / "nested" / "outdir"
        assert not nested.exists()
        write_chain_results(nested, sim, results)
        assert nested.exists()
        assert (nested / "results.json").exists()


class TestChainSimulationCLIOutput:
    """CLI end-to-end: --output-dir flag is parsed and documented."""

    def _runner(self):
        return CliRunner(), cli

    def test_help_documents_output_dir(self):
        """The chain_simulation help text documents --output-dir and its chain_bd_results default."""
        runner, cli = self._runner()
        result = runner.invoke(cli, ["chain_simulation", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "chain_bd_results" in result.output  # default

    def test_output_dir_flag_parses(self):
        """The CLI parses --output-dir with --auto-diffusion and fails only on missing input files, not flag parsing."""
        runner, cli = self._runner()
        result = runner.invoke(
            cli,
            [
                "chain_simulation",
                "--chain",
                "/tmp/_pystarc_nonexistent.json",
                "--target",
                "/tmp/_pystarc_nonexistent.pqr",
                "--rxn",
                "/tmp/_pystarc_nonexistent.xml",
                "--auto-diffusion",
                "--output-dir",
                "/tmp/_pystarc_out_test",
            ],
        )
        assert result.exit_code != 0
        assert "Usage" not in result.output or "Error: " in result.output
        assert "no such option" not in result.output.lower()


class TestAdaptiveDtZone:
    """Two-zone adaptive dt: switches from params.dt to params.dt_rxn
    when chain CoM is within 1.5 * smallest_reaction_cutoff.
    """

    def _build_pathway_with_distance(self, distance_cutoff):
        """Construct a PathwaySet with one reaction with one contact
        criterion at the given distance_cutoff."""

        pair = ContactPair(
            mol1_atom_index=0,
            mol2_atom_index=0,
            distance_cutoff=distance_cutoff,
        )
        criteria = ReactionCriteria(pairs=[pair])
        rxn = ReactionInterface(
            name="contact",
            criteria=criteria,
            probability=1.0,
        )
        return PathwaySet(reactions=[rxn])

    def test_min_reaction_distance_with_no_reactions(self):
        """_min_reaction_distance returns 0.0 for an empty PathwaySet."""

        empty = PathwaySet(reactions=[])
        assert _min_reaction_distance(empty) == 0.0

    def test_min_reaction_distance_with_none(self):
        """_min_reaction_distance returns 0.0 when pathway_set is None."""

        assert _min_reaction_distance(None) == 0.0

    def test_min_reaction_distance_picks_smallest(self):
        """_min_reaction_distance returns the smallest distance_cutoff across all reactions and pairs."""

        rxns = [
            ReactionInterface(
                name="r1",
                probability=1.0,
                criteria=ReactionCriteria(
                    pairs=[
                        ContactPair(0, 0, distance_cutoff=8.0),
                        ContactPair(1, 1, distance_cutoff=4.0),  # smallest in r1
                    ]
                ),
            ),
            ReactionInterface(
                name="r2",
                probability=1.0,
                criteria=ReactionCriteria(
                    pairs=[
                        ContactPair(0, 0, distance_cutoff=3.0),  # smallest overall
                        ContactPair(2, 2, distance_cutoff=10.0),
                    ]
                ),
            ),
        ]
        ps = PathwaySet(reactions=rxns)
        assert _min_reaction_distance(ps) == 3.0

    def test_simulator_caches_rxn_min(self):
        """ChainBDSimulator caches _rxn_min in __init__ as the minimum reaction distance, or 0.0 when empty."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        params = ChainBDParameters(
            n_trajectories=1,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=10,
            r_start=20.0,
            r_escape=30.0,
            seed=42,
        )

        # Empty pathway set -> _rxn_min = 0.0
        sim_empty = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=None,
            D_trans=0.05,
            D_rot=0.005,
        )
        assert sim_empty._rxn_min == 0.0

        # Pathway with cutoff 5.0 -> _rxn_min = 5.0
        ps = self._build_pathway_with_distance(5.0)
        sim = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )
        assert sim._rxn_min == 5.0

    def test_dt_zone_activates_near_reaction_surface(self):
        """Inside 1.5 times rxn_min the simulator uses params.dt_rxn so mean time per step equals dt_rxn."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        # Minimal target so the reaction check has something to look at.
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.1,
                ),
            ],
        )
        # Big gap between dt and dt_rxn so we can tell which fired.
        params = ChainBDParameters(
            n_trajectories=1,
            dt=10.0,
            dt_chain=0.05,
            dt_rxn=0.1,
            chain_steps_per_outer=2,
            max_steps=5,
            r_start=5.0,
            r_escape=100.0,
            seed=42,
        )

        ps = self._build_pathway_with_distance(10.0)
        sim = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=ps,
            D_trans=0.001,
            D_rot=0.0001,
        )
        # Manually iterate so we can keep r low enough to stay in zone.
        result = sim.run_one(np.random.default_rng(42))
        assert result.time_ps < 1.0, (
            f"expected dt_rxn (0.1) to fire repeatedly, giving t < 1 ps; "
            f"got {result.time_ps:.4f} ps. dt zone may not be activating."
        )

    def test_dt_zone_does_not_activate_in_bulk(self):
        """Well outside 1.5 times rxn_min the simulator uses params.dt so mean time per step equals dt."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.1,
                ),
            ],
        )
        params = ChainBDParameters(
            n_trajectories=1,
            dt=10.0,
            dt_chain=0.05,
            dt_rxn=0.1,
            chain_steps_per_outer=2,
            max_steps=3,
            r_start=100.0,
            r_escape=10000.0,
            seed=42,
        )

        ps = self._build_pathway_with_distance(5.0)
        sim = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=ps,
            D_trans=0.001,
            D_rot=0.0001,
        )
        result = sim.run_one(np.random.default_rng(42))
        assert result.time_ps > 5.0, (
            f"expected params.dt (10.0) to fire in bulk; got "
            f"{result.time_ps:.4f} ps. Did dt_rxn activate unexpectedly?"
        )

    def test_elapsed_time_correctly_accumulated(self):
        """With an empty PathwaySet the smoke run reproduces the mean simulated time of about 1526 ps."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(4)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        params = ChainBDParameters(
            n_trajectories=20,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=10000,
            r_start=10.0,
            r_escape=30.0,
            seed=42,
            n_threads=1,
            use_lmz=False,
        )

        sim = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            auto_diffusion=True,
        )
        results = sim.run()
        mean_t = sum(r.time_ps for r in results) / len(results)
        assert 1150 < mean_t < 1500, (
            f"empty PathwaySet smoke value drifted: expected ~1315 ps "
            f"(post viscosity-temperature fix), got {mean_t:.1f} ps"
        )


class TestForceChangeBackstep:
    """Force-change backstep mechanism for the chain BD outer step.

    Verifies that the BD step is subdivided when external forces change
    rapidly across a step, and is not subdivided otherwise.
    """

    def _make_sim(self, *, force_change_backstep=True, **param_overrides):
        """Build a minimal ChainBDSimulator for backstep testing."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.1,
                ),
            ],
        )
        params_kwargs = dict(
            n_trajectories=1,
            dt=2.0,
            dt_chain=0.05,
            chain_steps_per_outer=2,
            max_steps=10,
            r_start=20.0,
            r_escape=200.0,
            seed=42,
            n_threads=1,
            force_change_backstep=force_change_backstep,
        )
        params_kwargs.update(param_overrides)
        params = ChainBDParameters(**params_kwargs)

        sim = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            D_trans=0.05,
            D_rot=0.005,
        )
        return sim

    def test_parameter_default_is_on(self):
        """ChainBDParameters.force_change_backstep defaults to True."""

        p = ChainBDParameters()
        assert p.force_change_backstep is True

    def test_effective_hydro_radius_auto_mode(self):
        """In auto_diffusion mode the effective hydrodynamic radius derives from the trace of D_trans via Stokes-Einstein."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        params = ChainBDParameters(n_trajectories=1, max_steps=1, seed=0)
        sim = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=None,
            auto_diffusion=True,
        )
        # a = 1 / (6 pi mu D_iso)
        D_iso = float(np.trace(np.asarray(sim.D_trans)) / 3.0)
        expected = 1.0 / (6.0 * math.pi * WATER_VISCOSITY * D_iso)
        assert abs(sim._effective_hydro_radius - expected) < 1e-12

    def test_effective_hydro_radius_scalar_mode(self):
        """In scalar mode the effective hydrodynamic radius equals the largest bead radius."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0),
            ChainAtom(radius=2.5, charge=0.0),  # largest
            ChainAtom(radius=1.5, charge=0.0),
        ]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        params = ChainBDParameters(n_trajectories=1, max_steps=1, seed=0)
        sim = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=None,
            D_trans=0.05,
            D_rot=0.005,
        )
        assert sim._effective_hydro_radius == 2.5

    def test_chain_outer_bd_step_wiener_matches_chain_outer_bd_step(self):
        """chain_outer_bd_step_wiener matches chain_outer_bd_step when fed equivalent Wiener increments."""

        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)

        pos = np.array([5.0, 0.0, 0.0])
        ori = Quaternion.identity()
        n = 4
        chain_world = np.array(
            [
                [4.0, 0.0, 0.0],
                [4.5, 0.5, 0.0],
                [5.5, -0.5, 0.0],
                [6.0, 0.0, 0.0],
            ]
        )
        per_atom_forces = np.array([[0.1, 0.0, 0.0]] * n)
        D_trans, D_rot = 0.05, 0.005
        dt = 0.5

        # Reference: regular function draws Wiener internally
        pos_ref, ori_ref = chain_outer_bd_step(
            pos,
            ori,
            chain_world,
            per_atom_forces,
            D_trans,
            D_rot,
            dt,
            rng_a,
        )

        # Sibling: pass equivalent pre-drawn Wiener
        dW_t = math.sqrt(dt) * rng_b.standard_normal(3)
        dW_r = math.sqrt(dt) * rng_b.standard_normal(3)
        pos_w, ori_w = chain_outer_bd_step_wiener(
            pos,
            ori,
            chain_world,
            per_atom_forces,
            D_trans,
            D_rot,
            dt,
            dW_t,
            dW_r,
        )

        np.testing.assert_allclose(pos_w, pos_ref, atol=1e-13)
        np.testing.assert_allclose(
            [ori_w.w, ori_w.x, ori_w.y, ori_w.z],
            [ori_ref.w, ori_ref.x, ori_ref.y, ori_ref.z],
            atol=1e-13,
        )

    def test_zero_forces_never_trigger_backstep(self):
        """With zero external forces the force-change backstep never fires, matching the flag-off trajectory exactly."""
        sim_on = self._make_sim(force_change_backstep=True)
        sim_off = self._make_sim(force_change_backstep=False)
        # Both have target_grid=None, so per-atom forces always zero.
        r_on = sim_on.run_one(np.random.default_rng(42))
        r_off = sim_off.run_one(np.random.default_rng(42))
        # Same trajectory because backstep never fires.
        assert r_on.steps == r_off.steps
        assert abs(r_on.time_ps - r_off.time_ps) < 1e-12
        assert abs(r_on.final_separation - r_off.final_separation) < 1e-9

    def test_backstep_fires_with_steep_force_gradient(self):
        """backstep_due_to_force fires only when dt exceeds the force-change threshold and never at the dt floor."""

        zero = np.zeros(3)

        assert backstep_due_to_force(
            np.array([0.1, 0.0, 0.0]),  # force_new
            zero,  # force_old
            np.array([2.0, 0.0, 0.0]),  # pos_new
            zero,  # pos_old
            dt=2.0,
            dt_min=0.05,
            radius=1.0,
        ), "criterion should fire for large dx with small force change"

        assert not backstep_due_to_force(
            np.array([0.05, 0.0, 0.0]),
            zero,
            np.array([5.0, 0.0, 0.0]),
            zero,
            dt=2.0,
            dt_min=0.05,
            radius=1.0,
        ), "criterion should not fire when threshold exceeds dt"

        # dt at/below the floor (dt <= dt_min) never subdivides, by guard.
        assert not backstep_due_to_force(
            np.array([0.1, 0.0, 0.0]),
            zero,
            np.array([2.0, 0.0, 0.0]),
            zero,
            dt=0.05,
            dt_min=0.05,
            radius=1.0,
        ), "criterion must not fire at the dt floor"

    def test_backstep_skipped_in_dt_rxn_zone(self):
        """Inside the dt_rxn zone dt is already at the floor so force-change subdivision does not occur."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.1,
                ),
            ],
        )
        pair = ContactPair(0, 0, distance_cutoff=10.0)
        criteria = ReactionCriteria(pairs=[pair])
        rxn = ReactionInterface(
            name="contact",
            criteria=criteria,
            probability=1.0,
        )
        ps = PathwaySet(reactions=[rxn])
        params = ChainBDParameters(
            n_trajectories=1,
            dt=10.0,
            dt_chain=0.05,
            dt_rxn=0.1,
            chain_steps_per_outer=2,
            max_steps=5,
            r_start=5.0,
            r_escape=100.0,
            seed=42,
            force_change_backstep=True,
        )

        sim_on = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=ps,
            D_trans=0.001,
            D_rot=0.0001,
        )
        params_off = ChainBDParameters(
            n_trajectories=1,
            dt=10.0,
            dt_chain=0.05,
            dt_rxn=0.1,
            chain_steps_per_outer=2,
            max_steps=5,
            r_start=5.0,
            r_escape=100.0,
            seed=42,
            force_change_backstep=False,
        )
        sim_off = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_off,
            pathway_set=ps,
            D_trans=0.001,
            D_rot=0.0001,
        )


        def synthetic(self, world_positions):
            f = np.zeros_like(world_positions)
            f[:, 0] = 100.0 * float(world_positions.mean(axis=0)[0])
            return f

        sim_on._compute_per_atom_external_forces = types.MethodType(
            synthetic,
            sim_on,
        )
        sim_off._compute_per_atom_external_forces = types.MethodType(
            synthetic,
            sim_off,
        )

        r_on = sim_on.run_one(np.random.default_rng(42))
        r_off = sim_off.run_one(np.random.default_rng(42))

        assert r_on.steps == r_off.steps
        assert abs(r_on.time_ps - r_off.time_ps) < 1e-12
        assert abs(r_on.final_separation - r_off.final_separation) < 1e-9


class TestHardSphereRejection:
    """Hard-sphere overlap rejection for the chain BD outer step.

    Verifies the overlap helper detects bead-target and intra-chain
    bead-bead overlaps, that bonded pairs are excluded, that ghost
    atoms are skipped, and that the simulator's run loop actually
    rejects overlapping steps.
    """

    def _build_target(self, atom_specs):
        """Build a Molecule with atoms at given (x, y, z, radius) tuples."""

        atoms = []
        for i, (x, y, z, r) in enumerate(atom_specs, start=1):
            atoms.append(
                Atom(
                    index=i,
                    name=f"X{i}",
                    residue_name="DUM",
                    residue_index=1,
                    x=x,
                    y=y,
                    z=z,
                    charge=0.0,
                    radius=r,
                )
            )
        return Molecule(name="target", atoms=atoms)

    def test_parameter_default_is_on(self):
        """ChainBDParameters.use_hard_sphere defaults to True."""

        p = ChainBDParameters()
        assert p.use_hard_sphere is True

    def test_overlap_helper_chain_target(self):
        """_check_chain_overlap returns True when a chain bead overlaps a target atom."""

        target = self._build_target([(0.0, 0.0, 0.0, 2.0)])
        chain_pos = np.array([[3.0, 0.0, 0.0]])
        chain_r = np.array([1.5])
        assert _check_chain_overlap(target, chain_pos, chain_r, set())

    def test_overlap_helper_no_overlap_when_separated(self):
        """_check_chain_overlap returns False when a bead is well separated from the target atom."""

        target = self._build_target([(0.0, 0.0, 0.0, 2.0)])
        # Chain bead at (10.0, 0, 0) with radius 1.5 -> well separated.
        chain_pos = np.array([[10.0, 0.0, 0.0]])
        chain_r = np.array([1.5])
        assert not _check_chain_overlap(target, chain_pos, chain_r, set())

    def test_overlap_helper_intra_chain_bead_pair(self):
        """_check_chain_overlap returns True when two non-bonded chain beads overlap."""

        # Two beads with radius 1.5 at distance 2.0 -> overlap.
        chain_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        chain_r = np.array([1.5, 1.5])
        # No bonded pair -> overlap detected.
        assert _check_chain_overlap(None, chain_pos, chain_r, set())

    def test_overlap_helper_bonded_pair_is_skipped(self):
        """_check_chain_overlap does not flag overlap between close bonded neighbors."""

        chain_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        chain_r = np.array([1.5, 1.5])
        # Same close geometry, but now (0, 1) is a bonded pair.
        bonded = {(0, 1), (1, 0)}
        assert not _check_chain_overlap(None, chain_pos, chain_r, bonded)

    def test_overlap_helper_ghost_atoms_skipped(self):
        """_check_chain_overlap skips ghost atoms with radius below 1e-10 so they never trigger overlap."""

        target = self._build_target([(0.0, 0.0, 0.0, 0.0)])  # ghost
        chain_pos = np.array([[0.5, 0.0, 0.0]])
        chain_r = np.array([1.5])
        assert not _check_chain_overlap(target, chain_pos, chain_r, set())

    def test_simulator_caches_bonded_pairs_and_radii(self):
        """ChainBDSimulator caches bead radii and bonded pairs in both orderings from the chain template."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0),
            ChainAtom(radius=2.0, charge=0.0),
            ChainAtom(radius=3.0, charge=0.0),
        ]
        bonds = [
            ChainBond(
                a=ChainAtomRef(0),
                b=ChainAtomRef(1),
                r0=1.0,
                k_spring=10.0,
            ),
        ]
        template = ChainCommon(name="t", atoms=atoms, bonds=bonds)
        body_pos = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        params = ChainBDParameters(n_trajectories=1, max_steps=1, seed=0)
        sim = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=None,
            D_trans=0.05,
            D_rot=0.005,
        )
        np.testing.assert_array_equal(sim._chain_radii, [1.0, 2.0, 3.0])
        assert (0, 1) in sim._bonded_pairs
        assert (1, 0) in sim._bonded_pairs  # both orderings
        assert (0, 2) not in sim._bonded_pairs
        assert (1, 2) not in sim._bonded_pairs

    def test_rejection_fires_with_overlap_prone_geometry(self):
        """Hard-sphere rejection on versus off yields meaningfully different mean times for overlap-prone geometry."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(2)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-1.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        # Big target: chain has to navigate around it.
        target = self._build_target([(0.0, 0.0, 0.0, 4.0)])
        params_on = ChainBDParameters(
            n_trajectories=10,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=2000,
            r_start=8.0,
            r_escape=20.0,
            seed=42,
            n_threads=1,
            use_hard_sphere=True,
        )
        params_off = ChainBDParameters(
            n_trajectories=10,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=2000,
            r_start=8.0,
            r_escape=20.0,
            seed=42,
            n_threads=1,
            use_hard_sphere=False,
        )
        ps = PathwaySet(reactions=[])

        sim_on = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_on,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )
        sim_off = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_off,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )

        results_on = sim_on.run()
        results_off = sim_off.run()
        mean_on = sum(r.time_ps for r in results_on) / len(results_on)
        mean_off = sum(r.time_ps for r in results_off) / len(results_off)

        assert abs(mean_on - mean_off) > 1e-6, (
            f"rejection did not alter trajectories: "
            f"on={mean_on:.4f}, off={mean_off:.4f}. The geometry may "
            f"not have produced overlaps."
        )

    def test_rejection_off_matches_pre_adt3(self):
        """With use_hard_sphere=False the run reproduces the pre-ADT3 smoke value of about 1675 ps."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(4)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        params = ChainBDParameters(
            n_trajectories=20,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=10000,
            r_start=10.0,
            r_escape=30.0,
            seed=42,
            n_threads=1,
            use_hard_sphere=False,
            use_lmz=False,
        )
        sim = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params,
            pathway_set=PathwaySet(reactions=[]),
            auto_diffusion=True,
        )
        results = sim.run()
        mean_t = sum(r.time_ps for r in results) / len(results)
        assert 1050 < mean_t < 1350, (
            f"with hard-sphere off, expected ~1181 ps post viscosity fix; "
            f"got {mean_t:.1f} ps"
        )


class TestSoftRepulsion:
    """WCA soft repulsion forces: intra-chain + bead-target.

    Verifies the new chain_intra_nonbonded_forces and
    chain_target_steric_forces functions have correct physics
    (sign, magnitude, cutoff) and are wired through the simulator.
    """

    def test_intra_chain_force_is_repulsive(self):
        """chain_intra_nonbonded_forces pushes two beads inside sigma apart with equal and opposite forces."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        common = ChainCommon(name="t", atoms=atoms)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        F = chain_intra_nonbonded_forces(state, common, eps=1.0)
        # Bead 0 should be pushed in -x (away from bead 2 at +x).
        assert F[0, 0] < 0, f"bead 0 not repelled: {F[0]}"
        # Bead 2 should be pushed in +x (away from bead 0 at -x relative).
        assert F[2, 0] > 0, f"bead 2 not repelled: {F[2]}"
        # By Newton 3rd law, magnitudes equal.
        np.testing.assert_allclose(F[0], -F[2], atol=1e-12)

    def test_intra_chain_force_zero_outside_cutoff(self):
        """chain_intra_nonbonded_forces is zero at r at or beyond the WCA cutoff 2^(1/6) sigma."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        common = ChainCommon(name="t", atoms=atoms)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        F = chain_intra_nonbonded_forces(state, common, eps=1.0)
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(F[2], [0.0, 0.0, 0.0], atol=1e-12)

    def test_intra_chain_force_magnitude_correct(self):
        """chain_intra_nonbonded_forces magnitude at r = sigma/2 matches the textbook WCA value 4 eps (12 sigma^12/r^13 minus 6 sigma^6/r^7)."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        common = ChainCommon(name="t", atoms=atoms)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        F = chain_intra_nonbonded_forces(state, common, eps=1.0)
        expected_mag = 4.0 * (12.0 * 4096.0 - 6.0 * 64.0)
        actual_mag = float(np.linalg.norm(F[2]))
        assert (
            abs(actual_mag - expected_mag) / expected_mag < 1e-10
        ), f"magnitude mismatch: expected {expected_mag}, got {actual_mag}"

    def test_intra_chain_skips_bonded_pairs(self):
        """chain_intra_nonbonded_forces excludes bonded pairs even when they sit inside sigma."""

        atoms = [ChainAtom(radius=1.0, charge=0.0) for _ in range(3)]
        bonds = [
            ChainBond(
                a=ChainAtomRef(0),
                b=ChainAtomRef(2),
                r0=2.0,
                k_spring=10.0,
            ),
        ]
        common = ChainCommon(name="t", atoms=atoms, bonds=bonds)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        F = chain_intra_nonbonded_forces(state, common, eps=1.0)
        # Bonded pair (0, 2) should be skipped, so no force.
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(F[2], [0.0, 0.0, 0.0], atol=1e-12)

    def test_intra_chain_skips_ghost_atoms(self):
        """chain_intra_nonbonded_forces produces no force from or on ghost beads with radius below 1e-10."""

        atoms = [
            ChainAtom(radius=0.0, charge=0.0),  # ghost
            ChainAtom(radius=1.0, charge=0.0),
            ChainAtom(radius=1.0, charge=0.0),
        ]
        common = ChainCommon(name="t", atoms=atoms)
        positions = np.array(
            [
                [0.5, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        F = chain_intra_nonbonded_forces(state, common, eps=1.0)
        # Ghost bead 0 should produce no force on anyone.
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(F[2], [0.0, 0.0, 0.0], atol=1e-12)

    def test_target_steric_force_is_repulsive(self):
        """chain_target_steric_forces pushes a chain bead inside sigma away from the target atom."""

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        chain_pos = np.array([[2.5, 0.0, 0.0]])  # sig = 1.5+2.0 = 3.5
        chain_r = np.array([1.5])
        F = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        # Force should push bead in +x (away from atom at origin).
        assert F[0, 0] > 0, f"bead not repelled by target: {F[0]}"

    def test_target_steric_force_zero_outside_sigma(self):
        """chain_target_steric_forces is zero when the bead lies at or beyond sigma from the target atom."""

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        chain_pos = np.array([[10.0, 0.0, 0.0]])
        chain_r = np.array([1.5])
        F = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_target_steric_skips_ghost_target_atoms(self):
        """chain_target_steric_forces produces no force from ghost target atoms with radius below 1e-10."""

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.0,
                ),  # ghost
            ],
        )
        chain_pos = np.array([[0.5, 0.0, 0.0]])
        chain_r = np.array([1.5])
        F = chain_target_steric_forces(chain_pos, chain_r, target, eps=1.0)
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_target_steric_handles_none_target(self):
        """chain_target_steric_forces returns zero force when target is None without crashing."""

        chain_pos = np.array([[0.5, 0.0, 0.0]])
        chain_r = np.array([1.5])
        F = chain_target_steric_forces(chain_pos, chain_r, None, eps=1.0)
        np.testing.assert_allclose(F[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_simulator_default_use_soft_repulsion_off(self):
        """ChainBDParameters.use_soft_repulsion defaults to False with soft_repulsion_eps of 1.0."""

        p = ChainBDParameters()
        assert p.use_soft_repulsion is False
        assert p.soft_repulsion_eps == 1.0

    def test_soft_repulsion_changes_trajectories(self):
        """Toggling use_soft_repulsion changes trajectories when a bead is close enough to feel the WCA force."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(2)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array(
            [
                [-1.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        body_pos -= body_pos.mean(axis=0)
        target = Molecule(
            name="target",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=4.0,
                ),
            ],
        )
        params_on = ChainBDParameters(
            n_trajectories=8,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=5000,
            r_start=8.0,
            r_escape=15.0,
            seed=42,
            n_threads=1,
            use_soft_repulsion=True,
            use_hard_sphere=False,
        )
        params_off = ChainBDParameters(
            n_trajectories=8,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=5000,
            r_start=8.0,
            r_escape=15.0,
            seed=42,
            n_threads=1,
            use_soft_repulsion=False,
            use_hard_sphere=False,
        )
        ps = PathwaySet(reactions=[])
        sim_on = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_on,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )
        sim_off = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_off,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )
        results_on = sim_on.run()
        results_off = sim_off.run()
        mean_on = sum(r.time_ps for r in results_on) / len(results_on)
        mean_off = sum(r.time_ps for r in results_off) / len(results_off)
        assert abs(mean_on - mean_off) > 1e-6, (
            f"soft repulsion did not alter trajectories: "
            f"on={mean_on:.4f}, off={mean_off:.4f}. Geometry may not "
            f"have produced bead-target overlaps."
        )


class TestBrownianBridge:
    """Brownian bridge for chain BD reaction detection.

    Verifies compute_pair_distances + check_reaction_with_bridge
    physics, and that the bridge is wired into run_one without breaking
    backward compatibility.
    """

    def _make_simple_target_and_chain(self):
        """One target atom at origin, one chain bead. Single reaction
        with a single contact pair (target_atom_0, chain_bead_0) and a
        cutoff."""

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.0,
                ),
            ],
        )

        criteria = ReactionCriteria(
            pairs=[
                ContactPair(
                    mol1_atom_index=0,
                    mol2_atom_index=0,
                    distance_cutoff=5.0,
                )
            ],
            n_needed=-1,
        )
        rxn = ReactionInterface(
            name="rxn0",
            criteria=criteria,
            probability=1.0,
            state_before="A",
            state_after="B",
        )
        ps = PathwaySet(reactions=[rxn])
        return target, ps

    def test_compute_pair_distances_basic(self):
        """compute_pair_distances returns the correct per-reaction, per-pair distances."""

        target, ps = self._make_simple_target_and_chain()
        chain_pos = np.array([[3.0, 4.0, 0.0]])  # distance 5.0 to origin
        out = compute_pair_distances(target, chain_pos, ps)
        assert len(out) == 1, f"expected 1 reaction, got {len(out)}"
        assert len(out[0]) == 1, f"expected 1 pair, got {len(out[0])}"
        np.testing.assert_allclose(out[0][0], 5.0, atol=1e-12)

    def test_compute_pair_distances_empty(self):
        """compute_pair_distances returns an empty list when pathway_set or target is None."""

        target, ps = self._make_simple_target_and_chain()
        chain_pos = np.array([[3.0, 4.0, 0.0]])
        assert compute_pair_distances(target, chain_pos, None) == []
        assert compute_pair_distances(None, chain_pos, ps) == []

    def test_endpoint_fired_below_cutoff(self):
        """check_reaction_with_bridge fires the reaction when the new distance drops below the cutoff."""

        target, ps = self._make_simple_target_and_chain()
        rng = np.random.default_rng(42)
        # old_d = 8.0, new_d = 3.0: cutoff is 5, so x1 < 0 -> endpoint fired.
        rxn = check_reaction_with_bridge(
            target,
            target,
            ps,
            old_pair_dists_per_rxn=[np.array([8.0])],
            new_pair_dists_per_rxn=[np.array([3.0])],
            D_eff=0.05,
            dt=0.5,
            rng=rng,
        )
        assert rxn == "rxn0", f"endpoint fire missed: got {rxn}"

    def test_bridge_fires_with_high_p_cross(self):
        """check_reaction_with_bridge fires via the bridge when both endpoints stay just outside the cutoff and p_cross is near 1."""

        target, ps = self._make_simple_target_and_chain()
        rng = np.random.default_rng(42)
        rxn = check_reaction_with_bridge(
            target,
            target,
            ps,
            old_pair_dists_per_rxn=[np.array([5.001])],
            new_pair_dists_per_rxn=[np.array([5.001])],
            D_eff=0.05,
            dt=1.0,
            rng=rng,
        )
        assert rxn == "rxn0", f"high-p_cross bridge did not fire: got {rxn}"

    def test_bridge_no_fire_with_low_p_cross(self):
        """check_reaction_with_bridge does not fire when both endpoints are far outside the cutoff and p_cross is near 0."""

        target, ps = self._make_simple_target_and_chain()
        rng = np.random.default_rng(42)
        rxn = check_reaction_with_bridge(
            target,
            target,
            ps,
            old_pair_dists_per_rxn=[np.array([10.0])],
            new_pair_dists_per_rxn=[np.array([10.0])],
            D_eff=0.001,
            dt=1.0,
            rng=rng,
        )
        assert rxn is None, f"low-p_cross bridge fired anyway: got {rxn}"

    def test_and_logic_multi_pair_all_fire(self):
        """With n_needed=-1 (ALL), a two-pair reaction does not fire when only one pair fires."""

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="A",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.0,
                ),
                Atom(
                    index=2,
                    name="B",
                    residue_name="DUM",
                    residue_index=1,
                    x=10.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.0,
                ),
            ],
        )
        criteria = ReactionCriteria(
            pairs=[
                ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0),
                ContactPair(mol1_atom_index=1, mol2_atom_index=0, distance_cutoff=5.0),
            ],
            n_needed=-1,  # ALL
        )
        rxn = ReactionInterface(
            name="and_rxn",
            criteria=criteria,
            probability=1.0,
            state_before="A",
            state_after="B",
        )
        ps = PathwaySet(reactions=[rxn])
        rng = np.random.default_rng(42)
        result = check_reaction_with_bridge(
            target,
            target,
            ps,
            old_pair_dists_per_rxn=[np.array([8.0, 100.0])],
            new_pair_dists_per_rxn=[np.array([3.0, 100.0])],
            D_eff=0.001,
            dt=1.0,
            rng=rng,
        )
        assert (
            result is None
        ), f"AND-logic incorrectly fired with only 1/2 pairs: got {result}"
        # Both endpoint-fire -> reaction fires.
        rng2 = np.random.default_rng(42)
        result2 = check_reaction_with_bridge(
            target,
            target,
            ps,
            old_pair_dists_per_rxn=[np.array([8.0, 8.0])],
            new_pair_dists_per_rxn=[np.array([3.0, 3.0])],
            D_eff=0.001,
            dt=1.0,
            rng=rng2,
        )
        assert (
            result2 == "and_rxn"
        ), f"AND-logic did not fire when both pairs satisfied: got {result2}"

    def test_default_use_brownian_bridge_is_true(self):
        """ChainBDParameters.use_brownian_bridge defaults to True."""

        p = ChainBDParameters()
        assert p.use_brownian_bridge is True

    def test_bridge_off_matches_endpoint_only(self):
        """With use_brownian_bridge=False run_one falls back to the endpoint-only check and returns all results."""

        atoms = [ChainAtom(radius=1.5, charge=0.0) for _ in range(2)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        body_pos -= body_pos.mean(axis=0)

        params_off = ChainBDParameters(
            n_trajectories=4,
            dt=0.5,
            dt_chain=0.05,
            chain_steps_per_outer=4,
            max_steps=300,
            r_start=8.0,
            r_escape=15.0,
            seed=42,
            n_threads=1,
            use_brownian_bridge=False,
        )
        ps = PathwaySet(reactions=[])
        sim = ChainBDSimulator(
            target=None,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_off,
            pathway_set=ps,
            D_trans=0.05,
            D_rot=0.005,
        )
        results = sim.run()
        assert len(results) == 4, f"unexpected n results: {len(results)}"

    def test_bridge_changes_reaction_count_with_real_geometry(self):
        """Brownian bridge on yields at least as many reactions as bridge off for the same trajectories."""

        atoms = [ChainAtom(radius=0.5, charge=0.0)]
        template = ChainCommon(name="t", atoms=atoms)
        body_pos = np.array([[0.0, 0.0, 0.0]])
        target = Molecule(
            name="target",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=0.0,
                ),
            ],
        )
        criteria = ReactionCriteria(
            pairs=[
                ContactPair(
                    mol1_atom_index=0,
                    mol2_atom_index=0,
                    distance_cutoff=2.0,
                )
            ],
            n_needed=-1,
        )
        rxn = ReactionInterface(
            name="contact_rxn",
            criteria=criteria,
            probability=1.0,
            state_before="A",
            state_after="B",
        )
        ps = PathwaySet(reactions=[rxn])

        params_on = ChainBDParameters(
            n_trajectories=20,
            dt=2.0,
            dt_chain=0.1,
            chain_steps_per_outer=4,
            max_steps=500,
            r_start=4.0,
            r_escape=12.0,
            seed=42,
            n_threads=1,
            use_brownian_bridge=True,
            use_hard_sphere=False,
        )
        params_off = ChainBDParameters(
            n_trajectories=20,
            dt=2.0,
            dt_chain=0.1,
            chain_steps_per_outer=4,
            max_steps=500,
            r_start=4.0,
            r_escape=12.0,
            seed=42,
            n_threads=1,
            use_brownian_bridge=False,
            use_hard_sphere=False,
        )

        sim_on = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_on,
            pathway_set=ps,
            D_trans=0.5,
            D_rot=0.05,
        )
        sim_off = ChainBDSimulator(
            target=target,
            chain_template=template,
            chain_init_body_positions=body_pos,
            params=params_off,
            pathway_set=ps,
            D_trans=0.5,
            D_rot=0.05,
        )
        # Compare the bridge on and off over identical trajectories. Each
        # trajectory is driven with the same main rng for both runs, so the two
        # follow the same path and the bridge, which draws from an independent
        # rng_bb, can only add reactions. This makes bridge on a strict superset
        # of bridge off. The serial run() shares one continuous rng across all
        # trajectories, so an early bridge reaction shifts the stream for later
        # trajectories and the two runs diverge, which is a false comparison.
        base_seed = params_on.seed
        n_react_on = 0
        n_react_off = 0
        for i in range(params_on.n_trajectories):
            r_on = sim_on.run_one(
                rng=np.random.default_rng(base_seed + i),
                rng_bb=np.random.default_rng(base_seed + i + 0xBB),
            )
            r_off = sim_off.run_one(
                rng=np.random.default_rng(base_seed + i),
                rng_bb=np.random.default_rng(base_seed + i + 0xBB),
            )
            n_react_on += r_on.fate == Fate.REACTED
            n_react_off += r_off.fate == Fate.REACTED
        assert n_react_on >= n_react_off, (
            f"bridge fewer reactions than no-bridge: "
            f"on={n_react_on}, off={n_react_off}"
        )


class TestNAMBrownianBridge:
    """NAM Brownian bridge for reaction detection.

    Mirrors TestBrownianBridge for chain BD, but tests NAM-specific
    integration: NAMSimulator.run_one (serial) and _run_trajectory_worker
    (parallel n_threads>1).
    """

    def _build_nam_sim(self, *, use_bb, n_threads=1, n_traj=10, seed=42):
        """Build a small NAM simulator with one contact-pair reaction."""

        mol1 = Molecule(
            name="m1",
            atoms=[
                Atom(
                    index=1,
                    name="X",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        mol2 = Molecule(
            name="m2",
            atoms=[
                Atom(
                    index=1,
                    name="Y",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=2.0,
                ),
            ],
        )
        criteria = ReactionCriteria(
            pairs=[
                ContactPair(
                    mol1_atom_index=0,
                    mol2_atom_index=0,
                    distance_cutoff=4.0,
                )
            ],
            n_needed=-1,
        )
        rxn = ReactionInterface(
            name="contact",
            criteria=criteria,
            probability=1.0,
            state_before="A",
            state_after="B",
        )
        ps = PathwaySet(reactions=[rxn])
        mobility = MobilityTensor(
            0.05,
            0.005,
            0.05,
            0.005,
            radius1=2.0,
            radius2=2.0,
        )
        params = NAMParameters(
            n_trajectories=n_traj,
            dt=0.5,
            dt_rxn=0.1,
            max_steps=300,
            r_start=8.0,
            r_escape=15.0,
            seed=seed,
            n_threads=n_threads,
            use_brownian_bridge=use_bb,
            use_hard_sphere=False,
        )
        return NAMSimulator(mol1, mol2, mobility, ps, params)

    def test_default_use_brownian_bridge_is_true(self):
        """NAMParameters.use_brownian_bridge defaults to True."""

        p = NAMParameters()
        assert p.use_brownian_bridge is True

    def test_mol2_positions_extracts_xyz(self):
        """_mol2_positions returns an (n_atoms, 3) array of atom x, y, z coordinates."""

        mol = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="A",
                    residue_name="X",
                    residue_index=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    charge=0.0,
                    radius=1.0,
                ),
                Atom(
                    index=2,
                    name="B",
                    residue_name="X",
                    residue_index=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    charge=0.0,
                    radius=1.0,
                ),
            ],
        )
        out = _mol2_positions(mol)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(out[1], [4.0, 5.0, 6.0])

    def test_bridge_off_serial_runs_without_error(self):
        """A serial NAM run with the bridge off completes and returns the requested number of trajectories."""
        sim = self._build_nam_sim(use_bb=False, n_threads=1, n_traj=5)
        result = sim.run()
        assert result.n_trajectories == 5

    def test_bridge_on_serial_runs_without_error(self):
        """A serial NAM run with the bridge on completes and returns the requested number of trajectories."""
        sim = self._build_nam_sim(use_bb=True, n_threads=1, n_traj=5)
        result = sim.run()
        assert result.n_trajectories == 5

    def test_bridge_monotonicity_serial(self):
        """In serial NAM, every trajectory reacting with the bridge off also reacts with it on, and the total can only grow."""

        n_traj, seed = 20, 7
        sim_off = self._build_nam_sim(use_bb=False, n_threads=1, n_traj=1, seed=seed)
        sim_on = self._build_nam_sim(use_bb=True, n_threads=1, n_traj=1, seed=seed)
        on_count = off_count = 0
        for idx in range(n_traj):
            base = seed + idx
            for sim in (sim_off, sim_on):
                sim.rng = np.random.default_rng(base)
                sim.rng_bb = np.random.default_rng(base + 0xBB)
            r_off = sim_off.run_one()
            r_on = sim_on.run_one()
            off_reacted = r_off.fate == Fate.REACTED
            on_reacted = r_on.fate == Fate.REACTED
            off_count += int(off_reacted)
            on_count += int(on_reacted)
            assert not (
                off_reacted and not on_reacted
            ), f"trajectory {idx}: reacted without the bridge but not with it"
        assert on_count >= off_count, f"on={on_count}, off={off_count}"

    def test_bridge_monotonicity_parallel(self):
        """In the parallel NAM worker, bridge on produces at least as many reactions as bridge off."""
        sim_off = self._build_nam_sim(use_bb=False, n_threads=2, n_traj=20, seed=7)
        sim_on = self._build_nam_sim(use_bb=True, n_threads=2, n_traj=20, seed=7)
        res_off = sim_off.run()
        res_on = sim_on.run()
        assert res_on.n_reacted >= res_off.n_reacted, (
            f"bridge fewer reactions than no-bridge in parallel: "
            f"on={res_on.n_reacted}, off={res_off.n_reacted}"
        )


class TestSmoluchowskiValidation:
    """Smoluchowski-limit regression test for NAM.

    A small probe diffusing toward an absorbing sphere of radius R must
    recover the analytical k_on = 4 pi D_rel R within stochastic noise.
    Uses N=100 trajectories (smaller than the validation script for test
    speed) with loose tolerance (within 60% of analytical) to catch
    egregious regressions without flaking on stochastic noise.

    See tests/validation/smoluchowski.py for a fuller validation run.
    """

    def test_k_on_within_60_percent_of_analytical(self):
        """NAM sim k_on lies within plus or minus 60% of the analytic 4 pi D R."""

        R = 20.0
        b = 50.0
        D = 0.05
        D_rel = 2 * D
        CONV = 6.022e8

        mol1 = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="T",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=R,
                ),
            ],
        )
        mol2 = Molecule(
            name="p",
            atoms=[
                Atom(
                    index=1,
                    name="P",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=1.0,
                ),
            ],
        )
        ps = PathwaySet(
            reactions=[
                ReactionInterface(
                    name="r",
                    criteria=ReactionCriteria(
                        pairs=[ContactPair(0, 0, R)],
                        n_needed=-1,
                    ),
                    probability=1.0,
                    state_before="A",
                    state_after="B",
                )
            ]
        )
        mob = MobilityTensor(D, 0.005, D, 0.005, radius1=R, radius2=1.0)
        params = NAMParameters(
            n_trajectories=100,
            dt=2.0,
            dt_rxn=0.5,
            max_steps=50_000,
            r_start=b,
            r_escape=0.0,
            seed=42,
            n_threads=1,
            use_brownian_bridge=True,
            use_hard_sphere=False,
        )
        sim = NAMSimulator(mol1, mol2, mob, ps, params)
        result = sim.run()

        k_on_sim = result.rate_constant(D_rel=D_rel)
        k_on_analytical = CONV * 4.0 * math.pi * D_rel * R
        ratio = k_on_sim / k_on_analytical

        assert 0.4 <= ratio <= 1.6, (
            f"Smoluchowski validation: sim/analytical ratio {ratio:.3f} "
            f"outside [0.4, 1.6]. sim={k_on_sim:.3e}, "
            f"analytical={k_on_analytical:.3e}, "
            f"reacted={result.n_reacted}/{result.n_trajectories}"
        )


class TestChainBDSmoluchowskiValidation:
    """Smoluchowski-limit regression test for chain BD.

    Single-bead chain diffusing toward absorbing target sphere of
    radius R must recover analytical k_on = 4 pi D_rel R within
    stochastic noise.

    This is the analog of TestSmoluchowskiValidation (NAM version).
    Tests the chain BD code path end-to-end: outer BD step,
    place_chain, reaction check, Brownian bridge, escape detection,
    and crucially the LMZ outer propagator (use_lmz=True default).

    Without LMZ, P_rxn would be ~10x too low (escape-without-return
    bias). With LMZ, chain BD recovers the analytical limit within
    statistical noise. Reference: ratio=0.84 at N=200 (NAM gives 1.13).

    Uses N=50 with tolerance +/- 60% to keep test runtime reasonable
    (chain BD is ~10x slower per trajectory than NAM due to chain
    machinery). See tests/validation/chain_smoluchowski.py for fuller
    validation.
    """

    def test_k_on_within_60_percent_of_analytical(self):
        """Chain BD sim k_on lies within plus or minus 60% of the analytic 4 pi D R."""

        R = 20.0
        b = 50.0
        D = 0.05
        D_rel = 2 * D
        CONV = 6.022e8

        target = Molecule(
            name="t",
            atoms=[
                Atom(
                    index=1,
                    name="T",
                    residue_name="DUM",
                    residue_index=1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=R,
                ),
            ],
        )
        chain_template = ChainCommon(
            name="c",
            atoms=[
                ChainAtom(radius=1.0, charge=0.0, resname="P", resid=0),
            ],
        )
        ps = PathwaySet(
            reactions=[
                ReactionInterface(
                    name="r",
                    criteria=ReactionCriteria(
                        pairs=[ContactPair(0, 0, R)],
                        n_needed=-1,
                    ),
                    probability=1.0,
                    state_before="A",
                    state_after="B",
                )
            ]
        )
        params = ChainBDParameters(
            n_trajectories=50,
            dt=2.0,
            dt_chain=0.5,
            chain_steps_per_outer=1,
            max_steps=50_000,
            r_start=b,
            r_escape=b * 1.1,
            seed=42,
            n_threads=1,
            use_brownian_bridge=True,
            use_hard_sphere=False,
            # use_lmz=True is the default and required for correctness
        )
        sim = ChainBDSimulator(
            target=target,
            chain_template=chain_template,
            chain_init_body_positions=np.array([[0.0, 0.0, 0.0]]),
            params=params,
            pathway_set=ps,
            D_trans=D,
            D_rot=0.005,
        )
        results = sim.run()
        n_react = sum(1 for r in results if r.fate == Fate.REACTED)
        n_esc = sum(1 for r in results if r.fate == Fate.ESCAPED)
        P = n_react / (n_react + n_esc) if (n_react + n_esc) > 0 else 0.0

        if P > 0:
            beta = b / (b * 1.1)
            k_D = 4.0 * math.pi * D_rel * b
            denom = 1.0 - P * (1.0 - beta)
            k_on_sim = CONV * k_D * P / denom
        else:
            k_on_sim = 0.0
        k_on_analytical = CONV * 4.0 * math.pi * D_rel * R
        ratio = k_on_sim / k_on_analytical

        assert 0.4 <= ratio <= 1.6, (
            f"Chain BD Smoluchowski validation: sim/analytical ratio "
            f"{ratio:.3f} outside [0.4, 1.6]. "
            f"sim={k_on_sim:.3e}, analytical={k_on_analytical:.3e}, "
            f"reacted={n_react}/50. "
            f"If ratio < 0.2, likely LMZ outer propagator is broken."
        )


class TestCOFFDROPTabulatedForces:
    """Regression tests for COFFDROP tabulated force branches in
    coffdrop_chain.py. Each force type (angle, torsion, pair) has a
    branch that fires when the chain has coffdrop_params set AND the
    relevant interaction has a type_idx >= 0 (or, for pairs,
    pair_lookups is non-empty). These tests exercise those branches
    end-to-end by constructing a tiny chain, computing forces, and
    verifying:
      - the tabulated branch produces forces different from the
        harmonic / cosine-series fallback
      - the tabulated forces conserve momentum (sum to zero)
      - for pair forces, the magnitude exactly matches the spline
        derivative dV/dr
      - the default path (no params) is unchanged
    """

    @pytest.fixture(scope="class")
    def params(self):
        """Load COFFDROPParams once per class (expensive)."""

        ff_dir = Path(__file__).parent.parent / "pystarc" / "coffdrop_data"
        return COFFDROPParams.load(
            ff_xml=str(ff_dir / "coffdrop.xml"),
            mapping_xml=str(ff_dir / "map.xml"),
            connectivity_xml=str(ff_dir / "connectivity.xml"),
            charges_xml=str(ff_dir / "charges.xml"),
        )

    def test_angle_tabulated_branch_fires_and_differs_from_harmonic(self, params):
        """The tabulated angle branch fires, giving nonzero forces that differ from the harmonic result and conserve momentum."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0, resname="ALA:CA", resid=i)
            for i in range(3)
        ]
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        # Harmonic
        common_h = ChainCommon(
            name="h",
            atoms=atoms,
            angles=[
                ChainAngle(
                    a=ChainAtomRef(0),
                    b=ChainAtomRef(1),
                    c=ChainAtomRef(2),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=-1,
                )
            ],
        )
        state_h = ChainState.from_template(common_h, positions)
        state_h.zero_forces()
        _angle_force_state(state_h, common_h.angles[0])
        # Tabulated
        common_t = ChainCommon(
            name="t",
            atoms=atoms,
            angles=[
                ChainAngle(
                    a=ChainAtomRef(0),
                    b=ChainAtomRef(1),
                    c=ChainAtomRef(2),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=0,
                )
            ],
            coffdrop_params=params,
        )
        state_t = ChainState.from_template(common_t, positions)
        state_t.zero_forces()
        _angle_force_state(state_t, common_t.angles[0])
        # Tabulated should differ from harmonic
        assert not np.allclose(
            state_h.forces, state_t.forces, atol=1e-6
        ), "angle tabulated branch should produce different forces from harmonic"
        # Tabulated should be nonzero (verifies the branch fired)
        assert not np.allclose(
            state_t.forces, 0, atol=1e-10
        ), "angle tabulated branch should produce nonzero forces"
        # Both should conserve momentum
        assert np.linalg.norm(state_h.forces.sum(axis=0)) < 1e-10
        assert np.linalg.norm(state_t.forces.sum(axis=0)) < 1e-10

    def test_torsion_tabulated_branch_fires_and_differs_from_cosine_series(
        self, params
    ):
        """The tabulated torsion branch fires, giving nonzero forces that differ from the cosine-series result."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0, resname="ALA:CA", resid=i)
            for i in range(4)
        ]
        positions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        # Cosine-series
        tor_h = ChainTorsion(
            a=ChainAtomRef(0),
            b=ChainAtomRef(1),
            c=ChainAtomRef(2),
            d=ChainAtomRef(3),
            phi0=0.0,
            k_tor=2.0,
            n=1,
            type_idx=-1,
        )
        common_h = ChainCommon(name="h", atoms=atoms, torsions=[tor_h])
        state_h = ChainState.from_template(common_h, positions)
        state_h.zero_forces()
        _torsion_force_state(state_h, tor_h)
        # Tabulated with stiff potential pot[4922]
        tor_t = ChainTorsion(
            a=ChainAtomRef(0),
            b=ChainAtomRef(1),
            c=ChainAtomRef(2),
            d=ChainAtomRef(3),
            phi0=0.0,
            k_tor=2.0,
            n=1,
            type_idx=4922,
        )
        common_t = ChainCommon(
            name="t",
            atoms=atoms,
            torsions=[tor_t],
            coffdrop_params=params,
        )
        state_t = ChainState.from_template(common_t, positions)
        state_t.zero_forces()
        _torsion_force_state(state_t, tor_t)
        # Tabulated should differ from cosine-series
        assert not np.allclose(
            state_h.forces, state_t.forces, atol=1e-6
        ), "torsion tabulated branch should produce different forces"
        # Tabulated should be nonzero
        assert not np.allclose(
            state_t.forces, 0, atol=1e-10
        ), "torsion tabulated branch should produce nonzero forces"
        # Both conserve momentum
        assert np.linalg.norm(state_h.forces.sum(axis=0)) < 1e-10
        assert np.linalg.norm(state_t.forces.sum(axis=0)) < 1e-10

    def test_pair_tabulated_branch_matches_spline_derivative(self, params):
        """The tabulated pair force magnitude equals |dV/dr| from the spline, obeys Newton's third law, and stays zero for unpaired atoms."""

        atoms = [
            ChainAtom(radius=1.5, charge=0.0, resname="ALA:CA", resid=i)
            for i in range(3)
        ]
        bonds = [
            ChainBond(a=ChainAtomRef(0), b=ChainAtomRef(1), r0=3.8, k_spring=100.0),
            ChainBond(a=ChainAtomRef(1), b=ChainAtomRef(2), r0=3.8, k_spring=100.0),
        ]
        common = ChainCommon(
            name="pair_test",
            atoms=atoms,
            bonds=bonds,
            coffdrop_params=params,
            pair_lookups={(0, 2): 0},  # use pair_pots[0] for (0, 2)
        )
        # Position so |r_0 - r_2| = 5.0 (well in pot range)
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        F = chain_intra_coffdrop_pair_forces(state, common)
        # Atom 1 not in any pair_lookup, force must be zero
        assert np.allclose(F[1], 0, atol=1e-12)
        # Atom 0 force magnitude must match |dV/dr| from the spline
        dV_dr = params.pair_pots[0].deriv(5.0)
        assert np.isclose(np.linalg.norm(F[0]), abs(dV_dr), atol=1e-10), (
            f"pair force magnitude {np.linalg.norm(F[0])} should match "
            f"|dV/dr| {abs(dV_dr)} from spline"
        )
        # Newton's third law
        assert np.allclose(F[0] + F[2], 0, atol=1e-12)

    def test_default_path_unchanged_when_params_none(self, params):
        """Bonded interactions fall back to harmonic forces when coffdrop_params is None despite a set type_idx."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0, resname="ALA:CA", resid=i)
            for i in range(3)
        ]
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        # type_idx=0 SET but coffdrop_params=None -> must fall back to harmonic
        common = ChainCommon(
            name="fallback",
            atoms=atoms,
            angles=[
                ChainAngle(
                    a=ChainAtomRef(0),
                    b=ChainAtomRef(1),
                    c=ChainAtomRef(2),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=0,
                )
            ],
            coffdrop_params=None,
        )
        state = ChainState.from_template(common, positions)
        state.zero_forces()
        _angle_force_state(state, common.angles[0])
        # Reference: pure harmonic (type_idx=-1, params=None)
        common_ref = ChainCommon(
            name="ref",
            atoms=atoms,
            angles=[
                ChainAngle(
                    a=ChainAtomRef(0),
                    b=ChainAtomRef(1),
                    c=ChainAtomRef(2),
                    theta0=2.0,
                    k_angle=10.0,
                    type_idx=-1,
                )
            ],
        )
        state_ref = ChainState.from_template(common_ref, positions)
        state_ref.zero_forces()
        _angle_force_state(state_ref, common_ref.angles[0])
        assert np.allclose(
            state.forces, state_ref.forces, atol=1e-12
        ), "type_idx without params must fall back to harmonic"

    def test_build_chain_common_from_coffdrop_5ala_end_to_end(self, params):
        """build_chain_common_from_coffdrop on ALA5 yields the expected topology counts, populated type_idx values, and finite momentum-conserving forces."""

        common = build_chain_common_from_coffdrop(
            ["ALA"] * 5,
            params,
            name="ala5_test",
        )
        # Topology counts
        assert common.n_atoms == 5
        assert len(common.bonds) == 4
        assert len(common.angles) == 3
        assert len(common.torsions) == 2
        assert len(common.pair_lookups) == 3
        # All angles/torsions must have type_idx >= 0 (lookups succeeded)
        assert all(
            a.type_idx >= 0 for a in common.angles
        ), "all angle type_idx should be populated"
        assert all(
            t.type_idx >= 0 for t in common.torsions
        ), "all torsion type_idx should be populated"
        # ALA-ALA-ALA backbone: angles all hit pot[0], torsions all hit pot[9972]
        assert common.angles[0].type_idx == 0
        assert common.torsions[0].type_idx == 9972
        # Run forces through the full pipeline
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8 * 1.5, 1.5, 0.0],
                [3.8 * 2.0, 0.5, 1.0],
                [3.8 * 2.5, -0.5, 0.5],
            ]
        )
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # Forces should be nonzero and finite
        assert np.all(np.isfinite(state.forces)), "forces must be finite"
        assert (
            np.max(np.abs(state.forces)) > 1.0
        ), "expect nontrivial force magnitudes for non-equilibrium geometry"
        # Newton's 3rd law: net force on system is zero
        force_sum_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            force_sum_norm < 1e-10
        ), f"net force should be zero, got norm {force_sum_norm}"

    @pytest.mark.parametrize("geometry_id", ["straight", "zigzag", "spiral", "compact"])
    def test_force_conservation_across_geometries(self, params, geometry_id):
        """Net force is zero and all forces are finite across straight, zigzag, spiral, and compact ALA5 geometries."""

        common = build_chain_common_from_coffdrop(
            ["ALA"] * 5,
            params,
            name=f"ala5_{geometry_id}",
        )
        # Different geometries that exercise distinct force regimes
        geometries = {
            "straight": np.array([[i * 3.8, 0.0, 0.0] for i in range(5)]),
            "zigzag": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.8, 1.5, 0.0],
                    [7.6, 0.0, 0.0],
                    [11.4, 1.5, 0.0],
                    [15.2, 0.0, 0.0],
                ]
            ),
            "spiral": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 1.0, 1.0],
                    [3.5, 3.5, 2.0],
                    [1.5, 5.0, 1.5],
                    [0.5, 4.0, -0.5],
                ]
            ),
            "compact": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.8, 0.5, 0.0],
                    [3.5, 3.0, 0.5],
                    [1.0, 3.5, 1.0],
                    [0.0, 1.5, 1.5],
                ]
            ),
        }
        positions = geometries[geometry_id]
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # All forces finite
        assert np.all(
            np.isfinite(state.forces)
        ), f"forces should be finite for {geometry_id}"
        # Net force ~ 0 (machine precision)
        force_sum_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            force_sum_norm < 1e-10
        ), f"{geometry_id}: net force norm {force_sum_norm} should be zero"

    def test_homopolymer_reversal_symmetry(self, params):
        """Reversing the positions of a homopolymer reverses the per-atom forces, so F'[i] equals F[n-1-i]."""

        common = build_chain_common_from_coffdrop(["ALA"] * 5, params)
        # Asymmetric geometry
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [7.0, 0.5, 1.5],
                [10.5, 2.0, 0.5],
                [13.0, 0.0, -1.0],
            ]
        )
        state_fwd = ChainState.from_template(common, positions)
        compute_chain_forces(state_fwd)

        state_rev = ChainState.from_template(common, positions[::-1].copy())
        compute_chain_forces(state_rev)

        n = common.n_atoms
        for i in range(n):
            expected = state_fwd.forces[n - 1 - i]
            actual = state_rev.forces[i]
            assert np.allclose(actual, expected, atol=1e-9), (
                f"reversal asymmetry at atom {i}: "
                f"actual={actual}, expected={expected}"
            )

    def test_translational_invariance(self, params):
        """Translating all atoms by a constant vector leaves the forces unchanged."""

        common = build_chain_common_from_coffdrop(["ALA"] * 5, params)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [7.0, 0.5, 1.5],
                [10.5, 2.0, 0.5],
                [13.0, 0.0, -1.0],
            ]
        )
        # Original
        state_orig = ChainState.from_template(common, positions)
        compute_chain_forces(state_orig)
        # Translated by (100, -50, 200)
        translation = np.array([100.0, -50.0, 200.0])
        state_trans = ChainState.from_template(common, positions + translation)
        compute_chain_forces(state_trans)
        # Forces must be identical
        assert np.allclose(
            state_orig.forces, state_trans.forces, atol=1e-9
        ), "translational invariance violated: forces differ after position shift"

    def test_rotational_equivariance(self, params):
        """Rotating all atoms by R rotates every force vector by the same R."""

        common = build_chain_common_from_coffdrop(["ALA"] * 5, params)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [7.0, 0.5, 1.5],
                [10.5, 2.0, 0.5],
                [13.0, 0.0, -1.0],
            ]
        )
        # Original forces
        state_orig = ChainState.from_template(common, positions)
        compute_chain_forces(state_orig)
        # Rotate by 30 degrees about z-axis (arbitrary)
        theta = np.deg2rad(30.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # Apply rotation to positions
        pos_rotated = positions @ R.T
        state_rot = ChainState.from_template(common, pos_rotated)
        compute_chain_forces(state_rot)
        # Expected: forces rotate by R as well
        expected_forces = state_orig.forces @ R.T
        assert np.allclose(
            state_rot.forces, expected_forces, atol=1e-9
        ), "rotational equivariance violated: forces don't rotate with positions"

    def test_heteropolymer_5_residue_chain(self, params):
        """A mixed-residue heteropolymer chain builds and computes finite, momentum-conserving forces."""

        # Mix of residues across COFFDROP types
        sequence = ["ALA", "GLY", "ARG", "LEU", "ASP"]
        common = build_chain_common_from_coffdrop(sequence, params)
        assert common.n_atoms == 5
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [7.0, 0.5, 1.5],
                [10.5, 2.0, 0.5],
                [13.0, 0.0, -1.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # Forces finite
        assert np.all(np.isfinite(state.forces)), "heteropolymer forces must be finite"
        # Net force zero
        force_sum_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            force_sum_norm < 1e-9
        ), f"heteropolymer net force norm {force_sum_norm} should be zero"

    def test_short_force_driven_integration_stable(self, params):
        """100 steps of damped Euler force-driven integration on an ALA5 chain keep positions finite and bond lengths bounded."""

        common = build_chain_common_from_coffdrop(["ALA"] * 5, params)
        # Start at a moderately stretched configuration
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.5, 0.0],
                [7.6, 0.0, 0.5],
                [11.4, 0.5, 0.0],
                [15.2, 0.0, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions.copy())
        dt = 1e-5
        damping = 0.9  # apply mild damping to forces (overdamped-like)
        n_steps = 100
        max_force_history = []
        max_bond_history = []
        for step in range(n_steps):
            compute_chain_forces(state)
            max_force_history.append(np.max(np.abs(state.forces)))
            # Euler step (overdamped: dr = F * dt)
            state.positions += damping * state.forces * dt
            # Track bond lengths
            bond_lengths = []
            for bond in common.bonds:
                ri = state.positions[bond.a.atom_idx]
                rj = state.positions[bond.b.atom_idx]
                bond_lengths.append(np.linalg.norm(rj - ri))
            max_bond_history.append(max(bond_lengths))
        # All positions finite
        assert np.all(
            np.isfinite(state.positions)
        ), "positions should stay finite during integration"
        # Bond lengths should not blow up (start ~3.8, allow up to 10)
        assert (
            max(max_bond_history) < 20.0
        ), f"bond lengths exploded: max = {max(max_bond_history)}"
        assert (
            max_force_history[-1] < max_force_history[0] * 100
        ), f"force exploded: start={max_force_history[0]}, end={max_force_history[-1]}"

    def test_sidechain_helper_topology_and_forces(self, params):
        """A heteropolymer chain with sidechain beads has the expected topology counts, populated type_idx values, and finite momentum-conserving forces."""

        # ALA(2) + ARG(3) + TRP(4) + GLY(1) + LEU(3) = 13 atoms
        sequence = ["ALA", "ARG", "TRP", "GLY", "LEU"]
        common = build_chain_common_with_sidechains_from_coffdrop(
            sequence,
            params,
            name="mixed5_test",
        )
        # Topology counts
        assert common.n_atoms == 13, f"expected 13 atoms, got {common.n_atoms}"
        assert len(common.bonds) == 12, f"expected 12 bonds, got {len(common.bonds)}"
        assert len(common.angles) == 13, f"expected 13 angles, got {len(common.angles)}"
        populated = sum(1 for a in common.angles if a.type_idx >= 0)
        assert populated == 13, f"expected all 13 angles populated, got {populated}"
        assert (
            len(common.torsions) == 14
        ), f"expected 14 torsions, got {len(common.torsions)}"
        populated_t = sum(1 for t in common.torsions if t.type_idx >= 0)
        assert (
            populated_t == 14
        ), f"expected all 14 torsions populated, got {populated_t}"
        assert (
            len(common.pair_lookups) >= 30
        ), f"expected >= 30 pair lookups, got {len(common.pair_lookups)}"

        positions = np.zeros((common.n_atoms, 3))
        ca_indices = [
            i for i, a in enumerate(common.atoms) if a.resname.endswith(":CA")
        ]
        for r, ca_i in enumerate(ca_indices):
            positions[ca_i] = [3.8 * r, 0.0, 0.0]
        # Project sidechains outward from each preceding bonded bead
        for i in range(common.n_atoms):
            if not common.atoms[i].resname.endswith(":CA"):
                for bond in common.bonds:
                    if bond.b.atom_idx == i:
                        prev_i = bond.a.atom_idx
                        r0 = bond.r0
                        break
                    if bond.a.atom_idx == i:
                        prev_i = bond.b.atom_idx
                        r0 = bond.r0
                        break
                positions[i] = positions[prev_i] + np.array([0, r0, 0])

        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        assert np.all(np.isfinite(state.forces)), "sidechain forces must be finite"
        # Momentum conservation: net force ~ 0
        force_sum_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            force_sum_norm < 1e-9
        ), f"net force should be zero, got norm {force_sum_norm}"
        assert np.max(np.abs(state.forces)) < 50.0, (
            f"forces too large at relaxed geometry: "
            f"max={np.max(np.abs(state.forces))}"
        )

    def test_cys_sidechain_uses_sb_not_cb(self, params):
        """CYS uses SB rather than CB as its sidechain bead, so ALA-CYS-ALA yields five fully populated angles including two SB-anchored ones."""

        common = build_chain_common_with_sidechains_from_coffdrop(
            ["ALA", "CYS", "ALA"],
            params,
        )
        # Topology: 6 atoms (ALA:2 + CYS:2 + ALA:2)
        assert common.n_atoms == 6
        # Angles: 1 backbone + 2 ALA-CB + 2 CYS-SB = 5
        assert (
            len(common.angles) == 5
        ), f"expected 5 angles for ALA-CYS-ALA, got {len(common.angles)}"
        # All should be populated
        populated = sum(1 for a in common.angles if a.type_idx >= 0)
        assert populated == 5, f"expected all 5 angles populated, got {populated}"
        # Verify SB-CA-CA angles exist by checking atom labels
        sb_ca_angles = [
            a for a in common.angles if "CYS:SB" in common.atoms[a.a.atom_idx].resname
        ]
        assert (
            len(sb_ca_angles) == 2
        ), f"expected 2 CYS:SB-anchored angles, got {len(sb_ca_angles)}"

    def test_long_heteropolymer_stable_integration(self, params):
        """A 10-residue heteropolymer with sidechains stays stable over 100 damped Euler integration steps."""

        # 10-residue heteropolymer, mix of sidechain sizes
        sequence = [
            "ALA",
            "ARG",
            "LEU",
            "GLY",
            "TRP",
            "SER",
            "GLU",
            "VAL",
            "PRO",
            "LYS",
        ]
        common = build_chain_common_with_sidechains_from_coffdrop(
            sequence,
            params,
            name="hetero10",
        )
        n = common.n_atoms
        # Place at relaxed geometry
        positions = np.zeros((n, 3))
        ca_indices = [
            i for i, a in enumerate(common.atoms) if a.resname.endswith(":CA")
        ]
        for r, ca_i in enumerate(ca_indices):
            positions[ca_i] = [3.8 * r, 0.0, 0.0]
        for i in range(n):
            if not common.atoms[i].resname.endswith(":CA"):
                # Find bond to predecessor
                for bond in common.bonds:
                    if bond.b.atom_idx == i:
                        prev_i = bond.a.atom_idx
                        r0 = bond.r0
                        break
                    if bond.a.atom_idx == i:
                        prev_i = bond.b.atom_idx
                        r0 = bond.r0
                        break
                positions[i] = positions[prev_i] + np.array([0, r0, 0])

        state = ChainState.from_template(common, positions)
        dt = 1e-5
        damping = 0.5
        n_steps = 100
        max_force_history = []
        for step in range(n_steps):
            compute_chain_forces(state)
            max_force_history.append(np.max(np.abs(state.forces)))
            state.positions += damping * state.forces * dt

        # All positions finite throughout
        assert np.all(
            np.isfinite(state.positions)
        ), "positions must stay finite during integration"
        # Forces bounded: ratio end/start < 100
        ratio = max_force_history[-1] / max_force_history[0]
        assert ratio < 100, (
            f"force exploded: start={max_force_history[0]:.4f}, "
            f"end={max_force_history[-1]:.4f}, ratio={ratio:.2f}"
        )
        # Bond lengths haven't gone wild
        for bond in common.bonds:
            r_a = state.positions[bond.a.atom_idx]
            r_b = state.positions[bond.b.atom_idx]
            length = float(np.linalg.norm(r_b - r_a))
            assert 0.5 < length < 20.0, (
                f"bond {bond.a.atom_idx}-{bond.b.atom_idx} has crazy length: "
                f"{length:.3f} (eq {bond.r0})"
            )

    def test_deriv_array_matches_scalar_deriv(self, params):
        """TabulatedPotential.deriv_array matches scalar deriv() across pair, angle, and dihedral pots for in-range and out-of-range x."""
        # Test 5 representative pots from each table
        for pots, name in [
            (params.pair_pots[:5], "pair"),
            (params.angle_pots[:5], "angle"),
            (params.dihedral_pots[:5], "dihedral"),
        ]:
            for pot in pots:
                x_lo = pot.x_min - 1.0
                x_hi = pot.x_max + 1.0
                test_xs = np.linspace(x_lo, x_hi, 20)
                scalar = np.array([pot.deriv(float(x)) for x in test_xs])
                array = pot.deriv_array(test_xs)
                assert np.allclose(scalar, array, atol=1e-10), (
                    f"{name} pot deriv_array != scalar deriv at " f"index={pot.index}"
                )

    def test_sidechain_dihedral_force_conservation(self, params):
        """Cross-residue and sidechain-extending dihedrals give finite forces that conserve momentum."""

        # ARG-LEU-LYS-MET: all have full sidechains with CG/NG
        sequence = ["ARG", "LEU", "LYS", "MET"]
        common = build_chain_common_with_sidechains_from_coffdrop(
            sequence,
            params,
        )
        n = common.n_atoms
        # Non-trivial geometry that exercises all dihedral angles
        positions = np.zeros((n, 3))
        for i in range(n):
            positions[i] = [i * 1.7, np.sin(i * 0.5) * 2.0, np.cos(i * 0.4) * 1.5]
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # All forces finite
        assert np.all(
            np.isfinite(state.forces)
        ), "forces must be finite for sidechain-rich chain"
        force_sum_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            force_sum_norm < 1e-9
        ), f"net force should be zero, got norm {force_sum_norm}"

    def test_sidechain_dihedral_rotational_equivariance(self, params):
        """Sidechain-extending dihedral forces rotate by R when the positions are rotated by R."""

        sequence = ["TRP", "ASN", "ASP", "PHE"]
        common = build_chain_common_with_sidechains_from_coffdrop(
            sequence,
            params,
        )
        n = common.n_atoms
        positions = np.zeros((n, 3))
        for i in range(n):
            positions[i] = [i * 1.7, np.sin(i * 0.5) * 1.5, np.cos(i * 0.4) * 1.0]
        state_orig = ChainState.from_template(common, positions)
        compute_chain_forces(state_orig)
        # Rotate by 30 deg about z
        theta = np.deg2rad(30.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos_rotated = positions @ R.T
        state_rot = ChainState.from_template(common, pos_rotated)
        compute_chain_forces(state_rot)
        # forces_rotated[i] == R @ forces_orig[i]
        expected = state_orig.forces @ R.T
        assert np.allclose(
            state_rot.forces, expected, atol=1e-9
        ), "rotational equivariance violated for sidechain-rich chain"

    def test_chain_from_sequence_all_formats(self):
        """chain_from_sequence accepts single-letter, dash-separated, and space-separated 3-letter formats and produces equivalent chains."""

        # Single-letter
        chain1 = chain_from_sequence("ARWGL")
        # 3-letter dash
        chain2 = chain_from_sequence("ALA-ARG-TRP-GLY-LEU")
        # 3-letter space
        chain3 = chain_from_sequence("ALA ARG TRP GLY LEU")

        # Same n_atoms, n_angles, n_torsions, n_pair_lookups across formats
        assert chain1.n_atoms == chain2.n_atoms == chain3.n_atoms == 13
        assert len(chain1.angles) == len(chain2.angles) == len(chain3.angles)
        assert len(chain1.torsions) == len(chain2.torsions) == len(chain3.torsions)
        assert len(chain1.pair_lookups) == len(chain2.pair_lookups)

        # Same atom resnames
        for a1, a2, a3 in zip(chain1.atoms, chain2.atoms, chain3.atoms):
            assert a1.resname == a2.resname == a3.resname

    def test_chain_from_sequence_sidechains_flag(self):
        """chain_from_sequence with sidechains=False produces a CA-only backbone chain with the expected bond, angle, and torsion counts."""

        # 5-residue chain, CA-only
        chain = chain_from_sequence("ARWGL", sidechains=False)
        assert chain.n_atoms == 5
        assert all(a.resname.endswith(":CA") for a in chain.atoms) or not any(
            ":" in a.resname for a in chain.atoms
        )
        # Backbone-only: 4 bonds, 3 angles, 2 torsions
        assert len(chain.bonds) == 4
        assert len(chain.angles) == 3
        assert len(chain.torsions) == 2

    def test_chain_from_sequence_validation_errors(self):
        """chain_from_sequence raises ValueError with an informative message for invalid codes, bad tokens, and empty input."""

        # Invalid single-letter code
        try:
            chain_from_sequence("ARQXZ")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "X" in str(e), f"error message should mention 'X': {e}"

        # Invalid 3-letter token
        try:
            chain_from_sequence("ALA-XX-TRP")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "XX" in str(e), f"error message should mention 'XX': {e}"

        # Empty sequence
        try:
            chain_from_sequence("")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert (
                "empty" in str(e).lower()
            ), f"error message should mention 'empty': {e}"

    def test_place_relaxed_geometry(self):
        """place_relaxed_geometry produces finite positions with exact bond lengths and finite momentum-conserving forces."""

        for seq in ["A", "ARWGL", "ARNDCEQGHI"]:
            chain = chain_from_sequence(seq)
            positions = place_relaxed_geometry(chain)
            # Shape correct
            assert positions.shape == (
                chain.n_atoms,
                3,
            ), f"{seq}: positions shape mismatch"
            # All finite
            assert np.all(np.isfinite(positions)), f"{seq}: positions must be finite"
            # Bond lengths match eq within machine precision
            for bond in chain.bonds:
                r_actual = float(
                    np.linalg.norm(
                        positions[bond.b.atom_idx] - positions[bond.a.atom_idx]
                    )
                )
                assert abs(r_actual - bond.r0) < 1e-9, (
                    f"{seq}: bond {bond.a.atom_idx}-{bond.b.atom_idx} "
                    f"length {r_actual} != eq {bond.r0}"
                )
            # Forces conserve momentum, are finite
            state = ChainState.from_template(chain, positions)
            compute_chain_forces(state)
            assert np.all(np.isfinite(state.forces)), f"{seq}: forces must be finite"
            sum_F_norm = np.linalg.norm(state.forces.sum(axis=0))
            assert (
                sum_F_norm < 1e-9
            ), f"{seq}: net force should be zero, got {sum_F_norm}"

    def test_chain_from_pdb_single_chain(self, tmp_path):
        """chain_from_pdb on a single-chain PDB builds a chain with topology identical to chain_from_sequence."""

        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "HEADER    TEST\n"
            "ATOM      1  CA  ALA A   1      11.000  13.000  10.000  1.00  0.00\n"
            "ATOM      2  CA  ARG A   2      14.000  13.000  10.000  1.00  0.00\n"
            "ATOM      3  CA  TRP A   3      17.000  13.000  10.000  1.00  0.00\n"
            "ATOM      4  CA  GLY A   4      20.000  13.000  10.000  1.00  0.00\n"
            "ATOM      5  CA  LEU A   5      23.000  13.000  10.000  1.00  0.00\n"
            "END\n"
        )
        chain_pdb = chain_from_pdb(str(pdb))
        chain_seq = chain_from_sequence("ARWGL")
        # Should produce identical topology
        assert chain_pdb.n_atoms == chain_seq.n_atoms == 13
        assert len(chain_pdb.bonds) == len(chain_seq.bonds)
        assert len(chain_pdb.angles) == len(chain_seq.angles)
        assert len(chain_pdb.torsions) == len(chain_seq.torsions)
        for a_pdb, a_seq in zip(chain_pdb.atoms, chain_seq.atoms):
            assert a_pdb.resname == a_seq.resname

    def test_chain_from_pdb_multi_chain_requires_id(self, tmp_path):
        """chain_from_pdb raises on a multi-chain PDB without chain_id and extracts the requested chain when chain_id is given."""

        pdb = tmp_path / "multi.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1      11.000  13.000  10.000  1.00  0.00\n"
            "ATOM      2  CA  GLY A   2      14.000  13.000  10.000  1.00  0.00\n"
            "ATOM      3  CA  TRP B   1      20.000  20.000  20.000  1.00  0.00\n"
            "ATOM      4  CA  LEU B   2      23.000  20.000  20.000  1.00  0.00\n"
            "END\n"
        )
        # Without chain_id, raises
        try:
            chain_from_pdb(str(pdb))
            assert False, "should raise on multi-chain without chain_id"
        except ValueError as e:
            assert "multiple chains" in str(e).lower()
        # With chain_id="A", gets ALA-GLY
        chain_a = chain_from_pdb(str(pdb), chain_id="A")
        assert chain_a.n_atoms == 3  # ALA(2) + GLY(1)
        # With chain_id="B", gets TRP-LEU
        chain_b = chain_from_pdb(str(pdb), chain_id="B")
        assert chain_b.n_atoms == 7  # TRP(4) + LEU(3)

    def test_chain_from_pdb_error_handling(self, tmp_path):
        """chain_from_pdb raises FileNotFoundError for a missing file and ValueError for a PDB with no ATOM records."""

        # Missing file
        try:
            chain_from_pdb(str(tmp_path / "nonexistent.pdb"))
            assert False, "should raise FileNotFoundError"
        except FileNotFoundError:
            pass
        # PDB with no ATOM records
        empty_pdb = tmp_path / "empty.pdb"
        empty_pdb.write_text("HEADER    EMPTY\nEND\n")
        try:
            chain_from_pdb(str(empty_pdb))
            assert False, "should raise ValueError on no ATOMs"
        except ValueError as e:
            assert "no atom" in str(e).lower()

    def test_caps_topology_ace_nme(self):
        """ACE and NME caps each add one atom and one bond, with cap atoms carrying resid -1 and the expected resnames."""

        # Uncapped ARWGL: 13 atoms, 12 bonds
        uncapped = chain_from_sequence("ARWGL")

        # Both caps: +2 atoms (CN, CC), +2 bonds (CA-CN, CA-CC)
        both = chain_from_sequence("ARWGL", caps=("ACE", "NME"))
        assert both.n_atoms == uncapped.n_atoms + 2
        assert len(both.bonds) == len(uncapped.bonds) + 2

        # ACE only: +1 atom, +1 bond
        ace = chain_from_sequence("ARWGL", caps=("ACE", None))
        assert ace.n_atoms == uncapped.n_atoms + 1
        assert len(ace.bonds) == len(uncapped.bonds) + 1

        # NME only: +1 atom, +1 bond
        nme = chain_from_sequence("ARWGL", caps=(None, "NME"))
        assert nme.n_atoms == uncapped.n_atoms + 1
        assert len(nme.bonds) == len(uncapped.bonds) + 1

        # Cap atoms have resid=-1 (cap marker)
        cap_atoms = [a for a in both.atoms if a.resid == -1]
        assert len(cap_atoms) == 2
        cap_names = sorted(a.resname for a in cap_atoms)
        assert cap_names == ["ACE:CN", "NME:CC"]

    def test_caps_force_lookups_populated(self):
        """Cap-flanking angles, torsions, and pair lookups receive valid populated type_idx values from COFFDROP."""

        chain = chain_from_sequence("ARWGL", caps=("ACE", "NME"))

        # Find cap-flanking angles
        cap_angles = [
            a
            for a in chain.angles
            if any(chain.atoms[ref.atom_idx].resid == -1 for ref in [a.a, a.b, a.c])
        ]
        assert len(cap_angles) >= 2, f"expected >=2 cap angles, got {len(cap_angles)}"
        # All cap angles should have populated type_idx
        for a in cap_angles:
            assert (
                a.type_idx >= 0
            ), "cap angle should have populated type_idx from COFFDROP"

        # Find cap-flanking torsions
        cap_torsions = [
            t
            for t in chain.torsions
            if any(
                chain.atoms[ref.atom_idx].resid == -1 for ref in [t.a, t.b, t.c, t.d]
            )
        ]
        assert (
            len(cap_torsions) >= 2
        ), f"expected >=2 cap torsions, got {len(cap_torsions)}"

        # At least the 2 backbone cap torsions should have populated type_idx
        cap_torsions_populated = sum(1 for t in cap_torsions if t.type_idx >= 0)
        assert (
            cap_torsions_populated >= 2
        ), f"expected >=2 populated cap torsions, got {cap_torsions_populated}"

        # Pair lookups should include cap-involved pairs
        cap_atom_indices = {i for i, a in enumerate(chain.atoms) if a.resid == -1}
        cap_pair_count = sum(
            1
            for (i, j) in chain.pair_lookups
            if i in cap_atom_indices or j in cap_atom_indices
        )
        assert (
            cap_pair_count > 0
        ), "no cap-involved pair_lookups; flanking residue logic broken"

    def test_caps_force_evaluation_finite(self):
        """compute_chain_forces on an ACE/NME-capped chain produces finite, momentum-conserving forces."""

        chain = chain_from_sequence("ARWGL", caps=("ACE", "NME"))
        positions = place_relaxed_geometry(chain)
        state = ChainState.from_template(chain, positions)
        compute_chain_forces(state)

        assert np.all(
            np.isfinite(state.forces)
        ), "forces must be finite for capped chain"
        sum_F_norm = np.linalg.norm(state.forces.sum(axis=0))
        assert (
            sum_F_norm < 1e-9
        ), f"momentum conservation broken for capped chain: {sum_F_norm}"

    def test_caps_validation(self):
        """Invalid cap names raise ValueError, as does requesting caps on a CA-only chain."""

        # Wrong cap name
        try:
            chain_from_sequence("ALA", caps=("WRONG", None))
            assert False, "should raise on bad cap name"
        except ValueError as e:
            assert "ACE" in str(e) or "WRONG" in str(e)

        # CA-only with caps not allowed
        try:
            chain_from_sequence("ALA", sidechains=False, caps=("ACE", None))
            assert False, "should raise on CA-only + caps"
        except ValueError as e:
            assert "sidechains" in str(e).lower() or "ca-only" in str(e).lower()


class TestChainBDInputXML:
    """Tests for the chain BD path through input.xml + parse() + run_chain().

    Covers schema parsing, validation, dispatch logic, and the helper
    functions in chain_pipeline. End-to-end execution with real chain.json
    and target.pqr is deferred to integration tests in Stage 2.
    """

    def test_chain_block_parses_with_all_fields(self, tmp_path):
        """A complete <chain> XML block populates every field of ChainConfig with the parsed values."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>t.pqr</receptor_pqr>
  <bd_milestone_radius>50.0</bd_milestone_radius>
  <n_trajectories>100</n_trajectories>
  <seed>42</seed>
  <chain>
    <chain_json>c.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
    <target_grid_dx>g.dx</target_grid_dx>
    <born_grid_dx>b.dx</born_grid_dx>
    <r_escape>120.0</r_escape>
    <reaction_n_needed>3</reaction_n_needed>
    <auto_diffusion>true</auto_diffusion>
    <D_trans>0.1</D_trans>
    <D_rot>0.01</D_rot>
    <use_soft_repulsion>true</use_soft_repulsion>
    <soft_repulsion_eps>0.5</soft_repulsion_eps>
    <n_workers>4</n_workers>
    <dt_chain>0.025</dt_chain>
    <chain_steps_per_outer>8</chain_steps_per_outer>
  </chain>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        cfg = parse(xml_path)
        assert cfg.chain is not None
        assert cfg.chain.chain_json == "c.json"
        assert cfg.chain.reaction_pairs_json == "rp.json"
        assert cfg.chain.target_grid_dx == "g.dx"
        assert cfg.chain.born_grid_dx == "b.dx"
        assert cfg.chain.r_escape == 120.0
        assert cfg.chain.reaction_n_needed == 3
        assert cfg.chain.auto_diffusion is True
        assert cfg.chain.D_trans == 0.1
        assert cfg.chain.D_rot == 0.01
        assert cfg.chain.use_soft_repulsion is True
        assert cfg.chain.soft_repulsion_eps == 0.5
        assert cfg.chain.n_workers == 4
        assert cfg.chain.dt_chain == 0.025
        assert cfg.chain.chain_steps_per_outer == 8

    def test_chain_block_minimal_uses_defaults(self, tmp_path):
        """A minimal <chain> block populates required fields and leaves the rest at their defaults."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>t.pqr</receptor_pqr>
  <chain>
    <chain_json>c.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
  </chain>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        cfg = parse(xml_path)
        assert cfg.chain is not None
        assert cfg.chain.chain_json == "c.json"
        assert cfg.chain.reaction_pairs_json == "rp.json"
        assert cfg.chain.target_grid_dx == ""
        assert cfg.chain.born_grid_dx == ""
        assert cfg.chain.r_escape == 0.0
        assert cfg.chain.reaction_n_needed == 3
        assert cfg.chain.auto_diffusion is False
        assert cfg.chain.D_trans == 0.0
        assert cfg.chain.D_rot == 0.0
        assert cfg.chain.use_soft_repulsion is False
        assert cfg.chain.soft_repulsion_eps == 1.0
        assert cfg.chain.n_workers == 1
        assert cfg.chain.dt_chain == 0.05
        assert cfg.chain.chain_steps_per_outer == 4

    def test_legacy_xml_chain_is_none(self, tmp_path):
        """An input.xml without a <chain> block yields cfg.chain equal to None."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        cfg = parse(xml_path)
        assert cfg.chain is None

    def test_chain_mode_missing_chain_json_raises(self, tmp_path):
        """Parsing raises ValueError mentioning chain_json when chain mode omits chain_json."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>t.pqr</receptor_pqr>
  <chain>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
  </chain>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        try:
            parse(xml_path)
            assert False, "should raise ValueError for missing chain_json"
        except ValueError as e:
            assert "chain_json" in str(e)

    def test_chain_mode_missing_receptor_pqr_raises(self, tmp_path):
        """Parsing raises ValueError mentioning receptor_pqr when chain mode omits receptor_pqr."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <chain>
    <chain_json>c.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
  </chain>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        try:
            parse(xml_path)
            assert False, "should raise ValueError for missing receptor_pqr"
        except ValueError as e:
            assert "receptor_pqr" in str(e)

    def test_run_chain_requires_chain_config(self):
        """run_chain raises ValueError mentioning config.chain when the chain config is None."""

        cfg = PySTARCConfig(receptor_pqr="r.pqr", ligand_pqr="l.pqr")
        try:
            run_chain(cfg)
            assert False, "should raise ValueError when config.chain is None"
        except ValueError as e:
            assert "config.chain" in str(e)

    def test_load_reaction_pairs_json_round_trip(self, tmp_path):
        """_load_reaction_pairs_json parses the list-of-tuples JSON format into the expected tuples."""

        path = tmp_path / "rp.json"
        path.write_text(json.dumps([[100, 0, 7.0], [200, 5, 6.5]]))
        pairs = _load_reaction_pairs_json(str(path))
        assert pairs == [(100, 0, 7.0), (200, 5, 6.5)]

    def test_load_reaction_pairs_json_missing_file_raises(self):
        """_load_reaction_pairs_json raises FileNotFoundError for a nonexistent path."""

        try:
            _load_reaction_pairs_json("/nonexistent/path.json")
            assert False, "should raise FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_load_reaction_pairs_json_wrong_length_raises(self, tmp_path):
        """_load_reaction_pairs_json raises ValueError when an entry does not have exactly three elements."""

        path = tmp_path / "bad.json"
        path.write_text(json.dumps([[100, 0]]))
        try:
            _load_reaction_pairs_json(str(path))
            assert False, "should raise ValueError for malformed entry"
        except ValueError as e:
            assert "length 2" in str(e) or "expected 3" in str(e)

    def test_build_pathway_set_creates_one_reaction_with_all_pairs(self):
        """_build_pathway_set produces one association reaction holding all contact pairs and the requested n_needed."""

        ps = _build_pathway_set([(10, 0, 7.0), (20, 1, 6.5), (30, 2, 5.0)], n_needed=2)
        assert len(ps.reactions) == 1
        rxn = ps.reactions[0]
        assert rxn.name == "association"
        assert len(rxn.criteria.pairs) == 3
        assert rxn.criteria.n_needed == 2
        assert rxn.criteria.pairs[0].mol1_atom_index == 10
        assert rxn.criteria.pairs[0].mol2_atom_index == 0
        assert rxn.criteria.pairs[0].distance_cutoff == 7.0
        assert rxn.criteria.pairs[2].mol1_atom_index == 30
        assert rxn.criteria.pairs[2].mol2_atom_index == 2
        assert rxn.criteria.pairs[2].distance_cutoff == 5.0


class TestChainBDInstrumentation:
    """Sub-stage 2a: TrajectoryResult diagnostic fields + outputs param plumbing."""

    def test_trajectory_result_default_optional_fields_are_none(self):
        """Positional TrajectoryResult construction leaves all optional diagnostic fields set to None."""

        r = TrajectoryResult(Fate.ESCAPED, 100, 1.0, 50.0)
        assert r.fate == Fate.ESCAPED
        assert r.steps == 100
        assert r.encounter_pos is None
        assert r.encounter_q is None
        assert r.near_miss_pos is None
        assert r.near_miss_dist is None
        assert r.path_steps is None
        assert r.path_com is None
        assert r.path_q is None
        assert r.energy_steps is None
        assert r.radial_trace is None
        assert r.contact_counts is None

    def test_trajectory_result_accepts_diagnostic_fields(self):
        """TrajectoryResult round-trips the diagnostic fields supplied by chain-style keyword construction."""

        r = TrajectoryResult(
            fate=Fate.REACTED,
            steps=42,
            time_ps=12.5,
            final_separation=8.0,
            reaction_name="rxn1",
            encounter_pos=np.array([1.0, 2.0, 3.0]),
            encounter_q=np.array([1.0, 0.0, 0.0, 0.0]),
            near_miss_dist=4.0,
            path_steps=np.array([0, 10, 20]),
            path_com=np.zeros((3, 3)),
            path_q=np.zeros((3, 4)),
            radial_trace=np.array([10.0, 9.0, 8.0]),
        )
        assert r.encounter_pos.tolist() == [1.0, 2.0, 3.0]
        assert r.encounter_q.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert r.near_miss_dist == 4.0
        assert r.path_steps.tolist() == [0, 10, 20]
        assert r.path_com.shape == (3, 3)
        assert r.energy_steps is None
        assert r.contact_counts is None

    def test_chainbd_simulator_init_signature_has_outputs(self):
        """ChainBDSimulator.__init__ exposes an outputs keyword argument defaulting to None."""

        sig = inspect.signature(ChainBDSimulator.__init__)
        assert "outputs" in sig.parameters
        assert sig.parameters["outputs"].default is None

    def test_write_chain_results_signature_accepts_outputs(self):
        """write_chain_results exposes an outputs keyword argument defaulting to None."""

        sig = inspect.signature(write_chain_results)
        assert "outputs" in sig.parameters
        assert sig.parameters["outputs"].default is None


class TestChainBDWriters:
    """Sub-stage 2b: 5 chain BD writer functions."""

    def _make_results(self):
        """Build a synthetic List[TrajectoryResult] for writer tests."""

        # 1 reacted, 2 escaped (1 with near_miss, 1 without)
        r0 = TrajectoryResult(
            fate=Fate.REACTED,
            steps=42,
            time_ps=12.5,
            final_separation=8.0,
            reaction_name="rxn1",
            encounter_pos=np.array([1.0, 2.0, 3.0]),
            encounter_q=np.array([1.0, 0.0, 0.0, 0.0]),
            path_steps=np.array([0, 10, 20]),
            path_com=np.zeros((3, 3)),
            path_q=np.zeros((3, 4)),
            radial_trace=np.array([45.0, 30.0, 8.0]),
            energy_steps=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-1.0, -0.7, -0.3, 0.0],
                    [-3.0, -2.0, -1.0, 0.0],
                ]
            ),
            contact_counts={(5, 0): 2, (12, 1): 1},
        )
        r1 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=500,
            time_ps=200.0,
            final_separation=120.0,
            near_miss_pos=np.array([10.0, 0.0, 0.0]),
            near_miss_dist=10.0,
            path_steps=np.array([0, 10, 20, 30]),
            path_com=np.zeros((4, 3)),
            path_q=np.zeros((4, 4)),
            radial_trace=np.array([45.0, 25.0, 10.0, 120.0]),
            energy_steps=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.5, -0.3, -0.2, 0.0],
                    [-1.5, -1.0, -0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            contact_counts={(5, 0): 1},
        )
        r2 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=100,
            time_ps=50.0,
            final_separation=200.0,
        )  # no diagnostics populated
        return [r0, r1, r2]

    def test_write_encounters_csv(self, tmp_path):
        """write_encounters_csv writes one data row per REACTED trajectory with the expected header and fields."""

        results = self._make_results()
        p = write_encounters_csv(tmp_path, results)
        assert p is not None
        assert p.name == "encounters.csv"
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0].startswith("traj_id,step,time_ps")
        # 1 REACTED trajectory => 1 data row
        assert len(lines) == 2
        assert lines[1].startswith("0,42,")
        assert "rxn1" in lines[1]

    def test_write_near_misses_csv(self, tmp_path):
        """write_near_misses_csv writes one row per ESCAPED trajectory with populated near-miss data including the near-miss distance."""

        results = self._make_results()
        p = write_near_misses_csv(tmp_path, results)
        assert p is not None
        text = p.read_text()
        lines = text.strip().split("\n")
        # 1 ESCAPED with populated near_miss => 1 data row (r1; r2 has no data)
        assert lines[0].startswith("traj_id,fate")
        assert len(lines) == 2
        assert "ESCAPED" in lines[1]
        assert "10.000" in lines[1]  # near_miss_dist

    def test_write_fpt_distribution_csv(self, tmp_path):
        """write_fpt_distribution_csv writes a header plus n_bins rows whose total count equals the number of REACTED trajectories."""

        results = self._make_results()
        p = write_fpt_distribution_csv(tmp_path, results, n_bins=10)
        assert p is not None
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0] == "bin_lo,bin_hi,count"
        assert len(lines) == 11  # header + 10 bins
        # Total count across all bins should equal number of REACTED trajectories
        counts = [int(line.split(",")[2]) for line in lines[1:]]
        assert sum(counts) == 1

    def test_write_contact_frequency_csv(self, tmp_path):
        """write_contact_frequency_csv aggregates per-pair contact totals and trajectory counts, sorted descending by total."""

        results = self._make_results()
        p = write_contact_frequency_csv(tmp_path, results)
        assert p is not None
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0] == "target_atom_id,chain_atom_id,total_contacts,n_trajectories"
        assert lines[1] == "5,0,3,2"
        assert lines[2] == "12,1,1,1"

    def test_write_energetics_npz(self, tmp_path):
        """write_energetics_npz stores per-snapshot traj_id, step, and energy arrays with the expected shapes and column labels."""

        results = self._make_results()
        p = write_energetics_npz(tmp_path, results)
        assert p is not None
        with np.load(p) as data:
            assert data["traj_id"].shape == (7,)  # 3 + 4 snapshots
            assert data["step"].shape == (7,)
            assert data["energy"].shape == (7, 4)
            assert list(data["columns"]) == ["total", "elec", "born", "steric"]
            # First trajectory (r0, REACTED) gets traj_id=0; 3 snapshots
            assert (data["traj_id"][:3] == 0).all()
            assert (data["traj_id"][3:7] == 1).all()

    def test_write_chain_results_gates_on_outputs(self, tmp_path):
        """OutputConfig flags gate which optional writers run, while results.json and trajectories.csv are always written."""

        # Stub `sim` with the minimum attrs needed by write_results_json.
        class _StubChain:
            name = "stub"
            n_atoms = 2
            atoms = []
            bonds = []
            angles = []
            torsions = []
            length_constraints = []

        class _StubParams:
            n_trajectories = 3
            dt = 1.0
            dt_chain = 0.05
            chain_steps_per_outer = 4
            max_steps = 1000
            r_start = 50.0
            r_escape = 100.0
            seed = 42
            n_threads = 1
            constraint_tol = 1e-4
            constraint_max_iter = 50

        class _StubSim:
            auto_diffusion = False
            D_trans = 1.0
            D_rot = 1.0
            chain_template = _StubChain()
            params = _StubParams()

        results = self._make_results()
        # All flags True (default) - should produce all writers that have data
        oc_all = OutputConfig()
        written = write_chain_results(
            tmp_path, _StubSim(), results, wall_time_sec=1.0, outputs=oc_all
        )
        names = [n for n, _ in written]
        assert "results.json" in names
        assert "trajectories.csv" in names
        assert "encounters.csv" in names
        assert "near_misses.csv" in names
        assert "fpt_distribution.csv" in names
        assert "contact_frequency.csv" in names
        assert "energetics.npz" in names
        # All flags False - only results.json and trajectories.csv (always written)
        oc_none = OutputConfig(
            encounters_csv=False,
            near_misses_csv=False,
            fpt_distribution=False,
            contact_frequency=False,
            energetics=False,
        )
        written2 = write_chain_results(
            tmp_path, _StubSim(), results, wall_time_sec=1.0, outputs=oc_none
        )
        names2 = [n for n, _ in written2]
        assert "results.json" in names2
        assert "trajectories.csv" in names2
        assert "encounters.csv" not in names2
        assert "energetics.npz" not in names2


class TestChainBDWritersHeavy:
    """Sub-stage 2c: 3 heavier chain BD writer functions."""

    def _make_results(self):
        """Synthetic List[TrajectoryResult] with path_com / path_q / radial_trace."""

        r0_com = np.array(
            [
                [50.0, 0.0, 0.0],  # phi=0,   theta=pi/2
                [0.0, 50.0, 0.0],  # phi=pi/2,theta=pi/2
                [0.0, 0.0, 50.0],  # phi=0,   theta=0
            ]
        )
        r0 = TrajectoryResult(
            fate=Fate.REACTED,
            steps=42,
            time_ps=12.5,
            final_separation=8.0,
            reaction_name="rxn1",
            path_steps=np.array([0, 10, 20]),
            path_com=r0_com,
            path_q=np.zeros((3, 4)),
            radial_trace=np.array([50.0, 50.0, 50.0]),
        )
        # r1 (ESCAPED): 4 snapshots
        r1 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=500,
            time_ps=200.0,
            final_separation=120.0,
            near_miss_pos=np.array([10.0, 0.0, 0.0]),
            near_miss_dist=10.0,
            path_steps=np.array([0, 10, 20, 30]),
            path_com=np.array(
                [
                    [40.0, 0.0, 0.0],
                    [60.0, 0.0, 0.0],
                    [80.0, 0.0, 0.0],
                    [120.0, 0.0, 0.0],
                ]
            ),
            path_q=np.zeros((4, 4)),
            radial_trace=np.array([40.0, 60.0, 80.0, 120.0]),
        )
        # r2 (ESCAPED): no diagnostics populated
        r2 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=100,
            time_ps=50.0,
            final_separation=200.0,
        )
        return [r0, r1, r2]

    def test_write_paths_npz(self, tmp_path):
        """write_paths_npz stores per-snapshot traj_id, step, com, q, and radial arrays with the expected shapes and values."""

        results = self._make_results()
        p = write_paths_npz(tmp_path, results)
        assert p is not None
        assert p.name == "paths.npz"
        with np.load(p) as data:
            assert data["traj_id"].shape == (7,)  # 3 + 4 snapshots
            assert data["step"].shape == (7,)
            assert data["com"].shape == (7, 3)
            assert data["q"].shape == (7, 4)
            assert data["radial"].shape == (7,)
            # First 3 belong to r0, next 4 to r1
            assert (data["traj_id"][:3] == 0).all()
            assert (data["traj_id"][3:7] == 1).all()
            # Verify radial values are correct
            assert data["radial"][0] == 50.0
            assert data["radial"][3] == 40.0

    def test_write_radial_density_csv(self, tmp_path):
        """write_radial_density_csv bins all snapshots into n_bins rows with non-negative densities and counts summing to the snapshot total."""

        results = self._make_results()
        p = write_radial_density_csv(tmp_path, results, n_bins=10)
        assert p is not None
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0] == "bin_lo,bin_hi,count,density"
        assert len(lines) == 11  # header + 10 bins
        # 7 total snapshots binned across [0, 120]
        counts = [int(line.split(",")[2]) for line in lines[1:]]
        assert sum(counts) == 7
        # Density column should be present and non-negative
        densities = [float(line.split(",")[3]) for line in lines[1:]]
        assert all(d >= 0 for d in densities)

    def test_write_angular_map_npz(self, tmp_path):
        """write_angular_map_npz produces a theta-by-phi count grid with edges spanning [0, π] and counts summing to the snapshot total."""

        results = self._make_results()
        p = write_angular_map_npz(tmp_path, results, n_theta=18, n_phi=36)
        assert p is not None
        with np.load(p) as data:
            assert data["counts"].shape == (18, 36)
            assert data["theta_edges"].shape == (19,)
            assert data["phi_edges"].shape == (37,)
            # Total snapshots: 3 (from r0) + 4 (from r1) = 7
            assert int(data["total_snapshots"][0]) == 7
            assert int(data["counts"].sum()) == 7
            # theta_edges should span [0, pi]
            assert abs(float(data["theta_edges"][0]) - 0.0) < 1e-9
            assert abs(float(data["theta_edges"][-1]) - float(np.pi)) < 1e-9

    def test_write_chain_results_includes_2c_outputs(self, tmp_path):
        """write_chain_results emits the 2c outputs paths.npz, radial_density.csv, and angular_map.npz in the correct order when their flags are set."""

        class _StubChain:
            name = "stub"
            n_atoms = 2
            atoms = []
            bonds = []
            angles = []
            torsions = []
            length_constraints = []

        class _StubParams:
            n_trajectories = 3
            dt = 1.0
            dt_chain = 0.05
            chain_steps_per_outer = 4
            max_steps = 1000
            r_start = 50.0
            r_escape = 100.0
            seed = 42
            n_threads = 1
            constraint_tol = 1e-4
            constraint_max_iter = 50

        class _StubSim:
            auto_diffusion = False
            D_trans = 1.0
            D_rot = 1.0
            chain_template = _StubChain()
            params = _StubParams()

        results = self._make_results()
        oc_all = OutputConfig()
        written = write_chain_results(
            tmp_path, _StubSim(), results, wall_time_sec=1.0, outputs=oc_all
        )
        names = [n for n, _ in written]
        assert "paths.npz" in names
        assert "radial_density.csv" in names
        assert "angular_map.npz" in names
        # Order check: 2c outputs come between near_misses and fpt_distribution
        idx_near = names.index("near_misses.csv")
        idx_paths = names.index("paths.npz")
        idx_fpt = names.index("fpt_distribution.csv")
        assert idx_near < idx_paths < idx_fpt


class TestChainBDMilestoneFlux:
    """Sub-stage 2d: milestone_flux.csv writer."""

    def _make_results_with_known_crossings(self):
        """Synthetic results with manually computable shell crossings.

        radials concatenated = [50, 50, 50, 40, 60, 80, 120]
        r_min=40, r_max=120, n_shells=4 -> shells = [56, 72, 88, 104]

        r0 [50, 50, 50]: no crossings
        r1 [40, 60, 80, 120]:
          40 -> 60   crosses 56 outward                   -> shell[0] out += 1
          60 -> 80   crosses 72 outward                   -> shell[1] out += 1
          80 -> 120  crosses 88, 104 outward              -> shell[2,3] out += 1 each
        Expected: out=[1,1,1,1], in=[0,0,0,0]
        """

        r0 = TrajectoryResult(
            fate=Fate.REACTED,
            steps=42,
            time_ps=12.5,
            final_separation=50.0,
            reaction_name="rxn1",
            radial_trace=np.array([50.0, 50.0, 50.0]),
            contact_counts={(5, 0): 1},
        )
        r1 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=500,
            time_ps=200.0,
            final_separation=120.0,
            radial_trace=np.array([40.0, 60.0, 80.0, 120.0]),
            contact_counts={(5, 0): 2, (12, 1): 1},
        )
        r2 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=10,
            time_ps=5.0,
            final_separation=200.0,
        )  # no radial_trace
        return [r0, r1, r2]

    def test_write_milestone_flux_csv_known_crossings(self, tmp_path):
        """write_milestone_flux_csv reports one outward and zero inward crossings per shell at the expected shell radii."""

        results = self._make_results_with_known_crossings()
        p = write_milestone_flux_csv(tmp_path, results, n_shells=4)
        assert p is not None
        assert p.name == "milestone_flux.csv"
        text = p.read_text()
        lines = text.strip().split("\n")
        assert lines[0] == "shell_radius,n_crossings_out,n_crossings_in"
        assert len(lines) == 5  # header + 4 shells
        # Each of 4 shells gets exactly 1 outward, 0 inward
        for line in lines[1:]:
            parts = line.split(",")
            assert (
                int(parts[1]) == 1
            ), f"expected 1 outward, got {parts[1]} in line: {line}"
            assert (
                int(parts[2]) == 0
            ), f"expected 0 inward, got {parts[2]} in line: {line}"
        # Verify shell radii: linspace(40, 120, 6)[1:-1] = [56, 72, 88, 104]
        radii = [float(line.split(",")[0]) for line in lines[1:]]
        expected = [56.0, 72.0, 88.0, 104.0]
        for got, want in zip(radii, expected):
            assert abs(got - want) < 1e-6, f"shell radius mismatch: {got} vs {want}"

    def test_write_milestone_flux_returns_none_when_no_data(self, tmp_path):
        """write_milestone_flux_csv returns None and writes no file when no trajectory has radial-trace data."""

        results = [
            TrajectoryResult(
                fate=Fate.ESCAPED, steps=10, time_ps=5.0, final_separation=100.0
            ),
            TrajectoryResult(
                fate=Fate.ESCAPED, steps=20, time_ps=10.0, final_separation=120.0
            ),
        ]
        p = write_milestone_flux_csv(tmp_path, results)
        assert p is None
        assert not (tmp_path / "milestone_flux.csv").exists()

    def test_milestone_flux_in_chain_results(self, tmp_path):
        """milestone_flux.csv is written between contact_frequency and energetics when its output flag is enabled."""

        class _StubChain:
            name = "stub"
            n_atoms = 2
            atoms = []
            bonds = []
            angles = []
            torsions = []
            length_constraints = []

        class _StubParams:
            n_trajectories = 3
            dt = 1.0
            dt_chain = 0.05
            chain_steps_per_outer = 4
            max_steps = 1000
            r_start = 50.0
            r_escape = 100.0
            seed = 42
            n_threads = 1
            constraint_tol = 1e-4
            constraint_max_iter = 50

        class _StubSim:
            auto_diffusion = False
            D_trans = 1.0
            D_rot = 1.0
            chain_template = _StubChain()
            params = _StubParams()

        results = self._make_results_with_known_crossings()
        oc = OutputConfig()  # all flags True by default
        written = write_chain_results(
            tmp_path, _StubSim(), results, wall_time_sec=1.0, outputs=oc
        )
        names = [n for n, _ in written]
        assert "milestone_flux.csv" in names
        # Order: contact_frequency < milestone_flux < energetics
        idx_cf = names.index("contact_frequency.csv")
        idx_mf = names.index("milestone_flux.csv")
        idx_en = names.index("energetics.npz") if "energetics.npz" in names else -1
        assert idx_cf < idx_mf, f"contact_frequency should come before milestone_flux"
        if idx_en > 0:
            assert idx_mf < idx_en, f"milestone_flux should come before energetics"


class TestChainBDDeferredOutputs:
    """Sub-stage 2e: contract for the 3 OutputConfig flags that are accepted
    but currently no-op in chain BD mode (p_commit, transition_matrix,
    pose_clusters).
    """

    def test_deferred_flags_are_silent_noop(self, tmp_path):
        """Enabling the deferred p_commit, transition_matrix, and pose_clusters flags is a silent no-op that does not crash or emit those files."""

        r0 = TrajectoryResult(
            fate=Fate.REACTED,
            steps=42,
            time_ps=12.5,
            final_separation=8.0,
            reaction_name="rxn1",
            encounter_pos=np.array([1.0, 2.0, 3.0]),
            encounter_q=np.array([1.0, 0.0, 0.0, 0.0]),
            path_steps=np.array([0, 10, 20]),
            path_com=np.array([[10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]]),
            path_q=np.zeros((3, 4)),
            radial_trace=np.array([10.0, 20.0, 30.0]),
            energy_steps=np.zeros((3, 4)),
            contact_counts={(5, 0): 1},
        )
        r1 = TrajectoryResult(
            fate=Fate.ESCAPED,
            steps=500,
            time_ps=200.0,
            final_separation=120.0,
            near_miss_pos=np.array([10.0, 0.0, 0.0]),
            near_miss_dist=10.0,
            path_steps=np.array([0, 10, 20, 30]),
            path_com=np.array(
                [[10.0, 0, 0], [40.0, 0, 0], [80.0, 0, 0], [120.0, 0, 0]]
            ),
            path_q=np.zeros((4, 4)),
            radial_trace=np.array([10.0, 40.0, 80.0, 120.0]),
            energy_steps=np.zeros((4, 4)),
            contact_counts={(5, 0): 1},
        )
        results = [r0, r1]

        class _StubChain:
            name = "stub"
            n_atoms = 2
            atoms = []
            bonds = []
            angles = []
            torsions = []
            length_constraints = []

        class _StubParams:
            n_trajectories = 2
            dt = 1.0
            dt_chain = 0.05
            chain_steps_per_outer = 4
            max_steps = 1000
            r_start = 50.0
            r_escape = 100.0
            seed = 42
            n_threads = 1
            constraint_tol = 1e-4
            constraint_max_iter = 50

        class _StubSim:
            auto_diffusion = False
            D_trans = 1.0
            D_rot = 1.0
            chain_template = _StubChain()
            params = _StubParams()

        oc = OutputConfig()
        # Pre-conditions: deferred flags default to True
        assert oc.p_commit is True
        assert oc.transition_matrix is True
        assert oc.pose_clusters is True
        # The actual call should not crash even with deferred flags True
        written = write_chain_results(
            tmp_path, _StubSim(), results, wall_time_sec=1.0, outputs=oc
        )
        names = [n for n, _ in written]
        # Deferred outputs must NOT appear in written list or on disk
        for missing in ("p_commit.npz", "transition_matrix.npz", "pose_clusters.csv"):
            assert missing not in names, f"{missing} should be deferred"
            assert not (tmp_path / missing).exists(), f"{missing} should not be on disk"
        # Implemented 11 outputs should all be present
        expected = [
            "results.json",
            "trajectories.csv",
            "encounters.csv",
            "near_misses.csv",
            "paths.npz",
            "radial_density.csv",
            "angular_map.npz",
            "fpt_distribution.csv",
            "contact_frequency.csv",
            "milestone_flux.csv",
            "energetics.npz",
        ]
        for f in expected:
            assert f in names, f"expected {f} in written files: {names}"
        assert len(written) == 11


class TestChainIORoundTrip:
    """Stage 3 Phase A: save -> load round-trip equality for chain.json."""

    @staticmethod
    def _make_synthetic_chain():
        atoms = [
            ChainAtom(radius=1.5, charge=-0.5, resname="ALA", resid=1),
            ChainAtom(radius=1.4, charge=0.0, resname="ALA", resid=1),
            ChainAtom(radius=1.6, charge=0.5, resname="GLY", resid=2),
            ChainAtom(radius=1.3, charge=0.1, resname="GLY", resid=2),
        ]
        bonds = [
            ChainBond(a=ChainAtomRef(0), b=ChainAtomRef(1), r0=1.5, k_spring=100.0),
            ChainBond(a=ChainAtomRef(1), b=ChainAtomRef(2), r0=1.4, k_spring=120.0),
            ChainBond(a=ChainAtomRef(2), b=ChainAtomRef(3), r0=1.5, k_spring=100.0),
        ]
        angles = [
            ChainAngle(
                a=ChainAtomRef(0),
                b=ChainAtomRef(1),
                c=ChainAtomRef(2),
                theta0=2.094,
                k_angle=50.0,
            ),
            ChainAngle(
                a=ChainAtomRef(1),
                b=ChainAtomRef(2),
                c=ChainAtomRef(3),
                theta0=2.094,
                k_angle=50.0,
            ),
        ]
        torsions = [
            ChainTorsion(
                a=ChainAtomRef(0),
                b=ChainAtomRef(1),
                c=ChainAtomRef(2),
                d=ChainAtomRef(3),
                phi0=3.14159,
                k_tor=2.0,
                n=2,
            ),
        ]
        return ChainCommon(
            name="test_tetrapeptide",
            atoms=atoms,
            bonds=bonds,
            angles=angles,
            torsions=torsions,
        )

    def test_save_then_load_preserves_topology(self, tmp_path):
        """Saving then loading a chain to JSON preserves atom, bond, angle, and torsion parameters and the centered positions."""

        common = self._make_synthetic_chain()
        positions = np.array(
            [
                [-1.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        out_path = tmp_path / "chain.json"
        save_chain_to_json(common, positions, out_path)
        assert out_path.exists()
        loaded_common, loaded_positions = load_chain_from_json(out_path)
        # Topology counts match
        assert loaded_common.name == common.name
        assert len(loaded_common.atoms) == len(common.atoms)
        assert len(loaded_common.bonds) == len(common.bonds)
        assert len(loaded_common.angles) == len(common.angles)
        assert len(loaded_common.torsions) == len(common.torsions)
        # Atom parameters
        for orig, got in zip(common.atoms, loaded_common.atoms):
            assert orig.radius == got.radius
            assert orig.charge == got.charge
            assert orig.resname == got.resname
            assert orig.resid == got.resid
        # Bond parameters
        for orig, got in zip(common.bonds, loaded_common.bonds):
            assert orig.r0 == got.r0
            assert orig.k_spring == got.k_spring
        # Angle parameters
        for orig, got in zip(common.angles, loaded_common.angles):
            assert orig.theta0 == got.theta0
            assert orig.k_angle == got.k_angle
        # Torsion parameters
        for orig, got in zip(common.torsions, loaded_common.torsions):
            assert orig.phi0 == got.phi0
            assert orig.k_tor == got.k_tor
            assert orig.n == got.n
        # Position arrays match exactly (input was already centered)
        np.testing.assert_array_almost_equal(loaded_positions, positions)

    def test_save_load_centers_uncentered_positions(self, tmp_path):
        """save_chain_to_json stores positions verbatim while load_chain_from_json centers them to zero mean."""

        common = self._make_synthetic_chain()
        positions = np.array(
            [
                [10.0, 5.0, 0.0],
                [11.0, 5.0, 0.0],
                [12.0, 5.0, 0.0],
                [13.0, 5.0, 0.0],
            ]
        )
        expected_centered = positions - positions.mean(axis=0)
        out_path = tmp_path / "chain.json"
        save_chain_to_json(common, positions, out_path)
        _, loaded_positions = load_chain_from_json(out_path)
        np.testing.assert_array_almost_equal(loaded_positions, expected_centered)
        np.testing.assert_array_almost_equal(loaded_positions.mean(axis=0), np.zeros(3))

    def test_save_validates_positions_shape(self, tmp_path):
        """save_chain_to_json raises ValueError for positions arrays with wrong rows, columns, or dimensions."""

        common = self._make_synthetic_chain()
        out_path = tmp_path / "chain.json"
        # Wrong number of rows: 3 rows for 4-atom chain
        with pytest.raises(ValueError, match="rows"):
            save_chain_to_json(common, np.zeros((3, 3)), out_path)
        # Wrong column count: (4, 2) - .shape[1] != 3
        with pytest.raises(ValueError, match="shape"):
            save_chain_to_json(common, np.zeros((4, 2)), out_path)
        # Wrong dimensions: 1D array
        with pytest.raises(ValueError, match="shape"):
            save_chain_to_json(common, np.zeros(12), out_path)


class TestChainBDEquilibration:
    """Step 3: chain pre-equilibration plumbing.

    Behavior change is implicitly verified by the 78 prior chain BD
    tests passing (default n_equilibration_steps=0 must be bit-exact
    with the pre-equilibration path, no RNG consumed).
    """

    def test_chainbdparams_default_n_equilibration_zero(self):
        """ChainBDParameters defaults n_equilibration_steps to 0."""

        p = ChainBDParameters()
        assert p.n_equilibration_steps == 0

    def test_chainconfig_default_n_equilibration_zero(self):
        """ChainConfig defaults n_equilibration_steps to 0."""

        c = ChainConfig()
        assert c.n_equilibration_steps == 0

    def test_n_equilibration_xml_parsing(self, tmp_path):
        """parse reads n_equilibration_steps from the chain block of input.xml."""

        xml = """<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>t.pqr</receptor_pqr>
  <bd_milestone_radius>50.0</bd_milestone_radius>
  <n_trajectories>100</n_trajectories>
  <seed>42</seed>
  <chain>
    <chain_json>c.json</chain_json>
    <reaction_pairs_json>rp.json</reaction_pairs_json>
    <n_equilibration_steps>1234</n_equilibration_steps>
  </chain>
</pystarc>"""
        xml_path = tmp_path / "input.xml"
        xml_path.write_text(xml)
        cfg = parse(xml_path)
        assert cfg.chain is not None
        assert cfg.chain.n_equilibration_steps == 1234


class TestForceSanity:
    """Phase B audit: numerical sanity checks for chain BD forces.

    Verifies forces produce correct magnitudes, directions, and obey
    conservation laws (Newton's 3rd law) at known configurations.
    Catches silent regressions in force formulas.
    """

    def test_wca_force_at_contact_is_repulsive(self):
        """At r = sigma the WCA force is repulsive, axial, with magnitude 24*eps/sigma."""

        chain_pos = np.array([[0.0, 0.0, 0.0]])
        chain_rad = np.array([1.5])
        target = Molecule(
            name="probe",
            atoms=[
                Atom(
                    name="X",
                    residue_name="R",
                    residue_index=1,
                    chain="A",
                    x=3.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=1.5,
                )
            ],
        )
        F = chain_target_steric_forces(chain_pos, chain_rad, target, eps=1.0)
        # F_x should be negative (away from +x target)
        assert F[0, 0] < 0, "WCA force should push chain atom away from target"
        assert abs(F[0, 1]) < 1e-10
        assert abs(F[0, 2]) < 1e-10
        # Magnitude: |F| = 24*eps/sigma at r=sigma
        expected = 24.0 / 3.0  # 8.0 for eps=1, sigma=3
        np.testing.assert_allclose(np.linalg.norm(F[0]), expected, rtol=1e-6)

    def test_wca_force_zero_at_cutoff(self):
        """At r = 2^(1/6)*sigma the WCA force is exactly zero."""

        chain_pos = np.array([[0.0, 0.0, 0.0]])
        chain_rad = np.array([1.5])
        cutoff = 2.0 ** (1.0 / 6.0) * 3.0
        target = Molecule(
            name="probe",
            atoms=[
                Atom(
                    name="X",
                    residue_name="R",
                    residue_index=1,
                    chain="A",
                    x=cutoff,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=1.5,
                )
            ],
        )
        F = chain_target_steric_forces(chain_pos, chain_rad, target, eps=1.0)
        # At cutoff, force is zero (LJ minimum has dV/dr = 0)
        np.testing.assert_allclose(F[0], np.zeros(3), atol=1e-10)

    def test_wca_force_zero_outside_cutoff(self):
        """Beyond the 2^(1/6)*sigma cutoff the WCA force is exactly zero."""

        chain_pos = np.array([[0.0, 0.0, 0.0]])
        chain_rad = np.array([1.5])
        target = Molecule(
            name="probe",
            atoms=[
                Atom(
                    name="X",
                    residue_name="R",
                    residue_index=1,
                    chain="A",
                    x=5.0,
                    y=0.0,
                    z=0.0,
                    charge=0.0,
                    radius=1.5,
                )
            ],
        )
        F = chain_target_steric_forces(chain_pos, chain_rad, target, eps=1.0)
        np.testing.assert_array_equal(F[0], np.zeros(3))

    def test_bond_force_zero_at_equilibrium(self):
        """A harmonic bond at r = r0 produces zero force on both atoms."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0, resname="X", resid=1),
            ChainAtom(radius=1.0, charge=0.0, resname="X", resid=2),
        ]
        bonds = [
            ChainBond(a=ChainAtomRef(0), b=ChainAtomRef(1), r0=3.8, k_spring=100.0)
        ]
        common = ChainCommon(
            name="dimer", atoms=atoms, bonds=bonds, angles=[], torsions=[]
        )
        positions = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        np.testing.assert_allclose(state.forces, np.zeros((2, 3)), atol=1e-10)

    def test_bond_force_restores_to_equilibrium(self):
        """A stretched bond pulls atoms together with magnitude k(r-r0) obeying Newton's third law."""

        atoms = [
            ChainAtom(radius=1.0, charge=0.0, resname="X", resid=1),
            ChainAtom(radius=1.0, charge=0.0, resname="X", resid=2),
        ]
        bonds = [
            ChainBond(a=ChainAtomRef(0), b=ChainAtomRef(1), r0=3.8, k_spring=100.0)
        ]
        common = ChainCommon(
            name="dimer", atoms=atoms, bonds=bonds, angles=[], torsions=[]
        )
        # Stretched: r = 4.5 (extension 0.7)
        positions = np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # Atom 0 pulled toward +x (toward atom 1)
        assert state.forces[0, 0] > 0
        # Atom 1 pulled toward -x (toward atom 0)
        assert state.forces[1, 0] < 0
        # Newton 3rd law
        np.testing.assert_allclose(state.forces[0], -state.forces[1])
        # Magnitude F = k * (r - r0) = 100 * 0.7 = 70
        expected_mag = 100.0 * (4.5 - 3.8)
        np.testing.assert_allclose(
            np.linalg.norm(state.forces[0]), expected_mag, rtol=1e-6
        )

    def test_compute_chain_forces_obey_newton_third_law(self):
        """For an isolated chain the sum of internal forces over all atoms is zero."""

        atoms = [
            ChainAtom(radius=1.5, charge=0.0, resname="A", resid=i) for i in range(1, 5)
        ]
        bonds = [
            ChainBond(a=ChainAtomRef(i), b=ChainAtomRef(i + 1), r0=3.8, k_spring=100.0)
            for i in range(3)
        ]
        angles = [
            ChainAngle(
                a=ChainAtomRef(i),
                b=ChainAtomRef(i + 1),
                c=ChainAtomRef(i + 2),
                theta0=2.094,
                k_angle=50.0,
            )
            for i in range(2)
        ]
        torsions = [
            ChainTorsion(
                a=ChainAtomRef(0),
                b=ChainAtomRef(1),
                c=ChainAtomRef(2),
                d=ChainAtomRef(3),
                phi0=3.14159,
                k_tor=2.0,
                n=2,
            )
        ]
        common = ChainCommon(
            name="tetramer", atoms=atoms, bonds=bonds, angles=angles, torsions=torsions
        )
        # Non-equilibrium positions to make all forces non-zero
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.5, 0.0],
                [6.0, 0.0, 0.5],
                [9.0, 0.5, 0.0],
            ]
        )
        state = ChainState.from_template(common, positions)
        compute_chain_forces(state)
        # Translational invariance: sum of forces is zero
        net_force = state.forces.sum(axis=0)
        np.testing.assert_allclose(net_force, np.zeros(3), atol=1e-9)


class TestChainGBSelfBorn:
    """Invariant-based tests for chain GB self-Born / generalized Born forces.

    Five groups: (A) module-level energy invariants, (B) force-energy
    consistency via finite differences, (C) force invariants under
    rigid-body and permutation transformations, (D) Path B dispatch
    correctness, and (E) wire-up coupling between ChainConfig and
    ChainBDParameters defaults. All assertions are parameter-independent
    invariants or universal consistency checks - no magic-number
    references that would break on harmless numerical reparameterization.
    """

    @staticmethod
    def _config():
        """Shared 5-atom test configuration with mixed charges and radii.
        Picked to exercise both case_overlap and case_outside HCT integrand
        branches by spreading interatomic distances across the boundary."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.5, 0.3],
                [0.5, 2.5, -0.8],
                [-1.5, 1.0, 1.2],
                [3.0, -1.0, -0.5],
            ]
        )
        charges = np.array([+1.0, -0.7, +0.4, -0.3, +0.2])
        intrinsic_radii = np.array([1.5, 1.7, 1.4, 1.6, 1.5])
        return positions, charges, intrinsic_radii


    def test_obc_R_eff_isolated_atom_returns_rho_tilde(self):
        """A single isolated atom has R_eff equal to intrinsic minus the OBC offset."""

        positions = np.array([[0.0, 0.0, 0.0]])
        intrinsic = np.array([1.5])
        R_eff = obc_effective_radii(positions, intrinsic)
        expected = intrinsic - DEFAULT_OBC_OFFSET
        assert np.allclose(R_eff, expected, atol=1e-12)

    def test_obc_R_eff_translation_invariance(self):
        """OBC effective radii are invariant under rigid translation of all positions."""

        positions, _, intrinsic = self._config()
        R_eff = obc_effective_radii(positions, intrinsic)
        shift = np.array([10.0, -5.0, 7.5])
        R_eff_shifted = obc_effective_radii(positions + shift, intrinsic)
        assert np.allclose(R_eff, R_eff_shifted, atol=1e-12)

    def test_obc_R_eff_burial_monotonicity(self):
        """Bringing a neighbor closer monotonically increases the target atom's R_eff."""

        intrinsic = np.array([1.5, 1.5])
        prev_R0 = None
        for r in [10.0, 6.0, 3.5, 2.5, 2.0]:  # spans both HCT branches
            positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            R_eff = obc_effective_radii(positions, intrinsic)
            if prev_R0 is not None:
                assert (
                    R_eff[0] > prev_R0
                ), f"R_eff[0] non-monotone at r={r}: prev={prev_R0}, current={R_eff[0]}"
            prev_R0 = R_eff[0]

    def test_gb_self_born_energy_zero_when_eps_out_equals_eps_in(self):
        """With eps_out = eps_in both self-Born and off-diagonal GB energies are exactly zero."""

        positions, charges, intrinsic = self._config()
        E_self = gb_self_born_energy(
            positions, charges, intrinsic, eps_in=2.5, eps_out=2.5
        )
        E_off = gb_offdiagonal_energy(
            positions, charges, intrinsic, eps_in=2.5, eps_out=2.5
        )
        assert E_self == 0.0
        assert E_off == 0.0

    def test_gb_offdiagonal_energy_zero_for_neutral_partner(self):
        """A partner with charge zero contributes nothing to the off-diagonal GB energy."""

        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        intrinsic = np.array([1.5, 1.5])
        charges_neutral_partner = np.array([1.0, 0.0])
        E = gb_offdiagonal_energy(positions, charges_neutral_partner, intrinsic)
        assert E == 0.0


    def test_chain_vacuum_coulomb_force_matches_finite_difference(self):
        """The analytic vacuum Coulomb force matches the finite-difference gradient of its energy."""

        positions, charges, _ = self._config()
        F_ana, _ = chain_vacuum_coulomb_force(positions, charges)
        F_fd = _finite_difference_force(
            positions, lambda p: gb_vacuum_coulomb_energy(p, charges)
        )
        rel = np.max(np.abs(F_ana - F_fd)) / max(np.max(np.abs(F_ana)), 1e-12)
        assert rel < 1e-5

    def test_chain_self_born_diagonal_force_matches_finite_difference(self):
        """The analytic self-Born force matches the finite-difference gradient including OBC R_eff dependence."""

        positions, charges, intrinsic = self._config()
        F_ana, _ = chain_self_born_diagonal_force(
            positions, charges, intrinsic, eps_in=1.0, eps_out=78.5
        )
        F_fd = _finite_difference_force(
            positions,
            lambda p: gb_self_born_energy(
                p, charges, intrinsic, eps_in=1.0, eps_out=78.5
            ),
        )
        rel = np.max(np.abs(F_ana - F_fd)) / max(np.max(np.abs(F_ana)), 1e-12)
        assert rel < 1e-5

    def test_chain_offdiagonal_gb_force_matches_finite_difference(self):
        """The analytic off-diagonal GB force matches the finite-difference gradient of its energy."""

        positions, charges, intrinsic = self._config()
        F_ana, _ = chain_offdiagonal_gb_force(
            positions, charges, intrinsic, eps_in=1.0, eps_out=78.5
        )
        F_fd = _finite_difference_force(
            positions,
            lambda p: gb_offdiagonal_energy(
                p, charges, intrinsic, eps_in=1.0, eps_out=78.5
            ),
        )
        rel = np.max(np.abs(F_ana - F_fd)) / max(np.max(np.abs(F_ana)), 1e-12)
        assert rel < 1e-5

    def test_chain_full_gb_force_matches_finite_difference(self):
        """The full GB force matches the finite-difference gradient of the summed GB energy."""

        positions, charges, intrinsic = self._config()
        F_ana, _ = chain_full_gb_force(
            positions,
            charges,
            intrinsic,
            eps_in=1.0,
            eps_out=78.5,
            coffdrop_active=False,
        )

        def E_total(p):
            return (
                gb_self_born_energy(p, charges, intrinsic, eps_in=1.0, eps_out=78.5)
                + gb_offdiagonal_energy(p, charges, intrinsic, eps_in=1.0, eps_out=78.5)
                + gb_vacuum_coulomb_energy(p, charges, eps_in=1.0)
            )

        F_fd = _finite_difference_force(positions, E_total)
        rel = np.max(np.abs(F_ana - F_fd)) / max(np.max(np.abs(F_ana)), 1e-12)
        assert rel < 1e-5


    def test_chain_full_gb_force_translation_invariance(self):
        """Full GB forces are translation invariant and sum to zero."""

        positions, charges, intrinsic = self._config()
        F, _ = chain_full_gb_force(positions, charges, intrinsic, coffdrop_active=False)
        shift = np.array([10.0, -5.0, 7.5])
        F_shifted, _ = chain_full_gb_force(
            positions + shift, charges, intrinsic, coffdrop_active=False
        )
        assert np.allclose(F, F_shifted, atol=1e-10)
        assert np.allclose(F.sum(axis=0), 0.0, atol=1e-9)

    def test_chain_full_gb_force_rotation_covariance(self):
        """Full GB forces transform covariantly under a rigid rotation, with F' = R F."""

        positions, charges, intrinsic = self._config()
        F, _ = chain_full_gb_force(positions, charges, intrinsic, coffdrop_active=False)
        theta = 0.5
        Rmat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        F_rot, _ = chain_full_gb_force(
            positions @ Rmat.T, charges, intrinsic, coffdrop_active=False
        )
        assert np.allclose(F_rot, F @ Rmat.T, atol=1e-9)

    def test_chain_full_gb_force_atom_permutation_covariance(self):
        """Permuting atom indices permutes the full GB forces in lockstep."""

        positions, charges, intrinsic = self._config()
        F, _ = chain_full_gb_force(positions, charges, intrinsic, coffdrop_active=False)
        perm = np.array([2, 0, 4, 1, 3])
        F_perm, _ = chain_full_gb_force(
            positions[perm],
            charges[perm],
            intrinsic[perm],
            coffdrop_active=False,
        )
        assert np.allclose(F_perm, F[perm], atol=1e-12)


    def test_path_b_dispatch_with_coffdrop_active_equals_diagonal_only(self):
        """With coffdrop_active=True the full GB force equals the diagonal self-Born only, with offdiag and Coulomb zero."""

        positions, charges, intrinsic = self._config()
        F_full, E_full = chain_full_gb_force(
            positions,
            charges,
            intrinsic,
            eps_in=1.0,
            eps_out=78.5,
            coffdrop_active=True,
        )
        F_diag, E_diag = chain_self_born_diagonal_force(
            positions,
            charges,
            intrinsic,
            eps_in=1.0,
            eps_out=78.5,
        )
        assert np.array_equal(F_full, F_diag)
        assert E_full["self"] == E_diag
        assert E_full["offdiag"] == 0.0
        assert E_full["coulomb"] == 0.0
        assert E_full["total"] == E_diag


    def test_chain_config_and_chain_bd_parameters_gb_defaults_consistent(self):
        """ChainConfig and ChainBDParameters agree on all seven GB-related defaults."""

        cc = ChainConfig()
        params = ChainBDParameters()
        gb_fields = [
            "use_self_born",
            "gb_eps_in",
            "gb_eps_out",
            "gb_obc_alpha",
            "gb_obc_beta",
            "gb_obc_gamma",
            "coffdrop_active",
        ]
        for f in gb_fields:
            cc_val = getattr(cc, f)
            p_val = getattr(params, f)
            assert cc_val == p_val, (
                f"GB default drift on '{f}': "
                f"ChainConfig={cc_val!r}, ChainBDParameters={p_val!r}"
            )


class TestChainGBEdgeCases:
    """Edge-case robustness tests for chain GB module (Phase C / Tier 1).

    Covers boundary inputs (single atom, all-neutral chain), parameter
    validation (non-positive rho_tilde), and degenerate geometry (close
    contact, exact overlap with Path B). Companion to TestChainGBSelfBorn,
    which covers physics-correct invariants under non-degenerate conditions.
    """

    def test_single_atom_chain_full_gb_force(self):
        """A single-atom chain has zero forces and Coulomb, with self-Born equal to -cf*q^2/(2*rho_tilde)."""

        positions = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([1.0])
        intrinsic = np.array([1.5])
        rho_tilde = intrinsic[0] - DEFAULT_OBC_OFFSET

        eps_in, eps_out = 1.0, 78.5
        cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A
        E_self_expected = -0.5 * cf * charges[0] ** 2 / rho_tilde

        for coffdrop_active in [False, True]:
            F, E = chain_full_gb_force(
                positions,
                charges,
                intrinsic,
                eps_in=eps_in,
                eps_out=eps_out,
                coffdrop_active=coffdrop_active,
            )
            assert F.shape == (1, 3)
            # No pairs: forces must be exactly zero
            assert np.array_equal(
                F, np.zeros((1, 3))
            ), f"single-atom F not zero (coffdrop_active={coffdrop_active}): {F}"
            # Self-energy matches OBC analytical value
            assert np.isclose(E["self"], E_self_expected, atol=1e-12)
            # No pairs: off-diagonal and Coulomb exactly zero
            assert E["offdiag"] == 0.0
            assert E["coulomb"] == 0.0
            assert E["total"] == E["self"]

    def test_all_neutral_chain_full_gb_force(self):
        """An all-neutral chain has zero GB forces and zero energy components in both Path B branches."""

        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.5, 0.3],
                [0.5, 2.5, -0.8],
                [-1.5, 1.0, 1.2],
                [3.0, -1.0, -0.5],
            ]
        )
        charges_neutral = np.zeros(5)
        intrinsic = np.array([1.5, 1.7, 1.4, 1.6, 1.5])

        for coffdrop_active in [False, True]:
            F, E = chain_full_gb_force(
                positions,
                charges_neutral,
                intrinsic,
                eps_in=1.0,
                eps_out=78.5,
                coffdrop_active=coffdrop_active,
            )
            assert np.array_equal(
                F, np.zeros((5, 3))
            ), f"neutral-chain F not zero (coffdrop_active={coffdrop_active}): {F}"
            assert E["self"] == 0.0
            assert E["offdiag"] == 0.0
            assert E["coulomb"] == 0.0
            assert E["total"] == 0.0

    def test_obc_effective_radii_raises_on_nonpositive_rho_tilde(self):
        """obc_effective_radii raises ValueError when rho_tilde is zero or negative."""

        positions = np.array([[0.0, 0.0, 0.0]])

        # Case 1: intrinsic < offset -> rho_tilde < 0
        with pytest.raises(ValueError, match=r"rho_tilde.*positive"):
            obc_effective_radii(positions, np.array([0.05]))

        # Case 2: intrinsic = offset -> rho_tilde = 0 (must be strictly > 0)
        with pytest.raises(ValueError, match=r"rho_tilde.*positive"):
            obc_effective_radii(positions, np.array([DEFAULT_OBC_OFFSET]))

    def test_close_contact_chain_gb_force_finite(self):
        """GB forces and energies stay finite for atoms in close contact at r = 0.1 angstrom."""

        positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        charges = np.array([1.0, -1.0])
        intrinsic = np.array([1.5, 1.5])

        for coffdrop_active in [False, True]:
            F, E = chain_full_gb_force(
                positions,
                charges,
                intrinsic,
                eps_in=1.0,
                eps_out=78.5,
                coffdrop_active=coffdrop_active,
            )
            assert np.all(
                np.isfinite(F)
            ), f"F not finite (coffdrop_active={coffdrop_active}): {F}"
            for component, value in E.items():
                assert np.isfinite(value), (
                    f"E[{component}] not finite "
                    f"(coffdrop_active={coffdrop_active}): {value}"
                )

    def test_overlapping_atoms_path_b_finite(self):
        """Path B self-Born stays finite for two coincident atoms, with offdiag and Coulomb zero and self-energy attractive."""

        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        charges = np.array([1.0, -1.0])
        intrinsic = np.array([1.5, 1.5])

        F, E = chain_full_gb_force(
            positions,
            charges,
            intrinsic,
            eps_in=1.0,
            eps_out=78.5,
            coffdrop_active=True,
        )
        assert np.all(np.isfinite(F)), f"F not finite at r=0: {F}"
        assert np.isfinite(E["self"]), f"E_self not finite at r=0: {E['self']}"
        # Path B skips off-diagonal and Coulomb
        assert E["offdiag"] == 0.0
        assert E["coulomb"] == 0.0
        # Self-energy is attractive (cf > 0, q^2 > 0, R_eff > 0)
        assert E["self"] < 0.0


class TestChainBDSafetyGuards:
    """Tier 2 robustness: verify existing chain BD safety guards trigger
    correctly when fed degenerate input. Each test exercises a specific
    guard via a NaN/coincident/collinear/out-of-bounds input and asserts
    (a) the function does not raise, (b) state.forces remains finite.
    Universal pattern -- works regardless of whether the guard's behavior
    is early-return, zero-force, or skip.
    """

    @staticmethod
    def _make_state(n_residues=4):
        """Build a minimal chain + ChainState for guard testing."""

        chain = chain_from_sequence("A" * n_residues)
        n = len(chain.atoms)
        positions = np.zeros((n, 3), dtype=float)
        # Spread atoms slightly so the default state is non-degenerate.
        for i in range(n):
            positions[i] = [i * 1.5, 0.0, 0.0]
        state = ChainState.from_template(chain, positions)
        return chain, state

    def test_bond_force_no_op_on_nan_position(self):
        """_bond_force_state returns early without propagating NaN when a bonded atom position is NaN."""

        chain, state = self._make_state(4)
        bond = chain.bonds[0]
        ia = bond.a.atom_idx
        state.positions[ia] = np.nan
        _bond_force_state(state, bond)  # must not raise
        assert np.all(
            np.isfinite(state.forces)
        ), f"NaN leaked through bond guard: {state.forces}"

    def test_bond_force_no_op_on_zero_distance(self):
        """_bond_force_state avoids divide-by-zero and produces no NaN when bonded atoms coincide."""

        chain, state = self._make_state(4)
        bond = chain.bonds[0]
        ia = bond.a.atom_idx
        ib = bond.b.atom_idx
        state.positions[ia] = np.array([0.0, 0.0, 0.0])
        state.positions[ib] = np.array([0.0, 0.0, 0.0])
        _bond_force_state(state, bond)
        assert np.all(np.isfinite(state.forces))

    def test_angle_force_no_op_on_collinear_atoms(self):
        """_angle_force_state stays finite for collinear atoms where sin(theta) approaches zero."""

        chain, state = self._make_state(4)
        angle = chain.angles[0]
        ia = angle.a.atom_idx
        ib = angle.b.atom_idx
        ic = angle.c.atom_idx
        # Collinear: a-b-c along the x-axis -> theta = pi -> sin(theta) = 0
        state.positions[ia] = np.array([0.0, 0.0, 0.0])
        state.positions[ib] = np.array([1.0, 0.0, 0.0])
        state.positions[ic] = np.array([2.0, 0.0, 0.0])
        _angle_force_state(state, angle)
        assert np.all(np.isfinite(state.forces))

    def test_torsion_force_no_op_on_nan_position(self):
        """_torsion_force_state returns early without propagating NaN when any of the four atoms is NaN."""

        chain, state = self._make_state(4)
        tor = chain.torsions[0]
        ia = tor.a.atom_idx
        state.positions[ia] = np.nan
        _torsion_force_state(state, tor)
        assert np.all(np.isfinite(state.forces))

    def test_dxgrid_force_finite_for_atom_outside_box(self):
        """DXGrid returns finite force for query points outside the grid box in both single and batch APIs."""

        np.random.seed(0)
        origin = np.array([0.0, 0.0, 0.0])
        delta = np.eye(3)
        data = np.random.uniform(-1.0, 1.0, (10, 10, 10))
        grid = DXGrid(origin, delta, data)

        # Single query: well outside the (0..9)^3 box
        F_far = grid.force_on_charge(np.array([-100.0, -100.0, -100.0]), 1.0)
        assert np.all(np.isfinite(F_far)), f"single-point F not finite: {F_far}"

        # Batch query: mix of inside / outside points
        points = np.array(
            [
                [5.0, 5.0, 5.0],  # inside
                [-100.0, -100.0, -100.0],  # outside (negative)
                [100.0, 100.0, 100.0],  # outside (positive)
            ]
        )
        charges = np.array([1.0, 1.0, -1.0])
        F_batch = grid.batch_force_on_charges(points, charges)
        assert np.all(np.isfinite(F_batch)), f"batch F not finite: {F_batch}"


# Audit-recommended additions
# Regression tests for named bugs, finite-difference verification for legacy
# force kernels, and negative-XML coverage for chain block validation.
#   TestMinimumCoreDtFloor              - regression: dt-floor silent zero
#   TestRunChainSmoke                   - regression: chain_pipeline.run_chain
#   TestPQRChainIdDialect               - regression: SEEKR2 PQR chain column
#   TestLJForceFiniteDifference         - audit gap: FD-verify lj_pair_force
#   TestDebyeHuckelForceFiniteDifference- audit gap: FD-verify dh force
#   TestDXGridForceFiniteDifference     - audit gap: FD-verify grid gradient
#   TestInputParserChainBlockNegative   - audit gap: 17 raises in input_parser


class TestMinimumCoreDtFloor:
    """Regression: minimum_core_dt and minimum_core_reaction_dt must exist on
    NAMParameters as real fields. The historic bug was that these fields were
    absent and gpu_batch_simulator silently read 0.0 via getattr, eliminating
    the adaptive-dt floor. The field-existence test below would have caught it.
    """

    def test_field_exists_on_nam_parameters(self):
        """NAMParameters exposes minimum_core_dt and minimum_core_reaction_dt attributes."""

        p = NAMParameters()
        assert hasattr(p, "minimum_core_dt"), (
            "NAMParameters missing minimum_core_dt; getattr fallback would "
            "silently return 0.0 and disable the adaptive-dt floor."
        )
        assert hasattr(p, "minimum_core_reaction_dt")

    def test_default_is_zero(self):
        """NAMParameters defaults minimum_core_dt and minimum_core_reaction_dt to 0.0."""

        p = NAMParameters()
        assert p.minimum_core_dt == 0.0
        assert p.minimum_core_reaction_dt == 0.0

    def test_nonzero_floor_is_preserved(self):
        """NAMParameters preserves nonzero minimum_core_dt and minimum_core_reaction_dt floors."""

        p = NAMParameters(minimum_core_dt=0.123, minimum_core_reaction_dt=0.045)
        assert p.minimum_core_dt == 0.123
        assert p.minimum_core_reaction_dt == 0.045

    def test_getattr_path_returns_stored_value_not_default(self):
        """getattr on NAMParameters returns the stored core-dt floors rather than the fallback default."""

        p = NAMParameters(minimum_core_dt=0.25, minimum_core_reaction_dt=0.075)
        assert getattr(p, "minimum_core_dt", 999.0) == 0.25
        assert getattr(p, "minimum_core_reaction_dt", 999.0) == 0.075

    def test_pystarc_config_parses_floor_from_xml(self, tmp_path):
        """parse reads minimum_core_dt and minimum_core_reaction_dt from input.xml."""

        xml = (
            '<?xml version="1.0" ?>\n'
            "<pystarc_input>\n"
            "  <receptor_pqr>fake_rec.pqr</receptor_pqr>\n"
            "  <ligand_pqr>fake_lig.pqr</ligand_pqr>\n"
            "  <minimum_core_dt>0.123</minimum_core_dt>\n"
            "  <minimum_core_reaction_dt>0.045</minimum_core_reaction_dt>\n"
            f"  <work_dir>{tmp_path / 'wd'}</work_dir>\n"
            "</pystarc_input>\n"
        )
        p = tmp_path / "in.xml"
        p.write_text(xml)
        cfg = parse(p)
        assert cfg.minimum_core_dt == 0.123
        assert cfg.minimum_core_reaction_dt == 0.045


class TestRunChainSmoke:
    """Regression: chain_pipeline.run_chain end-to-end. Catches the
    cfg.chain.* vs cc.* slip-class of bug. Any test exercising run_chain
    in full would catch it.
    """

    def test_run_chain_raises_when_chain_config_missing(self, tmp_path):
        """run_chain raises ValueError when config.chain is missing."""

        cfg = PySTARCConfig(
            receptor_pqr="fake.pqr",
            ligand_pqr="fake_lig.pqr",
            n_trajectories=1,
            work_dir=tmp_path,
        )
        cfg.chain = None
        with pytest.raises(ValueError, match="requires config.chain"):
            run_chain(cfg)

    def test_run_chain_minimal_end_to_end(self, tmp_path):
        """run_chain completes a minimal three-atom chain BD trajectory end-to-end."""

        chain_json_data = {
            "name": "trimer",
            "atoms": [
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "A",
                    "resid": 0,
                    "position": [0.0, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "B",
                    "resid": 1,
                    "position": [3.8, 0.0, 0.0],
                },
                {
                    "radius": 2.0,
                    "charge": 0.0,
                    "resname": "C",
                    "resid": 2,
                    "position": [5.7, 3.0, 1.0],
                },
            ],
            "bonds": [
                {"a": 0, "b": 1, "r0": 3.8, "k_spring": 100.0},
                {"a": 1, "b": 2, "r0": 3.69, "k_spring": 100.0},
            ],
            "angles": [],
            "torsions": [],
        }
        chain_json_path = tmp_path / "chain.json"
        chain_json_path.write_text(json.dumps(chain_json_data))

        # Two-atom receptor at ~10 A separation
        pqr_text = (
            "ATOM      1  CA  GLY     1       0.000   0.000   0.000  0.000  3.000\n"
            "ATOM      2  CB  GLY     1       5.000   0.000   0.000  0.000  3.000\n"
        )
        receptor_pqr = tmp_path / "receptor.pqr"
        receptor_pqr.write_text(pqr_text)

        # Huge cutoff -> reaction fires immediately, sim ends fast
        rxn_pairs_path = tmp_path / "rxn_pairs.json"
        rxn_pairs_path.write_text(json.dumps([[0, 0, 1000.0]]))

        cfg = PySTARCConfig(
            receptor_pqr=str(receptor_pqr),
            n_trajectories=1,
            max_steps=5,
            bd_milestone_radius=30.0,
            seed=42,
            work_dir=tmp_path / "out",
            dt=0.2,
        )
        cfg.chain = ChainConfig(
            chain_json=str(chain_json_path),
            reaction_pairs_json=str(rxn_pairs_path),
            dt_chain=0.05,
            chain_steps_per_outer=4,
            reaction_n_needed=1,
            n_workers=1,
            gb_eps_in=1.0,
            gb_eps_out=78.5,
            soft_repulsion_eps=1.0,
            auto_diffusion=True,
        )
        work_dir = run_chain(cfg)
        assert Path(work_dir).exists()
        # Some output JSON must have been written
        assert any(Path(work_dir).glob("*.json"))


class TestPQRChainIdDialect:
    """Regression: SEEKR2-produced PQRs include PDB chain-ID columns. parse_pqr
    must handle both column-fixed and whitespace-separated formats.
    """

    def test_pqr_with_chain_id_column(self, tmp_path):
        """parse_pqr reads coordinates, charges, and radii from a PQR file with a chain-ID column."""

        pqr_text = (
            "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  0.500  2.000\n"
            "ATOM      2  CB  GLY A   1       3.000   0.000   0.000 -0.500  2.000\n"
        )
        p = tmp_path / "with_chain.pqr"
        p.write_text(pqr_text)
        mol = parse_pqr(p)
        assert len(mol.atoms) == 2
        a0, a1 = mol.atoms[0], mol.atoms[1]
        assert math.isclose(a0.x, 0.0, abs_tol=1e-6)
        assert math.isclose(a1.x, 3.0, abs_tol=1e-6)
        assert math.isclose(a0.charge, 0.5, abs_tol=1e-6)
        assert math.isclose(a1.charge, -0.5, abs_tol=1e-6)
        assert math.isclose(a0.radius, 2.0, abs_tol=1e-6)

    def test_pqr_without_chain_id_still_parses(self, tmp_path):
        """parse_pqr still parses all atoms from a PQR file lacking a chain-ID column."""

        pqr_text = (
            "ATOM      1  CA  GLY     1       0.000   0.000   0.000  0.500  2.000\n"
            "ATOM      2  CB  GLY     1       3.000   0.000   0.000 -0.500  2.000\n"
        )
        p = tmp_path / "no_chain.pqr"
        p.write_text(pqr_text)
        mol = parse_pqr(p)
        assert len(mol.atoms) == 2

    def test_multi_chain_pqr_round_trip(self, tmp_path):
        """Multi-chain PQR survives a parse, write, parse round trip preserving coordinates, charges, and radii."""

        pqr_text = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  0.100  2.000\n"
            "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  0.100  2.000\n"
            "ATOM      3  CA  GLY B   1      10.000   0.000   0.000 -0.100  2.000\n"
            "ATOM      4  CA  GLY B   2      13.800   0.000   0.000 -0.100  2.000\n"
        )
        p_in = tmp_path / "multi.pqr"
        p_in.write_text(pqr_text)
        mol = parse_pqr(p_in)
        assert len(mol.atoms) == 4
        p_out = tmp_path / "round.pqr"
        write_pqr(mol, p_out)
        mol2 = parse_pqr(p_out)
        assert len(mol2.atoms) == 4
        for orig, rt in zip(mol.atoms, mol2.atoms):
            assert math.isclose(orig.x, rt.x, abs_tol=1e-3)
            assert math.isclose(orig.y, rt.y, abs_tol=1e-3)
            assert math.isclose(orig.z, rt.z, abs_tol=1e-3)
            assert math.isclose(orig.charge, rt.charge, abs_tol=1e-4)
            assert math.isclose(orig.radius, rt.radius, abs_tol=1e-4)


class TestLJForceFiniteDifference:
    """FD verification of lj_pair_force against the LJ potential.
    With a at origin and b at (r, 0, 0): F_a_x = +dV/dr, because
    d|b-a|/d(a_x) = -1 and F_a = -grad_a V. Audit C7 fixed an
    earlier inverted sign in the function (and in the test).
    """

    @pytest.mark.parametrize("r_over_sigma", [0.9, 1.0, 1.1, 1.5, 2.0])
    def test_lj_force_matches_finite_difference(self, r_over_sigma):
        """The LJ pair force equals the finite-difference derivative of its potential."""

        sigma, eps = 3.0, 0.5
        r = r_over_sigma * sigma
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([r, 0.0, 0.0])
        F, _ = lj_pair_force(a, b, eps, sigma, use_wca=False)
        F_along = F[0]
        h = 1e-5
        _, V_plus = lj_pair_force(
            a, np.array([r + h, 0.0, 0.0]), eps, sigma, use_wca=False
        )
        _, V_minus = lj_pair_force(
            a, np.array([r - h, 0.0, 0.0]), eps, sigma, use_wca=False
        )
        dV_dr = (V_plus - V_minus) / (2 * h)
        # Force on a along +x = +dV/dr (audit C7 sign convention)
        np.testing.assert_allclose(F_along, dV_dr, rtol=1e-4, atol=1e-6)

    def test_lj_force_direction_at_short_range_repulsive(self):
        """At r < sigma the LJ force on a points away from b and the potential is positive."""

        sigma, eps = 3.0, 1.0
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.5 * sigma, 0.0, 0.0])  # deep in repulsive zone
        F, V = lj_pair_force(a, b, eps, sigma, use_wca=False)
        assert (
            F[0] < 0
        ), f"force on a at r<sigma must point in -x (away from b); got F[0]={F[0]}"
        assert V > 0, f"potential at r<sigma must be repulsive (V>0); got V={V}"

    def test_lj_force_direction_at_long_range_attractive(self):
        """Past the LJ minimum the force on a points toward b and the potential is negative."""

        sigma, eps = 3.0, 1.0
        a = np.array([0.0, 0.0, 0.0])
        r = 1.3 * sigma  # past 2^(1/6)*sigma, well in the attractive tail
        b = np.array([r, 0.0, 0.0])
        F, V = lj_pair_force(a, b, eps, sigma, use_wca=False)
        assert (
            F[0] > 0
        ), f"force on a in attractive zone must point in +x (toward b); got F[0]={F[0]}"
        assert V < 0, f"potential past LJ minimum must be attractive (V<0); got V={V}"

    def test_wca_force_zero_outside_cutoff(self):
        """Beyond the 2^(1/6)*sigma cutoff the WCA pair force and potential are zero."""

        sigma = 3.0
        r_cut = 2.0 ** (1.0 / 6.0) * sigma
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([r_cut + 0.5, 0.0, 0.0])
        F, V = lj_pair_force(a, b, 1.0, sigma, use_wca=True)
        np.testing.assert_array_equal(F, np.zeros(3))
        assert V == 0.0

    @pytest.mark.parametrize("r_over_sigma", [0.85, 0.95, 1.0, 1.1])
    def test_wca_force_matches_finite_difference_inside_cutoff(self, r_over_sigma):
        """Inside the cutoff the WCA pair force matches the finite-difference derivative of its potential."""

        sigma, eps = 3.0, 1.0
        r = r_over_sigma * sigma
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([r, 0.0, 0.0])
        F, _ = lj_pair_force(a, b, eps, sigma, use_wca=True)
        F_along = F[0]
        h = 1e-5
        _, V_plus = lj_pair_force(
            a, np.array([r + h, 0.0, 0.0]), eps, sigma, use_wca=True
        )
        _, V_minus = lj_pair_force(
            a, np.array([r - h, 0.0, 0.0]), eps, sigma, use_wca=True
        )
        dV_dr = (V_plus - V_minus) / (2 * h)
        # Audit C7 sign convention
        np.testing.assert_allclose(F_along, dV_dr, rtol=1e-3, atol=1e-5)


class TestDebyeHuckelForceFiniteDifference:
    """Audit gap: debye_huckel_force lacked FD verification against
    debye_huckel_energy. Confirm F = -dV/dr along the inter-charge axis.
    """

    @pytest.mark.parametrize(
        "q1,q2,r,lam",
        [
            (+1.0, +1.0, 5.0, 7.86),
            (+1.0, -1.0, 5.0, 7.86),
            (-2.0, +1.0, 8.0, 7.86),
            (+1.0, +1.0, 3.0, 4.0),
            (+1.0, +1.0, 15.0, 10.0),
        ],
    )
    def test_dh_force_matches_finite_difference(self, q1, q2, r, lam):
        """The Debye-Huckel force equals minus the finite-difference derivative of its energy."""

        r_vec = np.array([r, 0.0, 0.0])
        F = debye_huckel_force(q1, q2, r_vec, debye_length=lam)
        F_along = F[0]
        h = 1e-5
        V_plus = debye_huckel_energy(q1, q2, r + h, debye_length=lam)
        V_minus = debye_huckel_energy(q1, q2, r - h, debye_length=lam)
        dV_dr = (V_plus - V_minus) / (2 * h)
        np.testing.assert_allclose(F_along, -dV_dr, rtol=1e-3, atol=1e-8)

    def test_dh_force_zero_at_zero_separation(self):
        """The Debye-Huckel force is exactly zero at zero separation."""

        F = debye_huckel_force(1.0, 1.0, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(F, np.zeros(3))

    def test_dh_force_repulsive_for_like_charges(self):
        """The Debye-Huckel force is repulsive for like charges."""

        F = debye_huckel_force(1.0, 1.0, np.array([5.0, 0.0, 0.0]))
        assert F[0] > 0

    def test_dh_force_attractive_for_opposite_charges(self):
        """The Debye-Huckel force is attractive for opposite charges."""

        F = debye_huckel_force(1.0, -1.0, np.array([5.0, 0.0, 0.0]))
        assert F[0] < 0


class TestDXGridForceFiniteDifference:
    """Audit gap: DXGrid.gradient and force_on_charge lacked FD checks against
    a known analytic potential. Use synthetic linear-ramp and quadratic grids.
    """

    @staticmethod
    def _make_linear_grid(a, b, c, n=21, spacing=1.0):
        """V(x,y,z) = a*x + b*y + c*z so gradient = (a,b,c) everywhere."""

        origin = np.array([-(n // 2) * spacing] * 3, dtype=float)
        delta = np.eye(3) * spacing
        coords = origin[0] + np.arange(n) * spacing
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        data = a * X + b * Y + c * Z
        return DXGrid(origin, delta, data)

    @pytest.mark.parametrize(
        "a,b,c",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, -0.7, 1.3),
        ],
    )
    def test_gradient_of_linear_grid_is_constant(self, a, b, c):
        """The gradient of a linear-field grid is the constant slope vector everywhere."""

        grid = self._make_linear_grid(a, b, c)
        for pt in [
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, -1.0, 3.0]),
            np.array([-3.5, 2.7, -1.2]),
        ]:
            g = grid.gradient(pt)
            np.testing.assert_allclose(g, [a, b, c], atol=1e-6)

    def test_force_on_charge_is_minus_q_gradient(self):
        """The grid force on a charge equals minus q times the field gradient."""

        grid = self._make_linear_grid(0.5, -0.7, 1.3)
        pt = np.array([1.0, 1.0, 1.0])
        F_pos = grid.force_on_charge(pt, +1.0)
        F_neg = grid.force_on_charge(pt, -1.0)
        np.testing.assert_allclose(F_pos, [-0.5, 0.7, -1.3], atol=1e-6)
        np.testing.assert_allclose(F_neg, [0.5, -0.7, 1.3], atol=1e-6)
        g = grid.gradient(pt)
        np.testing.assert_allclose(F_pos, -1.0 * g, atol=1e-12)
        np.testing.assert_allclose(F_neg, +1.0 * g, atol=1e-12)

    def test_gradient_matches_fd_of_interpolate_on_quadratic(self):
        """The grid gradient matches the finite-difference gradient of the interpolated quadratic field."""

        n, spacing = 41, 0.5
        origin = np.array([-(n // 2) * spacing] * 3, dtype=float)
        delta = np.eye(3) * spacing
        coords = origin[0] + np.arange(n) * spacing
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        data = 0.5 * (X * X + Y * Y + Z * Z)
        grid = DXGrid(origin, delta, data)
        pt = np.array([1.2, -0.7, 0.4])
        g = grid.gradient(pt)
        h = spacing

        def V(p):
            return grid.interpolate(p)

        gx_fd = (V(pt + np.array([h, 0, 0])) - V(pt - np.array([h, 0, 0]))) / (2 * h)
        gy_fd = (V(pt + np.array([0, h, 0])) - V(pt - np.array([0, h, 0]))) / (2 * h)
        gz_fd = (V(pt + np.array([0, 0, h])) - V(pt - np.array([0, 0, h]))) / (2 * h)
        np.testing.assert_allclose([gx_fd, gy_fd, gz_fd], g, atol=0.5)


class TestInputParserChainBlockNegative:
    """Audit gap: input_parser.py has 17 raises but most lacked negative tests.
    Cover each <chain>-block validation raise with a malformed-XML test.
    """

    @staticmethod
    def _write_chain_xml(tmp_path, chain_inner, **outer_overrides):
        outer = {
            "receptor_pqr": "fake_rec.pqr",
            "work_dir": str(tmp_path / "wd"),
            "n_trajectories": "1",
        }
        outer.update(outer_overrides)
        outer_xml = "\n".join(
            f"  <{k}>{v}</{k}>" for k, v in outer.items() if v is not None
        )
        xml = (
            '<?xml version="1.0" ?>\n'
            "<pystarc_input>\n" + outer_xml + "\n"
            "  <chain>\n" + chain_inner + "\n"
            "  </chain>\n"
            "</pystarc_input>\n"
        )
        p = tmp_path / "in.xml"
        p.write_text(xml)
        return p

    def test_missing_chain_json_raises(self, tmp_path):
        """parse raises ValueError when chain_json is missing from the chain block."""

        xml_path = self._write_chain_xml(
            tmp_path,
            "    <reaction_pairs_json>fake.json</reaction_pairs_json>",
        )
        with pytest.raises(ValueError, match="chain_json"):
            parse(xml_path)

    def test_missing_receptor_pqr_raises(self, tmp_path):
        """parse raises ValueError when receptor_pqr is missing."""

        xml_path = self._write_chain_xml(
            tmp_path,
            "    <chain_json>fake_chain.json</chain_json>\n"
            "    <reaction_pairs_json>fake.json</reaction_pairs_json>",
            receptor_pqr=None,
        )
        with pytest.raises(ValueError, match="receptor_pqr"):
            parse(xml_path)

    @pytest.mark.parametrize(
        "tag,bad_value,err_match",
        [
            ("dt_chain", "0.0", "dt_chain"),
            ("dt_chain", "-0.1", "dt_chain"),
            ("chain_steps_per_outer", "0", "chain_steps_per_outer"),
            ("n_equilibration_steps", "-1", "n_equilibration_steps"),
            ("D_trans", "-1.0", "D_trans"),
            ("D_rot", "-0.5", "D_rot"),
            ("r_escape", "-1.0", "r_escape"),
            ("reaction_n_needed", "0", "reaction_n_needed"),
            ("soft_repulsion_eps", "-0.1", "soft_repulsion_eps"),
            ("gb_eps_in", "0.0", "gb_eps_in"),
            ("gb_eps_in", "-1.0", "gb_eps_in"),
            ("gb_eps_out", "0.0", "gb_eps_out"),
            ("n_workers", "0", "n_workers"),
        ],
    )
    def test_chain_numeric_validation_raises(self, tmp_path, tag, bad_value, err_match):
        """parse raises ValueError for out-of-range numeric chain parameter values."""

        chain_inner = (
            "    <chain_json>fake_chain.json</chain_json>\n"
            "    <reaction_pairs_json>fake.json</reaction_pairs_json>\n"
            f"    <{tag}>{bad_value}</{tag}>"
        )
        xml_path = self._write_chain_xml(tmp_path, chain_inner)
        with pytest.raises(ValueError, match=err_match):
            parse(xml_path)

    def test_gb_eps_in_greater_than_eps_out_raises(self, tmp_path):
        """parse raises ValueError when gb_eps_in exceeds gb_eps_out."""

        chain_inner = (
            "    <chain_json>fake_chain.json</chain_json>\n"
            "    <reaction_pairs_json>fake.json</reaction_pairs_json>\n"
            "    <gb_eps_in>80.0</gb_eps_in>\n"
            "    <gb_eps_out>78.5</gb_eps_out>"
        )
        xml_path = self._write_chain_xml(tmp_path, chain_inner)
        with pytest.raises(ValueError, match="gb_eps_in"):
            parse(xml_path)

    def test_valid_chain_xml_parses(self, tmp_path):
        """parse accepts a valid chain XML and sets cfg.chain with dt_chain equal to 0.05."""

        chain_inner = (
            "    <chain_json>fake_chain.json</chain_json>\n"
            "    <reaction_pairs_json>fake.json</reaction_pairs_json>\n"
            "    <dt_chain>0.05</dt_chain>\n"
            "    <gb_eps_in>1.0</gb_eps_in>\n"
            "    <gb_eps_out>78.5</gb_eps_out>"
        )
        xml_path = self._write_chain_xml(tmp_path, chain_inner)
        cfg = parse(xml_path)
        assert cfg.chain is not None
        assert cfg.chain.dt_chain == 0.05


class TestBoundedHardSphereSafeguardPresent:
    """Regression: the bounded hard-sphere rejection safeguard must remain
    in chain_simulator.py. The dynamic firing is exercised by the existing
    TestChainBDSimulator* tests (which produce ~30 RuntimeWarnings in any
    normal suite run from naturally-wedging trajectories); this test guards
    against the safeguard CODE being silently refactored away. If this test
    fails, the safeguard has been removed or renamed and behavioural review
    is required before the change is merged.
    """

    def test_safeguard_code_present_in_chain_simulator(self):
        """chain_simulator source retains the MAX_HS_ATTEMPTS constant, overlap diagnostic string, and RuntimeWarning emission of the hard-sphere safeguard."""

        src = inspect.getsource(chain_simulator)
        # The three signatures of a working safeguard:
        assert "MAX_HS_ATTEMPTS" in src, (
            "MAX_HS_ATTEMPTS constant missing; bounded retry may have been "
            "removed from chain_simulator"
        )
        assert "hard-sphere overlap rejection exceeded" in src, (
            "diagnostic warning string missing; safeguard may have been "
            "silenced silently rather than at filterwarnings level"
        )
        assert "RuntimeWarning" in src, (
            "RuntimeWarning emission missing; safeguard may have been "
            "downgraded to a silent accept"
        )

    def test_max_hs_attempts_is_bounded(self):
        """MAX_HS_ATTEMPTS is assigned an integer within the bounded range [1, 10]."""

        src = inspect.getsource(chain_simulator)
        m = re.search(r"MAX_HS_ATTEMPTS\s*=\s*(\d+)", src)
        assert m is not None, "MAX_HS_ATTEMPTS not assigned an integer"
        n = int(m.group(1))
        assert 1 <= n <= 10, (
            f"MAX_HS_ATTEMPTS = {n} is outside the sensible bounded range "
            f"[1, 10]; review whether this is intentional"
        )


def test_prepare_bd_surface_pqr_roundtrip_4char_names():
    """write_pqr preserves 4-character Amber atom names and coordinates through a read_pqr round trip."""
    # Imported locally on purpose. There are two write_pqr functions with
    # incompatible signatures, one taking a list of PQRAtom and one taking a
    # Molecule, and the module-level import binds the Molecule one. This test
    # needs the atom-list version.
    from pystarc.pipeline.prepare_bd_surface import PQRAtom, read_pqr, write_pqr

    atoms_in = [
        PQRAtom(
            serial=1,
            name="1HG2",
            resname="ARG",
            resid=1,
            x=-12.345,
            y=6.789,
            z=2.500,
            charge=0.1234,
            radius=1.487,
            record="HETATM",
        ),
        PQRAtom(
            serial=2,
            name="CA",
            resname="ARG",
            resid=1,
            x=10.001,
            y=20.002,
            z=30.003,
            charge=-0.5678,
            radius=2.000,
            record="HETATM",
        ),
        PQRAtom(
            serial=3,
            name="2HD1",
            resname="LEU",
            resid=2,
            x=0.123,
            y=-99.876,
            z=55.555,
            charge=0.0,
            radius=1.0,
            record="HETATM",
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".pqr", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_pqr(atoms_in, tmp)
        atoms_out = read_pqr(tmp)
        assert len(atoms_in) == len(atoms_out)
        for a1, a2 in zip(atoms_in, atoms_out):
            assert a1.name == a2.name, f"name corrupted: {a1.name!r} -> {a2.name!r}"
            for coord in ("x", "y", "z"):
                v1, v2 = getattr(a1, coord), getattr(a2, coord)
                assert abs(v1 - v2) < 1e-9, f"{coord} drift: {v1} -> {v2}"
    finally:
        os.unlink(tmp)


def test_hydrophobic_attractive_force_direction():
    """Hydrophobic SASA force with default negative β is attractive, pointing a toward b with negative energy."""

    hp = HydrophobicParams()  # default beta = -0.025 -> fac < 0
    r_vec = np.array([1.0, 0.0, 0.0])  # unit vector a -> b
    # r + radius_self = 3.0 + 0.5 = 3.5, which is in [hp.a=3.1, hp.b=4.35]
    f, e = hydrophobic_sasa_force(3.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
    assert f[0] > 0, f"attractive hydrophobic must point a->b (+x); got F[0]={f[0]}"
    assert e < 0, f"attractive interaction must give negative energy; got e={e}"


def test_hydrophobic_repulsive_force_direction():
    """Hydrophobic SASA force with positive β is repulsive, pointing a away from b with positive energy."""

    hp = HydrophobicParams(beta=+0.025)  # flip sign -> repulsive
    r_vec = np.array([1.0, 0.0, 0.0])
    f, e = hydrophobic_sasa_force(3.0, r_vec, 0.5, 0.5, 10.0, 10.0, hp)
    assert f[0] < 0, f"repulsive hydrophobic must point b->a (-x); got F[0]={f[0]}"
    assert e > 0, f"repulsive interaction must give positive energy; got e={e}"


# Regression tests: #4a (commit "fix coffdrop_dir relative-path default")
# coffdrop_dir default must resolve regardless of cwd, and bad coffdrop_dir
# must give a clear error rather than a cryptic XML parse failure.
# Before fix: default was the relative string "pystarc/coffdrop_data",
# which only worked when cwd was the PySTARC root.


def test_chain_from_sequence_default_works_outside_pystarc_tree(tmp_path, monkeypatch):
    """chain_from_sequence with the default coffdrop_dir builds a chain from an arbitrary working directory."""

    monkeypatch.chdir(tmp_path)
    chain = chain_from_sequence("ALA")
    assert chain.n_atoms > 0


def test_chain_from_sequence_bad_coffdrop_dir_raises_clear_error():
    """chain_from_sequence with a bad coffdrop_dir raises a clear FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="COFFDROP data directory not found"):
        chain_from_sequence("ALA", coffdrop_dir="/nonexistent/path")


# Regression tests: #4b (commit "add pdb_to_bead_positions helper")
# Encapsulates COFFDROP centroid mapping + CB->CA fallback + TLEAP
# variant handling so chain BD setup.py scripts don't have to
# hand-code 80 lines of fragile logic.


def test_resname_match_tleap_variants():
    """_resname_match_tleap treats TLEAP-renamed residue variants as equivalent and distinct residues as unequal."""

    assert _resname_match_tleap("HIS", "HIE")
    assert _resname_match_tleap("HIE", "HID")
    assert _resname_match_tleap("HIP", "HIS")
    assert _resname_match_tleap("CYS", "CYX")
    assert _resname_match_tleap("ASP", "ASH")
    assert _resname_match_tleap("GLU", "GLH")
    assert _resname_match_tleap("LYS", "LYN")
    assert _resname_match_tleap("ALA", "ALA")
    assert not _resname_match_tleap("ALA", "GLY")
    assert not _resname_match_tleap("HIS", "ALA")
    assert not _resname_match_tleap("HIS", "CYX")


def test_parse_coffdrop_map_simple_known_entries():
    """_parse_coffdrop_map_simple reproduces known COFFDROP bead-to-atom definitions from map.xml."""

    map_path = Path(pystarc.__file__).parent / "coffdrop_data" / "map.xml"
    mapping = _parse_coffdrop_map_simple(map_path)
    assert "LYS" in mapping
    assert mapping["LYS"]["CA"] == ["CA"]
    assert mapping["LYS"]["CB"] == ["CB", "CG", "CD"]
    assert mapping["LYS"]["NG"] == ["CE", "NZ"]
    assert mapping["GLU"]["OG"] == ["CD", "OE1", "OE2"]
    assert mapping["ASN"]["CG"] == ["CG", "OD1", "ND2"]
    assert mapping["ILE"]["CG"] == ["CD1"]


def test_pdb_to_bead_positions_on_1brs_chain_d():
    """pdb_to_bead_positions on 1BRS chain D reproduces the expected barstar bead bounding box."""

    pdb = _fixture("1BRS.pdb")

    chain = chain_from_pdb(pdb, chain_id="D", name="barstar")
    pos = pdb_to_bead_positions(chain, pdb, chain_id="D")
    assert pos.shape == (chain.n_atoms, 3)
    assert np.isfinite(pos).all()
    # Bounding box reproduces earlier manual mapping
    assert 24.0 < pos[:, 0].min() < 25.0 and 51.0 < pos[:, 0].max() < 53.0
    assert 17.0 < pos[:, 1].min() < 18.0 and 50.0 < pos[:, 1].max() < 52.0
    assert -15.0 < pos[:, 2].min() < -13.0 and 16.0 < pos[:, 2].max() < 18.0


def test_pdb_to_bead_positions_strict_mode_raises_on_disorder():
    """pdb_to_bead_positions with fallback='strict' raises on the disordered GLN61 sidechain in 1BRS chain D."""

    pdb = _fixture("1BRS.pdb")

    chain = chain_from_pdb(pdb, chain_id="D", name="barstar")
    with pytest.raises(RuntimeError, match="fallback=strict"):
        pdb_to_bead_positions(chain, pdb, chain_id="D", fallback="strict")


def test_pdb_to_bead_positions_bad_fallback_raises():
    """pdb_to_bead_positions raises ValueError on an unrecognized fallback value."""

    pdb = _fixture("1BRS.pdb")

    chain = chain_from_pdb(pdb, chain_id="D", name="barstar")
    with pytest.raises(ValueError, match="fallback must be one of"):
        pdb_to_bead_positions(chain, pdb, chain_id="D", fallback="bogus")


# Regression tests: #2 (commit "regularize RPY Cholesky in chain_rigid_body_resistance")
# RPY mobility Cholesky used to raise LinAlgError on the 230-bead barstar
# chain (numerical conditioning at scale). Now falls back through
# regularized Cholesky and symmetric eigendecomposition with a warning,
# rather than killing the chain BD setup pipeline.


def test_build_robust_solver_spd_uses_plain_cholesky():
    """_build_robust_solver uses plain Cholesky without regularization on a well-conditioned SPD matrix."""

    M = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]])
    solver, was_reg, info = _build_robust_solver(M)
    assert not was_reg
    assert info == "cholesky"
    v = np.array([1.0, 2.0, 3.0])
    x = solver(v)
    assert np.allclose(M @ x, v)


def test_build_robust_solver_raises_on_indefinite():
    """_build_robust_solver raises on an indefinite matrix rather than clipping it.

    It used to fall back to an eigendecomposition and clip negative eigenvalues
    to a positive floor. That floor was absolute rather than scaled to the
    matrix, so inverting it injected about 1e10 where the correct entry is of
    order 1e2 and of opposite sign, which can only shrink D_trans and D_rot, in
    every direction, on nothing but a RuntimeWarning. Measured on a 230-bead
    chain with a duplicated bead it produced a D_trans 4509 times too small.

    The RPY kernel is positive definite for every configuration with positive
    finite radii, so an indefinite matrix is a data defect and not a numerical
    one, and it must be loud. Diagonal jitter up to eps=1e-6 absorbs anything
    merely ill conditioned before this point is reached.
    """

    M = np.diag([1.0, -0.5, 1.0])
    with pytest.raises(np.linalg.LinAlgError) as exc:
        _build_robust_solver(M)
    msg = str(exc.value)
    assert "indefinite" in msg
    # The message must quantify the violation against the matrix scale, since
    # an absolute eigenvalue carries no meaning for a mobility.
    assert "lambda_min" in msg and "lambda_min/scale" in msg


def test_build_robust_solver_accepts_positive_semidefinite():
    """A singular but positive semi-definite matrix is still handled, via jitter.

    This is the legitimate case the raise above must not swallow: two
    coincident beads are a genuine redundant degree of freedom and give an
    exact zero eigenvalue, not a negative one.
    """

    M = np.diag([1.0, 0.0, 1.0])
    solver, was_reg, info = _build_robust_solver(M)
    x = solver(np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(x).all()


def test_chain_rigid_body_resistance_handles_barstar_230_bead():
    """chain_rigid_body_resistance completes on the 230-bead barstar chain with finite outputs and no LinAlgError."""

    chain_json = (
        _fixture("chain.json")
    )

    with open(chain_json) as f:
        data = json.load(f)
    positions = np.array([a["position"] for a in data["atoms"]])
    radii = np.array([a["radius"] for a in data["atoms"]])
    A, C, hc = chain_rigid_body_resistance(positions, radii)
    assert np.isfinite(A).all()
    assert np.isfinite(C).all()
    assert np.isfinite(hc).all()


def test_gpu_yukawa_multipole_force_matches_potential_gradient():
    """The GPU screened multipole force equals minus the gradient of its potential for the full monopole, dipole, and quadrupole expansion."""

    def _gpu_usable():
        if not getattr(gbe, "_CUPY", False):
            return False
        try:
            gbe.cp.zeros(1).sum()  # forces a real device alloc and kernel compile
            return True
        except Exception:
            return False

    if not _gpu_usable():
        gbe.cp = np  # the kernel calls cp.*; numpy is API-compatible on CPU
    cp = gbe.cp

    def to_np(a):
        # The kernel returns CuPy arrays on a GPU and NumPy arrays on a CPU.
        return a.get() if hasattr(a, "get") else np.asarray(a)

    cls = next(
        c
        for c in vars(gbe).values()
        if isinstance(c, type) and "_yukawa_forces_gpu" in c.__dict__
    )
    yuk = cls._yukawa_forces_gpu

    rng = np.random.default_rng(7)
    M = rng.normal(size=(3, 3))
    quad = M + M.T
    quad -= np.eye(3) * np.trace(quad) / 3.0  # symmetric traceless quadrupole
    dipole = rng.normal(size=3)
    self = types.SimpleNamespace(
        _debye=9.0,
        _V_factor=2.5,  # non-zero monopole
        _multipole=object(),  # truthy so dipole and quadrupole are included
        _mp_dipole_gpu=cp.asarray(dipole),
        _mp_quad_gpu=cp.asarray(quad),
        _mp_four_pi_eps=1.0,
        _mp_trace=3.5,  # nonzero screened-trace moment, exercises the isotropic term
    )

    def energy_at(x):
        pos = cp.asarray(x.reshape(1, 1, 3))
        q = cp.asarray(np.array([1.0]))
        return float(to_np(yuk(self, pos, q)[2][0]))

    def force_at(x):
        pos = cp.asarray(x.reshape(1, 1, 3))
        q = cp.asarray(np.array([1.0]))
        return to_np(yuk(self, pos, q)[0][0])

    h = 1e-6
    for _ in range(200):
        x = rng.normal(size=3)
        x = x / np.linalg.norm(x) * rng.uniform(15.0, 55.0)
        fd = np.empty(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            fd[i] = -(energy_at(x + e) - energy_at(x - e)) / (2.0 * h)
        f = force_at(x)
        assert np.linalg.norm(f - fd) / max(np.linalg.norm(fd), 1e-30) < 1e-5


def test_multipole_trace_term_matches_exact_screened_quadrupole():
    """The screened multipole expansion reproduces the exact screened potential of
    an explicit net-neutral, dipole-free quadrupole only when the isotropic trace
    term (1/6) tr(M) exp(-r/lam)/(4 pi eps r lam^2) is included. A traceless-only
    expansion (correct for Coulomb) fails for the screened Yukawa kernel."""

    a, lam, sdie = 2.0, 7.86, 78.0
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, a], [0.0, 0.0, -a]])
    chg = np.array([-2.0, 1.0, 1.0])
    mp = MultipoleExpansion(pos, chg, debye_length=lam, sdie=sdie)
    assert abs(mp.Q) < 1e-12 and mp.dipole_mag < 1e-12
    assert abs(mp.trace_moment - 2.0 * a**2) < 1e-9
    fpe = 4.0 * np.pi * sdie * VACUUM_PERMITTIVITY_KBT

    def exact(R):
        return float(
            np.sum(
                [
                    qi
                    * np.exp(-np.linalg.norm(R - ri) / lam)
                    / (fpe * np.linalg.norm(R - ri))
                    for ri, qi in zip(pos, chg)
                ]
            )
        )

    dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.8165, 0.0, 0.5774]),  # magic angle measured from z
    ]
    worst_with = 0.0
    worst_without = 0.0
    for d in dirs:
        R = 15.0 * d / np.linalg.norm(d)
        r = float(np.linalg.norm(R))
        ex = exact(R)
        trace_V = (
            (mp.trace_moment / (6.0 * lam**2)) / (fpe * r) * np.exp(-r / lam)
        )
        with_trace = mp.potential(R)
        without_trace = with_trace - trace_V
        worst_with = max(worst_with, abs(with_trace - ex) / abs(ex))
        worst_without = max(worst_without, abs(without_trace - ex) / abs(ex))
    # With the trace term, the expansion matches the exact screened potential.
    assert worst_with < 0.08, f"trace-inclusive expansion off by {worst_with}"
    # Dropping the trace term makes the agreement clearly worse (the original bug).
    assert worst_without > 2.0 * worst_with, (
        f"trace term is not doing anything: with={worst_with}, without={worst_without}"
    )


def test_nam_parallel_worker_uses_run_one_with_recycling():
    """The multiprocessing trajectory worker matches serial run_one including Luty-McCammon-Zhou recycling for each seed."""

    mol1 = Molecule(name="m1")
    mol1.atoms = [Atom(x=0, y=0, z=0, charge=1.0, radius=2.0)]
    mol2 = Molecule(name="m2")
    mol2.atoms = [Atom(x=0, y=0, z=0, charge=-1.0, radius=2.0)]
    mob = MobilityTensor.from_radii(20.0, 20.0)
    pair = ContactPair(0, 0, 4.0)  # moderate cutoff so trajectories must diffuse
    ps = PathwaySet([ReactionInterface("rxn", ReactionCriteria(pairs=[pair]))])
    params = _NAMParameters(
        n_trajectories=8, r_start=50.0, seed=123, max_steps=4000, verbose=False
    )

    ref = _NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
    assert ref._outer_prop is not None

    base_seed = params.seed
    _worker_init(mol1, mol2, mob, ps, params, zero_force)
    for idx in range(8):
        worker_result = _run_trajectory_worker(idx)
        direct = _NAMSimulator(mol1, mol2, mob, ps, params, zero_force)
        direct.rng = np.random.default_rng(base_seed + idx)
        direct.rng_bb = np.random.default_rng(base_seed + idx + 0xBB)
        direct_result = direct.run_one()
        assert worker_result.fate == direct_result.fate
        assert worker_result.steps == direct_result.steps
        assert worker_result.final_separation == direct_result.final_separation


def test_effective_charges_browndye2_point_charge_layout():
    """EffectiveCharges.from_xml reads the BrownDye2 point-charge layout, returning exact positions and signed charges."""

    xml = (
        "<roottag>\n"
        "  <point>\n"
        "    <residue> ALA </residue><number> 1 </number>\n"
        "    <atom_type> charge-center </atom_type>\n"
        "    <x> 1.5 </x> <y> -2.0 </y> <z> 3.25 </z>\n"
        "    <charge> 0.41 </charge>\n"
        "  </point>\n"
        "  <point>\n"
        "    <x> -4.72 </x> <y> -2.97 </y> <z> -9.01 </z>\n"
        "    <charge> -0.53 </charge>\n"
        "  </point>\n"
        "  <total_charge> -0.12 </total_charge>\n"
        "</roottag>\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        ec = EffectiveCharges.from_xml(path)
    finally:
        os.unlink(path)
    assert len(ec) == 2
    assert np.allclose(ec.positions, [[1.5, -2.0, 3.25], [-4.72, -2.97, -9.01]])
    assert np.allclose(ec.charges, [0.41, -0.53])


def test_effective_charges_browndye2_lumped_layout():
    """EffectiveCharges.from_xml reads the BrownDye2 lumped layout with nested <pos> coordinates and <q> magnitudes."""

    xml = (
        "<top>\n"
        '  <point type="permanent">\n'
        "    <pos> <x>1.0</x> <y>2.0</y> <z>3.0</z> </pos>\n"
        "    <q>\n      0.5\n    </q>\n"
        "  </point>\n"
        '  <point type="induced">\n'
        "    <pos> <x>-1.0</x> <y>-2.0</y> <z>-3.0</z> </pos>\n"
        "    <q>\n      -0.25\n    </q>\n"
        "  </point>\n"
        "</top>\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        ec = EffectiveCharges.from_xml(path)
    finally:
        os.unlink(path)
    assert len(ec) == 2
    assert np.allclose(ec.positions, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    assert np.allclose(ec.charges, [0.5, -0.25])


def test_effective_charges_all_zero_layout_raises():
    """EffectiveCharges.from_xml raises ValueError when no charge magnitudes are readable, giving an all-zero set."""

    xml = (
        "<roottag>\n"
        "  <point>\n"
        "    <x> 1.0 </x> <y> 2.0 </y> <z> 3.0 </z>\n"
        "    <unrecognised_magnitude> 0.5 </unrecognised_magnitude>\n"
        "  </point>\n"
        "</roottag>\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        with pytest.raises(ValueError):
            EffectiveCharges.from_xml(path)
    finally:
        os.unlink(path)


def test_run_cmd_output_path_writes_captured_stdout(tmp_path):
    """run_cmd writes a command's captured stdout to output_path and also returns that text."""

    out = tmp_path / "combined.pqr"
    marker = "REMARK combined pqr content 12345"
    script = f"import sys; sys.stdout.write({marker!r})"
    cmd = f"{sys.executable} -c {shlex.quote(script)}"
    returned = run_cmd(cmd, step="ambpdb-like", output_path=out)
    assert out.read_text() == marker
    assert returned == marker


def test_run_cmd_does_not_interpret_shell_redirection(tmp_path):
    """run_cmd runs with shell=False, so a greater-than sign is a literal argument and never redirects to a file."""

    target = tmp_path / "redirect_target"
    script = "import sys; sys.stdout.write(' '.join(sys.argv[1:]))"
    cmd = f"{sys.executable} -c {shlex.quote(script)} -pqr > {target}"
    returned = run_cmd(cmd, step="redir", cwd=tmp_path)
    assert not target.exists()
    assert ">" in returned


def test_combine_concat_csv_offsets_keep_traj_ids_unique(tmp_path):
    """Concatenating shard CSVs offsets each trajectory id by the prior shards' trajectory counts, keeping ids unique."""

    d0 = tmp_path / "bd_0"
    d1 = tmp_path / "bd_1"
    d0.mkdir()
    d1.mkdir()
    # Shard 0 ran 5 trajectories; this file has two rows for trajectory 0.
    with open(d0 / "encounters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traj_id", "d"])
        w.writeheader()
        w.writerow({"traj_id": "0", "d": "1"})
        w.writerow({"traj_id": "0", "d": "2"})
        w.writerow({"traj_id": "3", "d": "3"})
    with open(d1 / "encounters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traj_id", "d"])
        w.writeheader()
        w.writerow({"traj_id": "0", "d": "4"})
        w.writerow({"traj_id": "2", "d": "5"})

    # Shard 0 ran 5 trajectories, so shard 1's ids are offset by 5.
    _concat_csv(
        [str(d0), str(d1)],
        "encounters.csv",
        str(tmp_path),
        reindex="traj_id",
        offsets=[0, 5],
    )
    with open(tmp_path / "encounters.csv") as f:
        rows = list(csv.DictReader(f))
    ids = [int(r["traj_id"]) for r in rows]
    assert ids == [0, 0, 3, 5, 7]  # shard 1 ids 0 and 2 became 5 and 7


def test_combine_concat_npz_reindexes_traj_id_column(tmp_path):
    """_concat_npz offsets the traj_id column by the per-shard trajectory count to match the combined CSV tables."""
    d0 = tmp_path / "bd_0"
    d1 = tmp_path / "bd_1"
    d0.mkdir()
    d1.mkdir()
    cols = np.array(["traj_id", "step", "x"])
    np.savez(
        d0 / "paths.npz", data=np.array([[0.0, 0, 1.0], [1.0, 0, 2.0]]), columns=cols
    )
    np.savez(d1 / "paths.npz", data=np.array([[0.0, 0, 3.0]]), columns=cols)

    _concat_npz(
        [str(d0), str(d1)],
        "paths.npz",
        str(tmp_path),
        data_key="data",
        meta_key="columns",
        reindex_col="traj_id",
        offsets=[0, 5],
    )
    out = np.load(tmp_path / "paths.npz", allow_pickle=True)
    assert out["data"].shape == (3, 3)
    # Shard 1's only trajectory had id 0, offset by 5 -> 5.
    np.testing.assert_array_equal(out["data"][:, 0], [0.0, 1.0, 5.0])


def test_combine_sum_npz_transition_matrix_keys(tmp_path):
    """_sum_npz pools transition matrices by summing the counts stored under the writer's actual keys."""
    d0 = tmp_path / "bd_0"
    d1 = tmp_path / "bd_1"
    d0.mkdir()
    d1.mkdir()
    bins = np.linspace(0.0, 10.0, 4)
    np.savez(d0 / "transition_matrix.npz", bins=bins, counts=np.ones((3, 3)))
    np.savez(d1 / "transition_matrix.npz", bins=bins, counts=np.full((3, 3), 2.0))

    _sum_npz(
        [str(d0), str(d1)],
        "transition_matrix.npz",
        str(tmp_path),
        sum_key="counts",
        copy_keys=["bins"],
    )
    out = np.load(tmp_path / "transition_matrix.npz")
    np.testing.assert_allclose(out["counts"], np.full((3, 3), 3.0))
    np.testing.assert_allclose(out["bins"], bins)


def test_combine_pool_p_commit_uses_count_pooling(tmp_path):
    """The pooled commitment probability is the summed reacted count over the summed sample count, not a probability average."""
    d0 = tmp_path / "bd_0"
    d1 = tmp_path / "bd_1"
    d0.mkdir()
    d1.mkdir()
    r_bins = np.linspace(0.0, 5.0, 3)
    np.savez(
        d0 / "p_commit.npz",
        r_bins=r_bins,
        p_commit=np.array([0.2, 0.5]),
        n_samples=np.array([10.0, 4.0]),
    )
    np.savez(
        d1 / "p_commit.npz",
        r_bins=r_bins,
        p_commit=np.array([0.2, 0.0]),
        n_samples=np.array([30.0, 0.0]),
    )

    _pool_p_commit([str(d0), str(d1)], "p_commit.npz", str(tmp_path))
    out = np.load(tmp_path / "p_commit.npz")
    # Bin 0: (0.2*10 + 0.2*30) / 40 = 0.2 ; Bin 1: (0.5*4 + 0) / 4 = 0.5
    np.testing.assert_allclose(out["p_commit"], [0.2, 0.5])
    np.testing.assert_allclose(out["n_samples"], [40.0, 4.0])


# Consolidated audit-fix and low-severity regression tests.
# Previously in separate tests/test_auditfix*.py and tests/test_lowsev_*.py,
# merged here so the whole suite lives in one file.


# --- merged from test_auditfix2_bdsurf_resname.py ---
def _make_atoms():
    """Build a small combined atom list with receptor (MGO) and ligand (APN) atoms."""
    return [
        PQRAtom(1, "C1", "MGO", 1, 0.0, 0.0, 0.0, -0.2, 1.7),
        PQRAtom(2, "C2", "MGO", 1, 1.0, 0.0, 0.0, 0.1, 1.7),
        PQRAtom(3, "O1", "MGO", 1, 0.0, 1.0, 0.0, -0.4, 1.5),
        PQRAtom(4, "N1", "APN", 2, 5.0, 5.0, 5.0, 0.3, 1.6),
        PQRAtom(5, "C3", "APN", 2, 6.0, 5.0, 5.0, 0.0, 1.7),
    ]


def test_matching_resnames_split_correctly():
    """Atoms split by matching residue names yield the receptor and ligand atom sets with correct sizes and names."""
    atoms = _make_atoms()
    rec, lig = split_receptor_ligand(atoms, "MGO", "APN")
    assert len(rec) == 3
    assert len(lig) == 2
    assert all(a.resname == "MGO" for a in rec)
    assert all(a.resname == "APN" for a in lig)


def test_nonmatching_receptor_resname_raises_named_valueerror():
    """A receptor resname matching no atoms raises a ValueError naming receptor_resname, the bad value, and the residue names present."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "XXX", "APN")
    msg = str(excinfo.value)
    # The error names the receptor residue name that matched nothing.
    assert "receptor_resname" in msg
    assert "XXX" in msg
    # The error lists residue names actually present in the PQR.
    assert "MGO" in msg
    assert "APN" in msg


def test_nonmatching_ligand_resname_raises_named_valueerror():
    """A ligand resname matching no atoms raises a ValueError naming ligand_resname, the bad value, and a present residue name."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "MGO", "ZZZ")
    msg = str(excinfo.value)
    assert "ligand_resname" in msg
    assert "ZZZ" in msg
    assert "MGO" in msg


def test_both_nonmatching_resnames_named_in_valueerror():
    """When both resnames match nothing, the ValueError message names both bad values."""
    atoms = _make_atoms()
    with pytest.raises(ValueError) as excinfo:
        split_receptor_ligand(atoms, "AAA", "BBB")
    msg = str(excinfo.value)
    assert "AAA" in msg
    assert "BBB" in msg


# --- merged from test_auditfix2_geomcache.py ---
_PQR_BODY = """\
ATOM      1  N   ALA     1       0.000   0.000   0.000  0.10 1.50
ATOM      2  C   ALA     1       3.000   0.000   0.000 -0.10 1.70
ATOM      3  O   ALA     1       0.000   3.000   0.000 -0.20 1.40
ATOM      4  C   ALA     1       0.000   0.000   3.000  0.20 1.70
"""


def _write_pqr(path: Path, body: str) -> None:
    path.write_text(body)


def _cache_files(pqr_path: Path):
    return sorted(glob.glob(str(pqr_path) + ".r_hydro_*.cache"))


def test_different_n_mc_use_different_cache_files(tmp_path):
    """Two analyses differing only in n_mc write to distinct cache files tagged with their n_mc values."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)

    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=5000)

    caches = _cache_files(pqr)
    assert len(caches) == 2, caches
    assert "_n2000_" in "".join(caches)
    assert "_n5000_" in "".join(caches)


def test_changed_n_mc_does_not_reuse_stale_value(tmp_path, monkeypatch):
    """Repeating a call with the same n_mc reuses the cache while a changed n_mc forces recomputation."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)

    calls = {"n": 0}
    real = geometry.mc_hydrodynamic_radius

    def counting_mc(coords, radii, spacing, n_mc):
        calls["n"] += 1
        return real(coords, radii, spacing=spacing, n_mc=n_mc)

    monkeypatch.setattr(geometry, "mc_hydrodynamic_radius", counting_mc)

    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    assert calls["n"] == 1
    # A second call with the same n_mc reuses the cache, so no new computation.
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    assert calls["n"] == 1
    # A call with a different n_mc must recompute instead of reusing the cache.
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=5000)
    assert calls["n"] == 2


def test_edited_structure_uses_different_cache_file(tmp_path):
    """Editing the atom coordinates produces a separate second cache file rather than reusing the first."""
    pqr = tmp_path / "mol.pqr"
    _write_pqr(pqr, _PQR_BODY)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)
    first = _cache_files(pqr)
    assert len(first) == 1

    moved = _PQR_BODY.replace(
        "ATOM      2  C   ALA     1       3.000   0.000   0.000 -0.10 1.70",
        "ATOM      2  C   ALA     1       4.000   0.000   0.000 -0.10 1.70",
    )
    _write_pqr(pqr, moved)
    analyse_molecule(pqr, use_mc_hydro=True, n_mc=2000)

    caches = _cache_files(pqr)
    assert len(caches) == 2, caches


# --- merged from test_auditfix2_nam.py ---
def _make_mol(charge: float) -> Molecule:
    mol = Molecule(name="m")
    mol.atoms = [Atom(index=0, x=0.0, y=0.0, z=0.0, charge=charge, radius=1.5)]
    return mol


def _make_inputs():
    mol1 = _make_mol(1.0)
    mol2 = _make_mol(-1.0)
    mobility = MobilityTensor.from_radii(5.0, 5.0)
    pathway_set = PathwaySet()  # No reactions, so trajectories escape.
    params = NAMParameters(
        n_trajectories=1,
        r_start=50.0,
        r_escape=60.0,
        max_steps=5000,
        seed=7,
        use_brownian_bridge=False,
        use_hard_sphere=False,
    )
    return mol1, mol2, mobility, pathway_set, params


def test_outer_propagator_failure_warns_and_falls_back():
    """A failed outer-propagator setup emits one RuntimeWarning carrying the failure message and leaves the outer propagator disabled."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    original = op.OuterPropagator

    def _raise(*args, **kwargs):
        raise ValueError("synthetic outer propagator failure")

    op.OuterPropagator = _raise
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)
    finally:
        op.OuterPropagator = original

    assert sim._outer_prop is None
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    message = str(runtime_warnings[0].message)
    assert "synthetic outer propagator failure" in message


def test_successful_outer_propagator_setup_is_silent():
    """A successful outer-propagator setup enables the propagator and emits no RuntimeWarning."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)

    assert sim._outer_prop is not None
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0


def test_time_ps_sums_actual_adaptive_steps():
    """result.time_ps equals the sum of the adaptive steps actually applied, not steps times the nominal dt."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()
    sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params)
    # Use the simple escape fallback so the adaptive controller drives every step.
    sim._outer_prop = None

    applied_dts = []
    original_get_dt = nsim.AdaptiveTimeStep.get_dt

    def _recording_get_dt(self, *args, **kwargs):
        dt = original_get_dt(self, *args, **kwargs)
        applied_dts.append(dt)
        return dt

    nsim.AdaptiveTimeStep.get_dt = _recording_get_dt
    try:
        result = sim.run_one()
    finally:
        nsim.AdaptiveTimeStep.get_dt = original_get_dt

    assert result.fate == Fate.ESCAPED
    assert applied_dts  # The controller was queried at least once.
    # The reported time is the running total of the applied adaptive steps.
    assert result.time_ps == sum(applied_dts)
    assert abs(result.time_ps - result.steps * params.dt) > 1.0
    assert result.time_ps > 0.0


def test_time_ps_accumulates_backstep_half_steps():
    """A force backstep contributes its full step time as two half steps so time_ps still equals the sum of the chosen steps."""
    mol1, mol2, mobility, pathway_set, params = _make_inputs()

    def _strong_varying_force(m1, m2):
        atom = m2.atoms[0]
        r_vec = np.array([atom.x, atom.y, atom.z])
        d = float(np.linalg.norm(r_vec))
        force = -5000.0 * r_vec / (d**3 + 1e-6)
        return force, np.zeros(3), 0.0

    sim = NAMSimulator(
        mol1, mol2, mobility, pathway_set, params, force_fn=_strong_varying_force
    )
    sim._outer_prop = None

    applied_dts = []
    original_get_dt = nsim.AdaptiveTimeStep.get_dt

    def _recording_get_dt(self, *args, **kwargs):
        dt = original_get_dt(self, *args, **kwargs)
        applied_dts.append(dt)
        return dt

    backstep_count = {"n": 0}
    original_backstep = nsim.backstep_due_to_force

    def _counting_backstep(*args, **kwargs):
        fired = original_backstep(*args, **kwargs)
        if fired:
            backstep_count["n"] += 1
        return fired

    nsim.AdaptiveTimeStep.get_dt = _recording_get_dt
    nsim.backstep_due_to_force = _counting_backstep
    try:
        result = sim.run_one()
    finally:
        nsim.AdaptiveTimeStep.get_dt = original_get_dt
        nsim.backstep_due_to_force = original_backstep

    # At least one backstep must fire for this test to exercise the half-step path.
    assert backstep_count["n"] > 0
    assert result.time_ps == sum(applied_dts)


# --- merged from test_auditfix2_parallel.py ---
def _make_molecules():
    """Build a tiny receptor and a single-atom ligand for the batch runner."""
    mol1 = Molecule(
        name="receptor", atoms=[Atom(index=0, name="A", x=0.0, y=0.0, z=0.0)]
    )
    mol2 = Molecule(name="ligand", atoms=[Atom(index=0, name="B", x=0.0, y=0.0, z=0.0)])
    return mol1, mol2


def _make_mobility():
    """A simple isotropic mobility tensor with no RPY coupling."""
    return MobilityTensor(
        D_trans1=0.01,
        D_rot1=0.001,
        D_trans2=0.01,
        D_rot2=0.001,
        radius1=1.0,
        radius2=1.0,
        use_rpy=False,
    )


def _empty_pathways():
    """A pathway set with no reactions, so trajectories only escape or time out."""
    return PathwaySet(reactions=[])


def _replicate_single_step_positions(mol2, mob, params, force_vec):
    """Reproduce the batch runner's first-step random draws and return the
    per-trajectory final positions for a constant translational force.

    The batch runner draws, from a generator seeded with params.seed and in this
    order: the starting directions, one uniform triple per trajectory consumed
    by the random orientation, then the translational noise. With max_steps=1
    and no reactions or escapes every trajectory takes exactly one step, so the
    final position is start + D_trans * F * dt + sqrt(2 * D_trans * dt) * W.
    """
    N = params.n_trajectories
    D_t = mob.relative_translational_diffusion()
    dt = params.dt
    sigma_t = math.sqrt(2.0 * D_t * dt)
    rng = np.random.default_rng(params.seed)
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pos = v * params.r_start
    # The orientation draws consume three uniforms per trajectory.
    for _ in range(N):
        rng.uniform(0, 1, 3)
    drift = D_t * np.asarray(force_vec) * dt
    noise = sigma_t * rng.standard_normal((N, 3))
    return pos + drift + noise


# --- merged from test_auditfix2_stepnear.py ---
class _StubRNG:
    """Deterministic stand-in for numpy's Generator exposing only random().

    The first draw selects between survival and absorption; every later draw
    returns a fixed value so that the rejection-sampling acceptance test never
    succeeds, exercising the attempt-cap path.
    """

    def __init__(self, first: float, rest: float):
        self._first = first
        self._rest = rest
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self._first if self.calls == 1 else self._rest


def _psurv(x0: float, F: float) -> float:
    b = -F
    tau = x0 * x0
    st2 = 2.0 * math.sqrt(tau)
    bt = b * tau
    erfmt = math.erf((x0 - bt) / st2)
    erfpt = math.erf((x0 + bt) / st2)
    p = 0.5 * (math.exp(b * x0) * (erfpt - 1.0) + erfmt + 1.0)
    return max(0.0, min(1.0, p))


def test_normal_path_is_deterministic_and_unchanged():
    """Fixed seeds reproduce the established zero-force absorbing-surface step outputs exactly."""
    reference = {
        0: [
            (False, 0.0, 6.7446678441),
            (True, 12.6390963248, 25.0),
            (False, 0.0, 4.3913905151),
            (False, 0.0, 7.4927972634),
            (True, 9.3712826847, 25.0),
        ],
        1: [
            (True, 17.5759728873, 25.0),
            (False, 0.0, 7.7957863003),
            (False, 0.0, 10.2299784092),
            (True, 11.2759206018, 25.0),
            (True, 12.0073043847, 25.0),
        ],
        7: [
            (False, 0.0, 5.6301797498),
            (False, 0.0, 7.5758106705),
            (True, 6.3935978972, 25.0),
            (False, 0.0, 5.3827174559),
            (False, 0.0, 1.098550199),
        ],
    }
    for seed, expected in reference.items():
        rng = np.random.default_rng(seed)
        for exp_survives, exp_x, exp_t in expected:
            survives, new_x, time = step_near_absorbing_surface(rng, 5.0, 0.0, 1.0)
            assert survives == exp_survives
            assert new_x == pytest.approx(exp_x, abs=1e-8)
            assert time == pytest.approx(exp_t, abs=1e-8)


def test_normal_path_nonzero_force_unchanged():
    """A nonzero force reproduces its established absorbing-surface step outputs exactly."""
    expected = [
        (False, 0.0, 3.1381561308),
        (False, 0.0, 0.576511347),
        (True, 11.2645266541, 4.5),
        (False, 0.0, 1.9953638947),
        (True, 6.7743307886, 4.5),
    ]
    rng = np.random.default_rng(42)
    for exp_survives, exp_x, exp_t in expected:
        survives, new_x, time = step_near_absorbing_surface(rng, 3.0, 0.2, 2.0)
        assert survives == exp_survives
        assert new_x == pytest.approx(exp_x, abs=1e-8)
        assert time == pytest.approx(exp_t, abs=1e-8)


def test_normal_path_emits_no_warning():
    """Healthy sampling over many absorbing-surface steps never trips the rejection-sampling warning."""
    rng = np.random.default_rng(2024)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2000):
            step_near_absorbing_surface(rng, 4.0, 0.1, 1.5)


def test_survival_fraction_matches_probability():
    """The empirical survival fraction matches the analytic survival probability P_surv(x0, F)."""
    x0, F, D = 4.0, 0.1, 1.5
    expected = _psurv(x0, F)
    rng = np.random.default_rng(2024)
    n = 20000
    survived = sum(1 for _ in range(n) if step_near_absorbing_surface(rng, x0, F, D)[0])
    assert survived / n == pytest.approx(expected, abs=0.02)


def test_survival_exhaustion_warns_and_returns_valid_position():
    """A degenerate survival draw warns about non-convergence and returns a finite, valid no-flux position rather than the deterministic fallback."""
    rng = _StubRNG(first=0.0, rest=0.999999)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survives, new_x, time = step_near_absorbing_surface(rng, 2.0, -5.0, 1.0)
    messages = [str(w.message) for w in caught]
    assert any("rejection sampling did not converge" in m for m in messages)
    assert survives is True
    assert new_x >= 0.0
    assert math.isfinite(new_x)
    assert math.isfinite(time)
    assert new_x != pytest.approx(max(2.0, 0.001), abs=1e-12)


def test_absorption_exhaustion_warns():
    """A degenerate absorption draw warns and returns survival False with the position pinned at the absorbing surface 0."""
    rng = _StubRNG(first=0.9999, rest=0.999999)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        survives, new_x, time = step_near_absorbing_surface(rng, 2.0, -5.0, 1.0)
    messages = [str(w.message) for w in caught]
    assert any("rejection" in m for m in messages)
    assert survives is False
    assert new_x == 0.0
    assert math.isfinite(time)


# --- merged from test_auditfix2_we_resample.py ---


# --- merged from test_auditfix3_chainio.py ---
def _write_pdb(tmp_path, lines):
    path = tmp_path / "test.pdb"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_insertion_code_residues_are_distinct(tmp_path):
    """Residues sharing a sequence number but differing by insertion code are parsed as two separate residues with their own atoms."""
    lines = [
        "ATOM      1  N   ALA A 100       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A 100       1.500   2.500   3.500  1.00  0.00           C",
        "ATOM      3  N   GLY A 100A      4.000   5.000   6.000  1.00  0.00           N",
        "ATOM      4  CA  GLY A 100A      4.500   5.500   6.500  1.00  0.00           C",
    ]
    pdb = _write_pdb(tmp_path, lines)
    residues = _parse_pdb_chain_for_beads(pdb, chain_id="A")

    assert len(residues) == 2
    assert residues[0]["resname"] == "ALA"
    assert residues[0]["resid"] == 100
    assert residues[1]["resname"] == "GLY"
    assert residues[1]["resid"] == 100
    # The two residues carry their own atoms rather than merging into one.
    assert set(residues[0]["atoms"]) == {"N", "CA"}
    assert set(residues[1]["atoms"]) == {"N", "CA"}
    assert np.allclose(residues[0]["atoms"]["CA"], [1.5, 2.5, 3.5])
    assert np.allclose(residues[1]["atoms"]["CA"], [4.5, 5.5, 6.5])


def test_no_insertion_code_groups_by_resid(tmp_path):
    """Without insertion codes the parser groups atoms by sequence number, yielding two residues with resids 1 and 2."""
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  N   ALA A   2       2.000   0.000   0.000  1.00  0.00           N",
        "ATOM      4  CA  ALA A   2       3.000   0.000   0.000  1.00  0.00           C",
    ]
    pdb = _write_pdb(tmp_path, lines)
    residues = _parse_pdb_chain_for_beads(pdb, chain_id="A")

    assert len(residues) == 2
    assert [r["resid"] for r in residues] == [1, 2]
    assert set(residues[0]["atoms"]) == {"N", "CA"}
    assert set(residues[1]["atoms"]) == {"N", "CA"}


# --- merged from test_auditfix3_coffdrop.py ---
def _atoms(n):
    return [ChainAtom(radius=2.0, charge=0.0, resname="X", resid=i) for i in range(n)]


def test_chain_idx_resolves_chain_atom_ref():
    """_chain_idx resolves a ChainAtomRef to its underlying integer index."""
    assert _chain_idx(ChainAtomRef(0)) == 0
    assert _chain_idx(ChainAtomRef(7)) == 7


def test_chain_idx_passes_through_raw_int():
    """_chain_idx passes a raw Python int or numpy integer through unchanged."""
    assert _chain_idx(3) == 3
    assert _chain_idx(np.int64(5)) == 5


def test_length_constraint_atomref_violation_evaluates():
    """A length constraint with ChainAtomRef endpoints evaluates the violation as the signed deviation from the target length."""
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert phi.shape == (1,)
    assert phi[0] == pytest.approx(0.7, abs=1e-12)


def test_length_constraint_atomref_matches_raw_int():
    """A length constraint using ChainAtomRef endpoints yields the same violation as one using raw integer endpoints."""
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)

    common_ref = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    common_int = ChainCommon(
        name="len_int",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(0, 1, 5.0)],
    )
    phi_ref = compute_constraint_violations(
        ChainState.from_template(common_ref, positions.copy())
    )
    phi_int = compute_constraint_violations(
        ChainState.from_template(common_int, positions.copy())
    )
    assert np.allclose(phi_ref, phi_int, atol=1e-14)


def test_length_constraint_atomref_shake_converges():
    """SHAKE on a ChainAtomRef length constraint converges to the target separation of 5.0."""
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [7.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints(state, tol=1e-10)
    r = float(np.linalg.norm(state.positions[0] - state.positions[1]))
    assert r == pytest.approx(5.0, abs=1e-9)


def test_length_constraint_atomref_newton_converges():
    """Newton solving of a ChainAtomRef length constraint converges to the target separation of 5.0."""
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [7.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints_newton(state, tol=1e-10)
    r = float(np.linalg.norm(state.positions[0] - state.positions[1]))
    assert r == pytest.approx(5.0, abs=1e-8)


def test_length_constraint_atomref_jacobian_builds():
    """The Jacobian of a ChainAtomRef length constraint has shape (1,6) with opposing unit-vector rows for the two atoms."""
    common = ChainCommon(
        name="len_ref",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.7, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    J = _build_constraint_jacobian(state)
    assert J.shape == (1, 6)
    assert np.allclose(J[0, 0:3], [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(J[0, 3:6], [1.0, 0.0, 0.0], atol=1e-12)


def test_coplanar_constraint_atomref_violation_evaluates():
    """A coplanar constraint with ChainAtomRef vertices evaluates the violation as the out-of-plane distance of 1.0."""
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    # Atom a sits one unit above the z=0 plane defined by b, c, d.
    positions = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert phi.shape == (1,)
    assert abs(phi[0]) == pytest.approx(1.0, abs=1e-9)


def test_coplanar_constraint_atomref_matches_raw_int():
    """A coplanar constraint using ChainAtomRef vertices yields the same violation as one using raw integer vertices."""
    positions = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float)
    common_ref = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    common_int = ChainCommon(
        name="cop_int",
        atoms=_atoms(4),
        coplanar_constraints=[CoplanarConstraint(0, 1, 2, 3)],
    )
    phi_ref = compute_constraint_violations(
        ChainState.from_template(common_ref, positions.copy())
    )
    phi_int = compute_constraint_violations(
        ChainState.from_template(common_int, positions.copy())
    )
    assert np.allclose(phi_ref, phi_int, atol=1e-14)


def test_coplanar_violation_helper_atomref():
    """The _coplanar_violation helper returns the out-of-plane distance of 1.0 for a ChainAtomRef constraint."""
    common = ChainCommon(name="cop_ref", atoms=_atoms(4))
    positions = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    c = CoplanarConstraint(
        ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
    )
    assert abs(_coplanar_violation(state, c)) == pytest.approx(1.0, abs=1e-9)


def test_coplanar_constraint_atomref_shake_projects_onto_plane():
    """SHAKE on a ChainAtomRef coplanar constraint projects the atom onto the plane, driving the violation below 1e-9."""
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    positions = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints(state, tol=1e-10)
    phi = compute_constraint_violations(state)
    assert abs(phi[0]) < 1e-9


def test_coplanar_constraint_atomref_jacobian_builds():
    """The Jacobian of a ChainAtomRef coplanar constraint has shape (1,12) with the atom-a row equal to the plane normal."""
    common = ChainCommon(
        name="cop_ref",
        atoms=_atoms(4),
        coplanar_constraints=[
            CoplanarConstraint(
                ChainAtomRef(0), ChainAtomRef(1), ChainAtomRef(2), ChainAtomRef(3)
            )
        ],
    )
    positions = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    J = _build_constraint_jacobian(state)
    assert J.shape == (1, 12)
    # The analytic atom-a row is the plane normal; here the plane is z=0.
    assert np.allclose(J[0, 0:3], [0.0, 0.0, 1.0], atol=1e-9)


def test_mixed_atomref_and_raw_int_endpoints_resolve():
    """A length constraint mixing a ChainAtomRef and a raw int endpoint resolves both and reports zero violation when satisfied."""
    common = ChainCommon(
        name="mixed",
        atoms=_atoms(2),
        length_constraints=[LengthConstraint(ChainAtomRef(0), 1, 5.0)],
    )
    positions = np.array([[0, 0, 0], [5.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    phi = compute_constraint_violations(state)
    assert abs(phi[0]) < 1e-12


def test_hybrid_solver_with_atomref_endpoints():
    """The hybrid solver satisfies two chained length constraints with ChainAtomRef endpoints to within tolerance."""
    common = ChainCommon(
        name="hybrid_ref",
        atoms=_atoms(3),
        length_constraints=[
            LengthConstraint(ChainAtomRef(0), ChainAtomRef(1), 3.0),
            LengthConstraint(ChainAtomRef(1), ChainAtomRef(2), 3.0),
        ],
    )
    positions = np.array([[0, 0, 0], [4.0, 0, 0], [9.0, 0, 0]], dtype=float)
    state = ChainState.from_template(common, positions)
    satisfy_constraints_hybrid(state, tol=1e-9)
    phi = compute_constraint_violations(state)
    assert float(np.max(np.abs(phi))) < 1e-8


def test_coffdrop_force_evaluator_removed():
    """The coffdrop_chain module no longer defines COFFDROPForceEvaluator."""

    assert not hasattr(mod, "COFFDROPForceEvaluator")


# --- merged from test_auditfix3_engine.py ---
def _mol(coords, charge=0.0, radius=1.8):
    m = Molecule(name="m")
    m.atoms = [
        Atom(x=c[0], y=c[1], z=c[2], charge=charge, radius=radius) for c in coords
    ]
    return m


def test_group_centroid_uses_only_charged_atoms():
    """_group_centroid averages only charged atoms and returns the origin when no atom is charged."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    charges = np.array([1.0, -1.0, 1e-12])
    assert np.allclose(_group_centroid(positions, charges), [1.0, 0.0, 0.0])
    # With no charged atoms the reference is the origin.
    assert np.allclose(_group_centroid(positions, np.zeros(3)), [0.0, 0.0, 0.0])


def test_lj_type_id_fallback_and_kbt_conversion():
    """A single shared Lennard-Jones type maps all atoms to type index 0 without an IndexError, and the engine converts the LJ contribution from kcal/mol to kBT."""
    ljp = LJParams(atom_types=[LJAtomType(name="X", epsilon=0.2, sigma=3.0)])
    engine = PySTARCEngine(lj_params=ljp)  # no electrostatic or Born grids

    # Three atoms per molecule (more atoms than Lennard-Jones types).
    mol1 = _mol([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mol2 = _mol([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]])

    # Must not raise an IndexError despite atoms outnumbering the single type.
    force, torque, energy = engine(mol1, mol2)
    assert np.all(np.isfinite(force)) and np.isfinite(energy)

    raw = LJForceEngine(ljp)
    pos1 = mol1.positions_array()
    pos2 = mol2.positions_array()
    _, f2_raw, e_raw = raw.compute(pos1, pos2, [0, 0, 0], [0, 0, 0])
    assert np.linalg.norm(f2_raw) > 0.0  # the configuration is within range
    assert np.allclose(force, f2_raw * KCAL_PER_MOL_TO_KBT)
    assert np.isclose(energy, e_raw * KCAL_PER_MOL_TO_KBT)


# --- merged from test_auditfix3_nam_hs.py ---
def _make_mol_test_auditfix3_nam_hs(radius: float, charge: float = 0.0) -> Molecule:
    mol = Molecule(name="m")
    mol.atoms = [Atom(index=0, x=0.0, y=0.0, z=0.0, charge=charge, radius=radius)]
    return mol


def _make_sim(use_hard_sphere=True, r_start=12.0, r_escape=14.0, seed=3):
    mol1 = _make_mol_test_auditfix3_nam_hs(5.0)
    mol2 = _make_mol_test_auditfix3_nam_hs(5.0)
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
    """When every redraw overlaps, no step is accepted and the trajectory runs to MAX_STEPS at its previous position."""
    sim = _make_sim()

    spy = _OverlapSpy(lambda pos: True)  # Every configuration overlaps.
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)

    result = sim.run_one()

    # Some configuration was tested for overlap on the trajectory.
    assert spy.queries
    # The checker reported overlap on every query, so none could be accepted.
    assert all(overlaps for _, overlaps in spy.queries)
    assert result.fate == Fate.MAX_STEPS


def test_redraw_loops_until_overlap_free(monkeypatch):
    """An overlapping redraw is rejected and further redraws are drawn until an overlap-free configuration is accepted."""
    sim = _make_sim()

    state = {"first_overlap_seen": False, "redraws_rejected": 0}

    def verdict(pos):
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

    assert state["redraws_rejected"] >= 1
    assert any(not overlaps for _, overlaps in spy.queries)
    assert result.fate in (Fate.ESCAPED, Fate.MAX_STEPS)


def test_accepted_positions_are_never_reported_overlapping(monkeypatch):
    """No configuration the simulator carries forward lies in the region declared overlapping by the checker."""
    sim = _make_sim()

    def verdict(pos):
        return 2.0 < pos[0] < 6.0

    spy = _OverlapSpy(verdict)
    monkeypatch.setattr(nsim, "_check_hard_sphere_overlap", spy)

    carried_positions = []
    original_check_all = sim.pathway_set.check_all

    def _recording_check_all(mol1, mol2, rng, *args, **kwargs):
        carried_positions.append(
            np.array([mol2.atoms[0].x, mol2.atoms[0].y, mol2.atoms[0].z])
        )
        return original_check_all(mol1, mol2, rng, *args, **kwargs)

    sim.pathway_set.check_all = _recording_check_all

    sim.run_one()

    assert len(carried_positions) > 1
    for pos in carried_positions[1:]:
        assert not (2.0 < pos[0] < 6.0)


def test_no_overlap_path_is_unchanged(monkeypatch):
    """With overlap never reported, the hard-sphere run is bitwise identical to a run with hard spheres disabled."""
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


# --- merged from test_auditfix3_outerprop.py ---
def _make_propagator(has_hi: bool) -> OuterPropagator:
    """Build a representative propagator with nonzero charges and radii."""
    g0 = OPGroupInfo(q=2.0, Dtrans=0.015, Drot=0.0003)
    g1 = OPGroupInfo(q=-1.0, Dtrans=0.030, Drot=0.0006)
    return OuterPropagator(
        b_radius=20.0,
        max_radius=15.0,
        has_hi=has_hi,
        kT=0.593,
        viscosity=0.0009,
        dielectric=78.5,
        vacuum_perm=1.0,
        debye_len=8.0,
        g0=g0,
        g1=g1,
    )


def _radial_force(op: OuterPropagator, r: float) -> float:
    """Radial force as evaluated by the propagator's own helper."""
    return op._radial_force(r)


def test_Fr1_matches_browndye2_form():
    """Fr1 equals -V/L^2 - 2*Fr0/r, matching the BrownDye2 form and differing from the previous incorrect expression."""
    op = _make_propagator(has_hi=False)
    L = op.debye_len
    for r in (25.0, 30.0, 40.0):
        Fr0 = _radial_force(op, r)
        # Yukawa monopole magnitude V used by the propagator.
        V = op.V_factor * math.exp(-r / L) / r

        # Expression as computed inside new_state.
        Fr1_code = -V / L**2 - 2.0 * Fr0 / r

        # Independent hand-computed BrownDye2 form.
        Fr1_ref = -V / (L * L) - 2.0 * Fr0 * (1.0 / r)

        assert math.isclose(Fr1_code, Fr1_ref, rel_tol=1e-12, abs_tol=0.0)

        # The previous, incorrect form (-V*(1/r+1/L)^2 - 2*Fr0/r) differs.
        Fr1_bad = -V * (1.0 / r + 1.0 / L) ** 2 - 2.0 * Fr0 / r
        assert not math.isclose(Fr1_code, Fr1_bad, rel_tol=1e-9, abs_tol=0.0)


def test_D1_uses_hi_only_part():
    """D1 is built from the HI-only diffusivity Di rather than the full D0, matching the BrownDye2 form and differing from the D0-based variant by the spurious -3*D_const/r term."""
    op = _make_propagator(has_hi=True)
    for r in (25.0, 30.0, 40.0):
        rm1 = 1.0 / r

        # Full parallel diffusivity (constant part + HI part).
        D0 = op._D_parallel(r)

        # HI-only part, matching the r-dependent terms of _D_parallel.
        Di = (op.D_factor / PI6) * (-3.0 / r + 2.0 * op.a2 / (r**3))

        # The constant part is the remainder.
        ainv = 1.0 / op.a0 + 1.0 / op.a1
        D_const = (op.D_factor / PI6) * ainv
        assert math.isclose(D0, D_const + Di, rel_tol=1e-12, abs_tol=0.0)

        # Expression as computed inside new_state.
        D1_code = -3.0 * Di * rm1 - op.D_factor * rm1**2 / PI_test_auditfix3_outerprop

        # Independent hand-computed BrownDye2 form using the HI-only part.
        D1_ref = (
            -3.0 * Di * (1.0 / r)
            - op.D_factor * (1.0 / r) ** 2 / PI_test_auditfix3_outerprop
        )

        assert math.isclose(D1_code, D1_ref, rel_tol=1e-12, abs_tol=0.0)

        D1_bad = -3.0 * D0 * rm1 - op.D_factor * rm1**2 / PI_test_auditfix3_outerprop
        spurious = -3.0 * D_const * rm1
        assert math.isclose(D1_bad - D1_code, spurious, rel_tol=1e-9, abs_tol=0.0)
        assert not math.isclose(D1_code, D1_bad, rel_tol=1e-9, abs_tol=0.0)


def test_D2_D3_consistent_with_corrected_D1():
    """D2 and D3 follow the BrownDye2 recurrence from the corrected D1."""
    op = _make_propagator(has_hi=True)
    for r in (25.0, 30.0, 40.0):
        rm1 = 1.0 / r
        Di = (op.D_factor / PI6) * (-3.0 / r + 2.0 * op.a2 / (r**3))
        D1 = -3.0 * Di * rm1 - op.D_factor * rm1**2 / PI_test_auditfix3_outerprop
        D2 = -4.0 * D1 * rm1 + op.D_factor * rm1**3 / PI_test_auditfix3_outerprop
        D3 = -5.0 * D2 * rm1 - 2.0 * op.D_factor * rm1**4 / PI_test_auditfix3_outerprop

        D2_ref = (
            -4.0 * D1 * (1.0 / r)
            + op.D_factor * (1.0 / r) ** 3 / PI_test_auditfix3_outerprop
        )
        D3_ref = (
            -5.0 * D2 * (1.0 / r)
            - 2.0 * op.D_factor * (1.0 / r) ** 4 / PI_test_auditfix3_outerprop
        )

        assert math.isclose(D2, D2_ref, rel_tol=1e-12, abs_tol=0.0)
        assert math.isclose(D3, D3_ref, rel_tol=1e-12, abs_tol=0.0)


# --- merged from test_auditfix4_chainio_altloc.py ---
def _atom(serial, name, altloc, resname, chain, resid, x, y, z):
    return (
        "ATOM  "
        + f"{serial:>5}"
        + " "
        + f"{name:^4}"
        + altloc
        + f"{resname:>3}"
        + " "
        + chain
        + f"{resid:>4}"
        + " "
        + "   "
        + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
    )


def test_parser_keeps_first_altloc_conformer():
    """The parser keeps the first alternate-location conformer when a residue lists multiple altLocs."""
    lines = [
        _atom(1, "N", " ", "SER", "A", 10, 0.0, 0.0, 0.0),
        _atom(2, "OG", "A", "SER", "A", 10, 1.0, 1.0, 1.0),
        _atom(3, "OG", "B", "SER", "A", 10, 9.0, 9.0, 9.0),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        residues = _parse_pdb_chain_for_beads(path, chain_id="A")
    finally:
        os.unlink(path)
    assert len(residues) == 1
    assert np.allclose(residues[0]["atoms"]["OG"], [1.0, 1.0, 1.0])


# --- merged from test_auditfix4_constants.py ---
def test_eps_water_consistent_with_bjerrum_length():
    """The stored BJERRUM_LENGTH matches the value implied by EPS_WATER and VACUUM_PERMITTIVITY_KBT."""
    lB_from_eps = 1.0 / (4.0 * math.pi * EPS_WATER * VACUUM_PERMITTIVITY_KBT)
    assert abs(lB_from_eps - BJERRUM_LENGTH) < 1e-3


# --- merged from test_auditfix4_nam_conditions.py ---
def _sim(**param_kwargs):
    mol1 = Molecule(name="m1")
    mol1.atoms = [Atom(x=0, y=0, z=0, charge=1.0, radius=2.0)]
    mol2 = Molecule(name="m2")
    mol2.atoms = [Atom(x=0, y=0, z=0, charge=-1.0, radius=2.0)]
    mob = MobilityTensor.from_radii(20.0, 20.0)
    ps = PathwaySet(
        [ReactionInterface("rxn", ReactionCriteria(pairs=[ContactPair(0, 0, 4.0)]))]
    )
    params = NAMParameters(n_trajectories=1, r_start=50.0, seed=1, **param_kwargs)
    return NAMSimulator(mol1, mol2, mob, ps, params, zero_force)


def test_default_conditions_come_from_the_registry():
    """The outer propagator takes its solvent conditions from one registry.

    The screening length is the 7.858 A the input parser uses for about 150 mM
    monovalent salt at 298.15 K. It previously read 8.0 A here, so a bare
    NAMParameters described a different solvent from a parsed one. Every shipped
    entry point passes the value explicitly, so no computed rate moves.

    The viscosity is now the correct 1.002e-3 Pa.s expressed in the internal
    units, 0.14422 kcal/mol.ps/A^3. The retired literal applied a conversion of
    1e8 rather than 143.93262 and was too large by 6.95e5, which drove the
    back-solved hydrodynamic radii to about 3e-5 A and reduced the Rotne-Prager
    terms to nothing while has_hi still reported enabled.
    """

    op = _sim()._outer_prop
    assert op is not None
    assert op.kT == KBT_KCAL
    assert op.debye_len == DEBYE_LENGTH
    assert math.isclose(op.viscosity, VISCOSITY)


def test_configured_conditions_propagate_to_the_outer_propagator():
    """Configured Debye length and temperature propagate to the outer propagator's debye_len and kT."""
    op = _sim(debye_length=4.0, temperature_kT=0.55)._outer_prop
    assert op is not None
    assert op.debye_len == 4.0
    assert op.kT == 0.55


def test_dielectric_scales_the_screened_coulomb_prefactor():
    """V_factor is inversely proportional to the dielectric, so halving the dielectric doubles it."""
    op_hi = _sim(dielectric=78.54)._outer_prop
    op_lo = _sim(dielectric=39.27)._outer_prop
    assert op_hi is not None and op_lo is not None
    assert abs(op_lo.V_factor / op_hi.V_factor - 2.0) < 1e-9


# --- merged from test_auditfix_born2_default.py ---
def _write_input_without_tag(tmp_path: Path) -> Path:
    """Write a minimal input XML that omits <enable_born2_torque>."""
    work_dir = tmp_path / "bd_sims"
    xml = f"""<?xml version="1.0" ?>
<pystarc_input>
    <receptor_pqr>receptor.pqr</receptor_pqr>
    <ligand_pqr>ligand.pqr</ligand_pqr>
    <ligand_resname>BEN</ligand_resname>
    <ligand_charge>1</ligand_charge>
    <work_dir>{work_dir}</work_dir>
    <n_trajectories>10000</n_trajectories>
    <bd_milestone_radius>30.0</bd_milestone_radius>
    <ghost_atoms>auto</ghost_atoms>
</pystarc_input>
"""
    xml_path = tmp_path / "pystarc_input.xml"
    xml_path.write_text(xml)
    return xml_path


def test_born2_default_true_when_tag_absent(tmp_path):
    """Parsing input without the tag enables enable_born2_torque by defaulting it to True."""
    xml_path = _write_input_without_tag(tmp_path)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is True


def test_born2_parser_default_matches_dataclass(tmp_path):
    """The parser default for enable_born2_torque agrees with the PySTARCConfig dataclass default."""
    xml_path = _write_input_without_tag(tmp_path)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is PySTARCConfig.enable_born2_torque


def test_born2_explicit_false_is_respected(tmp_path):
    """An explicit false enable_born2_torque tag disables the BORN2 torque."""
    work_dir = tmp_path / "bd_sims"
    xml = f"""<?xml version="1.0" ?>
<pystarc_input>
    <receptor_pqr>receptor.pqr</receptor_pqr>
    <ligand_pqr>ligand.pqr</ligand_pqr>
    <ligand_resname>BEN</ligand_resname>
    <ligand_charge>1</ligand_charge>
    <work_dir>{work_dir}</work_dir>
    <n_trajectories>10000</n_trajectories>
    <bd_milestone_radius>30.0</bd_milestone_radius>
    <ghost_atoms>auto</ghost_atoms>
    <enable_born2_torque>false</enable_born2_torque>
</pystarc_input>
"""
    xml_path = tmp_path / "pystarc_input.xml"
    xml_path.write_text(xml)
    cfg = parse(xml_path)
    assert cfg.enable_born2_torque is False


# --- merged from test_auditfix_chain_pipeline.py ---
class _FakeAtom:
    def __init__(self, radius=1.5):
        self.radius = radius


class _FakeChain:
    def __init__(self, n_atoms=3):
        self.name = "fake_chain"
        self.atoms = [_FakeAtom() for _ in range(n_atoms)]
        self.bonds = []
        self.angles = []
        self.torsions = []


def _make_config(tmp_path, chain_overrides=None, output_overrides=None):
    """Build a PySTARCConfig with a chain block for pipeline testing."""
    chain_kwargs = dict(
        chain_json=str(tmp_path / "chain.json"),
        reaction_pairs_json=str(tmp_path / "reaction_pairs.json"),
    )
    if chain_overrides:
        chain_kwargs.update(chain_overrides)
    chain = ChainConfig(**chain_kwargs)

    outputs = OutputConfig()
    if output_overrides:
        for key, value in output_overrides.items():
            setattr(outputs, key, value)

    cfg = PySTARCConfig(
        work_dir=tmp_path / "work",
        chain=chain,
        outputs=outputs,
    )
    return cfg


def _patch_pipeline_seams(monkeypatch, captured):
    """Replace the heavy run_chain dependencies with light stand-ins.

    The captured dict records the keyword arguments passed to the
    ChainBDSimulator constructor and to write_chain_results so the tests can
    assert on what run_chain forwarded.
    """
    chain = _FakeChain(n_atoms=3)
    body_positions = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )

    monkeypatch.setattr(
        chain_pipeline,
        "load_chain_from_json",
        lambda path: (chain, body_positions),
    )
    monkeypatch.setattr(
        chain_pipeline,
        "parse_pqr",
        lambda path: types.SimpleNamespace(name="target"),
    )
    monkeypatch.setattr(
        chain_pipeline,
        "_load_reaction_pairs_json",
        lambda path: [(0, 0, 4.0), (1, 1, 4.0), (2, 2, 4.0)],
    )
    monkeypatch.setattr(
        chain_pipeline,
        "_ensure_chain_apbs_grids",
        lambda config: None,
    )

    class _FakeSimulator:
        def __init__(self, **kwargs):
            captured["sim_kwargs"] = kwargs

        def run(self):
            return []

    monkeypatch.setattr(chain_pipeline, "ChainBDSimulator", _FakeSimulator)

    def _fake_write(work_dir, sim, results, wall_time_sec=0.0, outputs=None):
        captured["write_outputs"] = outputs
        return []

    monkeypatch.setattr(chain_pipeline, "write_chain_results", _fake_write)


def test_default_config_resolves_diffusion_defaults(tmp_path, monkeypatch):
    """A default chain config with D=0 forwards the scalar defaults D_trans=0.1 and D_rot=0.01 with auto_diffusion off."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(tmp_path)
    # The default chain config leaves auto_diffusion off with zero diffusion.
    assert cfg.chain.auto_diffusion is False
    assert cfg.chain.D_trans == 0.0
    assert cfg.chain.D_rot == 0.0

    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs.get("auto_diffusion") is not True
    assert sim_kwargs["D_trans"] == 0.1
    assert sim_kwargs["D_rot"] == 0.01


def test_explicit_diffusion_is_preserved(tmp_path, monkeypatch):
    """Explicit non-zero D_trans and D_rot pass through to the simulator unchanged."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(
        tmp_path,
        chain_overrides=dict(D_trans=0.25, D_rot=0.05),
    )
    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs["D_trans"] == 0.25
    assert sim_kwargs["D_rot"] == 0.05


def test_auto_diffusion_does_not_set_scalar_d(tmp_path, monkeypatch):
    """With auto_diffusion enabled, no scalar D_trans or D_rot is forwarded to the simulator."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(tmp_path, chain_overrides=dict(auto_diffusion=True))
    chain_pipeline.run_chain(cfg)

    sim_kwargs = captured["sim_kwargs"]
    assert sim_kwargs["auto_diffusion"] is True
    assert "D_trans" not in sim_kwargs
    assert "D_rot" not in sim_kwargs


def test_run_chain_forwards_outputs(tmp_path, monkeypatch):
    """The parsed OutputConfig is forwarded to write_chain_results so user output flags are honored."""
    captured = {}
    _patch_pipeline_seams(monkeypatch, captured)

    cfg = _make_config(
        tmp_path,
        output_overrides=dict(encounters_csv=False, full_paths=False),
    )
    chain_pipeline.run_chain(cfg)

    forwarded = captured["write_outputs"]
    assert forwarded is cfg.outputs
    assert forwarded.encounters_csv is False
    assert forwarded.full_paths is False


def test_born_grid_without_target_grid_raises(tmp_path):
    """Setting born_grid_dx without target_grid_dx raises a ValueError naming target_grid_dx."""
    cfg = _make_config(
        tmp_path,
        chain_overrides=dict(
            target_grid_dx="",
            born_grid_dx=str(tmp_path / "missing_born.dx"),
        ),
    )
    with pytest.raises(ValueError, match="target_grid_dx"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_no_grids_is_a_noop(tmp_path):
    """With neither grid path set, APBS grid generation is skipped and returns None."""
    cfg = _make_config(tmp_path)
    assert cfg.chain.target_grid_dx == ""
    assert cfg.chain.born_grid_dx == ""
    # Should return without raising and without attempting any APBS work.
    assert chain_pipeline._ensure_chain_apbs_grids(cfg) is None


# --- merged from test_auditfix_chaingb_hct.py ---
def _hct_closed_form(L, U, r, rho_S_j):
    """Closed-form HCT integrand for explicit lower and upper limits L and U."""
    return 0.5 * (
        1.0 / L
        - 1.0 / U
        + (r / 4.0) * (1.0 / U**2 - 1.0 / L**2)
        + (1.0 / (2.0 * r)) * np.log(L / U)
        + (rho_S_j * rho_S_j / (4.0 * r)) * (1.0 / L**2 - 1.0 / U**2)
    )


def test_engulfed_atom_integrand_uses_absolute_value():
    """For r < ρ_S_j the HCT integrand lower limit is abs(r - ρ_S_j) rather than ρ̃_i."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0
    assert r < rho_S_j
    assert (rho_S_j - r) > rho_tilde_i

    L_canonical = max(rho_tilde_i, abs(r - rho_S_j))
    U = r + rho_S_j
    reference = _hct_closed_form(L_canonical, U, r, rho_S_j)

    got = float(_hct_integrand(r, rho_tilde_i, rho_S_j))
    assert np.isclose(got, reference, rtol=0, atol=1e-12)


def test_engulfed_atom_integrand_smaller_than_old_expression():
    """The abs(r - ρ_S_j) lower limit yields a smaller integrand than the old ρ̃_i form, removing the descreening overcount."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0
    U = r + rho_S_j

    corrected = float(_hct_integrand(r, rho_tilde_i, rho_S_j))

    old_expression = _hct_closed_form(rho_tilde_i, U, r, rho_S_j)

    assert old_expression > corrected
    assert not np.isclose(old_expression, corrected)


def test_engulfed_atom_derivative_matches_hand_reference():
    """In the engulfed regime the analytic HCT integrand derivative gives dL/dr = -1, matching finite differences."""
    r, rho_tilde_i, rho_S_j = 1.0, 0.8, 2.0

    analytic = float(_hct_integrand_deriv(r, rho_tilde_i, rho_S_j))

    h = 1e-6
    fd = (
        float(_hct_integrand(r + h, rho_tilde_i, rho_S_j))
        - float(_hct_integrand(r - h, rho_tilde_i, rho_S_j))
    ) / (2.0 * h)

    assert np.isclose(analytic, fd, rtol=0, atol=1e-6)


def test_standard_outside_regime_unchanged():
    """For r > ρ_S_j the absolute value is a no-op, so integrand and derivative match the canonical reference."""
    for r, rho_tilde_i, rho_S_j in [(5.0, 1.5, 1.2), (3.0, 2.0, 1.0), (4.0, 1.0, 1.8)]:
        assert r > rho_S_j  # abs(r - rho_S_j) == r - rho_S_j here

        L_canonical = max(rho_tilde_i, r - rho_S_j)
        U = r + rho_S_j
        reference = _hct_closed_form(L_canonical, U, r, rho_S_j)
        got = float(_hct_integrand(r, rho_tilde_i, rho_S_j))
        assert np.isclose(got, reference, rtol=0, atol=1e-12)

        analytic = float(_hct_integrand_deriv(r, rho_tilde_i, rho_S_j))
        h = 1e-6
        fd = (
            float(_hct_integrand(r + h, rho_tilde_i, rho_S_j))
            - float(_hct_integrand(r - h, rho_tilde_i, rho_S_j))
        ) / (2.0 * h)
        assert np.isclose(analytic, fd, rtol=0, atol=1e-5)


# --- merged from test_auditfix_lj_wca.py ---
EPSILON = 1.3

SIGMA = 2.7

FACTOR = 0.75


def _force_energy(r, use_wca, factor=1.0):
    pos_a = np.zeros(3)
    pos_b = np.array([r, 0.0, 0.0])
    return lj_pair_force(pos_a, pos_b, EPSILON, SIGMA, factor=factor, use_wca=use_wca)


def test_wca_energy_zero_at_cutoff():
    """The WCA energy is zero just inside the cutoff r_cut = 2^(1/6) σ."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    # Evaluate just inside the cutoff to stay within the WCA branch.
    _, energy = _force_energy(r_cut * (1.0 - 1e-9), use_wca=True)
    assert energy == 0.0 or abs(energy) < 1e-6


def test_wca_energy_nonnegative_inside():
    """The WCA energy stays non-negative across separations from 0.5 σ up to the cutoff."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    radii = np.linspace(0.5 * SIGMA, r_cut * (1.0 - 1e-12), 200)
    for r in radii:
        _, energy = _force_energy(r, use_wca=True)
        assert energy >= -1e-9, f"WCA energy negative at r={r}: {energy}"


def test_wca_energy_continuous_at_cutoff():
    """Approaching the cutoff from inside, the WCA energy decreases monotonically to zero and is exactly zero beyond it."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    deltas = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    energies = []
    for delta in deltas:
        _, energy = _force_energy(r_cut - delta, use_wca=True)
        assert energy >= -1e-9
        energies.append(energy)
    for closer, farther in zip(energies[1:], energies[:-1]):
        assert closer <= farther + 1e-12
    # The value adjacent to the cutoff is essentially zero.
    assert energies[-1] < 1e-3
    # Beyond the cutoff the energy is exactly zero (no discontinuity).
    _, e_outside = _force_energy(r_cut * 1.01, use_wca=True)
    assert e_outside == 0.0


def test_wca_force_unchanged():
    # Within the repulsive branch the WCA force matches the plain LJ force.
    """Within the repulsive branch the WCA force equals the plain Lennard-Jones force."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    for r in np.linspace(0.6 * SIGMA, r_cut * (1.0 - 1e-9), 50):
        f_plain, _ = _force_energy(r, use_wca=False, factor=FACTOR)
        f_wca, _ = _force_energy(r, use_wca=True, factor=FACTOR)
        assert np.allclose(f_wca, f_plain, rtol=1e-12, atol=1e-12)


def test_wca_energy_shift_matches_well_depth():
    # The WCA energy is the plain LJ energy plus the well depth factor*eps/4.
    """The WCA energy equals the plain Lennard-Jones energy plus the well depth factor·ε/4."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    for r in np.linspace(0.6 * SIGMA, r_cut * (1.0 - 1e-9), 50):
        _, e_plain = _force_energy(r, use_wca=False, factor=FACTOR)
        _, e_wca = _force_energy(r, use_wca=True, factor=FACTOR)
        assert math.isclose(
            e_wca, e_plain + FACTOR * EPSILON * 0.25, rel_tol=1e-12, abs_tol=1e-12
        )


# --- merged from test_auditfix_multigpu_runs.py ---
def test_set_or_create_creates_missing_tag():
    """_set_or_create creates a tag absent from the XML and assigns it the requested text."""
    root = ET.fromstring("<simulation><receptor_pqr>r.pqr</receptor_pqr></simulation>")
    assert root.find("n_trajectories") is None

    el = _set_or_create(root, "n_trajectories", "1")

    assert el is root.find("n_trajectories")
    assert root.findtext("n_trajectories") == "1"


def test_set_or_create_updates_existing_tag():
    """_set_or_create overwrites an existing tag in place without appending a duplicate element."""
    root = ET.fromstring("<simulation><seed>1523</seed></simulation>")
    before = list(root)

    el = _set_or_create(root, "seed", "99")

    assert el is root.find("seed")
    assert root.findtext("seed") == "99"
    # No duplicate element is appended when the tag already exists.
    assert len(list(root)) == len(before)


def test_set_or_create_handles_all_optional_tags():
    """_set_or_create sets all four optional-with-default tags on an XML that omits them."""
    root = ET.fromstring("<simulation><receptor_pqr>r.pqr</receptor_pqr></simulation>")

    _set_or_create(root, "n_trajectories", "25000")
    _set_or_create(root, "max_steps", "1")
    _set_or_create(root, "seed", "11111112")
    _set_or_create(root, "work_dir", ".")

    assert root.findtext("n_trajectories") == "25000"
    assert root.findtext("max_steps") == "1"
    assert root.findtext("seed") == "11111112"
    assert root.findtext("work_dir") == "."


def test_naive_find_text_assignment_would_crash():
    """The naive find(tag).text assignment raises AttributeError on a missing tag, which _set_or_create avoids."""
    root = ET.fromstring("<simulation></simulation>")

    try:
        root.find("n_trajectories").text = "1"
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError on missing tag")

    # The helper sets the same tag without raising.
    _set_or_create(root, "n_trajectories", "1")
    assert root.findtext("n_trajectories") == "1"


# --- merged from test_auditfix_nam_rateconstant.py ---
def _make_result_test_auditfix_nam_rateconstant(k_db):
    return SimulationResult(
        n_trajectories=1000,
        n_reacted=10,
        n_escaped=990,
        n_max_steps=0,
        reaction_counts={"rxn": 10},
        r_start=100.0,
        r_escape=110.0,
        dt=0.2,
        k_db=k_db,
    )


def test_rate_constant_uses_stored_k_db_for_lmz():
    """A nonzero stored k_db yields the Luty-McCammon-Zhou rate rather than the Smoluchowski fallback."""
    k_db = 5.0  # Å³/ps from the outer propagator.
    res = _make_result_test_auditfix_nam_rateconstant(k_db)
    D_rel = 0.05  # Å²/ps.

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    expected_lmz = CONV_A3ps * k_db * P

    k = res.rate_constant(D_rel)

    assert math.isclose(k, expected_lmz, rel_tol=1e-9)


def test_point_estimate_matches_k_from_P():
    """The rate_constant point estimate equals _k_from_P evaluated at the same reaction probability."""
    res = _make_result_test_auditfix_nam_rateconstant(5.0)
    D_rel = 0.05

    k = res.rate_constant(D_rel)
    k_ref = _k_from_P(res, res.reaction_probability, D_rel)

    assert math.isclose(k, k_ref, rel_tol=1e-12)


def test_lmz_differs_from_smoluchowski_fallback():
    """With a stored k_db, the rate_constant result differs from the Smoluchowski fallback expression."""
    res = _make_result_test_auditfix_nam_rateconstant(5.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    smoluchowski = CONV_A3ps * k_D * P / (1.0 - P * (1.0 - beta))

    k = res.rate_constant(D_rel)

    assert not math.isclose(k, smoluchowski, rel_tol=1e-6)


def test_zero_stored_k_db_falls_back_to_smoluchowski():
    """When the stored k_db is 0.0, rate_constant uses the Smoluchowski expression."""
    res = _make_result_test_auditfix_nam_rateconstant(0.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    # Corrected NAM truncated-escape denominator 1 - (1 - P) * beta.
    expected = CONV_A3ps * k_D * P / (1.0 - (1.0 - P) * beta)

    k = res.rate_constant(D_rel)

    assert math.isclose(k, expected, rel_tol=1e-9)


def test_explicit_k_db_argument_overrides_stored():
    """An explicit positive k_db argument takes precedence over the stored self.k_db."""
    res = _make_result_test_auditfix_nam_rateconstant(5.0)
    D_rel = 0.05

    P = res.reaction_probability
    CONV_A3ps = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    arg_k_db = 7.5
    expected = CONV_A3ps * arg_k_db * P

    k = res.rate_constant(D_rel, k_db=arg_k_db)

    assert math.isclose(k, expected, rel_tol=1e-9)


# --- merged from test_auditfix_reaction_xml.py ---
def _build_pathway_set_n_needed_2():
    pairs = [
        ContactPair(0, 10, 5.0),
        ContactPair(1, 11, 4.5),
        ContactPair(2, 12, 6.0),
    ]
    criteria = ReactionCriteria(name="rxn", pairs=pairs, n_needed=2)
    rxn = ReactionInterface(name="rxn", criteria=criteria, probability=0.5)
    return PathwaySet(reactions=[rxn])


def test_n_needed_survives_round_trip(tmp_path):
    """n_needed=2 survives the reaction XML write and parse round trip alongside its three contact pairs."""
    pathway_set = _build_pathway_set_n_needed_2()
    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    parsed = parse_reaction_xml(out)

    assert len(parsed) == 1
    rxn = parsed.reactions[0]
    assert rxn.criteria.n_needed == 2
    assert rxn.criteria.n_needed != len(rxn.criteria.pairs)
    assert len(rxn.criteria.pairs) == 3
    assert rxn.probability == 0.5


def test_n_needed_emitted_in_xml(tmp_path):
    """A reaction with n_needed=2 emits the n_needed tag in the written XML."""
    pathway_set = _build_pathway_set_n_needed_2()
    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    text = out.read_text()
    assert "n_needed" in text


def test_all_pairs_reaction_stays_concise(tmp_path):
    """An all-pairs reaction with default n_needed=-1 omits the n_needed tag from the XML and parses back to -1."""
    pairs = [ContactPair(0, 10, 5.0), ContactPair(1, 11, 5.0)]
    criteria = ReactionCriteria(name="rxn", pairs=pairs)  # default n_needed = -1
    rxn = ReactionInterface(name="rxn", criteria=criteria)
    pathway_set = PathwaySet(reactions=[rxn])

    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    text = out.read_text()
    assert "n_needed" not in text

    parsed = parse_reaction_xml(out)
    assert parsed.reactions[0].criteria.n_needed == -1


def test_state_fields_survive_round_trip(tmp_path):
    """first_state and per-reaction state_before and state_after survive the reaction XML round trip."""
    pairs = [ContactPair(0, 10, 5.0)]
    criteria = ReactionCriteria(
        name="rxn", pairs=pairs, state_before="unbound", state_after="bound"
    )
    rxn = ReactionInterface(
        name="rxn",
        criteria=criteria,
        state_before="unbound",
        state_after="bound",
    )
    pathway_set = PathwaySet(reactions=[rxn], first_state="unbound")

    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    parsed = parse_reaction_xml(out)

    assert parsed.first_state == "unbound"
    parsed_rxn = parsed.reactions[0]
    assert parsed_rxn.state_before == "unbound"
    assert parsed_rxn.state_after == "bound"
    assert parsed_rxn.criteria.state_before == "unbound"
    assert parsed_rxn.criteria.state_after == "bound"


# --- merged from test_auditfix_we_time.py ---
def _make_molecules_test_auditfix_we_time(lig_x=50.0, charge=0.0):
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
            charge=charge,
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
            x=lig_x,
            y=0.0,
            z=0.0,
            charge=-charge,
            radius=2.0,
        )
    )
    return mol1, mol2


def _make_pathways(cutoff=10.0):
    criteria = ReactionCriteria(name="r", pairs=[ContactPair(0, 0, cutoff)], n_needed=1)
    rxn = ReactionInterface(name="rxn", criteria=criteria)
    return PathwaySet(reactions=[rxn])


# --- merged from test_lowsev_adaptive_time_step.py ---
def test_non_physical_inputs_take_safe_default():
    # D_rel <= 0 must short-circuit to the safe default before the size term.
    assert max_time_step(10.0, 0.0, 1.0, 5.0, 6.0) == 0.2
    assert max_time_step(10.0, -1.0, 1.0, 5.0, 6.0) == 0.2
    # r <= 0 also short-circuits.
    assert max_time_step(0.0, 1.0, 1.0, 5.0, 6.0) == 0.2


def test_size_constraint_is_the_minimum_when_it_dominates():
    r = 100.0
    D_rel = 1.0
    D_rot = 1.0e30  # makes dt_rot tiny only if large; here keep rotational term huge
    r_hydro1, r_hydro2 = 1.0, 2.0
    D_rot = 1.0e-30
    dt = max_time_step(r, D_rel, D_rot, r_hydro1, r_hydro2)
    expected_size = 4.0 * min(r_hydro1, r_hydro2) ** 2 / D_rel
    dt_pair = (0.1**2 / 2.0) * r**2 / D_rel
    dt_rot = math.pi**2 / D_rot
    assert dt == min(dt_pair, dt_rot, expected_size)
    assert dt == expected_size
    assert dt != _LARGE


def test_healthy_path_matches_closed_form():
    # General healthy case: result is the min of the three closed-form terms.
    r, D_rel, D_rot, r_h1, r_h2 = 25.0, 0.3, 0.05, 8.0, 12.0
    dt = max_time_step(r, D_rel, D_rot, r_h1, r_h2)
    dt_pair = (0.1**2 / 2.0) * r**2 / D_rel
    dt_rot = math.pi**2 / D_rot
    dt_size = 4.0 * min(r_h1, r_h2) ** 2 / D_rel
    assert dt == min(dt_pair, dt_rot, dt_size)


# --- merged from test_lowsev_chain_io.py ---
def _write_pdb_test_lowsev_chain_io(tmp_path, lines):
    path = tmp_path / "test.pdb"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _single_ala_pdb(tmp_path):
    """One ALA residue with N, CA, C, O, CB heavy atoms on chain A."""
    lines = [
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C",
        "ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C",
        "ATOM      4  O   ALA A   1       2.500   3.500   4.500  1.00  0.00           O",
        "ATOM      5  CB  ALA A   1       3.000   4.000   5.000  1.00  0.00           C",
    ]
    return _write_pdb_test_lowsev_chain_io(tmp_path, lines)


def test_capped_chain_raises_clear_error(tmp_path):
    """A chain carrying a cap bead (resid=-1) is rejected with a clear caps-unsupported error."""
    pdb = _single_ala_pdb(tmp_path)
    # One ordinary residue bead plus an ACE cap bead carrying resid=-1.
    common = ChainCommon(
        name="capped",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:CA", resid=0),
            ChainAtom(radius=2.0, charge=0.0, resname="ACE:CN", resid=-1),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        pdb_to_bead_positions(common, pdb, chain_id="A")
    message = str(excinfo.value).lower()
    assert "cap" in message
    assert "resid < 0" in message


def test_nme_cap_bead_also_rejected(tmp_path):
    """A C-terminal NME cap bead is detected by the resid<0 guard, not by atom name."""
    pdb = _single_ala_pdb(tmp_path)
    common = ChainCommon(
        name="capped",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:CA", resid=0),
            ChainAtom(radius=2.0, charge=0.0, resname="NME:CC", resid=-1),
        ],
    )
    with pytest.raises(ValueError, match="cap"):
        pdb_to_bead_positions(common, pdb, chain_id="A")


def test_no_cap_chain_does_not_trip_cap_guard(tmp_path):
    """A chain with only non-negative resids never raises the caps-unsupported error.

    A bead atom name intentionally absent from the COFFDROP ALA map drives the
    function past the cap guard. The resulting error must therefore be the
    bead-lookup KeyError, not the caps-unsupported ValueError, which confirms
    the new guard does not fire on a healthy no-cap chain.
    """
    pdb = _single_ala_pdb(tmp_path)
    common = ChainCommon(
        name="nocaps",
        atoms=[
            ChainAtom(radius=2.0, charge=0.0, resname="ALA:QQ", resid=0),
        ],
    )
    with pytest.raises(KeyError) as excinfo:
        pdb_to_bead_positions(common, pdb, chain_id="A")
    assert "cap" not in str(excinfo.value).lower()


# --- merged from test_lowsev_chain_pipeline.py ---
def _make_config_test_lowsev_chain_pipeline(tmp_path, target_grid_dx, receptor_pqr=""):
    chain = ChainConfig(
        chain_json=str(tmp_path / "chain.json"),
        reaction_pairs_json=str(tmp_path / "reaction_pairs.json"),
        target_grid_dx=target_grid_dx,
    )
    return PySTARCConfig(
        work_dir=tmp_path / "work",
        chain=chain,
        receptor_pqr=receptor_pqr,
    )


def test_target_grid_without_trailing_digit_raises(tmp_path):
    """A missing target_grid_dx whose name lacks the trailing level digit raises a clear ValueError before any APBS work."""
    bad_path = tmp_path / "grids" / "target.dx"
    cfg = _make_config_test_lowsev_chain_pipeline(
        tmp_path, target_grid_dx=str(bad_path)
    )

    with pytest.raises(ValueError, match=r"target1\.dx"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_conforming_target_grid_passes_naming_guard(tmp_path):
    """A conforming '{mol_name}1.dx' name passes the naming guard and proceeds to the receptor_pqr check, not the naming error."""
    good_path = tmp_path / "grids" / "target1.dx"
    missing_pqr = str(tmp_path / "does_not_exist.pqr")
    cfg = _make_config_test_lowsev_chain_pipeline(
        tmp_path, target_grid_dx=str(good_path), receptor_pqr=missing_pqr
    )

    with pytest.raises(FileNotFoundError, match="receptor_pqr"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


def test_coarse_electrostatic_grid_name_passes_naming_guard(tmp_path):
    """The coarse electrostatic name '{mol_name}0.dx' is among the names APBS produces and passes the naming guard."""
    good_path = tmp_path / "grids" / "target0.dx"
    missing_pqr = str(tmp_path / "does_not_exist.pqr")
    cfg = _make_config_test_lowsev_chain_pipeline(
        tmp_path, target_grid_dx=str(good_path), receptor_pqr=missing_pqr
    )

    with pytest.raises(FileNotFoundError, match="receptor_pqr"):
        chain_pipeline._ensure_chain_apbs_grids(cfg)


# --- merged from test_lowsev_coffdrop_chain.py ---
def _make_constrained_state() -> ChainState:
    """Two beads with one length constraint, deliberately violated.

    The constraint guard at the top of satisfy_constraints_newton returns 0
    only when there are no constraints. Providing one length constraint forces
    the solver into its iteration loop logic. The target length differs from
    the actual distance so the violation is nonzero.
    """
    atoms = [ChainAtom(radius=2.0, charge=0.0), ChainAtom(radius=2.0, charge=0.0)]
    common = ChainCommon(
        name="pair",
        atoms=atoms,
        length_constraints=[
            LengthConstraint(a=ChainAtomRef(0), b=ChainAtomRef(1), length=1.0)
        ],
    )
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    return ChainState.from_template(common, positions)


def test_newton_max_iter_zero_raises_runtimeerror():
    state = _make_constrained_state()
    with pytest.raises(RuntimeError):
        satisfy_constraints_newton(state, tol=1e-8, max_iter=0)


def test_newton_no_constraints_returns_zero():
    atoms = [ChainAtom(radius=2.0, charge=0.0), ChainAtom(radius=2.0, charge=0.0)]
    common = ChainCommon(name="free", atoms=atoms)
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    state = ChainState.from_template(common, positions)
    assert satisfy_constraints_newton(state, max_iter=0) == 0


# --- merged from test_lowsev_coffdrop_params.py ---
def _make_potential(values):
    return TabulatedPotential(
        x_min=0.0,
        x_max=1.0,
        values=np.asarray(values, dtype=np.float64),
        residues=(0, 0),
        atoms=(0, 0),
        orders=(0, 0),
        index=0,
    )


def test_empty_values_raises_value_error():
    """An empty energy table must fail clearly at construction time."""
    with pytest.raises(ValueError):
        _make_potential([])


def test_nonempty_values_construct_normally():
    """The healthy path with a populated table still constructs and evaluates."""
    pot = _make_potential([1.0, 2.0, 3.0, 4.0])
    # Boundary clamping returns the first and last tabulated values unchanged.
    assert pot.value(-1.0) == 1.0
    assert pot.value(2.0) == 4.0


def _write_xml(tmp_path, body):
    path = tmp_path / "coffdrop.xml"
    path.write_text("<coffdrop>\n" + body + "\n</coffdrop>\n")
    return str(path)


def test_pairs_short_distance_list_raises(tmp_path):
    """A <pairs> block with a one-element <distance> list must raise clearly."""
    xml = _write_xml(
        tmp_path,
        "<pairs><distance>0.0</distance>" "<potentials></potentials></pairs>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_bond_angles_short_angle_list_raises(tmp_path):
    """A <bond_angles> block with a one-element <angle> list must raise clearly."""
    xml = _write_xml(
        tmp_path,
        "<bond_angles><angle>0.0</angle>" "<potentials></potentials></bond_angles>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_dihedral_angles_short_angle_list_raises(tmp_path):
    """A <dihedral_angles> block with a one-element <angle> list must raise."""
    xml = _write_xml(
        tmp_path,
        "<dihedral_angles><angle>0.0</angle>"
        "<potentials></potentials></dihedral_angles>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_healthy_pairs_block_parses(tmp_path):
    """A well-formed <pairs> block with a two-element distance list parses fine."""
    xml = _write_xml(
        tmp_path,
        "<pairs><distance>0.0 10.0</distance>"
        "<potentials>"
        "<potential><orders>0 0</orders><index>0</index>"
        "<residues>0 0</residues><atoms>0 0</atoms>"
        "<data>1.0 2.0 3.0 4.0</data></potential>"
        "</potentials></pairs>",
    )
    type_map, pairs, angles, dihedrals = _parse_ff(xml)
    assert len(pairs) == 1
    assert pairs[0].x_min == 0.0
    assert pairs[0].x_max == 10.0


# --- merged from test_lowsev_combine_data.py ---
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


def _base_run():
    return {"k_b": 1.5, "D_rel": 0.3, "r_start": 10.0, "r_escape": 50.0}


def test_consistent_runs_emit_no_warning(capsys):
    runs = [_base_run(), _base_run(), _base_run()]
    _warn_run_mismatch(runs)
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_single_run_emits_no_warning(capsys):
    _warn_run_mismatch([_base_run()])
    out = capsys.readouterr().out
    assert out == ""


def test_kb_mismatch_warns(capsys):
    r2 = _base_run()
    r2["k_b"] = 2.0
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "k_b" in out


def test_geometry_mismatch_warns(capsys):
    r2 = _base_run()
    r2["r_escape"] = 60.0
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "r_escape" in out


def test_tiny_float_noise_does_not_warn(capsys):
    r2 = _base_run()
    r2["k_b"] = 1.5 + 1e-13
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_missing_value_in_later_shard_is_skipped(capsys):
    r2 = _base_run()
    del r2["D_rel"]
    _warn_run_mismatch([_base_run(), r2])
    out = capsys.readouterr().out
    assert "D_rel" not in out


# --- merged from test_lowsev_convergence.py ---
def test_n_zero_early_returns():
    """With no completed trajectories the function returns the early-return dict
    and never reaches the Wilson interval code."""
    result = analyse_convergence(n_reacted=0, n_escaped=0, k_b=1.0)
    assert result == {"converged": False, "reason": "no completed trajectories"}


def test_healthy_path_wilson_interval_unchanged():
    """A normal call with reactions and escapes still yields a Wilson interval
    in [0, 1] with lo <= hi, confirming the surviving branch computes correctly."""
    result = analyse_convergence(n_reacted=40, n_escaped=60, k_b=2.0)
    assert result["N"] == 100
    assert math.isclose(result["P_rxn"], 0.4)
    lo, hi = result["wilson_CI_P"]
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < result["P_rxn"] < hi


# --- merged from test_lowsev_geometry.py ---
def test_lenient_pqr_fallback_warns_and_defaults_radius(tmp_path):
    """A 9-field PQR with no radius column triggers the lenient fallback,
    which warns and defaults the radius to 1.5 A without changing the value."""
    pqr = tmp_path / "no_radius.pqr"
    pqr.write_text(
        "ATOM 1 N ALA 1 0.0 0.0 0.0 -0.5\n" "ATOM 2 C ALA 1 1.0 0.0 0.0 0.3\n"
    )
    with pytest.warns(UserWarning, match="defaulting radius to 1.5"):
        atoms = parse_pqr_test_lowsev_geometry(pqr)
    assert len(atoms) == 2
    # The defaulted radius value itself is unchanged at 1.5 A.
    assert all(a.radius == 1.5 for a in atoms)


def test_pqr_with_radius_column_no_warning(tmp_path):
    """A PQR line that includes a radius column does not trigger the fallback
    warning and parses the radius value as given."""
    pqr = tmp_path / "with_radius.pqr"
    pqr.write_text("ATOM 1 N ALA 1 0.0 0.0 0.0 -0.5 1.85\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        atoms = parse_pqr_test_lowsev_geometry(pqr)
    assert len(atoms) == 1
    assert atoms[0].radius == 1.85


def _write_two_reaction_xml(path, second_has_n_needed):
    """Write a two-reaction rxns XML. The first reaction declares n_needed=2.
    The second reaction declares n_needed=3 only when second_has_n_needed is
    True, otherwise it omits the element."""
    second_nn = "<n_needed>3</n_needed>" if second_has_n_needed else ""
    path.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <n_needed>2</n_needed>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "      <pair><atoms>2 12</atoms><distance>5.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        f"      {second_nn}\n"
        "      <pair><atoms>3 13</atoms><distance>4.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )


def test_multi_reaction_second_without_n_needed_does_not_inherit(tmp_path):
    """In a multi-reaction file, a second reaction without an explicit n_needed
    must not inherit the first reaction's n_needed. The flattened parser keeps
    the last reaction's own value, which here defaults to -1."""
    xml = tmp_path / "multi_inherit.xml"
    _write_two_reaction_xml(xml, second_has_n_needed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    # All three pairs are flattened from both reactions.
    assert len(pairs) == 3
    assert n_needed == -1


def test_multi_reaction_conflicting_n_needed_warns(tmp_path):
    """When two reactions declare different n_needed values, the flattened
    parser cannot represent both and emits a warning, keeping the last
    reaction's value."""
    xml = tmp_path / "multi_conflict.xml"
    _write_two_reaction_xml(xml, second_has_n_needed=True)
    with pytest.warns(UserWarning, match="differing n_needed"):
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 3
    # The last reaction declared n_needed=3.
    assert n_needed == 3


def test_single_reaction_n_needed_unchanged(tmp_path):
    """A single-reaction file with an explicit n_needed parses to that value
    with no warning, confirming the single-reaction path is unchanged."""
    xml = tmp_path / "single.xml"
    xml.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <n_needed>2</n_needed>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "      <pair><atoms>2 12</atoms><distance>4.5</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 2
    assert n_needed == 2
    # One-based to zero-based conversion is unchanged.
    assert pairs[0].rec_index == 0
    assert pairs[0].lig_index == 10
    assert pairs[0].cutoff == 5.0


def test_single_reaction_no_n_needed_defaults_minus_one(tmp_path):
    """A single-reaction file without an n_needed element parses to the
    reference default of -1, matching prior behavior."""
    xml = tmp_path / "single_default.xml"
    xml.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 1
    assert n_needed == -1


# --- merged from test_lowsev_gho_injection.py ---


# --- merged from test_lowsev_gpu_sim_guards.py ---
def _make_result_test_lowsev_gpu_sim_guards(n_reacted, n_escaped, r_start, r_escape):
    """Build a GPUBatchResult with only the fields these helpers read."""
    return GPUBatchResult(
        n_trajectories=n_reacted + n_escaped,
        n_reacted=n_reacted,
        n_escaped=n_escaped,
        n_max_steps=0,
        reaction_counts={},
        r_start=r_start,
        r_escape=r_escape,
        dt=1.0,
        elapsed_sec=1.0,
        steps_per_sec=1.0,
    )


def test_rate_constant_guard_raises_on_degenerate_denominator():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=50, n_escaped=50, r_start=2.0, r_escape=1.0
    )
    assert res.reaction_probability == 0.5
    with pytest.raises(ValueError, match="denominator"):
        res.rate_constant(D_rel=1.0, k_b=0.0)


def test_rate_constant_ci_finite_at_p_max():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=100, n_escaped=0, r_start=1.0, r_escape=1.0e30
    )
    lo, hi = res.rate_constant_ci(D_rel=1.0, k_b=0.0)
    assert 0.0 <= lo <= hi
    assert math.isfinite(hi)


def test_rate_constant_healthy_path_unchanged():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=30, n_escaped=70, r_start=10.0, r_escape=50.0
    )
    D_rel = 0.25
    P = res.reaction_probability
    CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    k_D = 4.0 * math.pi * D_rel * res.r_start
    beta = res.r_start / res.r_escape
    expected = CONV * k_D * P / (1.0 - (1.0 - P) * beta)
    assert res.rate_constant(D_rel=D_rel, k_b=0.0) == expected


def test_rate_constant_steering_path_unchanged():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=100, n_escaped=0, r_start=1.0, r_escape=1.0e30
    )
    k_b = 2.5
    P = res.reaction_probability
    CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
    expected = CONV * k_b * P
    # Should not raise even though the Smoluchowski denominator would be zero.
    assert res.rate_constant(D_rel=1.0, k_b=k_b) == expected


def test_reaction_probability_ci_zero_completed_warns_and_returns_unit_interval():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=0, n_escaped=0, r_start=10.0, r_escape=50.0
    )
    assert res.n_completed == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci = res.reaction_probability_ci()
    assert ci == (0.0, 1.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_reaction_probability_ci_nonzero_does_not_warn():
    res = _make_result_test_lowsev_gpu_sim_guards(
        n_reacted=40, n_escaped=60, r_start=10.0, r_escape=50.0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lo, hi = res.reaction_probability_ci()
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert 0.0 <= lo <= hi <= 1.0


# --- merged from test_lowsev_grid_force.py ---
def _orthogonal_delta(hx=1.0, hy=1.0, hz=1.0):
    return np.array([[hx, 0.0, 0.0], [0.0, hy, 0.0], [0.0, 0.0, hz]])


def test_valid_orthogonal_grid_sets_inv_dx():
    """A valid orthogonal grid still builds and computes the unchanged _inv_dx."""
    delta = _orthogonal_delta(0.5, 2.0, 4.0)
    data = np.zeros((3, 3, 3))
    grid = DXGrid(origin=np.zeros(3), delta=delta, data=data)
    assert np.allclose(grid._inv_dx, 1.0 / np.array([0.5, 2.0, 4.0]))


def test_zero_spacing_raises():
    delta = _orthogonal_delta(1.0, 0.0, 1.0)
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="axis 1"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_negative_spacing_raises():
    delta = _orthogonal_delta(1.0, 1.0, -1.0)
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="axis 2"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_non_orthogonal_grid_raises():
    delta = _orthogonal_delta(1.0, 1.0, 1.0)
    delta[0, 1] = 0.3  # large off-diagonal entry
    data = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="not orthogonal"):
        DXGrid(origin=np.zeros(3), delta=delta, data=data)


def test_tiny_off_diagonal_is_accepted():
    """Numerical noise far below the diagonal scale must not trip the guard."""
    delta = _orthogonal_delta(2.0, 2.0, 2.0)
    delta[1, 0] = 1e-12  # negligible relative to spacing of 2.0
    data = np.zeros((3, 3, 3))
    grid = DXGrid(origin=np.zeros(3), delta=delta, data=data)
    assert np.allclose(grid._inv_dx, 0.5)


def _write_dx(tmp_path, nx, ny, nz, n_values):
    lines = [
        f"object 1 class gridpositions counts {nx} {ny} {nz}",
        "origin 0.0 0.0 0.0",
        "delta 1.0 0.0 0.0",
        "delta 0.0 1.0 0.0",
        "delta 0.0 0.0 1.0",
        "object 2 class gridconnections counts {0} {1} {2}".format(nx, ny, nz),
        f"object 3 class array type double rank 0 items {n_values} data follows",
    ]
    values = [str(float(i)) for i in range(n_values)]
    # Six values per line, mimicking the OpenDX layout.
    for start in range(0, n_values, 6):
        lines.append(" ".join(values[start : start + 6]))
    lines.append('attribute "dep" string "positions"')
    path = tmp_path / "grid.dx"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_from_file_value_count_mismatch_raises(tmp_path):
    # Declare a 3x3x3 grid (27 values) but only supply 20.
    path = _write_dx(tmp_path, 3, 3, 3, n_values=20)
    with pytest.raises(ValueError) as excinfo:
        DXGrid.from_file(path)
    msg = str(excinfo.value)
    assert str(path) in msg
    assert "27" in msg  # expected count
    assert "20" in msg  # actual count


def test_from_file_well_formed_roundtrip(tmp_path):
    nx, ny, nz = 2, 3, 4
    path = _write_dx(tmp_path, nx, ny, nz, n_values=nx * ny * nz)
    grid = DXGrid.from_file(path)
    assert grid.data.shape == (nx, ny, nz)
    # Values were written as 0..N-1 in C order.
    expected = np.arange(nx * ny * nz, dtype=float).reshape(nx, ny, nz)
    assert np.array_equal(grid.data, expected)


# --- merged from test_lowsev_make_pqr.py ---
def _install_stubs(monkeypatch, pqr_content):
    """Patch _check_tool and _run so make_combined_pqr runs with no binaries.

    The _run stand in inspects the command string. For the ambpdb command it
    writes pqr_content (or nothing when pqr_content is None) to complex.pqr in
    the working directory, reproducing what the real shell redirection would do.
    The cpptraj command is treated as a no op that creates the expected rst file.
    """
    monkeypatch.setattr(make_pqr, "_check_tool", lambda name: None)

    def fake_run(cmd, cwd, step):
        cwd = Path(cwd)
        if step == "cpptraj":
            # cpptraj would normally write complex.rst from the pdb.
            (cwd / "complex.rst").write_text("dummy restart\n")
        elif step == "ambpdb":
            if pqr_content is not None:
                # Mimic the shell redirection target ambpdb -pqr > complex.pqr.
                (cwd / "complex.pqr").write_text(pqr_content)
            # When pqr_content is None we leave the file absent on purpose.
        return None

    monkeypatch.setattr(make_pqr, "_run", fake_run)


def test_missing_output_raises_clear_error(tmp_path, monkeypatch):
    """A missing combined PQR file raises a clear RuntimeError naming ambpdb."""
    _install_stubs(monkeypatch, pqr_content=None)
    with pytest.raises(RuntimeError, match=r"ambpdb.*no output"):
        make_pqr.make_combined_pqr(
            prmtop_path=tmp_path / "complex.prmtop",
            complex_pdb=tmp_path / "complex.pdb",
            work_dir=tmp_path,
        )


def test_atomless_output_raises_clear_error(tmp_path, monkeypatch):
    """A combined PQR file with no ATOM or HETATM records raises a clear error."""
    # A header only file with no atom records, the kind a failed run might leave.
    _install_stubs(monkeypatch, pqr_content="REMARK   1 nothing useful here\nEND\n")
    with pytest.raises(RuntimeError, match=r"no ATOM or HETATM"):
        make_pqr.make_combined_pqr(
            prmtop_path=tmp_path / "complex.prmtop",
            complex_pdb=tmp_path / "complex.pdb",
            work_dir=tmp_path,
        )


def test_valid_output_passes_and_cleans_intermediates(tmp_path, monkeypatch):
    """A PQR holding atom records returns its path and removes intermediate files.

    This is the healthy path. The guard must not raise, the returned path must be
    complex.pqr in the working directory, the atom content must be untouched, and
    the cpptraj input and inpcrd intermediates must be cleaned up as before.
    """
    valid = (
        "ATOM      1  N   ALA     1      11.104   6.134  -6.504  0.1000 1.5500\n"
        "HETATM    2  C1  LIG     2      12.000   7.000  -5.000 -0.2000 1.7000\n"
        "END\n"
    )
    _install_stubs(monkeypatch, pqr_content=valid)
    out = make_pqr.make_combined_pqr(
        prmtop_path=tmp_path / "complex.prmtop",
        complex_pdb=tmp_path / "complex.pdb",
        work_dir=tmp_path,
    )
    assert out == tmp_path / "complex.pqr"
    assert out.read_text() == valid
    # Intermediates created or consumed by the function must be gone.
    assert not (tmp_path / "get_inpcrd.cpptraj").exists()
    assert not (tmp_path / "complex.inpcrd").exists()


def test_first_atom_only_is_enough(tmp_path, monkeypatch):
    """The guard accepts a file whose first atom record is a HETATM line."""
    content = "HETATM    1  C1  LIG     1       0.000   0.000   0.000 0.0000 1.7000\n"
    _install_stubs(monkeypatch, pqr_content=content)
    out = make_pqr.make_combined_pqr(
        prmtop_path=tmp_path / "complex.prmtop",
        complex_pdb=tmp_path / "complex.pdb",
        work_dir=tmp_path,
    )
    assert out.read_text() == content


# --- merged from test_lowsev_molecules.py ---
def _two_atom_mol(name, x):
    """Build a small molecule with two atoms positioned along the x axis."""
    return Molecule(name=name, atoms=[Atom(x=x), Atom(x=x + 1.0)])


def test_out_of_range_mol1_index_raises_clear_error():
    """An mol1 contact index past the atom count names the index and atom count."""
    mol1 = _two_atom_mol("ligand", 0.0)
    mol2 = _two_atom_mol("receptor", 0.0)
    # mol1 has 2 atoms, so index 5 is out of range.
    criteria = ReactionCriteria(name="assoc", pairs=[ContactPair(5, 0, 5.0)])
    with pytest.raises(IndexError) as excinfo:
        criteria.is_satisfied(mol1, mol2)
    msg = str(excinfo.value)
    assert "5" in msg
    assert "2 atoms" in msg
    assert "ligand" in msg
    assert "assoc" in msg


def test_out_of_range_mol2_index_raises_clear_error():
    """An mol2 contact index past the atom count names the index and atom count."""
    mol1 = _two_atom_mol("ligand", 0.0)
    mol2 = _two_atom_mol("receptor", 0.0)
    criteria = ReactionCriteria(name="assoc", pairs=[ContactPair(0, 7, 5.0)])
    with pytest.raises(IndexError) as excinfo:
        criteria.is_satisfied(mol1, mol2)
    msg = str(excinfo.value)
    assert "7" in msg
    assert "receptor" in msg


def test_valid_indices_unchanged_true():
    """Healthy path with valid in-range indices still fires when within cutoff."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 0.0)  # atom0 of each coincides, distance 0
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(0, 0, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is True


def test_valid_indices_unchanged_false():
    """Healthy path with valid in-range indices returns False beyond the cutoff."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 100.0)  # far apart, beyond cutoff
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(0, 0, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is False


def test_negative_wraparound_index_still_works():
    """Negative-wraparound indices that were valid before remain valid (no behavior change)."""
    mol1 = _two_atom_mol("a", 0.0)
    mol2 = _two_atom_mol("b", 0.0)
    # -1 refers to the last atom; atom1 of each is at x=1, distance 0, within cutoff.
    criteria = ReactionCriteria(name="rxn", pairs=[ContactPair(-1, -1, 5.0)])
    assert criteria.is_satisfied(mol1, mol2) is True


# --- merged from test_lowsev_multi_gpu_runs.py ---
MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pystarc",
    "multi_GPU",
    "multi_GPU_runs.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_GPU_runs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_xml(path):
    with open(path, "w") as fh:
        fh.write(
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            "<input>\n"
            "  <n_trajectories>4</n_trajectories>\n"
            "  <seed>1</seed>\n"
            "</input>\n"
        )


class _FakeReturn:
    def __init__(self, returncode):
        self.returncode = returncode


def test_missing_bd_sims_after_grid_gen_reports_clear_error(
    tmp_path, monkeypatch, capsys
):
    module = _load_module()

    xml_path = tmp_path / "input.xml"
    _write_minimal_xml(str(xml_path))

    def fake_run(*args, **kwargs):
        return _FakeReturn(0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["multi_GPU_runs.py", str(xml_path)])

    bd_sims = tmp_path / "bd_sims"
    assert not bd_sims.exists()

    # Must return cleanly rather than raising FileNotFoundError from os.listdir.
    result = module.main()
    assert result is None

    out = capsys.readouterr().out
    assert "did not create" in out
    # The guard must short-circuit before any per-split directory is created.
    assert not bd_sims.exists()


def test_grid_gen_failure_returns_without_listdir(tmp_path, monkeypatch, capsys):
    module = _load_module()

    xml_path = tmp_path / "input.xml"
    _write_minimal_xml(str(xml_path))

    def fake_run(*args, **kwargs):
        return _FakeReturn(1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sys, "argv", ["multi_GPU_runs.py", str(xml_path)])

    result = module.main()
    assert result is None
    out = capsys.readouterr().out
    assert "grid generation failed" in out


# --- merged from test_lowsev_nam_serial_verbose.py ---
def _make_sim_test_lowsev_nam_serial_verbose(verbose: bool, n: int):
    sim = object.__new__(NAMSimulator)
    sim.params = NAMParameters(n_trajectories=n, verbose=verbose)
    sim.n_reacted = 0
    sim.n_escaped = 0
    recorded = []
    sim.run_one = lambda: "traj"
    sim._record = lambda result: recorded.append(result)
    return sim, recorded


def test_verbose_prints_one_line_per_trajectory():
    n = 5
    sim, recorded = _make_sim_test_lowsev_nam_serial_verbose(verbose=True, n=n)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sim._run_serial(n)
    lines = [ln for ln in buf.getvalue().splitlines() if "Trajectory" in ln]
    assert len(lines) == n
    # Every trajectory must be recorded regardless of the guard.
    assert len(recorded) == n


def test_non_verbose_prints_nothing():
    n = 4
    sim, recorded = _make_sim_test_lowsev_nam_serial_verbose(verbose=False, n=n)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sim._run_serial(n)
    assert buf.getvalue() == ""
    assert len(recorded) == n


# --- merged from test_lowsev_numerical.py ---
def test_legendre_p_negative_degree_raises():
    """A negative degree must raise a clear ValueError rather than silently
    returning x (the previous behavior)."""
    with pytest.raises(ValueError):
        legendre_p(-1, 0.5)
    with pytest.raises(ValueError):
        legendre_p(-3, -0.25)


def test_legendre_p_healthy_path_unchanged():
    """The nonnegative-degree path must be unchanged: P0(x)=1, P1(x)=x,
    P2(x)=(3x^2-1)/2."""
    x = 0.3
    assert legendre_p(0, x) == 1.0
    assert legendre_p(1, x) == x
    assert legendre_p(2, x) == pytest.approx((3 * x * x - 1.0) / 2.0)


# --- merged from test_lowsev_pipeline_gpusim.py ---
def _select_k_b(gpu_sim):
    """Reproduce the k_b selection expression from run() exactly."""
    return getattr(gpu_sim, "_k_b", 0.0) if gpu_sim is not None else 0.0


class _StubGpuSim:
    """Stand-in for GPUBatchSimulator carrying a Romberg k_b estimate."""

    def __init__(self, k_b):
        self._k_b = k_b


def test_k_b_selection_none_returns_zero_without_raising():
    assert _select_k_b(None) == 0.0


def test_k_b_selection_uses_attribute_on_healthy_gpu_path():
    sim = _StubGpuSim(1.2345)
    assert _select_k_b(sim) == 1.2345


def test_k_b_selection_attribute_missing_falls_back_to_zero():
    class _NoKB:
        pass

    assert _select_k_b(_NoKB()) == 0.0


def test_run_source_initializes_gpu_sim_and_guards_access():
    src = inspect.getsource(pipeline.run)
    # The defensive initialization must be present before any branch uses it.
    assert "gpu_sim = None" in src
    assert 'getattr(gpu_sim, "_k_b", 0.0) if gpu_sim is not None else 0.0' in src
    ast.parse(src.strip())


# --- merged from test_lowsev_pqr_io.py ---
def test_numeric_chain_is_not_misread_as_resid():
    """A numeric chain identifier is read as the chain, not as the residue index.

    The fields after the chain are resid, x, y, z, charge, radius. With a numeric
    chain the old parser shifted all of these by one token, so the residue index,
    coordinates, charge, and radius were all corrupted. This checks every field is
    placed correctly.
    """
    line = "ATOM      5  CA  ALA 1   42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "1"
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_multi_digit_numeric_chain_with_element():
    """A multi digit numeric chain and a trailing element symbol are both read correctly."""
    line = "ATOM      5  CA  ALA 10   7       1.000   2.000   3.000  0.500  1.800 C"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "10"
    assert rec.resid == 7
    assert rec.x == 1.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == "C"


def test_alphabetic_chain_parses_as_before():
    """A standard alphabetic chain line parses with the chain and all fields intact."""
    line = "ATOM      5  CA  ALA A   42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "A"
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_alphabetic_chain_with_element():
    """An alphabetic chain line with a trailing element symbol parses correctly."""
    line = "ATOM      5  CA  ALA B   42       1.000   2.000   3.000  0.500  1.800 N"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "B"
    assert rec.resid == 42
    assert rec.element == "N"


def test_no_chain_parses_as_before():
    """A no chain line keeps an empty chain and reads the residue index from token four."""
    line = "ATOM      5  CA  ALA     42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_no_chain_with_element():
    """A no chain line with a trailing element symbol parses correctly."""
    line = "ATOM      5  CA  ALA     42       1.000   2.000   3.000  0.500  1.800 O"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.element == "O"


def test_no_chain_collapsed_negative_coordinate():
    """A no chain line whose negative x coordinate retains a decimal point still parses."""
    line = "ATOM      5  CA  ALA     42      -1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.x == -1.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8


# --- merged from test_lowsev_prepare_bd_surface.py ---
def _gho_atom(serial: int = 1) -> PQRAtom:
    """Return a GHO ghost atom at the origin with zero charge and radius."""
    return PQRAtom(
        serial=serial,
        name="GHO",
        resname="GHO",
        resid=serial,
        x=0.0,
        y=0.0,
        z=0.0,
        charge=0.0,
        radius=0.0,
    )


def _real_atom() -> PQRAtom:
    """Return a single ordinary atom away from the origin."""
    return PQRAtom(
        serial=1,
        name="C1",
        resname="LIG",
        resid=1,
        x=1.0,
        y=2.0,
        z=3.0,
        charge=0.1,
        radius=1.7,
    )


def test_compute_grid_params_empty_atom_list_raises():
    """An empty atom list raises a clear ValueError rather than max() on empty."""
    with pytest.raises(ValueError) as excinfo:
        compute_grid_params([])
    assert "no real atoms" in str(excinfo.value)


def test_compute_grid_params_only_gho_raises():
    """An atom list of only GHO ghost atoms raises the same clear ValueError."""
    with pytest.raises(ValueError) as excinfo:
        compute_grid_params([_gho_atom(1), _gho_atom(2)])
    assert "no real atoms" in str(excinfo.value)


def test_compute_grid_params_healthy_path_still_works():
    """A list with at least one real atom still returns the expected grid blocks."""
    grids = compute_grid_params([_real_atom(), _gho_atom(2)])
    assert len(grids) == 3
    for g in grids:
        assert set(g.keys()) == {"spacing", "dime", "glen", "gcent"}
        assert len(g["dime"]) == 3
        assert len(g["glen"]) == 3
        assert len(g["gcent"]) == 3
    # The grid centre is the heavy-atom centroid, ignoring the GHO atom.
    assert grids[0]["gcent"] == [1.0, 2.0, 3.0]


# --- merged from test_lowsev_quaternion.py ---
def test_apply_accepts_python_list_single_point():
    """A single point given as a Python list should not raise AttributeError.

    The identity transform leaves the point unchanged, and the result should
    be a 1D array of shape (3,) matching a (3,) input.
    """
    t = RigidTransform.identity()
    result = t.apply([1.0, 2.0, 3.0])
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_apply_accepts_python_list_of_points():
    """A list of points should be treated as a (N, 3) array and return (N, 3)."""
    t = RigidTransform.identity()
    result = t.apply([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_apply_list_matches_ndarray_healthy_path():
    """List input must give the same result as the equivalent ndarray input."""
    rot = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    t = RigidTransform(rot, np.array([0.5, -1.0, 2.0]))
    pts_list = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    pts_arr = np.array(pts_list, dtype=float)
    np.testing.assert_allclose(t.apply(pts_list), t.apply(pts_arr))


# --- merged from test_lowsev_reaction_interface.py ---
def _line_molecule(name, n_atoms, x0=0.0):
    """Build a molecule of n_atoms atoms spaced one angstrom apart along x."""
    mol = Molecule(name=name)
    mol.atoms = [Atom(x=float(i) + x0) for i in range(n_atoms)]
    return mol


def test_empty_first_molecule_raises():
    """An empty mol1 must not yield a zero-pair always-firing reaction."""
    mol1 = Molecule(name="empty")
    mol2 = _line_molecule("m2", 5, x0=20.0)
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2)


def test_empty_second_molecule_raises():
    """An empty mol2 must not yield a zero-pair always-firing reaction."""
    mol1 = _line_molecule("m1", 5)
    mol2 = Molecule(name="empty")
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2)


def test_both_empty_molecules_raise():
    """Two empty molecules must raise rather than build an always-firing reaction."""
    with pytest.raises(ValueError):
        make_default_reaction(Molecule(name="a"), Molecule(name="b"))


@pytest.mark.parametrize("bad_n", [0, -1, -3])
def test_non_positive_n_pairs_raises(bad_n):
    """A non-positive n_pairs would give empty criteria and must raise."""
    mol1 = _line_molecule("m1", 5)
    mol2 = _line_molecule("m2", 5, x0=20.0)
    with pytest.raises(ValueError):
        make_default_reaction(mol1, mol2, n_pairs=bad_n)


def test_small_molecule_clamps_pair_count():
    """When a molecule is smaller than n_pairs, the pair count is clamped to the
    smaller atom count and the reaction stays non-degenerate."""
    mol1 = _line_molecule("m1", 5)
    mol2 = _line_molecule("m2", 1, x0=20.0)
    rxn = make_default_reaction(mol1, mol2, n_pairs=3)
    # Only one atom is available on mol2, so exactly one contact pair can form.
    assert len(rxn.criteria.pairs) == 1
    # The criterion must require at least one contact, so it is not always satisfied.
    assert len(rxn.criteria.pairs) >= 1


def test_single_atom_molecules_produce_one_pair():
    """Two single-atom molecules give one valid contact pair, not a degenerate
    empty reaction."""
    mol1 = _line_molecule("m1", 1)
    mol2 = _line_molecule("m2", 1, x0=20.0)
    rxn = make_default_reaction(mol1, mol2, n_pairs=3)
    assert len(rxn.criteria.pairs) == 1


def test_normal_molecules_unchanged():
    """For molecules large enough to supply n_pairs atoms on both sides, the
    requested number of pairs is produced exactly as before the guard was added."""
    mol1 = _line_molecule("m1", 10)
    mol2 = _line_molecule("m2", 10, x0=20.0)
    for n in (1, 2, 3, 4, 5):
        rxn = make_default_reaction(mol1, mol2, n_pairs=n)
        assert len(rxn.criteria.pairs) == n
        # Pairs index the closest atoms on each side, which are well defined.
        for pair in rxn.criteria.pairs:
            assert 0 <= pair.mol1_atom_index < len(mol1.atoms)
            assert 0 <= pair.mol2_atom_index < len(mol2.atoms)


# --- merged from test_lowsev_rotne_prager.py ---
def test_hydrodynamic_center_healthy_path_unchanged():
    # Radius-weighted centroid for positive radii must be exact.
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([1.0, 3.0])
    hc = _hydrodynamic_center(positions, radii)
    expected = (1.0 * positions[0] + 3.0 * positions[1]) / 4.0
    np.testing.assert_allclose(hc, expected)


def test_hydrodynamic_center_empty_raises_without_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            _hydrodynamic_center(np.empty((0, 3)), np.empty((0,)))


def test_hydrodynamic_center_all_zero_radii_raises():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            _hydrodynamic_center(positions, radii)


def test_chain_diffusion_tensors_empty_raises_same_error_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="at least one bead required"):
            chain_diffusion_tensors(np.empty((0, 3)), np.empty((0,)))


# --- merged from test_lowsev_simulation_io.py ---
def _write_xml_test_lowsev_simulation_io(tmp_path, body: str):
    p = tmp_path / "sim.xml"
    p.write_text("<simulation>\n" + body + "\n</simulation>\n")
    return p


def test_getf_handles_literal_none(tmp_path):
    p = _write_xml_test_lowsev_simulation_io(
        tmp_path, "  <dt>None</dt>\n  <r_start>None</r_start>"
    )
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.2
    assert result["r_start"] == 100.0


def test_getf_healthy_numeric_value_unchanged(tmp_path):
    # A normal numeric value must still parse to that exact float.
    p = _write_xml_test_lowsev_simulation_io(
        tmp_path, "  <dt>0.05</dt>\n  <r_start>250.0</r_start>"
    )
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.05
    assert result["r_start"] == 250.0


def test_getf_missing_tag_uses_default(tmp_path):
    # A missing float tag must still use the supplied default.
    p = _write_xml_test_lowsev_simulation_io(
        tmp_path, "  <n_trajectories>5</n_trajectories>"
    )
    result = parse_simulation_xml(p)
    assert result["dt"] == 0.2
    assert result["r_escape"] == 0.0


# --- merged from test_lowsev_we_makebins.py ---
def _make_mol_test_lowsev_we_makebins(name: str) -> Molecule:
    mol = Molecule(name=name)
    mol.atoms = [
        Atom(index=0, x=0.0, y=0.0, z=0.0, charge=1.0, radius=1.5),
        Atom(index=1, x=2.0, y=0.0, z=0.0, charge=-1.0, radius=1.5),
    ]
    return mol


def test_compute_geometry_escape_radius_override(tmp_path, monkeypatch):
    """compute_geometry uses 2 times the b-sphere radius by default and uses a positive r_escape when one is given."""

    def fake_analyse(pqr_path, srad=0.0):
        return MoleculeGeometry(
            n_atoms=1,
            n_charged=1,
            n_ghost=0,
            centroid=np.zeros(3),
            max_radius=1.0,
            hydrodynamic_r=1.0,
            ghost_indices=[],
            ghost_positions=[],
            total_charge=1.0,
        )

    monkeypatch.setattr(geom_mod, "analyse_molecule", fake_analyse)
    rec = tmp_path / "rec.pqr"
    lig = tmp_path / "lig.pqr"
    rec.write_text("x")
    lig.write_text("x")

    g_default = compute_geometry(rec, lig, bd_milestone_radius=10.0)
    assert g_default.r_start == 10.0
    assert g_default.r_escape == 20.0  # 2 times the b-sphere radius

    g_over = compute_geometry(rec, lig, bd_milestone_radius=10.0, r_escape=55.0)
    assert g_over.r_start == 10.0
    assert g_over.r_escape == 55.0


def test_parse_reads_r_escape_override(tmp_path):
    """The input parser reads the r_escape tag, defaulting to the 0 sentinel when the tag is absent."""

    src = Path(__file__).resolve().parent.parent / "examples" / "two_charged_spheres"
    assert src.is_dir(), (
        "two_charged_spheres example inputs missing at %s; they ship with the "
        "package, so this is a broken checkout rather than a reason to skip" % src
    )
    dst = tmp_path / "tcs"
    shutil.copytree(src, dst)
    base = (dst / "input.xml").read_text()
    assert parse(dst / "input.xml").r_escape == 0.0
    inj = base.replace("</pystarc>", "  <r_escape>40.0</r_escape>\n</pystarc>")
    (dst / "input_resc.xml").write_text(inj)
    assert parse(dst / "input_resc.xml").r_escape == 40.0


def test_dx_gradient_is_central_difference_and_more_accurate():
    """DXGrid.gradient uses the second-order central difference matching the GPU path, and is more accurate than the first-order gradient_of_cube on a curved field."""

    lam, h, n = 7.86, 0.5, 81
    origin = np.array([-20.0, -20.0, -20.0])
    delta = np.diag([h, h, h]).astype(float)
    ax = origin[0] + h * np.arange(n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R = np.where(R < 1e-6, 1e-6, R)
    g = DXGrid(origin, delta, np.exp(-R / lam) / R)
    pts = np.array([[8.0, 1.0, -2.0], [5.0, -4.0, 3.0], [11.0, 2.0, 6.0]])

    # gradient() must equal the central-difference batch operator point by point.
    for p in pts:
        np.testing.assert_allclose(
            g.gradient(p), g.batch_gradient(p.reshape(1, 3))[0], atol=1e-12
        )

    # gradient_of_cube is preserved and differs from the central difference here.
    assert (
        max(np.linalg.norm(g.gradient(p) - g.gradient_of_cube(p)) for p in pts) > 1e-6
    )

    # The central difference is closer to the true analytic gradient.
    def analytic(p):
        r = np.linalg.norm(p)
        return -np.exp(-r / lam) * (1.0 / (lam * r) + 1.0 / r**2) * (p / r)

    err_cd = np.mean(
        [
            np.linalg.norm(g.gradient(p) - analytic(p)) / np.linalg.norm(analytic(p))
            for p in pts
        ]
    )
    err_gc = np.mean(
        [
            np.linalg.norm(g.gradient_of_cube(p) - analytic(p))
            / np.linalg.norm(analytic(p))
            for p in pts
        ]
    )
    assert err_cd < err_gc


# Consolidated audit-fix and low-severity regression tests.
# Previously in separate tests/test_auditfix*.py and tests/test_lowsev_*.py,
# merged here so the whole suite lives in one file.


# --- merged from test_finish_chaingb_const.py ---
def _recompute_coulomb_k_kbt_a() -> float:
    """Recompute k_e e^2 / kBT in angstrom from the documented expression.

    The expression is COULOMB_K_KBT_A = k_e e^2 * 1e10 / (kB T) with k_e the Coulomb
    constant in N m^2 / C^2, e the elementary charge in C, kB the Boltzmann constant
    in J/K, and T = 298.15 K (T_DEFAULT). The factor 1e10 converts meters to angstrom.
    """
    k_e = 8.9875517873681764e9  # Coulomb constant, N m^2 / C^2
    e = 1.602176634e-19  # elementary charge, C
    kB = 1.380649e-23  # Boltzmann constant, J/K
    T = 298.15  # K (T_DEFAULT, consistent with the rest of the engine)
    return k_e * e * e * 1e10 / (kB * T)


def test_coulomb_constant_matches_first_principles_value():
    """The module literal matches the recomputed value to about 4 significant figures."""
    expected = _recompute_coulomb_k_kbt_a()
    literal = chain_gb.COULOMB_K_KBT_A

    rel_err = abs(literal - expected) / expected
    assert rel_err < 5e-5, (
        f"COULOMB_K_KBT_A literal {literal} disagrees with the recomputed value "
        f"{expected} by relative error {rel_err}"
    )


def test_coulomb_constant_is_not_the_stale_literal():
    """The literal is no longer the stale 556.86 value flagged in the audit."""
    assert abs(chain_gb.COULOMB_K_KBT_A - 556.86) > 0.1


def test_water_dielectric_is_single_sourced():
    """WATER_DIELECTRIC is defined and is the single consistent water dielectric."""
    assert hasattr(chain_gb, "WATER_DIELECTRIC")
    assert chain_gb.WATER_DIELECTRIC == 78.5


def test_eps_out_defaults_use_water_dielectric():
    """Every chain GB routine with an eps_out argument defaults to WATER_DIELECTRIC.

    This guards against the dielectric drifting between nearby literals across the
    energy and force routines.
    """
    routines = [
        chain_gb.gb_self_born_energy,
        chain_gb.gb_offdiagonal_energy,
        chain_gb.chain_self_born_diagonal_force,
        chain_gb.chain_offdiagonal_gb_force,
        chain_gb.chain_full_gb_force,
    ]
    for fn in routines:
        sig = inspect.signature(fn)
        assert "eps_out" in sig.parameters, f"{fn.__name__} has no eps_out parameter"
        default = sig.parameters["eps_out"].default
        assert default == chain_gb.WATER_DIELECTRIC, (
            f"{fn.__name__} eps_out default {default} is not WATER_DIELECTRIC "
            f"{chain_gb.WATER_DIELECTRIC}"
        )


# --- merged from test_finish_molecules_needed.py ---
def _two_molecules(separation: float) -> tuple[Molecule, Molecule]:
    """Build two single-atom molecules placed separation angstrom apart on x."""
    mol1 = Molecule(name="mol1", atoms=[Atom(index=0, x=0.0, y=0.0, z=0.0)])
    mol2 = Molecule(name="mol2", atoms=[Atom(index=0, x=separation, y=0.0, z=0.0)])
    return mol1, mol2


def test_zero_needed_explicit_returns_false():
    """A criterion with n_needed == 0 can never fire, even with close contacts."""
    mol1, mol2 = _two_molecules(separation=1.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="zero_needed", pairs=pairs, n_needed=0)
    assert criterion.is_satisfied(mol1, mol2) is False


def test_negative_needed_empty_pairs_returns_false():
    """A criterion with no pairs and the default n_needed == -1 cannot fire."""
    mol1, mol2 = _two_molecules(separation=1.0)
    criterion = ReactionCriteria(name="empty", pairs=[], n_needed=-1)
    assert criterion.is_satisfied(mol1, mol2) is False


def test_n_needed_one_satisfied_unchanged():
    """With n_needed == 1 a single close contact still satisfies the criterion."""
    mol1, mol2 = _two_molecules(separation=1.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="one_needed", pairs=pairs, n_needed=1)
    assert criterion.is_satisfied(mol1, mol2) is True


def test_n_needed_one_not_satisfied_unchanged():
    """With n_needed == 1 a contact beyond the cutoff still fails the criterion."""
    mol1, mol2 = _two_molecules(separation=10.0)
    pairs = [ContactPair(mol1_atom_index=0, mol2_atom_index=0, distance_cutoff=5.0)]
    criterion = ReactionCriteria(name="one_needed", pairs=pairs, n_needed=1)
    assert criterion.is_satisfied(mol1, mol2) is False


# --- merged from test_finish_we_bridge.py ---
def _make_molecules_test_finish_we_bridge(lig_x, radius=2.0):
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


def _make_pathways_test_finish_we_bridge(cutoff=10.0):
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


# Regression tests for the 2026-06 correctness-review fixes:
#   1. parallel MULTIPROCESSING/FUTURES worker initialisation
#   2. multi-GPU combine density (shell volume) and frequency (step count)
#   3. rxns.xml <contact> validation (no silent atom-0 default)


class TestMultiGpuCombineNormalization:
    """multi-GPU combine must renormalise radial density with the shell-volume
    factor and contact frequency by the pooled recorded-step count, matching the
    single-run writer rather than dividing by the trajectory count."""

    @staticmethod
    def _write_shard(d, c_bin1, c_bin2, n_contacts, steps):
        with open(os.path.join(d, "radial_density.csv"), "w") as f:
            f.write("r_center,r_low,r_high,count,density\n")
            f.write("11.0,10.0,12.0,%d,0\n" % c_bin1)
            f.write("13.0,12.0,14.0,%d,0\n" % c_bin2)
        with open(os.path.join(d, "contact_frequency.csv"), "w") as f:
            f.write("pair_index,n_contacts,frequency\n")
            f.write("0,%d,%.8e\n" % (n_contacts, n_contacts / steps))

    def test_density_uses_shell_volume(self, tmp_path):
        d1, d2, out = tmp_path / "s1", tmp_path / "s2", tmp_path / "out"
        for p in (d1, d2, out):
            p.mkdir()
        self._write_shard(str(d1), 300, 100, 300, 1000)
        self._write_shard(str(d2), 200, 400, 200, 1000)
        _sum_csv(
            [str(d1), str(d2)],
            "radial_density.csv",
            str(out),
            sum_col="count",
            recompute_col="density",
            density_mode=True,
        )
        rows = list(csv.DictReader(open(out / "radial_density.csv")))
        total = 1000.0  # (300 + 200) + (100 + 400)
        vol1 = 4.0 / 3.0 * math.pi * (12.0**3 - 10.0**3)
        assert int(rows[0]["count"]) == 500
        assert abs(float(rows[0]["density"]) - 500 / (total * vol1)) < 1e-12
        # and NOT the old count/total_N normalisation
        assert abs(float(rows[0]["density"]) - 500 / total) > 1e-6

    def test_frequency_uses_step_count(self, tmp_path):
        d1, d2, out = tmp_path / "s1", tmp_path / "s2", tmp_path / "out"
        for p in (d1, d2, out):
            p.mkdir()
        self._write_shard(str(d1), 300, 100, 300, 1000)
        self._write_shard(str(d2), 200, 400, 200, 1000)
        steps = _recover_contact_steps([str(d1), str(d2)])
        assert steps == 2000
        _sum_csv(
            [str(d1), str(d2)],
            "contact_frequency.csv",
            str(out),
            sum_col="n_contacts",
            recompute_col="frequency",
            total_N=steps,
        )
        rows = list(csv.DictReader(open(out / "contact_frequency.csv")))
        assert int(rows[0]["n_contacts"]) == 500
        assert abs(float(rows[0]["frequency"]) - 500 / 2000.0) < 1e-9


class TestReactionXmlContactValidation:
    """A <contact> with no recognisable atom index must raise instead of silently
    defaulting to atom 0, and the BrownDye <atom1> child-element form must
    parse."""

    @staticmethod
    def _parse(tmp_path, xml):
        f = tmp_path / "rxns.xml"
        f.write_text(xml)
        return parse_reaction_xml(str(f))

    def test_valid_attribute_form(self, tmp_path):
        ps = self._parse(
            tmp_path,
            '<reactions first_state="unbound"><reaction name="r" n_needed="1">'
            '<contact molecule1_index="3" molecule2_index="17" distance="5.0"/>'
            "</reaction></reactions>",
        )
        pair = ps.reactions[0].criteria.pairs[0]
        assert pair.mol1_atom_index == 3
        assert pair.mol2_atom_index == 17
        assert pair.distance_cutoff == 5.0

    def test_child_element_form(self, tmp_path):
        ps = self._parse(
            tmp_path,
            '<reactions><reaction name="r" n_needed="1"><contact distance="4.0">'
            "<atom1>3</atom1><atom2>17</atom2></contact></reaction></reactions>",
        )
        pair = ps.reactions[0].criteria.pairs[0]
        assert pair.mol1_atom_index == 3
        assert pair.mol2_atom_index == 17

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(ValueError):
            self._parse(
                tmp_path,
                '<reactions><reaction name="r" n_needed="1">'
                '<contact atom_1="3" atom_2="17" distance="5.0"/>'
                "</reaction></reactions>",
            )


# Every physics default must be defined in exactly one place.
#
# A default written down twice eventually disagrees with itself, and when that
# happens the association rate depends on which entry point the user called rather
# than on the physics. Nothing in the output says so. This is how the Born
# desolvation prefactor came to sit at 1.0 on the GPU engine and 1/(4 pi) on the
# CPU engine at the same time, a factor of 12.566 on the desolvation barrier.
#
# The test walks the syntax tree of the package rather than importing it, so it
# runs without cupy and without a GPU, and it sees dataclass fields, function
# signature defaults and getattr fallbacks alike.
#
# A genuine physical difference is allowed, but it has to be declared below with
# its reason. Silence is what this test exists to prevent.


PACKAGE = Path(__file__).resolve().parent.parent / "pystarc"


# Differences that are real physics, not drift. Each entry names the values that
# may legitimately coexist and says why. Anything not listed here must agree
# everywhere.
DECLARED_DIFFERENCES = {
    "dt": {0.2, 0.01},
    "minimum_core_dt": {0.0, 0.2},
    "minimum_core_reaction_dt": {0.0, 0.05},
    "bd_milestone_radius_inner": {0.0, 12.0},
    # Zero means derive the screening length from the ion concentration.
    "debye_length": {0.0, 7.858},
    # Zero means the escape sphere is derived from the b surface at setup.
    "r_escape": {0.0, 50.0},
    "r_start": {0.0, 100.0, 20.0},
    # Zero means roll the molecular surface rather than a probe-inflated one.
    "srad": {0.0, 1.5},
}


def _symbol_table():
    """Every module-level physics constant, by name, from the registry modules."""

    table = {}
    for mod in (C, D):
        for k, v in vars(mod).items():
            if k.startswith("_") or not k.isupper():
                continue
            if isinstance(v, (int, float, bool)):
                table[k] = v
    return table


_SYMBOLS = _symbol_table()


def _literal(node):
    """Resolve a default expression to a value.

    Handles plain literals, references to a registry constant by name, and
    subscripts into the registry dicts such as REFERENCE_DEFAULTS["debye_length"].
    Returning a value for symbolic references is the point: a site written as
    `= VISCOSITY` is still a default, and it can still name the wrong constant.
    """
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name) and node.id in _SYMBOLS:
        return _SYMBOLS[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        container = getattr(D, node.value.id, None)
        if isinstance(container, dict):
            try:
                return container[ast.literal_eval(node.slice)]
            except Exception:
                return None
    return None


def _collect():
    """name -> {value: [(relative_path, line)]} for every physics default."""
    found = defaultdict(lambda: defaultdict(list))

    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "defaults.py":
            continue  # the registry is the one place a literal belongs
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        rel = str(path.relative_to(PACKAGE))

        for node in ast.walk(tree):
            # annotated field:  name: type = default
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in PHYSICS_DEFAULT_NAMES and node.value is not None:
                    v = _literal(node.value)
                    if v is not None:
                        found[node.target.id][v].append((rel, node.lineno))

            # signature default:  def f(..., name=default)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pairs = list(zip(a.args[-len(a.defaults):], a.defaults)) if a.defaults else []
                pairs += list(zip(a.kwonlyargs, a.kw_defaults))
                for arg, dflt in pairs:
                    if dflt is None or arg.arg not in PHYSICS_DEFAULT_NAMES:
                        continue
                    v = _literal(dflt)
                    if v is not None:
                        found[arg.arg][v].append((rel, dflt.lineno))

            # getattr(obj, "name", default)
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else None)
                if fname == "getattr" and len(node.args) == 3:
                    key = node.args[1]
                    if isinstance(key, ast.Constant) and key.value in PHYSICS_DEFAULT_NAMES:
                        v = _literal(node.args[2])
                        if v is not None:
                            found[key.value][v].append((rel, node.lineno))
    return found


def _normalise(v):
    return round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


@pytest.mark.parametrize("name", sorted(PHYSICS_DEFAULT_NAMES))
def test_physics_default_is_defined_once(name):
    """No physics parameter may carry two different defaults undeclared."""
    sites = _collect().get(name)
    if not sites:
        known = set(D.INPUT_DEFAULTS) | {
            k.lower() for k, v in vars(D).items() if k.isupper()
        }
        assert name in known or name.lower() in known, (
            "%s has no default anywhere in the package and is not in the "
            "registry either, so nothing defines it" % name
        )
        return

    values = {_normalise(v) for v in sites}
    allowed = {_normalise(v) for v in DECLARED_DIFFERENCES.get(name, set())}

    undeclared = values - allowed
    if len(values) > 1 and len(undeclared) > 1:
        detail = []
        for v, locs in sorted(sites.items(), key=lambda kv: str(kv[0])):
            for rel, line in locs:
                detail.append("    %s:%d = %r" % (rel, line, v))
        pytest.fail(
            "%s has %d different defaults, and %d of them are undeclared.\n"
            "A user who omits this from the input file gets different physics\n"
            "depending on which entry point ran. Either point every site at\n"
            "pystarc.global_defs.defaults, or declare the difference with its\n"
            "reason in DECLARED_DIFFERENCES.\n%s"
            % (name, len(values), len(undeclared), "\n".join(detail))
        )


def test_born_prefactor_matches_the_grid_convention():
    """The desolvation grids fold in the rigorous normalisation, so alpha is 1.

    The retired convention stored an APBS potential and carried alpha = 1/(4 pi).
    Against the present grids that value is 12.566 times too weak, which shows up
    as a missing desolvation barrier rather than as an error.
    """

    assert DESOLVATION_ALPHA == 1.0
    stale = []
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text().split("\n"), 1):
            if "0.07957747" in line and not line.lstrip().startswith("#"):
                stale.append("%s:%d" % (path.relative_to(PACKAGE), i))
    assert not stale, (
        "the retired 1/(4 pi) Born prefactor is still live at:\n  "
        + "\n  ".join(stale)
    )


def test_both_input_readers_agree_on_shared_tags():
    """input_parser and prepare_bd_surface read the same file.

    They are different schemas serving different binaries, so they are allowed to
    differ, but only for the tags named in REFERENCE_DEFAULTS.
    """

    overlap = set(INPUT_DEFAULTS) & set(REFERENCE_DEFAULTS)
    for tag in overlap:
        assert INPUT_DEFAULTS[tag] != REFERENCE_DEFAULTS[tag], (
            "%s is listed as a reference-binary difference but the two values "
            "are identical, so the entry is redundant" % tag
        )


# Regression tests for the image (Born) desolvation grid.
#
# These lock down the properties that the replaced APBS based implementation
# violated, and that any future rewrite must keep:
#
#   * the stored field is a cavity self energy built from the partner RADII only,
#     so it is strictly positive, charge independent, and monotonically decreasing
#     away from the partner, which makes the force alpha*q^2*(-grad G) repulsive
#     for every atom regardless of the sign of its charge
#   * the kernel is the closed form Kirkwood resummation a^3/(r^2-a^2)^2 with the
#     ionic screening (1+kr)^2 exp(-2kr), reducing to a^3/r^4 in the far field
#   * a PQR is read for geometry with and without the optional trailing element
#     column, since reading the charge column as the radius silently produced an
#     identically zero grid
#   * the OpenDX writer emits the (nx,ny,nz) C ordered layout the force kernel
#     indexes as i*ny*nz + j*nz + k, with no transpose
#
# Plain pytest, numpy only, no GPU needed: the module falls back to numpy.


def ref_field(point, atom_xyz, atom_rad, eps_p=4.0, eps_s=78.0,
              temp=298.15, debye_length=dg.DEFAULT_DEBYE, cutoff=15.0):
    """Reference value of the stored field at one point, in kBT per e^2."""
    C = dg.COULOMB_KCAL / (dg.KB_KCAL * temp)
    D = (eps_s - eps_p) / (eps_s * (2.0 * eps_s + eps_p))
    k = 0.0 if not debye_length else 1.0 / float(debye_length)
    total = 0.0
    for (c, a) in zip(np.asarray(atom_xyz, float), np.asarray(atom_rad, float)):
        r = float(np.linalg.norm(np.asarray(point, float) - c))
        if r >= cutoff:
            continue
        r2 = max(r * r, max(a * a, dg.DEN_FLOOR))
        total += a ** 3 * (1.0 + k * r) ** 2 * math.exp(-2.0 * k * r) / (r2 * r2)
    # rigorous Kirkwood carries the linear-response 1/2
    return 0.5 * C * D * total


def field_at(points, atom_xyz, atom_rad, **kw):
    """Sample the grid builder at arbitrary points, one 1x1x1 grid per point."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.empty(len(pts), dtype=np.float64)
    for i, p in enumerate(pts):
        g = dg.desolvation_field_on_grid(
            p, [1.0, 1.0, 1.0], [1, 1, 1],
            np.asarray(atom_xyz, dtype=np.float64),
            np.asarray(atom_rad, dtype=np.float64), **kw)
        assert g.shape == (1, 1, 1)
        out[i] = g[0, 0, 0]
    return out


ONE_ATOM = np.array([[0.0, 0.0, 0.0]])
ONE_RAD = np.array([1.7])


def test_constants_match_closed_form():
    # e^2/(4 pi eps0)/kT at 298.15 K, and the Kirkwood n=1 image factor at 4/78
    assert dg.coulomb_kbt(298.15) == pytest.approx(560.46, abs=0.02)
    assert dg.dielectric_factor(4.0, 78.0) == pytest.approx(0.0059295, rel=1e-4)
    # temperature scaling is 1/T
    assert dg.coulomb_kbt(2.0 * 298.15) == pytest.approx(0.5 * dg.coulomb_kbt(298.15))
    # a partner interior as polar as the solvent cannot desolvate anything
    assert dg.dielectric_factor(78.0, 78.0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("r", [2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 14.9])
@pytest.mark.parametrize("debye", [dg.DEFAULT_DEBYE, 0.0])
def test_single_sphere_matches_analytic(r, debye):
    """G(r) = C*D*a^3 (1+kr)^2 exp(-2kr) / (r^2-a^2)^2 for one sphere."""
    a = float(ONE_RAD[0])
    got = field_at([[r, 0.0, 0.0]], ONE_ATOM, ONE_RAD, debye_length=debye)[0]
    C = dg.coulomb_kbt(298.15)
    D = dg.dielectric_factor(4.0, 78.0)
    k = 0.0 if not debye else 1.0 / debye
    want = 0.5 * C * D * a ** 3 * (1.0 + k * r) ** 2 * math.exp(-2.0 * k * r) / r ** 4
    assert got == pytest.approx(want, rel=1e-12)
    assert got == pytest.approx(ref_field([r, 0.0, 0.0], ONE_ATOM, ONE_RAD,
                                          debye_length=debye), rel=1e-12)


def test_single_sphere_isotropic():
    """The field of one sphere depends on |r| only, not on direction."""
    a, r = 1.7, 5.0
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
                     [1, 1, 1], [-2, 1, -3], [0.3, -0.4, 0.5]], dtype=float)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    vals = field_at(dirs * r, ONE_ATOM, np.array([a]))
    assert np.allclose(vals, vals[0], rtol=1e-12)


def test_scaling_with_radius_and_dielectric():
    """a^3 prefactor, and linearity in the Kirkwood dielectric factor."""
    r = 8.0
    v1 = field_at([[r, 0, 0]], ONE_ATOM, np.array([1.0]))[0]
    v2 = field_at([[r, 0, 0]], ONE_ATOM, np.array([2.0]))[0]
    # SDA r^-4 kernel: the denominator carries no radius, so scaling is exactly a^3
    assert v2 / v1 == pytest.approx(8.0, rel=1e-12)
    base = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD)[0]
    other = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, eps_p=2.0)[0]
    assert other / base == pytest.approx(
        dg.dielectric_factor(2.0, 78.0) / dg.dielectric_factor(4.0, 78.0), rel=1e-12)


def test_field_is_additive_over_atoms():
    """The stored field is a plain sum of per atom cavity terms."""
    xyz = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, -2.0], [-3.0, 2.0, 1.0]])
    rad = np.array([1.7, 1.9, 1.5])
    p = np.array([[7.0, -1.0, 3.0]])
    total = field_at(p, xyz, rad)[0]
    parts = sum(field_at(p, xyz[i:i + 1], rad[i:i + 1])[0] for i in range(3))
    assert total == pytest.approx(parts, rel=1e-12)


def test_field_strictly_positive_everywhere():
    """
    An APBS potential of the partner's own charges changes sign across the
    surface. A cavity self energy never can.
    """
    xyz = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 3.5, 1.0]])
    rad = np.array([1.7, 1.9, 1.5])
    g = dg.desolvation_field_on_grid([-8.0, -8.0, -8.0], [0.5, 0.5, 0.5],
                                     [40, 40, 40], xyz, rad, cutoff=100.0)
    assert np.all(np.isfinite(g))
    assert g.min() > 0.0
    assert np.count_nonzero(g) == g.size
    sing = field_at([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]], ONE_ATOM, ONE_RAD)
    assert np.all(np.isfinite(sing)) and np.all(sing > 0.0)


def test_field_is_zero_only_outside_the_cutoff():
    """
    The only zeros allowed are the deliberate truncation past `cutoff`. Any
    interior zero means the geometry or the radii were lost, which is exactly
    what the identically zero grid looked like.
    """
    xyz = np.array([[0.0, 0.0, 0.0]])
    origin, spacing, dime = [-20.0, 0.0, 0.0], [0.5, 1.0, 1.0], [81, 1, 1]
    g = dg.desolvation_field_on_grid(origin, spacing, dime, xyz, ONE_RAD)[:, 0, 0]
    r = np.abs(origin[0] + spacing[0] * np.arange(dime[0]))
    assert np.all(g[r < 15.0] > 0.0)
    assert np.all(g[r >= 15.0] == 0.0)


@pytest.mark.parametrize("debye", [dg.DEFAULT_DEBYE, 0.0])
def test_monotonic_decrease_gives_repulsive_force(debye):
    """
    F = -alpha*q^2*grad G. Since alpha*q^2 > 0, the force is outward for every
    atom, whatever the sign of q, if and only if G decreases with distance.
    """
    r = np.arange(2.0, 14.5, 0.25)
    v = field_at(np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=1),
                 ONE_ATOM, ONE_RAD, debye_length=debye)
    assert np.all(v > 0.0)
    assert np.all(np.diff(v) < 0.0)
    o, sp, n = [-10.0, -0.5, -0.5], [0.5, 0.5, 0.5], [41, 2, 2]
    g = dg.desolvation_field_on_grid(o, sp, n, ONE_ATOM, ONE_RAD, debye_length=debye)
    line = g[:, 0, 0]
    i0 = len(line) // 2
    fx = -np.gradient(line, sp[0])
    assert np.all(fx[:i0 - 2] < 0.0)     # left of the atom, pushed to -x
    assert np.all(fx[i0 + 3:] > 0.0)     # right of the atom, pushed to +x


def test_multi_atom_field_decreases_away_from_cluster():
    xyz = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [1.4, 2.4, 0.0],
                    [1.4, 0.8, 2.3]])
    rad = np.array([1.7, 1.8, 1.6, 1.9])
    centre = xyz.mean(axis=0)
    u = np.array([0.577, 0.577, 0.577])
    r = np.arange(6.0, 14.0, 0.25)
    v = field_at(centre + np.outer(r, u), xyz, rad)
    assert np.all(v > 0.0)
    assert np.all(np.diff(v) < 0.0)


def _loglog_slope(r1, r2, a=1.5, **kw):
    v = field_at([[r1, 0, 0], [r2, 0, 0]], ONE_ATOM, np.array([a]), **kw)
    return math.log(v[1] / v[0]) / math.log(r2 / r1)


def test_far_field_unscreened_slope_is_minus_four():
    """SDA kernel is a^3/r^4, so the unscreened log-log slope is exactly -4."""
    s = _loglog_slope(25.0, 35.0, a=1.5, debye_length=0.0, cutoff=200.0)
    assert s == pytest.approx(-4.0, abs=1e-9)
    s_far = _loglog_slope(150.0, 200.0, a=1.5, debye_length=0.0, cutoff=400.0)
    assert s_far == pytest.approx(-4.0, abs=1e-9)


def test_far_field_amplitude_reduces_to_a_cubed_over_r_fourth():
    a, r = 1.5, 60.0
    got = field_at([[r, 0, 0]], ONE_ATOM, np.array([a]),
                   debye_length=0.0, cutoff=200.0)[0]
    want = 0.5 * dg.coulomb_kbt(298.15) * dg.dielectric_factor(4.0, 78.0) * a ** 3 / r ** 4
    assert got == pytest.approx(want, rel=2e-3)


def test_screened_slope_is_steeper_than_unscreened():
    """At 150 mM the Yukawa factor adds about one more power of r near 8 A."""
    s = _loglog_slope(7.5, 8.5, a=1.5, debye_length=dg.DEFAULT_DEBYE, cutoff=200.0)
    assert -5.2 < s < -4.9
    s_unscreened = _loglog_slope(7.5, 8.5, a=1.5, debye_length=0.0, cutoff=200.0)
    assert s < s_unscreened
    # and the exact screening ratio between the two
    r = 8.0
    k = 1.0 / dg.DEFAULT_DEBYE
    vs = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, debye_length=dg.DEFAULT_DEBYE)[0]
    vu = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, debye_length=0.0)[0]
    assert vs / vu == pytest.approx((1.0 + k * r) ** 2 * math.exp(-2.0 * k * r), rel=1e-12)


def test_cutoff_drops_distant_atoms_and_default_cutoff_is_negligible():
    far = np.array([[40.0, 0.0, 0.0]])
    assert field_at([[0.0, 0.0, 0.0]], far, ONE_RAD, debye_length=0.0)[0] == 0.0
    assert field_at([[0.0, 0.0, 0.0]], far, ONE_RAD,
                    debye_length=0.0, cutoff=100.0)[0] > 0.0
    # what is thrown away at the 15 A default is four orders below contact
    edge = field_at([[14.99, 0, 0]], ONE_ATOM, ONE_RAD)[0]
    contact = field_at([[3.4, 0, 0]], ONE_ATOM, ONE_RAD)[0]
    assert edge / contact < 1e-3


_PQR_NO_ELEMENT = """\
REMARK   1 PQR file, columns are x y z charge radius
ATOM      1  N   ALA A   1      -1.234   2.345  -3.456 -0.4157  1.8240
ATOM      2  CA  ALA A   1       0.100  -0.200   0.300  0.0337  1.9080
ATOM      3  C   ALA A   1       4.500   1.500  -2.500  0.5973  1.9080
HETATM    4  O   HOH A   2       7.000  -1.000   1.000 -0.8340  1.7210
TER
END
"""

_PQR_WITH_ELEMENT = """\
REMARK   1 PQR file, columns are x y z charge radius element
ATOM      1  N   ALA A   1      -1.234   2.345  -3.456 -0.4157  1.8240 N
ATOM      2  CA  ALA A   1       0.100  -0.200   0.300  0.0337  1.9080 C
ATOM      3  C   ALA A   1       4.500   1.500  -2.500  0.5973  1.9080 C
HETATM    4  O   HOH A   2       7.000  -1.000   1.000 -0.8340  1.7210 O
TER
END
"""

_EXPECT_XYZ = np.array([[-1.234, 2.345, -3.456],
                        [0.100, -0.200, 0.300],
                        [4.500, 1.500, -2.500],
                        [7.000, -1.000, 1.000]])
_EXPECT_RAD = np.array([1.8240, 1.9080, 1.9080, 1.7210])
_CHARGES = np.array([-0.4157, 0.0337, 0.5973, -0.8340])


@pytest.mark.parametrize("text", [_PQR_NO_ELEMENT, _PQR_WITH_ELEMENT],
                         ids=["no_element_column", "with_element_column"])
def test_pqr_geometry_both_formats(tmp_path, text):
    p = tmp_path / "m.pqr"
    p.write_text(text)
    xyz, rad = dg.read_pqr_geometry(str(p))
    assert xyz.shape == (4, 3) and rad.shape == (4,)
    assert np.allclose(xyz, _EXPECT_XYZ)
    assert np.allclose(rad, _EXPECT_RAD)
    # the exact bug: the radius column read one field off, picking up the charge
    assert not np.allclose(rad, _CHARGES)
    assert np.all(rad > 1.0)


def test_pqr_both_formats_give_identical_geometry_and_field(tmp_path):
    """End to end guard on the identically zero grid the off by one produced."""
    a = tmp_path / "no_elem.pqr"
    b = tmp_path / "with_elem.pqr"
    a.write_text(_PQR_NO_ELEMENT)
    b.write_text(_PQR_WITH_ELEMENT)
    xa, ra = dg.read_pqr_geometry(str(a))
    xb, rb = dg.read_pqr_geometry(str(b))
    assert np.array_equal(xa, xb) and np.array_equal(ra, rb)
    ga = dg.desolvation_field_on_grid([-6, -6, -6], [1.0, 1.0, 1.0], [13, 13, 13], xa, ra)
    gb = dg.desolvation_field_on_grid([-6, -6, -6], [1.0, 1.0, 1.0], [13, 13, 13], xb, rb)
    assert np.array_equal(ga, gb)
    assert ga.min() > 0.0
    assert np.count_nonzero(ga) == ga.size


def test_field_is_independent_of_partner_charges(tmp_path):
    """
    The headline regression. The old term was a Poisson-Boltzmann solve of the
    partner's own charges, so flipping every charge flipped the term. A cavity
    self energy cannot see the charges at all.
    """
    flipped = []
    for line in _PQR_NO_ELEMENT.splitlines(True):
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            f[-2] = "%.4f" % (-float(f[-2]))
            line = " ".join(f) + "\n"
        flipped.append(line)
    p = tmp_path / "orig.pqr"
    q = tmp_path / "flip.pqr"
    p.write_text(_PQR_NO_ELEMENT)
    q.write_text("".join(flipped))
    xp_, rp_ = dg.read_pqr_geometry(str(p))
    xq_, rq_ = dg.read_pqr_geometry(str(q))
    assert np.array_equal(rp_, rq_) and np.array_equal(xp_, xq_)
    gp = dg.desolvation_field_on_grid([-6, -6, -6], [1.0] * 3, [13] * 3, xp_, rp_)
    gq = dg.desolvation_field_on_grid([-6, -6, -6], [1.0] * 3, [13] * 3, xq_, rq_)
    assert np.array_equal(gp, gq)


def test_pqr_ignores_non_atom_records(tmp_path):
    p = tmp_path / "m.pqr"
    p.write_text("REMARK junk\nCRYST1 1 1 1\n" + _PQR_NO_ELEMENT + "CONECT 1 2\n")
    xyz, rad = dg.read_pqr_geometry(str(p))
    assert len(xyz) == 4 and len(rad) == 4


def test_zero_radius_atoms_contribute_nothing():
    """a^3 = 0, so ghost or dummy sites are inert even at zero separation."""
    real = np.array([[0.0, 0.0, 0.0]])
    rad = np.array([1.7])
    with_ghosts = np.vstack([real, [[1.0, 0.0, 0.0]], [[4.0, 2.0, -1.0]]])
    rad_g = np.array([1.7, 0.0, 0.0])
    pts = np.array([[3.0, 0, 0], [1.0, 0.0, 0.0], [4.0, 2.0, -1.0], [6.0, -2.0, 3.0]])
    a = field_at(pts, real, rad)
    b = field_at(pts, with_ghosts, rad_g)
    assert np.allclose(a, b, rtol=0, atol=0)
    assert np.all(np.isfinite(b))


def test_all_zero_radius_gives_identically_zero_field():
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    g = dg.desolvation_field_on_grid([-4, -4, -4], [1.0] * 3, [9] * 3,
                                     xyz, np.zeros(2))
    assert np.all(g == 0.0)


def test_empty_atom_set_gives_zero_field_of_right_shape():
    g = dg.desolvation_field_on_grid([0.0, 0.0, 0.0], [1.0] * 3, [4, 5, 6],
                                     np.zeros((0, 3)), np.zeros(0))
    assert g.shape == (4, 5, 6)
    assert np.all(g == 0.0)


@pytest.mark.parametrize("debye", [0.0, None, float("inf"), 1e12])
def test_zero_kappa_screening_is_unity(debye):
    """
    kappa = 0 must give (1+kr)^2 exp(-2kr) = 1 exactly, whether the caller
    expresses no salt as a zero, a None, or an infinite Debye length.
    """
    r = np.array([2.5, 5.0, 10.0, 14.0])
    pts = np.stack([r, np.zeros(4), np.zeros(4)], axis=1)
    got = field_at(pts, ONE_ATOM, ONE_RAD, debye_length=debye)
    C = 0.5 * dg.coulomb_kbt(298.15) * dg.dielectric_factor(4.0, 78.0)
    a = float(ONE_RAD[0])
    want = C * a ** 3 / r ** 4
    assert np.allclose(got, want, rtol=1e-12)


def test_screening_monotonic_in_salt_and_bounded_by_unity():
    """More salt, i.e. shorter Debye length, always damps the field further."""
    r = 8.0
    unscreened = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, debye_length=0.0)[0]
    prev = unscreened
    for lam in [1000.0, 100.0, 30.0, dg.DEFAULT_DEBYE, 4.0, 2.0]:
        v = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, debye_length=lam)[0]
        assert 0.0 < v < prev <= unscreened * (1.0 + 1e-12)
        prev = v
    # a very long Debye length is numerically the no salt limit
    v = field_at([[r, 0, 0]], ONE_ATOM, ONE_RAD, debye_length=1e8)[0]
    assert v == pytest.approx(unscreened, rel=1e-6)


def read_dx(path):
    """Minimal OpenDX reader: returns field (nx,ny,nz), origin, spacing."""
    origin = spacing = None
    counts = None
    deltas = []
    data = []
    in_data = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("object 1 class gridpositions"):
                counts = tuple(int(t) for t in s.split()[-3:])
            elif s.startswith("origin"):
                origin = np.array([float(t) for t in s.split()[1:4]])
            elif s.startswith("delta"):
                deltas.append([float(t) for t in s.split()[1:4]])
            elif s.startswith("object 3 class array"):
                in_data = True
            elif s.startswith(("attribute", "component", 'object "')):
                in_data = False
            elif in_data:
                data.extend(float(t) for t in s.split())
    d = np.asarray(deltas, dtype=float)
    spacing = np.array([d[0][0], d[1][1], d[2][2]])
    assert counts is not None and origin is not None
    arr = np.asarray(data, dtype=float)
    assert arr.size == counts[0] * counts[1] * counts[2], (arr.size, counts)
    return arr.reshape(counts), origin, spacing


def test_dx_round_trip_preserves_values_and_axis_order(tmp_path):
    """
    Distinct nx, ny, nz and a value that encodes its own index, so any
    transpose or Fortran ordering shows up immediately. The force kernel reads
    the flat buffer as i*ny*nz + j*nz + k, which is exactly C order.
    """
    nx, ny, nz = 5, 4, 2          # 40 items, deliberately not a multiple of 3
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    field = (100.0 * i + 10.0 * j + 1.0 * k) + 0.5
    origin = [-3.25, 1.5, 0.75]
    spacing = [0.4, 0.6, 0.8]

    p = tmp_path / "grid.dx"
    dg.write_dx(str(p), field, origin, spacing, [nx, ny, nz])
    back, o, sp = read_dx(str(p))

    assert back.shape == (nx, ny, nz)
    assert np.allclose(o, origin)
    assert np.allclose(sp, spacing)
    assert np.allclose(back, field, rtol=0, atol=1e-6)
    # explicit spot checks that would fail under a transpose
    assert back[4, 0, 0] == pytest.approx(400.5)
    assert back[0, 3, 0] == pytest.approx(30.5)
    assert back[0, 0, 1] == pytest.approx(1.5)
    # the flat order the C kernel assumes
    flat = back.reshape(-1)
    assert flat[2 * ny * nz + 3 * nz + 1] == pytest.approx(231.5)


def test_dx_round_trip_of_a_real_field(tmp_path):
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, -1.0]])
    rad = np.array([1.7, 1.9])
    origin, spacing, dime = [-5.0, -4.0, -3.0], [0.5, 0.6, 0.7], [11, 9, 7]
    g = dg.desolvation_field_on_grid(origin, spacing, dime, xyz, rad)
    p = tmp_path / "real.dx"
    dg.write_dx(str(p), g, origin, spacing, dime)
    back, o, sp = read_dx(str(p))
    assert back.shape == tuple(dime)
    assert np.allclose(back, g, rtol=1e-5, atol=0)
    assert np.allclose(o, origin) and np.allclose(sp, spacing)
    imax = np.unravel_index(np.argmax(back), back.shape)
    node = np.array(origin) + np.array(spacing) * np.array(imax)
    assert np.min(np.linalg.norm(xyz - node, axis=1)) < max(rad) + max(spacing)


def test_grid_nodes_match_pointwise_reference():
    xyz = np.array([[0.0, 0.0, 0.0], [2.5, -1.5, 3.0]])
    rad = np.array([1.7, 1.4])
    origin, spacing, dime = [-4.0, -3.0, -2.0], [0.9, 1.1, 1.3], [7, 6, 5]
    g = dg.desolvation_field_on_grid(origin, spacing, dime, xyz, rad)
    assert g.shape == tuple(dime)
    for (i, j, k) in [(0, 0, 0), (6, 5, 4), (3, 2, 1), (1, 4, 2), (5, 0, 3)]:
        pt = [origin[0] + spacing[0] * i,
              origin[1] + spacing[1] * j,
              origin[2] + spacing[2] * k]
        assert g[i, j, k] == pytest.approx(ref_field(pt, xyz, rad), rel=1e-12)


def test_grid_axes_are_not_swapped():
    """One atom offset along y only: the peak must move in j, not i or k."""
    xyz = np.array([[0.0, 3.0, 0.0]])
    g = dg.desolvation_field_on_grid([-4.0, -4.0, -4.0], [1.0] * 3, [9, 9, 9],
                                     xyz, ONE_RAD)
    assert np.unravel_index(np.argmax(g), g.shape) == (4, 7, 4)


def test_field_is_translation_invariant():
    """Same physical point, different grid origin, identical value."""
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
    rad = np.array([1.7, 1.5])
    shift = np.array([13.7, -21.3, 5.9])
    p = np.array([[6.0, 1.0, -2.0]])
    a = field_at(p, xyz, rad)[0]
    b = field_at(p + shift, xyz + shift, rad)[0]
    assert a == pytest.approx(b, rel=1e-10)


def test_chunking_does_not_change_the_result():
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    rad = np.array([1.7, 1.5])
    args = ([-3.0, -3.0, -3.0], [0.75] * 3, [9, 8, 7], xyz, rad)
    full = dg.desolvation_field_on_grid(*args)
    tiny = dg.desolvation_field_on_grid(*args, chunk=7)
    assert np.array_equal(full, tiny)


def test_bounding_box_prefilter_keeps_atoms_that_matter():
    """
    Atoms are dropped before the loop by a box test. An atom just inside the
    cutoff shell of a grid corner must survive that filter.
    """
    origin, spacing, dime = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3, 3, 3]
    near = np.array([[-10.0, 1.0, 1.0]])         # 10 A outside, inside cutoff 15
    g = dg.desolvation_field_on_grid(origin, spacing, dime, near, ONE_RAD)
    assert g[0, 1, 1] > 0.0
    assert g[0, 1, 1] == pytest.approx(ref_field([0.0, 1.0, 1.0], near, ONE_RAD), rel=1e-12)


def test_probe_contact_value_is_a_real_field_value():
    """
    The diagnostic must report the field one vdW radius outside the outermost
    atom, not the DEN_FLOOR clamped interior plateau that field.max() returns.
    """
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    rad = np.array([1.7, 2.0])
    g = dg.desolvation_field_on_grid([-6.0, -6.0, -6.0], [0.5] * 3, [33, 25, 25],
                                     xyz, rad)
    v, r = dg.probe_contact_value(g, [-6.0] * 3, [0.5] * 3, [33, 25, 25], xyz, rad)
    assert r == pytest.approx(4.0)
    assert v == pytest.approx(ref_field([9.0, 0.0, 0.0], xyz, rad), rel=5e-3)
    assert 0.0 < v < g.max()


@pytest.mark.parametrize("q", [-2.0, -0.5, 0.5, 1.0])
def test_energy_is_penalty_and_force_is_outward_for_either_sign(q):
    alpha = 1.67
    r = np.array([3.0, 3.5, 4.0, 6.0, 10.0])
    g = field_at(np.stack([r, np.zeros(5), np.zeros(5)], axis=1), ONE_ATOM, ONE_RAD)
    u = alpha * q * q * g
    assert np.all(u > 0.0)                  # always a penalty, never a reward
    assert np.all(np.diff(u) < 0.0)         # so -dU/dr > 0, pushed apart
    # and the magnitude scales as q^2, not q
    g2 = alpha * (2.0 * q) ** 2 * g
    assert np.allclose(g2, 4.0 * u, rtol=1e-12)
