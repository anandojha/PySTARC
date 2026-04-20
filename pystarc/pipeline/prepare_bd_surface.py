"""
PySTARC - BD surface preparation pipeline
============================================
Generates a complete b_surface/ directory from a PDB + parm7 file.

Pipeline steps:
  1. Strip water from PDB -> gas-phase complex
  2. cpptraj: PDB -> inpcrd
  3. ambpdb:  parm7 + inpcrd -> combined PQR
  4. Split combined PQR -> receptor.pqr + ligand.pqr
  5. Centre each molecule at origin (subtract centroid)
  6. Assign each ligand atom its own residue number (pqr_resid_for_each_atom)
  7. Inject GHO ghost atom at (0,0,0) into both PQRs
  8. Convert PQRs to the reference implementation XML format -> receptor.xml + ligand.xml
  9. Generate 3-level APBS inputs (coarse/medium/fine) -> run APBS -> DX files
  10. Compute Debye length from ion concentration
  11. Generate rxns.xml (milestone reaction criterion)
  12. Generate input.xml (the reference implementation nam_simulation ready)

Usage:
  python prepare_bd_surface.py pystarc_input.xml
  # or called from run_pystarc.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from itertools import groupby
from pathlib import Path
import numpy as np
import subprocess
import shutil
import math
import sys
import os

# Constants
# Comprehensive solvent/ion residue set - covers TIP3P, TIP4P, SPC/E,
# CHARMM, GROMOS, AMBER, GROMACS naming conventions for water and ions.
SOLVENT_RESIDUES = {
    # Water models
    "WAT",
    "HOH",
    "TIP",
    "TIP3",
    "TIP4",
    "TIP5",
    "SOL",
    "TP3",
    "SPC",
    "SPCE",
    "T3P",
    # Monovalent ions (AMBER, CHARMM, GROMACS names)
    "Na+",
    "Cl-",
    "K+",
    "Na",
    "Cl",
    "K",
    "NA",
    "CL",
    "POT",
    "SOD",
    "CLA",
    "NA+",
    "CL-",
    # Divalent ions
    "Mg+",
    "Ca+",
    "Zn+",
    "Mg",
    "Ca",
    "Zn",
    "MG",
    "CA",
    "ZN",
    "MG2",
    "CAL",
    "ZN2",
}
# Debye length formula: lambda_D = sqrt(eps0 * eps_r * kB * T / (2 * Na * e^2 * I))
# At 298.15 K, water, 1:1 salt: lambda_D(A) = 3.04 / sqrt(c_molar)
DEBYE_PREFACTOR_ANGSTROM = 3.04  # A * sqrt(mol/L) at 298 K, water, 1:1 electrolyte


# Configuration
@dataclass
class BDSurfaceConfig:
    """All parameters for the BD surface preparation pipeline."""

    # Input files
    # [REQUIRED] - set these for your system
    pdb: Path = Path("complex.pdb")  # PDB file (water ok, stripped auto)
    parm7: Path = Path("complex.parm7")  # AMBER topology
    receptor_resname: str = ""  # 3-letter receptor residue name (e.g. "MGO", "HSP")
    ligand_resname: str = ""  # 3-letter ligand residue name  (e.g. "APN", "BEN")
    # Output
    work_dir: Path = Path("b_surface")
    # PQR options
    ligand_atom_per_residue: bool = True
    inject_gho: bool = True
    # APBS
    pdie: float = 4.0
    sdie: float = 78.0
    apbs_fine_spacing: float = 0.5
    apbs_n_grids: int = 3
    apbs_srfm: str = "smol"
    apbs_chgm: str = "spl2"
    apbs_srad: float = 1.5
    temperature: float = 298.15
    # Ionic strength
    ion_concentration: float = 0.150  # mol/L
    ion_type: str = "NaCl"
    debye_length: float = 0.0  # 0.0 = compute from concentration
    # Reaction criterion
    bd_milestone_radius: float = 30.0  # A - b-sphere (>= 3×(r_rec+r_lig))
    bd_milestone_radius_inner: float = 12.0  # A (0.0 = disabled)
    # BD simulation
    n_trajectories: int = 10000
    max_n_steps: int = 100000000
    seed: int = 11111113
    n_threads: int = 24
    gpu: bool = True
    hydrodynamic_interactions: bool = True
    minimum_core_dt: float = 0.2
    minimum_core_reaction_dt: float = 0.05
    desolvation_parameter: float = 1.0
    relative_viscosity: float = 1.0
    confidence_interval: float = 0.95

    def compute_debye_length(self) -> float:
        """Compute Debye length in Angstroms from ion concentration."""
        if self.debye_length > 0:
            return self.debye_length
        if self.ion_concentration <= 0:
            return 1.79769e308  # infinity - no screening
        # 1:1 electrolyte (NaCl, KCl): lambda_D = 3.04/sqrt(c) A at 298 K
        # Temperature correction: multiply by sqrt(T/298.15)
        t_factor = math.sqrt(self.temperature / 298.15)
        return DEBYE_PREFACTOR_ANGSTROM * t_factor / math.sqrt(self.ion_concentration)

    @property
    def kT(self) -> float:
        """kT in units of kBT (dimensionless, with small T correction)."""
        # Match SEEKR: kT = kB*T / kB*298.15 = T/298.15 * 1.0
        return self.temperature / 298.15


def parse_config(xml_path: Path) -> BDSurfaceConfig:
    """Parse pystarc_input.xml into a BDSurfaceConfig."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get(tag, default=None, cast=str):
        node = root.find(tag)
        if node is None or not (node.text or "").strip():
            return default
        val = node.text.strip()
        if cast is bool:
            return val.lower() in ("true", "1", "yes")
        return cast(val)

    return BDSurfaceConfig(
        pdb=Path(get("pdb", "hostguest.pdb")),
        parm7=Path(get("parm7", "hostguest.parm7")),
        receptor_resname=get("receptor_resname", "MGO"),
        ligand_resname=get("ligand_resname", "APN"),
        work_dir=Path(get("work_dir", "b_surface")),
        ligand_atom_per_residue=get("ligand_atom_per_residue", True, bool),
        inject_gho=get("inject_gho", True, bool),
        pdie=get("pdie", 4.0, float),
        sdie=get("sdie", 78.0, float),
        apbs_fine_spacing=get("apbs_fine_spacing", 0.5, float),
        apbs_n_grids=get("apbs_n_grids", 3, int),
        apbs_srfm=get("apbs_srfm", "smol"),
        apbs_chgm=get("apbs_chgm", "spl2"),
        apbs_srad=get("apbs_srad", 1.5, float),
        temperature=get("temperature", 298.15, float),
        ion_concentration=get("ion_concentration", 0.150, float),
        ion_type=get("ion_type", "NaCl"),
        debye_length=get("debye_length", 0.0, float),
        bd_milestone_radius=get("bd_milestone_radius", 30.0, float),
        bd_milestone_radius_inner=get("bd_milestone_radius_inner", 12.0, float),
        n_trajectories=get("n_trajectories", 10000, int),
        max_n_steps=get("max_n_steps", 100000000, int),
        seed=get("seed", 11111113, int),
        n_threads=get("n_threads", 24, int),
        gpu=get("gpu", True, bool),
        hydrodynamic_interactions=get("hydrodynamic_interactions", True, bool),
        minimum_core_dt=get("minimum_core_dt", 0.2, float),
        minimum_core_reaction_dt=get("minimum_core_reaction_dt", 0.05, float),
        desolvation_parameter=get("desolvation_parameter", 1.0, float),
        relative_viscosity=get("relative_viscosity", 1.0, float),
        confidence_interval=get("confidence_interval", 0.95, float),
    )


# PQR utilities
@dataclass
class PQRAtom:
    serial: int
    name: str
    resname: str
    resid: int
    x: float
    y: float
    z: float
    charge: float
    radius: float
    record: str = "HETATM"

    def to_pqr_line(self) -> str:
        # Always use HETATM for non-standard residues
        # Prevents parser failures
        return (
            f"HETATM{self.serial:5d}  {self.name:<4s} {self.resname:<4s}"
            f"{self.resid:5d}    {self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
            f"  {self.charge:7.4f}  {self.radius:6.4f}\n"
        )


def read_pqr(path: Path) -> List[PQRAtom]:
    """Parse a PQR file into a list of PQRAtom.

    Delegates to the canonical PQR parser in pystarc.structures.pqr_io
    and reshapes the result into the local PQRAtom dataclass used by
    the b-surface preparation pipeline.
    """
    from pystarc.structures.pqr_io import parse_pqr_records
    return [
        PQRAtom(
            serial=r.serial,
            name=r.name,
            resname=r.resname,
            resid=r.resid,
            x=r.x,
            y=r.y,
            z=r.z,
            charge=r.charge,
            radius=r.radius,
            record=r.record_type,
        )
        for r in parse_pqr_records(path)
    ]


def write_pqr(atoms: List[PQRAtom], path: Path):
    with open(path, "w") as f:
        for a in atoms:
            f.write(a.to_pqr_line())
        f.write("END\n")


def centroid(atoms: List[PQRAtom]) -> np.ndarray:
    coords = np.array([[a.x, a.y, a.z] for a in atoms if a.name != "GHO"])
    return coords.mean(axis=0)


def centre_at_origin(atoms: List[PQRAtom]) -> List[PQRAtom]:
    """Subtract centroid from all atom positions (excluding GHO)."""
    ctr = centroid(atoms)
    for a in atoms:
        a.x -= ctr[0]
        a.y -= ctr[1]
        a.z -= ctr[2]
    return atoms


def assign_each_atom_own_residue(atoms: List[PQRAtom]) -> List[PQRAtom]:
    """Give each atom its own residue number (pqr_resid_for_each_atom)."""
    for i, a in enumerate(atoms):
        a.resid = i + 1
    return atoms


def inject_gho(atoms: List[PQRAtom]) -> List[PQRAtom]:
    """Append GHO ghost atom at (0,0,0) with zero charge and radius."""
    next_serial = max(a.serial for a in atoms) + 1
    next_resid = max(a.resid for a in atoms) + 1
    atoms.append(
        PQRAtom(
            serial=next_serial,
            name="GHO",
            resname="GHO",
            resid=next_resid,
            x=0.0,
            y=0.0,
            z=0.0,
            charge=0.0,
            radius=0.0,
            record="HETATM",
        )
    )
    return atoms


def pqr_to_xml(atoms: List[PQRAtom], path: Path):
    """
    Convert PQR to the atom XML format.
    Matches output of the pqr2xml tool.
    Each residue groups its atoms; with ligand_atom_per_residue=True
    each atom is its own residue.
    """
    lines = ["<roottag>\n"]
    # Group atoms by residue
    for resid, group in groupby(atoms, key=lambda a: a.resid):
        group = list(group)
        resname = group[0].resname
        lines.append(f"  <residue>\n")
        lines.append(f"    <n>{resname}</n>\n")
        lines.append(f"    <number>{resid}</number>\n")
        for a in group:
            lines.append(f"    <atom>\n")
            lines.append(f"      <n>{a.name}</n>\n")
            lines.append(f"      <number>{a.serial}</number>\n")
            lines.append(f"      <x>{a.x:.6f}</x>\n")
            lines.append(f"      <y>{a.y:.6f}</y>\n")
            lines.append(f"      <z>{a.z:.6f}</z>\n")
            lines.append(f"      <charge>{a.charge:.6f}</charge>\n")
            lines.append(f"      <radius>{a.radius:.6f}</radius>\n")
            lines.append(f"    </atom>\n")
        lines.append(f"  </residue>\n")
    lines.append("</roottag>\n")
    path.write_text("".join(lines))


# APBS grid sizing
def compute_grid_params(
    atoms: List[PQRAtom], fine_spacing: float = 0.5, n_grids: int = 3, srad: float = 1.5
) -> List[dict]:
    """
    Compute APBS mg-manual grid parameters for 3 nested grids.
    Returns list of dicts with keys: spacing, dime, glen, gcent.
      glen_fine[i]   = max(16, ceil(2*(max_abs_coord[i]+atomic_r+srad)/16)*16)
      glen_coarse[i] = 4 * glen_fine[i]
      glen_medium[i] = 2 * glen_fine[i]
      sp_coarse = fine_spacing * 2^round(log2(max_glen_coarse/16/fine_spacing))
      sp_medium = sp_coarse / 4
      dime[i]   = glen[i]/spacing + 1  (rounded to odd)
      gcent     = centroid of heavy atoms
    """
    atoms_real = [a for a in atoms if a.name != "GHO"]
    coords = np.array([[a.x, a.y, a.z] for a in atoms_real])
    gcent = coords.mean(axis=0).tolist()
    # Fine glen per axis: cover all atoms + atomic radii + srad probe
    glen_fine = []
    for i, ax in enumerate(["x", "y", "z"]):
        max_half = max(abs(getattr(a, ax)) + a.radius + srad for a in atoms_real)
        glen = max(16, math.ceil(max_half * 2 / 16) * 16)
        if i == 2:
            glen_z_padded = max(16, math.ceil(max_half * 3 / 16) * 16)
            if glen_z_padded > glen:
                glen = glen_z_padded
        glen_fine.append(glen)
    glen_coarse = [g * 4 for g in glen_fine]
    glen_medium = [g * 2 for g in glen_fine]
    glen_fine_xy = max(glen_fine[:2])
    sp_coarse = float(glen_fine_xy) / 4.0
    # Round to nearest power of 2 * fine_spacing for clean grid
    exp = round(math.log2(sp_coarse / fine_spacing))
    sp_coarse = fine_spacing * (2 ** max(0, exp))
    sp_medium = sp_coarse / 4  # 3 levels: coarse/4x -> medium/2x -> fine/1x
    grids = []
    for sp, glens in [
        (sp_coarse, glen_coarse),
        (sp_medium, glen_medium),
        (fine_spacing, glen_fine),
    ][:n_grids]:
        dimes = [int(g / sp) + 1 for g in glens]
        dimes = [d + 1 if d % 2 == 0 else d for d in dimes]
        grids.append(
            {
                "spacing": sp,
                "dime": dimes,
                "glen": glens,
                "gcent": gcent,
            }
        )
    return grids


def write_apbs_inputs(
    atoms: List[PQRAtom],
    mol_xml: Path,
    mol_name: str,
    work_dir: Path,
    cfg: BDSurfaceConfig,
) -> List[Path]:
    """
    Write 3 APBS input files (coarse/medium/fine) for one molecule.
    Returns list of .in file paths.
    """
    grids = compute_grid_params(
        atoms, cfg.apbs_fine_spacing, cfg.apbs_n_grids, cfg.apbs_srad
    )
    in_files = []
    # Ion string for APBS
    debye = cfg.compute_debye_length()
    if cfg.ion_concentration > 0 and debye < 1e10:
        # NaCl: Na+ conc=c, Cl- conc=c
        ion_lines = (
            f"  ion charge +1 conc {cfg.ion_concentration:.4f} radius 0.95\n"
            f"  ion charge -1 conc {cfg.ion_concentration:.4f} radius 1.81\n"
        )
    else:
        ion_lines = ""
    for i, g in enumerate(grids):
        in_path = work_dir / f"{mol_name}{i}.in"
        dime_str = " ".join(str(d) for d in g["dime"])
        glen_str = " ".join(f"{v:.4g}" for v in g["glen"])
        gcent_str = " ".join(f"{v:.4g}" for v in g["gcent"])
        lines = [f"read\n  mol xml {mol_xml.name}\n"]
        if i > 0:
            prev_dx = work_dir / f"{mol_name}{i-1}.dx"
            lines.append(f"  pot dx {prev_dx.name}\n")
        lines.append("end\n\nelec\n  mg-manual\n")
        lines.append(f"  dime {dime_str} \n")
        lines.append(f"  glen {glen_str} \n")
        lines.append(f"  gcent {gcent_str} \n")
        lines.append(f"  mol 1\n  lpbe\n")
        if i == 0:
            lines.append(f"  bcfl sdh\n")
        else:
            lines.append(f"  usemap pot 1\n  bcfl map\n")
        lines.append(f"\n{ion_lines}")
        lines.append(f"  pdie {cfg.pdie}\n  sdie {cfg.sdie}\n")
        lines.append(f"  srfm {cfg.apbs_srfm}\n  chgm {cfg.apbs_chgm}\n")
        lines.append(f"  sdens 10.0\n  srad {cfg.apbs_srad}\n")
        lines.append(f"  temp {cfg.temperature}\n")
        lines.append(f"  write pot dx {mol_name}{i}\nend\nquit\n")
        in_path.write_text("".join(lines))
        in_files.append(in_path)
    return in_files


def run_apbs(in_files: List[Path], work_dir: Path):
    """Run APBS for each .in file sequentially."""
    for in_file in in_files:
        print(f"    $ apbs {in_file.name}")
        result = subprocess.run(
            ["apbs", in_file.name], cwd=work_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"APBS failed on {in_file.name}:\n{result.stdout[-2000:]}"
            )
        # Move generated DX file if APBS wrote it with path prefix
        dx_name = in_file.stem.replace(".in", "") + ".dx"
        # APBS sometimes writes to cwd
        if not (work_dir / dx_name).exists():
            for candidate in Path(".").glob(f"{in_file.stem}*.dx"):
                shutil.move(str(candidate), work_dir / candidate.name)


# rxns.xml
def write_rxns_xml(
    cfg: BDSurfaceConfig, rec_n_atoms: int, lig_n_atoms: int, work_dir: Path
):
    """
    Write rxns.xml reaction criterion file.
    Uses GHO-GHO distance (last atom of each molecule).
    """
    rec_gho = rec_n_atoms  # last atom = GHO
    lig_gho = lig_n_atoms  # last atom = GHO
    lines = ['<?xml version="1.0" ?>\n<roottag>\n']
    lines.append("   <first_state>b_surface</first_state>\n")
    lines.append("   <reactions>\n")
    outer_r = cfg.bd_milestone_radius
    n_outer = int(round(outer_r))
    lines.append("      <reaction>\n")
    lines.append(f"         <name>b_{n_outer}</name>\n")
    lines.append(f"         <state_before>b_surface</state_before>\n")
    lines.append(f"         <state_after>{n_outer}</state_after>\n")
    lines.append("         <criterion>\n")
    lines.append("            <molecules>\n")
    lines.append("               <molecule0>receptor receptor</molecule0>\n")
    lines.append("               <molecule1>ligand ligand</molecule1>\n")
    lines.append("            </molecules>\n")
    lines.append("            <n_needed>1</n_needed>\n")
    lines.append("            <pair>\n")
    lines.append(f"               <atoms>{rec_gho} {lig_gho}</atoms>\n")
    lines.append(f"               <distance>{outer_r:.1f}</distance>\n")
    lines.append("            </pair>\n")
    lines.append("         </criterion>\n")
    lines.append("      </reaction>\n")
    if cfg.bd_milestone_radius_inner > 0:
        inner_r = cfg.bd_milestone_radius_inner
        n_inner = int(round(inner_r))
        lines.append("      <reaction>\n")
        lines.append(f"         <name>{n_outer}_{n_inner}</name>\n")
        lines.append(f"         <state_before>{n_outer}</state_before>\n")
        lines.append(f"         <state_after>{n_inner}</state_after>\n")
        lines.append("         <criterion>\n")
        lines.append("            <molecules>\n")
        lines.append("               <molecule0>receptor receptor</molecule0>\n")
        lines.append("               <molecule1>ligand ligand</molecule1>\n")
        lines.append("            </molecules>\n")
        lines.append("            <n_needed>1</n_needed>\n")
        lines.append("            <pair>\n")
        lines.append(f"               <atoms>{rec_gho} {lig_gho}</atoms>\n")
        lines.append(f"               <distance>{inner_r:.1f}</distance>\n")
        lines.append("            </pair>\n")
        lines.append("         </criterion>\n")
        lines.append("      </reaction>\n")
    lines.append("   </reactions>\n</roottag>\n")
    (work_dir / "rxns.xml").write_text("".join(lines))


# input.xml
def write_input_xml(
    cfg: BDSurfaceConfig, n_rec_grids: int, n_lig_grids: int, work_dir: Path
):
    """
    Write the nam_simulation input.xml.
    """
    debye = cfg.compute_debye_length()
    kT = cfg.kT
    # Ion lines
    if cfg.ion_concentration > 0 and debye < 1e10:
        ion_xml = (
            f"\n      <ion>\n"
            f"        <q>1</q><conc>{cfg.ion_concentration}</conc><radius>0.95</radius>\n"
            f"      </ion>\n"
            f"      <ion>\n"
            f"        <q>-1</q><conc>{cfg.ion_concentration}</conc><radius>1.81</radius>\n"
            f"      </ion>\n    "
        )
    else:
        ion_xml = ""
    hi = "true" if cfg.hydrodynamic_interactions else "false"
    # Build DX grid entries
    rec_grids = "\n".join(
        f'          <grid source="make_apbs_inputs"> receptor{i}.dx </grid>'
        for i in range(n_rec_grids)
    )
    lig_grids = "\n".join(
        f'          <grid source="make_apbs_inputs"> ligand{i}.dx </grid>'
        for i in range(n_lig_grids)
    )
    xml = f"""<root>
  <n_threads> {cfg.n_threads} </n_threads>
  <seed> {cfg.seed} </seed>
  <o> results1.xml </o>
  <n_trajectories> {cfg.n_trajectories} </n_trajectories>
  <n_trajectories_per_output> 1000 </n_trajectories_per_output>
  <max_n_steps> {cfg.max_n_steps} </max_n_steps>
  <trajectory_file> traj </trajectory_file>
  <n_steps_per_output> 100000 </n_steps_per_output>
  <s>  
    <start_at_site> false </start_at_site>
    <reaction_file> rxns.xml </reaction_file>
    <hydrodynamic_interactions> {hi} </hydrodynamic_interactions>
    <time_step_tolerances>    
      <minimum_core_dt> {cfg.minimum_core_dt} </minimum_core_dt>
      <minimum_core_reaction_dt> {cfg.minimum_core_reaction_dt} </minimum_core_reaction_dt>
    </time_step_tolerances>
    <solvent>    
      <dielectric> {cfg.sdie} </dielectric>
      <relative_viscosity> {cfg.relative_viscosity} </relative_viscosity>
      <kT> {kT:.16f} </kT>
      <desolvation_parameter> {cfg.desolvation_parameter} </desolvation_parameter>
      <ions> {ion_xml}</ions>
      <debye_length> {debye:.5e} </debye_length>
    </solvent>
    <group>    
      <n> receptor </n>
      <core>      
        <n> receptor </n>
        <atoms> receptor.xml </atoms>
        <all_in_surface> false </all_in_surface>
        <is_protein> false </is_protein>
        <dielectric> {cfg.pdie} </dielectric>
        <grid_spacing> {cfg.apbs_fine_spacing} </grid_spacing>
        <electric_field checked="true">        
{rec_grids}
        </electric_field>
      </core>
    </group>
    <group>    
      <n> ligand </n>
      <core>      
        <n> ligand </n>
        <atoms> ligand.xml </atoms>
        <all_in_surface> false </all_in_surface>
        <is_protein> false </is_protein>
        <dielectric> {cfg.pdie} </dielectric>
        <grid_spacing> {cfg.apbs_fine_spacing} </grid_spacing>
        <electric_field checked="true">        
{lig_grids}
        </electric_field>
      </core>
    </group>
  </s>
</root>
"""
    (work_dir / "input.xml").write_text(xml)


# Main pipeline
def run_cmd(cmd: str, cwd: Path = None, step: str = "") -> str:
    """Run a shell command, raise RuntimeError on failure."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed:\n  cmd: {cmd}\n"
            f"  stdout: {result.stdout[-1000:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    return result.stdout


def prepare_bd_surface(cfg: BDSurfaceConfig, input_xml_dir: Path):
    W = input_xml_dir / cfg.work_dir
    W.mkdir(parents=True, exist_ok=True)
    pdb = (input_xml_dir / cfg.pdb).resolve()
    parm = (input_xml_dir / cfg.parm7).resolve()
    # Validate required fields
    if not cfg.receptor_resname.strip():
        raise ValueError(
            "receptor_resname is required. Set <receptor_resname>XXX</receptor_resname> "
            "in the pystarc_input.xml (3-letter residue name of the receptor)."
        )
    if not cfg.ligand_resname.strip():
        raise ValueError(
            "ligand_resname is required. Set <ligand_resname>XXX</ligand_resname> "
            "in the pystarc_input.xml (3-letter residue name of the ligand)."
        )
    if not cfg.pdb.exists() and not (input_xml_dir / cfg.pdb).exists():
        raise FileNotFoundError(f"PDB file not found: {cfg.pdb}")
    if not cfg.parm7.exists() and not (input_xml_dir / cfg.parm7).exists():
        raise FileNotFoundError(f"parm7 file not found: {cfg.parm7}")
    print("=" * 64)
    print("  PySTARC - BD surface preparation pipeline")
    print("=" * 64)
    print(f"  PDB            : {pdb.name}")
    print(f"  parm7          : {parm.name}")
    print(f"  Receptor       : {cfg.receptor_resname}")
    print(f"  Ligand         : {cfg.ligand_resname}")
    print(f"  Output         : {W}")
    print(f"  Ions           : {cfg.ion_concentration*1000:.0f} mM {cfg.ion_type}")
    debye = cfg.compute_debye_length()
    print(f"  Debye length   : {debye:.2f} A")
    print(f"  Milestone r    : {cfg.bd_milestone_radius:.1f} A")
    print()
    # -- Step 1: Strip water from PDB
    print("[1] Stripping water from PDB ...")
    dry_pdb = W / "complex_nowater.pdb"
    SOLVENT = SOLVENT_RESIDUES  # use the comprehensive module-level set
    kept = 0
    with open(pdb) as fin, open(dry_pdb, "w") as fout:
        for line in fin:
            tag = line[:6].strip()
            if tag in ("ATOM", "HETATM"):
                res = line[17:20].strip().upper()
                if res in SOLVENT:
                    continue
            fout.write(line)
            if tag in ("ATOM", "HETATM"):
                kept += 1
    print(f"  {kept} atoms retained after stripping solvent")
    # -- Step 2: cpptraj PDB -> inpcrd
    print("\n[2] cpptraj: PDB -> inpcrd ...")
    pdb_stem = pdb.stem
    inpcrd = W / f"{pdb_stem}.inpcrd"
    rst = W / f"{pdb_stem}.rst"
    cpptraj_in = W / "get_inpcrd.cpptraj"
    cpptraj_in.write_text(
        f"parm {parm}\n" f"trajin {pdb}\n" f"trajout {rst.name}\n" f"run\n"
    )
    run_cmd(f"cpptraj -i {cpptraj_in.name}", cwd=W, step="cpptraj")
    shutil.move(str(rst), str(inpcrd))
    cpptraj_in.unlink(missing_ok=True)
    print(f"  -> {inpcrd.name}")
    # -- Step 3: ambpdb -> combined PQR
    combined_pqr_name = f"{pdb_stem}.pqr"
    print(f"\n[3] ambpdb: parm7 + inpcrd -> {combined_pqr_name} ...")
    combined_pqr = W / combined_pqr_name
    run_cmd(
        f"ambpdb -p {parm} -c {inpcrd.name} -pqr > {combined_pqr_name}",
        cwd=W,
        step="ambpdb",
    )
    print(f"  -> {combined_pqr.name}")
    # -- Step 4: Split combined PQR -> receptor + ligand
    print("\n[4] Splitting PQR into receptor and ligand ...")
    all_atoms = read_pqr(combined_pqr)
    rec_atoms = [a for a in all_atoms if a.resname == cfg.receptor_resname]
    lig_atoms = [a for a in all_atoms if a.resname == cfg.ligand_resname]
    print(f"  Receptor ({cfg.receptor_resname}): {len(rec_atoms)} atoms")
    print(f"  Ligand   ({cfg.ligand_resname}):   {len(lig_atoms)} atoms")
    # -- Step 5: Centre each molecule at origin
    print("\n[5] Centering molecules at origin ...")
    rec_ctr = centroid(rec_atoms)
    lig_ctr = centroid(lig_atoms)
    print(f"  Receptor centroid: {rec_ctr.round(3)}")
    print(f"  Ligand centroid:   {lig_ctr.round(3)}")
    rec_atoms = centre_at_origin(rec_atoms)
    lig_atoms = centre_at_origin(lig_atoms)
    # Re-number serials from 1
    for i, a in enumerate(rec_atoms):
        a.serial = i + 1
    for i, a in enumerate(lig_atoms):
        a.serial = i + 1
    # Step 6: Ligand - each atom gets own residue
    if cfg.ligand_atom_per_residue:
        print("\n[6] Assigning each ligand atom its own residue number ...")
        lig_atoms = assign_each_atom_own_residue(lig_atoms)
    # Step 7: Inject GHO ghost atom
    if cfg.inject_gho:
        print("\n[7] Injecting GHO ghost atom at centroid (0,0,0) ...")
        rec_atoms = inject_gho(rec_atoms)
        lig_atoms = inject_gho(lig_atoms)
        print(f"  Receptor: now {len(rec_atoms)} atoms (last = GHO)")
        print(f"  Ligand:   now {len(lig_atoms)} atoms (last = GHO)")
    rec_pqr = W / "receptor.pqr"
    lig_pqr = W / "ligand.pqr"
    write_pqr(rec_atoms, rec_pqr)
    write_pqr(lig_atoms, lig_pqr)
    print(f"  -> {rec_pqr.name}, {lig_pqr.name}")
    # Step 8: PQR to XML
    print("\n[8] Converting PQR to the XML format ...")
    rec_xml = W / "receptor.xml"
    lig_xml = W / "ligand.xml"
    pqr_to_xml(rec_atoms, rec_xml)
    pqr_to_xml(lig_atoms, lig_xml)
    print(f"  -> {rec_xml.name}, {lig_xml.name}")
    # Step 9: APBS inputs + run
    print("\n[9] Generating APBS inputs and running APBS ...")
    rec_in = write_apbs_inputs(rec_atoms, rec_xml, "receptor", W, cfg)
    lig_in = write_apbs_inputs(lig_atoms, lig_xml, "ligand", W, cfg)
    print(f"  Receptor APBS inputs: {[f.name for f in rec_in]}")
    run_apbs(rec_in, W)
    print(f"  Ligand APBS inputs:   {[f.name for f in lig_in]}")
    run_apbs(lig_in, W)
    rec_dx = [W / f"receptor{i}.dx" for i in range(cfg.apbs_n_grids)]
    lig_dx = [W / f"ligand{i}.dx" for i in range(cfg.apbs_n_grids)]
    for dx in rec_dx + lig_dx:
        if dx.exists():
            size = dx.stat().st_size // 1024
            print(f"  -> {dx.name}  ({size} KB)")
        else:
            print(f"  WARNING: {dx.name} not found!")
    # Step 10: rxns.xml
    print("\n[10] Writing rxns.xml ...")
    write_rxns_xml(cfg, len(rec_atoms), len(lig_atoms), W)
    print(
        f"  Reaction: receptor GHO ({len(rec_atoms)}) - "
        f"ligand GHO ({len(lig_atoms)}) < {cfg.bd_milestone_radius:.1f} A"
    )
    # Step 11: input.xml
    print(
        "\n[11] Writing input.xml (the reference implementation nam_simulation ready) ..."
    )
    write_input_xml(cfg, cfg.apbs_n_grids, cfg.apbs_n_grids, W)
    print()
    print("=" * 64)
    print(f"  b_surface/ preparation complete!")
    print(f"  Output directory: {W}")
    print()
    print("  Files generated:")
    for f in sorted(W.iterdir()):
        if f.is_file():
            print(f"    {f.name:<30s} {f.stat().st_size//1024:6d} KB")
    print()
    print("  To run with the reference implementation:")
    print(f"    cd {W}")
    print(f"    nam_simulation input.xml")
    print()
    print("  To run with PySTARC:")
    print(f"    python run_pystarc.py pystarc_input.xml")
    print("=" * 64)
    return W


# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python prepare_bd_surface.py pystarc_input.xml")
        sys.exit(1)
    xml_path = Path(sys.argv[1])
    cfg = parse_config(xml_path)
    prepare_bd_surface(cfg, xml_path.parent)
