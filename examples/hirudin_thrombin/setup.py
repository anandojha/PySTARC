#!/usr/bin/env python3
"""
PySTARC chain BD setup script for hirudin C-tail / thrombin exosite I.

Generates chain.json (chain topology + initial body-frame positions) and
input.xml (chain BD pipeline configuration) from flat parameters at the
top of the file. Does NOT run APBS — pre-computed DX grids must be
present in apbs_output/ before launching; setup.py warns (non-fatal)
about missing grids and still writes chain.json + input.xml.

Run from the example directory:
    cd examples/hirudin_thrombin
    python setup.py
    python ../../run_pystarc.py input.xml
"""

import numpy as np
import os
import sys

#######################################################################################################################
# User settings

# Inputs (already present in this example directory)
THROMBIN_PQR              = "thrombin.pqr"                          # Rigid target PQR
HIRUDIN_PQR               = "hirudin_ctail.pqr"                     # Reference (chain is built from sequence below)
REACTION_PAIRS_JSON       = "reaction_pairs.json"                   # [target_atom_idx, chain_atom_idx, distance_A] triples

# Chain construction (sequence string, dash-separated 3-letter codes; caps for N/C termini)
SEQUENCE                  = "ASN-GLY-ASP-PHE-GLU-GLU-ILE-PRO-GLU-GLU-TYR-LEU-GLN"  # hirudin C-tail (residues 53-65)
CAP_N                     = "ACE"                                   # N-terminal cap
CAP_C                     = "NME"                                   # C-terminal cap
CHAIN_NAME                = "hirudin_ctail"
CHAIN_JSON                = "chain.json"                            # Output chain topology + positions

# DX grids (pre-computed by APBS; place in apbs_output/ before launching)
TARGET_GRID_DX            = "apbs_output/thrombin1.dx"              # Electrostatic potential grid
BORN_GRID_DX              = "apbs_output/thrombin1_born.dx"         # Born desolvation grid

# Reaction criterion
REACTION_N_NEEDED         = "3"                                     # Min contact pairs that must be satisfied

# BD core (shared with rigid-body)
BD_MILESTONE_RADIUS       = "50.0"                                  # b-surface start radius (A)
R_ESCAPE                  = "80.0"                                  # q-sphere radius (A); 0 -> 1.1 * BD_MILESTONE_RADIUS
N_TRAJECTORIES            = "1000"                                  # Number of BD trajectories (production: 10000+)
MAX_STEPS                 = "1000000"                               # Max steps per trajectory
DT                        = "0.2"                                   # Outer (rigid-body) timestep (ps)
TEMPERATURE               = "298.15"                                # Temperature (K)
SEED                      = "1"                                     # Random seed (any integer)

# Chain BD inner integration
DT_CHAIN                  = "0.025"                                 # Inner internal-coordinate timestep (ps)
CHAIN_STEPS_PER_OUTER     = "8"                                     # Inner steps per outer step
N_EQUILIBRATION_STEPS     = "0"                                     # Pre-equilibration steps for chain internal coords (production: 4000+)

# Diffusion mode
AUTO_DIFFUSION            = "true"                                  # true -> RPY tensors from geometry; false -> scalar D
D_TRANS                   = "0.0"                                   # Translational D (A^2/ps); 0 -> default 0.1
D_ROT                     = "0.0"                                   # Rotational D (rad^2/ps); 0 -> default 0.01

# Soft repulsion (WCA chain-target steering)
USE_SOFT_REPULSION        = "true"                                  # Enable WCA chain-target steering layer
SOFT_REPULSION_EPS        = "1.0"                                   # WCA epsilon (kBT)

# Desolvation
DESOLVATION_ALPHA         = "0.07957747"                            # 1/(4*pi), engine.py default

# Output / parallelism
WORK_DIR                  = "bd_sims"                               # Output directory
SAVE_INTERVAL             = "10"                                    # Output save interval (%)
N_WORKERS                 = "1"                                     # Parallel worker count for trajectory dispatch
GPU                       = "false"                                 # Chain BD is CPU
CONVERGENCE_CHECK         = "false"                                 # Convergence checking (chain BD safer with false)
CONVERGENCE_INTERVAL      = "10"                                    # Convergence print interval (%)
CONVERGENCE_TOL           = "0.05"                                  # Convergence tolerance
#######################################################################################################################

# Locate pystarc working tree (autodetect, mirrors trypsin)
PYSTARC_DIR = None
script_dir = os.path.dirname(os.path.abspath(__file__))
candidates = []
for i in range(10):
    prefix = os.path.join(script_dir, *[".."] * i) if i > 0 else script_dir
    candidates.append(os.path.join(prefix, "pystarc"))
    candidates.append(os.path.join(prefix, "PySTARC", "pystarc"))
for candidate in candidates:
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "coffdrop_data")):
        PYSTARC_DIR = os.path.abspath(candidate)
        break
if PYSTARC_DIR is None:
    try:
        import pystarc as _ps
        candidate = os.path.dirname(os.path.abspath(_ps.__file__))
        if os.path.isdir(os.path.join(candidate, "coffdrop_data")):
            PYSTARC_DIR = candidate
    except ImportError:
        pass
if PYSTARC_DIR is None:
    print("Error: Could not find pystarc/coffdrop_data/ directory.")
    print(f"  Searched relative to: {script_dir}")
    sys.exit(1)
COFFDROP_DIR = os.path.join(PYSTARC_DIR, "coffdrop_data")
# Make working-tree pystarc importable (overrides any installed version)
sys.path.insert(0, os.path.dirname(PYSTARC_DIR))
print(f"PySTARC: {PYSTARC_DIR}")
print(f"Coffdrop data: {COFFDROP_DIR}")

# Step 1: verify input files
print("\nStep 1: Verify input files")
for f in [THROMBIN_PQR, HIRUDIN_PQR, REACTION_PAIRS_JSON]:
    if not os.path.exists(f):
        print(f"Error: {f} not found in current directory")
        sys.exit(1)
    print(f"  {f}")

# Step 2: build chain via chain_from_sequence + place_relaxed_geometry
print(f"\nStep 2: Build chain from sequence")
print(f"  sequence: {SEQUENCE}")
print(f"  caps:     ({CAP_N}, {CAP_C})")
from pystarc.simulation.coffdrop_chain import (
    chain_from_sequence, place_relaxed_geometry,
)
from pystarc.structures.chain_io import save_chain_to_json
chain = chain_from_sequence(
    SEQUENCE,
    coffdrop_dir=COFFDROP_DIR,
    name=CHAIN_NAME,
    caps=(CAP_N, CAP_C),
)
chain_charges = np.array([a.charge for a in chain.atoms])
n_atoms = len(chain.atoms)
print(f"  atoms:           {n_atoms}")
print(f"  bonds:           {len(chain.bonds)}")
print(f"  angles:          {len(chain.angles)}")
print(f"  torsions:        {len(chain.torsions)}")
print(f"  net charge:      {chain_charges.sum():+.3f} e")
body_positions = place_relaxed_geometry(chain)
body_positions = body_positions - body_positions.mean(axis=0)
body_radius = float(np.linalg.norm(body_positions, axis=1).max())
print(f"  max body radius: {body_radius:.2f} A")

# Step 3: save chain.json
print(f"\nStep 3: Save chain.json")
out_path = save_chain_to_json(chain, body_positions, CHAIN_JSON)
print(f"  wrote: {out_path}")

# Step 4: write input.xml
print(f"\nStep 4: Write input.xml")
input_xml = f"""<?xml version="1.0"?>
<pystarc>
  <!-- Shared rigid-body fields (used by chain BD pipeline) -->
  <receptor_pqr>{THROMBIN_PQR}</receptor_pqr>
  <bd_milestone_radius>{BD_MILESTONE_RADIUS}</bd_milestone_radius>
  <n_trajectories>{N_TRAJECTORIES}</n_trajectories>
  <max_steps>{MAX_STEPS}</max_steps>
  <dt>{DT}</dt>
  <temperature>{TEMPERATURE}</temperature>
  <seed>{SEED}</seed>
  <work_dir>{WORK_DIR}</work_dir>
  <gpu>{GPU}</gpu>
  <desolvation_alpha>{DESOLVATION_ALPHA}</desolvation_alpha>
  <save_interval>{SAVE_INTERVAL}</save_interval>
  <convergence_check>{CONVERGENCE_CHECK}</convergence_check>
  <convergence_interval>{CONVERGENCE_INTERVAL}</convergence_interval>
  <convergence_tol>{CONVERGENCE_TOL}</convergence_tol>
  <!-- Chain BD specific block -->
  <chain>
    <chain_json>{CHAIN_JSON}</chain_json>
    <reaction_pairs_json>{REACTION_PAIRS_JSON}</reaction_pairs_json>
    <target_grid_dx>{TARGET_GRID_DX}</target_grid_dx>
    <born_grid_dx>{BORN_GRID_DX}</born_grid_dx>
    <r_escape>{R_ESCAPE}</r_escape>
    <reaction_n_needed>{REACTION_N_NEEDED}</reaction_n_needed>
    <auto_diffusion>{AUTO_DIFFUSION}</auto_diffusion>
    <D_trans>{D_TRANS}</D_trans>
    <D_rot>{D_ROT}</D_rot>
    <use_soft_repulsion>{USE_SOFT_REPULSION}</use_soft_repulsion>
    <soft_repulsion_eps>{SOFT_REPULSION_EPS}</soft_repulsion_eps>
    <n_workers>{N_WORKERS}</n_workers>
    <dt_chain>{DT_CHAIN}</dt_chain>
    <chain_steps_per_outer>{CHAIN_STEPS_PER_OUTER}</chain_steps_per_outer>
    <n_equilibration_steps>{N_EQUILIBRATION_STEPS}</n_equilibration_steps>
  </chain>
</pystarc>
"""
with open("input.xml", "w") as fh:
    fh.write(input_xml)
print(f"  wrote: input.xml")

# Step 5: verify DX grids (warning, not fatal)
print(f"\nStep 5: Check APBS DX grids (non-fatal)")
missing = [g for g in [TARGET_GRID_DX, BORN_GRID_DX] if not os.path.exists(g)]
if missing:
    print(f"  WARNING: DX grids missing: {missing}")
    print(f"  Required before running. Copy from archive:")
    print(f"    mkdir -p apbs_output")
    print(f"    cp /mnt/home/aojha/Downloads/pystarc_backups/PySTARC_v2_results_archive/hirudin_thrombin/apbs_output/thrombin1.dx apbs_output/")
    print(f"    cp /mnt/home/aojha/Downloads/pystarc_backups/PySTARC_v2_results_archive/hirudin_thrombin/apbs_output/thrombin1_born.dx apbs_output/")
else:
    for g in [TARGET_GRID_DX, BORN_GRID_DX]:
        size_mb = os.path.getsize(g) / 1e6
        print(f"  {g}  ({size_mb:.1f} MB)")

print()
print("=" * 70)
print("Setup complete. Files written:")
print(f"  {CHAIN_JSON}")
print(f"  input.xml")
print()
print("To run the chain BD simulation (after copying DX grids):")
print(f"  python ../../run_pystarc.py input.xml")
print("=" * 70)
