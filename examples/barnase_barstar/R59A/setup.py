#!/usr/bin/env python3
"""
PySTARC setup script for barnase-barstar protein-protein benchmark.

PDB: 1BRS (Buckle, Schreiber & Fersht, 1994)
  Barnase (chain A, 110 residues) + Barstar (chain D, 89 residues)

Experimental k_on: 3-6 × 10^8 M^-1 s^-1 at 50 mM ionic strength

For the barnase-barstar protein-protein complex, splitting is done 
by residue number after tleap renumbering (barnase = 1-108, barstar = 109+).
"""

import numpy as np
import subprocess
import shutil
import sys
import os
import re

# User settings
PDB_ID                  = "1BRS"                  # PDB accession code
CHAIN_REC               = "A"                     # Receptor chain ID in PDB
CHAIN_LIG               = "D"                     # Ligand chain ID in PDB
MUTATIONS               = [("A",59,"ARG","ALA")]  # R59A
REC_RESID_MAX           = 108                     # Last residue of receptor (after tleap renumbering)
RECEPTOR_RESNAME        = "BN"                    # Receptor label for XML
LIGAND_RESNAME          = "BS"                    # Ligand label for XML
RECEPTOR_PQR            = "receptor.pqr"          # Output receptor PQR filename
LIGAND_PQR              = "ligand.pqr"            # Output ligand PQR filename
RXNS_XML                = "rxns.xml"              # Output reaction criterion filename
# R83 barnase -> ARG81 (NH2)  <->  D35 barstar -> ASP147 (OD1) Reaction criterion (Gabdoulline & Wade, 1997)
# R59 pair removed (mutated to ALA)
RXN_TARGETS_REC         = [(81, 'NH2', 'ARG')]
RXN_TARGETS_LIG         = [(147, 'OD1', 'ASP')]
RXN_CUTOFF              = 7.0                     # Reaction distance cutoff (A)
N_NEEDED                = 1                       # Pairs that must be satisfied simultaneously
# Simulation parameters
BD_MILESTONE_RADIUS     = "80.0000"               # b-surface start radius (A)
R_HYDRO_REC             = "0"                     # Receptor hydrodynamic radius (0=auto)
R_HYDRO_LIG             = "0"                     # Ligand hydrodynamic radius (0=auto)
DEBYE_LENGTH            = "13.6"                  # Debye screening length (A) for 50 mM
ION_CONCENTRATION       = "0.05"                  # Salt concentration (M)
ION_RADIUS_POS          = "0.95"                  # Cation radius - Na+ Pauling (A)
ION_RADIUS_NEG          = "1.81"                  # Anion radius - Cl- Pauling (A)
PDIE                    = "4.0"                   # Protein dielectric constant
SDIE                    = "78.0"                  # Solvent dielectric constant
SRAD                    = "1.4"                   # Solvent probe radius (A)
APBS_CGLEN              = "0"                     # APBS coarse grid length (0=auto)
APBS_FGLEN              = "192"                   # APBS fine grid length (A)
APBS_DIME               = "257"                   # APBS grid points per dimension
APBS_COARSE_DIME        = "0"                     # APBS coarse grid dime (0=auto)
APBS_FINE_DIME          = "0"                     # APBS fine grid dime (0=auto)
GPU_FORCE_BATCH         = "1000"                  # GPU force batch size (0=auto)
DESOLVATION_ALPHA       = "0.0795775"             # Desolvation coupling constant
HYDRODYNAMIC_INTERACTIONS = "true"                # Include HI corrections
OVERLAP_CHECK           = "true"                  # Reject overlapping configurations
MULTIPOLE_FALLBACK      = "true"                  # Yukawa fallback outside grid
LJ_FORCES               = "false"                 # Lennard-Jones short-range forces
N_TRAJECTORIES          = "100000"                # Number of BD trajectories
MAX_STEPS               = "10000000"              # Max steps per trajectory
DT                      = "0.2"                   # Base timestep (ps)
MINIMUM_CORE_DT         = "0.2"                   # Minimum timestep near core (ps)
MAX_DT                  = "100.0"                 # Max timestep ceiling (ps) - critical for protein-protein
TEMPERATURE             = "298.15"                # Temperature (K)
SEED                    = "1"                     # Random seed (any integer)
CHECKPOINT_INTERVAL     = "0"                     # Checkpoint frequency (0=off)
CONVERGENCE_INTERVAL    = "10"                    # Convergence print interval (%)
CONVERGENCE_CHECK       = "true"                  # Enable convergence checking
CONVERGENCE_TOL         = "0.05"                  # Convergence tolerance
GPU                     = "true"                  # Use GPU acceleration
N_THREADS               = "32"                    # CPU threads for APBS/IO
WORK_DIR                = "bd_sims"               # Output directory
SAVE_INTERVAL           = "10"                    # Output save interval (%)

PARAMS = {
    'receptor_resname':          RECEPTOR_RESNAME,
    'ligand_resname':            LIGAND_RESNAME,
    'receptor_pqr':              RECEPTOR_PQR,
    'ligand_pqr':                LIGAND_PQR,
    'rxns_xml':                  RXNS_XML,
    'bd_milestone_radius':       BD_MILESTONE_RADIUS,
    'r_hydro_rec':               R_HYDRO_REC,
    'r_hydro_lig':               R_HYDRO_LIG,
    'debye_length':              DEBYE_LENGTH,
    'ion_concentration':         ION_CONCENTRATION,
    'ion_radius_pos':            ION_RADIUS_POS,
    'ion_radius_neg':            ION_RADIUS_NEG,
    'pdie':                      PDIE,
    'sdie':                      SDIE,
    'srad':                      SRAD,
    'apbs_cglen':                APBS_CGLEN,
    'apbs_fglen':                APBS_FGLEN,
    'apbs_dime':                 APBS_DIME,
    'apbs_coarse_dime':          APBS_COARSE_DIME,
    'apbs_fine_dime':            APBS_FINE_DIME,
    'gpu_force_batch':           GPU_FORCE_BATCH,
    'desolvation_alpha':         DESOLVATION_ALPHA,
    'hydrodynamic_interactions': HYDRODYNAMIC_INTERACTIONS,
    'overlap_check':             OVERLAP_CHECK,
    'multipole_fallback':        MULTIPOLE_FALLBACK,
    'lj_forces':                 LJ_FORCES,
    'n_trajectories':            N_TRAJECTORIES,
    'max_steps':                 MAX_STEPS,
    'dt':                        DT,
    'minimum_core_dt':           MINIMUM_CORE_DT,
    'max_dt':                    MAX_DT,
    'temperature':               TEMPERATURE,
    'seed':                      SEED,
    'checkpoint_interval':       CHECKPOINT_INTERVAL,
    'convergence_interval':      CONVERGENCE_INTERVAL,
    'convergence_check':         CONVERGENCE_CHECK,
    'convergence_tol':           CONVERGENCE_TOL,
    'gpu':                       GPU,
    'n_threads':                 N_THREADS,
    'work_dir':                  WORK_DIR,
    'save_interval':             SAVE_INTERVAL,
}

#  Find pystarc templates
PYSTARC_DIR = None
script_dir = os.path.dirname(os.path.abspath(__file__))
candidates = []
for i in range(10):
    prefix = os.path.join(script_dir, *[".."] * i) if i > 0 else script_dir
    candidates.append(os.path.join(prefix, "pystarc"))
    candidates.append(os.path.join(prefix, "PySTARC", "pystarc"))
for candidate in candidates:
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "templates")):
        PYSTARC_DIR = os.path.abspath(candidate)
        break
if PYSTARC_DIR is None:
    try:
        import pystarc
        candidate = os.path.dirname(os.path.abspath(pystarc.__file__))
        if os.path.isdir(os.path.join(candidate, "templates")):
            PYSTARC_DIR = candidate
    except ImportError:
        pass
if PYSTARC_DIR is None:
    print("Error: Could not find pystarc/templates/ directory.")
    print(f"  Searched relative to: {script_dir}")
    sys.exit(1)
TEMPLATE_DIR = os.path.join(PYSTARC_DIR, "templates")
print(f"Templates: {TEMPLATE_DIR}")

# Copy templates to current directory
shutil.copy(os.path.join(TEMPLATE_DIR, "input.xml"), "input.xml")
shutil.copy(os.path.join(TEMPLATE_DIR, "rxns.xml"), "rxns.xml")
print("Copied: input.xml, rxns.xml")

# Step 1: Download PDB (if needed)
print(f"\nStep 1: Get {PDB_ID}.pdb")
pdb_file = f"{PDB_ID}.pdb"
if not os.path.exists(pdb_file):
    import urllib.request
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, pdb_file)
    print(f"  Downloaded {pdb_file}")
else:
    print(f"  {pdb_file} already exists, skipping")
    
# Step 2: Extract chains
print(f"\nStep 2: Extract chains {CHAIN_REC} + {CHAIN_LIG}")
complex_pdb = f"complex_{CHAIN_REC}{CHAIN_LIG}.pdb"
with open(pdb_file) as fh:
    lines = [l for l in fh if (l.startswith("ATOM") or l.startswith("TER"))
             and (len(l) > 21 and l[21] in (CHAIN_REC, CHAIN_LIG))]
with open(complex_pdb, 'w') as fh:
    fh.write(''.join(lines) + 'END\n')
n_atoms = sum(1 for l in lines if l.startswith("ATOM"))
print(f"  {complex_pdb}: {n_atoms} ATOM lines")

# Step 2b: Apply mutations
print(f"\nStep 2b: Apply mutations")
with open(complex_pdb) as fh:
    pdb_lines = fh.readlines()
ala_atoms = {"N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"}
mutated_lines = []
for line in pdb_lines:
    if not line.startswith("ATOM"):
        mutated_lines.append(line)
        continue
    chain = line[21]
    resid = int(line[22:26])
    resname = line[17:20].strip()
    atomname = line[12:16].strip()
    mutated = False
    for m_chain, m_resid, m_old, m_new in MUTATIONS:
        if chain == m_chain and resid == m_resid and resname == m_old:
            if atomname in ala_atoms:
                line = line[:17] + f"{m_new:>3s}" + line[20:]
                mutated_lines.append(line)
            mutated = True
            break
    if not mutated:
        mutated_lines.append(line)
with open(complex_pdb, "w") as fh:
    fh.writelines(mutated_lines)
for m_chain, m_resid, m_old, m_new in MUTATIONS:
    print(f"  {m_old}{m_resid} (chain {m_chain}) -> {m_new}{m_resid}")

# Step 3: Parameterize with tleap (ff14SB)
print("\nStep 3: Running tleap (ff14SB)")
tleap_input = f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p
complex = loadpdb {complex_pdb}
check complex
saveamberparm complex complex.prmtop complex.rst7
savepdb complex complex.pdb
quit
"""
with open("tleap.in", "w") as fh:
    fh.write(tleap_input)
r = subprocess.run("tleap -f tleap.in", shell=True, capture_output=True, text=True)
with open("leap.log", "w") as fh:
    fh.write(r.stdout + r.stderr)
if not os.path.exists("complex.prmtop"):
    print("  ERROR: tleap failed. Check leap.log")
    sys.exit(1)
print("  complex.prmtop, complex.rst7, complex.pdb created")

# Step 4: Generate PQR and split by residue number
print("\nStep 4: Generate PQR files (split by residue number)")
# cpptraj: ensure proper rst7 format
r = subprocess.run("cpptraj", input=f"parm complex.prmtop\ntrajin complex.rst7\ntrajout _full.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
if r.returncode != 0:
    print(f"Error cpptraj: {r.stderr[:200]}")
    sys.exit(1)
# ambpdb: generate PQR
r = subprocess.run("ambpdb -p complex.prmtop -c _full.rst7 -pqr",
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    print(f"Error ambpdb: {r.stderr[:200]}")
    sys.exit(1)
with open("complex.pqr", "w") as fh:
    fh.write(r.stdout)
# Split by residue number (tleap renumbers sequentially)
rec_lines, lig_lines = [], []
for line in r.stdout.strip().split('\n'):
    if not line.startswith("ATOM"):
        continue
    resi = int(line[22:26].strip())
    if resi <= REC_RESID_MAX:
        rec_lines.append(line)
    else:
        lig_lines.append(line)
with open(RECEPTOR_PQR, "w") as fh:
    fh.write('\n'.join(rec_lines) + '\nEND\n')
with open(LIGAND_PQR, "w") as fh:
    fh.write('\n'.join(lig_lines) + '\nEND\n')
if os.path.exists("_full.rst7"):
    os.remove("_full.rst7")
rec_charge = sum(float(l.split()[8]) for l in rec_lines)
lig_charge = sum(float(l.split()[8]) for l in lig_lines)
print(f"  receptor.pqr (barnase): {len(rec_lines)} atoms, Q={rec_charge:+.2f} e")
print(f"  ligand.pqr (barstar):   {len(lig_lines)} atoms, Q={lig_charge:+.2f} e")

# Step 5: Find reaction criterion atom indices
print("\nStep 5: Find reaction criterion atoms")

def find_atoms(pqr_file, targets):
    """Find atom line numbers matching (resid, name, resname) targets."""
    found = {}
    atom_num = 0
    with open(pqr_file) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                atom_num += 1
                parts = line.split()
                resi = int(parts[4])
                name = parts[2]
                resn = parts[3]
                for tresi, tname, tresn in targets:
                    if resi == tresi and name == tname and resn == tresn:
                        found[(tresi, tname)] = atom_num
                        print(f"  {pqr_file} atom {atom_num}: {name} {resn}{resi}")
    return found

rec_atoms = find_atoms(RECEPTOR_PQR, RXN_TARGETS_REC)
lig_atoms = find_atoms(LIGAND_PQR, RXN_TARGETS_LIG)
if len(rec_atoms) != len(RXN_TARGETS_REC):
    print(f"Error: Missing receptor atoms. Found: {rec_atoms}")
    sys.exit(1)
if len(lig_atoms) != len(RXN_TARGETS_LIG):
    print(f"Error: Missing ligand atoms. Found: {lig_atoms}")
    sys.exit(1)
# Build pairs: match targets in order
rxn_pairs = []
for (r_resi, r_name, _), (l_resi, l_name, _) in zip(RXN_TARGETS_REC, RXN_TARGETS_LIG):
    r_idx = rec_atoms[(r_resi, r_name)]
    l_idx = lig_atoms[(l_resi, l_name)]
    rxn_pairs.append((r_idx, l_idx))
    print(f"  Pair: rec[{r_idx}] <-> lig[{l_idx}]  ({RXN_CUTOFF} A)")
    
# Step 6: Fill rxns.xml
print("\nStep 6: Fill rxns.xml")
with open(RXNS_XML) as fh:
    rxns_content = fh.read()
pair_lines = ""
for r_idx, l_idx in rxn_pairs:
    pair_lines += f'        <pair><atoms> {r_idx} {l_idx} </atoms><distance> {RXN_CUTOFF:.1f} </distance></pair>\n'

rxns_content = re.sub(r'<n_needed>\s*</n_needed>',
                      f'<n_needed> {N_NEEDED} </n_needed>', rxns_content)
rxns_content = re.sub(r'\s*<pair><atoms>\s*</atoms><distance>\s*</distance></pair>\s*\n',
                      '\n' + pair_lines, rxns_content)
with open(RXNS_XML, "w") as fh:
    fh.write(rxns_content)
print(f"  Filled: rxns.xml ({len(rxn_pairs)} pairs, n_needed={N_NEEDED})")

# Step 7: Fill input.xml
print("\nStep 7: Fill PySTARC input.xml")
with open("input.xml") as fh:
    input_content = fh.read()
for tag, val in PARAMS.items():
    input_content = re.sub(f'<{tag}>\\s*</{tag}>', f'<{tag}>{val}</{tag}>', input_content)
with open("input.xml", "w") as fh:
    fh.write(input_content)
print("  Filled: input.xml")

# Step 8: Geometry check
print("\nStep 8: Geometry check")
rec_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rec_lines])
lig_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lig_lines])
rec_maxr = np.max(np.linalg.norm(rec_xyz - rec_xyz.mean(axis=0), axis=1))
lig_maxr = np.max(np.linalg.norm(lig_xyz - lig_xyz.mean(axis=0), axis=1))
b = float(BD_MILESTONE_RADIUS)
fglen = float(APBS_FGLEN)
print(f"  Receptor max radius: {rec_maxr:.1f} A")
print(f"  Ligand max radius:   {lig_maxr:.1f} A")
print(f"  b-surface:           {b:.0f} A")
print(f"  fglen:               {fglen:.0f} A (covers +/- {fglen/2:.0f} A)")
print(f"  b + lig_maxr:        {b + lig_maxr:.0f} A", end="")
if b + lig_maxr < fglen / 2:
    print("  OK (within grid)")
else:
    print("  WARNING: atoms may fall outside grid")

# Step 9: Clean up intermediate files
for f in ["complex_AD.pdb", "tleap.in", "leap.log"]:
    if os.path.exists(f):
        os.remove(f)
    
# Summary
print(" Setup complete")
print(f"  PDB:          {PDB_ID} (chains {CHAIN_REC} + {CHAIN_LIG})")
print(f"  Receptor:     barnase ({len(rec_lines)} atoms, Q={rec_charge:+.1f} e)")
print(f"  Ligand:       barstar ({len(lig_lines)} atoms, Q={lig_charge:+.1f} e)")
print(f"  Force field:  ff14SB")
print(f"  Ionic str:    {float(ION_CONCENTRATION)*1000:.0f} mM (Debye = {DEBYE_LENGTH} A)")
print(f"  b-surface:    {BD_MILESTONE_RADIUS} A")
print(f"  R_hydro:      auto (both)")
print(f"  max_dt:       {MAX_DT} ps")
print(f"  Rxn criterion: {len(rxn_pairs)} pairs at {RXN_CUTOFF} A, n_needed={N_NEEDED}")
for i, (r_idx, l_idx) in enumerate(rxn_pairs):
    r_resi, r_name, r_resn = RXN_TARGETS_REC[i]
    l_resi, l_name, l_resn = RXN_TARGETS_LIG[i]
    print(f"    Pair {i+1}: {r_resn}{r_resi} {r_name} <-> {l_resn}{l_resi} {l_name}")
print(f"  Exp k_on:     ~6.5 x 10^7 (R59A) M-1 s-1")
print(f"  Expect ~4.5x slower than WT")
print(f"\n  To run:")
print(f"    python run_pystarc.py input.xml")
