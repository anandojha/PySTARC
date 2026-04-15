#!/usr/bin/env python3
"""
PySTARC setup script for p38 MAPK / SB203580 protein-ligand benchmark.

PDB: 1A9U (Wang et al., 1998)
  p38 MAPK alpha (chain A, DFG-in conformation) + SB203580 (type I inhibitor)

Experimental k_on: 1.5 × 10^7 M^-1 s^-1 (Miao et al., 2018)
Browndye BD:       6.6 × 10^7 M^-1 s^-1 (Huang, Biophys J, 2021)

For the p38-SB203580 protein-ligand complex, the receptor is the protein
(ATOM records, residues 4-354, skipping His-tag) and the ligand is the
small molecule SB203580 (HETATM records, residue SB2).

The ligand is parameterized with antechamber (GAFF2 + AM1-BCC charges).
"""
import urllib.request
import numpy as np
import subprocess
import shutil
import sys
import os
import re

# User settings
PDB_ID                  = "1A9U"                  # PDB accession code
CHAIN_REC               = "A"                     # Protein chain ID in PDB
LIGAND_RESID            = "SB2"                   # Ligand residue name in PDB
REC_RESID_MIN           = 4                       # First protein residue (skip His-tag)
LIGAND_NET_CHARGE       = 0                       # Net charge of SB203580 at pH 7
RECEPTOR_RESNAME        = "P38"                   # Receptor label for XML
LIGAND_RESNAME          = "SB2"                   # Ligand label for XML
RECEPTOR_PQR            = "receptor.pqr"          # Output receptor PQR filename
LIGAND_PQR              = "ligand.pqr"            # Output ligand PQR filename
RXNS_XML                = "rxns.xml"              # Output reaction criterion filename
# Reaction criterion from crystal contacts (heavy atom polar pairs)
# MET106 N   (hinge H-bond)      <->  NB1  (pyridine N)
# LYS50  NZ  (catalytic lysine)  <->  NC3  (imidazole N)
# VAL102 O   (backbone carbonyl) <->  FD3  (fluorine)
# THR103 N   (backbone amide)    <->  FD3  (fluorine)
RXN_TARGETS_REC         = [(106, 'N', 'MET'), (50, 'NZ', 'LYS'), (102, 'O', 'VAL'), (103, 'N', 'THR')]
RXN_TARGETS_LIG         = [('NB1',), ('NC3',), ('FD3',), ('FD3',)]
RXN_CUTOFFS             = [7.0, 7.0, 7.0, 7.0]    # Per-pair cutoffs (A)
N_NEEDED                = 2                       # Pairs that must be satisfied simultaneously
# Simulation parameters
BD_MILESTONE_RADIUS     = "60.0"                  # b-surface start radius (A)
R_HYDRO_REC             = "0"                     # Receptor hydrodynamic radius (0=auto)
R_HYDRO_LIG             = "0"                     # Ligand hydrodynamic radius (0=auto)
DEBYE_LENGTH            = "7.86"                  # Debye screening length (A) for 150 mM
ION_CONCENTRATION       = "0.15"                  # Salt concentration (M)
ION_RADIUS_POS          = "0.95"                  # Cation radius - Na+ Pauling (A)
ION_RADIUS_NEG          = "1.81"                  # Anion radius - Cl- Pauling (A)
PDIE                    = "4.0"                   # Protein dielectric constant
SDIE                    = "78.0"                  # Solvent dielectric constant
SRAD                    = "1.4"                   # Solvent probe radius (A)
APBS_CGLEN              = "0"                     # APBS coarse grid length (0=auto)
APBS_FGLEN              = "128"                   # APBS fine grid length (A)
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
MAX_STEPS               = "1000000"               # Max steps per trajectory
DT                      = "0.2"                   # Base timestep (ps)
MINIMUM_CORE_DT         = "0.2"                   # Minimum timestep near core (ps)
MAX_DT                  = "0"                     # Max timestep ceiling (ps) - not needed for protein-ligand
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
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, pdb_file)
    print(f"  Downloaded {pdb_file}")
else:
    print(f"  {pdb_file} already exists, skipping")

# Step 2: Extract protein (skip His-tag) and ligand separately
print(f"\nStep 2: Extract protein (chain {CHAIN_REC}, residues >= {REC_RESID_MIN}) and ligand ({LIGAND_RESID})")
# Extract protein ATOM lines, skip His-tag (residues < REC_RESID_MIN)
protein_lines = []
with open(pdb_file) as fh:
    for line in fh:
        if line.startswith("ATOM") and len(line) > 21 and line[21] == CHAIN_REC:
            resid = int(line[22:26].strip())
            if resid >= REC_RESID_MIN:
                protein_lines.append(line)
with open("protein.pdb", "w") as fh:
    fh.write(''.join(protein_lines) + 'END\n')
print(f"  protein.pdb: {len(protein_lines)} ATOM lines")

# Extract ligand HETATM lines
ligand_lines = []
with open(pdb_file) as fh:
    for line in fh:
        if line.startswith("HETATM") and LIGAND_RESID in line:
            ligand_lines.append(line)
with open("ligand_raw.pdb", "w") as fh:
    fh.write(''.join(ligand_lines) + 'END\n')
print(f"  ligand_raw.pdb: {len(ligand_lines)} HETATM lines")

# Step 3a: Parameterize protein with tleap (ff14SB)
print("\nStep 3a: Running tleap for protein (ff14SB)")
tleap_input = f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p
protein = loadpdb protein.pdb
check protein
saveamberparm protein protein.prmtop protein.rst7
savepdb protein protein_leap.pdb
quit
"""
with open("tleap_protein.in", "w") as fh:
    fh.write(tleap_input)
r = subprocess.run("tleap -f tleap_protein.in", shell=True, capture_output=True, text=True)
with open("leap_protein.log", "w") as fh:
    fh.write(r.stdout + r.stderr)
if not os.path.exists("protein.prmtop"):
    print("  ERROR: tleap failed. Check leap_protein.log")
    sys.exit(1)
print("  protein.prmtop, protein.rst7 created")

# Step 3b: Parameterize ligand with antechamber (GAFF2 + AM1-BCC)
print(f"\nStep 3b: Running antechamber for ligand (GAFF2, net charge = {LIGAND_NET_CHARGE})")
r = subprocess.run(
    f"antechamber -i ligand_raw.pdb -fi pdb -o ligand.mol2 -fo mol2 "
    f"-c bcc -at gaff2 -nc {LIGAND_NET_CHARGE}",
    shell=True, capture_output=True, text=True
)
if not os.path.exists("ligand.mol2"):
    print(f"  ERROR: antechamber failed.\n  {r.stderr[:500]}")
    sys.exit(1)
print("  ligand.mol2 created (AM1-BCC charges)")

r = subprocess.run(
    "parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2",
    shell=True, capture_output=True, text=True
)
if not os.path.exists("ligand.frcmod"):
    print(f"  ERROR: parmchk2 failed.\n  {r.stderr[:500]}")
    sys.exit(1)
print("  ligand.frcmod created")

tleap_lig = f"""source leaprc.gaff2
loadamberparams ligand.frcmod
lig = loadmol2 ligand.mol2
check lig
saveamberparm lig ligand.prmtop ligand.rst7
savepdb lig ligand_leap.pdb
quit
"""
with open("tleap_ligand.in", "w") as fh:
    fh.write(tleap_lig)
r = subprocess.run("tleap -f tleap_ligand.in", shell=True, capture_output=True, text=True)
with open("leap_ligand.log", "w") as fh:
    fh.write(r.stdout + r.stderr)
if not os.path.exists("ligand.prmtop"):
    print("  ERROR: tleap for ligand failed. Check leap_ligand.log")
    sys.exit(1)
print("  ligand.prmtop, ligand.rst7 created")

# Step 4: Generate PQR files
print("\nStep 4: Generate PQR files")
# Receptor PQR
r = subprocess.run("cpptraj", input="parm protein.prmtop\ntrajin protein.rst7\ntrajout _prot.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Error cpptraj (protein): {r.stderr[:200]}")
    sys.exit(1)
r = subprocess.run("ambpdb -p protein.prmtop -c _prot.rst7 -pqr",
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Error ambpdb (protein): {r.stderr[:200]}")
    sys.exit(1)
rec_lines = [l for l in r.stdout.strip().split('\n') if l.startswith("ATOM")]
with open(RECEPTOR_PQR, "w") as fh:
    fh.write('\n'.join(rec_lines) + '\nEND\n')
rec_charge = sum(float(l.split()[8]) for l in rec_lines)
print(f"  receptor.pqr (p38): {len(rec_lines)} atoms, Q={rec_charge:+.1f} e")
# Ligand PQR
r = subprocess.run("cpptraj", input="parm ligand.prmtop\ntrajin ligand.rst7\ntrajout _lig.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Error cpptraj (ligand): {r.stderr[:200]}")
    sys.exit(1)
r = subprocess.run("ambpdb -p ligand.prmtop -c _lig.rst7 -pqr",
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Error ambpdb (ligand): {r.stderr[:200]}")
    sys.exit(1)
lig_lines = [l for l in r.stdout.strip().split('\n') if l.startswith("ATOM")]
with open(LIGAND_PQR, "w") as fh:
    fh.write('\n'.join(lig_lines) + '\nEND\n')
lig_charge = sum(float(l.split()[8]) for l in lig_lines)
print(f"  ligand.pqr (SB203580): {len(lig_lines)} atoms, Q={lig_charge:+.1f} e")

# Clean temp files
for f in ["_prot.rst7", "_lig.rst7", "sqm.in", "sqm.out", "sqm.pdb",
          "ANTECHAMBER_AC.AC", "ANTECHAMBER_AC.AC0", "ANTECHAMBER_BOND_TYPE.AC",
          "ANTECHAMBER_BOND_TYPE.AC0", "ANTECHAMBER_AM1BCC.AC",
          "ANTECHAMBER_AM1BCC_PRE.AC", "ATOMTYPE.INF"]:
    if os.path.exists(f):
        os.remove(f)

# Step 5: Find reaction criterion atom indices
print("\nStep 5: Find reaction criterion atoms")

def find_rec_atoms(pqr_file, targets):
    """Find receptor atom line numbers matching (resid, name, resname) targets."""
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

def find_lig_atoms(pqr_file, targets):
    """Find ligand atom line numbers matching (name,) targets."""
    found = {}
    atom_num = 0
    with open(pqr_file) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                atom_num += 1
                parts = line.split()
                name = parts[2]
                for (tname,) in targets:
                    if name == tname and tname not in found:
                        found[tname] = atom_num
                        print(f"  {pqr_file} atom {atom_num}: {name}")
    return found

rec_atoms = find_rec_atoms(RECEPTOR_PQR, RXN_TARGETS_REC)
lig_atoms = find_lig_atoms(LIGAND_PQR, RXN_TARGETS_LIG)
if len(rec_atoms) != len(RXN_TARGETS_REC):
    print(f"Error: Missing receptor atoms. Found: {rec_atoms}")
    sys.exit(1)
# Check unique ligand atoms found
unique_lig_names = set(t[0] for t in RXN_TARGETS_LIG)
if len(lig_atoms) != len(unique_lig_names):
    print(f"Error: Missing ligand atoms. Found: {lig_atoms}")
    sys.exit(1)
# Build pairs: match targets in order
rxn_pairs = []
for i, ((r_resi, r_name, _), (l_name,)) in enumerate(zip(RXN_TARGETS_REC, RXN_TARGETS_LIG)):
    r_idx = rec_atoms[(r_resi, r_name)]
    l_idx = lig_atoms[l_name]
    cutoff = RXN_CUTOFFS[i]
    rxn_pairs.append((r_idx, l_idx, cutoff))
    print(f"  Pair {i+1}: rec[{r_idx}] <-> lig[{l_idx}]  ({cutoff} A)")

# Step 6: Fill rxns.xml
print("\nStep 6: Fill rxns.xml")
with open(RXNS_XML) as fh:
    rxns_content = fh.read()
pair_lines = ""
for r_idx, l_idx, cutoff in rxn_pairs:
    pair_lines += f'        <pair><atoms> {r_idx} {l_idx} </atoms><distance> {cutoff:.1f} </distance></pair>\n'

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
for f in ["tleap_protein.in", "tleap_ligand.in", "leap_protein.log",
          "leap_ligand.log", "leap.log", "ligand.frcmod", "ligand.mol2",
          "ligand_raw.pdb", "protein.pdb"]:
    if os.path.exists(f):
        os.remove(f)
# Rename tleap outputs to match PQR naming
if os.path.exists("protein_leap.pdb"):
    os.rename("protein_leap.pdb", "receptor.pdb")
if os.path.exists("ligand_leap.pdb"):
    os.rename("ligand_leap.pdb", "ligand.pdb")

# Summary
print("\n Setup complete")
print(f"  PDB:          {PDB_ID} (chain {CHAIN_REC}, DFG-in)")
print(f"  Receptor:     p38 MAPK ({len(rec_lines)} atoms, Q={rec_charge:+.1f} e)")
print(f"  Ligand:       SB203580 ({len(lig_lines)} atoms, Q={lig_charge:+.1f} e)")
print(f"  Force field:  ff14SB (protein) + GAFF2/AM1-BCC (ligand)")
print(f"  Ionic str:    {float(ION_CONCENTRATION)*1000:.0f} mM (Debye = {DEBYE_LENGTH} A)")
print(f"  b-surface:    {BD_MILESTONE_RADIUS} A")
print(f"  R_hydro:      auto (both)")
print(f"  max_dt:       {MAX_DT} ps (no cap, protein-ligand)")
print(f"  Rxn criterion: {len(rxn_pairs)} pairs, n_needed={N_NEEDED}")
for i, (r_idx, l_idx, cutoff) in enumerate(rxn_pairs):
    r_resi, r_name, r_resn = RXN_TARGETS_REC[i]
    l_name = RXN_TARGETS_LIG[i][0]
    print(f"    Pair {i+1}: {r_resn}{r_resi} {r_name} <-> {l_name} ({cutoff} A)")
print(f"\n  To run:")
print(f"    python run_pystarc.py input.xml")
