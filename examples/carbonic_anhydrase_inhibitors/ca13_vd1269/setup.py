#!/usr/bin/env python3
"""
PySTARC setup script for CA XIII / acetazolamide (AZM) protein-ligand benchmark.

PDB: 3CZV (Baranauskiene & Matulis, 2012)
  Human carbonic anhydrase XIII (chain A, 258 residues) + acetazolamide

Experimental intrinsic k_on: 1.5 x 10^6 M^-1 s^-1 (Linkuviene et al., J. Med. Chem. 2018)
  Observed k_on at pH 7.0: 4.1 x 10^5 M^-1 s^-1 (SPR)
  BD models the deprotonated (binding-competent) sulfonamide, so the intrinsic
  rate is the correct comparison target.

Receptor: CA XIII chain A (residues 4-261) + Zn2+ (HETATM)
Ligand: acetazolamide deprotonated (NH-SO2R, net charge = -1)
  Generated from SMILES via rdkit -> SDF -> obabel mol2 -> antechamber
  (antechamber cannot handle thiadiazole ring directly from PDB)
"""

import numpy as np
import subprocess
import shutil
import sys
import os
import re

# User settings
PDB_ID                  = "3CZV"                  # PDB accession code
CHAIN_REC               = "A"                     # Protein chain ID in PDB
LIGAND_RESID            = "V17"                   # Ligand residue name
# Deprotonated AZM SMILES: sulfonamide NH- (charge -1)
LIGAND_SMILES           = "COC1=CC=C(CNC2=C(C(F)=C(F)C(F)=C2F)S([NH-])(=O)=O)C=C1"
LIGAND_NET_CHARGE       = -1                      # Net charge of deprotonated AZM
RECEPTOR_RESNAME        = "CA13"                  # Receptor label for XML
LIGAND_RESNAME          = "V17"                   # Ligand label for XML
RECEPTOR_PQR            = "receptor.pqr"          # Output receptor PQR filename
LIGAND_PQR              = "ligand.pqr"            # Output ligand PQR filename
RXNS_XML                = "rxns.xml"              # Output reaction criterion filename
# Reaction criterion from crystal contacts (PDB numbering -> tleap offset -3):
#   Thr199 OG1 <-> AZM sulfonamide N (Zn-coordinating N, gatekeeper H-bond)
#   Glu106 OE1 <-> AZM amide N (relay H-bond)
# tleap residues: Thr199->196, Glu106->103
RXN_TARGETS_REC         = [(196, 'OG1', 'THR'), (103, 'OE1', 'GLU')]
RXN_TARGETS_LIG_NAMES   = ['N1', 'N']             # Sulfonamide N and amide N in ligand PQR
RXN_CUTOFF              = 5.0                     # Reaction distance cutoff (A)
N_NEEDED                = 2                       # Pairs that must be satisfied simultaneously
# Simulation parameters
BD_MILESTONE_RADIUS     = "60.0"                  # b-surface start radius (A)
R_HYDRO_REC             = "0"                     # Receptor hydrodynamic radius (0=auto)
R_HYDRO_LIG             = "0"                     # Ligand hydrodynamic radius (0=auto)
DEBYE_LENGTH            = "9.62"                  # Debye screening length (A) for 100 mM
ION_CONCENTRATION       = "0.10"                  # Salt concentration (M) - SPR conditions
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
MAX_STEPS               = "10000000"              # Max steps per trajectory
DT                      = "0.2"                   # Base timestep (ps)
MINIMUM_CORE_DT         = "0.2"                   # Minimum timestep near core (ps)
MAX_DT                  = "0"                     # Max timestep ceiling (0=off, small ligand)
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

# Find pystarc templates
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
    sys.exit(1)
TEMPLATE_DIR = os.path.join(PYSTARC_DIR, "templates")
print(f"Templates: {TEMPLATE_DIR}")

# Copy templates
shutil.copy(os.path.join(TEMPLATE_DIR, "input.xml"), "input.xml")
shutil.copy(os.path.join(TEMPLATE_DIR, "rxns.xml"), "rxns.xml")
print("Copied: input.xml, rxns.xml")

# Step 1: Download PDB
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

# Step 2: Extract chain A protein + Zn (remove OXT to avoid tleap error)
print(f"\nStep 2: Extract chain {CHAIN_REC} protein + Zn")
with open(pdb_file) as fh:
    lines = []
    for l in fh:
        if l.startswith("ATOM") and len(l) > 21 and l[21] == CHAIN_REC:
            if "OXT" not in l:
                lines.append(l)
        if l.startswith("HETATM") and " ZN" in l and len(l) > 21 and l[21] == CHAIN_REC:
            lines.append(l)
with open("_protein.pdb", 'w') as fh:
    fh.write(''.join(lines) + 'END\n')
n_protein = sum(1 for l in lines if l.startswith("ATOM"))
n_zn = sum(1 for l in lines if "ZN" in l and l.startswith("HETATM"))
print(f"  _protein.pdb: {n_protein} protein atoms + {n_zn} Zn")

# Step 3: Generate deprotonated AZM from SMILES
print(f"\nStep 3: Generate deprotonated {LIGAND_RESID} from SMILES")
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles(LIGAND_SMILES)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
Chem.MolToMolFile(mol, '_azm.sdf')
n_lig_atoms = mol.GetNumAtoms()
lig_charge = Chem.GetFormalCharge(mol)
print(f"  _azm.sdf: {n_lig_atoms} atoms, charge = {lig_charge}")
r = subprocess.run("obabel _azm.sdf -o mol2 -O _azm.mol2",
                   shell=True, capture_output=True, text=True)
if not os.path.exists("_azm.mol2"):
    print(f"  ERROR: obabel failed: {r.stderr[:300]}")
    sys.exit(1)
print("  _azm.mol2 created")

# Step 4: Parameterize ligand with antechamber (GAFF2 + AM1-BCC)
print(f"\nStep 4: Parameterize {LIGAND_RESID} with antechamber")
r = subprocess.run(
    f"antechamber -i _azm.mol2 -fi mol2 -o ligand.mol2 -fo mol2 "
    f"-c bcc -at gaff2 -nc {LIGAND_NET_CHARGE} -rn {LIGAND_RESID} -pf y -j 4",
    shell=True, capture_output=True, text=True
)
if not os.path.exists("ligand.mol2"):
    print(f"  ERROR: antechamber failed:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    sys.exit(1)
print("  ligand.mol2 created")
r = subprocess.run("parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2",
                   shell=True, capture_output=True, text=True)
if not os.path.exists("ligand.frcmod"):
    print(f"  ERROR: parmchk2 failed: {r.stderr[:300]}")
    sys.exit(1)
print("  ligand.frcmod created")

# Step 5: Parameterize receptor with tleap (ff14SB + Zn ion params)
print("\nStep 5: Parameterize receptor with tleap")
tleap_rec = """source leaprc.protein.ff14SB
source leaprc.water.tip3p
loadAmberParams frcmod.ions234lm_126_tip3p
protein = loadpdb _protein.pdb
check protein
saveamberparm protein protein.prmtop protein.rst7
savepdb protein receptor.pdb
quit
"""
with open("_tleap_rec.in", "w") as fh:
    fh.write(tleap_rec)
r = subprocess.run("tleap -f _tleap_rec.in", shell=True, capture_output=True, text=True)
if not os.path.exists("protein.prmtop") or os.path.getsize("protein.prmtop") == 0:
    print(f"  ERROR: tleap receptor failed.")
    print(r.stdout[-500:])
    sys.exit(1)
print("  protein.prmtop, protein.rst7, receptor.pdb created")

# Step 6: Parameterize ligand with tleap
print("\nStep 6: Parameterize ligand with tleap")
tleap_lig = f"""source leaprc.gaff2
loadAmberParams ligand.frcmod
{LIGAND_RESID} = loadmol2 ligand.mol2
check {LIGAND_RESID}
saveamberparm {LIGAND_RESID} ligand.prmtop ligand.rst7
savepdb {LIGAND_RESID} ligand.pdb
quit
"""
with open("_tleap_lig.in", "w") as fh:
    fh.write(tleap_lig)
r = subprocess.run("tleap -f _tleap_lig.in", shell=True, capture_output=True, text=True)
if not os.path.exists("ligand.prmtop") or os.path.getsize("ligand.prmtop") == 0:
    print(f"  ERROR: tleap ligand failed.")
    print(r.stdout[-500:])
    sys.exit(1)
print("  ligand.prmtop, ligand.rst7, ligand.pdb created")

# Step 7: Generate PQR files
print("\nStep 7: Generate PQR files")

def make_pqr(prmtop, rst7, outfile):
    r = subprocess.run("cpptraj", input=f"parm {prmtop}\ntrajin {rst7}\ntrajout _tmp.rst7\nrun\n",
                       shell=False, capture_output=True, text=True)
    r = subprocess.run(f"ambpdb -p {prmtop} -c _tmp.rst7 -pqr",
                       shell=True, capture_output=True, text=True)
    with open(outfile, "w") as fh:
        fh.write(r.stdout)
    if os.path.exists("_tmp.rst7"):
        os.remove("_tmp.rst7")
    atom_lines = [l for l in r.stdout.strip().split('\n') if l.startswith("ATOM")]
    n_atoms = len(atom_lines)
    charge = sum(float(l.split()[8]) for l in atom_lines)
    return n_atoms, charge

n_rec, q_rec = make_pqr("protein.prmtop", "protein.rst7", RECEPTOR_PQR)
n_lig, q_lig = make_pqr("ligand.prmtop", "ligand.rst7", LIGAND_PQR)
print(f"  {RECEPTOR_PQR}: {n_rec} atoms, Q={q_rec:+.2f} e")
print(f"  {LIGAND_PQR}: {n_lig} atoms, Q={q_lig:+.2f} e")

# Step 8: Find reaction criterion atom indices
print("\nStep 8: Find reaction criterion atoms")

def find_rec_atoms(pqr_file, targets):
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

def find_lig_atoms(pqr_file, names):
    found = {}
    atom_num = 0
    with open(pqr_file) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                atom_num += 1
                parts = line.split()
                name = parts[2]
                if name in names and name not in found:
                    found[name] = atom_num
                    print(f"  {pqr_file} atom {atom_num}: {name}")
    return found

rec_atoms = find_rec_atoms(RECEPTOR_PQR, RXN_TARGETS_REC)
lig_atoms = find_lig_atoms(LIGAND_PQR, RXN_TARGETS_LIG_NAMES)

# Auto-detect offset if atoms not found
if len(rec_atoms) != len(RXN_TARGETS_REC):
    print("  Trying alternative offsets...")
    for offset in range(-5, 6):
        alt = [(r+offset, n, rn) for r, n, rn in RXN_TARGETS_REC]
        test = find_rec_atoms(RECEPTOR_PQR, alt)
        if len(test) == len(RXN_TARGETS_REC):
            print(f"  Found with offset {offset}")
            rec_atoms = test
            RXN_TARGETS_REC[:] = alt
            break
    else:
        print("  ERROR: Could not find receptor atoms")
        sys.exit(1)

if len(lig_atoms) != len(RXN_TARGETS_LIG_NAMES):
    print(f"  ERROR: Missing ligand atoms. Found: {lig_atoms}")
    sys.exit(1)

# Build pairs in order
rxn_pairs = []
for (tresi, tname, tresn), lname in zip(RXN_TARGETS_REC, RXN_TARGETS_LIG_NAMES):
    r_idx = rec_atoms[(tresi, tname)]
    l_idx = lig_atoms[lname]
    rxn_pairs.append((r_idx, l_idx))
    print(f"  Pair: rec[{r_idx}] ({tresn}{tresi} {tname}) <-> lig[{l_idx}] ({lname})  ({RXN_CUTOFF} A)")

# Step 9: Fill rxns.xml
print("\nStep 9: Fill rxns.xml")
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

# Step 10: Fill input.xml
print("\nStep 10: Fill PySTARC input.xml")
with open("input.xml") as fh:
    input_content = fh.read()
for tag, val in PARAMS.items():
    input_content = re.sub(f'<{tag}>\\s*</{tag}>', f'<{tag}>{val}</{tag}>', input_content)
with open("input.xml", "w") as fh:
    fh.write(input_content)
print("  Filled: input.xml")

# Step 11: Geometry check
print("\nStep 11: Geometry check")
rec_lines = [l for l in open(RECEPTOR_PQR) if l.startswith("ATOM")]
lig_lines_pqr = [l for l in open(LIGAND_PQR) if l.startswith("ATOM")]
rec_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rec_lines])
lig_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lig_lines_pqr])
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

# Step 12: Clean up all intermediate files
print("\nStep 12: Clean up")
cleanup = [
    "_protein.pdb", "_azm.sdf", "_azm.mol2", "_tleap_rec.in", "_tleap_lig.in",
    "_tmp.rst7", "_grid_gen.xml", "leap.log", "ANTECHAMBER_AC.AC", 
    "ANTECHAMBER_AC.AC0", "ANTECHAMBER_AM1BCC.AC", "ANTECHAMBER_AM1BCC_PRE.AC", 
    "ANTECHAMBER_BOND_TYPE.AC","ANTECHAMBER_BOND_TYPE.AC0", "ANTECHAMBER_GAS.COM",
    "ANTECHAMBER_GAS.LOG", "ANTECHAMBER_GAS_OUT.LOG", "ANTECHAMBER_PREP.AC", 
    "ANTECHAMBER_PREP.AC0", "ATOMTYPE.INF", "sqm.in", "sqm.out", "sqm.pdb",
    "ligand.frcmod", "ligand.mol2"
]
removed = 0
for f in cleanup:
    if os.path.exists(f):
        os.remove(f)
        removed += 1
print(f"  Removed {removed} intermediate files")

# Summary
print("\n Setup complete")
print(f"  PDB:          {PDB_ID} (chain {CHAIN_REC})")
print(f"  Receptor:     CA XIII ({n_rec} atoms, Q={q_rec:+.1f} e, includes Zn2+)")
print(f"  Ligand:       AZM deprotonated ({n_lig} atoms, Q={q_lig:+.1f} e)")
print(f"  Force field:  ff14SB (protein) + GAFF2/AM1-BCC (ligand)")
print(f"  Ionic str:    {float(ION_CONCENTRATION)*1000:.0f} mM (Debye = {DEBYE_LENGTH} A)")
print(f"  b-surface:    {BD_MILESTONE_RADIUS} A")
print(f"  R_hydro:      auto (both)")
print(f"  max_dt:       {MAX_DT} ps (0 = off)")
print(f"  Rxn criterion: {len(rxn_pairs)} pairs at {RXN_CUTOFF} A, n_needed={N_NEEDED}")
for i, (r_idx, l_idx) in enumerate(rxn_pairs):
    tresi, tname, tresn = RXN_TARGETS_REC[i]
    lname = RXN_TARGETS_LIG_NAMES[i]
    print(f"    Pair {i+1}: {tresn}{tresi} {tname} <-> {lname}")
print(f"  Exp k_on (intrinsic): 1.5 x 10^6 M-1 s-1 (Linkuviene et al., 2018)")
print(f"  Exp k_on (observed):  4.1 x 10^5 M-1 s-1 (pH 7.0)")
print(f"\n  To run:")
print(f"    python run_pystarc.py input.xml")
