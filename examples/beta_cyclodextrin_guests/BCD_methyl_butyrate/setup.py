#!/usr/bin/env python3
"""
PySTARC setup script.
"""
import numpy as np
import subprocess
import shutil
import sys
import os
import re

#######################################################################################################################
# User settings
PDB                     = "complex.pdb"           # Bound-state PDB (may contain water/ions)
PRMTOP                  = "complex.parm7"         # AMBER topology file
LIGAND_RESNAME          = "APN"                   # Ligand residue name in PDB
EXCLUDED                = {"WAT"}                 # Residues to strip (water, ions)
CONTACT_CUTOFF          = 5.0                     # Max distance (A) to find contacts in bound state
BUFFER                  = 2.0                     # Added to crystal distance for reaction cutoff
N_NEEDED                = 4                       # Pairs that must be satisfied simultaneously
N_PAIRS                 = 8                       # Max number of contact pairs to use
CONTACT_MODE            = "all"                   # all=any heavy, polar=N/O/S both, nonpolar=C both, any_polar=N/O/S either, receptor_polar=N/O/S rec only
RECEPTOR_RESNAME        = "MGO"                   # Receptor residue name for XML
RECEPTOR_PQR            = "receptor.pqr"          # Output receptor PQR filename
LIGAND_PQR              = "ligand.pqr"            # Output ligand PQR filename
RXNS_XML                = "rxns.xml"              # Output reaction criterion filename
BD_MILESTONE_RADIUS     = "30.0000"               # b-surface start radius (A)
R_HYDRO_REC             = "0.0000"                # Receptor hydrodynamic radius (0=auto)
R_HYDRO_LIG             = "0.0000"                # Ligand hydrodynamic radius (0=auto)
DEBYE_LENGTH            = "7.85751"               # Debye screening length (A)
ION_CONCENTRATION       = "0.15"                  # Salt concentration (M)
ION_RADIUS_POS          = "0.95"                  # Cation radius - Na+ Pauling (A)
ION_RADIUS_NEG          = "1.81"                  # Anion radius - Cl- Pauling (A)
PDIE                    = "4.0"                   # Protein dielectric constant
SDIE                    = "78.0"                  # Solvent dielectric constant
SRAD                    = "1.4"                   # Solvent probe radius (A)
APBS_CGLEN              = "0"                     # APBS coarse grid length (0=auto)
APBS_FGLEN              = "96"                    # APBS fine grid length (A)
APBS_DIME               = "257"                   # APBS grid points per dimension
APBS_COARSE_DIME        = "0"                     # APBS coarse grid dime (0=auto)
APBS_FINE_DIME          = "0"                     # APBS fine grid dime (0=auto)
GPU_FORCE_BATCH         = "0"                     # GPU force batch size (0=auto)
DESOLVATION_ALPHA       = "0.0795775"             # Desolvation coupling constant
HYDRODYNAMIC_INTERACTIONS = "true"                # Include HI corrections
OVERLAP_CHECK           = "true"                  # Reject overlapping configurations
MULTIPOLE_FALLBACK      = "true"                  # Yukawa fallback outside grid
LJ_FORCES               = "false"                 # Lennard-Jones short-range forces
N_TRAJECTORIES          = "100000"                # Number of BD trajectories
MAX_STEPS               = "1000000"               # Max steps per trajectory
DT                      = "0.2"                   # Base timestep (ps)
MINIMUM_CORE_DT         = "0.2"                   # Minimum timestep near core (ps)
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
#######################################################################################################################

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

#  Contact filter
polar_set = {'N', 'O', 'S'}
def contact_filter(r_name, l_name, mode):
    r_elem = r_name[0]
    l_elem = l_name[0]
    if mode == "all":
        return True
    elif mode == "polar":
        return r_elem in polar_set and l_elem in polar_set
    elif mode == "nonpolar":
        return r_elem == 'C' and l_elem == 'C'
    elif mode == "any_polar":
        return r_elem in polar_set or l_elem in polar_set
    elif mode == "receptor_polar":
        return r_elem in polar_set
    else:
        print(f"Error: Unknown CONTACT_MODE '{mode}'")
        print("  Options: all, polar, nonpolar, any_polar, receptor_polar")
        sys.exit(1)

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

# Check input files
for f in [PRMTOP, PDB]:
    if not os.path.exists(f):
        print(f"Error: {f} not found")
        sys.exit(1)

#  Step 1: Generate PQR files
print("\nStep 1: Generate PQR files")
print("  cpptraj: PDB -> rst7")
r = subprocess.run("cpptraj",
                   input=f"parm {PRMTOP}\ntrajin {PDB}\ntrajout _full.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
if r.returncode != 0:
    print(f"Error cpptraj: {r.stderr[:200]}")
    sys.exit(1)
print("  ambpdb: prmtop + rst7 -> PQR")
r = subprocess.run(f"ambpdb -p {PRMTOP} -c _full.rst7 -pqr",
                   shell=True, capture_output=True, text=True)
if r.returncode != 0:
    print(f"Error ambpdb: {r.stderr[:200]}")
    sys.exit(1)
rec_lines, lig_lines = [], []
for line in r.stdout.strip().split('\n'):
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    resname = line.split()[3]
    if resname == LIGAND_RESNAME:
        lig_lines.append(line)
    elif resname not in EXCLUDED:
        rec_lines.append(line)
with open("receptor.pqr", "w") as fh:
    fh.write('\n'.join(rec_lines) + '\nEND\n')
with open("ligand.pqr", "w") as fh:
    fh.write('\n'.join(lig_lines) + '\nEND\n')
if os.path.exists("_full.rst7"):
    os.remove("_full.rst7")
rec_charge = sum(float(l.split()[8]) for l in rec_lines)
lig_charge = sum(float(l.split()[8]) for l in lig_lines)
print(f"  receptor.pqr: {len(rec_lines)} atoms, Q={rec_charge:+.2f} e")
print(f"  ligand.pqr:   {len(lig_lines)} atoms, Q={lig_charge:+.2f} e")

#  Step 2: Find binding-site contacts
print(f"\nStep 2: Find binding-site contacts (mode: {CONTACT_MODE})")
rec_pdb, lig_pdb = [], []
with open(PDB) as fh:
    for line in fh:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        a = {
            'name': line[12:16].strip(),
            'res': line[17:20].strip(),
            'resi': int(line[22:26]),
            'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        }
        if a['res'] == LIGAND_RESNAME:
            lig_pdb.append(a)
        elif a['res'] not in EXCLUDED:
            rec_pdb.append(a)
rec_xyz = np.array([a['xyz'] for a in rec_pdb])
lig_xyz = np.array([a['xyz'] for a in lig_pdb])
dmat = np.sqrt(((rec_xyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(axis=2))
all_contacts = []
for i, r in enumerate(rec_pdb):
    if r['name'].startswith('H'):
        continue
    for j, l in enumerate(lig_pdb):
        if l['name'].startswith('H'):
            continue
        d = dmat[i, j]
        if d < CONTACT_CUTOFF:
            all_contacts.append((d, r, l))
all_contacts.sort()
filtered_contacts = [(d, r, l) for d, r, l in all_contacts
                     if contact_filter(r['name'], l['name'], CONTACT_MODE)]
print(f"  Heavy-atom contacts within {CONTACT_CUTOFF} A: {len(all_contacts)}")
print(f"  Filtered contacts ({CONTACT_MODE}): {len(filtered_contacts)}")
print(f"\n  Closest contact per receptor residue:")
print(f"  {'Dist':>6s}  {'Rec atom':<6s}  {'Rec res':<10s}  {'Lig atom':<6s}  {'Type'}")
print(f"  {'-' * 48}")
seen = set()
for d, r, l in filtered_contacts:
    if r['resi'] not in seen:
        seen.add(r['resi'])
        r_elem = r['name'][0]
        l_elem = l['name'][0]
        if r_elem in polar_set and l_elem in polar_set:
            tag = "polar"
        elif r_elem == 'C' and l_elem == 'C':
            tag = "nonpolar"
        else:
            tag = "mixed"
        print(f"  {d:6.2f}  {r['name']:<6s}  {r['res']:<4s}{r['resi']:<6d}  {l['name']:<6s}  {tag}")

#  Step 3: Map contacts to PQR line numbers
print("\nStep 3: Map contacts to PQR line numbers")
def load_pqr(filename):
    atoms = []
    with open(filename) as fh:
        for i, line in enumerate(fh, 1):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                atoms.append({'line': i, 'name': parts[2], 'res': parts[3], 'resi': int(parts[4])})
    return atoms
rec_pqr = load_pqr("receptor.pqr")
lig_pqr = load_pqr("ligand.pqr")
rec_lookup = {(a['resi'], a['name']): a['line'] for a in rec_pqr}
lig_lookup = {(a['resi'], a['name']): a['line'] for a in lig_pqr}
mapped = []
seen_resi = set()
print(f"  {'Crystal':>8s}  {'Cutoff':>7s}  {'Rec line':>8s}  {'Rec':>15s}  {'Lig line':>8s}  {'Lig':>6s}  {'Status'}")
print(f"  {'-' * 70}")
for d, r, l in filtered_contacts:
    if r['resi'] in seen_resi:
        continue
    seen_resi.add(r['resi'])
    rl = rec_lookup.get((r['resi'], r['name']))
    ll = lig_lookup.get((l['resi'], l['name']))
    status = "OK" if (rl and ll) else "MISSING"
    cutoff = round((d + BUFFER) * 2) / 2
    print(f"  {d:8.2f}  {cutoff:7.1f}  {str(rl or '???'):>8s}  {r['name']:<4s} {r['res']}{r['resi']:<6d}  "
          f"{str(ll or '???'):>8s}  {l['name']:<6s}  {status}")
    if rl and ll:
        mapped.append((d, r, l, rl, ll))
mapped = mapped[:N_PAIRS]
print(f"\n  Mapped: {len(mapped)} pairs (top {N_PAIRS}), n_needed: {N_NEEDED}")
if len(mapped) < N_NEEDED:
    print(f"Error: Only {len(mapped)} pairs mapped, need at least {N_NEEDED}")
    sys.exit(1)

#  Step 4: Fill rxns.xml
print("\nStep 4: Fill rxns.xml")
with open("rxns.xml") as fh:
    rxns_content = fh.read()
pair_lines = ""
for d, r, l, rl, ll in mapped:
    cutoff = round((d + BUFFER) * 2) / 2
    pair_lines += f'        <pair><atoms> {rl} {ll} </atoms><distance> {cutoff:.1f} </distance></pair>\n'
rxns_content = re.sub(r'<n_needed>\s*</n_needed>', f'<n_needed> {N_NEEDED} </n_needed>', rxns_content)
rxns_content = re.sub(r'\s*<pair><atoms>\s*</atoms><distance>\s*</distance></pair>\s*\n', '\n' + pair_lines, rxns_content)
with open("rxns.xml", "w") as fh:
    fh.write(rxns_content)
print(f"  Filled: rxns.xml ({len(mapped)} pairs)")

#  Step 5: Fill input.xml
print("\nStep 5: Fill PySTARC input.xml")
with open("input.xml") as fh:
    input_content = fh.read()
for tag, val in PARAMS.items():
    input_content = re.sub(f'<{tag}>\\s*</{tag}>', f'<{tag}>{val}</{tag}>', input_content)
with open("input.xml", "w") as fh:
    fh.write(input_content)
print(f"  Filled: input.xml")

#  Summary
print("\nSetup complete")
print(f"  Contact mode       : {CONTACT_MODE}")
print(f"  receptor.pqr       : {len(rec_lines)} atoms, Q={rec_charge:+.2f} e")
print(f"  ligand.pqr         : {len(lig_lines)} atoms, Q={lig_charge:+.2f} e")
print(f"  rxns.xml           : {len(mapped)} pairs, n_needed={N_NEEDED}")
print(f"  input.xml          : {N_TRAJECTORIES} trajectories")
print(f"\n  To run:")
print(f"    python run_pystarc.py input.xml")
