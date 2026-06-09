#!/usr/bin/env python3

import numpy as np
import subprocess
import shutil
import sys
import re
import os 

SOURCE_PDB = "complex.pdb"
RECEPTOR_PQR = "receptor.pqr"
LIGAND_PQR   = "ligand.pqr"
RXNS_XML     = "rxns.xml"

# Auto-discovered HSP90 pocket Ca atoms + closest ligand atoms (from crystal pose)
RXN_TARGETS_REC = [(35, 'CA', 'ASN'), (122, 'CA', 'PHE'), (36, 'CA', 'SER')]   # [(resid, 'CA', resname), ...]
RXN_TARGETS_LIG = [('C6x',), ('O1x',), ('C8x',)]   # [('O2x',), ...] - original espaloma atom names
RXN_CUTOFFS = [5.0, 5.0, 5.0]

PARAMS = {
    "receptor_resname": "HSP90", "ligand_resname": "UNK",
    "receptor_pqr": RECEPTOR_PQR, "ligand_pqr": LIGAND_PQR, "rxns_xml": RXNS_XML,
    "bd_milestone_radius": "55.0", "r_hydro_rec": "0", "r_hydro_lig": "0",
    "debye_length": "7.86", "ion_concentration": "0.15",
    "ion_radius_pos": "0.95", "ion_radius_neg": "1.81",
    "pdie": "4.0", "sdie": "78.0", "srad": "1.4",
    "apbs_cglen": "0", "apbs_fglen": "144", "apbs_dime": "257",
    "apbs_coarse_dime": "0", "apbs_fine_dime": "0",
    "gpu_force_batch": "1000", "desolvation_alpha": "0.0795775",
    "hydrodynamic_interactions": "true", "overlap_check": "false",
    "multipole_fallback": "true", "lj_forces": "false", "enable_born2_torque": "false",
    "n_trajectories": "100000", "max_steps": "1000000",
    "dt": "0.2", "minimum_core_dt": "0.2", "max_dt": "0",
    "temperature": "298.15", "seed": "1",
    "checkpoint_interval": "0", "convergence_interval": "10",
    "convergence_check": "true", "convergence_tol": "0.05",
    "gpu": "true", "n_threads": "32", "work_dir": "bd_sims", "save_interval": "10",
}

# Locate pystarc templates
PYSTARC_DIR = None
sd = os.path.dirname(os.path.abspath(__file__))
for i in range(10):
    pf = os.path.join(sd, *[".."]*i) if i else sd
    for sub in ("pystarc", "PySTARC/pystarc"):
        c = os.path.join(pf, sub)
        if os.path.isdir(os.path.join(c, "templates")):
            PYSTARC_DIR = os.path.abspath(c); break
    if PYSTARC_DIR: break
if PYSTARC_DIR is None:
    try: import pystarc; PYSTARC_DIR = os.path.dirname(os.path.abspath(pystarc.__file__))
    except ImportError: pass
if PYSTARC_DIR is None:
    print("ERROR: pystarc/ not found"); sys.exit(1)
TPL = os.path.join(PYSTARC_DIR, "templates")
shutil.copy(os.path.join(TPL, "input.xml"), "input.xml")
shutil.copy(os.path.join(TPL, "rxns.xml"),  "rxns.xml")

# Step 1: Extract protein and ligand from seekrflow PDB
print(f"Step 1: Extract protein + ligand from seekrflow PDB (43)")
prot_lines, lig_lines = [], []
with open(SOURCE_PDB) as f:
    for line in f:
        if line.startswith("ATOM"):
            an = line[12:16].strip()
            if an[0] == 'H' or (len(an) > 1 and an[0].isdigit() and an[1] == 'H'): continue
            prot_lines.append(line)
        elif line.startswith("HETATM"):
            rn = line[17:20].strip()
            if rn in ('HOH','WAT','NA','CL','MG','ZN','ACE','NME'): continue
            lig_lines.append(line)
with open("protein.pdb", "w") as f:
    f.write(''.join(prot_lines) + 'END\n')
with open("ligand_raw.pdb", "w") as f:
    f.write(''.join(lig_lines) + 'END\n')
print(f"  protein.pdb: {len(prot_lines)} ATOM lines")
print(f"  ligand_raw.pdb: {len(lig_lines)} HETATM lines")

# Step 2: tleap for protein
print("Step 2: tleap for protein (ff14SB)")
with open("tleap_protein.in", "w") as f:
    f.write("source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\nprotein = loadpdb protein.pdb\ncheck protein\nsaveamberparm protein protein.prmtop protein.rst7\nsavepdb protein protein_leap.pdb\nquit\n")
r = subprocess.run("tleap -f tleap_protein.in", shell=True, capture_output=True, text=True)
with open("leap_protein.log", "w") as f: f.write(r.stdout + r.stderr)
if not os.path.exists("protein.prmtop") or os.path.getsize("protein.prmtop") == 0:
    print("  ERROR: tleap failed (empty/missing protein.prmtop). Check leap_protein.log"); sys.exit(1)
print("  protein.prmtop, protein.rst7 created")

# Step 3: OpenEye + antechamber for ligand
print("Step 3: Ligand parameterization (OpenEye AM1-BCC + GAFF2)")
from openeye import oechem, oequacpac
ifs = oechem.oemolistream("ligand_raw.pdb"); ifs.SetFormat(oechem.OEFormat_PDB)
mol = oechem.OEMol(); oechem.OEReadMolecule(ifs, mol)
oechem.OEAddExplicitHydrogens(mol)
NETQ = sum(a.GetFormalCharge() for a in mol.GetAtoms())
print(f"  Detected net formal charge: {NETQ:+d} e ({mol.NumAtoms()} atoms)")
if not oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCCharges()):
    print("  ERROR: OEAssignCharges failed"); sys.exit(1)
ofs = oechem.oemolostream("ligand_oe.mol2"); ofs.SetFormat(oechem.OEFormat_MOL2)
oechem.OEWriteMolecule(ofs, mol); ofs.close()
r = subprocess.run(f"antechamber -i ligand_oe.mol2 -fi mol2 -o ligand.mol2 -fo mol2 -at gaff2 -nc {NETQ}",
                   shell=True, capture_output=True, text=True)
if not os.path.exists("ligand.mol2"):
    print(f"  ERROR: antechamber failed\n  {r.stderr[:500]}"); sys.exit(1)
r = subprocess.run("parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2", shell=True, capture_output=True, text=True)
if not os.path.exists("ligand.frcmod"):
    print(f"  ERROR: parmchk2 failed"); sys.exit(1)
with open("tleap_ligand.in", "w") as f:
    f.write("source leaprc.gaff2\nloadamberparams ligand.frcmod\nlig = loadmol2 ligand.mol2\ncheck lig\nsaveamberparm lig ligand.prmtop ligand.rst7\nsavepdb lig ligand_leap.pdb\nquit\n")
r = subprocess.run("tleap -f tleap_ligand.in", shell=True, capture_output=True, text=True)
with open("leap_ligand.log", "w") as f: f.write(r.stdout + r.stderr)
if not os.path.exists("ligand.prmtop") or os.path.getsize("ligand.prmtop") == 0:
    print("  ERROR: tleap (ligand) failed (empty/missing ligand.prmtop). Check leap_ligand.log"); sys.exit(1)
print("  ligand.prmtop, ligand.rst7 created")

# Step 4: Generate PQRs via ambpdb (standard format)
print("Step 4: Generate PQRs")
r = subprocess.run("cpptraj", input="parm protein.prmtop\ntrajin protein.rst7\ntrajout _prot.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
r = subprocess.run("ambpdb -p protein.prmtop -c _prot.rst7 -pqr", shell=True, capture_output=True, text=True)
rec_pqr_lines = [l for l in r.stdout.strip().split('\n') if l.startswith("ATOM")]
with open(RECEPTOR_PQR, "w") as f: f.write('\n'.join(rec_pqr_lines) + '\nEND\n')
rec_q = sum(float(l.split()[8]) for l in rec_pqr_lines)
print(f"  receptor.pqr: {len(rec_pqr_lines)} atoms, Q={rec_q:+.1f} e")
r = subprocess.run("cpptraj", input="parm ligand.prmtop\ntrajin ligand.rst7\ntrajout _lig.rst7\nrun\n",
                   shell=False, capture_output=True, text=True)
r = subprocess.run("ambpdb -p ligand.prmtop -c _lig.rst7 -pqr", shell=True, capture_output=True, text=True)
lig_pqr_lines = [l for l in r.stdout.strip().split('\n') if l.startswith("ATOM")]
with open(LIGAND_PQR, "w") as f: f.write('\n'.join(lig_pqr_lines) + '\nEND\n')
lig_q = sum(float(l.split()[8]) for l in lig_pqr_lines)
print(f"  ligand.pqr:   {len(lig_pqr_lines)} atoms, Q={lig_q:+.1f} e")
for f in ["_prot.rst7", "_lig.rst7", "ligand_oe.mol2", "sqm.in", "sqm.out", "sqm.pdb",
          "ANTECHAMBER_AC.AC", "ANTECHAMBER_AC.AC0", "ANTECHAMBER_BOND_TYPE.AC",
          "ANTECHAMBER_BOND_TYPE.AC0", "ANTECHAMBER_AM1BCC.AC",
          "ANTECHAMBER_AM1BCC_PRE.AC", "ATOMTYPE.INF"]:
    if os.path.exists(f): os.remove(f)

# Step 5: Find reaction atoms by closest heavy-atom contacts (contact method).
# Parameters match the validated beta-CD setup. Reads rec_pqr_lines / lig_pqr_lines
# (built in Step 4) so indices match the final receptor.pqr / ligand.pqr atom order.
print("Step 5: Find reaction atoms (contact method)")
CONTACT_CUTOFF = 5.0
BUFFER = 2.0
N_PAIRS = 8
N_NEEDED = 6
CONTACT_MODE = "all"
_POLAR = {"N", "O", "S"}

def _pqr_atoms(lines):
    # Returns list of (idx_1based, atom_name, resi, np.array([x,y,z])) in file order.
    out = []
    n = 0
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        n += 1
        nm = ln[12:16].strip()
        try:
            resi = int(ln[22:26])
        except ValueError:
            resi = int(ln.split()[4])
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        out.append((n, nm, resi, xyz))
    return out

def _contact_ok(rn, ln_, mode):
    re_, le_ = rn[0], ln_[0]
    if mode == "all":
        return True
    if mode == "polar":
        return re_ in _POLAR and le_ in _POLAR
    if mode == "nonpolar":
        return re_ == "C" and le_ == "C"
    if mode == "any_polar":
        return re_ in _POLAR or le_ in _POLAR
    if mode == "receptor_polar":
        return re_ in _POLAR
    print(f"  ERR unknown CONTACT_MODE {mode}"); sys.exit(1)

_rec_atoms_xyz = _pqr_atoms(rec_pqr_lines)
_lig_atoms_xyz = _pqr_atoms(lig_pqr_lines)
_contacts = []
for ri, rnm, rresi, rxyz in _rec_atoms_xyz:
    if rnm.startswith("H"):
        continue
    for li, lnm, lresi, lxyz in _lig_atoms_xyz:
        if lnm.startswith("H"):
            continue
        d = float(np.linalg.norm(rxyz - lxyz))
        if d < CONTACT_CUTOFF and _contact_ok(rnm, lnm, CONTACT_MODE):
            _contacts.append((d, ri, rnm, rresi, li, lnm))
_contacts.sort(key=lambda t: t[0])
pairs = []
_seen_resi = set()
for d, ri, rnm, rresi, li, lnm in _contacts:
    if rresi in _seen_resi:
        continue
    _seen_resi.add(rresi)
    cutoff = 5.0
    pairs.append((ri, li, cutoff))
    print(f"  Pair {len(pairs)}: rec[{ri}] {rnm}{rresi} <-> lig[{li}] {lnm}  d={d:.2f} cut={cutoff:.1f}")
    if len(pairs) >= N_PAIRS:
        break
if len(pairs) < N_NEEDED:
    print(f"  ERR only {len(pairs)} contact pairs found, need >= {N_NEEDED}"); sys.exit(1)

# Step 6: Fill rxns.xml (per-pair cutoffs)
with open(RXNS_XML) as f: txt = f.read()
ln = "".join(f"        <pair><atoms> {a} {b} </atoms><distance> {c:.1f} </distance></pair>\n" for a,b,c in pairs)
txt = re.sub(r"<n_needed>\s*</n_needed>", f"<n_needed> {N_NEEDED} </n_needed>", txt)
txt = re.sub(r"\s*<pair><atoms>\s*</atoms><distance>\s*</distance></pair>\s*\n", "\n"+ln, txt)
with open(RXNS_XML, "w") as f: f.write(txt)
print(f"Step 6: rxns.xml filled ({len(pairs)} pairs, n_needed={N_NEEDED})")

# Step 7: Fill input.xml
with open("input.xml") as f: txt = f.read()
for k,v in PARAMS.items():
    txt = re.sub(f"<{k}>\\s*</{k}>", f"<{k}>{v}</{k}>", txt)
with open("input.xml", "w") as f: f.write(txt)
print("Step 7: input.xml filled")

# Step 8: Geometry check
rec_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rec_pqr_lines])
lig_xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lig_pqr_lines])
rmax = np.max(np.linalg.norm(rec_xyz - rec_xyz.mean(0), axis=1))
lmax = np.max(np.linalg.norm(lig_xyz - lig_xyz.mean(0), axis=1))
b = float(PARAMS["bd_milestone_radius"]); fg = float(PARAMS["apbs_fglen"])
print(f"Step 8: rmax={rmax:.1f}A  lmax={lmax:.1f}A  b={b:.0f}A  fglen={fg:.0f}A  ", end="")
print("OK" if b+lmax < fg/2 else "WARNING grid")

# Cleanup
for f in ["tleap_protein.in", "tleap_ligand.in", "leap_protein.log",
          "leap_ligand.log", "leap.log", "ligand.frcmod", "ligand.mol2",
          "ligand_raw.pdb", "protein.pdb"]:
    if os.path.exists(f): os.remove(f)
if os.path.exists("protein_leap.pdb"): os.rename("protein_leap.pdb", "receptor.pdb")
if os.path.exists("ligand_leap.pdb"): os.rename("ligand_leap.pdb", "ligand.pdb")

print(f"\nSetup done: HSP90 43")
print(f"  Receptor: HSP90 ({len(rec_pqr_lines)} atoms, Q={rec_q:+.1f} e)")
print(f"  Ligand:   ({len(lig_pqr_lines)} atoms, Q={lig_q:+.1f} e, formal={NETQ:+d})")
print(f"  Rxn: {len(pairs)} pairs at 7.0 A, n_needed={N_NEEDED}")
