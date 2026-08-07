#!/usr/bin/env python3

from openeye import oechem, oequacpac
import xml.etree.ElementTree as ET
from rdkit.Chem import AllChem
from openeye import oechem
from rdkit import Chem
import urllib.request
import numpy as np
import collections
import subprocess
import math
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "prmtop":      dict(need=("prmtop", "complex_pdb"), opt=(), note="Uses a topology and PDB that already exist"),
    "complex_pdb": dict(need=("complex_pdb",), opt=("chain", "ligand_net_charge", "ligand_ccd"), note="Takes one PDB holding the receptor and the ligand"),
    "rcsb":        dict(need=("pdb_id", "ligand_resid"), opt=("chain", "rec_resid_min", "ligand_net_charge", "ligand_ccd"), note="Downloads the entry and takes the ligand as deposited"),
    "rcsb_smiles": dict(need=("pdb_id", "ligand_resid", "smiles"), opt=("chain", "keep_hetatm", "ligand_net_charge"), note="Takes the receptor from the entry and builds the ligand from SMILES"),
}

MODES = {
    "auto":     dict(need=("search_cutoff", "atom_filter", "n_pairs", "n_needed"), opt=("pair_cutoff", "buffer", "rounding", "excluded"), note="Locates contacts automatically and keeps one per residue"),
    "explicit": dict(need=("receptor", "ligand", "n_needed"), opt=("receptor_numbering", "cutoffs", "buffer", "rounding", "atom_filter"), note="Pairs receptor atoms you name one to one with ligand atoms"),
}

NUMBERING = {
    "pdb":      "Read as deposited",
    "internal": "Read as renumbered from 1",
}

ROUNDINGS = {
    "nearest": "Takes the distance to the nearest half angstrom",
    "up":      "Takes the distance up to the next half angstrom",
}

ATOM_FILTERS = {
    "all":            "Matches any heavy atom against any heavy atom",
    "polar":          "Matches N O or S on both sides",
    "nonpolar":       "Matches carbon on both sides",
    "any_polar":      "Matches N O or S on either side",
    "receptor_polar": "Matches N O or S on the receptor side only",
}

KEYS = {
    "ligand_source":       "Source of the receptor and ligand structures.",
    "prmtop":              "Existing AMBER topology.",
    "complex_pdb":         "Single PDB containing receptor and ligand.",
    "pdb_id":              "RCSB entry to download.",
    "ligand_resid":        "Residue name of the ligand within the RCSB entry.",
    "smiles":              "Ligand SMILES, used in place of the deposited coordinates.",
    "chain":               "Receptor chain retained. Blank retains all chains.",
    "rec_resid_min":       "Lowest receptor residue number retained.",
    "ligand_net_charge":   "Formal charge of the ligand. Blank defers to OpenEye.",
    "ligand_ccd":          "Component supplying bond orders when the composition check fails.",
    "keep_hetatm":         "Het groups retained with the receptor.",
    "ligand_formula":      "Expected ligand composition. The build aborts on a mismatch.",
    "receptor_resname":    "Receptor label written to input.xml.",
    "ligand_resname":      "Ligand label written to input.xml.",
    "mode":                "How the atom pairs of the reaction criterion are chosen.",
    "receptor_numbering":  "Numbering for the target residues.",
    "search_cutoff":       "Search radius for contacts (in angstroms).",
    "n_pairs":             "Number of contacts retained.",
    "receptor":            "Receptor side of each pair. The pair counts when this atom is within the cutoff of its ligand atom.",
    "ligand":              "Ligand side of each pair. Use auto to choose the nearest atom that atom_filter permits.",
    "atom_filter":         "Element pairings permitted.",
    "excluded":            "Atom names omitted from the search.",
    "pair_cutoff":         "One distance defining every pair (in angstroms).",
    "cutoffs":             "Distance defining each pair (in angstroms), in the same order as receptor.",
    "buffer":              "Each cutoff is the measured contact distance and the buffer (in angstroms).",
    "rounding":            "Rule reducing the buffered distance to a half angstrom.",
    "n_needed":            "Pairs that must be inside their cutoff at once for a trajectory to count as reacted.",
    "bd_milestone_radius": "Radius of the starting sphere (in angstroms).",
    "n_trajectories":      "Number of Brownian dynamics trajectories.",
    "apbs_fglen":          "Fine electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_dime":           "Grid points per axis of the electrostatic grid.",
    "apbs_cglen":          "Coarse electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_coarse_dime":    "Grid points per axis of the coarse grid. 0 uses apbs_dime.",
    "apbs_fine_dime":      "Grid points per axis of the fine grid. 0 uses apbs_dime.",
    "debye_length":        "Electrostatic screening length of the salt (in angstroms).",
    "ion_concentration":   "Salt concentration (in molar).",
    "ion_radius_pos":      "Radius of the positive ion (in angstroms).",
    "ion_radius_neg":      "Radius of the negative ion (in angstroms).",
    "pdie":                "Dielectric constant of the protein.",
    "sdie":                "Dielectric constant of the solvent.",
    "srad":                "Solvent probe radius (in angstroms).",
    "desolvation_alpha":   "Scaling applied to the desolvation penalty.",
    "enable_born2_torque": "Include the reciprocal desolvation torque on the receptor.",
    "hydrodynamic_interactions": "Include hydrodynamic coupling between the two molecules.",
    "multipole_fallback":  "Extend the electrostatics beyond the grid by multipole expansion.",
    "lj_forces":           "Apply repulsive forces at close contact.",
    "overlap_check":       "Forbid the ligand from entering the receptor volume.",
    "gpu_force_batch":     "Trajectories per GPU force evaluation. 0 selects automatically.",
    "max_steps":           "Step limit per trajectory.",
    "dt":                  "Largest permitted time step (in picoseconds).",
    "minimum_core_dt":     "Smallest permitted time step. 0 imposes no floor.",
    "max_dt":              "Ceiling on the time step. 0 imposes no ceiling.",
    "r_hydro_rec":         "Receptor hydrodynamic radius. 0 derives it from the PQR.",
    "r_hydro_lig":         "Ligand hydrodynamic radius. 0 derives it from the PQR file.",
    "temperature":         "Temperature (in kelvin).",
    "seed":                "Random number seed.",
    "gpu":                 "Evaluate forces on the GPU.",
    "n_threads":           "Number of CPU threads.",
    "convergence_check":   "Run convergence analysis after the simulation.",
    "convergence_interval": "Reporting interval for the running rate.",
    "convergence_tol":     "Relative error defining convergence.",
    "checkpoint_interval": "Trajectories between checkpoints. 0 disables checkpointing.",
    "save_interval":       "Steps between saved positions. 0 saves endpoints only.",
}

# Run parameters written into input.xml
BD_PARAMS = (
    "bd_milestone_radius", "n_trajectories", "apbs_fglen", "apbs_dime",
    "apbs_cglen", "apbs_coarse_dime", "apbs_fine_dime", "debye_length",
    "ion_concentration", "ion_radius_pos", "ion_radius_neg", "pdie", "sdie",
    "srad", "desolvation_alpha", "enable_born2_torque",
    "hydrodynamic_interactions", "multipole_fallback", "lj_forces",
    "overlap_check", "gpu_force_batch", "max_steps", "dt", "minimum_core_dt",
    "max_dt", "r_hydro_rec", "r_hydro_lig", "temperature", "seed", "gpu",
    "n_threads", "convergence_check", "convergence_interval", "convergence_tol",
    "checkpoint_interval", "save_interval",
)

BOOL = ("enable_born2_torque", "hydrodynamic_interactions", "multipole_fallback",
        "lj_forces", "overlap_check", "gpu", "convergence_check")

POLAR = {"N", "O", "S"}

# structure to PQR 
SKIP_HET = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "ACE", "NME",
            "Na+", "Cl-", "K+", "CAL"}
JUNK = ["_protein.rst7", "_ligand.rst7", "ligand_oe.mol2", "sqm.in", "sqm.out",
        "sqm.pdb", "ANTECHAMBER_AC.AC", "ANTECHAMBER_AC.AC0",
        "ANTECHAMBER_BOND_TYPE.AC", "ANTECHAMBER_BOND_TYPE.AC0",
        "ANTECHAMBER_AM1BCC.AC", "ANTECHAMBER_AM1BCC_PRE.AC", "ATOMTYPE.INF",
        "ligand_raw.sdf", "ligand_obabel.mol2"]

LEFTOVER = ["tleap_protein.in", "tleap_ligand.in", "tleap_protein.log",
            "tleap_ligand.log", "leap.log", "ligand.frcmod", "ligand.mol2",
            "ligand_raw.pdb", "protein.pdb", "protein_leap.pdb",
            "ligand_leap.pdb", "protein.prmtop", "protein.rst7",
            "ligand.prmtop", "ligand.rst7"]

def tidy(W, cfg):
    gone = list(LEFTOVER)
    code = cfg.get("ligand_ccd") or cfg.get("ligand_resid")
    if code:
        gone.append(f"{code}.cif")
    for f in gone:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)

def _cast(t):
    t = (t or "").strip()
    for f in (int, float):
        try:
            return f(t)
        except ValueError:
            pass
    return t

def _value(el):
    kids = list(el)
    if not kids:
        return _cast(el.text)
    out = []
    for c in kids:
        out.append(tuple(_cast(v) for v in c.attrib.values()) if c.attrib
                   else _cast(c.text))
    return out

def read_config(path):
    cfg = {}
    for sec in ET.parse(path).getroot():
        vals = {el.tag: v for el in sec if (v := _value(el)) != ""}
        if sec.tag == "criterion":
            cfg["criterion"] = vals
        else:
            cfg.update(vals)
    return cfg

def _sh(W, cmd, produces):
    r = subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True)
    if not os.path.exists(os.path.join(W, produces)):
        sys.exit(f"  ERROR: {cmd.split()[0]} made no {produces}\n{r.stderr[:400]}")
    return r

def _leap(W, script, body, produces):
    open(os.path.join(W, script), "w").write(body)
    r = subprocess.run(f"tleap -f {script}", shell=True, cwd=W,
                       capture_output=True, text=True)
    open(os.path.join(W, script.replace(".in", ".log")), "w").write(r.stdout + r.stderr)
    p = os.path.join(W, produces)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        sys.exit(f"  ERROR: tleap made no {produces}, see {script[:-3]}.log")

def _pqr(W, stem, out):
    """cpptraj then ambpdb, the same two calls the per system scripts used."""
    subprocess.run("cpptraj", input=f"parm {stem}.prmtop\ntrajin {stem}.rst7\n"
                   f"trajout _{stem}.rst7\nrun\n", cwd=W,
                   capture_output=True, text=True)
    r = subprocess.run(f"ambpdb -p {stem}.prmtop -c _{stem}.rst7 -pqr",
                       shell=True, cwd=W, capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l.startswith("ATOM")]
    if not lines:
        sys.exit(f"  ERROR: ambpdb produced nothing for {stem}")
    open(os.path.join(W, out), "w").write("\n".join(lines) + "\nEND\n")
    q = sum(float(l.split()[8]) for l in lines)
    print(f"  {out}: {len(lines)} atoms, Q={q:+.1f} e")

def source_prmtop(cfg, W):
    """A topology and coordinates that already exist, holding both molecules.

    One ambpdb run gives the whole complex, which is split by residue name.
    Anything in SKIP_HET, water and loose counter ions, is dropped.
    """
    top, pdb = cfg["prmtop"], cfg["complex_pdb"]
    for f in (top, pdb):
        if not os.path.exists(os.path.join(W, f)):
            sys.exit(f"  ERROR: no {f}")
    subprocess.run("cpptraj", input=f"parm {top}\ntrajin {pdb}\n"
                   f"trajout _full.rst7\nrun\n", cwd=W,
                   capture_output=True, text=True)
    r = subprocess.run(f"ambpdb -p {top} -c _full.rst7 -pqr", shell=True, cwd=W,
                       capture_output=True, text=True)
    lig_name = cfg["ligand_resname"]
    rec, lig = [], []
    for line in r.stdout.strip().split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res = line[17:20].strip()
        if res == lig_name:
            lig.append(line)
        elif res not in SKIP_HET:
            rec.append(line)
    if not lig:
        sys.exit(f"  ERROR: no residue {lig_name} in {top}")
    if not rec:
        sys.exit(f"  ERROR: nothing left for the receptor in {top}")
    for out, lines in (("receptor.pqr", rec), ("ligand.pqr", lig)):
        open(os.path.join(W, out), "w").write("\n".join(lines) + "\nEND\n")
        q = sum(float(l.split()[8]) for l in lines)
        print(f"  {out}: {len(lines)} atoms, Q={q:+.1f} e")
    p = os.path.join(W, "_full.rst7")
    if os.path.exists(p):
        os.remove(p)

def ccd_bond_orders(code, W):
    """Bond orders from the deposited chemical component, by atom name."""
    cif = os.path.join(W, f"{code}.cif")
    if not os.path.exists(cif):
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/ligands/download/{code}.cif", cif)
    out, inblk = {}, False
    for line in open(cif):
        if line.startswith("_chem_comp_bond."):
            inblk = True
            continue
        if inblk:
            p = line.split()
            if len(p) >= 4 and p[0] == code:
                o = {"SING": 1, "DOUB": 2, "TRIP": 3}.get(p[3])
                if o:
                    out[frozenset((p[1].strip(chr(34)), p[2].strip(chr(34))))] = o
            elif line.startswith("#"):
                inblk = False
    return out

def _formula_counts(formula):
    c = collections.Counter()
    for e, n in re.findall(r"([A-Z][a-z]?)\s*(\d*)", formula):
        c[e.upper()] += int(n) if n else 1
    return c

def _mol_counts(mol):
    c = collections.Counter()
    for a in mol.GetAtoms():
        c[oechem.OEGetAtomicSymbol(a.GetAtomicNum()).upper()] += 1
    return c

def _parameterise(cfg, W):
    _leap(W, "tleap_protein.in",
          "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
          "protein = loadpdb protein.pdb\ncheck protein\n"
          "saveamberparm protein protein.prmtop protein.rst7\n"
          "savepdb protein protein_leap.pdb\nquit\n", "protein.prmtop")

    def _read(use_ccd):
        ifs = oechem.oemolistream(os.path.join(W, "ligand_raw.pdb"))
        ifs.SetFormat(oechem.OEFormat_PDB)
        m = oechem.OEMol()
        oechem.OEReadMolecule(ifs, m)
        n = 0
        if use_ccd:
            ref = ccd_bond_orders(use_ccd, W)
            for b in m.GetBonds():
                k = frozenset((b.GetBgn().GetName().strip(),
                               b.GetEnd().GetName().strip()))
                if k in ref and b.GetOrder() != ref[k]:
                    b.SetOrder(ref[k])
                    n += 1
            if n:
                oechem.OEFindRingAtomsAndBonds(m)
                oechem.OEAssignAromaticFlags(m)
                # implicit counts are fixed at perception, so redo them or the
                # hydrogens follow the bonds that were just replaced
                oechem.OEAssignImplicitHydrogens(m)
                oechem.OEAssignFormalCharges(m)
        oechem.OEAddExplicitHydrogens(m)
        return m, n
    mol, _n = _read(None)
    want = _formula_counts(cfg["ligand_formula"])
    got = _mol_counts(mol)
    if got != want:
        code = cfg.get("ligand_ccd") or cfg.get("ligand_resid")
        if not code:
            sys.exit(f"  ERROR: composition {dict(got)} against {dict(want)} "
                     f"and no component code to check the bond orders against")
        print(f"  composition off by "
              f"{ {k: got.get(k, 0) - want.get(k, 0) for k in set(got) | set(want) if got.get(k, 0) != want.get(k, 0)} }"
              f", retrying with the {code} bond orders")
        mol, n = _read(code)
        print(f"  {n} bond orders taken from the {code} definition")

    netq = cfg.get("ligand_net_charge")
    if netq is None:
        netq = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    print(f"  ligand net charge {netq:+d} e, {mol.NumAtoms()} atoms with hydrogens")
    if not oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCCharges()):
        sys.exit("  ERROR: OEAssignCharges failed")
    ofs = oechem.oemolostream(os.path.join(W, "ligand_oe.mol2"))
    ofs.SetFormat(oechem.OEFormat_MOL2)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()
    _sh(W, f"antechamber -i ligand_oe.mol2 -fi mol2 -o ligand.mol2 -fo mol2 "
            f"-at gaff2 -nc {netq}", "ligand.mol2")
    _sh(W, "parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2",
        "ligand.frcmod")
    _leap(W, "tleap_ligand.in",
          "source leaprc.gaff2\nloadamberparams ligand.frcmod\n"
          "lig = loadmol2 ligand.mol2\ncheck lig\n"
          "saveamberparm lig ligand.prmtop ligand.rst7\n"
          "savepdb lig ligand_leap.pdb\nquit\n", "ligand.prmtop")
    _pqr(W, "protein", "receptor.pqr")
    _pqr(W, "ligand", "ligand.pqr")
    for f in JUNK:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)

def source_complex_pdb(cfg, W):
    """Split the PDB, parameterise both, ambpdb."""
    src = os.path.join(W, cfg["complex_pdb"])
    prot, lig = [], []
    for line in open(src):
        if line.startswith("ATOM"):
            a = line[12:16].strip()
            if a[0] == "H" or (len(a) > 1 and a[0].isdigit() and a[1] == "H"):
                continue
            prot.append(line)
        elif line.startswith("HETATM"):
            if line[17:20].strip() in SKIP_HET:
                continue
            lig.append(line)
    open(os.path.join(W, "protein.pdb"), "w").write("".join(prot) + "END\n")
    open(os.path.join(W, "ligand_raw.pdb"), "w").write("".join(lig) + "END\n")
    print(f"  protein {len(prot)} atoms, ligand {len(lig)} atoms")
    _parameterise(cfg, W)

def source_rcsb(cfg, W):
    """Fetch the entry, extract chain and ligand, parameterise.

    CONECT records for the ligand come across too, so OpenEye reads the bond
    orders from the deposit instead of guessing them from geometry.
    """
    pid = cfg["pdb_id"]
    entry = os.path.join(W, f"{pid}.pdb")
    if not os.path.exists(entry):
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pid}.pdb", entry)
        print(f"  downloaded {pid}.pdb")

    chain = cfg.get("chain")
    rmin = cfg.get("rec_resid_min")
    resid = cfg["ligand_resid"]
    prot, lig, serials = [], [], set()
    for line in open(entry):
        if line.startswith("ATOM") and len(line) > 21:
            if chain and line[21] != chain:
                continue
            if rmin is not None and int(line[22:26]) < rmin:
                continue
            prot.append(line)
        elif line.startswith("HETATM") and resid in line:
            lig.append(line)
            try:
                serials.add(int(line[6:11]))
            except ValueError:
                pass
    conect = [l for l in open(entry) if l.startswith("CONECT")
              and l[6:11].strip().isdigit() and int(l[6:11]) in serials]
    open(os.path.join(W, "protein.pdb"), "w").write("".join(prot) + "END\n")
    open(os.path.join(W, "ligand_raw.pdb"), "w").write(
        "".join(lig) + "".join(conect) + "END\n")
    print(f"  protein {len(prot)} atoms, ligand {len(lig)} atoms, "
          f"{len(conect)} CONECT")
    _parameterise(cfg, W)
    return {"rmap": resid_map(entry, chain, rmin),
            "lmap": atom_map(os.path.join(W, "ligand_raw.pdb"))}

def _parameterise_smiles(cfg, W):
    """tleap the protein, RDKit and antechamber the ligand, ambpdb both.

    Reads protein.pdb, writes receptor.pqr and ligand.pqr. The conformer is
    embedded from the config seed, so the geometry and therefore the charges
    are reproducible.
    """
    mol = Chem.MolFromSmiles(cfg["smiles"])
    if mol is None:
        sys.exit(f"  ERROR: RDKit cannot read the SMILES {cfg['smiles']!r}")
    mol = Chem.AddHs(mol)
    par = AllChem.ETKDG()
    par.randomSeed = int(cfg["seed"])
    if AllChem.EmbedMolecule(mol, par) != 0:
        sys.exit("  ERROR: RDKit could not embed the ligand")
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, os.path.join(W, "ligand_raw.sdf"))
    netq = cfg.get("ligand_net_charge")
    if netq is None:
        netq = Chem.GetFormalCharge(mol)
    print(f"  ligand net charge {netq:+d} e, {mol.GetNumAtoms()} atoms with hydrogens")

    _leap(W, "tleap_protein.in",
          "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
          "loadAmberParams frcmod.ions234lm_126_tip3p\n"
          "protein = loadpdb protein.pdb\ncheck protein\n"
          "saveamberparm protein protein.prmtop protein.rst7\n"
          "savepdb protein protein_leap.pdb\nquit\n", "protein.prmtop")
    _sh(W, "obabel ligand_raw.sdf -o mol2 -O ligand_obabel.mol2",
        "ligand_obabel.mol2")
    _sh(W, f"antechamber -i ligand_obabel.mol2 -fi mol2 -o ligand.mol2 -fo mol2 "
           f"-c bcc -at gaff2 -nc {netq} -rn {cfg['ligand_resid']} -pf y -j 4 "
           f"-dr no", "ligand.mol2")
    _sh(W, "parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2",
        "ligand.frcmod")
    _leap(W, "tleap_ligand.in",
          "source leaprc.gaff2\nloadamberparams ligand.frcmod\n"
          "lig = loadmol2 ligand.mol2\ncheck lig\n"
          "saveamberparm lig ligand.prmtop ligand.rst7\n"
          "savepdb lig ligand_leap.pdb\nquit\n", "ligand.prmtop")
    _pqr(W, "protein", "receptor.pqr")
    _pqr(W, "ligand", "ligand.pqr")
    for f in JUNK:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)

def source_rcsb_smiles(cfg, W):
    """Receptor from the entry, ligand from SMILES.

    keep_hetatm names the het groups that stay with the receptor, such as a
    catalytic metal. OXT is dropped because tleap will not load a terminal
    oxygen it did not add itself.
    """
    pid = cfg["pdb_id"]
    entry = os.path.join(W, f"{pid}.pdb")
    if not os.path.exists(entry):
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pid}.pdb", entry)
        print(f"  downloaded {pid}.pdb")
    chain = cfg.get("chain")
    keep = {_name(h) for h in (cfg.get("keep_hetatm") or ())}
    prot, het = [], []
    for line in open(entry):
        if len(line) < 22 or (chain and line[21] != chain):
            continue
        if line.startswith("ATOM"):
            if line[12:16].strip() != "OXT":
                prot.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() in keep:
            het.append(line)
    open(os.path.join(W, "protein.pdb"), "w").write(
        "".join(prot) + "".join(het) + "END\n")
    print(f"  protein {len(prot)} atoms, {len(het)} kept het atoms")
    _parameterise_smiles(cfg, W)

HANDLERS = {"prmtop": source_prmtop, "complex_pdb": source_complex_pdb,
            "rcsb": source_rcsb, "rcsb_smiles": source_rcsb_smiles}

# reaction criterion and output files
def read_pqr(path):
    at = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        t = line.split()
        while t and not _isnum(t[-1]):
            t.pop()
        at.append(dict(name=line[12:16].strip(), resname=line[17:20].strip(),
                       resid=int(line[22:26]),
                       xyz=np.array([float(v) for v in t[-5:-2]]),
                       q=float(t[-2]), r=float(t[-1])))
    return at

def _isnum(t):
    try:
        float(t)
        return True
    except ValueError:
        return False

def check_formula(lig, formula):
    want = collections.Counter()
    for e, n in re.findall(r"([A-Z][a-z]?)\s*(\d*)", formula):
        want[e.upper()] += int(n) if n else 1
    got = collections.Counter(
        re.match(r"([A-Za-z])", a["name"]).group(1).upper() for a in lig)
    if got != want:
        d = {k: got.get(k, 0) - want.get(k, 0)
             for k in set(got) | set(want) if got.get(k, 0) != want.get(k, 0)}
        sys.exit(f"  ERROR: ligand composition off by {d}\n"
                 f"  built {dict(got)}\n  want  {dict(want)}  ({formula})")
    print(f"  composition verified: {formula}")

def _name(x):
    return x[0] if isinstance(x, (list, tuple)) else x

def half(d, how):
    return math.ceil(d * 2) / 2 if how == "up" else round(d * 2) / 2

def keeper(kind):
    return {"all": lambda a, b: True,
            "polar": lambda a, b: a in POLAR and b in POLAR,
            "nonpolar": lambda a, b: a == "C" and b == "C",
            "any_polar": lambda a, b: a in POLAR or b in POLAR,
            "receptor_polar": lambda a, b: a in POLAR}[kind]

def pairs_auto(rec, lig, c):
    keep = keeper(c["atom_filter"])
    skip = {_name(x) for x in (c.get("excluded") or ())}
    found = []
    for i, a in enumerate(rec):
        if a["name"].startswith("H") or a["name"] in skip:
            continue
        for j, b in enumerate(lig):
            if b["name"].startswith("H") or b["name"] in skip:
                continue
            d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
            if d < c["search_cutoff"] and keep(a["name"][0], b["name"][0]):
                found.append((d, i, j))
    found.sort()
    out, seen = [], set()
    for d, i, j in found:
        if rec[i]["resid"] in seen:
            continue
        seen.add(rec[i]["resid"])
        cut = (half(d + c["buffer"], c.get("rounding", "nearest"))
               if "buffer" in c else c["pair_cutoff"])
        out.append((i, j, cut, d))
        if len(out) >= c["n_pairs"]:
            break
    return out

def resid_map(pdb_path, chain=None, resid_min=None):
    seen = []
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        if chain and line[21] != chain:
            continue
        r = int(line[22:26])
        if resid_min is not None and r < resid_min:
            continue
        if r not in seen:
            seen.append(r)
    return {r: i + 1 for i, r in enumerate(seen)}

def atom_map(pdb_path, record="HETATM"):
    out, pos = {}, 0
    for line in open(pdb_path):
        if line.startswith(record):
            pos += 1
            nm = line[12:16].strip()
            out.setdefault(nm, pos)
    return out

def pairs_explicit(rec, lig, c, rmap=None, lmap=None):
    R, L = c["receptor"], c["ligand"]
    if L == "auto":
        L = ["auto"] * len(R)
    if len(R) != len(L):
        sys.exit(f"  ERROR: {len(R)} receptor targets against {len(L)} ligand "
                 f"atoms, explicit mode pairs them one to one")
    if "cutoffs" in c:
        cuts = c["cutoffs"]
        if not isinstance(cuts, (list, tuple)):
            cuts = [cuts] * len(R)
        if len(cuts) != len(R):
            sys.exit(f"  ERROR: {len(cuts)} cutoffs for {len(R)} pairs")
    else:
        cuts = [None] * len(R)
    out = []
    pdb_num = c.get("receptor_numbering", "internal") == "pdb"
    for (resid, aname, resname), lname, cut in zip(R, L, cuts):
        nm = _name(lname)
        if pdb_num:
            if not rmap:
                sys.exit("  ERROR: receptor_numbering is pdb but no map given")
            if resid not in rmap:
                sys.exit(f"  ERROR: residue {resid} is not in the source PDB")
            resid = rmap[resid]
        i = next((k for k, a in enumerate(rec) if a["resid"] == resid
                  and a["name"] == aname and a["resname"] == resname), None)
        if i is None:
            sys.exit(f"  ERROR: no {resname}{resid} {aname} in the receptor")
        if nm == "auto":
            keep = keeper(c["atom_filter"])
            near = [(float(np.linalg.norm(rec[i]["xyz"] - b["xyz"])), k)
                    for k, b in enumerate(lig) if not b["name"].startswith("H")
                    and keep(rec[i]["name"][0], b["name"][0])]
            if not near:
                sys.exit(f"  ERROR: no {c['atom_filter']} ligand atom for "
                         f"{resname}{resid} {aname}")
            j = min(near)[1]
        elif lmap is not None:
            if nm not in lmap:
                sys.exit(f"  ERROR: no ligand atom {nm} in the source PDB")
            j = lmap[nm] - 1
        else:
            j = next((k for k, b in enumerate(lig) if b["name"] == nm), None)
            if j is None:
                sys.exit(f"  ERROR: no ligand atom {nm}")
        d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
        if cut is None:
            cut = half(d + c["buffer"], c.get("rounding", "nearest"))
        out.append((i, j, cut, d))
    return out

def write_rxns(path, pairs, n_needed):
    body = "".join(
        f"        <pair><atoms> {i + 1} {j + 1} </atoms>"
        f"<distance> {cut:.1f} </distance></pair>\n"
        for i, j, cut, _d in pairs)
    open(path, "w").write(
        '<?xml version="1.0" ?>\n<top>\n  <first_state> start </first_state>\n'
        '  <reactions>\n    <reaction>\n      <n> association </n>\n'
        '      <state_before> start </state_before>\n'
        '      <state_after> end </state_after>\n      <criterion>\n'
        '        <molecules>\n          <molecule0> rec rec </molecule0>\n'
        '          <molecule1> lig lig </molecule1>\n        </molecules>\n'
        f'        <n_needed> {n_needed} </n_needed>\n{body}'
        '      </criterion>\n    </reaction>\n  </reactions>\n</top>\n')

TOP_LEVEL = ({"ligand_source", "ligand_formula", "receptor_resname",
              "ligand_resname", "criterion", "work_dir"}
             | set(BD_PARAMS))

def _check(choice, table, given, what, ignore=frozenset()):
    if choice not in table:
        sys.exit(f"{what} must be one of {sorted(table)}, got {choice!r}\n"
                 + "\n".join(f"    {k:<14}{v['note']}, needs {', '.join(v['need'])}"
                              for k, v in table.items()))
    spec = table[choice]
    missing = [k for k in spec["need"]
               if given.get(k) is None or given.get(k) in ("", [], ())]
    if missing:
        sys.exit(f"{what} {choice!r} needs {missing}, which is blank or absent")
    extra = set(given) - set(spec["need"]) - set(spec["opt"]) - {"mode"} - ignore
    if extra:
        sys.exit(f"{what} {choice!r} does not use {sorted(extra)}, so remove it "
                 f"rather than leave it dead")

def build(**cfg):
    src = cfg.get("ligand_source")
    _check(src, SOURCES, cfg, "ligand_source", ignore=TOP_LEVEL)
    c = cfg.get("criterion", {})
    _check(c.get("mode"), MODES, c, "criterion mode")
    unknown = set(cfg) - TOP_LEVEL - set(SOURCES[src]["need"]) \
        - set(SOURCES[src]["opt"])
    if unknown:
        sys.exit(f"CONFIG does not use {sorted(unknown)}")
    if c["mode"] == "auto":
        if c.get("atom_filter") not in ATOM_FILTERS:
            sys.exit(f"atom_filter must be one of {sorted(ATOM_FILTERS)}")
        if ("pair_cutoff" in c) == ("buffer" in c):
            sys.exit("give exactly one of pair_cutoff or buffer")
    else:
        if ("cutoffs" in c) == ("buffer" in c):
            sys.exit("give exactly one of cutoffs or buffer")
        auto = c["ligand"] == "auto" or "auto" in c["ligand"]
        if auto and c.get("atom_filter") not in ATOM_FILTERS:
            sys.exit(f"ligand auto needs atom_filter, one of {sorted(ATOM_FILTERS)}")
        if not auto and "atom_filter" in c:
            sys.exit("atom_filter only applies with ligand auto, so remove it")
    if c.get("receptor_numbering", "internal") not in NUMBERING:
        sys.exit(f"receptor_numbering must be one of {sorted(NUMBERING)}")
    if c.get("rounding", "nearest") not in ROUNDINGS:
        sys.exit(f"rounding must be one of {sorted(ROUNDINGS)}")
    if "rounding" in c and "buffer" not in c:
        sys.exit("rounding only applies with buffer, so remove it")
    W = cfg.get("work_dir", HERE)
    extra = HANDLERS[src](cfg, W) or {}
    rec = read_pqr(os.path.join(W, "receptor.pqr"))
    lig = read_pqr(os.path.join(W, "ligand.pqr"))
    check_formula(lig, cfg["ligand_formula"])
    pairs = (pairs_auto(rec, lig, c) if c["mode"] == "auto"
             else pairs_explicit(rec, lig, c, extra.get("rmap"),
                                 extra.get("lmap")))
    if len(pairs) < c["n_needed"]:
        sys.exit(f"  ERROR: {len(pairs)} pairs found, n_needed is {c['n_needed']}")
    write_rxns(os.path.join(W, "rxns.xml"), pairs, c["n_needed"])
    write_input(os.path.join(W, "input.xml"), cfg)
    print(f"  {len(pairs)} pairs, n_needed {c['n_needed']}")
    geometry_check(rec, lig, cfg)
    tidy(W, cfg)

def find_templates():
    here = os.path.dirname(os.path.abspath(__file__))
    for i in range(10):
        base = os.path.join(here, *[".."] * i) if i else here
        for sub in ("pystarc", "PySTARC/pystarc"):
            t = os.path.join(base, sub, "templates")
            if os.path.isdir(t):
                return os.path.abspath(t)
    try:
        import pystarc
        return os.path.join(os.path.dirname(os.path.abspath(pystarc.__file__)),
                            "templates")
    except ImportError:
        sys.exit("  ERROR: pystarc templates not found")

def write_input(path, cfg):
    txt = open(os.path.join(find_templates(), "input.xml")).read()
    vals = {"receptor_resname": cfg["receptor_resname"],
            "ligand_resname": cfg["ligand_resname"],
            "receptor_pqr": "receptor.pqr", "ligand_pqr": "ligand.pqr",
            "rxns_xml": "rxns.xml", "work_dir": cfg.get("bd_work_dir", "bd_sims")}
    missing = [k for k in BD_PARAMS if k not in cfg]
    if missing:
        sys.exit(f"  ERROR: config.xml has no {', '.join(missing)}")
    vals.update({k: cfg[k] for k in BD_PARAMS})
    for k, v in vals.items():
        txt = re.sub(rf"<{k}>\s*</{k}>", f"<{k}>{v}</{k}>", txt)
    open(path, "w").write(txt)
    left = re.findall(r"<([a-z_]+)>\s*</\1>", txt)
    if left:
        sys.exit(f"  ERROR: input.xml still has empty tags {left}")

def geometry_check(rec, lig, cfg):
    R = np.array([a["xyz"] for a in rec])
    L = np.array([a["xyz"] for a in lig])
    rmax = float(np.max(np.linalg.norm(R - R.mean(0), axis=1)))
    lmax = float(np.max(np.linalg.norm(L - L.mean(0), axis=1)))
    b = float(cfg["bd_milestone_radius"])
    fg = float(cfg["apbs_fglen"])
    ok = b + lmax < fg / 2
    print(f"  rmax={rmax:.1f} lmax={lmax:.1f} b={b:.0f} fglen={fg:.0f}  "
          f"{'OK' if ok else 'WARNING, b-surface outside the grid'}")

STRUCTURE = ("ligand_source", "prmtop", "complex_pdb", "pdb_id",
             "ligand_resid", "smiles", "chain", "rec_resid_min",
             "ligand_net_charge", "ligand_ccd", "keep_hetatm")
NAMES = ("ligand_formula", "receptor_resname", "ligand_resname")
CRITERION = ("mode", "receptor_numbering", "search_cutoff", "n_pairs",
             "receptor", "ligand", "atom_filter", "excluded", "pair_cutoff",
             "cutoffs", "buffer", "rounding", "n_needed")

def _el(t, v, ind):
    if t == "receptor":
        rows = "".join(f'{ind}  <target resid="{r}" atom="{a}" resname="{n}"/>\n'
                       for r, a, n in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    if isinstance(v, (list, tuple)):
        rows = "".join(f'{ind}  <atom name="{_name(x)}"/>\n' for x in v) if t == "ligand" \
            else "".join(f"{ind}  <item>{_name(x)}</item>\n" for x in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    return f"{ind}<{t}>{v}</{t}>"

def _listed(names):
    n = list(names)
    return " and ".join(n) if len(n) < 3 else ", ".join(n[:-1]) + ", and " + n[-1]

def _doc(t):
    table = {"ligand_source": SOURCES, "mode": MODES,
             "atom_filter": ATOM_FILTERS, "rounding": ROUNDINGS,
             "receptor_numbering": NUMBERING}.get(t)
    if table:
        return f"{KEYS[t]} Options: {_listed(table)}."
    if t in BOOL:
        return f"{KEYS[t]} Options: {_listed(('true', 'false'))}."
    return KEYS[t]

def _section(order, src, ind, w):
    out = []
    for t in order:
        if t not in src:
            continue
        first, _, rest = _el(t, src[t], ind).partition("\n")
        if t in KEYS:
            first = f"{first:<{w}}<!-- {_doc(t)} -->"
        out.append(first + ("\n" + rest if rest else ""))
    return "\n".join(out)

def config_xml(cfg):
    c = cfg.get("criterion", {})
    secs = [(STRUCTURE, cfg), (NAMES, cfg), (CRITERION, c),
            (BD_PARAMS, cfg)]
    w = max((len(_el(t, src[t], "    ").partition("\n")[0])
             for order, src in secs for t in order if t in src), default=0) + 2
    return f"""<?xml version="1.0" ?>
<config>
  <structure>
{_section(STRUCTURE, cfg, "    ", w)}
  </structure>
  <names>
{_section(NAMES, cfg, "    ", w)}
  </names>
  <criterion>
{_section(CRITERION, c, "    ", w)}
  </criterion>
  <run>
{_section(BD_PARAMS, cfg, "    ", w)}
  </run>
</config>
"""

if __name__ == "__main__":
    if "--template" in sys.argv:
        blank = {t: "" for t in ("ligand_source",) + NAMES}
        blank["criterion"] = {"mode": "", "n_needed": ""}
        blank.update({t: "" for t in BD_PARAMS})
        sys.stdout.write(config_xml(blank))
    else:
        given = [x for x in sys.argv[1:] if not x.startswith("-")]
        path = os.path.abspath(given[0] if given else "config.xml")
        if not os.path.exists(path):
            sys.exit(f"  ERROR: no {path}")
        cfg = read_config(path)
        cfg.setdefault("work_dir", os.path.dirname(path))
        build(**cfg)