#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit import Chem
import urllib.request
import numpy as np
import collections
import subprocess
import shutil
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "pqr":         dict(need=("receptor_pqr_file", "ligand_pqr_file"), opt=(), note="Uses receptor and ligand PQR files that already exist"),
    "prmtop":      dict(need=("prmtop", "complex_pdb"), opt=(), note="Uses a topology and PDB that already exist"),
    "complex_pdb": dict(need=("complex_pdb",), opt=("chain", "ligand_net_charge", "ligand_ccd", "ligand_protonate", "ligand_deprotonate"), note="Takes one PDB holding the receptor and the ligand"),
    "rcsb":        dict(need=("pdb_id", "ligand_resid"), opt=("chain", "rec_resid_min", "ligand_net_charge", "ligand_ccd", "ligand_protonate", "ligand_deprotonate"), note="Downloads the entry and takes the ligand as deposited"),
    "rcsb_smiles": dict(need=("pdb_id", "ligand_resid", "smiles"), opt=("chain", "keep_hetatm", "ligand_net_charge"), note="Takes the receptor from the entry and builds the ligand from SMILES"),
}

MODES = {
    "auto":     dict(need=("search_cutoff", "atom_filter", "n_pairs", "pair_cutoff", "n_needed"), opt=("excluded",), note="Locates contacts automatically and keeps one per residue"),
    "explicit": dict(need=("receptor", "ligand", "pair_cutoff", "n_needed"), opt=("receptor_numbering", "atom_filter"), note="Pairs receptor atoms you name one to one with ligand atoms"),
    "center":   dict(need=("pair_cutoff",), opt=(), note="Pairs the centroid of the receptor with the centroid of the ligand"),
}

NUMBERING = {
    "pdb":      "Read as deposited",
    "internal": "Read as renumbered from 1",
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
    "receptor_pqr_file":   "Existing receptor PQR with charges and radii. Not named receptor.pqr, which the run remakes.",
    "ligand_pqr_file":     "Existing ligand PQR with charges and radii. Not named ligand.pqr, which the run remakes.",
    "prmtop":              "Existing AMBER topology.",
    "complex_pdb":         "Single PDB containing receptor and ligand.",
    "pdb_id":              "RCSB entry to download.",
    "ligand_resid":        "Residue name of the ligand within the RCSB entry.",
    "smiles":              "Ligand SMILES, used in place of the deposited coordinates.",
    "chain":               "Receptor chain retained. Blank retains all chains.",
    "rec_resid_min":       "Lowest receptor residue number retained.",
    "ligand_net_charge":   "Formal charge of the ligand. Blank takes it from the component definition or the SMILES. Required when the ligand has neither.",
    "ligand_ccd":          "Component supplying the bond orders and formal charges. Blank uses the residue name of the ligand when it is a component code.",
    "ligand_protonate":    "Component atoms that each take one proton on top of the component definition. Each adds one to the net charge.",
    "ligand_deprotonate":  "Component atoms that each give up one proton of the component definition. Each takes one from the net charge.",
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
    "pair_cutoff":         "Reaction distance for every pair (in angstroms).",
    "n_needed":            "Pairs that must be inside their cutoff at once for a trajectory to count as reacted. Not used by mode center.",
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
    "soft_repulsion":      "Soft excluded volume of BrownDye 2 between receptor and ligand atoms in place of the overlap check.",
    "gpu_force_batch":     "Trajectories per GPU force evaluation. 0 selects automatically.",
    "max_steps":           "Step limit per trajectory.",
    "dt":                  "Largest permitted time step (in picoseconds).",
    "minimum_core_dt":     "Smallest permitted time step. 0 imposes no floor.",
    "minimum_core_reaction_dt": "Smallest time step near the reaction surface. 0 imposes no floor.",
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
    "full_paths":          "Save positions and orientations every save_interval steps.",
    "energetics":          "Save forces and energies every save_interval steps.",
}

# Run parameters written into input.xml grouped by section of config.xml
GEOMETRY = ("bd_milestone_radius", "r_hydro_rec", "r_hydro_lig")
ELECTROSTATICS = (
    "debye_length", "ion_concentration", "ion_radius_pos", "ion_radius_neg",
    "pdie", "sdie", "srad", "apbs_fglen", "apbs_dime", "apbs_cglen",
    "apbs_coarse_dime", "apbs_fine_dime",
)
PHYSICS = (
    "soft_repulsion", "overlap_check", "lj_forces", "hydrodynamic_interactions",
    "desolvation_alpha", "enable_born2_torque", "multipole_fallback",
)
SIMULATION = (
    "n_trajectories", "max_steps", "dt", "minimum_core_dt",
    "minimum_core_reaction_dt", "max_dt", "temperature", "seed",
    "convergence_check", "convergence_interval", "convergence_tol",
    "checkpoint_interval",
)
COMPUTE = ("gpu", "gpu_force_batch", "n_threads")
OUTPUTS = ("save_interval", "full_paths", "energetics")
BD_PARAMS = GEOMETRY + ELECTROSTATICS + PHYSICS + SIMULATION + COMPUTE + OUTPUTS

# Settings a config.xml may leave out and the value input.xml then gets.
# These are the defaults of pystarc/pipeline/input_parser.py.
OPTIONAL = {"soft_repulsion": "false", "minimum_core_reaction_dt": "0",
            "full_paths": "true", "energetics": "true"}

BOOL = ("enable_born2_torque", "hydrodynamic_interactions", "multipole_fallback",
        "lj_forces", "overlap_check", "soft_repulsion", "gpu", "convergence_check",
        "full_paths", "energetics")

POLAR = {"N", "O", "S"}

# structure to PQR 
SKIP_HET = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "ACE", "NME",
            "Na+", "Cl-", "K+", "CAL"}
JUNK = ["_protein.rst7", "_ligand.rst7", "ligand_h.sdf", "sqm.in", "sqm.out",
        "sqm.pdb", "ANTECHAMBER_AC.AC", "ANTECHAMBER_AC.AC0",
        "ANTECHAMBER_BOND_TYPE.AC", "ANTECHAMBER_BOND_TYPE.AC0",
        "ANTECHAMBER_AM1BCC.AC", "ANTECHAMBER_AM1BCC_PRE.AC", "ATOMTYPE.INF",
        "ligand_raw.sdf", "ligand_obabel.mol2"]

LEFTOVER = ["tleap_protein.in", "tleap_ligand.in", "tleap_protein.log",
            "tleap_ligand.log", "leap.log", "ligand.frcmod", "ligand.mol2",
            "ligand_raw.pdb", "protein.pdb", "protein_leap.pdb",
            "ligand_leap.pdb", "protein.prmtop", "protein.rst7",
            "ligand.prmtop", "ligand.rst7"]

def tidy(W, cfg, cif=None):
    gone = list(LEFTOVER)
    code = cif or cfg.get("ligand_ccd") or cfg.get("ligand_resid")
    if code:
        gone.append(f"{code}.cif")
    for f in gone:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)

# Config values read as numbers. Every other value stays the text of the
# config, so a PDB id, a component code or a chain stays as written.
NUMBER = {"rec_resid_min": int, "ligand_net_charge": int, "search_cutoff": float,
          "n_pairs": int, "pair_cutoff": float, "n_needed": int}

def _number(tag, text, kind=None):
    try:
        return (kind or NUMBER[tag])(text)
    except ValueError:
        sys.exit(f"  ERROR: {tag} must be a number, got {text!r}")

def _value(el):
    kids = list(el)
    if not kids:
        text = (el.text or "").strip()
        return _number(el.tag, text) if text and el.tag in NUMBER else text
    out = []
    for c in kids:
        a = {k: v.strip() for k, v in c.attrib.items()}
        if {"resid", "atom", "resname"} <= set(a):
            out.append((_number("resid", a["resid"], int), a["atom"], a["resname"]))
        elif a:
            out.append(tuple(a.values()))
        else:
            out.append((c.text or "").strip())
    return out

def _list(v):
    """A list key as a list. A key written as plain text holds one entry."""
    if v is None or v == "":
        return []
    return [v] if isinstance(v, (str, tuple)) else list(v)

def read_config(path):
    cfg = {}
    for sec in ET.parse(path).getroot():
        vals = {el.tag: v for el in sec if (v := _value(el)) != ""}
        if sec.tag == "criterion":
            cfg["criterion"] = vals
        else:
            cfg.update(vals)
    return cfg

def _gone(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

def _sh(W, cmd, produces):
    # The output is removed first, so a file left by an earlier run never
    # passes for the output of this one.
    p = os.path.join(W, produces)
    _gone(p)
    r = subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(p):
        made = "" if os.path.exists(p) else f" and made no {produces}"
        tail = (r.stderr or r.stdout).strip()[-600:]
        sys.exit(f"  ERROR: {cmd.split()[0]} ended with exit {r.returncode}{made}\n{tail}")
    return r

def _leap(W, script, body, produces):
    stem = os.path.join(W, produces.rsplit(".", 1)[0])
    _gone(stem + ".prmtop", stem + ".rst7")
    open(os.path.join(W, script), "w").write(body)
    r = subprocess.run(f"tleap -f {script}", shell=True, cwd=W,
                       capture_output=True, text=True)
    open(os.path.join(W, script.replace(".in", ".log")), "w").write(r.stdout + r.stderr)
    p = os.path.join(W, produces)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        sys.exit(f"  ERROR: tleap made no {produces}, see {script[:-3]}.log")

def _pqr(W, stem, out):
    """cpptraj then ambpdb, the same two calls the per system scripts used."""
    rst = os.path.join(W, f"_{stem}.rst7")
    _gone(rst)
    subprocess.run("cpptraj", input=f"parm {stem}.prmtop\ntrajin {stem}.rst7\n"
                   f"trajout _{stem}.rst7\nrun\n", cwd=W,
                   capture_output=True, text=True)
    if not os.path.exists(rst):
        sys.exit(f"  ERROR: cpptraj made no _{stem}.rst7 from {stem}.prmtop")
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
    full = os.path.join(W, "_full.rst7")
    _gone(full)
    subprocess.run("cpptraj", input=f"parm {top}\ntrajin {pdb}\n"
                   f"trajout _full.rst7\nrun\n", cwd=W,
                   capture_output=True, text=True)
    if not os.path.exists(full):
        sys.exit(f"  ERROR: cpptraj could not read {pdb} with {top}")
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

def _download(url, path, what):
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        _gone(path)
        sys.exit(f"  ERROR: could not fetch {what} from {url} ({e})")

def _cif_tokens(path):
    """Tokens of a CIF file. A quoted value or a semicolon text field is one value."""
    toks, lines, i = [], open(path).read().split("\n"), 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith(";"):
            buf = [line[1:]]
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            toks.append(("value", "\n".join(buf).strip()))
            continue
        j, n = 0, len(line)
        while j < n:
            ch = line[j]
            if ch.isspace():
                j += 1
            elif ch == "#":
                break
            elif ch in "'\"":
                # A quote ends a value only when a blank or the line end follows
                k = j + 1
                while k < n and not (line[k] == ch and (k + 1 == n or line[k + 1].isspace())):
                    k += 1
                toks.append(("value", line[j + 1:k]))
                j = k + 1
            else:
                k = j
                while k < n and not line[k].isspace():
                    k += 1
                w = line[j:k]
                low = w.lower()
                kind = ("tag" if w.startswith("_") else "loop" if low == "loop_"
                        else "data" if low.startswith("data_") else "value")
                toks.append((kind, w))
                j = k
    return toks

def _cif_rows(path, category):
    """Rows of one CIF category as dicts, written as a loop or as single values."""
    toks, rows, single, i = _cif_tokens(path), [], {}, 0
    while i < len(toks):
        kind, w = toks[i]
        if kind == "loop":
            i += 1
            tags = []
            while i < len(toks) and toks[i][0] == "tag":
                tags.append(toks[i][1])
                i += 1
            vals = []
            while i < len(toks) and toks[i][0] == "value":
                vals.append(toks[i][1])
                i += 1
            if tags and tags[0].split(".")[0] == category:
                cols = [x.split(".", 1)[1] for x in tags]
                rows += [dict(zip(cols, vals[r:r + len(cols)]))
                         for r in range(0, len(vals) - len(cols) + 1, len(cols))]
        elif kind == "tag" and i + 1 < len(toks) and toks[i + 1][0] == "value":
            if w.split(".")[0] == category:
                single[w.split(".", 1)[1]] = toks[i + 1][1]
            i += 2
        else:
            i += 1
    return rows + ([single] if single else [])

ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}

def ccd_definition(code, W):
    """Bonds with their orders, formal charges, elements and leaving atoms of
    the deposited chemical component, by atom name."""
    cif = os.path.join(W, f"{code}.cif")
    if not os.path.exists(cif):
        _download(f"https://files.rcsb.org/ligands/download/{code}.cif", cif,
                  f"the chemical component {code}. Name the component in ligand_ccd, "
                  f"or call the ligand residue one of {sorted(GENERIC_RESNAMES)} "
                  f"and set ligand_net_charge")
    bonds, charges, elements, leaving = {}, {}, {}, set()
    for row in _cif_rows(cif, "_chem_comp_atom"):
        nm = row.get("atom_id")
        if not nm or row.get("comp_id", code) != code:
            continue
        q = row.get("charge", "0")
        charges[nm] = int(q) if q.lstrip("+-").isdigit() else 0
        elements[nm] = row.get("type_symbol", "").capitalize()
        if row.get("pdbx_leaving_atom_flag") == "Y":
            leaving.add(nm)
    for row in _cif_rows(cif, "_chem_comp_bond"):
        if row.get("comp_id", code) != code:
            continue
        o = ORDERS.get(row.get("value_order", "").upper())
        if o is None:
            sys.exit(f"  ERROR: bond {row.get('atom_id_1')} {row.get('atom_id_2')} of "
                     f"{code} has the order {row.get('value_order')!r}, which setup.py "
                     f"does not read")
        bonds[frozenset((row["atom_id_1"], row["atom_id_2"]))] = o
    if not elements:
        sys.exit(f"  ERROR: {code}.cif holds no atoms of {code}")
    return bonds, charges, elements, leaving

def _formula_counts(formula):
    c = collections.Counter()
    for e, n in re.findall(r"([A-Z][a-z]?)\s*(\d*)", formula):
        c[e.upper()] += int(n) if n else 1
    return c

def _mol_counts(mol):
    c = collections.Counter()
    for a in mol.GetAtoms():
        c[a.GetSymbol().upper()] += 1
    return c

# Residue names that stand for an unnamed ligand and so name no component
GENERIC_RESNAMES = {"UNK", "UNL", "LIG", "DRG", "INH", "MOL"}

def _restore_names(W, mol):
    """antechamber numbers the atoms afresh. Give the atoms that came from the
    structure their names back. An added hydrogen keeps its antechamber name
    unless the name is taken, then it gets the next free name of its element."""
    path = os.path.join(W, "ligand.mol2")
    lines = open(path).read().split("\n")
    i0 = lines.index("@<TRIPOS>ATOM") + 1
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith("@<TRIPOS>"))
    rows = [i for i in range(i0, i1) if lines[i].strip()]
    if len(rows) != mol.GetNumAtoms():
        sys.exit(f"  ERROR: ligand.mol2 has {len(rows)} atoms, the ligand has "
                 f"{mol.GetNumAtoms()}")
    given = {}
    for i, a in zip(rows, mol.GetAtoms()):
        f = lines[i].split()
        el = {"cl": "Cl", "br": "Br"}.get(f[5][:2].lower(), f[5][0].upper())
        if el != a.GetSymbol():
            sys.exit(f"  ERROR: atom {f[0]} of ligand.mol2 is {el} where the ligand "
                     f"has {a.GetSymbol()}, so the atom order changed")
        info = a.GetPDBResidueInfo()
        if info is not None:
            given[i] = info.GetName().strip()
    taken = set(given.values())
    if len(taken) != len(given):
        sys.exit("  ERROR: two ligand atoms of the structure share a name")
    for i, a in zip(rows, mol.GetAtoms()):
        f = lines[i].split()
        nm = given.get(i)
        if nm is None:
            nm, k = f[1], 1
            while nm in taken:
                nm, k = f"{a.GetSymbol().upper()}{k}", k + 1
            taken.add(nm)
        f[1] = nm
        lines[i] = (f"{f[0]:>7} {f[1]:<8} {f[2]:>10} {f[3]:>10} {f[4]:>10} "
                    f"{f[5]:<6} {f[6]:>3} {f[7]:<8} {f[8]:>10}")
    open(path, "w").write("\n".join(lines))

def _parameterise(cfg, W):
    _leap(W, "tleap_protein.in",
          "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
          "protein = loadpdb protein.pdb\ncheck protein\n"
          "saveamberparm protein protein.prmtop protein.rst7\n"
          "savepdb protein protein_leap.pdb\nquit\n", "protein.prmtop")
    raw = os.path.join(W, "ligand_raw.pdb")
    resn_pdb = next((l[17:20].strip() for l in open(raw)
                     if l.startswith(("ATOM", "HETATM"))), "")
    # The component definition gives the bonds, their orders and the formal
    # charges. Its code is ligand_ccd, then ligand_resid, then the residue
    # name of the ligand when the name is not a placeholder.
    code = cfg.get("ligand_ccd") or cfg.get("ligand_resid") or (
        resn_pdb if resn_pdb and resn_pdb not in GENERIC_RESNAMES else None)
    m = Chem.MolFromPDBFile(raw, removeHs=False, sanitize=False)
    if m is None:
        sys.exit("  ERROR: RDKit cannot read ligand_raw.pdb")
    has_h = any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
    netq = cfg.get("ligand_net_charge")
    prot = [_name(x) for x in _list(cfg.get("ligand_protonate"))]
    deprot = [_name(x) for x in _list(cfg.get("ligand_deprotonate"))]
    if set(prot) & set(deprot):
        sys.exit(f"  ERROR: {sorted(set(prot) & set(deprot))} named in both "
                 f"ligand_protonate and ligand_deprotonate")
    if code:
        bonds, charges, elements, leaving = ccd_definition(code, W)
        name = lambda a: a.GetPDBResidueInfo().GetName().strip()
        rw = Chem.RWMol(m)
        for i, j in [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rw.GetBonds()]:
            rw.RemoveBond(i, j)
        # Hydrogens of the structure that the component does not name are
        # dropped. AddHs places the hydrogens the definition asks for.
        drop = [a.GetIdx() for a in rw.GetAtoms()
                if a.GetAtomicNum() == 1 and name(a) not in elements]
        for k in sorted(drop, reverse=True):
            rw.RemoveAtom(k)
        if drop:
            print(f"  {len(drop)} hydrogens of the structure are not named in "
                  f"{code} and are placed again")
        unknown = sorted({name(a) for a in rw.GetAtoms()} - set(elements))
        if unknown:
            sys.exit(f"  ERROR: atoms {unknown} are not in the {code} definition, "
                     f"so the ligand atom names do not follow the component")
        wrong = [f"{name(a)} is {a.GetSymbol()} here and {elements[name(a)]} in {code}"
                 for a in rw.GetAtoms() if elements[name(a)] != a.GetSymbol()]
        if wrong:
            sys.exit(f"  ERROR: {'; '.join(wrong)}, so {code} does not describe this ligand")
        heavy = {name(a) for a in rw.GetAtoms() if a.GetAtomicNum() > 1}
        absent = sorted(nm for nm, e in elements.items()
                        if e != "H" and nm not in heavy and nm not in leaving)
        if absent:
            sys.exit(f"  ERROR: the structure lacks the {code} atoms {absent}, "
                     f"so the ligand is not complete")
        idx = {name(a): a.GetIdx() for a in rw.GetAtoms()}
        if len(idx) != rw.GetNumAtoms():
            sys.exit("  ERROR: two ligand atoms of the structure share a name")
        kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE}
        n = 0
        for pair, o in bonds.items():
            a1, a2 = tuple(pair)
            if a1 in idx and a2 in idx:
                rw.AddBond(idx[a1], idx[a2], kinds[o])
                n += 1
        for a in rw.GetAtoms():
            a.SetFormalCharge(charges.get(name(a), 0))
        # A proton named in ligand_protonate raises the formal charge of its
        # atom by one and one named in ligand_deprotonate lowers it by one.
        # AddHs then places the hydrogens from the geometry. A hydrogen the
        # structure holds on a deprotonated atom is removed.
        gone = []
        for key, names, step in (("ligand_protonate", prot, 1),
                                 ("ligand_deprotonate", deprot, -1)):
            for nm in names:
                if nm not in idx:
                    sys.exit(f"  ERROR: {key} names {nm}, which is not a ligand atom")
                a = rw.GetAtomWithIdx(idx[nm])
                if a.GetSymbol() not in ("N", "O", "S"):
                    sys.exit(f"  ERROR: {key} names {nm}, a {a.GetSymbol()} atom. "
                             f"Protons move on N, O or S only")
                a.SetFormalCharge(a.GetFormalCharge() + step)
                hs = [b.GetIdx() for b in a.GetNeighbors() if b.GetAtomicNum() == 1]
                if step < 0 and hs:
                    gone.append(hs[-1])
        for k in sorted(gone, reverse=True):
            rw.RemoveAtom(k)
        m = rw.GetMol()
        q_ccd = sum(a.GetFormalCharge() for a in m.GetAtoms())
        print(f"  {n} bonds with their orders and the formal charges from the "
              f"{code} definition"
              + (f", one proton added on each of {', '.join(prot)}" if prot else "")
              + (f", one proton removed from each of {', '.join(deprot)}" if deprot else "")
              + f", net charge {q_ccd:+d} e")
        if netq is None:
            netq = q_ccd
        elif netq != q_ccd:
            sys.exit(f"  ERROR: ligand_net_charge {netq:+d} differs from the "
                     f"{q_ccd:+d} of the {code} definition with the named protons")
    else:
        if prot or deprot:
            sys.exit("  ERROR: ligand_protonate and ligand_deprotonate need a component "
                     "definition, name the component in ligand_ccd")
        if netq is None:
            sys.exit(f"  ERROR: no component definition for {resn_pdb or 'the ligand'}, "
                     f"so set ligand_net_charge, or name the component in ligand_ccd")
        if not has_h:
            # RDKit reads bond orders from a geometry only with every hydrogen present
            sys.exit(f"  ERROR: the ligand {resn_pdb} has no hydrogens and no component "
                     f"definition, so its bond orders cannot be read from the structure. "
                     f"Name the component in ligand_ccd, or give the ligand its "
                     f"hydrogens in the PDB")
        try:
            rdDetermineBonds.DetermineBondOrders(m, charge=netq)
        except Exception as e:
            sys.exit(f"  ERROR: RDKit finds no bond orders for net charge {netq:+d} "
                     f"({e}). Check ligand_net_charge and the hydrogens of the ligand")
        print("  bond orders from the geometry, with the hydrogens of the PDB")
    for a in m.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(m)
    except Exception as e:
        sys.exit(f"  ERROR: the ligand chemistry does not sanitize ({e}). "
                 f"Check ligand_protonate and the component")
    m = Chem.AddHs(m, addCoords=True)
    want, got = _formula_counts(cfg["ligand_formula"]), _mol_counts(m)
    if got != want:
        off = {k: got.get(k, 0) - want.get(k, 0)
               for k in sorted(set(got) | set(want)) if got.get(k, 0) != want.get(k, 0)}
        sys.exit(f"  ERROR: ligand composition off by {off} against "
                 f"{cfg['ligand_formula']}. Check ligand_net_charge and the component")
    print(f"  ligand net charge {netq:+d} e, {m.GetNumAtoms()} atoms with hydrogens")
    Chem.MolToMolFile(m, os.path.join(W, "ligand_h.sdf"))
    _sh(W, "obabel ligand_h.sdf -o mol2 -O ligand_obabel.mol2", "ligand_obabel.mol2")
    resn = cfg.get("ligand_resid") or cfg["ligand_resname"]
    _sh(W, f"antechamber -i ligand_obabel.mol2 -fi mol2 -o ligand.mol2 -fo mol2 "
           f"-c bcc -at gaff2 -nc {netq} -rn {resn} -pf y -j 4 -dr no",
        "ligand.mol2")
    _restore_names(W, m)
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
    return code

def _one_conformer(lines):
    """Lines without an altloc, and of each residue the first altloc it lists."""
    first, out = {}, []
    for l in lines:
        key = (l[21], l[22:27], l[17:20])
        if l[16] != " ":
            first.setdefault(key, l[16])
        if l[16] in (" ", first.get(key)):
            out.append(l)
    return out

def _residues(lines):
    """Lines grouped by residue (chain, number with insertion code, name), in file order."""
    groups = {}
    for l in lines:
        groups.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), []).append(l)
    return groups

def _label(key):
    return f"{key[2]} {key[0].strip()}{key[1]}"

def source_complex_pdb(cfg, W):
    """Split the PDB, parameterise both, ambpdb."""
    src = os.path.join(W, cfg["complex_pdb"])
    prot, het = [], []
    for line in open(src):
        if line.startswith("ATOM"):
            a = line[12:16].strip()
            if a[0] == "H" or (len(a) > 1 and a[0].isdigit() and a[1] == "H"):
                continue
            prot.append(line)
        elif line.startswith("HETATM"):
            if line[17:20].strip() in SKIP_HET:
                continue
            het.append(line)
    groups = _residues(_one_conformer(het))
    if not groups:
        sys.exit(f"  ERROR: no ligand HETATM in {cfg['complex_pdb']}")
    key = next(iter(groups))
    if len(groups) > 1:
        # Several het groups. The ligand is the one named in the config.
        want = {cfg.get("ligand_ccd"), cfg.get("ligand_resname")} - {None, ""}
        pick = [k for k in groups if k[2] in want]
        if len(pick) != 1:
            sys.exit(f"  ERROR: {cfg['complex_pdb']} holds the het groups "
                     f"{', '.join(_label(k) for k in groups)}. Name the ligand residue "
                     f"in ligand_resname and keep one copy, or remove the others")
        key = pick[0]
        print(f"  ligand {_label(key)}, left out "
              f"{', '.join(_label(k) for k in groups if k != key)}")
    lig = groups[key]
    open(os.path.join(W, "protein.pdb"), "w").write("".join(prot) + "END\n")
    open(os.path.join(W, "ligand_raw.pdb"), "w").write("".join(lig) + "END\n")
    print(f"  protein {len(prot)} atoms, ligand {len(lig)} atoms")
    return {"cif": _parameterise(cfg, W)}

def source_rcsb(cfg, W):
    """Fetch the entry, extract chain and ligand, parameterise.

    CONECT records for the ligand come across too and the bond orders come
    from the component definition when the composition check asks for them.
    """
    pid = cfg["pdb_id"]
    entry = os.path.join(W, f"{pid}.pdb")
    if not os.path.exists(entry):
        _download(f"https://files.rcsb.org/download/{pid}.pdb", entry, f"the entry {pid}")
        print(f"  downloaded {pid}.pdb")

    chain = cfg.get("chain")
    rmin = cfg.get("rec_resid_min")
    resid = cfg["ligand_resid"]
    prot, het = [], []
    for line in open(entry):
        if line.startswith("ATOM") and len(line) > 21:
            if chain and line[21] != chain:
                continue
            if rmin is not None and int(line[22:26]) < rmin:
                continue
            prot.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() == resid:
            het.append(line)
    groups = _residues(_one_conformer(het))
    if not groups:
        sys.exit(f"  ERROR: no {resid} in {pid}")
    # The copy in the receptor chain, else the first copy of the entry
    key = next((k for k in groups if chain and k[0] == chain), next(iter(groups)))
    if len(groups) > 1:
        print(f"  ligand {_label(key)}, other copies left out: "
              f"{', '.join(_label(k) for k in groups if k != key)}")
    lig = groups[key]
    serials = {int(l[6:11]) for l in lig if l[6:11].strip().isdigit()}
    conect = [l for l in open(entry) if l.startswith("CONECT")
              and l[6:11].strip().isdigit() and int(l[6:11]) in serials]
    open(os.path.join(W, "protein.pdb"), "w").write("".join(prot) + "END\n")
    open(os.path.join(W, "ligand_raw.pdb"), "w").write(
        "".join(lig) + "".join(conect) + "END\n")
    print(f"  protein {len(prot)} atoms, ligand {len(lig)} atoms, "
          f"{len(conect)} CONECT")
    code = _parameterise(cfg, W)
    return {"rmap": resid_map(entry, chain, rmin), "cif": code}

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
        _download(f"https://files.rcsb.org/download/{pid}.pdb", entry, f"the entry {pid}")
        print(f"  downloaded {pid}.pdb")
    chain = cfg.get("chain")
    keep = {_name(h) for h in _list(cfg.get("keep_hetatm"))}
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

def source_pqr(cfg, W):
    """Receptor and ligand PQR files that already exist, charges and radii included."""
    for key, out in (("receptor_pqr_file", "receptor.pqr"),
                     ("ligand_pqr_file", "ligand.pqr")):
        src = os.path.join(W, cfg[key])
        if not os.path.exists(src):
            sys.exit(f"  ERROR: no {cfg[key]}")
        dst = os.path.join(W, out)
        if os.path.abspath(src) == os.path.abspath(dst):
            sys.exit(f"  ERROR: {key} must not be {out}, which the run remakes")
        # The run places its own GHO ghost atom last, so one left in the file
        # by an earlier run is taken out here.
        lines = open(src).read().split("\n")
        ghost = lambda l: l.startswith(("ATOM", "HETATM")) and l[12:16].strip().upper() == "GHO"
        n_gho = sum(1 for l in lines if ghost(l))
        open(dst, "w").write("\n".join(l for l in lines if not ghost(l)))
        n = sum(1 for l in open(dst) if l.startswith(("ATOM", "HETATM")))
        print(f"  {out}: {n} atoms from {cfg[key]}"
              + (f", {n_gho} GHO ghost atom left out" if n_gho else ""))

HANDLERS = {"pqr": source_pqr, "prmtop": source_prmtop,
            "complex_pdb": source_complex_pdb,
            "rcsb": source_rcsb, "rcsb_smiles": source_rcsb_smiles}

# reaction criterion and output files
def _element(name, column=""):
    """Element of a PQR atom, upper case. The element column when the file has
    one, else the name, with chlorine and bromine told by two letters."""
    if column.isalpha() and len(column) <= 2:
        return column.upper()
    nm = name.lstrip("0123456789")
    return nm[:2].upper() if nm[:2].upper() in ("CL", "BR") else nm[:1].upper()

def read_pqr(path):
    at = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        t = line.split()
        tail = []
        while t and not _isnum(t[-1]):
            tail.append(t.pop())
        name = line[12:16].strip()
        at.append(dict(name=name, el=_element(name, tail[0] if tail else ""),
                       resname=line[17:20].strip(), resid=int(line[22:26]),
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
    got = collections.Counter(a["el"] for a in lig)
    if got != want:
        d = {k: got.get(k, 0) - want.get(k, 0)
             for k in set(got) | set(want) if got.get(k, 0) != want.get(k, 0)}
        sys.exit(f"  ERROR: ligand composition off by {d}\n"
                 f"  built {dict(got)}\n  want  {dict(want)}  ({formula})")
    print(f"  composition verified: {formula}")

def _name(x):
    return x[0] if isinstance(x, (list, tuple)) else x

def keeper(kind):
    return {"all": lambda a, b: True,
            "polar": lambda a, b: a in POLAR and b in POLAR,
            "nonpolar": lambda a, b: a == "C" and b == "C",
            "any_polar": lambda a, b: a in POLAR or b in POLAR,
            "receptor_polar": lambda a, b: a in POLAR}[kind]

def pairs_auto(rec, lig, c):
    keep = keeper(c["atom_filter"])
    skip = {_name(x) for x in _list(c.get("excluded"))}
    found = []
    for i, a in enumerate(rec):
        if a["el"] == "H" or a["name"] in skip:
            continue
        for j, b in enumerate(lig):
            if b["el"] == "H" or b["name"] in skip:
                continue
            d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
            if d < c["search_cutoff"] and keep(a["el"], b["el"]):
                found.append((d, i, j))
    found.sort()
    out, seen = [], set()
    for d, i, j in found:
        if rec[i]["resid"] in seen:
            continue
        seen.add(rec[i]["resid"])
        out.append((i, j, c["pair_cutoff"], d))
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

def pairs_explicit(rec, lig, c, rmap=None):
    R, L = _list(c["receptor"]), c["ligand"]
    if any(not (isinstance(r, tuple) and len(r) == 3) for r in R):
        sys.exit('  ERROR: receptor needs <target resid="..." atom="..." '
                 'resname="..."/> entries')
    L = ["auto"] * len(R) if L == "auto" else _list(L)
    if len(R) != len(L):
        sys.exit(f"  ERROR: {len(R)} receptor targets against {len(L)} ligand "
                 f"atoms, explicit mode pairs them one to one")
    out = []
    pdb_num = c.get("receptor_numbering", "internal") == "pdb"
    for (resid, aname, resname), lname in zip(R, L):
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
                    for k, b in enumerate(lig) if b["el"] != "H"
                    and keep(rec[i]["el"], b["el"])]
            if not near:
                sys.exit(f"  ERROR: no {c['atom_filter']} ligand atom for "
                         f"{resname}{resid} {aname}")
            j = min(near)[1]
        else:
            # ligand.pqr carries the names of the structure
            js = [k for k, b in enumerate(lig) if b["name"] == nm]
            if len(js) != 1:
                sys.exit(f"  ERROR: {'no' if not js else 'more than one'} ligand atom "
                         f"{nm} in ligand.pqr")
            j = js[0]
        d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
        out.append((i, j, c["pair_cutoff"], d))
    return out

def pair_center(rec, lig, c):
    """One pair between the centroids. PySTARC appends a ghost atom at the
    centroid of each molecule as its last atom. The pair names the position
    after the last real atom on both sides."""
    R = np.mean([a["xyz"] for a in rec], axis=0)
    L = np.mean([a["xyz"] for a in lig], axis=0)
    return (len(rec), len(lig), c["pair_cutoff"], float(np.linalg.norm(R - L)))

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
    elif c["mode"] == "explicit":
        L = c["ligand"]
        auto = L == "auto" or any(_name(x) == "auto" for x in _list(L))
        if auto and c.get("atom_filter") not in ATOM_FILTERS:
            sys.exit(f"ligand auto needs atom_filter, one of {sorted(ATOM_FILTERS)}")
        if not auto and "atom_filter" in c:
            sys.exit("atom_filter only applies with ligand auto, so remove it")
    if c.get("receptor_numbering", "internal") not in NUMBERING:
        sys.exit(f"receptor_numbering must be one of {sorted(NUMBERING)}")
    if not isinstance(c["pair_cutoff"], (int, float)) or c["pair_cutoff"] <= 0:
        sys.exit("pair_cutoff must be a distance in angstroms above zero")
    if c["mode"] != "center" and c["n_needed"] < 1:
        sys.exit("n_needed must be at least 1")
    if c["mode"] == "auto":
        if c["search_cutoff"] <= 0 or c["n_pairs"] < 1:
            sys.exit("search_cutoff must be above zero and n_pairs at least 1")
        if c["n_needed"] > c["n_pairs"]:
            sys.exit(f"n_needed {c['n_needed']} is more than n_pairs {c['n_pairs']}")
    if c["mode"] == "explicit" and c["n_needed"] > len(_list(c["receptor"])):
        sys.exit(f"n_needed {c['n_needed']} is more than the "
                 f"{len(_list(c['receptor']))} receptor targets")
    missing = [k for k in NAMES if not cfg.get(k)]
    if missing:
        sys.exit(f"config.xml needs {missing} in the names section")
    W = cfg.get("work_dir", HERE)
    extra = HANDLERS[src](cfg, W) or {}
    rec = read_pqr(os.path.join(W, "receptor.pqr"))
    lig = read_pqr(os.path.join(W, "ligand.pqr"))
    check_formula(lig, cfg["ligand_formula"])
    if c["mode"] == "auto":
        pairs = pairs_auto(rec, lig, c)
    elif c["mode"] == "explicit":
        pairs = pairs_explicit(rec, lig, c, extra.get("rmap"))
    else:
        pairs = [pair_center(rec, lig, c)]
    n_needed = 1 if c["mode"] == "center" else c["n_needed"]
    if len(pairs) < n_needed:
        sys.exit(f"  ERROR: {len(pairs)} pairs found, n_needed is {n_needed}")
    write_rxns(os.path.join(W, "rxns.xml"), pairs, n_needed)
    write_input(os.path.join(W, "input.xml"), cfg)
    print(f"  {len(pairs)} pairs, n_needed {n_needed}")
    geometry_check(rec, lig, cfg)
    tidy(W, cfg, extra.get("cif"))

def find_templates():
    # The template must come from the code that runs the input. The run
    # scripts name that code in PYTHONPATH so it is tried first.
    for base in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        t = os.path.join(base, "pystarc", "templates")
        if base and os.path.isdir(t):
            return os.path.abspath(t)
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
    missing = [k for k in BD_PARAMS if k not in cfg and k not in OPTIONAL]
    if missing:
        sys.exit(f"  ERROR: config.xml has no {', '.join(missing)}")
    vals.update({k: cfg.get(k, OPTIONAL.get(k)) for k in BD_PARAMS})
    for k, v in vals.items():
        txt = re.sub(rf"<{k}>\s*</{k}>", f"<{k}>{v}</{k}>", txt)
    open(path, "w").write(txt)
    left = re.findall(r"<([a-z0-9_]+)>\s*</\1>", txt)
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

STRUCTURE = ("ligand_source", "receptor_pqr_file", "ligand_pqr_file",
             "prmtop", "complex_pdb", "pdb_id",
             "ligand_resid", "smiles", "chain", "rec_resid_min",
             "ligand_net_charge", "ligand_ccd", "ligand_protonate",
             "ligand_deprotonate", "keep_hetatm")
NAMES = ("ligand_formula", "receptor_resname", "ligand_resname")
CRITERION = ("mode", "receptor_numbering", "search_cutoff", "n_pairs",
             "receptor", "ligand", "atom_filter", "excluded", "pair_cutoff",
             "n_needed")

def _el(t, v, ind):
    if t == "receptor":
        rows = "".join(f'{ind}  <target resid="{r}" atom="{a}" resname="{n}"/>\n'
                       for r, a, n in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    if isinstance(v, (list, tuple)):
        rows = "".join(f'{ind}  <atom name="{_name(x)}"/>\n' for x in v) \
            if t in ("ligand", "ligand_protonate", "ligand_deprotonate") \
            else "".join(f"{ind}  <item>{_name(x)}</item>\n" for x in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    return f"{ind}<{t}>{v}</{t}>"

def _listed(names):
    n = list(names)
    return " and ".join(n) if len(n) < 3 else ", ".join(n[:-1]) + ", and " + n[-1]

def _doc(t):
    table = {"ligand_source": SOURCES, "mode": MODES,
             "atom_filter": ATOM_FILTERS,
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

SECTIONS = (
    ("structure", STRUCTURE), ("names", NAMES), ("criterion", CRITERION),
    ("geometry", GEOMETRY), ("electrostatics", ELECTROSTATICS),
    ("physics", PHYSICS), ("simulation", SIMULATION), ("compute", COMPUTE),
    ("outputs", OUTPUTS),
)

def config_xml(cfg):
    c = cfg.get("criterion", {})
    secs = [(order, c if name == "criterion" else cfg) for name, order in SECTIONS]
    w = max((len(_el(t, src[t], "    ").partition("\n")[0])
             for order, src in secs for t in order if t in src), default=0) + 2
    body = "".join(
        f"  <{name}>\n{_section(order, src, '    ', w)}\n  </{name}>\n"
        for (name, order), (_, src) in zip(SECTIONS, secs))
    return f'<?xml version="1.0" ?>\n<config>\n{body}</config>\n'

if __name__ == "__main__":
    if "--template" in sys.argv:
        blank = {t: "" for t in ("ligand_source",) + NAMES}
        blank["criterion"] = {"mode": "", "pair_cutoff": "", "n_needed": ""}
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