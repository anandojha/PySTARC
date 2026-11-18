"""
Builds the input files of a PySTARC run from a config.xml: receptor.pqr,
ligand.pqr, rxns.xml (or chain.json and reaction_pairs.json), input.xml and
build_record.json. The structure sources are pqr, prmtop, complex_pdb, rcsb,
rcsb_smiles and chain_pdb. The reaction criterion modes are contacts, centroid
and site.

    python run_pystarc.py config.xml --setup
    python run_pystarc.py --template
"""

import xml.etree.ElementTree as ET
import urllib.request
import numpy as np
import collections
import subprocess
import difflib
import shutil
import sys
import os
import re
import json
import math
import hashlib
from pystarc.structures.pqr_io import parse_pqr_records

SOURCES = {
    "pqr":         dict(need=("receptor_pqr_file", "ligand_pqr_file"), opt=(), note="Reads receptor and ligand PQR files that already exist"),
    "prmtop":      dict(need=("prmtop", "complex_pdb"), opt=(), note="Reads an AMBER topology and PDB that already exist"),
    "complex_pdb": dict(need=("complex_pdb",), opt=("ligand_net_charge", "ligand_ccd", "ligand_protonate", "ligand_deprotonate"), note="Takes one PDB holding the receptor and the ligand"),
    "rcsb":        dict(need=("pdb_id", "ligand_resid"), opt=("chain", "rec_resid_min", "ligand_net_charge", "ligand_ccd", "ligand_protonate", "ligand_deprotonate"), note="Downloads the entry and takes the ligand as deposited"),
    "rcsb_smiles": dict(need=("pdb_id", "ligand_resid", "smiles"), opt=("chain", "keep_hetatm", "ligand_net_charge"), note="Takes the receptor from the entry and builds the ligand from SMILES"),
    "chain_pdb":   dict(need=("complex_pdb", "receptor_pqr_file", "chain", "ligand_chain"), opt=(), note="Builds a flexible COFFDROP bead chain from a PDB complex and a receptor PQR"),
}

MODES = {
    "contacts": dict(need=("pair_cutoff", "n_needed"), opt=("receptor", "ligand", "receptor_numbering", "atom_filter", "search_cutoff", "n_pairs", "excluded"), note="Atom pairs listed in the config or picked from the crystal contacts"),
    "centroid": dict(need=("pair_cutoff",), opt=(), note="Pairs the centroid of the receptor with the centroid of the ligand"),
    "site":     dict(need=(), opt=("pair_cutoff", "margin", "receptor", "ligand", "receptor_numbering", "pocket_ligand", "pocket_ligand_resid"), note="Pairs a receptor anchor atom with a ligand anchor atom, or a ghost atom at the centroid of the bound ligand with the centroid of the ligand"),
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
    "receptor_polar": "Matches N O or S on the receptor side",
}

KEYS = {
    "ligand_source":       "Source of the receptor and ligand structures.",
    "receptor_pqr_file":   "Existing receptor PQR with charges and radii.",
    "ligand_pqr_file":     "Existing ligand PQR with charges and radii.",
    "prmtop":              "Existing AMBER topology.",
    "complex_pdb":         "Single PDB containing receptor and ligand.",
    "pdb_id":              "RCSB entry to download.",
    "ligand_resid":        "Residue name of the ligand within the RCSB entry.",
    "smiles":              "Ligand SMILES, in place of deposited coordinates.",
    "chain":               "Receptor chain retained. Blank retains all chains.",
    "ligand_chain":        "Ligand chain in complex_pdb for chain_pdb.",
    "rec_resid_min":       "Lowest receptor residue number retained.",
    "ligand_net_charge":   "Formal charge of the ligand. Blank takes the charge from the component definition or the SMILES.",
    "ligand_ccd":          "Component supplying bond orders and formal charges. Blank takes the residue name of the ligand when valid.",
    "ligand_protonate":    "Component atoms that each take one proton on top of the component definition.",
    "ligand_deprotonate":  "Component atoms that each drop one proton of the component definition.",
    "keep_hetatm":         "Het groups retained with the receptor.",
    "ligand_formula":      "Expected ligand composition. The build aborts on a mismatch.",
    "receptor_resname":    "Receptor label.",
    "ligand_resname":      "Ligand label.",
    "mode":                "How the atom pairs of the reaction criterion are chosen.",
    "receptor_numbering":  "Numbering for the target residues.",
    "search_cutoff":       "Mode contacts with pairs picked from the bound contacts. Search radius (in angstroms).",
    "n_pairs":             "Mode contacts with pairs picked from the bound contacts. Number of contacts retained, one or an even number.",
    "receptor":            "Receptor side of each listed pair of mode contacts, or the receptor anchor atom of mode site.",
    "ligand":              "Ligand side of each listed pair of mode contacts, or the ligand anchor atom of mode site. In mode contacts, auto chooses the nearest permitted atom.",
    "atom_filter":         "Element pairings permitted.",
    "excluded":            "Atom names omitted from the search.",
    "pair_cutoff":         "Reaction distance for every pair (in angstroms). Mode site takes pair_cutoff or margin.",
    "margin":              "Mode site distance added to the closest accessible approach (in angstroms), rounded to 0.01 A.",
    "pocket_ligand":       "Mode site without anchor atoms for a SMILES ligand. Residue name of the deposited ligand marking the site.",
    "pocket_ligand_resid": "Residue number of pocket_ligand when the chain holds several copies.",
    "n_needed":            "Mode contacts pairs that must be inside cutoff at once: 1 for one pair, half of an even pair count, or any positive count for chain_pdb.",
    "bd_milestone_radius": "Radius of the starting sphere (in angstroms).",
    "r_escape":            "Escape sphere radius (in angstroms). 0 derives twice bd_milestone_radius.",
    "n_trajectories":      "Number of Brownian dynamics trajectories.",
    "apbs_fglen":          "Fine electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_dime":           "Grid points per axis of the electrostatic grid.",
    "apbs_cglen":          "Coarse electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_coarse_dime":    "Grid points per axis of the coarse grid. 0 takes apbs_dime.",
    "apbs_fine_dime":      "Grid points per axis of the fine grid. 0 takes apbs_dime.",
    "debye_length":        "Electrostatic screening length of the salt (in angstroms). Blank derives it from ion_concentration.",
    "ion_concentration":   "Salt concentration (in molar).",
    "ion_radius_pos":      "Radius of the positive ion (in angstroms).",
    "ion_radius_neg":      "Radius of the negative ion (in angstroms).",
    "pdie":                "Dielectric constant of the protein.",
    "sdie":                "Dielectric constant of the solvent.",
    "srad":                "Solvent probe radius (in angstroms).",
    "desolvation_alpha":   "Scaling applied to the desolvation penalty.",
    "enable_born2_torque": "Add the desolvation of the receptor charges in the ligand Born field, a force and a torque on the ligand.",
    "hydrodynamic_interactions": "Include hydrodynamic coupling between the two molecules.",
    "multipole_fallback":  "Extend the electrostatics beyond the grid by multipole expansion.",
    "lj_forces":           "Apply repulsive forces at close contact.",
    "overlap_check":       "Forbid the ligand from entering the receptor volume.",
    "soft_repulsion":      "Soft excluded volume of BrownDye 2 between receptor and ligand atoms in place of the overlap check.",
    "gpu_force_batch":     "Trajectories per GPU force evaluation. 0 selects automatically.",
    "max_steps":           "Step limit per trajectory.",
    "dt":                  "Outer time step of the flexible chain model (in picoseconds). No effect on rigid body runs.",
    "minimum_core_dt":     "Smallest permitted time step. 0 imposes no floor.",
    "minimum_core_reaction_dt": "Smallest time step near the reaction surface (in picoseconds). 0 imposes no floor on the GPU.",
    "max_dt":              "Ceiling on the time step. 0 imposes no ceiling.",
    "r_hydro_rec":         "Receptor hydrodynamic radius. 0 derives it from the PQR.",
    "r_hydro_lig":         "Ligand hydrodynamic radius. 0 derives it from the PQR file.",
    "temperature":         "Temperature (in kelvin).",
    "seed":                "Random number seed.",
    "gpu":                 "Evaluate forces on the GPU.",
    "n_gpus":              "Number of GPUs for parallel trajectory splitting. 1 runs a single worker.",
    "n_threads":           "Number of CPU threads.",
    "work_dir":            "Output directory for grids, generated inputs and trajectory data.",
    "convergence_check":   "Run convergence analysis after the simulation.",
    "convergence_interval": "Reporting interval for the running rate.",
    "convergence_tol":     "Relative error defining convergence.",
    "checkpoint_interval": "Trajectories between checkpoints. 0 disables checkpointing.",
    "save_interval":       "Steps between saved positions. 0 saves endpoints.",
    "full_paths":          "Save positions and orientations every save_interval steps.",
    "energetics":          "Save the force on the ligand and the time step every save_interval steps.",
    "dt_chain":            "Inner time step of flexible chain internal coordinates (in picoseconds).",
    "chain_steps_per_outer": "Inner chain steps per outer step.",
    "n_equilibration_steps": "Pre-equilibration chain steps before each trajectory. 0 skips pre-equilibration.",
    "auto_diffusion":      "Compute chain diffusion tensors from RPY bead geometry.",
    "d_trans":             "Explicit chain translational diffusion when auto_diffusion is false (in A^2/ps).",
    "d_rot":               "Explicit chain rotational diffusion when auto_diffusion is false (in rad^2/ps).",
    "soft_repulsion_eps":  "Chain soft repulsion strength (in units of kT).",
    "target_grid_dx":      "Optional receptor electrostatic DX grid path for chain_pdb. Blank builds in work_dir.",
    "born_grid_dx":        "Optional receptor Born DX grid path for chain_pdb. Blank builds in work_dir.",
}

# Run parameters written into input.xml grouped by section of config.xml
GEOMETRY = ("bd_milestone_radius", "r_escape", "r_hydro_rec", "r_hydro_lig")
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
COMPUTE = ("gpu", "n_gpus", "gpu_force_batch", "n_threads", "work_dir")
OUTPUTS = ("save_interval", "full_paths", "energetics")
CHAIN = (
    "dt_chain", "chain_steps_per_outer", "n_equilibration_steps",
    "auto_diffusion", "d_trans", "d_rot", "soft_repulsion_eps",
    "target_grid_dx", "born_grid_dx",
)
INPUT_COMPUTE = ("gpu", "gpu_force_batch", "n_threads")
BD_PARAMS = GEOMETRY + ELECTROSTATICS + PHYSICS + SIMULATION + INPUT_COMPUTE + OUTPUTS

# Settings a config.xml may leave blank and the value input.xml then gets.
OPTIONAL = {"soft_repulsion": "false", "minimum_core_reaction_dt": "0",
            "full_paths": "true", "energetics": "true", "r_escape": "0",
            "debye_length": ""}

BOOL = ("enable_born2_torque", "hydrodynamic_interactions", "multipole_fallback",
        "lj_forces", "overlap_check", "soft_repulsion", "gpu", "convergence_check",
        "full_paths", "energetics", "auto_diffusion")

POLAR = {"N", "O", "S"}
SITE_ATOM = "PKT"
RECORD = "build_record.json"

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
          "n_pairs": int, "pair_cutoff": float, "n_needed": int, "margin": float,
          "pocket_ligand_resid": int, "n_gpus": int}

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
        a = {k: v.strip() for k, v in c.attrib.items() if v.strip()}
        if {"resid", "atom", "resname"} <= set(a):
            out.append((_number("resid", a["resid"], int), a["atom"], a["resname"]))
        elif a:
            out.append(tuple(a.values()))
        else:
            txt = (c.text or "").strip()
            if txt:
                out.append(txt)
    return out if out else ""

def _list(v):
    """A list key as a list. A key written as plain text holds one entry."""
    if v is None or v == "":
        return []
    return [v] if isinstance(v, (str, tuple)) else list(v)

def _src_file(cfg, W, rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    sdir = cfg.get("source_dir", W)
    cand = os.path.join(sdir, rel_path)
    if os.path.exists(cand):
        return cand
    return os.path.join(W, rel_path)

def read_config(path):
    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError(f"Expected <config> root in {path}, got <{root.tag}>")
    sec_map = dict(SECTIONS)
    tag_home = {t: sname for sname, order in SECTIONS for t in order}
    all_tags = list(tag_home)
    cfg = {}
    for sec in root:
        if sec.tag not in sec_map:
            hint = difflib.get_close_matches(sec.tag, list(sec_map), n=1)
            raise ValueError(
                f"Unknown section <{sec.tag}> in <config>"
                + (f": did you mean <{hint[0]}>?" if hint else ".")
            )
        allowed = set(sec_map[sec.tag])
        vals = {}
        for el in sec:
            if el.tag not in allowed:
                if el.tag in tag_home:
                    raise ValueError(
                        f"<{el.tag}> in <{sec.tag}>: place it inside <{tag_home[el.tag]}>."
                    )
                hint = difflib.get_close_matches(el.tag, all_tags, n=1)
                raise ValueError(
                    f"Unknown tag <{el.tag}> in <{sec.tag}>"
                    + (f": did you mean <{hint[0]}>?" if hint else ".")
                )
            v = _value(el)
            if v != "":
                vals[el.tag] = v
        if sec.tag == "criterion":
            cfg["criterion"] = vals
        elif sec.tag == "chain":
            cfg["chain_cfg"] = vals
        elif sec.tag == "compute":
            if "work_dir" in vals:
                cfg["bd_work_dir"] = vals.pop("work_dir")
            cfg.update(vals)
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
    top = _src_file(cfg, W, cfg["prmtop"])
    pdb = _src_file(cfg, W, cfg["complex_pdb"])
    for raw, f in ((cfg["prmtop"], top), (cfg["complex_pdb"], pdb)):
        if not os.path.exists(f):
            sys.exit(f"  ERROR: no {raw}")
    full = os.path.join(W, "_full.rst7")
    _gone(full)
    subprocess.run("cpptraj", input=f"parm {top}\ntrajin {pdb}\n"
                   f"trajout _full.rst7\nrun\n", cwd=W,
                   capture_output=True, text=True)
    if not os.path.exists(full):
        sys.exit(f"  ERROR: cpptraj could not read {cfg['complex_pdb']} with {cfg['prmtop']}")
    r = subprocess.run(["ambpdb", "-p", top, "-c", "_full.rst7", "-pqr"], cwd=W,
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

def ccd_definition(code, W, source_dir=None):
    """Bonds with their orders, formal charges, elements and leaving atoms of
    the deposited chemical component, by atom name."""
    cif_src = os.path.join(source_dir, f"{code}.cif") if source_dir else None
    cif = cif_src if (cif_src and os.path.exists(cif_src)) else os.path.join(W, f"{code}.cif")
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

def _parameterize(cfg, W):
    from rdkit.Chem import rdDetermineBonds
    from rdkit import Chem
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
        bonds, charges, elements, leaving = ccd_definition(code, W, cfg.get("source_dir"))
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
    """Split the PDB, parameterize both, ambpdb."""
    src = _src_file(cfg, W, cfg["complex_pdb"])
    if not os.path.exists(src):
        sys.exit(f"  ERROR: no {cfg['complex_pdb']}")
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
    return {"cif": _parameterize(cfg, W)}

def source_rcsb(cfg, W):
    """Fetch the entry, extract chain and ligand, parameterize.

    CONECT records for the ligand come across too and the bond orders come
    from the component definition when the composition check asks for them.
    """
    pid = cfg["pdb_id"]
    entry = _src_file(cfg, W, f"{pid}.pdb")
    if not os.path.exists(entry):
        entry = os.path.join(W, f"{pid}.pdb")
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
    code = _parameterize(cfg, W)
    return {"rmap": resid_map(entry, chain, rmin), "cif": code}

def _parameterize_smiles(cfg, W):
    """tleap the protein, RDKit and antechamber the ligand, ambpdb both.

    Reads protein.pdb, writes receptor.pqr and ligand.pqr. The conformer is
    embedded from the config seed, so the geometry and therefore the charges
    are reproducible.
    """
    from rdkit.Chem import AllChem
    from rdkit import Chem
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
    entry = _src_file(cfg, W, f"{pid}.pdb")
    if not os.path.exists(entry):
        entry = os.path.join(W, f"{pid}.pdb")
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
    _parameterize_smiles(cfg, W)

def source_pqr(cfg, W):
    """Receptor and ligand PQR files that already exist, charges and radii included."""
    for key, out in (("receptor_pqr_file", "receptor.pqr"),
                     ("ligand_pqr_file", "ligand.pqr")):
        src = _src_file(cfg, W, cfg[key])
        if not os.path.exists(src):
            sys.exit(f"  ERROR: no {cfg[key]}")
        dst = os.path.join(W, out)
        if os.path.abspath(src) == os.path.abspath(dst):
            sys.exit(f"  ERROR: {key} must not overwrite {out} in place")
        # The run places its own ghost atoms, so a GHO or PKT left in the file
        # by an earlier run is taken out here.
        lines = open(src).read().split("\n")
        ghost = lambda l: (l.startswith(("ATOM", "HETATM"))
                           and (l[12:16].strip().upper() in ("GHO", SITE_ATOM)
                                or (len(l.split()) >= 3 and l.split()[2].upper() in ("GHO", SITE_ATOM))))
        n_gho = sum(1 for l in lines if ghost(l))
        open(dst, "w").write("\n".join(l for l in lines if not ghost(l)))
        n = sum(1 for l in open(dst) if l.startswith(("ATOM", "HETATM")))
        print(f"  {out}: {n} atoms from {cfg[key]}"
              + (f", {n_gho} ghost atom left out" if n_gho else ""))

TLEAP_VARIANT_GROUPS = [
    {"HIS", "HIE", "HID", "HIP"},
    {"CYS", "CYX", "CYM"},
    {"ASP", "ASH"},
    {"GLU", "GLH"},
    {"LYS", "LYN"},
]

def _resname_match(r1, r2):
    return r1 == r2 or any(r1 in g and r2 in g for g in TLEAP_VARIANT_GROUPS)

def _parse_pdb_chain_heavy(pdb_path, chain_id):
    atoms = []
    for line in open(pdb_path):
        if not line.startswith("ATOM") or line[21] != chain_id:
            continue
        aname = line[12:16].strip()
        if aname.startswith("H") or (len(aname) > 1 and aname[0].isdigit() and aname[1] == "H"):
            continue
        atoms.append({
            "atom_name": aname,
            "resname": line[17:20].strip(),
            "resid": int(line[22:26]),
            "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        })
    return atoms

def _seq_residues(atoms):
    out, last = [], None
    for a in atoms:
        if a["resid"] != last:
            last = a["resid"]
            out.append({"seq_pos": len(out), "resname": a["resname"], "resid_native": last, "atoms": []})
        out[-1]["atoms"].append(a)
    return out

def _center_pqr_copy(src, dst):
    lines = open(src).read().splitlines()
    hits = [i for i, l in enumerate(lines)
            if l[:6].strip() in ("ATOM", "HETATM") and len(l) >= 54
            and l[12:16].strip().upper() not in ("GHO", SITE_ATOM)]
    if not hits:
        sys.exit(f"  ERROR: no fixed-column ATOM records in {src}")
    xyz = np.array([[float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])] for i in hits])
    shift = xyz.mean(axis=0)
    for k, i in enumerate(hits):
        v = xyz[k] - shift
        lines[i] = f"{lines[i][:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{lines[i][54:]}"
    open(dst, "w").write("\n".join(lines) + "\n")

def source_chain_pdb(cfg, W):
    """Builds receptor.pqr (centered), chain.json and reaction_pairs.json for chain BD."""
    from pystarc.simulation.coffdrop_chain import chain_from_pdb
    from pystarc.structures.chain_io import save_chain_to_json
    pdb_src = _src_file(cfg, W, cfg["complex_pdb"])
    pqr_src = _src_file(cfg, W, cfg["receptor_pqr_file"])
    for raw, f in ((cfg["complex_pdb"], pdb_src), (cfg["receptor_pqr_file"], pqr_src)):
        if not os.path.exists(f):
            sys.exit(f"  ERROR: no {raw}")
    rec_chain = cfg["chain"]
    lig_chain = cfg["ligand_chain"]
    dst_pqr = os.path.join(W, "receptor.pqr")
    _center_pqr_copy(pqr_src, dst_pqr)
    rec_pdb = _parse_pdb_chain_heavy(pdb_src, rec_chain)
    lig_pdb = _parse_pdb_chain_heavy(pdb_src, lig_chain)
    if not rec_pdb or not lig_pdb:
        sys.exit(f"  ERROR: empty chain {rec_chain} or {lig_chain} in {cfg['complex_pdb']}")
    rec_res_pdb = _seq_residues(rec_pdb)
    lig_res_pdb = _seq_residues(lig_pdb)
    rec_pqr_recs = [r for r in parse_pqr_records(dst_pqr) if r.record_type == "ATOM"]
    rec_pqr_atoms = [{"atom_idx": i, "atom_name": r.name, "resname": r.resname,
                      "resid": r.resid, "xyz": np.array([r.x, r.y, r.z])}
                     for i, r in enumerate(rec_pqr_recs)]
    rec_res_pqr = _seq_residues(rec_pqr_atoms)
    if len(rec_res_pdb) != len(rec_res_pqr):
        sys.exit(f"  ERROR: residue count mismatch: PDB={len(rec_res_pdb)} vs PQR={len(rec_res_pqr)}")
    for k, (r1, r2) in enumerate(zip(rec_res_pdb, rec_res_pqr)):
        if not _resname_match(r1["resname"], r2["resname"]):
            sys.exit(f"  ERROR: residue mismatch at position {k}: {r1['resname']} vs {r2['resname']}")
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    coffdrop_dir = os.path.join(pkg_dir, "coffdrop_data")
    chain = chain_from_pdb(pdb_src, chain_id=lig_chain, name=cfg["ligand_resname"],
                           sidechains=True, coffdrop_dir=coffdrop_dir)
    unique_resids = []
    for bead in chain.atoms:
        if not unique_resids or unique_resids[-1] != bead.resid:
            unique_resids.append(bead.resid)
    resid_to_sp = {r: k for k, r in enumerate(unique_resids)}
    map_root = ET.parse(os.path.join(coffdrop_dir, "map.xml")).getroot()
    bead_map = {
        re_el.findtext("name").strip(): {
            b_el.findtext("name").strip(): b_el.findtext("atoms").strip().split()
            for b_el in re_el.findall("bead")
        }
        for re_el in map_root.findall("residue")
    }
    def _resolve_res(rn):
        if rn in bead_map:
            return rn
        for g in TLEAP_VARIANT_GROUPS:
            if rn in g:
                for cand in g:
                    if cand in bead_map:
                        return cand
        return None
    body_positions = np.zeros((chain.n_atoms, 3))
    for i, bead in enumerate(chain.atoms):
        b_rn, b_an = bead.resname.split(":")
        res = lig_res_pdb[resid_to_sp[bead.resid]]
        mrn = _resolve_res(b_rn)
        if mrn is None or b_an not in bead_map[mrn]:
            sys.exit(f"  ERROR: unmapped COFFDROP bead {bead.resname}")
        coords = [a["xyz"] for an in bead_map[mrn][b_an]
                  for a in res["atoms"] if a["atom_name"] == an]
        if not coords:
            coords = [a["xyz"] for a in res["atoms"] if a["atom_name"] == "CA"]
        if not coords:
            sys.exit(f"  ERROR: no coordinates found for bead {bead.resname}")
        body_positions[i] = np.mean(coords, axis=0)
    chain_json_path = os.path.join(W, "chain.json")
    save_chain_to_json(chain, body_positions, chain_json_path)
    c = cfg["criterion"]
    rec_xyz = np.array([a["xyz"] for a in rec_pdb])
    lig_xyz = np.array([a["xyz"] for a in lig_pdb])
    hits = np.argwhere(np.linalg.norm(rec_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1) < c["search_cutoff"])
    rec_sp = {r["resid_native"]: r["seq_pos"] for r in rec_res_pdb}
    lig_sp = {r["resid_native"]: r["seq_pos"] for r in lig_res_pdb}
    beads_by_sp = {}
    for i, bead in enumerate(chain.atoms):
        beads_by_sp.setdefault(resid_to_sp[bead.resid], []).append((i, bead))
    pairs = set()
    for ia, id_ in hits:
        ra, la = rec_pdb[int(ia)], lig_pdb[int(id_)]
        sp_r = rec_sp.get(ra["resid"])
        rec_idx = next((a["atom_idx"] for a in rec_res_pqr[sp_r]["atoms"]
                        if a["atom_name"] == ra["atom_name"]), None) if sp_r is not None else None
        sp_l = lig_sp.get(la["resid"])
        cands = beads_by_sp.get(sp_l, []) if sp_l is not None else []
        tname = f"{la['resname']}:{la['atom_name']}"
        bead_idx = next((i for i, b in cands if b.resname == tname), None)
        if bead_idx is None and cands:
            bead_idx = min(cands, key=lambda ib: float(np.linalg.norm(body_positions[ib[0]] - la["xyz"])))[0]
        if rec_idx is not None and bead_idx is not None:
            pairs.add((int(rec_idx), int(bead_idx)))
    min_req = max(int(c.get("n_pairs", 0) or 0), int(c["n_needed"]))
    if len(pairs) < min_req:
        sys.exit(f"  ERROR: {len(pairs)} chain reaction pairs found (minimum {min_req}).")
    rp_list = [[rec_i, bd_i, float(c["pair_cutoff"])] for rec_i, bd_i in sorted(pairs)]
    with open(os.path.join(W, "reaction_pairs.json"), "w") as fh:
        json.dump(rp_list, fh, indent=2)
    print(f"  chain.json: {chain.n_atoms} beads; reaction_pairs.json: {len(rp_list)} pairs, n_needed {c['n_needed']}")
    return {"chain_pairs": rp_list, "n_beads": chain.n_atoms}

HANDLERS = {"pqr": source_pqr, "prmtop": source_prmtop,
            "complex_pdb": source_complex_pdb,
            "rcsb": source_rcsb, "rcsb_smiles": source_rcsb_smiles,
            "chain_pdb": source_chain_pdb}

# reaction criterion and output files
def _element(name, column="", resname=""):
    """Element of a PQR atom, upper case. The element column when the file has
    one, else the name, with two-letter halogens and metal ions distinguished."""
    if column.isalpha() and len(column) <= 2:
        return column.upper()
    ru = resname.strip().upper()
    nm = name.lstrip("0123456789").upper()
    if ru in ("CAL", "CA") and nm.startswith("CA"):
        return "CA"
    if ru in ("ZN", "MG", "FE", "CU") and nm.startswith(ru):
        return ru
    return nm[:2] if nm[:2] in ("CL", "BR") else nm[:1]

def read_pqr(path):
    return [dict(name=a.name, el=_element(a.name, a.element, a.resname), resname=a.resname,
                 resid=a.resid, xyz=np.array([a.x, a.y, a.z]), q=a.charge, r=a.radius)
            for a in parse_pqr_records(path)]

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

def round_quarter(x):
    return math.floor(x / 0.25 + 0.5) * 0.25

def pair_count_ok(n, need):
    """One pair with n_needed 1, or an even number of pairs with n_needed half of them."""
    return (n == 1 and need == 1) or (n >= 2 and n % 2 == 0 and need == n // 2)

def contacts_style(c):
    listed = [k for k in ("receptor", "ligand", "receptor_numbering") if k in c]
    picked = [k for k in ("search_cutoff", "n_pairs", "excluded") if k in c]
    if listed and picked:
        sys.exit(f"  ERROR: Mode contacts either lists its pairs or picks them from the crystal contacts, "
                 f"not both (given {listed} and {picked}).")
    need = ("receptor", "ligand") if listed else ("search_cutoff", "n_pairs", "atom_filter")
    missing = [k for k in need if c.get(k) in (None, "", [], ())]
    if missing:
        sys.exit(f"  ERROR: Mode contacts with {'listed pairs' if listed else 'pairs picked from the crystal contacts'} "
                 f"needs {missing}.")
    return "listed" if listed else "picked"

def check_criterion(c, src):
    _check(c.get("mode"), MODES, c, "criterion mode")
    mode = c["mode"]
    if "pair_cutoff" in c and (not isinstance(c["pair_cutoff"], (int, float)) or c["pair_cutoff"] <= 0):
        sys.exit("  ERROR: pair_cutoff must be a distance above zero (in angstroms).")
    if c.get("receptor_numbering", "internal") not in NUMBERING:
        sys.exit(f"  ERROR: receptor_numbering must be one of {sorted(NUMBERING)}.")
    if mode == "site":
        if sum(c.get(k) not in (None, "") for k in ("pair_cutoff", "margin")) != 1:
            sys.exit("  ERROR: Mode site takes exactly one of pair_cutoff and margin.")
        if "margin" in c and (not isinstance(c["margin"], (int, float)) or c["margin"] < 0):
            sys.exit("  ERROR: margin must be a distance of 0 or more (in angstroms).")
        anchors = [k for k in ("receptor", "ligand") if c.get(k) not in (None, "", [], ())]
        if anchors:
            if len(anchors) == 1 or len(_list(c["receptor"])) != 1 or len(_list(c["ligand"])) != 1:
                sys.exit("  ERROR: Mode site with anchor atoms takes one receptor target and one ligand atom.")
            if _name(_list(c["ligand"])[0]) == "auto":
                sys.exit("  ERROR: The ligand anchor atom of mode site must be named, since auto applies only to mode contacts.")
            if "pocket_ligand" in c or "pocket_ligand_resid" in c:
                sys.exit("  ERROR: pocket_ligand marks the site only without anchor atoms, so remove it.")
            return mode
        if "receptor_numbering" in c:
            sys.exit("  ERROR: receptor_numbering applies in mode site only to anchor atoms.")
        if src == "rcsb_smiles" and not c.get("pocket_ligand"):
            sys.exit("  ERROR: Mode site with a ligand built from SMILES needs pocket_ligand, "
                     "the deposited ligand whose centroid marks the site.")
        if src != "rcsb_smiles" and ("pocket_ligand" in c or "pocket_ligand_resid" in c):
            sys.exit("  ERROR: pocket_ligand applies only to a ligand built from SMILES.")
    if mode == "contacts":
        if src == "chain_pdb":
            if not isinstance(c.get("search_cutoff"), (int, float)) or c["search_cutoff"] <= 0:
                sys.exit("  ERROR: chain_pdb contacts needs search_cutoff above zero.")
            if not isinstance(c.get("n_needed"), int) or c["n_needed"] < 1:
                sys.exit("  ERROR: chain_pdb contacts needs n_needed >= 1.")
            return mode
        if contacts_style(c) == "picked":
            if c["atom_filter"] not in ATOM_FILTERS:
                sys.exit(f"  ERROR: atom_filter must be one of {sorted(ATOM_FILTERS)}.")
            if not isinstance(c["search_cutoff"], (int, float)) or c["search_cutoff"] <= 0:
                sys.exit("  ERROR: search_cutoff must be above zero.")
            n = c["n_pairs"]
        else:
            L = c["ligand"]
            auto = L == "auto" or any(_name(x) == "auto" for x in _list(L))
            if auto and c.get("atom_filter") not in ATOM_FILTERS:
                sys.exit(f"  ERROR: A ligand side of auto needs atom_filter, one of {sorted(ATOM_FILTERS)}.")
            if not auto and "atom_filter" in c:
                sys.exit("  ERROR: atom_filter applies only to a ligand side of auto, so remove it.")
            n = len(_list(c["receptor"]))
        if not pair_count_ok(n, c["n_needed"]):
            sys.exit(f"  ERROR: {n} pairs with n_needed {c['n_needed']} are not allowed. Use one pair with "
                     "n_needed 1, or an even number of pairs with n_needed half of them.")
    elif src == "chain_pdb":
        sys.exit("  ERROR: chain_pdb requires criterion mode 'contacts'.")
    return mode

def contacts_picked(rec, lig, c):
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

def receptor_atom(rec, target, c, rmap=None):
    """Index of the receptor atom of one <target resid atom resname/> entry."""
    if not (isinstance(target, tuple) and len(target) == 3):
        sys.exit('  ERROR: receptor needs <target resid="..." atom="..." '
                 'resname="..."/> entries')
    resid, aname, resname = target
    if c.get("receptor_numbering", "internal") == "pdb":
        if not rmap:
            sys.exit("  ERROR: receptor_numbering is pdb but no map given")
        if resid not in rmap:
            sys.exit(f"  ERROR: residue {resid} is not in the source PDB")
        resid = rmap[resid]
    i = next((k for k, a in enumerate(rec) if a["resid"] == resid
              and a["name"] == aname and a["resname"] == resname), None)
    if i is None:
        sys.exit(f"  ERROR: no {resname}{resid} {aname} in the receptor")
    return i

def ligand_atom(lig, name):
    """Index of the ligand atom of this name. ligand.pqr carries the names of the structure."""
    js = [k for k, b in enumerate(lig) if b["name"] == name]
    if len(js) != 1:
        sys.exit(f"  ERROR: {'no' if not js else 'more than one'} ligand atom "
                 f"{name} in ligand.pqr")
    return js[0]

def contacts_listed(rec, lig, c, rmap=None):
    R, L = _list(c["receptor"]), c["ligand"]
    L = ["auto"] * len(R) if L == "auto" else _list(L)
    if len(R) != len(L):
        sys.exit(f"  ERROR: {len(R)} receptor targets against {len(L)} ligand "
                 f"atoms, listed pairs match them one to one")
    out = []
    for target, lname in zip(R, L):
        i = receptor_atom(rec, target, c, rmap)
        if _name(lname) == "auto":
            keep = keeper(c["atom_filter"])
            near = [(float(np.linalg.norm(rec[i]["xyz"] - b["xyz"])), k)
                    for k, b in enumerate(lig) if b["el"] != "H"
                    and keep(rec[i]["el"], b["el"])]
            if not near:
                sys.exit(f"  ERROR: no {c['atom_filter']} ligand atom for "
                         f"{rec[i]['resname']}{rec[i]['resid']} {rec[i]['name']}")
            j = min(near)[1]
        else:
            j = ligand_atom(lig, _name(lname))
        d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
        out.append((i, j, c["pair_cutoff"], d))
    return out

def centroid_pair(rec, lig, c):
    """One pair between the GHO atoms that PySTARC appends at the centroid of each molecule."""
    R = np.mean([a["xyz"] for a in rec], axis=0)
    L = np.mean([a["xyz"] for a in lig], axis=0)
    return (len(rec), len(lig), c["pair_cutoff"], float(np.linalg.norm(R - L)))

def deposited_centroid(entry, chain, resname, resid=None):
    groups = {}
    for line in open(entry):
        if not line.startswith(("ATOM", "HETATM")) or line[17:20].strip() != resname:
            continue
        if (chain and line[21] != chain) or line[16] not in (" ", "A"):
            continue
        if resid is not None and int(line[22:26]) != resid:
            continue
        groups.setdefault(int(line[22:26]), []).append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not groups:
        sys.exit(f"  ERROR: Chain {chain} of {entry} has no {resname} {resid or ''}.")
    if len(groups) > 1:
        sys.exit(f"  ERROR: Chain {chain} of {entry} holds {resname} {sorted(groups)}. Give pocket_ligand_resid.")
    xyz = np.array(next(iter(groups.values())))
    return xyz.mean(axis=0), len(xyz)

def write_site_atom(W, rec, p):
    """Appends PKT at p to receptor.pqr with no charge and no radius, so that it
    moves with the receptor and exerts no force. PySTARC appends GHO after it."""
    path = os.path.join(W, "receptor.pqr")
    lines = [l for l in open(path).read().split("\n") if l.strip() and l.strip() != "END"]
    serial = max(int(l[6:11]) for l in lines if l.startswith(("ATOM", "HETATM"))) + 1
    resid = max(a["resid"] for a in rec) + 1
    lines.append(f"HETATM{serial:5d} {SITE_ATOM:<4s} {SITE_ATOM:<4s}{resid:5d}    "
                 f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  {0.0:7.4f}  {0.0:6.4f}")
    open(path, "w").write("\n".join(lines) + "\nEND\n")

def write_rxns(path, pairs, n_needed):
    body = "".join(
        f"        <pair>\n          <atoms> {i + 1} {j + 1} </atoms>\n"
        f"          <distance> {float(cut)!s} </distance>\n        </pair>\n"
        for i, j, cut, _d in pairs)
    open(path, "w").write(
        "<top>\n  <first_state> start </first_state>\n  <reactions>\n    <reaction>\n"
        "      <name> association </name>\n      <state_before> start </state_before>\n"
        "      <state_after> end </state_after>\n      <criterion>\n        <molecules>\n"
        "          <molecule0> rec rec </molecule0>\n          <molecule1> lig lig </molecule1>\n"
        f"        </molecules>\n        <n_needed> {n_needed} </n_needed>\n{body}"
        "      </criterion>\n    </reaction>\n  </reactions>\n</top>\n")

def site_pair(W, rec, lig, c, cfg, rec_out, rmap=None):
    """Returns the site pair: the anchor atoms when the config names them, else
    a ghost atom at the centroid of the bound ligand, placed here, and the
    centroid of the ligand. The closest approach scan of a margin runs on the
    GPU when CuPy is present."""
    anchor = None
    if c.get("receptor"):
        i = receptor_atom(rec, _list(c["receptor"])[0], c, rmap)
        anchor = ligand_atom(lig, _name(_list(c["ligand"])[0]))
        p = rec[i]["xyz"]
        rec_out["anchors"] = dict(receptor=[i + 1, rec[i]["resname"], rec[i]["resid"], rec[i]["name"]],
                                  ligand=[anchor + 1, lig[anchor]["name"]])
        rec_out["site_reference"] = (f"receptor atom {i + 1} ({rec[i]['resname']} {rec[i]['resid']} {rec[i]['name']}) "
                                     f"and ligand atom {anchor + 1} ({lig[anchor]['name']})")
    elif cfg["ligand_source"] == "rcsb_smiles":
        entry = _src_file(cfg, W, f"{cfg['pdb_id']}.pdb")
        p, n = deposited_centroid(entry, cfg.get("chain"),
                                  c["pocket_ligand"], c.get("pocket_ligand_resid"))
        rec_out["site_reference"] = f"{c['pocket_ligand']} {c.get('pocket_ligand_resid', '')} of {cfg['pdb_id']}, {n} atoms"
    else:
        p = np.mean([a["xyz"] for a in lig], axis=0)
        rec_out["site_reference"] = f"bound ligand as built, {len(lig)} atoms"
    cut = c.get("pair_cutoff")
    if c.get("margin") not in (None, ""):
        from pystarc.analysis.closest_approach import closest_approach
        res = closest_approach([a["xyz"] for a in rec], [a["r"] for a in rec], [a["xyz"] for a in lig],
                               [a["r"] for a in lig], p, anchor=anchor, bound=cfg["ligand_source"] != "rcsb_smiles")
        if res["closest"] is None:
            sys.exit("  ERROR: The scan found no accessible pose of the ligand near the site.")
        cut = round(res["closest"] + c["margin"], 2)
        rec_out.update(closest_approach=res["closest"], margin=c["margin"], scan=res)
        print(f"  The closest accessible approach is {res['closest']:.2f} A, so the cutoff is {cut} A "
              f"with a margin of {c['margin']} A.")
    if anchor is not None:
        print(f"  The site pair is {rec_out['site_reference']}.")
        return (i, anchor, cut, float(np.linalg.norm(p - lig[anchor]["xyz"])))
    write_site_atom(W, rec, p)
    rec_out["site_atom"] = [round(float(x), 3) for x in p]
    print(f"  The site ghost atom {SITE_ATOM} sits at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) as receptor atom {len(rec) + 1}.")
    return (len(rec), len(lig), cut, 0.0)

def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()

TOP_LEVEL = ({"ligand_source", "ligand_formula", "receptor_resname",
              "ligand_resname", "criterion", "work_dir", "bd_work_dir",
              "source_dir", "chain_cfg", "n_gpus", "resolve_paths_in_w"}
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

def write_chain_input(path, cfg, W):
    cc = cfg.get("chain_cfg", {})
    use_abs = bool(cfg.get("resolve_paths_in_w"))
    rec_pqr = os.path.join(W, "receptor.pqr") if use_abs else "receptor.pqr"
    chain_json = os.path.join(W, "chain.json") if use_abs else "chain.json"
    rp_json = os.path.join(W, "reaction_pairs.json") if use_abs else "reaction_pairs.json"
    sdir = cfg.get("source_dir", W)
    def _grid_path(raw, default_name):
        if not raw:
            return os.path.join(W, default_name) if use_abs else default_name
        if os.path.isabs(raw):
            return raw
        return os.path.join(sdir, raw) if use_abs else raw
    tg_dx = _grid_path(cc.get("target_grid_dx", ""), "receptor1.dx")
    bn_dx = _grid_path(cc.get("born_grid_dx", ""), "receptor0_born.dx")
    auto_diff = str(cc.get("auto_diffusion", "true"))
    diff_xml = "" if auto_diff.lower() in ("true", "yes", "1") else (
        f"\n    <D_trans>{cc.get('d_trans', '0.015')}</D_trans>"
        f"\n    <D_rot>{cc.get('d_rot', '0.00005')}</D_rot>"
    )
    debye_line = (f"  <debye_length>{cfg['debye_length']}</debye_length>\n"
                  if cfg.get("debye_length") not in (None, "") else "")
    bd_wdir = cfg.get("bd_work_dir", "bd_sims")
    txt = (
        '<?xml version="1.0"?>\n<pystarc>\n'
        f'  <receptor_resname>{cfg["receptor_resname"]}</receptor_resname>\n'
        f'  <ligand_resname>{cfg["ligand_resname"]}</ligand_resname>\n'
        f'  <receptor_pqr>{rec_pqr}</receptor_pqr>\n'
        f'{debye_line}'
        f'  <ion_concentration>{cfg.get("ion_concentration", "0.15")}</ion_concentration>\n'
        f'  <ion_radius_pos>{cfg.get("ion_radius_pos", "0.95")}</ion_radius_pos>\n'
        f'  <ion_radius_neg>{cfg.get("ion_radius_neg", "1.81")}</ion_radius_neg>\n'
        f'  <pdie>{cfg.get("pdie", "4.0")}</pdie>\n'
        f'  <sdie>{cfg.get("sdie", "78.0")}</sdie>\n'
        f'  <srad>{cfg.get("srad", "1.4")}</srad>\n'
        f'  <apbs_cglen>{cfg.get("apbs_cglen", "0")}</apbs_cglen>\n'
        f'  <apbs_fglen>{cfg["apbs_fglen"]}</apbs_fglen>\n'
        f'  <apbs_dime>{cfg["apbs_dime"]}</apbs_dime>\n'
        f'  <r_hydro_rec>{cfg.get("r_hydro_rec", "0")}</r_hydro_rec>\n'
        f'  <bd_milestone_radius>{cfg["bd_milestone_radius"]}</bd_milestone_radius>\n'
        f'  <n_trajectories>{cfg["n_trajectories"]}</n_trajectories>\n'
        f'  <max_steps>{cfg["max_steps"]}</max_steps>\n'
        f'  <dt>{cfg["dt"]}</dt>\n'
        f'  <temperature>{cfg["temperature"]}</temperature>\n'
        f'  <seed>{cfg["seed"]}</seed>\n'
        f'  <work_dir>{bd_wdir}</work_dir>\n'
        f'  <gpu>{cfg.get("gpu", "false")}</gpu>\n'
        f'  <hydrodynamic_interactions>{cfg.get("hydrodynamic_interactions", "true")}</hydrodynamic_interactions>\n'
        f'  <desolvation_alpha>{cfg["desolvation_alpha"]}</desolvation_alpha>\n'
        f'  <convergence_check>{cfg["convergence_check"]}</convergence_check>\n'
        f'  <convergence_interval>{cfg["convergence_interval"]}</convergence_interval>\n'
        f'  <convergence_tol>{cfg["convergence_tol"]}</convergence_tol>\n'
        f'  <checkpoint_interval>{cfg.get("checkpoint_interval", "0")}</checkpoint_interval>\n'
        '  <outputs>\n'
        f'    <save_interval>{cfg["save_interval"]}</save_interval>\n'
        f'    <full_paths>{cfg.get("full_paths", "true")}</full_paths>\n'
        f'    <energetics>{cfg.get("energetics", "true")}</energetics>\n'
        '  </outputs>\n'
        '  <chain>\n'
        f'    <chain_json>{chain_json}</chain_json>\n'
        f'    <reaction_pairs_json>{rp_json}</reaction_pairs_json>\n'
        f'    <target_grid_dx>{tg_dx}</target_grid_dx>\n'
        f'    <born_grid_dx>{bn_dx}</born_grid_dx>\n'
        f'    <r_escape>{cfg.get("r_escape", "0")}</r_escape>\n'
        f'    <reaction_n_needed>{cfg["criterion"]["n_needed"]}</reaction_n_needed>\n'
        f'    <auto_diffusion>{auto_diff}</auto_diffusion>{diff_xml}\n'
        f'    <use_soft_repulsion>{cfg.get("soft_repulsion", "true")}</use_soft_repulsion>\n'
        f'    <soft_repulsion_eps>{cc.get("soft_repulsion_eps", "1.0")}</soft_repulsion_eps>\n'
        f'    <n_workers>{cfg.get("n_threads", "1")}</n_workers>\n'
        f'    <dt_chain>{cc.get("dt_chain", "0.05")}</dt_chain>\n'
        f'    <chain_steps_per_outer>{cc.get("chain_steps_per_outer", "4")}</chain_steps_per_outer>\n'
        f'    <n_equilibration_steps>{cc.get("n_equilibration_steps", "0")}</n_equilibration_steps>\n'
        '  </chain>\n'
        '</pystarc>\n'
    )
    with open(path, "w") as fh:
        fh.write(txt)

def build(**cfg):
    src = cfg.get("ligand_source")
    _check(src, SOURCES, cfg, "ligand_source", ignore=TOP_LEVEL)
    c = cfg.get("criterion", {})
    mode = check_criterion(c, src)
    unknown = set(cfg) - TOP_LEVEL - set(SOURCES[src]["need"]) \
        - set(SOURCES[src]["opt"])
    if unknown:
        sys.exit(f"CONFIG does not use {sorted(unknown)}")
    if src != "chain_pdb" and cfg.get("chain_cfg"):
        sys.exit(f"  ERROR: <chain> settings {sorted(cfg['chain_cfg'])} apply to chain_pdb runs.")
    req_names = ("receptor_resname", "ligand_resname") if src == "chain_pdb" else NAMES
    missing = [k for k in req_names if not cfg.get(k)]
    if missing:
        sys.exit(f"config.xml needs {missing} in the names section")
    W = cfg["work_dir"]
    os.makedirs(W, exist_ok=True)
    extra = HANDLERS[src](cfg, W) or {}
    if src == "chain_pdb":
        write_chain_input(os.path.join(W, "input.xml"), cfg, W)
        record = dict(
            mode=mode,
            ligand_source=src,
            n_beads=extra["n_beads"],
            n_needed=c["n_needed"],
            pairs=extra["chain_pairs"],
            code=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            md5={f: _md5(os.path.join(W, f)) for f in ("receptor.pqr", "chain.json", "reaction_pairs.json", "input.xml")},
        )
        with open(os.path.join(W, RECORD), "w") as fh:
            json.dump(record, fh, indent=1, default=float)
        tidy(W, cfg)
        return
    rec = read_pqr(os.path.join(W, "receptor.pqr"))
    lig = read_pqr(os.path.join(W, "ligand.pqr"))
    check_formula(lig, cfg["ligand_formula"])
    record = dict(mode=mode, ligand_source=src)
    if mode == "contacts":
        record["contacts_style"] = contacts_style(c)
        pairs = (contacts_picked(rec, lig, c) if record["contacts_style"] == "picked"
                 else contacts_listed(rec, lig, c, extra.get("rmap")))
        n_needed = c["n_needed"]
        if not pair_count_ok(len(pairs), n_needed):
            sys.exit(f"  ERROR: {len(pairs)} pairs were found for n_needed {n_needed}.")
    elif mode == "centroid":
        pairs, n_needed = [centroid_pair(rec, lig, c)], 1
    else:
        pairs, n_needed = [site_pair(W, rec, lig, c, cfg, record, extra.get("rmap"))], 1
    write_rxns(os.path.join(W, "rxns.xml"), pairs, n_needed)
    write_input(os.path.join(W, "input.xml"), cfg)
    record.update(pairs=[[int(i) + 1, int(j) + 1, float(cut)] for i, j, cut, _d in pairs], n_needed=n_needed,
                  code=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  md5={f: _md5(os.path.join(W, f)) for f in ("receptor.pqr", "ligand.pqr", "rxns.xml", "input.xml")})
    with open(os.path.join(W, RECORD), "w") as fh:
        json.dump(record, fh, indent=1, default=float)
    print(f"  {len(pairs)} pairs, n_needed {n_needed}")
    geometry_check(rec, lig, cfg)
    tidy(W, cfg, extra.get("cif"))

def find_templates():
    """The input template of this package, the code that also runs the input."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

def write_input(path, cfg):
    txt = open(os.path.join(find_templates(), "input.xml")).read()
    use_abs = bool(cfg.get("resolve_paths_in_w"))
    W = cfg["work_dir"]
    vals = {"receptor_resname": cfg["receptor_resname"],
            "ligand_resname": cfg["ligand_resname"],
            "receptor_pqr": os.path.join(W, "receptor.pqr") if use_abs else "receptor.pqr",
            "ligand_pqr": os.path.join(W, "ligand.pqr") if use_abs else "ligand.pqr",
            "rxns_xml": os.path.join(W, "rxns.xml") if use_abs else "rxns.xml",
            "work_dir": cfg.get("bd_work_dir", "bd_sims")}
    missing = [k for k in BD_PARAMS if k not in cfg and k not in OPTIONAL]
    if missing:
        sys.exit(f"  ERROR: config.xml has no {', '.join(missing)}")
    vals.update({k: cfg.get(k, OPTIONAL.get(k)) for k in BD_PARAMS})
    for k, v in vals.items():
        txt = re.sub(rf"<{k}>\s*</{k}>", f"<{k}>{v}</{k}>", txt)
    with open(path, "w") as fh:
        fh.write(txt)
    left = [t for t in re.findall(r"<([a-z0-9_]+)>\s*</\1>", txt) if t != "debye_length"]
    if left:
        sys.exit(f"  ERROR: input.xml still has empty tags {left}")

def build_from_config(config_path, work_dir_override=None):
    """Read config.xml, build all simulation inputs in <work_dir>, and return (input_xml_path, cfg_dict)."""
    cpath = os.path.abspath(config_path)
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"Configuration file not found: {cpath}")
    cfg = read_config(cpath)
    source_dir = os.path.dirname(cpath)
    raw_bd = work_dir_override if work_dir_override is not None else cfg.get("bd_work_dir", "bd_sims")
    W = str(raw_bd) if os.path.isabs(str(raw_bd)) else os.path.join(source_dir, str(raw_bd))
    os.makedirs(W, exist_ok=True)
    run_cfg = dict(cfg, source_dir=source_dir, work_dir=W, bd_work_dir=W, resolve_paths_in_w=True)
    build(**run_cfg)
    return os.path.join(W, "input.xml"), cfg

def geometry_check(rec, lig, cfg):
    R = np.array([a["xyz"] for a in rec])
    L = np.array([a["xyz"] for a in lig])
    rmax = float(np.max(np.linalg.norm(R - R.mean(0), axis=1)))
    lmax = float(np.max(np.linalg.norm(L - L.mean(0), axis=1)))
    b = float(cfg["bd_milestone_radius"])
    fg = float(cfg["apbs_fglen"])
    if fg <= 0:
        # The fine edge the run derives for apbs_fglen 0 (run_apbs._compute_grid_params).
        pad = max(a["r"] for a in rec) + float(cfg["srad"])
        fg = max((float(np.max(R.max(0) - R.min(0))) + 2 * pad) * 1.05, 4 * pad * 1.025)
    ok = b + lmax < fg / 2
    print(f"  rmax={rmax:.1f} lmax={lmax:.1f} b={b:.0f} fglen={fg:.0f}  "
          f"{'OK' if ok else 'WARNING, b-surface outside the grid'}")

STRUCTURE = ("ligand_source", "receptor_pqr_file", "ligand_pqr_file",
             "prmtop", "complex_pdb", "pdb_id",
             "ligand_resid", "smiles", "chain", "ligand_chain", "rec_resid_min",
             "ligand_net_charge", "ligand_ccd", "ligand_protonate",
             "ligand_deprotonate", "keep_hetatm")
NAMES = ("receptor_resname", "ligand_resname", "ligand_formula")
CRITERION = ("mode", "search_cutoff", "n_pairs", "atom_filter", "excluded",
             "receptor_numbering", "receptor", "ligand",
             "pocket_ligand", "pocket_ligand_resid", "pair_cutoff",
             "margin", "n_needed")

def _el(t, v, ind):
    if v in (None, "", [], ()):
        return f"{ind}<{t}></{t}>"
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

def _section(order, src, ind, w, full_template=True):
    out = []
    for t in order:
        if not full_template and t not in src:
            continue
        val = src.get(t, "")
        first, _, rest = _el(t, val, ind).partition("\n")
        if t in KEYS:
            first = f"{first:<{w}}<!-- {_doc(t)} -->"
        out.append(first + ("\n" + rest if rest else ""))
    return "\n".join(out)

SECTIONS = (
    ("structure", STRUCTURE), ("names", NAMES), ("criterion", CRITERION),
    ("geometry", GEOMETRY), ("electrostatics", ELECTROSTATICS),
    ("physics", PHYSICS), ("simulation", SIMULATION), ("compute", COMPUTE),
    ("outputs", OUTPUTS), ("chain", CHAIN),
)

def config_xml(cfg, full_template=True):
    c = cfg.get("criterion", {})
    cc = cfg.get("chain_cfg", {})
    comp = dict(cfg)
    if "bd_work_dir" in comp and "work_dir" not in comp:
        comp["work_dir"] = comp["bd_work_dir"]
    def _src_for(name):
        if name == "criterion":
            return c
        if name == "chain":
            return cc
        return comp
    secs = [(order, _src_for(name)) for name, order in SECTIONS]
    w = max((len(_el(t, src.get(t, ""), "    ").partition("\n")[0])
             for order, src in secs for t in order
             if full_template or t in src), default=0) + 2
    body = "".join(
        f"  <{name}>\n{_section(order, src, '    ', w, full_template=full_template)}\n  </{name}>\n"
        for (name, order), (_, src) in zip(SECTIONS, secs))
    return f'<?xml version="1.0" ?>\n<config>\n{body}</config>\n'

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--template" in argv:
        sys.stdout.write(config_xml({}, full_template=True))
        return
    given = [x for x in argv if not x.startswith("-")]
    path = os.path.abspath(given[0] if given else "config.xml")
    if not os.path.exists(path):
        sys.exit(f"  ERROR: no {path}")
    try:
        cfg = read_config(path)
    except ValueError as exc:
        sys.exit(f"  ERROR: {exc}")
    cfg.setdefault("work_dir", os.path.dirname(path))
    build(**cfg)

if __name__ == "__main__":
    main()
