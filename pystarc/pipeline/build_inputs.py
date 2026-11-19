"""
Builds the input files of a PySTARC run from a config.xml: receptor.pqr,
ligand.pqr, rxns.xml (or chain.json and reaction_pairs.json), input.xml and
build_record.json. Receptor and ligand structures are passed both as .pdb or
both as .pqr files, with solute model rigid or chain. The reaction criterion
modes are contacts and centroid.

    python run_pystarc.py config.xml --setup
    python run_pystarc.py --template
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
from pystarc.structures.pqr_io import parse_pqr_records


MODELS = {
    "rigid": dict(need=("receptor", "ligand", "ligand_net_charge"), opt=(), note="Rigid-body Brownian dynamics"),
    "chain": dict(need=("receptor", "ligand", "ligand_net_charge"), opt=(), note="Flexible COFFDROP bead chain"),
}

MODES = {
    "contacts": dict(need=("pair_cutoff", "n_needed"), opt=("receptor", "ligand"), note="Explicit atom pairs named in the config"),
    "centroid": dict(need=("pair_cutoff", "n_needed"), opt=(), note="Pairs the centroid of the receptor with the centroid of the ligand"),
}

KEYS = {
    "receptor":                  "Prepared receptor structure (.pdb or .pqr, matching ligand format).",
    "ligand":                    "Prepared ligand structure (.pdb or .pqr, matching receptor format).",
    "ligand_net_charge":         "Formal integer charge of the ligand (in elementary charge units).",
    "model":                     "Solute representation.",
    "mode":                      "How the reaction criterion is defined.",
    "pair_cutoff":               "Reaction distance for every pair (in angstroms).",
    "n_needed":                  "Mode contacts pairs inside cutoff at once (1 for one pair, half of an even pair count, or positive count for model chain); 0 in mode centroid (rxns.xml writes 1).",
    "bd_milestone_radius":       "Radius of the starting sphere (in angstroms).",
    "r_escape":                  "Escape sphere radius (in angstroms). 0 derives twice bd_milestone_radius.",
    "n_trajectories":            "Number of Brownian dynamics trajectories.",
    "apbs_fglen":                "Fine electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_dime":                 "Grid points per axis of the electrostatic grid.",
    "apbs_cglen":                "Coarse electrostatic grid length (in angstroms). 0 derives it.",
    "apbs_coarse_dime":          "Grid points per axis of the coarse grid. 0 copies apbs_dime.",
    "apbs_fine_dime":            "Grid points per axis of the fine grid. 0 copies apbs_dime.",
    "debye_length":              "Electrostatic screening length of the salt (in angstroms). Blank derives it from ion_concentration.",
    "ion_concentration":         "Salt concentration (in molar).",
    "ion_radius_pos":            "Radius of the positive ion (in angstroms).",
    "ion_radius_neg":            "Radius of the negative ion (in angstroms).",
    "pdie":                      "Dielectric constant of the protein.",
    "sdie":                      "Dielectric constant of the solvent.",
    "srad":                      "Solvent probe radius (in angstroms).",
    "desolvation_alpha":         "Scaling applied to the desolvation penalty.",
    "enable_born2_torque":       "Add the desolvation of the receptor charges in the ligand Born field, a force and a torque on the ligand.",
    "hydrodynamic_interactions": "Include hydrodynamic coupling between the two molecules.",
    "multipole_fallback":        "Extend the electrostatics beyond the grid by multipole expansion.",
    "lj_forces":                 "Apply repulsive forces at close contact.",
    "overlap_check":             "Forbid the ligand from entering the receptor volume.",
    "soft_repulsion":            "Soft excluded volume of BrownDye 2 between receptor and ligand atoms in place of the overlap check.",
    "gpu_force_batch":           "Trajectories per GPU force evaluation. 0 selects automatically.",
    "max_steps":                 "Step limit per trajectory.",
    "dt":                        "Outer time step of the flexible chain model (in picoseconds). No effect on rigid body runs.",
    "minimum_core_dt":           "Smallest permitted time step. 0 imposes no floor.",
    "minimum_core_reaction_dt":  "Smallest time step near the reaction surface (in picoseconds). 0 imposes no floor on the GPU.",
    "max_dt":                    "Ceiling on the time step. 0 imposes no ceiling.",
    "r_hydro_rec":               "Receptor hydrodynamic radius. 0 derives it from the PQR.",
    "r_hydro_lig":               "Ligand hydrodynamic radius. 0 derives it from the PQR file.",
    "temperature":               "Temperature (in kelvin).",
    "seed":                      "Random number seed.",
    "gpu":                       "Evaluate forces on the GPU.",
    "n_gpus":                    "Number of GPUs for parallel trajectory splitting. 1 runs a single worker.",
    "n_threads":                 "Number of CPU threads.",
    "work_dir":                  "Output directory for grids, generated inputs and trajectory data.",
    "convergence_check":         "Run convergence analysis after the simulation.",
    "convergence_interval":      "Reporting interval for the running rate.",
    "convergence_tol":           "Relative error defining convergence.",
    "checkpoint_interval":       "Trajectories between checkpoints. 0 disables checkpointing.",
    "save_interval":             "Steps between saved positions. 0 saves endpoints.",
    "full_paths":                "Save positions and orientations every save_interval steps.",
    "energetics":                "Save the force on the ligand and the time step every save_interval steps.",
    "reaction_pairs_json":       "Pre-built chain reaction pairs JSON file for model chain. Blank builds from criterion pairs.",
    "dt_chain":                  "Inner time step of flexible chain internal coordinates (in picoseconds).",
    "chain_steps_per_outer":     "Inner chain steps per outer step.",
    "n_equilibration_steps":     "Pre-equilibration chain steps before each trajectory. 0 skips pre-equilibration.",
    "auto_diffusion":            "Compute chain diffusion tensors from RPY bead geometry.",
    "d_trans":                   "Explicit chain translational diffusion when auto_diffusion is false (in A^2/ps).",
    "d_rot":                     "Explicit chain rotational diffusion when auto_diffusion is false (in rad^2/ps).",
    "soft_repulsion_eps":        "Chain soft repulsion strength (in units of kT).",
    "target_grid_dx":            "Optional receptor electrostatic DX grid path for model chain. Blank builds in work_dir.",
    "born_grid_dx":              "Optional receptor Born DX grid path for model chain. Blank builds in work_dir.",
}

CRITERION_KEYS = {
    "receptor": "Receptor atom of each contact pair (<target resid=\"...\" atom=\"...\" resname=\"...\"/>).",
    "ligand":   "Ligand atom of each contact pair (<atom name=\"...\"/> or <target resid=\"...\" atom=\"...\" resname=\"...\"/>).",
}

STRUCTURE = ("receptor", "ligand", "ligand_net_charge", "model")
CRITERION = ("mode", "receptor", "ligand", "pair_cutoff", "n_needed")
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
    "reaction_pairs_json", "dt_chain", "chain_steps_per_outer",
    "n_equilibration_steps", "auto_diffusion", "d_trans", "d_rot",
    "soft_repulsion_eps", "target_grid_dx", "born_grid_dx",
)
INPUT_COMPUTE = ("gpu", "gpu_force_batch", "n_threads")
BD_PARAMS = GEOMETRY + ELECTROSTATICS + PHYSICS + SIMULATION + INPUT_COMPUTE + OUTPUTS

OPTIONAL = {"soft_repulsion": "false", "minimum_core_reaction_dt": "0",
            "full_paths": "true", "energetics": "true", "r_escape": "0",
            "debye_length": ""}

BOOL = ("enable_born2_torque", "hydrodynamic_interactions", "multipole_fallback",
        "lj_forces", "overlap_check", "soft_repulsion", "gpu", "convergence_check",
        "full_paths", "energetics", "auto_diffusion")

RECORD = "build_record.json"

RETAINED_METALS = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CAL"}
SKIP_SOLVENT = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K+"}
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "CYM", "GLN", "GLU", "GLH",
    "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME",
}
TLEAP_VARIANT_GROUPS = [
    {"HIS", "HIE", "HID", "HIP"},
    {"CYS", "CYX", "CYM"},
    {"ASP", "ASH"},
    {"GLU", "GLH"},
    {"LYS", "LYN"},
]

JUNK = ["_protein.rst7", "_ligand.rst7", "ligand_h.sdf", "sqm.in", "sqm.out",
        "sqm.pdb", "ANTECHAMBER_AC.AC", "ANTECHAMBER_AC.AC0",
        "ANTECHAMBER_BOND_TYPE.AC", "ANTECHAMBER_BOND_TYPE.AC0",
        "ANTECHAMBER_AM1BCC.AC", "ANTECHAMBER_AM1BCC_PRE.AC", "ATOMTYPE.INF",
        "ligand_obabel.mol2"]

LEFTOVER = ["tleap_protein.in", "tleap_ligand.in", "tleap_protein.log",
            "tleap_ligand.log", "leap.log", "ligand.frcmod", "ligand.mol2",
            "ligand_raw.pdb", "protein.pdb", "protein_leap.pdb",
            "ligand_leap.pdb", "protein.prmtop", "protein.rst7",
            "ligand.prmtop", "ligand.rst7", "_receptor_raw.pqr"]


def tidy(W, _cfg=None):
    for f in LEFTOVER:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)


NUMBER = {"ligand_net_charge": int, "pair_cutoff": float, "n_needed": int, "n_gpus": int}


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


SECTIONS = (
    ("structure", STRUCTURE), ("criterion", CRITERION),
    ("geometry", GEOMETRY), ("electrostatics", ELECTROSTATICS),
    ("physics", PHYSICS), ("simulation", SIMULATION), ("compute", COMPUTE),
    ("outputs", OUTPUTS), ("chain", CHAIN),
)


def read_config(path):
    root = ET.parse(path).getroot()
    if root.tag != "config":
        raise ValueError(f"Expected <config> root in {path}, got <{root.tag}>")
    sec_map = dict(SECTIONS)
    tag_homes = {}
    for sname, order in SECTIONS:
        for t in order:
            tag_homes.setdefault(t, []).append(sname)
    all_tags = list(tag_homes)
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
                if el.tag in tag_homes:
                    homes = " or ".join(f"<{h}>" for h in tag_homes[el.tag])
                    raise ValueError(f"<{el.tag}> in <{sec.tag}>: place it inside {homes}.")
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
    p = os.path.join(W, produces)
    _gone(p)
    r = subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(p):
        made = "" if os.path.exists(p) else f" and produced no {produces}"
        tail = (r.stderr or r.stdout).strip()[-600:]
        sys.exit(f"  ERROR: {cmd.split()[0]} ended with exit {r.returncode}{made}\n{tail}")
    return r


def _leap(W, script, body, produces):
    stem = os.path.join(W, produces.rsplit(".", 1)[0])
    _gone(stem + ".prmtop", stem + ".rst7")
    with open(os.path.join(W, script), "w") as fh:
        fh.write(body)
    r = subprocess.run(f"tleap -f {script}", shell=True, cwd=W, capture_output=True, text=True)
    with open(os.path.join(W, script.replace(".in", ".log")), "w") as fh:
        fh.write(r.stdout + r.stderr)
    p = os.path.join(W, produces)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        sys.exit(f"  ERROR: tleap produced no {produces}, see {script[:-3]}.log")


def _pqr(W, stem, out):
    rst = os.path.join(W, f"_{stem}.rst7")
    _gone(rst)
    subprocess.run(
        "cpptraj",
        input=f"parm {stem}.prmtop\ntrajin {stem}.rst7\ntrajout _{stem}.rst7\nrun\n",
        cwd=W, capture_output=True, text=True,
    )
    if not os.path.exists(rst):
        sys.exit(f"  ERROR: cpptraj produced no _{stem}.rst7 from {stem}.prmtop")
    r = subprocess.run(f"ambpdb -p {stem}.prmtop -c _{stem}.rst7 -pqr",
                       shell=True, cwd=W, capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l.startswith("ATOM")]
    if not lines:
        sys.exit(f"  ERROR: ambpdb produced nothing for {stem}")
    with open(os.path.join(W, out), "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    q = sum(float(l.split()[8]) for l in lines)
    print(f"  {out}: {len(lines)} atoms, Q={q:+.1f} e")


def _restore_names(W, mol):
    path = os.path.join(W, "ligand.mol2")
    with open(path) as fh:
        lines = fh.read().split("\n")
    i0 = lines.index("@<TRIPOS>ATOM") + 1
    i1 = next(i for i in range(i0, len(lines)) if lines[i].startswith("@<TRIPOS>"))
    rows = [i for i in range(i0, i1) if lines[i].strip()]
    if len(rows) != mol.GetNumAtoms():
        sys.exit(f"  ERROR: ligand.mol2 has {len(rows)} atoms, the ligand has {mol.GetNumAtoms()}")
    given = {}
    for i, a in zip(rows, mol.GetAtoms()):
        f = lines[i].split()
        el = {"cl": "Cl", "br": "Br"}.get(f[5][:2].lower(), f[5][0].upper())
        if el != a.GetSymbol():
            sys.exit(f"  ERROR: atom {f[0]} of ligand.mol2 is {el} where the ligand has {a.GetSymbol()}")
        info = a.GetPDBResidueInfo()
        if info is not None:
            given[i] = info.GetName().strip()
    taken = set(given.values())
    if len(taken) != len(given):
        sys.exit("  ERROR: two ligand atoms in the input PDB share a name")
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
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _check_single_conformer(lines, path):
    seen = {}
    for l in lines:
        if len(l) > 26 and l[16] != " ":
            seen.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), set()).add(l[16])
    multi = [f"{k[2]} {k[0].strip()}{k[1]} ({''.join(sorted(v))})" for k, v in seen.items() if len(v) > 1]
    if multi:
        sys.exit(f"  ERROR: {path} contains multiple altloc conformers for {', '.join(multi)}. "
                 f"Select a single conformer before running PySTARC")


def _ext(path):
    e = os.path.splitext(str(path))[1].lower()
    if e not in (".pdb", ".pqr"):
        sys.exit(f"  ERROR: structure file {path!r} must end with .pdb or .pqr")
    return e


def _copy_clean_pqr(src, dst, label):
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit(f"  ERROR: {label} must not overwrite {os.path.basename(dst)} in place")
    with open(src) as fh:
        lines = fh.read().split("\n")
    ghost = lambda l: (
        l.startswith(("ATOM", "HETATM"))
        and (l[12:16].strip().upper() in ("GHO", "PKT")
             or (len(l.split()) >= 3 and l.split()[2].upper() in ("GHO", "PKT")))
    )
    n_gho = sum(1 for l in lines if ghost(l))
    kept = [l for l in lines if not ghost(l)]
    with open(dst, "w") as fh:
        fh.write("\n".join(kept))
    n = sum(1 for l in kept if l.startswith(("ATOM", "HETATM")))
    print(f"  {os.path.basename(dst)}: {n} atoms from {label}"
          + (f", {n_gho} ghost atom removed" if n_gho else ""))


def _parameterize_protein_pdb(pdb_src, label, W, stem, out_pqr):
    with open(pdb_src) as fh:
        raw_lines = fh.readlines()
    _check_single_conformer(raw_lines, label)
    prot, metals, bad_het = [], [], set()
    for line in raw_lines:
        if line.startswith("ATOM"):
            a = line[12:16].strip()
            if a == "OXT" or a[0] == "H" or (len(a) > 1 and a[0].isdigit() and a[1] == "H"):
                continue
            prot.append(line)
        elif line.startswith("HETATM"):
            resn = line[17:20].strip()
            if resn in RETAINED_METALS:
                metals.append(line)
            elif resn not in SKIP_SOLVENT:
                bad_het.add(resn)
    if bad_het:
        sys.exit(f"  ERROR: {label} contains non-metal HETATM {sorted(bad_het)}. "
                 f"Place the diffusing ligand in <ligand> and strip extra heteroatoms from <receptor>")
    if not prot and not metals:
        sys.exit(f"  ERROR: no ATOM records in {label}")
    stage_pdb = f"{stem}.pdb"
    body = "".join(prot) + ("TER\n" if prot else "") + "".join(f"{m}TER\n" for m in metals) + "END\n"
    with open(os.path.join(W, stage_pdb), "w") as fh:
        fh.write(body)
    print(f"  {stem}: {len(prot)} protein atoms, {len(metals)} metal atoms")
    _leap(W, f"tleap_{stem}.in",
          "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
          "loadAmberParams frcmod.ions234lm_126_tip3p\n"
          f"mol = loadpdb {stage_pdb}\ncheck mol\n"
          f"saveamberparm mol {stem}.prmtop {stem}.rst7\n"
          f"savepdb mol {stem}_leap.pdb\nquit\n", f"{stem}.prmtop")
    _pqr(W, stem, out_pqr)


def _is_protein_chain_pdb(pdb_src):
    resnames = set()
    has_atom = False
    with open(pdb_src) as fh:
        for l in fh:
            if l.startswith("ATOM"):
                has_atom = True
                resnames.add(l[17:20].strip())
            elif l.startswith("HETATM"):
                rn = l[17:20].strip()
                if rn not in RETAINED_METALS and rn not in SKIP_SOLVENT:
                    return False
    return has_atom and resnames <= PROTEIN_RESIDUES


def _parameterize_small_mol_pdb(lig_src, label, netq, W):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    with open(lig_src) as fh:
        raw_lines = [l for l in fh if l.startswith(("ATOM", "HETATM"))]
    if not raw_lines:
        sys.exit(f"  ERROR: no ATOM/HETATM records in {label}")
    _check_single_conformer(raw_lines, label)
    raw = os.path.join(W, "ligand_raw.pdb")
    with open(raw, "w") as fh:
        fh.write("".join(raw_lines) + "END\n")
    resn_pdb = raw_lines[0][17:20].strip() or "LIG"
    m = Chem.MolFromPDBFile(raw, removeHs=False, sanitize=False)
    if m is None:
        sys.exit(f"  ERROR: RDKit cannot read {label}")
    has_h = any(a.GetAtomicNum() == 1 for a in m.GetAtoms())
    if not has_h:
        sys.exit(f"  ERROR: ligand {resn_pdb} has no hydrogens in {label}. "
                 f"Add explicit hydrogens to the ligand PDB before running PySTARC")
    try:
        rdDetermineBonds.DetermineBondOrders(m, charge=netq)
    except Exception as e:
        sys.exit(f"  ERROR: RDKit finds no bond orders for net charge {netq:+d} "
                 f"({e}). Check ligand_net_charge and the hydrogens of {label}")
    for a in m.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(m)
    except Exception as e:
        sys.exit(f"  ERROR: ligand chemistry does not sanitize ({e}). "
                 f"Check ligand_net_charge and the hydrogens of {label}")
    m = Chem.AddHs(m, addCoords=True)
    print(f"  ligand net charge {netq:+d} e, {m.GetNumAtoms()} atoms with hydrogens")
    Chem.MolToMolFile(m, os.path.join(W, "ligand_h.sdf"))
    _sh(W, "obabel ligand_h.sdf -o mol2 -O ligand_obabel.mol2", "ligand_obabel.mol2")
    _sh(W, f"antechamber -i ligand_obabel.mol2 -fi mol2 -o ligand.mol2 -fo mol2 "
           f"-c bcc -at gaff2 -nc {netq} -rn {resn_pdb[:3]} -pf y -j 4 -dr no",
        "ligand.mol2")
    _restore_names(W, m)
    _sh(W, "parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2", "ligand.frcmod")
    _leap(W, "tleap_ligand.in",
          "source leaprc.gaff2\nloadamberparams ligand.frcmod\n"
          "lig = loadmol2 ligand.mol2\ncheck lig\n"
          "saveamberparm lig ligand.prmtop ligand.rst7\n"
          "savepdb lig ligand_leap.pdb\nquit\n", "ligand.prmtop")
    _pqr(W, "ligand", "ligand.pqr")
    for f in JUNK:
        p = os.path.join(W, f)
        if os.path.exists(p):
            os.remove(p)


def _resname_match(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in TLEAP_VARIANT_GROUPS)


def _is_h_atom(aname, elem=""):
    el = (elem or "").strip().upper()
    if el == "H":
        return True
    return aname.startswith("H") or (len(aname) > 1 and aname[0].isdigit() and aname[1] == "H")


def _parse_pdb_heavy(path):
    atoms = []
    chains = set()
    with open(path) as fh:
        for l in fh:
            if not l.startswith("ATOM") or len(l) < 54:
                continue
            if l[16] not in (" ", "A"):
                continue
            aname = l[12:16].strip()
            el = l[76:78].strip() if len(l) >= 78 else ""
            if _is_h_atom(aname, el):
                continue
            chains.add(l[21])
            atoms.append({
                "atom_name": aname,
                "resname": l[17:20].strip(),
                "resid": int(l[22:26]),
                "xyz": np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]),
            })
    return atoms, sorted(chains)


def _seq_residues(atoms):
    out, cur = [], None
    for a in atoms:
        if cur != a["resid"]:
            cur = a["resid"]
            out.append({"seq_pos": len(out), "resname": a["resname"], "resid_native": cur, "atoms": []})
        out[-1]["atoms"].append(a)
    return out


def _center_pqr_copy(src, dst):
    with open(src) as fh:
        lines = fh.read().splitlines()
    hits = [i for i, l in enumerate(lines) if l.startswith("ATOM") and len(l) >= 54]
    if not hits:
        sys.exit(f"  ERROR: no fixed-column ATOM records in {src}")
    xyz = np.array([[float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])] for i in hits])
    shift = xyz.mean(axis=0)
    for k, i in enumerate(hits):
        v = xyz[k] - shift
        lines[i] = f"{lines[i][:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{lines[i][54:]}"
    with open(dst, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _check_net_charge(q_sum, netq, label):
    if not isinstance(netq, int):
        sys.exit(f"  ERROR: ligand_net_charge must be an integer, got {netq!r}")
    if abs(float(q_sum) - int(netq)) > 0.05:
        sys.exit(f"  ERROR: ligand_net_charge is {int(netq):+d} e, where {label} sums to {float(q_sum):+.2f} e")


def _build_chain(cfg, rec_src, rec_ext, lig_src, _lig_ext, W):
    from pystarc.simulation.coffdrop_chain import chain_from_pdb
    from pystarc.structures.chain_io import save_chain_to_json

    netq = int(cfg["ligand_net_charge"])

    dst_pqr = os.path.join(W, "receptor.pqr")
    rec_rmap = None
    if rec_ext == ".pqr":
        _center_pqr_copy(rec_src, dst_pqr)
    else:
        raw_pqr_path = os.path.join(W, "_receptor_raw.pqr")
        _parameterize_protein_pdb(rec_src, cfg["receptor"], W, "protein", "_receptor_raw.pqr")
        _center_pqr_copy(raw_pqr_path, dst_pqr)
        rec_rmap = resid_map(rec_src)

    lig_pdb, lig_chains = _parse_pdb_heavy(lig_src)
    if not lig_pdb:
        sys.exit(f"  ERROR: no heavy ATOM records in {cfg['ligand']}")
    if len(lig_chains) > 1:
        sys.exit(f"  ERROR: {cfg['ligand']} contains multiple chains {lig_chains}. Keep one chain.")
    lig_chain_id = lig_chains[0] if lig_chains and lig_chains[0].strip() else None
    lig_name = os.path.splitext(os.path.basename(lig_src))[0]
    lig_res_pdb = _seq_residues(lig_pdb)

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    coffdrop_dir = os.path.join(pkg_dir, "coffdrop_data")
    chain = chain_from_pdb(lig_src, chain_id=lig_chain_id, name=lig_name,
                           sidechains=True, coffdrop_dir=coffdrop_dir)
    _check_net_charge(sum(float(b.charge) for b in chain.atoms), netq, cfg["ligand"])
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
    save_chain_to_json(chain, body_positions, os.path.join(W, "chain.json"))

    c = cfg["criterion"]
    cc = cfg.get("chain_cfg", {})
    rp_out = os.path.join(W, "reaction_pairs.json")
    if cc.get("reaction_pairs_json"):
        rp_src = _src_file(cfg, W, cc["reaction_pairs_json"])
        if not os.path.exists(rp_src):
            sys.exit(f"  ERROR: no {cc['reaction_pairs_json']}")
        with open(rp_src) as fh:
            rp_list = json.load(fh)
        if os.path.abspath(rp_src) != os.path.abspath(rp_out):
            shutil.copyfile(rp_src, rp_out)
    else:
        rec_pqr = read_pqr(dst_pqr)
        lig_sp = {r["resid_native"]: r["seq_pos"] for r in lig_res_pdb}
        beads_by_sp = {}
        for i, bead in enumerate(chain.atoms):
            beads_by_sp.setdefault(resid_to_sp[bead.resid], []).append((i, bead))
        rp_list = []
        for rt, lt in zip(_list(c["receptor"]), _list(c["ligand"])):
            rec_i = _match_atom(rec_pqr, rt, rec_rmap, "receptor")
            if not (isinstance(lt, tuple) and len(lt) == 3):
                sys.exit('  ERROR: chain <ligand> needs <target resid="..." atom="..." resname="..."/> entries')
            l_resid, l_aname, l_resn = lt
            sp_l = lig_sp.get(l_resid)
            cands = beads_by_sp.get(sp_l, []) if sp_l is not None else []
            if not cands:
                sys.exit(f"  ERROR: no residue {l_resn}{l_resid} in {cfg['ligand']}")
            tname = f"{l_resn}:{l_aname}"
            bead_i = next((i for i, b in cands if b.resname == tname), cands[0][0])
            rp_list.append([int(rec_i), int(bead_i), float(c["pair_cutoff"])])
        with open(rp_out, "w") as fh:
            json.dump(rp_list, fh, indent=2)

    if len(rp_list) < int(c["n_needed"]):
        sys.exit(f"  ERROR: {len(rp_list)} chain reaction pairs found (minimum {c['n_needed']}).")
    print(f"  chain.json: {chain.n_atoms} beads; reaction_pairs.json: {len(rp_list)} pairs, n_needed {c['n_needed']}")
    return {"chain_pairs": rp_list, "n_beads": chain.n_atoms}


def _build_rigid(cfg, rec_src, rec_ext, lig_src, lig_ext, W):
    rec_rmap, lig_rmap = None, None
    if rec_ext == ".pqr":
        _copy_clean_pqr(rec_src, os.path.join(W, "receptor.pqr"), cfg["receptor"])
    else:
        _parameterize_protein_pdb(rec_src, cfg["receptor"], W, "protein", "receptor.pqr")
        rec_rmap = resid_map(rec_src)

    netq = int(cfg["ligand_net_charge"])
    if lig_ext == ".pqr":
        _copy_clean_pqr(lig_src, os.path.join(W, "ligand.pqr"), cfg["ligand"])
    elif _is_protein_chain_pdb(lig_src):
        _parameterize_protein_pdb(lig_src, cfg["ligand"], W, "ligand", "ligand.pqr")
        lig_rmap = resid_map(lig_src)
    else:
        _parameterize_small_mol_pdb(lig_src, cfg["ligand"], netq, W)
    lig_q = sum(a["q"] for a in read_pqr(os.path.join(W, "ligand.pqr")))
    _check_net_charge(lig_q, netq, "ligand.pqr")
    return {"rec_rmap": rec_rmap, "lig_rmap": lig_rmap}


def _element(name, column="", resname=""):
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


def _name(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def pair_count_ok(n, need):
    return (n == 1 and need == 1) or (n >= 2 and n % 2 == 0 and need == n // 2)


def check_criterion(c, model="rigid", has_chain_rp_json=False):
    _check(c.get("mode"), MODES, c, "criterion mode")
    mode = c["mode"]
    if not isinstance(c.get("pair_cutoff"), (int, float)) or c["pair_cutoff"] <= 0:
        sys.exit("  ERROR: pair_cutoff must be a distance above zero (in angstroms).")
    if mode == "contacts":
        if not isinstance(c.get("n_needed"), int) or c["n_needed"] < 1:
            sys.exit("  ERROR: mode 'contacts' needs integer n_needed >= 1.")
        if model == "chain" and has_chain_rp_json:
            return mode
        missing = [k for k in ("receptor", "ligand") if c.get(k) in (None, "", [], ())]
        if missing:
            sys.exit(f"  ERROR: mode 'contacts' needs explicit {missing} atom entries.")
        R, L = _list(c["receptor"]), _list(c["ligand"])
        if len(R) != len(L):
            sys.exit(f"  ERROR: {len(R)} receptor targets against {len(L)} ligand entries.")
        if model != "chain" and not pair_count_ok(len(R), c["n_needed"]):
            sys.exit(f"  ERROR: {len(R)} pairs with n_needed {c['n_needed']} are not allowed. Choose one pair with "
                     "n_needed 1, or an even number of pairs with n_needed half of that count.")
    elif model == "chain":
        sys.exit("  ERROR: model 'chain' requires criterion mode 'contacts'.")
    else:
        if not isinstance(c.get("n_needed"), int) or c["n_needed"] != 0:
            sys.exit("  ERROR: mode 'centroid' requires integer n_needed 0 (ignored, rxns.xml writes 1).")
    return mode


def resid_map(pdb_path):
    prot_seen, metal_seen = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                r = int(line[22:26])
                if r not in prot_seen:
                    prot_seen.append(r)
            elif line.startswith("HETATM") and line[17:20].strip() in RETAINED_METALS:
                r = int(line[22:26])
                if r not in metal_seen:
                    metal_seen.append(r)
    ordered = prot_seen + [r for r in metal_seen if r not in prot_seen]
    return {r: i + 1 for i, r in enumerate(ordered)}


def _match_atom(atoms, target, rmap, side_label):
    if not (isinstance(target, tuple) and len(target) == 3):
        sys.exit(f'  ERROR: {side_label} needs <target resid="..." atom="..." resname="..."/> entries')
    resid, aname, resname = target
    q_resid = rmap.get(resid, resid) if rmap else resid
    i = next((k for k, a in enumerate(atoms)
              if a["resid"] == q_resid and a["name"] == aname and _resname_match(a["resname"], resname)), None)
    if i is None and rmap:
        i = next((k for k, a in enumerate(atoms)
                  if a["resid"] == resid and a["name"] == aname and _resname_match(a["resname"], resname)), None)
    if i is None:
        sys.exit(f"  ERROR: no {resname}{resid} {aname} in the {side_label}")
    return i


def ligand_atom(lig, target, lig_rmap=None):
    if isinstance(target, tuple) and len(target) == 3:
        return _match_atom(lig, target, lig_rmap, "ligand")
    name = _name(target)
    js = [k for k, b in enumerate(lig) if b["name"] == name]
    if len(js) != 1:
        sys.exit(f"  ERROR: {'no' if not js else 'more than one'} ligand atom {name} in ligand.pqr")
    return js[0]


def contacts_listed(rec, lig, c, rec_rmap=None, lig_rmap=None):
    R, L = _list(c["receptor"]), _list(c["ligand"])
    if len(R) != len(L):
        sys.exit(f"  ERROR: {len(R)} receptor targets against {len(L)} ligand entries")
    out = []
    for r_target, l_target in zip(R, L):
        i = _match_atom(rec, r_target, rec_rmap, "receptor")
        j = ligand_atom(lig, l_target, lig_rmap)
        d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
        out.append((i, j, c["pair_cutoff"], d))
    return out


def centroid_pair(rec, lig, c):
    R = np.mean([a["xyz"] for a in rec], axis=0)
    L = np.mean([a["xyz"] for a in lig], axis=0)
    return (len(rec), len(lig), c["pair_cutoff"], float(np.linalg.norm(R - L)))


def write_rxns(path, pairs, n_needed):
    body = "".join(
        f"        <pair>\n          <atoms> {i + 1} {j + 1} </atoms>\n"
        f"          <distance> {float(cut)!s} </distance>\n        </pair>\n"
        for i, j, cut, _d in pairs)
    with open(path, "w") as fh:
        fh.write(
            "<top>\n  <first_state> start </first_state>\n  <reactions>\n    <reaction>\n"
            "      <name> association </name>\n      <state_before> start </state_before>\n"
            "      <state_after> end </state_after>\n      <criterion>\n        <molecules>\n"
            "          <molecule0> rec rec </molecule0>\n          <molecule1> lig lig </molecule1>\n"
            f"        </molecules>\n        <n_needed> {n_needed} </n_needed>\n{body}"
            "      </criterion>\n    </reaction>\n  </reactions>\n</top>\n"
        )


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


TOP_LEVEL = ({"model", "criterion", "work_dir", "bd_work_dir",
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
        sys.exit(f"{what} {choice!r} needs {missing}, found blank or absent")
    extra = set(given) - set(spec["need"]) - set(spec["opt"]) - {"mode"} - ignore
    if extra:
        sys.exit(f"{what} {choice!r} does not accept {sorted(extra)}, remove it")


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
    model = cfg.get("model", "rigid")
    _check(model, MODELS, cfg, "model", ignore=TOP_LEVEL)
    c = cfg.get("criterion", {})
    cc = cfg.get("chain_cfg", {})
    mode = check_criterion(c, model, has_chain_rp_json=bool(cc.get("reaction_pairs_json")))
    if model != "chain" and cc:
        sys.exit(f"  ERROR: <chain> settings {sorted(cc)} apply to model 'chain'.")
    W = cfg["work_dir"]
    os.makedirs(W, exist_ok=True)
    rec_src = _src_file(cfg, W, cfg["receptor"])
    lig_src = _src_file(cfg, W, cfg["ligand"])
    for raw, full in ((cfg["receptor"], rec_src), (cfg["ligand"], lig_src)):
        if not os.path.exists(full):
            sys.exit(f"  ERROR: no {raw}")
    rec_ext, lig_ext = _ext(cfg["receptor"]), _ext(cfg["ligand"])
    if rec_ext != lig_ext:
        sys.exit(f"  ERROR: <receptor> ({rec_ext}) and <ligand> ({lig_ext}) must both be .pdb or both be .pqr")

    if model == "chain":
        extra = _build_chain(cfg, rec_src, rec_ext, lig_src, lig_ext, W)
        write_chain_input(os.path.join(W, "input.xml"), cfg, W)
        record = dict(
            mode=mode,
            model=model,
            receptor=cfg["receptor"],
            ligand=cfg["ligand"],
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

    extra = _build_rigid(cfg, rec_src, rec_ext, lig_src, lig_ext, W)
    rec = read_pqr(os.path.join(W, "receptor.pqr"))
    lig = read_pqr(os.path.join(W, "ligand.pqr"))
    record = dict(mode=mode, model=model, receptor=cfg["receptor"], ligand=cfg["ligand"])
    if mode == "contacts":
        pairs = contacts_listed(rec, lig, c, extra.get("rec_rmap"), extra.get("lig_rmap"))
        n_needed = c["n_needed"]
        if not pair_count_ok(len(pairs), n_needed):
            sys.exit(f"  ERROR: {len(pairs)} pairs were found for n_needed {n_needed}.")
    else:
        pairs, n_needed = [centroid_pair(rec, lig, c)], 1
    write_rxns(os.path.join(W, "rxns.xml"), pairs, n_needed)
    write_input(os.path.join(W, "input.xml"), cfg)
    record.update(
        pairs=[[int(i) + 1, int(j) + 1, float(cut)] for i, j, cut, _d in pairs],
        n_needed=n_needed,
        code=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        md5={f: _md5(os.path.join(W, f)) for f in ("receptor.pqr", "ligand.pqr", "rxns.xml", "input.xml")},
    )
    with open(os.path.join(W, RECORD), "w") as fh:
        json.dump(record, fh, indent=1, default=float)
    print(f"  {len(pairs)} pairs, n_needed {n_needed}")
    geometry_check(rec, lig, cfg)
    tidy(W, cfg)


def find_templates():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")


def write_input(path, cfg):
    with open(os.path.join(find_templates(), "input.xml")) as fh:
        txt = fh.read()
    use_abs = bool(cfg.get("resolve_paths_in_w"))
    W = cfg["work_dir"]
    vals = {
        "receptor_pqr": os.path.join(W, "receptor.pqr") if use_abs else "receptor.pqr",
        "ligand_pqr": os.path.join(W, "ligand.pqr") if use_abs else "ligand.pqr",
        "rxns_xml": os.path.join(W, "rxns.xml") if use_abs else "rxns.xml",
        "work_dir": cfg.get("bd_work_dir", "bd_sims"),
    }
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
        pad = max(a["r"] for a in rec) + float(cfg["srad"])
        fg = max((float(np.max(R.max(0) - R.min(0))) + 2 * pad) * 1.05, 4 * pad * 1.025)
    ok = b + lmax < fg / 2
    print(f"  rmax={rmax:.1f} lmax={lmax:.1f} b={b:.0f} fglen={fg:.0f}  "
          f"{'OK' if ok else 'WARNING, b-surface outside the grid'}")


def _el(sec_name, t, v, ind):
    if v in (None, "", [], ()):
        return f"{ind}<{t}></{t}>"
    if sec_name == "criterion" and t == "receptor":
        rows = "".join(f'{ind}  <target resid="{r}" atom="{a}" resname="{n}"/>\n'
                       for r, a, n in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    if sec_name == "criterion" and t == "ligand" and isinstance(v, (list, tuple)):
        rows = "".join(
            f'{ind}  <target resid="{x[0]}" atom="{x[1]}" resname="{x[2]}"/>\n'
            if isinstance(x, tuple) and len(x) == 3
            else f'{ind}  <atom name="{_name(x)}"/>\n'
            for x in v
        )
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    if isinstance(v, (list, tuple)):
        rows = "".join(f"{ind}  <item>{_name(x)}</item>\n" for x in v)
        return f"{ind}<{t}>\n{rows}{ind}</{t}>"
    return f"{ind}<{t}>{v}</{t}>"


def _listed(names):
    n = list(names)
    return " and ".join(n) if len(n) < 3 else ", ".join(n[:-1]) + ", and " + n[-1]


def _doc(sec_name, t):
    if sec_name == "criterion" and t in CRITERION_KEYS:
        return CRITERION_KEYS[t]
    table = {"model": MODELS, "mode": MODES}.get(t)
    base = KEYS[t]
    if table:
        return f"{base} Options: {_listed(table)}."
    if t in BOOL:
        return f"{base} Options: {_listed(('true', 'false'))}."
    return base


def _section(sec_name, order, src, ind, w, full_template=True):
    out = []
    for t in order:
        if not full_template and t not in src:
            continue
        val = src.get(t, "")
        first, _, rest = _el(sec_name, t, v=val, ind=ind).partition("\n")
        if t in KEYS or (sec_name == "criterion" and t in CRITERION_KEYS):
            first = f"{first:<{w}}<!-- {_doc(sec_name, t)} -->"
        out.append(first + ("\n" + rest if rest else ""))
    return "\n".join(out)


def config_xml(cfg, full_template=True):
    crit = cfg.get("criterion", {})
    chain_src = cfg.get("chain_cfg", {})
    top = dict(cfg)
    if "bd_work_dir" in top and "work_dir" not in top:
        top["work_dir"] = top["bd_work_dir"]
    lines = []
    for sname, order in SECTIONS:
        src = crit if sname == "criterion" else (chain_src if sname == "chain" else top)
        for t in order:
            if not full_template and t not in src:
                continue
            first, _, _ = _el(sname, t, src.get(t, ""), "    ").partition("\n")
            lines.append(first)
    w = max([len(s) for s in lines] + [40]) + 2
    blocks = ['<?xml version="1.0" ?>', "<config>"]
    for sname, order in SECTIONS:
        src = crit if sname == "criterion" else (chain_src if sname == "chain" else top)
        body = _section(sname, order, src, "    ", w, full_template=full_template)
        if full_template or body:
            blocks.append(f"  <{sname}>\n{body}\n  </{sname}>")
    blocks.append("</config>\n")
    return "\n".join(blocks)


def write_config_xml(path, cfg):
    with open(path, "w") as fh:
        fh.write(config_xml(cfg, full_template=True))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--template"]:
        sys.stdout.write(config_xml({}, full_template=True))
    elif len(args) == 1 and os.path.exists(args[0]):
        build_from_config(args[0])
    else:
        sys.exit("Usage: python -m pystarc.pipeline.build_inputs <config.xml> | --template")
