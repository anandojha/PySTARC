#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import numpy as np
import json
import sys
import os
import shutil
import glob
import re

def read_config(path):
    cfg = {}
    for sec in ET.parse(path).getroot():
        for el in sec:
            cfg[el.tag] = (el.text or "").strip()
    return cfg

_here = os.path.dirname(os.path.abspath(__file__))
_given = [a for a in sys.argv[1:] if not a.startswith("-")]
CONFIG_PATH = os.path.abspath(_given[0] if _given else "config.xml")
if not os.path.exists(CONFIG_PATH):
    sys.exit(f"  ERROR: no {CONFIG_PATH}")
C = read_config(CONFIG_PATH)
_missing = [k for k in ['auto_diffusion', 'bd_milestone_radius', 'born_grid_dx', 'chain_json', 'chain_name', 'chain_steps_per_outer', 'convergence_check', 'convergence_interval', 'convergence_tol', 'd_rot', 'd_trans', 'desolvation_alpha', 'dt', 'dt_chain', 'gpu', 'input_xml', 'interface_cutoff_a', 'ligand_chain_id', 'max_steps', 'min_reaction_pairs', 'n_equilibration_steps', 'n_trajectories', 'n_workers', 'r_escape', 'reaction_distance_a', 'reaction_n_needed', 'reaction_pairs_json', 'receptor_chain_id', 'receptor_pqr', 'apbs_fglen', 'apbs_dime', 'save_interval', 'seed', 'soft_repulsion_eps', 'source_pdb', 'target_grid_dx', 'temperature', 'use_soft_repulsion', 'work_dir'] if k not in C]
if _missing:
    sys.exit(f"  ERROR: config.xml has no {', '.join(_missing)}")

SOURCE_PDB                = C["source_pdb"]
RECEPTOR_CHAIN_ID         = C["receptor_chain_id"]
LIGAND_CHAIN_ID           = C["ligand_chain_id"]
RECEPTOR_PQR              = C["receptor_pqr"]
APBS_FGLEN                = C["apbs_fglen"]
APBS_DIME                 = C["apbs_dime"]
TARGET_GRID_DX            = C["target_grid_dx"]
BORN_GRID_DX              = C["born_grid_dx"]
CHAIN_JSON                = C["chain_json"]
REACTION_PAIRS_JSON       = C["reaction_pairs_json"]
INPUT_XML                 = C["input_xml"]
CHAIN_NAME                = C["chain_name"]
INTERFACE_CUTOFF_A        = float(C["interface_cutoff_a"])
REACTION_DISTANCE_A       = float(C["reaction_distance_a"])
MIN_REACTION_PAIRS        = int(C["min_reaction_pairs"])
TEMPERATURE               = C["temperature"]
SEED                      = C["seed"]
BD_MILESTONE_RADIUS       = C["bd_milestone_radius"]
DT                        = C["dt"]
DT_CHAIN                  = C["dt_chain"]
CHAIN_STEPS_PER_OUTER     = C["chain_steps_per_outer"]
N_EQUILIBRATION_STEPS     = C["n_equilibration_steps"]
R_ESCAPE                  = C["r_escape"]
DESOLVATION_ALPHA         = C["desolvation_alpha"]
N_TRAJECTORIES            = C["n_trajectories"]
MAX_STEPS                 = C["max_steps"]
REACTION_N_NEEDED         = C["reaction_n_needed"]
N_WORKERS                 = C["n_workers"]
WORK_DIR                  = C["work_dir"]
_root = _here
while _root != "/" and not os.path.isfile(os.path.join(_root, "run_pystarc.py")):
    _root = os.path.dirname(_root)
PYSTARC_DIR               = _root                 # the PySTARC checkout, found by walking up

def parse_pdb_chain(pdb_path, chain_id, heavy_only=True):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            atom_name = line[12:16].strip()
            if heavy_only and atom_name.startswith("H"):
                continue
            resname = line[17:20].strip()
            resid = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append({
                "atom_name": atom_name,
                "resname": resname,
                "resid": resid,
                "xyz": np.array([x, y, z]),
            })
    return atoms

def parse_pqr(pqr_path):
    atoms = []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            atom_name = parts[2]
            resname = parts[3]
            resid = int(parts[4])
            x = float(parts[5])
            y = float(parts[6])
            z = float(parts[7])
            atoms.append({
                "atom_idx": len(atoms),
                "atom_name": atom_name,
                "resname": resname,
                "resid": resid,
                "xyz": np.array([x, y, z]),
            })
    return atoms

def center_pqr(src):
    lines = open(src).read().splitlines()
    hits = [i for i, l in enumerate(lines)
            if l[:6].strip() in ("ATOM", "HETATM") and len(l) >= 54]
    if not hits:
        sys.exit(f"  ERROR: no fixed column ATOM records in {src}")
    try:
        xyz = np.array([[float(lines[i][30:38]), float(lines[i][38:46]),
                         float(lines[i][46:54])] for i in hits])
    except ValueError:
        sys.exit(f"  ERROR: {src} does not use the fixed PDB coordinate columns")
    shift = xyz.mean(axis=0)
    for k, i in enumerate(hits):
        v = xyz[k] - shift
        if np.any(np.abs(v) >= 10000.0):
            sys.exit(f"  ERROR: coordinate overflows the 8 column field in {src}")
        lines[i] = f"{lines[i][:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{lines[i][54:]}"
    with open(src, "w") as f:
        f.write("\n".join(lines) + "\n")
    return shift, len(hits)

def build_sequence_position_index(atoms):
    residues = []
    last_resid = None
    last_seq_pos = -1
    for a in atoms:
        if a["resid"] != last_resid:
            last_resid = a["resid"]
            last_seq_pos += 1
            residues.append({
                "seq_pos": last_seq_pos,
                "resname": a["resname"],
                "resid_native": a["resid"],
                "atoms": [],
            })
        residues[-1]["atoms"].append(a)
    return residues


TLEAP_VARIANT_GROUPS = [
    {"HIS", "HIE", "HID", "HIP"},
    {"CYS", "CYX", "CYM"},
    {"ASP", "ASH"},
    {"GLU", "GLH"},
    {"LYS", "LYN"},
]

def resname_match(r1, r2):
    if r1 == r2:
        return True
    for group in TLEAP_VARIANT_GROUPS:
        if r1 in group and r2 in group:
            return True
    return False


def find_atom_in_residue(residue, atom_name):
    for a in residue["atoms"]:
        if a["atom_name"] == atom_name:
            return a
    return None

def parse_coffdrop_map(map_xml_path):
    """Parse COFFDROP map.xml: {resname: {bead_name: [atom_names]}}."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(map_xml_path)
    root = tree.getroot()
    mapping = {}
    for res_elem in root.findall("residue"):
        resname = res_elem.find("name").text.strip()
        beads = {}
        for bead_elem in res_elem.findall("bead"):
            bead_name = bead_elem.find("name").text.strip()
            atoms_text = bead_elem.find("atoms").text.strip()
            beads[bead_name] = atoms_text.split()
        mapping[resname] = beads
    return mapping

# Step 1: Parse sources and verify consistency
print("=" * 70)
print("PySTARC chain BD setup: barnase-barstar (WT)")
print("=" * 70)

for f in [SOURCE_PDB, RECEPTOR_PQR]:
    if not os.path.exists(f):
        sys.exit(f"ERROR: prerequisite file missing: {f}")
    sz = os.path.getsize(f)
    print(f"  OK  {f}  ({sz} bytes)")

print("\nParsing 1BRS chain A (barnase) and D (barstar)...")
barnase_1brs = parse_pdb_chain(SOURCE_PDB, RECEPTOR_CHAIN_ID, heavy_only=True)
barstar_1brs = parse_pdb_chain(SOURCE_PDB, LIGAND_CHAIN_ID, heavy_only=True)
print(f"  1BRS chain {RECEPTOR_CHAIN_ID}: {len(barnase_1brs)} heavy atoms")
print(f"  1BRS chain {LIGAND_CHAIN_ID}: {len(barstar_1brs)} heavy atoms")

barnase_residues_1brs = build_sequence_position_index(barnase_1brs)
barstar_residues_1brs = build_sequence_position_index(barstar_1brs)
print(f"  Barnase residues: {len(barnase_residues_1brs)} "
      f"(resid {barnase_residues_1brs[0]['resid_native']}-{barnase_residues_1brs[-1]['resid_native']})")
print(f"  Barstar residues: {len(barstar_residues_1brs)} "
      f"(resid {barstar_residues_1brs[0]['resid_native']}-{barstar_residues_1brs[-1]['resid_native']})")

_shift, _n = center_pqr(RECEPTOR_PQR)
print(f"\nCentering {RECEPTOR_PQR}")
print(f"  {_n} atoms translated by {-_shift[0]:.3f} {-_shift[1]:.3f} "
      f"{-_shift[2]:.3f} A, centroid now at the origin")

print("\nParsing barnase.pqr...")
barnase_pqr = parse_pqr(RECEPTOR_PQR)
barnase_residues_pqr = build_sequence_position_index(barnase_pqr)
print(f"  barnase.pqr: {len(barnase_pqr)} atoms, {len(barnase_residues_pqr)} residues "
      f"(resid {barnase_residues_pqr[0]['resid_native']}-{barnase_residues_pqr[-1]['resid_native']})")

print("\nVerifying residue-by-residue match (1BRS chain A vs barnase.pqr)...")
if len(barnase_residues_1brs) != len(barnase_residues_pqr):
    sys.exit(f"ERROR: residue count mismatch: 1BRS={len(barnase_residues_1brs)}, "
             f"pqr={len(barnase_residues_pqr)}")
mismatches = 0
for k, (r1, r2) in enumerate(zip(barnase_residues_1brs, barnase_residues_pqr)):
    if not resname_match(r1["resname"], r2["resname"]):
        print(f"  MISMATCH at seq_pos {k}: 1BRS={r1['resname']} vs pqr={r2['resname']}")
        mismatches += 1
if mismatches:
    sys.exit(f"ERROR: {mismatches} resname mismatches")
offset = barnase_residues_1brs[0]['resid_native'] - barnase_residues_pqr[0]['resid_native']
print(f"  OK  108 residues match. Numbering offset (1BRS - pqr) = {offset}")

# Step 2: Build chain from barstar
print("\nBuilding COFFDROP chain from barstar (1BRS chain D)...")
sys.path.insert(0, PYSTARC_DIR)
try:
    from pystarc.simulation.coffdrop_chain import chain_from_pdb
    from pystarc.structures.chain_io import save_chain_to_json
except ImportError as e:
    sys.exit(f"ERROR: PySTARC import failed: {e}")

coffdrop_dir = os.path.join(PYSTARC_DIR, "pystarc", "coffdrop_data")
chain = chain_from_pdb(SOURCE_PDB, chain_id=LIGAND_CHAIN_ID, name=CHAIN_NAME,
                       sidechains=True, coffdrop_dir=coffdrop_dir)
print(f"  chain.n_atoms  = {chain.n_atoms}")
print(f"  chain.bonds    = {len(chain.bonds)}")
print(f"  chain.angles   = {len(chain.angles)}")
print(f"  chain.torsions = {len(chain.torsions)}")

unique_resids_in_chain = []
for bead in chain.atoms:
    if not unique_resids_in_chain or unique_resids_in_chain[-1] != bead.resid:
        unique_resids_in_chain.append(bead.resid)
chain_resid_to_seq_pos = {r: k for k, r in enumerate(unique_resids_in_chain)}
print(f"  chain unique residues = {len(unique_resids_in_chain)} "
      f"(resid range {unique_resids_in_chain[0]}-{unique_resids_in_chain[-1]})")
if len(unique_resids_in_chain) != len(barstar_residues_1brs):
    sys.exit(f"ERROR: chain residue count ({len(unique_resids_in_chain)}) != "
             f"barstar 1BRS residue count ({len(barstar_residues_1brs)})")

# Step 3: Derive body-frame positions from 1BRS chain D
print("\nDeriving body-frame positions via COFFDROP centroid mapping...")
coffdrop_map_xml = os.path.join(PYSTARC_DIR, "pystarc", "coffdrop_data", "map.xml")
bead_mapping = parse_coffdrop_map(coffdrop_map_xml)
print(f"  Loaded COFFDROP map: {len(bead_mapping)} residue types")

def resolve_map_resname(rname):
    if rname in bead_mapping:
        return rname
    for group in TLEAP_VARIANT_GROUPS:
        if rname in group:
            for cand in group:
                if cand in bead_mapping:
                    return cand
    return None

body_positions = np.zeros((chain.n_atoms, 3))
missing = []
n_centroid = 0
n_fallback = 0
for i, bead in enumerate(chain.atoms):
    parts = bead.resname.split(":")
    if len(parts) != 2:
        sys.exit(f"ERROR: unexpected bead resname format: {bead.resname!r}")
    bead_resname, bead_atom_name = parts
    seq_pos = chain_resid_to_seq_pos[bead.resid]
    res = barstar_residues_1brs[seq_pos]
    if not resname_match(res["resname"], bead_resname):
        sys.exit(f"ERROR: bead {i} resname {bead_resname} != 1BRS seq_pos {seq_pos} resname {res['resname']}")
    map_resname = resolve_map_resname(bead_resname)
    if map_resname is None:
        missing.append((i, bead.resname, seq_pos, f"resname {bead_resname} not in map.xml"))
        continue
    if bead_atom_name not in bead_mapping[map_resname]:
        missing.append((i, bead.resname, seq_pos, f"bead {bead_atom_name} not in map[{map_resname}]"))
        continue
    atom_names = bead_mapping[map_resname][bead_atom_name]
    coords = []
    for an in atom_names:
        atom = find_atom_in_residue(res, an)
        if atom is not None:
            coords.append(atom["xyz"])
    if not coords:
        # Fallback: use residue CA if available
        ca_atom = find_atom_in_residue(res, "CA")
        if ca_atom is not None:
            body_positions[i] = ca_atom["xyz"]
            present = [a["atom_name"] for a in res["atoms"]]
            print(f"  fallback CA for bead {i} {bead.resname} (seq_pos {seq_pos}): "
                  f"expected {atom_names}, present {present}")
            n_fallback += 1
            continue
        missing.append((i, bead.resname, seq_pos, f"none of {atom_names} found, no CA fallback"))
        continue
    body_positions[i] = np.mean(coords, axis=0)
    if len(coords) > 1:
        n_centroid += 1

print(f"  {chain.n_atoms - len(missing)}/{chain.n_atoms} beads mapped "
      f"({n_centroid} via centroid of >=2 atoms)")
if missing:
    print(f"  WARNING: {len(missing)} beads could not be mapped:")
    for i, name, sp, reason in missing[:10]:
        print(f"    bead {i} {name} (seq_pos {sp}): {reason}")
    sys.exit("ERROR: aborting due to unmapped beads")

if not np.isfinite(body_positions).all():
    sys.exit("ERROR: non-finite values in body_positions")
print(f"  All {chain.n_atoms} beads mapped to 1BRS chain D atoms")
print(f"  Position range: "
      f"x[{body_positions[:,0].min():.2f},{body_positions[:,0].max():.2f}] "
      f"y[{body_positions[:,1].min():.2f},{body_positions[:,1].max():.2f}] "
      f"z[{body_positions[:,2].min():.2f},{body_positions[:,2].max():.2f}]")

save_chain_to_json(chain, body_positions, CHAIN_JSON)
print(f"  Wrote {CHAIN_JSON}")

# Step 4: Native interface contacts to reaction_pairs.json
print(f"\nFinding native interface contacts (cutoff {INTERFACE_CUTOFF_A:.1f} A)...")
chain_a_xyz = np.array([a["xyz"] for a in barnase_1brs])
chain_d_xyz = np.array([a["xyz"] for a in barstar_1brs])
dists = np.linalg.norm(chain_a_xyz[:, None, :] - chain_d_xyz[None, :, :], axis=-1)
contact_indices = np.argwhere(dists < INTERFACE_CUTOFF_A)
print(f"  {len(contact_indices)} atom-atom contacts within {INTERFACE_CUTOFF_A:.1f} A")

barnase_1brs_resid_to_seq_pos = {r["resid_native"]: r["seq_pos"] for r in barnase_residues_1brs}
barstar_1brs_resid_to_seq_pos = {r["resid_native"]: r["seq_pos"] for r in barstar_residues_1brs}

def barnase_1brs_to_pqr_idx(atom):
    seq_pos = barnase_1brs_resid_to_seq_pos.get(atom["resid"])
    if seq_pos is None or seq_pos >= len(barnase_residues_pqr):
        return None
    for a in barnase_residues_pqr[seq_pos]["atoms"]:
        if a["atom_name"] == atom["atom_name"]:
            return a["atom_idx"]
    return None

beads_by_seq_pos = {}
for i, bead in enumerate(chain.atoms):
    sp = chain_resid_to_seq_pos[bead.resid]
    beads_by_seq_pos.setdefault(sp, []).append((i, bead))

def barstar_1brs_to_chain_bead_idx(atom):
    seq_pos = barstar_1brs_resid_to_seq_pos.get(atom["resid"])
    if seq_pos is None:
        return None
    candidate_beads = beads_by_seq_pos.get(seq_pos, [])
    if not candidate_beads:
        return None
    target_name = f"{atom['resname']}:{atom['atom_name']}"
    for i, bead in candidate_beads:
        if bead.resname == target_name:
            return i
    target_xyz = atom["xyz"]
    best_i, best_d = None, float("inf")
    for i, bead in candidate_beads:
        d = np.linalg.norm(body_positions[i] - target_xyz)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

pairs = set()
unmappable = 0
for i_a, i_d in contact_indices:
    barnase_atom = barnase_1brs[int(i_a)]
    barstar_atom = barstar_1brs[int(i_d)]
    rec_idx = barnase_1brs_to_pqr_idx(barnase_atom)
    bead_idx = barstar_1brs_to_chain_bead_idx(barstar_atom)
    if rec_idx is None or bead_idx is None:
        unmappable += 1
        continue
    pairs.add((rec_idx, bead_idx))

print(f"  {len(pairs)} unique (receptor_atom, chain_bead) pairs")
if unmappable:
    print(f"  {unmappable} contacts unmappable (skipped)")
if len(pairs) < MIN_REACTION_PAIRS:
    sys.exit(f"ERROR: only {len(pairs)} reaction pairs found "
             f"(minimum: {MIN_REACTION_PAIRS}). Increase INTERFACE_CUTOFF_A.")

reaction_pairs_list = [[int(rec), int(bd), REACTION_DISTANCE_A]
                       for rec, bd in sorted(pairs)]
with open(REACTION_PAIRS_JSON, "w") as f:
    json.dump(reaction_pairs_list, f, indent=2)
print(f"  Wrote {REACTION_PAIRS_JSON} ({len(pairs)} pairs, n_needed={REACTION_N_NEEDED})")

# Step 5: Write input.xml
DIFFUSION = "" if C["auto_diffusion"].lower() == "true" else (
    f"\n    <D_trans>{C['d_trans']}</D_trans>"
    f"\n    <D_rot>{C['d_rot']}</D_rot>"
)

input_xml = f"""<?xml version="1.0"?>
<pystarc>
  <receptor_pqr>{RECEPTOR_PQR}</receptor_pqr>
  <apbs_fglen>{APBS_FGLEN}</apbs_fglen>
  <apbs_dime>{APBS_DIME}</apbs_dime>
  <bd_milestone_radius>{BD_MILESTONE_RADIUS}</bd_milestone_radius>
  <n_trajectories>{N_TRAJECTORIES}</n_trajectories>
  <max_steps>{MAX_STEPS}</max_steps>
  <dt>{DT}</dt>
  <temperature>{TEMPERATURE}</temperature>
  <seed>{SEED}</seed>
  <work_dir>{WORK_DIR}</work_dir>
  <gpu>{C['gpu']}</gpu>
  <desolvation_alpha>{DESOLVATION_ALPHA}</desolvation_alpha>
  <save_interval>{C['save_interval']}</save_interval>
  <convergence_check>{C['convergence_check']}</convergence_check>
  <convergence_interval>{C['convergence_interval']}</convergence_interval>
  <convergence_tol>{C['convergence_tol']}</convergence_tol>
  <chain>
    <chain_json>{CHAIN_JSON}</chain_json>
    <reaction_pairs_json>{REACTION_PAIRS_JSON}</reaction_pairs_json>
    <target_grid_dx>{TARGET_GRID_DX}</target_grid_dx>
    <born_grid_dx>{BORN_GRID_DX}</born_grid_dx>
    <r_escape>{R_ESCAPE}</r_escape>
    <reaction_n_needed>{REACTION_N_NEEDED}</reaction_n_needed>
    <auto_diffusion>{C['auto_diffusion']}</auto_diffusion>{DIFFUSION}
    <use_soft_repulsion>{C['use_soft_repulsion']}</use_soft_repulsion>
    <soft_repulsion_eps>{C['soft_repulsion_eps']}</soft_repulsion_eps>
    <n_workers>{N_WORKERS}</n_workers>
    <dt_chain>{DT_CHAIN}</dt_chain>
    <chain_steps_per_outer>{CHAIN_STEPS_PER_OUTER}</chain_steps_per_outer>
    <n_equilibration_steps>{N_EQUILIBRATION_STEPS}</n_equilibration_steps>
  </chain>
</pystarc>
"""

with open(INPUT_XML, "w") as f:
    f.write(input_xml)

def tidy():
    stale = ["barnase_centered.pqr", "barnase_centred.pqr",
             ".ipynb_checkpoints", "__pycache__"]
    stale += glob.glob("apbs_output/*.pqr") + glob.glob("apbs_output/*.in")
    stale += glob.glob("apbs_output/tmp") + glob.glob("apbs_output/io.mc")
    gone = []
    for p in stale:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            gone.append(p)
        elif os.path.exists(p):
            os.remove(p)
            gone.append(p)
    if os.path.isdir(WORK_DIR) and not os.listdir(WORK_DIR):
        os.rmdir(WORK_DIR)
        gone.append(WORK_DIR)
    return gone


_gone = tidy()
if _gone:
    print(f"\n  removed {len(_gone)} stale: {', '.join(sorted(_gone))}")

print("\n" + "=" * 70)
print("Setup complete.")
print(f"  {CHAIN_JSON}            {chain.n_atoms} beads, body positions from 1BRS chain D")
print(f"  {REACTION_PAIRS_JSON}   {len(pairs)} pairs, cutoff {REACTION_DISTANCE_A:.1f} A, n_needed {REACTION_N_NEEDED}")
print(f"  {INPUT_XML}")
print(f"\n  N_TRAJECTORIES = {N_TRAJECTORIES}")
print(f"  MAX_STEPS      = {MAX_STEPS}")
print(f"  N_WORKERS      = {N_WORKERS}")
print(f"  WORK_DIR       = {WORK_DIR}")
print("=" * 70)