#!/usr/bin/env python3
"""Standalone reaction-criterion builder for PySTARC config.xml.

Inspects prepared receptor and ligand structures (.pdb + .pdb or .pqr + .pqr)
before filling config.xml and prints a ready-to-paste <criterion> block (plus
reaction_pairs.json when --chain-pairs-out is requested). Independent of
pystarc, run_pystarc.py, and config.xml.

Supported workflows:
  1. Bound-pose contact pairs (polar or heavy-atom):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --atom-filter polar --n-pairs 6 --pair-cutoff 7.0
  2. Named active-site receptor targets (with explicit or auto-picked ligand atoms):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --receptor-targets 199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE --ligand-atoms N1 --pair-cutoff 7.0
  3. Centroid reaction criterion:
     python scripts/find_reaction_criteria.py --receptor receptor.pqr --ligand ligand.pqr --mode centroid --pair-cutoff 8.0
  4. Flexible-chain COFFDROP reaction_pairs.json (model chain):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --search-cutoff 6.0 --pair-cutoff 8.0 --n-needed 3 --chain-pairs-out reaction_pairs.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np


POLAR_ELEMENTS = {"N", "O", "S"}
RETAINED_METALS = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CAL"}
SKIP_SOLVENT = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K+"}
VARIANT_GROUPS = [
    {"HIS", "HIE", "HID", "HIP"},
    {"CYS", "CYX", "CYM"},
    {"ASP", "ASH"},
    {"GLU", "GLH"},
    {"LYS", "LYN"},
]

ATOM_FILTERS = {
    "all":            "Matches any heavy atom against any heavy atom",
    "polar":          "Matches N O or S on both sides",
    "nonpolar":       "Matches carbon on both sides",
    "any_polar":      "Matches N O or S on either side",
    "receptor_polar": "Matches N O or S on the receptor side",
}


def keeper(kind):
    return {
        "all": lambda a, b: True,
        "polar": lambda a, b: a in POLAR_ELEMENTS and b in POLAR_ELEMENTS,
        "nonpolar": lambda a, b: a == "C" and b == "C",
        "any_polar": lambda a, b: a in POLAR_ELEMENTS or b in POLAR_ELEMENTS,
        "receptor_polar": lambda a, b: a in POLAR_ELEMENTS,
    }[kind]


def _resname_match(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in VARIANT_GROUPS)


def _is_h_atom(aname, elem=""):
    el = (elem or "").strip().upper()
    if el == "H":
        return True
    return aname.startswith("H") or (len(aname) > 1 and aname[0].isdigit() and aname[1] == "H")


def _infer_element(aname, col_el="", resname=""):
    ce = (col_el or "").strip().upper()
    if ce.isalpha() and len(ce) <= 2:
        return ce
    ru = (resname or "").strip().upper()
    nm = aname.lstrip("0123456789").upper()
    if ru in ("CAL", "CA") and nm.startswith("CA"):
        return "CA"
    if ru in ("ZN", "MG", "FE", "CU") and nm.startswith(ru):
        return ru
    return nm[:2] if nm[:2] in ("CL", "BR") else nm[:1]


def _parse_pqr_records_standalone(path):
    records = []
    with open(path) as fh:
        for l in fh:
            if not l.startswith(("ATOM", "HETATM")):
                continue
            toks = l.split()
            if len(toks) < 9:
                continue
            aname = l[12:16].strip() if len(l) >= 54 else toks[2]
            if aname.upper() in ("GHO", "PKT"):
                continue
            if len(l) >= 54 and l[30:38].strip() and l[38:46].strip() and l[46:54].strip():
                try:
                    resname = l[17:20].strip()
                    resid = int(l[22:26])
                    x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
                    tail = l[54:].split()
                    col_el = tail[2] if len(tail) >= 3 and tail[2].isalpha() else ""
                    records.append((l[:6].strip(), aname, resname, resid, x, y, z, col_el))
                    continue
                except ValueError:
                    pass
            # Whitespace fallback (with or without chain column)
            if len(toks) >= 10 and not toks[4].lstrip("+-").isdigit():
                resname, resid = toks[3], int(toks[5])
                x, y, z = float(toks[6]), float(toks[7]), float(toks[8])
            else:
                resname, resid = toks[3], int(toks[4])
                x, y, z = float(toks[5]), float(toks[6]), float(toks[7])
            records.append((toks[0], toks[2], resname, resid, x, y, z, ""))
    return records


def parse_structure_atoms(path):
    """Parse ATOM/HETATM records from a .pdb or .pqr file into dicts with name, el, resname, resid, xyz."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found")
    atoms = []
    if path.lower().endswith(".pqr"):
        for _rt, aname, resname, resid, x, y, z, col_el in _parse_pqr_records_standalone(path):
            el = _infer_element(aname, col_el, resname)
            atoms.append(dict(name=aname, el=el, resname=resname, resid=resid,
                              xyz=np.array([x, y, z], dtype=float)))
        return atoms
    with open(path) as fh:
        for l in fh:
            if not l.startswith(("ATOM", "HETATM")) or len(l) < 54:
                continue
            if l[16] not in (" ", "A"):
                continue
            aname = l[12:16].strip()
            if aname.upper() in ("GHO", "PKT"):
                continue
            resname = l[17:20].strip()
            resid = int(l[22:26])
            col_el = l[76:78].strip() if len(l) >= 78 else ""
            el = _infer_element(aname, col_el, resname)
            atoms.append(dict(name=aname, el=el, resname=resname, resid=resid,
                              xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])], dtype=float)))
    return atoms


def contacts_picked(rec, lig, c):
    keep = keeper(c["atom_filter"])
    raw_excl = c.get("excluded") or ()
    skip = {(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_excl}
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


def contacts_for_targets(rec, lig, target_specs, lig_atom_specs, atom_filter, pair_cutoff):
    keep = keeper(atom_filter)
    r_list = []
    for spec in target_specs:
        parts = [p.strip() for p in spec.split(":") if p.strip()]
        if len(parts) != 3:
            sys.exit(f"ERROR: --receptor-targets entry {spec!r} must be RESID:ATOM:RESNAME")
        resid, aname, resn = int(parts[0]), parts[1], parts[2].upper()
        hit = next((i for i, a in enumerate(rec)
                    if a["resid"] == resid and a["name"] == aname and _resname_match(a["resname"], resn)), None)
        if hit is None:
            sys.exit(f"ERROR: receptor target {resn}{resid} {aname} absent from receptor structure")
        r_list.append(hit)

    if lig_atom_specs:
        if len(lig_atom_specs) == 1 and len(r_list) > 1:
            lig_atom_specs = lig_atom_specs * len(r_list)
        if len(lig_atom_specs) != len(r_list):
            sys.exit(f"ERROR: {len(r_list)} --receptor-targets against {len(lig_atom_specs)} --ligand-atoms")
        out = []
        for i, lspec in zip(r_list, lig_atom_specs):
            parts = [p.strip() for p in lspec.split(":") if p.strip()]
            if len(parts) == 3:
                lrid, lan, lrn = int(parts[0]), parts[1], parts[2].upper()
                j = next((k for k, b in enumerate(lig)
                          if b["resid"] == lrid and b["name"] == lan and _resname_match(b["resname"], lrn)), None)
            else:
                j = next((k for k, b in enumerate(lig) if b["name"] == lspec.strip()), None)
            if j is None:
                sys.exit(f"ERROR: ligand atom {lspec!r} absent from ligand structure")
            d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
            out.append((i, j, pair_cutoff, d))
        return out

    out = []
    for i in r_list:
        a = rec[i]
        cands = [(float(np.linalg.norm(a["xyz"] - b["xyz"])), j)
                 for j, b in enumerate(lig) if b["el"] != "H" and keep(a["el"], b["el"])]
        if not cands:
            sys.exit(f"ERROR: no ligand atom permitted by --atom-filter {atom_filter!r} for {a['resname']}{a['resid']}")
        d, j = min(cands)
        out.append((i, j, pair_cutoff, d))
    return out


def _seq_residues(atoms):
    out, cur = [], None
    for a in atoms:
        if cur != a["resid"]:
            cur = a["resid"]
            out.append({"seq_pos": len(out), "resname": a["resname"], "resid_native": cur, "atoms": []})
        out[-1]["atoms"].append(a)
    return out


def _parse_pdb_heavy_chain(path):
    atoms, chains = [], set()
    with open(path) as fh:
        for l in fh:
            if not l.startswith("ATOM") or len(l) < 54 or l[16] not in (" ", "A"):
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
                "xyz": np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])], dtype=float),
            })
    return atoms, sorted(chains)


def _parameterize_receptor_pdb_to_pqr(pdb_src, tmp_dir):
    prot, metals = [], []
    with open(pdb_src) as fh:
        for l in fh:
            if l.startswith("ATOM"):
                a = l[12:16].strip()
                if a == "OXT" or _is_h_atom(a):
                    continue
                prot.append(l)
            elif l.startswith("HETATM"):
                rn = l[17:20].strip()
                if rn in RETAINED_METALS:
                    metals.append(l)
                elif rn not in SKIP_SOLVENT:
                    sys.exit(f"ERROR: receptor PDB {pdb_src} holds non-metal HETATM {rn!r}")
    with open(os.path.join(tmp_dir, "protein.pdb"), "w") as fh:
        fh.write("".join(prot) + ("TER\n" if prot else "") + "".join(f"{m}TER\n" for m in metals) + "END\n")
    with open(os.path.join(tmp_dir, "tleap.in"), "w") as fh:
        fh.write(
            "source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n"
            "loadAmberParams frcmod.ions234lm_126_tip3p\n"
            "mol = loadpdb protein.pdb\n"
            "saveamberparm mol protein.prmtop protein.rst7\nquit\n"
        )
    r1 = subprocess.run(["tleap", "-f", "tleap.in"], cwd=tmp_dir, capture_output=True, text=True)
    if r1.returncode != 0 or not os.path.exists(os.path.join(tmp_dir, "protein.prmtop")):
        sys.exit("ERROR: tleap failed while indexing receptor PDB for --chain-pairs-out")
    subprocess.run(
        "cpptraj",
        input="parm protein.prmtop\ntrajin protein.rst7\ntrajout _protein.rst7\nrun\n",
        cwd=tmp_dir, capture_output=True, text=True,
    )
    r2 = subprocess.run(["ambpdb", "-p", "protein.prmtop", "-c", "_protein.rst7", "-pqr"],
                        cwd=tmp_dir, capture_output=True, text=True)
    pqr_path = os.path.join(tmp_dir, "receptor.pqr")
    lines = [l for l in r2.stdout.strip().splitlines() if l.startswith("ATOM")]
    with open(pqr_path, "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    return pqr_path


def _load_coffdrop_map(coffdrop_dir=""):
    if not coffdrop_dir:
        here = os.path.dirname(os.path.abspath(__file__))
        coffdrop_dir = os.path.join(os.path.dirname(here), "pystarc", "coffdrop_data")
    map_xml = os.path.join(coffdrop_dir, "map.xml")
    if not os.path.exists(map_xml):
        sys.exit(f"ERROR: COFFDROP map.xml not found at {map_xml}; pass --coffdrop-dir")
    root = ET.parse(map_xml).getroot()
    bead_atoms = {}
    sidechain_order = {}
    for re_el in root.findall("residue"):
        rname = re_el.findtext("name").strip()
        bdict, sc = {}, []
        for b_el in re_el.findall("bead"):
            bname = b_el.findtext("name").strip()
            atoms = b_el.findtext("atoms").strip().split()
            loc = (b_el.findtext("location") or "").strip()
            btype = (b_el.findtext("type") or "").strip()
            bdict[bname] = atoms
            if btype == "" and loc == "" and bname != "CA":
                sc.append(bname)
        bead_atoms[rname] = bdict
        sidechain_order[rname] = ["CA"] + sc
    return bead_atoms, sidechain_order


def build_chain_reaction_pairs(rec_path, lig_path, search_cutoff, pair_cutoff, n_needed, out_json, coffdrop_dir=""):
    """Build COFFDROP [rec_pqr_atom_idx, bead_idx, pair_cutoff] JSON from receptor + ligand structures."""
    if not lig_path.lower().endswith(".pdb"):
        sys.exit("ERROR: --chain-pairs-out requires a .pdb protein chain in --ligand")
    rec_ext = os.path.splitext(rec_path)[1].lower()
    if rec_ext == ".pqr":
        raw_pqr = [r for r in _parse_pqr_records_standalone(rec_path) if r[0] == "ATOM"]
        rec_heavy = [
            {"atom_idx": i, "atom_name": r[1], "resname": r[2], "resid": r[3],
             "xyz": np.array([r[4], r[5], r[6]], dtype=float)}
            for i, r in enumerate(raw_pqr) if not _is_h_atom(r[1], r[7])
        ]
    else:
        rec_pdb_atoms, rec_chains = _parse_pdb_heavy_chain(rec_path)
        if len(rec_chains) > 1:
            sys.exit(f"ERROR: {rec_path} contains multiple chains {rec_chains}")
        rec_res_pdb = _seq_residues(rec_pdb_atoms)
        with tempfile.TemporaryDirectory() as tmp:
            pqr_path = _parameterize_receptor_pdb_to_pqr(rec_path, tmp)
            pqr_raw = [r for r in _parse_pqr_records_standalone(pqr_path) if r[0] == "ATOM"]
        pqr_atoms = [{"atom_idx": i, "atom_name": r[1], "resname": r[2], "resid": r[3],
                      "xyz": np.array([r[4], r[5], r[6]], dtype=float)}
                     for i, r in enumerate(pqr_raw)]
        rec_res_pqr = _seq_residues(pqr_atoms)
        if len(rec_res_pdb) != len(rec_res_pqr):
            sys.exit(f"ERROR: residue count mismatch: PDB={len(rec_res_pdb)} vs PQR={len(rec_res_pqr)}")
        rec_heavy = []
        for r_pdb, r_pqr in zip(rec_res_pdb, rec_res_pqr):
            for a in r_pdb["atoms"]:
                idx = next((pa["atom_idx"] for pa in r_pqr["atoms"] if pa["atom_name"] == a["atom_name"]), None)
                if idx is not None:
                    rec_heavy.append(dict(a, atom_idx=idx))

    lig_pdb, lig_chains = _parse_pdb_heavy_chain(lig_path)
    if len(lig_chains) > 1:
        sys.exit(f"ERROR: {lig_path} contains multiple chains {lig_chains}")
    lig_res_pdb = sorted(_seq_residues(lig_pdb), key=lambda r: r["resid_native"])
    bead_atoms, sidechain_order = _load_coffdrop_map(coffdrop_dir)

    def _resolve_res(rn):
        if rn in bead_atoms:
            return rn
        for g in VARIANT_GROUPS:
            if rn in g:
                for cand in g:
                    if cand in bead_atoms:
                        return cand
        sys.exit(f"ERROR: residue {rn!r} absent from COFFDROP map.xml")

    beads_by_resid = {}
    body_positions = []
    for res in lig_res_pdb:
        mrn = _resolve_res(res["resname"])
        for bname in sidechain_order[mrn]:
            b_idx = len(body_positions)
            coords = [a["xyz"] for an in bead_atoms[mrn][bname]
                      for a in res["atoms"] if a["atom_name"] == an]
            if not coords:
                coords = [a["xyz"] for a in res["atoms"] if a["atom_name"] == "CA"]
            body_positions.append(np.mean(coords, axis=0))
            beads_by_resid.setdefault(res["resid_native"], []).append((b_idx, f"{mrn}:{bname}"))
    body_positions = np.array(body_positions)

    rec_xyz = np.array([a["xyz"] for a in rec_heavy])
    lig_xyz = np.array([a["xyz"] for a in lig_pdb])
    hits = np.argwhere(np.linalg.norm(rec_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1) < search_cutoff)
    pairs = set()
    for ia, id_ in hits:
        ra, la = rec_heavy[int(ia)], lig_pdb[int(id_)]
        cands = beads_by_resid.get(la["resid"], [])
        tname = f"{_resolve_res(la['resname'])}:{la['atom_name']}"
        bead_idx = next((i for i, b_rn in cands if b_rn == tname), None)
        if bead_idx is None and cands:
            bead_idx = min(cands, key=lambda ib: float(np.linalg.norm(body_positions[ib[0]] - la["xyz"])))[0]
        if bead_idx is not None:
            pairs.add((int(ra["atom_idx"]), int(bead_idx)))
    if len(pairs) < int(n_needed):
        sys.exit(f"ERROR: {len(pairs)} chain reaction pairs found (minimum {n_needed})")
    rp_list = [[rec_i, bd_i, float(pair_cutoff)] for rec_i, bd_i in sorted(pairs)]
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(rp_list, fh, indent=2)
    return rp_list, len(body_positions)


def _csv(val):
    return [x.strip() for x in val.split(",") if x.strip()] if val else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", required=True, help="Prepared receptor .pdb or .pqr file")
    ap.add_argument("--ligand", required=True, help="Prepared ligand .pdb or .pqr file")
    ap.add_argument("--mode", choices=("contacts", "centroid"), default="contacts", help="Criterion mode")
    ap.add_argument("--search-cutoff", type=float, default=6.0, help="Bound-pose contact search cutoff in A")
    ap.add_argument("--n-pairs", type=int, default=6, help="Maximum number of contact pairs")
    ap.add_argument("--atom-filter", choices=sorted(ATOM_FILTERS), default="polar", help="Element filter")
    ap.add_argument("--exclude", default="", help="Comma-separated atom names omitted from contact search")
    ap.add_argument("--receptor-targets", default="", help="Comma-separated RESID:ATOM:RESNAME receptor targets")
    ap.add_argument("--ligand-atoms", default="", help="Comma-separated ligand atom names (or RESID:ATOM:RESNAME) paired with --receptor-targets")
    ap.add_argument("--pair-cutoff", type=float, default=7.0, help="Reaction distance per pair in A")
    ap.add_argument("--n-needed", type=int, default=None, help="Explicit n_needed override (default: half of even pair count, or 1)")
    ap.add_argument("--chain-pairs-out", default="", help="Output JSON path for flexible-chain COFFDROP reaction pairs")
    ap.add_argument("--coffdrop-dir", default="", help="Optional path to pystarc/coffdrop_data directory")
    args = ap.parse_args(argv)

    rec_ext = os.path.splitext(args.receptor)[1].lower()
    lig_ext = os.path.splitext(args.ligand)[1].lower()
    if rec_ext not in (".pdb", ".pqr") or lig_ext not in (".pdb", ".pqr"):
        sys.exit("ERROR: --receptor and --ligand must end with .pdb or .pqr")
    if not args.chain_pairs_out and rec_ext != lig_ext:
        sys.exit(f"ERROR: --receptor ({rec_ext}) and --ligand ({lig_ext}) must share the same format")

    if args.chain_pairs_out:
        n_need = args.n_needed if args.n_needed is not None else 3
        rp_list, n_beads = build_chain_reaction_pairs(
            args.receptor, args.ligand, args.search_cutoff, args.pair_cutoff,
            n_need, args.chain_pairs_out, args.coffdrop_dir,
        )
        print(f"# Wrote {args.chain_pairs_out}: {len(rp_list)} pairs across {n_beads} COFFDROP beads", file=sys.stderr)
        print("  <criterion>")
        print("    <mode>contacts</mode>")
        print("    <receptor></receptor>")
        print("    <ligand></ligand>")
        print(f"    <pair_cutoff>{args.pair_cutoff}</pair_cutoff>")
        print(f"    <n_needed>{n_need}</n_needed>")
        print("  </criterion>")
        print(f"  <!-- In <chain>, set <reaction_pairs_json>{os.path.basename(args.chain_pairs_out)}</reaction_pairs_json> -->")
        return

    rec = parse_structure_atoms(args.receptor)
    lig = parse_structure_atoms(args.ligand)
    if not rec or not lig:
        sys.exit("ERROR: empty receptor or ligand structure")

    if args.mode == "centroid":
        r_xyz = np.array([a["xyz"] for a in rec])
        l_xyz = np.array([b["xyz"] for b in lig])
        rc, lc = r_xyz.mean(0), l_xyz.mean(0)
        d_cen = float(np.linalg.norm(rc - lc))
        rmax = float(np.max(np.linalg.norm(r_xyz - rc, axis=1)))
        lmax = float(np.max(np.linalg.norm(l_xyz - lc, axis=1)))
        print(f"# Centroid distance = {d_cen:.2f} A (rmax = {rmax:.1f} A, lmax = {lmax:.1f} A)", file=sys.stderr)
        print("  <criterion>")
        print("    <mode>centroid</mode>")
        print("    <receptor></receptor>")
        print("    <ligand></ligand>")
        print(f"    <pair_cutoff>{args.pair_cutoff}</pair_cutoff>")
        print("    <n_needed>0</n_needed>")
        print("  </criterion>")
        return

    if args.receptor_targets:
        picked = contacts_for_targets(
            rec, lig, _csv(args.receptor_targets), _csv(args.ligand_atoms),
            args.atom_filter, args.pair_cutoff,
        )
    else:
        picked = contacts_picked(rec, lig, dict(
            atom_filter=args.atom_filter,
            search_cutoff=args.search_cutoff,
            n_pairs=args.n_pairs,
            pair_cutoff=args.pair_cutoff,
            excluded=_csv(args.exclude),
        ))
        if len(picked) > 1 and len(picked) % 2 == 1:
            picked = picked[:-1]

    if not picked:
        sys.exit(f"ERROR: zero contact pairs found within {args.search_cutoff} A for filter {args.atom_filter!r}")
    n_needed = args.n_needed if args.n_needed is not None else (1 if len(picked) == 1 else len(picked) // 2)
    multi_res_lig = len({b["resid"] for b in lig}) > 1

    for k, (i, j, _cut, dist) in enumerate(picked, 1):
        a, b = rec[i], lig[j]
        print(f"# Pair {k}: {a['resname']}{a['resid']} {a['name']} <-> {b['resname']}{b['resid']} {b['name']}  ({dist:.2f} A)",
              file=sys.stderr)

    print("  <criterion>")
    print("    <mode>contacts</mode>")
    print("    <receptor>")
    for i, _j, _cut, _d in picked:
        a = rec[i]
        print(f'      <target resid="{a["resid"]}" atom="{a["name"]}" resname="{a["resname"]}"/>')
    print("    </receptor>")
    print("    <ligand>")
    for _i, j, _cut, _d in picked:
        b = lig[j]
        if multi_res_lig:
            print(f'      <target resid="{b["resid"]}" atom="{b["name"]}" resname="{b["resname"]}"/>')
        else:
            print(f'      <atom name="{b["name"]}"/>')
    print("    </ligand>")
    print(f"    <pair_cutoff>{args.pair_cutoff}</pair_cutoff>")
    print(f"    <n_needed>{n_needed}</n_needed>")
    print("  </criterion>")


if __name__ == "__main__":
    main()
