#!/usr/bin/env python3
"""Standalone reaction-criterion builder for any receptor-ligand complex.

Inspects prepared receptor and ligand structures (.pdb + .pdb or .pqr + .pqr)
and prints a ready-to-paste <criterion> block (or writes reaction_pairs.json
when --chain-pairs-out is set). Independent of PySTARC runtime code.

Workflows:
  1. Bound-pose contact pairs (polar, hetero, all, nonpolar, or custom --elements):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --atom-filter polar --n-pairs 6 --pair-cutoff 7.0
  2. Named active-site receptor targets (with explicit or auto-picked ligand atoms):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --receptor-targets 199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE --ligand-atoms N1 --pair-cutoff 7.0
  3. Centroid reaction criterion:
     python scripts/find_reaction_criteria.py --receptor receptor.pqr --ligand ligand.pqr --mode centroid --pair-cutoff 8.0
  4. Flexible-chain COFFDROP reaction_pairs.json (single or multi-chain receptor):
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
RETAINED_METALS = {"ZN", "MG", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CA", "CAL", "CD", "HG", "SR", "BA"}
SKIP_SOLVENT = {"HOH", "WAT", "TIP3", "SPC", "NA", "CL", "K", "Na+", "Cl-", "K+"}
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "CYM", "GLN", "GLU", "GLH",
    "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME",
}
TWO_LETTER_ELEMENTS = {
    "CL", "BR", "SE", "SI", "ZN", "MG", "MN", "FE", "CU", "CO", "NI",
    "CA", "NA", "PT", "AU", "AG", "HG", "CD", "SR", "BA", "AL",
}
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
    "hetero":         "Matches any non-carbon heavy atom (N O S P halogens metals) on both sides",
    "nonpolar":       "Matches carbon on both sides",
    "any_polar":      "Matches N O or S on either side",
    "receptor_polar": "Matches N O or S on the receptor side",
}


def keeper(kind, custom_elements=""):
    if custom_elements:
        if ":" not in custom_elements:
            sys.exit("ERROR: --elements expects REC_ELEMENTS:LIG_ELEMENTS (for example ZN,O,N:N,O)")
        r_part, l_part = custom_elements.split(":", 1)
        r_set = {x.strip().upper() for x in r_part.split(",") if x.strip()}
        l_set = {x.strip().upper() for x in l_part.split(",") if x.strip()}
        return lambda a, b: a in r_set and b in l_set
    return {
        "all": lambda a, b: True,
        "polar": lambda a, b: a in POLAR_ELEMENTS and b in POLAR_ELEMENTS,
        "hetero": lambda a, b: a not in ("C", "H") and b not in ("C", "H"),
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
    if ru in PROTEIN_RESIDUES:
        return nm[:1]
    if ru in ("CAL", "CA") and nm.startswith("CA"):
        return "CA"
    if ru in RETAINED_METALS and nm[:2] in TWO_LETTER_ELEMENTS:
        return nm[:2]
    return nm[:2] if nm[:2] in ("CL", "BR", "SE", "SI", "ZN", "MG", "MN", "FE", "CU") else nm[:1]


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
                    ch = l[21].strip()
                    resid = int(l[22:26])
                    rescode = l[22:27].strip()
                    x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
                    tail = l[54:].split()
                    col_el = tail[2] if len(tail) >= 3 and tail[2].isalpha() else ""
                    records.append((l[:6].strip(), aname, resname, ch, resid, rescode, x, y, z, col_el))
                    continue
                except ValueError:
                    pass
            if len(toks) >= 10 and not toks[4].lstrip("+-").isdigit():
                resname, ch, resid = toks[3], toks[4], int(toks[5])
                x, y, z = float(toks[6]), float(toks[7]), float(toks[8])
            else:
                resname, ch, resid = toks[3], "", int(toks[4])
                x, y, z = float(toks[5]), float(toks[6]), float(toks[7])
            records.append((toks[0], toks[2], resname, ch, resid, str(resid), x, y, z, ""))
    return records


def parse_structure_atoms(path):
    """Parse ATOM/HETATM records from a .pdb or .pqr file into atom dicts."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found")
    atoms = []
    if path.lower().endswith(".pqr"):
        for _rt, aname, resname, ch, resid, rescode, x, y, z, col_el in _parse_pqr_records_standalone(path):
            el = _infer_element(aname, col_el, resname)
            atoms.append(dict(name=aname, el=el, resname=resname, chain=ch, resid=resid, rescode=rescode,
                              xyz=np.array([x, y, z], dtype=float)))
    else:
        with open(path) as fh:
            for l in fh:
                if not l.startswith(("ATOM", "HETATM")) or len(l) < 54 or l[16] not in (" ", "A"):
                    continue
                aname = l[12:16].strip()
                if aname.upper() in ("GHO", "PKT"):
                    continue
                resname = l[17:20].strip()
                ch = l[21].strip()
                resid = int(l[22:26])
                rescode = l[22:27].strip()
                col_el = l[76:78].strip() if len(l) >= 78 else ""
                el = _infer_element(aname, col_el, resname)
                atoms.append(dict(name=aname, el=el, resname=resname, chain=ch, resid=resid, rescode=rescode,
                                  xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])], dtype=float)))
    # Assign 1-based sequential residue index matching tleap ordering (protein residues then retained metals)
    seq_order = []
    for a in atoms:
        rk = (a["chain"], a["rescode"])
        if rk not in seq_order:
            seq_order.append(rk)
    seq_map = {rk: idx + 1 for idx, rk in enumerate(seq_order)}
    for a in atoms:
        a["seq_resid"] = seq_map[(a["chain"], a["rescode"])]
    return atoms


def _has_duplicate_resid_across_chains(atoms):
    seen = {}
    for a in atoms:
        rk = (a["chain"], a["rescode"])
        seen.setdefault(a["resid"], set()).add(rk)
    return any(len(v) > 1 for v in seen.values())


def contacts_picked(rec, lig, c):
    keep = keeper(c["atom_filter"], c.get("custom_elements", ""))
    raw_excl = c.get("excluded") or ()
    skip = {(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_excl}
    one_per_res = not bool(c.get("allow_same_residue"))
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
        rk = (rec[i]["chain"], rec[i]["rescode"])
        if one_per_res and rk in seen:
            continue
        seen.add(rk)
        out.append((i, j, c["pair_cutoff"], d))
        if len(out) >= c["n_pairs"]:
            break
    return out


def _match_spec_atom(atoms, spec, label):
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if len(parts) == 1:
        hits = [k for k, a in enumerate(atoms) if a["name"] == parts[0]]
        if len(hits) != 1:
            sys.exit(f"ERROR: {'no' if not hits else 'ambiguous'} {label} atom {spec!r}; specify RESID:ATOM:RESNAME")
        return hits[0]
    if len(parts) == 3:
        ch, resid, aname, resn = None, int(parts[0]), parts[1], parts[2].upper()
    elif len(parts) == 4:
        ch, resid, aname, resn = parts[0], int(parts[1]), parts[2], parts[3].upper()
    else:
        sys.exit(f"ERROR: {label} entry {spec!r} must be ATOM, RESID:ATOM:RESNAME, or CHAIN:RESID:ATOM:RESNAME")
    hits = [k for k, a in enumerate(atoms)
            if (ch is None or a["chain"] == ch) and a["resid"] == resid
            and a["name"] == aname and _resname_match(a["resname"], resn)]
    if not hits:
        sys.exit(f"ERROR: {label} target {spec!r} absent from structure")
    if len(hits) > 1:
        sys.exit(f"ERROR: {label} target {spec!r} matches multiple chains; specify CHAIN:RESID:ATOM:RESNAME")
    return hits[0]


def contacts_for_targets(rec, lig, target_specs, lig_atom_specs, atom_filter, custom_elements, pair_cutoff):
    keep = keeper(atom_filter, custom_elements)
    r_list = [_match_spec_atom(rec, spec, "receptor") for spec in target_specs]

    if lig_atom_specs:
        if len(lig_atom_specs) == 1 and len(r_list) > 1:
            lig_atom_specs = lig_atom_specs * len(r_list)
        if len(lig_atom_specs) != len(r_list):
            sys.exit(f"ERROR: {len(r_list)} --receptor-targets against {len(lig_atom_specs)} --ligand-atoms")
        out = []
        for i, lspec in zip(r_list, lig_atom_specs):
            j = _match_spec_atom(lig, lspec, "ligand")
            d = float(np.linalg.norm(rec[i]["xyz"] - lig[j]["xyz"]))
            out.append((i, j, pair_cutoff, d))
        return out

    out = []
    for i in r_list:
        a = rec[i]
        cands = [(float(np.linalg.norm(a["xyz"] - b["xyz"])), j)
                 for j, b in enumerate(lig) if b["el"] != "H" and keep(a["el"], b["el"])]
        if not cands:
            sys.exit(f"ERROR: no ligand atom permitted by filter for {a['resname']}{a['resid']}")
        d, j = min(cands)
        out.append((i, j, pair_cutoff, d))
    return out


def _seq_residues(atoms):
    out, cur = [], None
    for a in atoms:
        rk = (a.get("chain", ""), a["resid"])
        if cur != rk:
            cur = rk
            out.append({"seq_pos": len(out), "resname": a["resname"], "chain": rk[0], "resid_native": a["resid"], "atoms": []})
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
            ch = l[21].strip()
            chains.add(ch)
            atoms.append({
                "atom_name": aname,
                "resname": l[17:20].strip(),
                "chain": ch,
                "resid": int(l[22:26]),
                "xyz": np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])], dtype=float),
            })
    return atoms, sorted(chains)


def _parameterize_receptor_pdb_to_pqr(pdb_src, tmp_dir):
    prot, metals = [], []
    prev_ch = None
    with open(pdb_src) as fh:
        for l in fh:
            if l.startswith("ATOM"):
                a = l[12:16].strip()
                if a == "OXT" or _is_h_atom(a):
                    continue
                ch = l[21]
                if prev_ch is not None and ch != prev_ch:
                    prot.append("TER\n")
                prev_ch = ch
                prot.append(l if l.endswith("\n") else l + "\n")
            elif l.startswith("HETATM"):
                rn = l[17:20].strip()
                if rn in RETAINED_METALS:
                    metals.append(l if l.endswith("\n") else l + "\n")
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
            {"atom_idx": i, "atom_name": r[1], "resname": r[2], "chain": r[3], "resid": r[4],
             "xyz": np.array([r[6], r[7], r[8]], dtype=float)}
            for i, r in enumerate(raw_pqr) if not _is_h_atom(r[1], r[9])
        ]
    else:
        rec_pdb_atoms, _rec_chains = _parse_pdb_heavy_chain(rec_path)
        rec_res_pdb = _seq_residues(rec_pdb_atoms)
        with tempfile.TemporaryDirectory() as tmp:
            pqr_path = _parameterize_receptor_pdb_to_pqr(rec_path, tmp)
            pqr_raw = [r for r in _parse_pqr_records_standalone(pqr_path) if r[0] == "ATOM"]
        pqr_atoms = [{"atom_idx": i, "atom_name": r[1], "resname": r[2], "chain": r[3], "resid": r[4],
                      "xyz": np.array([r[6], r[7], r[8]], dtype=float)}
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
        sys.exit(f"ERROR: {lig_path} contains multiple chains {lig_chains}; extract a single diffusing chain")
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
    ap.add_argument("--elements", default="", help="Custom element pair filter REC_ELS:LIG_ELS (for example ZN,O,N:N,O)")
    ap.add_argument("--allow-same-residue", action="store_true", help="Allow multiple pairs on one receptor residue")
    ap.add_argument("--exclude", default="", help="Comma-separated atom names omitted from contact search")
    ap.add_argument("--receptor-targets", default="", help="Comma-separated RESID:ATOM:RESNAME (or CHAIN:RESID:ATOM:RESNAME) targets")
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
            args.atom_filter, args.elements, args.pair_cutoff,
        )
    else:
        picked = contacts_picked(rec, lig, dict(
            atom_filter=args.atom_filter,
            custom_elements=args.elements,
            allow_same_residue=args.allow_same_residue,
            search_cutoff=args.search_cutoff,
            n_pairs=args.n_pairs,
            pair_cutoff=args.pair_cutoff,
            excluded=_csv(args.exclude),
        ))
        if len(picked) > 1 and len(picked) % 2 == 1 and args.n_needed is None:
            picked = picked[:-1]

    if not picked:
        sys.exit(f"ERROR: zero contact pairs found within {args.search_cutoff} A")
    n_needed = args.n_needed if args.n_needed is not None else (1 if len(picked) == 1 else len(picked) // 2)
    multi_res_lig = len({(b["chain"], b["rescode"]) for b in lig}) > 1
    dup_rec_resid = rec_ext == ".pdb" and _has_duplicate_resid_across_chains(rec)
    dup_lig_resid = lig_ext == ".pdb" and multi_res_lig and _has_duplicate_resid_across_chains(lig)
    if dup_rec_resid or dup_lig_resid:
        print("# Multi-chain duplicate PDB residue numbers mapped to sequential 1-based PQR residue numbers",
              file=sys.stderr)

    for k, (i, j, _cut, dist) in enumerate(picked, 1):
        a, b = rec[i], lig[j]
        rc_lbl = f"{a['chain']}:{a['resname']}{a['rescode']}" if a["chain"] else f"{a['resname']}{a['rescode']}"
        lc_lbl = f"{b['chain']}:{b['resname']}{b['rescode']}" if b["chain"] else f"{b['resname']}{b['rescode']}"
        print(f"# Pair {k}: {rc_lbl} {a['name']} <-> {lc_lbl} {b['name']}  ({dist:.2f} A)", file=sys.stderr)

    print("  <criterion>")
    print("    <mode>contacts</mode>")
    print("    <receptor>")
    for i, _j, _cut, _d in picked:
        a = rec[i]
        rid = a["seq_resid"] if dup_rec_resid else a["resid"]
        print(f'      <target resid="{rid}" atom="{a["name"]}" resname="{a["resname"]}"/>')
    print("    </receptor>")
    print("    <ligand>")
    for _i, j, _cut, _d in picked:
        b = lig[j]
        if multi_res_lig:
            lrid = b["seq_resid"] if dup_lig_resid else b["resid"]
            print(f'      <target resid="{lrid}" atom="{b["name"]}" resname="{b["resname"]}"/>')
        else:
            print(f'      <atom name="{b["name"]}"/>')
    print("    </ligand>")
    print(f"    <pair_cutoff>{args.pair_cutoff}</pair_cutoff>")
    print(f"    <n_needed>{n_needed}</n_needed>")
    print("  </criterion>")


if __name__ == "__main__":
    main()
