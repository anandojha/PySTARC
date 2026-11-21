#!/usr/bin/env python3
"""Standalone reaction-criterion builder for any receptor-ligand complex.

Inspects prepared receptor and ligand structures (.pdb + .pdb or .pqr + .pqr)
and prints a ready-to-paste <criterion> block (or writes reaction_pairs.json
when --chain-pairs-out is set). Independent of PySTARC runtime code.

Coordinate-based criterion selection (from atomic coordinates alone):
  - Geometric centers (centroid): symmetric sphere pairs or macrocyclic
      host cavities (receptor heavy-atom count < 120).
  - Solvent-accessible binding sites (polar contacts): exposed interfaces
      (solid-angle shielding fraction < 0.70) or small ligands (ligand
      heavy-atom count <= 15); 3-of-6 shortest bound polar contacts
      (N, O, S, one per receptor residue) at 7.0 A.
  - Deep binding sites (exposed rim contacts): deep pockets
      (shielding fraction >= 0.70 and ligand heavy atoms > 15) or series
      lacking an experimental bound pose; 3-of-6 conserved solvent-exposed
      receptor targets shared across the receptor family at a rigid-body
      sterically accessible cutoff.

Workflows:
  1. Diagnose binding-site geometry and rank solvent-exposed rim residues:
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --diagnose --suggest-rim
  2. Bound-pose contact pairs (polar, hetero, all, nonpolar, or custom --elements):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --atom-filter polar --n-pairs 6 --pair-cutoff 7.0
  3. Shared solvent-exposed receptor targets (explicit anchor or nearest heavy atoms):
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --receptor-targets 51:ND2:ASN,54:OD1:ASP,58:NZ:LYS,98:SD:MET,102:OD1:ASP,107:CG:LEU --atom-filter all --pair-cutoff 7.5
  4. Verify rigid-body steric accessibility of a candidate criterion:
     python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --atom-filter polar --pair-cutoff 7.0 --check-accessibility
  5. Centroid reaction criterion:
     python scripts/find_reaction_criteria.py --receptor receptor.pqr --ligand ligand.pqr --mode centroid --pair-cutoff 8.0
  6. Flexible-chain COFFDROP reaction_pairs.json (single or multi-chain receptor):
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


VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
    "ZN": 1.39, "MG": 1.73, "CA": 1.97, "FE": 1.40, "MN": 1.40,
    "CU": 1.40, "CO": 1.40, "NI": 1.63,
}

SIDECHAIN_TIP_ATOMS = {
    "ALA": "CB", "ARG": "CZ", "ASN": "ND2", "ASP": "OD1", "ASH": "OD1",
    "CYS": "SG", "CYX": "SG", "CYM": "SG", "GLN": "NE2", "GLU": "OE1",
    "GLH": "OE1", "GLY": "CA", "HIS": "NE2", "HID": "ND1", "HIE": "NE2",
    "HIP": "NE2", "ILE": "CD1", "LEU": "CG", "LYS": "NZ", "LYN": "NZ",
    "MET": "SD", "PHE": "CZ", "PRO": "CG", "SER": "OG", "THR": "OG1",
    "TRP": "CH2", "TYR": "OH", "VAL": "CB",
}


def _fibonacci_sphere(n=100):
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)


def compute_pocket_geometry(rec, lig):
    """Compute solid-angle shielding and solute size descriptors from coordinates."""
    r_xyz = np.array([a["xyz"] for a in rec if a["el"] != "H"], dtype=float)
    l_xyz = np.array([b["xyz"] for b in lig if b["el"] != "H"], dtype=float)
    if not len(r_xyz):
        r_xyz = np.array([a["xyz"] for a in rec], dtype=float)
    if not len(l_xyz):
        l_xyz = np.array([b["xyz"] for b in lig], dtype=float)

    n_rec = len(r_xyz)
    n_lig = len(l_xyz)
    rc = r_xyz.mean(axis=0)
    lc = l_xyz.mean(axis=0)
    d_com = float(np.linalg.norm(lc - rc))
    r_rec_max = float(np.max(np.linalg.norm(r_xyz - rc, axis=1)))
    r_lig_max = float(np.max(np.linalg.norm(l_xyz - lc, axis=1)))

    # Mean receptor heavy-atom coordination around ligand heavy atoms
    dmat = np.linalg.norm(l_xyz[:, None, :] - r_xyz[None, :, :], axis=-1)
    mean_n6 = float((dmat < 6.0).sum(axis=1).mean())
    mean_n8 = float((dmat < 8.0).sum(axis=1).mean())

    # Solid-angle shielding fraction via 100 Fibonacci rays from the ligand centroid
    dirs = _fibonacci_sphere(100)
    near_r = r_xyz[np.linalg.norm(r_xyz - lc, axis=1) < 12.0]
    if len(near_r):
        v = near_r - lc
        proj = dirs @ v.T
        v2 = np.sum(v * v, axis=1)[None, :]
        perp2 = np.maximum(0.0, v2 - proj * proj)
        blocked = np.any((proj > 0.0) & (perp2 < 9.0), axis=1)
        f_shielded = float(blocked.mean())
        open_dirs = dirs[~blocked]
    else:
        f_shielded = 0.0
        open_dirs = dirs

    if len(open_dirs) and float(np.linalg.norm(open_dirs.mean(axis=0))) > 1e-6:
        exit_vec = open_dirs.mean(axis=0)
        exit_vec = exit_vec / np.linalg.norm(exit_vec)
    else:
        diff = lc - rc
        nrm = float(np.linalg.norm(diff))
        exit_vec = diff / nrm if nrm > 1e-6 else np.array([0.0, 0.0, 1.0])

    if n_rec < 120:
        site_class = "Centroid (host/sphere)"
        rec_rule = "mode=centroid (symmetric sphere pair or macrocyclic host cavity)"
    elif f_shielded < 0.70 or n_lig <= 15:
        site_class = "Accessible site (polar)"
        rec_rule = "mode=contacts, --atom-filter polar --search-cutoff 6.0 --n-pairs 6 --pair-cutoff 7.0"
    else:
        site_class = "Deep site (exposed rim)"
        rec_rule = "mode=contacts, 3-of-6 shared solvent-exposed receptor targets at a sterically accessible cutoff"

    return {
        "n_rec_heavy": n_rec,
        "n_lig_heavy": n_lig,
        "d_com": d_com,
        "r_rec_max": r_rec_max,
        "r_lig_max": r_lig_max,
        "mean_n6": mean_n6,
        "mean_n8": mean_n8,
        "f_shielded": f_shielded,
        "exit_vec": exit_vec,
        "site_class": site_class,
        "recommendation": rec_rule,
    }


def suggest_pocket_rim_targets(rec, lig, n_targets=6, rim_atom="CB"):
    """Select n_targets solvent-exposed binding-site rim atoms from 3D geometry."""
    geom = compute_pocket_geometry(rec, lig)
    exit_vec = geom["exit_vec"]
    r_heavy_xyz = np.array([a["xyz"] for a in rec if a["el"] != "H"], dtype=float)
    l_heavy_xyz = np.array([b["xyz"] for b in lig if b["el"] != "H"], dtype=float)
    lc = l_heavy_xyz.mean(axis=0)

    # Orthonormal basis perpendicular to the pocket exit vector for azimuthal spread
    ref = np.array([1.0, 0.0, 0.0]) if abs(exit_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(exit_vec, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(exit_vec, e1)

    cands = []
    seen_res = set()
    for i, a in enumerate(rec):
        if a["resname"] not in PROTEIN_RESIDUES:
            continue
        wanted = SIDECHAIN_TIP_ATOMS.get(a["resname"], "CB") if rim_atom == "tip" else rim_atom
        if a["name"] != wanted:
            continue
        rk = (a["chain"], a["rescode"])
        if rk in seen_res:
            continue
        v = a["xyz"] - lc
        d_lc = float(np.linalg.norm(v))
        if d_lc < 3.5 or d_lc > 14.0:
            continue
        seen_res.add(rk)
        # Receptor heavy-atom coordination measures local exposure (lower = more exposed)
        n8_rec = int((np.linalg.norm(r_heavy_xyz - a["xyz"], axis=1) < 8.0).sum())
        outward = float(np.dot(v, exit_vec))
        azim = float(np.arctan2(np.dot(v, e2), np.dot(v, e1)))
        d_lig_min = float(np.min(np.linalg.norm(l_heavy_xyz - a["xyz"], axis=1)))
        cands.append({
            "idx": i, "atom": a, "d_lc": d_lc, "d_lig_min": d_lig_min,
            "n8_rec": n8_rec, "outward": outward, "azim": azim,
        })

    if not cands:
        return []

    # Score outward-facing solvent-exposed rim residues (higher outward, lower crowding, closer to ligand)
    out_vals = np.array([c["outward"] for c in cands])
    n8_vals = np.array([c["n8_rec"] for c in cands], dtype=float)
    d_vals = np.array([c["d_lig_min"] for c in cands])
    out_z = (out_vals - out_vals.mean()) / max(float(out_vals.std()), 1e-6)
    n8_z = (n8_vals - n8_vals.mean()) / max(float(n8_vals.std()), 1e-6)
    d_z = (d_vals - d_vals.mean()) / max(float(d_vals.std()), 1e-6)
    for k, c in enumerate(cands):
        c["score"] = float(1.2 * out_z[k] - 1.0 * n8_z[k] - 0.6 * d_z[k])

    pool = sorted(cands, key=lambda c: c["score"], reverse=True)[:max(18, n_targets * 3)]
    selected = [pool[0]]
    while len(selected) < min(n_targets, len(pool)):
        best = None
        best_obj = -1e9
        for c in pool:
            if any(c["idx"] == s["idx"] for s in selected):
                continue
            min_ang = min(abs(((c["azim"] - s["azim"] + np.pi) % (2 * np.pi)) - np.pi) for s in selected)
            min_sep = min(float(np.linalg.norm(c["atom"]["xyz"] - s["atom"]["xyz"])) for s in selected)
            obj = 0.5 * c["score"] + 1.5 * min_ang + 0.15 * min(min_sep, 10.0)
            if obj > best_obj:
                best_obj = obj
                best = c
        if best is None:
            break
        selected.append(best)

    selected.sort(key=lambda c: (c["atom"]["chain"], c["atom"]["resid"]))
    return selected


def check_rigid_accessibility(rec, lig, picked, n_needed, cutoffs=(7.0, 8.0, 9.0), n_samples=12000, seed=0):
    """Fast NumPy rigid-body steric accessibility audit across candidate cutoffs."""
    r_heavy = [a for a in rec if a["el"] != "H"]
    l_heavy = [b for b in lig if b["el"] != "H"]
    if not r_heavy or not l_heavy or not picked:
        return {}

    l_xyz = np.array([b["xyz"] for b in l_heavy], dtype=float)
    l_rad = np.array([VDW_RADII.get(b["el"], 1.70) for b in l_heavy], dtype=float)
    lc = l_xyz.mean(axis=0)
    l_rel = l_xyz - lc

    # Retain receptor heavy atoms within 18 A of the bound ligand centroid
    r_xyz_all = np.array([a["xyz"] for a in r_heavy], dtype=float)
    r_rad_all = np.array([VDW_RADII.get(a["el"], 1.70) for a in r_heavy], dtype=float)
    near_mask = np.linalg.norm(r_xyz_all - lc, axis=1) < 18.0
    r_xyz = r_xyz_all[near_mask]
    r_rad = r_rad_all[near_mask]
    if not len(r_xyz):
        return {float(c): n_samples for c in cutoffs}

    # Map reaction pair atoms into receptor coordinates and ligand relative vectors
    rec_targ = np.array([rec[i]["xyz"] for i, _j, _c, _d in picked], dtype=float)
    lig_targ_rel = np.array([lig[j]["xyz"] - lc for _i, j, _c, _d in picked], dtype=float)
    geom = compute_pocket_geometry(rec, lig)
    exit_vec = geom["exit_vec"]

    rng = np.random.default_rng(seed)
    # Sample rigid translations around the bound pose and outward along the exit vector
    n_half = n_samples // 2
    t_local = rng.normal(0.0, 2.2, size=(n_half, 3))
    t_exit = rng.uniform(0.0, 7.5, size=(n_samples - n_half, 1)) * exit_vec[None, :] + rng.normal(0.0, 1.8, size=(n_samples - n_half, 3))
    trans = np.vstack([t_local, t_exit])

    # Small and moderate random rigid rotations
    axes = rng.normal(0.0, 1.0, size=(n_samples, 3))
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    angles = np.concatenate([
        rng.uniform(0.0, np.deg2rad(35.0), size=n_half),
        rng.uniform(0.0, np.deg2rad(120.0), size=n_samples - n_half),
    ])

    # Evaluate in chunks to stay memory-light
    sum_rad = l_rad[:, None] + r_rad[None, :]
    clash_cut2 = (0.72 * sum_rad) ** 2
    k_idx = max(0, int(n_needed) - 1)

    d_clear_min = float("inf")
    counts = {float(c): 0 for c in cutoffs}
    chunk = 1000
    for i0 in range(0, n_samples, chunk):
        i1 = min(i0 + chunk, n_samples)
        ax = axes[i0:i1]
        ang = angles[i0:i1]
        c_a = np.cos(ang)[:, None, None]
        s_a = np.sin(ang)[:, None, None]
        ux, uy, uz = ax[:, 0], ax[:, 1], ax[:, 2]
        K = np.zeros((i1 - i0, 3, 3), dtype=float)
        K[:, 0, 1] = -uz; K[:, 0, 2] = uy
        K[:, 1, 0] = uz;  K[:, 1, 2] = -ux
        K[:, 2, 0] = -uy; K[:, 2, 1] = ux
        I3 = np.broadcast_to(np.eye(3), (i1 - i0, 3, 3))
        R = I3 + s_a * K + (1.0 - c_a) * (K @ K)

        shift = lc[None, :] + trans[i0:i1]
        # Pair distances first so poses far outside max(cutoffs) skip full clash check
        lt_pos = np.einsum("nij,pj->npi", R, lig_targ_rel) + shift[:, None, :]
        d_pairs = np.linalg.norm(lt_pos - rec_targ[None, :, :], axis=-1)
        d_k = np.sort(d_pairs, axis=1)[:, k_idx]
        cand_idx = np.where(d_k <= max(cutoffs) + 1.0)[0]
        if not len(cand_idx):
            continue

        # Rigid steric overlap check on candidate poses
        poses = np.einsum("nij,mj->nmi", R[cand_idx], l_rel) + shift[cand_idx, None, :]
        diff = poses[:, :, None, :] - r_xyz[None, None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        n_clash = np.sum(dist2 < clash_cut2[None, :, :], axis=(1, 2))
        clear = n_clash == 0
        if np.any(clear):
            dk_clear = d_k[cand_idx[clear]]
            d_clear_min = min(d_clear_min, float(dk_clear.min()))
            for cut in cutoffs:
                counts[float(cut)] += int(np.sum(dk_clear <= cut))

    min_hits = max(1, int(0.002 * n_samples))
    rec_cut = next((float(c) for c in sorted(cutoffs) if counts[float(c)] >= min_hits), None)
    if rec_cut is None:
        rec_cut = next((float(c) for c in sorted(cutoffs) if counts[float(c)] > 0), None)
    return {
        "d_clear_min": d_clear_min,
        "counts": counts,
        "recommended_cutoff": rec_cut,
        "min_hits": min_hits,
        "n_samples": n_samples,
    }


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
    ap.add_argument("--diagnose", action="store_true", help="Print geometric binding-site classification and default criterion")
    ap.add_argument("--suggest-rim", action="store_true", help="Rank solvent-exposed binding-site rim targets from 3D geometry")
    ap.add_argument("--rim-atom", default="CB", help="Atom type for --suggest-rim (CB or tip, default CB)")
    ap.add_argument("--check-accessibility", action="store_true", help="Run rigid-body steric accessibility audit across 7.0, 8.0, and 9.0 A")
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

    if args.diagnose or args.suggest_rim:
        geom = compute_pocket_geometry(rec, lig)
        print(f"# Receptor heavy atoms   : {geom['n_rec_heavy']}", file=sys.stderr)
        print(f"# Ligand heavy atoms     : {geom['n_lig_heavy']}", file=sys.stderr)
        print(f"# Centroid separation    : {geom['d_com']:.2f} A", file=sys.stderr)
        print(f"# Solid-angle shielding  : {geom['f_shielded']:.2f}  (mean rec heavy atoms <6A: {geom['mean_n6']:.1f}, <8A: {geom['mean_n8']:.1f})", file=sys.stderr)
        print(f"# Site geometry          : {geom['site_class']}", file=sys.stderr)
        print(f"# Default criterion      : {geom['recommendation']}", file=sys.stderr)
        if args.suggest_rim or geom["site_class"] == "Deep site (exposed rim)":
            rim = suggest_pocket_rim_targets(rec, lig, n_targets=args.n_pairs, rim_atom=args.rim_atom)
            if rim:
                spec_str = ",".join(f"{c['atom']['resid']}:{c['atom']['name']}:{c['atom']['resname']}" for c in rim)
                print(f"# Suggested {len(rim)} rim atoms ({args.rim_atom}): {spec_str}", file=sys.stderr)
                for k, c in enumerate(rim, 1):
                    a = c["atom"]
                    print(
                        f"#   Rim {k}: {a['resname']}{a['resid']}:{a['name']}  "
                        f"d_lig={c['d_lig_min']:.2f} A  n8_rec={c['n8_rec']}  outward={c['outward']:+.2f} A",
                        file=sys.stderr,
                    )
        if not args.check_accessibility and not args.receptor_targets and args.mode == "contacts" and not args.ligand_atoms:
            return

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

    if args.check_accessibility:
        acc = check_rigid_accessibility(rec, lig, picked, n_needed)
        dmin = acc.get("d_clear_min", float("inf"))
        cnts = acc.get("counts", {})
        rec_cut = acc.get("recommended_cutoff")
        dmin_str = f"{dmin:.2f} A" if np.isfinite(dmin) else ">10.0 A"
        rec_str = f"{rec_cut:.1f} A" if rec_cut is not None else ">9.0 A"
        print(
            f"# Rigid accessibility ({acc.get('n_samples', 0)} poses): "
            f"d_min(clash-free)={dmin_str}  "
            f"hits(7.0A={cnts.get(7.0, 0)}, 8.0A={cnts.get(8.0, 0)}, 9.0A={cnts.get(9.0, 0)})  "
            f"smallest accessible integer cutoff={rec_str}",
            file=sys.stderr,
        )

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
