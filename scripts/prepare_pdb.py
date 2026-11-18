#!/usr/bin/env python3
"""Standalone PDB preparation helper for PySTARC inputs.

Prepares a clean complex.pdb (receptor + protonated ligand) or receptor.pdb from
a raw PDB file or RCSB entry before running PySTARC:
  - fetches <PDB_ID>.pdb and <CCD>.cif from files.rcsb.org when requested
  - selects a receptor chain, residue number range, and single conformer (altloc)
  - sets histidine tautomers (HID, HIE, HIP) explicitly or from metal coordination
  - retains specified receptor HETATM cofactors/metals (such as ZN) after a TER card
  - assigns ligand bond orders and formal charges from the CCD .cif definition,
    applies requested N/O/S protonation or deprotonation, and places 3D hydrogens

Examples:
  python scripts/prepare_pdb.py --pdb 1A9U.pdb --chain A --resid-min 4 --ligand SB2 -o complex.pdb
  python scripts/prepare_pdb.py --pdb 2YKJ.pdb --chain A --ligand YKJ --ligand-altloc B -o complex.pdb
  python scripts/prepare_pdb.py --pdb 6EI5.pdb --chain A --ligand B5Q --protonate N30 -o complex.pdb
  python scripts/prepare_pdb.py --pdb 3CZV.pdb --chain A --keep-hetatm ZN,AZM --metal-his -o receptor.pdb
"""

import argparse
import os
import sys
import urllib.request
import numpy as np


ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}
METALS = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "CAL"}


def _download(url, dest, label):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:
        sys.exit(f"ERROR: failed to download {label} from {url} ({exc})")
    with open(dest, "wb") as fh:
        fh.write(data)


def _cif_tokens(path):
    out = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            out.append(("value", "\n".join(buf)))
            i += 1
            continue
        j, n = 0, len(line)
        while j < n:
            while j < n and line[j].isspace():
                j += 1
            if j >= n or line[j] == "#":
                break
            if line[j] in ("'", '"'):
                q = line[j]
                k = j + 1
                while k < n and not (line[k] == q and (k + 1 == n or line[k + 1].isspace())):
                    k += 1
                out.append(("value", line[j + 1:k]))
                j = k + 1
            else:
                k = j
                while k < n and not line[k].isspace():
                    k += 1
                w = line[j:k]
                if w == "loop_":
                    out.append(("loop", w))
                elif w.startswith("_"):
                    out.append(("tag", w))
                elif not w.startswith(("data_", "save_")):
                    out.append(("value", w))
                j = k
        i += 1
    return out


def _cif_rows(path, category):
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


def load_ccd(code, cif_path=None, cache_dir="."):
    path = cif_path or os.path.join(cache_dir, f"{code}.cif")
    downloaded = False
    if not os.path.exists(path):
        _download(f"https://files.rcsb.org/ligands/download/{code}.cif", path, f"{code}.cif")
        downloaded = True
    try:
        bonds, charges, elements, leaving = {}, {}, {}, set()
        for row in _cif_rows(path, "_chem_comp_atom"):
            nm = row.get("atom_id")
            if not nm or row.get("comp_id", code) != code:
                continue
            q = row.get("charge", "0")
            charges[nm] = int(q) if q.lstrip("+-").isdigit() else 0
            elements[nm] = row.get("type_symbol", "").capitalize()
            if row.get("pdbx_leaving_atom_flag") == "Y":
                leaving.add(nm)
        for row in _cif_rows(path, "_chem_comp_bond"):
            if row.get("comp_id", code) != code:
                continue
            order = ORDERS.get(row.get("value_order", "").upper())
            if order is None:
                sys.exit(f"ERROR: unsupported bond order {row.get('value_order')!r} in {code}.cif")
            bonds[frozenset((row["atom_id_1"], row["atom_id_2"]))] = order
        if not elements:
            sys.exit(f"ERROR: {path} contains no atoms for {code}")
        return bonds, charges, elements, leaving
    finally:
        if downloaded and os.path.exists(path):
            os.remove(path)


def _select_conformer(lines, preferred="A"):
    chosen = {}
    for l in lines:
        if len(l) < 27:
            continue
        key = (l[21], l[22:27], l[17:20])
        alt = l[16]
        if alt == " ":
            continue
        if key not in chosen or alt == preferred:
            if chosen.get(key) != preferred:
                chosen[key] = alt
    out = []
    for l in lines:
        key = (l[21], l[22:27], l[17:20])
        alt = l[16]
        if alt == " " or alt == chosen.get(key):
            out.append(l[:16] + " " + l[17:])
    return out


def _xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def _apply_his_tautomers(prot_lines, metal_lines, hid_set, hie_set, hip_set, auto_metal, cutoff=2.6):
    taut_map = {}
    if auto_metal and metal_lines:
        m_xyz = [_xyz(l) for l in metal_lines]
        for l in prot_lines:
            resn = l[17:20].strip()
            aname = l[12:16].strip()
            if resn in ("HIS", "HID", "HIE", "HIP") and aname in ("ND1", "NE2"):
                resid = int(l[22:26])
                pos = _xyz(l)
                if min(float(np.linalg.norm(pos - m)) for m in m_xyz) <= cutoff:
                    taut_map[(l[21], resid)] = "HID" if aname == "NE2" else "HIE"
    for r in hid_set:
        for ch in {l[21] for l in prot_lines}:
            taut_map[(ch, r)] = "HID"
    for r in hie_set:
        for ch in {l[21] for l in prot_lines}:
            taut_map[(ch, r)] = "HIE"
    for r in hip_set:
        for ch in {l[21] for l in prot_lines}:
            taut_map[(ch, r)] = "HIP"
    if not taut_map:
        return prot_lines, {}
    updated = []
    applied = {}
    for l in prot_lines:
        key = (l[21], int(l[22:26]))
        if key in taut_map and l[17:20].strip() in ("HIS", "HID", "HIE", "HIP"):
            new_res = taut_map[key]
            l = f"{l[:17]}{new_res:>3s}{l[20:]}"
            applied[key] = new_res
        updated.append(l)
    return updated, applied


def _build_ligand_pdb(lig_lines, code, cif_path, prot_atoms, deprot_atoms, tmp_dir):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    bonds, charges, elements, leaving = load_ccd(code, cif_path=cif_path, cache_dir=tmp_dir)
    block = "".join(lig_lines) + "END\n"
    m = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False, proximityBonding=False)
    if m is None:
        sys.exit(f"ERROR: RDKit could not parse ligand {code}")
    rw = Chem.RWMol(m)
    atom_name = lambda a: a.GetPDBResidueInfo().GetName().strip()
    drop = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1 and atom_name(a) not in elements]
    for k in sorted(drop, reverse=True):
        rw.RemoveAtom(k)
    unknown = sorted({atom_name(a) for a in rw.GetAtoms()} - set(elements))
    if unknown:
        sys.exit(f"ERROR: ligand atoms {unknown} absent from {code} CCD definition")
    heavy = {atom_name(a) for a in rw.GetAtoms() if a.GetAtomicNum() > 1}
    absent = sorted(nm for nm, el in elements.items() if el != "H" and nm not in heavy and nm not in leaving)
    if absent:
        sys.exit(f"ERROR: ligand lacks heavy atoms {absent} of {code}")
    idx = {atom_name(a): a.GetIdx() for a in rw.GetAtoms()}
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for pair, order in bonds.items():
        a1, a2 = tuple(pair)
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], kinds[order])
    for a in rw.GetAtoms():
        a.SetFormalCharge(charges.get(atom_name(a), 0))
    gone = []
    for flag, names, step in (("--protonate", prot_atoms, 1), ("--deprotonate", deprot_atoms, -1)):
        for nm in names:
            if nm not in idx:
                sys.exit(f"ERROR: {flag} atom {nm!r} not found in {code}")
            a = rw.GetAtomWithIdx(idx[nm])
            if a.GetSymbol() not in ("N", "O", "S"):
                sys.exit(f"ERROR: {flag} atom {nm!r} is {a.GetSymbol()}, expected N, O, or S")
            a.SetFormalCharge(a.GetFormalCharge() + step)
            hs = [b.GetIdx() for b in a.GetNeighbors() if b.GetAtomicNum() == 1]
            if step < 0 and hs:
                gone.append(hs[-1])
    for k in sorted(gone, reverse=True):
        rw.RemoveAtom(k)
    mol = rw.GetMol()
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    used = {a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms() if a.GetPDBResidueInfo()}
    hk = 1
    ref = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
    for a in mol.GetAtoms():
        if a.GetPDBResidueInfo() is None:
            while f"H{hk}" in used:
                hk += 1
            nm = f"H{hk}"
            used.add(nm)
            inf = Chem.AtomPDBResidueInfo()
            inf.SetName(f" {nm:<3s}" if len(nm) < 4 else nm[:4])
            inf.SetResidueName(ref.GetResidueName())
            inf.SetResidueNumber(ref.GetResidueNumber())
            inf.SetChainId(ref.GetChainId())
            inf.SetIsHeteroAtom(True)
            inf.SetOccupancy(1.0)
            a.SetPDBResidueInfo(inf)
    q_net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pdb_lines = [l + "\n" for l in Chem.MolToPDBBlock(mol, flavor=0).splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
    return pdb_lines, q_net, formula, mol.GetNumAtoms()


def _csv_list(val, cast=str):
    if not val:
        return []
    return [cast(x.strip()) for x in val.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", help="Local input PDB file")
    ap.add_argument("--pdb-id", help="4-character RCSB PDB identifier to download when --pdb is omitted")
    ap.add_argument("-o", "--output", default="complex.pdb", help="Output PDB path (default: complex.pdb)")
    ap.add_argument("--chain", default="", help="Receptor chain ID to retain (blank keeps all chains)")
    ap.add_argument("--resid-min", type=int, default=None, help="Lowest receptor residue number retained")
    ap.add_argument("--resid-max", type=int, default=None, help="Highest receptor residue number retained")
    ap.add_argument("--altloc", default="A", help="Preferred protein conformer indicator (default: A)")
    ap.add_argument("--ligand", default="", help="3-letter HETATM residue name of the ligand")
    ap.add_argument("--ligand-resid", type=int, default=None, help="Ligand residue number when multiple copies exist")
    ap.add_argument("--ligand-altloc", default="", help="Preferred ligand conformer indicator (defaults to --altloc)")
    ap.add_argument("--ccd", default="", help="CCD component code for ligand bond orders (defaults to --ligand)")
    ap.add_argument("--cif", default="", help="Optional local .cif path for the ligand CCD definition")
    ap.add_argument("--protonate", default="", help="Comma-separated ligand N/O/S atom names gaining one proton")
    ap.add_argument("--deprotonate", default="", help="Comma-separated ligand N/O/S atom names losing one proton")
    ap.add_argument("--keep-hetatm", default="", help="Comma-separated HETATM residue names kept with the receptor")
    ap.add_argument("--hid", default="", help="Comma-separated receptor residue numbers set to HID")
    ap.add_argument("--hie", default="", help="Comma-separated receptor residue numbers set to HIE")
    ap.add_argument("--hip", default="", help="Comma-separated receptor residue numbers set to HIP")
    ap.add_argument("--metal-his", action="store_true", help="Assign HID/HIE on histidines coordinating retained metals")
    args = ap.parse_args(argv)

    if not args.pdb and not args.pdb_id:
        ap.error("Provide --pdb or --pdb-id")

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    downloaded_pdb = None
    pdb_path = args.pdb
    if not pdb_path:
        pid = args.pdb_id.upper()
        pdb_path = os.path.join(out_dir, f"_{pid}_raw.pdb")
        _download(f"https://files.rcsb.org/download/{pid}.pdb", pdb_path, f"{pid}.pdb")
        downloaded_pdb = pdb_path

    try:
        with open(pdb_path) as fh:
            raw_lines = fh.readlines()
    finally:
        if downloaded_pdb and os.path.exists(downloaded_pdb):
            os.remove(downloaded_pdb)

    keep_het = set(_csv_list(args.keep_hetatm))
    prot_raw, kept_het_raw, lig_raw = [], [], []
    for l in raw_lines:
        if l.startswith("ATOM") and len(l) > 26:
            if args.chain and l[21] != args.chain:
                continue
            rid = int(l[22:26])
            if args.resid_min is not None and rid < args.resid_min:
                continue
            if args.resid_max is not None and rid > args.resid_max:
                continue
            if l[12:16].strip() == "OXT":
                continue
            prot_raw.append(l)
        elif l.startswith("HETATM") and len(l) > 26:
            resn = l[17:20].strip()
            if args.ligand and resn == args.ligand:
                if args.ligand_resid is None or int(l[22:26]) == args.ligand_resid:
                    lig_raw.append(l)
            if resn in keep_het and (not args.chain or l[21] == args.chain):
                kept_het_raw.append(l)

    if not prot_raw:
        sys.exit("ERROR: no matching ATOM records in input PDB")

    prot_lines = _select_conformer(prot_raw, preferred=args.altloc)
    kept_het_lines = _select_conformer(kept_het_raw, preferred=args.altloc)
    metal_lines = [l for l in kept_het_lines if l[17:20].strip().upper() in METALS]

    prot_lines, his_applied = _apply_his_tautomers(
        prot_lines, metal_lines,
        set(_csv_list(args.hid, int)),
        set(_csv_list(args.hie, int)),
        set(_csv_list(args.hip, int)),
        args.metal_his,
    )

    lig_out_lines = []
    lig_msg = "none"
    if args.ligand:
        lig_alt = args.ligand_altloc or args.altloc
        lig_conf = _select_conformer(lig_raw, preferred=lig_alt)
        groups = {}
        for l in lig_conf:
            groups.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), []).append(l)
        if not groups:
            sys.exit(f"ERROR: ligand {args.ligand!r} not found in {pdb_path}")
        key = next((k for k in groups if args.chain and k[0] == args.chain), next(iter(groups)))
        ccd_code = args.ccd or args.ligand
        prot_atoms = _csv_list(args.protonate)
        deprot_atoms = _csv_list(args.deprotonate)
        if set(prot_atoms) & set(deprot_atoms):
            sys.exit(f"ERROR: overlap between --protonate and --deprotonate: {sorted(set(prot_atoms) & set(deprot_atoms))}")
        lig_out_lines, q_net, formula, n_lig = _build_ligand_pdb(
            groups[key], ccd_code, args.cif or None, prot_atoms, deprot_atoms, out_dir
        )
        lig_msg = f"{args.ligand} ({n_lig} atoms, {formula}, Q={q_net:+d} e)"

    with open(args.output, "w") as out:
        for l in prot_lines:
            out.write(l if l.endswith("\n") else l + "\n")
        out.write("TER\n")
        for l in kept_het_lines:
            out.write(l if l.endswith("\n") else l + "\n")
            if l[17:20].strip().upper() in METALS:
                out.write("TER\n")
        for l in lig_out_lines:
            out.write(l if l.endswith("\n") else l + "\n")
        out.write("END\n")

    n_res = len({(l[21], l[22:27]) for l in prot_lines})
    his_str = ", ".join(f"{c}{r}:{t}" for (c, r), t in sorted(his_applied.items())) or "unchanged"
    print(f"Prepared {args.output}: {n_res} residues ({len(prot_lines)} protein atoms, His: {his_str}), "
          f"kept HETATM: {len(kept_het_lines)} atoms, ligand: {lig_msg}")


if __name__ == "__main__":
    main()
