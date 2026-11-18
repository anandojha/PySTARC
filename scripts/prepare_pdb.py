#!/usr/bin/env python3
"""Standalone structure preparation helper for PySTARC inputs.

Prepares separate receptor and ligand structure files (.pdb or .pqr) from raw
PDB entries, AMBER prmtop topologies, SMILES strings, or combined complex PDBs
before running PySTARC:
  - fetches <PDB_ID>.pdb and <CCD>.cif from files.rcsb.org when requested
  - selects a receptor chain, residue number range, and single conformer (altloc)
  - sets histidine tautomers (HID, HIE, HIP) explicitly or from metal coordination
  - retains specified receptor HETATM cofactors/metals (such as ZN) after a TER card
  - assigns crystal ligand bond orders and formal charges from the CCD .cif definition,
    applies requested N/O/S protonation or deprotonation, and places 3D hydrogens
  - embeds a 3D ligand conformer from SMILES and places its centroid on a pocket
    reference ligand when --smiles is passed
  - extracts a diffusing protein chain when --ligand-chain is passed
  - splits an existing prepared complex PDB via --split-complex
  - exports receptor.pqr and ligand.pqr from an AMBER topology via --prmtop

Examples:
  python scripts/prepare_pdb.py --pdb 1A9U.pdb --chain A --resid-min 4 --ligand SB2 --receptor-out receptor.pdb --ligand-out ligand.pdb
  python scripts/prepare_pdb.py --pdb 3CZV.pdb --chain A --keep-hetatm ZN --metal-his --smiles "CC(=O)..." --pocket-ligand AZM --pocket-ligand-resid 263 --receptor-out receptor.pdb --ligand-out ligand.pdb
  python scripts/prepare_pdb.py --prmtop complex.prmtop --pdb complex.pdb --ligand BEN --receptor-out receptor.pqr --ligand-out ligand.pqr
  python scripts/prepare_pdb.py --split-complex complex.pdb --receptor-out receptor.pdb --ligand-out ligand.pdb
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import numpy as np


ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}
METALS = {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "CAL"}
SKIP_HET = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "ACE", "NME", "Na+", "Cl-", "K+", "CAL"}


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


def deposited_centroid(entry, chain, resname, resid=None):
    """Centroid and atom count of a reference HETATM/ATOM group in a PDB file."""
    groups = {}
    with open(entry) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or line[17:20].strip() != resname:
                continue
            if (chain and line[21] != chain) or line[16] not in (" ", "A"):
                continue
            if resid is not None and int(line[22:26]) != resid:
                continue
            groups.setdefault(int(line[22:26]), []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    if not groups:
        sys.exit(f"ERROR: Chain {chain} of {entry} has no {resname} {resid or ''}.")
    if len(groups) > 1:
        sys.exit(f"ERROR: Chain {chain} of {entry} holds {resname} {sorted(groups)}. Specify --pocket-ligand-resid.")
    xyz = np.array(next(iter(groups.values())))
    return xyz.mean(axis=0), len(xyz)


def check_formula(mol, formula):
    """Verify RDKit molecule element counts against an expected formula string."""
    want = collections.Counter()
    for e, n in re.findall(r"([A-Z][a-z]?)\s*(\d*)", formula):
        want[e.upper()] += int(n) if n else 1
    got = collections.Counter(a.GetSymbol().upper() for a in mol.GetAtoms())
    if got != want:
        diff = {k: got.get(k, 0) - want.get(k, 0)
                for k in sorted(set(got) | set(want)) if got.get(k, 0) != want.get(k, 0)}
        sys.exit(f"ERROR: ligand composition off by {diff} against {formula}")


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


def _build_ligand_pdb(lig_lines, code, cif_path, prot_atoms, deprot_atoms, tmp_dir, expected_formula=""):
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
    if expected_formula:
        check_formula(mol, expected_formula)
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


def build_smiles_ligand_pdb(smiles, resname="LIG", seed=1, target_centroid=None, expected_formula=""):
    """Embed a 3D conformer from SMILES, shift onto target_centroid when given, and return HETATM lines."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit(f"ERROR: RDKit cannot read SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    if expected_formula:
        check_formula(mol, expected_formula)
    par = AllChem.ETKDG()
    par.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, par) != 0:
        sys.exit("ERROR: RDKit could not embed the SMILES conformer")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    if target_centroid is not None:
        shift = np.asarray(target_centroid, dtype=float) - coords.mean(axis=0)
        coords = coords + shift
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    counts = collections.Counter()
    rname = (resname or "LIG")[:3]
    for a in mol.GetAtoms():
        sym = a.GetSymbol().upper()
        counts[sym] += 1
        nm = f"{sym}{counts[sym]}"
        inf = Chem.AtomPDBResidueInfo()
        inf.SetName(f" {nm:<3s}" if len(nm) < 4 else nm[:4])
        inf.SetResidueName(f"{rname:>3s}")
        inf.SetResidueNumber(1)
        inf.SetChainId("L")
        inf.SetIsHeteroAtom(True)
        inf.SetOccupancy(1.0)
        a.SetPDBResidueInfo(inf)
    q_net = Chem.GetFormalCharge(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pdb_lines = [l + "\n" for l in Chem.MolToPDBBlock(mol, flavor=0).splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
    return pdb_lines, q_net, formula, mol.GetNumAtoms()


def split_complex_pdb(complex_path, receptor_out, ligand_out, ligand_resname=""):
    """Split an already-prepared complex PDB into receptor.pdb and ligand.pdb."""
    if not os.path.exists(complex_path):
        sys.exit(f"ERROR: {complex_path} not found")
    prot, metals, het = [], [], []
    with open(complex_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                prot.append(line if line.endswith("\n") else line + "\n")
            elif line.startswith("HETATM"):
                resn = line[17:20].strip()
                if resn.upper() in METALS and resn != ligand_resname:
                    metals.append(line if line.endswith("\n") else line + "\n")
                elif resn not in SKIP_HET:
                    het.append(line if line.endswith("\n") else line + "\n")
    groups = {}
    for l in het:
        groups.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), []).append(l)
    if not groups:
        sys.exit(f"ERROR: no ligand HETATM in {complex_path}")
    key = next(iter(groups))
    if len(groups) > 1:
        pick = [k for k in groups if k[2] == ligand_resname]
        if len(pick) != 1:
            sys.exit(f"ERROR: multiple HETATM groups in {complex_path}; pass --ligand <RESNAME>")
        key = pick[0]
    lig = groups[key]
    os.makedirs(os.path.dirname(os.path.abspath(receptor_out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ligand_out)) or ".", exist_ok=True)
    with open(receptor_out, "w") as fh:
        fh.writelines(prot)
        fh.write("TER\n")
        for m in metals:
            fh.write(m)
            fh.write("TER\n")
        fh.write("END\n")
    with open(ligand_out, "w") as fh:
        fh.writelines(lig)
        fh.write("END\n")
    print(f"Split {complex_path} -> {receptor_out} ({len(prot)} protein + {len(metals)} metal atoms), "
          f"{ligand_out} ({len(lig)} ligand atoms)")


def prmtop_to_pqrs(prmtop_path, pdb_path, ligand_resname, receptor_out, ligand_out):
    """Export receptor.pqr and ligand.pqr from an AMBER prmtop + coordinate PDB."""
    for p in (prmtop_path, pdb_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found")
    if not ligand_resname:
        sys.exit("ERROR: --prmtop requires --ligand <RESNAME> to split the ligand residue")
    top = os.path.abspath(prmtop_path)
    pdb = os.path.abspath(pdb_path)
    with tempfile.TemporaryDirectory() as tmp:
        rst = os.path.join(tmp, "_full.rst7")
        subprocess.run(
            "cpptraj",
            input=f"parm {top}\ntrajin {pdb}\ntrajout {rst}\nrun\n",
            cwd=tmp, capture_output=True, text=True,
        )
        if not os.path.exists(rst):
            sys.exit(f"ERROR: cpptraj failed on {prmtop_path} + {pdb_path}")
        r = subprocess.run(["ambpdb", "-p", top, "-c", rst, "-pqr"],
                           cwd=tmp, capture_output=True, text=True)
    rec, lig = [], []
    for line in r.stdout.strip().split("\n"):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        res = line[17:20].strip()
        if res == ligand_resname:
            lig.append(line)
        elif res not in SKIP_HET:
            rec.append(line)
    if not lig:
        sys.exit(f"ERROR: no residue {ligand_resname} in {prmtop_path}")
    if not rec:
        sys.exit(f"ERROR: no receptor atoms remaining in {prmtop_path}")
    for out_path, rows in ((receptor_out, rec), (ligand_out, lig)):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("\n".join(rows) + "\nEND\n")
        q = sum(float(l.split()[8]) for l in rows)
        print(f"Wrote {out_path}: {len(rows)} atoms, Q={q:+.1f} e")


def _csv_list(val, cast=str):
    if not val:
        return []
    return [cast(x.strip()) for x in val.split(",") if x.strip()]


def _write_receptor_pdb(path, prot_lines, kept_het_lines, tail_lines=()):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as out:
        for l in prot_lines:
            out.write(l if l.endswith("\n") else l + "\n")
        out.write("TER\n")
        for l in kept_het_lines:
            out.write(l if l.endswith("\n") else l + "\n")
            if l[17:20].strip().upper() in METALS:
                out.write("TER\n")
        for l in tail_lines:
            out.write(l if l.endswith("\n") else l + "\n")
        out.write("END\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", help="Local input PDB file")
    ap.add_argument("--pdb-id", help="4-character RCSB PDB identifier to download when --pdb is omitted")
    ap.add_argument("--prmtop", default="", help="AMBER topology (.prmtop/.parm7) to export into receptor and ligand PQRs")
    ap.add_argument("--split-complex", default="", help="Existing prepared complex PDB to split into receptor and ligand PDBs")
    ap.add_argument("--receptor-out", default="", help="Output path for the prepared receptor (.pdb or .pqr)")
    ap.add_argument("--ligand-out", default="", help="Output path for the prepared ligand (.pdb or .pqr)")
    ap.add_argument("-o", "--output", default="", help="Single output PDB path when writing one combined file")
    ap.add_argument("--chain", default="", help="Receptor chain ID to retain (blank keeps all chains)")
    ap.add_argument("--ligand-chain", default="", help="Protein ligand chain ID to extract into --ligand-out")
    ap.add_argument("--resid-min", type=int, default=None, help="Lowest receptor residue number retained")
    ap.add_argument("--resid-max", type=int, default=None, help="Highest receptor residue number retained")
    ap.add_argument("--altloc", default="A", help="Preferred protein conformer indicator (default: A)")
    ap.add_argument("--ligand", default="", help="3-letter residue name of the crystal or prmtop ligand")
    ap.add_argument("--ligand-resid", type=int, default=None, help="Ligand residue number when multiple copies exist")
    ap.add_argument("--ligand-altloc", default="", help="Preferred ligand conformer indicator (defaults to --altloc)")
    ap.add_argument("--ccd", default="", help="CCD component code for crystal ligand bond orders (defaults to --ligand)")
    ap.add_argument("--cif", default="", help="Optional local .cif path for the ligand CCD definition")
    ap.add_argument("--protonate", default="", help="Comma-separated ligand N/O/S atom names gaining one proton")
    ap.add_argument("--deprotonate", default="", help="Comma-separated ligand N/O/S atom names losing one proton")
    ap.add_argument("--smiles", default="", help="Ligand SMILES string to embed as a 3D conformer")
    ap.add_argument("--ligand-resname", default="LIG", help="3-letter residue name assigned to a SMILES ligand")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for RDKit ETKDG conformer embedding")
    ap.add_argument("--pocket-ligand", default="", help="Reference HETATM residue name marking the pocket centroid for --smiles")
    ap.add_argument("--pocket-ligand-resid", type=int, default=None, help="Residue number of --pocket-ligand")
    ap.add_argument("--formula", default="", help="Optional expected molecular formula check for the ligand")
    ap.add_argument("--keep-hetatm", default="", help="Comma-separated HETATM residue names kept with the receptor")
    ap.add_argument("--hid", default="", help="Comma-separated receptor residue numbers set to HID")
    ap.add_argument("--hie", default="", help="Comma-separated receptor residue numbers set to HIE")
    ap.add_argument("--hip", default="", help="Comma-separated receptor residue numbers set to HIP")
    ap.add_argument("--metal-his", action="store_true", help="Assign HID/HIE on histidines coordinating retained metals")
    args = ap.parse_args(argv)

    if args.split_complex:
        rec_out = args.receptor_out or "receptor.pdb"
        lig_out = args.ligand_out or "ligand.pdb"
        split_complex_pdb(args.split_complex, rec_out, lig_out, args.ligand)
        return

    if args.prmtop:
        if not args.pdb:
            ap.error("--prmtop requires --pdb <coord.pdb>")
        rec_out = args.receptor_out or "receptor.pqr"
        lig_out = args.ligand_out or "ligand.pqr"
        prmtop_to_pqrs(args.prmtop, args.pdb, args.ligand, rec_out, lig_out)
        return

    if not args.pdb and not args.pdb_id:
        ap.error("Specify --pdb, --pdb-id, --split-complex, or --prmtop")

    ref_out = args.receptor_out or args.output or "receptor.pdb"
    out_dir = os.path.dirname(os.path.abspath(ref_out)) or "."
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
        pocket_center = None
        if args.smiles and args.pocket_ligand:
            pocket_center, _ = deposited_centroid(
                pdb_path, args.chain or None, args.pocket_ligand, args.pocket_ligand_resid
            )
    finally:
        if downloaded_pdb and os.path.exists(downloaded_pdb):
            os.remove(downloaded_pdb)

    keep_het = set(_csv_list(args.keep_hetatm))
    if args.pocket_ligand:
        keep_het.discard(args.pocket_ligand)
    prot_raw, kept_het_raw, lig_raw, lig_chain_raw = [], [], [], []
    for l in raw_lines:
        if l.startswith("ATOM") and len(l) > 26:
            if l[12:16].strip() == "OXT":
                continue
            if args.ligand_chain and l[21] == args.ligand_chain:
                lig_chain_raw.append(l)
                continue
            if args.chain and l[21] != args.chain:
                continue
            rid = int(l[22:26])
            if args.resid_min is not None and rid < args.resid_min:
                continue
            if args.resid_max is not None and rid > args.resid_max:
                continue
            prot_raw.append(l)
        elif l.startswith("HETATM") and len(l) > 26:
            resn = l[17:20].strip()
            if args.ligand and resn == args.ligand:
                if args.ligand_resid is None or int(l[22:26]) == args.ligand_resid:
                    lig_raw.append(l)
            if resn in keep_het and (not args.chain or l[21] == args.chain):
                kept_het_raw.append(l)

    if not prot_raw and not (args.ligand_chain and lig_chain_raw and not args.receptor_out and not args.output):
        sys.exit("ERROR: no matching receptor ATOM records in input PDB")

    prot_lines = _select_conformer(prot_raw, preferred=args.altloc) if prot_raw else []
    kept_het_lines = _select_conformer(kept_het_raw, preferred=args.altloc)
    metal_lines = [l for l in kept_het_lines if l[17:20].strip().upper() in METALS]

    if prot_lines:
        prot_lines, his_applied = _apply_his_tautomers(
            prot_lines, metal_lines,
            set(_csv_list(args.hid, int)),
            set(_csv_list(args.hie, int)),
            set(_csv_list(args.hip, int)),
            args.metal_his,
        )
    else:
        his_applied = {}

    lig_out_lines = []
    lig_msg = "none"
    if sum(bool(x) for x in (args.ligand, args.smiles, args.ligand_chain)) > 1:
        sys.exit("ERROR: choose one of --ligand, --smiles, or --ligand-chain")
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
            groups[key], ccd_code, args.cif or None, prot_atoms, deprot_atoms, out_dir, args.formula
        )
        lig_msg = f"{args.ligand} ({n_lig} atoms, {formula}, Q={q_net:+d} e)"
    elif args.smiles:
        lig_out_lines, q_net, formula, n_lig = build_smiles_ligand_pdb(
            args.smiles, resname=args.ligand_resname, seed=args.seed,
            target_centroid=pocket_center, expected_formula=args.formula,
        )
        lig_msg = f"{args.ligand_resname} from SMILES ({n_lig} atoms, {formula}, Q={q_net:+d} e)"
    elif args.ligand_chain:
        if not lig_chain_raw:
            sys.exit(f"ERROR: no ATOM records for --ligand-chain {args.ligand_chain!r}")
        lig_out_lines = _select_conformer(lig_chain_raw, preferred=args.altloc)
        lig_msg = f"chain {args.ligand_chain} ({len(lig_out_lines)} atoms)"

    if args.output and not args.receptor_out and not args.ligand_out:
        _write_receptor_pdb(args.output, prot_lines, kept_het_lines, tail_lines=lig_out_lines)
        target_desc = args.output
    else:
        neither = not args.receptor_out and not args.ligand_out
        rec_out = args.receptor_out or ("receptor.pdb" if neither and prot_lines else "")
        if rec_out and prot_lines:
            _write_receptor_pdb(rec_out, prot_lines, kept_het_lines)
        lig_out = args.ligand_out or ("ligand.pdb" if neither and lig_out_lines else "")
        if lig_out and lig_out_lines:
            os.makedirs(os.path.dirname(os.path.abspath(lig_out)) or ".", exist_ok=True)
            with open(lig_out, "w") as out:
                for l in lig_out_lines:
                    out.write(l if l.endswith("\n") else l + "\n")
                out.write("END\n")
        target_desc = ", ".join(x for x in (rec_out if prot_lines else "", lig_out if lig_out_lines else "") if x)

    n_res = len({(l[21], l[22:27]) for l in prot_lines})
    his_str = ", ".join(f"{c}{r}:{t}" for (c, r), t in sorted(his_applied.items())) or "unchanged"
    print(f"Prepared {target_desc} -> {n_res} receptor residues ({len(prot_lines)} protein atoms, His {his_str}), "
          f"kept HETATM {len(kept_het_lines)} atoms, ligand {lig_msg}")


if __name__ == "__main__":
    main()
