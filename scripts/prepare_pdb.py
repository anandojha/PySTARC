#!/usr/bin/env python3
"""Standalone structure preparation script for any receptor-ligand complex.

Prepares separate receptor and ligand structure files (.pdb + .pdb or .pqr + .pqr)
and prints the <structure> block (with integer <ligand_net_charge>). Independent
of PySTARC runtime code.

Workflows:
  1. Crystal protein-ligand PDB (RCSB CCD .cif, single or multi-chain receptor):
     python scripts/prepare_pdb.py --pdb-id 1A9U --chain A --resid-min 4 --ligand SB2 --receptor-out receptor.pdb --ligand-out ligand.pdb
  2. Docked ligand from .sdf, .mol2, or non-CCD .pdb:
     python scripts/prepare_pdb.py --pdb receptor_raw.pdb --chain A,B --ligand-file docked.sdf --receptor-out receptor.pdb --ligand-out ligand.pdb
     python scripts/prepare_pdb.py --pdb docked_complex.pdb --chain A --ligand LIG --ligand-smiles "NC(=[NH2+])c1ccccc1" --receptor-out receptor.pdb --ligand-out ligand.pdb
  3. Protein-protein complex (single or multi-chain receptor/ligand, rigid or chain):
     python scripts/prepare_pdb.py --pdb 1BRS.pdb --chain A --ligand-chain D --model chain --receptor-out receptor.pdb --ligand-out ligand.pdb
  4. Receptor PDB plus 3D ligand embedded from SMILES:
     python scripts/prepare_pdb.py --pdb 3CZV.pdb --chain A --keep-hetatm ZN --metal-his --smiles "CC(=O)NC1=NN=C(S1)S([NH-])(=O)=O" --pocket-ligand AZM --pocket-ligand-resid 263 --receptor-out receptor.pdb --ligand-out ligand.pdb
  5. Split an existing complex PDB (small-molecule HETATM or --ligand-chain):
     python scripts/prepare_pdb.py --split-complex complex.pdb --ligand UNK --receptor-out receptor.pdb --ligand-out ligand.pdb
  6. Export receptor.pqr + ligand.pqr from AMBER prmtop + coordinate PDB:
     python scripts/prepare_pdb.py --prmtop complex.prmtop --pdb complex.pdb --ligand BEN --receptor-out receptor.pqr --ligand-out ligand.pqr
  7. Validate and export pre-parameterized receptor + ligand PQR files:
     python scripts/prepare_pdb.py --receptor-pqr t.pqr --ligand-pqr m.pqr --receptor-out receptor.pqr --ligand-out ligand.pqr
  8. Generate analytical charged-sphere receptor.pqr + ligand.pqr:
     python scripts/prepare_pdb.py --sphere-pair 1.0:1.0,-1.0:1.0 --receptor-out receptor.pqr --ligand-out ligand.pqr
"""

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import numpy as np


ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": 4, "DELO": 4}
METALS = {"ZN", "MG", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CA", "CAL", "CD", "HG", "SR", "BA", "AL"}
SKIP_HET = {"HOH", "WAT", "TIP3", "SPC", "NA", "CL", "K", "MG", "ZN", "CA", "CAL", "ACE", "NME", "Na+", "Cl-", "K+"}
RESIDUE_CHARGES = {"ARG": 1, "LYS": 1, "HIP": 1, "ASP": -1, "GLU": -1, "CYM": -1}
BACKBONE_KEEP = {"N", "CA", "C", "O", "CB"}
BACKBONE_GLY = {"N", "CA", "C", "O"}


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
        bonds, charges, elements, leaving, aromatic_atoms = {}, {}, {}, set(), set()
        for row in _cif_rows(path, "_chem_comp_atom"):
            nm = row.get("atom_id")
            if not nm or row.get("comp_id", code) != code:
                continue
            q = row.get("charge", "0")
            charges[nm] = int(q) if q.lstrip("+-").isdigit() else 0
            elements[nm] = row.get("type_symbol", "").capitalize()
            if row.get("pdbx_leaving_atom_flag") == "Y":
                leaving.add(nm)
            if row.get("pdbx_aromatic_flag") == "Y":
                aromatic_atoms.add(nm)
        for row in _cif_rows(path, "_chem_comp_bond"):
            if row.get("comp_id", code) != code:
                continue
            order = ORDERS.get(row.get("value_order", "").upper())
            if order is None:
                sys.exit(f"ERROR: unsupported bond order {row.get('value_order')!r} in {code}.cif")
            if row.get("pdbx_aromatic_flag") == "Y" and order == 4:
                aromatic_atoms.update((row["atom_id_1"], row["atom_id_2"]))
            bonds[frozenset((row["atom_id_1"], row["atom_id_2"]))] = order
        if not elements:
            sys.exit(f"ERROR: {path} contains no atoms for {code}")
        return bonds, charges, elements, leaving, aromatic_atoms
    finally:
        if downloaded and os.path.exists(path):
            os.remove(path)


def deposited_centroid(entry, chains, resname, resid=None):
    """Centroid and atom count of a reference HETATM/ATOM group in a PDB file."""
    chain_set = ({chains} if isinstance(chains, str) and chains
                 else (set(chains) if chains else set()))
    groups = {}
    with open(entry) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or line[17:20].strip() != resname:
                continue
            if (chain_set and line[21] not in chain_set) or line[16] not in (" ", "A"):
                continue
            if resid is not None and int(line[22:26]) != resid:
                continue
            groups.setdefault((line[21], int(line[22:26])), []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    if not groups:
        sys.exit(f"ERROR: {entry} has no {resname} {resid or ''} in chains {sorted(chain_set) or 'all'}.")
    if len(groups) > 1:
        sys.exit(f"ERROR: {entry} holds multiple {resname} copies {sorted(groups)}. Specify --pocket-ligand-resid.")
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
        grp = (l[21], l[22:27], l[17:20])
        alt = l[16]
        if alt == " ":
            continue
        if grp not in chosen or alt == preferred:
            if chosen.get(grp) != preferred:
                chosen[grp] = alt
    out = []
    for l in lines:
        grp = (l[21], l[22:27], l[17:20])
        alt = l[16]
        if alt == " " or alt == chosen.get(grp):
            out.append(l[:16] + " " + l[17:])
    return out


def _xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def _convert_mse_line(line):
    """Convert HETATM MSE selenomethionine record to standard ATOM MET record."""
    aname = line[12:16].strip()
    if aname == "SE":
        line = line[:12] + " SD " + line[16:]
        if len(line) >= 78:
            line = line[:76] + " S" + line[78:]
    return "ATOM  " + line[6:17] + "MET" + line[20:]


def _apply_mutations(prot_lines, mut_specs):
    """Truncate mutated residues to backbone (+ CB except GLY) and rename for tleap sidechain rebuild."""
    if not mut_specs:
        return prot_lines, []
    parsed = {}
    for spec in mut_specs:
        parts = [p.strip() for p in spec.split(":") if p.strip()]
        if len(parts) == 2:
            parsed[(None, parts[0])] = parts[1].upper()
        elif len(parts) == 3:
            parsed[(parts[0], parts[1])] = parts[2].upper()
        else:
            sys.exit(f"ERROR: --mutate expects RESID:RESNAME or CHAIN:RESID:RESNAME, got {spec!r}")
    out, applied = [], []
    seen = set()
    for l in prot_lines:
        ch = l[21]
        rcode = l[22:27].strip()
        rnum = l[22:26].strip()
        old_res = l[17:20].strip()
        new_res = (parsed.get((ch, rcode)) or parsed.get((ch, rnum))
                   or parsed.get((None, rcode)) or parsed.get((None, rnum)))
        if new_res is None:
            out.append(l)
            continue
        aname = l[12:16].strip()
        keep_atoms = BACKBONE_GLY if new_res == "GLY" else BACKBONE_KEEP
        if aname not in keep_atoms:
            continue
        out.append(f"{l[:17]}{new_res:>3s}{l[20:]}")
        tag = (ch, rcode, old_res, new_res)
        if tag not in seen:
            seen.add(tag)
            applied.append(f"{ch.strip()}{rcode}:{old_res}->{new_res}")
    if len(seen) < len(parsed):
        sys.exit(f"ERROR: one or more --mutate targets {mut_specs} were absent from receptor ATOM records")
    if shutil.which("tleap") is not None:
        out = _complete_mutated_sidechains(out, {(t[0], t[1]) for t in seen})
    return out, applied


def _complete_mutated_sidechains(prot_lines, mutated_keys):
    """Build missing heavy sidechain atoms on mutated residues via tleap when installed."""
    with tempfile.TemporaryDirectory() as tmp:
        in_pdb = os.path.join(tmp, "in.pdb")
        with open(in_pdb, "w") as fh:
            _write_chain_separated_atoms(fh, prot_lines)
            fh.write("END\n")
        with open(os.path.join(tmp, "tleap.in"), "w") as fh:
            fh.write("source leaprc.protein.ff14SB\nmol = loadpdb in.pdb\nsavepdb mol out.pdb\nquit\n")
        subprocess.run(["tleap", "-f", "tleap.in"], cwd=tmp, capture_output=True, text=True)
        out_pdb = os.path.join(tmp, "out.pdb")
        if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
            return prot_lines
        leap_lines = [l for l in open(out_pdb) if l.startswith("ATOM")]

    orig_res_order = []
    orig_groups = {}
    for l in prot_lines:
        rk = (l[21], l[22:27].strip())
        if rk not in orig_groups:
            orig_res_order.append(rk)
            orig_groups[rk] = []
        orig_groups[rk].append(l if l.endswith("\n") else l + "\n")

    leap_res_order = []
    leap_groups = {}
    for l in leap_lines:
        an = l[12:16].strip()
        if an == "OXT" or an.startswith("H") or (len(an) > 1 and an[0].isdigit() and an[1] == "H"):
            continue
        lrk = int(l[22:26])
        if lrk not in leap_groups:
            leap_res_order.append(lrk)
            leap_groups[lrk] = []
        leap_groups[lrk].append(l)

    if len(orig_res_order) != len(leap_res_order):
        return prot_lines

    merged = []
    serial = 0
    for rk, lrk in zip(orig_res_order, leap_res_order):
        existing = orig_groups[rk]
        if rk not in mutated_keys:
            for l in existing:
                serial += 1
                merged.append(f"ATOM  {serial:5d}{l[11:]}")
            continue
        have_names = {l[12:16].strip() for l in existing}
        tmpl = existing[-1]
        for l in existing:
            serial += 1
            merged.append(f"ATOM  {serial:5d}{l[11:]}")
        for ll in leap_groups[lrk]:
            an = ll[12:16].strip()
            if an in have_names:
                continue
            serial += 1
            el = an[:1]
            merged.append(
                f"ATOM  {serial:5d} {ll[12:16]} {tmpl[17:27]}   "
                f"{ll[30:54]}{tmpl[54:66]}          {el:>2s}\n"
            )
    return merged


def _parse_res_targets(specs, chains):
    out = set()
    for s in specs:
        if ":" in s:
            ch, rc = s.split(":", 1)
            out.add((ch.strip(), rc.strip()))
        else:
            for ch in chains:
                out.add((ch, s.strip()))
    return out


def _apply_his_and_cyx(prot_lines, metal_lines, hid_specs, hie_specs, hip_specs, cyx_specs,
                       auto_metal, auto_cyx, metal_cutoff=2.6, ss_cutoff=2.2):
    taut_map = {}
    if auto_metal and metal_lines:
        m_xyz = [_xyz(l) for l in metal_lines]
        for l in prot_lines:
            resn = l[17:20].strip()
            aname = l[12:16].strip()
            if resn in ("HIS", "HID", "HIE", "HIP") and aname in ("ND1", "NE2"):
                rcode = l[22:27].strip()
                pos = _xyz(l)
                if min(float(np.linalg.norm(pos - m)) for m in m_xyz) <= metal_cutoff:
                    taut_map[(l[21], rcode)] = "HID" if aname == "NE2" else "HIE"
    chains = {l[21] for l in prot_lines}
    for rk in _parse_res_targets(hid_specs, chains):
        taut_map[rk] = "HID"
    for rk in _parse_res_targets(hie_specs, chains):
        taut_map[rk] = "HIE"
    for rk in _parse_res_targets(hip_specs, chains):
        taut_map[rk] = "HIP"

    ss_keys = set(_parse_res_targets(cyx_specs, chains))
    if auto_cyx:
        sgs = [(l[21], l[22:27].strip(), _xyz(l)) for l in prot_lines
               if l[17:20].strip() in ("CYS", "CYX") and l[12:16].strip() == "SG"]
        for i in range(len(sgs)):
            for j in range(i + 1, len(sgs)):
                if float(np.linalg.norm(sgs[i][2] - sgs[j][2])) <= ss_cutoff:
                    ss_keys.add((sgs[i][0], sgs[i][1]))
                    ss_keys.add((sgs[j][0], sgs[j][1]))

    updated, applied = [], {}
    for l in prot_lines:
        rk = (l[21], l[22:27].strip())
        cur = l[17:20].strip()
        if rk in taut_map and cur in ("HIS", "HID", "HIE", "HIP"):
            new_res = taut_map[rk]
            l = f"{l[:17]}{new_res:>3s}{l[20:]}"
            applied[rk] = new_res
        elif rk in ss_keys and cur in ("CYS", "CYX"):
            l = f"{l[:17]}CYX{l[20:]}"
            applied[rk] = "CYX"
        updated.append(l)
    return updated, applied


def _ensure_pdb_atom_names(mol, default_resname="LIG"):
    from rdkit import Chem

    rname = (default_resname or "LIG")[:3]
    used = set()
    counts = collections.Counter()
    ref = None
    for a in mol.GetAtoms():
        inf = a.GetPDBResidueInfo()
        if inf is not None and inf.GetName().strip():
            nm = inf.GetName().strip()
            if nm not in used:
                used.add(nm)
                if ref is None:
                    ref = inf
                continue
            a.SetPDBResidueInfo(None)
    for a in mol.GetAtoms():
        if a.GetPDBResidueInfo() is None:
            sym = a.GetSymbol().upper()
            counts[sym] += 1
            while f"{sym}{counts[sym]}" in used:
                counts[sym] += 1
            nm = f"{sym}{counts[sym]}"
            used.add(nm)
            inf = Chem.AtomPDBResidueInfo()
            inf.SetName(f" {nm:<3s}" if len(nm) < 4 else nm[:4])
            inf.SetResidueName(ref.GetResidueName() if ref else f"{rname:>3s}")
            inf.SetResidueNumber(ref.GetResidueNumber() if ref else 1)
            inf.SetChainId(ref.GetChainId() if ref else "L")
            inf.SetIsHeteroAtom(True)
            inf.SetOccupancy(1.0)
            a.SetPDBResidueInfo(inf)


def _apply_prot_deprot_and_hs(rw, idx, label, prot_atoms, deprot_atoms, expected_formula="", default_resname="LIG"):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    gone = []
    for flag, names, step in (("--protonate", prot_atoms, 1), ("--deprotonate", deprot_atoms, -1)):
        for nm in names:
            if nm not in idx:
                sys.exit(f"ERROR: {flag} atom {nm!r} absent from {label}")
            a = rw.GetAtomWithIdx(idx[nm])
            if a.GetAtomicNum() <= 1:
                sys.exit(f"ERROR: {flag} atom {nm!r} must be a heavy atom")
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
    _ensure_pdb_atom_names(mol, default_resname=default_resname)
    q_net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pdb_lines = [l + "\n" for l in Chem.MolToPDBBlock(mol, flavor=0).splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
    return pdb_lines, q_net, formula, mol.GetNumAtoms()


def _build_ligand_from_ccd(lig_lines, code, cif_path, prot_atoms, deprot_atoms, tmp_dir, expected_formula=""):
    from rdkit import Chem

    bonds, charges, elements, leaving, aromatic_atoms = load_ccd(code, cif_path=cif_path, cache_dir=tmp_dir)
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
    kinds = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for pair, order in bonds.items():
        a1, a2 = tuple(pair)
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], kinds[order])
            if order == 4:
                rw.GetAtomWithIdx(idx[a1]).SetIsAromatic(True)
                rw.GetAtomWithIdx(idx[a2]).SetIsAromatic(True)
    for a in rw.GetAtoms():
        nm = atom_name(a)
        a.SetFormalCharge(charges.get(nm, 0))
        if nm in aromatic_atoms and any(b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds()):
            a.SetIsAromatic(True)
    return _apply_prot_deprot_and_hs(rw, idx, code, prot_atoms, deprot_atoms, expected_formula, default_resname=code)


def _build_ligand_from_template_or_charge(lig_lines, code, template_smiles, net_charge, prot_atoms, deprot_atoms,
                                          expected_formula=""):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    block = "".join(lig_lines) + "END\n"
    if template_smiles:
        tmpl = Chem.MolFromSmiles(template_smiles)
        if tmpl is None:
            sys.exit(f"ERROR: RDKit cannot parse --ligand-smiles {template_smiles!r}")
        m_heavy = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
        if m_heavy is None:
            sys.exit(f"ERROR: RDKit could not parse ligand {code}")
        mol = AllChem.AssignBondOrdersFromTemplate(tmpl, m_heavy)
        rw = Chem.RWMol(mol)
    else:
        m = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        if m is None:
            sys.exit(f"ERROR: RDKit could not parse ligand {code}")
        rdDetermineBonds.DetermineBondOrders(m, charge=int(net_charge))
        rw = Chem.RWMol(m)
    _ensure_pdb_atom_names(rw, default_resname=code)
    idx = {a.GetPDBResidueInfo().GetName().strip(): a.GetIdx()
           for a in rw.GetAtoms() if a.GetPDBResidueInfo() is not None}
    return _apply_prot_deprot_and_hs(rw, idx, code, prot_atoms, deprot_atoms, expected_formula, default_resname=code)


def build_external_ligand_file(lig_file, resname="LIG", template_smiles="", net_charge=None,
                               prot_atoms=(), deprot_atoms=(), expected_formula=""):
    """Load an external .sdf, .mol2, or .pdb ligand file and return prepared HETATM lines."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    if not os.path.exists(lig_file):
        sys.exit(f"ERROR: --ligand-file {lig_file} not found")
    ext = os.path.splitext(lig_file)[1].lower()
    if ext in (".sdf", ".mol"):
        suppl = Chem.SDMolSupplier(lig_file, removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    elif ext == ".mol2":
        mol = Chem.MolFromMol2File(lig_file, removeHs=False, sanitize=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(lig_file, removeHs=(bool(template_smiles)), sanitize=False)
    else:
        sys.exit(f"ERROR: --ligand-file {lig_file!r} must end in .sdf, .mol, .mol2, or .pdb")
    if mol is None:
        sys.exit(f"ERROR: RDKit could not read --ligand-file {lig_file}")

    if template_smiles:
        tmpl = Chem.MolFromSmiles(template_smiles)
        if tmpl is None:
            sys.exit(f"ERROR: RDKit cannot parse --ligand-smiles {template_smiles!r}")
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
    elif ext == ".pdb":
        if net_charge is None:
            sys.exit("ERROR: --ligand-file .pdb requires --ligand-charge or --ligand-smiles")
        rdDetermineBonds.DetermineBondOrders(mol, charge=int(net_charge))

    rw = Chem.RWMol(mol)
    _ensure_pdb_atom_names(rw, default_resname=resname)
    idx = {a.GetPDBResidueInfo().GetName().strip(): a.GetIdx() for a in rw.GetAtoms()}
    return _apply_prot_deprot_and_hs(rw, idx, lig_file, prot_atoms, deprot_atoms, expected_formula,
                                     default_resname=resname)


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
    _ensure_pdb_atom_names(mol, default_resname=resname)
    q_net = Chem.GetFormalCharge(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pdb_lines = [l + "\n" for l in Chem.MolToPDBBlock(mol, flavor=0).splitlines()
                 if l.startswith(("ATOM", "HETATM"))]
    return pdb_lines, q_net, formula, mol.GetNumAtoms()


def _chain_net_charge(atom_lines):
    seen = {}
    for l in atom_lines:
        if l.startswith("ATOM") and len(l) >= 27:
            seen.setdefault((l[21], l[22:27].strip()), l[17:20].strip())
    return sum(RESIDUE_CHARGES.get(rn, 0) for rn in seen.values())


def _write_chain_separated_atoms(out, atom_lines):
    prev_chain = None
    for l in atom_lines:
        ch = l[21] if len(l) > 21 else " "
        if prev_chain is not None and ch != prev_chain:
            out.write("TER\n")
        prev_chain = ch
        out.write(l if l.endswith("\n") else l + "\n")
    if prev_chain is not None:
        out.write("TER\n")


def _write_receptor_pdb(path, prot_lines, kept_het_lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as out:
        _write_chain_separated_atoms(out, prot_lines)
        for l in kept_het_lines:
            out.write(l if l.endswith("\n") else l + "\n")
            if l[17:20].strip().upper() in METALS:
                out.write("TER\n")
        out.write("END\n")


def split_complex_pdb(complex_path, receptor_out, ligand_out, ligand_resname="", ligand_chains=(),
                      keep_hetatm=(), ligand_charge=None, model="rigid"):
    """Split an existing complex PDB into receptor.pdb and ligand.pdb (small molecule or --ligand-chain)."""
    if not os.path.exists(complex_path):
        sys.exit(f"ERROR: {complex_path} not found")
    lchain_set = set(ligand_chains)
    keep_set = {x.upper() for x in keep_hetatm}
    prot, lig_prot, metals, het = [], [], [], []
    with open(complex_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                row = line if line.endswith("\n") else line + "\n"
                if lchain_set and line[21] in lchain_set:
                    lig_prot.append(row)
                else:
                    prot.append(row)
            elif line.startswith("HETATM"):
                resn = line[17:20].strip()
                row = line if line.endswith("\n") else line + "\n"
                if resn.upper() in keep_set or (resn.upper() in METALS and resn != ligand_resname):
                    metals.append(row)
                elif resn not in SKIP_HET:
                    het.append(row)
    if lchain_set:
        if not lig_prot:
            sys.exit(f"ERROR: no ATOM records for --ligand-chain {sorted(lchain_set)} in {complex_path}")
        lig = lig_prot
    else:
        groups = {}
        for l in het:
            groups.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), []).append(l)
        if not groups:
            sys.exit(f"ERROR: no ligand HETATM in {complex_path} (pass --ligand-chain for protein-protein PDBs)")
        chosen = next(iter(groups))
        if len(groups) > 1:
            pick = [k for k in groups if k[2] == ligand_resname]
            if len(pick) != 1:
                sys.exit(f"ERROR: multiple HETATM groups in {complex_path}, pass --ligand <RESNAME>")
            chosen = pick[0]
        lig = groups[chosen]
    os.makedirs(os.path.dirname(os.path.abspath(receptor_out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ligand_out)) or ".", exist_ok=True)
    _write_receptor_pdb(receptor_out, prot, metals)
    with open(ligand_out, "w") as fh:
        if lchain_set:
            _write_chain_separated_atoms(fh, lig)
        else:
            fh.writelines(lig)
        fh.write("END\n")
    if ligand_charge is not None:
        q_net = int(ligand_charge)
    elif lchain_set:
        q_net = _chain_net_charge(lig)
    else:
        q_net = 0
    print(f"# Split {complex_path} -> {receptor_out} ({len(prot)} protein + {len(metals)} metal atoms), "
          f"{ligand_out} ({len(lig)} ligand atoms)", file=sys.stderr)
    _print_structure_snippet(receptor_out, ligand_out, q_net, model)


def _pqr_atom_charge(line):
    """Extract partial charge from a fixed-width or whitespace-delimited PQR ATOM/HETATM line."""
    if len(line) >= 54:
        try:
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])
            tail = line[54:].split()
            if len(tail) >= 2:
                return float(tail[0])
        except ValueError:
            pass
    toks = line.split()
    if len(toks) < 9:
        sys.exit(f"ERROR: malformed PQR record: {line!r}")
    idx = -3 if toks[-1].isalpha() else -2
    return float(toks[idx])


def prepare_pqr_pair(rec_in, lig_in, receptor_out, ligand_out, model="rigid"):
    """Validate and export a pre-parameterized receptor + ligand PQR pair and print <structure>."""
    for p in (rec_in, lig_in):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found")
        if not p.lower().endswith(".pqr"):
            sys.exit(f"ERROR: expected a .pqr input, got {p}")
    q_lig = 0.0
    for src, dst in ((rec_in, receptor_out), (lig_in, ligand_out)):
        with open(src) as fh:
            raw = fh.read()
        rows = [l for l in raw.splitlines() if l.startswith(("ATOM", "HETATM"))]
        if not rows:
            sys.exit(f"ERROR: {src} contains zero ATOM/HETATM records")
        q = sum(_pqr_atom_charge(l) for l in rows)
        if dst == ligand_out:
            q_lig = q
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        print(f"# Prepared {dst}: {len(rows)} atoms, Q={q:+.2f} e", file=sys.stderr)
    _print_structure_snippet(receptor_out, ligand_out, round(q_lig), model)


def write_sphere_pair(spec, receptor_out, ligand_out):
    """Write single-atom charged-sphere receptor.pqr and ligand.pqr from Q_REC:R_REC,Q_LIG:R_LIG."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2 or any(":" not in p for p in parts):
        sys.exit(f"ERROR: --sphere-pair expects Q_REC:R_REC,Q_LIG:R_LIG (for example 1.0:1.0,-1.0:1.0), got {spec!r}")
    (q_rec, r_rec), (q_lig, r_lig) = (tuple(float(x) for x in p.split(":", 1)) for p in parts)
    if r_rec <= 0.0 or r_lig <= 0.0:
        sys.exit("ERROR: sphere radii must exceed zero")
    for dst, q, r in ((receptor_out, q_rec, r_rec), (ligand_out, q_lig, r_lig)):
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        with open(dst, "w") as fh:
            fh.write(f"ATOM      1  U   FOO    1     0.0  0.0  0.0  {q:4.1f} {r:3.1f}\n")
        print(f"# Wrote {dst}: 1 atom, Q={q:+.2f} e, R={r:.2f} A", file=sys.stderr)
    _print_structure_snippet(receptor_out, ligand_out, round(q_lig), "rigid")


def _parse_resid_set(spec):
    out = set()
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            lo, hi = p.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(p))
    return out


def prmtop_to_pqrs(prmtop_path, pdb_path, ligand_resname, ligand_residues, receptor_out, ligand_out):
    """Export receptor.pqr and ligand.pqr from an AMBER prmtop + coordinate PDB."""
    for p in (prmtop_path, pdb_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found")
    if not ligand_resname and not ligand_residues:
        sys.exit("ERROR: --prmtop requires --ligand <RESNAME> or --ligand-residues <START-END>")
    lig_rset = _parse_resid_set(ligand_residues) if ligand_residues else set()
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
        rid = int(line[22:26])
        if (lig_rset and rid in lig_rset) or (ligand_resname and res == ligand_resname):
            lig.append(line)
        elif res not in SKIP_HET:
            rec.append(line)
    if not lig:
        sys.exit(f"ERROR: no matching ligand atoms in {prmtop_path}")
    if not rec:
        sys.exit(f"ERROR: no receptor atoms remaining in {prmtop_path}")
    q_lig = 0.0
    for out_path, rows in ((receptor_out, rec), (ligand_out, lig)):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("\n".join(rows) + "\nEND\n")
        q = sum(_pqr_atom_charge(l) for l in rows)
        if out_path == ligand_out:
            q_lig = q
        print(f"# Wrote {out_path}: {len(rows)} atoms, Q={q:+.2f} e", file=sys.stderr)
    _print_structure_snippet(receptor_out, ligand_out, round(q_lig), "rigid")


def _csv_list(val, cast=str):
    if not val:
        return []
    return [cast(x.strip()) for x in val.split(",") if x.strip()]


def _print_structure_snippet(rec_out, lig_out, net_q, model):
    if not rec_out or not lig_out or net_q is None:
        return
    print(
        "  <structure>\n"
        f"    <receptor>{os.path.basename(rec_out)}</receptor>\n"
        f"    <ligand>{os.path.basename(lig_out)}</ligand>\n"
        f"    <ligand_net_charge>{int(net_q)}</ligand_net_charge>\n"
        f"    <model>{model}</model>\n"
        "  </structure>"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", help="Local input PDB file")
    ap.add_argument("--pdb-id", help="4-character RCSB PDB identifier to download when --pdb is omitted")
    ap.add_argument("--prmtop", default="", help="AMBER topology (.prmtop/.parm7) to export into receptor and ligand PQRs")
    ap.add_argument("--split-complex", default="", help="Existing prepared complex PDB to split into receptor and ligand PDBs")
    ap.add_argument("--receptor-pqr", default="", help="Existing pre-parameterized receptor PQR paired with --ligand-pqr")
    ap.add_argument("--ligand-pqr", default="", help="Existing pre-parameterized ligand PQR paired with --receptor-pqr")
    ap.add_argument("--sphere-pair", default="", help="Generate charged-sphere PQRs from Q_REC:R_REC,Q_LIG:R_LIG (for example 1.0:1.0,-1.0:1.0)")
    ap.add_argument("--receptor-out", default="", help="Output path for the prepared receptor (.pdb or .pqr)")
    ap.add_argument("--ligand-out", default="", help="Output path for the prepared ligand (.pdb or .pqr)")
    ap.add_argument("--chain", default="", help="Comma-separated receptor chain IDs to retain (blank keeps all chains)")
    ap.add_argument("--ligand-chain", default="", help="Comma-separated protein ligand chain IDs to extract into --ligand-out")
    ap.add_argument("--model", choices=("rigid", "chain"), default="rigid", help="Model tag printed in <structure> snippet")
    ap.add_argument("--resid-min", type=int, default=None, help="Lowest receptor residue number retained")
    ap.add_argument("--resid-max", type=int, default=None, help="Highest receptor residue number retained")
    ap.add_argument("--altloc", default="A", help="Preferred protein conformer indicator (default: A)")
    ap.add_argument("--mse-to-met", action="store_true", help="Convert selenomethionine HETATM MSE residues to ATOM MET")
    ap.add_argument("--mutate", default="", help="Comma-separated receptor point mutations RESID:RESNAME or CHAIN:RESID:RESNAME")
    ap.add_argument("--ligand", default="", help="3-letter residue name of the crystal, docked, or prmtop ligand")
    ap.add_argument("--ligand-residues", default="", help="Ligand residue numbers/ranges (e.g. 109-197) for protein-protein --prmtop")
    ap.add_argument("--ligand-resid", type=int, default=None, help="Ligand residue number when multiple copies exist")
    ap.add_argument("--ligand-altloc", default="", help="Preferred ligand conformer indicator (defaults to --altloc)")
    ap.add_argument("--ligand-file", default="", help="External docked or 3D ligand file (.sdf, .mol, .mol2, or .pdb)")
    ap.add_argument("--ccd", default="", help="CCD component code for crystal ligand bond orders (defaults to --ligand)")
    ap.add_argument("--cif", default="", help="Optional local .cif path for the ligand CCD definition")
    ap.add_argument("--ligand-smiles", default="", help="SMILES bond-order template for a non-CCD docked ligand")
    ap.add_argument("--ligand-charge", type=int, default=None, help="Formal net charge for a non-CCD .pdb ligand with explicit Hs")
    ap.add_argument("--protonate", default="", help="Comma-separated ligand heavy-atom names gaining one proton")
    ap.add_argument("--deprotonate", default="", help="Comma-separated ligand heavy-atom names losing one proton")
    ap.add_argument("--smiles", default="", help="Ligand SMILES string to embed as a 3D conformer")
    ap.add_argument("--ligand-resname", default="LIG", help="3-letter residue name assigned to a SMILES or external ligand")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for RDKit ETKDG conformer embedding")
    ap.add_argument("--pocket-ligand", default="", help="Reference HETATM residue name marking the pocket centroid for --smiles")
    ap.add_argument("--pocket-ligand-resid", type=int, default=None, help="Residue number of --pocket-ligand")
    ap.add_argument("--formula", default="", help="Optional expected molecular formula check for the ligand")
    ap.add_argument("--keep-hetatm", default="", help="Comma-separated HETATM residue names kept with the receptor")
    ap.add_argument("--hid", default="", help="Comma-separated receptor residue numbers (or CHAIN:RESID) set to HID")
    ap.add_argument("--hie", default="", help="Comma-separated receptor residue numbers (or CHAIN:RESID) set to HIE")
    ap.add_argument("--hip", default="", help="Comma-separated receptor residue numbers (or CHAIN:RESID) set to HIP")
    ap.add_argument("--cyx", default="", help="Comma-separated receptor cysteine numbers (or CHAIN:RESID) set to CYX")
    ap.add_argument("--metal-his", action="store_true", help="Assign HID/HIE on histidines coordinating retained metals")
    ap.add_argument("--auto-cyx", action="store_true", help="Assign CYX on receptor cysteines with SG-SG <= 2.2 A")
    args = ap.parse_args(argv)

    rec_chains = set(_csv_list(args.chain))
    lig_chains = set(_csv_list(args.ligand_chain))

    if args.sphere_pair:
        rec_out = args.receptor_out or "receptor.pqr"
        lig_out = args.ligand_out or "ligand.pqr"
        write_sphere_pair(args.sphere_pair, rec_out, lig_out)
        return

    if args.receptor_pqr or args.ligand_pqr:
        if not (args.receptor_pqr and args.ligand_pqr):
            ap.error("--receptor-pqr and --ligand-pqr must be passed together")
        rec_out = args.receptor_out or "receptor.pqr"
        lig_out = args.ligand_out or "ligand.pqr"
        prepare_pqr_pair(args.receptor_pqr, args.ligand_pqr, rec_out, lig_out, args.model)
        return

    if args.split_complex:
        rec_out = args.receptor_out or "receptor.pdb"
        lig_out = args.ligand_out or "ligand.pdb"
        split_complex_pdb(
            args.split_complex, rec_out, lig_out, args.ligand, lig_chains,
            keep_hetatm=_csv_list(args.keep_hetatm), ligand_charge=args.ligand_charge, model=args.model,
        )
        return

    if args.prmtop:
        if not args.pdb:
            ap.error("--prmtop requires --pdb <coord.pdb>")
        rec_out = args.receptor_out or "receptor.pqr"
        lig_out = args.ligand_out or "ligand.pqr"
        prmtop_to_pqrs(args.prmtop, args.pdb, args.ligand, args.ligand_residues, rec_out, lig_out)
        return

    if not args.pdb and not args.pdb_id and not args.ligand_file and not args.smiles:
        ap.error("Specify --pdb, --pdb-id, --ligand-file, --smiles, --split-complex, --prmtop, --receptor-pqr/--ligand-pqr, or --sphere-pair")

    ref_out = args.receptor_out or args.ligand_out or "receptor.pdb"
    out_dir = os.path.dirname(os.path.abspath(ref_out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    downloaded_pdb = None
    pdb_path = args.pdb
    if not pdb_path and args.pdb_id:
        pid = args.pdb_id.upper()
        pdb_path = os.path.join(out_dir, f"_{pid}_raw.pdb")
        _download(f"https://files.rcsb.org/download/{pid}.pdb", pdb_path, f"{pid}.pdb")
        downloaded_pdb = pdb_path

    raw_lines = []
    pocket_center = None
    if pdb_path:
        try:
            with open(pdb_path) as fh:
                raw_lines = fh.readlines()
            if args.smiles and args.pocket_ligand:
                pocket_center, _ = deposited_centroid(
                    pdb_path, rec_chains, args.pocket_ligand, args.pocket_ligand_resid
                )
        finally:
            if downloaded_pdb and os.path.exists(downloaded_pdb):
                os.remove(downloaded_pdb)

    keep_het = set(_csv_list(args.keep_hetatm))
    if args.pocket_ligand:
        keep_het.discard(args.pocket_ligand)
    prot_raw, kept_het_raw, lig_raw, lig_chain_raw = [], [], [], []
    for l in raw_lines:
        if l.startswith("HETATM") and args.mse_to_met and len(l) > 26 and l[17:20].strip() == "MSE":
            l = _convert_mse_line(l)
        if l.startswith("ATOM") and len(l) > 26:
            if l[12:16].strip() == "OXT":
                continue
            if lig_chains and l[21] in lig_chains:
                lig_chain_raw.append(l)
                continue
            if rec_chains and l[21] not in rec_chains:
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
            if resn in keep_het and (not rec_chains or l[21] in rec_chains):
                kept_het_raw.append(l)

    if pdb_path and not prot_raw and not (lig_chains and lig_chain_raw and not args.receptor_out):
        sys.exit("ERROR: no matching receptor ATOM records in input PDB")

    prot_lines = _select_conformer(prot_raw, preferred=args.altloc) if prot_raw else []
    kept_het_lines = _select_conformer(kept_het_raw, preferred=args.altloc)
    metal_lines = [l for l in kept_het_lines if l[17:20].strip().upper() in METALS]

    mut_applied = []
    if prot_lines and args.mutate:
        prot_lines, mut_applied = _apply_mutations(prot_lines, _csv_list(args.mutate))

    if prot_lines:
        prot_lines, his_applied = _apply_his_and_cyx(
            prot_lines, metal_lines,
            _csv_list(args.hid),
            _csv_list(args.hie),
            _csv_list(args.hip),
            _csv_list(args.cyx),
            args.metal_his,
            args.auto_cyx,
        )
    else:
        his_applied = {}

    lig_sources = sum(bool(x) for x in (args.ligand, args.ligand_file, args.smiles, lig_chains))
    if lig_sources > 1:
        sys.exit("ERROR: choose one of --ligand, --ligand-file, --smiles, or --ligand-chain")
    prot_atoms = _csv_list(args.protonate)
    deprot_atoms = _csv_list(args.deprotonate)
    if set(prot_atoms) & set(deprot_atoms):
        sys.exit(f"ERROR: overlap between --protonate and --deprotonate: {sorted(set(prot_atoms) & set(deprot_atoms))}")

    lig_out_lines = []
    lig_msg = "none"
    q_net = None
    if args.ligand:
        lig_alt = args.ligand_altloc or args.altloc
        lig_conf = _select_conformer(lig_raw, preferred=lig_alt)
        groups = {}
        for l in lig_conf:
            groups.setdefault((l[21], l[22:27].strip(), l[17:20].strip()), []).append(l)
        if not groups:
            sys.exit(f"ERROR: ligand {args.ligand!r} absent from {pdb_path}")
        chosen = next((k for k in groups if rec_chains and k[0] in rec_chains), next(iter(groups)))
        if args.ligand_smiles or args.ligand_charge is not None:
            lig_out_lines, q_net, formula, n_lig = _build_ligand_from_template_or_charge(
                groups[chosen], args.ligand, args.ligand_smiles, args.ligand_charge,
                prot_atoms, deprot_atoms, args.formula,
            )
        else:
            ccd_code = args.ccd or args.ligand
            lig_out_lines, q_net, formula, n_lig = _build_ligand_from_ccd(
                groups[chosen], ccd_code, args.cif or None, prot_atoms, deprot_atoms, out_dir, args.formula
            )
        lig_msg = f"{args.ligand} ({n_lig} atoms, {formula}, Q={q_net:+d} e)"
    elif args.ligand_file:
        lig_out_lines, q_net, formula, n_lig = build_external_ligand_file(
            args.ligand_file, resname=args.ligand_resname, template_smiles=args.ligand_smiles,
            net_charge=args.ligand_charge, prot_atoms=prot_atoms, deprot_atoms=deprot_atoms,
            expected_formula=args.formula,
        )
        lig_msg = f"{args.ligand_resname} from {os.path.basename(args.ligand_file)} ({n_lig} atoms, {formula}, Q={q_net:+d} e)"
    elif args.smiles:
        lig_out_lines, q_net, formula, n_lig = build_smiles_ligand_pdb(
            args.smiles, resname=args.ligand_resname, seed=args.seed,
            target_centroid=pocket_center, expected_formula=args.formula,
        )
        lig_msg = f"{args.ligand_resname} from SMILES ({n_lig} atoms, {formula}, Q={q_net:+d} e)"
    elif lig_chains:
        if not lig_chain_raw:
            sys.exit(f"ERROR: no ATOM records for --ligand-chain {sorted(lig_chains)}")
        lig_out_lines = _select_conformer(lig_chain_raw, preferred=args.altloc)
        q_net = _chain_net_charge(lig_out_lines)
        lig_msg = f"chain {','.join(sorted(lig_chains))} ({len(lig_out_lines)} atoms, Q={q_net:+d} e)"

    neither = not args.receptor_out and not args.ligand_out
    rec_out = args.receptor_out or ("receptor.pdb" if neither and prot_lines else "")
    if rec_out and prot_lines:
        _write_receptor_pdb(rec_out, prot_lines, kept_het_lines)
    lig_out = args.ligand_out or ("ligand.pdb" if neither and lig_out_lines else "")
    if lig_out and lig_out_lines:
        os.makedirs(os.path.dirname(os.path.abspath(lig_out)) or ".", exist_ok=True)
        with open(lig_out, "w") as out:
            if lig_chains:
                _write_chain_separated_atoms(out, lig_out_lines)
            else:
                for l in lig_out_lines:
                    out.write(l if l.endswith("\n") else l + "\n")
            out.write("END\n")
    target_desc = ", ".join(x for x in (rec_out if prot_lines else "", lig_out if lig_out_lines else "") if x)

    n_res = len({(l[21], l[22:27]) for l in prot_lines})
    his_str = ", ".join(f"{c.strip()}{r}:{t}" for (c, r), t in sorted(his_applied.items())) or "unchanged"
    mut_str = f", mutations {', '.join(mut_applied)}" if mut_applied else ""
    print(f"# Prepared {target_desc} -> {n_res} receptor residues ({len(prot_lines)} protein atoms, "
          f"states {his_str}{mut_str}), kept HETATM {len(kept_het_lines)} atoms, ligand {lig_msg}", file=sys.stderr)
    _print_structure_snippet(rec_out, lig_out, q_net, args.model)


if __name__ == "__main__":
    main()
