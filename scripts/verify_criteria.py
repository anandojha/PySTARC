#!/usr/bin/env python3
"""Verify that every scripts/*/prepare.sh criterion matches examples/*/config.xml.

Runs scripts/find_reaction_criteria.py with the exact flags written in each
rigid complex prepare.sh script, checks exact equality against the <criterion>
block in examples/*/config.xml, and prints the binding-site geometry audit
table across all 32 rigid benchmark complexes (plus the flexible-chain entry).

Usage:
    python scripts/verify_criteria.py
    python scripts/verify_criteria.py --check-accessibility
"""

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import find_reaction_criteria as frc  # noqa: E402


def _normalize_criterion(xml_text):
    root = ET.fromstring(xml_text)
    crit = root if root.tag == "criterion" else root.find("criterion")
    mode = crit.findtext("mode").strip()
    cutoff = float(crit.findtext("pair_cutoff").strip())
    n_needed = int(crit.findtext("n_needed").strip())
    rec = [
        (e.attrib["resid"], e.attrib["atom"], e.attrib["resname"])
        for e in crit.findall("receptor/target")
    ]
    lig = [tuple(sorted(e.attrib.items())) for e in crit.findall("ligand/*")]
    return (mode, cutoff, n_needed, rec, lig)


def _extract_frc_args(prepare_sh, rec_path, lig_path):
    text = prepare_sh.read_text()
    match = re.search(r'find_reaction_criteria\.py"\s*\\\s*(.*?)\)', text, re.S)
    if not match:
        raise ValueError(f"Missing find_reaction_criteria.py call in {prepare_sh}")
    tokens = match.group(1).replace("\\\n", " ").split()
    args = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx].strip('"')
        if tok == "--receptor":
            args.extend(["--receptor", str(rec_path)])
            idx += 2
        elif tok == "--ligand":
            args.extend(["--ligand", str(lig_path)])
            idx += 2
        else:
            args.append(tok)
            idx += 1
    return args


def _site_category(mode, cli_args):
    if mode == "centroid":
        return "Centroid (host/sphere)"
    if "--receptor-targets" in cli_args:
        return "Deep site (exposed rim)"
    return "Accessible site (polar)"


def _criterion_summary(rel, mode, cutoff, n_needed, args):
    if mode == "centroid":
        return f"Centroid ({cutoff:.1f} A)"
    if "--receptor-targets" not in args:
        return f"{n_needed}-of-6 polar ({cutoff:.1f} A)"
    if "hsp90_inhibitors" in rel:
        return f"{n_needed}-of-6 exposed tips ({cutoff:.1f} A)"
    return f"{n_needed}-of-6 exposed CB ({cutoff:.1f} A)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--check-accessibility",
        action="store_true",
        help="Run the NumPy rigid-body steric accessibility audit on each contact complex",
    )
    args = ap.parse_args(argv)

    frc_path = SCRIPTS_DIR / "find_reaction_criteria.py"
    configs = sorted((REPO_DIR / "examples").rglob("config.xml"))

    header = (
        f"{'Complex':47s} | {'Category':24s} | {'N_rec':>5s} | {'N_lig':>5s} | "
        f"{'Shld':>4s} | {'Criterion':27s} | {'N_traj':>10s} | {'Status'}"
    )
    print(header)
    print("-" * len(header))

    n_rigid = 0
    n_chain = 0
    mismatches = 0

    for cfg in configs:
        rel = str(cfg.relative_to(REPO_DIR / "examples").parent)
        root = ET.parse(cfg).getroot()
        model = root.findtext("structure/model").strip()
        n_traj = int(root.findtext("simulation/n_trajectories").strip())
        cutoff = float(root.findtext("criterion/pair_cutoff").strip())
        n_needed = int(root.findtext("criterion/n_needed").strip())
        rec_path = cfg.parent / root.findtext("structure/receptor").strip()
        lig_path = cfg.parent / root.findtext("structure/ligand").strip()

        if model == "chain":
            n_chain += 1
            rec = frc.parse_structure_atoms(str(rec_path))
            lig = frc.parse_structure_atoms(str(lig_path))
            geom = frc.compute_pocket_geometry(rec, lig)
            print(
                f"{rel:47s} | {'Chain (COFFDROP CPU)':24s} | {geom['n_rec_heavy']:5d} | "
                f"{geom['n_lig_heavy']:5d} | {geom['f_shielded']:4.2f} | "
                f"{f'Chain contacts ({cutoff:.1f} A)':27s} | {n_traj:10,d} | CHAIN"
            )
            continue

        n_rigid += 1
        prep_sh = SCRIPTS_DIR / rel / "prepare.sh"
        cli_args = _extract_frc_args(prep_sh, rec_path, lig_path)
        run_args = list(cli_args)
        if args.check_accessibility and "--mode" not in cli_args:
            run_args.append("--check-accessibility")

        proc = subprocess.run(
            [sys.executable, str(frc_path)] + run_args,
            capture_output=True,
            text=True,
            check=True,
        )
        cfg_norm = _normalize_criterion(cfg.read_text())
        gen_norm = _normalize_criterion(proc.stdout)
        ok = cfg_norm == gen_norm
        if not ok:
            mismatches += 1

        rec = frc.parse_structure_atoms(str(rec_path))
        lig = frc.parse_structure_atoms(str(lig_path))
        geom = frc.compute_pocket_geometry(rec, lig)
        mode = cfg_norm[0]
        cat_label = _site_category(mode, cli_args)
        crit_label = _criterion_summary(rel, mode, cutoff, n_needed, cli_args)
        status = "OK" if ok else "MISMATCH"
        if args.check_accessibility and mode == "contacts":
            acc_lines = [l for l in proc.stderr.splitlines() if "Rigid accessibility" in l]
            if acc_lines:
                m_cut = re.search(r"smallest accessible integer cutoff=([^ ]+ A)", acc_lines[0])
                if m_cut:
                    status = f"{status} (acc: {m_cut.group(1)})"

        print(
            f"{rel:47s} | {cat_label:24s} | {geom['n_rec_heavy']:5d} | "
            f"{geom['n_lig_heavy']:5d} | {geom['f_shielded']:4.2f} | "
            f"{crit_label:27s} | {n_traj:10,d} | {status}"
        )

    print("-" * len(header))
    print(f"Verified {n_rigid} rigid complexes ({mismatches} mismatches) and {n_chain} flexible-chain complex.")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
