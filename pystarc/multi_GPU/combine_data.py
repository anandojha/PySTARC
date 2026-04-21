#!/usr/bin/env python3
"""
Combine all output files from split PySTARC runs into bd_sims/.
Auto-detects bd_sims/bd_1, bd_sims/bd_2, ... directories.
Produces identical file formats as a single-GPU run.
"""

import numpy as np
import argparse
import json
import math
import csv
import glob
import os
import re


def main():
    parser = argparse.ArgumentParser(
        description="Combine split PySTARC k_on calculations"
    )
    parser.add_argument(
        "--bd-sims",
        default="bd_sims",
        help="Path to bd_sims directory (default: bd_sims)",
    )
    args = parser.parse_args()
    bd_sims = os.path.abspath(args.bd_sims)
    if not os.path.isdir(bd_sims):
        print(f"  Error: {bd_sims} not found.")
        return
    # Auto-detect bd_N directories
    pattern = re.compile(r"^bd_\d+$")
    subdirs = sorted(
        [
            os.path.join(bd_sims, d)
            for d in os.listdir(bd_sims)
            if os.path.isdir(os.path.join(bd_sims, d)) and pattern.match(d)
        ],
        key=lambda x: int(os.path.basename(x).split("_")[1]),
    )
    if not subdirs:
        print(f"  Error: no bd_N directories found in {bd_sims}")
        return
    # Load all results.json
    runs = []
    dirs_valid = []
    dirs_missing = []
    for d in subdirs:
        rj = os.path.join(d, "results.json")
        if not os.path.exists(rj):
            dirs_missing.append(os.path.basename(d))
            continue
        with open(rj) as f:
            runs.append(json.load(f))
        dirs_valid.append(d)
        print(
            f"  {os.path.basename(d)}: {runs[-1]['n_reacted']:,} reacted, {runs[-1]['n_escaped']:,} escaped"
        )
    if dirs_missing:
        print(f"  Skipped (not finished): {', '.join(dirs_missing)}")
    if not runs:
        print("  Error: no completed runs found")
        return
    # Pool counts
    nr = sum(r["n_reacted"] for r in runs)
    ne = sum(r["n_escaped"] for r in runs)
    n_max = sum(r.get("n_max_steps", 0) for r in runs)
    N = nr + ne
    P = nr / N if N > 0 else 0
    k_b = runs[0]["k_b"]
    D_rel = runs[0].get("D_rel", 0)
    r_start = runs[0].get("r_start", 0)
    r_escape = runs[0].get("r_escape", 0)
    CONV = 6.022e8
    k_on = CONV * k_b * P
    SE = math.sqrt(P * (1 - P) / N) if 0 < P < 1 else 0
    RSE = SE / P if P > 0 else float("inf")
    # Wilson 95% CI
    z = 1.96
    denom = 1 + z**2 / N
    centre = (P + z**2 / (2 * N)) / denom
    spread = z * math.sqrt(P * (1 - P) / N + z**2 / (4 * N**2)) / denom
    P_lo = max(0, centre - spread)
    P_hi = min(1, centre + spread)
    k_lo = CONV * k_b * P_lo
    k_hi = CONV * k_b * P_hi
    wall_time = sum(r.get("wall_time_sec", 0) for r in runs)
    total_steps = sum(
        r.get("steps_per_sec", 0) * r.get("wall_time_sec", 0) for r in runs
    )
    steps_sec = total_steps / wall_time if wall_time > 0 else 0
    if k_on > 0:
        exp = int(math.floor(math.log10(k_on)))
        man = k_on / 10**exp
        err = (k_hi - k_lo) / 2 / 10**exp
    else:
        exp, man, err = 0, 0, 0
    targets = {}
    if 0 < P < 1:
        for tol in [0.10, 0.05, 0.01]:
            targets[f"{int(tol*100)}%"] = int(math.ceil((1 - P) / (P * tol**2)))

    # Print summary
    print(f"\n  Combined results ({len(runs)}/{len(subdirs)} runs)")
    print(f"  N completed      = {N:,}")
    print(f"  Reacted          = {nr:,}")
    print(f"  Escaped          = {ne:,}")
    print(f"  Max-steps        = {n_max:,}")
    print(f"  P_rxn            = {P:.6f} ± {SE:.6f}")
    print(f"  Relative SE      = {RSE*100:.2f}%")
    print(f"  k_b              = {k_b:.4f} A3/ps")
    print(f"  k_on             = {k_on:.4e} M-1 s-1")
    print(f"  k_on (± error)     = ({man:.1f} ± {err:.1f}) x 10^{exp} M-1 s-1")
    print(f"  Wilson 95% CI    = [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1")
    if targets:
        print(f"  Trajectories needed:")
        for t, n in targets.items():
            status = "completed" if N >= n else "need more simulations"
            print(f"    For ±{t} RSE: {n:,} ({status})")
    print(f"  {'Converged' if RSE < 0.05 else 'Not converged'} (RSE {RSE*100:.2f}%)")
    # Save results.json
    results = {
        "k_on": k_on,
        "k_on_low": k_lo,
        "k_on_high": k_hi,
        "k_on_units": "M-1 s-1",
        "P_rxn": P,
        "P_rxn_low": P_lo,
        "P_rxn_high": P_hi,
        "k_b": k_b,
        "k_b_units": "A3/ps",
        "D_rel": D_rel,
        "D_rel_units": "A2/ps",
        "n_trajectories": N + n_max,
        "n_reacted": nr,
        "n_escaped": ne,
        "n_max_steps": n_max,
        "r_start": r_start,
        "r_escape": r_escape,
        "wall_time_sec": wall_time,
        "steps_per_sec": steps_sec,
        "confidence_level": 0.95,
        "log10_k_on": math.log10(k_on) if k_on > 0 else 0,
    }
    # Aggregate per-reaction fire counts from sub-runs when state-machine
    # mode was active. Each sub-run's results.json may carry a list of
    # reaction entries with name / n_fired / state_before / state_after.
    # Sum n_fired across sub-runs while preserving the reaction metadata.
    per_rxn = {}
    rxn_order = []
    for r in runs:
        for entry in r.get("completed_reactions", []):
            name = entry.get("name")
            if name is None:
                continue
            if name not in per_rxn:
                per_rxn[name] = {
                    "name": name,
                    "n_fired": 0,
                    "state_before": entry.get("state_before"),
                    "state_after": entry.get("state_after"),
                }
                rxn_order.append(name)
            per_rxn[name]["n_fired"] += int(entry.get("n_fired", 0))
    if per_rxn:
        results["completed_reactions"] = [per_rxn[n] for n in rxn_order]
        print("\n  Per-reaction firing counts (summed across sub-runs):")
        for entry in results["completed_reactions"]:
            print(
                f"    {entry['name']}: {entry['n_fired']:,} fires "
                f"({entry['state_before']} -> {entry['state_after']})"
            )
    _save_json(results, os.path.join(bd_sims, "results.json"))

    # Combine CSV files
    _concat_csv(dirs_valid, "trajectories.csv", bd_sims, reindex="traj_id")
    _concat_csv(dirs_valid, "encounters.csv", bd_sims, reindex="traj_id")
    _concat_csv(dirs_valid, "near_misses.csv", bd_sims, reindex="traj_id")
    _concat_csv(dirs_valid, "fpt_distribution.csv", bd_sims, reindex="traj_id")
    _concat_csv(dirs_valid, "pose_clusters.csv", bd_sims)
    _sum_csv(
        dirs_valid,
        "radial_density.csv",
        bd_sims,
        sum_col="count",
        recompute_col="density",
        total_N=N,
    )
    _sum_csv(
        dirs_valid,
        "contact_frequency.csv",
        bd_sims,
        sum_col="n_contacts",
        recompute_col="frequency",
        total_N=N,
    )
    _sum_csv(
        dirs_valid,
        "milestone_flux.csv",
        bd_sims,
        sum_cols=["flux_outward", "flux_inward", "net_flux"],
    )
    # Combine NPZ files
    _concat_npz(dirs_valid, "paths.npz", bd_sims, data_key="data", meta_key="columns")
    _concat_npz(
        dirs_valid, "energetics.npz", bd_sims, data_key="data", meta_key="columns"
    )
    _sum_npz(
        dirs_valid,
        "angular_map.npz",
        bd_sims,
        sum_key="counts",
        copy_keys=["theta_centers", "phi_centers"],
    )
    _sum_npz(
        dirs_valid,
        "transition_matrix.npz",
        bd_sims,
        sum_key="matrix",
        copy_keys=["milestones"],
    )
    _sum_npz(
        dirs_valid, "p_commit.npz", bd_sims, sum_key="counts", copy_keys=["milestones"]
    )
    # Convergence
    conv = {
        "N": N,
        "n_reacted": nr,
        "n_escaped": ne,
        "P_rxn": P,
        "SE": SE,
        "relative_SE": RSE,
        "relative_SE_pct": RSE * 100 if P > 0 else float("inf"),
        "k_on": k_on,
        "SE_kon": CONV * k_b * SE,
        "wilson_CI": [k_lo, k_hi],
        "wilson_CI_P": [P_lo, P_hi],
        "converged": RSE < 0.05 if P > 0 else False,
        "tol": 0.05,
        "tol_pct": 5.0,
        "N_needed": targets,
    }
    _save_json(conv, os.path.join(bd_sims, "convergence.json"))
    print(f"\n  All files saved -> {bd_sims}/")


# Helpers
def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    {os.path.basename(path)}")


def _concat_csv(dirs, filename, out_dir, reindex=None):
    rows = []
    offset = 0
    header = None
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            header = reader.fieldnames
            for row in reader:
                if reindex and reindex in row:
                    row[reindex] = str(int(row[reindex]) + offset)
                rows.append(row)
        if reindex:
            offset = len(rows)
    if not rows:
        return
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    size = os.path.getsize(out_path)
    label = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.1f} KB"
    print(f"    {filename} ({len(rows):,} rows, {label})")


def _sum_csv(
    dirs,
    filename,
    out_dir,
    sum_col=None,
    sum_cols=None,
    recompute_col=None,
    total_N=None,
):
    all_data = {}
    header = None
    key_cols = None
    cols_to_sum = [sum_col] if sum_col else (sum_cols or [])
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            header = reader.fieldnames
            if key_cols is None:
                key_cols = [
                    c for c in header if c not in cols_to_sum and c != recompute_col
                ]
            for row in reader:
                key = tuple(row[c] for c in key_cols)
                if key not in all_data:
                    all_data[key] = {c: row[c] for c in key_cols}
                    for sc in cols_to_sum:
                        all_data[key][sc] = 0.0
                for sc in cols_to_sum:
                    try:
                        all_data[key][sc] += float(row[sc])
                    except (ValueError, KeyError):
                        pass
    if not all_data:
        return
    rows = list(all_data.values())
    if recompute_col and sum_col and total_N and total_N > 0:
        for row in rows:
            row[recompute_col] = f"{float(row[sum_col]) / total_N:.8e}"
            row[sum_col] = int(float(row[sum_col]))
    elif sum_col:
        for row in rows:
            row[sum_col] = int(float(row[sum_col]))
    if sum_cols:
        for row in rows:
            for sc in sum_cols:
                if sc in row:
                    row[sc] = int(float(row[sc]))
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    {filename} ({len(rows):,} rows)")


def _concat_npz(dirs, filename, out_dir, data_key="data", meta_key="columns"):
    arrays = []
    meta = None
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath, allow_pickle=True)
        if data_key in npz:
            arrays.append(npz[data_key])
        if meta is None and meta_key in npz:
            meta = npz[meta_key]
    if not arrays:
        return
    combined = np.concatenate(arrays, axis=0)
    out_path = os.path.join(out_dir, filename)
    save_dict = {data_key: combined}
    if meta is not None:
        save_dict[meta_key] = meta
    np.savez(out_path, **save_dict)
    size = os.path.getsize(out_path)
    label = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.1f} KB"
    print(f"    {filename} ({combined.shape[0]:,} rows, {label})")


def _sum_npz(dirs, filename, out_dir, sum_key, copy_keys=None):
    total = None
    copies = {}
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath, allow_pickle=True)
        if sum_key in npz:
            arr = npz[sum_key]
            total = arr if total is None else total + arr
        if not copies and copy_keys:
            for ck in copy_keys:
                if ck in npz:
                    copies[ck] = npz[ck]
    if total is None:
        return
    out_path = os.path.join(out_dir, filename)
    save_dict = {sum_key: total}
    save_dict.update(copies)
    np.savez(out_path, **save_dict)
    print(f"    {filename} ({sum_key}: {total.shape})")


if __name__ == "__main__":
    main()
