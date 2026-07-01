#!/usr/bin/env python3
"""
Combine the output files from a split PySTARC run back into a single bd_sims
directory. The script automatically detects the per-shard directories
bd_sims/bd_1, bd_sims/bd_2, and so on, and writes combined files in exactly the
same formats that a single-GPU run would produce.
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
    """Detect the per-shard bd_N directories, pool their results, and write the
    combined output files. This recomputes the reaction probability and the
    association rate constant k_on from the pooled trajectory counts, reports the
    Wilson confidence interval and convergence, and merges the per-shard CSV and
    NPZ files into a single set in the bd_sims directory."""
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
    # Find every bd_N shard directory and sort them by the integer N.
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
    # Read the results.json from each shard, recording which shards have
    # finished and which are still missing their output.
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
    # Pool the trajectory counts across all completed shards. The reaction
    # probability P is the fraction of trajectories that reacted, and the
    # association rate constant follows from
    #     k_on = CONV × k_b × P,
    # where k_b is the rate at which trajectories cross the b-surface and CONV
    # converts from Å³/ps to M⁻¹ s⁻¹.
    nr = sum(r["n_reacted"] for r in runs)
    ne = sum(r["n_escaped"] for r in runs)
    n_max = sum(r.get("n_max_steps", 0) for r in runs)
    N = nr + ne
    # Total recorded contact steps across shards, the denominator for the pooled
    # contact_frequency column (frequency = contacts / recorded steps, not
    # contacts / trajectory count). Prefer the value carried in results.json;
    # fall back to recovering it from the per-shard contact_frequency.csv for
    # runs written before that field existed.
    total_contact_steps = sum(int(r.get("contact_total_steps", 0)) for r in runs)
    if total_contact_steps == 0:
        total_contact_steps = _recover_contact_steps(dirs_valid)
    P = nr / N if N > 0 else 0
    k_b = runs[0]["k_b"]
    D_rel = runs[0].get("D_rel", 0)
    r_start = runs[0].get("r_start", 0)
    r_escape = runs[0].get("r_escape", 0)
    # The pooled k_on, k_b, and geometry are taken from the first shard on the
    # assumption that every shard ran the same physical system. Verify that
    # assumption and warn if any later shard reports a different value for these
    # quantities, since pooling counts across physically different setups would
    # give a meaningless rate constant. The first-shard values are still used so
    # consistent runs are unaffected.
    _warn_run_mismatch(runs)
    CONV = 6.022e8
    k_on = CONV * k_b * P
    SE = math.sqrt(P * (1 - P) / N) if 0 < P < 1 else 0
    RSE = SE / P if P > 0 else float("inf")
    # Wilson 95% confidence interval for the reaction probability. The audit fix
    # of 2026-05-21 added a guard for the N=0 case and clamped the argument of
    # the square root to be non-negative.
    z = 1.96
    if N == 0:
        P_lo, P_hi = 0.0, 1.0
    else:
        denom = 1 + z**2 / N
        centre = (P + z**2 / (2 * N)) / denom
        sqrt_arg = max(P * (1 - P) / N + z**2 / (4 * N**2), 0.0)
        spread = z * math.sqrt(sqrt_arg) / denom
        P_lo = max(0, centre - spread)
        P_hi = min(1, centre + spread)
    k_lo = CONV * k_b * P_lo
    k_hi = CONV * k_b * P_hi
    wall_time = sum(r.get("wall_time_sec", 0) for r in runs)
    total_steps = sum(
        r.get("steps_per_sec", 0) * r.get("wall_time_sec", 0) for r in runs
    )
    steps_sec = total_steps / wall_time if wall_time > 0 else 0
    # Split k_on into a mantissa and a power of ten so it can be printed in
    # scientific notation together with its half-width error bar.
    if k_on > 0:
        exp = int(math.floor(math.log10(k_on)))
        man = k_on / 10**exp
        err = (k_hi - k_lo) / 2 / 10**exp
    else:
        exp, man, err = 0, 0, 0
    # Estimate how many trajectories would be needed to reach a given relative
    # standard error using N ≈ (1 - P) / (P × tol²) for each tolerance.
    targets = {}
    if 0 < P < 1:
        for tol in [0.10, 0.05, 0.01]:
            targets[f"{int(tol*100)}%"] = int(math.ceil((1 - P) / (P * tol**2)))

    # Print a human-readable summary of the combined run.
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
    # Assemble the combined results dictionary that will be written to disk.
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
    # When the simulation ran in state-machine mode, each sub-run records how
    # many times every reaction fired. A sub-run's results.json may carry a list
    # of reaction entries, each with a name, a fire count n_fired, and the states
    # before and after the reaction. Here we sum n_fired across all sub-runs
    # while keeping the reaction metadata and the original ordering.
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

    # Combine the per-shard CSV files. Tables of one row per trajectory are
    # concatenated with their trajectory ids reindexed so they stay unique,
    # while histogram-style tables are summed bin by bin.
    # The reindex offset for each shard is the total number of trajectories run
    # by the earlier shards, so a trajectory keeps one consistent id across every
    # combined table and archive.
    traj_offsets = []
    running = 0
    for r in runs:
        traj_offsets.append(running)
        running += int(r.get("n_trajectories", 0))
    _concat_csv(
        dirs_valid, "trajectories.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _concat_csv(
        dirs_valid, "encounters.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _concat_csv(
        dirs_valid, "near_misses.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _concat_csv(
        dirs_valid,
        "fpt_distribution.csv",
        bd_sims,
        reindex="traj_id",
        offsets=traj_offsets,
    )
    _concat_csv(dirs_valid, "pose_clusters.csv", bd_sims)
    _sum_csv(
        dirs_valid,
        "radial_density.csv",
        bd_sims,
        sum_col="count",
        recompute_col="density",
        density_mode=True,
    )
    _sum_csv(
        dirs_valid,
        "contact_frequency.csv",
        bd_sims,
        sum_col="n_contacts",
        recompute_col="frequency",
        total_N=total_contact_steps,
    )
    _sum_csv(
        dirs_valid,
        "milestone_flux.csv",
        bd_sims,
        sum_cols=["flux_outward", "flux_inward", "net_flux"],
    )
    # Combine the per-shard NPZ archives, concatenating the trajectory data
    # arrays and summing the binned counts and matrices.
    _concat_npz(
        dirs_valid,
        "paths.npz",
        bd_sims,
        data_key="data",
        meta_key="columns",
        reindex_col="traj_id",
        offsets=traj_offsets,
    )
    _concat_npz(
        dirs_valid,
        "energetics.npz",
        bd_sims,
        data_key="data",
        meta_key="columns",
        reindex_col="traj_id",
        offsets=traj_offsets,
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
        sum_key="counts",
        copy_keys=["bins"],
    )
    _pool_p_commit(dirs_valid, "p_commit.npz", bd_sims)
    # Write a separate convergence report summarising the pooled statistics and
    # whether the relative standard error has dropped below the 5% tolerance.
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


def _warn_run_mismatch(runs):
    """Warn if shards disagree on the physical parameters used for pooling.

    The combined k_on uses k_b and the geometry r_start and r_escape from the
    first completed shard, and pooling the trajectory counts is only valid when
    every shard simulated the same system. This routine compares k_b, D_rel,
    r_start, and r_escape across all shards against the first shard and prints a
    warning for each quantity that differs. Floating-point values are compared
    with a small relative and absolute tolerance so that ordinary round-trip
    noise in the JSON does not trigger a false warning. The first-shard values
    are still used by the caller, so consistent runs are unaffected.
    """
    if len(runs) < 2:
        return
    ref = runs[0]
    checks = [
        ("k_b", ref.get("k_b")),
        ("D_rel", ref.get("D_rel")),
        ("r_start", ref.get("r_start")),
        ("r_escape", ref.get("r_escape")),
    ]
    for key, ref_val in checks:
        if ref_val is None:
            continue
        for i, r in enumerate(runs[1:], start=1):
            val = r.get(key)
            if val is None:
                continue
            try:
                mismatch = not math.isclose(
                    float(val), float(ref_val), rel_tol=1e-9, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                mismatch = val != ref_val
            if mismatch:
                print(
                    f"  Warning: shard {i + 1} has {key}={val} but shard 1 has "
                    f"{key}={ref_val}; pooling assumes identical runs and uses "
                    f"the shard 1 value."
                )
                break


def _save_json(data, path):
    """Write a dictionary to a JSON file and report its basename."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    {os.path.basename(path)}")


def _concat_csv(dirs, filename, out_dir, reindex=None, offsets=None):
    """Concatenate the same CSV file from several shard directories into one.

    The rows from every shard are stacked in order. When a reindex column is
    given (typically the trajectory id), its values are offset shard by shard so
    that every trajectory keeps a unique id in the combined table. The offset for
    shard i is offsets[i], the total number of trajectories run by the earlier
    shards, which keeps the ids consistent with the other combined tables and
    archives regardless of how many rows each shard contributes to this file.
    Shards that do not contain the file are skipped, and nothing is written if no
    rows are found.
    """
    rows = []
    header = None
    for i, d in enumerate(dirs):
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        offset = offsets[i] if (reindex and offsets is not None) else 0
        with open(fpath) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            header = reader.fieldnames
            for row in reader:
                if reindex and reindex in row:
                    row[reindex] = str(int(row[reindex]) + offset)
                rows.append(row)
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


def _recover_contact_steps(dirs):
    """Recover the summed contact_total_steps across shards from the per-shard
    contact_frequency.csv files.

    A fallback for runs whose results.json predates the contact_total_steps
    field. Each shard wrote frequency = n_contacts / total_steps, so a shard's
    total_steps is recovered as round(n_contacts / frequency) from any row with a
    non-zero count, and the recovered per-shard totals are summed.
    """
    total = 0
    for d in dirs:
        fpath = os.path.join(d, "contact_frequency.csv")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                try:
                    n = float(row["n_contacts"])
                    fr = float(row["frequency"])
                except (KeyError, ValueError):
                    continue
                if n > 0 and fr > 0:
                    total += int(round(n / fr))
                    break
    return total


def _sum_csv(
    dirs,
    filename,
    out_dir,
    sum_col=None,
    sum_cols=None,
    recompute_col=None,
    total_N=None,
    density_mode=False,
):
    """Sum histogram-style CSV files across shards bin by bin.

    Rows are grouped by every column that is not being summed, so identical bins
    from different shards are merged, and the columns named in sum_col or
    sum_cols are added together.

    When recompute_col is given, that normalised column is rebuilt from the
    pooled counts so it stays consistent with the single-run writer:

      - density_mode=True renormalises a radial density as
        count / (total_count * shell_volume), where total_count is the sum over
        all bins and shell_volume = 4/3 pi (r_high^3 - r_low^3) read from the
        r_low and r_high columns, matching output_writer.py's density.
      - otherwise the column is recomputed as count / total_N, where total_N is
        the pooled denominator supplied by the caller (the recorded step count
        for contact_frequency, not the trajectory count).
    """
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
    if recompute_col and sum_col:
        if density_mode:
            total_count = sum(float(r[sum_col]) for r in rows)
            for row in rows:
                try:
                    r_low = float(row["r_low"])
                    r_high = float(row["r_high"])
                    vol = 4.0 / 3.0 * math.pi * (r_high**3 - r_low**3)
                except (KeyError, ValueError):
                    vol = 0.0
                cnt = float(row[sum_col])
                dens = (
                    cnt / (total_count * vol)
                    if (total_count > 0 and vol > 0)
                    else 0.0
                )
                row[recompute_col] = f"{dens:.8e}"
                row[sum_col] = int(cnt)
        else:
            denom = total_N if (total_N and total_N > 0) else 0
            for row in rows:
                cnt = float(row[sum_col])
                val = cnt / denom if denom > 0 else 0.0
                row[recompute_col] = f"{val:.8e}"
                row[sum_col] = int(cnt)
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


def _concat_npz(
    dirs,
    filename,
    out_dir,
    data_key="data",
    meta_key="columns",
    reindex_col=None,
    offsets=None,
):
    """Concatenate the data array from the same NPZ file across shards.

    The array stored under data_key is read from every shard and stacked along
    its first axis. The metadata stored under meta_key (for example the column
    names) is taken from the first shard that has it and carried through
    unchanged, since it is the same for every shard. When reindex_col names one
    of the columns (typically the trajectory id), the values in that column are
    offset by offsets[i] for shard i, so the ids match the combined CSV tables.
    """
    arrays = []
    meta = None
    for i, d in enumerate(dirs):
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath, allow_pickle=True)
        if meta is None and meta_key in npz:
            meta = npz[meta_key]
        if data_key in npz:
            arr = np.array(npz[data_key])
            if reindex_col is not None and offsets is not None and meta_key in npz:
                cols = [str(c) for c in npz[meta_key]]
                if reindex_col in cols and arr.ndim == 2:
                    arr[:, cols.index(reindex_col)] += offsets[i]
            arrays.append(arr)
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
    """Sum the array stored under sum_key across shards element by element.

    The arrays from every shard (for example counts, a transition matrix, or
    commitment counts) are added together. Any arrays named in copy_keys, such as
    the bin centres or milestone labels, are copied from the first shard that has
    them, since they are shared across shards.
    """
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


def _pool_p_commit(dirs, filename, out_dir):
    """Pool the commitment probability across shards by summing counts.

    Each shard's p_commit.npz stores the per-bin reaction probability p_commit
    and the per-bin sample count n_samples over a shared set of radial bins
    r_bins. Probabilities cannot be averaged across shards, so the per-bin
    reacted count is recovered as p_commit times n_samples, the reacted counts
    and the sample counts are each summed across shards, and the pooled
    probability is the summed reacted count divided by the summed sample count.
    """
    r_bins = None
    reacted = None
    n_total = None
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath, allow_pickle=True)
        if "p_commit" not in npz or "n_samples" not in npz:
            continue
        n = np.array(npz["n_samples"], dtype=float)
        rc = np.array(npz["p_commit"], dtype=float) * n
        reacted = rc if reacted is None else reacted + rc
        n_total = n if n_total is None else n_total + n
        if r_bins is None and "r_bins" in npz:
            r_bins = npz["r_bins"]
    if n_total is None:
        return
    pooled = np.divide(reacted, n_total, out=np.zeros_like(reacted), where=n_total > 0)
    out_path = os.path.join(out_dir, filename)
    save_dict = {"p_commit": pooled, "n_samples": n_total}
    if r_bins is not None:
        save_dict["r_bins"] = r_bins
    np.savez(out_path, **save_dict)
    print(f"    {filename} (p_commit: {pooled.shape})")


if __name__ == "__main__":
    main()
