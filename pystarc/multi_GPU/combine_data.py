#!/usr/bin/env python3
"""
Combine the output files from a split PySTARC run back into a single bd_sims
directory. The script pools the shards that split.json lists (or, without it,
every bd_N directory) and writes combined files in exactly the same formats
that a single-GPU run would produce. It exits with code 1 on errors.
"""

import os
import sys

_PYSTARC_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PYSTARC_ROOT not in sys.path:
    sys.path.insert(0, _PYSTARC_ROOT)

import numpy as np
from pystarc.global_defs.constants import MOLAR_RATE_CONV
import statistics
import argparse
import json
import math
import csv
import glob
import re


def _p_rxn_bounds(n_reacted, n_escaped, n_max):
    """Pooled P_rxn with every trajectory at the step limit counted as not
    reacted (the reported value), and with every one counted as reacted."""
    n = n_reacted + n_escaped + n_max
    if n == 0:
        return 0.0, 1.0
    return n_reacted / n, (n_reacted + n_max) / n


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
        return 1
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
    # The splitter's split.json names the shards of this run and their sizes.
    split = None
    split_path = os.path.join(bd_sims, "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = {s["dir"]: s for s in json.load(f)["shards"]}
        left_out = [
            os.path.basename(d) for d in subdirs if os.path.basename(d) not in split
        ]
        if left_out:
            print(f"  Left out (not in split.json): {', '.join(left_out)}")
        subdirs = [os.path.join(bd_sims, d) for d in split]
    if not subdirs:
        print(f"  Error: no bd_N directories found in {bd_sims}")
        return 1
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
        _cur = runs[-1]
        _csum = _cur.get("summary", {}) if isinstance(_cur.get("summary"), dict) else {}
        _nt = (
            _cur["n_trajectories"]
            if "n_trajectories" in _cur
            else _csum["n_trajectories"]
        )
        _nr_s = _cur["n_reacted"] if "n_reacted" in _cur else _csum["n_reacted"]
        _ne_s = _cur["n_escaped"] if "n_escaped" in _cur else _csum["n_escaped"]
        want = split[os.path.basename(d)]["n_trajectories"] if split else None
        if want is not None and _nt != want:
            print(
                f"  Error: {os.path.basename(d)} ran {_nt:,} "
                f"trajectories, split.json gives {want:,}."
            )
            return 1
        print(f"  {os.path.basename(d)}: {_nr_s:,} reacted, {_ne_s:,} escaped")
    if dirs_missing:
        print(f"  Skipped (not finished): {', '.join(dirs_missing)}")
    if not runs:
        print("  Error: no completed runs found")
        return 1
    if "chain" in runs[0] and isinstance(runs[0].get("summary"), dict):
        return _combine_chain_runs(runs, dirs_valid, subdirs, bd_sims)
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
    # A trajectory at the step limit counts as not reacted, so the denominator
    # of P_rxn is every trajectory that was run.
    N_p = N + n_max
    # Total recorded contact steps across shards, the denominator for the pooled
    # contact_frequency column (frequency = contacts / recorded steps, not
    # contacts / trajectory count). Prefer the value carried in results.json;
    # fall back to recovering it from the per-shard contact_frequency.csv for
    # runs written before that field existed.
    total_contact_steps = 0
    _missing_steps_dirs = []
    for _r, _d in zip(runs, dirs_valid):
        _s = int(_r.get("contact_total_steps", 0))
        if _s > 0:
            total_contact_steps += _s
        else:
            _missing_steps_dirs.append(_d)
    # Recover the step count per-shard for any shard whose results.json predates
    # the contact_total_steps field, so a mix of new and legacy shards does not
    # drop the legacy shards' steps from the pooled contact-frequency denominator.
    if _missing_steps_dirs:
        total_contact_steps += _recover_contact_steps(_missing_steps_dirs)
    P = nr / N_p if N_p > 0 else 0
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
    conf = runs[0].get("confidence_level", 0.95)
    conv_tol = runs[0].get("convergence_tol", 0.05)
    CONV = MOLAR_RATE_CONV
    k_on = CONV * k_b * P
    SE = math.sqrt(P * (1 - P) / N_p) if 0 < P < 1 else 0
    RSE = SE / P if P > 0 else float("inf")
    # Wilson interval for the reaction probability at the shards' confidence
    # level, with z = 1.96 at 0.95 as in convergence.json.
    z = (
        1.96
        if abs(conf - 0.95) < 1e-9
        else statistics.NormalDist().inv_cdf(0.5 + conf / 2)
    )
    if N_p == 0:
        P_lo, P_hi = 0.0, 1.0
    else:
        denom = 1 + z**2 / N_p
        centre = (P + z**2 / (2 * N_p)) / denom
        sqrt_arg = max(P * (1 - P) / N_p + z**2 / (4 * N_p**2), 0.0)
        spread = z * math.sqrt(sqrt_arg) / denom
        P_lo = max(0, centre - spread)
        P_hi = min(1, centre + spread)
    k_lo = CONV * k_b * P_lo
    k_hi = CONV * k_b * P_hi
    # Range of P_rxn and k_on between counting every trajectory at the step
    # limit as not reacted (the reported value) and as reacted.
    P_b_lo, P_b_hi = _p_rxn_bounds(nr, ne, n_max)
    k_bound_lo = CONV * k_b * P_b_lo
    k_bound_hi = CONV * k_b * P_b_hi
    # Shards run concurrently, so the wall clock is the slowest shard, not the sum.
    shard_walls = [r.get("wall_time_sec", 0) for r in runs]
    wall_time = max(shard_walls) if shard_walls else 0
    aggregate_gpu_sec = sum(shard_walls)  # total GPU seconds across shards
    # Aggregate throughput is all shard steps over the shared wall clock.
    total_steps = sum(
        r.get("total_steps") or r.get("steps_per_sec", 0) * r.get("wall_time_sec", 0)
        for r in runs
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
        for t in [0.10, 0.05, 0.01]:
            targets[f"{int(t*100)}%"] = int(math.ceil((1 - P) / (P * t**2)))

    # Print a human-readable summary of the combined run.
    print(f"\n  Combined results ({len(runs)}/{len(subdirs)} runs)")
    print(f"  N completed      = {N:,}")
    print(f"  Reacted          = {nr:,}")
    print(f"  Escaped          = {ne:,}")
    print(f"  Max-steps        = {n_max:,}  (counted as not reacted)")
    print(f"  P_rxn            = {P:.6f} ± {SE:.6f}")
    print(f"  Relative SE      = {RSE*100:.2f}%")
    print(f"  k_b              = {k_b:.4f} A3/ps")
    print(f"  k_on             = {k_on:.4e} M-1 s-1")
    if k_on > 0:
        print(f"  k_on (± error)     = ({man:.1f} ± {err:.1f}) x 10^{exp} M-1 s-1")
    print(
        f"  {'Wilson ' + format(100 * conf, 'g') + '% CI':<17s}= [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1"
    )
    if n_max:
        print(
            f"  k_on bound       = [{k_bound_lo:.4e}, {k_bound_hi:.4e}] M-1 s-1  "
            "(max-steps trajectories as not reacted, as reacted)"
        )
    if targets:
        print(f"  Trajectories needed:")
        for t, n in targets.items():
            status = "completed" if N_p >= n else "need more simulations"
            print(f"    For ±{t} RSE: {n:,} ({status})")
    is_converged = (0 < P < 1) and (RSE < conv_tol)
    print(f"  {'Converged' if is_converged else 'Not converged'} (RSE {RSE*100:.2f}%)")
    print(f"  Wall clock       = {wall_time:.1f} s  (slowest of {len(runs)} shard(s))")
    print(f"  Aggregate GPU-s  = {aggregate_gpu_sec:.1f} s  (sum over shards)")
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
        "P_rxn_bound_low": P_b_lo,
        "P_rxn_bound_high": P_b_hi,
        "k_on_bound_low": k_bound_lo,
        "k_on_bound_high": k_bound_hi,
        "r_start": r_start,
        "r_escape": r_escape,
        "wall_time_sec": wall_time,
        "aggregate_gpu_sec": aggregate_gpu_sec,
        "steps_per_sec": steps_sec,
        "confidence_level": conf,
        "convergence_tol": conv_tol,
    }
    if k_on > 0:
        results["log10_k_on"] = math.log10(k_on)
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
            key = (name, entry.get("state_before"), entry.get("state_after"))
            if key not in per_rxn:
                per_rxn[key] = {
                    "name": name,
                    "n_fired": 0,
                    "state_before": entry.get("state_before"),
                    "state_after": entry.get("state_after"),
                }
                rxn_order.append(key)
            per_rxn[key]["n_fired"] += int(entry.get("n_fired", 0))
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
        "bd_time_to_reaction.csv",
        bd_sims,
        reindex="traj_id",
        offsets=traj_offsets,
    )
    _concat_csv(
        dirs_valid,
        "pose_clusters.csv",
        bd_sims,
        reindex="traj_id",
        offsets=traj_offsets,
    )
    _sum_csv(
        dirs_valid,
        "radial_density.csv",
        bd_sims,
        sum_col="count",
        recompute_col="density",
        density_mode=True,
        time_col="time_ps",
    )
    _sum_csv(
        dirs_valid,
        "contact_frequency.csv",
        bd_sims,
        sum_col="n_contacts",
        recompute_col="frequency",
        total_N=total_contact_steps,
        time_col="time_ps",
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
        sum_key=["counts", "time_ps"],
        copy_keys=["theta_centers", "phi_centers"],
    )
    _sum_npz(
        dirs_valid,
        "transition_matrix.npz",
        bd_sims,
        sum_key="counts",
        copy_keys=["bins"],
    )
    # Write a separate convergence report summarising the pooled statistics and
    # whether the relative standard error has dropped below the tolerance.
    conv = {
        "N": N_p,
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
        "converged": (0 < P < 1) and (RSE < conv_tol),
        "confidence_level": conf,
        "tol": conv_tol,
        "tol_pct": conv_tol * 100,
        "N_needed": targets,
    }
    _save_json(conv, os.path.join(bd_sims, "convergence.json"))
    print(f"\n  All files saved -> {bd_sims}/")
    return 0


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
        ("confidence_level", ref.get("confidence_level")),
        ("convergence_tol", ref.get("convergence_tol")),
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


def _warn_chain_run_mismatch(runs):
    """Warn if chain shards disagree on physical rate or partner parameters."""
    if len(runs) < 2:
        return
    checks = [
        ("rate.k_b", lambda r: (r.get("rate") or {}).get("k_b")),
        ("rate.b_radius", lambda r: (r.get("rate") or {}).get("b_radius")),
        ("rate.q_radius", lambda r: (r.get("rate") or {}).get("q_radius")),
        ("rate.use_lmz", lambda r: (r.get("rate") or {}).get("use_lmz")),
        ("rate.nam_omega", lambda r: (r.get("rate") or {}).get("nam_omega")),
        (
            "chain.flexible_partner",
            lambda r: (r.get("chain") or {}).get("flexible_partner"),
        ),
    ]
    for label, getter in checks:
        ref_val = getter(runs[0])
        for i, r in enumerate(runs[1:], start=1):
            val = getter(r)
            if ref_val is None and val is None:
                continue
            if (ref_val is None) != (val is None):
                mismatch = True
            elif isinstance(ref_val, bool) or isinstance(val, bool):
                mismatch = bool(val) != bool(ref_val)
            else:
                try:
                    mismatch = not math.isclose(
                        float(val), float(ref_val), rel_tol=1e-9, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    mismatch = val != ref_val
            if mismatch:
                print(
                    f"  Warning: shard {i + 1} has {label}={val} but shard 1 has "
                    f"{label}={ref_val}; pooling assumes identical runs and selects "
                    f"the shard 1 value."
                )
                break


def _save_json(data, path, allow_nan=True):
    """Write a dictionary to a JSON file and report its basename."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str, allow_nan=allow_nan)
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
    time_col=None,
    sort_desc_col=None,
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

    time_col names a column of times in ps that is summed and kept as a float.
    """
    all_data = {}
    header = None
    key_cols = None
    cols_to_sum = ([sum_col] if sum_col else (sum_cols or [])) + (
        [time_col] if time_col else []
    )
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
                    cnt / (total_count * vol) if (total_count > 0 and vol > 0) else 0.0
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
    if time_col:
        for row in rows:
            row[time_col] = f"{float(row[time_col]):.6f}"
    if sort_desc_col:
        rows.sort(key=lambda r: -float(r.get(sort_desc_col, 0)))
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
    """Sum the arrays stored under sum_key (a name or a list of names) across
    shards element by element.

    The arrays from every shard (for example counts, times or a transition
    matrix) are added together. Any arrays named in copy_keys, such as the bin
    centres or milestone labels, are copied from the first shard that has
    them, since they are shared across shards.
    """
    keys = [sum_key] if isinstance(sum_key, str) else list(sum_key)
    totals = {}
    copies = {}
    for d in dirs:
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        npz = np.load(fpath, allow_pickle=True)
        for k in keys:
            if k in npz:
                totals[k] = npz[k] if k not in totals else totals[k] + npz[k]
        if not copies and copy_keys:
            for ck in copy_keys:
                if ck in npz:
                    copies[ck] = npz[ck]
    if not totals:
        return
    out_path = os.path.join(out_dir, filename)
    np.savez(out_path, **totals, **copies)
    print(f"    {filename} ({', '.join(f'{k}: {v.shape}' for k, v in totals.items())})")


def _combine_chain_runs(runs, dirs_valid, subdirs, bd_sims):
    """Pool shard outputs from flexible-chain BD runs into bd_sims."""
    _warn_chain_run_mismatch(runs)
    # Remove stale optional combined outputs so 0-row reruns leave no old files behind.
    for stale_name in (
        "encounters.csv",
        "near_misses.csv",
        "fpt_distribution.csv",
        "contact_frequency.csv",
        "radial_density.csv",
        "angular_map.npz",
        "milestone_flux.csv",
        "paths.npz",
        "energetics.npz",
    ):
        stale_path = os.path.join(bd_sims, stale_name)
        if os.path.exists(stale_path) or os.path.islink(stale_path):
            os.unlink(stale_path)
    sums = [r["summary"] for r in runs]
    rates = [r.get("rate", {}) for r in runs]
    nr = sum(int(s.get("n_reacted", 0)) for s in sums)
    ne = sum(int(s.get("n_escaped", 0)) for s in sums)
    n_max = sum(int(s.get("n_max_steps", 0)) for s in sums)
    n_timeouts = sum(int(s.get("n_escaped_via_timeout", 0)) for s in sums)
    N_p = sum(int(s.get("n_trajectories", 0)) for s in sums)
    P = nr / N_p if N_p > 0 else 0.0

    ref_rate = rates[0]
    k_b = ref_rate.get("k_b")
    conf = float(ref_rate.get("confidence_level", 0.95))
    conv_tol = float(ref_rate.get("convergence_tol", 0.05))
    CONV = MOLAR_RATE_CONV

    z = (
        1.96
        if abs(conf - 0.95) < 1.0e-9
        else statistics.NormalDist().inv_cdf(0.5 + 0.5 * conf)
    )
    if N_p == 0:
        P_lo, P_hi = 0.0, 0.0
    else:
        denom = 1.0 + z * z / N_p
        centre = (P + z * z / (2.0 * N_p)) / denom
        spread = (
            z
            * math.sqrt(max(P * (1.0 - P) / N_p + z * z / (4.0 * N_p * N_p), 0.0))
            / denom
        )
        P_lo = max(0.0, centre - spread)
        P_hi = min(1.0, centre + spread)

    SE = math.sqrt(P * (1.0 - P) / N_p) if 0.0 < P < 1.0 and N_p > 0 else 0.0
    RSE = SE / P if P > 0.0 else float("inf")
    P_b_lo, P_b_hi = _p_rxn_bounds(nr, ne, n_max)

    shard_walls = [float(s.get("wall_time_sec", 0.0)) for s in sums]
    wall_time = max(shard_walls) if shard_walls else 0.0
    agg_wall = sum(shard_walls)
    tot_steps = sum(
        float(s.get("mean_steps", 0.0)) * int(s.get("n_trajectories", 0)) for s in sums
    )

    pooled_rxn_counts = {}
    for s in sums:
        for rname, cnt in (s.get("reaction_counts") or {}).items():
            pooled_rxn_counts[rname] = pooled_rxn_counts.get(rname, 0) + int(cnt)

    w_n = [int(s.get("n_trajectories", 0)) for s in sums]
    summary_out = {
        "n_trajectories": N_p,
        "n_reacted": nr,
        "n_escaped": ne,
        "n_max_steps": n_max,
        "n_escaped_via_timeout": n_timeouts,
        "fraction_reacted": P,
        "fraction_escaped": (ne / N_p) if N_p > 0 else 0.0,
        "mean_time_ps": (
            sum(float(s.get("mean_time_ps", 0.0)) * n for s, n in zip(sums, w_n)) / N_p
            if N_p > 0
            else 0.0
        ),
        "mean_steps": (tot_steps / N_p) if N_p > 0 else 0.0,
        "mean_final_separation": (
            sum(
                float(s.get("mean_final_separation", 0.0)) * n
                for s, n in zip(sums, w_n)
            )
            / N_p
            if N_p > 0
            else 0.0
        ),
        "min_time_ps": min(
            (
                float(s.get("min_time_ps", 0.0))
                for s in sums
                if s.get("n_trajectories", 0) > 0
            ),
            default=0.0,
        ),
        "max_time_ps": max(
            (
                float(s.get("max_time_ps", 0.0))
                for s in sums
                if s.get("n_trajectories", 0) > 0
            ),
            default=0.0,
        ),
        "n_hard_sphere_redraws": sum(
            int(s.get("n_hard_sphere_redraws", 0)) for s in sums
        ),
        "n_hard_sphere_rejections": sum(
            int(s.get("n_hard_sphere_rejections", 0)) for s in sums
        ),
        "wall_time_sec": wall_time,
        "aggregate_gpu_sec": agg_wall,
        "steps_per_sec": (tot_steps / wall_time) if wall_time > 0.0 else 0.0,
    }
    if pooled_rxn_counts:
        summary_out["reaction_counts"] = pooled_rxn_counts

    rate_out = {
        "n_reacted": nr,
        "n_escaped": ne,
        "n_censored": n_max,
        "censored_fraction": (n_max / N_p) if N_p > 0 else 0.0,
        "use_lmz": bool(ref_rate.get("use_lmz", True)),
        "outer_propagator": bool(ref_rate.get("outer_propagator", k_b is not None)),
    }
    if k_b is not None:
        kb_f = float(k_b)
        nam_omega = float(ref_rate.get("nam_omega", 0.0))
        if rate_out["use_lmz"]:
            nam_omega = 0.0

        def _nam_beta(prob: float) -> float:
            if nam_omega <= 0.0:
                return prob
            denom_b = 1.0 - (1.0 - prob) * nam_omega
            return prob / denom_b if denom_b > 0.0 else 1.0

        beta = _nam_beta(P)
        beta_lo = _nam_beta(P_lo)
        beta_hi = _nam_beta(P_hi)
        beta_b_lo = _nam_beta(P_b_lo)
        beta_b_hi = _nam_beta(P_b_hi)
        if nam_omega > 0.0:
            denom_j = 1.0 - (1.0 - P) * nam_omega
            jac = (
                max(0.0, 1.0 - nam_omega) / (denom_j * denom_j)
                if denom_j > 0.0
                else 0.0
            )
        else:
            jac = 1.0
        SE_beta = jac * SE
        RSE_beta = SE_beta / beta if beta > 0.0 else float("inf")
        k_on = CONV * kb_f * beta
        SE_kon = CONV * kb_f * SE_beta
        k_lo = CONV * kb_f * beta_lo
        k_hi = CONV * kb_f * beta_hi
        rate_out.update(
            {
                "k_b": kb_f,
                "k_b_units": "A^3/ps",
                "b_radius": float(ref_rate.get("b_radius", 0.0)),
                "q_radius": float(ref_rate.get("q_radius", 0.0)),
                "return_prob": float(ref_rate.get("return_prob", 0.0)),
                "nam_omega": nam_omega,
                "P_rxn_raw": P,
                "P_rxn": beta,
                "P_rxn_low": beta_lo,
                "P_rxn_high": beta_hi,
                "k_on": k_on,
                "k_on_low": k_lo,
                "k_on_high": k_hi,
                "k_on_units": "M-1 s-1",
                "P_rxn_bound_low": beta_b_lo,
                "P_rxn_bound_high": beta_b_hi,
                "k_on_bound_low": CONV * kb_f * beta_b_lo,
                "k_on_bound_high": CONV * kb_f * beta_b_hi,
                "confidence_level": conf,
                "convergence_tol": conv_tol,
            }
        )
    else:
        nam_omega = 0.0
        beta = P
        beta_lo = P_lo
        beta_hi = P_hi
        SE_beta = SE
        RSE_beta = RSE
        k_on = k_lo = k_hi = 0.0
        SE_kon = 0.0
        rate_out["k_on"] = None
        rate_out["reason"] = ref_rate.get(
            "reason", "no outer propagator, so k_b is undefined"
        )

    diff_out = dict(runs[0].get("diffusion", {}))
    diff_out["n_relative_diffusion_clipped"] = sum(
        int(r.get("diffusion", {}).get("n_relative_diffusion_clipped", 0)) for r in runs
    )
    if "n_rebuilds" in diff_out:
        diff_out["n_rebuilds"] = sum(
            int(r.get("diffusion", {}).get("n_rebuilds", 0)) for r in runs
        )
        diff_out["n_rebuild_failures"] = sum(
            int(r.get("diffusion", {}).get("n_rebuild_failures", 0)) for r in runs
        )
    for dkey in ("D_trans_mean", "D_rot_mean"):
        entries = [
            (
                float(r.get("diffusion", {})[dkey]),
                float(s.get("mean_time_ps", 0.0)) * n_s,
                float(n_s),
            )
            for r, s, n_s in zip(runs, sums, w_n)
            if dkey in r.get("diffusion", {})
            and r.get("diffusion", {})[dkey] is not None
            and n_s > 0
        ]
        if not entries:
            continue
        use_time = all(t_w > 0.0 for _, t_w, _ in entries)
        weights = [t_w if use_time else n_w for _, t_w, n_w in entries]
        tot_w = sum(weights)
        if tot_w > 0.0:
            diff_out[dkey] = (
                sum(v * w for (v, _, _), w in zip(entries, weights)) / tot_w
            )

    print(f"\n  Combined chain BD results ({len(runs)}/{len(subdirs)} runs)")
    print(f"  Trajectories     = {N_p:,}")
    print(f"  Reacted          = {nr:,}")
    print(f"  Escaped          = {ne:,}")
    print(f"  Max-steps        = {n_max:,}")
    if nam_omega > 0.0:
        print(
            f"  P_rxn            = {beta:.6f} ± {SE_beta:.6f} "
            f"(raw {P:.6f}, NAM omega={nam_omega:.4f})"
        )
    else:
        print(f"  P_rxn            = {beta:.6f} ± {SE_beta:.6f}")
    if k_b is not None:
        print(f"  k_b              = {float(k_b):.4f} A3/ps")
        print(f"  k_on             = {k_on:.4e} M-1 s-1")
        print(f"  Wilson {100 * conf:g}% CI   = [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1")

    pooled_doc = {
        "summary": summary_out,
        "rate": rate_out,
        "diffusion": diff_out,
        "chain": runs[0].get("chain", {}),
        "params": dict(runs[0].get("params", {}), n_trajectories=N_p),
    }
    _save_json(pooled_doc, os.path.join(bd_sims, "results.json"))

    traj_offsets = []
    running = 0
    for n_s in w_n:
        traj_offsets.append(running)
        running += n_s

    _concat_csv(
        dirs_valid, "trajectories.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _concat_csv(
        dirs_valid, "encounters.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _concat_csv(
        dirs_valid, "near_misses.csv", bd_sims, reindex="traj_id", offsets=traj_offsets
    )
    _sum_csv(
        dirs_valid,
        "contact_frequency.csv",
        bd_sims,
        sum_cols=["total_contacts", "n_trajectories"],
        sort_desc_col="total_contacts",
    )
    _sum_npz(
        dirs_valid,
        "angular_map.npz",
        bd_sims,
        sum_key=["counts", "total_snapshots"],
        copy_keys=["theta_edges", "phi_edges"],
    )
    _concat_chain_flat_npz(dirs_valid, "paths.npz", bd_sims, traj_offsets)
    _concat_chain_flat_npz(dirs_valid, "energetics.npz", bd_sims, traj_offsets)
    _rebuild_chain_derived_csvs(dirs_valid, bd_sims, pooled_doc)

    targets = {}
    if 0.0 < P < 1.0:
        denom_t = 1.0 - (1.0 - P) * nam_omega
        rse_scale = (
            max(0.0, 1.0 - nam_omega) / denom_t
            if (nam_omega > 0.0 and denom_t > 0.0)
            else 1.0
        )
        for t in (0.10, 0.05, 0.01):
            targets[f"{int(t * 100)}%"] = int(
                math.ceil((rse_scale * rse_scale) * (1.0 - P) / (P * t * t))
            )
    rse_json = RSE_beta if math.isfinite(RSE_beta) else None
    rse_pct_json = (
        RSE_beta * 100.0 if (beta > 0.0 and math.isfinite(RSE_beta)) else None
    )
    conv = {
        "N": N_p,
        "n_reacted": nr,
        "n_escaped": ne,
        "n_max_steps": n_max,
        "P_rxn": beta,
        "P_rxn_raw": P,
        "SE": SE_beta,
        "relative_SE": rse_json,
        "relative_SE_pct": rse_pct_json,
        "k_on": k_on,
        "SE_kon": SE_kon,
        "wilson_CI": [k_lo, k_hi],
        "wilson_CI_P": [beta_lo, beta_hi],
        "converged": (0.0 < P < 1.0) and (RSE_beta < conv_tol),
        "confidence_level": conf,
        "tol": conv_tol,
        "tol_pct": conv_tol * 100.0,
        "N_needed": targets,
    }
    _save_json(conv, os.path.join(bd_sims, "convergence.json"), allow_nan=False)
    print(f"\n  All files saved -> {bd_sims}/")
    return 0


def _concat_chain_flat_npz(dirs, filename, out_dir, offsets):
    """Concatenate flat-schema chain NPZ archives (paths.npz or energetics.npz)."""
    shard_dicts = []
    all_keys = []
    cols = None
    for i, d in enumerate(dirs):
        fpath = os.path.join(d, filename)
        if not os.path.exists(fpath):
            continue
        with np.load(fpath, allow_pickle=False) as npz:
            sd = {}
            for k in npz.files:
                if k == "columns":
                    if cols is None:
                        cols = np.array(npz[k])
                    continue
                arr = np.array(npz[k])
                if k == "traj_id":
                    arr = arr + int(offsets[i])
                sd[k] = arr
                if k not in all_keys:
                    all_keys.append(k)
            if sd:
                shard_dicts.append(sd)
    if not shard_dicts:
        return
    # Pad optional columns missing in a subset of shards so row counts align.
    save_dict = {}
    for k in all_keys:
        ref = next(sd[k] for sd in shard_dicts if k in sd)
        pieces = []
        for sd in shard_dicts:
            if k in sd:
                pieces.append(sd[k])
            else:
                n_s = len(sd.get("traj_id", next(iter(sd.values()))))
                pieces.append(np.zeros((n_s,) + ref.shape[1:], dtype=ref.dtype))
        save_dict[k] = np.concatenate(pieces, axis=0)
    if cols is not None:
        save_dict["columns"] = cols
    out_path = os.path.join(out_dir, filename)
    np.savez(out_path, **save_dict)
    n_rows = len(save_dict.get("traj_id", next(iter(save_dict.values()))))
    print(f"    {filename} ({n_rows:,} rows)")


def _rebuild_chain_derived_csvs(dirs_valid, out_dir, pooled_doc):
    """Rebuild pooled fpt_distribution.csv, radial_density.csv, and milestone_flux.csv."""
    from pathlib import Path
    from pystarc.molsystem.system_state import Fate, TrajectoryResult
    from pystarc.pipeline.chain_output_writer import (
        write_fpt_distribution_csv,
        write_milestone_flux_csv,
        write_radial_density_csv,
    )

    out_p = Path(out_dir)
    want_fpt = any(
        os.path.exists(os.path.join(d, "fpt_distribution.csv")) for d in dirs_valid
    )
    traj_csv = out_p / "trajectories.csv"
    if want_fpt and traj_csv.exists():
        fpt_stubs = []
        with open(traj_csv) as f:
            for row in csv.DictReader(f):
                if row.get("fate", "").strip().upper() == "REACTED":
                    fpt_stubs.append(
                        TrajectoryResult(
                            fate=Fate.REACTED,
                            steps=int(row.get("steps", 0)),
                            time_ps=float(row.get("time_ps", 0.0)),
                            final_separation=float(row.get("final_separation", 0.0)),
                        )
                    )
        p = write_fpt_distribution_csv(out_p, fpt_stubs)
        if p is not None:
            print(f"    {p.name}")

    want_rad = any(
        os.path.exists(os.path.join(d, "radial_density.csv")) for d in dirs_valid
    )
    want_flux = any(
        os.path.exists(os.path.join(d, "milestone_flux.csv")) for d in dirs_valid
    )
    paths_npz = out_p / "paths.npz"
    if (want_rad or want_flux) and paths_npz.exists():
        with np.load(paths_npz, allow_pickle=False) as npz:
            tids = np.asarray(npz["traj_id"], dtype=int)
            rad = np.asarray(npz["radial"], dtype=float)
            ret = (
                np.asarray(npz["radial_returns"], dtype=int)
                if "radial_returns" in npz.files
                else None
            )
        if len(tids) > 0:
            order = np.argsort(tids, kind="mergesort")
            tids_s = tids[order]
            rad_s = rad[order]
            ret_s = ret[order] if ret is not None else None
            uniq, starts = np.unique(tids_s, return_index=True)
            ends = np.append(starts[1:], len(tids_s))
            path_stubs = []
            for s_i, e_i in zip(starts, ends):
                tr = TrajectoryResult(
                    fate=Fate.ESCAPED, steps=0, time_ps=0.0, final_separation=0.0
                )
                tr.radial_trace = rad_s[s_i:e_i]
                if ret_s is not None:
                    tr.radial_returns = ret_s[s_i:e_i]
                path_stubs.append(tr)
            if want_rad:
                p = write_radial_density_csv(out_p, path_stubs)
                if p is not None:
                    print(f"    {p.name}")
            if want_flux:
                b_rad = pooled_doc.get("rate", {}).get("b_radius") or pooled_doc.get(
                    "params", {}
                ).get("r_start")
                p = write_milestone_flux_csv(
                    out_p,
                    path_stubs,
                    b_radius=float(b_rad) if b_rad is not None else None,
                )
                if p is not None:
                    print(f"    {p.name}")
    elif want_rad or want_flux:
        # Pool shard CSV histograms directly when full_paths=false omitted paths.npz.
        if want_rad:
            _pool_chain_radial_density_csv(dirs_valid, out_dir)
        if want_flux:
            _pool_chain_milestone_flux_csv(dirs_valid, out_dir)


def _pool_chain_radial_density_csv(dirs_valid, out_dir, n_bins=50):
    """Pool shard radial_density.csv histograms by exact bin interval overlap."""
    bins_all = []
    max_r = 0.0
    for d in dirs_valid:
        fpath = os.path.join(d, "radial_density.csv")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                try:
                    lo = float(row["bin_lo"])
                    hi = float(row["bin_hi"])
                    cnt = float(row["count"])
                except (KeyError, ValueError):
                    continue
                if hi > max_r:
                    max_r = hi
                if cnt > 0.0:
                    bins_all.append((lo, hi, cnt))
    if not bins_all or max_r <= 0.0:
        return
    edges = np.linspace(0.0, max_r, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)
    for lo, hi, cnt in bins_all:
        if hi > lo:
            ov = np.maximum(0.0, np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]))
            ov_sum = float(ov.sum())
            if ov_sum > 0.0:
                counts += cnt * (ov / ov_sum)
            else:
                idx = int(
                    np.clip(
                        np.searchsorted(edges, 0.5 * (lo + hi), side="right") - 1,
                        0,
                        n_bins - 1,
                    )
                )
                counts[idx] += cnt
        else:
            idx = int(
                np.clip(
                    np.searchsorted(edges, lo, side="right") - 1,
                    0,
                    n_bins - 1,
                )
            )
            counts[idx] += cnt
    total = float(counts.sum())
    if total <= 0.0:
        return
    widths = np.diff(edges)
    dens = np.where(widths > 0.0, counts / (total * widths), 0.0)
    out_path = os.path.join(out_dir, "radial_density.csv")
    with open(out_path, "w", newline="") as f:
        f.write("bin_lo,bin_hi,count,density\n")
        for i in range(n_bins):
            f.write(
                f"{edges[i]:.4f},{edges[i + 1]:.4f},{int(round(counts[i]))},{dens[i]:.8e}\n"
            )
    print(f"    radial_density.csv ({n_bins} rows)")


def _pool_chain_milestone_flux_csv(dirs_valid, out_dir):
    """Pool shard milestone_flux.csv tables when paths.npz is absent.

    When every shard shares the same shell radii, crossing counts are summed
    directly. When shell radii differ across shards, each shard is linearly
    interpolated onto a common sorted n_shells interior radius grid and summed.
    """
    shard_tables = []
    for d in dirs_valid:
        fpath = os.path.join(d, "milestone_flux.csv")
        if not os.path.exists(fpath):
            continue
        rows = []
        with open(fpath) as f:
            for row in csv.DictReader(f):
                try:
                    rad = float(row["shell_radius"])
                    n_out = float(row["n_crossings_out"])
                    n_in = float(row["n_crossings_in"])
                except (KeyError, ValueError):
                    continue
                rows.append((rad, n_out, n_in))
        if not rows:
            continue
        rows.sort(key=lambda t: t[0])
        shard_tables.append(
            (
                np.array([t[0] for t in rows], dtype=float),
                np.array([t[1] for t in rows], dtype=float),
                np.array([t[2] for t in rows], dtype=float),
            )
        )
    if not shard_tables:
        return

    ref_r = shard_tables[0][0]
    same_grid = all(
        len(r) == len(ref_r) and np.allclose(r, ref_r, rtol=0.0, atol=1e-6)
        for r, _, _ in shard_tables[1:]
    )
    if same_grid:
        common_radii = ref_r
        total_out = sum(o for _, o, _ in shard_tables)
        total_in = sum(ni for _, _, ni in shard_tables)
    else:
        n_shells = len(ref_r)
        if n_shells == 1:
            common_radii = np.array(
                [float(np.mean([r[0] for r, _, _ in shard_tables]))], dtype=float
            )
            total_out = np.array(
                [sum(float(o[0]) for _, o, _ in shard_tables)], dtype=float
            )
            total_in = np.array(
                [sum(float(ni[0]) for _, _, ni in shard_tables)], dtype=float
            )
        else:
            r_min_all = math.inf
            r_max_all = -math.inf
            augmented = []
            for r_s, o_s, i_s in shard_tables:
                if len(r_s) >= 2:
                    dr_lo = r_s[1] - r_s[0]
                    dr_hi = r_s[-1] - r_s[-2]
                    r_lo = max(0.0, r_s[0] - dr_lo)
                    r_hi = r_s[-1] + dr_hi
                else:
                    r_lo = float(r_s[0])
                    r_hi = float(r_s[0])
                if r_lo < r_min_all:
                    r_min_all = r_lo
                if r_hi > r_max_all:
                    r_max_all = r_hi
                if len(r_s) >= 2 and r_lo < r_s[0] and r_hi > r_s[-1]:
                    x_aug = np.concatenate(([r_lo], r_s, [r_hi]))
                    o_aug = np.concatenate(([0.0], o_s, [0.0]))
                    i_aug = np.concatenate(([0.0], i_s, [0.0]))
                else:
                    x_aug, o_aug, i_aug = r_s, o_s, i_s
                augmented.append((x_aug, o_aug, i_aug))
            common_radii = np.linspace(r_min_all, r_max_all, n_shells + 2)[1:-1]
            total_out = np.zeros(n_shells, dtype=float)
            total_in = np.zeros(n_shells, dtype=float)
            for x_aug, o_aug, i_aug in augmented:
                if len(x_aug) == 1:
                    idx = int(np.argmin(np.abs(common_radii - x_aug[0])))
                    total_out[idx] += float(o_aug[0])
                    total_in[idx] += float(i_aug[0])
                else:
                    total_out += np.interp(
                        common_radii, x_aug, o_aug, left=0.0, right=0.0
                    )
                    total_in += np.interp(
                        common_radii, x_aug, i_aug, left=0.0, right=0.0
                    )

    out_path = os.path.join(out_dir, "milestone_flux.csv")
    with open(out_path, "w", newline="") as f:
        f.write("shell_radius,n_crossings_out,n_crossings_in\n")
        for r_val, n_o, n_i in zip(common_radii, total_out, total_in):
            f.write(f"{r_val:.4f},{int(round(n_o))},{int(round(n_i))}\n")
    print(f"    milestone_flux.csv ({len(common_radii)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
