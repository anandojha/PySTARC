"""
Choose the step limit (max_steps) for a production run from a pilot run.

Every trajectory runs until it reacts, escapes or reaches max_steps. A
trajectory at the limit counts as not reacted, so the limit biases P_rxn
downward by at most S(max_steps), the fraction of trajectories still running
at the limit, and k_on by at most S(max_steps)/P_rxn in relative terms. The
pilot gives the survival curve S(n), the fraction of trajectories that have
neither reacted nor escaped after n steps. The recommended limit is the
smallest multiple of the granularity at which

    S(max_steps) <= tol * P_rxn,

so that the trajectories at the limit cannot move k_on by more than tol. P_rxn
is taken from the pilot (reacted over every trajectory run) or given, for
example from an earlier run when the pilot is too small to see a reaction.

Beyond the observed range of the pilot, or below its statistical resolution
(3/N), S(n) is extrapolated with an exponential fitted to the tail of the
observed curve. Within the observed range the fit can raise the observed
survival but not lower it. A
tail that does not decay (trajectories at the pilot limit with no finish in
the last part of the pilot range) is reported as trapped: those trajectories
are stuck, not slow, and no step limit will resolve them. The usual causes are
a reaction criterion that cannot be met and the excluded-volume setting.

Usage:
    python -m pystarc.analysis.step_limit bd_sims/trajectories.csv [options]

The pilot is an ordinary run with a small number of trajectories and a
generous max_steps (the ceiling, 1,000,000 by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Outcome codes of trajectories.csv and the sim_data outcome array.
_REACTED = {"reacted", 1, "1"}
_ESCAPED = {"escaped", 2, "2"}
_AT_LIMIT = {"max_steps", 3, "3", "running", 0, "0"}

# Fraction of the pilot range, counted back from the pilot limit, in which at
# least one trajectory must finish for the tail to count as decaying.
_FLAT_WINDOW = 0.2
# Smallest number of distinct finishing lengths for the exponential tail fit.
_MIN_FIT_POINTS = 5
# The tail fit uses the finishing lengths above this quantile of all lengths.
_TAIL_QUANTILE = 0.90


def survival_curve(n_steps: Sequence[int]):
    """Return (lengths, S) with S[k] the fraction of trajectories longer than
    lengths[k], for the sorted distinct lengths."""
    n = np.asarray(n_steps, dtype=np.int64)
    if n.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    lengths, counts = np.unique(n, return_counts=True)
    survivors = n.size - np.cumsum(counts)
    return lengths, survivors / n.size


def _survival_at(n_steps: np.ndarray, cap: int) -> float:
    """Observed fraction of trajectories longer than cap."""
    return float(np.count_nonzero(n_steps > cap)) / n_steps.size


def _fit_tail(n_steps: np.ndarray, finished: np.ndarray) -> Optional[Dict]:
    """Fit ln S(n) = a + b n over the tail of the observed survival curve.

    The window holds the finishing lengths above the _TAIL_QUANTILE quantile of
    all lengths. Returns None when there are too few points or the slope is not
    negative.
    """
    if finished.size < _MIN_FIT_POINTS:
        return None
    lo = float(np.quantile(n_steps, _TAIL_QUANTILE))
    lengths = np.unique(finished[finished >= lo])
    if lengths.size < _MIN_FIT_POINTS:
        return None
    S = np.array([_survival_at(n_steps, int(l)) for l in lengths])
    keep = S > 0
    if np.count_nonzero(keep) < _MIN_FIT_POINTS:
        return None
    x = lengths[keep].astype(float)
    y = np.log(S[keep])
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None
    return {
        "tau_steps": float(-1.0 / slope),
        "intercept": float(intercept),
        "slope": float(slope),
        "n_points": int(x.size),
        "window": [int(x[0]), int(x[-1])],
    }


def _fit_survival(fit: Dict, cap: int) -> float:
    return float(min(1.0, math.exp(fit["intercept"] + fit["slope"] * cap)))


def recommend_max_steps(
    n_steps: Sequence[int],
    outcomes: Sequence,
    *,
    p_rxn: Optional[float] = None,
    tol: float = 0.01,
    granularity: int = 10_000,
    ceiling: int = 1_000_000,
    pilot_cap: Optional[int] = None,
) -> Dict:
    """Recommend max_steps for a production run from a pilot run.

    n_steps and outcomes are the per-trajectory step counts and outcomes of the
    pilot ("reacted", "escaped", "max_steps" or "running", or the codes 1, 2,
    3, 0). p_rxn is the reaction probability to use; by default it is reacted
    over every trajectory of the pilot. tol is the largest relative change of
    k_on that the trajectories at the limit may cause. The recommended limit is
    a multiple of granularity between granularity and ceiling.

    Returns a dictionary with the recommendation (max_steps), whether it meets
    the tolerance (meets_tolerance), the survival at the limit and how it was
    obtained (observed or extrapolated), the tail fit, and warnings.
    """
    n = np.asarray(n_steps, dtype=np.int64)
    out = np.asarray(list(outcomes), dtype=object)
    if n.size == 0 or n.size != out.size:
        raise ValueError("n_steps and outcomes must be non-empty and of equal length")
    if granularity <= 0 or ceiling < granularity:
        raise ValueError("granularity must be positive and ceiling at least granularity")
    if not 0 < tol < 1:
        raise ValueError("tol must be between 0 and 1")
    reacted = np.array([o in _REACTED for o in out])
    escaped = np.array([o in _ESCAPED for o in out])
    at_limit = np.array([o in _AT_LIMIT for o in out])
    if not np.all(reacted | escaped | at_limit):
        bad = sorted({str(o) for o in out[~(reacted | escaped | at_limit)]})
        raise ValueError(f"unknown outcome(s): {', '.join(bad)}")
    N = int(n.size)
    n_reacted = int(reacted.sum())
    n_escaped = int(escaped.sum())
    n_at_limit = int(at_limit.sum())
    warnings: List[str] = []

    # The reaction probability that sets the target survival.
    if p_rxn is None:
        p_source = "pilot"
        p_rxn = n_reacted / N
        if n_reacted == 0:
            raise ValueError(
                "the pilot has no reaction, so the tolerance has no reference; "
                "give p_rxn from an earlier run or a larger pilot"
            )
        if n_reacted < 10:
            warnings.append(
                f"only {n_reacted} reaction(s) in the pilot; P_rxn is uncertain "
                "and so is the target survival"
            )
    else:
        p_source = "given"
        if not 0 < p_rxn <= 1:
            raise ValueError("p_rxn must be between 0 and 1")
    target = tol * p_rxn

    # Censoring: trajectories at the pilot limit are longer than the limit.
    if pilot_cap is None:
        pilot_cap = int(n.max()) if n_at_limit else None
    if n_at_limit and pilot_cap is not None and int(n[at_limit].min()) < pilot_cap:
        warnings.append(
            "some trajectories reported at the limit have fewer steps than the "
            "pilot limit; they are treated as censored at their own length"
        )
    finished = n[reacted | escaped]
    # A tail that does not decay: trajectories at the pilot limit, and no
    # finish in the last part of the pilot range.
    tail_flat = False
    if n_at_limit and pilot_cap is not None:
        window_start = int((1.0 - _FLAT_WINDOW) * pilot_cap)
        if not np.any(finished > window_start):
            tail_flat = True
            warnings.append(
                f"trapped tail: {n_at_limit} of {N} trajectories reached the "
                f"pilot limit ({pilot_cap:,} steps) and none finished in the "
                f"last {int(_FLAT_WINDOW * 100)}% of the pilot range. These "
                "trajectories are stuck, not slow, and a larger step limit will "
                "not resolve them. Check the reaction criterion and the "
                "excluded-volume setting."
            )
    fit = None if tail_flat else _fit_tail(n, finished)
    resolution = 3.0 / N  # Rule of three: S = 0 observed means S < 3/N at 95%.

    def estimate(cap: int):
        """Survival at cap and how it was obtained."""
        observable = pilot_cap is None or cap < pilot_cap
        s_obs = _survival_at(n, cap) if observable else None
        if s_obs is not None and s_obs >= resolution:
            return s_obs, "observed"
        if tail_flat:
            # The survivors do not decay; the best estimate is the censored
            # fraction, a lower bound of the true survival.
            return (s_obs if s_obs is not None else n_at_limit / N), "trapped"
        if fit is not None:
            s_fit = _fit_survival(fit, cap)
            # Within the observed range the fit may raise a survival below the
            # resolution of the pilot, never lower it.
            if s_obs is not None and s_obs > s_fit:
                return s_obs, "observed"
            return s_fit, "extrapolated"
        if s_obs is not None:
            return s_obs, "observed"
        return (n_at_limit / N), "unresolved"

    chosen = None
    for cap in range(granularity, ceiling + 1, granularity):
        s, how = estimate(cap)
        if s <= target:
            chosen = (cap, s, how)
            break
    if chosen is None:
        cap = ceiling - ceiling % granularity
        s, how = estimate(cap)
        chosen = (cap, s, how)
        meets = False
        warnings.append(
            f"no step limit up to the ceiling ({ceiling:,}) brings the survival "
            f"({s:.2e}, {how}) below the target {target:.2e} = tol × P_rxn; "
            "the trajectories at the limit could move k_on by more than "
            f"{tol * 100:.1f}%. Raise the ceiling, or report the bound."
        )
    else:
        meets = True
    cap, s, how = chosen
    if how == "extrapolated" and fit is not None and target < resolution:
        warnings.append(
            f"the target survival {target:.2e} is below the resolution of a "
            f"pilot of {N:,} trajectories (about {resolution:.1e}); the "
            "recommendation rests on the exponential tail fit "
            f"(tau = {fit['tau_steps']:,.0f} steps from {fit['n_points']} points)"
        )
    return {
        "max_steps": int(cap),
        "meets_tolerance": bool(meets),
        "tol": float(tol),
        "p_rxn": float(p_rxn),
        "p_rxn_source": p_source,
        "target_survival": float(target),
        "survival_at_max_steps": float(s),
        "method": how,
        "n_pilot": N,
        "n_reacted": n_reacted,
        "n_escaped": n_escaped,
        "n_at_limit": n_at_limit,
        "pilot_cap": None if pilot_cap is None else int(pilot_cap),
        "tail_fit": fit,
        "tail_trapped": bool(tail_flat),
        "resolution": float(resolution),
        "granularity": int(granularity),
        "ceiling": int(ceiling),
        "warnings": warnings,
    }


def read_trajectories_csv(path: str):
    """Read the outcome and n_steps columns of a trajectories.csv."""
    outcomes, n_steps = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"outcome", "n_steps"} <= set(
            reader.fieldnames
        ):
            raise ValueError(f"{path}: expected the columns outcome and n_steps")
        for row in reader:
            outcomes.append(row["outcome"].strip())
            n_steps.append(int(float(row["n_steps"])))
    return n_steps, outcomes


def format_report(rec: Dict, n_steps: Optional[Sequence[int]] = None) -> str:
    """Human-readable report of a recommendation."""
    lines = [
        "  Step limit from the pilot",
        f"  Pilot            = {rec['n_pilot']:,} trajectories  "
        f"({rec['n_reacted']:,} reacted, {rec['n_escaped']:,} escaped, "
        f"{rec['n_at_limit']:,} at the limit"
        + (f" of {rec['pilot_cap']:,} steps)" if rec["pilot_cap"] else ")"),
        f"  P_rxn            = {rec['p_rxn']:.3e}  ({rec['p_rxn_source']})",
        f"  Target survival  = {rec['target_survival']:.2e}  "
        f"(tol {rec['tol'] * 100:.1f}% × P_rxn)",
    ]
    if n_steps is not None:
        n = np.asarray(n_steps, dtype=np.int64)
        lines.append("  Survival S(n), fraction still running after n steps:")
        g = rec["granularity"]
        caps = sorted(
            {g, 2 * g, 5 * g, 10 * g, 20 * g, 50 * g, rec["max_steps"], rec["ceiling"]}
        )
        for cap in caps:
            if cap > rec["ceiling"]:
                continue
            obs = (
                _survival_at(n, cap)
                if rec["pilot_cap"] is None or cap < rec["pilot_cap"]
                else None
            )
            if obs is None:
                s_txt = "beyond the pilot limit"
            else:
                s_txt = f"{obs:.2e}" if obs > 0 else "0 (none observed)"
            if rec["tail_fit"] is not None:
                s_txt += f"   fit {_fit_survival(rec['tail_fit'], cap):.2e}"
            lines.append(f"    n = {cap:>9,}   S = {s_txt}")
    if rec["tail_fit"] is not None:
        fit = rec["tail_fit"]
        lines.append(
            f"  Tail fit         = exponential, tau {fit['tau_steps']:,.0f} steps "
            f"({fit['n_points']} points, {fit['window'][0]:,}-{fit['window'][1]:,})"
        )
    lines.append(
        f"  max_steps        = {rec['max_steps']:,}  "
        f"(S = {rec['survival_at_max_steps']:.2e}, {rec['method']}; "
        + ("meets the tolerance)" if rec["meets_tolerance"] else "does NOT meet the tolerance)")
    )
    for w in rec["warnings"]:
        lines.append(f"  Warning: {w}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose max_steps for a production run from a pilot run."
    )
    parser.add_argument("trajectories", help="trajectories.csv of the pilot run")
    parser.add_argument(
        "--p-rxn",
        type=float,
        default=None,
        help="reaction probability to use instead of the pilot's own",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="largest relative change of k_on allowed from the trajectories at the limit (default 0.01)",
    )
    parser.add_argument(
        "--granularity", type=int, default=10_000, help="max_steps is a multiple of this (default 10000)"
    )
    parser.add_argument(
        "--ceiling", type=int, default=1_000_000, help="largest max_steps considered (default 1000000)"
    )
    parser.add_argument(
        "--pilot-cap",
        type=int,
        default=None,
        help="max_steps of the pilot run (default: the largest step count when some trajectory is at the limit)",
    )
    parser.add_argument("--json", default=None, help="also write the recommendation to this JSON file")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        n_steps, outcomes = read_trajectories_csv(args.trajectories)
        rec = recommend_max_steps(
            n_steps,
            outcomes,
            p_rxn=args.p_rxn,
            tol=args.tol,
            granularity=args.granularity,
            ceiling=args.ceiling,
            pilot_cap=args.pilot_cap,
        )
    except ValueError as e:
        print(f"  Error: {e}")
        return 2
    print(format_report(rec, n_steps))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Written: {args.json}")
    return 0 if rec["meets_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
