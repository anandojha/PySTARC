#!/usr/bin/env python3
import glob
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CONV = 6.02214076e23 * 1e-27 * 1e12          # A^3/ps to M^-1 s^-1

def wilson(reacted, n, z=1.96):
    """Wilson score interval on the reaction probability."""
    if n == 0:
        return 0.0, 0.0
    p = reacted / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def counts(res):
    """Reacted, escaped and step-limited counts, from either schema."""
    r = res.get("rate")
    if r:
        return r["n_reacted"], r["n_escaped"], r["n_censored"]
    s = res["summary"]
    return s["n_reacted"], s["n_escaped"], s.get("n_max_steps", 0)

paths = sorted(glob.glob(os.path.join(BASE, "shards", "shard_*",
                                      "bd_sims", "results.json")))
if not paths:
    sys.exit("  no finished shards under shards/")

tot_r = tot_e = tot_c = 0
k_b = b_rad = None
print(f"{'shard':<10}{'reacted':>9}{'escaped':>9}{'stepcap':>9}"
      f"{'%capped':>9}{'k_b':>10}{'hours':>8}")
for p in paths:
    res = json.load(open(p))
    n_r, n_e, n_c = counts(res)
    tot_r, tot_e, tot_c = tot_r + n_r, tot_e + n_e, tot_c + n_c
    rate = res.get("rate", {})
    kb = rate.get("k_b")
    br = rate.get("b_radius") or res["params"]["r_start"]
    if kb is not None:
        if k_b is None:
            k_b, b_rad = kb, br
        elif abs(kb - k_b) > 1e-6 * max(abs(kb), 1.0):
            sys.exit(f"  shards disagree on k_b: {k_b} and {kb}")
    hours = res["summary"].get("wall_time_sec", 0) / 3600.0
    name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    pct = 100.0 * n_c / max(n_r + n_e + n_c, 1)
    print(f"{name:<10}{n_r:>9}{n_e:>9}{n_c:>9}{pct:>8.0f}%"
          f"{(kb if kb is not None else float('nan')):>10.4f}{hours:>8.1f}")

n = tot_r + tot_e
print(f"{'total':<10}{tot_r:>9}{tot_e:>9}{tot_c:>9}"
      f"{100.0*tot_c/max(n+tot_c,1):>8.0f}%")
print()

if n == 0:
    sys.exit("  no trajectory committed to a reaction or an escape")
p = tot_r / n
lo, hi = wilson(tot_r, n)
print(f"  b surface        {b_rad:.1f} A")
print(f"  committed        {n} of {n + tot_c}")
print(f"  P_rxn            {p:.5f}   Wilson 95 percent {lo:.5f} to {hi:.5f}")
if k_b is None:
    sys.exit("  no k_b in any shard, so no rate can be reported")
print(f"  k_b              {k_b:.4f} A^3/ps")
print(f"  k_on             {CONV*k_b*p:.3e} M-1 s-1"
      f"   [{CONV*k_b*lo:.2e}, {CONV*k_b*hi:.2e}]")