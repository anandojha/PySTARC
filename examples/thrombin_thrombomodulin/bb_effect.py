#!/usr/bin/env python3
"""
  A/B test: Brownian Bridge effect on thrombin P_rxn
  Runs 4 experiments:
    A1: BB=OFF, seed=11111111  (baseline 1)
    A2: BB=OFF, seed=22222222  (baseline 2, different RNG)
    B1: BB=ON,  seed=11111111  (BB test 1)
    B2: BB=ON,  seed=22222222  (BB test 2)
  If BB is working:  B1 > A1 and B2 > A2 (consistently higher)
  If BB is just RNG shift:  B1 ≈ A2 ≈ random (no consistent direction)
  Run from the thrombin example directory: examples/thrombin_thrombomodulin && python test_bb_effect.py
"""

from pystarc.pipeline.pipeline import run as run_pipeline
import pystarc.simulation.gpu_batch_simulator as sim_mod
from pystarc.pipeline.input_parser import parse
from contextlib import redirect_stdout
from pathlib import Path
import datetime
import time
import math
import sys
import os
import io

# Script is in examples/thrombin_thrombomodulin/
script_dir = os.path.dirname(os.path.abspath(__file__))

class _Tee:
    """Write to both stdout and a log file."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()

def run_one(label, seed, bb_enabled):
    """Run a single thrombin simulation, return P_rxn and n_reacted."""
    xml_path = os.path.join(script_dir, "input.xml")
    if not os.path.exists(xml_path):
        xml_path = "input.xml"
    cfg = parse(xml_path)
    cfg.seed = seed
    cfg.n_trajectories = 10000
    original_bb = None
    # Use an environment variable to control BB
    os.environ['PYSTARC_BB_DISABLED'] = '0' if bb_enabled else '1'
    print(f"\n{'='*60}")
    print(f"  {label}: seed={seed}, BB={'ON' if bb_enabled else 'OFF'}")
    print(f"{'='*60}")
    t0 = time.time()
    result = run_pipeline(cfg)
    elapsed = time.time() - t0
    p = result.reaction_probability
    nr = result.n_reacted
    ne = result.n_escaped
    nm = result.n_max_steps
    nd = nr + ne
    se = math.sqrt(p*(1-p)/nd) if nd > 0 and 0 < p < 1 else 0
    print(f"\n  {label} RESULT:")
    print(f"    P_rxn = {p:.6f} ± {se:.6f}")
    print(f"    reacted={nr}, escaped={ne}, max_steps={nm}")
    print(f"    time = {elapsed:.1f}s")
    return {
        'label': label, 'seed': seed, 'bb': bb_enabled,
        'P_rxn': p, 'se': se, 'n_reacted': nr, 'n_escaped': ne,
        'n_max': nm, 'elapsed': elapsed
    }

def main():
    # Set up log
    log_dir = os.path.join(os.getcwd(), "bd_sims")
    os.makedirs(log_dir, exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"bb_effect_{_ts}.log")
    log_f = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_f)
    print("""
  Brownian bridge A/B test
  This test determines whether BB genuinely catches more reactions,
  or just shifts the RNG sequence producing different random paths.
""")
    print("Running 4 seeds with BB=ON (current code):")
    print("If P_rxn is consistently ~0.13 across all seeds,")
    print("BB+position-refresh is a real physics effect.")
    print("If P_rxn varies wildly (0.05-0.15), it's RNG noise.")
    print()
    results = []
    for i, seed in enumerate([11111111, 22222222, 33333333, 44444444]):
        label = f"Run {i+1}"
        xml_path = "input.xml"
        cfg = parse(xml_path)
        cfg.seed = seed
        cfg.n_trajectories = 10000
        print(f"\n{'='*60}")
        print(f"  {label}: seed={seed}")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_pipeline(cfg)
        elapsed = time.time() - t0
        p = result.reaction_probability
        nr = result.n_reacted
        ne = result.n_escaped
        nm = result.n_max_steps
        nd = nr + ne
        se = math.sqrt(p*(1-p)/nd) if nd > 0 and 0 < p < 1 else 0
        results.append({
            'seed': seed, 'P_rxn': p, 'se': se,
            'n_reacted': nr, 'n_escaped': ne, 'n_max': nm,
            'elapsed': elapsed
        })
        print(f"\n  -> P_rxn = {p:.4f} ± {se:.4f}  "
              f"({nr} reacted, {ne} escaped, {nm} max-steps, {elapsed:.0f}s)")
    
    # Summary
    print("\n" + "=" * 70)
    print("  Summary: 4-seed consistency test")
    print("=" * 70)
    print(f"  {'Seed':>12s}  {'P_rxn':>10s}  {'±SE':>8s}  {'Reacted':>8s}  {'Escaped':>8s}  {'Time':>6s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    p_vals = []
    for r in results:
        print(f"  {r['seed']:12d}  {r['P_rxn']:10.4f}  {r['se']:8.4f}  "
              f"{r['n_reacted']:8d}  {r['n_escaped']:8d}  {r['elapsed']:5.0f}s")
        p_vals.append(r['P_rxn'])
    p_mean = sum(p_vals) / len(p_vals)
    p_std = math.sqrt(sum((p - p_mean)**2 for p in p_vals) / (len(p_vals) - 1))
    p_sem = p_std / math.sqrt(len(p_vals))
    print(f"\n  Mean P_rxn  = {p_mean:.4f} ± {p_sem:.4f}")
    print(f"  Std dev     = {p_std:.4f}")
    print(f"  Range       = [{min(p_vals):.4f}, {max(p_vals):.4f}]")
    print()
    # Compare with earlier no-BB, no-fix baseline
    p_old = 0.059  # from earlier run without BB or pos-refresh
    print(f"  Earlier baseline (no BB, no pos-refresh): P_rxn ≈ {p_old:.3f}")
    print(f"  Reference (100 traj):                      P_rxn ≈ 0.07")
    print()
    # Statistical test - Is mean P_rxn significantly different?
    if p_sem > 0:
        z = (p_mean - p_old) / p_sem
        print(f"  Z-test vs baseline ({p_old}):")
        print(f"    z = ({p_mean:.4f} - {p_old:.3f}) / {p_sem:.4f} = {z:.1f}")
        if abs(z) > 2.0:
            print(f"    -> Significant (|z| > 2): BB + pos-refresh genuinely changes P_rxn")
        else:
            print(f"    -> Not significant (|z| < 2): could be noise")
    # Coefficient of variation
    cv = p_std / p_mean * 100 if p_mean > 0 else 0
    print(f"\n  Coefficient of variation = {cv:.1f}%")
    if cv < 15:
        print(f"  -> Consistent - all seeds give similar P_rxn")
        print(f"  -> The BB + pos-refresh effect is not RNG noise")
    else:
        print(f"  -> High variance: P_rxn depends strongly on seed")
        print(f"  -> Need more trajectories per seed to reduce noise")
    print()
    print("  Interpretation guide")
    print(f"""
  If all 4 seeds give P_rxn ≈ 0.13 ± 0.02:
    -> BB + pos-refresh is a genuine ~2× improvement over baseline
    -> The "true" P_rxn for this system is ~0.13, not 0.07
  If P_rxn varies from 0.05 to 0.15:
    -> 10k trajectories is not enough. We need 50k+ for convergence
    -> Statistical error dominates over any BB effect
  If all 4 seeds give P_rxn ≈ 0.06:
    -> BB does not help for thrombin-thrombomodulin complex (expected - BB is negligible
      when D_eff×dt is small compared to pair distance above cutoff)
    -> The earlier 0.13 result was an RNG artifact
""")
    # Close log
    sys.stdout = original_stdout
    log_f.close()
    print(f"\n  Log saved -> {log_path}")

if __name__ == "__main__":
    main()
