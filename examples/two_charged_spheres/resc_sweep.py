#!/usr/bin/env python3
"""Escape-radius convergence experiment for the two-charged-sphere control.

Runs the PySTARC Brownian-dynamics simulation at several escape radii with the
b-sphere start radius held fixed at bd_milestone_radius = 10 A, and compares the
resulting k_on. The recollision-corrected rate is predicted to be independent of
the escape radius (analytic k_on = 1.5598e10 M^-1 s^-1, computed in
resc_analytic_sweep.py). A k_on that is flat across escape radii, within the
statistical error bars, confirms that the simulation reproduces this
independence and validates the r_esc = 2b convention used in the manuscript. A
systematic drift would mean 2b is too small and a larger escape radius is needed.

The escape radius is set through the <r_escape> input tag. A value of 0 (the
default) means 2 x bd_milestone_radius, so r_esc = 20 A here is identical to the
default run.

Usage
-----
    cd examples/two_charged_spheres
    module load cuda
    python resc_sweep.py                 # sweep 20 40 80 A at b = 10 A
    python resc_sweep.py 20 40 80 160
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent.parent / "run_pystarc.py"
ANALYTIC_KON = 1.5598e10


def make_input(r_esc):
    """Write an input variant that fixes the escape radius and its own work_dir."""
    base = (HERE / "input.xml").read_text()
    tag = int(round(r_esc))
    work = f"bd_sims_resc{tag}"
    txt = re.sub(r"<work_dir>.*?</work_dir>", f"<work_dir>{work}</work_dir>", base)
    txt = txt.replace("</pystarc>", f"  <r_escape>{r_esc}</r_escape>\n</pystarc>")
    xml = HERE / f"input_resc{tag}.xml"
    xml.write_text(txt)
    return xml, HERE / work


def run_one(r_esc):
    xml, work = make_input(r_esc)
    print(f"\n========== r_esc = {r_esc} A   (work_dir {work.name}) ==========")
    subprocess.run([sys.executable, str(RUNNER), str(xml)], check=True)
    return json.loads((work / "results.json").read_text())


def main():
    radii = [float(x) for x in sys.argv[1:]] or [20.0, 40.0, 80.0]
    rows = [(r, run_one(r)) for r in radii]
    print("\n" + "=" * 96)
    print("  Escape-radius convergence, two charged spheres (b = 10 A fixed)")
    print(f"  Analytic k_on = {ANALYTIC_KON:.4e} M^-1 s^-1 (independent of escape radius)")
    print("=" * 96)
    hdr = (
        f"  {'r_esc/A':>8s}  {'k_b':>9s}  {'P_rxn':>9s}  {'k_on':>12s}  "
        f"{'95% CI (M^-1 s^-1)':>27s}  {'vs r_esc[0]':>11s}  {'vs analytic':>11s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    k0 = rows[0][1]["k_on"]
    for r, rj in rows:
        kon, lo, hi = rj["k_on"], rj["k_on_low"], rj["k_on_high"]
        kb, pr = rj.get("k_b", float("nan")), rj.get("P_rxn", float("nan"))
        print(
            f"  {r:8.1f}  {kb:9.4f}  {pr:9.5f}  {kon:12.4e}  "
            f"[{lo:.3e}, {hi:.3e}]  {(kon - k0) / k0:+11.2%}  "
            f"{(kon - ANALYTIC_KON) / ANALYTIC_KON:+11.2%}"
        )
    print(
        "\n  Read: if k_on is flat across escape radii within the 95% CIs, and matches\n"
        "  the analytic value, the recollision correction holds in simulation and the\n"
        "  r_esc = 2b convention is validated. A systematic drift of k_on with the\n"
        "  escape radius would mean 2b is too small and the larger escape radius is\n"
        "  the correct production convention."
    )


if __name__ == "__main__":
    main()
