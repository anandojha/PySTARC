#!/usr/bin/env python3
import numpy as np
import glob
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_root = HERE
while _root != "/" and not os.path.isfile(os.path.join(_root, "run_pystarc.py")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)
from pystarc.pipeline.input_parser import parse
from pystarc.pipeline.chain_pipeline import _ensure_chain_apbs_grids, parse_pqr

os.chdir(HERE)
xml = sys.argv[1] if len(sys.argv) > 1 else "input.xml"
cfg = parse(xml)
cc = cfg.chain

print(f"receptor   {cfg.receptor_pqr}")
print(f"fine span  {cfg.apbs_fglen} A over {cfg.apbs_dime} points")
print(f"target     {cc.target_grid_dx}")
print(f"born       {cc.born_grid_dx}")
print()


def header(path):
    n = sp = org = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) > 7 and t[:4] == ["object", "1", "class", "gridpositions"]:
                n = int(t[-3])
            elif t[:1] == ["origin"]:
                org = np.array([float(v) for v in t[1:4]])
            elif t[:1] == ["delta"] and sp is None and float(t[1]) > 0:
                sp = float(t[1])
            if n and sp and org is not None:
                return n, sp, org
    return n, sp, org

stale = []
if os.path.exists(cc.target_grid_dx):
    rec = parse_pqr(cfg.receptor_pqr)
    cen = np.array([[a.x, a.y, a.z] for a in rec.atoms]).mean(axis=0)
    n, sp, org = header(cc.target_grid_dx)
    box = org + sp * (n - 1) / 2.0
    if n != cfg.apbs_dime:
        stale.append(f"dime {n} against {cfg.apbs_dime}")
    if abs(sp * (n - 1) - cfg.apbs_fglen) > 1e-3:
        stale.append(f"span {sp*(n-1):.2f} A against {cfg.apbs_fglen} A")
    if np.linalg.norm(box - cen) > 0.05:
        stale.append(f"centre off the receptor by {np.linalg.norm(box-cen):.2f} A")

if stale:
    print("  stale grids: " + "; ".join(stale))
    for p in glob.glob("apbs_output/*.dx"):
        os.remove(p)
    print("  cleared, rebuilding")
    print()

_ensure_chain_apbs_grids(cfg)

print()
for p in (cc.target_grid_dx, cc.born_grid_dx):
    print(f"  {'OK ' if os.path.exists(p) else 'MISSING'}  {p}")
