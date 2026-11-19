#!/usr/bin/env python3
"""Standalone offline helper for inspecting candidate reaction criteria.

Collects all contact-picking, atom-filtering, site-atom, and closest-approach
functions removed from PySTARC into one standalone script. Never imported or
invoked by pystarc, run_pystarc.py, or config.xml.

Example:
  python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --atom-filter polar --n-pairs 6 --pair-cutoff 7.0
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from pystarc.forces.soft_repulsion import SoftRepulsion

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

SEED = 20261108
BATCH = 20_000
POLAR_ELEMENTS = {"N", "O", "S"}
SITE_ATOM = "PKT"

ATOM_FILTERS = {
    "all":            "Matches any heavy atom against any heavy atom",
    "polar":          "Matches N O or S on both sides",
    "nonpolar":       "Matches carbon on both sides",
    "any_polar":      "Matches N O or S on either side",
    "receptor_polar": "Matches N O or S on the receptor side",
}


def keeper(kind):
    return {
        "all": lambda a, b: True,
        "polar": lambda a, b: a in POLAR_ELEMENTS and b in POLAR_ELEMENTS,
        "nonpolar": lambda a, b: a == "C" and b == "C",
        "any_polar": lambda a, b: a in POLAR_ELEMENTS or b in POLAR_ELEMENTS,
        "receptor_polar": lambda a, b: a in POLAR_ELEMENTS,
    }[kind]


def round_quarter(x):
    return math.floor(x / 0.25 + 0.5) * 0.25


def contacts_picked(rec, lig, c):
    keep = keeper(c["atom_filter"])
    raw_excl = c.get("excluded") or ()
    skip = {(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_excl}
    found = []
    for i, a in enumerate(rec):
        if a["el"] == "H" or a["name"] in skip:
            continue
        for j, b in enumerate(lig):
            if b["el"] == "H" or b["name"] in skip:
                continue
            d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
            if d < c["search_cutoff"] and keep(a["el"], b["el"]):
                found.append((d, i, j))
    found.sort()
    out, seen = [], set()
    for d, i, j in found:
        if rec[i]["resid"] in seen:
            continue
        seen.add(rec[i]["resid"])
        out.append((i, j, c["pair_cutoff"], d))
        if len(out) >= c["n_pairs"]:
            break
    return out


def write_site_atom(work_dir, rec, p):
    path = os.path.join(work_dir, "receptor.pqr")
    with open(path) as fh:
        lines = [l for l in fh.read().split("\n") if l.strip() and l.strip() != "END"]
    serial = max(int(l[6:11]) for l in lines if l.startswith(("ATOM", "HETATM"))) + 1
    resid = max(a["resid"] for a in rec) + 1
    lines.append(f"HETATM{serial:5d} {SITE_ATOM:<4s} {SITE_ATOM:<4s}{resid:5d}    "
                 f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  {0.0:7.4f}  {0.0:6.4f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")


def parse_structure_atoms(path):
    """Parse ATOM/HETATM records from a .pdb or .pqr file into dicts with name, el, resname, resid, xyz."""
    atoms = []
    if path.lower().endswith(".pqr"):
        from pystarc.structures.pqr_io import parse_pqr_records
        for r in parse_pqr_records(path):
            nm = r.name.lstrip("0123456789").upper()
            el = (r.element.upper() if r.element.isalpha() and len(r.element) <= 2
                  else (nm[:2] if nm[:2] in ("CL", "BR", "ZN", "MG", "FE", "CU", "CA") else nm[:1]))
            atoms.append(dict(name=r.name, el=el, resname=r.resname, resid=r.resid,
                              xyz=np.array([r.x, r.y, r.z])))
        return atoms
    with open(path) as fh:
        for l in fh:
            if not l.startswith(("ATOM", "HETATM")) or len(l) < 54:
                continue
            aname = l[12:16].strip()
            resname = l[17:20].strip()
            resid = int(l[22:26])
            col_el = l[76:78].strip().upper() if len(l) >= 78 else ""
            nm = aname.lstrip("0123456789").upper()
            el = col_el if col_el else (nm[:2] if nm[:2] in ("CL", "BR", "ZN", "MG", "FE", "CU") else nm[:1])
            atoms.append(dict(name=aname, el=el, resname=resname, resid=resid,
                              xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return atoms


def _rotations(q, xp):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def scan(wall, body, ref, n, ball, rng, xp, e_max=2.0):
    best, n_ok, pose = math.inf, 0, None
    for s0 in range(0, n, BATCH):
        b = min(BATCH, n - s0)
        v = rng.normal(size=(b, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        r = ball * rng.random(b) ** (1.0 / 3.0)
        c = ref + v * r[:, None]
        q = rng.normal(size=(b, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        L = np.einsum("nij,aj->nai", _rotations(q, np), body) + c[:, None, :]
        _, _, e, _ = wall(xp.asarray(L), xp.asarray(c))
        ok = (e.get() if xp is not np else e) <= e_max
        n_ok += int(ok.sum())
        if ok.any():
            k = np.flatnonzero(ok)[r[ok].argmin()]
            if r[k] < best:
                best, pose = float(r[k]), L[k]
    return (None if not math.isfinite(best) else best), n_ok, pose


def _qmul(a, b, xp):
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return xp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _axis_angle(v, xp):
    ang = xp.linalg.norm(v, axis=1)
    ax = v / xp.maximum(ang, 1e-12)[:, None]
    return xp.concatenate([xp.cos(ang / 2)[:, None], ax * xp.sin(ang / 2)[:, None]], axis=1)


def descend(wall, bound, ref, n_starts, n_steps, xp, e_max=2.0, seed=SEED, anchor=None):
    c0 = bound.mean(axis=0)
    body = xp.asarray(bound - c0)
    rng = np.random.default_rng(seed)
    c = xp.asarray(c0 + rng.normal(scale=0.5, size=(n_starts, 3)))
    c[0] = xp.asarray(c0)
    ax = rng.normal(size=(n_starts, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ang = rng.uniform(0.0, np.radians(20.0), n_starts)
    ang[0] = 0.0
    q = _axis_angle(xp.asarray(ax * ang[:, None]), xp)
    ref_x = xp.asarray(ref)
    found = xp.full(n_starts, xp.nan)
    poses = xp.zeros((n_starts,) + body.shape)
    for _ in range(n_steps):
        L = xp.einsum("nij,aj->nai", _rotations(q, xp), body) + c[:, None, :]
        F, T, E, _ = wall(L, c)
        point = c if anchor is None else L[:, anchor, :]
        new = (E <= e_max) & xp.isnan(found)
        found = xp.where(new, xp.linalg.norm(point - ref_x, axis=1), found)
        poses = xp.where(new[:, None, None], L, poses)
        live = xp.isnan(found)
        if not bool(live.any()):
            break
        fn = xp.linalg.norm(F, axis=1)
        tn = xp.linalg.norm(T, axis=1)
        c = xp.where(live[:, None], c + 0.05 * F / xp.maximum(fn, 1e-12)[:, None], c)
        q = xp.where(live[:, None], _qmul(_axis_angle(0.02 * T / xp.maximum(tn, 1e-12)[:, None], xp), q, xp), q)
        q = q / xp.linalg.norm(q, axis=1)[:, None]
    f = found.get() if xp is not np else found
    ok = ~np.isnan(f)
    if not ok.any():
        return None, 0, None
    k = int(np.nanargmin(f))
    return float(f[k]), int(ok.sum()), (poses[k].get() if xp is not np else poses[k])


def closest_approach(rec_xyz, rec_radii, lig_xyz, lig_radii, ref, anchor=None, bound=True, ball=16.0,
                     n_coarse=2_000_000, n_dense=8_000_000, widen=1.5, descent_below=None, n_starts=2048,
                     n_steps=4000, e_max=2.0, seed=SEED):
    xp = cp if cp is not None else np
    rec_xyz, lig_xyz = np.asarray(rec_xyz, float), np.asarray(lig_xyz, float)
    ref = np.asarray(ref, float)
    wall = SoftRepulsion(rec_xyz, np.asarray(rec_radii, float), np.asarray(lig_radii, float), xp=xp)
    body = lig_xyz - (lig_xyz.mean(axis=0) if anchor is None else lig_xyz[anchor])
    rng = np.random.default_rng(seed)
    coarse, n0, p0 = scan(wall, body, ref, n_coarse, ball, rng, xp, e_max)
    dense, n1, p1 = None, 0, None
    if coarse is not None:
        dense, n1, p1 = scan(wall, body, ref, n_dense, coarse + widen, rng, xp, e_max)
    scans = [x for x in (coarse, dense) if x is not None]
    descent, n2, p2 = None, 0, None
    if bound and (descent_below is None or (scans and min(scans) < descent_below)):
        descent, n2, p2 = descend(wall, lig_xyz, ref, n_starts, n_steps, xp, e_max, seed, anchor)
    found = [(x, p) for x, p in ((coarse, p0), (dense, p1), (descent, p2)) if x is not None]
    best, pose = min(found, key=lambda f: f[0]) if found else (None, None)
    return dict(closest=None if best is None else round(best, 2), anchor=anchor, coarse=coarse, dense=dense,
                descent=descent, accessible_coarse=n0, accessible_dense=n1, descent_reached=n2, ball=ball,
                n_coarse=n_coarse, n_dense=n_dense, widen=widen, descent_below=descent_below, n_starts=n_starts,
                n_steps=n_steps, e_max=e_max, seed=seed, device="gpu" if xp is not np else "cpu",
                pose=None if pose is None else pose.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", required=True, help="Prepared receptor .pdb or .pqr file")
    ap.add_argument("--ligand", required=True, help="Prepared ligand .pdb or .pqr file")
    ap.add_argument("--search-cutoff", type=float, default=6.0, help="Bound-pose contact search cutoff in A")
    ap.add_argument("--n-pairs", type=int, default=6, help="Maximum number of contact pairs")
    ap.add_argument("--atom-filter", choices=sorted(ATOM_FILTERS), default="polar", help="Element filter")
    ap.add_argument("--pair-cutoff", type=float, default=7.0, help="Reaction distance per pair in A")
    args = ap.parse_args(argv)

    rec = parse_structure_atoms(args.receptor)
    lig = parse_structure_atoms(args.ligand)
    picked = contacts_picked(rec, lig, dict(
        atom_filter=args.atom_filter,
        search_cutoff=args.search_cutoff,
        n_pairs=args.n_pairs,
        pair_cutoff=args.pair_cutoff,
    ))
    if len(picked) > 1 and len(picked) % 2 == 1:
        picked = picked[:-1]
    n_needed = 1 if len(picked) == 1 else len(picked) // 2
    multi_res_lig = len({b["resid"] for b in lig}) > 1
    print("<criterion>")
    print("  <mode>contacts</mode>")
    print("  <receptor>")
    for i, _j, _cut, _d in picked:
        a = rec[i]
        print(f'    <target resid="{a["resid"]}" atom="{a["name"]}" resname="{a["resname"]}"/>')
    print("  </receptor>")
    print("  <ligand>")
    for _i, j, _cut, _d in picked:
        b = lig[j]
        if multi_res_lig:
            print(f'    <target resid="{b["resid"]}" atom="{b["name"]}" resname="{b["resname"]}"/>')
        else:
            print(f'    <atom name="{b["name"]}"/>')
    print("  </ligand>")
    print(f"  <pair_cutoff>{args.pair_cutoff}</pair_cutoff>")
    print(f"  <n_needed>{n_needed}</n_needed>")
    print("</criterion>")


if __name__ == "__main__":
    main()
