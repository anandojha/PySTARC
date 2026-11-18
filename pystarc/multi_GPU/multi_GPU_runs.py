#!/usr/bin/env python3
"""
Split a single PySTARC simulation into N independent runs so they can be spread
across several GPUs. The script first runs one trajectory in <work_dir> to build
the APBS grids or reuse them by their key, then creates <work_dir>/bd_1,
<work_dir>/bd_2, and so on, each with its own seed and its share of the
trajectories (the first N mod n_splits shards run one more). Each run directory
links the grids, their key files and the hydrodynamic radius caches, and writes
its own centred PQR files. split.json records the shards for combine_data.py.
"""

import xml.etree.ElementTree as ET
from dataclasses import fields
import subprocess
import argparse
import json
import sys
import os


def _set_or_create(root, tag, value):
    """Set the text of the child element named tag, creating the element if the
    input XML does not already contain it. Tags such as n_trajectories, max_steps,
    seed, and work_dir are optional in a PySTARC input file and fall back to
    defaults when omitted, so they may be absent from an otherwise valid file."""
    el = root.find(tag)
    if el is None:
        el = ET.SubElement(root, tag)
    el.text = value
    return el


KEEP_SETUP_FILES = {"rxns.xml", "build_record.json", "chain.json", "reaction_pairs.json"}


def split_multi_gpu(xml_path: str, n_splits: int = 4) -> int:
    pystarc_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if pystarc_root not in sys.path:
        sys.path.insert(0, pystarc_root)
    from pystarc.pipeline.input_parser import PySTARCConfig

    defaults = {f.name: f.default for f in fields(PySTARCConfig)}
    runner = os.path.join(pystarc_root, "run_pystarc.py")
    combiner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "combine_data.py"
    )
    abs_xml = os.path.abspath(xml_path)
    tree = ET.parse(abs_xml)
    root = tree.getroot()
    if root.tag == "config":
        from pystarc.pipeline.build_inputs import build_from_config, read_config

        raw_cfg = read_config(abs_xml)
        src_dir = os.path.dirname(abs_xml)
        raw_bd = raw_cfg.get("bd_work_dir", str(defaults["work_dir"]))
        bd_check = raw_bd if os.path.isabs(raw_bd) else os.path.join(src_dir, raw_bd)
        for d in [bd_check] + [os.path.join(bd_check, f"bd_{i}") for i in range(1, n_splits + 1)]:
            if any(os.path.exists(os.path.join(d, f)) for f in ("results.json", "checkpoint.npz")):
                print(f"  Error: {d} holds an earlier run. Remove it or set another work_dir.")
                return 1
        abs_xml, _ = build_from_config(abs_xml)
        tree = ET.parse(abs_xml)
        root = tree.getroot()

    text = lambda tag: (root.findtext(tag) or "").strip()
    total_traj = int(text("n_trajectories") or defaults["n_trajectories"])
    base_seed = int(text("seed") or defaults["seed"])
    if not 1 <= n_splits <= total_traj:
        raise ValueError(f"--n-splits must be between 1 and n_trajectories ({total_traj})")
    per_split, extra = divmod(total_traj, n_splits)
    base_dir = os.path.dirname(abs_xml)
    w_txt = text("work_dir") or str(defaults["work_dir"])
    bd_sims = w_txt if os.path.isabs(w_txt) else os.path.join(base_dir, w_txt)
    for d in [bd_sims] + [os.path.join(bd_sims, f"bd_{i}") for i in range(1, n_splits + 1)]:
        if any(os.path.exists(os.path.join(d, f)) for f in ("results.json", "checkpoint.npz")):
            print(f"  Error: {d} holds an earlier run. Remove it or set another work_dir.")
            return 1
    # One trajectory in <work_dir> builds the grids, or reuses them by their key.
    print("  Building or reusing the APBS grids (1 trajectory) ...")
    grid_xml = os.path.join(base_dir, "_grid_gen.xml")
    _set_or_create(root, "n_trajectories", "1")
    _set_or_create(root, "max_steps", "1")
    _set_or_create(root, "n_gpus", "1")
    tree.write(grid_xml, xml_declaration=True, encoding="UTF-8")
    ret = subprocess.run([sys.executable, runner, grid_xml], cwd=base_dir)
    if os.path.exists(grid_xml):
        os.remove(grid_xml)
    if ret.returncode != 0:
        print("  Error: grid generation failed.")
        return 1
    if not os.path.isdir(bd_sims):
        print(f"  Error: grid generation did not create {bd_sims}.")
        return 1
    # Keep the grid, key, cache, structure and setup files of the grid pass.
    keep_input = os.path.abspath(abs_xml) == os.path.abspath(os.path.join(bd_sims, "input.xml"))
    for f in os.listdir(bd_sims):
        fpath = os.path.join(bd_sims, f)
        if not os.path.isfile(fpath):
            continue
        if f.endswith((".dx", ".key", ".cache", ".pqr")) or f in KEEP_SETUP_FILES:
            continue
        if keep_input and f == "input.xml":
            continue
        os.remove(fpath)
    # Reload the original XML, since the grid pass overwrote its counts.
    tree = ET.parse(abs_xml)
    root = tree.getroot()
    print("  Grids ready.\n")
    link_files = [
        f
        for f in os.listdir(bd_sims)
        if os.path.isfile(os.path.join(bd_sims, f)) and f.endswith((".dx", ".key", ".cache"))
    ]
    shards = []
    for i in range(1, n_splits + 1):
        run_dir = os.path.join(bd_sims, f"bd_{i}")
        os.makedirs(run_dir, exist_ok=True)
        for f in ("receptor.pqr", "ligand.pqr"):
            if os.path.islink(os.path.join(run_dir, f)):
                os.remove(os.path.join(run_dir, f))
        for f in link_files:
            dst = os.path.join(run_dir, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join("..", f), dst)
        n_i = per_split + (1 if i <= extra else 0)
        seed_i = base_seed + i * 11111111
        _set_or_create(root, "n_trajectories", str(n_i))
        _set_or_create(root, "seed", str(seed_i))
        _set_or_create(root, "n_gpus", "1")
        _set_or_create(root, "work_dir", ".")
        # Turn any relative input paths into absolute ones so they resolve
        # cleanly when the run executes from inside the shard folder.
        for tag in ["pdb", "rxns_xml", "receptor_pqr", "ligand_pqr"]:
            el = root.find(tag)
            if el is not None and el.text and el.text.strip() and not os.path.isabs(el.text.strip()):
                el.text = os.path.join(base_dir, el.text.strip())
        tree.write(os.path.join(run_dir, "input.xml"), xml_declaration=True, encoding="UTF-8")
        shards.append({"dir": f"bd_{i}", "seed": seed_i, "n_trajectories": n_i})
        print(f"  {run_dir}: {n_i:,} trajectories, seed={seed_i}")
    with open(os.path.join(bd_sims, "split.json"), "w") as f:
        json.dump(
            {"n_splits": n_splits, "n_trajectories": total_traj, "shards": shards},
            f,
            indent=2,
        )
    print(f"\n  Total: {total_traj:,} trajectories across {n_splits} runs")
    print(f"\n  To run each split:")
    for s in shards:
        print(f"    cd {os.path.join(bd_sims, s['dir'])} && python {runner} input.xml")
    print(f"\n  Then combine:")
    print(f"    python {combiner} --bd-sims {bd_sims}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Split PySTARC run into N independent jobs"
    )
    parser.add_argument("xml", help="Configuration or input XML file")
    parser.add_argument(
        "--n-splits", type=int, default=4, help="Number of splits (default 4)"
    )
    args = parser.parse_args()
    try:
        return split_multi_gpu(args.xml, n_splits=args.n_splits)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
