#!/usr/bin/env python3
"""
Split a PySTARC simulation into N independent runs for multi-GPU execution.
Creates bd_sims/bd_1/, bd_sims/bd_2/, ... each with adjusted seed and trajectory count.
DX grids are symlinked from bd_sims/ to avoid copying large files.
If bd_sims/ does not exist, runs 1 trajectory first to generate APBS grids.
"""
import xml.etree.ElementTree as ET
import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Split PySTARC run into N independent jobs")
    parser.add_argument("xml", help="Input XML file")
    parser.add_argument("--n-splits", type=int, default=4, help="Number of splits (default 4)")
    args = parser.parse_args()
    pystarc_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runner = os.path.join(pystarc_root, "run_pystarc.py")
    combiner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combine_data.py")
    tree = ET.parse(args.xml)
    root = tree.getroot()
    total_traj = int(root.findtext("n_trajectories", "100000"))
    base_seed = int(root.findtext("seed", "1"))
    per_split = total_traj // args.n_splits
    base_dir = os.path.dirname(os.path.abspath(args.xml))
    bd_sims = os.path.join(base_dir, "bd_sims")
    # Auto-generate APBS grids if bd_sims/ does not exist
    if not os.path.isdir(bd_sims) or not any(f.endswith(".dx") for f in os.listdir(bd_sims)):
        print("  No APBS grids found. Generating grids (1 trajectory) ...")
        grid_xml = os.path.join(base_dir, "_grid_gen.xml")
        root.find("n_trajectories").text = "1"
        root.find("max_steps").text = "1"
        tree.write(grid_xml, xml_declaration=True, encoding="UTF-8")
        ret = subprocess.run([sys.executable, runner, grid_xml], cwd=base_dir)
        os.remove(grid_xml)
        if ret.returncode != 0:
            print("  Error: grid generation failed.")
            return
        # Clean grid-gen artifacts (keep only .dx, .cache, .pqr)
        for f in os.listdir(bd_sims):
            fpath = os.path.join(bd_sims, f)
            if os.path.isfile(fpath) and not f.endswith((".dx", ".cache", ".pqr")):
                os.remove(fpath)
        # Reload original XML
        tree = ET.parse(args.xml)
        root = tree.getroot()
        print("  Grids ready.\n")
    link_exts = (".dx", ".cache", ".pqr")
    link_files = [f for f in os.listdir(bd_sims)
                  if os.path.isfile(os.path.join(bd_sims, f)) and f.endswith(link_exts)]
    for i in range(1, args.n_splits + 1):
        run_dir = os.path.join(bd_sims, f"bd_{i}")
        os.makedirs(run_dir, exist_ok=True)
        for f in link_files:
            dst = os.path.join(run_dir, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join("..", f), dst)
        root.find("n_trajectories").text = str(per_split)
        root.find("seed").text = str(base_seed + i * 11111111)
        root.find("work_dir").text = "."
        # Resolve relative paths to absolute so they work from bd_sims/bd_N/
        for tag in ["rxns_xml", "receptor_pqr", "ligand_pqr"]:
            el = root.find(tag)
            if el is not None and el.text and not os.path.isabs(el.text.strip()):
                el.text = os.path.join(base_dir, el.text.strip())
        out_xml = os.path.join(run_dir, "input.xml")
        tree.write(out_xml, xml_declaration=True, encoding="UTF-8")
        print(f"  bd_sims/bd_{i}/: {per_split:,} trajectories, seed={base_seed + i * 11111111}")
    print(f"\n  Total: {per_split * args.n_splits:,} trajectories across {args.n_splits} runs")
    print(f"\n  To run each split:")
    for i in range(1, args.n_splits + 1):
        print(f"    cd bd_sims/bd_{i} && python {runner} input.xml && cd ../..")
    print(f"\n  Then combine:")
    # combine auto-detects bd_sims/bd_N directories
    print(f"    python {combiner}")

if __name__ == "__main__":
    main()
