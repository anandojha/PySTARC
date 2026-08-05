"""
First hitting point distribution export for the rigid protein-ligand engine.

A bimolecular association splits naturally into two regimes. The long approach
through solvent is diffusive and is what Brownian dynamics computes cheaply.
The final few angstroms, where the ligand settles into the pocket and
desolvates, needs atomistic molecular dynamics. Composing the two into one
rate requires handing configurations from the first stage to the second, and
the handoff is only unbiased if molecular dynamics is seeded from the poses
that diffusion actually delivers, in the proportions it delivers them.

That ensemble is the first hitting point distribution. For every trajectory
that reaches the reaction criterion, the configuration is recorded at the
*first* moment the criterion fires, not at any later crossing, so the
distribution carries the statistical weight of the diffusive stage.

The rigid engine already records exactly this. When the criterion first fires
the ligand centroid and orientation quaternion are stored, and the writer in
pipeline/output_writer.py emits them as encounters.csv with the columns

    traj_id, step, x, y, z, q0, q1, q2, q3, n_pairs_satisfied

For a rigid body those seven numbers are a complete description of the pose,
so the atomic coordinates follow exactly, with no loss, from

    r_i = R(q) · b_i + c

where b_i is atom i of the ligand in its body frame, meaning its coordinates
with the ligand centroid subtracted, R(q) is the rotation matrix of the unit
quaternion q ordered (w, x, y, z), and c is the recorded centroid. The engine
applies precisely this transform internally, so the reconstruction here is not
an approximation of what was simulated but the same operation.

The receptor is translated to the origin during setup, so the reconstructed
poses are already in the receptor frame and carry no further metadata.

This module is post-processing. It reads finished output and writes structures,
so it neither changes nor reruns any simulation.

The chain engine cannot be supported. A flexible chain has internal degrees of
freedom that a centroid and a quaternion do not describe, and the per bead
coordinates are never buffered, so its encounter conformations are
unrecoverable in principle rather than merely missing.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pystarc.structures.pqr_io import parse_pqr, write_pqr
from pystarc.transforms.quaternion import Quaternion

ENCOUNTERS = "encounters.csv"
LIGAND = "ligand.pqr"


def read_encounters(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read encounters.csv and return trajectory ids, steps, centroids, quaternions.

    The quaternion columns are q0 through q3 in (w, x, y, z) order, matching
    both the engine and the Quaternion constructor.
    """
    traj: List[int] = []
    step: List[int] = []
    cen: List[List[float]] = []
    quat: List[List[float]] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            traj.append(int(row["traj_id"]))
            step.append(int(row["step"]))
            cen.append([float(row["x"]), float(row["y"]), float(row["z"])])
            quat.append([float(row["q0"]), float(row["q1"]),
                         float(row["q2"]), float(row["q3"])])
    return (np.array(traj, dtype=np.int64), np.array(step, dtype=np.int64),
            np.array(cen, dtype=float), np.array(quat, dtype=float))


def poses(ligand_pqr: str | Path, cen: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Reconstruct every encounter pose as an (n_enc, n_atom, 3) coordinate array.

    The quaternions are renormalised before use. Integration leaves them unit
    to within round off, so the correction is small, but an unnormalised
    quaternion scales the rotation matrix by its squared norm and would stretch
    the ligand rather than merely rotate it.
    """
    mol = parse_pqr(ligand_pqr)
    body = mol.positions_array() - mol.centroid()
    out = np.empty((len(cen), body.shape[0], 3), dtype=float)
    for i, (c, q) in enumerate(zip(cen, quat)):
        R = Quaternion(*q).normalized().to_rotation_matrix()
        out[i] = body @ R.T + c
    return out


def write_fhpd(bd_sims: str | Path, out_dir: str | Path | None = None,
               pdb: bool = True, pqr: bool = True) -> int:
    """Export the first hitting point distribution for one finished system.

    Reads encounters.csv and ligand.pqr from bd_sims, writes one PQR per
    encounter and a single multi model PDB. Returns the number of poses
    written.
    """
    bd_sims = Path(bd_sims)
    enc_path, lig_path = bd_sims / ENCOUNTERS, bd_sims / LIGAND
    for p in (enc_path, lig_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found, so this system has no FHPD to export")

    out_dir = Path(out_dir) if out_dir is not None else bd_sims / "fhpd"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj, step, cen, quat = read_encounters(enc_path)
    if len(traj) == 0:
        print(f"  {bd_sims}: encounters.csv holds no rows, nothing to export")
        return 0

    norms = np.linalg.norm(quat, axis=1)
    coords = poses(lig_path, cen, quat)
    mol = parse_pqr(lig_path)

    if pqr:
        for i in range(len(traj)):
            for a, xyz in zip(mol.atoms, coords[i]):
                a.position = xyz
            write_pqr(mol, out_dir / f"fhpd_{i + 1:05d}.pqr")

    if pdb:
        with open(out_dir / "fhpd.pdb", "w") as fh:
            fh.write("REMARK  PySTARC first hitting point distribution\n")
            fh.write(f"REMARK  {len(traj)} poses, receptor frame, receptor at the origin\n")
            for i in range(len(traj)):
                fh.write(f"MODEL     {i + 1:4d}\n")
                fh.write(f"REMARK  traj_id {traj[i]}  step {step[i]}\n")
                for j, (a, xyz) in enumerate(zip(mol.atoms, coords[i])):
                    name = a.name if a.name else "X"
                    resname = a.residue_name if a.residue_name else "UNK"
                    # PDB is fixed column, not whitespace separated. Serial is
                    # 7-11, atom name 13-16, altLoc 17 and blank, resName 18-20,
                    # chain 22, resSeq 23-26, then coordinates at 31-38, 39-46
                    # and 47-54. Dropping the blank altLoc shifts every
                    # coordinate one column left and a strict parser misreads it.
                    fh.write(
                        f"HETATM{j + 1:5d} {name:<4s} {resname:>3s} A"
                        f"{a.residue_index:4d}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        f"  1.00  0.00\n"
                    )
                fh.write("ENDMDL\n")
            fh.write("END\n")

    r = np.linalg.norm(cen, axis=1)
    print(f"  {bd_sims}")
    print(f"    poses            {len(traj)}")
    print(f"    atoms per pose   {coords.shape[1]}")
    print(f"    centroid radius  {r.min():.2f} to {r.max():.2f} A, mean {r.mean():.2f}")
    print(f"    quaternion norm  {norms.min():.6f} to {norms.max():.6f}")
    print(f"    step at contact  {step.min()} to {step.max()}, median {int(np.median(step))}")
    print(f"    written to       {out_dir}")
    return len(traj)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export the first hitting point distribution from a finished "
                    "rigid protein-ligand run.")
    ap.add_argument("bd_sims", nargs="+",
                    help="one or more bd_sims directories holding encounters.csv and ligand.pqr")
    ap.add_argument("--out", default=None,
                    help="output directory, default is fhpd/ inside each bd_sims")
    ap.add_argument("--no-pqr", action="store_true", help="skip the per pose PQR files")
    ap.add_argument("--no-pdb", action="store_true", help="skip the multi model PDB")
    args = ap.parse_args()

    total = 0
    for d in args.bd_sims:
        try:
            total += write_fhpd(d, args.out, pdb=not args.no_pdb, pqr=not args.no_pqr)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
    print(f"\n  {total} poses across {len(args.bd_sims)} system(s)")


if __name__ == "__main__":
    main()
