"""
PySTARC pipeline - Step 2: Parameterize ligand
=============================================
Runs AmberTools to assign force field parameters to the ligand:
  1. antechamber  - assign AM1-BCC partial charges -> ligand.mol2
  2. parmchk2     - find missing parameters        -> ligand.frcmod
  3. tleap        - build Amber library file       -> ligand.lib
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import subprocess
import shutil
import os


def _run(cmd: str, cwd: Path, step: str):
    """Run a shell command, raise on failure with clear message."""
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed (exit {result.returncode}):\n"
            f"  cmd : {cmd}\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    return result


def _check_tool(name: str):
    if not shutil.which(name):
        raise EnvironmentError(
            f"'{name}' not found in PATH.\n"
            f"Install AmberTools:  conda install -c conda-forge ambertools -y"
        )


def parameterize(
    ligand_pdb: Path,
    ligand_resname: str,
    ligand_charge: int,
    work_dir: Path,
    ligand_ff: str = "gaff",
) -> Tuple[Path, Path, Path]:
    """
    Parameterize the ligand using AmberTools.
    Parameters
    ----------
    ligand_pdb     : path to ligand-only PDB
    ligand_resname : 3-letter residue name (e.g. 'BEN')
    ligand_charge  : net formal charge (integer, e.g. 1)
    work_dir       : working directory for all intermediate files
    ligand_ff      : 'gaff' or 'gaff2'
    Returns
    -------
    (mol2_path, frcmod_path, lib_path)
    """
    for tool in ["antechamber", "parmchk2", "tleap"]:
        _check_tool(tool)
    work_dir = Path(work_dir)
    ligand_resname = ligand_resname.strip().upper()
    resname_lower = ligand_resname.lower()
    mol2_path = work_dir / f"{resname_lower}.mol2"
    frcmod_path = work_dir / f"{resname_lower}.frcmod"
    lib_path = work_dir / f"{resname_lower}.lib"
    # 1. antechamber: AM1-BCC partial charges
    print("  antechamber - AM1-BCC charges ...")
    _run(
        f"antechamber -i {ligand_pdb.resolve()} -fi pdb "
        f"-bk {ligand_resname} "
        f"-o {mol2_path.name} -fo mol2 "
        f"-c bcc -nc {ligand_charge}",
        cwd=work_dir,
        step="antechamber",
    )
    # 2. parmchk2: missing force field parameters
    print("parmchk2 - missing parameters ...")
    _run(
        f"parmchk2 -i {mol2_path.name} -f mol2 -o {frcmod_path.name}",
        cwd=work_dir,
        step="parmchk2",
    )
    # 3. tleap: build Amber library file
    print("  tleap - building ligand library ...")
    tleap_script = work_dir / "save_ligand_lib.tleap"
    tleap_script.write_text(
        f"source leaprc.{ligand_ff}\n"
        f"{ligand_resname} = loadmol2 {mol2_path.name}\n"
        f"saveoff {ligand_resname} {lib_path.name}\n"
        f"quit\n"
    )
    _run(f"tleap -f {tleap_script.name}", cwd=work_dir, step="tleap-savelib")
    if not lib_path.exists():
        raise RuntimeError(f"tleap did not produce {lib_path}")
    # Cleanup antechamber intermediates
    for pattern in ["ANTECHAMBER*", "ATOMTYPE.INF", "sqm.*", "leap.log"]:
        for f in work_dir.glob(pattern):
            f.unlink(missing_ok=True)
    print(f"  Ligand mol2   : {mol2_path}")
    print(f"  Ligand frcmod : {frcmod_path}")
    print(f"  Ligand lib    : {lib_path}")
    return mol2_path, frcmod_path, lib_path
