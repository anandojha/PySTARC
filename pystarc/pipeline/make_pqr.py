"""
PySTARC pipeline - Step 3: Build PQR files
=========================================

No solvation needed for BD. We build a gas-phase (no water) complex,
convert to PQR, then split into receptor.pqr and ligand.pqr.
Steps:
  3a. tleap (no solvent) - load protein ff + ligand lib + complex.pdb
                         -> complex.prmtop + complex.pdb
  3b. ambpdb             -> complex.pqr  (charges + radii on every atom)
  3c. split              -> receptor.pqr (protein only)
                           ligand.pqr   (ligand only, each atom own residue)
"""

from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import subprocess
import shutil


def _run(cmd: str, cwd: Path, step: str):
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        # tleap writes errors to leap.log, not stderr - read it
        leap_log = ""
        for log_name in ["leap.log", "leap.out"]:
            log_path = cwd / log_name
            if log_path.exists():
                leap_log = log_path.read_text()[-3000:]
                break
        raise RuntimeError(
            f"'{step}' failed (exit {result.returncode}):\n"
            f"  cmd    : {cmd}\n"
            f"  stdout : {result.stdout[-500:]}\n"
            f"  stderr : {result.stderr[-500:]}\n"
            f"  leap.log (last 3000 chars):\n{leap_log if leap_log else '(no leap.log found)'}"
        )
    return result


def _check_tool(name: str):
    if not shutil.which(name):
        raise EnvironmentError(
            f"'{name}' not found in PATH.\n"
            f"Install AmberTools:  conda install -c conda-forge ambertools -y"
        )


# tleap (gas phase, no solvent)
def build_complex(
    pdb_path: Path,
    mol2_path: Path,
    frcmod_path: Path,
    lib_path: Path,
    ligand_resname: str,
    work_dir: Path,
    protein_ff: str = "ff14SB",
    ligand_ff: str = "gaff",
) -> Tuple[Path, Path]:
    """
    Build gas-phase complex with tleap (no water, no ions - BD does not need them).
    Returns (prmtop_path, complex_pdb_path).
    """
    _check_tool("tleap")
    ligand_resname = ligand_resname.strip().upper()
    prmtop_path = work_dir / "complex.prmtop"
    complex_pdb = work_dir / "complex.pdb"
    # Strip water/ions before passing to tleap - BD is gas-phase.
    # tleap cannot type WAT residues without a water force field loaded,
    # and solvation is unnecessary for Brownian dynamics preprocessing.
    SOLVENT = {"WAT", "HOH", "TIP", "TIP3", "SOL", "TP3", "SPC"}
    stripped_pdb = work_dir / "complex_nowater.pdb"
    kept = 0
    skipped_res: set = set()
    with open(pdb_path) as fin, open(stripped_pdb, "w") as fout:
        for line in fin:
            tag = line[:6].strip()
            if tag in ("ATOM", "HETATM"):
                res = line[17:20].strip().upper()
                if res in SOLVENT:
                    skipped_res.add(res)
                    continue
            fout.write(line)
            if tag in ("ATOM", "HETATM"):
                kept += 1
    if skipped_res:
        print(
            f"  [3a] Stripped solvent residues: {skipped_res} "
            f"({kept} atoms remain for gas-phase build)"
        )
    tleap_script = work_dir / "build_complex.tleap"
    tleap_script.write_text(
        f"source leaprc.protein.{protein_ff}\n"
        f"source leaprc.{ligand_ff}\n"
        f"set default PBRadii mbondi2\n"
        f"loadoff {lib_path.name}\n"
        f"loadAmberParams {frcmod_path.name}\n"
        f"complex = loadpdb {stripped_pdb.name}\n"
        f"saveamberparm complex {prmtop_path.name} complex.inpcrd\n"
        f"savepdb complex {complex_pdb.name}\n"
        f"quit\n"
    )
    print("  [3a] tleap - building gas-phase complex (no solvent) ...")
    print(
        f"    tleap script:\n"
        + "\n".join(f"      {l}" for l in tleap_script.read_text().splitlines())
    )
    _run(f"tleap -f {tleap_script.name}", cwd=work_dir, step="tleap-complex")
    # cleanup
    (work_dir / "complex.inpcrd").unlink(missing_ok=True)
    for f in work_dir.glob("leap.log"):
        f.unlink(missing_ok=True)
    return prmtop_path, complex_pdb


# ambpdb -> combined PQR
def make_combined_pqr(prmtop_path: Path, complex_pdb: Path, work_dir: Path) -> Path:
    """
    Run ambpdb to produce a PQR file (charges + radii for every atom).
    Returns path to combined PQR.
    """
    _check_tool("ambpdb")
    _check_tool("cpptraj")
    combined_pqr = work_dir / "complex.pqr"
    # First generate inpcrd from pdb using cpptraj
    cpptraj_in = work_dir / "get_inpcrd.cpptraj"
    cpptraj_in.write_text(
        f"parm {prmtop_path.name}\n"
        f"trajin {complex_pdb.name}\n"
        f"trajout complex.rst\n"
        f"run\n"
    )
    print("  [3b] cpptraj - generating inpcrd from pdb ...")
    _run(f"cpptraj -i {cpptraj_in.name}", cwd=work_dir, step="cpptraj")
    inpcrd = work_dir / "complex.inpcrd"
    rst = work_dir / "complex.rst"
    if rst.exists():
        rst.rename(inpcrd)
    print("  [3b] ambpdb - generating combined PQR ...")
    _run(
        f"ambpdb -p {prmtop_path.name} -c {inpcrd.name} -pqr > {combined_pqr.name}",
        cwd=work_dir,
        step="ambpdb",
    )
    # cleanup intermediates
    for f in [cpptraj_in, inpcrd]:
        f.unlink(missing_ok=True)
    return combined_pqr


# Split PQR into receptor + ligand
_SKIP_RESIDUES = {
    "WAT",
    "HOH",
    "TIP",
    "SOL",
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "Na+",
    "Cl-",
    "K+",
}


def _pqr_residue(line: str) -> str:
    return line[17:20].strip().upper()


def split_pqr(
    combined_pqr: Path, ligand_resname: str, work_dir: Path
) -> Tuple[Path, Path]:
    """
    Split combined PQR into receptor.pqr and ligand.pqr.

    For the ligand: renumber each atom so it gets its own unique residue
    number (the pqr_resid_for_each_atom step). This makes each atom an
    independent point charge in BD, which improves accuracy for small
    molecules.
    Returns (receptor_pqr, ligand_pqr).
    """
    ligand_resname = ligand_resname.strip().upper()
    rec_lines: List[str] = []
    lig_lines: List[str] = []
    with open(combined_pqr) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = _pqr_residue(line)
            if resname == ligand_resname:
                lig_lines.append(line)
            elif resname not in _SKIP_RESIDUES:
                rec_lines.append(line)
    if not lig_lines:
        raise ValueError(
            f"No ligand atoms (resname='{ligand_resname}') found in {combined_pqr}"
        )
    # Write receptor.pqr
    receptor_pqr = work_dir / "receptor.pqr"
    receptor_pqr.write_text("".join(rec_lines) + "END\n")
    # Renumber ligand: each atom gets its own residue number
    # This is the pqr_resid_for_each_atom step from seekrtools.
    # PQR format: cols 23-26 are residue sequence number (right-justified)
    renumbered = []
    for idx, line in enumerate(lig_lines, start=1):
        # Overwrite residue number field (cols 22-25, 0-based)
        new_line = line[:22] + f"{idx:4d}" + line[26:]
        renumbered.append(new_line)
    ligand_pqr = work_dir / "ligand.pqr"
    ligand_pqr.write_text("".join(renumbered) + "END\n")
    print(f"  Receptor PQR : {len(rec_lines):5d} atoms -> {receptor_pqr}")
    print(f"  Ligand PQR   : {len(lig_lines):5d} atoms -> {ligand_pqr} (renumbered)")
    return receptor_pqr, ligand_pqr
