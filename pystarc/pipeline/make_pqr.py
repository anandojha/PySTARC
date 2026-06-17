"""
PySTARC pipeline, Step 3: build the PQR files.

Brownian dynamics does not need explicit solvent, so we build a gas-phase
complex with no water, convert it to PQR, and then split it into a receptor
file and a ligand file. The three sub-steps are as follows. Step 3a runs tleap
with no solvent, loading the protein force field, the ligand library, and the
complex PDB, and writes complex.prmtop together with complex.pdb. Step 3b runs
ambpdb to produce complex.pqr, which carries a charge and a radius on every
atom. Step 3c splits that file into receptor.pqr, holding the protein only, and
ligand.pqr, holding the ligand only with each atom placed in its own residue.
"""

from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import subprocess
import shlex
import shutil


def _run(cmd: str, cwd: Path, step: str):
    print(f"    $ {cmd}")
    result = subprocess.run(shlex.split(cmd), shell=False, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        # tleap writes its errors to leap.log rather than stderr, so read that file.
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


# Build the gas-phase complex with tleap, using no solvent.
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
    Build the gas-phase complex with tleap. No water and no ions are added,
    because Brownian dynamics does not need them. Returns the pair
    (prmtop_path, complex_pdb_path).
    """
    _check_tool("tleap")
    ligand_resname = ligand_resname.strip().upper()
    prmtop_path = work_dir / "complex.prmtop"
    complex_pdb = work_dir / "complex.pdb"
    # Strip water and ions before handing the structure to tleap, since the
    # Brownian dynamics build is gas-phase. tleap cannot assign atom types to
    # WAT residues unless a water force field is loaded, and solvation is
    # unnecessary for Brownian dynamics preprocessing anyway.
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
    # Remove the intermediate files tleap leaves behind.
    (work_dir / "complex.inpcrd").unlink(missing_ok=True)
    for f in work_dir.glob("leap.log"):
        f.unlink(missing_ok=True)
    return prmtop_path, complex_pdb


# Run ambpdb to produce the combined PQR file.
def make_combined_pqr(prmtop_path: Path, complex_pdb: Path, work_dir: Path) -> Path:
    """
    Run ambpdb to produce a PQR file that carries a charge and a radius for
    every atom. Returns the path to the combined PQR file.
    """
    _check_tool("ambpdb")
    _check_tool("cpptraj")
    combined_pqr = work_dir / "complex.pqr"
    # First use cpptraj to generate an inpcrd file from the PDB.
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
    # Confirm that ambpdb actually produced usable output. The shell redirection
    # has no built in success check, so a silent failure would leave an empty or
    # missing file that only surfaces as a confusing error later in split_pqr.
    # Verify the file exists and holds at least one atom record here so that the
    # failure is reported at its source.
    if not combined_pqr.exists():
        raise RuntimeError(
            f"Step 'ambpdb' produced no output file at {combined_pqr}. Check that "
            f"ambpdb is installed and that the topology and coordinate files are valid."
        )
    has_atom = False
    with open(combined_pqr) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                has_atom = True
                break
    if not has_atom:
        raise RuntimeError(
            f"Step 'ambpdb' wrote {combined_pqr} but it contains no ATOM or HETATM "
            f"records. Check that the topology and coordinate files are valid."
        )
    # Remove the intermediate files.
    for f in [cpptraj_in, inpcrd]:
        f.unlink(missing_ok=True)
    return combined_pqr


# Split the combined PQR into a receptor file and a ligand file.
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
    """Extract the residue name from a single PQR line, reading columns 17 to 21.

    This uses the same column window as the canonical PQR parser so that
    four-character Amber residue names such as NTHR and CLYS are preserved
    whether they extend to the left (a non-space character in column 17) or to
    the right (a non-space character in column 21) of the standard field in
    columns 18 to 20.
    """
    return line[16:21].strip().upper()


def split_pqr(
    combined_pqr: Path, ligand_resname: str, work_dir: Path
) -> Tuple[Path, Path]:
    """
    Split the combined PQR into receptor.pqr and ligand.pqr.

    For the ligand, each atom is renumbered so that it gets its own unique
    residue number (the pqr_resid_for_each_atom step). This makes every atom an
    independent point charge in the Brownian dynamics run, which improves
    accuracy for small molecules. Returns the pair (receptor_pqr, ligand_pqr).
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
    # Write the receptor file.
    receptor_pqr = work_dir / "receptor.pqr"
    receptor_pqr.write_text("".join(rec_lines) + "END\n")
    # Renumber the ligand so that each atom gets its own residue number. This
    # is the pqr_resid_for_each_atom step from seekrtools. In the PQR format the
    # residue sequence number sits in columns 23 to 26, right-justified.
    renumbered = []
    for idx, line in enumerate(lig_lines, start=1):
        # Overwrite the residue-number field, which is columns 22 to 25 in zero-based indexing.
        new_line = line[:22] + f"{idx:4d}" + line[26:]
        renumbered.append(new_line)
    ligand_pqr = work_dir / "ligand.pqr"
    ligand_pqr.write_text("".join(renumbered) + "END\n")
    print(f"  Receptor PQR : {len(rec_lines):5d} atoms -> {receptor_pqr}")
    print(f"  Ligand PQR   : {len(lig_lines):5d} atoms -> {ligand_pqr} (renumbered)")
    return receptor_pqr, ligand_pqr
