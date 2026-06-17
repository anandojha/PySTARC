"""
PySTARC pipeline, step 1: extract the ligand and the receptor.

This module reads a complex PDB file containing the protein and the ligand
together and splits it into two files. The file ligand.pdb holds just the
ligand atoms, selected by residue name. The file receptor.pdb holds just the
protein atoms, with water, the ligand, and ions removed.
"""

from __future__ import annotations
from typing import List, Tuple
from pathlib import Path

# Residue names that are never part of the receptor protein.
_SOLVENT_RESIDUES = {
    "WAT",
    "HOH",
    "TIP",
    "SOL",  # water models
    "NA",
    "CL",
    "K",
    "MG",
    "CA",  # ions
    "Na+",
    "Cl-",
    "K+",  # Amber ion names
}
_SKIP_RECORD_TYPES = {
    "TER",
    "END",
    "CRYST1",
    "REMARK",
    "HEADER",
    "TITLE",
    "COMPND",
    "SOURCE",
    "SEQRES",
}


def _is_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _residue_name(line: str) -> str:
    """Return the residue name from a PDB ATOM or HETATM line, which sits in columns 17 to 20."""
    return line[17:20].strip()


def extract(
    pdb_path: str | Path, ligand_resname: str, work_dir: str | Path
) -> Tuple[Path, Path]:
    """
    Split a complex PDB into receptor.pdb and ligand.pdb.

    The argument pdb_path is the path to the combined PDB holding the protein
    and the ligand. The argument ligand_resname is the three-letter residue
    name of the ligand, for example 'BEN'. The argument work_dir is the output
    directory. The function returns the paths to receptor.pdb and ligand.pdb as
    a pair.
    """
    pdb_path = Path(pdb_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ligand_resname = ligand_resname.strip().upper()
    receptor_lines: List[str] = []
    ligand_lines: List[str] = []
    with open(pdb_path) as f:
        for line in f:
            if not _is_atom_line(line):
                continue
            resname = _residue_name(line).upper()
            if resname == ligand_resname:
                ligand_lines.append(line)
            elif resname not in _SOLVENT_RESIDUES:
                receptor_lines.append(line)
    if not ligand_lines:
        raise ValueError(
            f"No atoms with residue name '{ligand_resname}' found in {pdb_path}.\n"
            f"Check <ligand_resname> in your pystarc_input.xml."
        )
    if not receptor_lines:
        raise ValueError(
            f"No receptor atoms found in {pdb_path} after removing "
            f"ligand '{ligand_resname}' and solvent."
        )
    ligand_pdb = work_dir / "ligand.pdb"
    receptor_pdb = work_dir / "receptor.pdb"
    ligand_pdb.write_text("".join(ligand_lines) + "END\n")
    receptor_pdb.write_text("".join(receptor_lines) + "END\n")
    print(f"  Receptor : {len(receptor_lines):5d} atoms -> {receptor_pdb}")
    print(f"  Ligand   : {len(ligand_lines):5d} atoms -> {ligand_pdb}")
    return receptor_pdb, ligand_pdb
