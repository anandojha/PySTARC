"""
PQR file input and output for PySTARC.

This module provides the single canonical PQR parser used by all library code
and example scripts.

It handles several variations of the PQR format. Both ATOM and HETATM records
are read. The chain identifier at PDB column 22 is optional, so files with or
without a chain column are both supported. Collapsed spacing between adjacent
numeric fields is tolerated through the whitespace fallback, for example a
negative x coordinate that runs into the space before y, or only a single space
separating the charge from the radius. A trailing element symbol is captured
when present. Four-character Amber residue names that extend into the chain
column, such as NTHR or CLYS, are handled. Blank lines and lines beginning with
REMARK or END are skipped.

The public interface consists of four pieces. PQRRecord is a dataclass holding
all eleven PQR fields plus the element symbol. parse_pqr_records returns a list
of PQRRecord objects and is the single source of truth for parsing. parse_pqr
returns a Molecule and is a backward-compatible wrapper. write_pqr writes a
Molecule out as a PQR file.

The parsing strategy is to try strict PDB-column parsing first, then fall back
to a whitespace split with chain-column auto-detection whenever the strict
parse fails. Lines that both modes reject are skipped silently, matching the
behavior of the earlier parser.
"""

from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


# The public data type returned by the parser.
@dataclass
class PQRRecord:
    """A single atom record parsed from a PQR file.

    It holds the eleven standard PQR fields together with an optional trailing
    element symbol.
    """

    record_type: str
    serial: int
    name: str
    resname: str
    chain: str
    resid: int
    x: float
    y: float
    z: float
    charge: float
    radius: float
    element: str = ""


# The primary parser and single source of truth for reading PQR files.
def parse_pqr_records(path: str | Path) -> List[PQRRecord]:
    """Parse a PQR file into a list of PQRRecord objects.

    This reads the file line by line, skips blank lines and REMARK or END
    records, and keeps only ATOM and HETATM records. Each kept line is parsed
    first by the strict column parser and then, if that fails, by the
    whitespace fallback. Lines that both parsers reject are skipped.
    """
    path = Path(path)
    records: List[PQRRecord] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("REMARK") or stripped.startswith("END"):
                continue
            record_type = line[:6].strip()
            if record_type not in ("ATOM", "HETATM"):
                continue
            rec = _parse_strict(line, record_type)
            if rec is None:
                rec = _parse_whitespace(line, record_type)
            if rec is not None:
                records.append(rec)
    return records


# Strict parser based on the fixed PDB column positions.
def _parse_strict(line: str, record_type: str) -> Optional[PQRRecord]:
    """Parse one PQR line using the fixed PDB column positions.

    This reads each field from its standard PDB column range and returns None if
    the line is too short or any numeric field fails to convert, which lets the
    caller fall back to the whitespace parser.
    """
    if len(line) < 54:
        return None
    try:
        serial = int(line[6:11])
        name = line[12:16].strip()
        resname = line[16:21].strip()
        chain = line[21:22].strip()
        resid = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        tail = line[54:].split()
        if len(tail) < 2:
            return None
        charge = float(tail[0])
        radius = float(tail[1])
        element = tail[2] if len(tail) >= 3 else ""
        return PQRRecord(
            record_type=record_type,
            serial=serial,
            name=name,
            resname=resname,
            chain=chain,
            resid=resid,
            x=x,
            y=y,
            z=z,
            charge=charge,
            radius=radius,
            element=element,
        )
    except (ValueError, IndexError):
        return None


# Whitespace fallback parser used when strict column parsing fails.
def _parse_whitespace(line: str, record_type: str) -> Optional[PQRRecord]:
    """Parse one PQR line by splitting on whitespace and detecting the chain.

    This is the fallback for lines whose columns do not line up with the PDB
    specification, for example when collapsed spacing merges adjacent numeric
    fields. The chain column is detected from the structure of the residue and
    coordinate tokens rather than from whether the fifth token alone looks
    numeric, which keeps numeric chain identifiers from being misread as the
    residue index. After the optional chain the layout is fixed as resid, x, y,
    z, charge, radius, with resid an integer and the coordinates floating point
    values that always carry a decimal point. A chain column is therefore
    present when the sixth token is an integer residue index, regardless of
    whether the fifth token is alphabetic or numeric. No chain is present when
    the fifth token is the integer residue index and the sixth token is the x
    coordinate, which is not an integer. The remaining fields are read relative
    to the resulting offset.
    """
    parts = line.split()
    if len(parts) < 10:
        return None
    # Rescue a chain-bearing line whose residue index abuts the following
    # (often negative) coordinate and collapses two fields into one token, e.g.
    # "... A 603-12.345 6.789 ...". Splitting the leading integer off restores
    # the 11-token chain layout instead of silently dropping the atom.
    if len(parts) == 10 and not _is_int(parts[4]) and not _is_int(parts[5]):
        s = parts[5]
        j = 0
        while j < len(s) and s[j].isdigit():
            j += 1
        if 0 < j < len(s):
            parts = parts[:5] + [s[:j], s[j:]] + parts[6:]
    try:
        if len(parts) >= 11 and _is_int(parts[5]):
            chain = parts[4]
            resid = int(parts[5])
            off = 6
        elif _is_int(parts[4]):
            chain = ""
            resid = int(parts[4])
            off = 5
        else:
            return None
        serial = int(parts[1]) if _is_int(parts[1]) else 0
        name = parts[2]
        resname = parts[3]
        x = float(parts[off])
        y = float(parts[off + 1])
        z = float(parts[off + 2])
        charge = float(parts[off + 3])
        radius = float(parts[off + 4])
        element = parts[off + 5] if len(parts) > off + 5 else ""
        return PQRRecord(
            record_type=record_type,
            serial=serial,
            name=name,
            resname=resname,
            chain=chain,
            resid=resid,
            x=x,
            y=y,
            z=z,
            charge=charge,
            radius=radius,
            element=element,
        )
    except (ValueError, IndexError):
        return None


def _is_int(s: str) -> bool:
    """Return True if the string can be converted to an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


# Backward-compatible parser that returns a Molecule.
def parse_pqr(path: str | Path) -> Molecule:
    """Parse a PQR file into a Molecule.

    This keeps the original signature for existing callers. Callers that need
    the chain identifier, element symbol, or record type should use
    parse_pqr_records instead. Atoms with no chain are assigned chain "A".
    """
    path = Path(path)
    mol = Molecule(name=path.stem)
    for i, rec in enumerate(parse_pqr_records(path)):
        mol.atoms.append(
            Atom(
                index=i,
                name=rec.name,
                residue_name=rec.resname,
                residue_index=rec.resid,
                chain=rec.chain or "A",
                x=rec.x,
                y=rec.y,
                z=rec.z,
                charge=rec.charge,
                radius=rec.radius,
            )
        )
    return mol


def write_pqr(mol: Molecule, path: str | Path) -> None:
    """Write a Molecule to a .pqr file.

    The output begins with a REMARK header naming the molecule, followed by one
    ATOM record per atom with its name, residue name, residue index, Cartesian
    coordinates in angstrom, charge, and radius, and closes with an END line.
    Atoms with an empty name or residue name are written as "X" and "UNK".
    """
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"REMARK  Generated by PySTARC  molecule={mol.name}\n")
        for i, a in enumerate(mol.atoms):
            name = a.name if a.name else "X"
            resname = a.residue_name if a.residue_name else "UNK"
            fh.write(
                f"ATOM  {i+1:5d} {name:<4s} {resname:<4s} "
                f"{a.residue_index:4d}    "
                f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  "
                f"{a.charge:7.4f} {a.radius:6.4f}\n"
            )
        fh.write("END\n")
