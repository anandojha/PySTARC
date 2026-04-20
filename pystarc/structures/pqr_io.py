"""
PQR I/O for PySTARC.

Single canonical PQR parser used by all library code and example scripts.

Supported format variations:

  Record types
      ATOM and HETATM.
  Chain column
      Optional. Files with or without a chain identifier at PDB column 22
      are both handled.
  Numeric spacing
      Collapsed spacing between adjacent numeric fields (e.g. negative x
      coordinate eating the space before y, or single-space separation
      between charge and radius) is handled by the whitespace fallback.
  Trailing element symbol
      Optional. Captured when present.
  Extended residue names
      Four-character Amber residue names that extend into the chain
      column (e.g. NTHR, CLYS) are handled.
  Comment and blank lines
      Blank lines and lines beginning with REMARK or END are skipped.

Public API:

  PQRRecord          Dataclass with all eleven PQR fields plus element.
  parse_pqr_records  Returns list[PQRRecord]. Single source of truth.
  parse_pqr          Returns Molecule. Backward-compatible wrapper.
  write_pqr          Writes a Molecule out as PQR. Behavior unchanged.

Parse strategy:

  For each candidate line, strict PDB-column parsing is tried first and
  the whitespace-split fallback with chain-column auto-detection runs
  on any parse failure. Malformed lines that both modes reject are
  skipped silently, matching prior parser behavior.
"""

from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Public data type
@dataclass
class PQRRecord:
    """One parsed record from a PQR file."""
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

# Primary parser
def parse_pqr_records(path: str | Path) -> List[PQRRecord]:
    """Parse a PQR file into a list of PQRRecord."""
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

# Strict column parser (PDB spec)
def _parse_strict(line: str, record_type: str) -> Optional[PQRRecord]:
    """Parse one PQR line using fixed PDB column positions."""
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

# Whitespace fallback parser
def _parse_whitespace(line: str, record_type: str) -> Optional[PQRRecord]:
    """Parse one PQR line by whitespace split, detecting chain presence."""
    parts = line.split()
    if len(parts) < 10:
        return None
    try:
        if _is_int(parts[4]):
            chain = ""
            resid = int(parts[4])
            off = 5
        elif len(parts) >= 11 and _is_int(parts[5]):
            chain = parts[4]
            resid = int(parts[5])
            off = 6
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
    try:
        int(s)
        return True
    except ValueError:
        return False

# Backward-compatible API
def parse_pqr(path: str | Path) -> Molecule:
    """Parse a PQR file into a Molecule.

    Backward-compatible signature. Callers that need the chain identifier,
    element symbol, or record type should use parse_pqr_records instead.
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
    """Write a Molecule to a .pqr file."""
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"REMARK  Generated by PySTARC  molecule={mol.name}\n")
        for i, a in enumerate(mol.atoms):
            name = a.name if a.name else "X"
            resname = a.residue_name if a.residue_name else "UNK"
            fh.write(
                f"ATOM  {i+1:5d}  {name:<4s} {resname:<4s} "
                f"{a.residue_index:4d}    "
                f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  "
                f"{a.charge:7.4f} {a.radius:6.4f}\n"
            )
        fh.write("END\n")
