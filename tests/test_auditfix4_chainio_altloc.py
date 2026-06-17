"""The chain PDB parser must keep the first (altLoc A) conformer of an atom."""
import os
import tempfile

import numpy as np

from pystarc.structures.chain_io import _parse_pdb_chain_for_beads


def _atom(serial, name, altloc, resname, chain, resid, x, y, z):
    # Build a fixed-column PDB ATOM record (name in cols 13-16, altLoc 17,
    # resName 18-20, chain 22, resSeq 23-26, x/y/z in 31-54).
    return (
        "ATOM  "
        + f"{serial:>5}"
        + " "
        + f"{name:^4}"
        + altloc
        + f"{resname:>3}"
        + " "
        + chain
        + f"{resid:>4}"
        + " "
        + "   "
        + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
    )


def test_parser_keeps_first_altloc_conformer():
    # SER 10 has OG in two conformers: altLoc A at (1,1,1) first, altLoc B at
    # (9,9,9) second. The parser must keep the first (altLoc A).
    """The parser keeps the first alternate-location conformer when a residue lists multiple altLocs."""
    lines = [
        _atom(1, "N", " ", "SER", "A", 10, 0.0, 0.0, 0.0),
        _atom(2, "OG", "A", "SER", "A", 10, 1.0, 1.0, 1.0),
        _atom(3, "OG", "B", "SER", "A", 10, 9.0, 9.0, 9.0),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        residues = _parse_pdb_chain_for_beads(path, chain_id="A")
    finally:
        os.unlink(path)
    assert len(residues) == 1
    assert np.allclose(residues[0]["atoms"]["OG"], [1.0, 1.0, 1.0])
