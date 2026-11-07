"""
Atom numbers in rxns_xml are 1-based positions in the PQR.

BrownDye 2 reads the same numbers as PQR serials. The two agree when the
serials of a PQR run 1..N. When they do not, and the numbers are also valid
serials, the same file selects different atoms in the two programs, and
PySTARC warns.
"""

from __future__ import annotations

from pystarc.pipeline.geometry import (
    ReactionPair,
    check_rxn_atom_numbers,
    check_rxn_atom_range,
)


def _pqr(path, serials):
    lines = [
        f"ATOM  {s:5d}  CA  ALA A{i + 1:4d}    {float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  0.0000 1.5000"
        for i, s in enumerate(serials)
    ]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_no_warning_when_numbered_1_to_n(tmp_path, capsys):
    rec = _pqr(tmp_path / "r.pqr", [1, 2, 3, 4])
    lig = _pqr(tmp_path / "l.pqr", [1, 2])
    assert check_rxn_atom_numbers([ReactionPair(3, 1, 5.0)], rec, lig) == []
    assert capsys.readouterr().out == ""


def test_warns_when_serials_skip(tmp_path, capsys):
    """Numbers 1 and 4 are both serials and positions, and 4 names different atoms."""
    rec = _pqr(tmp_path / "r.pqr", [1, 2, 4, 5, 6])
    lig = _pqr(tmp_path / "l.pqr", [1, 2])
    msgs = check_rxn_atom_numbers(
        [ReactionPair(0, 0, 5.0), ReactionPair(3, 1, 5.0)], rec, lig
    )
    assert len(msgs) == 1
    assert "number 4 is the atom with serial 5" in msgs[0]
    assert "1 of the 2 receptor atoms" in msgs[0]
    assert "Warning" in capsys.readouterr().out


def test_warns_for_the_ligand_too(tmp_path):
    rec = _pqr(tmp_path / "r.pqr", [1, 2, 3])
    lig = _pqr(tmp_path / "l.pqr", [1, 3, 4])
    msgs = check_rxn_atom_numbers([ReactionPair(0, 2, 5.0)], rec, lig)
    assert len(msgs) == 1 and "ligand" in msgs[0]


def test_no_warning_for_offset_serials(tmp_path):
    """A ligand split from a complex keeps serials that start after the
    receptor's. Positions 4 and 12 cannot be serials, so they are not ambiguous."""
    rec = _pqr(tmp_path / "r.pqr", list(range(1, 148)))
    lig = _pqr(tmp_path / "l.pqr", list(range(148, 164)))
    pairs = [ReactionPair(61, 3, 5.0), ReactionPair(70, 11, 5.0)]
    assert check_rxn_atom_numbers(pairs, rec, lig) == []


def test_no_warning_when_only_unreferenced_atoms_differ(tmp_path):
    rec = _pqr(tmp_path / "r.pqr", [1, 2, 3, 9])
    lig = _pqr(tmp_path / "l.pqr", [1, 2])
    assert check_rxn_atom_numbers([ReactionPair(0, 0, 5.0)], rec, lig) == []


def test_rxn_atom_range_accepts_the_ghost_positions():
    """The last position of each molecule, the GHO ghost atom, is in range."""
    check_rxn_atom_range([ReactionPair(147, 21, 8.0)], 148, 22)


def test_rxn_atom_range_stops_a_pair_past_the_end():
    """A pair one past the end of the ligand or a zero position stops the run."""
    import pytest

    with pytest.raises(ValueError):
        check_rxn_atom_range([ReactionPair(147, 22, 8.0)], 148, 22)
    with pytest.raises(ValueError):
        check_rxn_atom_range([ReactionPair(-1, 0, 8.0)], 148, 22)
