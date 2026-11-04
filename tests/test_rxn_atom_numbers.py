"""
Atom numbers in rxns_xml are 1-based positions in the PQR (C23).

BrownDye 2 reads the same numbers as PQR serials. The shipped thrombin receptor
skips two serials, so its BrownDye 2 file selected atoms 2-7.5 Å from the
intended ones in PySTARC. The example now carries positions, and PySTARC warns
when a referenced atom's serial differs from its position.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pystarc.pipeline.geometry import (
    ReactionPair,
    _parse_rxns_xml_criteria,
    check_rxn_atom_numbers,
)
from pystarc.structures.pqr_io import parse_pqr_records

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "thrombin_thrombomodulin"


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


def test_no_warning_for_offset_ligand_serials(tmp_path):
    """A ligand split from a complex keeps serials 148.. ; positions 4 and 12
    cannot be serials, so they are not ambiguous (the beta-cyclodextrin rows)."""
    rec = _pqr(tmp_path / "r.pqr", list(range(1, 148)))
    lig = _pqr(tmp_path / "l.pqr", list(range(148, 164)))
    pairs = [ReactionPair(61, 3, 5.0), ReactionPair(70, 11, 5.0)]
    assert check_rxn_atom_numbers(pairs, rec, lig) == []


def test_no_warning_when_only_unreferenced_atoms_differ(tmp_path):
    rec = _pqr(tmp_path / "r.pqr", [1, 2, 3, 9])
    lig = _pqr(tmp_path / "l.pqr", [1, 2])
    assert check_rxn_atom_numbers([ReactionPair(0, 0, 5.0)], rec, lig) == []


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples not shipped")
class TestThrombinExample:
    def test_criterion_atoms_are_the_intended_polar_atoms(self):
        """The 21 receptor atoms are N, O or the side-chain polar atoms of the
        BrownDye 2 thrombin file, not the hydrogens and carbons that the serial
        numbers selected as positions."""
        pairs, n_needed = _parse_rxns_xml_criteria(EXAMPLE / "rxns.xml")
        rec = parse_pqr_records(EXAMPLE / "receptor.pqr")
        lig = parse_pqr_records(EXAMPLE / "ligand.pqr")
        assert len(pairs) == 21 and n_needed == 5
        names = [rec[p.rec_index].name.strip() for p in pairs]
        assert all(n[0] in "NO" for n in names), names
        assert all(lig[p.lig_index].name.strip()[0] in "NO" for p in pairs)
        # Lys113 NZ, not HE3 as in the published run.
        lys = [rec[p.rec_index] for p in pairs if rec[p.rec_index].resid == 113]
        assert [a.name.strip() for a in lys] == ["NZ"]

    def test_example_triggers_the_numbering_warning(self, capsys):
        """The example PQR skips serials, so the warning explains the convention."""
        pairs, _ = _parse_rxns_xml_criteria(EXAMPLE / "rxns.xml")
        msgs = check_rxn_atom_numbers(
            pairs, EXAMPLE / "receptor.pqr", EXAMPLE / "ligand.pqr"
        )
        assert len(msgs) == 1 and "receptor" in msgs[0]
