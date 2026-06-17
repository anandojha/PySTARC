"""Tests for robust chain detection in the PQR whitespace fallback parser.

These exercise pystarc.structures.pqr_io._parse_whitespace, the fallback used
when strict PDB column parsing fails. The fix makes chain detection robust to
numeric chain identifiers, which were previously misread as the residue index
and shifted every following field. Standard alphabetic chain lines and no chain
lines must continue to parse byte for byte identically.
"""

from pystarc.structures.pqr_io import _parse_whitespace


def test_numeric_chain_is_not_misread_as_resid():
    """A numeric chain identifier is read as the chain, not as the residue index.

    The fields after the chain are resid, x, y, z, charge, radius. With a numeric
    chain the old parser shifted all of these by one token, so the residue index,
    coordinates, charge, and radius were all corrupted. This checks every field is
    placed correctly.
    """
    line = "ATOM      5  CA  ALA 1   42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "1"
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_multi_digit_numeric_chain_with_element():
    """A multi digit numeric chain and a trailing element symbol are both read correctly."""
    line = "ATOM      5  CA  ALA 10   7       1.000   2.000   3.000  0.500  1.800 C"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "10"
    assert rec.resid == 7
    assert rec.x == 1.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == "C"


def test_alphabetic_chain_parses_as_before():
    """A standard alphabetic chain line parses with the chain and all fields intact."""
    line = "ATOM      5  CA  ALA A   42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "A"
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_alphabetic_chain_with_element():
    """An alphabetic chain line with a trailing element symbol parses correctly."""
    line = "ATOM      5  CA  ALA B   42       1.000   2.000   3.000  0.500  1.800 N"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == "B"
    assert rec.resid == 42
    assert rec.element == "N"


def test_no_chain_parses_as_before():
    """A no chain line keeps an empty chain and reads the residue index from token four."""
    line = "ATOM      5  CA  ALA     42       1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.x == 1.0
    assert rec.y == 2.0
    assert rec.z == 3.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
    assert rec.element == ""


def test_no_chain_with_element():
    """A no chain line with a trailing element symbol parses correctly."""
    line = "ATOM      5  CA  ALA     42       1.000   2.000   3.000  0.500  1.800 O"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.element == "O"


def test_no_chain_collapsed_negative_coordinate():
    """A no chain line whose negative x coordinate retains a decimal point still parses."""
    line = "ATOM      5  CA  ALA     42      -1.000   2.000   3.000  0.500  1.800"
    rec = _parse_whitespace(line, "ATOM")
    assert rec is not None
    assert rec.chain == ""
    assert rec.resid == 42
    assert rec.x == -1.0
    assert rec.charge == 0.5
    assert rec.radius == 1.8
