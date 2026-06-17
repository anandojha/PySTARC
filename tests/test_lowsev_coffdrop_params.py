"""
Low-severity robustness tests for pystarc.simulation.coffdrop_params.

These cover two bad-input crash fixes:

1. TabulatedPotential now raises a clear ValueError at construction when the
   energy values array is empty, instead of a lazy IndexError later when value()
   is evaluated.
2. _parse_ff now raises a clear ValueError when a <pairs>, <bond_angles>, or
   <dihedral_angles> block has a <distance>/<angle> list shorter than two
   values, instead of a bare IndexError while unpacking x_min and x_max.

The tests exercise pure-Python parsing and construction logic and need no GPU,
APBS, or external binaries.
"""

import numpy as np
import pytest

from pystarc.simulation.coffdrop_params import TabulatedPotential, _parse_ff


def _make_potential(values):
    return TabulatedPotential(
        x_min=0.0,
        x_max=1.0,
        values=np.asarray(values, dtype=np.float64),
        residues=(0, 0),
        atoms=(0, 0),
        orders=(0, 0),
        index=0,
    )


def test_empty_values_raises_value_error():
    """An empty energy table must fail clearly at construction time."""
    with pytest.raises(ValueError):
        _make_potential([])


def test_nonempty_values_construct_normally():
    """The healthy path with a populated table still constructs and evaluates."""
    pot = _make_potential([1.0, 2.0, 3.0, 4.0])
    # Boundary clamping returns the first and last tabulated values unchanged.
    assert pot.value(-1.0) == 1.0
    assert pot.value(2.0) == 4.0


def _write_xml(tmp_path, body):
    path = tmp_path / "coffdrop.xml"
    path.write_text("<coffdrop>\n" + body + "\n</coffdrop>\n")
    return str(path)


def test_pairs_short_distance_list_raises(tmp_path):
    """A <pairs> block with a one-element <distance> list must raise clearly."""
    xml = _write_xml(
        tmp_path,
        "<pairs><distance>0.0</distance>"
        "<potentials></potentials></pairs>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_bond_angles_short_angle_list_raises(tmp_path):
    """A <bond_angles> block with a one-element <angle> list must raise clearly."""
    xml = _write_xml(
        tmp_path,
        "<bond_angles><angle>0.0</angle>"
        "<potentials></potentials></bond_angles>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_dihedral_angles_short_angle_list_raises(tmp_path):
    """A <dihedral_angles> block with a one-element <angle> list must raise."""
    xml = _write_xml(
        tmp_path,
        "<dihedral_angles><angle>0.0</angle>"
        "<potentials></potentials></dihedral_angles>",
    )
    with pytest.raises(ValueError):
        _parse_ff(xml)


def test_healthy_pairs_block_parses(tmp_path):
    """A well-formed <pairs> block with a two-element distance list parses fine."""
    xml = _write_xml(
        tmp_path,
        "<pairs><distance>0.0 10.0</distance>"
        "<potentials>"
        "<potential><orders>0 0</orders><index>0</index>"
        "<residues>0 0</residues><atoms>0 0</atoms>"
        "<data>1.0 2.0 3.0 4.0</data></potential>"
        "</potentials></pairs>",
    )
    type_map, pairs, angles, dihedrals = _parse_ff(xml)
    assert len(pairs) == 1
    assert pairs[0].x_min == 0.0
    assert pairs[0].x_max == 10.0
