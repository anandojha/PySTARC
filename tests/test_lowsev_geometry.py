"""
Low-severity robustness tests for pystarc.pipeline.geometry.

These cover the warnings added for silent fallbacks and the per-reaction
n_needed scoping in the flattened rxns XML parser. None of them require a GPU,
APBS, or any external binary.

The fixes under test:
  parse_pqr        emits a warning when the lenient fallback defaults a missing
                   radius column to 1.5 A.
  _parse_rxns_xml_criteria
                   scopes n_needed per reaction so a reaction without an
                   explicit n_needed no longer inherits the value of an earlier
                   reaction, and warns when reactions declare conflicting
                   n_needed values.
"""

import warnings

import pytest

from pystarc.pipeline.geometry import parse_pqr, _parse_rxns_xml_criteria


def test_lenient_pqr_fallback_warns_and_defaults_radius(tmp_path):
    """A 9-field PQR with no radius column triggers the lenient fallback,
    which warns and defaults the radius to 1.5 A without changing the value."""
    pqr = tmp_path / "no_radius.pqr"
    pqr.write_text(
        "ATOM 1 N ALA 1 0.0 0.0 0.0 -0.5\n"
        "ATOM 2 C ALA 1 1.0 0.0 0.0 0.3\n"
    )
    with pytest.warns(UserWarning, match="defaulting radius to 1.5"):
        atoms = parse_pqr(pqr)
    assert len(atoms) == 2
    # The defaulted radius value itself is unchanged at 1.5 A.
    assert all(a.radius == 1.5 for a in atoms)


def test_pqr_with_radius_column_no_warning(tmp_path):
    """A PQR line that includes a radius column does not trigger the fallback
    warning and parses the radius value as given."""
    pqr = tmp_path / "with_radius.pqr"
    pqr.write_text("ATOM 1 N ALA 1 0.0 0.0 0.0 -0.5 1.85\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        atoms = parse_pqr(pqr)
    assert len(atoms) == 1
    assert atoms[0].radius == 1.85


def _write_two_reaction_xml(path, second_has_n_needed):
    """Write a two-reaction rxns XML. The first reaction declares n_needed=2.
    The second reaction declares n_needed=3 only when second_has_n_needed is
    True, otherwise it omits the element."""
    second_nn = "<n_needed>3</n_needed>" if second_has_n_needed else ""
    path.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <n_needed>2</n_needed>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "      <pair><atoms>2 12</atoms><distance>5.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        f"      {second_nn}\n"
        "      <pair><atoms>3 13</atoms><distance>4.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )


def test_multi_reaction_second_without_n_needed_does_not_inherit(tmp_path):
    """In a multi-reaction file, a second reaction without an explicit n_needed
    must not inherit the first reaction's n_needed. The flattened parser keeps
    the last reaction's own value, which here defaults to -1."""
    xml = tmp_path / "multi_inherit.xml"
    _write_two_reaction_xml(xml, second_has_n_needed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    # All three pairs are flattened from both reactions.
    assert len(pairs) == 3
    # The second reaction had no n_needed, so it defaults to -1 rather than
    # carrying over the first reaction's value of 2.
    assert n_needed == -1


def test_multi_reaction_conflicting_n_needed_warns(tmp_path):
    """When two reactions declare different n_needed values, the flattened
    parser cannot represent both and emits a warning, keeping the last
    reaction's value."""
    xml = tmp_path / "multi_conflict.xml"
    _write_two_reaction_xml(xml, second_has_n_needed=True)
    with pytest.warns(UserWarning, match="differing n_needed"):
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 3
    # The last reaction declared n_needed=3.
    assert n_needed == 3


def test_single_reaction_n_needed_unchanged(tmp_path):
    """A single-reaction file with an explicit n_needed parses to that value
    with no warning, confirming the single-reaction path is unchanged."""
    xml = tmp_path / "single.xml"
    xml.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <n_needed>2</n_needed>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "      <pair><atoms>2 12</atoms><distance>4.5</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 2
    assert n_needed == 2
    # One-based to zero-based conversion is unchanged.
    assert pairs[0].rec_index == 0
    assert pairs[0].lig_index == 10
    assert pairs[0].cutoff == 5.0


def test_single_reaction_no_n_needed_defaults_minus_one(tmp_path):
    """A single-reaction file without an n_needed element parses to the
    reference default of -1, matching prior behavior."""
    xml = tmp_path / "single_default.xml"
    xml.write_text(
        "<root>\n"
        "  <reaction>\n"
        "    <criterion>\n"
        "      <pair><atoms>1 11</atoms><distance>5.0</distance></pair>\n"
        "    </criterion>\n"
        "  </reaction>\n"
        "</root>\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs, n_needed = _parse_rxns_xml_criteria(xml)
    assert len(pairs) == 1
    assert n_needed == -1
