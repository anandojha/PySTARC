"""Reaction files are read strictly: pairs, n_needed, names, missing files and
the GHO criterion, with no silent fallback."""

from types import SimpleNamespace

import pytest

from pystarc.pipeline.geometry import (
    _parse_rxns_xml_criteria,
    _parse_rxns_xml_reaction_groups,
    auto_detect_reactions,
)
from pystarc.structures.molecules import needed_pairs


def _rxns(tmp_path, *reactions, name="rxns.xml"):
    p = tmp_path / name
    p.write_text("<top><first_state>a</first_state><reactions>" + "".join(reactions)
                 + "</reactions></top>")
    return p


def _reaction(pairs, n_needed=None, name="<name>bind</name>", states=""):
    nn = "" if n_needed is None else f"<n_needed>{n_needed}</n_needed>"
    return f"<reaction>{name}{states}<criterion>{nn}{pairs}</criterion></reaction>"


PAIR = "<pair><atoms>3 1</atoms><distance>5.0</distance></pair>"
PAIR2 = "<pair><atoms>4 2</atoms><distance>6.0</distance></pair>"


def test_atoms_and_distance_are_read_zero_based(tmp_path):
    pairs, nn = _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction(PAIR + PAIR2, 1)))
    assert [(p.rec_index, p.lig_index, p.cutoff) for p in pairs] == [(2, 0, 5.0), (3, 1, 6.0)]
    assert nn == 1


def test_atom1_atom2_form_needs_its_cutoff(tmp_path):
    ok = "<pair><atom1>5 -0.5 4.5</atom1><atom2>2</atom2></pair>"
    pairs, _ = _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction(ok)))
    assert (pairs[0].rec_index, pairs[0].lig_index, pairs[0].cutoff) == (4, 1, 4.5)
    with pytest.raises(ValueError, match="pair 1"):
        _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction("<pair><atom1>5</atom1><atom2>2</atom2></pair>")))


@pytest.mark.parametrize("pair", [
    "<pair><atoms>3</atoms><distance>5</distance></pair>",
    "<pair><atoms>3 1 2</atoms><distance>5</distance></pair>",
    "<pair><atoms>3 x</atoms><distance>5</distance></pair>",
    "<pair><atoms>3 1</atoms></pair>",
    "<pair><atoms>0 1</atoms><distance>5</distance></pair>",
    "<pair><atoms>3 1</atoms><distance>0</distance></pair>",
    "<pair><atoms>3 1</atoms><distance>-2</distance></pair>",
    "<pair><distance>5</distance></pair>",
])
def test_malformed_pairs_raise_naming_file_and_reaction(tmp_path, pair):
    with pytest.raises(ValueError, match="rxns.xml, reaction 'bind', pair 1"):
        _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction(pair)))


def test_invalid_xml_and_bad_criteria_raise(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<top><reaction>")
    with pytest.raises(ValueError, match="bad.xml is not valid XML"):
        _parse_rxns_xml_criteria(bad)
    with pytest.raises(ValueError, match="needs one <criterion>"):
        _parse_rxns_xml_criteria(_rxns(tmp_path, "<reaction><name>x</name></reaction>"))
    nested = f"<reaction><name>x</name><criterion><criterion>{PAIR}</criterion></criterion></reaction>"
    with pytest.raises(ValueError, match="without nested criteria"):
        _parse_rxns_xml_criteria(_rxns(tmp_path, nested))
    with pytest.raises(ValueError, match="has no pair"):
        _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction("")))


@pytest.mark.parametrize("nn,ok", [(None, -1), (-1, -1), (1, 1), (2, 2), (0, None), (3, None), ("1.5", None)])
def test_n_needed_is_all_or_1_to_the_pair_count(tmp_path, nn, ok):
    p = _rxns(tmp_path, _reaction(PAIR + PAIR2, nn))
    if ok is None:
        with pytest.raises(ValueError, match="n_needed"):
            _parse_rxns_xml_criteria(p)
    else:
        assert _parse_rxns_xml_criteria(p)[1] == ok


def test_needed_pairs_rule():
    assert needed_pairs(-1, 4) == 4 and needed_pairs(3, 4) == 3
    for bad in (0, 5, -2):
        with pytest.raises(ValueError, match="use -1 for all pairs or 1 to 4"):
            needed_pairs(bad, 4)


def test_names_come_from_name_then_n_then_position(tmp_path):
    p = _rxns(tmp_path, _reaction(PAIR, name="<name>a</name><n>b</n>"), _reaction(PAIR, name="<n>c</n>"),
              _reaction(PAIR, name=""))
    groups, first = _parse_rxns_xml_reaction_groups(p)
    assert [g.name for g in groups] == ["a", "c", "reaction_3"] and first == "a"


def test_state_machine_names_must_differ(tmp_path):
    p = _rxns(tmp_path, _reaction(PAIR, name="<name>a</name>"), _reaction(PAIR2, name="<name>a</name>"))
    with pytest.raises(ValueError, match="names repeat"):
        _parse_rxns_xml_reaction_groups(p)


def test_flattened_criteria_take_one_reaction_only(tmp_path):
    two = _rxns(tmp_path, _reaction(PAIR, name="<name>a</name>"), _reaction(PAIR2, 1, name="<name>b</name>"))
    pairs, nn = _parse_rxns_xml_criteria(two)
    assert len(pairs) == 2 and nn == -1
    with pytest.raises(ValueError, match="holds 2 reactions"):
        _parse_rxns_xml_criteria(two, single=True)
    assert _parse_rxns_xml_criteria(_rxns(tmp_path, _reaction(PAIR)), single=True)[1] == -1


def _geom(rec=(5,), lig=(0,)):
    return SimpleNamespace(receptor=SimpleNamespace(ghost_indices=list(rec)),
                           ligand=SimpleNamespace(ghost_indices=list(lig)))


def test_given_rxns_file_is_required_and_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="rxns_xml not found"):
        auto_detect_reactions(_geom(), "auto", str(tmp_path / "missing.xml"),
                              bd_milestone_radius=30.0, bd_milestone_radius_inner=10.0)
    empty = tmp_path / "empty.xml"
    empty.write_text("<top><reactions></reactions></top>")
    with pytest.raises(ValueError, match="holds no reaction"):
        auto_detect_reactions(_geom(), "auto", str(empty))
    stages, nn = auto_detect_reactions(_geom(), "auto", str(_rxns(tmp_path, _reaction(PAIR + PAIR2, 2))))
    assert len(stages[0]) == 2 and nn == 2


def test_gho_criterion_needs_an_inner_radius_inside_b():
    for inner in (0.0, 30.0, 35.0):
        with pytest.raises(ValueError, match="bd_milestone_radius_inner"):
            auto_detect_reactions(_geom(), "auto", "", bd_milestone_radius=30.0,
                                  bd_milestone_radius_inner=inner)
    stages, nn = auto_detect_reactions(_geom(), "auto", "", bd_milestone_radius=30.0,
                                       bd_milestone_radius_inner=12.0)
    assert stages[0][0].cutoff == 12.0 and nn == 1


def test_cli_reader_checks_n_needed(tmp_path):
    from pystarc.xml_io.simulation_io import parse_reaction_xml

    p = tmp_path / "p.xml"
    p.write_text('<reactions><reaction name="r" n_needed="3"><contact atom1="1" atom2="2" '
                 'distance="5"/></reaction></reactions>')
    with pytest.raises(ValueError, match="n_needed is 3"):
        parse_reaction_xml(p)
