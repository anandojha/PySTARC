"""Round-trip tests for reaction XML serialization of relaxed contact criteria."""

from pystarc.structures.molecules import ContactPair, ReactionCriteria
from pystarc.pathways.reaction_interface import ReactionInterface, PathwaySet
from pystarc.xml_io.simulation_io import parse_reaction_xml, write_reaction_xml


def _build_pathway_set_n_needed_2():
    pairs = [
        ContactPair(0, 10, 5.0),
        ContactPair(1, 11, 4.5),
        ContactPair(2, 12, 6.0),
    ]
    criteria = ReactionCriteria(name="rxn", pairs=pairs, n_needed=2)
    rxn = ReactionInterface(name="rxn", criteria=criteria, probability=0.5)
    return PathwaySet(reactions=[rxn])


def test_n_needed_survives_round_trip(tmp_path):
    pathway_set = _build_pathway_set_n_needed_2()
    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    parsed = parse_reaction_xml(out)

    assert len(parsed) == 1
    rxn = parsed.reactions[0]
    assert rxn.criteria.n_needed == 2
    assert rxn.criteria.n_needed != len(rxn.criteria.pairs)
    assert len(rxn.criteria.pairs) == 3
    assert rxn.probability == 0.5


def test_n_needed_emitted_in_xml(tmp_path):
    pathway_set = _build_pathway_set_n_needed_2()
    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    text = out.read_text()
    assert "n_needed" in text


def test_all_pairs_reaction_stays_concise(tmp_path):
    pairs = [ContactPair(0, 10, 5.0), ContactPair(1, 11, 5.0)]
    criteria = ReactionCriteria(name="rxn", pairs=pairs)  # default n_needed = -1
    rxn = ReactionInterface(name="rxn", criteria=criteria)
    pathway_set = PathwaySet(reactions=[rxn])

    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    text = out.read_text()
    assert "n_needed" not in text

    parsed = parse_reaction_xml(out)
    assert parsed.reactions[0].criteria.n_needed == -1


def test_state_fields_survive_round_trip(tmp_path):
    pairs = [ContactPair(0, 10, 5.0)]
    criteria = ReactionCriteria(
        name="rxn", pairs=pairs, state_before="unbound", state_after="bound"
    )
    rxn = ReactionInterface(
        name="rxn",
        criteria=criteria,
        state_before="unbound",
        state_after="bound",
    )
    pathway_set = PathwaySet(reactions=[rxn], first_state="unbound")

    out = tmp_path / "reactions.xml"
    write_reaction_xml(pathway_set, out)
    parsed = parse_reaction_xml(out)

    assert parsed.first_state == "unbound"
    parsed_rxn = parsed.reactions[0]
    assert parsed_rxn.state_before == "unbound"
    assert parsed_rxn.state_after == "bound"
    assert parsed_rxn.criteria.state_before == "unbound"
    assert parsed_rxn.criteria.state_after == "bound"
