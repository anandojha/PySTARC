"""
XML input and output for PySTARC simulations.

This module reads and writes the XML files used by the reference implementation.
These include the simulation input files, the reaction and contact files, and the
chain and molecule files.
"""

from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule, ContactPair, ReactionCriteria
from pystarc.pathways.reaction_interface import ReactionInterface, PathwaySet
from typing import List, Optional, Tuple, Dict
import xml.etree.ElementTree as ET
from pathlib import Path


# Parser for the reaction XML file.
def parse_reaction_xml(path: str | Path) -> PathwaySet:
    """
    Read a reaction XML file in the reference format and return the set of reaction
    pathways it defines. Each reaction carries a name and a reaction probability, and
    lists the contact pairs whose simultaneous formation defines the reaction. A
    contact pair names two atoms (one from each molecule) and the distance cutoff in
    angstrom within which they are considered in contact.

    A reaction may also carry the number of contact pairs that must form at the
    same time for the reaction to fire (n_needed) and, for state-machine
    reactions, the labels of the state before and after the reaction. When the
    pathway set runs in state-machine mode it also carries the label of the
    initial state. These are read from the elements shown below when present.

    The file has the following structure:

        <reactions first_state="unbound">
          <reaction name="rxn1" probability="1.0" n_needed="2"
                    state_before="unbound" state_after="bound">
            <contact molecule1_index="3" molecule2_index="17" distance="5.0"/>
            ...
          </reaction>
        </reactions>
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    first_state = root.get("first_state")
    pathway_set = PathwaySet(first_state=first_state)
    for rxn_elem in root.findall("reaction"):
        name = rxn_elem.get("name", "reaction")
        prob = float(rxn_elem.get("probability", "1.0"))
        pairs: List[ContactPair] = []
        for c in rxn_elem.findall("contact"):
            i1 = int(c.get("molecule1_index", c.get("atom1", "0")))
            i2 = int(c.get("molecule2_index", c.get("atom2", "0")))
            dist = float(c.get("distance", c.get("cutoff", "5.0")))
            pairs.append(ContactPair(i1, i2, dist))
        n_needed = int(rxn_elem.get("n_needed", "-1"))
        state_before = rxn_elem.get("state_before")
        state_after = rxn_elem.get("state_after")
        criteria = ReactionCriteria(
            name=name,
            pairs=pairs,
            n_needed=n_needed,
            state_before=state_before,
            state_after=state_after,
        )
        pathway_set.add(
            ReactionInterface(
                name=name,
                criteria=criteria,
                probability=prob,
                state_before=state_before,
                state_after=state_after,
            )
        )
    return pathway_set


def write_reaction_xml(pathway_set: PathwaySet, path: str | Path) -> None:
    """Write a set of reaction pathways to a reaction XML file in the reference format."""
    root = ET.Element("reactions")
    if pathway_set.first_state is not None:
        root.set("first_state", str(pathway_set.first_state))
    for rxn in pathway_set.reactions:
        rxn_elem = ET.SubElement(
            root, "reaction", name=rxn.name, probability=str(rxn.probability)
        )
        # The number of pairs needed is written only when it differs from the
        # default of requiring every pair, so that strict all-pairs reactions
        # keep their concise form.
        if rxn.criteria.n_needed >= 0 and rxn.criteria.n_needed != len(
            rxn.criteria.pairs
        ):
            rxn_elem.set("n_needed", str(rxn.criteria.n_needed))
        state_before = rxn.state_before
        if state_before is None:
            state_before = rxn.criteria.state_before
        state_after = rxn.state_after
        if state_after is None:
            state_after = rxn.criteria.state_after
        if state_before is not None:
            rxn_elem.set("state_before", str(state_before))
        if state_after is not None:
            rxn_elem.set("state_after", str(state_after))
        for pair in rxn.criteria.pairs:
            ET.SubElement(
                rxn_elem,
                "contact",
                molecule1_index=str(pair.mol1_atom_index),
                molecule2_index=str(pair.mol2_atom_index),
                distance=str(pair.distance_cutoff),
            )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)


# Parser for the simulation input XML file.
def parse_simulation_xml(path: str | Path) -> Dict:
    """
    Read a simulation input XML file and return its settings as a dictionary. The
    settings are the number of Brownian-dynamics trajectories (n_trajectories), the
    time step Δt (dt), the maximum number of steps per trajectory (max_steps), the
    b-surface starting radius and the escape radius in angstrom (r_start and r_escape),
    the random-number seed (seed), the PQR structure files for the two molecules
    (mol1_pqr and mol2_pqr), the reaction definition file (reaction_file), and the list
    of OpenDX electrostatic grid files (dx_files).
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    def get(tag: str, default=None):
        elem = root.find(tag)
        if elem is None:
            return default
        return elem.text.strip() if elem.text else default

    def getf(tag: str, default: float = 0.0) -> float:
        v = get(tag)
        # Treat a missing tag, an empty value, and the literal string "None"
        # (which is what str(None) produces when a None default is written out
        # and read back) as the default, matching geti. Without this guard a
        # round-tripped None reaches float("None") and raises ValueError.
        if not v or v == "None":
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def geti(tag: str, default: int = 0) -> int:
        v = get(tag)
        if not v or v == "None":
            return default
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    result = {
        "n_trajectories": geti("n_trajectories", 1000),
        "dt": getf("dt", 0.2),
        "max_steps": geti("max_steps", 1_000_000),
        "r_start": getf("r_start", 100.0),
        "r_escape": getf("r_escape", 0.0),
        "seed": geti("seed", 0) or None,
        "mol1_pqr": get("molecule1_pqr", "mol1.pqr"),
        "mol2_pqr": get("molecule2_pqr", "mol2.pqr"),
        "reaction_file": get("reaction_file", "reactions.xml"),
        "dx_files": [],
    }
    for dx in root.findall("dx_file"):
        if dx.text:
            result["dx_files"].append(dx.text.strip())
    return result


def write_simulation_xml(config: Dict, path: str | Path) -> None:
    """Write a simulation configuration dictionary to a simulation input XML file."""
    root = ET.Element("simulation")
    for key, val in config.items():
        if key == "dx_files":
            for f in val:
                ET.SubElement(root, "dx_file").text = str(f)
        else:
            ET.SubElement(root, key).text = str(val)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)
