"""
XML I/O for PySTARC simulation inputs and outputs.

Parses the reference implementation-compatible XML files:
- simulation input files
- reaction/contact files
- chain/molecule files
"""

from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule, ContactPair, ReactionCriteria
from pystarc.pathways.reaction_interface import ReactionInterface, PathwaySet
from typing import List, Optional, Tuple, Dict
import xml.etree.ElementTree as ET
from pathlib import Path


# Reaction XML parser 
def parse_reaction_xml(path: str | Path) -> PathwaySet:
    """
    Parse a the reference reaction XML file.
    Expected structure:
    <reactions>
      <reaction name="rxn1" probability="1.0">
        <contact molecule1_index="3" molecule2_index="17" distance="5.0"/>
        ...
      </reaction>
    </reactions>
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    pathway_set = PathwaySet()
    for rxn_elem in root.findall("reaction"):
        name = rxn_elem.get("name", "reaction")
        prob = float(rxn_elem.get("probability", "1.0"))
        pairs: List[ContactPair] = []
        for c in rxn_elem.findall("contact"):
            i1 = int(c.get("molecule1_index", c.get("atom1", "0")))
            i2 = int(c.get("molecule2_index", c.get("atom2", "0")))
            dist = float(c.get("distance", c.get("cutoff", "5.0")))
            pairs.append(ContactPair(i1, i2, dist))
        criteria = ReactionCriteria(name=name, pairs=pairs)
        pathway_set.add(ReactionInterface(name=name,
                                          criteria=criteria,
                                          probability=prob))
    return pathway_set


def write_reaction_xml(pathway_set: PathwaySet, path: str | Path) -> None:
    """Write a PathwaySet to the reaction XML file."""
    root = ET.Element("reactions")
    for rxn in pathway_set.reactions:
        rxn_elem = ET.SubElement(root, "reaction",
                                  name=rxn.name,
                                  probability=str(rxn.probability))
        for pair in rxn.criteria.pairs:
            ET.SubElement(rxn_elem, "contact",
                           molecule1_index=str(pair.mol1_atom_index),
                           molecule2_index=str(pair.mol2_atom_index),
                           distance=str(pair.distance_cutoff))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)

# Simulation input XML parser 
def parse_simulation_xml(path: str | Path) -> Dict:
    """
    Parse a simulation input XML file.
    Returns a dict with keys:
      n_trajectories, dt, max_steps, r_start, r_escape, seed,
      mol1_pqr, mol2_pqr, reaction_file, dx_files
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
        return float(v) if v else default

    def geti(tag: str, default: int = 0) -> int:
        v = get(tag)
        if not v or v == 'None':
            return default
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    result = {
        "n_trajectories": geti("n_trajectories",  1000),
        "dt":             getf("dt",              0.2),
        "max_steps":      geti("max_steps",       1_000_000),
        "r_start":        getf("r_start",         100.0),
        "r_escape":       getf("r_escape",        0.0),
        "seed":           geti("seed",            0) or None,
        "mol1_pqr":       get("molecule1_pqr",   "mol1.pqr"),
        "mol2_pqr":       get("molecule2_pqr",   "mol2.pqr"),
        "reaction_file":  get("reaction_file",   "reactions.xml"),
        "dx_files":       [],
    }
    for dx in root.findall("dx_file"):
        if dx.text:
            result["dx_files"].append(dx.text.strip())
    return result

def write_simulation_xml(config: Dict, path: str | Path) -> None:
    """Write a simulation configuration dict to XML."""
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