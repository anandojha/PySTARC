"""
PySTARC chain input and output.

This module loads and saves chain topologies (atoms, bonds, angles, and
torsions) together with the initial body-frame positions, using a JSON
format.

The JSON schema is intentionally minimal. It is a top-level object with a
"name" (a string), an "atoms" list of atom dictionaries, and optional lists
for "bonds", "angles", and "torsions". Each atom carries a radius, a charge,
a residue name, a residue identifier, and a three-component position. Bonded
interactions reference atoms by their integer index into the atoms list.

Positions on disk may be expressed in any frame. This module centers them at
the origin on load by subtracting the mean, so that the resulting body-frame
positions can be consumed directly by ChainBDSimulator.
"""

from __future__ import annotations
from pystarc.simulation.coffdrop_chain import (
    ChainAngle,
    ChainAtom,
    ChainAtomRef,
    ChainBond,
    ChainCommon,
    ChainTorsion,
)
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import numpy as np
import json


def load_chain_from_json(path: str | Path) -> Tuple[ChainCommon, np.ndarray]:
    """Load a chain topology and initial positions from a JSON file.

    The path argument is the path to a JSON file matching the schema
    documented at the top of this module.

    This function returns two objects. The first is a ChainCommon holding
    the parsed atoms, bonds, angles, and torsions. The second is the
    body-frame positions, an array of shape (n_atoms, 3) centered at the
    origin so that the mean of all atom positions is exactly zero.
    """
    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    name = data.get("name", "")
    raw_atoms = data.get("atoms", [])
    if not raw_atoms:
        raise ValueError(f"chain JSON at {path} has no atoms")
    atoms = []
    positions = []
    for i, atom_dict in enumerate(raw_atoms):
        atoms.append(
            ChainAtom(
                radius=float(atom_dict["radius"]),
                charge=float(atom_dict.get("charge", 0.0)),
                resname=atom_dict.get("resname", ""),
                resid=int(atom_dict.get("resid", i)),
            )
        )
        pos = atom_dict["position"]
        if len(pos) != 3:
            raise ValueError(
                f"atom {i} in {path} has position {pos}; expected 3 values"
            )
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    body_positions = np.array(positions, dtype=float)
    body_positions -= body_positions.mean(axis=0)
    bonds = []
    for bd in data.get("bonds", []):
        bonds.append(
            ChainBond(
                a=ChainAtomRef(int(bd["a"])),
                b=ChainAtomRef(int(bd["b"])),
                r0=float(bd["r0"]),
                k_spring=float(bd["k_spring"]),
            )
        )
    angles = []
    for ad in data.get("angles", []):
        angles.append(
            ChainAngle(
                a=ChainAtomRef(int(ad["a"])),
                b=ChainAtomRef(int(ad["b"])),
                c=ChainAtomRef(int(ad["c"])),
                theta0=float(ad["theta0"]),
                k_angle=float(ad["k_angle"]),
            )
        )
    torsions = []
    for td in data.get("torsions", []):
        torsions.append(
            ChainTorsion(
                a=ChainAtomRef(int(td["a"])),
                b=ChainAtomRef(int(td["b"])),
                c=ChainAtomRef(int(td["c"])),
                d=ChainAtomRef(int(td["d"])),
                phi0=float(td["phi0"]),
                k_tor=float(td["k_tor"]),
                n=int(td["n"]),
            )
        )
    common = ChainCommon(
        name=name,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
    )
    return common, body_positions


def _atom_ref_to_int(ref) -> int:
    """Extract the integer atom index from a ChainAtomRef regardless of how it
    is represented internally. This handles a ChainAtomRef that is an int
    subclass or otherwise supports __int__, one that is a dataclass with a
    value, idx, index, or atom_idx attribute, and as a final fallback one that
    is a NamedTuple whose first field holds the index.
    """
    try:
        return int(ref)
    except (TypeError, ValueError):
        pass
    for attr in ("value", "idx", "index", "atom_idx"):
        if hasattr(ref, attr):
            return int(getattr(ref, attr))
    if hasattr(ref, "_fields"):
        return int(ref[0])
    raise TypeError(f"Cannot extract int from ChainAtomRef: {ref!r}")


def save_chain_to_json(
    common: ChainCommon,
    body_positions: np.ndarray,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """Save a chain topology and body-frame positions to a JSON file.

    The output schema mirrors what load_chain_from_json consumes. It is a
    top-level object with a "name", an "atoms" list whose entries hold a
    radius, charge, resname, resid, and position, and optional "bonds",
    "angles", and "torsions" lists that reference atoms by integer index.

    Note that load_chain_from_json centers positions at the origin on load,
    whereas save_chain_to_json writes positions verbatim. As a consequence, a
    save followed by a load is equivalent up to centering, while a load
    followed by a save followed by a load is fully idempotent.

    The common argument is a ChainCommon holding the atoms, bonds, angles, and
    torsions. The body_positions argument is an array of shape (n_atoms, 3)
    giving the atom positions. The path argument is the output path, and its
    parent directory is created if it does not already exist. The indent
    argument sets the JSON indentation (the default is 2, and passing None
    produces compact output).

    The function returns a Path object pointing at the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body_positions = np.asarray(body_positions, dtype=float)
    if body_positions.ndim != 2 or body_positions.shape[1] != 3:
        raise ValueError(
            f"body_positions must have shape (n_atoms, 3); got {body_positions.shape}"
        )
    if body_positions.shape[0] != len(common.atoms):
        raise ValueError(
            f"body_positions has {body_positions.shape[0]} rows but chain has "
            f"{len(common.atoms)} atoms"
        )
    atoms_out = []
    for atom, pos in zip(common.atoms, body_positions):
        atoms_out.append(
            {
                "radius": float(atom.radius),
                "charge": float(atom.charge),
                "resname": str(atom.resname),
                "resid": int(atom.resid),
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            }
        )
    bonds_out = [
        {
            "a": _atom_ref_to_int(bd.a),
            "b": _atom_ref_to_int(bd.b),
            "r0": float(bd.r0),
            "k_spring": float(bd.k_spring),
        }
        for bd in common.bonds
    ]
    angles_out = [
        {
            "a": _atom_ref_to_int(ag.a),
            "b": _atom_ref_to_int(ag.b),
            "c": _atom_ref_to_int(ag.c),
            "theta0": float(ag.theta0),
            "k_angle": float(ag.k_angle),
        }
        for ag in common.angles
    ]
    torsions_out = [
        {
            "a": _atom_ref_to_int(tr.a),
            "b": _atom_ref_to_int(tr.b),
            "c": _atom_ref_to_int(tr.c),
            "d": _atom_ref_to_int(tr.d),
            "phi0": float(tr.phi0),
            "k_tor": float(tr.k_tor),
            "n": int(tr.n),
        }
        for tr in common.torsions
    ]
    data = {
        "name": str(common.name),
        "atoms": atoms_out,
        "bonds": bonds_out,
        "angles": angles_out,
        "torsions": torsions_out,
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=indent)
    return path


# Mapping from PDB atoms to COFFDROP beads.
#
# These helpers derive the body-frame positions of a COFFDROP chain from the
# atomic coordinates in a PDB. They encapsulate the centroid mapping defined in
# the COFFDROP map.xml (for example, the LYS:NG bead is the centroid of CE and
# NZ), the fallback from CB to CA for disordered sidechains, and the
# TLEAP-renamed residue variants (HIS/HIE/HID/HIP, CYS/CYX, ASP/ASH, GLU/GLH,
# and LYS/LYN). Collecting this logic here means end-user setup.py scripts no
# longer have to reimplement it every time chain BD is set up from a real PDB.

_TLEAP_VARIANT_GROUPS = (
    frozenset({"HIS", "HIE", "HID", "HIP"}),
    frozenset({"CYS", "CYX", "CYM"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"LYS", "LYN"}),
)


def _resname_match_tleap(r1: str, r2: str) -> bool:
    """Match residue names while allowing TLEAP-renamed variants.

    Tleap renames residues such as histidine and cysteine according to their
    protonation or disulfide state, so that HIS becomes HIE and CYS becomes CYX,
    and so on. Raw PDBs instead use the canonical name. This matcher treats such
    variant pairs as equal.
    """
    if r1 == r2:
        return True
    for group in _TLEAP_VARIANT_GROUPS:
        if r1 in group and r2 in group:
            return True
    return False


def _resolve_resname(resname: str, mapping: dict) -> Optional[str]:
    """Find a TLEAP variant of resname that exists as a key in mapping."""
    if resname in mapping:
        return resname
    for group in _TLEAP_VARIANT_GROUPS:
        if resname in group:
            for cand in group:
                if cand in mapping:
                    return cand
    return None


def _parse_pdb_chain_for_beads(
    pdb_path,
    chain_id: Optional[str] = None,
) -> List[dict]:
    """Parse the heavy atoms of one chain in a PDB.

    The function returns a list of residue dictionaries in N-to-C order. Each
    dictionary has a "resname" string, a "resid" integer, and an "atoms" entry
    mapping each atom name to its coordinate array.
    """
    residues: List[dict] = []
    chains_seen: set = set()
    last_key = None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            ch = line[21:22].strip() or "A"
            chains_seen.add(ch)
            if chain_id is not None and ch != chain_id:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            try:
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            resname = line[17:20].strip()
            icode = line[26:27].strip()
            res_key = (ch, resid, icode)
            if res_key != last_key:
                last_key = res_key
                residues.append({
                    "resname": resname,
                    "resid": resid,
                    "atoms": {},
                })
            # For a repeated atom name within one residue (alternate
            # conformers), keep the first one seen, which is the primary
            # conformer (altLoc A) in a conventionally ordered PDB.
            if atom_name not in residues[-1]["atoms"]:
                residues[-1]["atoms"][atom_name] = np.array([x, y, z])

    if chain_id is None and len(chains_seen) > 1:
        raise ValueError(
            f"PDB has multiple chains {sorted(chains_seen)}; specify chain_id=..."
        )
    if not residues:
        raise ValueError(
            f"No ATOM records found for chain_id={chain_id!r} in {pdb_path}"
        )
    return residues


def _parse_coffdrop_map_simple(map_xml_path) -> Dict[str, Dict[str, List[str]]]:
    """Parse the COFFDROP map.xml into a nested dictionary that maps each
    residue name to a dictionary mapping each bead name to its list of atom
    names.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(map_xml_path))
    root = tree.getroot()
    mapping: Dict[str, Dict[str, List[str]]] = {}
    for res_elem in root.findall("residue"):
        name_elem = res_elem.find("name")
        if name_elem is None or name_elem.text is None:
            continue
        resname = name_elem.text.strip()
        beads: Dict[str, List[str]] = {}
        for bead_elem in res_elem.findall("bead"):
            bead_name_elem = bead_elem.find("name")
            atoms_elem = bead_elem.find("atoms")
            if bead_name_elem is None or atoms_elem is None:
                continue
            if bead_name_elem.text is None or atoms_elem.text is None:
                continue
            bead_name = bead_name_elem.text.strip()
            beads[bead_name] = atoms_elem.text.strip().split()
        mapping[resname] = beads
    return mapping


def pdb_to_bead_positions(
    common: ChainCommon,
    pdb_path,
    chain_id: Optional[str] = None,
    coffdrop_dir: Optional[str] = None,
    fallback: str = "auto",
) -> np.ndarray:
    """Derive body-frame positions for a COFFDROP chain from PDB coordinates.

    Each chain bead is mapped to the centroid of the PDB atoms named by the
    COFFDROP map.xml convention. For example, the LYS:NG bead is the centroid
    of CE and NZ, the GLU:OG bead is the centroid of CD, OE1, and OE2, and the
    ASN:CG bead is the centroid of CG, OD1, and ND2. When the expected
    sidechain atoms are missing in the crystal structure, as happens for
    disordered residues, the mapping falls back first to CB and then to CA.

    Call this after chain_from_pdb to start a chain BD trajectory in the
    bound-state configuration rather than at place_relaxed_geometry.

    The common argument is the ChainCommon topology, typically produced by
    chain_from_pdb(pdb_path, chain_id=...). The pdb_path argument is the PDB
    file whose heavy-atom positions seed the bead coordinates. The chain_id
    argument is the chain identifier, which is required when the PDB contains
    more than one chain. The coffdrop_dir argument is the path to the COFFDROP
    data directory containing map.xml; when it is None, which is the default,
    the data bundled with the pystarc package is used.

    The fallback argument chooses the strategy for missing distal sidechain
    atoms. The default "auto" tries CB, then CA, then raises. The value "cb"
    uses only CB and otherwise raises. The value "ca" uses only CA and
    otherwise raises. The value "strict" never falls back and raises on any
    missing atom.

    The function returns an array of shape (common.n_atoms, 3) holding the
    body-frame positions.

    A ValueError is raised when the chain residue count does not match the PDB,
    or when a residue name does not match. A KeyError is raised when a bead
    residue name or a bead atom name is absent from the COFFDROP map. A
    RuntimeError is raised when a bead cannot be mapped under the chosen
    fallback mode. A FileNotFoundError is raised when coffdrop_dir does not
    contain map.xml.

    Example usage is as follows.

    >>> from pystarc.simulation.coffdrop_chain import chain_from_pdb
    >>> from pystarc.structures.chain_io import (
    ...     pdb_to_bead_positions, save_chain_to_json,
    ... )
    >>> chain = chain_from_pdb("complex.pdb", chain_id="D")
    >>> positions = pdb_to_bead_positions(chain, "complex.pdb", chain_id="D")
    >>> save_chain_to_json(chain, positions, "chain.json")
    """
    if fallback not in {"auto", "cb", "ca", "strict"}:
        raise ValueError(
            f"fallback must be one of auto/cb/ca/strict; got {fallback!r}"
        )

    # Fall back to the bundled COFFDROP data directory when none is given.
    if coffdrop_dir is None:
        coffdrop_dir = str(Path(__file__).parent.parent / "coffdrop_data")
    map_xml_path = Path(coffdrop_dir) / "map.xml"
    if not map_xml_path.exists():
        raise FileNotFoundError(
            f"COFFDROP map.xml not found at {map_xml_path}. "
            "Check coffdrop_dir or pystarc installation."
        )

    residues = _parse_pdb_chain_for_beads(pdb_path, chain_id=chain_id)
    bead_mapping = _parse_coffdrop_map_simple(str(map_xml_path))

    # Map the chain residue identifiers to sequence positions 0 to N-1.
    unique_resids: List[int] = []
    for bead in common.atoms:
        if not unique_resids or unique_resids[-1] != bead.resid:
            unique_resids.append(bead.resid)
    resid_to_seq_pos = {r: k for k, r in enumerate(unique_resids)}

    if len(unique_resids) != len(residues):
        raise ValueError(
            f"Chain has {len(unique_resids)} unique residues but PDB chain "
            f"{chain_id!r} has {len(residues)}. Wrong chain_id or inconsistent inputs."
        )

    positions = np.zeros((common.n_atoms, 3))
    for i, bead in enumerate(common.atoms):
        parts = bead.resname.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Bead {i} resname {bead.resname!r} not in 'RES:BEAD' format"
            )
        bead_resname, bead_atom_name = parts
        seq_pos = resid_to_seq_pos[bead.resid]
        res = residues[seq_pos]

        if not _resname_match_tleap(res["resname"], bead_resname):
            raise ValueError(
                f"Bead {i} resname {bead_resname} != PDB residue {seq_pos} "
                f"(resid {res['resid']}) resname {res['resname']}"
            )

        map_resname = _resolve_resname(bead_resname, bead_mapping)
        if map_resname is None:
            raise KeyError(f"Residue {bead_resname} not in COFFDROP map.xml")
        if bead_atom_name not in bead_mapping[map_resname]:
            raise KeyError(
                f"Bead {bead_atom_name} not defined for {map_resname} in map.xml"
            )

        atom_names = bead_mapping[map_resname][bead_atom_name]
        coords = [res["atoms"][an] for an in atom_names if an in res["atoms"]]

        if coords:
            positions[i] = np.mean(coords, axis=0)
            continue

        # Apply the fallback hierarchy when the distal atoms are missing.
        if fallback == "strict":
            raise RuntimeError(
                f"Bead {i} ({bead.resname}) at PDB residue "
                f"{res['resname']}{res['resid']}: none of {atom_names} found; "
                f"fallback=strict so cannot recover"
            )
        fallback_chain = []
        if fallback in ("auto", "cb"):
            fallback_chain.append("CB")
        if fallback in ("auto", "ca"):
            fallback_chain.append("CA")
        fb_pos = None
        for fb_atom in fallback_chain:
            if fb_atom in res["atoms"]:
                fb_pos = res["atoms"][fb_atom]
                break
        if fb_pos is None:
            raise RuntimeError(
                f"Bead {i} ({bead.resname}) at PDB residue "
                f"{res['resname']}{res['resid']}: none of {atom_names} found "
                f"and no fallback ({fallback_chain}) available"
            )
        positions[i] = fb_pos

    return positions
