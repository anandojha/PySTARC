"""
PySTARC chain I/O

Load and save chain topologies (atoms, bonds, angles, torsions) plus
initial body-frame positions to and from a JSON format.

The JSON schema is intentionally minimal: a top-level object with
"name" (str), "atoms" (list of atom dicts), and optional lists for
"bonds", "angles", "torsions". Each atom carries radius, charge,
resname, resid, and a 3-component position. Bonded interactions
reference atoms by integer index into the atoms list.

Positions on disk may be in any frame; this module centers them at
the origin on load (subtracts the mean) so the resulting body-frame
positions are immediately consumable by ChainBDSimulator.
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

    Parameters
    ----------
    path : path to JSON file matching the schema documented at the
           module top.

    Returns
    -------
    common : ChainCommon with the parsed atoms, bonds, angles, torsions.
    body_positions : (n_atoms, 3) array, centered at the origin so the
                     mean of all atom positions is exactly zero.
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
    """Extract integer index from a ChainAtomRef regardless of internal
    representation. Supports int subclass / __int__, dataclass with a
    .value/.idx/.index attribute, and NamedTuple fallback.
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

    The output schema mirrors what load_chain_from_json consumes: a
    top-level object with "name", "atoms" (list of {radius, charge,
    resname, resid, position}), and optional "bonds", "angles",
    "torsions" lists referencing atoms by integer index.

    Note: load_chain_from_json centers positions at the origin on load;
    save_chain_to_json writes positions verbatim. As a consequence,
    save -> load is centering-equivalent, while load -> save -> load is
    fully idempotent.

    Parameters
    ----------
    common         : ChainCommon with atoms, bonds, angles, torsions
    body_positions : (n_atoms, 3) array of atom positions
    path           : output path (parent directory created if missing)
    indent         : JSON indent (default 2; pass None for compact)

    Returns
    -------
    Path object pointing at the written file.
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


# ============================================================
# PDB -> bead position mapping
# ============================================================
#
# Helpers for deriving body-frame positions of a COFFDROP chain from
# atomic coordinates in a PDB. Encapsulates the centroid mapping
# (per COFFDROP map.xml: e.g. LYS:NG = centroid(CE, NZ)), CB->CA
# fallback for disordered sidechains, and TLEAP-renamed-residue
# variants (HIS/HIE/HID/HIP, CYS/CYX, ASP/ASH, GLU/GLH, LYS/LYN).
# Eliminates the need for end-user setup.py scripts to reimplement
# this every time chain BD is set up from a real PDB.

_TLEAP_VARIANT_GROUPS = (
    frozenset({"HIS", "HIE", "HID", "HIP"}),
    frozenset({"CYS", "CYX", "CYM"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"LYS", "LYN"}),
)


def _resname_match_tleap(r1: str, r2: str) -> bool:
    """Match residue names allowing TLEAP-renamed variants.

    Tleap renames histidine/cysteine/etc. based on protonation or
    disulfide state (HIS->HIE, CYS->CYX, ...); raw PDBs use the
    canonical name. This matcher treats variant pairs as equal.
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
    """Parse heavy atoms from one chain in a PDB.

    Returns a list of residue dicts in N-to-C order:
        [{"resname": str, "resid": int, "atoms": {atom_name: np.ndarray, ...}}, ...]
    """
    residues: List[dict] = []
    chains_seen: set = set()
    last_resid = None
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
            if resid != last_resid:
                last_resid = resid
                residues.append({
                    "resname": resname,
                    "resid": resid,
                    "atoms": {},
                })
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
    """Parse COFFDROP map.xml into {resname: {bead_name: [atom_names]}}."""
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

    Maps each chain bead to the centroid of PDB atoms specified by the
    COFFDROP map.xml convention (e.g. LYS:NG = centroid(CE, NZ),
    GLU:OG = centroid(CD, OE1, OE2), ASN:CG = centroid(CG, OD1, ND2)).
    When the expected sidechain atoms are missing in the crystal
    (disordered residues), falls back through CB then CA.

    Use this AFTER chain_from_pdb to start a chain BD trajectory in
    the bound-state configuration rather than at place_relaxed_geometry.

    Parameters
    ----------
    common : ChainCommon
        Chain topology, typically from chain_from_pdb(pdb_path, chain_id=...).
    pdb_path : str or Path
        PDB file whose heavy-atom positions seed the bead coordinates.
    chain_id : str, optional
        Chain identifier. Required if the PDB has multiple chains.
    coffdrop_dir : str, optional
        Path to the COFFDROP data directory containing map.xml. If None
        (default), uses the data bundled with the pystarc package.
    fallback : str
        Fallback strategy for missing distal sidechain atoms:
          - "auto" (default): try CB, then CA, then raise
          - "cb": only CB, else raise
          - "ca": only CA, else raise
          - "strict": never fall back; raise on any missing atom

    Returns
    -------
    np.ndarray
        Shape (common.n_atoms, 3) body-frame positions.

    Raises
    ------
    ValueError
        Chain residue count doesn't match the PDB; or resname mismatch.
    KeyError
        Bead resname or bead atom-name not in the COFFDROP map.
    RuntimeError
        Bead cannot be mapped under the chosen fallback mode.
    FileNotFoundError
        coffdrop_dir does not contain map.xml.

    Examples
    --------
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

    # Resolve coffdrop_dir default
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

    # Map chain residue indexing -> 0..N-1 sequence position
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

        # Fallback hierarchy for missing distal atoms
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
