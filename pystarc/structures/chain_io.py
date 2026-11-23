"""
PySTARC chain input and output.

This module loads and saves chain topologies (atoms, bonds, angles, and
torsions) together with the initial body-frame positions, using a JSON
format.

The JSON schema is intentionally minimal. It is a top-level object with a
"name" (a string), an "atoms" list of atom dictionaries, and optional lists
for "bonds", "angles", and "torsions". Each atom carries a radius, a charge,
a residue name, a residue identifier, and a three-component position. Bonded
interactions reference atoms by their integer index into the atoms list, and
type_idx selects their COFFDROP table (-1 for none). A chain built with the
COFFDROP tables also carries a "coffdrop" object with the four source files,
their SHA-256 and the pair table of every non-bonded pair, so a load restores
the same force field.

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
    CoplanarConstraint,
    LengthConstraint,
)
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import numpy as np
import hashlib
import json

# The bundled COFFDROP data, searched when a recorded file has moved.
COFFDROP_DATA = Path(__file__).resolve().parent.parent / "coffdrop_data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_coffdrop(block: dict, path: Path):
    """Load the COFFDROP parameters named in a chain JSON block.

    Each file is taken from its recorded path or, when that is gone, from the
    bundled data folder, and its SHA-256 must match the recorded one.
    """
    from pystarc.simulation.coffdrop_params import COFFDROPParams

    files = {}
    for key, rec in block["files"].items():
        found = None
        for cand in (Path(rec["path"]), COFFDROP_DATA / Path(rec["path"]).name):
            if cand.is_file() and _sha256(cand) == rec["sha256"]:
                found = cand
                break
        if found is None:
            raise ValueError(
                f"chain JSON {path}: no COFFDROP file with SHA-256 "
                f"{rec['sha256']} at {rec['path']} or in {COFFDROP_DATA}"
            )
        files[key] = str(found)
    return COFFDROPParams.load(**files)


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
                hydro_radius=float(atom_dict.get("hydro_radius", 0.0)),
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
                type_idx=int(bd.get("type_idx", -1)),
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
                type_idx=int(ad.get("type_idx", -1)),
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
                type_idx=int(td.get("type_idx", -1)),
            )
        )
    length_constraints = [
        LengthConstraint(
            a=ChainAtomRef(int(lc["a"])),
            b=ChainAtomRef(int(lc["b"])),
            length=float(lc["length"]),
        )
        for lc in data.get("length_constraints", [])
    ]
    coplanar_constraints = [
        CoplanarConstraint(
            a=ChainAtomRef(int(cc_item["a"])),
            b=ChainAtomRef(int(cc_item["b"])),
            c=ChainAtomRef(int(cc_item["c"])),
            d=ChainAtomRef(int(cc_item["d"])),
        )
        for cc_item in data.get("coplanar_constraints", [])
    ]
    raw_network = data.get("network")
    if raw_network is None:
        raw_network = data.get("elastic_network", [])
    network = [
        ChainBond(
            a=ChainAtomRef(int(bd["a"])),
            b=ChainAtomRef(int(bd["b"])),
            r0=float(bd["r0"]),
            k_spring=float(bd["k_spring"]),
            type_idx=int(bd.get("type_idx", -1)),
        )
        for bd in raw_network
    ]
    common = ChainCommon(
        name=name,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        length_constraints=length_constraints,
        coplanar_constraints=coplanar_constraints,
        network=network,
    )
    block = data.get("coffdrop")
    if block is not None:
        common.coffdrop_params = _load_coffdrop(block, path)
        common.pair_lookups = {
            (int(i), int(j)): int(t) for i, j, t in block.get("pairs", [])
        }
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
        entry = {
            "radius": float(atom.radius),
            "charge": float(atom.charge),
            "resname": str(atom.resname),
            "resid": int(atom.resid),
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        }
        if getattr(atom, "hydro_radius", 0.0) > 0.0:
            entry["hydro_radius"] = float(atom.hydro_radius)
        atoms_out.append(entry)
    bonds_out = [
        {
            "a": _atom_ref_to_int(bd.a),
            "b": _atom_ref_to_int(bd.b),
            "r0": float(bd.r0),
            "k_spring": float(bd.k_spring),
            "type_idx": int(bd.type_idx),
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
            "type_idx": int(ag.type_idx),
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
            "type_idx": int(tr.type_idx),
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
    if common.length_constraints:
        data["length_constraints"] = [
            {
                "a": _atom_ref_to_int(lc.a),
                "b": _atom_ref_to_int(lc.b),
                "length": float(lc.length),
            }
            for lc in common.length_constraints
        ]
    if common.coplanar_constraints:
        data["coplanar_constraints"] = [
            {
                "a": _atom_ref_to_int(cc_item.a),
                "b": _atom_ref_to_int(cc_item.b),
                "c": _atom_ref_to_int(cc_item.c),
                "d": _atom_ref_to_int(cc_item.d),
            }
            for cc_item in common.coplanar_constraints
        ]
    if common.network:
        data["network"] = [
            {
                "a": _atom_ref_to_int(bd.a),
                "b": _atom_ref_to_int(bd.b),
                "r0": float(bd.r0),
                "k_spring": float(bd.k_spring),
                "type_idx": int(bd.type_idx),
            }
            for bd in common.network
        ]
    params = common.coffdrop_params
    if params is not None:
        files = getattr(params, "files", {})
        if not files:
            raise ValueError(
                "the chain carries COFFDROP tables without their source files; "
                "load the parameters with COFFDROPParams.load"
            )
        data["coffdrop"] = {
            "files": {
                k: {"path": str(Path(v).resolve()), "sha256": _sha256(v)}
                for k, v in files.items()
            },
            "pairs": [
                [int(i), int(j), int(t)] for (i, j), t in common.pair_lookups.items()
            ],
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
    frozenset({"HIS", "HIE", "HID", "HIP", "HSE", "HSD", "HSP"}),
    frozenset({"CYS", "CYX", "CYM"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"LYS", "LYN"}),
)

_CANONICAL_RESNAMES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "HIP",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

_PROTONATION_CANONICAL = {
    "HIE": "HIS",
    "HID": "HIS",
    "HSE": "HIS",
    "HSD": "HIS",
    "HSP": "HIP",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
}

_GHOST_ATOM_NAMES = frozenset({"GHO", "PKT"})

_SKIP_RESNAMES = frozenset(
    {
        "GHO",
        "PKT",
        "HOH",
        "WAT",
        "TIP",
        "TIP3",
        "SOL",
        "NA",
        "CL",
        "K",
        "MG",
        "ZN",
        "CA",
        "FE",
        "MN",
        "CU",
        "CO",
        "NI",
        "CD",
        "HG",
        "SO4",
        "PO4",
    }
)


def _is_hydrogen(atom_name: str, element: str = "") -> bool:
    """Return True when a PDB atom name or element column identifies hydrogen."""
    elem = element.strip().upper()
    if elem in ("H", "D"):
        return True
    name = atom_name.strip().upper()
    if not name:
        return False
    if name[0] == "H":
        return True
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
        return True
    return False


def _normalize_pdb_resname(resname: str) -> str:
    """Normalize Amber terminal prefixes (NALA -> ALA, CSER -> SER) and
    protonation variants (HID/HIE/HSE/HSD -> HIS, HSP -> HIP, CYX/CYM -> CYS,
    ASH -> ASP, GLH -> GLU, LYN -> LYS).
    """
    r = resname.strip().upper()
    if len(r) == 4 and r[0] in ("N", "C"):
        tail = r[1:]
        if tail in _CANONICAL_RESNAMES or tail in _PROTONATION_CANONICAL:
            r = tail
    return _PROTONATION_CANONICAL.get(r, r)


def _resname_match_tleap(r1: str, r2: str) -> bool:
    """Match residue names while normalizing Amber terminal and protonation variants."""
    n1 = _normalize_pdb_resname(r1)
    n2 = _normalize_pdb_resname(r2)
    if n1 == n2:
        return True
    for group in _TLEAP_VARIANT_GROUPS:
        if n1 in group and n2 in group:
            return True
    return False


def _resolve_resname(resname: str, mapping: dict) -> Optional[str]:
    """Find a canonical or TLEAP variant of resname present in mapping."""
    if resname in mapping:
        return resname
    norm = _normalize_pdb_resname(resname)
    if norm in mapping:
        return norm
    for group in _TLEAP_VARIANT_GROUPS:
        if norm in group:
            for cand in group:
                if cand in mapping:
                    return cand
    return None


def _parse_pdb_chain_for_beads(
    pdb_path,
    chain_id: Optional[str] = None,
) -> List[dict]:
    """Parse heavy polymer atoms of one chain in a PDB in file order across TER segments."""
    residues: List[dict] = []
    res_index_map: Dict[Tuple[str, int, int, str], int] = {}
    chains_seen: set = set()
    seg_idx = 0
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("TER") and (len(line) == 3 or line[3].isspace()):
                seg_idx += 1
                continue
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            resname_raw = line[17:21].strip().upper()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if atom_name.upper() in _GHOST_ATOM_NAMES or resname_raw in _SKIP_RESNAMES:
                continue
            ch = line[21:22].strip() or "A"
            chains_seen.add(ch)
            if chain_id is not None and ch != chain_id:
                continue
            if _is_hydrogen(atom_name, element):
                continue
            try:
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            resname = _normalize_pdb_resname(resname_raw)
            icode = line[26:27].strip()
            res_key = (ch, seg_idx, resid, icode)
            if res_key not in res_index_map:
                res_index_map[res_key] = len(residues)
                residues.append(
                    {
                        "resname": resname,
                        "resid": resid,
                        "seg_idx": seg_idx,
                        "icode": icode,
                        "atoms": {},
                    }
                )
            r_entry = residues[res_index_map[res_key]]
            if atom_name not in r_entry["atoms"]:
                r_entry["atoms"][atom_name] = np.array([x, y, z])

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
        raise ValueError(f"fallback must be one of auto/cb/ca/strict; got {fallback!r}")

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

    # Cap beads (ACE:CN, NME:CC) carry resid=-1 because they have no PDB
    # residue of their own. Such beads cannot be matched to a PDB residue, so
    # capped chains are unsupported here. Reject them with a clear error rather
    # than letting resid=-1 leak into the residue-count check below.
    if any(bead.resid < 0 for bead in common.atoms):
        raise ValueError(
            "pdb_to_bead_positions does not support capped chains: one or more "
            "beads carry resid < 0, which marks an ACE or NME cap bead with no "
            "corresponding PDB residue. Build the chain without caps to seed "
            "positions from a PDB."
        )

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

    positions = np.zeros((common.n_atoms, 3), dtype=float)
    mapped = [False] * common.n_atoms
    missing_entries: List[Tuple[int, dict, List[str]]] = []

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
            mapped[i] = True
        else:
            missing_entries.append((i, res, atom_names))

    if missing_entries:
        # Build bond adjacency for outward extension of missing sidechain beads.
        bond_nbrs: Dict[int, List[Tuple[int, float]]] = {
            k: [] for k in range(common.n_atoms)
        }
        for bond in common.bonds:
            ia = _atom_ref_to_int(bond.a)
            ib = _atom_ref_to_int(bond.b)
            r0 = float(bond.r0)
            bond_nbrs[ia].append((ib, r0))
            bond_nbrs[ib].append((ia, r0))

        unit_axes = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        for i, res, atom_names in missing_entries:
            bead = common.atoms[i]
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

            # Locate bonded parent (preferring same residue) to extend outward by r0.
            parent_idx = None
            r0 = 2.5
            for nbr, blen in bond_nbrs[i]:
                if mapped[nbr] and common.atoms[nbr].resid == bead.resid:
                    parent_idx = nbr
                    r0 = blen
                    break
            if parent_idx is None:
                for nbr, blen in bond_nbrs[i]:
                    if mapped[nbr]:
                        parent_idx = nbr
                        r0 = blen
                        break

            anchor = positions[parent_idx] if parent_idx is not None else fb_pos
            direction = np.zeros(3, dtype=float)
            if parent_idx is not None:
                # Prefer same-residue grandparent (e.g. CA -> CB -> CG).
                gp_same = [
                    gp
                    for gp, _ in bond_nbrs[parent_idx]
                    if gp != i
                    and mapped[gp]
                    and common.atoms[gp].resid == common.atoms[parent_idx].resid
                ]
                if gp_same:
                    direction = positions[parent_idx] - positions[gp_same[0]]
                else:
                    # Parent is CA: sum outward vectors from bonded backbone neighbors.
                    gp_any = [
                        gp for gp, _ in bond_nbrs[parent_idx] if gp != i and mapped[gp]
                    ]
                    for gp in gp_any:
                        v = positions[parent_idx] - positions[gp]
                        vn = float(np.linalg.norm(v))
                        if vn > 1e-8:
                            direction += v / vn

            dnorm = float(np.linalg.norm(direction))
            if dnorm > 1e-8:
                u = direction / dnorm
            else:
                u = unit_axes[i % 3]
            positions[i] = anchor + r0 * u
            mapped[i] = True

    return positions


def prepare_chain_partner(
    pdb_path: str | Path,
    flexible_chain_id: str,
    chain_name: str = "",
    sidechains: bool = True,
    coffdrop_dir: Optional[str] = None,
    fallback: str = "auto",
) -> Tuple[ChainCommon, np.ndarray]:
    """Build a COFFDROP chain topology and centered body-frame positions for one PDB chain.

    Either protein partner of a complex PDB can be chosen through flexible_chain_id.
    The returned body positions have zero centroid and are ready for save_chain_to_json
    or direct simulator construction.
    """
    from pystarc.simulation.coffdrop_chain import chain_from_pdb

    label = chain_name or str(flexible_chain_id)
    common = chain_from_pdb(
        str(pdb_path),
        chain_id=flexible_chain_id,
        name=label,
        sidechains=sidechains,
        coffdrop_dir=coffdrop_dir,
    )
    body_positions = pdb_to_bead_positions(
        common,
        pdb_path,
        chain_id=flexible_chain_id,
        coffdrop_dir=coffdrop_dir,
        fallback=fallback,
    )
    body_positions = body_positions - body_positions.mean(axis=0)
    return common, body_positions


def interface_reaction_pairs(
    pdb_path: str | Path,
    rigid_chain_id: str,
    flexible_chain_id: str,
    rigid_pqr_path: str | Path,
    chain_common: ChainCommon,
    interface_cutoff: float = 6.0,
    reaction_distance: float = 8.0,
    coffdrop_dir: Optional[str] = None,
) -> List[Tuple[int, int, float]]:
    """Map heavy-atom interface contacts of a complex PDB to (rigid_pqr_idx, chain_bead_idx, cutoff) pairs.

    Contacts closer than interface_cutoff angstrom between rigid_chain_id and
    flexible_chain_id in pdb_path are mapped onto the atom indices of rigid_pqr_path
    and the COFFDROP bead indices of chain_common. Calling this function with the
    two chain identifiers swapped prepares reaction pairs for the opposite partner
    role.
    """
    from pystarc.structures.pqr_io import parse_pqr

    if coffdrop_dir is None:
        coffdrop_dir = str(COFFDROP_DATA)
    map_xml_path = Path(coffdrop_dir) / "map.xml"
    if not map_xml_path.exists():
        raise FileNotFoundError(f"COFFDROP map.xml not found at {map_xml_path}")
    bead_mapping = _parse_coffdrop_map_simple(str(map_xml_path))

    rigid_res = _parse_pdb_chain_for_beads(pdb_path, chain_id=rigid_chain_id)
    flex_res = _parse_pdb_chain_for_beads(pdb_path, chain_id=flexible_chain_id)
    rigid_mol = parse_pqr(str(rigid_pqr_path))

    pqr_residues: List[dict] = []
    last_resid = None
    for idx, atom in enumerate(rigid_mol.atoms):
        if atom.residue_index != last_resid:
            last_resid = atom.residue_index
            pqr_residues.append({"resname": atom.residue_name, "atoms": {}})
        aname = atom.name.strip()
        if aname not in pqr_residues[-1]["atoms"]:
            pqr_residues[-1]["atoms"][aname] = idx

    if len(pqr_residues) != len(rigid_res):
        raise ValueError(
            f"Rigid PQR {rigid_pqr_path} has {len(pqr_residues)} residues, "
            f"whereas PDB chain {rigid_chain_id!r} has {len(rigid_res)} residues."
        )
    for k, (rp, rmol) in enumerate(zip(rigid_res, pqr_residues)):
        if not _resname_match_tleap(rp["resname"], rmol["resname"]):
            raise ValueError(
                f"Residue mismatch at position {k}: PDB {rp['resname']} vs "
                f"PQR {rmol['resname']}"
            )

    unique_resids: List[int] = []
    for bead in chain_common.atoms:
        if bead.resid < 0:
            continue
        if not unique_resids or unique_resids[-1] != bead.resid:
            unique_resids.append(bead.resid)
    resid_to_seq = {r: k for k, r in enumerate(unique_resids)}
    if len(unique_resids) != len(flex_res):
        raise ValueError(
            f"Chain topology has {len(unique_resids)} residues, whereas PDB chain "
            f"{flexible_chain_id!r} has {len(flex_res)} residues."
        )

    atom_to_bead: Dict[Tuple[int, str], int] = {}
    ca_bead_of_seq: Dict[int, int] = {}
    for b_idx, bead in enumerate(chain_common.atoms):
        if bead.resid < 0:
            continue
        parts = bead.resname.split(":")
        if len(parts) != 2:
            continue
        b_res, b_name = parts
        seq_k = resid_to_seq[bead.resid]
        if b_name == "CA":
            ca_bead_of_seq[seq_k] = b_idx
        m_res = _resolve_resname(b_res, bead_mapping)
        if m_res is not None and b_name in bead_mapping[m_res]:
            for an in bead_mapping[m_res][b_name]:
                atom_to_bead[(seq_k, an)] = b_idx

    cut2 = float(interface_cutoff) ** 2
    seen_pairs = set()
    pairs_out: List[Tuple[int, int, float]] = []
    for r_seq, r_pdb in enumerate(rigid_res):
        pqr_atoms = pqr_residues[r_seq]["atoms"]
        for r_aname, r_xyz in r_pdb["atoms"].items():
            if r_aname not in pqr_atoms:
                continue
            t_idx = pqr_atoms[r_aname]
            for f_seq, f_pdb in enumerate(flex_res):
                for f_aname, f_xyz in f_pdb["atoms"].items():
                    d = r_xyz - f_xyz
                    if float(d @ d) < cut2:
                        b_idx = atom_to_bead.get(
                            (f_seq, f_aname), ca_bead_of_seq.get(f_seq)
                        )
                        if b_idx is not None and (t_idx, b_idx) not in seen_pairs:
                            seen_pairs.add((t_idx, b_idx))
                            pairs_out.append(
                                (int(t_idx), int(b_idx), float(reaction_distance))
                            )
    return pairs_out
