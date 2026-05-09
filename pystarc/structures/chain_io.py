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
from typing import Tuple
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
