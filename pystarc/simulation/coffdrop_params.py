"""
Parser for the COFFDROP parameter files used by PySTARC.

This module reads the four XML files that together define the COFFDROP
coarse-grained force field. The file coffdrop.xml holds the tabulated pair,
bond-angle, and dihedral potentials, with energies in kcal/mol, distances in
Å, and angles in degrees. The file map.xml gives the atom-to-bead mapping for
each residue. The file connectivity.xml lists the bond definitions, namely the
residue pairs, bead names, sequence orders, and equilibrium lengths. The file
charges.xml provides the partial charges on the named beads of each residue.
All four files are parsed directly from the XML formats published in the
COFFDROP data repository.

To use this module, load the four files and then evaluate the potentials. For
example, after

    from pystarc.simulation.coffdrop_params import COFFDROPParams
    params = COFFDROPParams.load(
        ff_xml       = "coffdrop.xml",
        mapping_xml  = "map.xml",
        connectivity_xml = "connectivity.xml",
        charges_xml  = "charges.xml",
    )

one can evaluate the pair potential and its derivative between two bead types
at a separation r in Å with params.pair_potential("ALA", "CA", "GLY", "CA",
r=5.0) and params.pair_force("ALA", "CA", "GLY", "CA", r=5.0). The bond-angle
and dihedral potentials are obtained in the same way from
params.angle_potential and params.dihedral_potential, with the angle given in
degrees.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import math
from scipy.interpolate import CubicSpline, PPoly, PchipInterpolator
from pystarc.global_defs.constants import KBT_KCAL
from pystarc.global_defs.defaults import TEMPERATURE

# Unit conversion from kcal/mol in coffdrop.xml to kBT at reference temperature.
_KCAL_TO_KBT = 1.0 / KBT_KCAL


# Bead mapping
@dataclass
class BeadDef:
    """One coarse-grained bead in the mapping file."""

    name: str  # the bead name, for example 'CA', 'CB', or 'NG'
    atoms: List[str]  # the all-atom names that map onto this bead
    location: str = ""  # position within the chain: 'begin', 'end', or empty
    btype: str = ""  # bead role: 'cap', 'terminus', or empty


@dataclass
class ResidueDef:
    """Per-residue bead definitions from map.xml."""

    name: str
    beads: List[BeadDef] = field(default_factory=list)


def _parse_mapping(xml_path: str) -> Dict[str, ResidueDef]:
    """Parse the atom-to-bead mapping XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mapping: Dict[str, ResidueDef] = {}
    for res_node in root.findall("residue"):
        resname = (res_node.findtext("name") or res_node.findtext("n") or "").strip()
        rdef = ResidueDef(name=resname)
        for bead_node in res_node.findall("bead"):
            bname = (
                bead_node.findtext("name") or bead_node.findtext("n") or ""
            ).strip()
            atoms_txt = (bead_node.findtext("atoms") or "").strip()
            atoms = atoms_txt.split()
            loc = (bead_node.findtext("location") or "").strip()
            btype = (bead_node.findtext("type") or "").strip()
            rdef.beads.append(
                BeadDef(name=bname, atoms=atoms, location=loc, btype=btype)
            )
        mapping[resname] = rdef
    return mapping


# Bond connectivity
@dataclass
class BondDef:
    """One bond from connectivity.xml."""

    residues: Tuple[str, str]  # the two residue names, where 'XXX' is a wildcard
    atoms: Tuple[str, str]  # the two bead names
    orders: Tuple[int, int]  # the sequence orders within the residue
    length: float  # the equilibrium length in Å
    index: int


def _parse_connectivity(xml_path: str) -> List[BondDef]:
    """Parse bond connectivity XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bonds = []
    for bond_node in root.findall("bond"):
        res_txt = bond_node.findtext("residues", "").split()
        atom_txt = bond_node.findtext("atoms", "").split()
        ord_txt = bond_node.findtext("orders", "").split()
        length = float(bond_node.findtext("length", "0"))
        idx = int(bond_node.findtext("index", "0"))
        if len(res_txt) < 2 or len(atom_txt) < 2:
            continue
        bonds.append(
            BondDef(
                residues=(res_txt[0], res_txt[1]),
                atoms=(atom_txt[0], atom_txt[1]),
                orders=(
                    (int(ord_txt[0]), int(ord_txt[1])) if len(ord_txt) >= 2 else (0, 0)
                ),
                length=length,
                index=idx,
            )
        )
    return bonds


# Charges
def _parse_charges(xml_path: str) -> Dict[Tuple[str, str], float]:
    """Parse bead charges XML. Returns {(resname, beadname): charge}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    charges: Dict[Tuple[str, str], float] = {}
    for ch_node in root.findall("charge"):
        res = (ch_node.findtext("residue") or "").strip()
        atom = (ch_node.findtext("atom") or "").strip()
        val = float(ch_node.findtext("value") or "0")
        charges[(res, atom)] = val
    return charges


# Tabulated potential


@dataclass
class TabulatedPotential:
    """
    One-dimensional tabulated potential evaluated by spline interpolation.
    """

    x_min: float
    x_max: float
    values: np.ndarray  # N energy values in kBT
    residues: Tuple  # residue type indices
    atoms: Tuple  # atom (bead) type indices
    orders: Tuple  # sequence order values
    index: int
    kind: str = ""
    eq_length: Optional[float] = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        n = len(self.values)
        if n == 0:
            raise ValueError(
                "TabulatedPotential requires at least one energy value, but "
                f"the values array is empty (index={self.index}). An empty "
                "<data> tag in the force-field XML causes this."
            )
        self._dx = (self.x_max - self.x_min) / (n - 1) if n > 1 else 1.0
        self.left_slope: float = 0.0
        self.is_periodic: bool = self.kind == "dihedral"
        if n >= 4:
            xs = self.x_min + self._dx * np.arange(n, dtype=np.float64)
            if self.kind == "pair":
                # Monotone cubic Hermite avoids Gibbs ringing across steep cores.
                self._spline = PchipInterpolator(xs, self.values)
            elif self.kind == "dihedral":
                has_ghost = (
                    abs(float(self.values[0]) - float(self.values[-2])) < 1e-6
                    and abs(float(self.values[1]) - float(self.values[-1])) < 1e-6
                )
                if has_ghost:
                    vals_per = np.array(self.values[1:], dtype=np.float64, copy=True)
                    vals_per[-1] = vals_per[0]
                    cs = CubicSpline(xs[1:], vals_per, bc_type="periodic")
                    self._spline = PPoly(
                        np.concatenate([cs.c[:, -1:], cs.c], axis=1), xs
                    )
                elif abs(float(self.values[0]) - float(self.values[-1])) < 1e-6:
                    vals_per = np.array(self.values, dtype=np.float64, copy=True)
                    vals_per[-1] = vals_per[0]
                    self._spline = CubicSpline(xs, vals_per, bc_type="periodic")
                else:
                    self._spline = CubicSpline(xs, self.values, bc_type="natural")
            else:
                self._spline = CubicSpline(xs, self.values, bc_type="natural")
        else:
            self._spline = None

        if self.kind == "pair" and float(self.values[0]) > 100.0:
            if self._spline is not None:
                self.left_slope = min(float(self._spline(self.x_min, 1)), -1.0e4)
            elif n >= 2:
                slope0 = float(self.values[1] - self.values[0]) / self._dx
                self.left_slope = min(slope0, -1.0e4)
            else:
                self.left_slope = -1.0e4
        elif self.kind == "bond":
            if self._spline is not None:
                slope0 = float(self._spline(self.x_min, 1))
                if slope0 < 0.0:
                    self.left_slope = slope0
            elif n >= 2:
                slope0 = float(self.values[1] - self.values[0]) / self._dx
                if slope0 < 0.0:
                    self.left_slope = slope0

    def value(self, x: float) -> float:
        """Spline interpolation, clamped at table boundaries."""
        if x <= self.x_min:
            return float(self.values[0])
        if x >= self.x_max:
            return float(self.values[-1])
        if self._spline is not None:
            return float(self._spline(x))
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        frac = t - i
        return float(self.values[i] * (1.0 - frac) + self.values[i + 1] * frac)

    def deriv(self, x: float) -> float:
        """First derivative of the spline, with repulsive left core on pair tables."""
        if not math.isfinite(x):
            return 0.0
        if x <= self.x_min:
            if self.left_slope != 0.0:
                return self.left_slope
            if not (self.is_periodic and x == self.x_min):
                return 0.0
        if x >= self.x_max:
            return 0.0
        if self._spline is not None:
            return float(self._spline(x, 1))
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        if i < 0:
            i = 0
        n = len(self.values)
        if i >= n - 1:
            i = n - 2
        return float((self.values[i + 1] - self.values[i]) / self._dx)

    def deriv_array(self, xs: np.ndarray) -> np.ndarray:
        """Vectorized first derivative matching scalar deriv() elementwise."""
        xs = np.asarray(xs, dtype=np.float64)
        out = np.zeros_like(xs, dtype=np.float64)
        finite = np.isfinite(xs)
        if self.is_periodic:
            in_range = finite & (xs >= self.x_min) & (xs < self.x_max)
        else:
            in_range = finite & (xs > self.x_min) & (xs < self.x_max)
        if in_range.any():
            xs_in = xs[in_range]
            if self._spline is not None:
                out[in_range] = self._spline(xs_in, 1)
            else:
                n = len(self.values)
                t = (xs_in - self.x_min) / self._dx
                i_arr = np.floor(t).astype(np.int64)
                i_arr = np.clip(i_arr, 0, n - 2)
                out[in_range] = (self.values[i_arr + 1] - self.values[i_arr]) / self._dx
        if self.left_slope != 0.0:
            left_mask = finite & (xs <= self.x_min)
            if left_mask.any():
                out[left_mask] = self.left_slope
        return out


# Force-field XML parser
def _txt_to_floats(txt: str) -> np.ndarray:
    return np.array([float(v) for v in txt.split()])


def _parse_ff(
    xml_path: str,
    nb_attractive_scale: float = 1.0,
    temperature: float = TEMPERATURE,
    return_bonds: bool = False,
):
    """Parse coffdrop.xml into type maps and tabulated pair, angle, dihedral, and optional bond tables."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    kcal_to_kbt = _KCAL_TO_KBT * (TEMPERATURE / float(temperature))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    type_map: Dict = {"atoms": {}, "residues": {}}
    types_node = root.find("types")
    if types_node is not None:
        for atype in types_node.findall("atoms/type"):
            n = (atype.findtext("name") or atype.findtext("n") or "").strip()
            i = int(atype.findtext("index", "0"))
            type_map["atoms"][n] = i
        for rtype in types_node.findall("residues/type"):
            n = (rtype.findtext("name") or rtype.findtext("n") or "").strip()
            i = int(rtype.findtext("index", "0"))
            type_map["residues"][n] = i

    # Non-bonded pair potentials: (0, 0) general pairs followed by non-(0, 0)
    # improper-bond tables ((1, 2) adjacent pairs plus intra-residue TRP-cap
    # tables (1, 1) CN-CD and (2, 2) CD-CC). Actual bond="true" tables go into
    # bond_pots so pair_pots keeps sequential 0-based indices matching XML order.
    pairs_00: List[TabulatedPotential] = []
    pairs_non00: List[TabulatedPotential] = []
    bond_pots: List[TabulatedPotential] = []
    pairs_node = root.find("pairs")
    if pairs_node is not None:
        dist_txt = pairs_node.findtext("distance", "").strip()
        dist_vals = [float(v) for v in dist_txt.split()]
        if len(dist_vals) < 2:
            raise ValueError(
                "The <pairs> block requires a <distance> list with at least "
                f"two values (x_min and x_max), but found {len(dist_vals)}."
            )
        x_min_p, x_max_p = dist_vals[0], dist_vals[1]
        for pot_node in pairs_node.findall("potentials/potential"):
            orders_txt = pot_node.findtext("orders", "0 0").split()
            orders = tuple(int(v) for v in orders_txt)
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            is_bond = pot_node.attrib.get("bond", "").strip().lower() == "true"
            data = _txt_to_floats(pot_node.findtext("data", "")) * kcal_to_kbt
            if is_bond:
                eq_txt = pot_node.findtext("eq_length")
                eq_len = (
                    float(eq_txt) if eq_txt is not None and eq_txt.strip() else None
                )
                bond_pots.append(
                    TabulatedPotential(
                        x_min=x_min_p,
                        x_max=x_max_p,
                        values=data,
                        residues=tuple(int(v) for v in res_txt),
                        atoms=tuple(int(v) for v in at_txt),
                        orders=orders,
                        index=0,
                        kind="bond",
                        eq_length=eq_len,
                    )
                )
                continue
            if nb_attractive_scale != 1.0:
                data[data < 0.0] *= float(nb_attractive_scale)
            pot = TabulatedPotential(
                x_min=x_min_p,
                x_max=x_max_p,
                values=data,
                residues=tuple(int(v) for v in res_txt),
                atoms=tuple(int(v) for v in at_txt),
                orders=orders,
                index=0,
                kind="pair",
            )
            if orders == (0, 0):
                pairs_00.append(pot)
            else:
                pairs_non00.append(pot)
    pairs: List[TabulatedPotential] = pairs_00 + pairs_non00
    for seq_idx, pot in enumerate(pairs):
        pot.index = seq_idx
    for seq_idx, pot in enumerate(bond_pots):
        pot.index = seq_idx

    # Bond-angle potentials
    angles: List[TabulatedPotential] = []
    angles_node = root.find("bond_angles")
    if angles_node is not None:
        ang_txt = angles_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        if len(ang_vals) < 2:
            raise ValueError(
                "The <bond_angles> block requires an <angle> list with at "
                f"least two values (x_min and x_max), but found {len(ang_vals)}."
            )
        x_min_a, x_max_a = ang_vals[0], ang_vals[1]
        for pot_node in angles_node.findall("potentials/potential"):
            idx = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            ord_txt = pot_node.findtext("orders", "").split()
            data = _txt_to_floats(pot_node.findtext("data", "")) * kcal_to_kbt
            angles.append(
                TabulatedPotential(
                    x_min=x_min_a,
                    x_max=x_max_a,
                    values=data,
                    residues=tuple(int(v) for v in res_txt),
                    atoms=tuple(int(v) for v in at_txt),
                    orders=tuple(int(v) for v in ord_txt),
                    index=idx,
                    kind="angle",
                )
            )

    # Dihedral potentials
    dihedrals: List[TabulatedPotential] = []
    dih_node = root.find("dihedral_angles")
    if dih_node is not None:
        ang_txt = dih_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        if len(ang_vals) < 2:
            raise ValueError(
                "The <dihedral_angles> block requires an <angle> list with at "
                f"least two values (x_min and x_max), but found {len(ang_vals)}."
            )
        x_min_d, x_max_d = ang_vals[0], ang_vals[1]
        for pot_node in dih_node.findall("potentials/potential"):
            idx = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            ord_txt = pot_node.findtext("orders", "").split()
            data = _txt_to_floats(pot_node.findtext("data", "")) * kcal_to_kbt
            dihedrals.append(
                TabulatedPotential(
                    x_min=x_min_d,
                    x_max=x_max_d,
                    values=data,
                    residues=tuple(int(v) for v in res_txt),
                    atoms=tuple(int(v) for v in at_txt),
                    orders=tuple(int(v) for v in ord_txt),
                    index=idx,
                    kind="dihedral",
                )
            )
    if return_bonds:
        return type_map, pairs, angles, dihedrals, bond_pots
    return type_map, pairs, angles, dihedrals


# Lookup helpers
def _match_pot(
    potentials: List[TabulatedPotential],
    res_indices: Tuple[int, ...],
    at_indices: Tuple[int, ...],
    orders: Tuple[int, ...],
    wildcard: int = 0,
) -> Optional[TabulatedPotential]:
    """
    Find the best-matching potential entry.

    Following the standard convention, the wildcard residue index 0 (written XXX
    in the data files) means "matches any residue". When both an exact residue
    match and a wildcard match are available, the exact match takes priority.
    """
    exact = None
    wild = None
    for pot in potentials:
        if len(pot.atoms) != len(at_indices):
            continue
        if pot.atoms != at_indices:
            continue
        if pot.orders != orders:
            continue
        # Check that the residues match, allowing the wildcard.
        res_match = all(
            pr == rr or pr == wildcard for pr, rr in zip(pot.residues, res_indices)
        )
        if not res_match:
            continue
        if pot.residues == res_indices:
            exact = pot
            break
        else:
            wild = pot
    return exact if exact is not None else wild


# Main parameter container
class COFFDROPParams:
    """
    Holds all four COFFDROP parameter files, loaded and indexed for fast lookup.

    The attribute mapping is a dictionary from residue name to ResidueDef. The
    attribute bonds is a list of BondDef. The attribute charges maps a
    (residue name, bead name) pair to the partial charge. The attribute
    type_map maps atom and residue names to integer indices and has the form
    {'atoms': {name: idx}, 'residues': {name: idx}}. The attributes pair_pots,
    angle_pots, and dihedral_pots are lists of TabulatedPotential objects for
    the non-bonded pairs, the bond angles, and the dihedrals, respectively.
    """

    def __init__(
        self,
        mapping: Dict[str, ResidueDef],
        bonds: List[BondDef],
        charges: Dict[Tuple[str, str], float],
        type_map: Dict,
        pair_pots: List[TabulatedPotential],
        angle_pots: List[TabulatedPotential],
        dihedral_pots: List[TabulatedPotential],
        nb_attractive_scale: float = 1.0,
        temperature: float = TEMPERATURE,
        bond_pots: Optional[List[TabulatedPotential]] = None,
    ):
        self.mapping = mapping
        self.bonds = bonds
        self.charges = charges
        self.type_map = type_map
        self.pair_pots = pair_pots
        self.angle_pots = angle_pots
        self.dihedral_pots = dihedral_pots
        self.bond_pots: List[TabulatedPotential] = (
            list(bond_pots) if bond_pots is not None else []
        )
        self.nb_attractive_scale = float(nb_attractive_scale)
        self.temperature = float(temperature)
        self._at_idx = type_map["atoms"]
        self._res_idx = type_map["residues"]
        self.files: Dict[str, str] = {}
        self._pot_index_cache: Dict[str, Tuple[int, int, Dict]] = {}
        self._diff_index_cache: Dict[str, Tuple[int, int, object]] = {}

    def _indexed_bucket(
        self, kind: str, at_indices: Tuple[int, ...], orders: Tuple[int, ...]
    ) -> List[TabulatedPotential]:
        pots = getattr(self, kind + "_pots")
        cached = self._pot_index_cache.get(kind)
        if cached is None or cached[0] != id(pots) or cached[1] != len(pots):
            buckets: Dict[
                Tuple[Tuple[int, ...], Tuple[int, ...]], List[TabulatedPotential]
            ] = {}
            for pot in pots:
                buckets.setdefault((pot.atoms, pot.orders), []).append(pot)
            cached = (id(pots), len(pots), buckets)
            self._pot_index_cache[kind] = cached
        return cached[2].get((at_indices, orders), [])

    def match_pot(
        self,
        kind: str,
        res_indices: Tuple[int, ...],
        at_indices: Tuple[int, ...],
        orders: Tuple[int, ...],
        wildcard: int = 0,
    ) -> Optional[TabulatedPotential]:
        """Indexed lookup matching _match_pot semantics over pair, angle, or dihedral tables."""
        bucket = self._indexed_bucket(kind, at_indices, orders)
        if not bucket:
            return None
        return _match_pot(bucket, res_indices, at_indices, orders, wildcard=wildcard)

    def _bond_diff_buckets(
        self,
    ) -> Dict[Tuple[Tuple[int, int], int], Dict[Tuple[int, int], TabulatedPotential]]:
        pots = self.bond_pots
        cached = self._diff_index_cache.get("bond")
        if cached is None or cached[0] != id(pots) or cached[1] != len(pots):
            buckets: Dict[
                Tuple[Tuple[int, int], int], Dict[Tuple[int, int], TabulatedPotential]
            ] = {}
            for pot in pots:
                if (
                    len(pot.orders) != 2
                    or len(pot.atoms) != 2
                    or len(pot.residues) != 2
                ):
                    continue
                diff = pot.orders[1] - pot.orders[0]
                rmap = buckets.setdefault((pot.atoms, diff), {})
                rmap.setdefault(pot.residues, pot)
            cached = (id(pots), len(pots), buckets)
            self._diff_index_cache["bond"] = cached
        return cached[2]  # type: ignore[return-value]

    def _improper_pair_diff_buckets(
        self,
    ) -> Dict[Tuple[Tuple[int, int], int], Dict[Tuple[int, int], TabulatedPotential]]:
        pots = self.pair_pots
        cached = self._diff_index_cache.get("improper_pair")
        if cached is None or cached[0] != id(pots) or cached[1] != len(pots):
            buckets: Dict[
                Tuple[Tuple[int, int], int], Dict[Tuple[int, int], TabulatedPotential]
            ] = {}
            for pot in pots:
                if pot.orders == (0, 0) or len(pot.orders) != 2:
                    continue
                if len(pot.atoms) != 2 or len(pot.residues) != 2:
                    continue
                diff = pot.orders[1] - pot.orders[0]
                rmap = buckets.setdefault((pot.atoms, diff), {})
                rmap.setdefault(pot.residues, pot)
            cached = (id(pots), len(pots), buckets)
            self._diff_index_cache["improper_pair"] = cached
        return cached[2]  # type: ignore[return-value]

    def _angle_diff_index(
        self,
    ) -> Tuple[
        Dict[
            Tuple[Tuple[int, int, int], Tuple[int, int]],
            Dict[Tuple[int, int, int], TabulatedPotential],
        ],
        Tuple[Tuple[int, int], ...],
    ]:
        pots = self.angle_pots
        cached = self._diff_index_cache.get("angle")
        if cached is None or cached[0] != id(pots) or cached[1] != len(pots):
            buckets: Dict[
                Tuple[Tuple[int, int, int], Tuple[int, int]],
                Dict[Tuple[int, int, int], TabulatedPotential],
            ] = {}
            diffs_seen = set()
            for pot in pots:
                if (
                    len(pot.orders) != 3
                    or len(pot.atoms) != 3
                    or len(pot.residues) != 3
                ):
                    continue
                diffs = (pot.orders[1] - pot.orders[0], pot.orders[2] - pot.orders[0])
                diffs_seen.add(diffs)
                rmap = buckets.setdefault((pot.atoms, diffs), {})
                rmap.setdefault(pot.residues, pot)
            cached = (id(pots), len(pots), (buckets, tuple(sorted(diffs_seen))))
            self._diff_index_cache["angle"] = cached
        return cached[2]  # type: ignore[return-value]

    def _dihedral_diff_buckets(
        self,
    ) -> Dict[
        Tuple[Tuple[int, int, int, int], Tuple[int, int, int]],
        Dict[Tuple[int, int, int, int], TabulatedPotential],
    ]:
        pots = self.dihedral_pots
        cached = self._diff_index_cache.get("dihedral")
        if cached is None or cached[0] != id(pots) or cached[1] != len(pots):
            buckets: Dict[
                Tuple[Tuple[int, int, int, int], Tuple[int, int, int]],
                Dict[Tuple[int, int, int, int], TabulatedPotential],
            ] = {}
            for pot in pots:
                if (
                    len(pot.orders) != 4
                    or len(pot.atoms) != 4
                    or len(pot.residues) != 4
                ):
                    continue
                diffs = (
                    pot.orders[1] - pot.orders[0],
                    pot.orders[2] - pot.orders[0],
                    pot.orders[3] - pot.orders[0],
                )
                rmap = buckets.setdefault((pot.atoms, diffs), {})
                rmap.setdefault(pot.residues, pot)
            cached = (id(pots), len(pots), buckets)
            self._diff_index_cache["dihedral"] = cached
        return cached[2]  # type: ignore[return-value]

    def angle_diffs(self) -> Tuple[Tuple[int, int], ...]:
        """Return sorted unique relative residue difference pairs present in angle_pots."""
        _, diffs_tuple = self._angle_diff_index()
        return diffs_tuple

    def match_bond_pot(
        self,
        res0: str,
        bead0: str,
        order0: int,
        res1: str,
        bead1: str,
        order1: int,
    ) -> Optional[TabulatedPotential]:
        """Look up a bond="true" TabulatedPotential by residue diff and BD2 wildcard order."""
        if not self.bond_pots:
            return None
        r0, r1 = self._ri(res0), self._ri(res1)
        a0, a1 = self._ai(bead0), self._ai(bead1)
        if a0 < 0 or a1 < 0:
            return None
        diff = order1 - order0
        buckets = self._bond_diff_buckets()
        fwd = buckets.get(((a0, a1), diff))
        rev = buckets.get(((a1, a0), -diff))
        for pr0, pr1 in ((r0, r1), (r0, 0), (0, r1), (0, 0)):
            if fwd is not None:
                pot = fwd.get((pr0, pr1))
                if pot is not None:
                    return pot
            if rev is not None:
                pot = rev.get((pr1, pr0))
                if pot is not None:
                    return pot
        return None

    def match_improper_pair(
        self,
        res_indices: Tuple[int, int],
        at_indices: Tuple[int, int],
        diff: int,
    ) -> Optional[TabulatedPotential]:
        """Look up a directed non-(0,0) improper-bond pair table with BD2 wildcard order."""
        rmap = self._improper_pair_diff_buckets().get((at_indices, diff))
        if not rmap:
            return None
        r0, r1 = res_indices
        for pat in ((r0, r1), (r0, 0), (0, r1), (0, 0)):
            pot = rmap.get(pat)
            if pot is not None:
                return pot
        return None

    def match_angle_diff(
        self,
        res_indices: Tuple[int, int, int],
        at_indices: Tuple[int, int, int],
        diffs: Tuple[int, int],
    ) -> Optional[TabulatedPotential]:
        """Look up an angle table by relative residue diffs using BD2 wildcard priority."""
        buckets, _ = self._angle_diff_index()
        rmap = buckets.get((at_indices, diffs))
        if not rmap:
            return None
        r0, r1, r2 = res_indices
        for pat in (
            (r0, r1, r2),
            (r0, r1, 0),
            (r0, 0, r2),
            (0, r1, r2),
            (r0, 0, 0),
            (0, r1, 0),
            (0, 0, r2),
            (0, 0, 0),
        ):
            pot = rmap.get(pat)
            if pot is not None:
                return pot
        return None

    def match_dihedral_diff(
        self,
        res_indices: Tuple[int, int, int, int],
        at_indices: Tuple[int, int, int, int],
        diffs: Tuple[int, int, int],
    ) -> Optional[TabulatedPotential]:
        """Look up a dihedral table by relative residue diffs with fewest-wildcard, lowest-index tie-break."""
        rmap = self._dihedral_diff_buckets().get((at_indices, diffs))
        if not rmap:
            return None
        r0, r1, r2, r3 = res_indices
        best_key: Optional[Tuple[int, int]] = None
        best_pot: Optional[TabulatedPotential] = None
        for mask in range(16):
            m0 = (mask >> 0) & 1
            m1 = (mask >> 1) & 1
            m2 = (mask >> 2) & 1
            m3 = (mask >> 3) & 1
            pat = (
                r0 if m0 else 0,
                r1 if m1 else 0,
                r2 if m2 else 0,
                r3 if m3 else 0,
            )
            pot = rmap.get(pat)
            if pot is not None:
                cand_key = (4 - (m0 + m1 + m2 + m3), pot.index)
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_pot = pot
        return best_pot

    @classmethod
    def load(
        cls,
        ff_xml: str,
        mapping_xml: str,
        connectivity_xml: str,
        charges_xml: str,
        nb_attractive_scale: float = 1.0,
        temperature: float = TEMPERATURE,
    ) -> "COFFDROPParams":
        """Load all four COFFDROP XML parameter files."""
        mapping = _parse_mapping(mapping_xml)
        bonds = _parse_connectivity(connectivity_xml)
        charges = _parse_charges(charges_xml)
        type_map, pair_pots, angle_pots, dihedral_pots, bond_pots = _parse_ff(
            ff_xml,
            nb_attractive_scale=nb_attractive_scale,
            temperature=temperature,
            return_bonds=True,
        )
        params = cls(
            mapping,
            bonds,
            charges,
            type_map,
            pair_pots,
            angle_pots,
            dihedral_pots,
            nb_attractive_scale=nb_attractive_scale,
            temperature=temperature,
            bond_pots=bond_pots,
        )
        params.files = dict(
            ff_xml=str(ff_xml),
            mapping_xml=str(mapping_xml),
            connectivity_xml=str(connectivity_xml),
            charges_xml=str(charges_xml),
        )
        return params

    # Public evaluation API
    def _ri(self, resname: str) -> int:
        """Return residue type index, falling back to XXX wildcard 0."""
        return self._res_idx.get(resname, 0)

    def _ai(self, beadname: str) -> int:
        """Return atom (bead) type index."""
        return self._at_idx.get(beadname, -1)

    def _lookup_pair(
        self,
        res0: str,
        bead0: str,
        res1: str,
        bead1: str,
        orders: Tuple[int, int] = (0, 0),
    ) -> Optional[TabulatedPotential]:
        ri = (self._ri(res0), self._ri(res1))
        ai = (self._ai(bead0), self._ai(bead1))
        pot = self.match_pot("pair", ri, ai, orders)
        if pot is None:
            ord_rev = (orders[1], orders[0])
            pot = self.match_pot("pair", (ri[1], ri[0]), (ai[1], ai[0]), ord_rev)
        return pot

    def pair_potential(
        self,
        res0: str,
        bead0: str,
        res1: str,
        bead1: str,
        r: float,
        orders: Tuple[int, int] = (0, 0),
    ) -> float:
        """Return non-bonded pair potential V(r) in kBT at separation r in A."""
        pot = self._lookup_pair(res0, bead0, res1, bead1, orders)
        return pot.value(r) if pot is not None else 0.0

    def pair_force(
        self,
        res0: str,
        bead0: str,
        res1: str,
        bead1: str,
        r: float,
        orders: Tuple[int, int] = (0, 0),
    ) -> float:
        """Return non-bonded pair derivative dV/dr in kBT/A at separation r."""
        pot = self._lookup_pair(res0, bead0, res1, bead1, orders)
        return pot.deriv(r) if pot is not None else 0.0

    def angle_potential(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        theta_deg: float,
    ) -> float:
        """Return bond-angle potential V(theta) in kBT, with theta in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = self.match_pot("angle", ri, ai, orders)
        return pot.value(theta_deg) if pot is not None else 0.0

    def angle_force(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        theta_deg: float,
    ) -> float:
        """Return bond-angle derivative dV/dtheta in kBT per degree."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = self.match_pot("angle", ri, ai, orders)
        return pot.deriv(theta_deg) if pot is not None else 0.0

    def dihedral_potential(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        phi_deg: float,
    ) -> float:
        """Return dihedral potential V(phi) in kBT, with phi in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = self.match_pot("dihedral", ri, ai, orders)
        return pot.value(phi_deg) if pot is not None else 0.0

    def dihedral_force(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        phi_deg: float,
    ) -> float:
        """Return dihedral derivative dV/dphi in kBT per degree."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = self.match_pot("dihedral", ri, ai, orders)
        return pot.deriv(phi_deg) if pot is not None else 0.0

    def bead_charge(self, resname: str, beadname: str) -> float:
        """Return the partial charge on a bead, in elementary charges."""
        return self.charges.get((resname, beadname), 0.0)

    def beads_for_residue(self, resname: str) -> Optional[List[BeadDef]]:
        """Return all bead definitions for a residue."""
        rdef = self.mapping.get(resname)
        return rdef.beads if rdef else None

    def bond_length(
        self, res0: str, bead0: str, order0: int, res1: str, bead1: str, order1: int
    ) -> Optional[float]:
        """Return equilibrium bond length in A, preferring coffdrop.xml eq_length over connectivity.xml."""
        bpot = self.match_bond_pot(res0, bead0, order0, res1, bead1, order1)
        if bpot is not None and bpot.eq_length is not None:
            return float(bpot.eq_length)
        for bond in self.bonds:
            r_match = (bond.residues[0] in (res0, "XXX")) and (
                bond.residues[1] in (res1, "XXX")
            )
            a_match = bond.atoms[0] == bead0 and bond.atoms[1] == bead1
            o_match = bond.orders[0] == order0 and bond.orders[1] == order1
            if r_match and a_match and o_match:
                return bond.length
            # Also check the bond written in the reverse order.
            r_match2 = (bond.residues[1] in (res0, "XXX")) and (
                bond.residues[0] in (res1, "XXX")
            )
            a_match2 = bond.atoms[1] == bead0 and bond.atoms[0] == bead1
            o_match2 = bond.orders[1] == order0 and bond.orders[0] == order1
            if r_match2 and a_match2 and o_match2:
                return bond.length
        return None

    def __repr__(self) -> str:
        return (
            f"COFFDROPParams("
            f"{len(self.mapping)} residues, "
            f"{len(self.bonds)} bonds, "
            f"{len(self.pair_pots)} pair pots, "
            f"{len(self.angle_pots)} angle pots, "
            f"{len(self.dihedral_pots)} dihedral pots)"
        )
