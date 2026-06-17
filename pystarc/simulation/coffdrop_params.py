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
from scipy.interpolate import CubicSpline

# Unit conversions. The energies in coffdrop.xml are given in kcal/mol, so this
# is the standard COFFDROP factor that converts them to units of kBT.
_KCAL_TO_KBT = 1.688656287
# Angles in the XML are in degrees. Internally we keep radians for forces.
_DEG_TO_RAD = math.pi / 180.0


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
    A one-dimensional tabulated potential evaluated by interpolation.
    """

    x_min: float
    x_max: float
    values: np.ndarray  # the N energy values in units of kBT
    residues: Tuple  # the residue type indices
    atoms: Tuple  # the atom (bead) type indices
    orders: Tuple  # the sequence order values
    index: int

    def __post_init__(self):
        n = len(self.values)
        self._dx = (self.x_max - self.x_min) / (n - 1) if n > 1 else 1.0
        # Build the cubic spline once when the object is constructed, so that
        # later calls to value() and deriv() simply evaluate it. We use natural
        # boundary conditions (the second derivative vanishes at the endpoints)
        # to match the Even_Spline semantics of BD2. A spline can only be built
        # when there are at least 4 points, so for shorter tables we fall back
        # to linear interpolation.
        if n >= 4:
            xs = self.x_min + self._dx * np.arange(n)
            self._spline = CubicSpline(xs, self.values, bc_type="natural")
        else:
            self._spline = None

    def value(self, x: float) -> float:
        """Cubic-spline interpolation, clamped at the table boundaries."""
        if x <= self.x_min:
            return float(self.values[0])
        if x >= self.x_max:
            return float(self.values[-1])
        if self._spline is not None:
            return float(self._spline(x))
        # Fall back to linear interpolation for short tables.
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        frac = t - i
        return float(self.values[i] * (1.0 - frac) + self.values[i + 1] * frac)

    def deriv(self, x: float) -> float:
        """First derivative of the spline, zero outside the table."""
        if x <= self.x_min or x >= self.x_max:
            return 0.0
        if self._spline is not None:
            return float(self._spline(x, 1))  # the second argument 1 selects the first derivative
        # Fall back to linear differences for short tables.
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        if i < 0:
            i = 0
        n = len(self.values)
        if i >= n - 1:
            i = n - 2
        return float((self.values[i + 1] - self.values[i]) / self._dx)

    def deriv_array(self, xs: np.ndarray) -> np.ndarray:
        """Vectorized first derivative that returns an array of derivatives.

        This has the same meaning as deriv() applied to each element, returning
        zero outside the table range, but it takes advantage of the array
        support in CubicSpline and so is much faster than a Python loop over
        scalar deriv() calls. The input xs is an array of x values of shape
        (N,), and the result is an array of shape (N,) giving the first
        derivative at each x, which is zero outside [x_min, x_max].
        """
        xs = np.asarray(xs)
        out = np.zeros_like(xs, dtype=np.float64)
        in_range = (xs > self.x_min) & (xs < self.x_max)
        if not in_range.any():
            return out
        if self._spline is not None:
            # CubicSpline accepts array input directly.
            out[in_range] = self._spline(xs[in_range], 1)
            return out
        # Fall back to linear differences, with the index arithmetic vectorized.
        n = len(self.values)
        t = (xs - self.x_min) / self._dx
        i_arr = np.floor(t).astype(np.int64)
        i_arr = np.clip(i_arr, 0, n - 2)
        diffs = (self.values[i_arr + 1] - self.values[i_arr]) / self._dx
        out[in_range] = diffs[in_range]
        return out


# Force-field XML parser
def _txt_to_floats(txt: str) -> np.ndarray:
    return np.array([float(v) for v in txt.split()])


def _parse_ff(
    xml_path: str,
) -> Tuple[
    Dict, List[TabulatedPotential], List[TabulatedPotential], List[TabulatedPotential]
]:
    """
    Parse coffdrop.xml.

    The function returns four objects. The first, type_map, maps atom and
    residue names to integer indices and has the form {'atoms': {name: index},
    'residues': {name: index}}. The remaining three are lists of
    TabulatedPotential objects holding, respectively, the non-bonded pair
    potentials, the bond-angle potentials, and the dihedral-angle potentials.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Type indices
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
    # Pair potentials
    pairs: List[TabulatedPotential] = []
    pairs_node = root.find("pairs")
    if pairs_node is not None:
        dist_txt = pairs_node.findtext("distance", "").strip()
        dist_vals = [float(v) for v in dist_txt.split()]
        x_min_p, x_max_p = dist_vals[0], dist_vals[1]
        for pot_node in pairs_node.findall("potentials/potential"):
            orders_txt = pot_node.findtext("orders", "0 0").split()
            orders = tuple(int(v) for v in orders_txt)
            # Orders of (0, 0) identify a non-bonded pair potential.
            if orders != (0, 0):
                continue
            idx = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            data = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            pairs.append(
                TabulatedPotential(
                    x_min=x_min_p,
                    x_max=x_max_p,
                    values=data,
                    residues=tuple(int(v) for v in res_txt),
                    atoms=tuple(int(v) for v in at_txt),
                    orders=orders,
                    index=idx,
                )
            )
    # Bond-angle potentials
    angles: List[TabulatedPotential] = []
    angles_node = root.find("bond_angles")
    if angles_node is not None:
        ang_txt = angles_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        x_min_a, x_max_a = ang_vals[0], ang_vals[1]
        for pot_node in angles_node.findall("potentials/potential"):
            idx = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            ord_txt = pot_node.findtext("orders", "").split()
            data = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            angles.append(
                TabulatedPotential(
                    x_min=x_min_a,
                    x_max=x_max_a,
                    values=data,
                    residues=tuple(int(v) for v in res_txt),
                    atoms=tuple(int(v) for v in at_txt),
                    orders=tuple(int(v) for v in ord_txt),
                    index=idx,
                )
            )
    # Dihedral potentials
    dihedrals: List[TabulatedPotential] = []
    dih_node = root.find("dihedral_angles")
    if dih_node is not None:
        ang_txt = dih_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        x_min_d, x_max_d = ang_vals[0], ang_vals[1]
        for pot_node in dih_node.findall("potentials/potential"):
            idx = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt = pot_node.findtext("atoms", "").split()
            ord_txt = pot_node.findtext("orders", "").split()
            data = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            dihedrals.append(
                TabulatedPotential(
                    x_min=x_min_d,
                    x_max=x_max_d,
                    values=data,
                    residues=tuple(int(v) for v in res_txt),
                    atoms=tuple(int(v) for v in at_txt),
                    orders=tuple(int(v) for v in ord_txt),
                    index=idx,
                )
            )
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
    ):
        self.mapping = mapping
        self.bonds = bonds
        self.charges = charges
        self.type_map = type_map
        self.pair_pots = pair_pots
        self.angle_pots = angle_pots
        self.dihedral_pots = dihedral_pots
        # Pre-build the name to index lookups.
        self._at_idx = type_map["atoms"]
        self._res_idx = type_map["residues"]

    @classmethod
    def load(
        cls, ff_xml: str, mapping_xml: str, connectivity_xml: str, charges_xml: str
    ) -> "COFFDROPParams":
        """
        Load all four COFFDROP files.

        The argument ff_xml is the path to coffdrop.xml, which holds the
        tabulated force-field potentials. The argument mapping_xml is the path
        to map.xml, which gives the atom-to-bead mapping. The argument
        connectivity_xml is the path to connectivity.xml, which holds the bond
        definitions. The argument charges_xml is the path to charges.xml, which
        gives the bead partial charges.
        """
        mapping = _parse_mapping(mapping_xml)
        bonds = _parse_connectivity(connectivity_xml)
        charges = _parse_charges(charges_xml)
        type_map, pair_pots, angle_pots, dihedral_pots = _parse_ff(ff_xml)
        return cls(
            mapping, bonds, charges, type_map, pair_pots, angle_pots, dihedral_pots
        )

    # Public evaluation API
    def _ri(self, resname: str) -> int:
        """Return the residue type index, using the XXX wildcard 0 if unknown."""
        return self._res_idx.get(resname, 0)

    def _ai(self, beadname: str) -> int:
        """Return the atom (bead) type index."""
        return self._at_idx.get(beadname, -1)

    def pair_potential(
        self,
        res0: str,
        bead0: str,
        res1: str,
        bead1: str,
        r: float,
        orders: Tuple[int, int] = (0, 0),
    ) -> float:
        """
        Return the non-bonded pair potential V(r) in kBT at separation r in Å.

        The default orders of (0, 0) select the non-bonded pair convention.
        """
        ri = (self._ri(res0), self._ri(res1))
        ai = (self._ai(bead0), self._ai(bead1))
        # The interaction is symmetric, so try both orderings of the pair.
        pot = _match_pot(self.pair_pots, ri, ai, orders)
        if pot is None:
            ai_rev = (ai[1], ai[0])
            ri_rev = (ri[1], ri[0])
            pot = _match_pot(self.pair_pots, ri_rev, ai_rev, orders)
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
        """
        Return the non-bonded pair force dV/dr in kBT/Å at separation r.

        A positive value corresponds to a repulsive force.
        """
        ri = (self._ri(res0), self._ri(res1))
        ai = (self._ai(bead0), self._ai(bead1))
        pot = _match_pot(self.pair_pots, ri, ai, orders)
        if pot is None:
            ai_rev = (ai[1], ai[0])
            ri_rev = (ri[1], ri[0])
            pot = _match_pot(self.pair_pots, ri_rev, ai_rev, orders)
        return pot.deriv(r) if pot is not None else 0.0

    def angle_potential(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        theta_deg: float,
    ) -> float:
        """Return the bond-angle potential V(θ) in kBT, with θ in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.angle_pots, ri, ai, orders)
        return pot.value(theta_deg) if pot is not None else 0.0

    def angle_force(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        theta_deg: float,
    ) -> float:
        """Return the bond-angle force dV/dθ in kBT per degree."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.angle_pots, ri, ai, orders)
        return pot.deriv(theta_deg) if pot is not None else 0.0

    def dihedral_potential(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        phi_deg: float,
    ) -> float:
        """Return the dihedral potential V(φ) in kBT, with φ in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.dihedral_pots, ri, ai, orders)
        return pot.value(phi_deg) if pot is not None else 0.0

    def dihedral_force(
        self,
        residues: Tuple[str, ...],
        beads: Tuple[str, ...],
        orders: Tuple[int, ...],
        phi_deg: float,
    ) -> float:
        """Return the dihedral force dV/dφ in kBT per degree."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.dihedral_pots, ri, ai, orders)
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
        """Return the equilibrium bond length in Å for a bond, or None if absent."""
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
