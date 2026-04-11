"""
PySTARC COFFDROP parameter file parser
=====================================
Reads the four COFFDROP data files:
1. **coffdrop.xml**       - tabulated pair, bond-angle and dihedral potentials
                            (units: kcal/mol, distances in Å, angles in degrees)
2. **mapping.xml**        - atom-to-bead mapping per residue
3. **connectivity.xml**   - bond definitions (residue pairs, bead names, orders,
                            equilibrium length)
4. **charges.xml**        - partial charges on named beads per residue
All four files are directly parsed from the XML formats provided on the
the COFFDROP data repository.

Usage
-----
    from pystarc.simulation.coffdrop_params import COFFDROPParams
    params = COFFDROPParams.load(
        ff_xml       = "coffdrop.xml",
        mapping_xml  = "mapping.xml",
        connectivity_xml = "connectivity.xml",
        charges_xml  = "charges.xml",
    )
    # Evaluate pair potential between two bead types at distance r (Å)
    V = params.pair_potential("ALA", "CA", "GLY", "CA", r=5.0)
    dVdr = params.pair_force("ALA", "CA", "GLY", "CA", r=5.0)
    # Evaluate bond-angle potential (degrees)
    V = params.angle_potential(res_triplet, atom_triplet, order_triplet, theta_deg)
    # Evaluate dihedral potential (degrees)
    V = params.dihedral_potential(res_quad, atom_quad, order_quad, phi_deg)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import math

# Unit conversion 
# coffdrop.xml energies are in kcal/mol.
_KCAL_TO_KBT = 1.688656287   # standard COFFDROP energy conversion
# Angles in the XML are in degrees; internally we keep radians for forces.
_DEG_TO_RAD = math.pi / 180.0

# Bead mapping 
@dataclass
class BeadDef:
    """One coarse-grained bead in the mapping file."""
    name:     str          # e.g. 'CA', 'CB', 'NG'
    atoms:    List[str]    # all-atom names that map to this bead
    location: str = ""     # 'begin' / 'end' / ''
    btype:    str = ""     # 'cap' / 'terminus' / ''

@dataclass
class ResidueDef:
    """Per-residue bead definitions from mapping.xml."""
    name:  str
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
            bname    = (bead_node.findtext("name") or bead_node.findtext("n") or "").strip()
            atoms_txt = (bead_node.findtext("atoms") or "").strip()
            atoms    = atoms_txt.split()
            loc      = (bead_node.findtext("location") or "").strip()
            btype    = (bead_node.findtext("type") or "").strip()
            rdef.beads.append(BeadDef(name=bname, atoms=atoms,
                                      location=loc, btype=btype))
        mapping[resname] = rdef
    return mapping

# Bond connectivity 
@dataclass
class BondDef:
    """One bond from connectivity.xml."""
    residues: Tuple[str, str]   # residue names ('XXX' = wildcard)
    atoms:    Tuple[str, str]   # bead names
    orders:   Tuple[int, int]   # sequence orders within residue
    length:   float             # equilibrium length [Å]
    index:    int

def _parse_connectivity(xml_path: str) -> List[BondDef]:
    """Parse bond connectivity XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bonds = []
    for bond_node in root.findall("bond"):
        res_txt  = bond_node.findtext("residues", "").split()
        atom_txt = bond_node.findtext("atoms",    "").split()
        ord_txt  = bond_node.findtext("orders",   "").split()
        length   = float(bond_node.findtext("length", "0"))
        idx      = int(bond_node.findtext("index", "0"))
        if len(res_txt) < 2 or len(atom_txt) < 2:
            continue
        bonds.append(BondDef(
            residues = (res_txt[0],  res_txt[1]),
            atoms    = (atom_txt[0], atom_txt[1]),
            orders   = (int(ord_txt[0]), int(ord_txt[1])) if len(ord_txt) >= 2 else (0, 0),
            length   = length,
            index    = idx,
        ))
    return bonds

# Charges 
def _parse_charges(xml_path: str) -> Dict[Tuple[str, str], float]:
    """Parse bead charges XML. Returns {(resname, beadname): charge}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    charges: Dict[Tuple[str, str], float] = {}
    for ch_node in root.findall("charge"):
        res  = (ch_node.findtext("residue") or "").strip()
        atom = (ch_node.findtext("atom") or "").strip()
        val  = float(ch_node.findtext("value") or "0")
        charges[(res, atom)] = val
    return charges

# Tabulated potential 

@dataclass
class TabulatedPotential:
    """
    A 1D tabulated potential with linear interpolation.
    """
    x_min:  float
    x_max:  float
    values: np.ndarray      # energy values [kBT], length N
    residues: Tuple         # tuple of residue type indices
    atoms:    Tuple         # tuple of atom (bead) type indices
    orders:   Tuple         # tuple of sequence order values
    index:    int

    def __post_init__(self):
        n = len(self.values)
        self._dx = (self.x_max - self.x_min) / (n - 1) if n > 1 else 1.0

    def value(self, x: float) -> float:
        """Linear interpolation"""
        n = len(self.values)
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        if i < 0:
            return float(self.values[0])
        if i >= n - 1:
            return float(self.values[-1])
        frac = t - i
        return float(self.values[i] * (1.0 - frac) + self.values[i + 1] * frac)

    def deriv(self, x: float) -> float:
        """First derivative - finite difference between adjacent grid points."""
        n = len(self.values)
        t = (x - self.x_min) / self._dx
        i = int(math.floor(t))
        if i < 0:
            i = 0
        if i >= n - 1:
            i = n - 2
        return float((self.values[i + 1] - self.values[i]) / self._dx)

# Force-field XML parser 
def _txt_to_floats(txt: str) -> np.ndarray:
    return np.array([float(v) for v in txt.split()])

def _parse_ff(xml_path: str
              ) -> Tuple[Dict, List[TabulatedPotential],
                         List[TabulatedPotential], List[TabulatedPotential]]:
    """
    Parse coffdrop.xml.
    Returns
    -------
    type_map     : {'atoms': {name: index}, 'residues': {name: index}}
    pairs        : list of TabulatedPotential (pair non-bonded)
    angles       : list of TabulatedPotential (bond angles)
    dihedrals    : list of TabulatedPotential (dihedral angles)
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
            # orders == (0, 0) means this is a non-bonded pair potential
            if orders != (0, 0):
                continue
            idx     = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt  = pot_node.findtext("atoms",    "").split()
            data    = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            pairs.append(TabulatedPotential(
                x_min    = x_min_p, x_max = x_max_p,
                values   = data,
                residues = tuple(int(v) for v in res_txt),
                atoms    = tuple(int(v) for v in at_txt),
                orders   = orders,
                index    = idx,
            ))
    # Bond-angle potentials
    angles: List[TabulatedPotential] = []
    angles_node = root.find("bond_angles")
    if angles_node is not None:
        ang_txt = angles_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        x_min_a, x_max_a = ang_vals[0], ang_vals[1]
        for pot_node in angles_node.findall("potentials/potential"):
            idx     = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt  = pot_node.findtext("atoms",    "").split()
            ord_txt = pot_node.findtext("orders",   "").split()
            data    = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            angles.append(TabulatedPotential(
                x_min    = x_min_a, x_max = x_max_a,
                values   = data,
                residues = tuple(int(v) for v in res_txt),
                atoms    = tuple(int(v) for v in at_txt),
                orders   = tuple(int(v) for v in ord_txt),
                index    = idx,
            ))
    # Dihedral potentials
    dihedrals: List[TabulatedPotential] = []
    dih_node = root.find("dihedral_angles")
    if dih_node is not None:
        ang_txt = dih_node.findtext("angle", "").strip()
        ang_vals = [float(v) for v in ang_txt.split()]
        x_min_d, x_max_d = ang_vals[0], ang_vals[1]
        for pot_node in dih_node.findall("potentials/potential"):
            idx     = int(pot_node.findtext("index", "0"))
            res_txt = pot_node.findtext("residues", "").split()
            at_txt  = pot_node.findtext("atoms",    "").split()
            ord_txt = pot_node.findtext("orders",   "").split()
            data    = _txt_to_floats(pot_node.findtext("data", "")) * _KCAL_TO_KBT
            dihedrals.append(TabulatedPotential(
                x_min    = x_min_d, x_max = x_max_d,
                values   = data,
                residues = tuple(int(v) for v in res_txt),
                atoms    = tuple(int(v) for v in at_txt),
                orders   = tuple(int(v) for v in ord_txt),
                index    = idx,
            ))
    return type_map, pairs, angles, dihedrals

# Lookup helpers 
def _match_pot(potentials: List[TabulatedPotential],
               res_indices: Tuple[int, ...],
               at_indices:  Tuple[int, ...],
               orders:      Tuple[int, ...],
               wildcard:    int = 0) -> Optional[TabulatedPotential]:
    """
    Find the best-matching potential entry.
    The standard approach uses wildcard residue index 0 (XXX) to denote
    "matches any residue". An exact residue match takes priority
    over a wildcard match.
    """
    exact   = None
    wild    = None
    for pot in potentials:
        if len(pot.atoms) != len(at_indices):
            continue
        if pot.atoms != at_indices:
            continue
        if pot.orders != orders:
            continue
        # Check residues
        res_match = all(
            pr == rr or pr == wildcard
            for pr, rr in zip(pot.residues, res_indices)
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
    All four COFFDROP parameter files loaded and indexed for fast lookup.
    Attributes
    ----------
    mapping       : {resname -> ResidueDef}
    bonds         : list of BondDef
    charges       : {(resname, beadname) -> float}
    type_map      : {'atoms': {name: idx}, 'residues': {name: idx}}
    pair_pots     : list of TabulatedPotential (non-bonded pairs)
    angle_pots    : list of TabulatedPotential (bond angles)
    dihedral_pots : list of TabulatedPotential (dihedrals)
    """

    def __init__(self,
                 mapping:   Dict[str, ResidueDef],
                 bonds:     List[BondDef],
                 charges:   Dict[Tuple[str, str], float],
                 type_map:  Dict,
                 pair_pots: List[TabulatedPotential],
                 angle_pots: List[TabulatedPotential],
                 dihedral_pots: List[TabulatedPotential]):
        self.mapping       = mapping
        self.bonds         = bonds
        self.charges       = charges
        self.type_map      = type_map
        self.pair_pots     = pair_pots
        self.angle_pots    = angle_pots
        self.dihedral_pots = dihedral_pots
        # Pre-build name -> index lookups
        self._at_idx  = type_map["atoms"]
        self._res_idx = type_map["residues"]

    @classmethod
    def load(cls,
             ff_xml:           str,
             mapping_xml:      str,
             connectivity_xml: str,
             charges_xml:      str) -> "COFFDROPParams":
        """
        Load all four COFFDROP files.
        Parameters
        ----------
        ff_xml           : path to coffdrop.xml (force-field tabulated potentials)
        mapping_xml      : path to mapping.xml  (atom-to-bead mapping)
        connectivity_xml : path to connectivity.xml (bond definitions)
        charges_xml      : path to charges.xml (bead partial charges)
        """
        mapping = _parse_mapping(mapping_xml)
        bonds   = _parse_connectivity(connectivity_xml)
        charges = _parse_charges(charges_xml)
        type_map, pair_pots, angle_pots, dihedral_pots = _parse_ff(ff_xml)
        return cls(mapping, bonds, charges, type_map,
                   pair_pots, angle_pots, dihedral_pots)
    
    # Public evaluation API 
    def _ri(self, resname: str) -> int:
        """Residue type index (0 = XXX wildcard if unknown)."""
        return self._res_idx.get(resname, 0)

    def _ai(self, beadname: str) -> int:
        """Atom (bead) type index."""
        return self._at_idx.get(beadname, -1)

    def pair_potential(self,
                       res0: str, bead0: str,
                       res1: str, bead1: str,
                       r: float,
                       orders: Tuple[int, int] = (0, 0)) -> float:
        """
        Non-bonded pair potential V(r) in kBT at separation r [Å].
        orders = (0,0) selects non-bonded pairs convention.
        """
        ri = (self._ri(res0), self._ri(res1))
        ai = (self._ai(bead0), self._ai(bead1))
        # Try both orderings (symmetric)
        pot = _match_pot(self.pair_pots, ri, ai, orders)
        if pot is None:
            ai_rev = (ai[1], ai[0])
            ri_rev = (ri[1], ri[0])
            pot = _match_pot(self.pair_pots, ri_rev, ai_rev, orders)
        return pot.value(r) if pot is not None else 0.0

    def pair_force(self,
                   res0: str, bead0: str,
                   res1: str, bead1: str,
                   r: float,
                   orders: Tuple[int, int] = (0, 0)) -> float:
        """
        Non-bonded pair force magnitude dV/dr [kBT/Å] at r.
        Positive = repulsive.
        """
        ri = (self._ri(res0), self._ri(res1))
        ai = (self._ai(bead0), self._ai(bead1))
        pot = _match_pot(self.pair_pots, ri, ai, orders)
        if pot is None:
            ai_rev = (ai[1], ai[0])
            ri_rev = (ri[1], ri[0])
            pot = _match_pot(self.pair_pots, ri_rev, ai_rev, orders)
        return pot.deriv(r) if pot is not None else 0.0

    def angle_potential(self,
                        residues: Tuple[str, ...],
                        beads:    Tuple[str, ...],
                        orders:   Tuple[int, ...],
                        theta_deg: float) -> float:
        """Bond-angle potential V(θ) in kBT, θ in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.angle_pots, ri, ai, orders)
        return pot.value(theta_deg) if pot is not None else 0.0

    def angle_force(self,
                    residues: Tuple[str, ...],
                    beads:    Tuple[str, ...],
                    orders:   Tuple[int, ...],
                    theta_deg: float) -> float:
        """Bond-angle force dV/dθ [kBT/deg]."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.angle_pots, ri, ai, orders)
        return pot.deriv(theta_deg) if pot is not None else 0.0

    def dihedral_potential(self,
                           residues: Tuple[str, ...],
                           beads:    Tuple[str, ...],
                           orders:   Tuple[int, ...],
                           phi_deg:  float) -> float:
        """Dihedral potential V(φ) in kBT, φ in degrees."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.dihedral_pots, ri, ai, orders)
        return pot.value(phi_deg) if pot is not None else 0.0

    def dihedral_force(self,
                       residues: Tuple[str, ...],
                       beads:    Tuple[str, ...],
                       orders:   Tuple[int, ...],
                       phi_deg:  float) -> float:
        """Dihedral force dV/dφ [kBT/deg]."""
        ri = tuple(self._ri(r) for r in residues)
        ai = tuple(self._ai(b) for b in beads)
        pot = _match_pot(self.dihedral_pots, ri, ai, orders)
        return pot.deriv(phi_deg) if pot is not None else 0.0

    def bead_charge(self, resname: str, beadname: str) -> float:
        """Partial charge on a bead (in elementary charges)."""
        return self.charges.get((resname, beadname), 0.0)

    def beads_for_residue(self, resname: str) -> Optional[List[BeadDef]]:
        """Return all bead definitions for a residue."""
        rdef = self.mapping.get(resname)
        return rdef.beads if rdef else None

    def bond_length(self,
                    res0: str, bead0: str, order0: int,
                    res1: str, bead1: str, order1: int) -> Optional[float]:
        """Equilibrium bond length [Å] for a given bond, or None if not found."""
        for bond in self.bonds:
            r_match = (
                (bond.residues[0] in (res0, "XXX")) and
                (bond.residues[1] in (res1, "XXX"))
            )
            a_match = (bond.atoms[0] == bead0 and bond.atoms[1] == bead1)
            o_match = (bond.orders[0] == order0 and bond.orders[1] == order1)
            if r_match and a_match and o_match:
                return bond.length
            # Reverse
            r_match2 = (
                (bond.residues[1] in (res0, "XXX")) and
                (bond.residues[0] in (res1, "XXX"))
            )
            a_match2 = (bond.atoms[1] == bead0 and bond.atoms[0] == bead1)
            o_match2 = (bond.orders[1] == order0 and bond.orders[0] == order1)
            if r_match2 and a_match2 and o_match2:
                return bond.length
        return None

    def __repr__(self) -> str:
        return (f"COFFDROPParams("
                f"{len(self.mapping)} residues, "
                f"{len(self.bonds)} bonds, "
                f"{len(self.pair_pots)} pair pots, "
                f"{len(self.angle_pots)} angle pots, "
                f"{len(self.dihedral_pots)} dihedral pots)")