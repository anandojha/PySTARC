"""
Core molecular structure types for PySTARC.

This module defines the basic data structures used throughout the package: a
single point-charge atom, a rigid molecule built from atoms, an axis-aligned
bounding box, and the contact-based reaction criteria that decide when two
molecules have associated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math


# A single atom.
@dataclass
class Atom:
    """A single point-charge atom with PQR data."""

    index: int = 0
    name: str = ""
    residue_name: str = ""
    residue_index: int = 0
    chain: str = "A"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    charge: float = 0.0
    radius: float = 1.5

    @property
    def position(self) -> np.ndarray:
        """Return the atom's Cartesian coordinates in angstrom as a length-3 array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @position.setter
    def position(self, xyz: np.ndarray) -> None:
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    def distance_to(self, other: "Atom") -> float:
        """Return the Euclidean distance in angstrom between this atom and another."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def __repr__(self) -> str:
        return (
            f"Atom({self.name!r} res={self.residue_name}{self.residue_index} "
            f"pos=({self.x:.2f},{self.y:.2f},{self.z:.2f}) "
            f"q={self.charge:.3f} r={self.radius:.2f})"
        )


# A rigid molecule made of atoms.
@dataclass
class Molecule:
    """Collection of atoms representing one rigid body."""

    name: str = ""
    atoms: List[Atom] = field(default_factory=list)

    # The methods below compute geometric properties of the molecule.
    def centroid(self) -> np.ndarray:
        """Return the unweighted mean position of all atoms in angstrom."""
        if not self.atoms:
            return np.zeros(3)
        pos = np.array([a.position for a in self.atoms])
        return pos.mean(axis=0)

    def center_of_mass(self) -> np.ndarray:
        """Return the center of mass.

        PQR files carry no atomic masses, so every atom is treated as having
        equal mass and this reduces to the plain centroid.
        """
        return self.centroid()

    def total_charge(self) -> float:
        """Return the net charge of the molecule, summed over all atomic charges."""
        return sum(a.charge for a in self.atoms)

    def radius_of_gyration(self) -> float:
        """Return the radius of gyration in angstrom about the centroid.

        This is the root-mean-square distance of the atoms from the centroid,
        Rg = sqrt(mean over atoms of |r - c|^2). Here r is each atom position and
        c is the centroid, both in angstrom.
        """
        if not self.atoms:
            return 0.0
        c = self.centroid()
        pos = np.array([a.position for a in self.atoms])
        return float(np.sqrt(((pos - c) ** 2).sum(axis=1).mean()))

    def bounding_radius(self) -> float:
        """Return the radius in angstrom of the smallest sphere about the centroid
        that encloses every atom, measured to each atom's surface rather than its
        center. It is the maximum over atoms of the centroid-to-center distance plus
        the atom radius."""
        if not self.atoms:
            return 0.0
        c = self.centroid()
        return max(np.linalg.norm(a.position - c) + a.radius for a in self.atoms)

    def positions_array(self) -> np.ndarray:
        """Return the atom positions in angstrom as an array of shape (N, 3)."""
        return np.array([a.position for a in self.atoms])

    def charges_array(self) -> np.ndarray:
        """Return the atomic charges as a length-N array."""
        return np.array([a.charge for a in self.atoms])

    def radii_array(self) -> np.ndarray:
        """Return the atomic radii in angstrom as a length-N array."""
        return np.array([a.radius for a in self.atoms])

    def translate(self, delta: np.ndarray) -> None:
        """Shift every atom in place by the displacement delta given in angstrom."""
        for a in self.atoms:
            a.x += delta[0]
            a.y += delta[1]
            a.z += delta[2]

    def rotate(self, R: np.ndarray) -> None:
        """Rotate every atom in place about the origin by the rotation matrix R."""
        for a in self.atoms:
            new_pos = R @ a.position
            a.position = new_pos

    def rotate_about_centroid(self, R: np.ndarray) -> None:
        """Rotate the molecule in place by R about its own centroid.

        The molecule is shifted so its centroid sits at the origin, rotated by R,
        and shifted back, which leaves the centroid fixed.
        """
        c = self.centroid()
        self.translate(-c)
        self.rotate(R)
        self.translate(c)

    def __len__(self) -> int:
        return len(self.atoms)

    def __repr__(self) -> str:
        return f"Molecule({self.name!r}, {len(self.atoms)} atoms, q={self.total_charge():.2f})"


# An axis-aligned bounding box.
@dataclass
class BoundingBox:
    """An axis-aligned box, stored as the minimum and maximum extent along each
    Cartesian axis in angstrom."""

    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    zmin: float = 0.0
    zmax: float = 0.0

    @classmethod
    def from_molecule(cls, mol: Molecule, padding: float = 0.0) -> "BoundingBox":
        """Build the tightest axis-aligned box enclosing the atom centers of mol.

        The optional padding in angstrom is added on every side, for example to
        leave room for atomic radii or a margin around the molecule.
        """
        if not mol.atoms:
            return cls()
        xs = [a.x for a in mol.atoms]
        ys = [a.y for a in mol.atoms]
        zs = [a.z for a in mol.atoms]
        return cls(
            xmin=min(xs) - padding,
            xmax=max(xs) + padding,
            ymin=min(ys) - padding,
            ymax=max(ys) + padding,
            zmin=min(zs) - padding,
            zmax=max(zs) + padding,
        )

    @property
    def center(self) -> np.ndarray:
        """Return the geometric center of the box in angstrom."""
        return np.array(
            [
                (self.xmin + self.xmax) / 2,
                (self.ymin + self.ymax) / 2,
                (self.zmin + self.zmax) / 2,
            ]
        )

    @property
    def size(self) -> np.ndarray:
        """Return the side lengths of the box along each axis in angstrom."""
        return np.array(
            [
                self.xmax - self.xmin,
                self.ymax - self.ymin,
                self.zmax - self.zmin,
            ]
        )

    def contains(self, point: np.ndarray) -> bool:
        """Return True if the point lies inside the box, with the boundary counted
        as inside."""
        return (
            self.xmin <= point[0] <= self.xmax
            and self.ymin <= point[1] <= self.ymax
            and self.zmin <= point[2] <= self.zmax
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x=[{self.xmin:.1f},{self.xmax:.1f}] "
            f"y=[{self.ymin:.1f},{self.ymax:.1f}] "
            f"z=[{self.zmin:.1f},{self.zmax:.1f}])"
        )


# A single contact between an atom of one molecule and an atom of the other.
@dataclass
class ContactPair:
    """One reaction contact, pairing a specific atom in mol1 with a specific atom
    in mol2. The contact is satisfied when the two atoms are closer than
    distance_cutoff."""

    mol1_atom_index: int = 0
    mol2_atom_index: int = 0
    distance_cutoff: float = 5.0  # contact distance threshold in Å

    def __repr__(self) -> str:
        return (
            f"ContactPair(mol1[{self.mol1_atom_index}] ↔ "
            f"mol2[{self.mol2_atom_index}], "
            f"cutoff={self.distance_cutoff:.1f} Å)"
        )


# A reaction criterion defined by a set of atomic contacts.
@dataclass
class ReactionCriteria:
    """A reaction criterion built from a set of contact pairs.

    The criterion fires once the number of satisfied contact pairs reaches the
    threshold n_needed. By default n_needed equals the number of pairs, so every
    contact must be satisfied at the same time, which behaves like a logical AND.
    Setting n_needed below the number of pairs relaxes this so that any sufficient
    subset of contacts triggers the reaction, which behaves more like a logical OR.
    """

    name: str = "reaction"
    pairs: List[ContactPair] = field(default_factory=list)
    n_needed: int = -1  # a value of -1 means require all pairs
    # The two labels below are only used when state-machine reactions are enabled.
    # Leaving them as None selects the flattened-reactions path instead.
    state_before: "Optional[str]" = None
    state_after: "Optional[str]" = None

    def is_satisfied(self, mol1: Molecule, mol2: Molecule) -> bool:
        """Return True if the reaction criterion is met for the given molecule pair.

        Each contact pair is checked by measuring the distance between its two
        atoms and counting the pair as satisfied when that distance is below the
        pair's cutoff. The criterion is met as soon as the count of satisfied pairs
        reaches the threshold, which is n_needed when set and otherwise the total
        number of pairs. A criterion with no required contacts is always satisfied.
        """
        threshold = len(self.pairs) if self.n_needed < 0 else self.n_needed
        if threshold == 0:
            return True
        n_satis = 0
        for pair in self.pairs:
            a1 = mol1.atoms[pair.mol1_atom_index]
            a2 = mol2.atoms[pair.mol2_atom_index]
            if a1.distance_to(a2) < pair.distance_cutoff:
                n_satis += 1
                if n_satis >= threshold:
                    return True
        return False

    def __repr__(self) -> str:
        return f"ReactionCriteria({self.name!r}, {len(self.pairs)} pairs, n_needed={self.n_needed})"
