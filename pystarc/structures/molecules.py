"""
Core molecular structure types for PySTARC.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math


# Atom
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
        return np.array([self.x, self.y, self.z], dtype=float)

    @position.setter
    def position(self, xyz: np.ndarray) -> None:
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    def distance_to(self, other: "Atom") -> float:
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


# Molecule
@dataclass
class Molecule:
    """Collection of atoms representing one rigid body."""

    name: str = ""
    atoms: List[Atom] = field(default_factory=list)

    # geometry helpers
    def centroid(self) -> np.ndarray:
        if not self.atoms:
            return np.zeros(3)
        pos = np.array([a.position for a in self.atoms])
        return pos.mean(axis=0)

    def center_of_mass(self) -> np.ndarray:
        """Uniform-mass centroid (PQR files lack mass data)."""
        return self.centroid()

    def total_charge(self) -> float:
        return sum(a.charge for a in self.atoms)

    def radius_of_gyration(self) -> float:
        if not self.atoms:
            return 0.0
        c = self.centroid()
        pos = np.array([a.position for a in self.atoms])
        return float(np.sqrt(((pos - c) ** 2).sum(axis=1).mean()))

    def bounding_radius(self) -> float:
        """Maximum distance from centroid to any atom surface."""
        if not self.atoms:
            return 0.0
        c = self.centroid()
        return max(np.linalg.norm(a.position - c) + a.radius for a in self.atoms)

    def positions_array(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms])

    def charges_array(self) -> np.ndarray:
        return np.array([a.charge for a in self.atoms])

    def radii_array(self) -> np.ndarray:
        return np.array([a.radius for a in self.atoms])

    def translate(self, delta: np.ndarray) -> None:
        for a in self.atoms:
            a.x += delta[0]
            a.y += delta[1]
            a.z += delta[2]

    def rotate(self, R: np.ndarray) -> None:
        """Rotate all atoms in-place around the origin."""
        for a in self.atoms:
            new_pos = R @ a.position
            a.position = new_pos

    def rotate_about_centroid(self, R: np.ndarray) -> None:
        c = self.centroid()
        self.translate(-c)
        self.rotate(R)
        self.translate(c)

    def __len__(self) -> int:
        return len(self.atoms)

    def __repr__(self) -> str:
        return f"Molecule({self.name!r}, {len(self.atoms)} atoms, q={self.total_charge():.2f})"


# BoundingBox
@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""

    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    zmin: float = 0.0
    zmax: float = 0.0

    @classmethod
    def from_molecule(cls, mol: Molecule, padding: float = 0.0) -> "BoundingBox":
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
        return np.array(
            [
                (self.xmin + self.xmax) / 2,
                (self.ymin + self.ymax) / 2,
                (self.zmin + self.zmax) / 2,
            ]
        )

    @property
    def size(self) -> np.ndarray:
        return np.array(
            [
                self.xmax - self.xmin,
                self.ymax - self.ymin,
                self.zmax - self.zmin,
            ]
        )

    def contains(self, point: np.ndarray) -> bool:
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


# ContactPair
@dataclass
class ContactPair:
    """A single reaction contact between an atom in mol1 and mol2."""

    mol1_atom_index: int = 0
    mol2_atom_index: int = 0
    distance_cutoff: float = 5.0  # Å

    def __repr__(self) -> str:
        return (
            f"ContactPair(mol1[{self.mol1_atom_index}] ↔ "
            f"mol2[{self.mol2_atom_index}], "
            f"cutoff={self.distance_cutoff:.1f} Å)"
        )


# ReactionCriteria
@dataclass
class ReactionCriteria:
    """
    Set of contact pairs defining a reaction criterion.
    Reaction criterion: fires when n_satisfied >= n_needed.
    Default n_needed = len(pairs) (all pairs must be satisfied).
    Setting n_needed < len(pairs) allows or-like logic.
    """

    name: str = "reaction"
    pairs: List[ContactPair] = field(default_factory=list)
    n_needed: int = -1  # -1 means all pairs (default: all pairs)

    def is_satisfied(self, mol1: Molecule, mol2: Molecule) -> bool:
        """
        n_satis = 0
        for pair in pairs:
            if distance < req_distance: n_satis++
            if n_satis >= n_needed: return True
        return False
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
