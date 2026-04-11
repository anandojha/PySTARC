"""
Reaction pathways and state machine for PySTARC.

"""

from __future__ import annotations
from pystarc.structures.molecules import Molecule, ReactionCriteria, ContactPair
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import random

@dataclass
class ReactionInterface:
    """
    One reaction pathway: a name + a list of contact criteria.
    All contacts must be satisfied simultaneously for the reaction to occur.
    """
    name: str
    criteria: ReactionCriteria
    probability: float = 1.0     # reaction probability when contacts are met

    def check(self, mol1: Molecule, mol2: Molecule) -> bool:
        """Return True if this reaction has fired."""
        if not self.criteria.is_satisfied(mol1, mol2):
            return False
        if self.probability >= 1.0:
            return True
        return random.random() < self.probability

    def __repr__(self) -> str:
        return (f"ReactionInterface({self.name!r}, "
                f"p={self.probability:.3f}, "
                f"{len(self.criteria.pairs)} contacts)")

class PathwaySet:
    """
    Collection of all reaction pathways for a simulation.
    Iterates through pathways in order, returns first match.
    """
    def __init__(self, reactions: Optional[List[ReactionInterface]] = None):
        self.reactions: List[ReactionInterface] = reactions or []

    def add(self, rxn: ReactionInterface) -> None:
        self.reactions.append(rxn)

    def check_all(self,
                  mol1: Molecule,
                  mol2: Molecule,
                  rng: Optional[np.random.Generator] = None) -> Optional[str]:
        """
        Check all pathways; return name of first that fires, or None.
        """
        for rxn in self.reactions:
            if rxn.criteria.is_satisfied(mol1, mol2):
                prob = rxn.probability
                if prob >= 1.0:
                    return rxn.name
                if rng is not None:
                    if rng.random() < prob:
                        return rxn.name
                else:
                    import random
                    if random.random() < prob:
                        return rxn.name
        return None

    def __len__(self) -> int:
        return len(self.reactions)

    def __repr__(self) -> str:
        names = [r.name for r in self.reactions]
        return f"PathwaySet({names})"

def make_default_reaction(mol1: Molecule,
                          mol2: Molecule,
                          cutoff: float = 5.0,
                          n_pairs: int = 3) -> ReactionInterface:
    """
    Build a default reaction using the n closest atom pairs at centroid approach.
    """
    c1 = mol1.centroid()
    c2 = mol2.centroid()
    # Pick atoms nearest to the opposing centroid
    def closest_atoms(mol: Molecule, target: np.ndarray, n: int) -> List[int]:
        dists = [np.linalg.norm(a.position - target) for a in mol.atoms]
        return sorted(range(len(dists)), key=lambda i: dists[i])[:n]
    idx1 = closest_atoms(mol1, c2, n_pairs)
    idx2 = closest_atoms(mol2, c1, n_pairs)
    pairs = [ContactPair(i, j, cutoff) for i, j in zip(idx1, idx2)]
    criteria = ReactionCriteria(name="default", pairs=pairs)
    return ReactionInterface(name="default_reaction", criteria=criteria)