"""
Reaction pathways and the state machine that PySTARC uses to detect when a
bimolecular reaction has occurred during a Brownian-dynamics trajectory.
"""

from __future__ import annotations
from pystarc.structures.molecules import Molecule, ReactionCriteria, ContactPair
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class ReactionInterface:
    """
    A single reaction pathway, consisting of a name and a list of contact
    criteria. Every contact must be satisfied at the same time for the reaction
    to occur.
    """

    name: str
    criteria: ReactionCriteria
    probability: float = 1.0  # Probability that the reaction fires once all contacts are met.
    # Labels for the state machine, used only when state_machine_reactions is True.
    # When both default to None, the simulator follows the flattened-reactions path instead.
    state_before: "Optional[str]" = None
    state_after: "Optional[str]" = None

    def check(
        self,
        mol1: Molecule,
        mol2: Molecule,
        rng: "Optional[np.random.Generator]" = None,
    ) -> bool:
        """Return True if this reaction has fired.

        The argument rng is an optional numpy Generator used for the probability
        gate when self.probability is below 1.0. Pass an explicit rng to make the
        result reproducible. When rng is None, np.random.default_rng() supplies
        the single random draw, which is not reproducible across runs but is
        isolated from any global random state.
        """
        if not self.criteria.is_satisfied(mol1, mol2):
            return False
        if self.probability >= 1.0:
            return True
        if rng is None:
            rng = np.random.default_rng()
        return float(rng.random()) < self.probability

    def __repr__(self) -> str:
        return (
            f"ReactionInterface({self.name!r}, "
            f"p={self.probability:.3f}, "
            f"{len(self.criteria.pairs)} contacts)"
        )


class PathwaySet:
    """
    The full collection of reaction pathways for a simulation. The pathways are
    examined in order, and the first one that matches is returned.
    """

    def __init__(
        self,
        reactions: Optional[List[ReactionInterface]] = None,
        first_state: Optional[str] = None,
    ):
        self.reactions: List[ReactionInterface] = reactions or []
        # Initial state label for trajectories run in state-machine reaction mode.
        # This holds the value parsed from <first_state> in rxns.xml. It is None
        # when state-machine mode is not used, and in that case the simulator
        # ignores it.
        self.first_state: Optional[str] = first_state

    def add(self, rxn: ReactionInterface) -> None:
        self.reactions.append(rxn)

    def check_all(
        self, mol1: Molecule, mol2: Molecule, rng: Optional[np.random.Generator] = None
    ) -> Optional[str]:
        """
        Check every pathway and return the name of the first one that fires, or
        None if none of them do.
        """
        for rxn in self.reactions:
            if rxn.criteria.is_satisfied(mol1, mol2):
                prob = rxn.probability
                if prob >= 1.0:
                    return rxn.name
                # Use the supplied rng so the result is reproducible, and fall
                # back to a fresh default_rng when none is supplied. The fresh
                # generator isolates the draw from any module-level global state,
                # but it is not reproducible across runs unless the caller passes
                # a seeded rng.
                _rng = rng if rng is not None else np.random.default_rng()
                if _rng.random() < prob:
                    return rxn.name
        return None

    def __len__(self) -> int:
        return len(self.reactions)

    def __repr__(self) -> str:
        names = [r.name for r in self.reactions]
        return f"PathwaySet({names})"


def make_default_reaction(
    mol1: Molecule, mol2: Molecule, cutoff: float = 5.0, n_pairs: int = 3
) -> ReactionInterface:
    """
    Build a default reaction from the n closest atom pairs as the two molecules
    approach along the line joining their centroids.
    """
    c1 = mol1.centroid()
    c2 = mol2.centroid()

    # Pick the atoms nearest to the opposing molecule's centroid.
    def closest_atoms(mol: Molecule, target: np.ndarray, n: int) -> List[int]:
        dists = [np.linalg.norm(a.position - target) for a in mol.atoms]
        return sorted(range(len(dists)), key=lambda i: dists[i])[:n]

    idx1 = closest_atoms(mol1, c2, n_pairs)
    idx2 = closest_atoms(mol2, c1, n_pairs)
    pairs = [ContactPair(i, j, cutoff) for i, j in zip(idx1, idx2)]
    criteria = ReactionCriteria(name="default", pairs=pairs)
    return ReactionInterface(name="default_reaction", criteria=criteria)
