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
    probability: float = (
        1.0  # Probability that the reaction fires once all contacts are met.
    )
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

    The number of contact pairs is clamped to the smaller atom count of the two
    molecules, because each pair consumes one atom from each side. A reaction with
    no contact pairs is degenerate: an empty ReactionCriteria is always satisfied
    and would fire on every step, so this function refuses to build one. If either
    molecule has no atoms, or the requested pair count n_pairs is not positive, a
    ValueError is raised with a clear message rather than returning a reaction that
    always fires.
    """
    n1 = len(mol1.atoms)
    n2 = len(mol2.atoms)
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"Cannot build a default reaction: molecule {mol1.name!r} has {n1} "
            f"atoms and molecule {mol2.name!r} has {n2} atoms. Each contact pair "
            f"needs one atom from each molecule, so both molecules must contain "
            f"at least one atom."
        )
    if n_pairs < 1:
        raise ValueError(
            f"Cannot build a default reaction with n_pairs={n_pairs}. A reaction "
            f"needs at least one contact pair, otherwise its criteria would be "
            f"empty and would fire on every step."
        )

    c1 = mol1.centroid()
    c2 = mol2.centroid()

    # Pick the atoms nearest to the opposing molecule's centroid.
    def closest_atoms(mol: Molecule, target: np.ndarray, n: int) -> List[int]:
        dists = [np.linalg.norm(a.position - target) for a in mol.atoms]
        return sorted(range(len(dists)), key=lambda i: dists[i])[:n]

    # Clamp the pair count so it never exceeds the atoms available on either
    # molecule. For molecules large enough to supply n_pairs atoms on both sides
    # this leaves the result unchanged, since closest_atoms already returned at
    # most n_pairs indices and zip truncated to the shorter list. The clamp makes
    # that truncation explicit and well defined for small molecules.
    n_eff = min(n_pairs, n1, n2)
    idx1 = closest_atoms(mol1, c2, n_eff)
    idx2 = closest_atoms(mol2, c1, n_eff)
    pairs = [ContactPair(i, j, cutoff) for i, j in zip(idx1, idx2)]
    criteria = ReactionCriteria(name="default", pairs=pairs)
    return ReactionInterface(name="default_reaction", criteria=criteria)
