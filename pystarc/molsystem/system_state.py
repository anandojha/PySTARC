"""
Data structures that track the state and outcome of a Brownian-dynamics trajectory.
"""

from __future__ import annotations
from pystarc.transforms.quaternion import Quaternion
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class Fate(Enum):
    """Terminal outcome of a BD trajectory."""

    ONGOING = auto()  # The trajectory is still running.
    REACTED = auto()  # The reaction criterion was satisfied.
    ESCAPED = auto()  # The molecule reached the escape radius.
    MAX_STEPS = auto()  # The trajectory exceeded the maximum number of steps.


@dataclass
class SystemState:
    """
    The instantaneous state of a two-molecule Brownian-dynamics system. Molecule 1
    is held fixed at the origin, which is the default for rigid-body BD, and
    molecule 2 diffuses relative to it.
    """

    # Position and orientation of molecule 2.
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    # Energetics. Energy is in units of kBT, force in kBT/Å, and torque in kBT/rad.
    energy: float = 0.0  # kBT
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # kBT/Å
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))  # kBT/rad
    # Bookkeeping for the current step and simulation time (in picoseconds).
    step: int = 0
    time: float = 0.0  # ps
    fate: Fate = Fate.ONGOING
    reaction_name: Optional[str] = None

    def separation(self) -> float:
        """Return the distance from the origin to the current position."""
        return float(np.linalg.norm(self.position))

    def copy(self) -> "SystemState":
        return SystemState(
            position=self.position.copy(),
            orientation=Quaternion(
                self.orientation.w,
                self.orientation.x,
                self.orientation.y,
                self.orientation.z,
            ),
            energy=self.energy,
            force=self.force.copy(),
            torque=self.torque.copy(),
            step=self.step,
            time=self.time,
            fate=self.fate,
            reaction_name=self.reaction_name,
        )

    def __repr__(self) -> str:
        r = self.separation()
        return (
            f"SystemState(step={self.step}, r={r:.2f}Å, "
            f"E={self.energy:.3f}kBT, fate={self.fate.name})"
        )


@dataclass
class TrajectoryResult:
    """Summary statistics for one completed Brownian-dynamics trajectory."""

    fate: Fate
    steps: int
    time_ps: float
    final_separation: float
    reaction_name: Optional[str] = None
    energy_at_reaction: float = 0.0
    # Per-trajectory diagnostics. Chain BD fills these in, whereas rigid-body BD
    # leaves them as None so that existing rigid-body construction sites are
    # unaffected.
    encounter_pos: Optional[np.ndarray] = None  # Center-of-mass position (3,) at the reaction.
    encounter_q: Optional[np.ndarray] = None  # Orientation quaternion (4,) at the reaction.
    near_miss_pos: Optional[np.ndarray] = None  # Center-of-mass position (3,) at closest approach.
    near_miss_dist: Optional[float] = None  # Closest separation reached during the trajectory.
    path_steps: Optional[np.ndarray] = None  # Step numbers (n_snap,) of the saved snapshots.
    path_com: Optional[np.ndarray] = None  # Center-of-mass positions (n_snap, 3) along the path.
    path_q: Optional[np.ndarray] = None  # Orientation quaternions (n_snap, 4) along the path.
    energy_steps: Optional[np.ndarray] = (
        None  # Energies (n_snap, 4) as total, electrostatic, Born, and steric.
    )
    radial_trace: Optional[np.ndarray] = None  # Separation magnitudes (n_snap,) along the path.
    contact_counts: Optional[Dict[Tuple[int, int], int]] = None

    @property
    def reacted(self) -> bool:
        return self.fate == Fate.REACTED

    @property
    def escaped(self) -> bool:
        return self.fate == Fate.ESCAPED

    def __repr__(self) -> str:
        return (
            f"TrajectoryResult(fate={self.fate.name}, "
            f"steps={self.steps}, t={self.time_ps:.1f}ps, "
            f"r_final={self.final_separation:.1f}Å)"
        )
