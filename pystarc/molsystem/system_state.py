"""
SystemState and SystemFate to track the state of a BD trajectory.
"""

from __future__ import annotations
from pystarc.transforms.quaternion import Quaternion
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np

class Fate(Enum):
    """Terminal outcome of a BD trajectory."""
    ONGOING    = auto()   # still running
    REACTED    = auto()   # reaction criterion satisfied
    ESCAPED    = auto()   # reached escape radius
    MAX_STEPS  = auto()   # exceeded maximum step count

@dataclass
class SystemState:
    """
    Instantaneous state of a two-molecule BD system.
    Molecule 1 is fixed at the origin (default for rigid-body BD).
    Molecule 2 diffuses relative to molecule 1.
    """
    # molecule 2 position and orientation 
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    # energetics 
    energy: float = 0.0        # kBT
    force:  np.ndarray = field(default_factory=lambda: np.zeros(3))   # kBT/Å
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))   # kBT/rad
    # bookkeeping 
    step: int = 0
    time: float = 0.0          # ps
    fate: Fate = Fate.ONGOING
    reaction_name: Optional[str] = None

    def separation(self) -> float:
        """Distance from origin to current position."""
        return float(np.linalg.norm(self.position))

    def copy(self) -> "SystemState":
        return SystemState(
            position=self.position.copy(),
            orientation=Quaternion(self.orientation.w, self.orientation.x,
                                   self.orientation.y, self.orientation.z),
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
        return (f"SystemState(step={self.step}, r={r:.2f}Å, "
                f"E={self.energy:.3f}kBT, fate={self.fate.name})")

@dataclass
class TrajectoryResult:
    """Summary statistics for one completed BD trajectory."""
    fate: Fate
    steps: int
    time_ps: float
    final_separation: float
    reaction_name: Optional[str] = None
    energy_at_reaction: float = 0.0
    @property
    def reacted(self) -> bool:
        return self.fate == Fate.REACTED
    @property
    def escaped(self) -> bool:
        return self.fate == Fate.ESCAPED
    def __repr__(self) -> str:
        return (f"TrajectoryResult(fate={self.fate.name}, "
                f"steps={self.steps}, t={self.time_ps:.1f}ps, "
                f"r_final={self.final_separation:.1f}Å)")