"""
Multipole effective charges for PySTARC.

The standard approach uses "effective charges", i.e., a set of point charges that
reproduce the electrostatic potential of the molecule outside a
bounding sphere. This is faster than evaluating the full APBS grid
for long-range interactions because we only need a small number of
effective charges (typically 20-100) instead of interpolating a
161^3 grid.

The method is described in:
  Gabdoulline & Wade (1996) "Simulation of the diffusional association
  of barnase and barstar". Biophys J 72:1917-1929.

The effective potential at point r outside the bounding sphere is:

    Φ_eff(r) = Σ_k  q_k * exp(-|r - r_k| / λ_D) / |r - r_k|  * l_B

where q_k, r_k are the effective charges and their positions,
and the sum runs over all effective charges (typically 20-100).

This is used for long-range forces when the ligand is outside the
finest APBS grid. Inside the finest grid, the APBS potential is
used directly.

For PySTARC, we implement this as a fallback for points outside all
loaded DX grids.

"""

from __future__ import annotations
from pystarc.global_defs.constants import BJERRUM_LENGTH, DEFAULT_DEBYE_LENGTH
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import math


class EffectiveCharges:
    """
    Effective point charges that reproduce long-range electrostatics.
    The potential at point r:
        Φ(r) = Σ_k q_k * l_B * exp(-|r-r_k|/λ_D) / |r-r_k|
    The force on a test charge q at point r:
        F(r) = -q * ∇Φ(r)
             = q * Σ_k q_k * l_B * exp(-|r-r_k|/λ_D) / |r-r_k|^2
                           * (1/λ_D + 1/|r-r_k|) * (r-r_k)/|r-r_k|
    """

    def __init__(
        self,
        positions: np.ndarray,  # (N,3) [Å]
        charges: np.ndarray,  # (N,)  [e]
        debye_length: float = DEFAULT_DEBYE_LENGTH,
        bjerrum_length: float = BJERRUM_LENGTH,
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.charges = np.asarray(charges, dtype=np.float64)
        self.debye_length = debye_length
        self.bjerrum_length = bjerrum_length

    def potential(self, r: np.ndarray) -> float:
        """
        Debye-Hückel potential at point r from all effective charges.
        Φ(r) = Σ_k q_k * l_B * exp(-d_k/λ_D) / d_k   [kBT/e]
        """
        d_vec = r[np.newaxis, :] - self.positions  # (N,3)
        d = np.linalg.norm(d_vec, axis=1)  # (N,)
        mask = d > 1e-10
        phi = np.sum(
            self.charges[mask]
            * self.bjerrum_length
            * np.exp(-d[mask] / self.debye_length)
            / d[mask]
        )
        return float(phi)

    def force_on_charge(self, r: np.ndarray, q: float) -> np.ndarray:
        """
        Force on test charge q at point r.
        F = -q ∇Φ(r)    [kBT/Å]
        """
        if abs(q) < 1e-9:
            return np.zeros(3)
        d_vec = r[np.newaxis, :] - self.positions  # (N,3)
        d = np.linalg.norm(d_vec, axis=1)  # (N,)
        mask = d > 1e-10
        # Gradient of Φ w.r.t. r:
        # ∂Φ/∂r = Σ_k q_k l_B exp(-d/λ) * [-(1/λ + 1/d)] * (r-r_k)/d
        # Force on q: F = -q ∂Φ/∂r
        inv_d = 1.0 / d[mask]
        exp_fac = np.exp(-d[mask] / self.debye_length)
        coeff = (
            self.charges[mask]
            * self.bjerrum_length
            * exp_fac
            * (1.0 / self.debye_length + inv_d)
            * inv_d
        )  # (N,)
        # d_vec[mask] / d[mask,None] = unit vectors
        unit = d_vec[mask] / d[mask, np.newaxis]  # (N,3)
        grad_phi = -(coeff[:, np.newaxis] * unit).sum(axis=0)  # (3,)
        return -q * grad_phi

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        debye_length: float = DEFAULT_DEBYE_LENGTH,
        bjerrum_length: float = BJERRUM_LENGTH,
    ) -> "EffectiveCharges":
        """
        Load effective charges from a the reference implementation XML file.
        Supports both *_cheby.xml and *_mpole.xml formats.
        XML format (the reference implementation):
            <charges>
              <charge>
                <x> -4.72 </x>
                <y> -2.97 </y>
                <z> -9.01 </z>
                <q> 0.523 </q>
              </charge>
              ...
            </charges>
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        positions = []
        charges = []
        # Handle both <charges> and <multipole> root tags
        charge_elements = root.findall(".//charge")
        if not charge_elements:
            charge_elements = root.findall(".//point_charge")
        for elem in charge_elements:
            x = float(elem.findtext("x", "0"))
            y = float(elem.findtext("y", "0"))
            z = float(elem.findtext("z", "0"))
            q = float(elem.findtext("q", "0"))
            positions.append([x, y, z])
            charges.append(q)
        if not positions:
            raise ValueError(f"No charges found in {xml_path}")
        return cls(
            positions=np.array(positions),
            charges=np.array(charges),
            debye_length=debye_length,
            bjerrum_length=bjerrum_length,
        )

    def __len__(self) -> int:
        return len(self.charges)

    def __repr__(self) -> str:
        return (
            f"EffectiveCharges({len(self)} charges, "
            f"q_net={self.charges.sum():.2f} e, "
            f"λ_D={self.debye_length:.3f} Å)"
        )


def load_effective_charges(
    directory: str | Path,
    prefix: str,
    debye_length: float = DEFAULT_DEBYE_LENGTH,
    bjerrum_length: float = BJERRUM_LENGTH,
) -> Optional[EffectiveCharges]:
    """
    Auto-detect and load effective charges from a the reference implementation directory.
    Looks for files in this priority order:
      1. <prefix>_cheby.xml     (Chebyshev effective charges - most accurate)
      2. <prefix>_mpole.xml     (multipole expansion)
    Returns None if no file is found (not an error - DX grids alone suffice).
    """
    d = Path(directory)
    for suffix in ["_cheby.xml", "_mpole.xml", "_charges.xml"]:
        p = d / f"{prefix}{suffix}"
        if p.exists():
            try:
                ec = EffectiveCharges.from_xml(p, debye_length, bjerrum_length)
                return ec
            except Exception:
                continue
    return None
