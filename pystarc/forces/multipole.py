"""
Multipole effective charges for PySTARC.

The standard approach uses effective charges, meaning a small set of point
charges that reproduce the electrostatic potential of the molecule outside a
bounding sphere. This is faster than evaluating the full APBS grid for
long-range interactions because we only need a small number of effective
charges (typically 20 to 100) instead of interpolating a 161^3 grid.

The method is described in Gabdoulline and Wade (1996), "Simulation of the
diffusional association of barnase and barstar", Biophys J 72:1917-1929.

The effective potential at a point r outside the bounding sphere is

    Φ_eff(r) = l_B Σ_k q_k exp(-|r - r_k| / λ_D) / |r - r_k|

Here q_k and r_k are the effective charges and their positions, l_B is the
Bjerrum length, λ_D is the Debye length, and the sum runs over all effective
charges (typically 20 to 100).

This is used for long-range forces when the ligand is outside the finest APBS
grid. Inside the finest grid, the APBS potential is used directly. In PySTARC
this serves as a fallback for points that fall outside all loaded DX grids.
"""

from __future__ import annotations
from pystarc.global_defs.constants import BJERRUM_LENGTH, DEFAULT_DEBYE_LENGTH
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import warnings
import math


class EffectiveCharges:
    """
    Effective point charges that reproduce long-range electrostatics.

    The potential at a point r is

        Φ(r) = Σ_k q_k l_B exp(-|r-r_k|/λ_D) / |r-r_k|

    and the force on a test charge q at point r is

        F(r) = -q ∇Φ(r)
             = q Σ_k q_k l_B exp(-|r-r_k|/λ_D) / |r-r_k|^2
                          (1/λ_D + 1/|r-r_k|) (r-r_k)/|r-r_k|

    Here q_k and r_k are the effective charges and their positions, l_B is the
    Bjerrum length, and λ_D is the Debye length.
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
        Return the Debye-Hückel potential at point r summed over all effective
        charges, in units of kBT/e.

        Φ(r) = Σ_k q_k l_B exp(-d_k/λ_D) / d_k

        Here d_k is the distance from r to the k-th effective charge.
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
        Return the force on a test charge q at point r, in units of kBT/Å.

        F = -q ∇Φ(r)
        """
        if abs(q) < 1e-9:
            return np.zeros(3)
        d_vec = r[np.newaxis, :] - self.positions  # (N,3)
        d = np.linalg.norm(d_vec, axis=1)  # (N,)
        mask = d > 1e-10
        # The gradient of Φ with respect to r is
        # ∂Φ/∂r = Σ_k q_k l_B exp(-d/λ) [-(1/λ + 1/d)] (r-r_k)/d,
        # and the force on the test charge is F = -q ∂Φ/∂r.
        inv_d = 1.0 / d[mask]
        exp_fac = np.exp(-d[mask] / self.debye_length)
        coeff = (
            self.charges[mask]
            * self.bjerrum_length
            * exp_fac
            * (1.0 / self.debye_length + inv_d)
            * inv_d
        )  # (N,)
        # Dividing each separation vector by its length gives the unit vectors.
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
        Load effective point charges from a BrownDye2 charge file.

        BrownDye2 writes effective charges as a list of <point> elements. The
        lumped-charge writer nests the coordinates in a <pos> block and stores
        the magnitude in <q>, while the test-charge and effective-volume writers
        place <x>, <y>, and <z> directly inside <point> and store the magnitude
        in <charge>. Both layouts are read here:

            <point>
              <x> 1.0 </x> <y> 2.0 </y> <z> 3.0 </z>
              <charge> 0.5 </charge>
            </point>

            <point type="permanent">
              <pos> <x>1.0</x> <y>2.0</y> <z>3.0</z> </pos>
              <q> 0.5 </q>
            </point>

        A PySTARC-native layout that wraps each charge in <charge> or
        <point_charge> with child <x>, <y>, <z>, and <q> is also accepted.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        positions = []
        charges = []

        def _coord(elem, axis):
            node = elem.find(f".//{axis}")
            text = node.text if node is not None else None
            return float(text) if text and text.strip() else 0.0

        point_elements = root.findall(".//point")
        if point_elements:
            for pt in point_elements:
                # The signed charge is in <q>, in <actual_charge> when <charge>
                # holds the squared charge, or in <charge> for the plain
                # test-charge and effective-volume writers.
                q_text = (
                    pt.findtext("q")
                    or pt.findtext("actual_charge")
                    or pt.findtext("charge")
                )
                q = float(q_text) if q_text and q_text.strip() else 0.0
                positions.append([_coord(pt, "x"), _coord(pt, "y"), _coord(pt, "z")])
                charges.append(q)
        else:
            # PySTARC-native fallback: each charge wrapped in <charge> or
            # <point_charge> with child <x>, <y>, <z>, and <q>.
            charge_elements = root.findall(".//charge")
            if not charge_elements:
                charge_elements = root.findall(".//point_charge")
            for elem in charge_elements:
                positions.append(
                    [
                        float(elem.findtext("x", "0")),
                        float(elem.findtext("y", "0")),
                        float(elem.findtext("z", "0")),
                    ]
                )
                charges.append(float(elem.findtext("q", "0")))

        if not positions:
            raise ValueError(f"No charges found in {xml_path}")
        charges_arr = np.array(charges, dtype=float)
        if not np.any(np.abs(charges_arr) > 1e-12):
            raise ValueError(
                f"All effective charges in {xml_path} are zero, which means the "
                f"charge-file layout was not recognised"
            )
        return cls(
            positions=np.array(positions, dtype=float),
            charges=charges_arr,
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
    Detect and load effective charges from a reference implementation
    directory. The search tries files in order of preference, first
    <prefix>_cheby.xml (Chebyshev effective charges, the most accurate) and
    then <prefix>_mpole.xml (a multipole expansion). When no file is found the
    function returns None, which is not an error because the DX grids alone are
    sufficient.
    """
    d = Path(directory)
    for suffix in ["_cheby.xml", "_mpole.xml", "_charges.xml"]:
        p = d / f"{prefix}{suffix}"
        if p.exists():
            try:
                return EffectiveCharges.from_xml(p, debye_length, bjerrum_length)
            except Exception as exc:
                warnings.warn(
                    f"Effective-charge file {p} exists but could not be parsed "
                    f"({exc}); skipping it."
                )
                continue
    return None
