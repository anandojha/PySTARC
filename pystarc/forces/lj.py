"""
PySTARC Lennard-Jones forces
==========================

Force formula:
    V = eps * ((sig/r)^12 - (sig/r)^6)
    F = eps * (12*(sig/r)^12 - 6*(sig/r)^6) / r^2 * r_vec

Mixing rules:
    eps_ij = sqrt(eps_i * eps_j)
    sig_ij = sig_i + sig_j

Optional WCA (Weeks-Chandler-Andersen) cutoff at r = 2^(1/6) * sig.

Hydrophobic SASA force:
    F_hydrophob = factor * area  when  a <= r+radius <= b
    where factor = beta * c / (b - a)
    Defaults: a=3.1 A, b=4.35 A, c=0.5, beta=-0.025 kcal/mol/A^2
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math

# LJ atom type parameters

@dataclass
class LJAtomType:
    """Per-atom-type LJ parameters."""
    name:    str
    epsilon: float   # kcal/mol
    sigma:   float   # A (radius, NOT diameter)

@dataclass
class LJParams:
    """
    System-level LJ parameters.
    atom_types: list of LJAtomType
    one_four_factor: scaling for 1-4 interactions (default 0.5)
    use_wca: if True, use WCA (purely repulsive) cutoff
    """
    atom_types:      List[LJAtomType] = field(default_factory=list)
    one_four_factor: float = 0.5
    use_wca:         bool  = False

    def epsilon(self, type_idx: int) -> float:
        return self.atom_types[type_idx].epsilon

    def sigma(self, type_idx: int) -> float:
        return self.atom_types[type_idx].sigma

# Hydrophobic SASA parameters
@dataclass
class HydrophobicParams:
    """
    SASA-based hydrophobic interaction parameters.
    """
    a:    float = 3.1      # A - inner cutoff
    b:    float = 4.35     # A - outer cutoff
    c:    float = 0.5      # dimensionless
    beta: float = -0.025   # kcal/mol/A^2 (negative = attractive)
    @property
    def factor(self) -> float:
        """beta * c / (b - a)  [kcal/mol/A^3]"""
        return self.beta * self.c / (self.b - self.a)

# Core LJ force function 
def lj_pair_force(
        pos_a:   np.ndarray,   # (3,) A
        pos_b:   np.ndarray,   # (3,) A
        epsilon: float,        # kcal/mol
        sigma:   float,        # A
        factor:  float = 1.0,  # scaling (e.g. 1-4 factor)
        use_wca: bool  = False,
) -> Tuple[np.ndarray, float]:
    """
    LJ force on atom a due to atom b, and the interaction energy.
        V   = factor * eps * ((sig/r)^12 - (sig/r)^6)
        F_r = factor * eps * (12*(sig/r)^12 - 6*(sig/r)^6) / r^2
    Returns
    -------
    (force_on_a, energy)   force_on_a is (3,) pointing a -> b direction
    """
    dpos = pos_b - pos_a
    r2   = float(np.dot(dpos, dpos))
    if r2 < 1e-6:
        return np.zeros(3), 0.0
    r    = math.sqrt(r2)
    sr   = sigma / r
    sr2  = sr * sr
    sr6  = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    # WCA: only repulsive part, cutoff at r = 2^(1/6) * sigma
    if use_wca:
        r_cut = 2.0 ** (1.0/6.0) * sigma
        if r > r_cut:
            return np.zeros(3), 0.0
    energy  = factor * epsilon * (sr12 - sr6)
    # F = -dV/dr * r_hat, magnitude = eps*(12*sr12 - 6*sr6)/r^2
    f_mag   = factor * epsilon * (12.0 * sr12 - 6.0 * sr6) / r2
    force_a = f_mag * dpos   # force on a in direction a->b
    return force_a, energy

# Hydrophobic SASA force 
def hydrophobic_sasa_force(
        r:      float,        # centre-to-centre distance (A)
        r_vec:  np.ndarray,   # (3,) unit vector a->b
        radius_a: float,      # VdW radius of atom a (A)
        radius_b: float,      # VdW radius of atom b (A)
        sasa_a:   float,      # SASA of atom a (A^2)
        sasa_b:   float,      # SASA of atom b (A^2)
        hp:       HydrophobicParams = HydrophobicParams(),
) -> Tuple[np.ndarray, float]:
    """
    SASA-based hydrophobic force.
    """
    fac = hp.factor   # kcal/mol/A^3 (negative = attractive)
    
    def sasa_contrib(radius_self: float, area_other: float) -> float:
        ri = r + radius_self
        if hp.a <= ri <= hp.b:
            return fac * area_other
        return 0.0
    f_scalar = sasa_contrib(radius_a, sasa_b) + sasa_contrib(radius_b, sasa_a)
    # Force acts along the intermolecular axis
    force_a  = f_scalar * r_vec
    # Approximate energy (trapezoid integral of force over contact range)
    energy   = f_scalar * (hp.b - hp.a) * 0.5
    return force_a, energy

# Full pairwise LJ + hydrophobic force engine 

class LJForceEngine:
    """
    Computes all pairwise LJ (and optionally hydrophobic) forces
    between two molecules.
    Usage:
        engine = LJForceEngine(lj_params, hydrophobic_params)
        total_force, total_energy = engine.compute(mol1, mol2)
    """
    def __init__(self,
                 lj_params:   Optional[LJParams]          = None,
                 hp_params:   Optional[HydrophobicParams] = None):
        self.lj = lj_params
        self.hp = hp_params
        
    def compute(self,
                positions1:  np.ndarray,   # (N1, 3) A
                positions2:  np.ndarray,   # (N2, 3) A
                type_ids1:   List[int],    # LJ type index for each atom in mol1
                type_ids2:   List[int],    # LJ type index for each atom in mol2
                radii1:      Optional[np.ndarray] = None,   # (N1,) A for SASA
                radii2:      Optional[np.ndarray] = None,   # (N2,) A for SASA
                sasa1:       Optional[np.ndarray] = None,   # (N1,) A^2
                sasa2:       Optional[np.ndarray] = None,   # (N2,) A^2
                ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute total LJ + hydrophobic forces on both molecules.
        Returns
        -------
        (force_on_mol1, force_on_mol2, total_energy)
        Both forces are (3,) net force vectors on the molecular centroid.
        """
        N1, N2 = len(positions1), len(positions2)
        f1 = np.zeros(3)
        f2 = np.zeros(3)
        E  = 0.0
        for i in range(N1):
            for j in range(N2):
                pos_i = positions1[i]
                pos_j = positions2[j]
                dpos  = pos_j - pos_i
                r     = float(np.linalg.norm(dpos))
                if r < 1e-6:
                    continue
                r_hat = dpos / r
                # LJ contribution
                if self.lj is not None:
                    ti = type_ids1[i]
                    tj = type_ids2[j]
                    # Mixing rules (the reference C++ implementation)
                    eps_ij = math.sqrt(self.lj.epsilon(ti) * self.lj.epsilon(tj))
                    sig_ij = self.lj.sigma(ti) + self.lj.sigma(tj)
                    f_lj, e_lj = lj_pair_force(
                        pos_i, pos_j, eps_ij, sig_ij, use_wca=self.lj.use_wca)
                    f1 += f_lj
                    f2 -= f_lj
                    E  += e_lj
                # Hydrophobic SASA contribution
                if (self.hp is not None and
                        radii1 is not None and radii2 is not None and
                        sasa1  is not None and sasa2  is not None):
                    f_hp, e_hp = hydrophobic_sasa_force(
                        r, r_hat,
                        float(radii1[i]), float(radii2[j]),
                        float(sasa1[i]),  float(sasa2[j]),
                        self.hp)
                    f1 += f_hp
                    f2 -= f_hp
                    E  += e_hp
        return f1, f2, E