"""
Lennard-Jones forces for PySTARC.

The pair potential and the radial force it produces are

    V = eps * ((sig/r)^12 - (sig/r)^6)
    F = eps * (12*(sig/r)^12 - 6*(sig/r)^6) / r^2 * r_vec

Here r is the centre-to-centre distance, eps is the well depth, and sig sets
the length scale of the interaction.

Parameters for a pair of unlike atoms are obtained from the per-type values by
the standard mixing rules

    eps_ij = sqrt(eps_i * eps_j)
    sig_ij = sig_i + sig_j

An optional Weeks-Chandler-Andersen (WCA) treatment keeps only the repulsive
branch of the potential by cutting it off at r = 2^(1/6) * sig and shifting the
energy up by the well depth, so that it is purely repulsive and vanishes at the
cutoff.

A hydrophobic force based on solvent-accessible surface area (SASA) can also be
applied. It acts only over a contact shell, contributing

    F_hydrophob = factor * area  when  a <= r + radius <= b

where factor = beta * c / (b - a). The default shell and coefficients are
a = 3.1 angstrom, b = 4.35 angstrom, c = 0.5, and beta = -0.025 kcal/mol/angstrom^2.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math

# Lennard-Jones parameters for a single atom type.


@dataclass
class LJAtomType:
    """Lennard-Jones parameters for one atom type."""

    name: str
    epsilon: float  # well depth in kcal/mol
    sigma: float  # length scale in angstrom (this is a radius, not a diameter)


@dataclass
class LJParams:
    """
    Lennard-Jones parameters for the whole system.

    The field atom_types holds one LJAtomType per atom type. The field
    one_four_factor scales 1-4 interactions and defaults to 0.5. When use_wca
    is True the potential is reduced to its purely repulsive WCA form.
    """

    atom_types: List[LJAtomType] = field(default_factory=list)
    one_four_factor: float = 0.5
    use_wca: bool = False

    def epsilon(self, type_idx: int) -> float:
        return self.atom_types[type_idx].epsilon

    def sigma(self, type_idx: int) -> float:
        return self.atom_types[type_idx].sigma


# Parameters for the SASA-based hydrophobic interaction.
@dataclass
class HydrophobicParams:
    """Parameters for the hydrophobic interaction based on solvent-accessible surface area."""

    a: float = 3.1  # inner edge of the contact shell in angstrom
    b: float = 4.35  # outer edge of the contact shell in angstrom
    c: float = 0.5  # dimensionless scaling coefficient
    beta: float = (
        -0.025
    )  # surface tension in kcal/mol/angstrom^2; a negative value is attractive

    @property
    def factor(self) -> float:
        """Return beta * c / (b - a), in kcal/mol/angstrom^3."""
        return self.beta * self.c / (self.b - self.a)


# The core Lennard-Jones pair force.
def lj_pair_force(
    pos_a: np.ndarray,  # position of atom a, shape (3,), in angstrom
    pos_b: np.ndarray,  # position of atom b, shape (3,), in angstrom
    epsilon: float,  # well depth in kcal/mol
    sigma: float,  # length scale in angstrom
    factor: float = 1.0,  # overall scaling, for example the 1-4 factor
    use_wca: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Return the Lennard-Jones force on atom a due to atom b together with the
    interaction energy. The potential and the magnitude of the radial force are

        V   = factor * eps * ((sig/r)^12 - (sig/r)^6)
        F_r = factor * eps * (12*(sig/r)^12 - 6*(sig/r)^6) / r^2

    where r is the distance between the two atoms.

    The return value is a tuple (force_on_a, energy). The force is the (3,)
    vector acting on atom a due to atom b, following the usual convention
    F_a = -grad_a V(|b-a|). For r < sigma the force points along (a - b)/r and
    is repulsive, pushing a away from b. For sigma < r < about 2*sigma it points
    along (b - a)/r and is attractive, pulling a toward the Lennard-Jones
    minimum. Audit C7 corrected an earlier inverted sign. The magnitude had
    always been right and only the direction was wrong. The fix is verified by
    the hand-computed dimer test in
    tests/test_pystarc.py::test_lj_force_direction_*.
    """
    dpos = pos_b - pos_a
    r2 = float(np.dot(dpos, dpos))
    if r2 < 1e-6:
        return np.zeros(3), 0.0
    r = math.sqrt(r2)
    sr = sigma / r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    # In the WCA treatment we keep only the repulsive branch and cut the
    # interaction off at r = 2^(1/6) * sigma, the location of the potential minimum.
    if use_wca:
        r_cut = 2.0 ** (1.0 / 6.0) * sigma
        if r > r_cut:
            return np.zeros(3), 0.0
        # Shift the energy up by the well depth so that the potential is purely
        # repulsive, vanishing at the cutoff r = 2^(1/6) * sigma where
        # (sig/r)^6 = 1/2 and (sig/r)^12 = 1/4, giving V(r_cut) = -factor*eps/4.
        energy = factor * epsilon * (sr12 - sr6) + factor * epsilon * 0.25
    else:
        energy = factor * epsilon * (sr12 - sr6)
    # The radial force magnitude is -dV/dr projected onto the unit separation,
    # which works out to eps*(12*sr12 - 6*sr6)/r^2.
    f_mag = factor * epsilon * (12.0 * sr12 - 6.0 * sr6) / r2
    # The force on a is F_a = -grad_a V(|b-a|) = V'(r) * (b - a)/r. Since
    # V'(r) = -f_mag * r, this reduces to F_a = -f_mag * dpos. This is the Audit C7 fix.
    force_a = -f_mag * dpos
    return force_a, energy


# The hydrophobic force from solvent-accessible surface area.
def hydrophobic_sasa_force(
    r: float,  # centre-to-centre distance in angstrom
    r_vec: np.ndarray,  # unit vector from a to b, shape (3,)
    radius_a: float,  # van der Waals radius of atom a in angstrom
    radius_b: float,  # van der Waals radius of atom b in angstrom
    sasa_a: float,  # solvent-accessible surface area of atom a in angstrom^2
    sasa_b: float,  # solvent-accessible surface area of atom b in angstrom^2
    hp: HydrophobicParams = HydrophobicParams(),
) -> Tuple[np.ndarray, float]:
    """Return the hydrophobic force based on solvent-accessible surface area."""
    fac = hp.factor  # in kcal/mol/angstrom^3; a negative value is attractive

    def sasa_contrib(radius_self: float, area_other: float) -> float:
        ri = r + radius_self
        if hp.a <= ri <= hp.b:
            return fac * area_other
        return 0.0

    f_scalar = sasa_contrib(radius_a, sasa_b) + sasa_contrib(radius_b, sasa_a)
    # This uses the same sign convention as lj_pair_force, namely F_a = -grad_a V.
    # With the attractive default fac < 0 the scalar f_scalar is negative, so
    # force_a = -f_scalar * r_vec points along +r_vec toward b and is attractive.
    # This is the Audit C7b fix.
    force_a = -f_scalar * r_vec
    # Approximate the energy as a trapezoidal integral of the force across the contact shell.
    energy = f_scalar * (hp.b - hp.a) * 0.5
    return force_a, energy


# The full pairwise engine for Lennard-Jones and hydrophobic forces.


class LJForceEngine:
    """
    Compute all pairwise Lennard-Jones forces, and optionally the hydrophobic
    forces, between two molecules. A typical use is to build the engine from
    its parameter sets and then call compute on a pair of molecules,

        engine = LJForceEngine(lj_params, hydrophobic_params)
        total_force, total_energy = engine.compute(mol1, mol2)
    """

    def __init__(
        self,
        lj_params: Optional[LJParams] = None,
        hp_params: Optional[HydrophobicParams] = None,
    ):
        self.lj = lj_params
        self.hp = hp_params

    def compute(
        self,
        positions1: np.ndarray,  # atom positions of molecule 1, shape (N1, 3), in angstrom
        positions2: np.ndarray,  # atom positions of molecule 2, shape (N2, 3), in angstrom
        type_ids1: List[int],  # Lennard-Jones type index for each atom in molecule 1
        type_ids2: List[int],  # Lennard-Jones type index for each atom in molecule 2
        radii1: Optional[
            np.ndarray
        ] = None,  # van der Waals radii for molecule 1, shape (N1,), in angstrom, used for SASA
        radii2: Optional[
            np.ndarray
        ] = None,  # van der Waals radii for molecule 2, shape (N2,), in angstrom, used for SASA
        sasa1: Optional[
            np.ndarray
        ] = None,  # per-atom SASA for molecule 1, shape (N1,), in angstrom^2
        sasa2: Optional[
            np.ndarray
        ] = None,  # per-atom SASA for molecule 2, shape (N2,), in angstrom^2
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the total Lennard-Jones and hydrophobic forces on both molecules.

        The return value is a tuple (force_on_mol1, force_on_mol2, total_energy).
        Each force is the (3,) net force vector acting on the centroid of that
        molecule.
        """
        N1, N2 = len(positions1), len(positions2)
        f1 = np.zeros(3)
        f2 = np.zeros(3)
        E = 0.0
        for i in range(N1):
            for j in range(N2):
                pos_i = positions1[i]
                pos_j = positions2[j]
                dpos = pos_j - pos_i
                r = float(np.linalg.norm(dpos))
                if r < 1e-6:
                    continue
                r_hat = dpos / r
                # Add the Lennard-Jones contribution for this pair.
                if self.lj is not None:
                    ti = type_ids1[i]
                    tj = type_ids2[j]
                    # Combine the per-type parameters with the mixing rules used by
                    # the reference C++ implementation.
                    eps_ij = math.sqrt(self.lj.epsilon(ti) * self.lj.epsilon(tj))
                    sig_ij = self.lj.sigma(ti) + self.lj.sigma(tj)
                    f_lj, e_lj = lj_pair_force(
                        pos_i, pos_j, eps_ij, sig_ij, use_wca=self.lj.use_wca
                    )
                    f1 += f_lj
                    f2 -= f_lj
                    E += e_lj
                # Add the hydrophobic SASA contribution for this pair.
                if (
                    self.hp is not None
                    and radii1 is not None
                    and radii2 is not None
                    and sasa1 is not None
                    and sasa2 is not None
                ):
                    f_hp, e_hp = hydrophobic_sasa_force(
                        r,
                        r_hat,
                        float(radii1[i]),
                        float(radii2[j]),
                        float(sasa1[i]),
                        float(sasa2[j]),
                        self.hp,
                    )
                    f1 += f_hp
                    f2 -= f_hp
                    E += e_hp
        return f1, f2, E
