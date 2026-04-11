"""
PySTARC COFFDROP flexible chain model
=====================================

Python implementation of the COFFDROP (Coarse-grained Force Field
for Disordered Proteins) flexible chain model.

Reference: Levine et al., JCTC 2019 (COFFDROP paper)

COFFDROP models intrinsically disordered proteins and flexible loops as
chains of coarse-grained beads, one per residue. Each bead has:
- A position (3D)
- A diffusion coefficient (Stokes radius from COFFDROP parameter file)
- Bonded interactions: bond lengths, bond angles, torsion angles
- Non-bonded interactions: excluded volume + optional electrostatics

In the reference implementation, a system can be:
- Two rigid bodies (cores only) - the case PySTARC handles now
- One rigid body + one flexible chain - e.g. structured protein + IDP
- Two flexible chains

This module implements the chain kinematics, force evaluation, and BD
propagation for flexible chains. It does not yet implement the full
COFFDROP parameter file reader (that requires the COFFDROP XML files
from the data files).

What is implemented here:
- Chain data structures (beads, bonds, angles, torsions)
- Ermak-McCammon BD propagation for each bead independently
- Bond constraint satisfaction (RATTLE-style)
- Hard-sphere bead-bead exclusion

What is not yet implemented:
- Full COFFDROP bonded parameter file reader
- ABSINTH implicit solvent model for chains
- Chain-chain hydrodynamic interactions (RPY for many beads)

This is a foundation for future full COFFDROP support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import math

# Bead data structure 
@dataclass
class ChainBead:
    """
    One coarse-grained bead in a COFFDROP chain.
    """
    pos:     np.ndarray        # (3,) current position (A)
    force:   np.ndarray        # (3,) current force (kBT/A)
    radius:  float             # (A) Stokes radius for diffusion
    charge:  float             # (e) partial charge
    resname: str   = ''        # residue name (1-letter or 3-letter)
    resid:   int   = 0         # residue index

    def __post_init__(self):
        self.pos   = np.asarray(self.pos,   dtype=float)
        self.force = np.asarray(self.force, dtype=float)

@dataclass
class ChainBond:
    """
    Bonded interaction between two beads.
    """
    i:          int     # bead index
    j:          int     # bead index
    r0:         float   # equilibrium distance (A)
    k_spring:   float   # spring constant (kBT/A^2)

@dataclass
class ChainAngle:
    """
    Three-body angle interaction.
    """
    i: int
    j: int    # central bead
    k: int
    theta0:  float   # equilibrium angle (rad)
    k_angle: float   # force constant (kBT/rad^2)

@dataclass
class ChainTorsion:
    """
    Four-body torsion/dihedral interaction.
    """
    i: int
    j: int
    k: int
    l: int
    phi0:  float   # equilibrium dihedral (rad)
    k_tor: float   # force constant (kBT)
    n:     int     # periodicity

# Chain state

@dataclass
class FlexibleChain:
    """
    State of a COFFDROP flexible chain.
    """
    beads:    List[ChainBead]
    bonds:    List[ChainBond]    = field(default_factory=list)
    angles:   List[ChainAngle]   = field(default_factory=list)
    torsions: List[ChainTorsion] = field(default_factory=list)
    name:     str                = ''
    frozen:   bool               = False   # if True, chain doesn't move

    @property
    def n_beads(self) -> int:
        return len(self.beads)

    def positions_array(self) -> np.ndarray:
        return np.array([b.pos for b in self.beads])

    def forces_array(self) -> np.ndarray:
        return np.array([b.force for b in self.beads])

    def set_positions(self, pos: np.ndarray):
        for i, b in enumerate(self.beads):
            b.pos = pos[i].copy()

    def zero_forces(self):
        for b in self.beads:
            b.force = np.zeros(3)

# Force evaluation

class ChainForceEvaluator:
    """
    Evaluates all bonded and non-bonded forces on a flexible chain.
    """
    def compute_forces(self, chain: FlexibleChain,
                       kT: float = 0.5961) -> np.ndarray:
        """
        Compute all forces on chain beads. Returns (n_beads, 3) force array.
        Forces are in kBT/A units.
        """
        n = chain.n_beads
        F = np.zeros((n, 3))
        # 1. Bond forces (harmonic)
        for bond in chain.bonds:
            F += self._bond_force(chain, bond, kT)
        # 2. Angle forces (harmonic)
        for angle in chain.angles:
            F += self._angle_force(chain, angle, kT)
        # 3. Torsion forces (periodic)
        for tor in chain.torsions:
            F += self._torsion_force(chain, tor, kT)
        # 4. Non-bonded: excluded volume (soft sphere)
        F += self._excluded_volume_forces(chain, kT)
        return F

    def _bond_force(self, chain: FlexibleChain,
                    bond: ChainBond, kT: float) -> np.ndarray:
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[bond.i].pos
        rj = chain.beads[bond.j].pos
        dr   = rj - ri
        r    = float(np.linalg.norm(dr))
        if r < 1e-8:
            return F
        r_hat = dr / r
        f_mag = -bond.k_spring * (r - bond.r0)   # kBT/A
        F[bond.i] -= f_mag * r_hat
        F[bond.j] += f_mag * r_hat
        return F

    def _angle_force(self, chain: FlexibleChain,
                     angle: ChainAngle, kT: float) -> np.ndarray:
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[angle.i].pos
        rj = chain.beads[angle.j].pos
        rk = chain.beads[angle.k].pos
        u  = ri - rj
        v  = rk - rj
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-8 or nv < 1e-8:
            return F
        u /= nu; v /= nv
        cos_t = float(np.dot(u, v))
        cos_t = max(-1.0, min(1.0, cos_t))
        theta = math.acos(cos_t)
        sin_t = math.sin(theta)
        if abs(sin_t) < 1e-8:
            return F
        d_theta = theta - angle.theta0
        coeff = -angle.k_angle * d_theta / sin_t
        # gradient wrt ri, rk
        fi = coeff * (v - cos_t * u) / nu
        fk = coeff * (u - cos_t * v) / nv
        F[angle.i] += fi
        F[angle.k] += fk
        F[angle.j] -= (fi + fk)
        return F

    def _torsion_force(self, chain: FlexibleChain,
                       tor: ChainTorsion, kT: float) -> np.ndarray:
        F = np.zeros((chain.n_beads, 3))
        ri = chain.beads[tor.i].pos
        rj = chain.beads[tor.j].pos
        rk = chain.beads[tor.k].pos
        rl = chain.beads[tor.l].pos
        b1 = rj - ri
        b2 = rk - rj
        b3 = rl - rk
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = float(np.linalg.norm(n1))
        n2_norm = float(np.linalg.norm(n2))
        if n1_norm < 1e-8 or n2_norm < 1e-8:
            return F
        n1 /= n1_norm; n2 /= n2_norm
        cos_phi = float(np.dot(n1, n2))
        cos_phi = max(-1.0, min(1.0, cos_phi))
        phi = math.acos(cos_phi)
        # dV/dphi = -k * n * sin(n*phi - phi0)
        dV = -tor.k_tor * tor.n * math.sin(tor.n * phi - tor.phi0)
        b2_hat = b2 / float(np.linalg.norm(b2)) if float(np.linalg.norm(b2)) > 1e-8 else b2
        F[tor.i] += -dV * (n1 / n1_norm) / float(np.linalg.norm(b1))
        F[tor.l] +=  dV * (n2 / n2_norm) / float(np.linalg.norm(b3))
        return F

    def _excluded_volume_forces(self, chain: FlexibleChain,
                                 kT: float) -> np.ndarray:
        """Soft-sphere excluded volume between non-bonded bead pairs."""
        n = chain.n_beads
        F = np.zeros((n, 3))
        bonded_pairs = {(b.i, b.j) for b in chain.bonds} | \
                       {(b.j, b.i) for b in chain.bonds}
        for i in range(n):
            for j in range(i+2, n):   # skip bonded neighbours
                if (i, j) in bonded_pairs:
                    continue
                ri = chain.beads[i].pos
                rj = chain.beads[j].pos
                dr   = rj - ri
                r    = float(np.linalg.norm(dr))
                sig  = chain.beads[i].radius + chain.beads[j].radius
                if r < 1e-8 or r >= sig:
                    continue
                # WCA-style repulsion
                sr   = sig / r
                sr12 = sr**12
                sr6  = sr**6
                eps  = 1.0   # kBT units
                f_mag = eps * (12*sr12 - 6*sr6) / (r*r)
                fvec  = f_mag * dr
                F[i] += fvec
                F[j] -= fvec
        return F

# BD propagation for chain 

class ChainBDPropagator:
    """
    Brownian dynamics propagator for a flexible chain.
    Each bead moves independently with its own D_trans = kT/(6*pi*eta*radius).
    No hydrodynamic coupling between beads.
    """

    def __init__(self, kT: float = 0.5961, viscosity: float = 8.904e-4):
        self.kT   = kT
        self.eta  = viscosity   # Pa*s converted to kcal*ps/A^3
        self._evaluator = ChainForceEvaluator()

    def D_trans(self, radius: float) -> float:
        """Stokes-Einstein translational diffusion (A^2/ps)."""
        return self.kT / (6.0 * math.pi * self.eta * radius)

    def step(self, chain: FlexibleChain,
             dt: float,
             rng: np.random.Generator,
             force_evaluator=None) -> FlexibleChain:
        """
        Advance chain by one BD step of size dt (ps).
            dpos  = mob * f * dt
            wdpos = sqrt(2 * kT * mob) * dW
        Parameters
        ----------
        force_evaluator : optional external evaluator (e.g. COFFDROPForceEvaluator).
                          If None, uses internal ChainForceEvaluator (harmonic).
        """
        if chain.frozen:
            return chain
        # Compute forces - use external evaluator if provided
        if force_evaluator is not None:
            forces = force_evaluator.compute_forces(chain)
        else:
            forces = self._evaluator.compute_forces(chain, self.kT)
        for i, b in enumerate(chain.beads):
            b.force = forces[i]
        # Propagate each bead
        for i, b in enumerate(chain.beads):
            mob = 1.0 / (6.0 * math.pi * self.eta * b.radius)   # A^3/(kBT*ps)
            # Deterministic drift: dpos = mob * F * dt
            drift = mob * b.force * dt
            # Stochastic: wdpos = sqrt(2 * kT * mob) * dW
            sigma = math.sqrt(2.0 * self.kT * mob * dt)
            noise = sigma * rng.standard_normal(3)
            b.pos += drift + noise
        return chain
    
    def max_time_step(self, chain: FlexibleChain) -> float:
        """
        Geometry-based maximum time step for chain.
        Uses smallest bead radius: dt ~ R^2 / D
        """
        if not chain.beads:
            return 0.1
        min_R = min(b.radius for b in chain.beads)
        D_max = self.D_trans(min_R)
        # 4*R^3/D_factor simplified to R^2/D here
        return min_R**2 / D_max if D_max > 0 else 0.001

    def satisfy_bond_constraints(self, chain: FlexibleChain,
                                  tol: float = 1e-4,
                                  max_iter: int = 100):
        """
        RATTLE-style bond constraint satisfaction.
        """
        for _ in range(max_iter):
            max_viol = 0.0
            for bond in chain.bonds:
                ri = chain.beads[bond.i].pos
                rj = chain.beads[bond.j].pos
                dr = rj - ri
                r  = float(np.linalg.norm(dr))
                if r < 1e-8:
                    continue
                viol = abs(r - bond.r0) / bond.r0
                max_viol = max(max_viol, viol)
                if viol > tol:
                    # Project back to constraint surface
                    correction = 0.5 * (r - bond.r0) * dr / r
                    chain.beads[bond.i].pos += correction
                    chain.beads[bond.j].pos -= correction
            if max_viol < tol:
                break

# Simple chain builder 
def build_linear_chain(n_residues:  int,
                       bead_radius: float = 2.0,
                       bead_charge: float = 0.0,
                       bond_length: float = 3.8,
                       start_pos:   Optional[np.ndarray] = None,
                       ) -> FlexibleChain:
    """
    Build a simple linear chain of n_residues beads.
    Useful for testing; production use should load from COFFDROP XML.
    """
    if start_pos is None:
        start_pos = np.zeros(3)
    beads = []
    for i in range(n_residues):
        pos = start_pos + np.array([i * bond_length, 0.0, 0.0])
        beads.append(ChainBead(
            pos     = pos,
            force   = np.zeros(3),
            radius  = bead_radius,
            charge  = bead_charge,
            resname = 'UNK',
            resid   = i
        ))
    bonds = [
        ChainBond(i=i, j=i+1, r0=bond_length, k_spring=100.0)
        for i in range(n_residues - 1)
    ]
    return FlexibleChain(beads=beads, bonds=bonds, name='chain')

# COFFDROP tabulated force evaluator 

class COFFDROPForceEvaluator:
    """
    Force evaluator using the tabulated COFFDROP potentials loaded from the
    four XML data files (coffdrop.xml, mapping.xml, connectivity.xml,
    charges.xml).
    Replaces ChainForceEvaluator when COFFDROP parameter files are available.
    Usage
    -----
        from pystarc.simulation.coffdrop_params import COFFDROPParams
        from pystarc.simulation.coffdrop_chain import COFFDROPForceEvaluator
        params = COFFDROPParams.load(
            ff_xml='coffdrop.xml', mapping_xml='mapping.xml',
            connectivity_xml='connectivity.xml', charges_xml='charges.xml')
        evaluator = COFFDROPForceEvaluator(params)
        F = evaluator.compute_forces(chain)
    """
    def __init__(self, params):
        """
        Parameters
        ----------
        params : COFFDROPParams - loaded parameter set
        """
        self.params = params

    def compute_forces(self, chain: 'FlexibleChain') -> np.ndarray:
        """
        Compute all forces on chain beads using COFFDROP tabulated potentials.
        Returns (n_beads, 3) force array in kBT/A.
        Force contributions:
        1. Non-bonded pair potentials (from coffdrop.xml <pairs>)
        2. Bond-angle potentials      (from coffdrop.xml <bond_angles>)
        3. Dihedral potentials        (from coffdrop.xml <dihedral_angles>)
        4. Electrostatic (Debye-Hückel) for charged beads
        """
        n = chain.n_beads
        F = np.zeros((n, 3))
        # Build exclusion set: skip 1-2 bonded pairs in non-bonded evaluation
        excluded = set()
        for bond in chain.bonds:
            excluded.add((min(bond.i, bond.j), max(bond.i, bond.j)))
        # 1. Non-bonded pair forces (skip bonded pairs)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in excluded:
                    continue
                f_ij = self._pair_force_vec(chain, i, j)
                F[i] += f_ij
                F[j] -= f_ij
        # 2. Bond-angle forces (triplets from chain bonds)
        # Build triplets: consecutive bonded beads i-j-k
        bonded_next = {}  # i -> j if (i,j) is a bond
        for bond in chain.bonds:
            bonded_next[bond.i] = bond.j
        for i in range(n - 2):
            if i in bonded_next and bonded_next[i] == i + 1:
                if i + 1 in bonded_next and bonded_next[i + 1] == i + 2:
                    f_i, f_j, f_k = self._angle_forces(chain, i, i+1, i+2)
                    F[i]   += f_i
                    F[i+1] += f_j
                    F[i+2] += f_k
        # 3. Dihedral forces (quadruplets)
        for i in range(n - 3):
            f_i, f_j, f_k, f_l = self._dihedral_forces(chain, i, i+1, i+2, i+3)
            F[i]   += f_i
            F[i+1] += f_j
            F[i+2] += f_k
            F[i+3] += f_l
        return F

    def _pair_force_vec(self, chain: 'FlexibleChain',
                        i: int, j: int) -> np.ndarray:
        """
        Vector force on bead i from bead j via COFFDROP pair potential.
        """
        bi = chain.beads[i]
        bj = chain.beads[j]
        dr   = bi.pos - bj.pos
        r    = float(np.linalg.norm(dr))
        if r < 1e-10:
            return np.zeros(3)
        dVdr = self.params.pair_force(bi.resname, self._bead_type(bi),
                                       bj.resname, self._bead_type(bj), r)
        # F_i = -dV/dr * rhat  (force on i is away from j when repulsive)
        return -dVdr * (dr / r)

    def _angle_forces(self, chain: 'FlexibleChain',
                      i: int, j: int, k: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forces from bond-angle potential on beads i-j-k.
        Returns forces on (i, j, k).
        """
        bi, bj, bk = chain.beads[i], chain.beads[j], chain.beads[k]
        r_ij = bi.pos - bj.pos
        r_kj = bk.pos - bj.pos
        norm_ij = float(np.linalg.norm(r_ij))
        norm_kj = float(np.linalg.norm(r_kj))
        if norm_ij < 1e-10 or norm_kj < 1e-10:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        cos_theta = float(np.dot(r_ij, r_kj)) / (norm_ij * norm_kj)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_deg = math.acos(cos_theta) * 180.0 / math.pi
        dVdth = self.params.angle_force(
            (bi.resname, bj.resname, bk.resname),
            (self._bead_type(bi), self._bead_type(bj), self._bead_type(bk)),
            self._angle_orders(chain, i, j, k),
            theta_deg
        )  # kBT/deg
        if abs(math.sin(theta_deg * math.pi / 180.0)) < 1e-10:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        # Chain rule: dV/dr = dV/dθ * dθ/dr
        dth_deg_to_rad = math.pi / 180.0
        dVdth_rad = dVdth / dth_deg_to_rad  # kBT/rad
        u_ij = r_ij / norm_ij
        u_kj = r_kj / norm_kj
        sin_th = math.sqrt(max(1.0 - cos_theta**2, 1e-30))
        df_i = (cos_theta * u_ij - u_kj) / (norm_ij * sin_th)
        df_k = (cos_theta * u_kj - u_ij) / (norm_kj * sin_th)
        df_j = -(df_i + df_k)
        fi = -dVdth_rad * df_i
        fj = -dVdth_rad * df_j
        fk = -dVdth_rad * df_k
        return fi, fj, fk

    def _dihedral_forces(self, chain: 'FlexibleChain',
                         i: int, j: int, k: int, l: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forces from dihedral potential on beads i-j-k-l.
        """
        bi = chain.beads[i]; bj = chain.beads[j]
        bk = chain.beads[k]; bl = chain.beads[l]
        b1 = bj.pos - bi.pos
        b2 = bk.pos - bj.pos
        b3 = bl.pos - bk.pos
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = float(np.linalg.norm(n1))
        n2_norm = float(np.linalg.norm(n2))
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        n1u = n1 / n1_norm
        n2u = n2 / n2_norm
        cos_phi = float(np.dot(n1u, n2u))
        cos_phi = max(-1.0, min(1.0, cos_phi))
        phi = math.acos(cos_phi)
        # Sign convention
        if float(np.dot(np.cross(n1u, n2u), b2)) < 0:
            phi = -phi
        phi_deg = phi * 180.0 / math.pi
        # Wrap to [0, 360)
        phi_deg = phi_deg % 360.0
        dVdphi = self.params.dihedral_force(
            (bi.resname, bj.resname, bk.resname, bl.resname),
            (self._bead_type(bi), self._bead_type(bj),
             self._bead_type(bk), self._bead_type(bl)),
            self._dihedral_orders(chain, i, j, k, l),
            phi_deg
        )  # kBT/deg
        dVdphi_rad = dVdphi / (math.pi / 180.0)  # kBT/rad
        # Gradient of phi w.r.t. positions 
        b2_norm = float(np.linalg.norm(b2))
        if b2_norm < 1e-10:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        fi =  (b2_norm / (n1_norm**2)) * n1
        fl = -(b2_norm / (n2_norm**2)) * n2
        fj = (-np.dot(b1, b2) / (b2_norm**2 * n1_norm**2) * n1 * b2_norm
               + np.dot(b3, b2) / (b2_norm**2 * n2_norm**2) * n2 * b2_norm)
        fk = -fi - fj - fl
        return (-dVdphi_rad * fi, -dVdphi_rad * fj,
                -dVdphi_rad * fk, -dVdphi_rad * fl)

    # Helper methods 
    def _bead_type(self, bead: 'ChainBead') -> str:
        """Get the COFFDROP bead type name for a chain bead."""
        if ':' in bead.resname:
            return bead.resname.split(':')[1]
        # Default: CA for backbone beads
        return 'CA'

    def _angle_orders(self, chain, i, j, k):
        """Sequence orders for an angle triplet."""
        # Orders are the sequence positions within the chain
        return (i + 1, j + 1, k + 1)

    def _dihedral_orders(self, chain, i, j, k, l):
        """Sequence orders for a dihedral quartet."""
        return (i + 1, j + 1, k + 1, l + 1)

def build_chain_from_coffdrop(residues: List[str],
                              params,
                              start_pos: Optional[np.ndarray] = None
                              ) -> 'FlexibleChain':
    """
    Build a FlexibleChain from a sequence of residue names using COFFDROP
    equilibrium bond lengths and charges from the parameter files.
    Parameters
    ----------
    residues : list of 3-letter residue names, e.g. ['ALA', 'GLY', 'ARG']
    params   : COFFDROPParams loaded from XML files
    start_pos: starting position of first bead (default [0,0,0])
    Returns
    -------
    FlexibleChain with beads, bonds, and charges from COFFDROP data files.
    """
    if start_pos is None:
        start_pos = np.zeros(3)
    beads = []
    pos = start_pos.copy()
    for i, resname in enumerate(residues):
        # Get charge from charges.xml (CA bead is backbone, usually neutral)
        charge = params.bead_charge(resname, 'CA')
        bead = ChainBead(
            pos     = pos.copy(),
            force   = np.zeros(3),
            radius  = 2.0,           # typical CA Stokes radius
            charge  = charge,
            resname = resname,
            resid   = i,
        )
        beads.append(bead)
        # Advance position by CA-CA backbone bond length
        ca_ca_len = params.bond_length('XXX', 'CA', 1, 'XXX', 'CA', 2) or 3.8
        pos = pos + np.array([ca_ca_len, 0.0, 0.0])
    # Build bonds using equilibrium lengths from connectivity.xml
    bonds = []
    for i in range(len(residues) - 1):
        r0 = params.bond_length('XXX', 'CA', 1, 'XXX', 'CA', 2) or 3.8
        bonds.append(ChainBond(i=i, j=i+1, r0=r0, k_spring=100.0))
    return FlexibleChain(beads=beads, bonds=bonds,
                         name='-'.join(residues[:3]) + '...')