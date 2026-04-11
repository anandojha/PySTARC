"""
Auxiliary preprocessing tools for PySTARC.

"""
from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule, BoundingBox
from pystarc.global_defs.constants import BJERRUM_LENGTH
from typing import List, Tuple, Optional, Dict
from pystarc.global_defs.constants import PI
import numpy as np
import math

# bounding_box 
def bounding_box(mol: Molecule, padding: float = 5.0) -> BoundingBox:
    """
    Compute axis-aligned bounding box of a molecule with optional padding.
    """
    return BoundingBox.from_molecule(mol, padding=padding)

# surface_spheres
def surface_spheres(mol: Molecule,
                    probe_radius: float = 1.4,
                    n_points: int = 92) -> List[np.ndarray]:
    """
    Generate surface probe positions using a Fibonacci sphere around each atom.
    Returns list of (x,y,z) probe positions on the molecular surface.
    """
    positions = []
    golden = (1 + math.sqrt(5)) / 2
    for atom in mol.atoms:
        r = atom.radius + probe_radius
        c = atom.position
        for i in range(n_points):
            theta = math.acos(1 - 2*(i+0.5)/n_points)
            phi   = 2*PI * i / golden
            x = c[0] + r * math.sin(theta) * math.cos(phi)
            y = c[1] + r * math.sin(theta) * math.sin(phi)
            z = c[2] + r * math.cos(theta)
            # Check not inside any other atom
            p = np.array([x, y, z])
            buried = any(
                np.linalg.norm(p - a.position) < a.radius + probe_radius
                for a in mol.atoms if a is not atom
            )
            if not buried:
                positions.append(p)
    return positions

# lumped_charges
def lumped_charges(mol: Molecule,
                   grid_spacing: float = 2.0) -> List[Tuple[np.ndarray, float]]:
    """
    Coarse-grain atomic charges onto a regular grid by nearest-grid-point.
    Returns list of (position, charge) tuples for non-zero grid points.
    """
    if not mol.atoms:
        return []
    bb = bounding_box(mol, padding=grid_spacing)
    # Build grid
    xs = np.arange(bb.xmin, bb.xmax + grid_spacing, grid_spacing)
    ys = np.arange(bb.ymin, bb.ymax + grid_spacing, grid_spacing)
    zs = np.arange(bb.zmin, bb.zmax + grid_spacing, grid_spacing)
    grid: Dict[Tuple[int,int,int], float] = {}
    for atom in mol.atoms:
        if atom.charge == 0.0:
            continue
        ix = int(round((atom.x - bb.xmin) / grid_spacing))
        iy = int(round((atom.y - bb.ymin) / grid_spacing))
        iz = int(round((atom.z - bb.zmin) / grid_spacing))
        key = (ix, iy, iz)
        grid[key] = grid.get(key, 0.0) + atom.charge
    result = []
    for (ix, iy, iz), q in grid.items():
        if abs(q) > 1e-8:
            pos = np.array([
                bb.xmin + ix * grid_spacing,
                bb.ymin + iy * grid_spacing,
                bb.zmin + iz * grid_spacing,
            ])
            result.append((pos, q))
    return result

# electrostatic_center
def electrostatic_center(mol: Molecule) -> np.ndarray:
    """
    Charge-weighted center of a molecule.
    Falls back to geometric centroid if total charge is zero.
    """
    total_q = sum(abs(a.charge) for a in mol.atoms)
    if total_q < 1e-10:
        return mol.centroid()
    pos = mol.positions_array()
    charges = np.abs(mol.charges_array())
    return (pos * charges[:, None]).sum(axis=0) / total_q

# hydrodynamic_radius
def hydrodynamic_radius_from_rg(mol: Molecule) -> float:
    """
    Approximate hydrodynamic radius from radius of gyration.
    r_h ≈ 0.77 × r_g  (empirical for globular proteins).
    """
    return 0.77 * mol.radius_of_gyration()

def hydrodynamic_radius_from_surface(mol: Molecule) -> float:
    """
    Approximate hydrodynamic radius as bounding radius of the molecule.
    """
    return mol.bounding_radius()

# contact_distances 
def contact_distances(mol1: Molecule,
                      mol2: Molecule,
                      cutoff: float = 8.0) -> List[Tuple[int, int, float]]:
    """
    Return all atom pairs (i, j, dist) within cutoff Å.
    Used to auto-generate reaction contacts.
    """
    pairs = []
    for i, a1 in enumerate(mol1.atoms):
        for j, a2 in enumerate(mol2.atoms):
            d = a1.distance_to(a2)
            if d <= cutoff:
                pairs.append((i, j, d))
    pairs.sort(key=lambda t: t[2])
    return pairs

# born_integral
def born_integral(charge: float, radius: float,
                  eps_in: float = 4.0,
                  eps_out: float = 78.54) -> float:
    """
    Born solvation energy of a sphere:
    ΔG_Born = -(q²/8π ε₀) × (1/ε_in - 1/ε_out) / r   [kBT]
    Returns energy in kBT (using Bjerrum length scale).
    """
    if radius < 1e-10:
        return 0.0
    return -(charge**2 * BJERRUM_LENGTH / (2 * radius)) * (1.0/eps_in - 1.0/eps_out)