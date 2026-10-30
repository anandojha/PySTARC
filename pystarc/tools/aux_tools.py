"""
Auxiliary preprocessing tools for PySTARC.

This module collects small helper routines that prepare molecular input for a
Brownian-dynamics association-rate calculation. It builds bounding boxes,
samples molecular surfaces, coarse-grains atomic charges onto a grid, locates
charge-weighted centers, estimates hydrodynamic radii, finds inter-molecular
contact pairs, and evaluates the Born solvation energy of a charged sphere.
"""

from __future__ import annotations
from pystarc.structures.molecules import Atom, Molecule, BoundingBox
from pystarc.global_defs.constants import BJERRUM_LENGTH
from typing import List, Tuple, Optional, Dict
from pystarc.global_defs.constants import PI
import numpy as np
import math


def bounding_box(mol: Molecule, padding: float = 5.0) -> BoundingBox:
    """
    Return the axis-aligned bounding box of a molecule, expanded on every side
    by the given padding in angstrom.
    """
    return BoundingBox.from_molecule(mol, padding=padding)


def surface_spheres(
    mol: Molecule, probe_radius: float = 1.4, n_points: int = 92
) -> List[np.ndarray]:
    """
    Generate probe positions on the solvent-accessible surface of a molecule.
    Each atom is surrounded by n_points points spread evenly over a sphere using
    the Fibonacci spiral construction, with the sphere radius set to the atom
    radius plus the probe radius. Points that fall inside any neighbouring atom
    are discarded, so only genuinely exposed surface points remain. The function
    returns a list of (x, y, z) positions on the molecular surface.
    """
    positions = []
    golden = (1 + math.sqrt(5)) / 2
    for atom in mol.atoms:
        r = atom.radius + probe_radius
        c = atom.position
        for i in range(n_points):
            theta = math.acos(1 - 2 * (i + 0.5) / n_points)
            phi = 2 * PI * i / golden
            x = c[0] + r * math.sin(theta) * math.cos(phi)
            y = c[1] + r * math.sin(theta) * math.sin(phi)
            z = c[2] + r * math.cos(theta)
            # Keep this point only if it does not lie inside any other atom.
            p = np.array([x, y, z])
            buried = any(
                np.linalg.norm(p - a.position) < a.radius + probe_radius
                for a in mol.atoms
                if a is not atom
            )
            if not buried:
                positions.append(p)
    return positions


def lumped_charges(
    mol: Molecule, grid_spacing: float = 2.0
) -> List[Tuple[np.ndarray, float]]:
    """
    Coarse-grain the atomic charges of a molecule onto a regular cubic grid.
    Each atom is assigned to its nearest grid point and its charge is added
    there, so several nearby atoms collapse into a single effective charge. The
    function returns a list of (position, charge) tuples for the grid points
    that carry a non-zero net charge.
    """
    if not mol.atoms:
        return []
    bb = bounding_box(mol, padding=grid_spacing)
    # Lay out the grid axes spanning the padded bounding box.
    xs = np.arange(bb.xmin, bb.xmax + grid_spacing, grid_spacing)
    ys = np.arange(bb.ymin, bb.ymax + grid_spacing, grid_spacing)
    zs = np.arange(bb.zmin, bb.zmax + grid_spacing, grid_spacing)
    grid: Dict[Tuple[int, int, int], float] = {}
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
            pos = np.array(
                [
                    bb.xmin + ix * grid_spacing,
                    bb.ymin + iy * grid_spacing,
                    bb.zmin + iz * grid_spacing,
                ]
            )
            result.append((pos, q))
    return result


def electrostatic_center(mol: Molecule) -> np.ndarray:
    """
    Return the charge-weighted center of a molecule, with each atom weighted by
    the magnitude of its charge. If the molecule carries essentially no charge,
    the function falls back to the geometric centroid.
    """
    total_q = sum(abs(a.charge) for a in mol.atoms)
    if total_q < 1e-10:
        return mol.centroid()
    pos = mol.positions_array()
    charges = np.abs(mol.charges_array())
    return (pos * charges[:, None]).sum(axis=0) / total_q


def hydrodynamic_radius_from_rg(mol: Molecule) -> float:
    """
    Estimate the hydrodynamic radius of a molecule from its radius of gyration.
    The estimate uses the empirical relation

        r_h ≈ 0.77 × r_g

    where r_h is the hydrodynamic radius and r_g is the radius of gyration. The
    factor 0.77 is an empirical value appropriate for globular proteins.
    """
    return 0.77 * mol.radius_of_gyration()


def hydrodynamic_radius_from_surface(mol: Molecule) -> float:
    """
    Estimate the hydrodynamic radius of a molecule as its bounding radius, that
    is, the radius of the smallest sphere that encloses all of its atoms.
    """
    return mol.bounding_radius()


def contact_distances(
    mol1: Molecule, mol2: Molecule, cutoff: float = 8.0
) -> List[Tuple[int, int, float]]:
    """
    Find all pairs of atoms from two molecules that lie within the cutoff
    distance in angstrom. Each pair is reported as (i, j, dist), where i and j
    index the atoms in the first and second molecule and dist is their
    separation. The pairs are sorted by increasing distance and are used to
    automatically generate the reaction contacts of an association event.
    """
    pairs = []
    for i, a1 in enumerate(mol1.atoms):
        for j, a2 in enumerate(mol2.atoms):
            d = a1.distance_to(a2)
            if d <= cutoff:
                pairs.append((i, j, d))
    pairs.sort(key=lambda t: t[2])
    return pairs


def born_integral(
    charge: float, radius: float, eps_in: float = 4.0, eps_out: float = 78.54
) -> float:
    """
    Compute the Born solvation energy of a charged sphere, that is, the work of
    moving a charge from a medium of dielectric constant ε_in into a medium of
    dielectric constant ε_out. The energy is

        ΔG_Born = -(q² / 8π ε₀) × (1/ε_in - 1/ε_out) / r

    where q is the charge, r is the sphere radius, ε_in is the interior
    dielectric constant, and ε_out is the exterior (solvent) dielectric
    constant. The result is returned in units of kBT, with the prefactor folded
    into the Bjerrum length so that no explicit ε₀ appears in the code.
    """
    if radius < 1e-10:
        return 0.0
    return -(charge**2 * BJERRUM_LENGTH / (2 * radius)) * (1.0 / eps_in - 1.0 / eps_out)
