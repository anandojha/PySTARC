"""
Hydrodynamic radius by Hansen Monte Carlo
==========================================

Background
-------------------
The hydrodynamic radius a determines how fast a molecule diffuses:
    D_trans = kBT / (6π η a)
    D_rot   = kBT / (8π η a³) = 3 D_trans / (4a²)

For a perfect sphere, a equals the geometric radius.  For a protein,
a is an effective radius that accounts for the irregular shape and
the solvent layer.

The Hansen algorithm (J. Chem. Phys. 121, 9111, 2004) computes a
by Monte Carlo simulation of random walkers outside the molecular
surface:
  1. Generate the solvent-excluded surface (atom centers + probe radius)
  2. Launch random walkers from the surface
  3. Each walker diffuses outward and its trajectory is recorded
  4. The mean inverse chord length <1/L> between surface points gives:
     a = 1 / <1/L>
This is equivalent to solving the exterior Stokes problem for the
hydrodynamic friction of the molecular shape.

Why not just use the radius of gyration?
  - R_g underestimates a by ~20% for typical proteins
  - R_g does not account for shape anisotropy
  - The Hansen MC gives the correct Stokes radius
"""

from __future__ import annotations
from typing import List, Tuple, NamedTuple
import numpy as np
import math

# Each entry is (vertex_a, vertex_b) as (ix,iy,iz) offsets in the 2x2x2 cube.
_EDGES = [
    ((0,0,0),(1,0,0)), ((1,0,0),(1,1,0)), ((1,1,0),(0,1,0)), ((0,1,0),(0,0,0)),
    ((0,0,1),(1,0,1)), ((1,0,1),(1,1,1)), ((1,1,1),(0,1,1)), ((0,1,1),(0,0,1)),
    ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((1,1,0),(1,1,1)), ((0,1,0),(0,1,1)),
]
_FDIAGS = [
    ((0,0,0),(1,1,0)), ((1,0,0),(0,1,0)), ((0,0,1),(1,1,1)), ((1,0,1),(0,1,1)),
    ((0,0,0),(0,1,1)), ((0,1,0),(0,0,1)), ((1,0,0),(1,1,1)), ((1,1,0),(1,0,1)),
    ((0,0,0),(1,0,1)), ((1,0,0),(0,0,1)), ((0,1,0),(1,1,1)), ((1,1,0),(0,1,1)),
]
_LDIAGS = [
    ((0,0,0),(1,1,1)), ((1,0,0),(0,1,1)),
    ((1,1,0),(0,0,1)), ((0,1,0),(1,0,1)),
]

# area table: 13 distinct fingerprint classes -> area weight
# tri  = sqrt(3)/8,  rect = sqrt(2)/2
_TRI  = math.sqrt(3.0) / 8.0
_RECT = math.sqrt(2.0) / 2.0
_SIG_AREAS = {
    (1, 3, 3, 1): _TRI,
    (2, 4, 6, 2): _RECT,
    (2, 6, 4, 2): 2.0 * _TRI,
    (2, 6, 6, 0): 2.0 * _TRI,
    (3, 5, 7, 3): 0.5 + 3.0 * _TRI,
    (3, 7, 7, 1): _RECT + _TRI,
    (3, 9, 3, 3): 3.0 * _TRI,
    (4, 4, 8, 4): 1.0,
    (4, 8, 6, 2): 0.5 + 4.0 * _TRI,
    (4, 6, 8, 2): math.sqrt(2.0),
    (4, 6, 6, 4): 6.0 * _TRI,
    (4, 8, 8, 0): 2.0 * _RECT,
    (4,12, 0, 4): 4.0 * _TRI,
}

def _fingerprint(verts: np.ndarray) -> tuple:
    """
    Compute (sum, nedges, nfdiags, nldiags) fingerprint for a 2×2×2 cube.
    verts[ix,iy,iz] = 1 (inside) or 0 (outside).
    """
    s = int(verts.sum())
    if s > 4:
        s = 8 - s
    def count(pairs):
        n = 0
        for (ax,ay,az),(bx,by,bz) in pairs:
            va, vb = verts[ax,ay,az], verts[bx,by,bz]
            if va != vb:   # one inside, one outside = surface crossing
                n += 1
        return n
    return (s, count(_EDGES), count(_FDIAGS), count(_LDIAGS))

def _surface_position(verts: np.ndarray,
                      ix: int, iy: int, iz: int,
                      hx: float, hy: float, hz: float) -> np.ndarray:
    """
    Average position of surface-crossing edge midpoints in a 2×2×2 cube.
    """
    total = np.zeros(3)
    n = 0
    for (ax,ay,az),(bx,by,bz) in _EDGES:
        va = verts[ax, ay, az]
        vb = verts[bx, by, bz]
        if va != vb:
            # midpoint of this edge in world coordinates
            mid = np.array([
                hx * ((ix + ax + ix + bx) * 0.5),
                hy * ((iy + ay + iy + by) * 0.5),
                hz * ((iz + az + iz + bz) * 0.5),
            ])
            total += mid
            n += 1
    return total / n if n > 0 else total

def _voxelise(coords: np.ndarray,
              radii:  np.ndarray,
              spacing: float = 0.5,
              padding: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build inside/outside voxel grid from atom positions and radii.
    Returns (grid[nx,ny,nz], origin[3], (hx,hy,hz)).
    A voxel is inside (1) if it is within the vdW radius of any atom.
    """
    lo = coords.min(axis=0) - radii.max() - padding
    hi = coords.max(axis=0) + radii.max() + padding
    hx = hy = hz = spacing
    nx = int(math.ceil((hi[0] - lo[0]) / hx)) + 2
    ny = int(math.ceil((hi[1] - lo[1]) / hy)) + 2
    nz = int(math.ceil((hi[2] - lo[2]) / hz)) + 2
    # Build grid: grid[ix, iy, iz] = 1 if inside any sphere
    xs = lo[0] + hx * np.arange(nx)
    ys = lo[1] + hy * np.arange(ny)
    zs = lo[2] + hz * np.arange(nz)
    grid = np.zeros((nx, ny, nz), dtype=np.int8)
    # Vectorised: for each atom, mark all voxels within radius
    for i in range(len(coords)):
        cx, cy, cz = coords[i]
        r = radii[i]
        # find index range
        ix0 = max(0, int((cx - r - lo[0]) / hx) - 1)
        ix1 = min(nx, int((cx + r - lo[0]) / hx) + 2)
        iy0 = max(0, int((cy - r - lo[1]) / hy) - 1)
        iy1 = min(ny, int((cy + r - lo[1]) / hy) + 2)
        iz0 = max(0, int((cz - r - lo[2]) / hz) - 1)
        iz1 = min(nz, int((cz + r - lo[2]) / hz) + 2)
        sx = xs[ix0:ix1] - cx
        sy = ys[iy0:iy1] - cy
        sz = zs[iz0:iz1] - cz
        d2 = (sx[:, None, None]**2 +
              sy[None, :, None]**2 +
              sz[None, None, :]**2)
        mask = d2 <= r * r
        grid[ix0:ix1, iy0:iy1, iz0:iz1] |= mask.astype(np.int8)
    return grid, lo, (hx, hy, hz)

class SurfacePoint(NamedTuple):
    area: float
    pos:  np.ndarray

def _extract_surface(grid: np.ndarray,
                     origin: np.ndarray,
                     spacing: Tuple[float, float, float]
                     ) -> List[SurfacePoint]:
    """
    Find all surface cubes (mixed 0/1 in 2×2×2 block) and compute
    their area weight + representative position.
    """
    hx, hy, hz = spacing
    nx, ny, nz = grid.shape
    surface = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            for iz in range(nz - 1):
                verts = grid[ix:ix+2, iy:iy+2, iz:iz+2]
                s = int(verts.sum())
                if s == 0 or s == 8:
                    continue   # fully outside or fully inside
                fp = _fingerprint(verts)
                if fp not in _SIG_AREAS:
                    continue   # degenerate cube - skip
                area = _SIG_AREAS[fp]
                pos  = _surface_position(verts, ix, iy, iz, hx, hy, hz)
                surface.append(SurfacePoint(area=area, pos=origin + pos))
    return surface

def mc_hydrodynamic_radius(
        coords:  np.ndarray,
        radii:   np.ndarray,
        spacing: float = 0.5,
        n_mc:    int   = 1_000_000,
        seed:    int   = 1111111,
) -> Tuple[float, np.ndarray, float]:
    """
    Compute hydrodynamic radius by the Hansen (2004) Monte Carlo algorithm.
    Exact translation of the reference C++ implementation``.
    Parameters
    ----------
    coords  : (N,3) atom positions [Å]
    radii   : (N,)  van-der-Waals radii [Å]
    spacing : voxel grid spacing [Å]
    n_mc    : number of MC surface-pair samples
    seed    : random seed (the reference implementation: 1111111)
    Returns
    -------
    r_h      : hydrodynamic radius [Å]
    center   : area-weighted surface centroid [Å] (3,)
    max_dist : max distance from center to any surface point [Å]
    """
    # 1. Voxelise
    grid, origin, sp = _voxelise(coords, radii, spacing=spacing)
    # 2. Extract surface cubes
    surface = _extract_surface(grid, origin, sp)
    if len(surface) < 2:
        raise ValueError("No surface found - check atom radii and grid spacing")
    areas = np.array([s.area for s in surface])
    poses = np.array([s.pos  for s in surface])   # (M, 3)
    nsu   = len(surface)
    # 3. Area-weighted centroid (the reference implementation: psum/a0sum)
    a0sum  = float(areas.sum())
    center = (areas[:, None] * poses).sum(axis=0) / a0sum
    # 4. max_dist from center to any surface point
    max_dist = float(np.max(np.linalg.norm(poses - center, axis=1)))
    # 5. Monte Carlo: sample n_mc pairs, weighted by area product
    rng = np.random.default_rng(seed)
    # Draw n_mc distinct pairs (i0 != i1)
    # For large nsu this is effectively always distinct on first draw
    i0 = rng.integers(0, nsu, size=n_mc)
    i1 = rng.integers(0, nsu, size=n_mc)
    same = (i0 == i1)
    while same.any():
        i1[same] = rng.integers(0, nsu, size=same.sum())
        same = (i0 == i1)
    a0  = areas[i0]                        # (n_mc,)
    a1  = areas[i1]                        # (n_mc,)
    dv  = poses[i0] - poses[i1]            # (n_mc, 3)
    r   = np.linalg.norm(dv, axis=1)       # (n_mc,)
    r   = np.maximum(r, 1e-12)             # avoid divide-by-zero
    aa      = a0 * a1
    asum    = float(aa.sum())
    inv_sum = float((aa / r).sum())
    # r_h = asum / inv_chord_sum
    r_h = asum / inv_sum if inv_sum > 0 else 0.0
    return r_h, center, max_dist