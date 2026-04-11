"""
APBS grid force interpolation
==============================

Background
-------------------
APBS solves the linearized Poisson-Boltzmann equation on a 3D grid,
producing a volumetric potential map φ(x,y,z) in kBT/e units.

To compute the electrostatic force on an atom at position r:
    F = -q × ∇φ(r),
we need the gradient of the potential at arbitrary (off-grid) points.

This is done by trilinear interpolation.
  1. Locate the grid cell containing the atom
  2. Compute fractional coordinates (fx, fy, fz) within the cell
  3. Interpolate the 8 corner values using trilinear weights:
     φ(r) = Σ w_ijk × φ_ijk
     where w = (1-fx)(1-fy)(1-fz), fx(1-fy)(1-fz), etc.
  4. Compute the gradient by central differences at half-spacing:
     ∂φ/∂x ≈ [φ(x+h/2) - φ(x-h/2)] / h
     
Grid boundary
--------------
APBS boundary conditions (Debye-Hückel) are only approximate.
Atoms within 3 grid spacings of the boundary receive forces from
the Yukawa multipole fallback instead, avoiding artifacts.

Grid selection
--------------
APBS produces two grids per molecule.
1. Coarse - Large domain, low resolution (for APBS boundary conditions)
2. Fine - Small domain, high resolution (for force evaluation)

At runtime, only the fine grid is used for forces.  The coarse grid
exists solely to provide accurate boundary conditions to APBS and
is dropped before the BD simulation begins.
"""

from __future__ import annotations
from pystarc.global_defs.constants import BJERRUM_LENGTH, DEFAULT_DEBYE_LENGTH, KBT_KCAL
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import math

# Screened Coulomb (Debye-Hückel)
def debye_huckel_energy(q1: float, q2: float,
                        r: float,
                        debye_length: float = DEFAULT_DEBYE_LENGTH,
                        bjerrum_length: float = BJERRUM_LENGTH) -> float:
    """
    E = q1 q2 l_B exp(-r / λ_D) / r    [kBT]
    Parameters
    ----------
    q1, q2 : charges in elementary charge units
    r      : separation in Å
    """
    if r < 1e-10:
        return 0.0
    return q1 * q2 * bjerrum_length * math.exp(-r / debye_length) / r

def debye_huckel_force(q1: float, q2: float,
                       r_vec: np.ndarray,
                       debye_length: float = DEFAULT_DEBYE_LENGTH,
                       bjerrum_length: float = BJERRUM_LENGTH) -> np.ndarray:
    """
    F = -∇E  in the direction of r_vec.
    Returns force on particle 1 (pointing away from particle 2 if same sign).
    """
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
        return np.zeros(3)
    E = debye_huckel_energy(q1, q2, r, debye_length, bjerrum_length)
    dE_dr = E * (-1.0/r - 1.0/debye_length)
    return -dE_dr * r_vec / r  # force = -dE/dr * r_hat ... but sign from convention

# DX grid reader

class DXGrid:
    """
    Volumetric potential grid loaded from an APBS .dx file.
    Provides trilinear interpolation of potential and gradient.
    """
    def __init__(self,
                 origin: np.ndarray,
                 delta: np.ndarray,                    # (3,3) matrix of grid spacings
                 data: np.ndarray):                    # (nx, ny, nz) potential in kBT/e
        self.origin = np.asarray(origin, dtype=float)
        self.delta  = np.asarray(delta,  dtype=float)  # (3,3)
        self.data   = np.asarray(data,   dtype=float)
        self.shape  = np.array(self.data.shape)
        # inverse delta for fast index computation (assumes orthogonal grid)
        self._inv_dx = 1.0 / np.diag(self.delta)

    @classmethod
    def from_file(cls, path: str | Path) -> "DXGrid":
        path = Path(path)
        origin = np.zeros(3)
        delta  = np.zeros((3, 3))
        shape  = np.zeros(3, dtype=int)
        raw_values: list[float] = []
        with open(path) as fh:
            in_data = False
            delta_row = 0
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if line.startswith("object 1"):
                    # object 1 class gridpositions counts nx ny nz
                    parts = line.split()
                    shape[:] = int(parts[-3]), int(parts[-2]), int(parts[-1])
                    continue
                if line.startswith("origin"):
                    parts = line.split()
                    origin[:] = float(parts[1]), float(parts[2]), float(parts[3])
                    continue
                if line.startswith("delta"):
                    parts = line.split()
                    delta[delta_row] = float(parts[1]), float(parts[2]), float(parts[3])
                    delta_row += 1
                    continue
                if line.startswith("object 3"):
                    in_data = True
                    continue
                if in_data:
                    if line.startswith("object") or line.startswith("attribute"):
                        break
                    raw_values.extend(float(v) for v in line.split())
        data = np.array(raw_values, dtype=float).reshape(
            shape[0], shape[1], shape[2])
        return cls(origin, delta, data)

    def _to_fractional(self, point: np.ndarray) -> np.ndarray:
        """Convert Å coordinate to fractional grid index."""
        diff = point - self.origin
        return diff * self._inv_dx   # element-wise for orthogonal grid

    def interpolate(self, point: np.ndarray) -> float:
        """Trilinear interpolation of potential at given Å coordinate."""
        idx = self._to_fractional(point)
        ix, iy, iz = idx[0], idx[1], idx[2]
        i0 = int(math.floor(ix))
        j0 = int(math.floor(iy))
        k0 = int(math.floor(iz))
        # bounds check
        nx, ny, nz = self.data.shape
        if not (0 <= i0 < nx-1 and 0 <= j0 < ny-1 and 0 <= k0 < nz-1):
            return 0.0
        fx = ix - i0
        fy = iy - j0
        fz = iz - k0
        d = self.data
        val = (d[i0,   j0,   k0  ] * (1-fx)*(1-fy)*(1-fz) +
               d[i0+1, j0,   k0  ] *    fx *(1-fy)*(1-fz) +
               d[i0,   j0+1, k0  ] * (1-fx)*   fy *(1-fz) +
               d[i0,   j0,   k0+1] * (1-fx)*(1-fy)*   fz  +
               d[i0+1, j0+1, k0  ] *    fx *   fy *(1-fz) +
               d[i0+1, j0,   k0+1] *    fx *(1-fy)*   fz  +
               d[i0,   j0+1, k0+1] * (1-fx)*   fy *   fz  +
               d[i0+1, j0+1, k0+1] *    fx *   fy *   fz)
        return float(val)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of potential using the exact method:
        trilinear interpolation of forward finite differences within the cube.
        Exact translation of Single_Grid::gradient_of_cube() in single_grid.hh:
          gz = (v[i,j,k+1] - v[i,j,k]) / hz   (then trilinearly interpolated)
          gy = (v[i,j+1,k] - v[i,j,k]) / hy
          gx = (v[i+1,j,k] - v[i,j,k]) / hx
        Returns (3,) vector in kBT/(e·Å).
        """
        idx = self._to_fractional(point)
        ix = int(math.floor(idx[0]))
        iy = int(math.floor(idx[1]))
        iz = int(math.floor(idx[2]))
        nx, ny, nz = self.data.shape
        if not (0 <= ix < nx-1 and 0 <= iy < ny-1 and 0 <= iz < nz-1):
            return np.zeros(3)
        ax = idx[0] - ix
        ay = idx[1] - iy
        az = idx[2] - iz
        apx = 1.0 - ax
        apy = 1.0 - ay
        apz = 1.0 - az
        d = self.data
        hx, hy, hz = self.delta[0, 0], self.delta[1, 1], self.delta[2, 2]
        # 8 cube corners: [ix+dx, iy+dy, iz+dz] for dx,dy,dz in {0,1}
        vmmm = float(d[ix,   iy,   iz  ])
        vmmp = float(d[ix,   iy,   iz+1])
        vmpm = float(d[ix,   iy+1, iz  ])
        vmpp = float(d[ix,   iy+1, iz+1])
        vpmm = float(d[ix+1, iy,   iz  ])
        vpmp = float(d[ix+1, iy,   iz+1])
        vppm = float(d[ix+1, iy+1, iz  ])
        vppp = float(d[ix+1, iy+1, iz+1])
        # z-component: gz = (vmmp-vmmm)/hz trilinearly weighted (the reference implementation exact)
        gzmm = (vmmp - vmmm) / hz
        gzmp = (vmpp - vmpm) / hz
        gzpm = (vpmp - vpmm) / hz
        gzpp = (vppp - vppm) / hz
        gzm  = apy * gzmm + ay * gzmp
        gzp  = apy * gzpm + ay * gzpp
        gz   = apx * gzm  + ax * gzp
        # y-component
        gymm = (vmpm - vmmm) / hy
        gymp = (vmpp - vmmp) / hy
        gypm = (vppm - vpmm) / hy
        gypp = (vppp - vpmp) / hy
        gym  = apz * gymm + az * gymp
        gyp  = apz * gypm + az * gypp
        gy   = apx * gym  + ax * gyp
        # x-component
        gxmm = (vpmm - vmmm) / hx
        gxmp = (vpmp - vmmp) / hx
        gxpm = (vppm - vmpm) / hx
        gxpp = (vppp - vmpp) / hx
        gxm  = apz * gxmm + az * gxmp
        gxp  = apz * gxpm + az * gxpp
        gx   = apy * gxm  + ay * gxp
        return np.array([gx, gy, gz])

    def force_on_charge(self, point: np.ndarray, charge: float) -> np.ndarray:
        """Force on a point charge at given position: F = -q ∇φ  [kBT/Å]."""
        return -charge * self.gradient(point)

    # Vectorised batch methods (50-100× faster than per-atom loops) 
    def batch_interpolate(self, points: np.ndarray) -> np.ndarray:
        """
        Trilinear interpolation for N points at once.
        Parameters
        ----------
        points : (N, 3) array of Å coordinates
        Returns
        -------
        (N,) array of potential values  [kBT/e]
        """
        pts = np.asarray(points, dtype=float)            # (N,3)
        idx = (pts - self.origin) * self._inv_dx         # (N,3) fractional
        i0 = np.floor(idx[:, 0]).astype(int)
        j0 = np.floor(idx[:, 1]).astype(int)
        k0 = np.floor(idx[:, 2]).astype(int)
        nx, ny, nz = self.data.shape
        valid = ((i0 >= 0) & (i0 < nx-1) &
                 (j0 >= 0) & (j0 < ny-1) &
                 (k0 >= 0) & (k0 < nz-1))
        fx = idx[:, 0] - i0
        fy = idx[:, 1] - j0
        fz = idx[:, 2] - k0
        out = np.zeros(len(pts))
        v = valid
        d = self.data
        out[v] = (d[i0[v],   j0[v],   k0[v]  ] * (1-fx[v])*(1-fy[v])*(1-fz[v]) +
                  d[i0[v]+1, j0[v],   k0[v]  ] *    fx[v] *(1-fy[v])*(1-fz[v]) +
                  d[i0[v],   j0[v]+1, k0[v]  ] * (1-fx[v])*   fy[v] *(1-fz[v]) +
                  d[i0[v],   j0[v],   k0[v]+1] * (1-fx[v])*(1-fy[v])*   fz[v]  +
                  d[i0[v]+1, j0[v]+1, k0[v]  ] *    fx[v] *   fy[v] *(1-fz[v]) +
                  d[i0[v]+1, j0[v],   k0[v]+1] *    fx[v] *(1-fy[v])*   fz[v]  +
                  d[i0[v],   j0[v]+1, k0[v]+1] * (1-fx[v])*   fy[v] *   fz[v]  +
                  d[i0[v]+1, j0[v]+1, k0[v]+1] *    fx[v] *   fy[v] *   fz[v])
        return out

    def batch_gradient(self, points: np.ndarray) -> np.ndarray:
        """
        Central-difference gradient for N points at once.
        Returns
        -------
        (N, 3) array of gradient vectors  [kBT/(e·Å)]
        """
        pts = np.asarray(points, dtype=float)
        h   = np.diag(self.delta) * 0.5     # half-step per axis
        grad = np.zeros_like(pts)
        for i in range(3):
            dp = pts.copy(); dp[:, i] += h[i]
            dm = pts.copy(); dm[:, i] -= h[i]
            grad[:, i] = (self.batch_interpolate(dp) -
                          self.batch_interpolate(dm)) / (2 * h[i])
        return grad

    def batch_force_on_charges(self,
                               points: np.ndarray,
                               charges: np.ndarray) -> np.ndarray:
        """
        Force on N point charges: F_i = -q_i ∇φ(r_i).
        Parameters
        ----------
        points  : (N, 3) positions  [Å]
        charges : (N,)  charges     [e]
        Returns
        -------
        (N, 3) force array  [kBT/Å]
        """
        grad = self.batch_gradient(points)              # (N,3)
        return -charges[:, None] * grad                 # (N,3)

    def __repr__(self) -> str:
        nx, ny, nz = self.data.shape
        return (f"DXGrid({nx}×{ny}×{nz}, origin={self.origin}, "
                f"spacing={np.diag(self.delta)})")