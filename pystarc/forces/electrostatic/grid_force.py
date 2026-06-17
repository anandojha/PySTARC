"""
Interpolation of electrostatic forces from an APBS potential grid.

APBS solves the linearized Poisson-Boltzmann equation on a three-dimensional
grid, producing a volumetric potential map φ(x,y,z) in units of kBT/e. To
compute the electrostatic force on an atom at position r we use

    F = -q × ∇φ(r),

which requires the gradient of the potential at arbitrary off-grid points. Here
q is the atomic charge and ∇φ is the gradient of the potential at r.

We obtain off-grid values by trilinear interpolation. First we locate the grid
cell containing the atom and compute the fractional coordinates (fx, fy, fz) of
the atom within that cell. The potential is then a weighted sum over the eight
corners of the cell,

    φ(r) = Σ w_ijk × φ_ijk,

where the weights are products of the fractional coordinates such as
(1-fx)(1-fy)(1-fz) for one corner and fx(1-fy)(1-fz) for the next. The gradient
is taken by central differences at half the grid spacing,

    ∂φ/∂x ≈ [φ(x+h/2) - φ(x-h/2)] / h,

where h is the grid spacing along the axis.

The APBS boundary conditions (Debye-Hückel) are only approximate, so atoms
within three grid spacings of the grid boundary receive their forces from the
Yukawa multipole fallback instead, which avoids boundary artifacts.

APBS produces two grids per molecule. The coarse grid covers a large domain at
low resolution and supplies the boundary conditions for APBS. The fine grid
covers a small domain at high resolution and is the one used for force
evaluation. At runtime only the fine grid is used for forces. The coarse grid
serves only to give APBS accurate boundary conditions and is dropped before the
Brownian-dynamics simulation begins.
"""

from __future__ import annotations
from pystarc.global_defs.constants import BJERRUM_LENGTH, DEFAULT_DEBYE_LENGTH, KBT_KCAL
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import math


# Screened-Coulomb (Debye-Hückel) interaction.
def debye_huckel_energy(
    q1: float,
    q2: float,
    r: float,
    debye_length: float = DEFAULT_DEBYE_LENGTH,
    bjerrum_length: float = BJERRUM_LENGTH,
) -> float:
    """
    Screened-Coulomb (Debye-Hückel) interaction energy between two point
    charges, returned in units of kBT,

        E = q1 q2 l_B exp(-r / λ_D) / r.

    Here q1 and q2 are the charges in units of the elementary charge, r is their
    separation in Å, l_B is the Bjerrum length, and λ_D is the Debye screening
    length.
    """
    if r < 1e-10:
        return 0.0
    return q1 * q2 * bjerrum_length * math.exp(-r / debye_length) / r


def debye_huckel_force(
    q1: float,
    q2: float,
    r_vec: np.ndarray,
    debye_length: float = DEFAULT_DEBYE_LENGTH,
    bjerrum_length: float = BJERRUM_LENGTH,
) -> np.ndarray:
    """
    Screened-Coulomb force on particle 1, F = -∇E, taken along the direction of
    r_vec, the vector from particle 2 to particle 1. For charges of the same
    sign the force points away from particle 2.
    """
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
        return np.zeros(3)
    E = debye_huckel_energy(q1, q2, r, debye_length, bjerrum_length)
    dE_dr = E * (-1.0 / r - 1.0 / debye_length)
    return -dE_dr * r_vec / r  # F = -dE/dr times the unit vector r_hat, with the sign set by the convention above.


# Reader for the OpenDX volumetric grid format.


class DXGrid:
    """
    A volumetric potential grid loaded from an APBS .dx file. It provides
    trilinear interpolation of the potential and its gradient.
    """

    def __init__(
        self,
        origin: np.ndarray,
        delta: np.ndarray,  # (3,3) matrix of grid spacings
        data: np.ndarray,
    ):  # (nx, ny, nz) array of potential values in kBT/e
        self.origin = np.asarray(origin, dtype=float)
        self.delta = np.asarray(delta, dtype=float)  # (3,3)
        self.data = np.asarray(data, dtype=float)
        self.shape = np.array(self.data.shape)
        # Inverse grid spacing along each axis, used for fast index lookup. This
        # assumes an orthogonal grid, so validate that assumption before using
        # the diagonal alone. The index lookup and interpolation only consume
        # the diagonal spacings, so a non-orthogonal grid or a non-positive
        # spacing would be silently mishandled without these guards.
        diag = np.diag(self.delta)
        for axis, spacing in enumerate(diag):
            if not spacing > 0.0:
                raise ValueError(
                    f"Grid spacing along axis {axis} must be strictly positive, "
                    f"got {spacing}."
                )
        off_diagonal = self.delta - np.diag(diag)
        max_off = float(np.max(np.abs(off_diagonal)))
        tol = 1e-9 * float(np.max(np.abs(diag)))
        if max_off > tol:
            raise ValueError(
                "Grid is not orthogonal: the off-diagonal entries of delta must "
                f"be approximately zero, but the largest is {max_off} (tolerance "
                f"{tol}). This interpolator only supports orthogonal grids."
            )
        self._inv_dx = 1.0 / diag

    @classmethod
    def from_file(cls, path: str | Path) -> "DXGrid":
        path = Path(path)
        origin = np.zeros(3)
        delta = np.zeros((3, 3))
        shape = np.zeros(3, dtype=int)
        raw_values: list[float] = []
        with open(path) as fh:
            in_data = False
            delta_row = 0
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if line.startswith("object 1"):
                    # This line has the form "object 1 class gridpositions counts nx ny nz".
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
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        expected = nx * ny * nz
        if len(raw_values) != expected:
            raise ValueError(
                f"Malformed DX file {path}: expected {expected} grid values "
                f"({nx} x {ny} x {nz}) but read {len(raw_values)}."
            )
        data = np.array(raw_values, dtype=float).reshape(nx, ny, nz)
        return cls(origin, delta, data)

    def _to_fractional(self, point: np.ndarray) -> np.ndarray:
        """Convert a coordinate in Å to a fractional grid index."""
        diff = point - self.origin
        return diff * self._inv_dx  # Element-wise division, valid for an orthogonal grid.

    def interpolate(self, point: np.ndarray) -> float:
        """Trilinear interpolation of the potential at a given coordinate in Å."""
        idx = self._to_fractional(point)
        ix, iy, iz = idx[0], idx[1], idx[2]
        i0 = int(math.floor(ix))
        j0 = int(math.floor(iy))
        k0 = int(math.floor(iz))
        # Return zero if the point falls outside the grid.
        nx, ny, nz = self.data.shape
        if not (0 <= i0 < nx - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nz - 1):
            return 0.0
        fx = ix - i0
        fy = iy - j0
        fz = iz - k0
        d = self.data
        val = (
            d[i0, j0, k0] * (1 - fx) * (1 - fy) * (1 - fz)
            + d[i0 + 1, j0, k0] * fx * (1 - fy) * (1 - fz)
            + d[i0, j0 + 1, k0] * (1 - fx) * fy * (1 - fz)
            + d[i0, j0, k0 + 1] * (1 - fx) * (1 - fy) * fz
            + d[i0 + 1, j0 + 1, k0] * fx * fy * (1 - fz)
            + d[i0 + 1, j0, k0 + 1] * fx * (1 - fy) * fz
            + d[i0, j0 + 1, k0 + 1] * (1 - fx) * fy * fz
            + d[i0 + 1, j0 + 1, k0 + 1] * fx * fy * fz
        )
        return float(val)

    def gradient_of_cube(self, point: np.ndarray) -> np.ndarray:
        """
        First-order gradient of the potential, computed by trilinearly
        interpolating the forward finite differences taken within the enclosing
        cube. This is an exact translation of Single_Grid::gradient_of_cube() in
        single_grid.hh and is kept as the BrownDye2 reference. It is the analytic
        gradient of the trilinear interpolant and is therefore only first order
        in the grid spacing. The production force path uses the second-order
        central difference in gradient(); on a screened-Coulomb field this form
        deviates from the true gradient by about 2.5 percent on average, where
        the central difference stays near 0.1 percent.
        Each component is a forward difference between adjacent corners divided
        by the grid spacing along that axis, for example

            gz = (v[i,j,k+1] - v[i,j,k]) / hz,
            gy = (v[i,j+1,k] - v[i,j,k]) / hy,
            gx = (v[i+1,j,k] - v[i,j,k]) / hx,

        and these differences are then interpolated trilinearly across the cube.
        Returns a length-three vector in units of kBT/(e·Å).
        """
        idx = self._to_fractional(point)
        ix = int(math.floor(idx[0]))
        iy = int(math.floor(idx[1]))
        iz = int(math.floor(idx[2]))
        nx, ny, nz = self.data.shape
        if not (0 <= ix < nx - 1 and 0 <= iy < ny - 1 and 0 <= iz < nz - 1):
            return np.zeros(3)
        ax = idx[0] - ix
        ay = idx[1] - iy
        az = idx[2] - iz
        apx = 1.0 - ax
        apy = 1.0 - ay
        apz = 1.0 - az
        d = self.data
        hx, hy, hz = self.delta[0, 0], self.delta[1, 1], self.delta[2, 2]
        # The eight cube corners at [ix+dx, iy+dy, iz+dz] for dx, dy, dz each in {0, 1}.
        vmmm = float(d[ix, iy, iz])
        vmmp = float(d[ix, iy, iz + 1])
        vmpm = float(d[ix, iy + 1, iz])
        vmpp = float(d[ix, iy + 1, iz + 1])
        vpmm = float(d[ix + 1, iy, iz])
        vpmp = float(d[ix + 1, iy, iz + 1])
        vppm = float(d[ix + 1, iy + 1, iz])
        vppp = float(d[ix + 1, iy + 1, iz + 1])
        # z-component: the forward difference (vmmp - vmmm)/hz weighted trilinearly, matching the reference implementation exactly.
        gzmm = (vmmp - vmmm) / hz
        gzmp = (vmpp - vmpm) / hz
        gzpm = (vpmp - vpmm) / hz
        gzpp = (vppp - vppm) / hz
        gzm = apy * gzmm + ay * gzmp
        gzp = apy * gzpm + ay * gzpp
        gz = apx * gzm + ax * gzp
        # y-component.
        gymm = (vmpm - vmmm) / hy
        gymp = (vmpp - vmmp) / hy
        gypm = (vppm - vpmm) / hy
        gypp = (vppp - vpmp) / hy
        gym = apz * gymm + az * gymp
        gyp = apz * gypm + az * gypp
        gy = apx * gym + ax * gyp
        # x-component.
        gxmm = (vpmm - vmmm) / hx
        gxmp = (vpmp - vmmp) / hx
        gxpm = (vppm - vmpm) / hx
        gxpp = (vppp - vmpp) / hx
        gxm = apz * gxmm + az * gxmp
        gxp = apz * gxpm + az * gxpp
        gx = apy * gxm + ay * gxp
        return np.array([gx, gy, gz])

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of the potential by central difference at half the grid
        spacing. This is the second-order operator used for the force, matching
        the production GPU kernel and batch_gradient, and it is consistent with
        the trilinear interpolation used for the energy. See gradient_of_cube()
        for the first-order BrownDye2 form. Returns a length-three vector in
        units of kBT/(e·Å).
        """
        return self.batch_gradient(np.asarray(point, dtype=float).reshape(1, 3))[0]

    def force_on_charge(self, point: np.ndarray, charge: float) -> np.ndarray:
        """Force on a point charge at the given position, F = -q ∇φ, in units of kBT/Å."""
        return -charge * self.gradient(point)

    # Vectorised batch methods, roughly 50 to 100 times faster than per-atom loops.
    def batch_interpolate(self, points: np.ndarray) -> np.ndarray:
        """
        Trilinear interpolation of the potential at N points at once. The input
        points is an (N, 3) array of coordinates in Å, and the result is a
        length-N array of potential values in units of kBT/e.
        """
        pts = np.asarray(points, dtype=float)  # (N,3)
        idx = (pts - self.origin) * self._inv_dx  # (N,3) fractional indices
        # Defensive handling of NaN and infinite values. The upstream Brownian-
        # dynamics propagators can transiently produce non-finite chain
        # positions when the WCA forces are large. Those positions correspond to
        # non-physical states that the caller rejects on the next step. We
        # pre-filter the non-finite entries so the cast below is clean and does
        # not flood the SLURM logs with RuntimeWarnings. The valid mask further
        # down additionally catches out-of-bounds indices.
        finite_mask = np.isfinite(idx).all(axis=1)
        idx_safe = np.where(finite_mask[:, None], idx, 0.0)
        with np.errstate(invalid="ignore"):
            i0 = np.floor(idx_safe[:, 0]).astype(int)
            j0 = np.floor(idx_safe[:, 1]).astype(int)
            k0 = np.floor(idx_safe[:, 2]).astype(int)
        nx, ny, nz = self.data.shape
        valid = finite_mask & (
            (i0 >= 0)
            & (i0 < nx - 1)
            & (j0 >= 0)
            & (j0 < ny - 1)
            & (k0 >= 0)
            & (k0 < nz - 1)
        )
        # Use idx_safe, which is zero-padded at the non-finite positions, so that
        # fx, fy, and fz stay finite. The non-finite entries are masked out below
        # by the valid mask, so their numeric value does not matter.
        fx = idx_safe[:, 0] - i0
        fy = idx_safe[:, 1] - j0
        fz = idx_safe[:, 2] - k0
        out = np.zeros(len(pts))
        v = valid
        d = self.data
        out[v] = (
            d[i0[v], j0[v], k0[v]] * (1 - fx[v]) * (1 - fy[v]) * (1 - fz[v])
            + d[i0[v] + 1, j0[v], k0[v]] * fx[v] * (1 - fy[v]) * (1 - fz[v])
            + d[i0[v], j0[v] + 1, k0[v]] * (1 - fx[v]) * fy[v] * (1 - fz[v])
            + d[i0[v], j0[v], k0[v] + 1] * (1 - fx[v]) * (1 - fy[v]) * fz[v]
            + d[i0[v] + 1, j0[v] + 1, k0[v]] * fx[v] * fy[v] * (1 - fz[v])
            + d[i0[v] + 1, j0[v], k0[v] + 1] * fx[v] * (1 - fy[v]) * fz[v]
            + d[i0[v], j0[v] + 1, k0[v] + 1] * (1 - fx[v]) * fy[v] * fz[v]
            + d[i0[v] + 1, j0[v] + 1, k0[v] + 1] * fx[v] * fy[v] * fz[v]
        )
        return out

    def batch_gradient(self, points: np.ndarray) -> np.ndarray:
        """
        Central-difference gradient of the potential at N points at once.
        Returns an (N, 3) array of gradient vectors in units of kBT/(e·Å).
        """
        pts = np.asarray(points, dtype=float)
        h = np.diag(self.delta) * 0.5  # Half the grid spacing along each axis.
        grad = np.zeros_like(pts)
        for i in range(3):
            dp = pts.copy()
            dp[:, i] += h[i]
            dm = pts.copy()
            dm[:, i] -= h[i]
            grad[:, i] = (self.batch_interpolate(dp) - self.batch_interpolate(dm)) / (
                2 * h[i]
            )
        return grad

    def batch_force_on_charges(
        self, points: np.ndarray, charges: np.ndarray
    ) -> np.ndarray:
        """
        Force on N point charges, F_i = -q_i ∇φ(r_i). The input points is an
        (N, 3) array of positions in Å and charges is a length-N array of
        charges in units of the elementary charge. The result is an (N, 3) array
        of forces in units of kBT/Å.
        """
        grad = self.batch_gradient(points)  # (N,3) gradient vectors
        return -charges[:, None] * grad  # (N,3) forces

    def __repr__(self) -> str:
        nx, ny, nz = self.data.shape
        return (
            f"DXGrid({nx}×{ny}×{nz}, origin={self.origin}, "
            f"spacing={np.diag(self.delta)})"
        )
