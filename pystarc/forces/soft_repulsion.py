"""
Soft excluded-volume repulsion between the receptor and the ligand atoms.

Every receptor atom i and ligand atom j interact through the purely repulsive
potential of BrownDye 2,

    V_ij(r) = kT (a_ij / r)^12,   a_ij = 0.9345 (r_i + r_j),

with the PQR radii of both atoms, hydrogens included. The pair interacts only
when r < 2 a_ij, where V has fallen to 2^-12 kT. The force on the ligand atom is

    F_j = sum_i 12 kT (a_ij / r)^12 / r^2 (x_j - x_i),

directed away from the receptor atom. Energies are in kT and forces in kT/Å,
the units of the other ligand forces. The stiffness of each pair is the radial
second derivative V'' = 156 kT (a_ij / r)^12 / r^2; summed over the pairs of a
pose it bounds the curvature of the wall that the pose feels.

The receptor is fixed in the simulation frame, so its atoms are sorted once
into cubic cells of the largest cutoff length. A ligand atom then interacts
only with the receptor atoms of its own cell and the 26 neighbouring cells. On
the GPU each (trajectory, ligand atom) pair is one CUDA thread; the per-atom
forces are reduced to the force and the torque about the ligand centroid of
each trajectory.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    _CUPY = True
except Exception:  # pragma: no cover - CuPy is optional
    cp = None
    _CUPY = False

# Contact distance a_ij = A_FACTOR (r_i + r_j) and interaction cutoff
# CUTOFF × a_ij of the BrownDye 2 repulsion.
A_FACTOR = 0.9345
CUTOFF = 2.0

_KERNEL_CODE = r"""
extern "C" __global__
void soft_repulsion_kernel(
    const double* lig_pos, const double* lig_rad,
    const double* rec_pos, const double* rec_rad, const int* cell_start,
    const double ox, const double oy, const double oz, const double h,
    const int nx, const int ny, const int nz,
    const double afac, const double cut,
    const int n_traj, const int n_lig,
    double* f_out, double* e_out, double* k_out)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_traj * n_lig) return;
    const int j = tid % n_lig;
    const double x = lig_pos[3 * tid];
    const double y = lig_pos[3 * tid + 1];
    const double z = lig_pos[3 * tid + 2];
    const double rj = lig_rad[j];
    const int cx = (int)floor((x - ox) / h);
    const int cy = (int)floor((y - oy) / h);
    const int cz = (int)floor((z - oz) / h);
    double fx = 0.0, fy = 0.0, fz = 0.0, e = 0.0, kk = 0.0;
    for (int ix = cx - 1; ix <= cx + 1; ++ix) {
        if (ix < 0 || ix >= nx) continue;
        for (int iy = cy - 1; iy <= cy + 1; ++iy) {
            if (iy < 0 || iy >= ny) continue;
            for (int iz = cz - 1; iz <= cz + 1; ++iz) {
                if (iz < 0 || iz >= nz) continue;
                const int c = (ix * ny + iy) * nz + iz;
                for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                    const double dx = x - rec_pos[3 * k];
                    const double dy = y - rec_pos[3 * k + 1];
                    const double dz = z - rec_pos[3 * k + 2];
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double a = afac * (rj + rec_rad[k]);
                    const double rc = cut * a;
                    if (r2 < rc * rc && r2 > 0.0) {
                        const double ar2 = a * a / r2;
                        const double ar6 = ar2 * ar2 * ar2;
                        const double ar12 = ar6 * ar6;
                        const double fox = 12.0 * ar12 / r2;
                        e += ar12;
                        kk += 13.0 * fox;
                        fx += fox * dx;
                        fy += fox * dy;
                        fz += fox * dz;
                    }
                }
            }
        }
    }
    f_out[3 * tid] = fx;
    f_out[3 * tid + 1] = fy;
    f_out[3 * tid + 2] = fz;
    e_out[tid] = e;
    k_out[tid] = kk;
}
"""


def pair_terms(r, a):
    """Return the energy (kT) and |F|/r (kT/Å²) of one pair at distance r.

    Beyond the cutoff CUTOFF × a both are zero.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    ar12 = (a / r) ** 12
    inside = r < CUTOFF * a
    return np.where(inside, ar12, 0.0), np.where(inside, 12.0 * ar12 / (r * r), 0.0)


def repulsion_reference(lig_pos, lig_radii, rec_pos, rec_radii, chunk=200_000):
    """All-pairs evaluation on the CPU, used as the reference.

    lig_pos has shape (n, N_lig, 3). Returns the per-atom forces (n, N_lig, 3),
    energies (n, N_lig) and stiffnesses (n, N_lig).
    """
    lig_pos = np.asarray(lig_pos, dtype=np.float64)
    n, m, _ = lig_pos.shape
    flat = lig_pos.reshape(-1, 3)
    rad = np.tile(np.asarray(lig_radii, dtype=np.float64), n)
    rec_pos = np.asarray(rec_pos, dtype=np.float64)
    rec_radii = np.asarray(rec_radii, dtype=np.float64)
    f = np.zeros_like(flat)
    e = np.zeros(len(flat))
    k = np.zeros(len(flat))
    step = max(1, chunk // max(1, len(rec_pos)))
    for s0 in range(0, len(flat), step):
        d = flat[s0 : s0 + step, None, :] - rec_pos[None, :, :]
        r = np.sqrt((d * d).sum(axis=-1))
        a = A_FACTOR * (rad[s0 : s0 + step, None] + rec_radii[None, :])
        ee, fox = pair_terms(r, a)
        e[s0 : s0 + step] = ee.sum(axis=1)
        k[s0 : s0 + step] = 13.0 * fox.sum(axis=1)
        f[s0 : s0 + step] = (fox[:, :, None] * d).sum(axis=1)
    return f.reshape(n, m, 3), e.reshape(n, m), k.reshape(n, m)


class SoftRepulsion:
    """Soft repulsion of a fixed receptor on many ligand poses.

    rec_pos (N_rec, 3) and rec_radii (N_rec,) describe the receptor in the
    simulation frame; lig_radii (N_lig,) are the ligand radii. Atoms with a
    zero or negative radius (ghost atoms) take no part. Calling the object with
    lig_pos (n, N_lig, 3) and centroids (n, 3) returns the force (n, 3), the
    torque about the centroid (n, 3), the energy (n,) and the summed pair
    stiffness (n,) of each pose.
    """

    def __init__(self, rec_pos, rec_radii, lig_radii, xp=np):
        self.xp = xp
        rec_pos = np.asarray(rec_pos, dtype=np.float64)
        rec_radii = np.asarray(rec_radii, dtype=np.float64)
        lig_radii = np.asarray(lig_radii, dtype=np.float64)
        real = rec_radii > 0.0
        rec_pos, rec_radii = rec_pos[real], rec_radii[real]
        # A ghost ligand atom gets radius 0 and a cutoff of 2 × 0.9345 r_i, so
        # its contribution is switched off through a mask instead.
        self.lig_mask = lig_radii > 0.0
        self.lig_radii = np.where(self.lig_mask, lig_radii, 0.0)
        r_lig_max = float(self.lig_radii.max()) if self.lig_radii.size else 0.0
        # The cell edge is the longest cutoff of any pair, so a ligand atom only
        # meets receptor atoms of its own and the neighbouring cells.
        self.h = CUTOFF * A_FACTOR * (float(rec_radii.max()) + r_lig_max)
        self.origin = rec_pos.min(axis=0)
        span = rec_pos.max(axis=0) - self.origin
        self.shape = (np.floor(span / self.h).astype(np.int64) + 1).tolist()
        nx, ny, nz = self.shape
        ijk = np.floor((rec_pos - self.origin) / self.h).astype(np.int64)
        cell = (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]
        order = np.argsort(cell, kind="stable")
        counts = np.bincount(cell, minlength=nx * ny * nz)
        self.cell_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.rec_pos = rec_pos[order]
        self.rec_radii = rec_radii[order]
        if xp is not np:
            self._kernel = xp.RawKernel(_KERNEL_CODE, "soft_repulsion_kernel")
            self._rec_pos_d = xp.asarray(self.rec_pos.ravel())
            self._rec_rad_d = xp.asarray(self.rec_radii)
            self._cell_start_d = xp.asarray(self.cell_start)
            self._lig_rad_d = xp.asarray(self.lig_radii)
            self._lig_mask_d = xp.asarray(self.lig_mask)

    def atom_terms(self, lig_pos):
        """Per-atom forces (n, N_lig, 3), energies (n, N_lig) and stiffnesses (n, N_lig)."""
        xp = self.xp
        if xp is np:
            f, e, k = self._cells_numpy(np.asarray(lig_pos, dtype=np.float64))
        else:
            lig_pos = xp.ascontiguousarray(lig_pos, dtype=xp.float64)
            n, m, _ = lig_pos.shape
            f = xp.zeros((n, m, 3), dtype=xp.float64)
            e = xp.zeros((n, m), dtype=xp.float64)
            k = xp.zeros((n, m), dtype=xp.float64)
            total = n * m
            if total:
                threads = 256
                blocks = (total + threads - 1) // threads
                nx, ny, nz = self.shape
                self._kernel(
                    (blocks,),
                    (threads,),
                    (
                        lig_pos, self._lig_rad_d, self._rec_pos_d, self._rec_rad_d,
                        self._cell_start_d,
                        np.float64(self.origin[0]), np.float64(self.origin[1]),
                        np.float64(self.origin[2]), np.float64(self.h),
                        np.int32(nx), np.int32(ny), np.int32(nz),
                        np.float64(A_FACTOR), np.float64(CUTOFF),
                        np.int32(n), np.int32(m), f, e, k,
                    ),
                )
            mask = self._lig_mask_d
            return f * mask[None, :, None], e * mask[None, :], k * mask[None, :]
        mask = self.lig_mask
        return f * mask[None, :, None], e * mask[None, :], k * mask[None, :]

    def _cells_numpy(self, lig_pos):
        """The cell-list search of the CUDA kernel, written with NumPy."""
        n, m, _ = lig_pos.shape
        f = np.zeros((n, m, 3))
        e = np.zeros((n, m))
        k = np.zeros((n, m))
        nx, ny, nz = self.shape
        c = np.floor((lig_pos - self.origin) / self.h).astype(np.int64)
        for t in range(n):
            for j in range(m):
                cx, cy, cz = c[t, j]
                for ix in range(cx - 1, cx + 2):
                    if ix < 0 or ix >= nx:
                        continue
                    for iy in range(cy - 1, cy + 2):
                        if iy < 0 or iy >= ny:
                            continue
                        for iz in range(cz - 1, cz + 2):
                            if iz < 0 or iz >= nz:
                                continue
                            cc = (ix * ny + iy) * nz + iz
                            k0, k1 = self.cell_start[cc], self.cell_start[cc + 1]
                            if k1 == k0:
                                continue
                            d = lig_pos[t, j] - self.rec_pos[k0:k1]
                            r = np.sqrt((d * d).sum(axis=1))
                            a = A_FACTOR * (self.lig_radii[j] + self.rec_radii[k0:k1])
                            ee, fox = pair_terms(r, a)
                            e[t, j] += ee.sum()
                            k[t, j] += 13.0 * fox.sum()
                            f[t, j] += (fox[:, None] * d).sum(axis=0)
        return f, e, k

    def __call__(self, lig_pos, centroids):
        xp = self.xp
        f_atom, e_atom, k_atom = self.atom_terms(lig_pos)
        force = f_atom.sum(axis=1)
        lever = lig_pos - centroids[:, None, :]
        torque = xp.cross(lever, f_atom, axis=2).sum(axis=1)
        return force, torque, e_atom.sum(axis=1), k_atom.sum(axis=1)


def surface_time_step(
    dt, room, force_mag, D, drift_max=0.5, room_min=0.1, stiffness=None, curv_max=0.05
):
    """Limit the time step of poses close to the receptor surface.

    room is the distance each pose can move before it reaches the contact
    distance of the excluded-volume model (zero in contact), force_mag the
    magnitude of the net force on the ligand in kT/Å and D the translational
    diffusion coefficient in Å²/ps. The step is the smallest of the given dt,
    the step whose 3D noise sqrt(6 D dt) is 0.7 of the room, dt = room²/(12 D),
    and the step whose drift D |F| dt is drift_max (Å). A pose in contact
    counts room_min (Å) of room, so its step stays finite when the run sets
    no minimum time step. When the summed wall stiffness K (kT/Å²) is given,
    the step also keeps D K dt below curv_max: an Euler step on a wall of
    curvature K spreads the walker by the factor 1 / (1 - D K dt / 2), and the
    force changes by about K × (step length) over one step, the quantity that
    the force-change backstep of BrownDye 2 limits. The minimum time step of
    the run is applied afterwards.
    """
    xp = cp.get_array_module(room) if _CUPY else np
    room = xp.maximum(room, room_min)
    dt_room = room * room / (12.0 * D)
    dt_drift = drift_max / (D * xp.maximum(force_mag, 1e-12))
    out = xp.minimum(dt, xp.minimum(dt_room, dt_drift))
    if stiffness is not None:
        out = xp.minimum(out, curv_max / (D * xp.maximum(stiffness, 1e-12)))
    return out


__all__ = [
    "A_FACTOR",
    "CUTOFF",
    "SoftRepulsion",
    "repulsion_reference",
    "pair_terms",
    "surface_time_step",
]
