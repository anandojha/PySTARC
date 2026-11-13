"""
Grid-filtered atom-pair overlap check for the GPU batch simulator.

A ligand atom j at position y overlaps the receptor when some receptor atom i
satisfies |y - x_i| < f (r_i + r_j), where f is the overlap fraction. With

    g(y) = min_i ( |y - x_i| - f r_i ),

the test reads g(y) < f r_j, so a single map of g over the receptor serves
every ligand atom. Each term of the minimum changes by at most the distance
moved, and so does g. With g stored on a cubic grid of spacing h, the value at
the nearest grid point g_n therefore bounds the true value from below by
g_n - h √3 / 2. A ligand atom with g_n - h √3 / 2 >= f r_j cannot overlap any
receptor atom. Only the few atoms closer to the threshold are tested against
the receptor atoms directly, with the same arithmetic as the all-pairs test, so
the decisions are identical to it. The work per step falls from N_lig × N_rec
pair distances to N_lig grid lookups plus the direct tests.

The receptor is fixed in the simulation frame, so the map is built once. The
arrays may be NumPy or CuPy, and the module passed as xp does the work. With
CuPy the map and the clearance are CUDA kernels. They round every operation as
the array code below does when CuPy runs it with its default reduction
accelerator (CUB), so their values are the same bits. With CUPY_ACCELERATORS
empty CuPy adds the three squares of a distance in another order, and the
array code then rounds some values differently from the kernels.
"""

from __future__ import annotations
import math
import numpy as np

_KERNELS = r"""
// (a^2 + b^2) + c^2 with every product and sum rounded and no fused
// multiply-add. CuPy sums a last axis of three in this order with its default
// reduction accelerator (CUB), and in another order without it.
__device__ double sum_sq(const double a, const double b, const double c)
{
    return __dadd_rn(__dadd_rn(__dmul_rn(a, a), __dmul_rn(b, b)), __dmul_rn(c, c));
}

// One thread per grid point; the receptor atoms pass through shared memory in
// tiles of 256 (blockDim.x is 256).
extern "C" __global__
void build_map(const double* rec_pos, const double* f_r, const int n_rec,
               const double ox, const double oy, const double oz, const double h,
               const long long nx, const long long ny, const long long nz,
               double* grid)
{
    __shared__ double tile[4 * 256];
    const long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const double x = __dadd_rn(ox, __dmul_rn(h, (double)(p / (ny * nz))));
    const double y = __dadd_rn(oy, __dmul_rn(h, (double)((p / nz) % ny)));
    const double z = __dadd_rn(oz, __dmul_rn(h, (double)(p % nz)));
    double g = 1.0 / 0.0;
    for (int a0 = 0; a0 < n_rec; a0 += 256) {
        const int a = a0 + threadIdx.x;
        __syncthreads();
        if (a < n_rec) {
            tile[4 * threadIdx.x] = rec_pos[3 * a];
            tile[4 * threadIdx.x + 1] = rec_pos[3 * a + 1];
            tile[4 * threadIdx.x + 2] = rec_pos[3 * a + 2];
            tile[4 * threadIdx.x + 3] = f_r[a];
        }
        __syncthreads();
        const int na = min(256, n_rec - a0);
        for (int b = 0; b < na; ++b) {
            const double* t = tile + 4 * b;
            g = fmin(g, sqrt(sum_sq(x - t[0], y - t[1], z - t[2])) - t[3]);
        }
    }
    if (p < nx * ny * nz) grid[p] = g;
}

// One thread per pose: the smallest room of its atoms, clipped at zero.
extern "C" __global__
void clearance(const double* lig_pos, const double* lig_rad, const double* grid,
               const int n, const int m,
               const double ox, const double oy, const double oz, const double h,
               const long long nx, const long long ny, const long long nz,
               const double slack, const double frac,
               const double lox, const double loy, const double loz,
               const double hix, const double hiy, const double hiz,
               const double fr_max, const double cx, const double cy, const double cz,
               const double r_ext, double* out)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n) return;
    double room = 1.0 / 0.0;
    for (int j = 0; j < m; ++j) {
        const double* q = lig_pos + 3 * ((long long)t * m + j);
        const double fr = __dmul_rn(frac, lig_rad[j]);
        const long long i = (long long)rint((q[0] - ox) / h);
        const long long k = (long long)rint((q[1] - oy) / h);
        const long long l = (long long)rint((q[2] - oz) / h);
        double r;
        if (i >= 0 && i < nx && k >= 0 && k < ny && l >= 0 && l < nz) {
            r = grid[(i * ny + k) * nz + l] - slack - fr;
        } else {
            const double bx = fmax(fmax(lox - q[0], q[0] - hix), 0.0);
            const double by = fmax(fmax(loy - q[1], q[1] - hiy), 0.0);
            const double bz = fmax(fmax(loz - q[2], q[2] - hiz), 0.0);
            const double d_in = sqrt(sum_sq(bx, by, bz));
            // CuPy evaluates d ** 2 as pow(d, 2.0), which is not always d * d.
            const double px = pow(q[0] - cx, 2.0), py = pow(q[1] - cy, 2.0);
            const double d_c = sqrt(__dadd_rn(__dadd_rn(px, py), pow(q[2] - cz, 2.0)));
            r = fmax(d_in - fr_max - fr, d_c - r_ext - fr);
        }
        room = fmin(room, r);
    }
    out[t] = fmax(room, 0.0);
}
"""


class OverlapGrid:
    """Map of g(y) over the receptor for the atom-pair overlap test.

    rec_pos has shape (N_rec, 3) and rec_radii shape (N_rec,). fraction is the
    overlap fraction f, max_lig_radius the largest ligand atom radius, and
    spacing the grid spacing h in angstrom.
    """

    def __init__(
        self,
        rec_pos,
        rec_radii,
        fraction: float,
        max_lig_radius: float,
        spacing: float = 0.3,
        xp=np,
        margin: float = 0.0,
    ):
        self.xp = xp
        self.fraction = float(fraction)
        self.spacing = float(spacing)
        self.rec_pos = xp.asarray(rec_pos, dtype=xp.float64)
        self.rec_radii = xp.asarray(rec_radii, dtype=xp.float64)
        # Largest drop of g between a point and its nearest grid point, padded
        # for the rounding in the stored values.
        self.slack = self.spacing * math.sqrt(3.0) / 2.0 + 1e-6
        # A ligand atom farther than reach from every receptor atom cannot
        # overlap, so the grid covers the receptor with reach to spare on every
        # side, and atoms off the grid are clear. The margin (Å) extends the map
        # further, so that clearance() reads the room of atoms near the surface
        # from the map instead of the coarser bounds used off the map.
        reach = margin + (
            self.fraction * (float(self.rec_radii.max()) + float(max_lig_radius))
            + self.slack
        )
        lo = self.rec_pos.min(axis=0) - reach
        hi = self.rec_pos.max(axis=0) + reach
        self.origin = lo
        self.shape = xp.ceil((hi - lo) / self.spacing).astype(xp.int64) + 1
        if xp is not np:
            module = xp.RawModule(code=_KERNELS)
            self._build_map = module.get_function("build_map")
            self._clearance = module.get_function("clearance")
            self._map_args = tuple(np.float64(v) for v in self.origin.tolist()) + (
                np.float64(self.spacing),
            ) + tuple(np.int64(n) for n in self.shape.tolist())
        self.grid = self._build()

    def _build(self, budget: int = 1 << 25):
        """Return g at every grid point, shape (nx, ny, nz)."""
        xp = self.xp
        nx, ny, nz = (int(n) for n in self.shape)
        total = nx * ny * nz
        grid = xp.empty(total, dtype=xp.float64)
        f_r = self.fraction * self.rec_radii
        if xp is not np:
            rec_pos = xp.ascontiguousarray(self.rec_pos)
            self._build_map(
                ((total + 255) // 256,),
                (256,),
                (rec_pos, f_r, np.int32(rec_pos.shape[0])) + self._map_args + (grid,),
            )
            return grid.reshape(nx, ny, nz)
        step = max(1, budget // (3 * self.rec_pos.shape[0]))
        for p0 in range(0, total, step):
            p1 = min(p0 + step, total)
            flat = xp.arange(p0, p1, dtype=xp.int64)
            ijk = xp.stack([flat // (ny * nz), (flat // nz) % ny, flat % nz], axis=1)
            pts = self.origin + self.spacing * ijk.astype(xp.float64)
            diff = pts[:, None, :] - self.rec_pos[None, :, :]
            dist = xp.sqrt((diff * diff).sum(axis=-1))
            grid[p0:p1] = (dist - f_r[None, :]).min(axis=1)
        return grid.reshape(nx, ny, nz)

    def clearance(self, lig_pos, lig_radii):
        """Return a (n,) lower bound on the room each pose has before it overlaps.

        For a ligand atom at y the room is g(y) - fraction * r_lig, the distance
        by which the atom can approach the receptor before the overlap test
        fires; the grid error (slack) is taken off. An atom off the map is at
        least its distance d to the bounding box of the receptor atoms from
        every receptor atom, so d - fraction (r_rec_max + r_lig) is a lower
        bound for it. The pose value is the smallest atom value, clipped at
        zero. lig_pos has shape (n, N_lig, 3).
        """
        xp = self.xp
        if xp is not np:
            return self._clearance_gpu(lig_pos, lig_radii)
        lig_radii = xp.asarray(lig_radii, dtype=xp.float64)
        ijk = xp.rint((lig_pos - self.origin) / self.spacing).astype(xp.int64)
        on_grid = ((ijk >= 0) & (ijk < self.shape)).all(axis=-1)
        ijk = xp.minimum(xp.maximum(ijk, 0), self.shape - 1)
        g_n = self.grid[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        room = g_n - self.slack - self.fraction * lig_radii[None, :]
        lo_in, hi_in, fr_max, c, r_ext = self._clearance_consts()
        out = xp.maximum(xp.maximum(lo_in - lig_pos, lig_pos - hi_in), 0.0)
        d_in = xp.sqrt((out * out).sum(axis=-1))
        room_off = d_in - fr_max - self.fraction * lig_radii[None, :]
        d_c = xp.sqrt(((lig_pos - c) ** 2).sum(axis=-1))
        room_off = xp.maximum(room_off, d_c - r_ext - self.fraction * lig_radii[None, :])
        room = xp.where(on_grid, room, room_off)
        return xp.maximum(room.min(axis=1), 0.0)

    def _clearance_gpu(self, lig_pos, lig_radii):
        """clearance() as one CUDA kernel."""
        xp = self.xp
        lig_pos = xp.ascontiguousarray(lig_pos, dtype=xp.float64)
        n, m = lig_pos.shape[0], lig_pos.shape[1]
        lig_radii = xp.asarray(lig_radii, dtype=xp.float64)
        lig_radii = xp.ascontiguousarray(xp.broadcast_to(lig_radii, (m,)))
        out = xp.empty(n, dtype=xp.float64)
        if not hasattr(self, "_clr_args"):
            lo_in, hi_in, fr_max, c, r_ext = self._clearance_consts()
            self._clr_args = tuple(
                np.float64(v)
                for v in [self.slack, self.fraction, *lo_in.tolist(), *hi_in.tolist(),
                          fr_max, *c.tolist(), r_ext]
            )
        if n:
            self._clearance(
                ((n + 127) // 128,),
                (128,),
                (lig_pos, lig_radii, self.grid, np.int32(n), np.int32(m))
                + self._map_args + self._clr_args + (out,),
            )
        return out

    def _clearance_consts(self):
        xp = self.xp
        if getattr(self, "_clr_consts", None) is None:
            # Computed once: the receptor bounding box, the largest scaled
            # radius, and the sphere around the receptor centroid that holds
            # every receptor atom with its scaled radius.
            c = self.rec_pos.mean(axis=0)
            r_ext = float(
                (
                    xp.sqrt(((self.rec_pos - c) ** 2).sum(axis=1))
                    + self.fraction * self.rec_radii
                ).max()
            )
            self._clr_consts = (
                self.rec_pos.min(axis=0),
                self.rec_pos.max(axis=0),
                self.fraction * float(self.rec_radii.max()),
                c,
                r_ext,
            )
        return self._clr_consts

    def overlaps(self, lig_pos, lig_radii, budget: int = 1 << 22):
        """Return a (n,) boolean array, true where any atom of pose n overlaps.

        lig_pos has shape (n, N_lig, 3) and lig_radii shape (N_lig,).
        """
        xp = self.xp
        n, m = lig_pos.shape[0], lig_pos.shape[1]
        lig_radii = xp.asarray(lig_radii, dtype=xp.float64)
        limit = self.fraction * lig_radii
        hit = xp.zeros(n, dtype=xp.bool_)
        step = max(1, budget // m)
        for t0 in range(0, n, step):
            t1 = min(t0 + step, n)
            ijk = xp.rint((lig_pos[t0:t1] - self.origin) / self.spacing).astype(
                xp.int64
            )
            on_grid = ((ijk >= 0) & (ijk < self.shape)).all(axis=-1)
            ijk = xp.minimum(xp.maximum(ijk, 0), self.shape - 1)
            g_n = self.grid[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
            near = on_grid & (g_n - self.slack < limit[None, :])
            if not bool(near.any()):
                continue
            ti, aj = xp.nonzero(near)
            ti = ti + t0
            close = self._pairwise(lig_pos[ti, aj], lig_radii[aj])
            hit[ti[close]] = True
        return hit

    def _pairwise(self, pos, radii, budget: int = 1 << 25):
        """All-pairs test of single atoms, (k, 3) and (k,), against the receptor."""
        xp = self.xp
        k = pos.shape[0]
        out = xp.zeros(k, dtype=xp.bool_)
        step = max(1, budget // (3 * self.rec_pos.shape[0]))
        for c0 in range(0, k, step):
            c1 = min(c0 + step, k)
            diff = pos[c0:c1, None, :] - self.rec_pos[None, :, :]
            dist_sq = (diff * diff).sum(axis=-1)
            thresh = self.fraction * (radii[c0:c1, None] + self.rec_radii[None, :])
            out[c0:c1] = (dist_sq < thresh * thresh).any(axis=1)
        return out
