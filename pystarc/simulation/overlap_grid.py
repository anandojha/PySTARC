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
arrays may be NumPy or CuPy, and the module passed as xp does the work.
"""

from __future__ import annotations
import math
import numpy as np


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
        self.grid = self._build()

    def _build(self, budget: int = 1 << 25):
        """Return g at every grid point, shape (nx, ny, nz)."""
        xp = self.xp
        nx, ny, nz = (int(n) for n in self.shape)
        total = nx * ny * nz
        grid = xp.empty(total, dtype=xp.float64)
        f_r = self.fraction * self.rec_radii
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
        lig_radii = xp.asarray(lig_radii, dtype=xp.float64)
        ijk = xp.rint((lig_pos - self.origin) / self.spacing).astype(xp.int64)
        on_grid = ((ijk >= 0) & (ijk < self.shape)).all(axis=-1)
        ijk = xp.minimum(xp.maximum(ijk, 0), self.shape - 1)
        g_n = self.grid[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        room = g_n - self.slack - self.fraction * lig_radii[None, :]
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
        lo_in, hi_in, fr_max, c, r_ext = self._clr_consts
        out = xp.maximum(xp.maximum(lo_in - lig_pos, lig_pos - hi_in), 0.0)
        d_in = xp.sqrt((out * out).sum(axis=-1))
        room_off = d_in - fr_max - self.fraction * lig_radii[None, :]
        d_c = xp.sqrt(((lig_pos - c) ** 2).sum(axis=-1))
        room_off = xp.maximum(room_off, d_c - r_ext - self.fraction * lig_radii[None, :])
        room = xp.where(on_grid, room, room_off)
        return xp.maximum(room.min(axis=1), 0.0)

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
