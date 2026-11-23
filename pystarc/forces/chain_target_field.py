"""Pure-NumPy CPU evaluator for target fields acting on a flexible chain.

Evaluates nested electrostatic DX grids with optional Yukawa multipole
far-field fallback, Born desolvation DX grids (direction 1), pairwise
reverse Born desolvation (direction 2), steric WCA repulsion with optional
core capping, hard-sphere overlap checks, and optional internal generalized
Born forces.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from pystarc.forces.chain_gb import WATER_DIELECTRIC, chain_full_gb_force
from pystarc.global_defs.constants import (
    COULOMB_KCAL,
    KBT_KCAL,
    VACUUM_PERMITTIVITY_KBT,
)

_BORN2_CUTOFF = 15.0
_BORN2_DEN_FLOOR = 0.5
_TWO_ONE_SIXTH = 2.0 ** (1.0 / 6.0)


def _extract_target_spheres(target) -> Tuple[np.ndarray, np.ndarray]:
    """Return positions and radii of target atoms with radius >= 1e-10 A."""
    if target is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    xyz = np.array([[a.x, a.y, a.z] for a in target.atoms], dtype=np.float64)
    rad = np.array([a.radius for a in target.atoms], dtype=np.float64)
    live = rad >= 1.0e-10
    return xyz.reshape(-1, 3)[live], rad[live]


def _eval_wca_pairs_cpu(
    r2: np.ndarray,
    sig: np.ndarray,
    eps: float,
    core_frac: float,
) -> np.ndarray:
    """Return pairwise force-over-r scalar multiplier for WCA with optional core cap."""
    rcut = _TWO_ONE_SIXTH * sig
    active = (r2 > 1.0e-20) & (r2 < rcut * rcut)
    f_over_r = np.zeros_like(r2)
    if not np.any(active):
        return f_over_r
    if core_frac > 0.0:
        r_core = core_frac * sig
        in_core = active & (r2 < r_core * r_core)
        out_core = active & (~in_core)
        if np.any(in_core):
            inv_f2 = 1.0 / (core_frac * core_frac)
            inv_f6 = inv_f2 * inv_f2 * inv_f2
            f_core = (4.0 * eps / r_core[in_core]) * (
                12.0 * inv_f6 * inv_f6 - 6.0 * inv_f6
            )
            f_over_r[in_core] = f_core / np.sqrt(r2[in_core])
        if np.any(out_core):
            sr2 = (sig[out_core] * sig[out_core]) / r2[out_core]
            sr6 = sr2 * sr2 * sr2
            f_over_r[out_core] = (
                4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r2[out_core]
            )
    else:
        sr2 = (sig[active] * sig[active]) / r2[active]
        sr6 = sr2 * sr2 * sr2
        f_over_r[active] = 4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r2[active]
    return f_over_r


def _eval_single_elec_grid(
    pts: np.ndarray,
    charges: np.ndarray,
    grid,
    want_energy: bool,
) -> Tuple[np.ndarray, float]:
    """Evaluate electrostatic force and optional energy on a single DX grid."""
    forces = grid.batch_force_on_charges(pts, charges)
    if want_energy:
        potentials = grid.batch_interpolate(pts)
        energy = float(np.sum(charges * potentials))
    else:
        energy = 0.0
    return forces, energy


def _eval_single_born_grid(
    pts: np.ndarray,
    charges: np.ndarray,
    grid,
    alpha: float,
    want_energy: bool,
) -> Tuple[np.ndarray, float]:
    """Evaluate Born desolvation force and optional energy on a single DX grid."""
    grads = grid.batch_gradient(pts)
    q2 = charges * charges
    forces = (-alpha * q2)[:, None] * grads
    if want_energy:
        potentials = grid.batch_interpolate(pts)
        energy = float(alpha * np.sum(q2 * potentials))
    else:
        energy = 0.0
    return forces, energy


class ChainTargetField:
    """Target grids, reverse Born desolvation, steric WCA, and overlap evaluator on CPU."""

    def __init__(
        self,
        target,
        chain_charges: np.ndarray,
        chain_radii: np.ndarray,
        elec_grids: Optional[list] = None,
        born_grids: Optional[list] = None,
        desolvation_alpha: float = 1.0,
        use_far_field: bool = False,
        target_charge: Optional[float] = None,
        debye_length: float = 7.858,
        sdie: float = 78.0,
        kT_scale: float = 1.0,
        multipole_expansion=None,
        gb_exclude_mask: Optional[np.ndarray] = None,
        xp=None,
        grid_float32: bool = False,
        overlap_scale: float = 1.0,
        wca_core_cap_fraction: float = 0.0,
        enable_born2: bool = False,
        born_alpha: float = 1.0,
        pdie: float = 4.0,
    ):
        if xp is not None and xp is not np:
            raise ValueError("ChainTargetField requires the NumPy CPU backend")
        self.xp = np
        self.target = target
        self.alpha = float(desolvation_alpha)
        self.kT_scale = float(kT_scale)
        self.grid_float32 = bool(grid_float32)
        self.overlap_scale = float(overlap_scale)
        self.wca_core_cap_fraction = float(wca_core_cap_fraction)
        if not (0.0 <= self.wca_core_cap_fraction < _TWO_ONE_SIXTH):
            raise ValueError(
                f"wca_core_cap_fraction must lie in [0, 2^(1/6)), got {wca_core_cap_fraction}"
            )
        self.gb_exclude_mask = (
            np.asarray(gb_exclude_mask, dtype=bool)
            if gb_exclude_mask is not None
            else None
        )
        self.chain_charges_np = np.asarray(chain_charges, dtype=np.float64)
        self.chain_radii_np = np.asarray(chain_radii, dtype=np.float64)
        self.chain_charges = self.chain_charges_np
        self.chain_radii = self.chain_radii_np
        self.n_atoms = len(self.chain_radii_np)

        self.elec_grids_cpu = sorted(
            list(elec_grids or []), key=lambda g: float(np.prod(np.diag(g.delta)))
        )
        self.born_grids_cpu = sorted(
            list(born_grids or []), key=lambda g: float(np.prod(np.diag(g.delta)))
        )

        if target_charge is None:
            if target is None and multipole_expansion is not None:
                target_charge = float(getattr(multipole_expansion, "Q", 0.0))
            else:
                target_charge = 0.0
        target_charge = float(target_charge)

        self.use_far_field = (
            bool(use_far_field) and debye_length is not None and debye_length > 0.0
        )
        self.multipole = multipole_expansion
        eps_s = float(sdie) * VACUUM_PERMITTIVITY_KBT * self.kT_scale
        self._V_factor = target_charge / (4.0 * math.pi * eps_s) if eps_s > 0 else 0.0
        self._debye = float(debye_length) if debye_length is not None else 0.0
        has_mp = multipole_expansion is not None and (
            float(np.linalg.norm(multipole_expansion.dipole)) > 1.0e-9
            or float(np.linalg.norm(multipole_expansion.quadrupole)) > 1.0e-9
            or abs(float(getattr(multipole_expansion, "trace_moment", 0.0))) > 1.0e-9
        )
        self.has_far = int(
            self.use_far_field and (abs(target_charge) > 1.0e-9 or has_mp)
        )
        far_c = np.zeros(19, dtype=np.float64)
        if self.has_far:
            far_c[0] = self._V_factor
            far_c[1] = self._debye
            if multipole_expansion is not None:
                far_c[2] = 4.0 * math.pi * eps_s
                far_c[3] = float(getattr(multipole_expansion, "trace_moment", 0.0)) / (
                    6.0 * self._debye**2
                )
                far_c[4:7] = np.asarray(multipole_expansion.dipole, dtype=np.float64)
                far_c[7:16] = np.asarray(
                    multipole_expansion.quadrupole, dtype=np.float64
                ).ravel()
                far_c[16] = float(np.linalg.norm(multipole_expansion.dipole)) > 1.0e-9
                far_c[17] = (
                    float(np.linalg.norm(multipole_expansion.quadrupole)) > 1.0e-9
                )
                far_c[18] = (
                    abs(float(getattr(multipole_expansion, "trace_moment", 0.0)))
                    > 1.0e-12
                )
        self._far_c_np = far_c
        self.far_consts = far_c

        # Positive-radius target spheres for steric forces and overlap checks.
        rec_pos, rec_rad = _extract_target_spheres(target)
        self.has_spheres = rec_rad.size > 0
        self._spheres_cpu = (rec_pos, rec_rad)

        # Reverse Born desolvation (direction 2) from target charges desolvated by chain beads.
        self.enable_born2 = bool(enable_born2)
        self.born_alpha = float(born_alpha)
        self.pdie = float(pdie)
        self.sdie = float(sdie)
        self._born2_cutoff = _BORN2_CUTOFF
        self._born2_kappa = (
            0.0
            if (not debye_length or debye_length <= 0.0 or math.isinf(debye_length))
            else 1.0 / float(debye_length)
        )
        diel_fac = (self.sdie - self.pdie) / (self.sdie * (2.0 * self.sdie + self.pdie))
        coul_kbt_t = COULOMB_KCAL / (KBT_KCAL * self.kT_scale)
        self.born2_prefactor = self.born_alpha * 0.5 * coul_kbt_t * diel_fac
        if self.enable_born2 and target is not None and len(target.atoms) > 0:
            chg_atoms = [a for a in target.atoms if abs(a.charge) > 1.0e-9]
            if chg_atoms:
                b2_pos = np.ascontiguousarray(
                    np.array(
                        [[a.x, a.y, a.z] for a in chg_atoms], dtype=np.float64
                    ).reshape(-1, 3)
                )
                b2_q2 = np.ascontiguousarray(
                    np.array([a.charge * a.charge for a in chg_atoms], dtype=np.float64)
                )
            else:
                b2_pos = np.zeros((0, 3), dtype=np.float64)
                b2_q2 = np.zeros(0, dtype=np.float64)
        else:
            b2_pos = np.zeros((0, 3), dtype=np.float64)
            b2_q2 = np.zeros(0, dtype=np.float64)

        self.has_born2 = b2_q2.size > 0
        self._born2_n_tgt = int(b2_q2.size)
        if b2_pos.size > 0:
            self._born2_min = b2_pos.min(axis=0).astype(np.float64)
            self._born2_max = b2_pos.max(axis=0).astype(np.float64)
        else:
            self._born2_min = np.zeros(3, dtype=np.float64)
            self._born2_max = np.zeros(3, dtype=np.float64)
        self._born2_pos_cpu = b2_pos
        self._born2_q2_cpu = b2_q2
        self.born2_pos = b2_pos
        self.born2_q2 = b2_q2

        if target is not None and len(target.atoms) > 0:
            self.target_all_xyz = np.array(
                [[a.x, a.y, a.z] for a in target.atoms], dtype=np.float64
            )
        else:
            self.target_all_xyz = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("xp", None)
        state["_xp_name"] = "numpy"
        return state

    def __setstate__(self, state):
        state = dict(state)
        xp_name = state.pop("_xp_name", "numpy")
        if xp_name != "numpy":
            raise ValueError(f"Unsupported serialized array backend {xp_name!r}")
        self.__dict__.update(state)
        self.xp = np

    @staticmethod
    def _effective_grid_bounds(g, margin: float):
        sp = np.diag(g.delta).astype(np.float64)
        sh = np.array(g.data.shape, dtype=np.float64)
        m_vec = np.minimum(float(margin), np.maximum(0.0, 0.5 * (sh - 2.0)))
        lo = np.where(m_vec > 0.0, g.origin + m_vec * sp, g.origin - 1.0e-12)
        hi = np.where(
            m_vec > 0.0,
            g.origin + (sh - 1.0 - m_vec) * sp,
            g.origin + (sh - 1.0) * sp,
        )
        return sp, lo, hi

    def _eval_far_cpu(self, sub_p: np.ndarray, sub_q: np.ndarray):
        c = self._far_c_np
        V, lam, fpe = c[0], c[1], c[2]
        p = c[4:7]
        Q = c[7:16].reshape(3, 3)
        r_raw = np.linalg.norm(sub_p, axis=1)
        valid = r_raw >= 1.0
        forces = np.zeros_like(sub_p)
        if not np.any(valid):
            return forces, 0.0
        vp = sub_p[valid]
        vq = sub_q[valid]
        sr = r_raw[valid]
        ex = np.exp(-sr / lam)
        h = vp / sr[:, None]
        phi = V * ex / sr
        dphi = V * ex * (-1.0 / (sr * sr) - 1.0 / (sr * lam))
        grad = dphi[:, None] * h
        if c[16] != 0.0:
            pr = h @ p
            a = 1.0 + sr / lam
            phi = phi + pr / (fpe * sr * sr) * a * ex
            dphi_d = (
                pr
                / fpe
                * ex
                * (-2.0 / (sr**3) - 2.0 / (sr * sr * lam) - 1.0 / (sr * lam * lam))
            )
            w = a * ex / (fpe * sr**3)
            grad = (
                grad + dphi_d[:, None] * h + w[:, None] * (p[None, :] - pr[:, None] * h)
            )
        if c[17] != 0.0:
            Qh = h @ Q.T
            rQr = np.sum(h * Qh, axis=1)
            b = 1.0 + sr / lam + sr * sr / (3.0 * lam * lam)
            phi = phi + rQr / (fpe * sr**3) * b * ex
            dphi_q = (
                rQr
                / fpe
                * ex
                * (
                    -3.0 / (sr**4)
                    - 3.0 / (sr**3 * lam)
                    - 4.0 / (3.0 * sr * sr * lam * lam)
                    - 1.0 / (3.0 * sr * lam**3)
                )
            )
            w = 2.0 * b * ex / (fpe * sr**4)
            grad = grad + dphi_q[:, None] * h + w[:, None] * (Qh - rQr[:, None] * h)
        if c[18] != 0.0:
            phi = phi + c[3] / (fpe * sr) * ex
            dphi_t = c[3] / fpe * ex * (-1.0 / (sr * sr) - 1.0 / (sr * lam))
            grad = grad + dphi_t[:, None] * h
        forces[valid] = -vq[:, None] * grad
        return forces, float(np.sum(vq * phi))

    def _eval_grid_cpu(self, world_pos: np.ndarray, is_born: bool, want_energy: bool):
        world_pos = np.asarray(world_pos, dtype=np.float64)
        B = world_pos.shape[0]
        forces = np.zeros(world_pos.shape, dtype=np.float64)
        energies = np.zeros(B, dtype=np.float64) if want_energy else None
        grids = self.born_grids_cpu if is_born else self.elec_grids_cpu
        if B == 0 or self.n_atoms == 0:
            return forces, energies
        if not grids and not (not is_born and self.has_far):
            return forces, energies
        if is_born and self.alpha == 0.0:
            return forces, energies
        for b in range(B):
            if len(grids) == 1 and not (not is_born and self.has_far):
                if is_born:
                    fb, eb = _eval_single_born_grid(
                        world_pos[b],
                        self.chain_charges_np,
                        grids[0],
                        self.alpha,
                        want_energy,
                    )
                else:
                    fb, eb = _eval_single_elec_grid(
                        world_pos[b],
                        self.chain_charges_np,
                        grids[0],
                        want_energy,
                    )
                forces[b] = fb
                if want_energy:
                    energies[b] = eb
            else:
                pts = world_pos[b]
                finite = np.all(np.isfinite(pts), axis=1)
                assigned = np.zeros(self.n_atoms, dtype=bool)
                eb_tot = 0.0
                for idx, g in enumerate(grids):
                    m = (
                        0.0
                        if ((is_born or not self.has_far) and idx == len(grids) - 1)
                        else (1.0 if is_born else 3.0)
                    )
                    _, lo, hi = self._effective_grid_bounds(g, m)
                    in_box = (
                        ~assigned
                        & finite
                        & (np.abs(self.chain_charges_np) > 0.0)
                        & np.all((pts > lo) & (pts < hi), axis=1)
                    )
                    if not in_box.any():
                        continue
                    sub_pts = pts[in_box]
                    sub_q = self.chain_charges_np[in_box]
                    if is_born:
                        fb, eb = _eval_single_born_grid(
                            sub_pts, sub_q, g, self.alpha, want_energy
                        )
                    else:
                        fb, eb = _eval_single_elec_grid(sub_pts, sub_q, g, want_energy)
                    forces[b, in_box] = fb
                    eb_tot += eb
                    assigned |= in_box
                if not is_born and self.has_far:
                    rem = ~assigned & finite & (np.abs(self.chain_charges_np) > 0.0)
                    if rem.any():
                        ff, fe = self._eval_far_cpu(
                            pts[rem], self.chain_charges_np[rem]
                        )
                        forces[b, rem] = ff
                        eb_tot += fe
                if want_energy:
                    energies[b] = eb_tot
        return forces, energies

    def _eval_born2_cpu(self, world_pos: np.ndarray, want_energy: bool):
        world_pos = np.asarray(world_pos, dtype=np.float64)
        B = world_pos.shape[0]
        forces = np.zeros((B, self.n_atoms, 3), dtype=np.float64)
        energies = np.zeros(B, dtype=np.float64) if want_energy else None
        if (
            not self.enable_born2
            or not self.has_born2
            or self.born2_prefactor == 0.0
            or B == 0
            or self.n_atoms == 0
        ):
            return forces, energies
        live_ch = self.chain_radii_np >= 1.0e-10
        if not np.any(live_ch):
            return forces, energies
        tgt_pos = self._born2_pos_cpu
        tgt_q2 = self._born2_q2_cpu
        rad_live = self.chain_radii_np[live_ch]
        a3 = rad_live * rad_live * rad_live
        floor2 = np.maximum(rad_live * rad_live, _BORN2_DEN_FLOOR)
        cut = self._born2_cutoff
        pad = cut + 1.0e-9
        c2 = cut * cut
        kap = self._born2_kappa
        pref = self.born2_prefactor
        live_indices = np.flatnonzero(live_ch)
        box_lo = self._born2_min - pad
        box_hi = self._born2_max + pad

        for b in range(B):
            pts = world_pos[b, live_ch]
            finite = np.all(np.isfinite(pts), axis=1)
            if not np.any(finite):
                continue
            pts_fin = pts[finite]
            near_bead = np.all((pts_fin >= box_lo) & (pts_fin <= box_hi), axis=1)
            if not np.any(near_bead):
                continue
            pts_ok = pts_fin[near_bead]
            p_lo = pts_ok.min(axis=0) - pad
            p_hi = pts_ok.max(axis=0) + pad
            near_tgt = np.all((tgt_pos >= p_lo) & (tgt_pos <= p_hi), axis=1)
            if not np.any(near_tgt):
                continue
            t_pos = tgt_pos[near_tgt]
            t_q2 = tgt_q2[near_tgt]
            a3_ok = a3[finite][near_bead]
            fl2_ok = floor2[finite][near_bead]
            dr = pts_ok[:, None, :] - t_pos[None, :, :]
            r2 = np.einsum("kmd,kmd->km", dr, dr)
            in_cut = r2 < c2
            if not np.any(in_cut):
                continue
            above = in_cut & (r2 > fl2_ok[:, None])
            f_ok = np.zeros_like(pts_ok)
            e_tot = 0.0
            if np.any(above):
                ik, im = np.nonzero(above)
                r2_a = r2[ik, im]
                r_a = np.sqrt(r2_a)
                kr = kap * r_a
                ex = np.exp(-2.0 * kr)
                opk = 1.0 + kr
                inv_r2 = 1.0 / r2_a
                inv_r4 = inv_r2 * inv_r2
                scale_a = pref * a3_ok[ik] * t_q2[im]
                if want_energy:
                    e_tot += float(np.sum(scale_a * opk * opk * ex * inv_r4))
                fac = (
                    scale_a
                    * 2.0
                    * ex
                    * opk
                    * (2.0 + 2.0 * kr + kr * kr)
                    * inv_r4
                    * inv_r2
                )
                fvec = fac[:, None] * dr[ik, im]
                for d in range(3):
                    f_ok[:, d] = np.bincount(
                        ik, weights=fvec[:, d], minlength=len(pts_ok)
                    )
            if want_energy:
                inside_fl = in_cut & (r2 <= fl2_ok[:, None])
                if np.any(inside_fl):
                    ik, im = np.nonzero(inside_fl)
                    fl2_i = fl2_ok[ik]
                    reff = np.sqrt(fl2_i)
                    kr = kap * reff
                    opk = 1.0 + kr
                    scale_i = pref * a3_ok[ik] * t_q2[im]
                    e_tot += float(
                        np.sum(
                            scale_i * opk * opk * np.exp(-2.0 * kr) / (fl2_i * fl2_i)
                        )
                    )
                energies[b] = e_tot
            forces[b, live_indices[finite][near_bead]] = f_ok
        return forces, energies

    def evaluate_elec(self, world_pos, want_energy: bool = False):
        """Evaluate electrostatic grid and far-field multipole forces and optional energies."""
        return self._eval_grid_cpu(world_pos, False, want_energy)

    def evaluate_born(self, world_pos, want_energy: bool = False):
        """Evaluate Born desolvation grid (direction 1) forces and optional energies."""
        return self._eval_grid_cpu(world_pos, True, want_energy)

    def evaluate_born2(self, world_pos, want_energy: bool = False):
        """Evaluate reverse Born desolvation (direction 2) forces and optional energies."""
        return self._eval_born2_cpu(world_pos, want_energy)

    def add_steric_forces(self, world_pos, forces, eps: float = 1.0) -> None:
        """Accumulate chain-target WCA repulsion forces into forces."""
        if forces.dtype != np.float64 or not forces.flags.c_contiguous:
            raise ValueError("forces must be a C-contiguous float64 array")
        if not self.has_spheres:
            return
        world_pos = np.asarray(world_pos, dtype=np.float64)
        B = world_pos.shape[0]
        if B == 0 or self.n_atoms == 0:
            return
        t_pos, t_rad = self._spheres_cpu
        live_ch = self.chain_radii_np >= 1.0e-10
        if not np.any(live_ch):
            return
        ch_rad = self.chain_radii_np[live_ch]
        sig = ch_rad[:, None] + t_rad[None, :]
        for b in range(B):
            pts = world_pos[b, live_ch]
            dr = pts[:, None, :] - t_pos[None, :, :]
            r2 = np.einsum("ikd,ikd->ik", dr, dr)
            f_over_r = _eval_wca_pairs_cpu(
                r2, sig, float(eps), self.wca_core_cap_fraction
            )
            nz = (r2 > 1.0e-20) & (f_over_r != 0.0)
            if not np.any(nz):
                continue
            ia, ka = np.nonzero(nz)
            fvec = f_over_r[ia, ka, None] * dr[ia, ka]
            F_act = np.zeros((len(ch_rad), 3), dtype=np.float64)
            for c in range(3):
                F_act[:, c] = np.bincount(ia, weights=fvec[:, c], minlength=len(ch_rad))
            forces[b, live_ch] += F_act

    def evaluate_external_forces(
        self,
        world_pos,
        soft_repulsion: bool = False,
        soft_repulsion_eps: float = 1.0,
        self_born: bool = False,
        gb_params: Optional[dict] = None,
        want_energy: bool = False,
    ):
        """Sum target electrostatic, Born, steric, and optional self-Born forces."""
        world_pos = np.asarray(world_pos, dtype=np.float64)
        forces, e_elec = self.evaluate_elec(world_pos, want_energy=want_energy)
        if self.born_grids_cpu and self.alpha != 0.0:
            f_born, e_born = self.evaluate_born(world_pos, want_energy=want_energy)
            forces = forces + f_born
        else:
            e_born = (
                np.zeros(world_pos.shape[0], dtype=np.float64) if want_energy else None
            )
        if self.enable_born2 and self.has_born2 and self.born2_prefactor != 0.0:
            f_b2, e_b2 = self.evaluate_born2(world_pos, want_energy=want_energy)
            forces = forces + f_b2
            if want_energy:
                e_born = e_born + e_b2
        if soft_repulsion and self.has_spheres:
            self.add_steric_forces(world_pos, forces, eps=soft_repulsion_eps)
        if self_born:
            gp = gb_params or {}
            eps_in = gp.get("eps_in", 1.0)
            eps_out = gp.get("eps_out", WATER_DIELECTRIC)
            coffdrop_active = gp.get("coffdrop_active", False)
            excl_mask = gp.get("exclude_pair_mask", self.gb_exclude_mask)
            obc_kwargs = gp.get("obc_kwargs")
            kT_scale = gp.get("kT_scale", self.kT_scale)
            kappa = gp.get("kappa", 0.0)
            for b in range(world_pos.shape[0]):
                f_gb, _ = chain_full_gb_force(
                    world_pos[b],
                    self.chain_charges_np,
                    self.chain_radii_np,
                    eps_in=eps_in,
                    eps_out=eps_out,
                    coffdrop_active=coffdrop_active,
                    exclude_pair_mask=excl_mask,
                    obc_kwargs=obc_kwargs,
                    kT_scale=kT_scale,
                    kappa=kappa,
                )
                forces[b] += f_gb
        return forces, e_elec, e_born

    def check_overlap(self, world_pos):
        """Return (B,) boolean array True where any bead overlaps a target sphere or is non-finite."""
        world_pos = np.asarray(world_pos, dtype=np.float64)
        B = world_pos.shape[0]
        if B == 0 or self.n_atoms == 0:
            return np.zeros(B, dtype=bool)
        if not self.has_spheres:
            return ~np.all(np.isfinite(world_pos), axis=(1, 2))
        out_b = np.zeros(B, dtype=bool)
        live = self.chain_radii_np >= 1.0e-10
        t_pos, t_rad = self._spheres_cpu
        scale = self.overlap_scale
        for b in range(B):
            pts = world_pos[b]
            if not np.all(np.isfinite(pts)):
                out_b[b] = True
            elif np.any(live):
                d = pts[live][:, None, :] - t_pos[None, :, :]
                r2 = np.einsum("ikd,ikd->ik", d, d)
                sig = scale * (self.chain_radii_np[live][:, None] + t_rad[None, :])
                out_b[b] = bool((r2 < sig * sig).any())
        return out_b
