"""
GPU force evaluation, overlap checks, and internal propagation for flexible-chain
Brownian dynamics.

Two-body interactions within the chain (harmonic bonds, elastic network springs,
tabulated COFFDROP pair potentials, and intra-chain WCA repulsion on pairs that
have no table) are stored as symmetric compressed-sparse-row neighbor lists
sorted by neighbor index. Each thread owns one bead and sums its two-body forces
in registers without atomic writes. Three-body angles and four-body
Blondel-Karplus torsions are partitioned into conflict-free color groups once at
construction so that threads in each group write to disjoint beads in a fixed
order.

For chains of up to 600 beads, all internal Ermak-McCammon steps of an outer
step run in shared memory inside a single kernel launch, with one thread block
per active trajectory. Each substep evaluates the bonded and non-bonded forces,
adds the body-frame external force when requested, caps any per-bead drift above
5 A, draws counter-based Philox Gaussian increments, projects constraints with
SHAKE, centers the body coordinates at the origin, and removes rigid rotation by
an unweighted Kabsch fit computed from a two-sided Jacobi 3x3 singular value
decomposition. Longer chains run the same device routine against global-memory
work buffers.

Target electrostatic and Born desolvation grids are evaluated per bead with
probe clamping at the grid boundaries, matching DXGrid.batch_gradient. Target
spheres of positive radius are placed into a cubic spatial cell list of width
2^(1/6) (r_chain,max + r_target,max) so that both bead-target WCA forces and
hard-sphere overlap checks inspect only the 27 neighboring cells.
"""

from __future__ import annotations

from pystarc.forces.chain_gb import (
    COULOMB_K_KBT_A,
    DEFAULT_HCT_SCALE,
    DEFAULT_OBC_ALPHA,
    DEFAULT_OBC_BETA,
    DEFAULT_OBC_GAMMA,
    DEFAULT_OBC_OFFSET,
    WATER_DIELECTRIC,
)
from pystarc.simulation.coffdrop_chain import (
    ChainCommon,
    ChainState,
    _chain_idx,
    _spline_table,
    _term_arrays,
    bonded_exclusions,
    compute_chain_forces,
    satisfy_constraints_hybrid,
)
from pystarc.simulation.gpu_random import DEVICE_CODE as _RNG_DEVICE_CODE
from pystarc.global_defs.constants import VACUUM_PERMITTIVITY_KBT
from pystarc.simulation.chain_simulator import target_spheres
from pystarc.motion.do_bd_step import WATER_VISCOSITY
from typing import List, Optional, Tuple
import numpy as np
import math

try:
    import cupy as cp

    _CUPY = True
except ImportError:
    cp = None
    _CUPY = False

_SHMEM_MAX_ATOMS = 600

_CHAIN_CUDA_CODE = (
    _RNG_DEVICE_CODE
    + r"""
__device__ double eval_spline_deriv(
    const double* coef4, const long long* start,
    const double* x_min, const double* x_max,
    const double* dx, const long long* n_int,
    long long t, double x)
{
    double x0 = x_min[t];
    if (!(x > x0 && x < x_max[t])) return 0.0;
    double h = dx[t];
    long long m = (long long)floor((x - x0) / h);
    long long ni = n_int[t];
    if (m < 0) m = 0;
    if (m >= ni) m = ni - 1;
    double s = x - (x0 + h * (double)m);
    const double* c = coef4 + 4 * (start[t] + m);
    return (3.0 * c[0] * s + 2.0 * c[1]) * s + c[2];
}

__device__ int pos_ok(const double* p)
{
    for (int d = 0; d < 3; ++d) {
        double v = p[d];
        if (!isfinite(v) || fabs(v) > 1.0e6) return 0;
    }
    return 1;
}

__device__ void eval_two_body_atom(
    int i, const double* pos,
    const int* bond_ptr, const int* bond_nbr, const double* bond_r0, const double* bond_k,
    const int* pair_ptr, const int* pair_nbr, const long long* pair_t, int has_pairs,
    const double* pair_coef4, const long long* pair_start,
    const double* pair_xmin, const double* pair_xmax,
    const double* pair_dx, const long long* pair_nint,
    const int* wca_ptr, const int* wca_nbr, const double* wca_sig,
    int has_wca, double wca_eps,
    double* out_f)
{
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const double* pi = pos + 3 * i;
    int oki = pos_ok(pi);
    double xi = pi[0], yi = pi[1], zi = pi[2];

    if (oki) {
        for (int e = bond_ptr[i]; e < bond_ptr[i + 1]; ++e) {
            int j = bond_nbr[e];
            const double* pj = pos + 3 * j;
            if (!pos_ok(pj)) continue;
            double dx = xi - pj[0], dy = yi - pj[1], dz = zi - pj[2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1.0e-8) continue;
            double f = -bond_k[e] * (r - bond_r0[e]) / r;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }
    }

    if (has_pairs) {
        for (int e = pair_ptr[i]; e < pair_ptr[i + 1]; ++e) {
            int j = pair_nbr[e];
            const double* pj = pos + 3 * j;
            double dx = xi - pj[0], dy = yi - pj[1], dz = zi - pj[2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (!(r > 1.0e-10)) continue;
            double dV = eval_spline_deriv(
                pair_coef4, pair_start, pair_xmin, pair_xmax,
                pair_dx, pair_nint, pair_t[e], r);
            if (dV == 0.0) continue;
            double f = -dV / r;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }
    }

    if (has_wca) {
        const double two_1_6 = 1.122462048309373;
        for (int e = wca_ptr[i]; e < wca_ptr[i + 1]; ++e) {
            int j = wca_nbr[e];
            const double* pj = pos + 3 * j;
            double dx = xi - pj[0], dy = yi - pj[1], dz = zi - pj[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double sig = wca_sig[e];
            double rcut = two_1_6 * sig;
            if (!(r2 >= 1.0e-20 && r2 < rcut * rcut)) continue;
            double sr2 = (sig * sig) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double f = 4.0 * wca_eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r2;
            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }
    }

    out_f[0] = fx;
    out_f[1] = fy;
    out_f[2] = fz;
}

__device__ void accum_angle_term(
    int m, const double* pos, double* F,
    const int* ang_a, const int* ang_b, const int* ang_c,
    const double* ang_theta0, const double* ang_k, const long long* ang_type,
    int has_ang_tab, const double* ang_coef4, const long long* ang_start,
    const double* ang_xmin, const double* ang_xmax,
    const double* ang_dx, const long long* ang_nint)
{
    int ia = ang_a[m], ib = ang_b[m], ic = ang_c[m];
    const double* pa = pos + 3 * ia;
    const double* pb = pos + 3 * ib;
    const double* pc = pos + 3 * ic;
    if (!pos_ok(pa) || !pos_ok(pb) || !pos_ok(pc)) return;
    double ux = pa[0] - pb[0], uy = pa[1] - pb[1], uz = pa[2] - pb[2];
    double vx = pc[0] - pb[0], vy = pc[1] - pb[1], vz = pc[2] - pb[2];
    double nu = sqrt(ux * ux + uy * uy + uz * uz);
    double nv = sqrt(vx * vx + vy * vy + vz * vz);
    if (nu < 1.0e-8 || nv < 1.0e-8) return;
    ux /= nu; uy /= nu; uz /= nu;
    vx /= nv; vy /= nv; vz /= nv;
    double cos_t = fmin(1.0, fmax(-1.0, ux * vx + uy * vy + uz * vz));
    double theta = acos(cos_t);
    double sin_t = sin(theta);
    if (fabs(sin_t) < 1.0e-8) return;
    const double rad2deg = 57.2957795130823208768;
    long long typ = ang_type[m];
    double dV;
    if (has_ang_tab && typ >= 0) {
        dV = eval_spline_deriv(
            ang_coef4, ang_start, ang_xmin, ang_xmax,
            ang_dx, ang_nint, typ, theta * rad2deg) * rad2deg;
    } else {
        dV = ang_k[m] * (theta - ang_theta0[m]);
    }
    double coeff = dV / sin_t;
    double fix = (coeff / nu) * (vx - cos_t * ux);
    double fiy = (coeff / nu) * (vy - cos_t * uy);
    double fiz = (coeff / nu) * (vz - cos_t * uz);
    double fkx = (coeff / nv) * (ux - cos_t * vx);
    double fky = (coeff / nv) * (uy - cos_t * vy);
    double fkz = (coeff / nv) * (uz - cos_t * vz);
    F[3 * ia]     += fix; F[3 * ia + 1] += fiy; F[3 * ia + 2] += fiz;
    F[3 * ic]     += fkx; F[3 * ic + 1] += fky; F[3 * ic + 2] += fkz;
    F[3 * ib]     -= (fix + fkx);
    F[3 * ib + 1] -= (fiy + fky);
    F[3 * ib + 2] -= (fiz + fkz);
}

__device__ void accum_torsion_term(
    int m, const double* pos, double* F,
    const int* tor_a, const int* tor_b, const int* tor_c, const int* tor_d,
    const double* tor_phi0, const double* tor_k, const double* tor_n,
    const long long* tor_type, int has_tor_tab,
    const double* tor_coef4, const long long* tor_start,
    const double* tor_xmin, const double* tor_xmax,
    const double* tor_dx, const long long* tor_nint)
{
    int ia = tor_a[m], ib = tor_b[m], ic = tor_c[m], il = tor_d[m];
    const double* pa = pos + 3 * ia;
    const double* pb = pos + 3 * ib;
    const double* pc = pos + 3 * ic;
    const double* pl = pos + 3 * il;
    if (!pos_ok(pa) || !pos_ok(pb) || !pos_ok(pc) || !pos_ok(pl)) return;
    double fx = pa[0] - pb[0], fy = pa[1] - pb[1], fz = pa[2] - pb[2];
    double gx = pb[0] - pc[0], gy = pb[1] - pc[1], gz = pb[2] - pc[2];
    double hx = pl[0] - pc[0], hy = pl[1] - pc[1], hz = pl[2] - pc[2];
    double ax = fy * gz - fz * gy;
    double ay = fz * gx - fx * gz;
    double az = fx * gy - fy * gx;
    double bx = hy * gz - hz * gy;
    double by = hz * gx - hx * gz;
    double bz = hx * gy - hy * gx;
    double a2 = ax * ax + ay * ay + az * az;
    double b2 = bx * bx + by * by + bz * bz;
    double gn = sqrt(gx * gx + gy * gy + gz * gz);
    if (a2 < 1.0e-24 || b2 < 1.0e-24 || gn < 1.0e-12) return;
    double fb = fx * bx + fy * by + fz * bz;
    double ab = ax * bx + ay * by + az * bz;
    double phi = atan2(-gn * fb, ab);
    const double rad2deg = 57.2957795130823208768;
    long long typ = tor_type[m];
    double dV;
    if (has_tor_tab && typ >= 0) {
        double deg = fmod(phi * rad2deg, 360.0);
        if (deg < 0.0) deg += 360.0;
        if (deg >= 360.0) deg -= 360.0;
        dV = eval_spline_deriv(
            tor_coef4, tor_start, tor_xmin, tor_xmax,
            tor_dx, tor_nint, typ, deg) * rad2deg;
    } else {
        double nn = tor_n[m];
        dV = tor_k[m] * nn * sin(nn * phi - tor_phi0[m]);
    }
    double fg = (fx * gx + fy * gy + fz * gz) / gn;
    double hg = (hx * gx + hy * gy + hz * gz) / gn;
    double ca = gn / a2, cb = gn / b2;
    double gix = -ca * ax, giy = -ca * ay, giz = -ca * az;
    double glx =  cb * bx, gly =  cb * by, glz =  cb * bz;
    double cja = (gn + fg) / a2, cjb = hg / b2;
    double gjx = cja * ax - cjb * bx;
    double gjy = cja * ay - cjb * by;
    double gjz = cja * az - cjb * bz;
    double gkx = -gix - gjx - glx;
    double gky = -giy - gjy - gly;
    double gkz = -giz - gjz - glz;
    F[3 * ia] -= dV * gix; F[3 * ia + 1] -= dV * giy; F[3 * ia + 2] -= dV * giz;
    F[3 * ib] -= dV * gjx; F[3 * ib + 1] -= dV * gjy; F[3 * ib + 2] -= dV * gjz;
    F[3 * ic] -= dV * gkx; F[3 * ic + 1] -= dV * gky; F[3 * ic + 2] -= dV * gkz;
    F[3 * il] -= dV * glx; F[3 * il + 1] -= dV * gly; F[3 * il + 2] -= dV * glz;
}

// Two-sided Jacobi 3x3 SVD returning proper orthogonal Kabsch rotation Q.
__device__ int kabsch_rotation_3x3(const double* H, double* Q)
{
    double B[9], U[9], V[9];
    for (int i = 0; i < 9; ++i) {
        B[i] = H[i];
        U[i] = 0.0;
        V[i] = 0.0;
    }
    U[0] = U[4] = U[8] = 1.0;
    V[0] = V[4] = V[8] = 1.0;

    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 25; ++sweep) {
        double off = fabs(B[1]) + fabs(B[2]) + fabs(B[3])
                   + fabs(B[5]) + fabs(B[6]) + fabs(B[7]);
        double diag = fabs(B[0]) + fabs(B[4]) + fabs(B[8]);
        if (off <= 1.0e-15 * fmax(diag, 1.0e-300)) break;
        for (int kp = 0; kp < 3; ++kp) {
            int p = pairs[kp][0], q = pairs[kp][1];
            double w = B[3 * p + p], x = B[3 * p + q];
            double y = B[3 * q + p], z = B[3 * q + q];
            double alpha = w + z, beta = y - x;
            double c1 = 1.0, s1 = 0.0;
            double r = hypot(alpha, beta);
            if (r > 0.0) {
                c1 = alpha / r;
                s1 = beta / r;
            }
            for (int k = 0; k < 3; ++k) {
                double bp = B[3 * p + k], bq = B[3 * q + k];
                B[3 * p + k] =  c1 * bp + s1 * bq;
                B[3 * q + k] = -s1 * bp + c1 * bq;
                double up = U[3 * k + p], uq = U[3 * k + q];
                U[3 * k + p] =  c1 * up + s1 * uq;
                U[3 * k + q] = -s1 * up + c1 * uq;
            }
            double wp = B[3 * p + p], xp = B[3 * p + q], zp = B[3 * q + q];
            double c2 = 1.0, s2 = 0.0;
            if (fabs(xp) > 1.0e-16 * (fabs(wp) + fabs(zp))) {
                double tau = (zp - wp) / (2.0 * xp);
                double sgn_tau = (tau >= 0.0) ? 1.0 : -1.0;
                double t = sgn_tau / (fabs(tau) + sqrt(1.0 + tau * tau));
                c2 = 1.0 / sqrt(1.0 + t * t);
                s2 = t * c2;
            }
            for (int k = 0; k < 3; ++k) {
                double bp = B[3 * p + k], bq = B[3 * q + k];
                B[3 * p + k] = c2 * bp - s2 * bq;
                B[3 * q + k] = s2 * bp + c2 * bq;
            }
            for (int k = 0; k < 3; ++k) {
                double bp = B[3 * k + p], bq = B[3 * k + q];
                B[3 * k + p] = c2 * bp - s2 * bq;
                B[3 * k + q] = s2 * bp + c2 * bq;
                double up = U[3 * k + p], uq = U[3 * k + q];
                U[3 * k + p] = c2 * up - s2 * uq;
                U[3 * k + q] = s2 * up + c2 * uq;
                double vp = V[3 * k + p], vq = V[3 * k + q];
                V[3 * k + p] = c2 * vp - s2 * vq;
                V[3 * k + q] = s2 * vp + c2 * vq;
            }
        }
    }

    double S[3] = {B[0], B[4], B[8]};
    for (int k = 0; k < 3; ++k) {
        if (S[k] < 0.0) {
            S[k] = -S[k];
            for (int r = 0; r < 3; ++r) U[3 * r + k] = -U[3 * r + k];
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (S[j] > S[i]) {
                double ts = S[i]; S[i] = S[j]; S[j] = ts;
                for (int r = 0; r < 3; ++r) {
                    double tu = U[3 * r + i]; U[3 * r + i] = U[3 * r + j]; U[3 * r + j] = tu;
                    double tv = V[3 * r + i]; V[3 * r + i] = V[3 * r + j]; V[3 * r + j] = tv;
                }
            }
        }
    }
    if (!(S[0] > 0.0 && S[1] >= 1.0e-12 * S[0])) return 0;

    double W[9];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            W[3 * a + b] = V[3 * a + 0] * U[3 * b + 0]
                         + V[3 * a + 1] * U[3 * b + 1]
                         + V[3 * a + 2] * U[3 * b + 2];
    double det = W[0] * (W[4] * W[8] - W[5] * W[7])
               - W[1] * (W[3] * W[8] - W[5] * W[6])
               + W[2] * (W[3] * W[7] - W[4] * W[6]);
    double sgn = (det < 0.0) ? -1.0 : 1.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            Q[3 * a + b] = V[3 * a + 0] * U[3 * b + 0]
                         + V[3 * a + 1] * U[3 * b + 1]
                         + sgn * V[3 * a + 2] * U[3 * b + 2];
    return 1;
}

__device__ int shake_constraints_single(
    double* pos,
    int n_len, const int* lc_a, const int* lc_b, const double* lc_len,
    int n_cop, const int* cc_a, const int* cc_b, const int* cc_c, const int* cc_d,
    double tol, int max_iter)
{
    if (n_len == 0 && n_cop == 0) return 1;
    for (int it = 0; it < max_iter; ++it) {
        double max_viol = 0.0;
        for (int c = 0; c < n_len; ++c) {
            int ia = lc_a[c], ib = lc_b[c];
            double dx = pos[3 * ib] - pos[3 * ia];
            double dy = pos[3 * ib + 1] - pos[3 * ia + 1];
            double dz = pos[3 * ib + 2] - pos[3 * ia + 2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1.0e-12) continue;
            double viol = r - lc_len[c];
            if (fabs(viol) > max_viol) max_viol = fabs(viol);
            double half_over_r = 0.5 * viol / r;
            pos[3 * ia]     += half_over_r * dx;
            pos[3 * ia + 1] += half_over_r * dy;
            pos[3 * ia + 2] += half_over_r * dz;
            pos[3 * ib]     -= half_over_r * dx;
            pos[3 * ib + 1] -= half_over_r * dy;
            pos[3 * ib + 2] -= half_over_r * dz;
        }
        for (int c = 0; c < n_cop; ++c) {
            int ia = cc_a[c], ib = cc_b[c], ic = cc_c[c], id = cc_d[c];
            double d2x = pos[3 * ic] - pos[3 * ib];
            double d2y = pos[3 * ic + 1] - pos[3 * ib + 1];
            double d2z = pos[3 * ic + 2] - pos[3 * ib + 2];
            double d3x = pos[3 * id] - pos[3 * ib];
            double d3y = pos[3 * id + 1] - pos[3 * ib + 1];
            double d3z = pos[3 * id + 2] - pos[3 * ib + 2];
            double px = d2y * d3z - d2z * d3y;
            double py = d2z * d3x - d2x * d3z;
            double pz = d2x * d3y - d2y * d3x;
            double pn = sqrt(px * px + py * py + pz * pz);
            if (pn < 1.0e-12) continue;
            double nx = px / pn, ny = py / pn, nz = pz / pn;
            double cx = (pos[3 * ib] + pos[3 * ic] + pos[3 * id]) / 3.0;
            double cy = (pos[3 * ib + 1] + pos[3 * ic + 1] + pos[3 * id + 1]) / 3.0;
            double cz = (pos[3 * ib + 2] + pos[3 * ic + 2] + pos[3 * id + 2]) / 3.0;
            double phi = (pos[3 * ia] - cx) * nx
                       + (pos[3 * ia + 1] - cy) * ny
                       + (pos[3 * ia + 2] - cz) * nz;
            if (fabs(phi) > max_viol) max_viol = fabs(phi);
            pos[3 * ia]     -= phi * nx;
            pos[3 * ia + 1] -= phi * ny;
            pos[3 * ia + 2] -= phi * nz;
        }
        if (max_viol < tol) return 1;
    }
    return 0;
}

extern "C" __global__
void chain_forces_only_kernel(
    const double* __restrict__ body_pos,
    double* __restrict__ out_forces,
    int n_traj, int n_atoms,
    const int* __restrict__ bond_ptr, const int* __restrict__ bond_nbr,
    const double* __restrict__ bond_r0, const double* __restrict__ bond_k,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_nbr,
    const long long* __restrict__ pair_t, int has_pairs,
    const double* __restrict__ pair_coef4, const long long* __restrict__ pair_start,
    const double* __restrict__ pair_xmin, const double* __restrict__ pair_xmax,
    const double* __restrict__ pair_dx, const long long* __restrict__ pair_nint,
    const int* __restrict__ wca_ptr, const int* __restrict__ wca_nbr,
    const double* __restrict__ wca_sig, int has_wca, double wca_eps,
    int n_ang_colors, const int* __restrict__ ang_color_off,
    const int* __restrict__ ang_a, const int* __restrict__ ang_b, const int* __restrict__ ang_c,
    const double* __restrict__ ang_theta0, const double* __restrict__ ang_k,
    const long long* __restrict__ ang_type, int has_ang_tab,
    const double* __restrict__ ang_coef4, const long long* __restrict__ ang_start,
    const double* __restrict__ ang_xmin, const double* __restrict__ ang_xmax,
    const double* __restrict__ ang_dx, const long long* __restrict__ ang_nint,
    int n_tor_colors, const int* __restrict__ tor_color_off,
    const int* __restrict__ tor_a, const int* __restrict__ tor_b,
    const int* __restrict__ tor_c, const int* __restrict__ tor_d,
    const double* __restrict__ tor_phi0, const double* __restrict__ tor_k,
    const double* __restrict__ tor_n, const long long* __restrict__ tor_type,
    int has_tor_tab,
    const double* __restrict__ tor_coef4, const long long* __restrict__ tor_start,
    const double* __restrict__ tor_xmin, const double* __restrict__ tor_xmax,
    const double* __restrict__ tor_dx, const long long* __restrict__ tor_nint)
{
    int s = blockIdx.x;
    if (s >= n_traj) return;
    int tid = threadIdx.x, nt = blockDim.x;
    const double* pos = body_pos + (long long)s * n_atoms * 3;
    double* F = out_forces + (long long)s * n_atoms * 3;

    for (int i = tid; i < n_atoms; i += nt) {
        double f[3];
        eval_two_body_atom(
            i, pos, bond_ptr, bond_nbr, bond_r0, bond_k,
            pair_ptr, pair_nbr, pair_t, has_pairs,
            pair_coef4, pair_start, pair_xmin, pair_xmax, pair_dx, pair_nint,
            wca_ptr, wca_nbr, wca_sig, has_wca, wca_eps, f);
        F[3 * i]     = f[0];
        F[3 * i + 1] = f[1];
        F[3 * i + 2] = f[2];
    }
    __syncthreads();

    for (int c = 0; c < n_ang_colors; ++c) {
        int m0 = ang_color_off[c], m1 = ang_color_off[c + 1];
        for (int m = m0 + tid; m < m1; m += nt) {
            accum_angle_term(
                m, pos, F, ang_a, ang_b, ang_c, ang_theta0, ang_k, ang_type,
                has_ang_tab, ang_coef4, ang_start, ang_xmin, ang_xmax, ang_dx, ang_nint);
        }
        __syncthreads();
    }

    for (int c = 0; c < n_tor_colors; ++c) {
        int m0 = tor_color_off[c], m1 = tor_color_off[c + 1];
        for (int m = m0 + tid; m < m1; m += nt) {
            accum_torsion_term(
                m, pos, F, tor_a, tor_b, tor_c, tor_d, tor_phi0, tor_k, tor_n, tor_type,
                has_tor_tab, tor_coef4, tor_start, tor_xmin, tor_xmax, tor_dx, tor_nint);
        }
        __syncthreads();
    }
}

__device__ void inner_step_body(
    double* pos, double* old_pos, double* F, double* red, double* sh_Q,
    const double* ext_f, int has_ext,
    const double* supplied_z, int has_supplied_z,
    int n_atoms, long long traj_id, long long rng_step, unsigned int stream,
    unsigned int k0, unsigned int k1,
    double dt_chain, const double* mob, const double* sigma,
    const int* bond_ptr, const int* bond_nbr, const double* bond_r0, const double* bond_k,
    const int* pair_ptr, const int* pair_nbr, const long long* pair_t, int has_pairs,
    const double* pair_coef4, const long long* pair_start,
    const double* pair_xmin, const double* pair_xmax,
    const double* pair_dx, const long long* pair_nint,
    const int* wca_ptr, const int* wca_nbr, const double* wca_sig,
    int has_wca, double wca_eps,
    int n_ang_colors, const int* ang_color_off,
    const int* ang_a, const int* ang_b, const int* ang_c,
    const double* ang_theta0, const double* ang_k, const long long* ang_type,
    int has_ang_tab, const double* ang_coef4, const long long* ang_start,
    const double* ang_xmin, const double* ang_xmax,
    const double* ang_dx, const long long* ang_nint,
    int n_tor_colors, const int* tor_color_off,
    const int* tor_a, const int* tor_b, const int* tor_c, const int* tor_d,
    const double* tor_phi0, const double* tor_k, const double* tor_n,
    const long long* tor_type, int has_tor_tab,
    const double* tor_coef4, const long long* tor_start,
    const double* tor_xmin, const double* tor_xmax,
    const double* tor_dx, const long long* tor_nint,
    int apply_constraints,
    int n_len_c, const int* lc_a, const int* lc_b, const double* lc_len,
    int n_cop_c, const int* cc_a, const int* cc_b, const int* cc_c, const int* cc_d,
    double c_tol, int c_max_iter, int remove_rot, int* c_failed)
{
    int tid = threadIdx.x, nt = blockDim.x;
    if (remove_rot) {
        double ox = 0.0, oy = 0.0, oz = 0.0;
        for (int i = tid; i < n_atoms; i += nt) {
            ox += pos[3 * i];
            oy += pos[3 * i + 1];
            oz += pos[3 * i + 2];
        }
        red[tid]          = ox;
        red[nt + tid]     = oy;
        red[2 * nt + tid] = oz;
        __syncthreads();
        for (int w = nt / 2; w > 0; w >>= 1) {
            if (tid < w) {
                red[tid]          += red[tid + w];
                red[nt + tid]     += red[nt + tid + w];
                red[2 * nt + tid] += red[2 * nt + tid + w];
            }
            __syncthreads();
        }
        double mx = red[0] / (double)n_atoms;
        double my = red[nt] / (double)n_atoms;
        double mz = red[2 * nt] / (double)n_atoms;
        for (int i = tid; i < n_atoms; i += nt) {
            old_pos[3 * i]     = pos[3 * i]     - mx;
            old_pos[3 * i + 1] = pos[3 * i + 1] - my;
            old_pos[3 * i + 2] = pos[3 * i + 2] - mz;
        }
        __syncthreads();
    }

    for (int i = tid; i < n_atoms; i += nt) {
        double f[3];
        eval_two_body_atom(
            i, pos, bond_ptr, bond_nbr, bond_r0, bond_k,
            pair_ptr, pair_nbr, pair_t, has_pairs,
            pair_coef4, pair_start, pair_xmin, pair_xmax, pair_dx, pair_nint,
            wca_ptr, wca_nbr, wca_sig, has_wca, wca_eps, f);
        if (has_ext) {
            f[0] += ext_f[3 * i];
            f[1] += ext_f[3 * i + 1];
            f[2] += ext_f[3 * i + 2];
        }
        F[3 * i]     = f[0];
        F[3 * i + 1] = f[1];
        F[3 * i + 2] = f[2];
    }
    __syncthreads();

    for (int c = 0; c < n_ang_colors; ++c) {
        int m0 = ang_color_off[c], m1 = ang_color_off[c + 1];
        for (int m = m0 + tid; m < m1; m += nt) {
            accum_angle_term(
                m, pos, F, ang_a, ang_b, ang_c, ang_theta0, ang_k, ang_type,
                has_ang_tab, ang_coef4, ang_start, ang_xmin, ang_xmax, ang_dx, ang_nint);
        }
        __syncthreads();
    }

    for (int c = 0; c < n_tor_colors; ++c) {
        int m0 = tor_color_off[c], m1 = tor_color_off[c + 1];
        for (int m = m0 + tid; m < m1; m += nt) {
            accum_torsion_term(
                m, pos, F, tor_a, tor_b, tor_c, tor_d, tor_phi0, tor_k, tor_n, tor_type,
                has_tor_tab, tor_coef4, tor_start, tor_xmin, tor_xmax, tor_dx, tor_nint);
        }
        __syncthreads();
    }

    uint2 key = make_uint2(k0, k1);
    for (int i = tid; i < n_atoms; i += nt) {
        double mi = mob[i] * dt_chain;
        double dx = mi * F[3 * i];
        double dy = mi * F[3 * i + 1];
        double dz = mi * F[3 * i + 2];
        double dm = sqrt(dx * dx + dy * dy + dz * dz);
        if (dm > 5.0) {
            double sc = 5.0 / dm;
            dx *= sc; dy *= sc; dz *= sc;
        }
        double z0, z1, z2, spare;
        if (has_supplied_z) {
            z0 = supplied_z[3 * i];
            z1 = supplied_z[3 * i + 1];
            z2 = supplied_z[3 * i + 2];
        } else {
            box_muller(philox(philox_counter(traj_id, rng_step, stream, (unsigned int)(2 * i)), key), &z0, &z1);
            box_muller(philox(philox_counter(traj_id, rng_step, stream, (unsigned int)(2 * i + 1)), key), &z2, &spare);
        }
        double si = sigma[i];
        pos[3 * i]     += dx + si * z0;
        pos[3 * i + 1] += dy + si * z1;
        pos[3 * i + 2] += dz + si * z2;
    }
    __syncthreads();

    int has_constr = apply_constraints && (n_len_c > 0 || n_cop_c > 0);
    if (remove_rot) {
        if (has_constr) {
            if (tid == 0) {
                int ok = shake_constraints_single(
                    pos, n_len_c, lc_a, lc_b, lc_len,
                    n_cop_c, cc_a, cc_b, cc_c, cc_d, c_tol, c_max_iter);
                if (!ok) *c_failed = 1;
            }
            __syncthreads();
        }
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = tid; i < n_atoms; i += nt) {
            sx += pos[3 * i];
            sy += pos[3 * i + 1];
            sz += pos[3 * i + 2];
        }
        red[tid]          = sx;
        red[nt + tid]     = sy;
        red[2 * nt + tid] = sz;
        __syncthreads();
        for (int w = nt / 2; w > 0; w >>= 1) {
            if (tid < w) {
                red[tid]          += red[tid + w];
                red[nt + tid]     += red[nt + tid + w];
                red[2 * nt + tid] += red[2 * nt + tid + w];
            }
            __syncthreads();
        }
        double cx = red[0] / (double)n_atoms;
        double cy = red[nt] / (double)n_atoms;
        double cz = red[2 * nt] / (double)n_atoms;
        for (int i = tid; i < n_atoms; i += nt) {
            pos[3 * i]     -= cx;
            pos[3 * i + 1] -= cy;
            pos[3 * i + 2] -= cz;
        }
        __syncthreads();

        if (n_atoms >= 3) {
            double h_loc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = tid; i < n_atoms; i += nt) {
                const double* p = pos + 3 * i;
                const double* o = old_pos + 3 * i;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        h_loc[3 * a + b] += p[a] * o[b];
            }
            for (int k = 0; k < 9; ++k) red[k * nt + tid] = h_loc[k];
            __syncthreads();
            for (int w = nt / 2; w > 0; w >>= 1) {
                if (tid < w) {
                    for (int k = 0; k < 9; ++k)
                        red[k * nt + tid] += red[k * nt + tid + w];
                }
                __syncthreads();
            }
            if (tid == 0) {
                double H[9];
                for (int k = 0; k < 9; ++k) H[k] = red[k * nt];
                sh_Q[9] = (double)kabsch_rotation_3x3(H, sh_Q);
            }
            __syncthreads();
            if (sh_Q[9] > 0.5) {
                for (int i = tid; i < n_atoms; i += nt) {
                    double px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2];
                    pos[3 * i]     = sh_Q[0] * px + sh_Q[1] * py + sh_Q[2] * pz;
                    pos[3 * i + 1] = sh_Q[3] * px + sh_Q[4] * py + sh_Q[5] * pz;
                    pos[3 * i + 2] = sh_Q[6] * px + sh_Q[7] * py + sh_Q[8] * pz;
                }
                __syncthreads();
            }
        }
    } else {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = tid; i < n_atoms; i += nt) {
            sx += pos[3 * i];
            sy += pos[3 * i + 1];
            sz += pos[3 * i + 2];
        }
        red[tid]          = sx;
        red[nt + tid]     = sy;
        red[2 * nt + tid] = sz;
        __syncthreads();
        for (int w = nt / 2; w > 0; w >>= 1) {
            if (tid < w) {
                red[tid]          += red[tid + w];
                red[nt + tid]     += red[nt + tid + w];
                red[2 * nt + tid] += red[2 * nt + tid + w];
            }
            __syncthreads();
        }
        double cx = red[0] / (double)n_atoms;
        double cy = red[nt] / (double)n_atoms;
        double cz = red[2 * nt] / (double)n_atoms;
        for (int i = tid; i < n_atoms; i += nt) {
            pos[3 * i]     -= cx;
            pos[3 * i + 1] -= cy;
            pos[3 * i + 2] -= cz;
        }
        __syncthreads();
        if (has_constr) {
            if (tid == 0) {
                int ok = shake_constraints_single(
                    pos, n_len_c, lc_a, lc_b, lc_len,
                    n_cop_c, cc_a, cc_b, cc_c, cc_d, c_tol, c_max_iter);
                if (!ok) *c_failed = 1;
            }
            __syncthreads();
        }
    }
}

extern "C" __global__
void chain_inner_steps_shmem_kernel(
    double* __restrict__ body_pos,
    const double* __restrict__ ext_body_forces, int has_ext,
    const double* __restrict__ supplied_normals, int has_supplied_z,
    const long long* __restrict__ traj_ids,
    const long long* __restrict__ step_base,
    const int* __restrict__ n_inner_arr,
    int n_traj, int n_atoms,
    unsigned int stream, unsigned int k0, unsigned int k1,
    double dt_chain, const double* __restrict__ mob, const double* __restrict__ sigma,
    const int* __restrict__ bond_ptr, const int* __restrict__ bond_nbr,
    const double* __restrict__ bond_r0, const double* __restrict__ bond_k,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_nbr,
    const long long* __restrict__ pair_t, int has_pairs,
    const double* __restrict__ pair_coef4, const long long* __restrict__ pair_start,
    const double* __restrict__ pair_xmin, const double* __restrict__ pair_xmax,
    const double* __restrict__ pair_dx, const long long* __restrict__ pair_nint,
    const int* __restrict__ wca_ptr, const int* __restrict__ wca_nbr,
    const double* __restrict__ wca_sig, int has_wca, double wca_eps,
    int n_ang_colors, const int* __restrict__ ang_color_off,
    const int* __restrict__ ang_a, const int* __restrict__ ang_b, const int* __restrict__ ang_c,
    const double* __restrict__ ang_theta0, const double* __restrict__ ang_k,
    const long long* __restrict__ ang_type, int has_ang_tab,
    const double* __restrict__ ang_coef4, const long long* __restrict__ ang_start,
    const double* __restrict__ ang_xmin, const double* __restrict__ ang_xmax,
    const double* __restrict__ ang_dx, const long long* __restrict__ ang_nint,
    int n_tor_colors, const int* __restrict__ tor_color_off,
    const int* __restrict__ tor_a, const int* __restrict__ tor_b,
    const int* __restrict__ tor_c, const int* __restrict__ tor_d,
    const double* __restrict__ tor_phi0, const double* __restrict__ tor_k,
    const double* __restrict__ tor_n, const long long* __restrict__ tor_type,
    int has_tor_tab,
    const double* __restrict__ tor_coef4, const long long* __restrict__ tor_start,
    const double* __restrict__ tor_xmin, const double* __restrict__ tor_xmax,
    const double* __restrict__ tor_dx, const long long* __restrict__ tor_nint,
    int apply_constraints,
    int n_len_c, const int* __restrict__ lc_a, const int* __restrict__ lc_b, const double* __restrict__ lc_len,
    int n_cop_c, const int* __restrict__ cc_a, const int* __restrict__ cc_b,
    const int* __restrict__ cc_c, const int* __restrict__ cc_d,
    double c_tol, int c_max_iter, int remove_rot,
    int* __restrict__ c_failed_out)
{
    extern __shared__ double shmem[];
    int s = blockIdx.x;
    if (s >= n_traj) return;
    int tid = threadIdx.x, nt = blockDim.x;
    double* sh_pos = shmem;
    double* sh_old = sh_pos + 3 * n_atoms;
    double* sh_F   = sh_old + 3 * n_atoms;
    double* sh_red = sh_F   + 3 * n_atoms;
    double* sh_Q   = sh_red + 9 * nt;

    double* g_pos = body_pos + (long long)s * n_atoms * 3;
    const double* g_ext = has_ext ? (ext_body_forces + (long long)s * n_atoms * 3) : NULL;

    for (int k = tid; k < 3 * n_atoms; k += nt) sh_pos[k] = g_pos[k];
    __syncthreads();

    int n_in = n_inner_arr[s];
    long long tid_global = traj_ids[s];
    long long sb = step_base[s];
    int failed_local = 0;

    for (int sub = 0; sub < n_in; ++sub) {
        const double* g_z = has_supplied_z
            ? (supplied_normals + ((long long)s * n_in + sub) * n_atoms * 3)
            : NULL;
        inner_step_body(
            sh_pos, sh_old, sh_F, sh_red, sh_Q,
            g_ext, has_ext, g_z, has_supplied_z,
            n_atoms, tid_global, sb + sub, stream, k0, k1,
            dt_chain, mob, sigma,
            bond_ptr, bond_nbr, bond_r0, bond_k,
            pair_ptr, pair_nbr, pair_t, has_pairs,
            pair_coef4, pair_start, pair_xmin, pair_xmax, pair_dx, pair_nint,
            wca_ptr, wca_nbr, wca_sig, has_wca, wca_eps,
            n_ang_colors, ang_color_off, ang_a, ang_b, ang_c,
            ang_theta0, ang_k, ang_type, has_ang_tab,
            ang_coef4, ang_start, ang_xmin, ang_xmax, ang_dx, ang_nint,
            n_tor_colors, tor_color_off, tor_a, tor_b, tor_c, tor_d,
            tor_phi0, tor_k, tor_n, tor_type, has_tor_tab,
            tor_coef4, tor_start, tor_xmin, tor_xmax, tor_dx, tor_nint,
            apply_constraints,
            n_len_c, lc_a, lc_b, lc_len,
            n_cop_c, cc_a, cc_b, cc_c, cc_d,
            c_tol, c_max_iter, remove_rot, &failed_local);
    }

    for (int k = tid; k < 3 * n_atoms; k += nt) g_pos[k] = sh_pos[k];
    if (tid == 0 && failed_local) c_failed_out[s] = 1;
}

extern "C" __global__
void chain_inner_steps_global_kernel(
    double* __restrict__ body_pos,
    double* __restrict__ work_old,
    double* __restrict__ work_F,
    const double* __restrict__ ext_body_forces, int has_ext,
    const double* __restrict__ supplied_normals, int has_supplied_z,
    const long long* __restrict__ traj_ids,
    const long long* __restrict__ step_base,
    const int* __restrict__ n_inner_arr,
    int n_traj, int n_atoms,
    unsigned int stream, unsigned int k0, unsigned int k1,
    double dt_chain, const double* __restrict__ mob, const double* __restrict__ sigma,
    const int* __restrict__ bond_ptr, const int* __restrict__ bond_nbr,
    const double* __restrict__ bond_r0, const double* __restrict__ bond_k,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_nbr,
    const long long* __restrict__ pair_t, int has_pairs,
    const double* __restrict__ pair_coef4, const long long* __restrict__ pair_start,
    const double* __restrict__ pair_xmin, const double* __restrict__ pair_xmax,
    const double* __restrict__ pair_dx, const long long* __restrict__ pair_nint,
    const int* __restrict__ wca_ptr, const int* __restrict__ wca_nbr,
    const double* __restrict__ wca_sig, int has_wca, double wca_eps,
    int n_ang_colors, const int* __restrict__ ang_color_off,
    const int* __restrict__ ang_a, const int* __restrict__ ang_b, const int* __restrict__ ang_c,
    const double* __restrict__ ang_theta0, const double* __restrict__ ang_k,
    const long long* __restrict__ ang_type, int has_ang_tab,
    const double* __restrict__ ang_coef4, const long long* __restrict__ ang_start,
    const double* __restrict__ ang_xmin, const double* __restrict__ ang_xmax,
    const double* __restrict__ ang_dx, const long long* __restrict__ ang_nint,
    int n_tor_colors, const int* __restrict__ tor_color_off,
    const int* __restrict__ tor_a, const int* __restrict__ tor_b,
    const int* __restrict__ tor_c, const int* __restrict__ tor_d,
    const double* __restrict__ tor_phi0, const double* __restrict__ tor_k,
    const double* __restrict__ tor_n, const long long* __restrict__ tor_type,
    int has_tor_tab,
    const double* __restrict__ tor_coef4, const long long* __restrict__ tor_start,
    const double* __restrict__ tor_xmin, const double* __restrict__ tor_xmax,
    const double* __restrict__ tor_dx, const long long* __restrict__ tor_nint,
    int apply_constraints,
    int n_len_c, const int* __restrict__ lc_a, const int* __restrict__ lc_b, const double* __restrict__ lc_len,
    int n_cop_c, const int* __restrict__ cc_a, const int* __restrict__ cc_b,
    const int* __restrict__ cc_c, const int* __restrict__ cc_d,
    double c_tol, int c_max_iter, int remove_rot,
    int* __restrict__ c_failed_out)
{
    extern __shared__ double shmem[];
    int s = blockIdx.x;
    if (s >= n_traj) return;
    int nt = blockDim.x;
    double* sh_red = shmem;
    double* sh_Q   = sh_red + 9 * nt;

    double* g_pos = body_pos + (long long)s * n_atoms * 3;
    double* g_old = work_old + (long long)s * n_atoms * 3;
    double* g_F   = work_F   + (long long)s * n_atoms * 3;
    const double* g_ext = has_ext ? (ext_body_forces + (long long)s * n_atoms * 3) : NULL;

    int n_in = n_inner_arr[s];
    long long tid_global = traj_ids[s];
    long long sb = step_base[s];
    int failed_local = 0;

    for (int sub = 0; sub < n_in; ++sub) {
        const double* g_z = has_supplied_z
            ? (supplied_normals + ((long long)s * n_in + sub) * n_atoms * 3)
            : NULL;
        inner_step_body(
            g_pos, g_old, g_F, sh_red, sh_Q,
            g_ext, has_ext, g_z, has_supplied_z,
            n_atoms, tid_global, sb + sub, stream, k0, k1,
            dt_chain, mob, sigma,
            bond_ptr, bond_nbr, bond_r0, bond_k,
            pair_ptr, pair_nbr, pair_t, has_pairs,
            pair_coef4, pair_start, pair_xmin, pair_xmax, pair_dx, pair_nint,
            wca_ptr, wca_nbr, wca_sig, has_wca, wca_eps,
            n_ang_colors, ang_color_off, ang_a, ang_b, ang_c,
            ang_theta0, ang_k, ang_type, has_ang_tab,
            ang_coef4, ang_start, ang_xmin, ang_xmax, ang_dx, ang_nint,
            n_tor_colors, tor_color_off, tor_a, tor_b, tor_c, tor_d,
            tor_phi0, tor_k, tor_n, tor_type, has_tor_tab,
            tor_coef4, tor_start, tor_xmin, tor_xmax, tor_dx, tor_nint,
            apply_constraints,
            n_len_c, lc_a, lc_b, lc_len,
            n_cop_c, cc_a, cc_b, cc_c, cc_d,
            c_tol, c_max_iter, remove_rot, &failed_local);
    }
    if (threadIdx.x == 0 && failed_local) c_failed_out[s] = 1;
}

__device__ double trilinear_interp(
    const double* grid_data, const double* origin, const double* inv_sp,
    int nx, int ny, int nz, double x, double y, double z)
{
    double ix = (x - origin[0]) * inv_sp[0];
    double iy = (y - origin[1]) * inv_sp[1];
    double iz = (z - origin[2]) * inv_sp[2];
    if (!isfinite(ix) || !isfinite(iy) || !isfinite(iz)) return 0.0;
    int i0 = (int)floor(ix), j0 = (int)floor(iy), k0 = (int)floor(iz);
    if (i0 < 0 || i0 >= nx - 1 || j0 < 0 || j0 >= ny - 1 || k0 < 0 || k0 >= nz - 1)
        return 0.0;
    double fx = ix - i0, fy = iy - j0, fz = iz - k0;
    long long sy = (long long)ny * nz, sz = nz;
    return (grid_data[i0 * sy + j0 * sz + k0]                   * (1.0 - fx) * (1.0 - fy) * (1.0 - fz) +
            grid_data[(i0 + 1) * sy + j0 * sz + k0]             * fx         * (1.0 - fy) * (1.0 - fz) +
            grid_data[i0 * sy + (j0 + 1) * sz + k0]             * (1.0 - fx) * fy         * (1.0 - fz) +
            grid_data[i0 * sy + j0 * sz + (k0 + 1)]             * (1.0 - fx) * (1.0 - fy) * fz         +
            grid_data[(i0 + 1) * sy + (j0 + 1) * sz + k0]       * fx         * fy         * (1.0 - fz) +
            grid_data[(i0 + 1) * sy + j0 * sz + (k0 + 1)]       * fx         * (1.0 - fy) * fz         +
            grid_data[i0 * sy + (j0 + 1) * sz + (k0 + 1)]       * (1.0 - fx) * fy         * fz         +
            grid_data[(i0 + 1) * sy + (j0 + 1) * sz + (k0 + 1)] * fx         * fy         * fz);
}

__device__ void eval_dxgrid_clamped(
    const double* grid_data, const double* b, const int* n,
    double px, double py, double pz, double q, double alpha, int is_born,
    double* f, double* e)
{
    const double* origin = b;
    const double* inv_sp = b + 3;
    const double* sp = b + 6;
    int nx = n[0], ny = n[1], nz = n[2];
    double p[3] = {px, py, pz};
    double g[3];
    for (int d = 0; d < 3; ++d) {
        double h = 0.5 * sp[d];
        double lo = origin[d];
        double hi = origin[d] + ((double)(n[d] - 1) - 1.0e-6) * sp[d];
        double dp[3] = {px, py, pz};
        double dm[3] = {px, py, pz};
        dp[d] = fmin(hi, fmax(lo, p[d] + h));
        dm[d] = fmin(hi, fmax(lo, p[d] - h));
        double span = dp[d] - dm[d];
        if (fabs(span) > 1.0e-12) {
            double vp = trilinear_interp(grid_data, origin, inv_sp, nx, ny, nz, dp[0], dp[1], dp[2]);
            double vm = trilinear_interp(grid_data, origin, inv_sp, nx, ny, nz, dm[0], dm[1], dm[2]);
            g[d] = (vp - vm) / span;
        } else {
            g[d] = 0.0;
        }
    }
    double phi = trilinear_interp(grid_data, origin, inv_sp, nx, ny, nz, px, py, pz);
    double coeff   = is_born ? (-alpha * q * q) : (-q);
    double e_coeff = is_born ? ( alpha * q * q) : ( q);
    f[0] = coeff * g[0];
    f[1] = coeff * g[1];
    f[2] = coeff * g[2];
    *e = e_coeff * phi;
}

__device__ void eval_far_term(
    const double* c, double x, double y, double z, double q, double* f, double* e)
{
    double V = c[0], lam = c[1], fpe = c[2];
    const double* p = c + 4;
    const double* Q = c + 7;
    double sr = fmax(sqrt(x * x + y * y + z * z), 1.0);
    double ex = exp(-sr / lam);
    double hx = x / sr, hy = y / sr, hz = z / sr;
    double phi = V * ex / sr;
    double dphi = V * ex * (-1.0 / (sr * sr) - 1.0 / (sr * lam));
    double tdx = 0.0, tdy = 0.0, tdz = 0.0, tqx = 0.0, tqy = 0.0, tqz = 0.0;
    if (c[16] != 0.0) {
        double pr = hx * p[0] + hy * p[1] + hz * p[2];
        double a = 1.0 + sr / lam;
        phi += pr / (fpe * sr * sr) * a * ex;
        dphi += pr / fpe * ex * (-2.0 / (sr * sr * sr) - 2.0 / (sr * sr * lam) - 1.0 / (sr * lam * lam));
        double w = a * ex / (fpe * sr * sr * sr);
        tdx = w * (p[0] - pr * hx);
        tdy = w * (p[1] - pr * hy);
        tdz = w * (p[2] - pr * hz);
    }
    if (c[17] != 0.0) {
        double Qx = Q[0] * hx + Q[1] * hy + Q[2] * hz;
        double Qy = Q[3] * hx + Q[4] * hy + Q[5] * hz;
        double Qz = Q[6] * hx + Q[7] * hy + Q[8] * hz;
        double rQr = hx * Qx + hy * Qy + hz * Qz;
        double b = 1.0 + sr / lam + sr * sr / (3.0 * lam * lam);
        phi += rQr / (fpe * sr * sr * sr) * b * ex;
        dphi += rQr / fpe * ex * (-3.0 / (sr * sr * sr * sr) - 3.0 / (sr * sr * sr * lam)
                                  - 4.0 / (3.0 * sr * sr * lam * lam) - 1.0 / (3.0 * sr * lam * lam * lam));
        double w = 2.0 * b * ex / (fpe * sr * sr * sr * sr);
        tqx = w * (Qx - rQr * hx);
        tqy = w * (Qy - rQr * hy);
        tqz = w * (Qz - rQr * hz);
    }
    if (c[18] != 0.0) {
        phi += c[3] / (fpe * sr) * ex;
        dphi += c[3] / fpe * ex * (-1.0 / (sr * sr) - 1.0 / (sr * lam));
    }
    f[0] = -q * (dphi * hx + tdx + tqx);
    f[1] = -q * (dphi * hy + tdy + tqy);
    f[2] = -q * (dphi * hz + tdz + tqz);
    *e = q * phi;
}

extern "C" __global__
void chain_grid_per_bead_kernel(
    const double* __restrict__ world_pos,
    const double* __restrict__ charges,
    const double* __restrict__ grid_values,
    const long long* __restrict__ grid_start,
    const double* __restrict__ grid_box,
    const int* __restrict__ grid_n,
    int n_grids,
    const double* __restrict__ far_consts, int has_far,
    double alpha, int is_born,
    int n_traj, int n_atoms,
    double* __restrict__ out_forces,
    double* __restrict__ out_atom_energy,
    int want_energy)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)n_traj * n_atoms;
    if (tid >= total) return;
    int atom = (int)(tid % n_atoms);
    double q = charges[atom];
    const double* p = world_pos + 3 * tid;
    double px = p[0], py = p[1], pz = p[2];
    double f[3] = {0.0, 0.0, 0.0};
    double e = 0.0;

    if (fabs(q) > 0.0 && isfinite(px) && isfinite(py) && isfinite(pz)) {
        if (n_grids == 1 && !has_far) {
            eval_dxgrid_clamped(
                grid_values + grid_start[0], grid_box, grid_n,
                px, py, pz, q, alpha, is_born, f, &e);
        } else {
            int chosen = -1;
            for (int g = 0; g < n_grids; ++g) {
                const double* b = grid_box + 15 * g;
                if (px > b[9] && py > b[10] && pz > b[11] &&
                    px < b[12] && py < b[13] && pz < b[14]) {
                    chosen = g;
                    break;
                }
            }
            if (chosen >= 0) {
                eval_dxgrid_clamped(
                    grid_values + grid_start[chosen],
                    grid_box + 15 * chosen, grid_n + 3 * chosen,
                    px, py, pz, q, alpha, is_born, f, &e);
            } else if (has_far && !is_born) {
                eval_far_term(far_consts, px, py, pz, q, f, &e);
            }
        }
    }
    out_forces[3 * tid]     = f[0];
    out_forces[3 * tid + 1] = f[1];
    out_forces[3 * tid + 2] = f[2];
    if (want_energy) out_atom_energy[tid] = e;
}

extern "C" __global__
void chain_target_wca_kernel(
    const double* __restrict__ world_pos,
    const double* __restrict__ chain_rad,
    const double* __restrict__ rec_pos,
    const double* __restrict__ rec_rad,
    const int* __restrict__ cell_start,
    double ox, double oy, double oz, double h,
    int nx, int ny, int nz,
    double r_rec_max, double eps,
    int n_traj, int n_atoms,
    double* __restrict__ out_forces)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)n_traj * n_atoms;
    if (tid >= total) return;
    int atom = (int)(tid % n_atoms);
    double ri = chain_rad[atom];
    if (ri < 1.0e-10) return;
    const double* p = world_pos + 3 * tid;
    double x = p[0], y = p[1], z = p[2];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;
    const double two_1_6 = 1.122462048309373;
    double reach = two_1_6 * (ri + r_rec_max) + 1.0e-6;
    if (x < ox - reach || y < oy - reach || z < oz - reach ||
        x > ox + nx * h + reach || y > oy + ny * h + reach || z > oz + nz * h + reach)
        return;
    int cx = (int)floor((x - ox) / h);
    int cy = (int)floor((y - oy) / h);
    int cz = (int)floor((z - oz) / h);
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int ix = cx - 1; ix <= cx + 1; ++ix) {
        if (ix < 0 || ix >= nx) continue;
        for (int iy = cy - 1; iy <= cy + 1; ++iy) {
            if (iy < 0 || iy >= ny) continue;
            for (int iz = cz - 1; iz <= cz + 1; ++iz) {
                if (iz < 0 || iz >= nz) continue;
                int c = (ix * ny + iy) * nz + iz;
                for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                    double dx = x - rec_pos[3 * k];
                    double dy = y - rec_pos[3 * k + 1];
                    double dz = z - rec_pos[3 * k + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double sig = ri + rec_rad[k];
                    double rcut = two_1_6 * sig;
                    if (r2 > 1.0e-20 && r2 < rcut * rcut) {
                        double sr2 = (sig * sig) / r2;
                        double sr6 = sr2 * sr2 * sr2;
                        double f = 4.0 * eps * (12.0 * sr6 * sr6 - 6.0 * sr6) / r2;
                        fx += f * dx;
                        fy += f * dy;
                        fz += f * dz;
                    }
                }
            }
        }
    }
    out_forces[3 * tid]     += fx;
    out_forces[3 * tid + 1] += fy;
    out_forces[3 * tid + 2] += fz;
}

extern "C" __global__
void chain_target_overlap_kernel(
    const double* __restrict__ world_pos,
    const double* __restrict__ chain_rad,
    const double* __restrict__ rec_pos,
    const double* __restrict__ rec_rad,
    const int* __restrict__ cell_start,
    double ox, double oy, double oz, double h,
    int nx, int ny, int nz,
    double r_rec_max,
    int n_traj, int n_atoms,
    int* __restrict__ out_overlap)
{
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)n_traj * n_atoms;
    if (tid >= total) return;
    int s = (int)(tid / n_atoms);
    if (out_overlap[s]) return;
    int atom = (int)(tid % n_atoms);
    double ri = chain_rad[atom];
    if (ri < 1.0e-10) return;
    const double* p = world_pos + 3 * tid;
    double x = p[0], y = p[1], z = p[2];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;
    double reach = ri + r_rec_max + 1.0e-6;
    if (x < ox - reach || y < oy - reach || z < oz - reach ||
        x > ox + nx * h + reach || y > oy + ny * h + reach || z > oz + nz * h + reach)
        return;
    int cx = (int)floor((x - ox) / h);
    int cy = (int)floor((y - oy) / h);
    int cz = (int)floor((z - oz) / h);
    for (int ix = cx - 1; ix <= cx + 1; ++ix) {
        if (ix < 0 || ix >= nx) continue;
        for (int iy = cy - 1; iy <= cy + 1; ++iy) {
            if (iy < 0 || iy >= ny) continue;
            for (int iz = cz - 1; iz <= cz + 1; ++iz) {
                if (iz < 0 || iz >= nz) continue;
                int c = (ix * ny + iy) * nz + iz;
                for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                    double dx = x - rec_pos[3 * k];
                    double dy = y - rec_pos[3 * k + 1];
                    double dz = z - rec_pos[3 * k + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double sig = ri + rec_rad[k];
                    if (r2 < sig * sig) {
                        out_overlap[s] = 1;
                        return;
                    }
                }
            }
        }
    }
}
"""
)

_CHAIN_MODULE = None


def _get_chain_module():
    global _CHAIN_MODULE
    if _CHAIN_MODULE is None:
        if not _CUPY:
            raise RuntimeError("CuPy is required for GPU chain kernels")
        _CHAIN_MODULE = cp.RawModule(code=_CHAIN_CUDA_CODE)
    return _CHAIN_MODULE


def _build_symmetric_csr(
    n_atoms: int,
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    *payloads: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Build deterministic symmetric CSR adjacency arrays sorted by neighbor index."""
    if i_arr.size == 0:
        ptr = np.zeros(n_atoms + 1, dtype=np.int32)
        nbr = np.zeros(0, dtype=np.int32)
        return (ptr, nbr) + tuple(p[:0].copy() for p in payloads)
    src = np.concatenate([i_arr, j_arr]).astype(np.int32)
    dst = np.concatenate([j_arr, i_arr]).astype(np.int32)
    order = np.lexsort((dst, src))
    src = src[order]
    dst = dst[order]
    counts = np.bincount(src, minlength=n_atoms).astype(np.int32)
    ptr = np.zeros(n_atoms + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts, dtype=np.int32)
    duplicated = [np.concatenate([p, p])[order] for p in payloads]
    return (ptr, dst, *duplicated)


def _color_terms(atom_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy conflict-free coloring so terms in one color touch disjoint atoms."""
    n_terms = atom_rows.shape[0]
    if n_terms == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int32)
    groups: List[List[int]] = []
    used_sets: List[set] = []
    for m in range(n_terms):
        atoms = [int(v) for v in atom_rows[m]]
        atom_set = set(atoms)
        placed = False
        if len(atom_set) == len(atoms):
            for g, used in zip(groups, used_sets):
                if used.isdisjoint(atom_set):
                    g.append(m)
                    used.update(atom_set)
                    placed = True
                    break
        if not placed:
            groups.append([m])
            used_sets.append(set(atom_set))
    perm = np.array([m for g in groups for m in g], dtype=np.int64)
    offsets = np.zeros(len(groups) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(g) for g in groups], dtype=np.int32)
    return perm, offsets


def _upload_spline_table(params, kind: str, xp):
    if params is None:
        return (
            0,
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.int64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.int64),
        )
    pots = getattr(params, kind + "_pots", None)
    if not pots:
        return (
            0,
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.int64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.float64),
            xp.zeros(0, dtype=xp.int64),
        )
    cache = params.__dict__.setdefault("_tables", {})
    if kind not in cache:
        cache[kind] = _spline_table(pots)
    tab = cache[kind]
    coef4 = np.ascontiguousarray(tab["coef"].T, dtype=np.float64).ravel()
    return (
        1,
        xp.asarray(coef4, dtype=xp.float64),
        xp.asarray(tab["start"], dtype=xp.int64),
        xp.asarray(tab["x_min"], dtype=xp.float64),
        xp.asarray(tab["x_max"], dtype=xp.float64),
        xp.asarray(tab["dx"], dtype=xp.float64),
        xp.asarray(tab["n_int"], dtype=xp.int64),
    )


class ChainInternalPlan:
    """Device-resident topology and force tables for batched internal chain BD."""

    def __init__(
        self,
        common: ChainCommon,
        dt_chain: float,
        kT: float = 1.0,
        viscosity: float = WATER_VISCOSITY,
        soft_repulsion: bool = False,
        soft_repulsion_eps: float = 1.0,
        xp=None,
    ):
        if xp is None:
            xp = cp if _CUPY else np
        self.xp = xp
        self.common = common
        self.n_atoms = common.n_atoms
        self.dt_chain = float(dt_chain)
        self.kT = float(kT)
        self.viscosity = float(viscosity)
        self.soft_repulsion = bool(soft_repulsion)
        self.soft_repulsion_eps = float(soft_repulsion_eps)

        radii = np.array([a.radius for a in common.atoms], dtype=np.float64)
        safe_rad = np.where(radii > 0.0, radii, 1.0)
        mob = np.where(radii > 0.0, 1.0 / (6.0 * math.pi * self.viscosity * safe_rad), 0.0)
        sigma = np.sqrt(2.0 * self.kT * mob * self.dt_chain)
        self.radii = xp.asarray(radii, dtype=xp.float64)
        self.mob = xp.asarray(mob, dtype=xp.float64)
        self.sigma = xp.asarray(sigma, dtype=xp.float64)

        terms = _term_arrays(common)
        ba, bb, br0, bk = terms["bond"]
        na, nb, nr0, nk = terms["network"]
        all_ba = np.concatenate([ba, na]) if na.size else ba
        all_bb = np.concatenate([bb, nb]) if nb.size else bb
        all_r0 = np.concatenate([br0, nr0]) if nr0.size else br0
        all_k = np.concatenate([bk, nk]) if nk.size else bk
        b_ptr, b_nbr, b_r0, b_k = _build_symmetric_csr(
            self.n_atoms, all_ba, all_bb, all_r0, all_k
        )
        self.bond_ptr = xp.asarray(b_ptr, dtype=xp.int32)
        self.bond_nbr = xp.asarray(b_nbr, dtype=xp.int32)
        self.bond_r0 = xp.asarray(b_r0, dtype=xp.float64)
        self.bond_k = xp.asarray(b_k, dtype=xp.float64)

        params = common.coffdrop_params
        if params is not None and common.pair_lookups:
            ijt = np.array(
                [(i, j, t) for (i, j), t in common.pair_lookups.items()],
                dtype=np.int64,
            ).reshape(-1, 3)
            p_ptr, p_nbr, p_t = _build_symmetric_csr(
                self.n_atoms, ijt[:, 0], ijt[:, 1], ijt[:, 2]
            )
            self.has_pairs = 1
            self.pair_ptr = xp.asarray(p_ptr, dtype=xp.int32)
            self.pair_nbr = xp.asarray(p_nbr, dtype=xp.int32)
            self.pair_t = xp.asarray(p_t, dtype=xp.int64)
        else:
            self.has_pairs = 0
            self.pair_ptr = xp.zeros(self.n_atoms + 1, dtype=xp.int32)
            self.pair_nbr = xp.zeros(0, dtype=xp.int32)
            self.pair_t = xp.zeros(0, dtype=xp.int64)
        (
            _hp,
            self.pair_coef4,
            self.pair_start,
            self.pair_xmin,
            self.pair_xmax,
            self.pair_dx,
            self.pair_nint,
        ) = _upload_spline_table(params if self.has_pairs else None, "pair", xp)

        if self.soft_repulsion and self.n_atoms >= 2:
            skip = bonded_exclusions(common)
            if self.has_pairs:
                ij = np.array(list(common.pair_lookups), dtype=np.int64).reshape(-1, 2)
                skip[ij[:, 0], ij[:, 1]] = True
                skip[ij[:, 1], ij[:, 0]] = True
            iu, ju = np.triu_indices(self.n_atoms, k=1)
            live = radii >= 1.0e-10
            m = ~skip[iu, ju] & live[iu] & live[ju]
            iu, ju = iu[m], ju[m]
            w_ptr, w_nbr, w_sig = _build_symmetric_csr(
                self.n_atoms, iu, ju, radii[iu] + radii[ju]
            )
            self.has_wca = int(w_nbr.size > 0)
            self.wca_ptr = xp.asarray(w_ptr, dtype=xp.int32)
            self.wca_nbr = xp.asarray(w_nbr, dtype=xp.int32)
            self.wca_sig = xp.asarray(w_sig, dtype=xp.float64)
        else:
            self.has_wca = 0
            self.wca_ptr = xp.zeros(self.n_atoms + 1, dtype=xp.int32)
            self.wca_nbr = xp.zeros(0, dtype=xp.int32)
            self.wca_sig = xp.zeros(0, dtype=xp.float64)

        aa, ab, ac, ath0, ak, atyp = terms["angle"]
        if aa.size:
            ang_rows = np.stack([aa, ab, ac], axis=1)
            perm, off = _color_terms(ang_rows)
            aa, ab, ac = aa[perm], ab[perm], ac[perm]
            ath0, ak, atyp = ath0[perm], ak[perm], atyp[perm]
            self.n_ang_colors = len(off) - 1
            self.ang_color_off = xp.asarray(off, dtype=xp.int32)
        else:
            self.n_ang_colors = 0
            self.ang_color_off = xp.zeros(1, dtype=xp.int32)
        self.ang_a = xp.asarray(aa, dtype=xp.int32)
        self.ang_b = xp.asarray(ab, dtype=xp.int32)
        self.ang_c = xp.asarray(ac, dtype=xp.int32)
        self.ang_theta0 = xp.asarray(ath0, dtype=xp.float64)
        self.ang_k = xp.asarray(ak, dtype=xp.float64)
        self.ang_type = xp.asarray(atyp, dtype=xp.int64)
        (
            self.has_ang_tab,
            self.ang_coef4,
            self.ang_start,
            self.ang_xmin,
            self.ang_xmax,
            self.ang_dx,
            self.ang_nint,
        ) = _upload_spline_table(params, "angle", xp)

        ta, tb, tc, td, tphi0, tk, tn, ttyp = terms["torsion"]
        if ta.size:
            tor_rows = np.stack([ta, tb, tc, td], axis=1)
            perm, off = _color_terms(tor_rows)
            ta, tb, tc, td = ta[perm], tb[perm], tc[perm], td[perm]
            tphi0, tk, tn, ttyp = tphi0[perm], tk[perm], tn[perm], ttyp[perm]
            self.n_tor_colors = len(off) - 1
            self.tor_color_off = xp.asarray(off, dtype=xp.int32)
        else:
            self.n_tor_colors = 0
            self.tor_color_off = xp.zeros(1, dtype=xp.int32)
        self.tor_a = xp.asarray(ta, dtype=xp.int32)
        self.tor_b = xp.asarray(tb, dtype=xp.int32)
        self.tor_c = xp.asarray(tc, dtype=xp.int32)
        self.tor_d = xp.asarray(td, dtype=xp.int32)
        self.tor_phi0 = xp.asarray(tphi0, dtype=xp.float64)
        self.tor_k = xp.asarray(tk, dtype=xp.float64)
        self.tor_n = xp.asarray(tn, dtype=xp.float64)
        self.tor_type = xp.asarray(ttyp, dtype=xp.int64)
        (
            self.has_tor_tab,
            self.tor_coef4,
            self.tor_start,
            self.tor_xmin,
            self.tor_xmax,
            self.tor_dx,
            self.tor_nint,
        ) = _upload_spline_table(params, "dihedral", xp)

        lc = common.length_constraints
        self.n_len_c = len(lc)
        self.lc_a = xp.asarray([_chain_idx(c.a) for c in lc], dtype=xp.int32)
        self.lc_b = xp.asarray([_chain_idx(c.b) for c in lc], dtype=xp.int32)
        self.lc_len = xp.asarray([c.length for c in lc], dtype=xp.float64)

        cc = common.coplanar_constraints
        self.n_cop_c = len(cc)
        self.cc_a = xp.asarray([_chain_idx(c.a) for c in cc], dtype=xp.int32)
        self.cc_b = xp.asarray([_chain_idx(c.b) for c in cc], dtype=xp.int32)
        self.cc_c = xp.asarray([_chain_idx(c.c) for c in cc], dtype=xp.int32)
        self.cc_d = xp.asarray([_chain_idx(c.d) for c in cc], dtype=xp.int32)

        self._work_old = None
        self._work_F = None
        if xp is not np:
            mod = _get_chain_module()
            self._forces_kernel = mod.get_function("chain_forces_only_kernel")
            self._inner_shmem_kernel = mod.get_function("chain_inner_steps_shmem_kernel")
            self._inner_global_kernel = mod.get_function("chain_inner_steps_global_kernel")

    def _force_table_args(self):
        return (
            self.bond_ptr,
            self.bond_nbr,
            self.bond_r0,
            self.bond_k,
            self.pair_ptr,
            self.pair_nbr,
            self.pair_t,
            np.int32(self.has_pairs),
            self.pair_coef4,
            self.pair_start,
            self.pair_xmin,
            self.pair_xmax,
            self.pair_dx,
            self.pair_nint,
            self.wca_ptr,
            self.wca_nbr,
            self.wca_sig,
            np.int32(self.has_wca),
            np.float64(self.soft_repulsion_eps),
            np.int32(self.n_ang_colors),
            self.ang_color_off,
            self.ang_a,
            self.ang_b,
            self.ang_c,
            self.ang_theta0,
            self.ang_k,
            self.ang_type,
            np.int32(self.has_ang_tab),
            self.ang_coef4,
            self.ang_start,
            self.ang_xmin,
            self.ang_xmax,
            self.ang_dx,
            self.ang_nint,
            np.int32(self.n_tor_colors),
            self.tor_color_off,
            self.tor_a,
            self.tor_b,
            self.tor_c,
            self.tor_d,
            self.tor_phi0,
            self.tor_k,
            self.tor_n,
            self.tor_type,
            np.int32(self.has_tor_tab),
            self.tor_coef4,
            self.tor_start,
            self.tor_xmin,
            self.tor_xmax,
            self.tor_dx,
            self.tor_nint,
        )

    def evaluate_forces_batch(self, body_pos):
        """Return (B, n_atoms, 3) internal chain forces in kBT/A."""
        xp = self.xp
        body_pos = xp.ascontiguousarray(body_pos, dtype=xp.float64)
        n_traj = body_pos.shape[0]
        if n_traj == 0:
            return xp.zeros((0, self.n_atoms, 3), dtype=xp.float64)
        if xp is np:
            out = np.zeros((n_traj, self.n_atoms, 3), dtype=np.float64)
            st = ChainState.from_template(self.common, body_pos[0].copy())
            for b in range(n_traj):
                st.positions[:] = body_pos[b]
                compute_chain_forces(
                    st,
                    kT=self.kT,
                    soft_repulsion=self.soft_repulsion,
                    soft_repulsion_eps=self.soft_repulsion_eps,
                )
                out[b] = st.forces
            return out
        out = xp.empty_like(body_pos)
        threads = min(256, max(32, 1 << (max(self.n_atoms - 1, 1)).bit_length()))
        self._forces_kernel(
            (n_traj,),
            (threads,),
            (body_pos, out, np.int32(n_traj), np.int32(self.n_atoms))
            + self._force_table_args(),
        )
        return out

    def step_internal_batch(
        self,
        body_pos,
        ext_body_forces,
        traj_ids,
        step_base,
        n_inner_arr,
        rng_key: np.ndarray,
        stream: int,
        apply_constraints: bool = True,
        constraint_tol: float = 1.0e-6,
        constraint_max_iter: int = 200,
        remove_rigid_rotation: bool = False,
        supplied_normals=None,
        force_global_kernel: bool = False,
    ) -> None:
        """Advance body_pos (B, n_atoms, 3) in place over n_inner_arr[b] substeps."""
        xp = self.xp
        n_traj = body_pos.shape[0]
        if n_traj == 0:
            return
        if xp is np:
            from pystarc.simulation.chain_simulator import chain_internal_bd_step

            st = ChainState.from_template(self.common, body_pos[0].copy())
            for b in range(n_traj):
                st.positions[:] = body_pos[b]
                n_in = int(n_inner_arr[b])
                ext_b = None if ext_body_forces is None else ext_body_forces[b]
                if supplied_normals is not None:
                    z_b = np.asarray(supplied_normals[b], dtype=np.float64)
                    if z_b.ndim == 2:
                        z_b = z_b[None, :, :]

                    class _FixedNormalRNG:
                        def __init__(self, z_seq):
                            self._z = z_seq
                            self._idx = 0

                        def standard_normal(self, shape):
                            val = self._z[self._idx]
                            self._idx += 1
                            return val.reshape(shape)

                    rng = _FixedNormalRNG(z_b)
                else:
                    seq = np.random.SeedSequence(
                        [
                            int(rng_key[0]),
                            int(rng_key[1]),
                            int(traj_ids[b]),
                            int(step_base[b]),
                            int(stream),
                        ]
                    )
                    rng = np.random.default_rng(seq)
                for _ in range(n_in):
                    chain_internal_bd_step(
                        st,
                        dt=self.dt_chain,
                        rng=rng,
                        kT=self.kT,
                        viscosity=self.viscosity,
                        apply_constraints=apply_constraints,
                        constraint_tol=constraint_tol,
                        constraint_max_iter=constraint_max_iter,
                        soft_repulsion=self.soft_repulsion,
                        soft_repulsion_eps=self.soft_repulsion_eps,
                        remove_rigid_rotation=remove_rigid_rotation,
                        external_forces=ext_b,
                    )
                body_pos[b] = st.positions
            return

        has_ext = ext_body_forces is not None
        ext_arg = (
            xp.ascontiguousarray(ext_body_forces, dtype=xp.float64)
            if has_ext
            else body_pos
        )
        has_z = supplied_normals is not None
        z_arg = (
            xp.ascontiguousarray(supplied_normals, dtype=xp.float64)
            if has_z
            else body_pos
        )
        traj_ids = xp.ascontiguousarray(traj_ids, dtype=xp.int64)
        step_base = xp.ascontiguousarray(step_base, dtype=xp.int64)
        n_inner_arr = xp.ascontiguousarray(n_inner_arr, dtype=xp.int32)
        c_failed = xp.zeros(n_traj, dtype=xp.int32)
        threads = 128 if self.n_atoms <= 160 else 256

        constr_args = (
            np.int32(1 if apply_constraints else 0),
            np.int32(self.n_len_c),
            self.lc_a,
            self.lc_b,
            self.lc_len,
            np.int32(self.n_cop_c),
            self.cc_a,
            self.cc_b,
            self.cc_c,
            self.cc_d,
            np.float64(constraint_tol),
            np.int32(constraint_max_iter),
            np.int32(1 if remove_rigid_rotation else 0),
            c_failed,
        )

        if self.n_atoms <= _SHMEM_MAX_ATOMS and not force_global_kernel:
            shmem_bytes = (9 * self.n_atoms + 9 * threads + 10) * 8
            self._inner_shmem_kernel(
                (n_traj,),
                (threads,),
                (
                    body_pos,
                    ext_arg,
                    np.int32(1 if has_ext else 0),
                    z_arg,
                    np.int32(1 if has_z else 0),
                    traj_ids,
                    step_base,
                    n_inner_arr,
                    np.int32(n_traj),
                    np.int32(self.n_atoms),
                    np.uint32(stream),
                    np.uint32(rng_key[0]),
                    np.uint32(rng_key[1]),
                    np.float64(self.dt_chain),
                    self.mob,
                    self.sigma,
                )
                + self._force_table_args()
                + constr_args,
                shared_mem=shmem_bytes,
            )
        else:
            if self._work_old is None or self._work_old.shape[0] < n_traj:
                self._work_old = xp.empty((n_traj, self.n_atoms, 3), dtype=xp.float64)
                self._work_F = xp.empty((n_traj, self.n_atoms, 3), dtype=xp.float64)
            shmem_bytes = (9 * threads + 10) * 8
            self._inner_global_kernel(
                (n_traj,),
                (threads,),
                (
                    body_pos,
                    self._work_old[:n_traj],
                    self._work_F[:n_traj],
                    ext_arg,
                    np.int32(1 if has_ext else 0),
                    z_arg,
                    np.int32(1 if has_z else 0),
                    traj_ids,
                    step_base,
                    n_inner_arr,
                    np.int32(n_traj),
                    np.int32(self.n_atoms),
                    np.uint32(stream),
                    np.uint32(rng_key[0]),
                    np.uint32(rng_key[1]),
                    np.float64(self.dt_chain),
                    self.mob,
                    self.sigma,
                )
                + self._force_table_args()
                + constr_args,
                shared_mem=shmem_bytes,
            )

        if (self.n_len_c > 0 or self.n_cop_c > 0) and int(c_failed.sum()) > 0:
            bad = cp.asnumpy(cp.where(c_failed != 0)[0])
            st = ChainState.from_template(self.common, np.zeros((self.n_atoms, 3)))
            for idx in bad:
                st.positions[:] = cp.asnumpy(body_pos[int(idx)])
                satisfy_constraints_hybrid(
                    st,
                    tol=constraint_tol,
                    shake_max_iter=constraint_max_iter,
                    newton_max_iter=50,
                )
                if remove_rigid_rotation:
                    st.positions -= st.positions.mean(axis=0)
                body_pos[int(idx)] = cp.asarray(st.positions)


def _hct_integrand_xp(r, rho_tilde_i, rho_S_j, xp):
    case_engulf = rho_tilde_i >= r + rho_S_j
    L = xp.where(case_engulf, 1.0, xp.maximum(rho_tilde_i, xp.abs(r - rho_S_j)))
    U = xp.where(case_engulf, 1.0, r + rho_S_j)
    L = xp.maximum(L, 1.0e-10)
    U = xp.maximum(U, 1.0e-10)
    r_safe = xp.maximum(r, 1.0e-10)
    inv_L = 1.0 / L
    inv_U = 1.0 / U
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U
    val = 0.5 * (
        inv_L
        - inv_U
        + (r_safe / 4.0) * (inv_U2 - inv_L2)
        + (1.0 / (2.0 * r_safe)) * xp.log(L / U)
        + (rho_S_j * rho_S_j / (4.0 * r_safe)) * (inv_L2 - inv_U2)
    )
    return xp.where(case_engulf, 0.0, val)


def _hct_integrand_deriv_xp(r, rho_tilde_i, rho_S_j, xp):
    case_engulf = rho_tilde_i >= r + rho_S_j
    abs_rm = xp.abs(r - rho_S_j)
    wins = rho_tilde_i >= abs_rm
    L = xp.where(case_engulf, 1.0, xp.maximum(rho_tilde_i, abs_rm))
    U = xp.where(case_engulf, 1.0, r + rho_S_j)
    L = xp.maximum(L, 1.0e-10)
    U = xp.maximum(U, 1.0e-10)
    r_safe = xp.maximum(r, 1.0e-10)
    dL = xp.where(wins, 0.0, xp.sign(r - rho_S_j))
    inv_L = 1.0 / L
    inv_U = 1.0 / U
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U
    inv_L3 = inv_L2 * inv_L
    inv_U3 = inv_U2 * inv_U
    inv_r2 = 1.0 / (r_safe * r_safe)
    dT1 = -inv_L2 * dL
    dT2 = inv_U2
    dT3 = 0.25 * (inv_U2 - inv_L2) + (r_safe / 4.0) * (
        -2.0 * inv_U3 + 2.0 * inv_L3 * dL
    )
    dT4 = -0.5 * xp.log(L / U) * inv_r2 + (1.0 / (2.0 * r_safe)) * (
        dL * inv_L - inv_U
    )
    rho_S_sq = rho_S_j * rho_S_j
    dT5 = -0.25 * rho_S_sq * inv_r2 * (inv_L2 - inv_U2) + (
        rho_S_sq / (4.0 * r_safe)
    ) * (-2.0 * inv_L3 * dL + 2.0 * inv_U3)
    return xp.where(case_engulf, 0.0, 0.5 * (dT1 + dT2 + dT3 + dT4 + dT5))


def chain_full_gb_force_batch(
    positions_batch,
    charges,
    intrinsic_radii,
    eps_in: float = 1.0,
    eps_out: float = WATER_DIELECTRIC,
    coffdrop_active: bool = False,
    exclude_pair_mask=None,
    obc_kwargs: Optional[dict] = None,
    xp=None,
    chunk_size: int = 64,
):
    """Batched GB-OBC2 per-bead forces (B, n_atoms, 3) and total energies (B,)."""
    if xp is None:
        xp = cp if (_CUPY and isinstance(positions_batch, cp.ndarray)) else np
    pos_all = xp.asarray(positions_batch, dtype=xp.float64)
    q = xp.asarray(charges, dtype=xp.float64)
    rho = xp.asarray(intrinsic_radii, dtype=xp.float64)
    B, n, _ = pos_all.shape

    kw = obc_kwargs or {}
    obc_alpha = float(kw.get("obc_alpha", DEFAULT_OBC_ALPHA))
    obc_beta = float(kw.get("obc_beta", DEFAULT_OBC_BETA))
    obc_gamma = float(kw.get("obc_gamma", DEFAULT_OBC_GAMMA))
    obc_offset = float(kw.get("obc_offset", DEFAULT_OBC_OFFSET))
    hct_scale = float(kw.get("hct_scale", DEFAULT_HCT_SCALE))

    rho_tilde = rho - obc_offset
    if bool((rho_tilde <= 0.0).any()):
        raise ValueError("rho_tilde must be positive for all atoms")
    rho_S = hct_scale * rho

    cf = (1.0 / eps_in - 1.0 / eps_out) * COULOMB_K_KBT_A
    eye = xp.eye(n, dtype=bool)[None, :, :]
    off_diag = ~eye
    upper = xp.triu(xp.ones((n, n), dtype=bool), k=1)[None, :, :]
    qq = q[:, None] * q[None, :]
    if not coffdrop_active:
        allowed_coul = ~xp.eye(n, dtype=bool)
        if exclude_pair_mask is not None:
            excl = xp.asarray(exclude_pair_mask, dtype=bool)
            allowed_coul = allowed_coul & ~(excl | excl.T)
        upper_coul = xp.triu(xp.ones((n, n), dtype=bool), k=1) & allowed_coul

    out_f = xp.empty_like(pos_all)
    out_e = xp.empty(B, dtype=xp.float64)
    step = max(1, min(chunk_size, int(256 * 1024 * 1024 // max(1, n * n * 24))))

    for b0 in range(0, B, step):
        b1 = min(b0 + step, B)
        pos = pos_all[b0:b1]
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        r = xp.linalg.norm(diff, axis=3)
        r_safe = xp.where(eye, 1.0, xp.maximum(r, 1.0e-10))
        unit = xp.where(off_diag[:, :, :, None], diff / r_safe[:, :, :, None], 0.0)

        rt_i = rho_tilde[None, :, None]
        rs_j = rho_S[None, None, :]
        integ = xp.where(eye, 0.0, _hct_integrand_xp(r, rt_i, rs_j, xp))
        K = xp.where(eye, 0.0, _hct_integrand_deriv_xp(r, rt_i, rs_j, xp))

        I = xp.sum(integ, axis=2)
        psi = rho_tilde[None, :] * I
        arg = obc_alpha * psi - obc_beta * psi * psi + obc_gamma * psi * psi * psi
        th = xp.tanh(arg)
        sech2 = 1.0 - th * th
        inv_R = xp.maximum(1.0 / rho_tilde[None, :] - th / rho[None, :], 1.0e-10)
        R_eff = 1.0 / inv_R
        darg = obc_alpha - 2.0 * obc_beta * psi + 3.0 * obc_gamma * psi * psi
        D_chain = R_eff * R_eff * sech2 / rho[None, :] * darg * rho_tilde[None, :]

        q2 = (q * q)[None, :]
        e_chunk = -0.5 * cf * xp.sum(q2 / R_eff, axis=1)
        eta = 0.5 * cf * q2 / (R_eff * R_eff) * D_chain
        ku = xp.einsum("bkj,bkjd->bkd", K, unit)
        f_chunk = -eta[:, :, None] * ku + xp.einsum("bi,bik,bikd->bkd", eta, K, unit)

        if not coffdrop_active:
            R_i = R_eff[:, :, None]
            R_j = R_eff[:, None, :]
            A = R_i * R_j
            A_safe = xp.where(eye, 1.0, xp.maximum(A, 1.0e-10))
            D_grid = -r_safe * r_safe / (4.0 * A_safe)
            expD = xp.exp(D_grid)
            f_GB = xp.sqrt(r_safe * r_safe + A_safe * expD)
            inv_f = xp.where(eye, 0.0, 1.0 / f_GB)
            inv_f2 = inv_f * inv_f
            e_chunk = e_chunk - cf * xp.sum(
                xp.where(upper, qq[None, :, :] * inv_f, 0.0), axis=(1, 2)
            )
            df_dr = xp.where(eye, 0.0, r_safe * (1.0 - 0.25 * expD) * inv_f)
            w_dir = xp.where(eye, 0.0, qq[None, :, :] * inv_f2 * df_dr)
            f_dir = -cf * xp.einsum("bkm,bkmd->bkd", w_dir, unit)
            df_dR = xp.where(eye, 0.0, 0.5 * R_j * expD * (1.0 - D_grid) * inv_f)
            sum_m = xp.sum((inv_f2 * df_dR) * q[None, None, :], axis=2)
            eta_off = D_chain * cf * q[None, :] * sum_m
            f_ind = -eta_off[:, :, None] * ku + xp.einsum(
                "bi,bik,bikd->bkd", eta_off, K, unit
            )

            K_fac = COULOMB_K_KBT_A / eps_in
            inv_r = xp.where(eye, 0.0, 1.0 / r_safe)
            inv_r3 = inv_r * inv_r * inv_r
            e_chunk = e_chunk + K_fac * xp.sum(
                xp.where(upper_coul[None, :, :], qq[None, :, :] * inv_r, 0.0),
                axis=(1, 2),
            )
            w_coul = qq[None, :, :] * inv_r3 * allowed_coul[None, :, :]
            f_coul = K_fac * xp.einsum("bkm,bkmd->bkd", w_coul, diff)
            f_chunk = f_chunk + f_dir + f_ind + f_coul

        out_f[b0:b1] = f_chunk
        out_e[b0:b1] = e_chunk

    return out_f, out_e


def _build_target_cells(rec_pos: np.ndarray, rec_rad: np.ndarray, h: float):
    if rec_rad.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(2, dtype=np.int32),
            np.zeros(3, dtype=np.float64),
            1.0,
            (1, 1, 1),
            0.0,
        )
    h = max(float(h), 1.0)
    origin = rec_pos.min(axis=0) - 1.0e-6
    span = rec_pos.max(axis=0) - origin
    dims = tuple(int(v) for v in (np.floor(span / h).astype(np.int64) + 1))
    nx, ny, nz = dims
    cx = np.clip(np.floor((rec_pos[:, 0] - origin[0]) / h).astype(np.int64), 0, nx - 1)
    cy = np.clip(np.floor((rec_pos[:, 1] - origin[1]) / h).astype(np.int64), 0, ny - 1)
    cz = np.clip(np.floor((rec_pos[:, 2] - origin[2]) / h).astype(np.int64), 0, nz - 1)
    cell_id = (cx * ny + cy) * nz + cz
    order = np.argsort(cell_id, kind="stable")
    sorted_pos = np.ascontiguousarray(rec_pos[order], dtype=np.float64)
    sorted_rad = np.ascontiguousarray(rec_rad[order], dtype=np.float64)
    counts = np.bincount(cell_id[order], minlength=nx * ny * nz).astype(np.int32)
    cell_start = np.zeros(nx * ny * nz + 1, dtype=np.int32)
    cell_start[1:] = np.cumsum(counts, dtype=np.int32)
    return sorted_pos, sorted_rad, cell_start, origin, h, dims, float(rec_rad.max())


class GPUChainTargetField:
    """Target grids, steric cell list, and overlap evaluator for chain BD."""

    def __init__(
        self,
        target,
        chain_charges: np.ndarray,
        chain_radii: np.ndarray,
        elec_grids: Optional[list] = None,
        born_grids: Optional[list] = None,
        desolvation_alpha: float = 1.0,
        use_far_field: bool = False,
        target_charge: float = 0.0,
        debye_length: float = 7.858,
        sdie: float = 78.0,
        kT_scale: float = 1.0,
        multipole_expansion=None,
        xp=None,
    ):
        if xp is None:
            xp = cp if _CUPY else np
        self.xp = xp
        self.target = target
        self.alpha = float(desolvation_alpha)
        self.chain_charges_np = np.asarray(chain_charges, dtype=np.float64)
        self.chain_radii_np = np.asarray(chain_radii, dtype=np.float64)
        self.chain_charges = xp.asarray(self.chain_charges_np, dtype=xp.float64)
        self.chain_radii = xp.asarray(self.chain_radii_np, dtype=xp.float64)
        self.n_atoms = len(self.chain_radii_np)

        self.elec_grids_cpu = sorted(
            list(elec_grids or []), key=lambda g: float(np.prod(np.diag(g.delta)))
        )
        self.born_grids_cpu = sorted(
            list(born_grids or []), key=lambda g: float(np.prod(np.diag(g.delta)))
        )

        self.use_far_field = bool(use_far_field) and debye_length > 0.0
        self.multipole = multipole_expansion
        eps_s = sdie * VACUUM_PERMITTIVITY_KBT * kT_scale
        self._V_factor = target_charge / (4.0 * math.pi * eps_s) if eps_s > 0 else 0.0
        self._debye = float(debye_length)
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
                far_c[2] = multipole_expansion.four_pi_eps
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
        self.far_consts = xp.asarray(far_c, dtype=xp.float64)

        self.elec_uploaded = self._pack_grids(
            self.elec_grids_cpu,
            margin=3.0 if (len(self.elec_grids_cpu) > 1 or self.has_far) else 0.0,
            coarsest_zero_margin=(not self.has_far),
        )
        self.born_uploaded = self._pack_grids(
            self.born_grids_cpu,
            margin=1.0 if len(self.born_grids_cpu) > 1 else 0.0,
            coarsest_zero_margin=True,
        )

        rec_pos, rec_rad = target_spheres(target)
        r_chain_max = (
            float(self.chain_radii_np.max()) if self.chain_radii_np.size else 0.0
        )
        r_rec_max = float(rec_rad.max()) if rec_rad.size else 0.0
        cell_h = 2.0 ** (1.0 / 6.0) * (r_chain_max + r_rec_max) + 1.0e-5
        (
            s_pos,
            s_rad,
            c_start,
            c_orig,
            c_h,
            c_dims,
            r_rmax,
        ) = _build_target_cells(rec_pos, rec_rad, cell_h)
        self.has_spheres = rec_rad.size > 0
        self.rec_pos = xp.asarray(s_pos, dtype=xp.float64)
        self.rec_rad = xp.asarray(s_rad, dtype=xp.float64)
        self.cell_start = xp.asarray(c_start, dtype=xp.int32)
        self.cell_origin = c_orig
        self.cell_h = c_h
        self.cell_dims = c_dims
        self.r_rec_max = r_rmax
        self._spheres_cpu = (rec_pos, rec_rad)

        if target is not None and len(target.atoms) > 0:
            self.target_all_xyz = np.array(
                [[a.x, a.y, a.z] for a in target.atoms], dtype=np.float64
            )
        else:
            self.target_all_xyz = None

        if xp is not np:
            mod = _get_chain_module()
            self._grid_kernel = mod.get_function("chain_grid_per_bead_kernel")
            self._wca_kernel = mod.get_function("chain_target_wca_kernel")
            self._overlap_kernel = mod.get_function("chain_target_overlap_kernel")

    def _pack_grids(self, grids: list, margin: float, coarsest_zero_margin: bool):
        xp = self.xp
        if not grids:
            return None
        starts = np.cumsum([0] + [g.data.size for g in grids], dtype=np.int64)
        values = xp.asarray(
            np.concatenate([g.data.ravel() for g in grids]), dtype=xp.float64
        )
        boxes = []
        dims = []
        n_g = len(grids)
        for idx, g in enumerate(grids):
            sp = np.diag(g.delta).astype(np.float64)
            inv_sp = 1.0 / sp
            nx, ny, nz = g.data.shape
            sh = np.array([nx, ny, nz], dtype=np.float64)
            m = 0.0 if (coarsest_zero_margin and idx == n_g - 1) else margin
            lo = g.origin + m * sp if m > 0.0 else g.origin - 1.0e-12
            hi = (
                g.origin + (sh - 1.0 - m) * sp
                if m > 0.0
                else g.origin + (sh - 1.0) * sp
            )
            boxes.append(np.concatenate([g.origin, inv_sp, sp, lo, hi]))
            dims.append([nx, ny, nz])
        return {
            "n_grids": n_g,
            "values": values,
            "start": xp.asarray(starts[:-1], dtype=xp.int64),
            "box": xp.asarray(np.concatenate(boxes), dtype=xp.float64),
            "n": xp.asarray(np.array(dims, dtype=np.int32).ravel(), dtype=xp.int32),
        }

    def _eval_far_cpu(self, sub_p: np.ndarray, sub_q: np.ndarray):
        c = self._far_c_np
        V, lam, fpe = c[0], c[1], c[2]
        p = c[4:7]
        Q = c[7:16].reshape(3, 3)
        sr = np.maximum(np.linalg.norm(sub_p, axis=1), 1.0)
        ex = np.exp(-sr / lam)
        h = sub_p / sr[:, None]
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
            grad = grad + dphi_d[:, None] * h + w[:, None] * (p[None, :] - pr[:, None] * h)
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
        return -sub_q[:, None] * grad, float(np.sum(sub_q * phi))

    def _eval_grid_cpu(self, world_pos: np.ndarray, is_born: bool, want_energy: bool):
        from pystarc.simulation.chain_simulator import (
            evaluate_born_force_on_chain,
            evaluate_target_grid_force_on_chain,
        )

        B = world_pos.shape[0]
        forces = np.zeros_like(world_pos)
        energies = np.zeros(B, dtype=np.float64) if want_energy else None
        grids = self.born_grids_cpu if is_born else self.elec_grids_cpu
        if not grids and not (not is_born and self.has_far):
            return forces, energies
        if is_born and self.alpha == 0.0:
            return forces, energies
        for b in range(B):
            if len(grids) == 1 and not (not is_born and self.has_far):
                if is_born:
                    fb, eb = evaluate_born_force_on_chain(
                        world_pos[b], self.chain_charges_np, grids[0], alpha=self.alpha
                    )
                else:
                    fb, eb = evaluate_target_grid_force_on_chain(
                        world_pos[b], self.chain_charges_np, grids[0]
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
                    sp = np.diag(g.delta)
                    m = (
                        0.0
                        if ((is_born or not self.has_far) and idx == len(grids) - 1)
                        else (1.0 if is_born else 3.0)
                    )
                    lo = g.origin + m * sp if m > 0.0 else g.origin - 1.0e-12
                    hi = g.origin + (np.array(g.data.shape) - 1 - m) * sp
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
                        fb, eb = evaluate_born_force_on_chain(
                            sub_pts, sub_q, g, alpha=self.alpha
                        )
                    else:
                        fb, eb = evaluate_target_grid_force_on_chain(sub_pts, sub_q, g)
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

    def _eval_grid_gpu(self, world_pos, is_born: bool, want_energy: bool):
        xp = self.xp
        B = world_pos.shape[0]
        forces = xp.zeros((B, self.n_atoms, 3), dtype=xp.float64)
        energies = xp.zeros(B, dtype=xp.float64) if want_energy else None
        packed = self.born_uploaded if is_born else self.elec_uploaded
        has_far = 0 if is_born else self.has_far
        if packed is None and not has_far:
            return forces, energies
        if is_born and self.alpha == 0.0:
            return forces, energies
        total = B * self.n_atoms
        if total == 0:
            return forces, energies
        atom_e = xp.empty((B, self.n_atoms), dtype=xp.float64) if want_energy else forces
        n_grids = 0 if packed is None else packed["n_grids"]
        vals = forces if packed is None else packed["values"]
        st = xp.zeros(1, dtype=xp.int64) if packed is None else packed["start"]
        bx = forces if packed is None else packed["box"]
        gn = xp.zeros(3, dtype=xp.int32) if packed is None else packed["n"]
        self._grid_kernel(
            ((total + 255) // 256,),
            (256,),
            (
                xp.ascontiguousarray(world_pos, dtype=xp.float64),
                self.chain_charges,
                vals,
                st,
                bx,
                gn,
                np.int32(n_grids),
                self.far_consts,
                np.int32(has_far),
                np.float64(self.alpha),
                np.int32(1 if is_born else 0),
                np.int32(B),
                np.int32(self.n_atoms),
                forces,
                atom_e,
                np.int32(1 if want_energy else 0),
            ),
        )
        if want_energy:
            energies = atom_e.sum(axis=1)
        return forces, energies

    def evaluate_elec(self, world_pos, want_energy: bool = False):
        if self.xp is np:
            return self._eval_grid_cpu(world_pos, False, want_energy)
        return self._eval_grid_gpu(world_pos, False, want_energy)

    def evaluate_born(self, world_pos, want_energy: bool = False):
        if self.xp is np:
            return self._eval_grid_cpu(world_pos, True, want_energy)
        return self._eval_grid_gpu(world_pos, True, want_energy)

    def add_steric_forces(self, world_pos, forces, eps: float = 1.0) -> None:
        if not self.has_spheres:
            return
        B = world_pos.shape[0]
        if B == 0:
            return
        if self.xp is np:
            from pystarc.simulation.chain_simulator import chain_target_steric_forces

            for b in range(B):
                forces[b] += chain_target_steric_forces(
                    world_pos[b], self.chain_radii_np, self._spheres_cpu, eps=eps
                )
            return
        total = B * self.n_atoms
        nx, ny, nz = self.cell_dims
        self._wca_kernel(
            ((total + 255) // 256,),
            (256,),
            (
                self.xp.ascontiguousarray(world_pos, dtype=self.xp.float64),
                self.chain_radii,
                self.rec_pos,
                self.rec_rad,
                self.cell_start,
                np.float64(self.cell_origin[0]),
                np.float64(self.cell_origin[1]),
                np.float64(self.cell_origin[2]),
                np.float64(self.cell_h),
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.float64(self.r_rec_max),
                np.float64(eps),
                np.int32(B),
                np.int32(self.n_atoms),
                forces,
            ),
        )

    def evaluate_external_forces(
        self,
        world_pos,
        soft_repulsion: bool = False,
        soft_repulsion_eps: float = 1.0,
        self_born: bool = False,
        gb_params: Optional[dict] = None,
        want_energy: bool = False,
    ):
        forces, e_elec = self.evaluate_elec(world_pos, want_energy=want_energy)
        if self.born_grids_cpu and self.alpha != 0.0:
            f_born, e_born = self.evaluate_born(world_pos, want_energy=want_energy)
            forces = forces + f_born
        else:
            e_born = (
                self.xp.zeros(world_pos.shape[0], dtype=self.xp.float64)
                if want_energy
                else None
            )
        if soft_repulsion and self.has_spheres:
            self.add_steric_forces(world_pos, forces, eps=soft_repulsion_eps)
        if self_born:
            gp = gb_params or {}
            f_gb, _ = chain_full_gb_force_batch(
                world_pos,
                self.chain_charges,
                self.chain_radii,
                eps_in=gp.get("eps_in", 1.0),
                eps_out=gp.get("eps_out", WATER_DIELECTRIC),
                coffdrop_active=gp.get("coffdrop_active", False),
                exclude_pair_mask=None,
                obc_kwargs=gp.get("obc_kwargs"),
                xp=self.xp,
            )
            forces = forces + f_gb
        return forces, e_elec, e_born

    def check_overlap(self, world_pos):
        """Return (B,) boolean array True where any bead overlaps a target sphere."""
        xp = self.xp
        B = world_pos.shape[0]
        if not self.has_spheres or B == 0:
            return xp.zeros(B, dtype=bool)
        if xp is np:
            from pystarc.simulation.chain_simulator import _check_chain_overlap

            return np.array(
                [
                    _check_chain_overlap(
                        self._spheres_cpu, world_pos[b], self.chain_radii_np
                    )
                    for b in range(B)
                ],
                dtype=bool,
            )
        out = xp.zeros(B, dtype=xp.int32)
        total = B * self.n_atoms
        nx, ny, nz = self.cell_dims
        self._overlap_kernel(
            ((total + 255) // 256,),
            (256,),
            (
                xp.ascontiguousarray(world_pos, dtype=xp.float64),
                self.chain_radii,
                self.rec_pos,
                self.rec_rad,
                self.cell_start,
                np.float64(self.cell_origin[0]),
                np.float64(self.cell_origin[1]),
                np.float64(self.cell_origin[2]),
                np.float64(self.cell_h),
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                np.float64(self.r_rec_max),
                np.int32(B),
                np.int32(self.n_atoms),
                out,
            ),
        )
        return out != 0
