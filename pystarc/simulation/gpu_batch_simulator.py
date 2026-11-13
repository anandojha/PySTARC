"""
PySTARC batch GPU Brownian-dynamics simulator.

This module runs all trajectories simultaneously on the GPU. The key
difference from NAMSimulator is in how the work is distributed. NAMSimulator
uses a Python loop that advances one trajectory at a time and calls the GPU
once per step. GPUBatchSimulator instead holds all N_traj trajectories as GPU
arrays and runs the entire Brownian-dynamics loop on the GPU, with Python only
checking termination.

The state is laid out as a set of GPU arrays. The positions array has shape
(N_traj, 3) and holds the ligand centroid position for each trajectory, and
the orientations array has shape (N_traj, 4) and holds a quaternion for each
trajectory. All random numbers are generated on the GPU, and the force
evaluation, the Brownian-dynamics step, and the reaction check are all GPU
operations. The Python loop only checks the escape and reaction masks and
recycles finished trajectories.

This design saturates the GPU, launching N_traj × N_atoms threads per kernel.
"""

from __future__ import annotations
from pystarc.simulation.diffusional_rotation import diffusional_rotation
from pystarc.simulation.diffusional_rotation import quat_multiply, quat_rotate_vector
from pystarc.simulation.gpu_random import GPURandom
from pystarc.simulation.nam_simulator import NAMParameters, SimulationResult
from pystarc.simulation.overlap_grid import OverlapGrid
from pystarc.forces.soft_repulsion import A_FACTOR as SOFT_A_FACTOR, SoftRepulsion
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from pystarc.lib.numerical import romberg_integrate
from pystarc.structures.molecules import needed_pairs
from pystarc.global_defs.constants import VACUUM_PERMITTIVITY_KBT
from typing import List, Optional, Tuple, Dict
from scipy.special import erf as scipy_erf
from dataclasses import dataclass
import scipy.stats as stats
from pathlib import Path
import numpy as np
import functools
import math
import warnings
import copy
import time
from pystarc.global_defs.defaults import (
    SAVE_INTERVAL,
    HYDRODYNAMIC_INTERACTIONS,
    SOLVENT_DIELECTRIC,
)

try:
    import cupy as cp

    _CUPY = True
except ImportError:
    _CUPY = False


def _radial_force_derivative(V0: float, r, debye: float, expf):
    """Radial force F0 = -dV/dr of the screened Coulomb potential
    V = V0 exp(-r/λ)/r between the net charges, and its derivative dF0/dr.

    expf is exp(-r/λ), passed in so that the caller chooses NumPy or CuPy.
    With V written including its 1/r, as in BrownDye 2 (outer_propagator.cc),
        F0 = V (1/r + 1/λ),
        dF0/dr = -V/λ² - 2 F0/r = -V0 exp(-r/λ) [2/r³ + 2/(r²λ) + 1/(rλ²)].
    """
    V = V0 * expf / r
    Fr0 = V * (1.0 / r + 1.0 / debye)
    Fr1 = -V / debye**2 - 2.0 * Fr0 / r
    return Fr0, Fr1


def _split_rotational_diffusion(mob, D_r: float) -> Tuple[float, float]:
    """Return (D_rot1, D_rot2), the receptor's and the ligand's rotational diffusion.

    A mobility that does not carry the per-body split cannot propagate the
    receptor's rotation of the separation vector, so it falls back to spinning
    the ligand with the full relative D_r, and says so.
    """
    if hasattr(mob, "D_rot1") and hasattr(mob, "D_rot2"):
        return float(mob.D_rot1), float(mob.D_rot2)
    warnings.warn(
        f"{type(mob).__name__} has no D_rot1/D_rot2; the receptor's rotation of "
        "the separation vector is not propagated and the ligand spins with the "
        "full relative rotational diffusion.",
        RuntimeWarning,
        stacklevel=3,
    )
    return 0.0, float(D_r)


def _return_rotation_host(
    q: np.ndarray,
    pos: np.ndarray,
    tau_lig: float,
    tau_rec: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate returning trajectories for the time they spent in the outer region.

    The two bodies tumble independently over the outer excursion. The ligand's
    rotation dq1 (tau_lig = t D_rot2) turns only its orientation. The
    receptor's rotation dq0 (tau_rec = t D_rot1) turns the whole relative
    configuration by dq0⁻¹ about the origin, the re-entry point on the
    b-sphere included, exactly as the CPU outer propagator does. The new
    orientation is dq0⁻¹ dq1 q. q is (N, 4) in (w, x, y, z) order and pos is
    (N, 3); the inputs are not modified. The loop takes the same rotation on
    the GPU (GPURandom.return_rotation); this is its host reference.
    """
    q_new = np.array(q, dtype=float, copy=True)
    pos_new = np.array(pos, dtype=float, copy=True)
    for i in range(len(q_new)):
        if tau_lig > 0.0:
            q_new[i] = quat_multiply(diffusional_rotation(rng, tau_lig), q_new[i])
        if tau_rec > 0.0:
            dq0 = diffusional_rotation(rng, tau_rec)
            dq0_inv = np.array([dq0[0], -dq0[1], -dq0[2], -dq0[3]])
            q_new[i] = quat_multiply(dq0_inv, q_new[i])
            pos_new[i] = quat_rotate_vector(dq0_inv, pos_new[i])
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True)
    return q_new, pos_new


# CUDA acos and atan2 differ from NumPy's in the last bits, so a pose within
# _ANGLE_EDGE of an angular bin edge is binned with NumPy on the host.
_ANGLE_EDGE = 1e-9

# Per-step records, one thread per running trajectory, with every comparison,
# bin and sum made as NumPy makes it. cnt[0] counts the trajectories still
# running and cnt[1] the poses left in defer. The time maps add dt, in units
# of 1/ticks ps, at the pose at the start of the step (r0, c0).
_TICKS = 2.0**20
_BOOK_KERNEL_CODE = r"""
__device__ int bin_of(const double* e, const int ne, const double v)
{
    int lo = 0, hi = ne;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (e[mid] < v || (v != v && e[mid] == e[mid])) lo = mid + 1;
        else hi = mid;
    }
    lo -= 1;
    return lo < 0 ? 0 : (lo > ne - 2 ? ne - 2 : lo);
}

extern "C" __global__
void book_kernel(
    const long long* idx, const double* r, const double* dt, const int n,
    const double* pos, const double* q, const int* status, const long long step,
    const long long* snap, const int n_snap,
    const double* gho, const double* rec, const double* cut, const int n_pairs,
    const double* rad_e, const int n_rad, const double* trans_e, const int n_trans,
    const double* ms_r, const int n_ms,
    const int n_theta, const int n_phi, const double edge, long long* n_steps,
    double* min_dist, long long* step_at_min, double* nm_pos, double* nm_q,
    double* total_time, double* prev_r,
    unsigned long long* rad, unsigned long long* ang,
    unsigned long long* ms_out, unsigned long long* ms_in,
    unsigned long long* trans, unsigned long long* contact,
    double* defer, unsigned long long* cnt,
    const double* r0, const double* c0, const double ticks,
    unsigned long long* rad_t, unsigned long long* ang_t, unsigned long long* con_t)
{
    const double pi = 3.141592653589793;
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    bool live = false;
    if (j < n) {
        const long long t = idx[j];
        const double rj = r[j];
        const double x = pos[3 * t], y = pos[3 * t + 1], z = pos[3 * t + 2];
        live = status[t] == 0;
        n_steps[t] += 1;
        if (rj < min_dist[t]) {
            min_dist[t] = rj;
            step_at_min[t] = step;
            nm_pos[3 * t] = x;
            nm_pos[3 * t + 1] = y;
            nm_pos[3 * t + 2] = z;
            for (int k = 0; k < 4; ++k) nm_q[4 * t + k] = q[4 * t + k];
        }
        total_time[t] = __dadd_rn(total_time[t], dt[j]);
        atomicAdd(rad + bin_of(rad_e, n_rad, rj), 1ULL);
        const unsigned long long w = __double2ull_rn(dt[j] * ticks);
        atomicAdd(rad_t + bin_of(rad_e, n_rad, r0[j]), w);
        {
            const double s = r0[j] < 1e-30 ? 1e-30 : r0[j];
            double c = c0[3 * j + 2] / s;
            c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
            const double a = acos(c) / pi * n_theta;
            const double b = (atan2(c0[3 * j + 1], c0[3 * j]) + pi) / (2.0 * pi) * n_phi;
            const int ti = a >= n_theta - 1 ? n_theta - 1 : (a > 0.0 ? (int)a : 0);
            const int pj = b >= n_phi - 1 ? n_phi - 1 : (b > 0.0 ? (int)b : 0);
            atomicAdd(ang_t + ti * n_phi + pj, w);
        }
        const double rs = rj < 1e-30 ? 1e-30 : rj;
        double c = __ddiv_rn(z, rs);
        c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
        const double u = __dmul_rn(__ddiv_rn(acos(c), pi), (double)n_theta);
        const double v = __dmul_rn(
            __ddiv_rn(__dadd_rn(atan2(y, x), pi), 2.0 * pi), (double)n_phi);
        if (fabs(u - rint(u)) > edge && fabs(v - rint(v)) > edge) {
            atomicAdd(ang + (int)u * n_phi + (int)v, 1ULL);
        } else {
            const unsigned long long k = atomicAdd(cnt + 1, 1ULL);
            defer[4 * k] = x;
            defer[4 * k + 1] = y;
            defer[4 * k + 2] = z;
            defer[4 * k + 3] = rj;
        }
        bool keep = true;
        for (int k = 0; k < n_snap; ++k) keep = keep && snap[k] != t;
        if (keep) {
            const double p = prev_r[t];
            for (int k = 0; k < n_ms; ++k) {
                if (p < ms_r[k] && rj >= ms_r[k]) atomicAdd(ms_out + k, 1ULL);
                if (p >= ms_r[k] && rj < ms_r[k]) atomicAdd(ms_in + k, 1ULL);
            }
            const int b0 = bin_of(trans_e, n_trans, p);
            const int b1 = bin_of(trans_e, n_trans, rj);
            atomicAdd(trans + b0 * (n_trans - 1) + b1, 1ULL);
        }
        prev_r[t] = rj;
        for (int k = 0; k < n_pairs; ++k) {
            const double* g = gho + 3 * ((long long)j * n_pairs + k);
            const double dx = __dsub_rn(g[0], rec[3 * k]);
            const double dy = __dsub_rn(g[1], rec[3 * k + 1]);
            const double dz = __dsub_rn(g[2], rec[3 * k + 2]);
            const double d2 = __dadd_rn(
                __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
            if (__dsqrt_rn(d2) < cut[k]) {
                atomicAdd(contact + k, 1ULL);
                atomicAdd(con_t + k, w);
            }
        }
    }
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if ((threadIdx.x & 31) == 0 && m) atomicAdd(cnt, (unsigned long long)__popc(m));
}
"""


def _angular_bins_host(d, n_theta, n_phi, counts):
    """Add the poses d, rows (x, y, z, r), to the angular map with NumPy."""
    cen = np.ascontiguousarray(d[:, :3])
    r_safe = np.maximum(np.ascontiguousarray(d[:, 3]), 1e-30)
    theta = np.arccos(np.clip(cen[:, 2] / r_safe, -1, 1))
    phi = np.arctan2(cen[:, 1], cen[:, 0]) + np.pi
    ti = np.clip((theta / np.pi * n_theta).astype(int), 0, n_theta - 1)
    pj = np.clip((phi / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    np.add.at(counts, (ti, pj), 1)


# Fused kernels of the BD step, operation by operation as the CuPy chain of
# the reference methods, compiled without FMA contraction. A norm of a single
# vector follows CuPy's tree (x0² + x2²) + x1² (one). The kernels that square
# with pow() keep contraction on and unfuse their products with __dmul_rn.
# place() sums in the order cuBLAS uses for the einsum at most batch sizes.
_NO_FMA = ("--fmad=false",)

_STEP_DEVICE = r"""
// min and max that keep NaN, as cp.minimum and cp.maximum do.
__device__ double nmin(double a, double b) { return a != a || b != b ? a + b : min(a, b); }
__device__ double nmax(double a, double b) { return a != a || b != b ? a + b : max(a, b); }

__device__ double norm3(const double* v, bool one) {
    if (one) return sqrt(v[0] * v[0] + v[2] * v[2] + v[1] * v[1]);
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__device__ void cross3(const double* a, const double* b, double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ void rotmat(const double* q, double* m) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    m[0] = 1.0 - 2.0 * (y * y + z * z);
    m[1] = 2.0 * (x * y - w * z);
    m[2] = 2.0 * (x * z + w * y);
    m[3] = 2.0 * (x * y + w * z);
    m[4] = 1.0 - 2.0 * (x * x + z * z);
    m[5] = 2.0 * (y * z - w * x);
    m[6] = 2.0 * (x * z - w * y);
    m[7] = 2.0 * (y * z + w * x);
    m[8] = 1.0 - 2.0 * (x * x + y * y);
}

// Ligand pair atom m g + c, the product summed by FMA in index order as
// cuBLAS does for the einsum at most batch sizes.
__device__ double place(const double* m, const double* g, int a, double c) {
    return fma(m[3 * a + 2], g[2], fma(m[3 * a + 1], g[1], m[3 * a] * g[0])) + c;
}

// q <- dq q, dq the rotation by the vector w (_apply_rotation_gpu).
__device__ void turn_quat(double* q, const double* w, bool one) {
    double a = norm3(w, one);
    double h = a * 0.5, c = cos(h), s = sin(h);
    double u[3];
    for (int k = 0; k < 3; k++) u[k] = (a > 1e-10 ? w[k] / a : 0.0) * s;
    double n0 = c * q[0] - u[0] * q[1] - u[1] * q[2] - u[2] * q[3];
    double n1 = c * q[1] + u[0] * q[0] + u[1] * q[3] - u[2] * q[2];
    double n2 = c * q[2] - u[0] * q[3] + u[1] * q[0] + u[2] * q[1];
    double n3 = c * q[3] + u[0] * q[2] - u[1] * q[1] + u[2] * q[0];
    double n = one ? sqrt(n0 * n0 + n2 * n2 + (n1 * n1 + n3 * n3))
                   : sqrt(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3);
    q[0] = n0 / n;
    q[1] = n1 / n;
    q[2] = n2 / n;
    q[3] = n3 / n;
}

// v <- the rotation by the vector w applied to v (_rotate_vectors_gpu).
__device__ void turn_vector(double* v, const double* w, bool one) {
    double a = norm3(w, one);
    double k[3], x[3];
    for (int j = 0; j < 3; j++) k[j] = a > 1e-10 ? w[j] / a : 0.0;
    double c = cos(a), s = sin(a);
    double kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    cross3(k, v, x);
    for (int j = 0; j < 3; j++) x[j] = v[j] * c + x[j] * s + k[j] * kv * (1.0 - c);
    for (int j = 0; j < 3; j++) v[j] = x[j];
}
"""

_POSE_CODE = r"""
double c[3], qq[4], m[9];
for (int k = 0; k < 3; k++) c[k] = pos[3 * j + k];
for (int k = 0; k < 4; k++) qq[k] = q[4 * j + k];
rotmat(qq, m);
for (int k = 0; k < 3; k++) cen[3 * i + k] = c[k];
for (int k = 0; k < 4; k++) quat[4 * i + k] = qq[k];
for (int k = 0; k < 9; k++) rot[9 * i + k] = m[k];
r = norm3(c, _ind.size() == 1);
"""

_PAIRS_CODE = r"""
double m[9], g[3], c[3], d[3];
for (int k = 0; k < 9; k++) m[k] = rot[9 * i + k];
for (int k = 0; k < 3; k++) c[k] = cen[3 * i + k];
long n = 0;
for (long p = 0; p < npair; p++) {
    for (int a = 0; a < 3; a++) g[a] = lig[3 * p + a];
    for (int a = 0; a < 3; a++) {
        double x = place(m, g, a, c[a]);
        gho[3 * (npair * i + p) + a] = x;
        d[a] = x - rec[3 * p + a];
    }
    double dp = norm3(d, _ind.size() * npair == 1);
    dist[npair * i + p] = dp;
    n += dp < cut[p];
}
nsat = n;
hit = n >= need;
"""

_ESCAPE_CODE = r"""
esc = r >= r_esc && !hit;
if (hit) atomicAdd(&cnt[0], 1);
if (esc) atomicAdd(&cnt[1], 1);
"""

# Pair, force and edge rules of the time step.
_DT_CODE = r"""
outer = r > r_outer;
double dt_pair = c_pair * r * r / (D_t + __dmul_rn(D_r1 * r, r));
double dt_force = dt_pair;
if (force) {
    double V = V0 * exp(-r / debye) / r;
    double F0 = V * (1.0 / r + inv_debye);
    double DF1 = fabs(D_t * (-V / debye2 - 2.0 * F0 / r));
    if (DF1 > 1e-15 && r < debye3) dt_force = alpha / DF1;
}
double e = nmin(nmax(r - r_b, 1e-3), nmax(r_esc - r, 1e-3));
double dt_edge = pow(e, 2.0) / D18;
dt = outer ? nmin(nmin(dt_force, dt_edge), dt_pair) : nmin(dt_force, dt_pair);
"""

# Surface rule (surface_time_step with its limits 0.5 A, 0.1 A and 0.05),
# then the cap and the floor of the run.
_SURFACE_CODE = r"""
double t = dt0;
if (surface) {
    double f[3], tq[3];
    for (int k = 0; k < 3; k++) f[k] = frc[3 * i + k];
    for (int k = 0; k < 3; k++) tq[k] = trq[3 * i + k];
    double lev = nmin(r + rho, extent);
    double D = D_lig + D_r1 * lev * lev;
    bool one = _ind.size() == 1;
    double fm = (D_t * norm3(f, one) + D_spin * norm3(tq, one)) / D;
    double w = nmax(room[i], 0.1);
    t = nmin(t, nmin(w * w / (12.0 * D), 0.5 / (D * nmax(fm, 1e-12))));
    if (soft) t = nmin(t, 0.05 / (D * nmax(stiff[i], 1e-12)));
}
if (max_dt > 0) t = nmin(t, max_dt);
if (min_dt > 0) t = nmax(t, min_dt);
dt = t;
"""

# RPY D_par(r) and the divergence drift dD/dr dt r_hat, bounded by a tenth
# of the random step.
_HI_CODE = r"""
double rm1 = 1.0 / nmax(r, 0.01);
double rm2 = rm1 * rm1, rm3 = rm2 * rm1;
double D_far = D0 - __dmul_rn(dpre, __dmul_rn(3.0, rm1) - __dmul_rn(a2x2, rm3));
double D_ov = D0 - __dmul_rn(c_ov, c8S - __dmul_rn(c6am2, rm1) - __dmul_rn(3.0, r)
                                   + __dmul_rn(am2sq, rm3));
double Dp = nmax(r > a_sum ? D_far : (r > a_dif ? D_ov : Deng), 1e-6);
double dD_far = dpre * (__dmul_rn(3.0, rm2) - __dmul_rn(a2x6 * rm1, rm3));
double dD_ov = c3ov * pow(1.0 - __dmul_rn(am2, rm2), 2.0);
double dD = r > a_sum ? dD_far : (r > a_dif ? dD_ov : 0.0);
double disp = dD * dt;
double cap = 0.1 * sqrt(2.0 * Dp * dt);
disp = disp < -cap ? -cap : (disp > cap ? cap : disp);
for (int k = 0; k < 3; k++) dvg[3 * i + k] = disp * (cen[3 * i + k] * rm1);
D = Dp;
"""

# Translation, rotation of the ligand about its centre and, when D_r1 > 0,
# the receptor's tumbling as a turn of the relative configuration by -omega1
# about the origin.
_MOVE_CODE = r"""
double p[3], qq[4], f[3], tq[3], w[3];
for (int k = 0; k < 3; k++) p[k] = pos[3 * j + k];
for (int k = 0; k < 4; k++) qq[k] = q[4 * j + k];
for (int k = 0; k < 3; k++) f[k] = frc[3 * i + k];
for (int k = 0; k < 3; k++) tq[k] = trq[3 * i + k];
sig = sqrt(2.0 * D * dt);
for (int k = 0; k < 3; k++) {
    double d = D * dt * f[k];
    if (hi) d = d + dvg[3 * i + k];
    drift[3 * i + k] = d;
    p[k] = p[k] + (d + sig * nt[3 * i + k]);
}
bool one = _ind.size() == 1;
double s2 = sqrt(Dr2x2 * dt);
for (int k = 0; k < 3; k++) w[k] = Dr2 * dt * tq[k] + s2 * nr[3 * i + k];
turn_quat(qq, w, one);
if (tumble) {
    double c[3], x[3];
    for (int k = 0; k < 3; k++) c[k] = cen[3 * i + k];
    cross3(c, f, x);
    double s1 = sqrt(Dr1x2 * dt);
    for (int k = 0; k < 3; k++) w[k] = -(Dr1 * dt * -(tq[k] + x[k]) + s1 * nr1[3 * i + k]);
    turn_vector(p, w, one);
    turn_quat(qq, w, one);
}
for (int k = 0; k < 3; k++) pos[3 * j + k] = p[k];
for (int k = 0; k < 4; k++) q[4 * j + k] = qq[k];
"""

# Pair distances after the step and the Brownian-bridge crossing probability
# exp(-x0 x1 / (D dt)), with the rotation part of D from the pose at the
# start of the step (_bridge_D_rot).
_BRIDGE_CODE = r"""
double qq[4], m[9], c[3], g[3], b[3], d[3], s[3], l[3], bx[3], lx[3];
for (int k = 0; k < 4; k++) qq[k] = q[4 * j + k];
for (int k = 0; k < 3; k++) c[k] = pos[3 * j + k];
rotmat(qq, m);
for (long p = 0; p < npair; p++) {
    long ip = npair * i + p;
    for (int a = 0; a < 3; a++) g[a] = lig[3 * p + a];
    for (int a = 0; a < 3; a++) b[a] = rec[3 * p + a];
    for (int a = 0; a < 3; a++) d[a] = place(m, g, a, c[a]) - b[a];
    double dp = norm3(d, _ind.size() * npair == 1);
    double x0 = d0[ip] - cut[p], x1 = dp - cut[p];
    double h = nmax(d0[ip], 1e-12);
    for (int a = 0; a < 3; a++) s[a] = (gho0[3 * ip + a] - b[a]) / h;
    for (int a = 0; a < 3; a++) l[a] = gho0[3 * ip + a] - cen[3 * i + a];
    cross3(b, s, bx);
    cross3(l, s, lx);
    double Drot = Dr1 * (bx[0] * bx[0] + bx[1] * bx[1] + bx[2] * bx[2])
                + Dr2 * (lx[0] * lx[0] + lx[1] * lx[1] + lx[2] * lx[2]);
    double Ddt = nmax((D + Drot) * dt, 1e-30);
    d1[ip] = dp;
    above[ip] = x0 > 0 && x1 > 0;
    pc[ip] = above[ip] ? exp(-x0 * x1 / Ddt) : 0.0;
}
"""

_FIRE_CODE = r"""
bool keep = !(reverted && rev[i]);
long n = 0, m = 0;
for (long p = 0; p < npair; p++) {
    long ip = npair * i + p;
    bool ins = d1[ip] < cut[p];
    bool f = (u[ip] < pc[ip] || ins) && keep;
    fired[ip] = f;
    n += f;
    m += ins && keep;
}
total = n;
inside = m;
hit = n >= need;
"""


@functools.lru_cache(maxsize=None)
def _step_kernels():
    """The step kernels, compiled on first use."""
    k = cp.ElementwiseKernel
    d = dict(preamble=_STEP_DEVICE, options=_NO_FMA)
    return {
        "pose": k(
            "int64 j, raw float64 pos, raw float64 q",
            "raw float64 cen, raw float64 quat, raw float64 rot, float64 r",
            _POSE_CODE, "pystarc_pose", **d,
        ),
        "pairs": k(
            "raw float64 rot, raw float64 cen, raw float64 lig, raw float64 rec, "
            "raw float64 cut, int64 npair, int64 need",
            "raw float64 gho, raw float64 dist, int64 nsat, bool hit",
            _PAIRS_CODE, "pystarc_pairs", **d,
        ),
        "escape": k(
            "float64 r, bool hit, float64 r_esc", "bool esc, raw int32 cnt",
            _ESCAPE_CODE, "pystarc_escape",
        ),
        "dt": k(
            "float64 r, float64 r_outer, float64 c_pair, float64 D_t, float64 D_r1, "
            "bool force, float64 V0, float64 debye, float64 inv_debye, "
            "float64 debye2, float64 debye3, float64 alpha, float64 r_b, "
            "float64 r_esc, float64 D18",
            "float64 dt, bool outer",
            _DT_CODE, "pystarc_dt", preamble=_STEP_DEVICE,
        ),
        "surface": k(
            "float64 r, float64 dt0, raw float64 frc, raw float64 trq, "
            "raw float64 room, raw float64 stiff, bool surface, bool soft, "
            "float64 rho, float64 extent, float64 D_t, float64 D_r1, "
            "float64 D_lig, float64 D_spin, float64 max_dt, float64 min_dt",
            "float64 dt",
            _SURFACE_CODE, "pystarc_surface", **d,
        ),
        "hi": k(
            "float64 r, float64 dt, raw float64 cen, float64 D0, float64 dpre, "
            "float64 Deng, float64 a_sum, float64 a_dif, float64 a2x2, float64 a2x6, "
            "float64 c_ov, float64 c8S, float64 c6am2, float64 am2sq, "
            "float64 c3ov, float64 am2",
            "float64 D, raw float64 dvg",
            _HI_CODE, "pystarc_hi", preamble=_STEP_DEVICE,
        ),
        "move": k(
            "int64 j, float64 dt, float64 D, raw float64 cen, raw float64 frc, "
            "raw float64 trq, raw float64 dvg, raw float64 nt, raw float64 nr, "
            "raw float64 nr1, bool hi, bool tumble, float64 Dr1, float64 Dr2, "
            "float64 Dr1x2, float64 Dr2x2",
            "raw float64 pos, raw float64 q, raw float64 drift, float64 sig",
            _MOVE_CODE, "pystarc_move", **d,
        ),
        "bridge": k(
            "int64 j, float64 D, float64 dt, raw float64 pos, raw float64 q, "
            "raw float64 lig, raw float64 rec, raw float64 cut, raw float64 gho0, "
            "raw float64 d0, raw float64 cen, int64 npair, float64 Dr1, float64 Dr2",
            "raw float64 d1, raw float64 pc, raw bool above",
            _BRIDGE_CODE, "pystarc_bridge", **d,
        ),
        "fire": k(
            "raw float64 u, raw float64 pc, raw float64 d1, raw float64 cut, "
            "raw bool rev, bool reverted, int64 npair, int64 need",
            "raw bool fired, int64 total, int64 inside, bool hit",
            _FIRE_CODE, "pystarc_fire",
        ),
    }


def _pose_gpu(pos, q, idx):
    """Centroids, quaternions, rotation matrices and radii of the poses idx."""
    n = len(idx)
    cen = cp.empty((n, 3))
    quat = cp.empty((n, 4))
    rot = cp.empty((n, 3, 3))
    r = cp.empty(n)
    _step_kernels()["pose"](idx, pos, q, cen, quat, rot, r)
    return cen, quat, rot, r


def _pairs_gpu(rot, cen, lig, rec, cut, need):
    """Ligand pair atoms, pair distances, the number of pairs inside their
    cutoffs and whether that number reaches need, per pose."""
    n, npair = len(cen), len(cut)
    gho = cp.empty((n, npair, 3))
    dist = cp.empty((n, npair))
    nsat = cp.empty(n, dtype=cp.int64)
    hit = cp.empty(n, dtype=cp.bool_)
    _step_kernels()["pairs"](rot, cen, lig, rec, cut, npair, need, gho, dist, nsat, hit)
    return gho, dist, nsat, hit


@dataclass
class GPUBatchResult:
    """Results from batch GPU BD simulation."""

    n_trajectories: int
    n_reacted: int
    n_escaped: int
    n_max_steps: int
    reaction_counts: Dict[str, int]
    r_start: float
    r_escape: float
    dt: float
    elapsed_sec: float
    steps_per_sec: float
    sim_data: Dict = None  # Collected trajectory data passed to the output writer.
    total_steps: int = 0  # BD steps over all trajectories, resumed runs included.

    # Every trajectory runs until it reacts, escapes or reaches max_steps. A
    # trajectory at the step limit (n_max_steps) counts as not reacted, as in
    # BrownDye2, so the denominator of P_rxn is every trajectory that was run.
    @property
    def n_completed(self) -> int:
        """Trajectories that reacted or escaped."""
        return self.n_reacted + self.n_escaped

    @property
    def n_run(self) -> int:
        """Every trajectory that was run: reacted, escaped or at the step limit."""
        return self.n_reacted + self.n_escaped + self.n_max_steps

    @property
    def reaction_probability(self) -> float:
        n = self.n_run
        return self.n_reacted / n if n > 0 else 0.0

    def reaction_probability_bounds(self):
        """P_rxn with every trajectory at the step limit counted as not reacted
        (the reported value), and with every one counted as reacted."""
        n = self.n_run
        if n == 0:
            return (0.0, 1.0)
        return (self.n_reacted / n, (self.n_reacted + self.n_max_steps) / n)

    def reaction_probability_ci(self, confidence: float = 0.95):
        n = self.n_run
        if n == 0:
            warnings.warn(
                "reaction_probability_ci: no trajectories were run; the "
                "reaction probability is undefined and the returned CI [0, 1] "
                "is uninformative.",
                RuntimeWarning,
                stacklevel=2,
            )
            return (0.0, 1.0)
        # z = 1.96 at 0.95 as in convergence.json and the pooled results.
        z = 1.96 if abs(confidence - 0.95) < 1e-9 else float(stats.norm.ppf(0.5 + confidence / 2.0))
        p = self.reaction_probability
        z2n = z**2 / n
        denom = 1.0 + z2n
        centre = (p + z2n / 2.0) / denom
        margin = z * math.sqrt(max(p * (1 - p) / n + z2n / (4.0 * n), 0)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    @staticmethod
    def _rate_from_p(P: float, k_b: float) -> float:
        """k_on = CONV k_b P (SI Step 1), CONV = N_A 1e-30/1e-12/1e-3 taking
        Å³/ps to M⁻¹ s⁻¹."""
        if not (math.isfinite(k_b) and k_b > 0):
            raise ValueError(f"k_b must be finite and positive, got {k_b}")
        return 6.022e23 * 1e-30 / 1e-12 / 1e-3 * k_b * P

    def rate_constant(self, D_rel: float, k_b: float) -> float:
        return self._rate_from_p(self.reaction_probability, k_b)

    def rate_constant_ci(self, D_rel: float, k_b: float, confidence: float = 0.95):
        p_lo, p_hi = self.reaction_probability_ci(confidence)
        return (self._rate_from_p(p_lo, k_b), self._rate_from_p(p_hi, k_b))

    def rate_constant_bounds(self, D_rel: float, k_b: float):
        """k_on with every trajectory at the step limit counted as not reacted
        (the reported value), and with every one counted as reacted."""
        p_lo, p_hi = self.reaction_probability_bounds()
        return (self._rate_from_p(p_lo, k_b), self._rate_from_p(p_hi, k_b))

    def summary(self, D_rel: float, k_b: float, confidence: float = 0.95) -> str:
        k = self.rate_constant(D_rel, k_b)
        k_lo, k_hi = self.rate_constant_ci(D_rel, k_b, confidence)
        p = self.reaction_probability
        p_lo, p_hi = self.reaction_probability_ci(confidence)
        pct = f"{100 * confidence:g}"
        lines = [
            f"  Trajectories : {self.n_trajectories:,}",
            f"  Completed    : {self.n_completed:,}  "
            f"({self.n_reacted:,} reacted + {self.n_escaped:,} escaped)",
        ]
        if self.n_max_steps:
            lines.append(
                f"  Max-steps    : {self.n_max_steps:,}  (counted as not reacted)"
            )
        lines += [
            f"  P_rxn        : {p:.6f}  ({pct}% CI: [{p_lo:.6f}, {p_hi:.6f}])",
            f"  k_on         : {k:.4e} M-1 s-1",
            f"  {pct}% CI     : [{k_lo:.4e}, {k_hi:.4e}] M-1 s-1",
        ]
        if self.n_max_steps:
            b_lo, b_hi = self.rate_constant_bounds(D_rel, k_b)
            lines.append(
                f"  k_on bound   : [{b_lo:.4e}, {b_hi:.4e}] M-1 s-1  "
                "(max-steps trajectories as not reacted, as reacted)"
            )
        if k > 0:
            lines.append(f"  log10(k_on)  : {math.log10(k):.3f}")
        lines += [
            f"  Wall time    : {self.elapsed_sec:.1f} s",
            f"  BD steps/sec : {self.steps_per_sec:,.0f}",
            f"  Backend      : CUPY (batch GPU)",
        ]
        return "\n".join(lines)


class GPUBatchSimulator:
    """
    Batch GPU Brownian-dynamics simulator in which all N_traj trajectories run
    simultaneously on the GPU. Unlike the sequential NAMSimulator, this keeps
    GPU utilization high (above 80%).

    Parameters
    ----------
    mol1        : receptor Molecule, held fixed.
    mol2        : ligand Molecule, which diffuses.
    mob         : MobilityTensor.
    pathway_set : the reaction criteria.
    params      : NAMParameters.
    batch_engine: the GPUBatchForceEngine.
    """

    def __init__(
        self, mol1, mol2, mob, pathway_set, params: NAMParameters, batch_engine
    ):
        if not _CUPY:
            raise RuntimeError("CuPy required for GPU batch simulation.")
        self.mol1 = mol1
        self.mol2 = mol2
        self.mob = mob
        self.pathway_set = pathway_set
        self.params = params
        self.engine = batch_engine
        # Extract the ligand geometry, storing atom positions relative to the
        # centroid.
        c0 = mol2.centroid()
        self._mol2_pos0 = cp.asarray(
            mol2.positions_array() - c0, dtype=cp.float64
        )  # (N_lig, 3)
        self._charges = cp.asarray(mol2.charges_array(), dtype=cp.float64)  # (N_lig,)
        self._N_lig = len(mol2.atoms)
        # Van der Waals radii are placed on the GPU for the atom-pair overlap
        # check. They are loaded unconditionally so that overlap_check=true
        # works without lj_forces=true. Ghost atoms (radius 0) take no part.
        self._mol1_radii_overlap = cp.asarray(
            mol1.radii_array(), dtype=cp.float64
        )  # (N_rec,)
        self._mol2_radii_overlap = cp.asarray(
            mol2.radii_array(), dtype=cp.float64
        )  # (N_lig,)
        # Outer propagator (LMZ) parameters, computed once before run().
        self._return_prob = 0.0  # Set in run(): P(return to b | at trigger_r).
        self._k_b = 0.0  # Set in run(): the Romberg k_b, the encounter rate from the Romberg integral.
        self._qb_factor = 1.1  # Trigger at 1.1 × b_sphere (the standard qb_factor).
        self._use_hi = getattr(
            params, "hydrodynamic_interactions", HYDRODYNAMIC_INTERACTIONS
        )
        # Ghost atom indices used by the reaction check.
        self._rec_gho_indices = []
        self._lig_gho_indices = []
        self._rxn_cutoffs = []
        for rxn in pathway_set.reactions:
            for pair in rxn.criteria.pairs:
                self._rec_gho_indices.append(pair.mol1_atom_index)
                self._lig_gho_indices.append(pair.mol2_atom_index)
                self._rxn_cutoffs.append(pair.distance_cutoff)
        self._rxn_cutoffs = np.array(self._rxn_cutoffs)
        self._rxn_cutoff_min = (
            float(self._rxn_cutoffs.min()) if len(self._rxn_cutoffs) else 5.0
        )
        # n_needed is the minimum number of pairs that must fire
        # simultaneously. A value of -1 means all pairs, and it is taken from
        # the first reaction's ReactionCriteria.
        if pathway_set.reactions:
            c0 = pathway_set.reactions[0].criteria
            self._rxn_n_needed = needed_pairs(c0.n_needed, len(self._rxn_cutoffs), c0.name)
        else:
            self._rxn_n_needed = 1
        # Receptor ghost atom positions, which are fixed because the receptor
        # never moves.
        if self._rec_gho_indices:
            rec_pos = mol1.positions_array()
            self._rec_gho_pos = cp.asarray(
                rec_pos[self._rec_gho_indices], dtype=cp.float64
            )
            self._rxn_cutoffs_gpu = cp.asarray(self._rxn_cutoffs, dtype=cp.float64)
        else:
            self._rec_gho_pos = None
            self._rxn_cutoffs_gpu = None
        # State-machine reaction metadata. This is populated only when at least
        # one reaction in pathway_set carries a non-None state_before label.
        # Otherwise all fields are left empty or False and the simulator uses
        # the flattened-pairs path. Each reaction has a source state
        # (state_before) and a destination state (state_after). A reaction
        # fires on a trajectory only when that trajectory is currently in the
        # source state, and upon firing the trajectory moves to the destination
        # state. Each reaction is assumed to have exactly one pair, so the
        # per-reaction cutoff and GHO indices below pick pair[0] from each
        # reaction's criterion.
        self._sm_active = False
        self._sm_rxn_names = []
        self._sm_rxn_state_before = []
        self._sm_rxn_state_after = []
        self._sm_rxn_cutoffs = []
        self._sm_rxn_gho_rec_idx = []
        self._sm_rxn_gho_lig_idx = []
        _sm_any_labels = any(
            getattr(rxn, "state_before", None) is not None
            for rxn in pathway_set.reactions
        )
        if not _sm_any_labels and len(pathway_set.reactions) > 1:
            raise ValueError(
                f"{len(pathway_set.reactions)} reactions without state labels would "
                "share the first one's n_needed. Put all pairs in one reaction, or "
                "label each reaction for the state machine."
            )
        if any(getattr(rxn, "probability", 1.0) < 1.0 for rxn in pathway_set.reactions):
            raise NotImplementedError("Reaction probabilities below 1 are not implemented on the GPU.")
        if _sm_any_labels and pathway_set.reactions:
            # Validate all-or-none labeling. If any reaction labels a source
            # state, every reaction must label state_before. An unlabeled
            # reaction would silently fall to index -1 in
            # _sm_rxn_state_before_idx and never fire on any trajectory, which
            # looks like a missing pathway rather than a configuration error,
            # so fail loudly here.
            _sm_missing_before = [
                rxn.name
                for rxn in pathway_set.reactions
                if getattr(rxn, "state_before", None) is None
            ]
            if _sm_missing_before:
                raise ValueError(
                    "State-machine reactions partially labeled: at least "
                    "one reaction has state_before set, but these reactions "
                    f"do not: {_sm_missing_before[:5]}"
                    f"{' (and more)' if len(_sm_missing_before) > 5 else ''}"
                    ". Either label every reaction with state_before or none."
                )
            for rxn in pathway_set.reactions:
                if not rxn.criteria.pairs:
                    continue
                if len(rxn.criteria.pairs) > 1:
                    raise ValueError(
                        f"State-machine reaction '{rxn.name}' has "
                        f"{len(rxn.criteria.pairs)} contact pairs, but the "
                        "state-machine path supports exactly one pair per "
                        "reaction (the per-reaction cutoff and GHO indices use "
                        "pair[0]). Split it into one reaction per pair, or run "
                        "without state_machine_reactions to use the flattened "
                        "n_needed path."
                    )
                needed_pairs(rxn.criteria.n_needed, 1, rxn.name)
                pair0 = rxn.criteria.pairs[0]
                self._sm_rxn_names.append(rxn.name)
                self._sm_rxn_state_before.append(getattr(rxn, "state_before", None))
                self._sm_rxn_state_after.append(getattr(rxn, "state_after", None))
                self._sm_rxn_cutoffs.append(float(pair0.distance_cutoff))
                self._sm_rxn_gho_rec_idx.append(int(pair0.mol1_atom_index))
                self._sm_rxn_gho_lig_idx.append(int(pair0.mol2_atom_index))
            self._sm_active = len(self._sm_rxn_names) > 0
        # Integer encoding of the state labels. GPU kernels work with integer
        # arrays rather than string labels, so each distinct state gets a
        # stable integer index. The first_state is always index 0 so that every
        # trajectory can be initialized with current_state = 0 at trajectory
        # start. Terminal states are those that never appear as a state_before
        # in any reaction. A trajectory entering a terminal state has no further
        # reactions to fire and is marked reacted.
        self._sm_state_to_idx = {}
        self._sm_idx_to_state = []
        self._sm_rxn_state_before_idx = []
        self._sm_rxn_state_after_idx = []
        self._sm_state_is_terminal = []
        self._sm_first_state = None
        if self._sm_active:
            fs = getattr(pathway_set, "first_state", None)
            if fs is None and self._sm_rxn_state_before:
                fs = self._sm_rxn_state_before[0]
            self._sm_first_state = fs

            def _get_or_add_state(label):
                if label is None:
                    return -1
                if label not in self._sm_state_to_idx:
                    idx = len(self._sm_idx_to_state)
                    self._sm_state_to_idx[label] = idx
                    self._sm_idx_to_state.append(label)
                return self._sm_state_to_idx[label]

            if self._sm_first_state is not None:
                _get_or_add_state(self._sm_first_state)
            for sb, sa in zip(self._sm_rxn_state_before, self._sm_rxn_state_after):
                self._sm_rxn_state_before_idx.append(_get_or_add_state(sb))
                self._sm_rxn_state_after_idx.append(_get_or_add_state(sa))
            sources = set(self._sm_rxn_state_before_idx)
            self._sm_state_is_terminal = [
                (i not in sources) for i in range(len(self._sm_idx_to_state))
            ]
            # GPU-side per-reaction arrays consumed by the state-machine check.
            rec_pos_full = mol1.positions_array()
            self._sm_rec_gho_pos_gpu = cp.asarray(
                rec_pos_full[self._sm_rxn_gho_rec_idx], dtype=cp.float64
            )
            self._sm_rxn_cutoffs_gpu = cp.asarray(
                self._sm_rxn_cutoffs, dtype=cp.float64
            )
            self._sm_rxn_state_before_idx_gpu = cp.asarray(
                self._sm_rxn_state_before_idx, dtype=cp.int32
            )
            self._sm_rxn_state_after_idx_gpu = cp.asarray(
                self._sm_rxn_state_after_idx, dtype=cp.int32
            )
            self._sm_state_is_terminal_gpu = cp.asarray(
                self._sm_state_is_terminal, dtype=cp.bool_
            )
            self._sm_lig_gho_indices_cached = cp.asarray(
                self._sm_rxn_gho_lig_idx, dtype=cp.int64
            )
        else:
            self._sm_rec_gho_pos_gpu = None
            self._sm_rxn_cutoffs_gpu = None
            self._sm_rxn_state_before_idx_gpu = None
            self._sm_rxn_state_after_idx_gpu = None
            self._sm_state_is_terminal_gpu = None
            self._sm_lig_gho_indices_cached = None

    def _compute_return_prob(self) -> float:
        """
        Compute the probability of returning to the b-sphere given that the
        ligand is currently at trigger_r, together with the b-sphere rate k_b.
        The return probability is

            P_return(r to b) = h(trigger_r) / h(b)

        where h(r) is evaluated by the substitution s = 1/r, so the integral
        runs from s = 0, meaning r = infinity, down to s = 1/b, and its
        integrand carries 1/D(r) rather than 1/s². Running to r = infinity
        keeps the rate invariant to the b surface placement, and carrying
        1/D(r) couples the normalisation to the hydrodynamics. Here r is a
        separation and U
        is the orientation-averaged potential of mean force in units of kBT. This method also computes and stores self._k_b
        from the Romberg integral

            k_b = 4π / integral from 0 to 1/b of exp(U(1/s)) / D ds.
        """
        b = self.params.r_start
        # The return probability is now applied at r_escape, so the Romberg
        # integral must use r_escape as the outer boundary. The reference uses
        # qradius = 20 × max_mol_radius and applies the return probability at
        # qradius, but since PySTARC applies it at r_escape we set q_out =
        # r_escape.
        q_out = self.params.r_escape
        trigger = self._qb_factor * b
        if not q_out > trigger:
            raise ValueError(
                f"r_escape {q_out} Å must exceed {self._qb_factor} x b = {trigger:g} Å"
            )
        q_rec = float(self.mol1.total_charge())
        q_lig = float(self.mol2.total_charge())
        debye = self.engine._debye
        # Vacuum permittivity in kBT units. See constants.py.
        eps0 = VACUUM_PERMITTIVITY_KBT
        # The configured solvent permittivity. The screened Coulomb forces use
        # it through the APBS grids, so k_b is normalised with the same value.
        sdie = float(getattr(self.params, "dielectric", SOLVENT_DIELECTRIC))
        if not (debye > 0 and sdie > 0):
            raise ValueError(f"debye_length {debye} and sdie {sdie} must be positive")
        eps = sdie * eps0
        fpe = 4.0 * math.pi * eps
        D_t = float(self.mob.relative_translational_diffusion())
        # The potential V(r) is expressed in kBT_298 units. At a temperature T
        # other than 298.15 K the Boltzmann factor is exp(V/kT_scale), where
        # kT_scale = T/298.15.
        _kT_scale = getattr(self.params, "_kT_scale", 1.0)
        # Keep the multipole treatment consistent. The BD trajectories
        # integrate the full multipole field of the receptor, so the k_b
        # Smoluchowski integral must use the orientation-averaged potential of
        # mean force rather than the monopole-only U(r). The leading angular
        # averages of the dipole and traceless quadrupole vanish, and the
        # corrections enter at the second cumulant:
        #   V_pmf(r) = V_0(r) - <(V - <V>)²>_Omega / (2 kBT)
        #   <V_dip²>_Omega  = q_lig² × |p_rec|² / 3       × g_1(r)²
        #   <V_quad²>_Omega = q_lig² × (2/15) Tr(Q_rec²)  × g_2(r)²
        # Here g_1 and g_2 are the Yukawa radial factors used in
        # _yukawa_forces_gpu. For typical biological systems at the b-sphere the
        # correction is small, but it closes the NAM decomposition loop now that
        # the trajectories see the full multipole (including the transverse
        # component) past the grid.
        _mp_dip_gpu = getattr(self.engine, "_mp_dipole_gpu", None)
        _mp_quad_gpu = getattr(self.engine, "_mp_quad_gpu", None)
        if _mp_dip_gpu is not None and _mp_quad_gpu is not None:
            _p_rec = cp.asnumpy(_mp_dip_gpu).astype(float)
            _Q_rec = cp.asnumpy(_mp_quad_gpu).astype(float)
            _p_sq = float(np.dot(_p_rec, _p_rec))
            # For a symmetric Q the trace of Q² is the elementwise sum of
            # squares.
            _Tr_Q_sq = float(np.sum(_Q_rec * _Q_rec))
        else:
            _p_sq = 0.0
            _Tr_Q_sq = 0.0
        _has_multipole_force = abs(q_lig) > 1e-9 and (_p_sq > 1e-18 or _Tr_Q_sq > 1e-18)
        # Stash these for the diagnostic print in run().
        self._kb_multipole_active = bool(_has_multipole_force)
        self._kb_mp_p_sq = _p_sq
        self._kb_mp_Tr_Q_sq = _Tr_Q_sq
        # A neutral ligand feels no steering, so U(r) vanishes, but solvent is
        # still squeezed from between the surfaces as they approach. The ordinary
        # path below reduces to k_b = 4 pi D b when hydrodynamics is off, so it
        # also covers the neutral case.
        self._kb_pure_diffusion = bool(
            abs(q_rec * q_lig) < 1e-9 and not _has_multipole_force
        )

        def U(r):
            # Monopole-monopole Yukawa term, always present though possibly
            # zero.
            V0 = q_rec * q_lig * math.exp(-r / debye) / (fpe * r)
            if not _has_multipole_force:
                return V0
            # Second-cumulant correction to the potential of mean force, in
            # kBT_298 units. The factor 1/_kT_scale absorbs kBT in the cumulant
            # divisor so that the final integrand exp(V/kT_T) is
            # Boltzmann-weighted correctly at the actual run temperature.
            e_rl = math.exp(-r / debye)
            g1 = e_rl * (1.0 + r / debye) / (fpe * r * r)
            g2 = (
                e_rl
                * (1.0 + r / debye + r * r / (3.0 * debye * debye))
                / (fpe * r * r * r)
            )
            V1_sq = (q_lig * q_lig) * (_p_sq / 3.0) * g1 * g1
            V2_sq = (q_lig * q_lig) * (2.0 / 15.0) * _Tr_Q_sq * g2 * g2
            return V0 - 0.5 * (V1_sq + V2_sq) / _kT_scale

        # With hydrodynamic interactions the parallel diffusion coefficient is
        # D_parallel(r) = dpre × (ainv - 3/r + 2 a2/r³), and without them it is
        # the constant D_parallel(r) = dpre × ainv = D_t. The reference
        # variables are D_factor = kT/mu, dpre = D_factor/(6π), the hydrodynamic
        # radii a0 and a1, ainv = 1/a0 + 1/a1, and a2 = 0.5 (a0² + a1²). At s=0
        # the integrand is 1/(dpre × ainv) whether or not hydrodynamic
        # interactions are included.
        use_hi = self._use_hi
        if use_hi:
            a0 = float(self.mob.radius1)  # Receptor hydrodynamic radius.
            a1 = float(self.mob.radius2)  # Ligand hydrodynamic radius.
            a2 = 0.5 * (a0**2 + a1**2)
            # Since dpre × ainv = D_t, we have dpre = D_t / ainv.
            ainv = 1.0 / a0 + 1.0 / a1
            dpre = D_t / ainv

        def intgd_romberg(s):
            if s == 0.0:
                return (
                    1.0 / D_t
                )  # The same value with or without hydrodynamic interactions.
            r = 1.0 / s
            v = U(r) / _kT_scale  # Scale V to kBT_T units.
            if use_hi:
                D_para = dpre * (ainv - 3.0 / r + 2.0 * a2 / (r**3))
                return math.exp(v) / D_para
            else:
                return math.exp(v) / D_t

        # k_b(b) and k_b(q_out) from the Romberg integrals out to b and to the
        # escape sphere; the return probability is their ratio.
        self._k_b = 4.0 * math.pi / romberg_integrate(intgd_romberg, 0.0, 1.0 / b, tol=1e-8)
        k_q = 4.0 * math.pi / romberg_integrate(intgd_romberg, 0.0, 1.0 / q_out, tol=1e-8)
        if not all(math.isfinite(k) and k > 0 for k in (self._k_b, k_q)):
            raise ValueError(
                f"k_b = {self._k_b} and k_b(r_esc) = {k_q} must be finite and "
                f"positive (debye_length {debye}, sdie {sdie}, charges {q_rec:+g} "
                f"and {q_lig:+g} e, b {b}, r_esc {q_out})"
            )
        return min(1.0, self._k_b / k_q)

    def _hi_constants(self, D_t: float) -> Tuple[float, ...]:
        """Constants of the RPY D_par(r) that the step uses with hydrodynamic
        interactions.

        Returns (dpre, D0, Deng, a2, S, Ad, am2, p). The prefactor dpre is
        kT/(6π η). D0 = dpre (1/a0 + 1/a1) is the value D_par reaches far
        from contact. Both come from the mobility D_t as in the k_b integrand
        of _compute_return_prob. For a mobility built from radii D_par then
        tends to D_t at the temperature and viscosity of the mobility. The step without hydrodynamic
        interactions, k_b and p_b use the same D_t. Deng is the constant
        D_par once the smaller sphere lies inside the larger. a2, S, Ad, am2
        and p depend on the radii alone.
        """
        a0 = float(getattr(self.mob, "radius1", 0.0))
        a1 = float(getattr(self.mob, "radius2", 0.0))
        # Checked first because ainv divides by both radii.
        if not (a0 > 0.0 and a1 > 0.0):
            raise ValueError(
                "hydrodynamic interactions need two strictly positive "
                f"radii, got a0={a0} A, a1={a1} A"
            )
        # dpre × ainv = D_t gives dpre = D_t / ainv.
        ainv = 1.0 / a0 + 1.0 / a1
        dpre = D_t / ainv
        D0 = D_t
        Deng = D0 - 2.0 * dpre / max(a0, a1)
        a2 = 0.5 * (a0**2 + a1**2)
        # The pair mobility is split at contact, r = a0 + a1, and
        # again where the smaller sphere is swallowed by the larger,
        # r = |a0 - a1|. Inside that second boundary the relative mobility
        # is the difference of the two self mobilities and does not depend
        # on r at all. Precomputed so the step loop stays elementwise.
        S = a0 + a1
        Ad = abs(a0 - a1)
        return dpre, D0, Deng, a2, S, Ad, Ad * Ad, a0 * a1

    def run(self) -> GPUBatchResult:
        """Run all trajectories as a single GPU batch."""
        # Warm up the GPU by forcing CuPy and CUDA kernel JIT compilation before
        # the timing starts. Without this the first run would pay a JIT cost of
        # roughly 2 to 5 seconds inside the timed region.
        _w = cp.zeros(1, dtype=cp.float64)
        _w + 1.0
        cp.cuda.Stream.null.synchronize()
        del _w
        t0 = time.time()
        N = self.params.n_trajectories
        # Random numbers keyed by the seed, the trajectory, its step count and
        # the draw (gpu_random), so none depends on the other trajectories.
        rng = GPURandom(self.params.seed)
        D_t = float(self.mob.relative_translational_diffusion())
        D_r = float(self.mob.relative_rotational_diffusion())
        # The relative rotational diffusion is split between the two bodies.
        # The ligand's share D_r2 spins it about its own centre; the
        # receptor's share D_r1 is applied as a rigid rotation of the relative
        # position and orientation about the frame origin (the receptor
        # centre, and the centre of the b-sphere and escape sphere), which
        # gives the ligand the tangential diffusion 4 D_r1 r² that the
        # receptor's tumbling induces.
        D_r1, D_r2 = _split_rotational_diffusion(self.mob, D_r)
        r_b = self.params.r_start
        r_esc = self.params.r_escape
        print(
            f"  GPU Batch BD: {N:,} trajectories x up to "
            f"{self.params.max_steps:,} steps"
        )
        _bounds = (
            f"minimum_core_dt={getattr(self.params, 'minimum_core_dt', 0.0)}ps  "
            f"max_dt={getattr(self.params, 'max_dt', 0.0) or 'none'}"
        )
        if self._sm_active:
            _bounds += f"  minimum_core_reaction_dt={getattr(self.params, 'minimum_core_reaction_dt', 0.0)}ps"
        print(f"  D_rel={D_t:.5f} A2/ps  {_bounds}")
        print(f"  Progress columns:")
        print(f"    step      = BD time-step counter (all trajectories combined)")
        print(f"    done      = trajectories finished (reacted + escaped) / total")
        print(f"    reacted   = ligand entered reaction zone (criterion fired)")
        print(f"    escaped   = ligand diffused beyond escape radius")
        print(f"    steps/sec = GPU throughput (all active trajectories)")
        # Set up the outer propagator (LMZ). The return probability is the
        # Smoluchowski hitting probability P_return = h(trigger_r) / h(b), with
        # trigger_r = QB_FACTOR × b = 1.1 × b. When r exceeds trigger_r the
        # trajectory snaps back to b with probability P_return, otherwise it
        # escapes.
        trigger_r = self._qb_factor * r_b  # Kept for reference only.
        self._return_prob = self._compute_return_prob()
        _mp_active = bool(getattr(self, "_kb_multipole_active", False))
        if _mp_active:
            _p_mag = math.sqrt(getattr(self, "_kb_mp_p_sq", 0.0))
            _q_mag = math.sqrt(getattr(self, "_kb_mp_Tr_Q_sq", 0.0))
            _mp_tag = (
                f"  [multipole PMF: |p|={_p_mag:.3f} e·Å, "
                f"sqrt(TrQ²)={_q_mag:.3f} e·Å²]"
            )
        else:
            _mp_tag = ""
        print(
            f"  Outer propagator (LMZ): return_prob={self._return_prob:.4f}"
            f"  (applied at r_esc={r_esc:.1f}Å)  k_b={self._k_b:.4f} Å³/ps{_mp_tag}"
        )
        # Initialise all trajectories on the GPU with random start positions on
        # the b-sphere.
        v = rng.normal(
            cp.arange(N, dtype=cp.int64), cp.zeros(N, dtype=cp.int64), 3, rng.START
        )
        v /= cp.linalg.norm(v, axis=1, keepdims=True)
        pos = v * r_b  # (N, 3) centroid positions.
        # Random initial quaternions, ordered (w, x, y, z).
        q = self._random_quaternions_gpu(N, rng)  # (N, 4)
        # Trajectory status, where 0 is running, 1 is reacted, 2 is escaped, and
        # 3 is max-steps.
        status = cp.zeros(N, dtype=cp.int32)
        n_steps = cp.zeros(N, dtype=cp.int64)
        # Track whether each trajectory is currently in the outer region.
        n_reacted = 0
        n_escaped = 0
        n_maxsteps = 0
        total_steps = 0
        # Per-trajectory state-machine tracking, allocated only when
        # state-machine mode is active. The array _sm_current_state has shape
        # (N,) and int32 type and holds the current state index for each
        # trajectory, initialized to 0 (the first_state). The array
        # _sm_rxn_fire_counts has shape (n_rxn,) and int64 type and holds the
        # number of times each reaction has fired across all trajectories,
        # accumulated on the CPU to avoid GPU atomics.
        if self._sm_active:
            self._sm_current_state = cp.zeros(N, dtype=cp.int32)
            self._sm_rxn_fire_counts = np.zeros(len(self._sm_rxn_names), dtype=np.int64)
            # Keep separate debug counters for fires from the discrete path and
            # the bridge path.
            self._sm_rxn_fire_counts_discrete = np.zeros(
                len(self._sm_rxn_names), dtype=np.int64
            )
            self._sm_rxn_fire_counts_bridge = np.zeros(
                len(self._sm_rxn_names), dtype=np.int64
            )
            # For debugging, track which unique trajectories fired each reaction.
            self._sm_rxn_fired_traj_sets = [set() for _ in self._sm_rxn_names]
        else:
            self._sm_current_state = None
            self._sm_rxn_fire_counts = None
        _resume_step = 0
        # Per-trajectory history, encounters and the per-step records. The
        # time maps add dt in fixed point (_TICKS per ps), so their sums do not
        # depend on the order of the atomic adds.
        _start_pos = cp.asnumpy(pos)
        _start_q = cp.asnumpy(q)
        _min_dist = np.full(N, 1e30)
        _step_at_min = np.zeros(N, dtype=np.int64)
        _total_time = np.zeros(N)
        _n_returns = np.zeros(N, dtype=np.int64)
        _bb_triggered = np.zeros(N, dtype=np.int64)
        _prev_r = cp.linalg.norm(pos, axis=1)
        _nm_pos = cp.asnumpy(pos)
        _nm_q = cp.asnumpy(q)
        _enc_pos, _enc_q = [], []
        _enc_traj, _enc_step, _enc_npairs, _enc_bridge = [], [], [], []
        _path_data = []
        _energy_data = []
        _rad_bins = np.linspace(0, float(r_esc * 1.2), 201)
        _rad_counts = np.zeros(len(_rad_bins) - 1, dtype=np.int64)
        _rad_time = np.zeros(len(_rad_bins) - 1, dtype=np.uint64)
        _n_theta = 36
        _n_phi = 72
        _ang_counts = np.zeros((_n_theta, _n_phi), dtype=np.int64)
        _ang_time = np.zeros((_n_theta, _n_phi), dtype=np.uint64)
        # Milestone flux measured on shells at 10%, 20%, up to 100% of r_esc.
        _ms_radii = np.linspace(float(r_b), float(r_esc), 11)
        _ms_flux_out = np.zeros(len(_ms_radii), dtype=np.int64)
        _ms_flux_in = np.zeros(len(_ms_radii), dtype=np.int64)
        _n_rxn_pairs = len(self._rxn_cutoffs) if self._rxn_cutoffs.size > 0 else 0
        _contact_counts = np.zeros(max(_n_rxn_pairs, 1), dtype=np.int64)
        _contact_time = np.zeros(max(_n_rxn_pairs, 1), dtype=np.uint64)
        _contact_total_steps = 0
        _trans_n = 50
        _trans_bins = np.linspace(0, float(r_esc * 1.2), _trans_n + 1)
        _trans_mat = np.zeros((_trans_n, _trans_n), dtype=np.int64)
        _output_cfg = getattr(self.params, "_output_cfg", None)
        _save_interval = getattr(_output_cfg, "save_interval", SAVE_INTERVAL)
        # A checkpoint is read only when checkpoints are on, and only by the run
        # that wrote it: N, seed, max_steps, the pipeline's run key (settings
        # and input files) and every array shape must match.
        _ckpt_interval = getattr(self.params, "checkpoint_interval", 0)
        _ckpt_dir = Path(getattr(self.params, "_work_dir", "bd_sims"))
        _ckpt_path = _ckpt_dir / "checkpoint.npz"
        _run_id = {
            "run_N": str(N),
            "run_seed": str(self.params.seed),
            "run_max_steps": str(self.params.max_steps),
            "run_key": str(getattr(self.params, "_run_key", "")),
        }
        _n_sm = len(self._sm_rxn_names) if self._sm_active else 0
        _like = {
            "pos": pos, "q": q, "status": status, "n_steps": n_steps,
            "prev_r": _prev_r,
            "start_pos": _start_pos, "start_q": _start_q, "min_dist": _min_dist,
            "step_at_min": _step_at_min, "total_time": _total_time,
            "n_returns": _n_returns, "bb_triggered": _bb_triggered,
            "near_miss_pos": _nm_pos, "near_miss_q": _nm_q,
            "rad_counts": _rad_counts, "rad_time": _rad_time,
            "ang_counts": _ang_counts, "ang_time": _ang_time,
            "ms_flux_out": _ms_flux_out, "ms_flux_in": _ms_flux_in,
            "contact_counts": _contact_counts, "contact_time": _contact_time,
            "trans_mat": _trans_mat,
            "sm_current_state": np.zeros(N if self._sm_active else 0, dtype=np.int32),
            "sm_rxn_fire_counts": np.zeros(_n_sm, dtype=np.int64),
        }
        if _ckpt_path.exists() and _ckpt_interval <= 0:
            warnings.warn(
                f"{_ckpt_path} is left untouched and not resumed, since "
                "checkpoint_interval is 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif _ckpt_path.exists():
            ckpt = np.load(str(_ckpt_path))
            _diff = [
                f"{k} {v} here, {str(ckpt[k]) if k in ckpt.files else 'missing'} "
                "in the checkpoint"
                for k, v in _run_id.items()
                if k not in ckpt.files or str(ckpt[k]) != v
            ]
            _diff += [
                f"{k} shape {a.shape} here, "
                f"{ckpt[k].shape if k in ckpt.files else 'missing'} in the checkpoint"
                for k, a in _like.items()
                if k not in ckpt.files or ckpt[k].shape != a.shape
            ]
            if _diff:
                raise RuntimeError(
                    f"{_ckpt_path} was written by another run ("
                    + "; ".join(_diff)
                    + "). Delete it to start this run fresh."
                )
            _r = {
                k: np.ascontiguousarray(ckpt[k], dtype=a.dtype) for k, a in _like.items()
            }
            pos, q = cp.asarray(_r["pos"]), cp.asarray(_r["q"])
            status, n_steps = cp.asarray(_r["status"]), cp.asarray(_r["n_steps"])
            _prev_r = cp.asarray(_r["prev_r"])
            _start_pos, _start_q = _r["start_pos"], _r["start_q"]
            _min_dist, _step_at_min = _r["min_dist"], _r["step_at_min"]
            _total_time, _n_returns = _r["total_time"], _r["n_returns"]
            _bb_triggered = _r["bb_triggered"]
            _nm_pos, _nm_q = _r["near_miss_pos"], _r["near_miss_q"]
            _rad_counts, _rad_time = _r["rad_counts"], _r["rad_time"]
            _ang_counts, _ang_time = _r["ang_counts"], _r["ang_time"]
            _ms_flux_out, _ms_flux_in = _r["ms_flux_out"], _r["ms_flux_in"]
            _contact_counts, _contact_time = _r["contact_counts"], _r["contact_time"]
            _trans_mat = _r["trans_mat"]
            if self._sm_active:
                self._sm_current_state = cp.asarray(_r["sm_current_state"])
                self._sm_rxn_fire_counts = _r["sm_rxn_fire_counts"]
            n_reacted, n_escaped = int(ckpt["n_reacted"]), int(ckpt["n_escaped"])
            n_maxsteps, total_steps = int(ckpt["n_maxsteps"]), int(ckpt["total_steps"])
            _contact_total_steps = int(ckpt["contact_total_steps"])
            _enc_traj = ckpt["enc_traj"].astype(np.int64).tolist()
            _enc_step = ckpt["enc_step"].astype(np.int64).tolist()
            _enc_npairs = ckpt["enc_npairs"].astype(np.int64).tolist()
            _enc_bridge = ckpt["enc_bridge"].astype(np.int64).tolist()
            _enc_pos = [np.asarray(ckpt["enc_pos"], dtype=np.float64)]
            _enc_q = [np.asarray(ckpt["enc_q"], dtype=np.float64)]
            # The checkpoint holds the state after iteration step.
            _resume_step = int(ckpt["step"]) + 1
            t0 -= float(ckpt["elapsed"])
            print(
                f"  Resumed from checkpoint: step={_resume_step}, "
                f"{int((status != 0).sum())}/{N} done "
                f"({n_reacted} react, {n_escaped} esc)"
            )
        if _ckpt_interval > 0:
            _ckpt_dir.mkdir(parents=True, exist_ok=True)
        # The records stay on the GPU until a checkpoint or the end. Poses at an
        # angular bin edge are binned on the host into _ang_host.
        _min_dist = cp.asarray(_min_dist)
        _step_at_min = cp.asarray(_step_at_min)
        _total_time = cp.asarray(_total_time)
        _nm_pos = cp.asarray(_nm_pos)
        _nm_q = cp.asarray(_nm_q)
        _rad_counts = cp.asarray(_rad_counts)
        _rad_time = cp.asarray(_rad_time)
        _ang_host = np.zeros_like(_ang_counts)
        _ang_counts = cp.asarray(_ang_counts)
        _ang_time = cp.asarray(_ang_time)
        _ms_flux_out = cp.asarray(_ms_flux_out)
        _ms_flux_in = cp.asarray(_ms_flux_in)
        _trans_mat = cp.asarray(_trans_mat)
        _contact_counts = cp.asarray(_contact_counts)
        _contact_time = cp.asarray(_contact_time)
        _rad_bins_gpu = cp.asarray(_rad_bins)
        _trans_bins_gpu = cp.asarray(_trans_bins)
        _ms_radii_gpu = cp.asarray(_ms_radii)
        _rec_gho_book = self._rec_gho_pos if _n_rxn_pairs else cp.zeros((0, 3))
        _cut_book = self._rxn_cutoffs_gpu if _n_rxn_pairs else cp.zeros(0)
        _ang_defer = cp.empty((N, 4), dtype=cp.float64)
        _book_cnt = cp.zeros(2, dtype=cp.uint64)
        _book = cp.RawKernel(_BOOK_KERNEL_CODE, "book_kernel")
        # Pre-compute the adaptive timestep constants to avoid Python overhead
        # per step.
        _adt_debye = float(self.engine._debye)
        # The configured solvent permittivity times ε_0 in kBT units at the run
        # temperature, so the adaptive timestep sees the same solvent as the
        # forces.
        _adt_eps = (
            float(getattr(self.params, "dielectric", SOLVENT_DIELECTRIC))
            * VACUUM_PERMITTIVITY_KBT
            * getattr(self.params, "_kT_scale", 1.0)
        )
        _adt_V0 = (
            float(self.mol1.total_charge())
            * float(self.mol2.total_charge())
            / (4.0 * math.pi * _adt_eps)
        )
        _adt_has_force = abs(_adt_V0) > 1e-9
        # The reference minimum_core_dt sets a floor on the adaptive timestep.
        _min_core_dt = float(getattr(self.params, "minimum_core_dt", 0.0))
        # Floor on the timestep near the reaction surface. In the SEEKR2
        # standard configuration this is 0.05 ps, four times smaller than
        # minimum_core_dt = 0.2, which lets the timestep shrink near the
        # reaction surface while still bounding it from below. This matches the
        # minimum_core_reaction_dt semantics of the reference BD engine.
        _min_core_rxn_dt = float(getattr(self.params, "minimum_core_reaction_dt", 0.0))
        # The Brownian bridge needs the diffusion coefficient D of each pair
        # distance over one step. Translation moves the pair with D_trans. With
        # hydrodynamic interactions it moves the pair with D_par at the
        # separation at the start of the step instead (_bridge_D_dt). The two
        # rotations of the step move the pair atoms through their lever arms.
        # The receptor lever arm g_rec runs from the origin to the receptor pair
        # atom. The ligand lever arm R g_lig runs from the ligand centroid to
        # the ligand pair atom. To first order only the part of a lever arm
        # across the pair changes the pair distance. So
        #   D = D_trans + D_r1 |g_rec × s|² + D_r2 |R g_lig × s|²
        # with s the unit pair vector at the start of the step (_bridge_D_rot).
        # The bridge takes the D_r1 and D_r2 of the step. For charged spheres
        # both lever arms are zero and D = D_trans exactly. The lever arm
        # lengths L give the upper bound D_trans + D_r1 L_rec² + D_r2 L_lig².
        # A pair reaches the bound when both lever arms stand at right angles
        # to it.
        if self._rec_gho_pos is not None and len(self._rec_gho_indices) > 0:
            # Lever arm lengths for the upper bound in the log.
            L_rec = np.linalg.norm(cp.asnumpy(self._rec_gho_pos), axis=1)
            L_lig = np.linalg.norm(
                cp.asnumpy(self._mol2_pos0[self._lig_gho_indices]), axis=1
            )
            D_max_np = D_t + D_r1 * L_rec**2 + D_r2 * L_lig**2  # (n_pairs,)
            print(
                f"  BB D_eff: D_trans={D_t:.5f}, D_rot_rec={D_r1:.6f}, "
                f"D_rot_lig={D_r2:.6f}"
            )
            print(
                f"  BB D_eff per pair: min={D_max_np.min():.5f}, "
                f"max={D_max_np.max():.5f}, mean={D_max_np.mean():.5f} Å²/ps "
                f"({D_max_np.mean()/D_t:.1f}x D_trans) at most"
            )
            print(
                "  BB D_eff takes the lever arms across the pair at the start "
                "of each step"
            )
            if self._use_hi:
                print(
                    "  BB D_eff with HI takes D_par(r) at the start of each step "
                    "in place of D_trans"
                )
        # Print startup diagnostics in verbose mode.
        print(f"\n  Diagnostic dump (verbose mode)")
        print(
            f"  Receptor: Q={float(self.mol1.total_charge()):+.2f} e, "
            f"n_atoms={len(self.mol1.atoms)}"
        )
        print(
            f"  Ligand:   Q={float(self.mol2.total_charge()):+.2f} e, "
            f"n_atoms={len(self.mol2.atoms)}, n_lig_gpu={self._N_lig}"
        )
        _q = cp.asnumpy(self._charges)
        print(f"  Charges GPU: {len(_q)} atoms, range=[{_q.min():.2f}, {_q.max():.2f}]")
        _m2p = cp.asnumpy(self._mol2_pos0)
        print(
            f"  Mol2 pos0: {_m2p.shape[0]} atoms, "
            f"extent=[{_m2p.min(0).tolist()}, {_m2p.max(0).tolist()}]"
        )
        print(f"  Yukawa V_factor (Q_rec*Q_lig/(4pi*eps)): {_adt_V0:.6f} kBT·Å")
        print(f"  Has electrostatic force: {_adt_has_force}")
        print(
            f"  Engine has Yukawa fallback: {getattr(self.engine, '_has_yukawa', False)}"
        )
        print(
            f"  Engine V_factor (Q_rec only): "
            f"{getattr(self.engine, '_V_factor', 'N/A')}"
        )
        print(
            f"  Engine receptor_charge: "
            f"{getattr(self.engine, '_rec_charge', 'N/A')}"
        )
        print(f"  Debye length: {_adt_debye:.3f} Å")
        print(f"  r_b (b-sphere): {r_b:.3f} Å")
        print(f"  r_esc (escape): {r_esc:.3f} Å")
        print(f"  return_prob: {self._return_prob:.6f}")
        print(f"  k_b: {self._k_b:.6f} Å³/ps")
        # Grid information.
        if hasattr(self.engine, "_elec_grids_gpu"):
            print(f"  Elec grids on GPU: {len(self.engine._elec_grids_gpu)}")
            for i, g in enumerate(self.engine._elec_grids_gpu):
                lo_np = (
                    g["lo"] if isinstance(g["lo"], np.ndarray) else np.array(g["lo"])
                )
                hi_np = (
                    g["hi"] if isinstance(g["hi"], np.ndarray) else np.array(g["hi"])
                )
                print(
                    f"    Grid {i}: nx={g['nx']} ny={g['ny']} nz={g['nz']}  "
                    f"lo=[{lo_np[0]:.3f},{lo_np[1]:.3f},{lo_np[2]:.3f}]  "
                    f"hi=[{hi_np[0]:.3f},{hi_np[1]:.3f},{hi_np[2]:.3f}]"
                )
        if hasattr(self.engine, "_born_grids_gpu"):
            print(f"  Born grids on GPU: {len(self.engine._born_grids_gpu)}")
        # Reaction criterion information.
        print(f"  Reaction pairs: {len(self._rxn_cutoffs)}")
        print(f"  n_needed: {self._rxn_n_needed}")
        if self._rec_gho_pos is not None:
            _gho = cp.asnumpy(self._rec_gho_pos)
            print(
                f"  Rec GHO: {_gho.shape[0]} positions, "
                f"r_range=[{float(cp.linalg.norm(self._rec_gho_pos, axis=1).min()):.1f}, "
                f"{float(cp.linalg.norm(self._rec_gho_pos, axis=1).max()):.1f}] Å from origin"
            )
        _cuts = self._rxn_cutoffs
        if len(set(_cuts)) == 1:
            print(f"  Rxn cutoffs: all {_cuts[0]:.1f} Å ({len(_cuts)} pairs)")
        else:
            print(f"  Rxn cutoffs: {_cuts}")
        print(f" End startup diagnostics\n")
        # Pre-compute the hydrodynamic-interaction constants to avoid Python
        # overhead per step.
        if self._use_hi:
            (
                _hi_dpre,
                _hi_D0,
                _hi_Deng,
                _hi_a2,
                _hi_S,
                _hi_Ad,
                _hi_am2,
                _hi_p,
            ) = self._hi_constants(D_t)
        # Set up the convergence milestones and the checkpointing.
        _conv_interval = getattr(self.params, "convergence_interval", 10)
        if _conv_interval > 0:
            _conv_milestones = set(
                int(N * p / 100) for p in range(_conv_interval, 100, _conv_interval)
            )
        else:
            _conv_milestones = set()
        _conv_CONV = 6.022e23 * 1e-30 / 1e-12 / 1e-3
        _last_conv_done = 0
        _progress_interval_sec = 10.0  # Print progress every this many seconds.
        _last_progress_time = time.time()
        _last_ckpt_done = 0
        # The overlap check rejects a move that brings a ligand atom closer to a
        # receptor atom than 0.5 (r_lig + r_rec), about 1.7 Å heavy to heavy,
        # which keeps hydrogen bonds and salt bridges. Ghost atoms (radius 0)
        # take no part, as in the soft wall. The receptor is fixed in the
        # simulation frame, so the map that OverlapGrid uses is built once here.
        _use_overlap = getattr(self.params, "_overlap_check", True)
        # The soft repulsion of BrownDye 2, V = kT (0.9345 (r_i + r_j) / r)^12
        # between every receptor and ligand atom, replaces the hard check.
        _use_soft = bool(getattr(self.params, "_soft_repulsion", False))
        # With the Brownian bridge off a reaction counts only when the criterion
        # holds at the end of a step (input key brownian_bridge).
        _use_bridge = bool(getattr(self.params, "use_brownian_bridge", True))
        if not _use_bridge:
            print("  Brownian bridge off. Reactions are checked at the end of each step only.")
        if _use_soft:
            _use_overlap = False
        # With either excluded-volume model the time step has to resolve the
        # receptor surface. The pair step from the centre distance alone is tens
        # of ps at the surface of a protein, which moves the ligand several Å
        # into the receptor at every step.
        _surface_dt = _use_overlap or _use_soft
        if getattr(self.engine, "_use_lj", False) and not _surface_dt:
            raise ValueError(
                "lj_forces needs overlap_check or soft_repulsion, which limit the "
                "time step at the receptor surface."
            )
        _overlap_grid = None
        _room_grid = None
        _soft = None
        _n_overlap_rejected = 0
        _OVERLAP_VDW_FRACTION = 0.5
        _SURFACE_MARGIN = 8.0  # Å of map beyond the reach of the contact test
        rec_pos_np = self.mol1.positions_array()  # (N_rec, 3)
        _rec_real = self.mol1.radii_array() > 0.0
        _room_lig_idx = cp.asarray(
            np.nonzero(self.mol2.radii_array() > 0.0)[0], dtype=cp.int64
        )
        _room_lig_radii = self._mol2_radii_overlap[_room_lig_idx]
        _lig_real_pos0 = self._mol2_pos0[_room_lig_idx]
        if _use_overlap:
            _overlap_grid = OverlapGrid(
                cp.asarray(rec_pos_np[_rec_real], dtype=cp.float64),
                self._mol1_radii_overlap[cp.asarray(_rec_real)],
                _OVERLAP_VDW_FRACTION,
                float(_room_lig_radii.max()),
                xp=cp,
                margin=_SURFACE_MARGIN,
            )
            _room_grid = _overlap_grid
            print(
                f"  Overlap check: atom-pair vdW × {_OVERLAP_VDW_FRACTION:.2f}  "
                f"({self._N_lig} lig × {len(rec_pos_np)} rec atoms, map "
                f"{' × '.join(str(int(n)) for n in _overlap_grid.shape)} at "
                f"{_overlap_grid.spacing} Å)"
            )
        if _use_soft:
            _soft = SoftRepulsion(
                rec_pos_np, self.mol1.radii_array(), self.mol2.radii_array(), xp=cp
            )
            # The room before contact is measured against the contact distance
            # of the repulsion, 0.9345 (r_i + r_j), where V = 1 kT.
            _room_grid = OverlapGrid(
                cp.asarray(rec_pos_np[_rec_real], dtype=cp.float64),
                self._mol1_radii_overlap[cp.asarray(_rec_real)],
                SOFT_A_FACTOR,
                float(_room_lig_radii.max()),
                xp=cp,
                margin=_SURFACE_MARGIN,
            )
            print(
                f"  Soft repulsion: kT (0.9345 (r_i + r_j) / r)^12 up to "
                f"r = 2 × 0.9345 (r_i + r_j)  ({self._N_lig} lig × "
                f"{len(rec_pos_np)} rec atoms, cells "
                f"{' × '.join(str(int(n)) for n in _soft.shape)} of {_soft.h:.2f} Å)"
            )
        # Near the surface an atom moves by translation, by the spin of the
        # ligand about its centroid (lever up to rho_max) and by the tumbling of
        # the receptor (lever up to the receptor extent), so the surface limits
        # use the atom mobility D_t + D_r2 rho_max^2 + D_r1 min(r + rho_max, R)^2.
        _rho_max = float(cp.linalg.norm(self._mol2_pos0, axis=1).max())
        _rec_extent = float(
            np.linalg.norm(rec_pos_np[_rec_real], axis=1).max()
        ) if np.any(_rec_real) else 0.0
        if _surface_dt:
            print(
                "  Surface time step: sqrt(6 D dt) <= 0.7 × room and "
                "D |F| dt <= 0.5 Å near the receptor"
                + ("; D K dt <= 0.05 on the soft wall" if _use_soft else "")
            )
        # Constants of the fused step kernels, evaluated as the step did.
        K = _step_kernels()
        _gho_ref = (
            self._mol2_pos0[self._lig_gho_indices]
            if self._rec_gho_pos is not None
            else None
        )
        _frac = 0.1
        _force_alpha = 0.01
        _max_dt = float(getattr(self.params, "max_dt", 0))
        _dt_par = (
            self._qb_factor * r_b,
            _frac * _frac / 2.0,
            D_t,
            D_r1,
            _adt_has_force,
            _adt_V0,
            _adt_debye,
            1.0 / _adt_debye if _adt_has_force else 0.0,
            _adt_debye**2,
            3.0 * _adt_debye,
            _force_alpha,
            r_b,
            r_esc,
            18.0 * D_t,
        )
        _surf_par = (
            _surface_dt,
            _soft is not None,
            _rho_max,
            _rec_extent,
            D_t,
            D_r1,
            D_t + D_r2 * _rho_max**2,
            D_r2 * _rho_max,
            _max_dt,
            _min_core_dt,
        )
        if self._use_hi:
            _hi_par = (
                _hi_D0,
                _hi_dpre,
                _hi_Deng,
                _hi_S,
                _hi_Ad,
                2.0 * _hi_a2,
                6.0 * _hi_a2,
                _hi_dpre / (8.0 * _hi_p),
                8.0 * _hi_S,
                6.0 * _hi_am2,
                _hi_am2 * _hi_am2,
                3.0 * _hi_dpre / (8.0 * _hi_p),
                _hi_am2,
            )
        _move_par = (D_r1 > 0.0, D_r1, D_r2, 2.0 * D_r1, 2.0 * D_r2)
        # Main Brownian-dynamics loop.
        assert pos.flags.c_contiguous and q.flags.c_contiguous
        assert status.dtype == cp.int32 and n_steps.dtype == cp.int64
        # The pass at step max_steps tests the pose after the last move for a
        # reaction and an escape, and moves nothing.
        for step in range(_resume_step, self.params.max_steps + 1):
            run_idx = cp.where(status == 0)[0]
            n_run = len(run_idx)
            if n_run == 0:
                break
            _n_done = n_reacted + n_escaped
            # Place the ligand atoms for all running trajectories. To stay
            # memory-efficient with large ligands we avoid creating the full
            # (N_run, N_lig, 3) array. Instead we compute the small set of GHO
            # positions for the reaction check and batch the full pos_lig only
            # for the force evaluation. One kernel gathers the centroids and
            # orientations and forms R(q) and the radii.
            pos_run, q_run, R, r_mag = _pose_gpu(pos, q, run_idx)
            # Force batch of at most 4 GiB / 150 bytes of atom-trajectory pairs,
            # which keeps every index of the force kernels below 2^31.
            N_lig = self._mol2_pos0.shape[0]
            _cfg_batch = getattr(self.params, "gpu_force_batch", 0)
            if _cfg_batch > 0:
                _force_batch = _cfg_batch
            else:
                _force_batch = max(1000, int(4 * 1024**3 / max(1, N_lig * 150)))
            _force_batch = min(_force_batch, n_run)  # never exceed running count
            _n_force_batches = (n_run + _force_batch - 1) // _force_batch
            if step == 0:
                print(
                    f"  Force batch: {_force_batch:,} traj/batch "
                    f"({_n_force_batches} batches for {n_run:,} running)"
                )
            # Run the reaction check on the small set of GHO atoms. In
            # state-machine mode this dispatches to a per-reaction check that
            # respects the trajectory state and fires only matching reactions,
            # and otherwise it uses the flattened-pairs check.
            if self._sm_active:
                # State-aware reaction check that returns a reaction index per
                # trajectory, or -1 if no reaction fired this step.
                current_state_run = self._sm_current_state[run_idx]
                fired_rxn_idx = self._check_reactions_gpu_gho_state_machine(
                    pos_run, q_run, current_state_run
                )  # (N_run,) int32
                reacted = fired_rxn_idx >= 0  # (N_run,) boolean.
                still_running_mask = status[run_idx] == 0
                # Trajectories that fired a reaction this step and are still
                # running.
                fired_mask = reacted & still_running_mask
                if bool(cp.any(fired_mask)):
                    fired_global = run_idx[fired_mask]
                    fired_rxn_idx_masked = fired_rxn_idx[fired_mask]
                    # Update the state, setting current_state[traj] to the
                    # state_after of the fired reaction.
                    new_states = self._sm_rxn_state_after_idx_gpu[fired_rxn_idx_masked]
                    self._sm_current_state[fired_global] = new_states
                    # Increment the per-reaction fire counter on the CPU.
                    fired_rxn_idx_np = cp.asnumpy(fired_rxn_idx_masked)
                    _fired_global_np = cp.asnumpy(fired_global)
                    for _j, _rxn_i in enumerate(fired_rxn_idx_np):
                        self._sm_rxn_fire_counts[int(_rxn_i)] += 1
                        self._sm_rxn_fire_counts_discrete[int(_rxn_i)] += 1
                        self._sm_rxn_fired_traj_sets[int(_rxn_i)].add(
                            int(_fired_global_np[_j])
                        )
                    # Mark trajectories as reacted only if their new state is
                    # terminal, meaning no further reactions are possible from
                    # that state.
                    terminal_after = self._sm_state_is_terminal_gpu[new_states]
                    newly_reacted = fired_global[terminal_after]
                else:
                    newly_reacted = cp.array([], dtype=run_idx.dtype)
                if len(newly_reacted) > 0:
                    status[newly_reacted] = 1
                    n_reacted += int(len(newly_reacted))
                    _nr_np = cp.asnumpy(newly_reacted)
                    _enc_traj.extend(_nr_np.tolist())
                    _enc_step.extend([step] * len(_nr_np))
                    _enc_pos.append(cp.asnumpy(pos[newly_reacted]))
                    _enc_q.append(cp.asnumpy(q[newly_reacted]))
                    _enc_npairs.extend([1] * len(_nr_np))
                    _enc_bridge.extend([0] * len(_nr_np))
            else:
                # Flattened-pairs reaction check on the GHO atoms only, which
                # are placed, measured and counted in one kernel.
                if self._rec_gho_pos is not None:
                    _, _, _n_sat, reacted = _pairs_gpu(
                        R,
                        pos_run,
                        _gho_ref,
                        self._rec_gho_pos,
                        self._rxn_cutoffs_gpu,
                        self._rxn_n_needed,
                    )  # (N_run,) pair count and (N_run,) boolean.
                else:
                    reacted = cp.zeros(n_run, dtype=cp.bool_)
                    _n_sat = cp.zeros(n_run, dtype=cp.int64)
            # Trajectories at r_escape that did not react, and one read of how
            # many reacted and how many reached r_escape. Every running
            # trajectory has status 0 here, so no other mask is needed.
            at_escape = cp.empty(n_run, dtype=cp.bool_)
            _counts = cp.zeros(2, dtype=cp.int32)
            K["escape"](r_mag, reacted, r_esc, at_escape, _counts)
            n_hit, n_esc = _counts.get().tolist()
            if not self._sm_active and n_hit > 0:
                # Mark the newly reacted trajectories.
                newly_reacted = run_idx[reacted]
                status[newly_reacted] = 1
                n_reacted += int(len(newly_reacted))
                # Record the encounter snapshots. The pair count is masked with
                # the same selection as the positions, because _n_sat is indexed
                # by position within run_idx rather than by trajectory id.
                _nr_np = cp.asnumpy(newly_reacted)
                _enc_traj.extend(_nr_np.tolist())
                _enc_step.extend([step] * len(_nr_np))
                _enc_pos.append(cp.asnumpy(pos[newly_reacted]))
                _enc_q.append(cp.asnumpy(q[newly_reacted]))
                _enc_npairs.extend(cp.asnumpy(_n_sat[reacted]).tolist())
                _enc_bridge.extend([0] * len(_nr_np))
            # Global indices of trajectories snapped back to the b-surface this
            # step. Their radius jumps discontinuously from the outer region to
            # r_b, so they are excluded from the milestone-flux and transition
            # diagnostics below to avoid counting snap artifacts as flux.
            _snapped_gids = np.empty(0, dtype=np.int64)
            # For trajectories that reach r_escape, apply the return
            # probability.
            if n_esc > 0:
                esc_idx = run_idx[at_escape]
                u_outer = rng.uniform(esc_idx, n_steps, 1, rng.ESCAPE)[:, 0]
                returns = u_outer < self._return_prob
                # These trajectories are snapped back to the b-sphere.
                if cp.any(returns):
                    ret_idx = esc_idx[returns]
                    new_ret = ret_idx[status[ret_idx] == 0]
                    if len(new_ret) > 0:
                        dirs = pos[new_ret] / cp.linalg.norm(
                            pos[new_ret], axis=1, keepdims=True
                        )
                        pos[new_ret] = dirs * r_b
                        _snapped_gids = cp.asnumpy(new_ret)
                        # Apply diffusional rotation over the estimated time
                        # spent in the outer region. The snap propagator does
                        # not track the per-trajectory elapsed outer time, so we
                        # estimate it from the free-diffusion radial timescale
                        #   t_outer ≈ (r_esc² - r_b²) / (6 D_t).
                        # Then tau = t_outer × D_r feeds diffusional_rotation,
                        # drawn on the GPU, which handles all regimes,
                        # taking a Gaussian step for small tau, a recursive split
                        # for intermediate tau, and a uniform fallback for tau ≥
                        # 4. A future fix for a real GPU outer propagator will
                        # replace this with a per-trajectory sampled t_outer.
                        n_ret = int(len(new_ret))
                        t_outer_est = max(
                            0.0, (r_esc * r_esc - r_b * r_b) / (6.0 * D_t)
                        )
                        # The two bodies tumble independently over the outer
                        # excursion (see _return_rotation_host).
                        tau_lig = t_outer_est * D_r2
                        tau_rec = t_outer_est * D_r1
                        if tau_lig > 0.0 or tau_rec > 0.0:
                            q_new, pos_new = rng.return_rotation(
                                q[new_ret],
                                pos[new_ret],
                                new_ret,
                                n_steps,
                                tau_lig,
                                tau_rec,
                            )
                            q[new_ret] = q_new
                            pos[new_ret] = pos_new
                        # When both taus are 0 the rotation is degenerate, so q
                        # and pos are left unchanged.
                        if not getattr(self, "_diff_rot_logged", False):
                            print(
                                f"  Return rotation: "
                                f"t_outer_est={t_outer_est:.1f} ps, "
                                f"tau_lig={tau_lig:.3f} (D_r2={D_r2:.4e}), "
                                f"tau_rec={tau_rec:.3f} (D_r1={D_r1:.4e}) rad^2/ps"
                            )
                            self._diff_rot_logged = True
                        # Track the returns.
                        _ret_np = cp.asnumpy(new_ret)
                        _n_returns[_ret_np] += 1
                # Handle the true escapes.
                if cp.any(~returns):
                    true_esc = esc_idx[~returns]
                    new_esc = true_esc[status[true_esc] == 0]
                    if len(new_esc) > 0:
                        status[new_esc] = 2
                        n_escaped += int(len(new_esc))
            if step == self.params.max_steps:
                break
            # A trajectory is still running if it has not reacted, escaped, or
            # otherwise finished.
            sr_mask = status[run_idx] == 0
            # Gather the poses again when some returned to the b-sphere.
            if len(_snapped_gids) > 0:
                pos_run, q_run, R, r_mag = _pose_gpu(pos, q, run_idx)
            # pos_lig is not computed here, since it is batched during the force
            # evaluation. The force calculation runs as a memory-safe GPU batch.
            if n_reacted + n_escaped == _n_done:
                sr_idx, R_sr, cen_sr, r_sr = run_idx, R, pos_run, r_mag
            else:
                _keep = cp.where(sr_mask)[0]
                sr_idx = run_idx[_keep]
                if len(sr_idx) == 0:
                    continue
                R_sr = R[_keep]  # (N_sr, 3, 3)
                cen_sr = pos_run[_keep]  # (N_sr, 3) centroids
                r_sr = r_mag[_keep]
            N_sr = len(sr_idx)
            # Batch the force computation to limit GPU memory use.
            forces_gpu = cp.zeros((N_sr, 3), dtype=cp.float64)
            torques_gpu = cp.zeros((N_sr, 3), dtype=cp.float64)
            _room_gpu = (
                cp.full(N_sr, cp.inf, dtype=cp.float64) if _surface_dt else None
            )
            _stiff_gpu = (
                cp.zeros(N_sr, dtype=cp.float64) if _soft is not None else None
            )
            for _fb0 in range(0, N_sr, _force_batch):
                _fb1 = min(_fb0 + _force_batch, N_sr)
                _R_b = R_sr[_fb0:_fb1]
                _cen_b = cen_sr[_fb0:_fb1]
                _pos_lig_b = (
                    cp.einsum("nij,kj->nki", _R_b, self._mol2_pos0) + _cen_b[:, None, :]
                )
                _f_b, _t_b, _ = self.engine(
                    _pos_lig_b, self._charges, R_matrices=_R_b, centroids=_cen_b
                )
                forces_gpu[_fb0:_fb1] = _f_b
                torques_gpu[_fb0:_fb1] = _t_b
                if _soft is not None:
                    _f_s, _t_s, _, _k_s = _soft(_pos_lig_b, _cen_b)
                    forces_gpu[_fb0:_fb1] += _f_s
                    torques_gpu[_fb0:_fb1] += _t_s
                    _stiff_gpu[_fb0:_fb1] = _k_s
                if _surface_dt:
                    _room_gpu[_fb0:_fb1] = _room_grid.clearance(
                        _pos_lig_b[:, _room_lig_idx], _room_lig_radii
                    )
                del _pos_lig_b
            # forces_gpu now has shape (N_sr, 3).
            # Save the old pair distances for the Brownian bridge. After the BD
            # step we will check whether the continuous Brownian path crossed
            # the reaction surface even when both endpoints lie above it. The
            # exact crossing probability is
            #   P(cross) = exp(-x₀ × x₁ / (D × Δt)),
            # where x = pair_distance - cutoff is the distance above the
            # reaction surface.
            _old_pair_dists = None
            if self._rec_gho_pos is not None:
                # Compute only the GHO positions, a small (N_sr × n_gho × 3)
                # array, and the pair distances (N_sr, n_pairs).
                _lig_gho_old, _old_pair_dists, _, _ = _pairs_gpu(
                    R_sr,
                    cen_sr,
                    _gho_ref,
                    self._rec_gho_pos,
                    self._rxn_cutoffs_gpu,
                    self._rxn_n_needed,
                )
            # Choose an adaptive time step as the minimum of three criteria.
            # The pair criterion is
            #   dt_pair = (frac²/2) × r² / D  with frac = 0.1,
            # the maximum safe Δt at separation r, which keeps the mean
            # displacement below 10% of the separation per step. At r = 172 this
            # gives dt_pair = 0.005 × 172² / 0.0187 ≈ 7912 ps, so σ ≈ 17 Å, and
            # at r = 10 it gives dt_pair = 0.005 × 100 / 0.434 ≈ 1.15 ps. The
            # force criterion is
            #   dt_force = alpha / |D × Fr1|  with alpha = 0.01 and
            #   Fr1 = dF0/dr, the derivative of the screened Coulomb force,
            # which controls force accuracy and is only active near the Yukawa
            # region. The edge criterion is
            #   dt_edge = min(r-b, q-r)² / (18 D),
            # which prevents overshooting the b-sphere and r_escape boundaries.
            # The final step is dt = min(dt_pair, dt_force, dt_edge).
            # The receptor's tumbling moves the ligand tangentially with
            # diffusion D_r1 r² per direction, which the same bound covers, so
            # dt_pair takes D_t + D_r1 r². The force criterion applies only
            # where the Yukawa force is significant (_radial_force_derivative).
            # PySTARC uses a Brownian bridge, which catches mid-step crossings
            # without shrinking Δt and so avoids the trapping problem. The two
            # zones are combined as follows. In the outer zone (r > 1.1 × b) the
            # step is min(dt_force, dt_edge, dt_pair), and in the inner zone (r ≤
            # 1.1 × b) it is min(dt_force, dt_pair).
            dt_arr = cp.empty(N_sr)
            in_outer = cp.empty(N_sr, dtype=cp.bool_)
            K["dt"](r_sr, *_dt_par, dt_arr, in_outer)
            # Refine the timestep near the reaction surface in state-machine
            # mode. Reference BD engines refine the timestep so that the
            # diffusive step stays much smaller than the distance to the
            # reaction surface. The criterion is that 12σ should stay below the
            # minimum reaction coordinate reachable from the current state,
            # which rearranges to dt ≤ rxn_coord² / (288 D). Trajectories that
            # are already inside the surface, or that have no reachable
            # reactions from their current state, use a large sentinel value so
            # that this term does not constrain them.
            if (
                self._sm_active
                and _old_pair_dists is not None
                and self._sm_rxn_state_before_idx_gpu is not None
            ):
                _sm_state_for_dt = self._sm_current_state[sr_idx]
                _sm_reachable = (
                    _sm_state_for_dt[:, None]
                    == self._sm_rxn_state_before_idx_gpu[None, :]
                )  # (N_sr, n_pairs)
                _rxn_coord_per_pair = cp.maximum(
                    _old_pair_dists - self._rxn_cutoffs_gpu[None, :],
                    cp.zeros_like(_old_pair_dists),
                )  # (N_sr, n_pairs)
                _big = cp.asarray(1.0e30, dtype=cp.float64)
                _rxn_coord_masked = cp.where(_sm_reachable, _rxn_coord_per_pair, _big)
                _rxn_coord_per_traj = cp.min(_rxn_coord_masked, axis=1)  # (N_sr,)
                # This refinement activates only when the current diffusive
                # width would approach the distance to the reaction surface,
                # that is when 12σ > rxn_coord, matching the adaptive criterion
                # of the reference BD engine. When the trigger fires, Δt is
                # reduced toward rxn_coord²/(288 D_t) but never below the
                # configured reaction-surface floor _min_core_rxn_dt, which is
                # 0.05 ps in SEEKR2. Trajectories far from any reachable
                # reaction surface are not slowed.
                _sigma_now = cp.sqrt(2.0 * D_t * dt_arr)
                _needs_refinement = 12.0 * _sigma_now > _rxn_coord_per_traj
                _rxn_coord_sq = _rxn_coord_per_traj * _rxn_coord_per_traj
                _dt_rxn_crit = _rxn_coord_sq / (288.0 * D_t)
                if _min_core_rxn_dt > 0:
                    _dt_rxn_floored = cp.maximum(
                        _dt_rxn_crit,
                        cp.asarray(_min_core_rxn_dt, dtype=cp.float64),
                    )
                else:
                    _dt_rxn_floored = _dt_rxn_crit
                dt_arr = cp.where(
                    _needs_refinement,
                    cp.minimum(dt_arr, _dt_rxn_floored),
                    dt_arr,
                )
            # Surface rule (room before contact, drift, wall stiffness), then the
            # cap of the run and the minimum_core_dt floor.
            _dt_cap = cp.empty(N_sr)
            K["surface"](
                r_sr,
                dt_arr,
                forces_gpu,
                torques_gpu,
                _room_gpu if _surface_dt else dt_arr,
                _stiff_gpu if _soft is not None else dt_arr,
                *_surf_par,
                _dt_cap,
            )
            dt_arr = _dt_cap
            # Compute the diffusion coefficient for the BD step,
            #   D_parallel(r) = D_factor/pi6 × (ainv + Di(r)),
            # where Di captures the Oseen hydrodynamic coupling. D_parallel is
            # applied isotropically, the same for all three noise components,
            # rather than as an anisotropic radial and tangential split, which
            # belongs in the outer propagator loop where PySTARC uses the return
            # probability. Note that D_par_arr is the translational D_parallel as
            # a function of r, given per trajectory, while D_r below is the
            # scalar rotational diffusion coefficient. The two are distinct, and
            # the _arr suffix is not a typo.
            if self._use_hi:
                # RPY D_par(r) in its three regimes (far, partial overlap,
                # engulfed), with dD/dr branched with D so that the drift is the
                # divergence of the D in use, capped at a tenth of the random
                # step below contact.
                D_par_arr = cp.empty(N_sr)
                _div_drift = cp.empty((N_sr, 3))
                K["hi"](r_sr, dt_arr, cen_sr, *_hi_par, D_par_arr, _div_drift)
            else:
                D_par_arr = cp.full(N_sr, D_t, dtype=cp.float64)
                # A constant D has vanishing divergence.
                _div_drift = None
            # Translational update following the Ermak-McCammon scheme.
            noise_t = rng.normal(sr_idx, n_steps, 3, rng.NOISE_T)
            # Rotational update of the ligand about its own centre.
            noise_r = rng.normal(sr_idx, n_steps, 3, rng.NOISE_R2)
            # Rotational update of the receptor, applied as a rigid rotation of
            # the post-translation relative configuration about the origin.
            if D_r1 > 0.0:
                noise_r1 = rng.normal(sr_idx, n_steps, 3, rng.NOISE_R1)
            # Save the old positions in case the overlap check rejects the move.
            _old_pos_sr = pos[sr_idx].copy() if _use_overlap else None
            _old_q_sr = q[sr_idx].copy() if _use_overlap else None
            # The three updates in one kernel, which writes pos and q in place.
            drift = cp.empty((N_sr, 3))
            sigma_t = cp.empty((N_sr, 1))
            K["move"](
                sr_idx,
                dt_arr,
                D_par_arr,
                cen_sr,
                forces_gpu,
                torques_gpu,
                forces_gpu if _div_drift is None else _div_drift,
                noise_t,
                noise_r,
                noise_r1 if D_r1 > 0.0 else noise_r,
                _div_drift is not None,
                *_move_par,
                pos,
                q,
                drift,
                sigma_t[:, 0],
            )
            # Overlap check: a move that brings a ligand atom within
            # OVERLAP_VDW_FRACTION × (r_vdw_lig + r_vdw_rec) of a receptor atom
            # is rejected, and the clock still advances. The lab-frame ligand
            # atoms follow from the updated centroid and quaternion. OverlapGrid
            # clears most atoms with one lookup in a precomputed map of the
            # receptor and tests only the atoms near the threshold against every
            # receptor atom, so its decisions are those of the all-pairs test at
            # a fraction of the cost.
            # Track which still-running trajectories have their move reverted by
            # the overlap check this step, so the bridge crossing below can skip
            # them. None means no overlap check ran.
            _reverted_mask = None
            if _use_overlap and _overlap_grid is not None:
                _new_R_sr = self._quats_to_rotmats(q[sr_idx])  # (N_sr, 3, 3)
                _new_lig_atoms = (
                    cp.einsum("nij,kj->nki", _new_R_sr, _lig_real_pos0)
                    + pos[sr_idx][:, None, :]
                )  # (N_sr, N_lig_real, 3)
                _inside_all = _overlap_grid.overlaps(
                    _new_lig_atoms, _room_lig_radii
                )  # (N_sr,)
                if cp.any(_inside_all):
                    _inside_idx = cp.where(_inside_all)[0]
                    pos[sr_idx[_inside_idx]] = _old_pos_sr[_inside_idx]
                    q[sr_idx[_inside_idx]] = _old_q_sr[_inside_idx]
                    _n_overlap_rejected += int(_inside_idx.shape[0])
                # The reverted trajectories did not move this step. Record them in
                # sr_idx order, aligned with the bridge arrays below.
                _reverted_mask = _inside_all
            # Brownian bridge crossing check at the reaction surface. To catch
            # reactions that the discrete endpoint checks miss, PySTARC uses the
            # exact Brownian bridge crossing probability
            #   P(path crossed boundary | start x0, end x1, both > 0)
            #       = exp(-x0 × x1 / (D × Δt)),
            # where x = pair_distance - cutoff is the height above the reaction
            # surface. This is exact for free diffusion and accurate when the
            # drift is much smaller than the noise. In state-machine mode the
            # bridge is checked per reaction, with the trajectory's current state
            # gating which reactions can fire. The pre-step distances stored in
            # _old_pair_dists align with the per-reaction arrays because each
            # state-machine reaction has exactly one pair, matching the
            # flattened-pair layout used elsewhere.
            if (
                _use_bridge
                and _old_pair_dists is not None
                and len(sr_idx) > 0
                and self._rec_gho_pos is not None
            ):
                # Pair distances after the step and the crossing probability of
                # each pair with both ends above its cutoff, with the rotation
                # part of D from the pose at the start of the step.
                _new_pair_dists = cp.empty((N_sr, _n_rxn_pairs))
                _p_cross = cp.empty((N_sr, _n_rxn_pairs))
                _both_above = cp.empty((N_sr, _n_rxn_pairs), dtype=cp.bool_)
                K["bridge"](
                    sr_idx,
                    D_par_arr,
                    dt_arr,
                    pos,
                    q,
                    _gho_ref,
                    self._rec_gho_pos,
                    self._rxn_cutoffs_gpu,
                    _lig_gho_old,
                    _old_pair_dists,
                    cen_sr,
                    _n_rxn_pairs,
                    D_r1,
                    D_r2,
                    _new_pair_dists,
                    _p_cross,
                    _both_above,
                )
                # Every pair is tested at the end of every step. It fires when its
                # path crossed (u < P) or it ended inside its cutoff, unless the
                # overlap check undid the move. bb_triggered marks a reaction
                # with fewer than n_needed pairs inside at the end (in the state
                # machine, the fired pair ended outside).
                _u_bb = rng.uniform(sr_idx, n_steps, _n_rxn_pairs, rng.BRIDGE)
                _pair_fired = cp.empty((N_sr, _n_rxn_pairs), dtype=cp.bool_)
                _total_fired = cp.empty(N_sr, dtype=cp.int64)
                _n_inside = cp.empty(N_sr, dtype=cp.int64)
                _bb_reacted = cp.empty(N_sr, dtype=cp.bool_)
                K["fire"](
                    _u_bb,
                    _p_cross,
                    _new_pair_dists,
                    self._rxn_cutoffs_gpu,
                    _both_above if _reverted_mask is None else _reverted_mask,
                    _reverted_mask is not None,
                    _n_rxn_pairs,
                    self._rxn_n_needed,
                    _pair_fired,
                    _total_fired,
                    _n_inside,
                    _bb_reacted,
                )
                if self._sm_active:
                    # Gate each firing on the trajectory's state, fire the
                    # lowest-index matching reaction, and end the trajectory
                    # only when the new state is terminal.
                    _sm_cur_state_sr = self._sm_current_state[sr_idx]
                    _sm_state_ok = (
                        _sm_cur_state_sr[:, None]
                        == self._sm_rxn_state_before_idx_gpu[None, :]
                    )
                    _sm_fired_masked = _pair_fired & _sm_state_ok
                    _sm_any_fire = _sm_fired_masked.any(axis=1)
                    if bool(cp.any(_sm_any_fire)):
                        _sm_fired_rxn_idx = cp.argmax(_sm_fired_masked, axis=1).astype(
                            cp.int32
                        )
                        _sm_cross = ~(
                            _new_pair_dists[cp.arange(N_sr), _sm_fired_rxn_idx]
                            < self._rxn_cutoffs_gpu[_sm_fired_rxn_idx]
                        )
                        _still_running = status[sr_idx] == 0
                        _sm_proceed = _sm_any_fire & _still_running
                        if bool(cp.any(_sm_proceed)):
                            _sm_bb_new = sr_idx[_sm_proceed]
                            _sm_rxn_for_new = _sm_fired_rxn_idx[_sm_proceed]
                            _new_states_bb = self._sm_rxn_state_after_idx_gpu[
                                _sm_rxn_for_new
                            ]
                            self._sm_current_state[_sm_bb_new] = _new_states_bb
                            _sm_rxn_for_new_np = cp.asnumpy(_sm_rxn_for_new)
                            _sm_bb_new_np = cp.asnumpy(_sm_bb_new)
                            for _j, _rxn_i in enumerate(_sm_rxn_for_new_np):
                                self._sm_rxn_fire_counts[int(_rxn_i)] += 1
                                self._sm_rxn_fire_counts_bridge[int(_rxn_i)] += 1
                                self._sm_rxn_fired_traj_sets[int(_rxn_i)].add(
                                    int(_sm_bb_new_np[_j])
                                )
                            _terminal_after_bb = self._sm_state_is_terminal_gpu[
                                _new_states_bb
                            ]
                            _sm_bb_terminal = _sm_bb_new[_terminal_after_bb]
                            if len(_sm_bb_terminal) > 0:
                                status[_sm_bb_terminal] = 1
                                n_reacted += int(len(_sm_bb_terminal))
                                _bb_np = cp.asnumpy(_sm_bb_terminal)
                                _bb_x = cp.asnumpy(
                                    _sm_cross[_sm_proceed][_terminal_after_bb]
                                ).astype(np.int64)
                                _bb_triggered[_bb_np] = _bb_x
                                _enc_traj.extend(_bb_np.tolist())
                                _enc_step.extend([step] * len(_bb_np))
                                _enc_pos.append(cp.asnumpy(pos[_sm_bb_terminal]))
                                _enc_q.append(cp.asnumpy(q[_sm_bb_terminal]))
                                _enc_npairs.extend([1] * len(_bb_np))
                                _enc_bridge.extend(_bb_x.tolist())
                elif cp.any(_bb_reacted):
                    _bb_global = sr_idx[_bb_reacted]
                    _bb_fired = _total_fired[_bb_reacted]
                    _bb_inside = _n_inside[_bb_reacted]
                    _bb_keep = status[_bb_global] == 0
                    _bb_new = _bb_global[_bb_keep]
                    if len(_bb_new) > 0:
                        status[_bb_new] = 1
                        n_reacted += int(len(_bb_new))
                        _bb_np = cp.asnumpy(_bb_new)
                        _bb_x = cp.asnumpy(
                            _bb_inside[_bb_keep] < self._rxn_n_needed
                        ).astype(np.int64)
                        _bb_triggered[_bb_np] = _bb_x
                        _enc_traj.extend(_bb_np.tolist())
                        _enc_step.extend([step] * len(_bb_np))
                        _enc_pos.append(cp.asnumpy(pos[_bb_new]))
                        _enc_q.append(cp.asnumpy(q[_bb_new]))
                        _enc_npairs.extend(cp.asnumpy(_bb_fired[_bb_keep]).tolist())
                        _enc_bridge.extend(_bb_x.tolist())
            total_steps += N_sr
            # Per-step records. The step counts take the pose after the step, the
            # time maps and contacts the pose the step's dt belongs to. Returns
            # to the b-sphere stay out of the milestone flux and transitions.
            _cur_r = cp.linalg.norm(pos[sr_idx], axis=1)
            _gho_book = (
                cp.ascontiguousarray(_lig_gho_old) if _n_rxn_pairs else _rec_gho_book
            )
            _book_cnt.fill(0)
            _book(
                ((N_sr + 255) // 256,),
                (256,),
                (
                    sr_idx, _cur_r, dt_arr, np.int32(N_sr), pos, q, status,
                    np.int64(step), cp.asarray(_snapped_gids),
                    np.int32(len(_snapped_gids)),
                    _gho_book, _rec_gho_book, _cut_book, np.int32(_n_rxn_pairs),
                    _rad_bins_gpu, np.int32(len(_rad_bins)),
                    _trans_bins_gpu, np.int32(len(_trans_bins)),
                    _ms_radii_gpu, np.int32(len(_ms_radii)),
                    np.int32(_n_theta), np.int32(_n_phi), np.float64(_ANGLE_EDGE),
                    n_steps, _min_dist, _step_at_min, _nm_pos, _nm_q,
                    _total_time, _prev_r, _rad_counts, _ang_counts,
                    _ms_flux_out, _ms_flux_in, _trans_mat, _contact_counts,
                    _ang_defer, _book_cnt, r_sr, cen_sr, np.float64(_TICKS),
                    _rad_time, _ang_time, _contact_time,
                ),
            )
            if _n_rxn_pairs:
                _contact_total_steps += N_sr
            # Record paths and energetics every save_interval steps.
            if _save_interval > 0 and step % _save_interval == 0:
                if _output_cfg is None or getattr(_output_cfg, "full_paths", True):
                    _sr_np = cp.asnumpy(sr_idx)
                    _snap = np.column_stack(
                        [
                            _sr_np.astype(np.float64),
                            np.full(len(_sr_np), step, dtype=np.float64),
                            cp.asnumpy(pos[sr_idx]),
                            # All four components: the norm gives only |q3|,
                            # and q and -q are not kept in one hemisphere.
                            cp.asnumpy(q[sr_idx]),
                        ]
                    )  # (n_running, 9)
                    _path_data.append(_snap)

                if _output_cfg is None or getattr(_output_cfg, "energetics", True):
                    _sr_np = cp.asnumpy(sr_idx)
                    _esnap = np.column_stack(
                        [
                            _sr_np.astype(np.float64),
                            np.full(len(_sr_np), step, dtype=np.float64),
                            cp.asnumpy(forces_gpu),
                            cp.asnumpy(dt_arr),
                        ]
                    )  # (n_running, 6)
                    _energy_data.append(_esnap)
            # Progress report. Every trajectory that is not running has
            # finished.
            _n_live, _n_edge = cp.asnumpy(_book_cnt).tolist()
            if _n_edge:
                _angular_bins_host(
                    cp.asnumpy(_ang_defer[:_n_edge]), _n_theta, _n_phi, _ang_host
                )
            done = N - _n_live
            # The step limit is the only end of a trajectory other than a
            # reaction or an escape. There is no early exit: a trajectory that
            # is still running lingers near the receptor and is more likely
            # than average to react, so stopping the run early biases k_on
            # downward. Choose max_steps from a pilot run with
            # pystarc.analysis.step_limit so that the trajectories at the limit
            # cannot move k_on by more than the tolerance.
            # Live convergence report.
            if done in _conv_milestones and done > _last_conv_done:
                _last_conv_done = done
                _n_comp = n_reacted + n_escaped
                if _n_comp > 0:
                    _p_live = n_reacted / _n_comp
                    _k_live = _conv_CONV * self._k_b * _p_live if self._k_b > 0 else 0
                    _pct = done / N * 100
                    _elapsed = time.time() - t0
                    print(
                        f"  -- {_pct:.0f}% converged --  "
                        f"P_rxn={_p_live:.5f}  "
                        f"k_on={_k_live:.3e}  "
                        f"({n_reacted:,} react / {n_escaped:,} esc)  "
                        f"[{_elapsed:.0f}s]"
                    )
            # Time-based progress, printed every 10 seconds.
            _now = time.time()
            if _now - _last_progress_time >= _progress_interval_sec:
                _last_progress_time = _now
                _n_comp = n_reacted + n_escaped
                _pct_done = done / N * 100
                _elapsed = _now - t0
                if _n_comp > 0:
                    _p_live = n_reacted / _n_comp
                    _k_live = _conv_CONV * self._k_b * _p_live if self._k_b > 0 else 0
                    # Format k_on in mantissa-and-exponent form.
                    if _k_live > 0:
                        _exp = int(math.floor(math.log10(_k_live)))
                        _man = _k_live / 10**_exp
                        print(
                            f"  [T] {_elapsed:6.0f}s  "
                            f"done={done:,}/{N:,} ({_pct_done:.1f}%)  "
                            f"reacted={n_reacted:,}  escaped={n_escaped:,}  "
                            f"P_rxn={_p_live:.5f}  "
                            f"k_on={_man:.2f}e{_exp}"
                        )
                    else:
                        print(
                            f"  [T] {_elapsed:6.0f}s  "
                            f"done={done:,}/{N:,} ({_pct_done:.1f}%)  "
                            f"reacted={n_reacted:,}  escaped={n_escaped:,}"
                        )
            # Checkpoint, written to a temporary file and renamed.
            if _ckpt_interval > 0 and done - _last_ckpt_done >= _ckpt_interval:
                _last_ckpt_done = done
                _ckpt_tmp = _ckpt_dir / "checkpoint.tmp.npz"
                np.savez_compressed(
                    str(_ckpt_tmp),
                    **_run_id,
                    pos=cp.asnumpy(pos),
                    q=cp.asnumpy(q),
                    status=cp.asnumpy(status),
                    n_steps=cp.asnumpy(n_steps),
                    n_reacted=n_reacted,
                    n_escaped=n_escaped,
                    n_maxsteps=n_maxsteps,
                    total_steps=total_steps,
                    step=step,
                    elapsed=time.time() - t0,
                    start_pos=_start_pos,
                    start_q=_start_q,
                    min_dist=cp.asnumpy(_min_dist),
                    step_at_min=cp.asnumpy(_step_at_min),
                    total_time=cp.asnumpy(_total_time),
                    n_returns=_n_returns,
                    bb_triggered=_bb_triggered,
                    prev_r=cp.asnumpy(_prev_r),
                    contact_counts=cp.asnumpy(_contact_counts),
                    contact_time=cp.asnumpy(_contact_time),
                    contact_total_steps=_contact_total_steps,
                    sm_current_state=(
                        cp.asnumpy(self._sm_current_state)
                        if self._sm_active
                        else np.zeros(0, dtype=np.int32)
                    ),
                    sm_rxn_fire_counts=(
                        np.asarray(self._sm_rxn_fire_counts, dtype=np.int64)
                        if self._sm_active
                        else np.zeros(0, dtype=np.int64)
                    ),
                    near_miss_pos=cp.asnumpy(_nm_pos),
                    near_miss_q=cp.asnumpy(_nm_q),
                    rad_counts=cp.asnumpy(_rad_counts),
                    rad_time=cp.asnumpy(_rad_time),
                    ang_counts=cp.asnumpy(_ang_counts) + _ang_host,
                    ang_time=cp.asnumpy(_ang_time),
                    ms_flux_out=cp.asnumpy(_ms_flux_out),
                    ms_flux_in=cp.asnumpy(_ms_flux_in),
                    trans_mat=cp.asnumpy(_trans_mat),
                    enc_pos=(np.vstack(_enc_pos) if _enc_pos else np.zeros((0, 3))),
                    enc_q=(np.vstack(_enc_q) if _enc_q else np.zeros((0, 4))),
                    enc_traj=np.asarray(_enc_traj, dtype=np.int64),
                    enc_step=np.asarray(_enc_step, dtype=np.int64),
                    enc_npairs=np.asarray(_enc_npairs, dtype=np.int64),
                    enc_bridge=np.asarray(_enc_bridge, dtype=np.int64),
                )
                _ckpt_tmp.replace(_ckpt_path)
                print(f" Checkpoint saved ({done:,}/{N:,} done) -> {_ckpt_path}")
            # Detailed diagnostics in verbose mode, printed every 1000 steps.
            if step % 1000 == 0:
                elapsed = time.time() - t0
                sps = total_steps / elapsed if elapsed > 0 else 0
                # Force statistics.
                f_mag = float(cp.linalg.norm(forces_gpu, axis=1).mean())
                f_max = float(cp.linalg.norm(forces_gpu, axis=1).max())
                f_min = float(cp.linalg.norm(forces_gpu, axis=1).min())
                # Position statistics.
                r_mean = float(r_sr.mean())
                r_min = float(r_sr.min())
                r_max = float(r_sr.max())
                # Timestep statistics.
                dt_mean = float(dt_arr.mean())
                dt_min = float(dt_arr.min())
                dt_max_ = float(dt_arr.max())
                # Drift and noise magnitude.
                drift_mag = float(cp.linalg.norm(drift, axis=1).mean())
                noise_mag = float((sigma_t[:, 0]).mean())
                # Count how many trajectories are near b versus far from b.
                n_near_b = int((~in_outer).sum())
                n_far = int(in_outer.sum())
                # Outer propagator events this step.
                n_at_esc = int(at_escape.sum()) if "at_escape" in dir() else 0
                print(
                    f"  step {step:6d} | done={done:6d}/{N} | "
                    f"reacted={n_reacted} escaped={n_escaped} | "
                    f"{sps:,.0f} steps/sec"
                )
                print(
                    f"    r: mean={r_mean:.2f} min={r_min:.2f} max={r_max:.2f} Å  "
                    f"(b={r_b:.1f}, esc={r_esc:.1f})"
                )
                print(
                    f"    |F|: mean={f_mag:.6f} max={f_max:.6f} min={f_min:.6f} kBT/Å"
                )
                print(
                    f"    dt: mean={dt_mean:.4f} min={dt_min:.4f} max={dt_max_:.4f} ps"
                )
                print(
                    f"    drift: {drift_mag:.6f} Å  noise(sigma): {noise_mag:.4f} Å  "
                    f"drift/noise={drift_mag/(noise_mag+1e-30):.4f}"
                )
                print(
                    f"    zone: {n_near_b} inner (r<=b) + {n_far} outer (r>b)  "
                    f"| at_escape: {n_at_esc}"
                )
                # For the first three steps, dump individual trajectory details.
                if step <= 2:
                    n_show = min(5, int(sr_mask.sum()))
                    print(f"    [TRACE] First {n_show} running trajectories:")
                    for ti in range(n_show):
                        _pos = cp.asnumpy(pos[sr_idx[ti]])
                        _r = float(cp.linalg.norm(pos[sr_idx[ti]]))
                        _f = cp.asnumpy(forces_gpu[ti])
                        _dt = float(dt_arr[ti])
                        _drft = cp.asnumpy(drift[ti])
                        print(
                            f"      traj[{int(sr_idx[ti])}]: pos=({_pos[0]:.4f},"
                            f"{_pos[1]:.4f},{_pos[2]:.4f}) r={_r:.4f}  "
                            f"F=({_f[0]:.6f},{_f[1]:.6f},{_f[2]:.6f})  "
                            f"|F|={np.linalg.norm(_f):.6f}  dt={_dt:.4f}  "
                            f"drift=({_drft[0]:.6f},{_drft[1]:.6f},{_drft[2]:.6f})"
                        )
        _min_dist = cp.asnumpy(_min_dist)
        _step_at_min = cp.asnumpy(_step_at_min)
        _total_time = cp.asnumpy(_total_time)
        _nm_pos = cp.asnumpy(_nm_pos)
        _nm_q = cp.asnumpy(_nm_q)
        _rad_counts = cp.asnumpy(_rad_counts)
        _ang_counts = cp.asnumpy(_ang_counts) + _ang_host
        _ms_flux_out = cp.asnumpy(_ms_flux_out)
        _ms_flux_in = cp.asnumpy(_ms_flux_in)
        _trans_mat = cp.asnumpy(_trans_mat)
        _contact_counts = cp.asnumpy(_contact_counts)
        _rad_time = cp.asnumpy(_rad_time) / _TICKS
        _ang_time = cp.asnumpy(_ang_time) / _TICKS
        _contact_time = cp.asnumpy(_contact_time) / _TICKS
        # With save_interval 0 the paths hold the start pose (step -1) and the
        # last pose (step n_steps - 1) of every trajectory.
        if _save_interval == 0 and getattr(_output_cfg, "full_paths", True):
            _ids = np.arange(N, dtype=np.float64)
            _path_data = [
                np.column_stack([_ids, np.full(N, -1.0), _start_pos, _start_q]),
                np.column_stack(
                    [_ids, cp.asnumpy(n_steps) - 1.0, cp.asnumpy(pos), cp.asnumpy(q)]
                ),
            ]
        # Any trajectory still running after the loop ends is counted as
        # max-steps.
        n_maxsteps = int((status == 0).sum())
        # Count the reacted and escaped trajectories from their final status.
        n_reacted = int((status == 1).sum())
        n_escaped = int((status == 2).sum())
        elapsed = time.time() - t0
        sps = total_steps / elapsed if elapsed > 0 else 0
        print(
            f"  BD complete: {n_reacted} reacted, {n_escaped} escaped, "
            f"{n_maxsteps} max-steps"
        )
        print(f"  Total steps: {total_steps:,}  ({sps:,.0f} steps/sec)")
        if _n_overlap_rejected > 0:
            print(f"  Overlap rejections: {_n_overlap_rejected:,}")
        # Final convergence report.
        # Trajectories at the step limit count as not reacted.
        _n_comp_final = n_reacted + n_escaped + n_maxsteps
        if _n_comp_final > 0 and self._k_b > 0:
            _p_final = n_reacted / _n_comp_final
            _k_final = _conv_CONV * self._k_b * _p_final
            print(f"  Final: P_rxn={_p_final:.6f}  k_on={_k_final:.4e} M-1 s-1")
        # Delete the checkpoint on successful completion.
        if _ckpt_interval > 0:
            if _ckpt_path.exists():
                _ckpt_path.unlink()
                print(f"  Checkpoint removed (run completed successfully)")
            (_ckpt_dir / "checkpoint.tmp.npz").unlink(missing_ok=True)
        # Package the collected data.
        _outcome = cp.asnumpy(status)
        # A trajectory still running when the loop ended reached the step limit.
        _outcome[_outcome == 0] = 3
        _nsteps_np = cp.asnumpy(n_steps)
        # Near-miss data is kept only for escaped trajectories.
        _esc_mask = _outcome == 2
        _nm_traj_arr = np.where(_esc_mask)[0]
        # Stack the encounter lists into arrays.
        if _enc_pos:
            _enc_pos_arr = np.vstack(_enc_pos)
            _enc_q_arr = np.vstack(_enc_q)
        else:
            _enc_pos_arr = np.zeros((0, 3))
            _enc_q_arr = np.zeros((0, 4))
        sim_data = {
            "outcome": _outcome,
            "n_steps": _nsteps_np,
            "start_pos": _start_pos,
            "start_q": _start_q,
            "min_dist": _min_dist,
            "step_at_min": _step_at_min,
            "total_time_ps": _total_time,
            "n_returns": _n_returns,
            "bb_triggered": _bb_triggered,
            "encounter_pos": _enc_pos_arr,
            "encounter_q": _enc_q_arr,
            "encounter_traj": np.array(_enc_traj, dtype=np.int64),
            "encounter_step": np.array(_enc_step, dtype=np.int64),
            "encounter_n_pairs": np.array(_enc_npairs, dtype=np.int64),
            "encounter_bridge_only": np.array(_enc_bridge, dtype=np.int64),
            "near_miss_pos": _nm_pos[_esc_mask],
            "near_miss_q": _nm_q[_esc_mask],
            "near_miss_traj": _nm_traj_arr,
            "near_miss_dist": _min_dist[_esc_mask],
            "path_steps": _path_data,
            "energy_steps": _energy_data,
            "radial_bins": _rad_bins,
            "radial_counts": _rad_counts,
            "radial_time_ps": _rad_time,
            # Bin centres. phi is the azimuth atan2(y, x), binned from -pi.
            "angular_theta": (np.arange(_n_theta) + 0.5) * np.pi / _n_theta,
            "angular_phi": (np.arange(_n_phi) + 0.5) * 2 * np.pi / _n_phi - np.pi,
            "angular_counts": _ang_counts,
            "angular_time_ps": _ang_time,
            "milestone_radii": _ms_radii,
            "milestone_flux_out": _ms_flux_out,
            "milestone_flux_in": _ms_flux_in,
            "contact_pair_counts": _contact_counts,
            "contact_pair_time_ps": _contact_time,
            "contact_total_steps": _contact_total_steps,
            "trans_bins": _trans_bins,
            "trans_matrix": _trans_mat,
        }
        # Report the fire-counter breakdown for debugging.
        if self._sm_active:
            print("  [DEBUG] Fire-counter breakdown:")
            for i, name in enumerate(self._sm_rxn_names):
                d = int(self._sm_rxn_fire_counts_discrete[i])
                b = int(self._sm_rxn_fire_counts_bridge[i])
                t = int(self._sm_rxn_fire_counts[i])
                u = len(self._sm_rxn_fired_traj_sets[i])
                print(
                    f"    {name}: total={t}  discrete={d}  bridge={b}  unique_trajs={u}"
                )
        # Per-reaction firing counts for state-machine mode. Each entry records
        # the reaction name, the state transition, and how many trajectories
        # fired that reaction. The list is omitted when state-machine mode was
        # not active.
        if self._sm_active and self._sm_rxn_fire_counts is not None:
            sim_data["completed_reactions"] = [
                {
                    "name": self._sm_rxn_names[i],
                    "n_fired": int(self._sm_rxn_fire_counts[i]),
                    "state_before": self._sm_rxn_state_before[i],
                    "state_after": self._sm_rxn_state_after[i],
                }
                for i in range(len(self._sm_rxn_names))
            ]

        return GPUBatchResult(
            n_trajectories=N,
            n_reacted=n_reacted,
            n_escaped=n_escaped,
            n_max_steps=n_maxsteps,
            reaction_counts={"stage_0": n_reacted},
            r_start=self.params.r_start,
            r_escape=self.params.r_escape,
            dt=float(getattr(self.params, "minimum_core_dt", 0.0)),
            elapsed_sec=elapsed,
            steps_per_sec=sps,
            sim_data=sim_data,
            total_steps=total_steps,
        )

    # GPU helper methods.
    def _random_quaternions_gpu(self, N: int, rng) -> "cp.ndarray":
        """Generate N random unit quaternions on the GPU."""
        u = rng.uniform(
            cp.arange(N, dtype=cp.int64), cp.zeros(N, dtype=cp.int64), 3, rng.ORIENT
        )
        q = cp.stack(
            [
                cp.sqrt(1 - u[:, 0]) * cp.sin(2 * cp.pi * u[:, 1]),
                cp.sqrt(1 - u[:, 0]) * cp.cos(2 * cp.pi * u[:, 1]),
                cp.sqrt(u[:, 0]) * cp.sin(2 * cp.pi * u[:, 2]),
                cp.sqrt(u[:, 0]) * cp.cos(2 * cp.pi * u[:, 2]),
            ],
            axis=1,
        )
        return q

    def _quats_to_rotmats(self, q: "cp.ndarray") -> "cp.ndarray":
        """Convert (N, 4) quaternions to (N, 3, 3) rotation matrices on the GPU."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        N = q.shape[0]
        R = cp.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            axis=1,
        ).reshape(N, 3, 3)
        return R

    def _apply_rotation_gpu(self, q: "cp.ndarray", omega: "cp.ndarray") -> "cp.ndarray":
        """Apply the rotation vector omega to the quaternions q and return the normalised result."""
        angle = cp.linalg.norm(omega, axis=1, keepdims=True)  # (N,1)
        safe = angle > 1e-10
        axis = cp.where(safe, omega / cp.where(safe, angle, 1.0), 0.0)
        half = angle * 0.5
        dq = cp.stack(
            [
                cp.cos(half[:, 0]),
                axis[:, 0] * cp.sin(half[:, 0]),
                axis[:, 1] * cp.sin(half[:, 0]),
                axis[:, 2] * cp.sin(half[:, 0]),
            ],
            axis=1,
        )  # (N,4)
        # Quaternion multiplication q_new = dq ⊗ q. The angular increment omega
        # (torque drift plus noise) is expressed in the lab frame, so the
        # increment is applied as a left product, matching the return-snap path
        # and the analytic Langevin equilibrium for a dipole in a field.
        w1, x1, y1, z1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        qnew = cp.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            axis=1,
        )
        # Normalise the result.
        qnew /= cp.linalg.norm(qnew, axis=1, keepdims=True)
        return qnew

    def _receptor_rotation_gpu(
        self,
        pos: "cp.ndarray",
        q: "cp.ndarray",
        lever: "cp.ndarray",
        forces: "cp.ndarray",
        torques: "cp.ndarray",
        dt_arr: "cp.ndarray",
        noise: "cp.ndarray",
        D_r1: float,
    ) -> Tuple["cp.ndarray", "cp.ndarray"]:
        """Apply the receptor's rotational step to (N, 3) positions and (N, 4) quaternions.

        The frame is fixed to the receptor, so the receptor's lab-frame angular
        displacement omega1 turns the whole relative configuration by -omega1
        about the origin: the separation vector and the ligand orientation
        together, by the same rotation. omega1 = D_r1 dt tau1 + sqrt(2 D_r1 dt)
        noise, with the torque on the receptor about the origin given by
        Newton's third law, tau1 = -(tau2 + r x F), and r the pre-step centroid
        (lever) where the force was evaluated. This is the batched counterpart
        of do_bd_step.receptor_rotation_vector and apply_frame_rotation.
        """
        torque_rec = -(torques + cp.cross(lever, forces))
        sigma = cp.sqrt(2.0 * D_r1 * dt_arr)[:, None]
        omega1 = D_r1 * dt_arr[:, None] * torque_rec + sigma * noise
        return (
            self._rotate_vectors_gpu(pos, -omega1),
            self._apply_rotation_gpu(q, -omega1),
        )

    def _bridge_D_dt(
        self,
        D_par_arr: "cp.ndarray",
        dt_arr: "cp.ndarray",
        D_t: float,
        D_rot: "cp.ndarray" = None,
    ) -> "cp.ndarray":
        """D times dt of each reaction pair distance over one step for the bridge.

        The translational part is the D the step moved the pair with. With
        hydrodynamic interactions that is D_par at the separation at the start
        of the step (D_par_arr per trajectory). Without them it is the far field
        D_trans. D_rot is the rotation part from _bridge_D_rot with shape
        (N, n_pairs). Returns shape (N, n_pairs) or (N, 1) when D_rot is None.
        """
        D = D_par_arr[:, None] if self._use_hi else D_t
        if D_rot is None:
            return D * dt_arr[:, None]
        return (D + D_rot) * dt_arr[:, None]

    def _bridge_D_rot(
        self,
        lig_gho: "cp.ndarray",
        centroids: "cp.ndarray",
        pair_dists: "cp.ndarray",
        D_r1: float,
        D_r2: float,
    ) -> "cp.ndarray":
        """Rotation part of the diffusion coefficient of each pair distance.

        lig_gho holds the (N, n_pairs, 3) ligand pair atoms and centroids the
        (N, 3) ligand centroids at the start of the step. pair_dists holds the
        (N, n_pairs) pair distances of the same pose. The step turns the ligand
        about its centroid by omega2 with D_r2. It turns the relative
        configuration about the origin by -omega1 with D_r1. To first order
        the two turns move the pair vector d by
            omega2 × R g_lig - omega1 × (g_rec + d)
        with R g_lig the ligand pair atom from the ligand centroid and g_rec
        the receptor pair atom from the origin. The term omega1 × d stands at
        right angles to d and leaves the distance unchanged. With s = d / |d|
        the distance changes by omega2 · (R g_lig × s) - omega1 · (g_rec × s).
        The noise in omega1 and omega2 is independent with variance 2 D_r1 dt
        and 2 D_r2 dt per axis. So the pair distance diffuses with
            D_r1 |g_rec × s|² + D_r2 |R g_lig × s|²
        on top of the translation. A lever arm along the pair adds nothing. A
        lever arm at right angles to the pair adds D_r L². No lever arm of
        length L adds more. Returns shape (N, n_pairs).
        """
        rec = cp.broadcast_to(self._rec_gho_pos, lig_gho.shape)
        # A pair at zero distance has no direction and gets no rotation part.
        s = (lig_gho - rec) / cp.maximum(pair_dists, 1e-12)[:, :, None]
        rec_x = cp.cross(rec, s)
        lig_x = cp.cross(lig_gho - centroids[:, None, :], s)
        return D_r1 * cp.sum(rec_x * rec_x, axis=2) + D_r2 * cp.sum(
            lig_x * lig_x, axis=2
        )

    def _rotate_vectors_gpu(self, v: "cp.ndarray", omega: "cp.ndarray") -> "cp.ndarray":
        """Rotate the (N, 3) vectors v by the (N, 3) rotation vectors omega.

        This is the vector counterpart of _apply_rotation_gpu: the rotation
        that omega applies to a quaternion from the left is applied here to a
        lab-frame vector, by the Rodrigues formula
        v cos θ + (k × v) sin θ + k (k · v)(1 - cos θ) with k the unit axis
        and θ the angle. Rotations below 1e-10 rad leave v unchanged.
        """
        angle = cp.linalg.norm(omega, axis=1, keepdims=True)  # (N,1)
        safe = angle > 1e-10
        k = cp.where(safe, omega / cp.where(safe, angle, 1.0), 0.0)
        cos_t = cp.cos(angle)
        sin_t = cp.sin(angle)
        k_dot_v = cp.sum(k * v, axis=1, keepdims=True)
        return v * cos_t + cp.cross(k, v) * sin_t + k * k_dot_v * (1.0 - cos_t)

    def _check_reactions_gpu_gho_state_machine(
        self,
        pos_run: "cp.ndarray",
        q_run: "cp.ndarray",
        current_state_run: "cp.ndarray",
    ) -> "cp.ndarray":
        """
        State-aware reaction check.

        For each running trajectory this determines which reaction, if any,
        fires this step. A reaction r fires on trajectory t when the trajectory
        is in the right state, current_state_run[t] == state_before_idx[r], and
        when the GHO-GHO distance for reaction r's pair is below cutoff[r]. If
        several reactions satisfy both conditions for the same trajectory, the
        lowest-indexed reaction wins, following the reaction order in rxns.xml.

        Parameters
        ----------
        pos_run            : (N_run, 3)   ligand centroid positions.
        q_run              : (N_run, 4)   ligand orientation quaternions.
        current_state_run  : (N_run,)     int32 current state per trajectory.

        Returns
        -------
        fired_rxn_idx : (N_run,) int32
            The index of the reaction that fired, or -1 if none fired.
        """
        N_run = pos_run.shape[0]
        n_rxn = len(self._sm_rxn_names)
        if n_rxn == 0 or N_run == 0:
            return cp.full(N_run, -1, dtype=cp.int32)
        # Gather the ligand-frame GHO positions for each reaction, shape
        # (n_rxn, 3).
        lig_gho_local = self._mol2_pos0[self._sm_lig_gho_indices_cached]
        # Rotation matrices from the quaternions, shape (N_run, 3, 3).
        R = self._quats_to_rotmats(q_run)
        # Rotated and translated ligand GHO per reaction, shape (N_run, n_rxn,
        # 3).
        lig_gho_lab = cp.einsum("nij,kj->nki", R, lig_gho_local) + pos_run[:, None, :]
        # Distance from each trajectory's per-reaction ligand GHO to the
        # corresponding fixed receptor GHO, shape (N_run, n_rxn).
        rec_gho = self._sm_rec_gho_pos_gpu[None, :, :]
        dists = cp.linalg.norm(lig_gho_lab - rec_gho, axis=2)
        # Gate on both the cutoff and the state match.
        cutoff_ok = dists < self._sm_rxn_cutoffs_gpu[None, :]
        state_ok = (
            current_state_run[:, None] == self._sm_rxn_state_before_idx_gpu[None, :]
        )
        fire_mask = cutoff_ok & state_ok  # (N_run, n_rxn)
        # Take the lowest-index reaction that fires per trajectory, or -1 if
        # none fires.
        any_fire = fire_mask.any(axis=1)
        fired_rxn_idx = cp.argmax(fire_mask, axis=1).astype(cp.int32)
        fired_rxn_idx = cp.where(any_fire, fired_rxn_idx, -1)
        return fired_rxn_idx
