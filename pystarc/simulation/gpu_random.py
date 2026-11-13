"""
Random numbers of the GPU Brownian-dynamics loop, drawn on the GPU.

Every number comes from the counter-based generator Philox4x32-10 (Salmon,
Moraes, Dror and Shaw, SC11 2011; the generator of Random123 and cuRAND). The
key is a hash of the run seed. The counter holds the trajectory id, the step
count of that trajectory, the stream (one per draw statement of the simulator)
and the index of the draw. A trajectory's numbers therefore depend only on the
seed, its id and its own step count, not on the slot it occupies, the other
trajectories running beside it, how the work is batched or whether the run
was resumed from a checkpoint.

One Philox call gives 128 bits, read as two uniforms of 53 bits on [0, 1), the
resolution of NumPy's random(). Standard normals come from the Box-Muller
transform of two such uniforms, z = sqrt(-2 ln u1) (cos 2 pi u2, sin 2 pi u2),
with u1 on (0, 1]. Everything is in float64. DEVICE_CODE holds the device
functions so that other kernels can draw the same numbers.
"""

from __future__ import annotations

import numpy as np

from pystarc.simulation.diffusional_rotation import (
    _PHIS_ARR,
    _RPROB_0P5_ARR,
    _RPROB_1P0_ARR,
    _RPROB_2P0_ARR,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover - CuPy is optional
    cp = None

DEVICE_CODE = r"""
// Philox4x32-10 (Salmon et al. 2011), as in Random123 and cuRAND.
__device__ uint4 philox(uint4 c, uint2 k)
{
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        const unsigned int lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
        const unsigned int lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// Counter of draw j of a stream, for one trajectory at its step count.
__device__ uint4 philox_counter(long long traj, long long step, unsigned int stream,
                                unsigned int j)
{
    const unsigned long long s = (unsigned long long)step;
    return make_uint4(j | (stream << 16), (unsigned int)traj, (unsigned int)s,
                      (unsigned int)(s >> 32));
}

// The top 53 bits of two words, an integer in [0, 2^53).
__device__ double bits53(unsigned int a, unsigned int b)
{
    return (double)((((unsigned long long)a << 32) | b) >> 11);
}

// Uniform on [0, 1).
__device__ double u53(unsigned int a, unsigned int b)
{
    return bits53(a, b) / 9007199254740992.0;
}

// Two standard normals by Box-Muller; u1 is on (0, 1] so that the log is finite.
__device__ void box_muller(uint4 r, double* z0, double* z1)
{
    const double u1 = (bits53(r.x, r.y) + 1.0) / 9007199254740992.0;
    const double u2 = u53(r.z, r.w);
    const double rad = sqrt(-2.0 * log(u1));
    double s, c;
    sincospi(2.0 * u2, &s, &c);
    *z0 = rad * c;
    *z1 = rad * s;
}

// Draws taken in turn from one counter, for a number of draws that varies.
struct Draws {
    uint4 c;
    uint2 k;
};

__device__ uint4 next_block(Draws* d)
{
    const uint4 r = philox(d->c, d->k);
    d->c.x += 1;
    return r;
}

__device__ void normals3(Draws* d, double* z)
{
    double spare;
    box_muller(next_block(d), &z[0], &z[1]);
    box_muller(next_block(d), &z[2], &spare);
}

// q = a b, quaternions in (w, x, y, z) order.
__device__ void quat_mul(const double* a, const double* b, double* q)
{
    q[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    q[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    q[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    q[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

// The rotation of diffusional_rotation.py at tau <= 0.25: omega = sqrt(2 tau) xi.
__device__ void gauss_rotation(Draws* d, double tau, double* q)
{
    double w[3];
    normals3(d, w);
    const double sq = sqrt(2.0 * tau);
    for (int a = 0; a < 3; ++a) w[a] *= sq;
    const double angle = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (angle < 1e-12) {
        q[0] = 1.0;
        q[1] = q[2] = q[3] = 0.0;
        return;
    }
    const double sh = sin(0.5 * angle);
    q[0] = cos(0.5 * angle);
    for (int a = 0; a < 3; ++a) q[a + 1] = w[a] / angle * sh;
}

// A rotation whose angle is drawn from a tabulated CDF (np.interp) about a
// random axis, as _spline_rot_from_table.
__device__ void table_rotation(Draws* d, const double* phis, const double* rprob,
                               int nt, double* q)
{
    const uint4 r = next_block(d);
    const double pc = u53(r.x, r.y);
    int lo = 0, hi = nt - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (rprob[mid] <= pc) lo = mid;
        else hi = mid;
    }
    double phi = phis[lo];
    if (pc != rprob[lo]) {
        const double slope = (phis[lo + 1] - phis[lo]) / (rprob[lo + 1] - rprob[lo]);
        phi = slope * (pc - rprob[lo]) + phis[lo];
    }
    double axis[3];
    normals3(d, axis);
    const double norm = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (norm < 1e-14) {
        q[0] = 1.0;
        q[1] = q[2] = q[3] = 0.0;
        return;
    }
    const double sh = sin(phi / 2.0);
    q[0] = cos(phi / 2.0);
    for (int a = 0; a < 3; ++a) q[a + 1] = axis[a] / norm * sh;
}

// diffusional_rotation(rng, tau) of diffusional_rotation.py: the recursion over
// the checkpoints at tau = 2, 1, 0.5 and 0.25 written as a loop, the rotation
// of each checkpoint drawn first. tab holds phis and the CDFs at 0.5, 1 and 2.
__device__ void diffusional_rotation(Draws* d, double tau, const double* tab, int nt,
                                     double* q)
{
    double dq[4], t[4];
    q[0] = 1.0;
    q[1] = q[2] = q[3] = 0.0;
    if (tau >= 4.0) {
        box_muller(next_block(d), &q[0], &q[1]);
        box_muller(next_block(d), &q[2], &q[3]);
        const double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (int a = 0; a < 4; ++a) q[a] /= norm;
        return;
    }
    while (tau > 0.0) {
        if (tau <= 0.25) {
            gauss_rotation(d, tau, dq);
            tau = 0.0;
        } else if (tau < 0.5) {
            gauss_rotation(d, 0.25, dq);
            tau -= 0.25;
        } else if (tau < 1.0) {
            table_rotation(d, tab, tab + nt, nt, dq);
            tau -= 0.5;
        } else if (tau < 2.0) {
            table_rotation(d, tab, tab + 2 * nt, nt, dq);
            tau -= 1.0;
        } else {
            table_rotation(d, tab, tab + 3 * nt, nt, dq);
            tau -= 2.0;
        }
        quat_mul(dq, q, t);
        for (int a = 0; a < 4; ++a) q[a] = t[a];
    }
}
"""

_KERNEL_CODE = r"""
extern "C" __global__
void draw_uniform(const long long* traj, const long long* n_steps, double* out,
                  const long long rows, const int n, const unsigned int stream,
                  const unsigned int k0, const unsigned int k1)
{
    const int nb = (n + 1) / 2;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= rows * nb) return;
    const long long i = t / nb;
    const int j = (int)(t - i * nb);
    const long long id = traj[i];
    const uint4 c = philox_counter(id, n_steps[id], stream, j);
    const uint4 r = philox(c, make_uint2(k0, k1));
    out[i * n + 2 * j] = u53(r.x, r.y);
    if (2 * j + 1 < n) out[i * n + 2 * j + 1] = u53(r.z, r.w);
}

extern "C" __global__
void draw_normal(const long long* traj, const long long* n_steps, double* out,
                 const long long rows, const int n, const unsigned int stream,
                 const unsigned int k0, const unsigned int k1)
{
    const int nb = (n + 1) / 2;
    const long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= rows * nb) return;
    const long long i = t / nb;
    const int j = (int)(t - i * nb);
    const long long id = traj[i];
    const uint4 c = philox_counter(id, n_steps[id], stream, j);
    const uint4 r = philox(c, make_uint2(k0, k1));
    double z0, z1;
    box_muller(r, &z0, &z1);
    out[i * n + 2 * j] = z0;
    if (2 * j + 1 < n) out[i * n + 2 * j + 1] = z1;
}

// _return_rotation_host for one trajectory per thread: the ligand turns by
// dq1, then the relative configuration by dq0^-1 about the origin.
extern "C" __global__
void return_rotation(const double* q_in, const double* p_in, const long long* traj,
                     const long long* n_steps, double* q_out, double* p_out,
                     const long long rows, const double tau_lig, const double tau_rec,
                     const double* tab, const int nt, const unsigned int stream,
                     const unsigned int k0, const unsigned int k1)
{
    const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rows) return;
    const long long id = traj[i];
    Draws d = {philox_counter(id, n_steps[id], stream, 0), make_uint2(k0, k1)};
    double q[4], p[3], dq[4], t[4];
    for (int a = 0; a < 4; ++a) q[a] = q_in[4 * i + a];
    for (int a = 0; a < 3; ++a) p[a] = p_in[3 * i + a];
    if (tau_lig > 0.0) {
        diffusional_rotation(&d, tau_lig, tab, nt, dq);
        quat_mul(dq, q, t);
        for (int a = 0; a < 4; ++a) q[a] = t[a];
    }
    if (tau_rec > 0.0) {
        diffusional_rotation(&d, tau_rec, tab, nt, dq);
        const double w = dq[0], x = -dq[1], y = -dq[2], z = -dq[3];
        const double inv[4] = {w, x, y, z};
        quat_mul(inv, q, t);
        for (int a = 0; a < 4; ++a) q[a] = t[a];
        const double R[9] = {
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
        double v[3];
        for (int a = 0; a < 3; ++a)
            v[a] = R[3 * a] * p[0] + R[3 * a + 1] * p[1] + R[3 * a + 2] * p[2];
        for (int a = 0; a < 3; ++a) p[a] = v[a];
    }
    const double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int a = 0; a < 4; ++a) q_out[4 * i + a] = q[a] / norm;
    for (int a = 0; a < 3; ++a) p_out[3 * i + a] = p[a];
}
"""


class GPURandom:
    """Counter-based random numbers for the trajectories of one run.

    The draws take traj, the ids of the drawing trajectories, and n_steps, the
    step count of every trajectory indexed by id, and return one row per
    trajectory. Each draw statement of the simulator has its own stream.
    """

    START, ORIENT, ESCAPE, RETURN, NOISE_T, NOISE_R2, NOISE_R1, BRIDGE = range(8)

    def __init__(self, seed):
        # A 64-bit key hashed from the seed; seed None takes fresh entropy.
        self.key = np.random.SeedSequence(seed).generate_state(2)
        mod = cp.RawModule(code=DEVICE_CODE + _KERNEL_CODE)
        self._uniform = mod.get_function("draw_uniform")
        self._normal = mod.get_function("draw_normal")
        self._return = mod.get_function("return_rotation")
        self._tables = cp.asarray(
            np.concatenate([_PHIS_ARR, _RPROB_0P5_ARR, _RPROB_1P0_ARR, _RPROB_2P0_ARR])
        )

    def _draw(self, kernel, traj, n_steps, n, stream):
        traj = cp.ascontiguousarray(traj, dtype=cp.int64)
        n_steps = cp.ascontiguousarray(n_steps, dtype=cp.int64)
        out = cp.empty((len(traj), n), dtype=cp.float64)
        total = len(traj) * ((n + 1) // 2)
        if total:
            kernel(
                ((total + 127) // 128,),
                (128,),
                (
                    traj, n_steps, out, np.int64(len(traj)), np.int32(n),
                    np.uint32(stream), self.key[0], self.key[1],
                ),
            )
        return out

    def uniform(self, traj, n_steps, n, stream):
        """(len(traj), n) uniforms on [0, 1)."""
        return self._draw(self._uniform, traj, n_steps, n, stream)

    def normal(self, traj, n_steps, n, stream):
        """(len(traj), n) standard normals."""
        return self._draw(self._normal, traj, n_steps, n, stream)

    def return_rotation(self, q, pos, traj, n_steps, tau_lig, tau_rec):
        """_return_rotation_host on the GPU for the trajectories traj.

        q is (n, 4) and pos (n, 3). The ligand's rotation is drawn before the
        receptor's, both from stream RETURN. Returns the new q and pos.
        """
        q = cp.ascontiguousarray(q, dtype=cp.float64)
        pos = cp.ascontiguousarray(pos, dtype=cp.float64)
        traj = cp.ascontiguousarray(traj, dtype=cp.int64)
        n_steps = cp.ascontiguousarray(n_steps, dtype=cp.int64)
        q_new = cp.empty_like(q)
        pos_new = cp.empty_like(pos)
        n = len(traj)
        if n:
            self._return(
                ((n + 127) // 128,),
                (128,),
                (
                    q, pos, traj, n_steps, q_new, pos_new, np.int64(n),
                    np.float64(tau_lig), np.float64(tau_rec), self._tables,
                    np.int32(len(_PHIS_ARR)), np.uint32(self.RETURN),
                    self.key[0], self.key[1],
                ),
            )
        return q_new, pos_new
