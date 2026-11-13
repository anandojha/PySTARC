"""
GPU force evaluation engine.

This module computes the total force on each ligand atom from the receptor's
electrostatic field. Three contributions are summed.

The first contribution is the electrostatic (Poisson-Boltzmann) force,

    F_elec = -q_i × ∇φ_rec(r_i).

Here q_i is the charge of ligand atom i and φ_rec is the receptor's
electrostatic potential, which is precomputed by APBS on a 3D grid. The
gradient ∇φ is evaluated by trilinear interpolation with central differences
taken at half the grid spacing. For atoms outside the grid (using a
three-spacing safety margin), the Yukawa multipole far-field is used instead.

The second contribution is the Born desolvation force,

    F_born = -α × q_i² × ∇φ_born(r_i).

When a charged atom approaches the receptor it partially displaces the
high-dielectric solvent and pays an energetic penalty. The Born potential
φ_born is the vacuum electrostatic potential (ε = 1 everywhere, no ions) and
α = 1/(4π) ≈ 0.0796. This term is computed in both directions unless
enable_born2_torque is false. Direction 1 evaluates the receptor Born grid at
the ligand atom positions. Direction 2 evaluates the ligand Born grid at the
receptor atom positions.

The third contribution is the Yukawa monopole fallback. For atoms far from the
receptor (outside the APBS grid), the screened Coulomb force gives the correct
long-range behavior,

    F = q_rec × q_lig / (4πε) × (1/r² + 1/(rλ)) × exp(-r/λ) × r̂,

where λ is the Debye screening length.

For large ligands the simulator evaluates the forces in batches of

    N_batch = 4 GB / (N_lig_atoms × 150 bytes)

trajectories, which keeps every index of the force kernels below 2^31.
"""

from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import numpy as np
import math

from pystarc.global_defs.constants import VACUUM_PERMITTIVITY_KBT

try:
    import cupy as cp

    _CUPY = True
except ImportError:
    _CUPY = False


# Grid forces, one warp per trajectory and one lane per atom. Each atom goes to
# the finest grid whose usable box holds it, or to the far field. The warp sums
# in the order CuPy uses over the atom axis (16 partial sums, then a halving
# tree), and the grids are added in order.

_BATCH_KERNEL_CODE = r"""
__device__ int pick_grid(const double* box, int n_grid, double x, double y, double z)
{
    for (int g = 0; g < n_grid; g++) {
        const double* b = box + 15 * g;
        if (x > b[9] && y > b[10] && z > b[11] && x < b[12] && y < b[13] && z < b[14])
            return g;
    }
    return n_grid;
}

__device__ void grid_term(
    const double* grid_data, const double* b, const int* n,
    double px, double py, double pz, double q, double alpha, int is_born,
    double* f, double* e)
{
    const double* origin = b;
    const double* inv_sp = b + 3;
    const double* sp = b + 6;
    int nx = n[0], ny = n[1], nz = n[2];
    auto interp = [&](double x, double y, double z) -> double {
        double ix = (x - origin[0]) * inv_sp[0];
        double iy = (y - origin[1]) * inv_sp[1];
        double iz = (z - origin[2]) * inv_sp[2];
        int i0=(int)floor(ix), j0=(int)floor(iy), k0=(int)floor(iz);
        if (i0<0||i0>=nx-1||j0<0||j0>=ny-1||k0<0||k0>=nz-1) return 0.0;
        double fx=ix-i0, fy=iy-j0, fz=iz-k0;
        return (grid_data[(i0  )*ny*nz+(j0  )*nz+(k0  )]*(1-fx)*(1-fy)*(1-fz) +
                grid_data[(i0+1)*ny*nz+(j0  )*nz+(k0  )]*fx    *(1-fy)*(1-fz) +
                grid_data[(i0  )*ny*nz+(j0+1)*nz+(k0  )]*(1-fx)*fy    *(1-fz) +
                grid_data[(i0  )*ny*nz+(j0  )*nz+(k0+1)]*(1-fx)*(1-fy)*fz     +
                grid_data[(i0+1)*ny*nz+(j0+1)*nz+(k0  )]*fx    *fy    *(1-fz) +
                grid_data[(i0+1)*ny*nz+(j0  )*nz+(k0+1)]*fx    *(1-fy)*fz     +
                grid_data[(i0  )*ny*nz+(j0+1)*nz+(k0+1)]*(1-fx)*fy    *fz     +
                grid_data[(i0+1)*ny*nz+(j0+1)*nz+(k0+1)]*fx    *fy    *fz);
    };
    double h0=sp[0]*0.5, h1=sp[1]*0.5, h2=sp[2]*0.5;
    double gx = (interp(px+h0,py,pz) - interp(px-h0,py,pz)) / (2*h0);
    double gy = (interp(px,py+h1,pz) - interp(px,py-h1,pz)) / (2*h1);
    double gz = (interp(px,py,pz+h2) - interp(px,py,pz-h2)) / (2*h2);
    double phi = interp(px, py, pz);
    double coeff   = is_born ? (-alpha*q*q) : (-q);
    double e_coeff = is_born ? ( alpha*q*q) : ( q );
    f[0] = __dmul_rn(coeff, gx);
    f[1] = __dmul_rn(coeff, gy);
    f[2] = __dmul_rn(coeff, gz);
    *e = __dmul_rn(e_coeff, phi);
}

// Yukawa monopole with the dipole, quadrupole and screened trace terms, as in
// _yukawa_forces_gpu. c holds V_factor, debye, 4 pi eps, tr(M)/(6 debye^2),
// the dipole, the quadrupole and the three on/off flags.
__device__ void far_term(const double* c, double x, double y, double z, double q,
                         double* f, double* e)
{
    double V = c[0], lam = c[1], fpe = c[2];
    const double* p = c + 4;
    const double* Q = c + 7;
    double sr = fmax(sqrt(x*x + y*y + z*z), 1.0);
    double ex = exp(-sr / lam);
    double hx = x / sr, hy = y / sr, hz = z / sr;
    double phi = V * ex / sr;
    double dphi = V * ex * (-1.0 / (sr*sr) - 1.0 / (sr*lam));
    double tdx = 0.0, tdy = 0.0, tdz = 0.0, tqx = 0.0, tqy = 0.0, tqz = 0.0;
    if (c[16] != 0.0) {
        double pr = hx*p[0] + hy*p[1] + hz*p[2];
        double a = 1.0 + sr/lam;
        phi += pr / (fpe*sr*sr) * a * ex;
        dphi += pr / fpe * ex * (-2.0/(sr*sr*sr) - 2.0/(sr*sr*lam) - 1.0/(sr*lam*lam));
        double w = a * ex / (fpe*sr*sr*sr);
        tdx = w * (p[0] - pr*hx);
        tdy = w * (p[1] - pr*hy);
        tdz = w * (p[2] - pr*hz);
    }
    if (c[17] != 0.0) {
        double Qx = Q[0]*hx + Q[1]*hy + Q[2]*hz;
        double Qy = Q[3]*hx + Q[4]*hy + Q[5]*hz;
        double Qz = Q[6]*hx + Q[7]*hy + Q[8]*hz;
        double rQr = hx*Qx + hy*Qy + hz*Qz;
        double b = 1.0 + sr/lam + sr*sr/(3.0*lam*lam);
        phi += rQr / (fpe*sr*sr*sr) * b * ex;
        dphi += rQr / fpe * ex * (-3.0/(sr*sr*sr*sr) - 3.0/(sr*sr*sr*lam)
                                  - 4.0/(3.0*sr*sr*lam*lam) - 1.0/(3.0*sr*lam*lam*lam));
        double w = 2.0 * b * ex / (fpe*sr*sr*sr*sr);
        tqx = w * (Qx - rQr*hx);
        tqy = w * (Qy - rQr*hy);
        tqz = w * (Qz - rQr*hz);
    }
    if (c[18] != 0.0) {
        phi += c[3] / (fpe*sr) * ex;
        dphi += c[3] / fpe * ex * (-1.0/(sr*sr) - 1.0/(sr*lam));
    }
    f[0] = __dmul_rn(-q, dphi*hx + tdx + tqx);
    f[1] = __dmul_rn(-q, dphi*hy + tdy + tqy);
    f[2] = __dmul_rn(-q, dphi*hz + tdz + tqz);
    *e = __dmul_rn(q, phi);
}

extern "C" __global__
void batch_force_kernel(
    const double* __restrict__ pos, long long s0, long long s1, long long s2,
    const double* __restrict__ charges,
    const double* __restrict__ centroids, int has_centroids,
    const double* __restrict__ grid_values,
    const long long* __restrict__ grid_start,
    const double* __restrict__ grid_box,
    const int* __restrict__ grid_n,
    int n_grid,
    const double* __restrict__ far, int has_far,
    double alpha, int is_born,
    int N_traj, int N_atoms,
    double* __restrict__ force,
    double* __restrict__ torque,
    double* __restrict__ energy)
{
    int traj = blockIdx.x * blockDim.y + threadIdx.y;
    if (traj >= N_traj) return;
    int lane = threadIdx.x;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    if (has_centroids) {
        cx = centroids[3*traj]; cy = centroids[3*traj+1]; cz = centroids[3*traj+2];
    }
    double total[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int g = 0; g < n_grid + has_far; g++) {
        double s[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int a0 = 0; a0 < N_atoms; a0 += 32) {
            double c[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            int atom = a0 + lane;
            if (atom < N_atoms) {
                const double* p = pos + traj*s0 + atom*s1;
                double px = p[0], py = p[s2], pz = p[2*s2];
                double q = charges[atom];
                if (pick_grid(grid_box, n_grid, px, py, pz) == g
                        && (g == n_grid || fabs(q) >= 1e-9)) {
                    if (g < n_grid)
                        grid_term(grid_values + grid_start[g], grid_box + 15*g, grid_n + 3*g,
                                  px, py, pz, q, alpha, is_born, c, c + 6);
                    else
                        far_term(far, px, py, pz, q, c, c + 6);
                    if (has_centroids) {
                        double lx = px - cx, ly = py - cy, lz = pz - cz;
                        c[3] = __dsub_rn(__dmul_rn(ly, c[2]), __dmul_rn(lz, c[1]));
                        c[4] = __dsub_rn(__dmul_rn(lz, c[0]), __dmul_rn(lx, c[2]));
                        c[5] = __dsub_rn(__dmul_rn(lx, c[1]), __dmul_rn(ly, c[0]));
                    }
                }
            }
            for (int i = 0; i < 7; i++)
                s[i] = (s[i] + c[i]) + __shfl_down_sync(0xffffffffu, c[i], 16);
        }
        for (int o = 8; o > 0; o >>= 1)
            for (int i = 0; i < 7; i++)
                s[i] += __shfl_down_sync(0xffffffffu, s[i], o);
        for (int i = 0; i < 7; i++)
            total[i] += s[i];
    }
    if (lane == 0) {
        for (int i = 0; i < 3; i++) {
            force[3*traj + i] = total[i];
            torque[3*traj + i] = total[3 + i];
        }
        energy[traj] = total[6];
    }
}
"""


# Born direction 2 with one block of threads per trajectory, on the tables of
# the uploaded ligand Born grids (_GridList).

_BORN2_KERNEL_CODE = r"""
// Receptor atom in the ligand frame, p = R^T d, with the rounding of a gemm
// k loop, and the finest grid whose usable box holds it (-1 for none).
__device__ int born2_locate(const double* R, const double* par, int n_grids,
                            double dx, double dy, double dz, double* p)
{
    p[0] = fma(R[6], dz, fma(R[3], dy, R[0]*dx));
    p[1] = fma(R[7], dz, fma(R[4], dy, R[1]*dx));
    p[2] = fma(R[8], dz, fma(R[5], dy, R[2]*dx));
    for (int g = 0; g < n_grids; g++) {
        const double* lo = par + 15*g + 9;
        const double* hi = lo + 3;
        if (p[0] > lo[0] && p[1] > lo[1] && p[2] > lo[2] &&
            p[0] < hi[0] && p[1] < hi[1] && p[2] < hi[2]) return g;
    }
    return -1;
}

// Force -alpha q^2 grad phi at p, as in batch_force_kernel.
__device__ void born2_force(const double* grid_data, const double* origin,
                            const double* inv_sp, const double* sp,
                            int nx, int ny, int nz, double alpha, double q,
                            const double* p, double* f)
{
    double px = p[0], py = p[1], pz = p[2];
    auto interp = [&](double x, double y, double z) -> double {
        double ix = (x - origin[0]) * inv_sp[0];
        double iy = (y - origin[1]) * inv_sp[1];
        double iz = (z - origin[2]) * inv_sp[2];
        int i0=(int)floor(ix), j0=(int)floor(iy), k0=(int)floor(iz);
        if (i0<0||i0>=nx-1||j0<0||j0>=ny-1||k0<0||k0>=nz-1) return 0.0;
        double fx=ix-i0, fy=iy-j0, fz=iz-k0;
        return (grid_data[(i0  )*ny*nz+(j0  )*nz+(k0  )]*(1-fx)*(1-fy)*(1-fz) +
                grid_data[(i0+1)*ny*nz+(j0  )*nz+(k0  )]*fx    *(1-fy)*(1-fz) +
                grid_data[(i0  )*ny*nz+(j0+1)*nz+(k0  )]*(1-fx)*fy    *(1-fz) +
                grid_data[(i0  )*ny*nz+(j0  )*nz+(k0+1)]*(1-fx)*(1-fy)*fz     +
                grid_data[(i0+1)*ny*nz+(j0+1)*nz+(k0  )]*fx    *fy    *(1-fz) +
                grid_data[(i0+1)*ny*nz+(j0  )*nz+(k0+1)]*fx    *(1-fy)*fz     +
                grid_data[(i0  )*ny*nz+(j0+1)*nz+(k0+1)]*(1-fx)*fy    *fz     +
                grid_data[(i0+1)*ny*nz+(j0+1)*nz+(k0+1)]*fx    *fy    *fz);
    };
    double h0=sp[0]*0.5, h1=sp[1]*0.5, h2=sp[2]*0.5;
    double gx = (interp(px+h0,py,pz) - interp(px-h0,py,pz)) / (2*h0);
    double gy = (interp(px,py+h1,pz) - interp(px,py-h1,pz)) / (2*h1);
    double gz = (interp(px,py,pz+h2) - interp(px,py,pz-h2)) / (2*h2);
    double coeff = -alpha*q*q;
    f[0] = coeff * gx;
    f[1] = coeff * gy;
    f[2] = coeff * gz;
}

extern "C" __global__
void born2_kernel(
    const double* __restrict__ cen,
    const double* __restrict__ rot,
    const double* __restrict__ rec,
    const double* __restrict__ chg,
    const double* __restrict__ data,
    const double* __restrict__ par,
    const int* __restrict__ dims,
    const long long* __restrict__ start,
    int n_grids, int n_rec,
    double alpha, double rec_r, double reach,
    double* __restrict__ force,
    double* __restrict__ torque)
{
    extern __shared__ double red[];
    int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    double cx = cen[3*t], cy = cen[3*t+1], cz = cen[3*t+2];
    // |x - c| >= |c| - rec_r for every receptor atom x.
    if (sqrt(cx*cx + cy*cy + cz*cz) - rec_r > reach) {
        if (tid < 3) force[3*t+tid] = torque[3*t+tid] = 0.0;
        return;
    }
    const double* R = rot + 9*t;
    double s[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int a = tid; a < n_rec; a += nt) {
        double q = chg[a];
        if (fabs(q) < 1e-9) continue;
        double dx = rec[3*a] - cx, dy = rec[3*a+1] - cy, dz = rec[3*a+2] - cz;
        if (dx*dx + dy*dy + dz*dz >= reach*reach) continue;
        double p[3], f[3];
        int g = born2_locate(R, par, n_grids, dx, dy, dz, p);
        if (g < 0) continue;
        const double* o = par + 15*g;
        born2_force(data + start[g], o, o + 3, o + 6,
                    dims[3*g], dims[3*g+1], dims[3*g+2], alpha, q, p, f);
        s[0] += f[0];
        s[1] += f[1];
        s[2] += f[2];
        // Torque p x f about the ligand centre.
        s[3] += p[1]*f[2] - p[2]*f[1];
        s[4] += p[2]*f[0] - p[0]*f[2];
        s[5] += p[0]*f[1] - p[1]*f[0];
    }
    // Sum over the threads in a fixed order.
    for (int k = 0; k < 6; k++) red[k*nt + tid] = s[k];
    __syncthreads();
    for (int w = nt/2; w > 0; w /= 2) {
        if (tid < w)
            for (int k = 0; k < 6; k++) red[k*nt + tid] += red[k*nt + tid + w];
        __syncthreads();
    }
    // Ligand frame to lab frame, then minus for the reaction on the ligand.
    if (tid < 3) {
        const double* Ri = R + 3*tid;
        force[3*t+tid] = -(Ri[0]*red[0] + Ri[1]*red[nt] + Ri[2]*red[2*nt]);
        torque[3*t+tid] = -(Ri[0]*red[3*nt] + Ri[1]*red[4*nt] + Ri[2]*red[5*nt]);
    }
}
"""


class _GridList(list):
    """Uploaded grids, finest first, with the tables the force kernel reads:
    values (the values of all grids in one array), start (where each grid
    begins in values), box (origin, 1/spacing, spacing, usable lo and hi) and
    n (points per axis)."""


class GPUBatchForceEngine:
    """
    Batch GPU force engine with a Yukawa far-field fallback.

    The argument elec_grids is a list of DXGrid objects for electrostatics,
    ordered finest first, and born_grids is the corresponding list for Born
    desolvation, also finest first. The argument alpha is the Born desolvation
    parameter. receptor_charge is the total charge of the receptor molecule in
    units of e, debye_length is the Debye screening length in angstrom, and
    sdie is the solvent dielectric constant. kT_scale is T / 298.15 K, which
    puts the far field and the WCA well depth in kBT at the run temperature.
    """

    def __init__(
        self,
        elec_grids,
        born_grids,
        alpha: float = 1.0,
        receptor_charge: float = 0.0,
        debye_length: float = 7.858,
        sdie: float = 78.0,
        lig_born_grids=None,
        rec_positions=None,
        rec_charges=None,
        multipole_expansion=None,
        rec_radii=None,
        lig_radii=None,
        use_lj=False,
        enable_born2_torque=True,
        kT_scale=1.0,
    ):
        if not _CUPY:
            raise RuntimeError(
                "CuPy not installed. Install with: pip install cupy-cuda12x"
            )
        self.alpha = alpha
        self._kernel = cp.RawKernel(_BATCH_KERNEL_CODE, "batch_force_kernel")
        # System parameters used by the Yukawa far-field fallback.
        self._rec_charge = receptor_charge
        self._debye = debye_length
        self._sdie = sdie
        # The prefactor of the far-field potential is V_factor = Q_rec / (4π εs),
        # where εs = sdie × (vacuum permittivity). The vacuum permittivity in
        # the reference units used here is 0.000142 e²/(kBT·A) at 298.15 K.
        eps_s = sdie * VACUUM_PERMITTIVITY_KBT * kT_scale
        self._V_factor = receptor_charge / (4.0 * math.pi * eps_s) if eps_s > 0 else 0.0
        # Multipole expansion (dipole and quadrupole) used in the far field.
        self._multipole = multipole_expansion
        if self._multipole is not None:
            self._mp_dipole_gpu = cp.asarray(self._multipole.dipole, dtype=cp.float64)
            self._mp_quad_gpu = cp.asarray(self._multipole.quadrupole, dtype=cp.float64)
            self._mp_four_pi_eps = self._multipole.four_pi_eps
            # Primitive second-moment trace tr(M) = Σ q_i |r_i|²; drives the
            # screened isotropic far-field term (an effective monopole of charge
            # tr(M)/(6λ²)) that a traceless-only quadrupole drops for a screened
            # kernel.
            self._mp_trace = float(getattr(self._multipole, "trace_moment", 0.0))
        # The far field is active when the receptor carries either a net monopole
        # charge or a non-negligible dipole/quadrupole. Gating on the monopole
        # alone silently dropped the multipole steering for neutral receptors
        # (for example beta-cyclodextrin), zeroing the far-field force outside the
        # finest grid for the exact case the multipole expansion exists to cover.
        # _yukawa_forces_gpu carries both the monopole and the multipole terms.
        _has_multipole = self._multipole is not None and (
            float(cp.linalg.norm(self._mp_dipole_gpu)) > 1e-9
            or float(cp.linalg.norm(self._mp_quad_gpu)) > 1e-9
            or abs(self._mp_trace) > 1e-9
        )
        self._has_yukawa = (
            abs(receptor_charge) > 1e-9 or _has_multipole
        ) and debye_length > 0
        # Counter used to emit diagnostics on the first few calls.
        self._call_count = 0
        # Upload the grids to the GPU once.
        self._elec_grids_gpu = self._upload_grids(elec_grids)
        self._born_grids_gpu = self._upload_grids(born_grids, margin=1.0)
        # The reference grid architecture uses one fine grid per molecule plus a
        # multipole or analytical field for atoms outside it. The coarse
        # electrostatic grid exists only to provide Dirichlet boundary
        # conditions for the fine grid's APBS solve (the bcfl map). At runtime
        # the fine grid handles atoms inside it and the Chebyshev or Yukawa
        # field handles atoms outside.
        #
        # We retain all APBS grids at runtime. The force kernel of _eval_batch
        # processes each atom on exactly one grid, namely the finest grid that
        # contains it. Atoms outside all grids fall through to the Yukawa
        # fallback. The previous code dropped every grid but the finest,
        # assuming the coarse grids existed only as APBS boundary conditions.
        # In fact the coarse APBS field is more accurate than truncated
        # multipole extrapolation in the intermediate range, and it matches the
        # BD2 runtime behavior.
        if len(self._elec_grids_gpu) > 1:
            _fine = self._elec_grids_gpu[0]
            _coarse = self._elec_grids_gpu[-1]
            _fine_ext = float(max(abs(_fine["lo"][0]), abs(_fine["hi"][0])))
            _coarse_ext = float(max(abs(_coarse["lo"][0]), abs(_coarse["hi"][0])))
            print(
                f"  Elec: {len(self._elec_grids_gpu)} grids retained "
                f"(fine ±{_fine_ext:.0f}Å -> coarse ±{_coarse_ext:.0f}Å)"
                + ("  + Yukawa far-field" if self._has_yukawa else "")
            )
        # The desolvation grid is a cavity self energy that decays as 1/r^4, so
        # the coarse grid carries a physical far-field force. The fine Born box
        # is about
        # 28 A half-width while the field has support to about 43 A, and
        # clipping it leaves a force discontinuity of tens of kBT/A at the face.
        if len(self._born_grids_gpu) > 1:
            _b_ext = float(
                max(
                    abs(self._born_grids_gpu[-1]["lo"][0]),
                    abs(self._born_grids_gpu[-1]["hi"][0]),
                )
            )
            print(
                f"  Born: {len(self._born_grids_gpu)} desolvation grids retained "
                f"(to {_b_ext:.0f}A)"
            )
        # The call core_desolvation_force_on_1(state0, state1) gives the
        # receptor Born force on the ligand, and core_desolvation_force_on_1(
        # state1, state0) gives the ligand Born force on the receptor.
        # Direction 2 requires the ligand Born grids and the receptor atom
        # positions.
        self._enable_born2_torque = enable_born2_torque
        self._lig_born_grids_gpu = []
        self._rec_pos_gpu = None
        self._rec_charges_gpu = None
        if (
            enable_born2_torque
            and alpha > 1e-12
            and lig_born_grids
            and rec_positions is not None
            and rec_charges is not None
        ):
            self._lig_born_grids_gpu = self._upload_grids(lig_born_grids, margin=1.0)
            # We retain all ligand Born grids, by the same reasoning used for
            # the receptor side above. The direction 2 kernel takes each
            # receptor atom on the finest grid that holds it, as _eval_batch
            # does, and the coarse ligand Born grids give receptor atoms in the
            # intermediate range a proper APBS field.
            self._rec_pos_gpu = cp.asarray(
                rec_positions, dtype=cp.float64
            )  # shape (N_rec, 3)
            self._rec_charges_gpu = cp.asarray(
                rec_charges, dtype=cp.float64
            )  # shape (N_rec,)
            self._init_born_reverse()
            print(
                f"  Born both-directions: {len(self._lig_born_grids_gpu)} lig born grid(s), "
                f"{len(rec_charges)} rec atoms"
            )
        else:
            print("  Born direction 2 off (enable_born2_torque false, alpha 0 or no ligand Born grid)")
        print(
            f"  GPUBatchForceEngine: {len(self._elec_grids_gpu)} elec + "
            f"{len(self._born_grids_gpu)} born grids on GPU"
        )
        if self._has_yukawa:
            print(
                f"  Yukawa far-field: Q_rec={receptor_charge:+.2f} e, "
                f"debye={debye_length:.3f} A, sdie={sdie:.1f}"
            )
        # Lennard-Jones (WCA repulsive) forces between the atoms of radius > 0.
        self._use_lj = use_lj
        if use_lj:
            if rec_positions is None or rec_radii is None or lig_radii is None:
                raise ValueError(
                    "lj_forces needs the receptor positions and both radius sets."
                )
            rec_r = np.asarray(rec_radii, dtype=np.float64)
            lig_r = np.asarray(lig_radii, dtype=np.float64)
            self._wca_rec = cp.asarray(
                np.asarray(rec_positions)[rec_r > 0.0], dtype=cp.float64
            )
            self._wca_rec_radii = cp.asarray(rec_r[rec_r > 0.0])
            self._wca_lig = cp.asarray(np.nonzero(lig_r > 0.0)[0])
            self._wca_lig_radii = cp.asarray(lig_r[lig_r > 0.0])
            # WCA well depth 0.1 kcal/mol, 0.17 kBT at 298.15 K.
            self._lj_epsilon = 0.17 / kT_scale
            # A pair interacts below 2^(1/6) sigma, so a trajectory feels the
            # wall only when its centroid is within this reach plus the ligand
            # extent of the origin.
            self._lj_reach = float(
                cp.linalg.norm(self._wca_rec, axis=1).max()
            ) + 1.122462 * float(self._wca_rec_radii.max() + self._wca_lig_radii.max())
            print(
                f"  LJ (WCA repulsive): {len(self._wca_rec)} rec + {len(self._wca_lig)} "
                f"lig atoms of radius > 0, reach {self._lj_reach:.1f} A + ligand extent"
            )

    def _upload_grids(self, grids, margin=3.0):
        """Upload a list of DXGrid objects to the GPU, sorted finest first.

        An atom is evaluated on a grid only margin spacings in from its faces.
        The central difference probes ±0.5 spacing, which one spacing covers,
        and focused APBS grids built with bcfl=map carry about 3 outer shells
        of interpolated boundary values, so the APBS grids keep 3 and the
        analytic Born grids take 1.
        """
        if not grids:
            return []
        sorted_grids = sorted(grids, key=lambda g: float(g.delta[0, 0]))
        # The values of all grids in one array, grid k from start[k] on.
        start = np.cumsum([0] + [g.data.size for g in sorted_grids])
        values = cp.asarray(
            np.concatenate([g.data.ravel() for g in sorted_grids]), dtype=cp.float64
        )
        uploaded = _GridList()
        for k, g in enumerate(sorted_grids):
            data = values[start[k] : start[k + 1]]
            origin = cp.asarray(g.origin, dtype=cp.float64)
            sp = cp.array([g.delta[i, i] for i in range(3)], dtype=cp.float64)
            inv_sp = 1.0 / sp
            nx, ny, nz = g.data.shape
            sp_np = np.array([g.delta[i, i] for i in range(3)])
            dims = np.array([nx, ny, nz])
            lo = g.origin + margin * sp_np
            hi = g.origin + (dims - 1 - margin) * sp_np
            uploaded.append(
                {
                    "data": data,
                    "origin": origin,
                    "sp": sp,
                    "inv_sp": inv_sp,
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "lo": lo,
                    "hi": hi,
                }
            )
        uploaded.values = values
        uploaded.start = cp.asarray(start[:-1], dtype=cp.int64)
        uploaded.box = cp.stack(
            [
                cp.concatenate(
                    (g["origin"], g["inv_sp"], g["sp"], cp.asarray(g["lo"]), cp.asarray(g["hi"]))
                )
                for g in uploaded
            ]
        )
        uploaded.n = cp.array([(g["nx"], g["ny"], g["nz"]) for g in uploaded], dtype=cp.int32)
        return uploaded

    def _eval_batch(
        self,
        positions_gpu,  # shape (N_traj, N_atoms, 3)
        charges_gpu,  # shape (N_atoms,)
        grids_gpu,
        alpha: float,
        is_born: bool,
        centroids=None,  # shape (N_traj, 3), the reference point for torque accumulation
    ):
        """
        Evaluate the forces, torques, and energies for all trajectories.

        One kernel launch covers every atom of every trajectory. Each atom is
        assigned to the finest grid whose usable box holds it, and for
        electrostatics an atom outside every grid gets the Yukawa and
        multipole far field. Torque is computed about each trajectory's
        centroid. When centroids is None the torques are returned as zeros,
        which matches the behavior expected by legacy callers.
        """
        N_traj, N_atoms, _ = positions_gpu.shape
        if not grids_gpu:
            forces = cp.zeros((N_traj, 3), dtype=cp.float64)
            torques = cp.zeros((N_traj, 3), dtype=cp.float64)
            energies = cp.zeros((N_traj,), dtype=cp.float64)
            if not is_born and self._has_yukawa:
                yf, yt, ye = self._yukawa_forces_gpu(
                    positions_gpu, charges_gpu, centroids=centroids
                )
                forces += yf
                torques += yt
                energies += ye
            return forces, torques, energies
        forces = cp.empty((N_traj, 3), dtype=cp.float64)
        torques = cp.empty((N_traj, 3), dtype=cp.float64)
        energies = cp.empty((N_traj,), dtype=cp.float64)
        has_cen = centroids is not None
        # The far field applies to electrostatics only. The Born field is cut
        # at 15 A, which the Born grids cover.
        has_far = not is_born and self._has_yukawa
        self._kernel(
            ((N_traj + 3) // 4,),
            (32, 4),
            (
                positions_gpu,
                *[np.int64(s // 8) for s in positions_gpu.strides],
                cp.ascontiguousarray(charges_gpu, dtype=cp.float64),
                cp.ascontiguousarray(centroids, dtype=cp.float64) if has_cen else None,
                np.int32(has_cen),
                grids_gpu.values,
                grids_gpu.start,
                grids_gpu.box,
                grids_gpu.n,
                np.int32(len(grids_gpu)),
                self._far_field() if has_far else None,
                np.int32(has_far),
                np.float64(alpha),
                np.int32(1 if is_born else 0),
                np.int32(N_traj),
                np.int32(N_atoms),
                forces,
                torques,
                energies,
            ),
        )
        # Log the first few calls.
        if self._call_count < 3 and not is_born:
            n_total = N_traj * N_atoms
            on_grid = cp.zeros((N_traj, N_atoms), dtype=cp.bool_)
            for g in grids_gpu:
                on_grid |= cp.all(positions_gpu > cp.asarray(g["lo"]), axis=2) & cp.all(
                    positions_gpu < cp.asarray(g["hi"]), axis=2
                )
            n_assigned = int(cp.sum(on_grid))
            n_not = n_total - n_assigned
            if self._has_yukawa and n_not > 0:
                print(
                    f"    [FORCE VERBOSE] Yukawa fallback: "
                    f"{n_not}/{n_total} atom-traj pairs "
                    f"({100*n_not/n_total:.1f}%)"
                )
            print(
                f"    [FORCE VERBOSE call#{self._call_count}] ELEC: "
                f"{n_assigned}/{n_total} atom-traj pairs assigned to grids "
                f"({100*n_assigned/n_total:.1f}%)"
            )
            f_mag = cp.linalg.norm(forces, axis=1)
            print(
                f"    [FORCE VERBOSE] |F| stats: "
                f"mean={float(f_mag.mean()):.8f} "
                f"max={float(f_mag.max()):.8f} "
                f"min={float(f_mag.min()):.8f} kBT/Å"
            )
        return forces, torques, energies

    def _far_field(self):
        """Constants of the far field for the force kernel, uploaded once.
        Only called with the far field on, so the Debye length is positive."""
        if getattr(self, "_far", None) is None:
            c = np.zeros(19)
            c[0], c[1] = self._V_factor, self._debye
            if self._multipole is not None:
                c[2] = self._mp_four_pi_eps
                c[3] = self._mp_trace / (6.0 * self._debye**2)
                c[4:7] = cp.asnumpy(self._mp_dipole_gpu)
                c[7:16] = cp.asnumpy(self._mp_quad_gpu).ravel()
                c[16] = float(cp.linalg.norm(self._mp_dipole_gpu)) > 1e-9
                c[17] = float(cp.linalg.norm(self._mp_quad_gpu)) > 1e-9
                c[18] = abs(self._mp_trace) > 1e-12
            self._far = cp.asarray(c)
        return self._far

    def _yukawa_forces_gpu(self, positions_gpu, charges_gpu, mask=None, centroids=None):
        """
        Analytical screened Coulomb force from the receptor charge
        distribution.

        When a multipole expansion is provided this includes the monopole,
        dipole, and quadrupole terms. Otherwise it uses the monopole only,
        that is, the net charge Q. The receptor is placed at the origin, and
        the force on atom i is

            F_i = -q_i × ∇φ,

        in units of kBT/A, matching the forces obtained from the APBS grids.
        """
        N_traj, N_atoms, _ = positions_gpu.shape
        debye = self._debye
        r_mag = cp.linalg.norm(positions_gpu, axis=2)  # shape (N_traj, N_atoms)
        safe_r = cp.maximum(r_mag, 1.0)
        exp_term = cp.exp(-safe_r / debye)
        r_hat = positions_gpu / safe_r[:, :, None]
        # Monopole potential: φ = V_factor × exp(-r/λ) / r.
        V_fac = self._V_factor
        phi = V_fac * exp_term / safe_r
        dphi_dr = V_fac * exp_term * (-1.0 / safe_r**2 - 1.0 / (safe_r * debye))
        # Dipole and quadrupole terms, when a multipole expansion is available.
        if self._multipole is not None:
            p_gpu = self._mp_dipole_gpu  # dipole vector, shape (3,)
            Q_gpu = self._mp_quad_gpu  # quadrupole tensor, shape (3, 3)
            fpe = self._mp_four_pi_eps
            lam = debye
            # Dipole potential: φ_1 = (p · r̂) / (4πε r²) × (1 + r/λ) × exp(-r/λ).
            p_dot_r = cp.sum(
                r_hat * p_gpu[None, None, :], axis=2
            )  # shape (N_traj, N_atoms)
            p_mag = float(cp.linalg.norm(p_gpu))
            if p_mag > 1e-9:
                phi_dip = p_dot_r / (fpe * safe_r**2) * (1.0 + safe_r / lam) * exp_term
                phi += phi_dip
                # Leading radial term of d(φ_dip)/dr.
                dphi_dip_dr = (
                    p_dot_r
                    / fpe
                    * exp_term
                    * (
                        -2.0 / safe_r**3
                        - 2.0 / (safe_r**2 * lam)
                        - 1.0 / (safe_r * lam**2)
                    )
                )
                dphi_dr += dphi_dip_dr
            # Quadrupole potential:
            # φ_2 = (r̂ᵀ Q r̂) / (4πε r³) × (1 + r/λ + r²/(3λ²)) × exp(-r/λ).
            q_mag = float(cp.linalg.norm(Q_gpu))
            if q_mag > 1e-9:
                # The scalar r̂ᵀ Q r̂ for each atom has shape (N_traj, N_atoms),
                # formed from r_hat of shape (N_traj, N_atoms, 3).
                rQr = cp.sum(r_hat * cp.einsum("ij,...j->...i", Q_gpu, r_hat), axis=2)
                phi_quad = (
                    rQr
                    / (fpe * safe_r**3)
                    * (1.0 + safe_r / lam + safe_r**2 / (3.0 * lam**2))
                    * exp_term
                )
                phi += phi_quad
                dphi_quad_dr = (
                    rQr
                    / fpe
                    * exp_term
                    * (
                        -3.0 / safe_r**4
                        - 3.0 / (safe_r**3 * lam)
                        - 4.0 / (3.0 * safe_r**2 * lam**2)
                        - 1.0 / (3.0 * safe_r * lam**3)
                    )
                )
                dphi_dr += dphi_quad_dr
            # Screened isotropic trace term (effective monopole of charge
            # Q_eff = tr(M)/(6λ²)). It is spherically symmetric, so it
            # contributes to φ and the radial gradient only, with no transverse
            # part. This is exactly the term a traceless-only quadrupole drops
            # for a screened kernel (∇²G = G/λ² ≠ 0).
            _mp_trace = float(getattr(self, "_mp_trace", 0.0))
            if abs(_mp_trace) > 1e-12:
                q_eff_tr = _mp_trace / (6.0 * lam**2)
                phi += q_eff_tr / (fpe * safe_r) * exp_term
                dphi_dr += (
                    q_eff_tr
                    / fpe
                    * exp_term
                    * (-1.0 / safe_r**2 - 1.0 / (safe_r * lam))
                )
        # The gradient is ∇φ = (dφ/dr) r̂ plus a transverse contribution from
        # the non-spherical terms. The radial part is exact for the monopole,
        # whose potential is spherically symmetric. For the dipole the
        # potential V_dip = (p·r̂)/(4πε r²)·(1+r/λ)·exp(-r/λ) carries angular
        # dependence through (p·r̂), so ∇V_dip has a transverse component
        # proportional to (p - (p·r̂)r̂) that a radial-only formula would drop.
        # Including it is essential for the orientational steering produced by
        # the ligand-receptor dipole interaction beyond the fine APBS grid.
        grad_phi = dphi_dr[:, :, None] * r_hat
        if (
            self._multipole is not None
            and float(cp.linalg.norm(self._mp_dipole_gpu)) > 1e-9
        ):
            p_gpu_t = self._mp_dipole_gpu
            fpe_t = self._mp_four_pi_eps
            lam_t = debye
            g_f = (1.0 + safe_r / lam_t) * exp_term  # shape (N_traj, N_atoms)
            transverse_factor = g_f / (fpe_t * safe_r**3)
            p_dot_r_t = cp.sum(r_hat * p_gpu_t[None, None, :], axis=2)
            p_perp = (
                p_gpu_t[None, None, :] - p_dot_r_t[:, :, None] * r_hat
            )  # shape (N_traj, N_atoms, 3)
            grad_phi = grad_phi + transverse_factor[:, :, None] * p_perp
        # The quadrupole potential also carries angular dependence, through the
        # scalar r̂ᵀ Q r̂, so its gradient has a transverse component that a
        # radial-only formula would drop. Writing the quadrupole potential as
        # h(r) times r̂ᵀ Q r̂, the transverse part is h(r) times the gradient of
        # r̂ᵀ Q r̂, which equals (2/r) (Q r̂ - (r̂ᵀ Q r̂) r̂). The combined factor
        # is 2 (1 + r/λ + r²/(3λ²)) exp(-r/λ) / (4πε r⁴). Including this term is
        # what gives the quadrupolar orientational steering for neutral hosts,
        # where the monopole and dipole both vanish.
        if (
            self._multipole is not None
            and float(cp.linalg.norm(self._mp_quad_gpu)) > 1e-9
        ):
            Q_gpu_t = self._mp_quad_gpu
            fpe_t = self._mp_four_pi_eps
            lam_t = debye
            quad_transverse_factor = (
                2.0
                * (1.0 + safe_r / lam_t + safe_r**2 / (3.0 * lam_t**2))
                * exp_term
                / (fpe_t * safe_r**4)
            )
            Q_rhat = cp.einsum("ij,...j->...i", Q_gpu_t, r_hat)
            rQr_t = cp.sum(r_hat * Q_rhat, axis=2)
            q_perp = Q_rhat - rQr_t[:, :, None] * r_hat
            grad_phi = grad_phi + quad_transverse_factor[:, :, None] * q_perp
        # The force on each atom is F_i = -q_i × ∇φ.
        q_3d = charges_gpu[None, :, None]
        atom_forces = -q_3d * grad_phi
        atom_energies = charges_gpu[None, :] * phi
        if mask is not None:
            mask_3d = mask[:, :, None].astype(cp.float64)
            atom_forces = atom_forces * mask_3d
            atom_energies = atom_energies * mask.astype(cp.float64)
        forces = atom_forces.sum(axis=1)
        energies = atom_energies.sum(axis=1)
        # Torque about each trajectory's centroid, given by the sum over atoms
        # of lever_i × F_i.
        if centroids is not None:
            lever_arms = positions_gpu - centroids[:, None, :]
            atom_torques = cp.cross(lever_arms, atom_forces, axis=2)
            torques = atom_torques.sum(axis=1)
        else:
            torques = cp.zeros((N_traj, 3), dtype=cp.float64)
        return forces, torques, energies

    def _wca_forces_gpu(self, lig_positions, centroids):
        """
        WCA (purely repulsive Lennard-Jones) forces between the atoms of
        radius > 0, with the torque about each trajectory's centroid.

        The pair size is σ_ij = rec_radius_i + lig_radius_j from the Lorentz
        combining rule, and the WCA cutoff keeps only the repulsive part,
        r < 2^(1/6) × σ_ij. The work is processed in chunks to limit GPU
        memory.

        Returns a (force, torque) pair, each of shape (N_traj, 3).
        """
        N_traj = lig_positions.shape[0]
        lig = lig_positions[:, self._wca_lig]
        N_lig = lig.shape[1]
        N_rec = self._wca_rec.shape[0]
        eps = self._lj_epsilon
        rho = float(cp.linalg.norm(lig - centroids[:, None, :], axis=2).max())
        active = cp.linalg.norm(centroids, axis=1) < self._lj_reach + rho
        n_active = int(active.sum())
        lj_forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        lj_torques = cp.zeros((N_traj, 3), dtype=cp.float64)
        if n_active == 0:
            return lj_forces, lj_torques
        active_idx = cp.where(active)[0]
        # Process in chunks to limit the memory of the (chunk × N_lig × N_rec ×
        # 3) array.
        CHUNK = max(1, min(50, int(2e9 / (N_lig * N_rec * 8 * 3))))
        for c0 in range(0, n_active, CHUNK):
            c1 = min(c0 + CHUNK, n_active)
            idx = active_idx[c0:c1]
            lp = lig[idx]  # shape (nc, N_lig, 3)
            rp = self._wca_rec  # shape (N_rec, 3)
            # Pairwise displacement vectors, shape (nc, N_lig, N_rec, 3).
            diff = lp[:, :, None, :] - rp[None, None, :, :]  # broadcast
            r2 = (diff * diff).sum(axis=3)  # shape (nc, N_lig, N_rec)
            r = cp.sqrt(cp.maximum(r2, 1e-6))
            # Pair size σ_ij = lig_radius + rec_radius.
            sig = self._wca_lig_radii[None, :, None] + self._wca_rec_radii[None, None, :]
            # WCA cutoff at r < 2^(1/6) × σ.
            r_cut = 1.122462 * sig  # 2^(1/6) ≈ 1.122462
            in_range = r < r_cut
            # WCA force from U = 4 eps [(σ/r)¹² - (σ/r)⁶], so
            # F = 4 eps × (12 (σ/r)¹² - 6 (σ/r)⁶) / r² × r_vec. The leading 4
            # sets the well depth to eps (0.1 kcal/mol); it matches the chain
            # reference in chain_simulator.py and the LJ/WCA wall in BrownDye2.
            sr = sig / r
            sr2 = sr * sr
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            f_mag = (
                4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / r2
            )  # shape (nc, N_lig, N_rec)
            f_mag = cp.where(in_range, f_mag, 0.0)
            # Force on each ligand atom from each receptor atom. f_mag already
            # carries the 1/r^2 factor (|F|/r), so the vector force is
            # f_mag * (lig - rec) with no further division by r.
            f_vec = f_mag[:, :, :, None] * diff  # shape (nc, N_lig, N_rec, 3)
            # Force per ligand atom, summed over receptor atoms, shape (nc,
            # N_lig, 3).
            f_per_lig = f_vec.sum(axis=2)
            # Net force per trajectory, summed over ligand atoms.
            lj_forces[idx] = f_per_lig.sum(axis=1)
            # Per-atom torque about the ligand centroid, cross(r - c, f_per_lig).
            lever = lp - centroids[idx][:, None, :]  # shape (nc, N_lig, 3)
            atom_torques = cp.cross(lever, f_per_lig, axis=2)  # shape (nc, N_lig, 3)
            lj_torques[idx] = atom_torques.sum(axis=1)
        return lj_forces, lj_torques

    def __call__(self, lig_positions, lig_charges, R_matrices=None, centroids=None):
        """
        Compute the net force, torque, and energy on the ligand for all
        trajectories.

        Born desolvation is computed in both directions. Direction 1, given by
        core_desolvation_force_on_1(state0, state1), is the receptor Born force
        on the ligand. Direction 2, given by core_desolvation_force_on_1(
        state1, state0), is the ligand Born force on the receptor, evaluated
        unless enable_born2_torque is false. The force on
        the receptor from direction 2 is negated by Newton's third law to give
        the reaction force on the ligand.

        The argument lig_positions has shape (N_traj, N_lig, 3) and holds the
        ligand atom positions in the lab frame, lig_charges has shape (N_lig,)
        and holds the ligand atom charges, R_matrices has shape (N_traj, 3, 3)
        and holds the rotation matrices that map the ligand frame to the lab
        frame, and centroids has shape (N_traj, 3) and holds the ligand
        centroid positions.
        """
        N_traj, N_atoms, _ = lig_positions.shape
        if self._call_count < 3:
            r_mag = cp.linalg.norm(
                lig_positions[:, 0, :], axis=1
            )  # radial distance of the centroid
            print(
                f"    [ENGINE call#{self._call_count}] N_traj={N_traj} N_atoms={N_atoms}  "
                f"r_centroid: mean={float(r_mag.mean()):.3f} "
                f"min={float(r_mag.min()):.3f} max={float(r_mag.max()):.3f}"
            )
        # Electrostatic force on the ligand from the receptor field. The kernel
        # sums start from +0.0, so they are the totals bit for bit.
        if self._elec_grids_gpu:
            forces, torques, energies = self._eval_batch(
                lig_positions,
                lig_charges,
                self._elec_grids_gpu,
                0.0,
                False,
                centroids=centroids,
            )
            if self._call_count < 3:
                fm = float(cp.linalg.norm(forces, axis=1).mean())
                tm = float(cp.linalg.norm(torques, axis=1).mean())
                print(
                    f"    [COMPONENT] ELEC:   |F|_mean={fm:.6e} kBT/Å  "
                    f"|T|_mean={tm:.6e} kBT"
                )
        else:
            forces = cp.zeros((N_traj, 3), dtype=cp.float64)
            torques = cp.zeros((N_traj, 3), dtype=cp.float64)
            energies = cp.zeros((N_traj,), dtype=cp.float64)
            if self._has_yukawa:
                f, t, e = self._yukawa_forces_gpu(
                    lig_positions, lig_charges, centroids=centroids
                )
                forces += f
                torques += t
                energies += e
        # Born desolvation, direction 1: the receptor Born grid acting on the
        # ligand atoms. With alpha = 0 every term is zero and it is skipped.
        if self._born_grids_gpu:
            fm = tm = 0.0
            if self.alpha != 0.0:
                f, t, e = self._eval_batch(
                    lig_positions,
                    lig_charges,
                    self._born_grids_gpu,
                    self.alpha,
                    True,
                    centroids=centroids,
                )
                forces += f
                torques += t
                energies += e
                if self._call_count < 3:
                    fm = float(cp.linalg.norm(f, axis=1).mean())
                    tm = float(cp.linalg.norm(t, axis=1).mean())
            if self._call_count < 3:
                print(
                    f"    [COMPONENT] BORN1:  |F|_mean={fm:.6e} kBT/Å  "
                    f"|T|_mean={tm:.6e} kBT"
                )
        # Born desolvation, direction 2: the ligand Born grid acting on the
        # receptor atoms. Evaluating the ligand Born grid at the receptor atom
        # positions gives the force on the receptor. By Newton's third law the
        # force on the ligand is minus the force on the receptor.
        if (
            self._enable_born2_torque
            and self._lig_born_grids_gpu
            and self._rec_pos_gpu is not None
            and R_matrices is not None
            and centroids is not None
            and self.alpha > 1e-12
        ):
            f2, t2 = self._eval_born_reverse(R_matrices, centroids, N_traj)
            # For parity with BD2 we add both the direction 1 Born force (added
            # above) and the Newton's-third-law reciprocal of the direction 2
            # Born force into the ligand net force. This matches add_core_forces
            # in BD2's forces_impl.hh (around lines 998 to 1005), which sums
            # b_force10 and b_force11 into force1. The term f2 was previously
            # zeroed as a workaround for an apparent spurious barrier at r ≈ 22
            # to 30 A, and it is restored here for BD2 parity. If that barrier
            # reappears in benchmarks, investigate the ligand Born grid
            # construction (see the diagnostic near lines 209 to 223).
            forces += f2
            torques += t2
            if self._call_count < 3:
                fm = float(cp.linalg.norm(f2, axis=1).mean())
                tm = float(cp.linalg.norm(t2, axis=1).mean())
                print(
                    f"    [COMPONENT] BORN2:  |F|_mean={fm:.6e} kBT/Å  "
                    f"|T|_mean={tm:.6e} kBT  "
                    f"- lig Born on {self._rec_pos_gpu.shape[0]} rec atoms"
                )
        # Lennard-Jones (WCA repulsive) forces, computed only when atoms are
        # very close.
        if self._use_lj:
            lj_f, lj_t = self._wca_forces_gpu(lig_positions, centroids)
            forces += lj_f
            torques += lj_t
            if self._call_count < 3:
                fm = float(cp.linalg.norm(lj_f, axis=1).mean())
                tm = float(cp.linalg.norm(lj_t, axis=1).mean())
                print(
                    f"    [COMPONENT] WCA:    |F|_mean={fm:.6e} kBT/Å  "
                    f"|T|_mean={tm:.6e} kBT"
                )
        self._call_count += 1
        return forces, torques, energies

    def _init_born_reverse(self):
        """Device inputs of the direction 2 kernel, built once."""
        grids = self._lig_born_grids_gpu
        self._b2_rec = cp.ascontiguousarray(self._rec_pos_gpu)
        self._b2_chg = cp.ascontiguousarray(self._rec_charges_gpu)
        # Reach: the farthest box corner, or the farthest nonzero node of any
        # grid plus the 2.5 spacings an interpolation reads, if smaller.
        corner = max(
            float(np.linalg.norm(np.maximum(np.abs(g["lo"]), np.abs(g["hi"]))))
            for g in grids
        )
        far = 0.0
        for g in grids:
            x, y, z = (
                g["origin"][k] + g["sp"][k] * cp.arange(n)
                for k, n in enumerate((g["nx"], g["ny"], g["nz"]))
            )
            d2 = (x[:, None, None] ** 2 + y[None, :, None] ** 2) + z[None, None, :] ** 2
            d2 = d2[g["data"].reshape(d2.shape) != 0.0]
            if d2.size:
                far = max(far, float(cp.sqrt(d2.max())) + 2.5 * float(g["sp"].max()))
        self._b2_reach = 1e-6 + min(corner, far)
        self._b2_rec_r = float(cp.linalg.norm(self._b2_rec, axis=1).max())
        self._b2_kernel = cp.RawKernel(_BORN2_KERNEL_CODE, "born2_kernel")

    def _eval_born_reverse(self, R_matrices, centroids, N_traj):
        """
        Born direction 2: evaluate the ligand Born grid at the receptor atom
        positions.

        For each trajectory the receptor atoms are first transformed into the
        ligand frame as Rᵀ (rec_pos - centroid). Each atom is evaluated on the
        finest ligand Born grid whose usable box holds it, which gives the
        force on the receptor atoms expressed in the ligand frame, and it is
        rotated back to the lab frame as R F_lig. By Newton's third law the
        force on the ligand is minus the force on the receptor, and likewise
        the torque on the ligand about its centre is minus the torque on the
        receptor about the same point, matching the b_torque11 contribution in
        BD2's add_core_forces. One kernel does all of this with one block of
        threads per trajectory. It skips a trajectory whose centre is too far
        for any receptor atom to reach a box, and atoms beyond the reach.

        Returns result_f of shape (N_traj, 3), the net force on the ligand from
        direction 2, and result_t of shape (N_traj, 3), the corresponding net
        torque on the ligand.
        """
        result_f = cp.empty((N_traj, 3), dtype=cp.float64)
        result_t = cp.empty((N_traj, 3), dtype=cp.float64)
        threads = 128
        self._b2_kernel(
            (N_traj,),
            (threads,),
            (
                cp.ascontiguousarray(centroids, dtype=cp.float64),
                cp.ascontiguousarray(R_matrices, dtype=cp.float64),
                self._b2_rec,
                self._b2_chg,
                self._lig_born_grids_gpu.values,
                self._lig_born_grids_gpu.box,
                self._lig_born_grids_gpu.n,
                self._lig_born_grids_gpu.start,
                np.int32(len(self._lig_born_grids_gpu)),
                np.int32(self._b2_rec.shape[0]),
                np.float64(self.alpha),
                np.float64(self._b2_rec_r),
                np.float64(self._b2_reach),
                result_f,
                result_t,
            ),
            shared_mem=6 * threads * 8,
        )
        return result_f, result_t
