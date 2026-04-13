"""
GPU force evaluation engine
===========================

Background
-------------------
This module computes the total force on each ligand atom from the
receptor's electrostatic field.  Three contributions are summed.

1. ELECTROSTATIC (Poisson-Boltzmann)
   F_elec = -q_i × ∇φ_rec(r_i)
   The receptor's electrostatic potential φ_rec is precomputed by
   APBS on a 3D grid.  The gradient ∇φ is evaluated by trilinear
   interpolation with central differences at half-grid-spacing.
   For atoms outside the grid (with 3-spacing safety margin), the
   Yukawa multipole far-field is used instead.

2. BORN DESOLVATION
   F_born = -α × q_i² × ∇φ_born(r_i)
   When a charged atom approaches the receptor, it partially
   displaces the high-dielectric solvent, paying an energetic
   penalty.  The Born potential φ_born is the vacuum electrostatic
   potential (ε=1 everywhere, no ions).  α = 1/(4π) ≈ 0.0796.

   Computed in BOTH directions:
   1. Direction 1: receptor Born grid at ligand atom positions
   2. Direction 2: ligand Born grid at receptor atom positions
   Direction 2 is GPU-memory-intensive for large receptors and
   is computed in 500 MB chunks.

3. YUKAWA MONOPOLE FALLBACK
   For atoms far from the receptor (outside APBS grid), the
   screened Coulomb force provides the correct long-range behavior:
   F = q_rec × q_lig / (4πε) × (1/r² + 1/(rλ)) × exp(-r/λ) × r̂
   where λ is the Debye screening length.

Force batching
--------------
For large ligands (e.g. thrombomodulin, 1652 atoms), evaluating
forces for all 10⁶ trajectories simultaneously would exceed GPU
memory.  The engine automatically batches:

    N_batch = 4 GB / (N_lig_atoms × 150 bytes)

The 150 bytes/atom accounts for ~6 internal arrays per batch
(positions, forces, energies, masks, etc.).
"""

from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import numpy as np
import math

try:
    import cupy as cp

    _CUPY = True
except ImportError:
    _CUPY = False


# CUDA kernel: one thread per (trajectory, atom) pair

_BATCH_KERNEL_CODE = r"""
extern "C" __global__
void batch_force_kernel(
    const double* __restrict__ positions,
    const double* __restrict__ charges,
    const double* __restrict__ grid_data,
    const double* __restrict__ origin,
    const double* __restrict__ inv_sp,
    const double* __restrict__ sp,
    double alpha,
    int is_born,
    int nx, int ny, int nz,
    int N_traj, int N_atoms,
    double* __restrict__ atom_forces,
    double* __restrict__ atom_energies)
{
    int traj = blockIdx.x * blockDim.x + threadIdx.x;
    int atom  = blockIdx.y * blockDim.y + threadIdx.y;
    if (traj >= N_traj || atom >= N_atoms) return;
    double q = charges[atom];
    int out_idx = traj * N_atoms + atom;
    atom_forces[out_idx*3+0] = 0;
    atom_forces[out_idx*3+1] = 0;
    atom_forces[out_idx*3+2] = 0;
    atom_energies[out_idx]   = 0;
    if (fabs(q) < 1e-9) return;
    int pos_idx = (traj * N_atoms + atom) * 3;
    double px = positions[pos_idx+0];
    double py = positions[pos_idx+1];
    double pz = positions[pos_idx+2];
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
    atom_forces[out_idx*3+0] = coeff * gx;
    atom_forces[out_idx*3+1] = coeff * gy;
    atom_forces[out_idx*3+2] = coeff * gz;
    atom_energies[out_idx]   = e_coeff * phi;
}
"""


class GPUBatchForceEngine:
    """
    Batch GPU force engine with Yukawa far-field fallback.
    Parameters
    ----------
    elec_grids      : list of DXGrid for electrostatics (finest first)
    born_grids      : list of DXGrid for Born desolvation (finest first)
    alpha           : Born desolvation parameter
    receptor_charge : total charge of receptor molecule (in e)
    debye_length    : Debye screening length (in A)
    sdie            : solvent dielectric constant
    """

    def __init__(
        self,
        elec_grids,
        born_grids,
        alpha: float = 0.07957747,
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
    ):
        if not _CUPY:
            raise RuntimeError(
                "CuPy not installed. Install with: pip install cupy-cuda12x"
            )
        self.alpha = alpha
        self._kernel = cp.RawKernel(_BATCH_KERNEL_CODE, "batch_force_kernel")
        # System parameters for Yukawa far-field fallback
        self._rec_charge = receptor_charge
        self._debye = debye_length
        self._sdie = sdie
        # V_factor = Q_rec / (4*pi*eps_s)  where eps_s = sdie * vacuum_permittivity
        # vacuum_permittivity in the reference units: 0.000142 e^2/(kBT*A)
        eps_s = sdie * 0.000142
        self._V_factor = receptor_charge / (4.0 * math.pi * eps_s) if eps_s > 0 else 0.0
        self._has_yukawa = abs(receptor_charge) > 1e-9 and debye_length > 0
        # Multipole expansion (dipole + quadrupole) for far-field
        self._multipole = multipole_expansion
        if self._multipole is not None:
            self._mp_dipole_gpu = cp.asarray(self._multipole.dipole, dtype=cp.float64)
            self._mp_quad_gpu = cp.asarray(self._multipole.quadrupole, dtype=cp.float64)
            self._mp_four_pi_eps = self._multipole.four_pi_eps
        # Verbose call counter for first-call diagnostics
        self._call_count = 0
        # Upload grids to GPU once
        self._elec_grids_gpu = self._upload_grids(elec_grids)
        self._born_grids_gpu = self._upload_grids(born_grids)
        # The reference grid architecture: one fine grid per molecule + multipole/analytical
        # field for atoms outside.
        # The coarse electrostatic grid exists only to provide Dirichlet
        # boundary conditions for the fine grid's APBS solve (bcfl map).
        # At runtime: fine grid for atoms inside, Chebyshev/Yukawa for outside.
        if self._has_yukawa and len(self._elec_grids_gpu) > 1:
            finest = self._elec_grids_gpu[0]  # sorted finest-first
            fine_extent = float(max(abs(finest["lo"][0]), abs(finest["hi"][0])))
            print(
                f"  Elec: using fine grid only (±{fine_extent:.0f}Å) + Yukawa far-field "
                f"(dropping {len(self._elec_grids_gpu)-1} coarse grid(s) - BC only)"
            )
            self._elec_grids_gpu = [finest]
        if len(self._born_grids_gpu) > 1:
            finest_born = self._born_grids_gpu[0]
            print(
                f"  Born: using finest grid only "
                f"(dropping {len(self._born_grids_gpu)-1} coarse grid(s))"
            )
            self._born_grids_gpu = [finest_born]
        # core_desolvation_force_on_1(state0, state1)  -> rec Born on lig
        # core_desolvation_force_on_1(state1, state0)  -> lig Born on rec
        # Direction 2 needs ligand Born grids + receptor atom positions
        self._lig_born_grids_gpu = []
        self._rec_pos_gpu = None
        self._rec_charges_gpu = None
        if lig_born_grids and rec_positions is not None and rec_charges is not None:
            self._lig_born_grids_gpu = self._upload_grids(lig_born_grids)
            # Keep only finest lig Born grid (auto-sized to ligand extent)
            if len(self._lig_born_grids_gpu) > 1:
                self._lig_born_grids_gpu = [self._lig_born_grids_gpu[0]]
            self._rec_pos_gpu = cp.asarray(
                rec_positions, dtype=cp.float64
            )  # (N_rec, 3)
            self._rec_charges_gpu = cp.asarray(
                rec_charges, dtype=cp.float64
            )  # (N_rec,)
            print(
                f"  Born both-directions: {len(self._lig_born_grids_gpu)} lig born grid(s), "
                f"{len(rec_charges)} rec atoms"
            )
        print(
            f"  GPUBatchForceEngine: {len(self._elec_grids_gpu)} elec + "
            f"{len(self._born_grids_gpu)} born grids on GPU"
        )
        if self._has_yukawa:
            print(
                f"  Yukawa far-field: Q_rec={receptor_charge:+.2f} e, "
                f"debye={debye_length:.3f} A, sdie={sdie:.1f}"
            )
        # LJ (WCA repulsive) forces using PQR radii
        self._use_lj = use_lj
        self._rec_radii_gpu = None
        self._lig_radii_gpu = None
        if use_lj and rec_radii is not None and lig_radii is not None:
            self._rec_radii_gpu = cp.asarray(rec_radii, dtype=cp.float64)
            self._lig_radii_gpu = cp.asarray(lig_radii, dtype=cp.float64)
            # WCA epsilon in kBT units (0.1 kcal/mol / 0.593 kcal/mol/kBT ≈ 0.17 kBT)
            self._lj_epsilon = 0.17
            # Activation radius: only compute LJ when centroid is close
            _max_sig = float(self._rec_radii_gpu.max() + self._lig_radii_gpu.max())
            self._lj_activation = _max_sig * 2.5
            print(
                f"  LJ (WCA repulsive): {len(rec_radii)} rec + {len(lig_radii)} lig atoms, "
                f"activation={self._lj_activation:.1f} A"
            )

    def _upload_grids(self, grids):
        """Upload DXGrid list to GPU, sorted finest first.
        Grid bounds account for:
        1. Gradient probe width (central diff needs ±0.5*sp)
        2. APBS boundary artifacts (focused grids with bcfl=map have
           ~3 shells contaminated by coarse grid BC interpolation)
        Valid range: [origin + 3*sp, origin + (nx-4)*sp].
        This 3-spacing margin prevents force spikes at grid boundaries.
        """
        if not grids:
            return []
        sorted_grids = sorted(grids, key=lambda g: float(g.delta[0, 0]))
        uploaded = []
        for g in sorted_grids:
            data = cp.asarray(g.data.ravel(), dtype=cp.float64)
            origin = cp.asarray(g.origin, dtype=cp.float64)
            sp = cp.array([g.delta[i, i] for i in range(3)], dtype=cp.float64)
            inv_sp = 1.0 / sp
            nx, ny, nz = g.data.shape
            # BC-safe bounds: 3 spacings from each edge
            # (gradient probe needs 0.5*sp + APBS focused BC contaminates ~2-3 shells)
            sp_np = np.array([g.delta[i, i] for i in range(3)])
            dims = np.array([nx, ny, nz])
            margin = 3.0  # grid spacings from edge
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
        return uploaded

    def _eval_batch(
        self,
        positions_gpu,  # (N_traj, N_atoms, 3)
        charges_gpu,  # (N_atoms,)
        grids_gpu,
        alpha: float,
        is_born: bool,
    ):
        """
        Evaluate forces for all trajectories against grids.
        Finest-grid-first assignment. Yukawa fallback for unassigned atoms.
        """
        N_traj, N_atoms, _ = positions_gpu.shape
        total_forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        total_energies = cp.zeros((N_traj,), dtype=cp.float64)
        if not grids_gpu:
            if not is_born and self._has_yukawa:
                yf, ye = self._yukawa_forces_gpu(positions_gpu, charges_gpu)
                total_forces += yf
                total_energies += ye
            return total_forces, total_energies
        assigned = cp.zeros((N_traj, N_atoms), dtype=cp.bool_)
        for g in grids_gpu:
            nx, ny, nz = g["nx"], g["ny"], g["nz"]
            atom_forces = cp.zeros((N_traj, N_atoms, 3), dtype=cp.float64)
            atom_energies = cp.zeros((N_traj, N_atoms), dtype=cp.float64)
            lo = cp.asarray(g["lo"], dtype=cp.float64)
            hi = cp.asarray(g["hi"], dtype=cp.float64)
            in_grid = cp.all(positions_gpu > lo, axis=2) & cp.all(
                positions_gpu < hi, axis=2
            )
            to_process = in_grid & ~assigned
            assigned |= to_process
            if not cp.any(to_process):
                continue
            pos_masked = positions_gpu.copy()
            pos_masked[~to_process] = 0.0
            chg_masked = cp.zeros((N_traj, N_atoms), dtype=cp.float64)
            chg_masked[to_process] = charges_gpu[cp.where(to_process)[1]]
            threads = (16, 16)
            blocks = (
                (N_traj + threads[0] - 1) // threads[0],
                (N_atoms + threads[1] - 1) // threads[1],
            )
            self._kernel(
                blocks,
                threads,
                (
                    pos_masked.ravel(),
                    chg_masked.ravel(),
                    g["data"],
                    g["origin"],
                    g["inv_sp"],
                    g["sp"],
                    np.float64(alpha),
                    np.int32(1 if is_born else 0),
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nz),
                    np.int32(N_traj),
                    np.int32(N_atoms),
                    atom_forces.ravel(),
                    atom_energies.ravel(),
                ),
            )
            total_forces += atom_forces.sum(axis=1)
            total_energies += atom_energies.sum(axis=1)
        # Yukawa far-field fallback for atoms outside all grids
        # (electrostatic only - Born decays too fast to matter)
        if not is_born and self._has_yukawa:
            not_assigned = ~assigned
            n_not = int(cp.sum(not_assigned))
            if n_not > 0:
                yf, ye = self._yukawa_forces_gpu(
                    positions_gpu, charges_gpu, mask=not_assigned
                )
                total_forces += yf
                total_energies += ye
                if self._call_count < 3:
                    print(
                        f"    [FORCE VERBOSE] Yukawa fallback: "
                        f"{n_not}/{N_traj*N_atoms} atom-traj pairs "
                        f"({100*n_not/(N_traj*N_atoms):.1f}%)"
                    )
        # Log first few calls
        if self._call_count < 3 and not is_born:
            n_assigned = int(cp.sum(assigned))
            n_total = N_traj * N_atoms
            print(
                f"    [FORCE VERBOSE call#{self._call_count}] "
                f"{'ELEC' if not is_born else 'BORN'}: "
                f"{n_assigned}/{n_total} atom-traj pairs assigned to grids "
                f"({100*n_assigned/n_total:.1f}%)"
            )
            # Force magnitude stats
            f_mag = cp.linalg.norm(total_forces, axis=1)
            print(
                f"    [FORCE VERBOSE] |F| stats: "
                f"mean={float(f_mag.mean()):.8f} "
                f"max={float(f_mag.max()):.8f} "
                f"min={float(f_mag.min()):.8f} kBT/Å"
            )
        return total_forces, total_energies

    def _yukawa_forces_gpu(self, positions_gpu, charges_gpu, mask=None):
        """
        Analytical screened Coulomb force from receptor charge distribution.
        When multipole_expansion is set: monopole + dipole + quadrupole.
        Otherwise: monopole only (net charge Q).
        Receptor at origin. Force on atom i:
            F_i = -q_i * grad(phi)   [kBT/A, matches APBS grid forces]
        """
        N_traj, N_atoms, _ = positions_gpu.shape
        debye = self._debye
        r_mag = cp.linalg.norm(positions_gpu, axis=2)  # (N_traj, N_atoms)
        safe_r = cp.maximum(r_mag, 1.0)
        exp_term = cp.exp(-safe_r / debye)
        r_hat = positions_gpu / safe_r[:, :, None]
        # Monopole: phi = V_factor * exp(-r/λ) / r
        V_fac = self._V_factor
        phi = V_fac * exp_term / safe_r
        dphi_dr = V_fac * exp_term * (-1.0 / safe_r**2 - 1.0 / (safe_r * debye))
        # Dipole + Quadrupole (if multipole expansion available)
        if self._multipole is not None:
            p_gpu = self._mp_dipole_gpu  # (3,)
            Q_gpu = self._mp_quad_gpu  # (3, 3)
            fpe = self._mp_four_pi_eps
            lam = debye
            # Dipole: phi_1 = (p · r̂) / (4πε r²) × (1 + r/λ) × exp(-r/λ)
            p_dot_r = cp.sum(r_hat * p_gpu[None, None, :], axis=2)  # (N_traj, N_atoms)
            p_mag = float(cp.linalg.norm(p_gpu))
            if p_mag > 1e-9:
                phi_dip = p_dot_r / (fpe * safe_r**2) * (1.0 + safe_r / lam) * exp_term
                phi += phi_dip
                # d(phi_dip)/dr - leading radial term
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
            # Quadrupole: phi_2 = (r̂ᵀ Q r̂) / (4πε r³) × (1 + r/λ + r²/(3λ²)) × exp(-r/λ)
            q_mag = float(cp.linalg.norm(Q_gpu))
            if q_mag > 1e-9:
                # r̂ᵀ Q r̂ for each atom: (N_traj, N_atoms)
                # r_hat: (N_traj, N_atoms, 3)
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
                        - 1.0 / (safe_r**2 * lam**2)
                        - 1.0 / (3.0 * safe_r * lam**3)
                    )
                )
                dphi_dr += dphi_quad_dr
        # grad(phi) = d(phi)/dr * r_hat
        grad_phi = dphi_dr[:, :, None] * r_hat
        # F_i = -q_i * grad(phi)
        q_3d = charges_gpu[None, :, None]
        atom_forces = -q_3d * grad_phi
        atom_energies = charges_gpu[None, :] * phi
        if mask is not None:
            mask_3d = mask[:, :, None].astype(cp.float64)
            atom_forces = atom_forces * mask_3d
            atom_energies = atom_energies * mask.astype(cp.float64)
        forces = atom_forces.sum(axis=1)
        energies = atom_energies.sum(axis=1)
        return forces, energies

    def _wca_forces_gpu(self, lig_positions, centroids):
        """
        WCA (purely repulsive LJ) forces using PQR radii.

        Only activates for trajectories where centroid is close to receptor.
        sigma_ij = rec_radius_i + lig_radius_j (Lorentz combining rule).
        WCA cutoff: r < 2^(1/6) × sigma_ij (only repulsive part).
        Processed in chunks to limit GPU memory.
        """
        N_traj = lig_positions.shape[0]
        N_lig = lig_positions.shape[1]
        N_rec = self._rec_pos_gpu.shape[0]
        eps = self._lj_epsilon
        # Find trajectories close enough for LJ
        r_cen = cp.linalg.norm(centroids, axis=1)
        active = r_cen < self._lj_activation
        n_active = int(active.sum())
        lj_forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        if n_active == 0:
            return lj_forces
        active_idx = cp.where(active)[0]
        # Process in chunks to limit memory: (chunk × N_lig × N_rec × 3)
        CHUNK = max(1, min(50, int(2e9 / (N_lig * N_rec * 8 * 3))))
        for c0 in range(0, n_active, CHUNK):
            c1 = min(c0 + CHUNK, n_active)
            idx = active_idx[c0:c1]
            nc = len(idx)
            # lig_pos: (nc, N_lig, 3), rec_pos: (N_rec, 3)
            lp = lig_positions[idx]  # (nc, N_lig, 3)
            rp = self._rec_pos_gpu  # (N_rec, 3)
            # Pairwise distances: (nc, N_lig, N_rec, 3)
            diff = lp[:, :, None, :] - rp[None, None, :, :]  # broadcast
            r2 = (diff * diff).sum(axis=3)  # (nc, N_lig, N_rec)
            r = cp.sqrt(cp.maximum(r2, 1e-6))
            # sigma_ij = rec_radius + lig_radius
            sig = (
                self._lig_radii_gpu[None, :, None] + self._rec_radii_gpu[None, None, :]
            )
            # WCA cutoff: r < 2^(1/6) * sigma
            r_cut = 1.122462 * sig  # 2^(1/6) ≈ 1.122462
            in_range = r < r_cut
            # WCA force: F = eps * (12*(sig/r)^12 - 6*(sig/r)^6) / r^2 * r_vec
            sr = sig / r
            sr2 = sr * sr
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            f_mag = eps * (12.0 * sr12 - 6.0 * sr6) / r2  # (nc, N_lig, N_rec)
            f_mag = cp.where(in_range, f_mag, 0.0)
            # Force on ligand atom from each rec atom: f_mag * (lig - rec) / r
            f_vec = (
                f_mag[:, :, :, None] * diff / r[:, :, :, None]
            )  # (nc, N_lig, N_rec, 3)
            # Sum over rec atoms and lig atoms -> net force per trajectory
            net_f = f_vec.sum(axis=(1, 2))  # (nc, 3)
            lj_forces[idx] = net_f
        return lj_forces

    def __call__(self, lig_positions, lig_charges, R_matrices=None, centroids=None):
        """
        Compute net force, torque, energy on ligand for all trajectories.
        Born desolvation is computed in both directions:
        1. core_desolvation_force_on_1(state0, state1) -> rec Born on lig
        2. core_desolvation_force_on_1(state1, state0) -> lig Born on rec
        The force on rec from direction 2 is negated (Newton's 3rd) to get
        the reaction force on lig.
        Parameters
        ----------
        lig_positions : (N_traj, N_lig, 3) ligand atom positions in lab frame
        lig_charges   : (N_lig,) ligand atom charges
        R_matrices    : (N_traj, 3, 3) rotation matrices (lig frame -> lab)
        centroids     : (N_traj, 3) ligand centroid positions
        """
        N_traj, N_atoms, _ = lig_positions.shape
        forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        energies = cp.zeros((N_traj,), dtype=cp.float64)
        if self._call_count < 3:
            r_mag = cp.linalg.norm(lig_positions[:, 0, :], axis=1)  # centroid r
            print(
                f"    [ENGINE call#{self._call_count}] N_traj={N_traj} N_atoms={N_atoms}  "
                f"r_centroid: mean={float(r_mag.mean()):.3f} "
                f"min={float(r_mag.min()):.3f} max={float(r_mag.max()):.3f}"
            )
        # Electrostatic force on ligand from receptor field
        if self._elec_grids_gpu:
            f, e = self._eval_batch(
                lig_positions, lig_charges, self._elec_grids_gpu, 0.0, False
            )
            forces += f
            energies += e
            if self._call_count < 3:
                fm = float(cp.linalg.norm(f, axis=1).mean())
                print(f"    [COMPONENT] ELEC:   |F|_mean={fm:.6e} kBT/Å")
        elif self._has_yukawa:
            f, e = self._yukawa_forces_gpu(lig_positions, lig_charges)
            forces += f
            energies += e
        # Born desolvation direction 1: rec Born grid on lig atoms
        if self._born_grids_gpu:
            f, e = self._eval_batch(
                lig_positions, lig_charges, self._born_grids_gpu, self.alpha, True
            )
            forces += f
            energies += e
            if self._call_count < 3:
                fm = float(cp.linalg.norm(f, axis=1).mean())
                print(f"    [COMPONENT] BORN1:  |F|_mean={fm:.6e} kBT/Å")
        # Born desolvation direction 2: lig Born grid on rec atoms
        # Evaluates lig Born grid at rec atom positions -> force on rec.
        # Newton's 3rd law: force on lig = -force on rec.
        if (
            self._lig_born_grids_gpu
            and self._rec_pos_gpu is not None
            and R_matrices is not None
            and centroids is not None
            and self.alpha > 1e-12
        ):
            f2 = self._eval_born_reverse(R_matrices, centroids, N_traj)
            forces += f2
            if self._call_count < 3:
                fm = float(cp.linalg.norm(f2, axis=1).mean())
                print(
                    f"    [COMPONENT] BORN2:  |F|_mean={fm:.6e} kBT/Å  - lig Born on {self._rec_pos_gpu.shape[0]} rec atoms"
                )
        # Torque placeholder
        mask = cp.abs(lig_charges) > 1e-9
        torques = cp.zeros((N_traj, 3), dtype=cp.float64)
        # LJ (WCA repulsive) forces - only when atoms are very close
        if self._use_lj and self._rec_radii_gpu is not None:
            lj_f = self._wca_forces_gpu(lig_positions, centroids)
            forces += lj_f
        self._call_count += 1
        return forces, torques, energies

    def _eval_born_reverse(self, R_matrices, centroids, N_traj):
        """
        Born direction 2: evaluate lig Born grid at rec atom positions.
        For each trajectory:
          1. Transform rec atoms into lig frame: R^T @ (rec_pos - centroid)
          2. Evaluate lig Born grid -> per-atom force in lig frame
          3. Rotate back to lab frame: R @ F_lig
          4. Sum over rec atoms -> total force on rec
          5. Negate -> force on lig (Newton's 3rd)
        Chunked by trajectory to limit GPU memory (N_rec can be large).
        """
        N_rec = self._rec_pos_gpu.shape[0]
        result = cp.zeros((N_traj, 3), dtype=cp.float64)
        # Get ligand Born grid extent - skip if no rec atoms can reach it
        if self._lig_born_grids_gpu:
            g = self._lig_born_grids_gpu[0]
            lig_grid_radius = float(max(abs(g["lo"][0]), abs(g["hi"][0])))
        else:
            return result
        # Chunk size: limit to ~500 MB for the (chunk, N_rec, 3) array
        max_bytes = 500 * 1024 * 1024
        chunk_size = max(1, int(max_bytes / (N_rec * 3 * 8)))
        chunk_size = min(chunk_size, N_traj)
        for c0 in range(0, N_traj, chunk_size):
            c1 = min(c0 + chunk_size, N_traj)
            nc = c1 - c0
            # Skip chunk if all centroids are too far for any rec atom
            # to fall inside the ligand Born grid
            r_cen = cp.linalg.norm(centroids[c0:c1], axis=1)
            rec_max_r = float(cp.linalg.norm(self._rec_pos_gpu, axis=1).max())
            min_possible_dist = float(r_cen.min()) - rec_max_r
            if min_possible_dist > lig_grid_radius:
                continue  # no rec atoms can be inside lig Born grid
            # rec_pos_rel: (nc, N_rec, 3) = rec_pos - centroid
            rec_pos_rel = self._rec_pos_gpu[None, :, :] - centroids[c0:c1, None, :]
            # Transform to lig frame: R^T @ rec_pos_rel
            R_T = cp.swapaxes(R_matrices[c0:c1], 1, 2)
            rec_in_lig = cp.einsum("nij,nkj->nki", R_T, rec_pos_rel)
            # Evaluate lig Born grid at rec positions in lig frame
            f_lig, e_lig = self._eval_batch(
                rec_in_lig,
                self._rec_charges_gpu,
                self._lig_born_grids_gpu,
                self.alpha,
                True,
            )
            # Rotate back to lab frame: R @ f_lig
            f_lab = cp.einsum("nij,nj->ni", R_matrices[c0:c1], f_lig)
            # Newton's 3rd: force on lig = -force on rec
            result[c0:c1] = -f_lab
        return result
