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
α = 1/(4π) ≈ 0.0796. This term is computed in both directions. Direction 1
evaluates the receptor Born grid at the ligand atom positions. Direction 2
evaluates the ligand Born grid at the receptor atom positions. Direction 2 is
demanding on GPU memory for large receptors and is computed in 500 MB chunks.

The third contribution is the Yukawa monopole fallback. For atoms far from the
receptor (outside the APBS grid), the screened Coulomb force gives the correct
long-range behavior,

    F = q_rec × q_lig / (4πε) × (1/r² + 1/(rλ)) × exp(-r/λ) × r̂,

where λ is the Debye screening length.

For large ligands (for example thrombomodulin, 1652 atoms), evaluating forces
for all 10⁶ trajectories at once would exceed GPU memory, so the engine
batches automatically using

    N_batch = 4 GB / (N_lig_atoms × 150 bytes).

The figure of 150 bytes per atom accounts for roughly 6 internal arrays per
batch (positions, forces, energies, masks, and so on).
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


# CUDA kernel with one thread per (trajectory, atom) pair.

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
    double q = charges[traj * N_atoms + atom];
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
    Batch GPU force engine with a Yukawa far-field fallback.

    The argument elec_grids is a list of DXGrid objects for electrostatics,
    ordered finest first, and born_grids is the corresponding list for Born
    desolvation, also finest first. The argument alpha is the Born desolvation
    parameter. receptor_charge is the total charge of the receptor molecule in
    units of e, debye_length is the Debye screening length in angstrom, and
    sdie is the solvent dielectric constant.
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
        # the reference units used here is 0.000142 e²/(kBT·A).
        eps_s = sdie * VACUUM_PERMITTIVITY_KBT
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
        self._born_grids_gpu = self._upload_grids(born_grids)
        # The reference grid architecture uses one fine grid per molecule plus a
        # multipole or analytical field for atoms outside it. The coarse
        # electrostatic grid exists only to provide Dirichlet boundary
        # conditions for the fine grid's APBS solve (the bcfl map). At runtime
        # the fine grid handles atoms inside it and the Chebyshev or Yukawa
        # field handles atoms outside.
        #
        # We retain all APBS grids at runtime. The `assigned` mask in
        # _eval_batch ensures that each atom is processed by exactly one grid,
        # namely the finest grid that contains it. Atoms outside all grids fall
        # through to the Yukawa fallback. The previous code dropped every grid
        # but the finest, assuming the coarse grids existed only as APBS
        # boundary conditions. In fact the coarse APBS field is more accurate
        # than truncated multipole extrapolation in the intermediate range, and
        # it matches the BD2 runtime behavior.
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
        self._lig_born_grids_gpu = []
        self._rec_pos_gpu = None
        self._rec_charges_gpu = None
        if lig_born_grids and rec_positions is not None and rec_charges is not None:
            self._lig_born_grids_gpu = self._upload_grids(lig_born_grids)
            # We retain all ligand Born grids, by the same reasoning used for
            # the receptor side above. The `assigned` mask in _eval_batch makes
            # finest-first assignment correct for the multi-grid case, and the
            # coarse ligand Born grids give receptor atoms in the intermediate
            # range a proper APBS field.
            self._rec_pos_gpu = cp.asarray(
                rec_positions, dtype=cp.float64
            )  # shape (N_rec, 3)
            self._rec_charges_gpu = cp.asarray(
                rec_charges, dtype=cp.float64
            )  # shape (N_rec,)
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
        # Lennard-Jones (WCA repulsive) forces computed from the PQR radii.
        self._use_lj = use_lj
        self._enable_born2_torque = enable_born2_torque
        self._rec_radii_gpu = None
        self._lig_radii_gpu = None
        if use_lj and rec_radii is not None and lig_radii is not None:
            self._rec_radii_gpu = cp.asarray(rec_radii, dtype=cp.float64)
            self._lig_radii_gpu = cp.asarray(lig_radii, dtype=cp.float64)
            # WCA well depth in kBT units: 0.1 kcal/mol ÷ 0.5925 kcal/mol/kBT ≈ 0.17 kBT.
            self._lj_epsilon = 0.17
            # Activation radius beyond which the LJ term is skipped. It is only
            # computed when the ligand centroid is close to the receptor.
            _max_sig = float(self._rec_radii_gpu.max() + self._lig_radii_gpu.max())
            self._lj_activation = _max_sig * 2.5
            print(
                f"  LJ (WCA repulsive): {len(rec_radii)} rec + {len(lig_radii)} lig atoms, "
                f"activation={self._lj_activation:.1f} A"
            )

    def _upload_grids(self, grids):
        """Upload a list of DXGrid objects to the GPU, sorted finest first.

        The usable grid bounds account for two effects. First, the central
        difference used for the gradient probes the grid at ±0.5 of the
        spacing. Second, focused APBS grids built with bcfl=map have roughly 3
        outer shells contaminated by interpolation of the coarse-grid boundary
        conditions. The valid range is therefore [origin + 3·sp,
        origin + (nx-4)·sp]. This three-spacing margin prevents force spikes at
        the grid boundaries.
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
            # Boundary-condition-safe bounds set 3 spacings in from each edge.
            # The gradient probe needs 0.5 of a spacing, and the focused APBS
            # boundary conditions contaminate roughly 2 to 3 shells.
            sp_np = np.array([g.delta[i, i] for i in range(3)])
            dims = np.array([nx, ny, nz])
            margin = 3.0  # number of grid spacings in from the edge
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
        positions_gpu,  # shape (N_traj, N_atoms, 3)
        charges_gpu,  # shape (N_atoms,)
        grids_gpu,
        alpha: float,
        is_born: bool,
        centroids=None,  # shape (N_traj, 3), the reference point for torque accumulation
    ):
        """
        Evaluate the forces, torques, and energies for all trajectories.

        Atoms are assigned to grids finest first, and atoms left unassigned use
        the Yukawa fallback. Torque is computed about each trajectory's
        centroid. When centroids is None the torques are returned as zeros,
        which matches the behavior expected by legacy callers.
        """
        N_traj, N_atoms, _ = positions_gpu.shape
        total_forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        total_torques = cp.zeros((N_traj, 3), dtype=cp.float64)
        total_energies = cp.zeros((N_traj,), dtype=cp.float64)
        # Lever arms measured from each trajectory's centroid, used to
        # accumulate torque.
        if centroids is not None:
            lever_arms = positions_gpu - centroids[:, None, :]
        else:
            lever_arms = None
        if not grids_gpu:
            if not is_born and self._has_yukawa:
                yf, yt, ye = self._yukawa_forces_gpu(
                    positions_gpu, charges_gpu, centroids=centroids
                )
                total_forces += yf
                total_torques += yt
                total_energies += ye
            return total_forces, total_torques, total_energies
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
            # Per-atom torque about each trajectory's centroid, given by the
            # lever arm crossed with the force.
            if lever_arms is not None:
                atom_torques = cp.cross(lever_arms, atom_forces, axis=2)
                total_torques += atom_torques.sum(axis=1)
        # Yukawa far-field fallback for atoms outside all grids. This applies
        # to electrostatics only, because the Born term decays too fast to
        # matter at these distances.
        if not is_born and self._has_yukawa:
            not_assigned = ~assigned
            n_not = int(cp.sum(not_assigned))
            if n_not > 0:
                yf, yt, ye = self._yukawa_forces_gpu(
                    positions_gpu, charges_gpu, mask=not_assigned, centroids=centroids
                )
                total_forces += yf
                total_torques += yt
                total_energies += ye
                if self._call_count < 3:
                    print(
                        f"    [FORCE VERBOSE] Yukawa fallback: "
                        f"{n_not}/{N_traj*N_atoms} atom-traj pairs "
                        f"({100*n_not/(N_traj*N_atoms):.1f}%)"
                    )
        # Log the first few calls.
        if self._call_count < 3 and not is_born:
            n_assigned = int(cp.sum(assigned))
            n_total = N_traj * N_atoms
            print(
                f"    [FORCE VERBOSE call#{self._call_count}] "
                f"{'ELEC' if not is_born else 'BORN'}: "
                f"{n_assigned}/{n_total} atom-traj pairs assigned to grids "
                f"({100*n_assigned/n_total:.1f}%)"
            )
            # Statistics on the force magnitude.
            f_mag = cp.linalg.norm(total_forces, axis=1)
            print(
                f"    [FORCE VERBOSE] |F| stats: "
                f"mean={float(f_mag.mean()):.8f} "
                f"max={float(f_mag.max()):.8f} "
                f"min={float(f_mag.min()):.8f} kBT/Å"
            )
        return total_forces, total_torques, total_energies

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
        WCA (purely repulsive Lennard-Jones) forces from the PQR radii, with
        per-atom torque about each trajectory's centroid.

        This activates only for trajectories whose centroid is close to the
        receptor. The pair size is σ_ij = rec_radius_i + lig_radius_j from the
        Lorentz combining rule, and the WCA cutoff keeps only the repulsive
        part, r < 2^(1/6) × σ_ij. The work is processed in chunks to limit GPU
        memory.

        Returns a (force, torque) pair, each of shape (N_traj, 3).
        """
        N_traj = lig_positions.shape[0]
        N_lig = lig_positions.shape[1]
        N_rec = self._rec_pos_gpu.shape[0]
        eps = self._lj_epsilon
        # Select the trajectories close enough to need the LJ term.
        r_cen = cp.linalg.norm(centroids, axis=1)
        active = r_cen < self._lj_activation
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
            nc = len(idx)
            # Ligand positions lp have shape (nc, N_lig, 3) and receptor
            # positions rp have shape (N_rec, 3).
            lp = lig_positions[idx]  # shape (nc, N_lig, 3)
            rp = self._rec_pos_gpu  # shape (N_rec, 3)
            # Pairwise displacement vectors, shape (nc, N_lig, N_rec, 3).
            diff = lp[:, :, None, :] - rp[None, None, :, :]  # broadcast
            r2 = (diff * diff).sum(axis=3)  # shape (nc, N_lig, N_rec)
            r = cp.sqrt(cp.maximum(r2, 1e-6))
            # Pair size σ_ij = lig_radius + rec_radius.
            sig = (
                self._lig_radii_gpu[None, :, None] + self._rec_radii_gpu[None, None, :]
            )
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
            f_mag = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / r2  # shape (nc, N_lig, N_rec)
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
        state1, state0), is the ligand Born force on the receptor. The force on
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
        forces = cp.zeros((N_traj, 3), dtype=cp.float64)
        torques = cp.zeros((N_traj, 3), dtype=cp.float64)
        energies = cp.zeros((N_traj,), dtype=cp.float64)
        if self._call_count < 3:
            r_mag = cp.linalg.norm(
                lig_positions[:, 0, :], axis=1
            )  # radial distance of the centroid
            print(
                f"    [ENGINE call#{self._call_count}] N_traj={N_traj} N_atoms={N_atoms}  "
                f"r_centroid: mean={float(r_mag.mean()):.3f} "
                f"min={float(r_mag.min()):.3f} max={float(r_mag.max()):.3f}"
            )
        # Electrostatic force on the ligand from the receptor field.
        if self._elec_grids_gpu:
            f, t, e = self._eval_batch(
                lig_positions,
                lig_charges,
                self._elec_grids_gpu,
                0.0,
                False,
                centroids=centroids,
            )
            forces += f
            torques += t
            energies += e
            if self._call_count < 3:
                fm = float(cp.linalg.norm(f, axis=1).mean())
                tm = float(cp.linalg.norm(t, axis=1).mean())
                print(
                    f"    [COMPONENT] ELEC:   |F|_mean={fm:.6e} kBT/Å  "
                    f"|T|_mean={tm:.6e} kBT"
                )
        elif self._has_yukawa:
            f, t, e = self._yukawa_forces_gpu(
                lig_positions, lig_charges, centroids=centroids
            )
            forces += f
            torques += t
            energies += e
        # Born desolvation, direction 1: the receptor Born grid acting on the
        # ligand atoms.
        if self._born_grids_gpu:
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
        if self._use_lj and self._rec_radii_gpu is not None:
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

    def _eval_born_reverse(self, R_matrices, centroids, N_traj):
        """
        Born direction 2: evaluate the ligand Born grid at the receptor atom
        positions.

        For each trajectory the receptor atoms are first transformed into the
        ligand frame as Rᵀ (rec_pos - centroid). The ligand Born grid is then
        evaluated to give the per-atom force on the receptor atoms expressed in
        the ligand frame, which is rotated back to the lab frame as R F_lig. By
        Newton's third law the force on the ligand is minus the force on the
        receptor, and likewise the torque on the ligand about its centre is
        minus the torque on the receptor about the same point, matching the
        b_torque11 contribution in BD2's add_core_forces. The work is chunked
        by trajectory to limit GPU memory, since N_rec can be large.

        Returns result_f of shape (N_traj, 3), the net force on the ligand from
        direction 2, and result_t of shape (N_traj, 3), the corresponding net
        torque on the ligand.
        """
        N_rec = self._rec_pos_gpu.shape[0]
        result_f = cp.zeros((N_traj, 3), dtype=cp.float64)
        result_t = cp.zeros((N_traj, 3), dtype=cp.float64)
        # Get the extent of the ligand Born grid and skip if no receptor atom
        # can reach it.
        if self._lig_born_grids_gpu:
            g = self._lig_born_grids_gpu[0]
            lig_grid_radius = float(max(abs(g["lo"][0]), abs(g["hi"][0])))
        else:
            return result_f, result_t
        # Choose the chunk size to keep the (chunk, N_rec, 3) array under about
        # 500 MB.
        max_bytes = 500 * 1024 * 1024
        chunk_size = max(1, int(max_bytes / (N_rec * 3 * 8)))
        chunk_size = min(chunk_size, N_traj)
        for c0 in range(0, N_traj, chunk_size):
            c1 = min(c0 + chunk_size, N_traj)
            nc = c1 - c0
            # Skip the chunk if all centroids are too far for any receptor atom
            # to fall inside the ligand Born grid.
            r_cen = cp.linalg.norm(centroids[c0:c1], axis=1)
            rec_max_r = float(cp.linalg.norm(self._rec_pos_gpu, axis=1).max())
            min_possible_dist = float(r_cen.min()) - rec_max_r
            if min_possible_dist > lig_grid_radius:
                continue  # no receptor atom can be inside the ligand Born grid
            # Receptor positions relative to the ligand centroid,
            # rec_pos_rel = rec_pos - centroid, with shape (nc, N_rec, 3).
            rec_pos_rel = self._rec_pos_gpu[None, :, :] - centroids[c0:c1, None, :]
            # Transform into the ligand frame as Rᵀ rec_pos_rel.
            R_T = cp.swapaxes(R_matrices[c0:c1], 1, 2)
            rec_in_lig = cp.einsum("nij,nkj->nki", R_T, rec_pos_rel)
            # Evaluate the ligand Born grid at the receptor positions in the
            # ligand frame. Setting zero_centroids places the torque reference
            # at the origin of the ligand frame, the ligand's centre.
            # _eval_batch then returns sum_j r_j_lig × f_j_lig, the torque on
            # the receptor atoms about the ligand's centre, expressed in the
            # ligand frame.
            zero_centroids = cp.zeros((nc, 3), dtype=cp.float64)
            f_lig, t_lig, e_lig = self._eval_batch(
                rec_in_lig,
                self._rec_charges_gpu,
                self._lig_born_grids_gpu,
                self.alpha,
                True,
                centroids=zero_centroids,
            )
            # Rotate the force to the lab frame, then negate it so that
            # Newton's third law gives the force on the ligand.
            f_lab = cp.einsum("nij,nj->ni", R_matrices[c0:c1], f_lig)
            result_f[c0:c1] = -f_lab
            # The torque on the ligand about its centre is minus the torque on
            # the receptor about the same point, since Newton's third law holds
            # for torques taken about a shared point. Rotate the ligand-frame
            # torque to the lab frame, then negate it.
            t_lab = cp.einsum("nij,nj->ni", R_matrices[c0:c1], t_lig)
            result_t[c0:c1] = -t_lab
        return result_f, result_t
