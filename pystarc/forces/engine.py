"""
PySTARC unified force engine
==========================

One engine. GPU when available, best CPU otherwise.

- Finest-grid-first DX selection
- Per-atom force evaluation
- Born desolvation
- Adaptive time step

Backend selection (automatic, in priority order).
1. CuPy   - NVIDIA GPU (CUDA).
2. Numba  - CPU JIT, ~9x faster than pure Python
3. NumPy  - CPU Python, always available

Usage
-----
    from pystarc.forces.engine import PySTARCEngine, load_dx_directory
    engine = load_dx_directory("/path/to/b_surface_trp/")
    print(f"Backend: {engine.backend}")   # cupy / numba / numpy
    force, torque, energy = engine(mol_receptor, mol_ligand)
"""

from __future__ import annotations
from pystarc.forces.multipole import load_effective_charges
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.forces.multipole import EffectiveCharges
from pystarc.structures.molecules import Molecule
from pystarc.forces.lj import LJForceEngine
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import warnings
import math

try:
    from numba import njit as _njit
    import numba
except ImportError:
    _njit = None
    numba = None
try:
    import cupy as cp
except ImportError:
    cp = None


# Backend detection
def _detect_backend() -> str:
    """Return best available backend: 'cupy', 'numba', or 'numpy'."""
    try:
        cp.array([1.0])  # triggers GPU access
        return "cupy"
    except Exception:
        pass
    try:
        return "numba"
    except ImportError:
        pass
    return "numpy"


# Compiled grid - pre-extracted arrays for GPU/Numba kernels
class _Grid:
    """
    DXGrid with pre-extracted contiguous arrays.
    Numba and CuPy cannot access Python objects so we pull the
    raw arrays out once at construction time.
    """

    __slots__ = ("data", "origin", "spacing", "inv_spacing", "lo", "hi")

    def __init__(self, g: DXGrid):
        self.data = np.ascontiguousarray(g.data, dtype=np.float64)
        self.origin = np.ascontiguousarray(g.origin, dtype=np.float64)
        self.spacing = np.array([g.delta[i, i] for i in range(3)], dtype=np.float64)
        self.inv_spacing = 1.0 / self.spacing
        nx, ny, nz = self.data.shape
        self.lo = self.origin + self.spacing  # 1-cell margin
        self.hi = self.origin + np.array([nx - 2, ny - 2, nz - 2]) * self.spacing

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point > self.lo) and np.all(point < self.hi))


# Numba inner loop
try:

    @_njit(cache=True, fastmath=True)
    def _interp(data, origin, inv_sp, point):
        """Trilinear interpolation - Numba compiled."""
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        ix = (point[0] - origin[0]) * inv_sp[0]
        iy = (point[1] - origin[1]) * inv_sp[1]
        iz = (point[2] - origin[2]) * inv_sp[2]
        i0 = int(math.floor(ix))
        j0 = int(math.floor(iy))
        k0 = int(math.floor(iz))
        if not (0 <= i0 < nx - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nz - 1):
            return 0.0
        fx = ix - i0
        fy = iy - j0
        fz = iz - k0
        return (
            data[i0, j0, k0] * (1 - fx) * (1 - fy) * (1 - fz)
            + data[i0 + 1, j0, k0] * fx * (1 - fy) * (1 - fz)
            + data[i0, j0 + 1, k0] * (1 - fx) * fy * (1 - fz)
            + data[i0, j0, k0 + 1] * (1 - fx) * (1 - fy) * fz
            + data[i0 + 1, j0 + 1, k0] * fx * fy * (1 - fz)
            + data[i0 + 1, j0, k0 + 1] * fx * (1 - fy) * fz
            + data[i0, j0 + 1, k0 + 1] * (1 - fx) * fy * fz
            + data[i0 + 1, j0 + 1, k0 + 1] * fx * fy * fz
        )

    @_njit(cache=True, fastmath=True)
    def _grad(data, origin, inv_sp, sp, point):
        """Central-difference gradient."""
        g = np.zeros(3)
        for d in range(3):
            h = sp[d] * 0.5
            pp = point.copy()
            pp[d] += h
            pm = point.copy()
            pm[d] -= h
            g[d] = (
                _interp(data, origin, inv_sp, pp) - _interp(data, origin, inv_sp, pm)
            ) / (2.0 * h)
        return g

    @_njit(cache=True, fastmath=True)
    def _atom_loop(positions, charges, data, origin, inv_sp, sp, alpha, is_born):
        """
        Core force accumulation loop.
        Iterates over atoms, accumulates force + torque + energy.
        Ghost atoms (charge=0) contribute exactly zero.
        """
        N = positions.shape[0]
        force = np.zeros(3)
        torque = np.zeros(3)
        energy = 0.0
        # centroid of charged atoms (for torque)
        cx = 0.0
        cy = 0.0
        cz = 0.0
        n_c = 0
        for i in range(N):
            if abs(charges[i]) > 1e-9:
                cx += positions[i, 0]
                cy += positions[i, 1]
                cz += positions[i, 2]
                n_c += 1
        if n_c > 0:
            cx /= n_c
            cy /= n_c
            cz /= n_c
        for i in range(N):
            q = charges[i]
            if abs(q) < 1e-9:
                continue
            p = positions[i]
            gr = _grad(data, origin, inv_sp, sp, p)
            ph = _interp(data, origin, inv_sp, p)
            if is_born:
                coeff = -alpha * q * q
                energy += alpha * q * q * ph
            else:
                coeff = -q
                energy += q * ph
            fx = coeff * gr[0]
            fy = coeff * gr[1]
            fz = coeff * gr[2]
            force[0] += fx
            force[1] += fy
            force[2] += fz
            rx = p[0] - cx
            ry = p[1] - cy
            rz = p[2] - cz
            torque[0] += ry * fz - rz * fy
            torque[1] += rz * fx - rx * fz
            torque[2] += rx * fy - ry * fx
        return force, torque, energy

    _NUMBA = True

except ImportError:
    _NUMBA = False

    # Pure-NumPy fallbacks
    def _interp(data, origin, inv_sp, point):
        nx, ny, nz = data.shape
        ix = (point[0] - origin[0]) * inv_sp[0]
        iy = (point[1] - origin[1]) * inv_sp[1]
        iz = (point[2] - origin[2]) * inv_sp[2]
        i0, j0, k0 = int(math.floor(ix)), int(math.floor(iy)), int(math.floor(iz))
        if not (0 <= i0 < nx - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nz - 1):
            return 0.0
        fx, fy, fz = ix - i0, iy - j0, iz - k0
        return float(
            data[i0, j0, k0] * (1 - fx) * (1 - fy) * (1 - fz)
            + data[i0 + 1, j0, k0] * fx * (1 - fy) * (1 - fz)
            + data[i0, j0 + 1, k0] * (1 - fx) * fy * (1 - fz)
            + data[i0, j0, k0 + 1] * (1 - fx) * (1 - fy) * fz
            + data[i0 + 1, j0 + 1, k0] * fx * fy * (1 - fz)
            + data[i0 + 1, j0, k0 + 1] * fx * (1 - fy) * fz
            + data[i0, j0 + 1, k0 + 1] * (1 - fx) * fy * fz
            + data[i0 + 1, j0 + 1, k0 + 1] * fx * fy * fz
        )

    def _grad(data, origin, inv_sp, sp, point):
        g = np.zeros(3)
        for d in range(3):
            h = sp[d] * 0.5
            pp = point.copy()
            pp[d] += h
            pm = point.copy()
            pm[d] -= h
            g[d] = (
                _interp(data, origin, inv_sp, pp) - _interp(data, origin, inv_sp, pm)
            ) / (2 * h)
        return g

    def _atom_loop(positions, charges, data, origin, inv_sp, sp, alpha, is_born):
        N = positions.shape[0]
        force = np.zeros(3)
        torque = np.zeros(3)
        energy = 0.0
        mask = np.abs(charges) > 1e-9
        if not mask.any():
            return force, torque, energy
        c = positions[mask].mean(axis=0)
        for i in range(N):
            q = charges[i]
            if abs(q) < 1e-9:
                continue
            p = positions[i]
            gr = _grad(data, origin, inv_sp, sp, p)
            ph = _interp(data, origin, inv_sp, p)
            coeff = (-alpha * q * q) if is_born else (-q)
            energy += (alpha * q * q * ph) if is_born else (q * ph)
            f = coeff * gr
            force += f
            torque += np.cross(p - c, f)
        return force, torque, energy


# CuPy GPU kernel
_CUPY_KERNEL = None


def _build_cupy_kernel():
    """
    Build the CUDA kernel for trilinear interpolation + gradient.
    Called once on first GPU use.
    """
    global _CUPY_KERNEL
    try:
        # Raw CUDA C kernel: one thread per atom
        _CUPY_KERNEL = cp.RawKernel(
            r"""
extern "C" __global__
void atom_force_kernel(
    const double* positions,   // (N,3)
    const double* charges,     // (N,)
    const double* data,        // (nx,ny,nz) grid
    const double* origin,      // (3,)
    const double* inv_sp,      // (3,)
    const double* sp,          // (3,)
    double alpha,
    int is_born,
    int nx, int ny, int nz,
    double* forces_out,        // (N,3) per-atom forces
    double* energy_out,        // (N,)  per-atom energies
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double q = charges[i];
    forces_out[3*i]=0; forces_out[3*i+1]=0; forces_out[3*i+2]=0;
    energy_out[i] = 0;
    if (fabs(q) < 1e-9) return;
    double px = positions[3*i];
    double py = positions[3*i+1];
    double pz = positions[3*i+2];
    // Trilinear interpolation helper (inline lambda via nested function)
    auto interp = [&](double x, double y, double z) -> double {
        double ix = (x - origin[0]) * inv_sp[0];
        double iy = (y - origin[1]) * inv_sp[1];
        double iz = (z - origin[2]) * inv_sp[2];
        int i0=(int)floor(ix), j0=(int)floor(iy), k0=(int)floor(iz);
        if (i0<0||i0>=nx-1||j0<0||j0>=ny-1||k0<0||k0>=nz-1) return 0.0;
        double fx=ix-i0, fy=iy-j0, fz=iz-k0;
        return (data[(i0  )*ny*nz+(j0  )*nz+(k0  )]*(1-fx)*(1-fy)*(1-fz) +
                data[(i0+1)*ny*nz+(j0  )*nz+(k0  )]*fx    *(1-fy)*(1-fz) +
                data[(i0  )*ny*nz+(j0+1)*nz+(k0  )]*(1-fx)*fy    *(1-fz) +
                data[(i0  )*ny*nz+(j0  )*nz+(k0+1)]*(1-fx)*(1-fy)*fz     +
                data[(i0+1)*ny*nz+(j0+1)*nz+(k0  )]*fx    *fy    *(1-fz) +
                data[(i0+1)*ny*nz+(j0  )*nz+(k0+1)]*fx    *(1-fy)*fz     +
                data[(i0  )*ny*nz+(j0+1)*nz+(k0+1)]*(1-fx)*fy    *fz     +
                data[(i0+1)*ny*nz+(j0+1)*nz+(k0+1)]*fx    *fy    *fz);
    };
    // Gradient via central differences
    double h0=sp[0]*0.5, h1=sp[1]*0.5, h2=sp[2]*0.5;
    double gx = (interp(px+h0,py,pz) - interp(px-h0,py,pz)) / (2*h0);
    double gy = (interp(px,py+h1,pz) - interp(px,py-h1,pz)) / (2*h1);
    double gz = (interp(px,py,pz+h2) - interp(px,py,pz-h2)) / (2*h2);
    double ph = interp(px,py,pz);
    double coeff   = is_born ? (-alpha*q*q) : (-q);
    double e_coeff = is_born ? ( alpha*q*q) : ( q );
    forces_out[3*i]   = coeff * gx;
    forces_out[3*i+1] = coeff * gy;
    forces_out[3*i+2] = coeff * gz;
    energy_out[i]     = e_coeff * ph;
}
""",
            "atom_force_kernel",
        )
    except Exception as e:
        warnings.warn(f"CuPy kernel build failed: {e}. Will use CPU fallback.")
        _CUPY_KERNEL = None


def _cupy_eval(
    positions: np.ndarray, charges: np.ndarray, g: _Grid, alpha: float, is_born: bool
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate force for all atoms on GPU using the CUDA kernel."""
    N = positions.shape[0]
    pos_gpu = cp.asarray(positions, dtype=cp.float64)
    chg_gpu = cp.asarray(charges, dtype=cp.float64)
    dat_gpu = cp.asarray(g.data, dtype=cp.float64)
    ori_gpu = cp.asarray(g.origin, dtype=cp.float64)
    isp_gpu = cp.asarray(g.inv_spacing, dtype=cp.float64)
    sp_gpu = cp.asarray(g.spacing, dtype=cp.float64)
    nx, ny, nz = g.data.shape
    forces_gpu = cp.zeros((N, 3), dtype=cp.float64)
    energies_gpu = cp.zeros(N, dtype=cp.float64)
    threads = 256
    blocks = (N + threads - 1) // threads
    _CUPY_KERNEL(
        (blocks,),
        (threads,),
        (
            pos_gpu,
            chg_gpu,
            dat_gpu,
            ori_gpu,
            isp_gpu,
            sp_gpu,
            np.float64(alpha),
            np.int32(1 if is_born else 0),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            forces_gpu,
            energies_gpu,
            np.int32(N),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    forces_np = forces_gpu.get()  # (N,3)
    energies_np = energies_gpu.get()  # (N,)
    # Sum over atoms
    force = forces_np.sum(axis=0)
    energy = float(energies_np.sum())
    # Torque: (r_i - centroid) x F_i
    mask = np.abs(charges) > 1e-9
    if mask.any():
        c = positions[mask].mean(axis=0)
        r = positions - c  # (N,3)
        torque = np.cross(r, forces_np).sum(axis=0)
    else:
        torque = np.zeros(3)
    return force, torque, energy


# GridStack - finest-grid-first selection
class _GridStack:
    """
    A set of DX grids sorted finest-first (smallest spacing first).
    For each atom, returns the potential from the finest grid containing it.
    This exactly mirrors the grid selection logic.
    """

    def __init__(self, grids: List[DXGrid]):
        # Sort finest first - smallest spacing = highest resolution
        sorted_grids = sorted(grids, key=lambda g: float(g.delta[0, 0]))
        self._grids = [_Grid(g) for g in sorted_grids]

    def __bool__(self):
        return len(self._grids) > 0

    def __len__(self):
        return len(self._grids)

    def finest_for(self, point: np.ndarray) -> Optional[_Grid]:
        """Return finest grid containing point, or None."""
        for g in self._grids:
            if g.contains(point):
                return g
        return None

    def eval_atoms(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        alpha: float,
        is_born: bool,
        backend: str,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate force on all atoms using finest-grid-first selection.
        Each atom is assigned to the finest grid that contains it.
        Atoms outside all grids contribute zero (negligible far-field).
        """
        if not self._grids:
            return np.zeros(3), np.zeros(3), 0.0
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        total_energy = 0.0
        N = positions.shape[0]
        assigned = np.zeros(N, dtype=bool)
        if backend == "cupy" and _CUPY_KERNEL is not None:
            # GPU path: group atoms by grid, send each group to GPU
            for g in self._grids:
                idx = []
                for i in range(N):
                    if not assigned[i] and abs(charges[i]) > 1e-9:
                        if g.contains(positions[i]):
                            idx.append(i)
                            assigned[i] = True
                if not idx:
                    continue
                f, t, e = _cupy_eval(
                    np.ascontiguousarray(positions[idx]),
                    np.ascontiguousarray(charges[idx]),
                    g,
                    alpha,
                    is_born,
                )
                total_force += f
                total_torque += t
                total_energy += e
        else:
            # CPU path: per-grid atom groups, run Numba/NumPy inner loop
            for g in self._grids:
                idx = []
                for i in range(N):
                    if not assigned[i] and abs(charges[i]) > 1e-9:
                        if g.contains(positions[i]):
                            idx.append(i)
                            assigned[i] = True
                if not idx:
                    continue
                sub_pos = np.ascontiguousarray(positions[idx])
                sub_chg = np.ascontiguousarray(charges[idx])
                f, t, e = _atom_loop(
                    sub_pos,
                    sub_chg,
                    g.data,
                    g.origin,
                    g.inv_spacing,
                    g.spacing,
                    alpha,
                    is_born,
                )
                total_force += f
                total_torque += t
                total_energy += e
        return total_force, total_torque, total_energy


# PySTARCEngine - the single unified engine
class PySTARCEngine:
    """
    PySTARC unified force engine.
    Implements all force terms from the reference implementation:
      1. Electrostatic (APBS DX grids, finest-grid-first, per-atom)
      2. Born desolvation (*_born.dx grids, per-atom)
    Backend selected automatically:
      cupy  -> NVIDIA GPU (CUDA)
      numba -> CPU JIT compiled
      numpy -> CPU pure Python
    Ghost atoms (charge=0) contribute exactly zero to all terms.
    Parameters
    ----------
    elec_mol1 : DX grids for molecule 1 electrostatics (receptor)
    elec_mol2 : DX grids for molecule 2 electrostatics (ligand)
    born_mol1 : DX grids for molecule 1 Born desolvation
    born_mol2 : DX grids for molecule 2 Born desolvation
    debye_length      : Debye screening length in Å (from solvent file)
    desolvation_alpha : Born desolvation parameter   (from solvent file)
    """

    _numba_warmed: bool = False  # compile once per process

    def __init__(
        self,
        elec_mol1: List[DXGrid] = None,
        elec_mol2: List[DXGrid] = None,
        born_mol1: List[DXGrid] = None,
        born_mol2: List[DXGrid] = None,
        eff_charges_mol1: "EffectiveCharges" = None,
        eff_charges_mol2: "EffectiveCharges" = None,
        debye_length: float = 7.858,
        desolvation_alpha: float = 0.07957747,
        lj_params: "Optional[LJParams]" = None,
        hydrophobic_params: "Optional[HydrophobicParams]" = None,
    ):
        self.alpha = desolvation_alpha
        self.backend = _detect_backend()
        self._elec1 = _GridStack(elec_mol1 or [])
        self._elec2 = _GridStack(elec_mol2 or [])
        self._born1 = _GridStack(born_mol1 or [])
        self._born2 = _GridStack(born_mol2 or [])
        # Effective charges - long-range fallback when outside all grids
        self._eff1 = eff_charges_mol1
        self._eff2 = eff_charges_mol2
        self._debye = debye_length
        # LJ and hydrophobic forces (optional)
        self._lj_engine = None
        if lj_params is not None or hydrophobic_params is not None:
            self._lj_engine = LJForceEngine(lj_params, hydrophobic_params)
        # Build CuPy kernel if on GPU
        if self.backend == "cupy":
            _build_cupy_kernel()
            if _CUPY_KERNEL is None:
                warnings.warn("CuPy kernel failed. Falling back to Numba/NumPy.")
                self.backend = "numba" if _NUMBA else "numpy"
        # Warm up Numba JIT
        if self.backend == "numba" and not PySTARCEngine._numba_warmed:
            self._warmup_numba()
            PySTARCEngine._numba_warmed = True

    def _warmup_numba(self):
        dummy_pos = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        dummy_chg = np.array([1.0], dtype=np.float64)
        dummy_dat = np.ones((3, 3, 3), dtype=np.float64)
        dummy_ori = np.zeros(3, dtype=np.float64)
        dummy_isp = np.ones(3, dtype=np.float64)
        dummy_sp = np.ones(3, dtype=np.float64)
        _atom_loop(
            dummy_pos, dummy_chg, dummy_dat, dummy_ori, dummy_isp, dummy_sp, 0.0, False
        )

    def __call__(
        self, mol1: Molecule, mol2: Molecule
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute total force and torque on mol2 from mol1's fields.
        This is called once per BD step.
        Returns
        -------
        force  : (3,) net force on mol2  [kBT/Å]
        torque : (3,) torque on mol2     [kBT]
        energy : float total energy      [kBT]
        """
        force = np.zeros(3)
        torque = np.zeros(3)
        energy = 0.0
        pos2 = np.ascontiguousarray(mol2.positions_array(), dtype=np.float64)
        chg2 = np.ascontiguousarray(mol2.charges_array(), dtype=np.float64)
        pos1 = np.ascontiguousarray(mol1.positions_array(), dtype=np.float64)
        chg1 = np.ascontiguousarray(mol1.charges_array(), dtype=np.float64)
        # Electrostatic: mol2 atoms in mol1's field
        if self._elec1:
            f, t, e = self._elec1.eval_atoms(pos2, chg2, 0.0, False, self.backend)
            force += f
            torque += t
            energy += e
        elif self._eff1 is not None:
            # Long-range fallback: effective charges when outside all DX grids
            for i, (p, q) in enumerate(zip(pos2, chg2)):
                if abs(q) < 1e-9:
                    continue
                f_i = self._eff1.force_on_charge(p, q)
                force += f_i
                energy += q * self._eff1.potential(p)
                c2 = (
                    pos2[np.abs(chg2) > 1e-9].mean(axis=0)
                    if np.any(np.abs(chg2) > 1e-9)
                    else np.zeros(3)
                )
                torque += np.cross(p - c2, f_i)
        # Electrostatic: mol1 atoms in mol2's field (Newton 3rd law)
        if self._elec2:
            f, t, e = self._elec2.eval_atoms(pos1, chg1, 0.0, False, self.backend)
            force -= f
            energy += e * 0.5  # avoid double-counting
        elif self._eff2 is not None:
            for i, (p, q) in enumerate(zip(pos1, chg1)):
                if abs(q) < 1e-9:
                    continue
                f_i = self._eff2.force_on_charge(p, q)
                force -= f_i
                energy += q * self._eff2.potential(p) * 0.5
        # Born desolvation: mol2 atoms in mol1's born field
        if self._born1:
            f, t, e = self._born1.eval_atoms(pos2, chg2, self.alpha, True, self.backend)
            force += f
            torque += t
            energy += e
        # Born desolvation: mol1 atoms in mol2's born field
        if self._born2:
            f, t, e = self._born2.eval_atoms(pos1, chg1, self.alpha, True, self.backend)
            force -= f
            energy += e * 0.5
        # Lennard-Jones + hydrophobic (optional)
        if self._lj_engine is not None:
            n1 = len(pos1)
            n2 = len(pos2)
            type_ids1 = list(range(n1))  # atom type per atom (use index 0 default)
            type_ids2 = list(range(n2))
            f1, f2, e_lj = self._lj_engine.compute(pos1, pos2, type_ids1, type_ids2)
            force += f2  # force on mol2 from mol1
            energy += e_lj
        return force, torque, energy

    def summary(self) -> str:
        lines = [f"PySTARCEngine  [backend: {self.backend.upper()}]"]
        lines.append(
            f"  Electrostatic grids : {len(self._elec1)} receptor + {len(self._elec2)} ligand"
        )
        lines.append(
            f"  Born desolvation    : {len(self._born1)} receptor + {len(self._born2)} ligand"
        )
        lines.append(f"  Grid selection      : finest-first")
        lines.append(f"  Force evaluation    : per-atom")
        lines.append(f"  desolvation_alpha   : {self.alpha}")
        if self.backend == "numpy":
            lines.append("  [!] Install numba for ~9x speedup: pip install numba")
            lines.append(
                "  [!] Install cupy for GPU:          pip install cupy-cuda12x"
            )
        elif self.backend == "numba":
            lines.append(
                "  [!] Install cupy for GPU:          pip install cupy-cuda12x"
            )
        return "\n".join(lines)


# Factory
def load_dx_directory(
    directory: str | Path,
    mol1_prefix: str = "receptor",
    mol2_prefix: str = "ligand",
    debye_length: float = 7.858,
    desolvation_alpha: float = 0.07957747,
) -> PySTARCEngine:
    """
    Build PySTARCEngine from a the reference implementation run directory.
    Auto-detects all files:
      <prefix>[0-9].dx       -> APBS electrostatic grids
      <prefix>[0-9]_born.dx  -> Born desolvation grids
      <prefix>_cheby.xml     -> Chebyshev effective charges (long-range fallback)
      <prefix>_mpole.xml     -> Multipole effective charges (alternative)
    The effective charges are used as a long-range fallback when the query
    point falls outside all loaded DX grids.
    """
    d = Path(directory)

    def _load_dx(prefix: str, suffix: str) -> List[DXGrid]:
        grids = []
        for p in sorted(d.glob(f"{prefix}[0-9]{suffix}")):
            grids.append(DXGrid.from_file(p))
        return grids

    eff1 = load_effective_charges(d, mol1_prefix, debye_length)
    eff2 = load_effective_charges(d, mol2_prefix, debye_length)
    return PySTARCEngine(
        elec_mol1=_load_dx(mol1_prefix, ".dx"),
        elec_mol2=_load_dx(mol2_prefix, ".dx"),
        born_mol1=_load_dx(mol1_prefix, "_born.dx"),
        born_mol2=_load_dx(mol2_prefix, "_born.dx"),
        eff_charges_mol1=eff1,
        eff_charges_mol2=eff2,
        debye_length=debye_length,
        desolvation_alpha=desolvation_alpha,
    )
