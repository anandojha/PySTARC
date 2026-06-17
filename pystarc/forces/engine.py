"""
PySTARC unified force engine.

This module provides a single force engine that runs on the GPU when one is
available and otherwise picks the fastest CPU path. It selects the finest grid
first for each atom, evaluates forces per atom, includes Born desolvation, and
supports an adaptive time step.

The backend is chosen automatically in order of preference. CuPy runs on an
NVIDIA GPU through CUDA. Numba is a just-in-time compiled CPU path that is about
9 times faster than pure Python. NumPy is the pure-Python CPU path and is always
available.

To use the engine, load a run directory and call the engine with the two
molecules. For example, build an engine with
load_dx_directory("/path/to/b_surface_trp/"), inspect engine.backend to see
whether it is running on cupy, numba, or numpy, and then call
engine(mol_receptor, mol_ligand) to obtain the force, torque, and energy.
"""

from __future__ import annotations
from pystarc.forces.multipole import load_effective_charges
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.forces.multipole import EffectiveCharges
from pystarc.structures.molecules import Molecule
from pystarc.forces.lj import LJForceEngine
from pystarc.global_defs.constants import KCAL_PER_MOL_TO_KBT
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import warnings
import math

try:
    from numba import njit as _njit
    import numba
except ImportError:
    # Defensive fallback in case numba is missing or version-incompatible.
    # Without this stub, the line "@_njit(cache=True, ...)" in the kernels below
    # would crash with a TypeError at module import. The stub decorator returns
    # the function unchanged, which lets the module load so that callers can fall
    # back to the pure-NumPy code path. The production environment has numba, so
    # this only matters in test sandboxes, partial installs, or fresh clusters
    # where numba is not yet available.
    def _njit(*args, **kwargs):
        # Support two call patterns. The first is "@_njit" with no parentheses,
        # where args is (function,) and we return the function itself. The second
        # is "@_njit(cache=True, fastmath=True)", where we return a decorator.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    numba = None
try:
    import cupy as cp
except ImportError:
    cp = None


def _detect_backend() -> str:
    """Return the best available backend, one of 'cupy', 'numba', or 'numpy'."""
    try:
        cp.array([1.0])  # forces an attempt to access the GPU
        return "cupy"
    except Exception:
        pass
    try:
        # The import itself is the test. An ImportError here means numba is
        # unavailable and we fall through to the numpy fallback.
        import numba  # noqa: F401

        return "numba"
    except ImportError:
        pass
    return "numpy"


class _Grid:
    """
    A DXGrid with its arrays pre-extracted into contiguous form for the GPU and
    Numba kernels. Numba and CuPy cannot access Python objects, so we pull the
    raw arrays out once at construction time.
    """

    __slots__ = ("data", "origin", "spacing", "inv_spacing", "lo", "hi")

    def __init__(self, g: DXGrid):
        self.data = np.ascontiguousarray(g.data, dtype=np.float64)
        self.origin = np.ascontiguousarray(g.origin, dtype=np.float64)
        self.spacing = np.array([g.delta[i, i] for i in range(3)], dtype=np.float64)
        self.inv_spacing = 1.0 / self.spacing
        nx, ny, nz = self.data.shape
        self.lo = self.origin + self.spacing  # leave a one-cell margin
        self.hi = self.origin + np.array([nx - 2, ny - 2, nz - 2]) * self.spacing

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point > self.lo) and np.all(point < self.hi))


# The following block defines the Numba-compiled inner loops.
try:

    @_njit(cache=True, fastmath=True)
    def _interp(data, origin, inv_sp, point):
        """Trilinear interpolation of the grid at a point, Numba compiled."""
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
        """Gradient of the interpolated grid by central differences."""
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
        Core force accumulation loop. It iterates over the atoms and accumulates
        the net force, torque, and energy. Ghost atoms with zero charge
        contribute exactly nothing.
        """
        N = positions.shape[0]
        force = np.zeros(3)
        torque = np.zeros(3)
        energy = 0.0
        # Centroid of the charged atoms, used as the reference point for torque.
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

    # Pure-NumPy versions of the same routines, used when numba is unavailable.
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


# Holds the compiled CuPy GPU kernel once it has been built.
_CUPY_KERNEL = None


def _build_cupy_kernel():
    """
    Build the CUDA kernel that performs the trilinear interpolation and gradient
    on the GPU. It is called once on the first use of the GPU.
    """
    global _CUPY_KERNEL
    try:
        # Raw CUDA C kernel that assigns one thread per atom.
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
    """Evaluate the force on all atoms on the GPU using the CUDA kernel."""
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
    forces_np = forces_gpu.get()  # per-atom forces, shape (N, 3)
    energies_np = energies_gpu.get()  # per-atom energies, shape (N,)
    # Sum the per-atom contributions to get the net force and energy.
    force = forces_np.sum(axis=0)
    energy = float(energies_np.sum())
    # Torque about the centroid, summing (r_i - centroid) x F_i over the atoms.
    mask = np.abs(charges) > 1e-9
    if mask.any():
        c = positions[mask].mean(axis=0)
        r = positions - c  # position relative to the centroid, shape (N, 3)
        torque = np.cross(r, forces_np).sum(axis=0)
    else:
        torque = np.zeros(3)
    return force, torque, energy


def _group_centroid(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """
    Return the mean position of the charged atoms in a grid group, which is the
    reference point the inner kernels use when accumulating that group's torque.
    Atoms with negligible charge are excluded, matching the kernels.
    """
    mask = np.abs(charges) > 1e-9
    if mask.any():
        return positions[mask].mean(axis=0)
    return np.zeros(3)


class _GridStack:
    """
    A set of DX grids sorted from finest to coarsest, that is with the smallest
    spacing first. For each atom it returns the potential from the finest grid
    that contains the atom. This reproduces the grid selection logic of the
    reference implementation exactly.
    """

    def __init__(self, grids: List[DXGrid]):
        # Sort from finest to coarsest, since the smallest spacing gives the
        # highest resolution.
        sorted_grids = sorted(grids, key=lambda g: float(g.delta[0, 0]))
        self._grids = [_Grid(g) for g in sorted_grids]

    def __bool__(self):
        return len(self._grids) > 0

    def __len__(self):
        return len(self._grids)

    def finest_for(self, point: np.ndarray) -> Optional[_Grid]:
        """Return the finest grid that contains the point, or None if none do."""
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
        ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluate the force on all atoms, assigning each atom to the finest grid
        that contains it. Atoms that fall outside every grid contribute zero,
        since their far-field contribution is negligible.

        The inner kernels return each grid group's torque about that group's own
        centroid. When ref is given, every group's torque is re-expressed about
        that single fixed reference point so that the summed torque shares one
        consistent reference even when the atoms span several grids. When ref is
        None the per-group centroid is used as the reference.
        """
        if not self._grids:
            return np.zeros(3), np.zeros(3), 0.0
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        total_energy = 0.0
        N = positions.shape[0]
        assigned = np.zeros(N, dtype=bool)
        if backend == "cupy" and _CUPY_KERNEL is not None:
            # GPU path. Group the atoms by grid and send each group to the GPU.
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
                f, t, e = _cupy_eval(
                    sub_pos,
                    sub_chg,
                    g,
                    alpha,
                    is_born,
                )
                if ref is not None:
                    t = t + np.cross(_group_centroid(sub_pos, sub_chg) - ref, f)
                total_force += f
                total_torque += t
                total_energy += e
        else:
            # CPU path. Group the atoms by grid and run the Numba or NumPy inner
            # loop on each group.
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
                if ref is not None:
                    t = t + np.cross(_group_centroid(sub_pos, sub_chg) - ref, f)
                total_force += f
                total_torque += t
                total_energy += e
        return total_force, total_torque, total_energy


class PySTARCEngine:
    """
    PySTARC unified force engine. It implements all of the force terms from the
    reference implementation. The first term is the electrostatic interaction,
    evaluated per atom from APBS DX grids using finest-grid-first selection. The
    second term is Born desolvation, evaluated per atom from the *_born.dx grids.

    The backend is selected automatically. The cupy backend runs on an NVIDIA GPU
    through CUDA, the numba backend is just-in-time compiled on the CPU, and the
    numpy backend is pure Python on the CPU. Ghost atoms with zero charge
    contribute exactly nothing to any term.

    The constructor takes the following parameters. elec_mol1 and elec_mol2 are
    the electrostatic DX grids for molecule 1 (the receptor) and molecule 2 (the
    ligand). born_mol1 and born_mol2 are the corresponding Born desolvation
    grids. debye_length is the Debye screening length in Å, taken from the
    solvent file. desolvation_alpha is the Born desolvation parameter, also taken
    from the solvent file.
    """

    _numba_warmed: bool = False  # compile the Numba kernels once per process

    def __init__(
        self,
        elec_mol1: List[DXGrid] = None,
        elec_mol2: List[DXGrid] = None,
        born_mol1: List[DXGrid] = None,
        born_mol2: List[DXGrid] = None,
        eff_charges_mol1: "EffectiveCharges" = None,
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
        # Effective charges, used as a long-range fallback for the
        # electrostatic term. The fallback engages only when no electrostatic
        # grids are loaded; when grids are present the electrostatics come
        # entirely from the grids.
        self._eff1 = eff_charges_mol1
        self._debye = debye_length
        # Optional Lennard-Jones and hydrophobic forces.
        self._lj_engine = None
        if lj_params is not None or hydrophobic_params is not None:
            self._lj_engine = LJForceEngine(lj_params, hydrophobic_params)
        # Build the CuPy kernel when running on the GPU.
        if self.backend == "cupy":
            _build_cupy_kernel()
            if _CUPY_KERNEL is None:
                warnings.warn("CuPy kernel failed. Falling back to Numba/NumPy.")
                self.backend = "numba" if _NUMBA else "numpy"
        # Warm up the Numba just-in-time compiler so the first real call is fast.
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
        Compute the total force and torque on mol2 from the fields of mol1. This
        is called once per Brownian-dynamics step. It returns the net force on
        mol2 as a length-3 array in units of kBT/Å, the torque on mol2 as a
        length-3 array in units of kBT, and the total energy as a float in units
        of kBT.
        """
        force = np.zeros(3)
        torque = np.zeros(3)
        energy = 0.0
        pos2 = np.ascontiguousarray(mol2.positions_array(), dtype=np.float64)
        chg2 = np.ascontiguousarray(mol2.charges_array(), dtype=np.float64)
        pos1 = np.ascontiguousarray(mol1.positions_array(), dtype=np.float64)
        chg1 = np.ascontiguousarray(mol1.charges_array(), dtype=np.float64)
        # The torque on mol2 is taken about mol2's charged-atom centroid, used as
        # one fixed reference for every term so the summed torque is consistent.
        c2 = _group_centroid(pos2, chg2)
        # Electrostatic term, evaluating the mol2 atoms in the field of mol1.
        if self._elec1:
            f, t, e = self._elec1.eval_atoms(
                pos2, chg2, 0.0, False, self.backend, ref=c2
            )
            force += f
            torque += t
            energy += e
        elif self._eff1 is not None:
            # Long-range fallback that uses effective charges for the
            # electrostatic term. This branch runs only when no electrostatic
            # grids are loaded, and it then evaluates every mol2 atom from the
            # effective charges. The centroid c2 is loop-invariant because it
            # depends only on pos2 and chg2, so it is hoisted out of the loop.
            for i, (p, q) in enumerate(zip(pos2, chg2)):
                if abs(q) < 1e-9:
                    continue
                f_i = self._eff1.force_on_charge(p, q)
                force += f_i
                energy += q * self._eff1.potential(p)
                torque += np.cross(p - c2, f_i)
        # Born desolvation term, evaluating the mol2 atoms in the Born field of
        # mol1.
        if self._born1:
            f, t, e = self._born1.eval_atoms(
                pos2, chg2, self.alpha, True, self.backend, ref=c2
            )
            force += f
            torque += t
            energy += e
        # Born desolvation term, evaluating the mol1 atoms in the Born field of
        # mol2, to match the BD2 reference. This is a distinct physical
        # contribution and not the Newton's third law conjugate of the born1
        # term. It has been verified against Browndye2 in
        # forces_impl.hh:add_core_forces.
        if self._born2:
            # The grid evaluation gives the desolvation force and torque on mol1,
            # the torque taken about mol1's charged-atom centroid c1. The force
            # and torque on mol2 follow from Newton's third law: the force on
            # mol2 is minus the force on mol1, and the torque on mol2 about c2 is
            # minus the torque on mol1 re-expressed about c2. This mirrors the
            # force_and_torque0 reciprocity in BD2's add_core_forces.
            c1 = _group_centroid(pos1, chg1)
            f, t, e = self._born2.eval_atoms(
                pos1, chg1, self.alpha, True, self.backend, ref=c1
            )
            force += -f
            torque += np.cross(c2 - c1, f) - t
            energy += e
        # Optional Lennard-Jones and hydrophobic term.
        if self._lj_engine is not None:
            n1 = len(pos1)
            n2 = len(pos2)
            # Every atom shares a single Lennard-Jones type, so the type index
            # for each atom is 0.
            type_ids1 = [0] * n1
            type_ids2 = [0] * n2
            f1, f2, e_lj = self._lj_engine.compute(pos1, pos2, type_ids1, type_ids2)
            # The Lennard-Jones force and energy come out in kcal/mol and
            # kcal/mol/Angstrom, so they are converted to kBT before being added
            # to the accumulators.
            force += f2 * KCAL_PER_MOL_TO_KBT  # force on mol2 from mol1
            energy += e_lj * KCAL_PER_MOL_TO_KBT
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


def load_dx_directory(
    directory: str | Path,
    mol1_prefix: str = "receptor",
    mol2_prefix: str = "ligand",
    debye_length: float = 7.858,
    desolvation_alpha: float = 0.07957747,
) -> PySTARCEngine:
    """
    Build a PySTARCEngine from a reference-implementation run directory. The
    relevant files are detected automatically by their names. Files named
    <prefix>[0-9].dx are APBS electrostatic grids, files named
    <prefix>[0-9]_born.dx are Born desolvation grids, files named
    <prefix>_cheby.xml hold Chebyshev effective charges for the long-range
    fallback, and files named <prefix>_mpole.xml hold multipole effective charges
    as an alternative. The effective charges serve as a long-range fallback for
    the electrostatic term that engages only when no electrostatic DX grids are
    loaded; when grids are present the electrostatics come entirely from them.
    """
    d = Path(directory)

    def _load_dx(prefix: str, suffix: str) -> List[DXGrid]:
        grids = []
        for p in sorted(d.glob(f"{prefix}[0-9]{suffix}")):
            grids.append(DXGrid.from_file(p))
        return grids

    eff1 = load_effective_charges(d, mol1_prefix, debye_length)
    return PySTARCEngine(
        elec_mol1=_load_dx(mol1_prefix, ".dx"),
        elec_mol2=_load_dx(mol2_prefix, ".dx"),
        born_mol1=_load_dx(mol1_prefix, "_born.dx"),
        born_mol2=_load_dx(mol2_prefix, "_born.dx"),
        eff_charges_mol1=eff1,
        debye_length=debye_length,
        desolvation_alpha=desolvation_alpha,
    )
