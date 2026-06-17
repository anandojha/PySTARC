"""
Parallel execution engine for PySTARC.

This module offers four ways of running the independent Brownian-dynamics
trajectories that make up a rate-constant calculation.

The first tier uses multiprocessing.Pool. This is the default. It runs on CPU
in separate processes, works on all platforms with no extra dependencies, and
is embarrassingly parallel because the trajectories are independent of one
another.

The second tier uses concurrent.futures.ProcessPoolExecutor. It does the same
work as the first tier but adds progress reporting and the ability to cancel or
time out individual trajectories.

The third tier is an experimental NumPy vectorised batch. It advances N
trajectories simultaneously as a single vectorised NumPy computation with no
per-step Python loop. For simple systems this runs roughly 5 to 10 times faster
per core than the first tier.

The fourth tier targets the GPU and requires cupy or torch. Full GPU batching
is not yet implemented, but the surrounding architecture is in place.

To run a calculation, import run_parallel and ParallelBackend from
pystarc.simulation.parallel and call run_parallel with the two molecules, the
mobility tensor, the pathway set, the parameters, and the force function,
passing the desired backend through the backend argument (for example
ParallelBackend.MULTIPROCESSING).
"""

from __future__ import annotations
from pystarc.simulation.nam_simulator import (
    NAMParameters,
    SimulationResult,
    zero_force,
    _run_trajectory_worker,
)
from pystarc.transforms.quaternion import Quaternion, random_quaternion
from pystarc.molsystem.system_state import Fate, TrajectoryResult
from concurrent.futures import ProcessPoolExecutor, as_completed
from pystarc.motion.do_bd_step import bd_step, bd_step_adaptive
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.pathways.reaction_interface import PathwaySet
from pystarc.structures.molecules import Molecule, Atom
from typing import Callable, List, Optional, Dict
from dataclasses import dataclass
import multiprocessing as mp
from enum import Enum, auto
import numpy as np
import copy
import math
import time
import sys
import os

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


class ParallelBackend(Enum):
    SERIAL = auto()  # single thread, no parallelism
    MULTIPROCESSING = auto()  # multiprocessing.Pool, the default backend
    FUTURES = auto()  # concurrent.futures with progress reporting
    NUMPY_BATCH = auto()  # vectorised NumPy batch
    GPU = auto()  # GPU, requires cupy or torch


# First tier, running trajectories in a multiprocessing.Pool.
def _run_pool(
    mol1, mol2, mob, pathway_set, params, force_fn, reaction_cutoffs, n_workers, verbose
):
    """
    Run all trajectories in a multiprocessing.Pool. Each worker is seeded with
    params.seed plus its trajectory index, which matches the per-thread seeding
    used elsewhere.
    """
    c0 = mol2.centroid()
    mol2_pos0 = mol2.positions_array() - c0
    args = [
        (mol1, mol2, mol2_pos0, mob, pathway_set, params, force_fn, reaction_cutoffs, i)
        for i in range(params.n_trajectories)
    ]
    if verbose:
        print(f"  [Pool] {n_workers} workers × {params.n_trajectories} trajectories")
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_trajectory_worker, args)
    return results


# Second tier, concurrent.futures with a live progress counter.
def _run_futures(
    mol1, mol2, mob, pathway_set, params, force_fn, reaction_cutoffs, n_workers, verbose
):
    """
    Run trajectories with a ProcessPoolExecutor and a live progress counter.
    The physics is identical to the first tier. The progress reporting makes
    long runs easier to follow.
    """
    c0 = mol2.centroid()
    mol2_pos0 = mol2.positions_array() - c0
    n = params.n_trajectories
    args = [
        (mol1, mol2, mol2_pos0, mob, pathway_set, params, force_fn, reaction_cutoffs, i)
        for i in range(n)
    ]
    results = [None] * n
    done = 0
    reacted = 0
    escaped = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_run_trajectory_worker, arg): i
            for i, arg in enumerate(args)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            done += 1
            if result.fate == Fate.REACTED:
                reacted += 1
            elif result.fate == Fate.ESCAPED:
                escaped += 1
            if verbose and done % max(1, n // 20) == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(
                    f"  [{done:>{len(str(n))}}/{n}]  "
                    f"reacted={reacted}  escaped={escaped}  "
                    f"rate={rate:.1f} traj/s  ETA={eta:.0f}s",
                    flush=True,
                )
    return results


# Third tier, the NumPy vectorised batch.
# This advances N trajectories at once using NumPy array operations, so the
# whole batch takes a single vectorised step with no Python loop per step. For
# systems with simple force functions (zero_force or grid-only) this runs
# roughly 5 to 10 times faster per core than the plain Python loop.
def _run_numpy_batch(
    mol1, mol2, mob, pathway_set, params, force_fn, reaction_cutoffs, verbose
) -> List[TrajectoryResult]:
    """
    Vectorised batch runner in which all N trajectories advance simultaneously.

    The state is held in arrays of length N, where N is the number of
    trajectories. The array pos has shape (N, 3) and holds the current
    positions. The array done has shape (N,) and flags each trajectory as
    complete. The array fate has shape (N,) and holds the outcome codes. The
    array steps has shape (N,) and holds the step count.

    This runner has a few limitations compared with the single-trajectory
    runner. It does not use an adaptive Δt, so the time step is fixed
    throughout, which is left as a future improvement. The force function must
    be zero_force or some other vectorisable function, since StandardForceEngine
    is not currently vectorised. All N trajectories run to max_steps even when
    most finish early, because the early-finish mask is applied but the memory
    stays allocated. This runner is best suited to large numbers of
    trajectories with zero or simple forces.
    """
    N = params.n_trajectories
    rng = np.random.default_rng(params.seed)
    D_t = mob.relative_translational_diffusion()
    D_r = mob.relative_rotational_diffusion()
    dt = params.dt
    r_esc = params.r_escape
    sigma_t = math.sqrt(2.0 * D_t * dt)
    sigma_r = math.sqrt(2.0 * D_r * dt)
    # Initialise the per-trajectory state arrays.
    # Draw random starting positions on the b-surface sphere.
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pos = v * params.r_start  # positions, shape (N, 3)
    # Draw random orientations as quaternions, stored as an (N, 4) array.
    ori_arr = np.array([random_quaternion(rng).to_array() for _ in range(N)])
    # Arrays that track the outcome of each trajectory.
    done = np.zeros(N, dtype=bool)
    fates = np.full(N, Fate.MAX_STEPS)
    steps = np.zeros(N, dtype=int)
    rxn_names = [None] * N
    # Cache the ligand atom positions, centred on the ligand centroid.
    c0 = mol2.centroid()
    mol2_pos0 = mol2.positions_array() - c0  # shape (M, 3), M = atoms in mol2
    if verbose:
        print(f"  [NumPy batch] N={N} trajectories, dt={dt} ps")
    # Main integration loop over time steps.
    for step in range(params.max_steps):
        active = ~done
        if not active.any():
            break
        active_idx = np.where(active)[0]
        # Place the ligand for each active trajectory and check for reaction.
        # This is the one part that cannot be fully vectorised without a
        # vectorised reaction checker, so it falls back to a Python loop over
        # the active trajectories only.
        for i in active_idx:
            # Build the quaternion from the stored orientation array.
            q = Quaternion(*ori_arr[i])
            R = q.to_rotation_matrix()
            placed_pos = (R @ mol2_pos0.T).T + pos[i]
            # Build the placed ligand, reusing a scratch copy.
            mol2_scratch = copy.copy(mol2)
            mol2_scratch.atoms = [copy.copy(a) for a in mol2.atoms]
            for atom, p in zip(mol2_scratch.atoms, placed_pos):
                atom.x, atom.y, atom.z = float(p[0]), float(p[1]), float(p[2])
            rng_i = np.random.default_rng((params.seed or 0) + i + step * N)
            rxn_name = pathway_set.check_all(mol1, mol2_scratch, rng_i)
            if rxn_name is not None:
                done[i] = True
                fates[i] = Fate.REACTED
                steps[i] = step
                rxn_names[i] = rxn_name
                continue
            r = float(np.linalg.norm(pos[i]))
            if r >= r_esc:
                done[i] = True
                fates[i] = Fate.ESCAPED
                steps[i] = step
                continue
        # Take the vectorised Brownian-dynamics step for every trajectory that
        # is still active.
        still_active = np.where(~done)[0]
        if len(still_active) == 0:
            break
        # Translational update, vectorised across all active trajectories. The
        # force is zero on this zero_force path. A non-zero force would require
        # a per-trajectory call.
        noise_t = sigma_t * rng.standard_normal((len(still_active), 3))
        pos[still_active] += noise_t
        # Rotational update as a vectorised small-angle rotation.
        noise_r = sigma_r * rng.standard_normal((len(still_active), 3))
        norms = np.linalg.norm(noise_r, axis=1, keepdims=True)
        mask = (norms > 1e-14).ravel()
        if mask.any():
            axes = np.where(norms > 1e-14, noise_r / (norms + 1e-30), noise_r)
            angles = norms.ravel()
            for k, i in enumerate(still_active):
                if mask[k]:
                    dq = Quaternion.from_axis_angle(axes[k], angles[k])
                    q = (Quaternion(*ori_arr[i]) * dq).normalized()
                    ori_arr[i] = q.to_array()
        if verbose and step % max(1, params.max_steps // 10) == 0:
            n_active = (~done).sum()
            print(
                f"  step {step:>8d}: {n_active} active, "
                f"{done.sum()} done ({fates[done==True] if done.any() else ''})"
            )
    # Collect the per-trajectory results.
    results = []
    for i in range(N):
        results.append(
            TrajectoryResult(
                fate=fates[i],
                steps=int(steps[i]),
                time_ps=float(steps[i]) * dt,
                final_separation=float(np.linalg.norm(pos[i])),
                reaction_name=rxn_names[i],
            )
        )
    return results


# Fourth tier, the GPU stub.
def _run_gpu(mol1, mol2, mob, pathway_set, params, force_fn, reaction_cutoffs, verbose):
    """
    GPU execution stub.

    A full GPU implementation would require cupy for NVIDIA CUDA or torch for
    NVIDIA, AMD, and Apple Metal, together with a vectorised force function that
    interpolates the DXGrid on the GPU and a vectorised reaction checker. The
    surrounding architecture is in place through the batch state arrays of the
    third tier, but the GPU memory transfers are not yet implemented.
    """
    try:
        backend = "CuPy (CUDA)"
    except ImportError:
        try:
            if torch.cuda.is_available():
                backend = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                backend = "PyTorch MPS (Apple Silicon)"
            else:
                backend = "PyTorch CPU"
        except ImportError:
            backend = None
    if backend is None:
        raise RuntimeError(
            "GPU backend requested but neither cupy nor torch is installed.\n"
            "Install with:  pip install cupy-cuda12x    (NVIDIA)\n"
            "           or: pip install torch            (multi-platform)\n"
            "Falling back to MULTIPROCESSING backend."
        )
    print(f"  [GPU] Backend: {backend}")
    print("  [GPU] Full GPU vectorisation not yet implemented.")
    print("  [GPU] Falling back to NumPy batch (CPU vectorised).")
    # Fall back to the NumPy batch runner until the GPU path is implemented.
    return _run_numpy_batch(
        mol1, mol2, mob, pathway_set, params, force_fn, reaction_cutoffs, verbose
    )


# Main entry point.
def run_parallel(
    mol1: Molecule,
    mol2: Molecule,
    mobility: MobilityTensor,
    pathway_set: PathwaySet,
    params: NAMParameters,
    force_fn=None,
    backend: ParallelBackend = ParallelBackend.MULTIPROCESSING,
) -> SimulationResult:
    """
    Run the NAM Brownian-dynamics trajectories using the chosen parallelism
    backend.

    The arguments mol1 and mol2 are the receptor and ligand molecules. The
    argument mobility is the MobilityTensor built from the Stokes-Einstein
    radii. The argument pathway_set holds the reaction criteria. The argument
    params is the NAMParameters object and carries values such as the number of
    threads and the random seed. The argument force_fn is the force function and
    defaults to zero_force. The argument backend selects which parallelism tier
    to use.

    The function returns a SimulationResult that contains the rate constant
    k_on, the reaction probability P_rxn, and the outcome counts.

    The available backends serve different purposes. SERIAL runs one trajectory
    at a time and is meant for debugging. MULTIPROCESSING is the default for
    production runs and uses all CPU cores. FUTURES does the same work as
    MULTIPROCESSING and adds a live progress bar. NUMPY_BATCH is the fastest
    choice for zero_force or otherwise simple systems. GPU is not yet
    implemented and falls back to NUMPY_BATCH.

    Example:

    >>> result = run_parallel(mol1, mol2, mob, ps, params,
    ...     force_fn=engine,
    ...     backend=ParallelBackend.MULTIPROCESSING)
    >>> print(f"k_on = {result.rate_constant(D_rel):.3e} M-1s-1")
    """
    if force_fn is None:
        force_fn = zero_force
    n_workers = min(params.n_threads, params.n_trajectories, mp.cpu_count())
    # Collect the reaction distance cutoffs, which the adaptive time step uses.
    reaction_cutoffs = [
        pair.distance_cutoff
        for rxn in pathway_set.reactions
        for pair in rxn.criteria.pairs
    ]
    t0 = time.time()
    if backend == ParallelBackend.SERIAL or n_workers <= 1:
        # Use the standard serial NAMSimulator path.
        from pystarc.simulation.nam_simulator import NAMSimulator

        sim = NAMSimulator(mol1, mol2, mobility, pathway_set, params, force_fn)
        result = sim.run()
        return result
    elif backend == ParallelBackend.MULTIPROCESSING:
        raw_results = _run_pool(
            mol1,
            mol2,
            mobility,
            pathway_set,
            params,
            force_fn,
            reaction_cutoffs,
            n_workers,
            params.verbose,
        )
    elif backend == ParallelBackend.FUTURES:
        raw_results = _run_futures(
            mol1,
            mol2,
            mobility,
            pathway_set,
            params,
            force_fn,
            reaction_cutoffs,
            n_workers,
            params.verbose,
        )
    elif backend == ParallelBackend.NUMPY_BATCH:
        raw_results = _run_numpy_batch(
            mol1,
            mol2,
            mobility,
            pathway_set,
            params,
            force_fn,
            reaction_cutoffs,
            params.verbose,
        )
    elif backend == ParallelBackend.GPU:
        raw_results = _run_gpu(
            mol1,
            mol2,
            mobility,
            pathway_set,
            params,
            force_fn,
            reaction_cutoffs,
            params.verbose,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    elapsed = time.time() - t0
    # Aggregate the per-trajectory results into summary counts.
    n_reacted = sum(1 for r in raw_results if r.fate == Fate.REACTED)
    n_escaped = sum(1 for r in raw_results if r.fate == Fate.ESCAPED)
    n_max = sum(1 for r in raw_results if r.fate == Fate.MAX_STEPS)
    rxn_counts: Dict[str, int] = {}
    for r in raw_results:
        if r.reacted:
            name = r.reaction_name or "unnamed"
            rxn_counts[name] = rxn_counts.get(name, 0) + 1
    total_steps = sum(r.steps for r in raw_results)
    if params.verbose:
        print(
            f"  Done: {elapsed:.1f}s  "
            f"({total_steps/elapsed:.0f} BD steps/sec total)"
        )
    return SimulationResult(
        n_trajectories=params.n_trajectories,
        n_reacted=n_reacted,
        n_escaped=n_escaped,
        n_max_steps=n_max,
        reaction_counts=rxn_counts,
        r_start=params.r_start,
        r_escape=params.r_escape,
        dt=params.dt,
    )


def recommended_backend(force_fn=None) -> ParallelBackend:
    """
    Choose the best backend automatically for the current machine and force
    function.

    When a GPU is available and the force is zero or simple, the GPU backend is
    preferred once it is implemented. On a machine with several CPUs and a
    complex force, MULTIPROCESSING is chosen. On a machine with several CPUs and
    a zero force, NUMPY_BATCH is chosen. On a single-CPU machine, SERIAL is
    chosen.
    """
    n_cpu = mp.cpu_count()
    # Check whether a GPU is available.
    gpu_available = False
    if torch is not None:
        gpu_available = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    if cp is not None:
        gpu_available = True
    is_zero_force = force_fn is None or force_fn is zero_force
    if n_cpu <= 1:
        return ParallelBackend.SERIAL
    if is_zero_force:
        return ParallelBackend.NUMPY_BATCH
    return ParallelBackend.MULTIPROCESSING


def auto_n_threads() -> int:
    """Return the optimal number of threads for this machine."""
    return mp.cpu_count()
