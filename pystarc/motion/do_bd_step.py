"""
Ermak-McCammon BD integrator
============================

Background
-------------------
The Ermak-McCammon equation is the standard integrator for
overdamped (Brownian) dynamics in implicit solvent:

    r(t + Δt) = r(t) + (D₀/kBT) × F × Δt + √(2 D₀ Δt) × W

where:
  - r(t)  : position at time t (Å)
  - D₀    : relative translational diffusion coefficient (Å²/ps)
  - F     : total force on the ligand (kBT/Å)
  - Δt    : time step (ps)
  - W     : 3D Gaussian noise ~ N(0, I)

The first term (drift) represents deterministic motion under forces:
electrostatic, Born desolvation, and optional WCA steric.

The second term (noise) represents thermal fluctuations from solvent
molecule collisions.  The noise amplitude √(2 D₀ Δt) satisfies the
fluctuation-dissipation theorem, ensuring that the equilibrium
distribution is the Boltzmann distribution exp(-V/kBT).

Physical interpretation
-----------------------
- Drift/noise ratio: |D₀ F Δt| / √(2 D₀ Δt) = |F| × √(D₀ Δt / 2)
    For typical BD: drift/noise ~ 0.01-0.1 (noise dominates)
    Near strong electrostatic steering: drift/noise ~ 0.5-1.0

- At each step, the ligand makes a random walk biased by the force.
    Over many steps, the bias accumulates to give directed motion
    toward or away from the receptor.

- kBT = 1, such that D₀/kBT = D₀ (no explicit division).

Why overdamped?
---------------
Water is highly viscous at the molecular scale.  The momentum
relaxation time τ_p = m/(6πηa) ≈ 10 fs for a protein, which is
1000× shorter than the BD time step (~10-100 ps).  So inertia is
completely negligible and the velocity instantaneously adjusts to the
force.  This is the overdamped (high-friction) limit.
"""

from __future__ import annotations
from pystarc.transforms.quaternion import Quaternion, small_rotation_quaternion
from typing import Optional, Tuple
import numpy as np
import math

FORCE_CHANGE_ALPHA = 0.02
WATER_VISCOSITY = 0.243  # kBT.ps/A^3


def ermak_mccammon_translation(
    position: np.ndarray, force: np.ndarray, D_trans: float, dt: float, dW_or_rng
) -> np.ndarray:
    """
    Translational BD step.
    Last arg can be a pre-drawn dW array OR a numpy rng (backward compat).
    r(t+dt) = r(t) + D_t*F*dt + sqrt(2*D_t)*dW
    """
    if isinstance(dW_or_rng, np.ndarray):
        dW = dW_or_rng
    else:
        dW = math.sqrt(dt) * dW_or_rng.standard_normal(3)
    drift = D_trans * force * dt
    noise = math.sqrt(2.0 * D_trans) * dW
    return position + drift + noise


def ermak_mccammon_rotation(
    orientation: Quaternion, torque: np.ndarray, D_rot: float, dt: float, dW_or_rng
) -> Quaternion:
    """
    Rotational BD step.
    Last arg can be a pre-drawn dW_rot array OR a numpy rng (backward compatibility).
    """
    if isinstance(dW_or_rng, np.ndarray):
        dW_rot = dW_or_rng
    else:
        dW_rot = math.sqrt(dt) * dW_or_rng.standard_normal(3)
    drift_angle = D_rot * dt * torque
    noise_angle = math.sqrt(2.0 * D_rot) * dW_rot
    total_angle = drift_angle + noise_angle
    angle_mag = float(np.linalg.norm(total_angle))
    if angle_mag < 1e-14:
        return orientation
    axis = total_angle / angle_mag
    dq = Quaternion.from_axis_angle(axis, angle_mag)
    return (orientation * dq).normalized()


def backstep_due_to_force(
    force_new: np.ndarray,
    force_old: np.ndarray,
    pos_new: np.ndarray,
    pos_old: np.ndarray,
    dt: float,
    dt_min: float,
    radius: float = 1.0,
    viscosity: float = WATER_VISCOSITY,
) -> bool:
    """
        dx2_sum   += |dx|^2
        DdxdF_sum += (1/a) * dot(F_new - F_old, dx)
        det        = |6*pi*mu * dx2_sum / DdxdF_sum|
        backstep if dt > 0.02 * det  AND  dt > dt_min
    viscosity default = 0.243 kBT.ps/A^3  .
    """
    if dt <= dt_min:
        return False
    dx = pos_new - pos_old
    dF = force_new - force_old
    dx2 = float(np.dot(dx, dx))
    ainv = 1.0 / max(radius, 1e-6)
    dFdx = ainv * float(np.dot(dF, dx))
    if abs(dFdx) < 1e-30:
        return False
    PI6_MU = 6.0 * math.pi * viscosity
    det = abs(PI6_MU * dx2 / dFdx)
    return dt > FORCE_CHANGE_ALPHA * det


def bd_step(
    position: np.ndarray,
    orientation: Quaternion,
    force: np.ndarray,
    torque: np.ndarray,
    D_trans: float,
    D_rot: float,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Quaternion]:
    """Combined BD step drawing fresh Wiener increments."""
    dW_t = math.sqrt(dt) * rng.standard_normal(3)
    dW_r = math.sqrt(dt) * rng.standard_normal(3)
    new_pos = ermak_mccammon_translation(position, force, D_trans, dt, dW_t)
    new_ori = ermak_mccammon_rotation(orientation, torque, D_rot, dt, dW_r)
    return new_pos, new_ori


def bd_step_wiener(
    position: np.ndarray,
    orientation: Quaternion,
    force: np.ndarray,
    torque: np.ndarray,
    D_trans: float,
    D_rot: float,
    dt: float,
    dW_t: np.ndarray,
    dW_r: np.ndarray,
) -> Tuple[np.ndarray, Quaternion]:
    """BD step using pre-drawn Wiener increments (for subdivision)."""
    new_pos = ermak_mccammon_translation(position, force, D_trans, dt, dW_t)
    new_ori = ermak_mccammon_rotation(orientation, torque, D_rot, dt, dW_r)
    return new_pos, new_ori


def bd_step_adaptive(
    position: np.ndarray,
    orientation: Quaternion,
    force: np.ndarray,
    torque: np.ndarray,
    D_trans: float,
    D_rot: float,
    rng: np.random.Generator,
    reaction_distances: list,
    dt_min: float = 0.2,
    dt_min_rxn: float = 0.05,
) -> Tuple[np.ndarray, Quaternion, float]:
    """
    Adaptive time step BD step.
    Uses dt_min_rxn when close to reaction boundary, dt_min otherwise.
    Returns (new_pos, new_ori, dt_used).
    """
    r = float(np.linalg.norm(position))
    rxn_min = min(reaction_distances) if reaction_distances else 5.0
    dt = dt_min_rxn if r < 1.5 * rxn_min else dt_min
    new_pos, new_ori = bd_step(
        position, orientation, force, torque, D_trans, D_rot, dt, rng
    )
    return new_pos, new_ori, dt


def escape_radius(r_start: float) -> float:
    """
    Default escape radius (q-sphere).
    Use 5 * b_sphere as default as this ensures the escape sphere
    is always well beyond the b-sphere.
    """
    return 5.0 * r_start


# Tensor-aware sibling functions for BD step.
#
# These accept (3, 3) diffusion-coefficient tensors instead of scalars.
# Required for chains with full RPY hydrodynamics, where D_trans and D_rot
# are anisotropic. The scalar versions above remain in use by NAMSimulator
# and chain_outer_bd_step (the latter passes D * I via tensor wrapping
# once the chain BD pipeline migrates).
#
# Conventions match the scalar versions exactly. In particular, the
# Wiener increment dW is already sqrt(dt)-scaled; do not add another
# sqrt(dt) inside the function.


def _cholesky_2D_dt(D_tensor: np.ndarray, dt: float) -> np.ndarray:
    """Cholesky factor L of 2 * D * dt, used to scale Wiener noise.

    Returns lower-triangular L such that L L^T = 2 * D * dt.
    Raises a clear LinAlgError if D is not positive-definite (which
    indicates the user supplied a non-physical mobility tensor).
    """
    M = 2.0 * D_tensor * dt
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        eigvals = np.linalg.eigvalsh(D_tensor)
        raise np.linalg.LinAlgError(
            f"diffusion tensor is not positive-definite "
            f"(eigenvalues {eigvals}). A physical D must satisfy v^T D v > 0 "
            f"for all v != 0."
        ) from None


def ermak_mccammon_translation_tensor(
    position: np.ndarray,
    force: np.ndarray,
    D_trans: np.ndarray,
    dt: float,
    dW_or_rng,
) -> np.ndarray:
    """Translational BD step with anisotropic D_trans.

    r(t + dt) = r(t) + D_trans @ F * dt + L_t @ dW_t,
    where L_t L_t^T = 2 * D_trans * dt.

    Parameters
    ----------
    position  : (3,) current position.
    force     : (3,) total force in kBT/A.
    D_trans   : (3, 3) translational diffusion tensor (A^2/ps), symmetric
                positive-definite.
    dt        : timestep in ps.
    dW_or_rng : pre-scaled Wiener increment array of shape (3,), or a
                numpy random Generator. If a Generator, dW_t is drawn
                internally as sqrt(dt) * standard_normal(3) (matching
                the scalar function's behavior).
    """
    D_trans = np.asarray(D_trans, dtype=float)
    if D_trans.shape != (3, 3):
        raise ValueError(f"D_trans must have shape (3, 3); got {D_trans.shape}")
    if isinstance(dW_or_rng, np.ndarray):
        dW = dW_or_rng
    else:
        dW = math.sqrt(dt) * dW_or_rng.standard_normal(3)
    drift = (D_trans @ force) * dt
    L = _cholesky_2D_dt(D_trans, dt)
    noise = L @ (dW / math.sqrt(dt))  # equivalent to sqrt(2 D dt) * (dW / sqrt(dt))
    # Wait: dW is already sqrt(dt)-scaled. To match the scalar form
    #   noise = sqrt(2 D) * dW = sqrt(2 D dt) * (dW/sqrt(dt))
    # noise (= L @ N(0, I)) is wanted. Since dW = sqrt(dt) * N(0, I),
    # noise = L @ (dW / sqrt(dt)).
    return position + drift + noise


def ermak_mccammon_rotation_tensor(
    orientation: Quaternion,
    torque: np.ndarray,
    D_rot: np.ndarray,
    dt: float,
    dW_or_rng,
) -> Quaternion:
    """Rotational BD step with anisotropic D_rot.

    Same axis-angle update as the scalar version, but with anisotropic
    drift (D_rot @ torque * dt) and Cholesky-scaled noise.
    """
    D_rot = np.asarray(D_rot, dtype=float)
    if D_rot.shape != (3, 3):
        raise ValueError(f"D_rot must have shape (3, 3); got {D_rot.shape}")
    if isinstance(dW_or_rng, np.ndarray):
        dW_rot = dW_or_rng
    else:
        dW_rot = math.sqrt(dt) * dW_or_rng.standard_normal(3)
    drift_angle = (D_rot @ torque) * dt
    L = _cholesky_2D_dt(D_rot, dt)
    noise_angle = L @ (dW_rot / math.sqrt(dt))
    total_angle = drift_angle + noise_angle
    angle_mag = float(np.linalg.norm(total_angle))
    if angle_mag < 1e-14:
        return orientation
    axis = total_angle / angle_mag
    dq = Quaternion.from_axis_angle(axis, angle_mag)
    return (orientation * dq).normalized()


def bd_step_wiener_tensor(
    position: np.ndarray,
    orientation: Quaternion,
    force: np.ndarray,
    torque: np.ndarray,
    D_trans: np.ndarray,
    D_rot: np.ndarray,
    dt: float,
    dW_t: np.ndarray,
    dW_r: np.ndarray,
) -> Tuple[np.ndarray, Quaternion]:
    """Tensor BD step with pre-drawn Wiener increments.

    Sibling of bd_step_wiener; takes (3, 3) D_trans and D_rot tensors.
    Reduces to the scalar version exactly when D_trans = D * I and
    D_rot = D * I (validated by regression test).
    """
    new_pos = ermak_mccammon_translation_tensor(
        position,
        force,
        D_trans,
        dt,
        dW_t,
    )
    new_ori = ermak_mccammon_rotation_tensor(
        orientation,
        torque,
        D_rot,
        dt,
        dW_r,
    )
    return new_pos, new_ori
