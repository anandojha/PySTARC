"""
Ermak-McCammon Brownian-dynamics integrator.

The Ermak-McCammon equation is the standard integrator for overdamped
(Brownian) dynamics in implicit solvent. A single step advances the position by

    r(t + Δt) = r(t) + (D₀/kBT) × F × Δt + √(2 D₀ Δt) × W

Here r(t) is the position at time t in Å, D₀ is the relative translational
diffusion coefficient in Å²/ps, F is the total force on the ligand in kBT/Å,
Δt is the time step in ps, and W is 3D Gaussian noise drawn from N(0, I).

The first term is the deterministic drift, the motion under the total force
(electrostatic, Born desolvation, and optional WCA steric). The second term is
the thermal noise from solvent-molecule collisions. The noise amplitude
√(2 D₀ Δt) satisfies the fluctuation-dissipation theorem, which guarantees that
the equilibrium distribution is the Boltzmann distribution exp(-V/kBT).

The relative size of the two terms is set by the ratio
|D₀ F Δt| / √(2 D₀ Δt) = |F| × √(D₀ Δt / 2). For typical BD this ratio is about
0.01 to 0.1, so the noise dominates, but near strong electrostatic steering it
rises to roughly 0.5 to 1.0. At each step the ligand performs a random walk that
is biased by the force, and over many steps this bias accumulates into directed
motion toward or away from the receptor. We work in units where kBT = 1, so
D₀/kBT reduces to D₀ and no explicit division is needed.

The overdamped limit applies because water is highly viscous at the molecular
scale. The momentum relaxation time τ_p = m/(6πηa) ≈ 10 fs for a protein is
about 1000× shorter than the BD time step (~10 to 100 ps), so inertia is
completely negligible and the velocity instantaneously adjusts to the force.
This is the overdamped (high-friction) limit.
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
    """Take one translational Brownian-dynamics step.

    The update is r(t + Δt) = r(t) + D_trans × F × Δt + √(2 D_trans) × dW.
    The last argument may be either a pre-drawn Wiener increment dW (already
    scaled by √Δt) or a numpy random Generator, which is supported for backward
    compatibility. When a Generator is passed, dW is drawn internally.
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
    """Take one rotational Brownian-dynamics step.

    The orientation is updated by an axis-angle rotation whose angle combines a
    deterministic drift D_rot × Δt × torque with thermal noise √(2 D_rot) × dW.
    The last argument may be either a pre-drawn Wiener increment dW_rot (already
    scaled by √Δt) or a numpy random Generator, which is supported for backward
    compatibility. When a Generator is passed, dW_rot is drawn internally.
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
    # total_angle is a lab-frame angular displacement (the torque is lab-frame),
    # so the increment is applied as a left product dq ⊗ orientation, matching
    # the GPU path and the analytic Langevin equilibrium for a dipole in a field.
    return (dq * orientation).normalized()


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
    """Decide whether the last step changed the force too much and should be retaken.

    The test compares the time step against a force-change criterion. Writing dx
    for the displacement pos_new - pos_old and dF for the force change
    F_new - F_old, the criterion accumulates |dx|² and (1/a) × dot(dF, dx), then
    forms det = |6 × π × μ × |dx|² / ((1/a) × dot(dF, dx))|. A backstep is
    requested when both dt > 0.02 × det and dt > dt_min hold. The default
    viscosity is 0.243 kBT.ps/A^3.
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
    """Take one combined translational and rotational step, drawing fresh Wiener increments."""
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
    """Take one combined step using pre-drawn Wiener increments, as needed when subdividing a step."""
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
    """Take one combined step with an adaptive time step.

    The step uses the smaller dt_min_rxn when the ligand is close to a reaction
    boundary and the larger dt_min otherwise. It returns the new position, the
    new orientation, and the time step that was actually used.
    """
    r = float(np.linalg.norm(position))
    rxn_min = min(reaction_distances) if reaction_distances else 5.0
    dt = dt_min_rxn if r < 1.5 * rxn_min else dt_min
    new_pos, new_ori = bd_step(
        position, orientation, force, torque, D_trans, D_rot, dt, rng
    )
    return new_pos, new_ori, dt


def escape_radius(r_start: float) -> float:
    """Return the default escape radius (the q-sphere).

    The default is 5 × the b-sphere radius, which ensures the escape sphere is
    always well beyond the b-sphere.
    """
    return 5.0 * r_start


# Tensor-aware sibling functions for the BD step.
#
# These accept (3, 3) diffusion-coefficient tensors instead of scalars, which is
# required for chains with full Rotne-Prager-Yamakawa hydrodynamics, where
# D_trans and D_rot are anisotropic. The scalar versions above remain in use by
# NAMSimulator and by chain_outer_bd_step. The latter will pass D × I through
# tensor wrapping once the chain BD pipeline migrates.
#
# The conventions match the scalar versions exactly. In particular, the Wiener
# increment dW is already √Δt-scaled, so do not add another √Δt inside the
# function.


def _cholesky_2D_dt(D_tensor: np.ndarray, dt: float) -> np.ndarray:
    """Return the Cholesky factor L of 2 × D × Δt, used to scale the Wiener noise.

    The result is the lower-triangular L such that L Lᵀ = 2 × D × Δt. A clear
    LinAlgError is raised if D is not positive-definite, which indicates that the
    caller supplied a non-physical mobility tensor.
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
    """Take one translational Brownian-dynamics step with an anisotropic D_trans.

    The update is r(t + Δt) = r(t) + (D_trans @ F) × Δt + L_t @ dW_t, where the
    noise factor L_t satisfies L_t L_tᵀ = 2 × D_trans × Δt.

    Here position is the current (3,) position and force is the (3,) total force
    in kBT/Å. D_trans is the (3, 3) translational diffusion tensor in Å²/ps,
    which must be symmetric positive-definite, and dt is the time step in ps.
    The argument dW_or_rng is either a pre-scaled Wiener increment of shape (3,)
    or a numpy random Generator. When a Generator is passed, dW_t is drawn
    internally as √Δt × standard_normal(3), matching the scalar function.
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
    noise = L @ (dW / math.sqrt(dt))  # this equals √(2 D Δt) × (dW / √Δt)
    # The increment dW is already √Δt-scaled. The scalar form is
    # noise = √(2 D) × dW = √(2 D Δt) × (dW / √Δt). What we want here is
    # noise = L @ N(0, I), and since dW = √Δt × N(0, I) this becomes
    # noise = L @ (dW / √Δt).
    return position + drift + noise


def ermak_mccammon_rotation_tensor(
    orientation: Quaternion,
    torque: np.ndarray,
    D_rot: np.ndarray,
    dt: float,
    dW_or_rng,
) -> Quaternion:
    """Take one rotational Brownian-dynamics step with an anisotropic D_rot.

    This applies the same axis-angle update as the scalar version, but with an
    anisotropic drift (D_rot @ torque) × Δt and Cholesky-scaled noise.
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
    # total_angle is a lab-frame angular displacement (the torque is lab-frame),
    # so the increment is applied as a left product dq ⊗ orientation, matching
    # the GPU path and the analytic Langevin equilibrium for a dipole in a field.
    return (dq * orientation).normalized()


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
    """Take one combined tensor step with pre-drawn Wiener increments.

    This is the sibling of bd_step_wiener and takes (3, 3) D_trans and D_rot
    tensors. It reduces to the scalar version exactly when D_trans = D × I and
    D_rot = D × I, which is validated by a regression test.
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
