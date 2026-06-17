"""
Adaptive time step for Brownian dynamics.

In a Brownian-dynamics simulation the time step Δt must be small enough that
three conditions hold. The mean displacement should be small compared to the
intermolecular separation. The force-induced displacement should not cause
large energy changes. The trajectory should not overshoot the b-surface or the
escape sphere within a single step.

PySTARC enforces these conditions with three independent constraints and uses
the smallest of the three.

The pair constraint Δt_pair keeps the mean displacement small relative to the
separation. It is

    Δt_pair = f² × r² / (2 D₀)    with f = 0.1.

Here r is the centre-to-centre separation and D₀ is the relative translational
diffusion coefficient. This ensures the RMS displacement √(2 D₀ Δt) is at most
10% of r. At large r the allowed step grows large, making the simulation
efficient, and at small r it shrinks to preserve accuracy.

The force constraint Δt_force limits the displacement driven by the force. It is

    Δt_force = α / |D₀ F|    with α = 0.01.

Here F is the force, so the force-induced displacement D₀ F Δt is kept to about
1% of an Å. This avoids large energy changes per step that would violate the
constant-force approximation used by the Ermak-McCammon integrator.

The edge constraint Δt_edge prevents the trajectory from overshooting the
nearest boundary. It is

    Δt_edge = min(r - b, r_esc - r)² / (18 D₀).

Here b is the b-surface radius and r_esc is the escape-sphere radius. The factor
18 keeps the probability of crossing the boundary in one step below 1%.

The combined step Δt = min(Δt_pair, Δt_force, Δt_edge) therefore adapts on its
own, taking large steps in the diffusion-dominated far field and small steps
near the receptor where the force dominates.
"""

from __future__ import annotations
from typing import Optional
import math

_FRAC = 0.1  # The mean displacement must stay below this fraction of the separation in the pair constraint.
_GROWTH = 1.1  # The time step is allowed to grow by this factor each step.
_RXN_FRAC = 0.0001  # In the reaction zone the step is held to 0.01% of the reaction distance, matching Rxn_Tester.
_LARGE = 1.0e30


def max_time_step(
    r: float, D_rel: float, D_rot: float, r_hydro1: float, r_hydro2: float
) -> float:
    """
    Compute the largest time step the geometry allows.

    For two rigid bodies the translational pair constraint is

        dt_pair = (frac² / 2) × r² / D_parallel(r).

    The step is also bounded by rotational diffusion through

        dt_rot = π² / D_rot.

    The arguments are the current centre-to-centre separation r in Å, the
    relative translational diffusion coefficient D_rel in Å²/ps, the maximum
    rotational diffusion coefficient D_rot in rad²/ps, and the hydrodynamic
    radii r_hydro1 and r_hydro2 of the two molecules in Å. The function returns
    the maximum step in ps.
    """
    if r <= 0 or D_rel <= 0:
        return 0.2  # Fall back to a safe default when the inputs are non-physical.
    # Pair constraint, keeping the mean displacement below frac × r.
    dt_pair = (_FRAC**2 / 2.0) * r**2 / D_rel
    # Rotational constraint, dt_rot = π² / D_rot.
    if D_rot > 0:
        dt_rot = math.pi**2 / D_rot
    else:
        dt_rot = _LARGE
    # Size constraint. The rigorous form is dt_size = 4 R³ / D_factor with
    # D_factor = kT/μ the viscosity factor, which we approximate here as
    # dt_size ≈ r_hydro² / D_rel.
    r_min = min(r_hydro1, r_hydro2)
    dt_size = 4.0 * r_min**2 / D_rel if D_rel > 0 else _LARGE
    return min(dt_pair, dt_rot, dt_size)


def reaction_time_step(rho_min: float, D_rel: float) -> float:
    """
    Time step constraint near the reaction boundary.

    This step is much smaller than dt_pair and ensures the trajectory does not
    overshoot the reaction criterion distance. The arguments are the smallest
    active reaction criterion distance rho_min in Å and the relative diffusion
    coefficient D_rel in Å²/ps.
    """
    if rho_min <= 0 or D_rel <= 0:
        return 0.05
    return 0.5 * (_RXN_FRAC * rho_min) ** 2 / D_rel


class AdaptiveTimeStep:
    """
    Geometry-based adaptive time step controller.

    On the first call the step is set to max_time_step(r, D, Dr). On each later
    call the step grows by at most the factor 1.1, becoming
    min(last_dt × 1.1, max_time_step()). Near the reaction zone it is further
    capped at reaction_time_step(rho_min, D).

    A typical Brownian-dynamics loop creates one controller and queries it each
    step. For example, dt_ctrl = AdaptiveTimeStep() is built once, and inside
    the loop dt = dt_ctrl.get_dt(r, D_rel, D_rot, r_h1, r_h2, rxn_distances,
    dt_min, dt_rxn_min) returns the step to use for the following move.
    """

    def __init__(self):
        self._last_dt: Optional[float] = None

    def reset(self):
        """Reset after trajectory restart."""
        self._last_dt = None

    def get_dt(
        self,
        r: float,
        D_rel: float,
        D_rot: float,
        r_hydro1: float,
        r_hydro2: float,
        rxn_distances: list,
        dt_min: float = 0.001,
        dt_rxn_min: float = 1e-6,
    ) -> float:
        """
        Compute the time step for the current Brownian-dynamics step.

        The arguments are the current separation r in Å, the relative
        translational diffusion D_rel in Å²/ps, the maximum rotational diffusion
        D_rot in rad²/ps, the hydrodynamic radii r_hydro1 and r_hydro2 in Å, and
        the list rxn_distances of reaction criterion distances in Å. The value
        dt_min is the hard minimum step in ps taken from the time_step_parameters,
        and dt_rxn_min is the hard minimum step in ps used near the reaction
        boundary. The function returns the step to use in ps.
        """
        # Geometric maximum allowed by the current configuration.
        dt_geo = max_time_step(r, D_rel, D_rot, r_hydro1, r_hydro2)
        # Tighten the step near the reaction boundary.
        if rxn_distances:
            rho_min = min(rxn_distances)
            # Apply the reaction step once within 1.5 times the smallest criterion distance.
            if r < 1.5 * rho_min:
                dt_rxn = reaction_time_step(rho_min, D_rel)
                dt_geo = min(dt_geo, dt_rxn)
        # Grow the step from the previous one, but never beyond the geometric maximum.
        if self._last_dt is None:
            dt = dt_geo
        else:
            dt = min(self._last_dt * _GROWTH, dt_geo)
        # Enforce the hard minimum step.
        if rxn_distances and r < 1.5 * min(rxn_distances):
            dt = max(dt, dt_rxn_min)
        else:
            dt = max(dt, dt_min)
        self._last_dt = dt
        return dt
