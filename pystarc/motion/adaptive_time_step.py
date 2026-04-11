"""
Adaptive time step for Brownian dynamics
========================================

Background
-------------------
In BD simulation, the time step Δt must be small enough such that:
1. The mean displacement is small compared to the separation.
2. Force-induced displacement does not cause large energy changes.
3. Trajectories don't overshoot the b-surface or escape sphere.

PySTARC uses three independent constraints, taking the minimum:

1. Pair constraint (Δt_pair)
    Δt_pair = f² × r² / (2 D₀)    where f = 0.1

This ensures the RMS displacement √(2 D₀ Δt) is at most 10% of
the intermolecular separation r.  At large r, Δt can be very large, 
making the simulation efficient.  At small r, Δt shrinks to maintain 
accuracy.

2. Force constraint (Δt_force)
    Δt_force = α / |D₀ F|     where α = 0.01

This limits the force-induced displacement D₀ F Δt to ~1% of Å.
This prevents large energy changes per step that would violate
the constant-force approximation of the Ermak-McCammon integrator.

3. Edge constraint (Δt_edge)
    Δt_edge = min(r - b, r_esc - r)² / (18 D₀)

This prevents trajectories from overshooting the b-surface or
escape sphere in a single step.  The factor 18 ensures the
probability of crossing the boundary in one step is < 1%.

The combined Δt = min(Δt_pair, Δt_force, Δt_edge) adapts
automatically, i.e., large steps in the far field (diffusion-dominated) and
small steps near the receptor (force-dominated).
"""

from __future__ import annotations
from typing import Optional
import math

_FRAC     = 0.1      # mean displacement must be < frac * separation (pair_dt)
_GROWTH   = 1.1      # dt growth factor per step
_RXN_FRAC = 0.0001   # reaction zone: 0.01% of reaction distance (Rxn_Tester)
_LARGE    = 1.0e30

def max_time_step(r: float,
                  D_rel: float,
                  D_rot: float,
                  r_hydro1: float,
                  r_hydro2: float) -> float:
    """
    Compute the maximum allowed time step from geometry.
    For two rigid bodies:
      dt_pair = (frac^2/2) * r^2 / D_parallel(r)
    Additionally bounded by rotational diffusion:
      dt_rot = pi^2 / D_rot
    Parameters
    ----------
    r        : current centre-to-centre separation (A)
    D_rel    : relative translational diffusion coefficient (A^2/ps)
    D_rot    : maximum rotational diffusion coefficient (rad^2/ps)
    r_hydro1 : hydrodynamic radius of molecule 1 (A)
    r_hydro2 : hydrodynamic radius of molecule 2 (A)
    Returns
    -------
    dt_max (ps)
    """
    if r <= 0 or D_rel <= 0:
        return 0.2   # fallback
    # pair_dt: mean displacement < frac * r  
    dt_pair = (_FRAC**2 / 2.0) * r**2 / D_rel
    # rotational constraint: dt_rot = pi^2 / Dr 
    if D_rot > 0:
        dt_rot = math.pi**2 / D_rot
    else:
        dt_rot = _LARGE
    # size constraint: dt_size = 4*R^3 / D_factor
    # where D_factor = kT/mu (viscosity factor)
    # Approximated as: dt_size ~ r_hydro^2 / D_rel
    r_min   = min(r_hydro1, r_hydro2)
    dt_size = 4.0 * r_min**2 / D_rel if D_rel > 0 else _LARGE
    return min(dt_pair, dt_rot, dt_size)

def reaction_time_step(rho_min: float, D_rel: float) -> float:
    """
    Time step constraint near reaction boundary.
    This is much smaller than dt_pair - ensures we do not overshoot the
    reaction criterion distance.
    Parameters
    ----------
    rho_min : smallest active reaction criterion distance (A)
    D_rel   : relative diffusion coefficient (A^2/ps)
    """
    if rho_min <= 0 or D_rel <= 0:
        return 0.05
    return 0.5 * (_RXN_FRAC * rho_min)**2 / D_rel

class AdaptiveTimeStep:
    """
    Geometry-based adaptive time step controller.
    - First call: dt = max_time_step(r, D, Dr)
    - Each subsequent call: dt = min(last_dt * 1.1, max_time_step())
    - Near reaction zone: dt = min(dt, reaction_time_step(rho_min, D))
    Usage in BD loop:
        dt_ctrl = AdaptiveTimeStep()
        for step in range(max_steps):
            dt = dt_ctrl.get_dt(r, D_rel, D_rot, r_h1, r_h2,
                                rxn_distances, dt_min, dt_rxn_min)
            ... BD step ...
    """
    def __init__(self):
        self._last_dt: Optional[float] = None

    def reset(self):
        """Reset after trajectory restart."""
        self._last_dt = None

    def get_dt(self,
               r:           float,
               D_rel:       float,
               D_rot:       float,
               r_hydro1:    float,
               r_hydro2:    float,
               rxn_distances: list,
               dt_min:      float = 0.001,
               dt_rxn_min:  float = 1e-6) -> float:
        """
        Compute the time step for the current BD step.
        Parameters
        ----------
        r             : current separation (A)
        D_rel         : relative translational diffusion (A^2/ps)
        D_rot         : max rotational diffusion (rad^2/ps)
        r_hydro1/2    : hydrodynamic radii (A)
        rxn_distances : list of reaction criterion distances (A)
        dt_min        : hard minimum dt (ps) - from time_step_parameters
        dt_rxn_min    : hard minimum dt near reaction (ps)
        Returns
        -------
        dt (ps)
        """
        # geometric maximum
        dt_geo = max_time_step(r, D_rel, D_rot, r_hydro1, r_hydro2)
        # near-reaction constraint
        if rxn_distances:
            rho_min = min(rxn_distances)
            # use reaction dt when within 1.5x the smallest criterion distance
            if r < 1.5 * rho_min:
                dt_rxn = reaction_time_step(rho_min, D_rel)
                dt_geo = min(dt_geo, dt_rxn)
        # grow from last step 
        if self._last_dt is None:
            dt = dt_geo
        else:
            dt = min(self._last_dt * _GROWTH, dt_geo)
        # apply hard minimums
        if rxn_distances and r < 1.5 * min(rxn_distances):
            dt = max(dt, dt_rxn_min)
        else:
            dt = max(dt, dt_min)
        self._last_dt = dt
        return dt