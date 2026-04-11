"""
PySTARC step near absorbing surface
===================================

Resolves motion of a particle diffusing near an absorbing surface at x=0.
Computes:
  1. Whether the particle is absorbed (reaches x=0)
  2. If not, its new position x
  3. The elapsed time

This is the Lamm-Schulten (1981) method implemented for both the
b-sphere (inner boundary) and q-sphere (outer boundary).

The survival probability is:
    P_sur = 0.5 * ( exp(b*x0) * (erf((x0+bt)/2t^0.5) - 1)
                  + erf((x0-bt)/2t^0.5) + 1 )
where b = -F (force pointing away from boundary), tau = x0^2/D,
t = sqrt(tau).
"""

from __future__ import annotations
from scipy.special import erfinv as _scipy_erfinv
from typing import Tuple
import numpy as np
import math

def _inv_erf(x: float) -> float:
    """Inverse error function, matching the inv_erf."""
    return float(_scipy_erfinv(x))

def step_near_absorbing_surface(
        rng:  np.random.Generator,
        x0:   float,     # initial distance from absorbing surface (A)
        F:    float,     # radial force component (1/A, kT units, + = away from surface)
        D:    float,     # diffusion coefficient (A^2/ps)
) -> Tuple[bool, float, float]:
    """
    Propagate a particle near an absorbing surface at x=0.
    Parameters
    ----------
    rng  : random number generator
    x0   : initial distance from surface (must be > 0)
    F    : force pointing away from surface (kT/A units)
    D    : diffusion coefficient (A^2/ps)

    Returns
    -------
    (survives, new_x, time)
    survives : True if particle did NOT reach surface
    new_x    : new distance from surface (0 if absorbed)
    time     : elapsed time (ps)
    """
    b    = -F                          # b = -F by convention
    tau  = x0 * x0 / D                # characteristic time
    st   = math.sqrt(tau)
    st2  = 2.0 * st
    bt   = b * tau
    erfmt = math.erf((x0 - bt) / st2)
    erfpt = math.erf((x0 + bt) / st2)
    # survival probability 
    psurv = 0.5 * (math.exp(b * x0) * (erfpt - 1.0) + erfmt + 1.0)
    psurv = max(0.0, min(1.0, psurv))  # numerical safety
    survives = rng.random() < psurv
    if survives:
        # Sample new position from survival distribution
        # Rejection method: use no-flux distribution as proposal
        E = math.erf((x0 - bt) / st2)
        x = 0.0
        found = False
        max_attempts = 10000
        for _ in range(max_attempts):
            pc    = rng.random()
            iearg = pc * (E + 1.0) - E
            # clamp to valid range for erfinv
            iearg = max(-1.0 + 1e-12, min(1.0 - 1e-12, iearg))
            x = 2.0 * st * _inv_erf(iearg) - bt + x0
            if x < 0.0:
                continue  # try again
            t4 = 4.0 * tau
            p0 = math.exp(-((x - x0 + bt)**2) / t4)
            p1 = math.exp( b*x0 - ((x + x0 + bt)**2) / t4)
            p2 = -2.0 * math.exp(-((x + bt)**2 + x0*(x0 + 2.0*(x - bt))) / t4)
            p  = p0 + p1 + p2
            denom = 0.5 * (erfmt + 1.0)
            pu    = p0 / denom if denom > 1e-30 else 0.0
            if p < 0 or pu <= 0:
                continue
            found = rng.random() < (p / pu)
            if found:
                break
        if not found:
            x = max(x0, 0.001)  # fallback
        new_x = max(0.0, x)
        time  = tau / D
        return True, new_x, time

    else:
        # Particle absorbed: sample absorption time
        # p(tau) propto x0 * exp(-(x0-b*tau)^2 / (4*tau)) / (2*sqrt(pi*tau^3))
        x02 = x0 * x0
        b2  = b * b
        # find tau_max where dp/dtau = 0 (Taylor expansion for small b*x0)
        if abs(b * x0) < 0.5:
            tau_max = x02 * (1.0/6.0 - x02 * b2 / 216.0)
        else:
            tau_max = (math.sqrt(b2 * x02 + 9.0) - 3.0) / b2 if b2 > 0 else x02 / 6.0
        def pt(t_: float) -> float:
            if t_ <= 0:
                return 0.0
            return (x0 * math.exp(-((x0 - b*t_)**2) / (4.0*t_)) /
                    (2.0 * math.sqrt(math.pi * t_ * t_ * t_)))

        pmax = pt(tau_max)
        if pmax <= 0:
            pmax = 1e-30
        # rejection sampling for absorption time
        tau_samp = 0.0
        for _ in range(10000):
            tau_samp  = tau * rng.random()
            p_samp    = pmax * rng.random()
            if p_samp < pt(tau_samp):
                break
        time  = tau_samp / D
        return False, 0.0, time