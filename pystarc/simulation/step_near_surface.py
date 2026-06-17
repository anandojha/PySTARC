"""
PySTARC step near an absorbing surface.

This module resolves the motion of a particle diffusing near an absorbing
surface located at x = 0. It determines whether the particle is absorbed,
meaning it reaches x = 0, and if it survives it returns the new position x
and the elapsed time.

This is the Lamm-Schulten (1981) method, implemented for both the b-sphere
(the inner boundary) and the q-sphere (the outer boundary).

The survival probability is

    P_sur = 0.5 × ( exp(b·x0) · (erf((x0 + bt) / 2√τ) − 1)
                  + erf((x0 − bt) / 2√τ) + 1 ).

Here b = −F is the force pointing away from the boundary and τ = x0² has
units of Length² rather than time. When the elapsed time is needed it is
given by τ/D. This convention matches BrownDye2 in
step_near_absorbing_surface.hh, lines 37 to 38.
"""

from __future__ import annotations
from scipy.special import erfinv as _scipy_erfinv
from typing import Tuple
import warnings
import numpy as np
import math


def _inv_erf(x: float) -> float:
    """Return the inverse error function, matching inv_erf."""
    return float(_scipy_erfinv(x))


def step_near_absorbing_surface(
    rng: np.random.Generator,
    x0: float,  # initial distance from the absorbing surface, in angstrom
    F: float,  # radial force component in units of 1/Å (kT units), positive means away from the surface
    D: float,  # diffusion coefficient, in Å²/ps
) -> Tuple[bool, float, float]:
    """
    Propagate a particle near an absorbing surface at x = 0.

    The argument rng is the random number generator. x0 is the initial
    distance from the surface and must be positive. F is the force pointing
    away from the surface, in units of kT/Å. D is the diffusion coefficient,
    in Å²/ps.

    The function returns the tuple (survives, new_x, time). survives is True
    if the particle did not reach the surface. new_x is the new distance from
    the surface, which is 0 if the particle was absorbed. time is the elapsed
    time in picoseconds.
    """
    b = -F  # b is defined as −F by convention.
    # tau has units of Length², following the BrownDye2 convention, so the
    # elapsed time is tau/D. Earlier versions used tau = x0*x0/D, which
    # conflated tau with a time-typed quantity. That produced dimensionally
    # inconsistent erf arguments and a returned time that was off by a factor
    # of D.
    tau = x0 * x0  # Length², in Å²
    st = math.sqrt(tau)
    st2 = 2.0 * st
    bt = b * tau
    erfmt = math.erf((x0 - bt) / st2)
    erfpt = math.erf((x0 + bt) / st2)
    # Survival probability.
    psurv = 0.5 * (math.exp(b * x0) * (erfpt - 1.0) + erfmt + 1.0)
    psurv = max(0.0, min(1.0, psurv))  # Clamp into [0, 1] for numerical safety.
    survives = rng.random() < psurv
    if survives:
        # Sample the new position from the survival distribution using
        # rejection sampling, with the no-flux distribution as the proposal.
        E = math.erf((x0 - bt) / st2)
        x = 0.0
        last_proposal = None  # Most recent non-negative no-flux proposal draw.
        found = False
        max_attempts = 10000
        for _ in range(max_attempts):
            pc = rng.random()
            iearg = pc * (E + 1.0) - E
            # Clamp to the valid range for erfinv.
            iearg = max(-1.0 + 1e-12, min(1.0 - 1e-12, iearg))
            x = 2.0 * st * _inv_erf(iearg) - bt + x0
            if x < 0.0:
                continue  # Reject and try again.
            last_proposal = x
            t4 = 4.0 * tau
            p0 = math.exp(-((x - x0 + bt) ** 2) / t4)
            p1 = math.exp(b * x0 - ((x + x0 + bt) ** 2) / t4)
            p2 = -2.0 * math.exp(-((x + bt) ** 2 + x0 * (x0 + 2.0 * (x - bt))) / t4)
            p = p0 + p1 + p2
            denom = 0.5 * (erfmt + 1.0)
            pu = p0 / denom if denom > 1e-30 else 0.0
            if p < 0 or pu <= 0:
                continue
            found = rng.random() < (p / pu)
            if found:
                break
        if not found:
            # The acceptance ratio of this rejection step is bounded by one, so
            # for a well-posed survival distribution the loop terminates after a
            # finite number of draws. Reaching the attempt cap indicates a
            # degenerate, over-constrained configuration where the survival
            # density is effectively unsupported. Make this rare case visible
            # and return the most recent valid proposal draw, which is itself a
            # sample of the no-flux distribution truncated to x >= 0, rather
            # than a fixed deterministic value.
            warnings.warn(
                "step_near_absorbing_surface: survival rejection sampling did "
                "not converge within {0} attempts for x0={1!r}, F={2!r}, "
                "D={3!r}; returning the last no-flux proposal draw.".format(
                    max_attempts, x0, F, D
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            x = last_proposal if last_proposal is not None else max(x0, 0.001)
        new_x = max(0.0, x)
        time = tau / D
        return True, new_x, time

    else:
        # The particle was absorbed, so sample the absorption time. Its
        # density is p(τ) ∝ x0 · exp(−(x0 − b·τ)² / (4τ)) / (2√(π·τ³)).
        x02 = x0 * x0
        b2 = b * b
        # Find tau_max, where dp/dτ = 0, using a Taylor expansion for small b·x0.
        if abs(b * x0) < 0.5:
            tau_max = x02 * (1.0 / 6.0 - x02 * b2 / 216.0)
        else:
            tau_max = (math.sqrt(b2 * x02 + 9.0) - 3.0) / b2 if b2 > 0 else x02 / 6.0

        def pt(t_: float) -> float:
            if t_ <= 0:
                return 0.0
            return (
                x0
                * math.exp(-((x0 - b * t_) ** 2) / (4.0 * t_))
                / (2.0 * math.sqrt(math.pi * t_ * t_ * t_))
            )

        pmax = pt(tau_max)
        if pmax <= 0:
            pmax = 1e-30
        # Rejection sampling for the absorption time.
        tau_samp = 0.0
        accepted = False
        max_attempts = 10000
        for _ in range(max_attempts):
            tau_samp = tau * rng.random()
            p_samp = pmax * rng.random()
            if p_samp < pt(tau_samp):
                accepted = True
                break
        if not accepted:
            # The proposal envelope pmax bounds the absorption-time density, so
            # for a well-posed density the loop terminates after a finite number
            # of draws. Reaching the attempt cap indicates a degenerate
            # configuration; make this rare case visible and return the last
            # candidate time rather than silently treating it as accepted.
            warnings.warn(
                "step_near_absorbing_surface: absorption-time rejection "
                "sampling did not converge within {0} attempts for x0={1!r}, "
                "F={2!r}, D={3!r}; returning the last candidate time.".format(
                    max_attempts, x0, F, D
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        time = tau_samp / D
        return False, 0.0, time
