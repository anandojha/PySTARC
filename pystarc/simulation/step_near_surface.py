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
from scipy.special import erfcinv as _scipy_erfcinv
from scipy.special import erfcx as _scipy_erfcx
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
    b = -F  # Convention b = -F.
    tau = x0 * x0  # Length^2, in A^2.
    st = math.sqrt(tau)
    st2 = 2.0 * st
    bt = b * tau
    bx0 = b * x0
    erfmt = math.erf((x0 - bt) / st2)
    # Overflow-free survival probability via scaled complementary error function.
    if bx0 >= 1.0:
        zm = 0.5 * (bx0 - 1.0)
        zp = 0.5 * (bx0 + 1.0)
        psurv = (
            0.5
            * math.exp(-zm * zm)
            * (float(_scipy_erfcx(zm)) - float(_scipy_erfcx(zp)))
        )
    elif bx0 >= -1.0:
        term1 = -math.exp(-0.25 * (bx0 - 1.0) ** 2) * float(
            _scipy_erfcx(0.5 * (1.0 + bx0))
        )
        psurv = 0.5 * (term1 + math.erfc(0.5 * (bx0 - 1.0)))
    else:
        term1 = -math.exp(bx0) * math.erfc(0.5 * (1.0 + bx0))
        psurv = 0.5 * (term1 + math.erfc(0.5 * (bx0 - 1.0)))
    psurv = max(0.0, min(1.0, psurv))  # Clamp into [0, 1] for numerical safety.
    survives = rng.random() < psurv
    if survives:
        # Sample the new position by rejection with truncated Gaussian proposal.
        # Exact acceptance ratio is p(x) / p0(x) = 1 - exp(-x * x0 / tau).
        E = erfmt
        erfc_tail = math.erfc(0.5 * (bx0 - 1.0))
        x = 0.0
        last_proposal = None  # Most recent non-negative no-flux proposal draw.
        found = False
        max_attempts = 10000
        for _ in range(max_attempts):
            pc = rng.random()
            if bx0 > 4.0:
                y = (1.0 - pc) * erfc_tail
                if y <= 0.0:
                    continue
                z_inv = float(_scipy_erfcinv(y))
                if not math.isfinite(z_inv):
                    continue
                x = 2.0 * st * z_inv - bt + x0
            else:
                iearg = pc * (E + 1.0) - E
                iearg = max(-1.0 + 1e-12, min(1.0 - 1e-12, iearg))
                x = 2.0 * st * _inv_erf(iearg) - bt + x0
            if x < 0.0:
                continue  # Reject and try again.
            last_proposal = x
            ratio_arg = x * x0 / tau
            acc = -math.expm1(-ratio_arg) if ratio_arg > 0.0 else 0.0
            found = rng.random() < acc
            if found:
                break
        if not found:
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
        # Sample absorption time from p(t) propto x0 * exp(-(x0 - b*t)^2 / (4t)) / (2 sqrt(pi t^3)).
        x02 = x0 * x0
        b2 = b * b
        if abs(bx0) < 0.5:
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
        tau_samp = 0.0
        accepted = False
        max_attempts = 10000
        if b != 0.0 and abs(bx0) > 15.0:
            # Exact Michael-Schucany-Haas Inverse Gaussian sampler truncated to (0, tau].
            ab = abs(b)
            mu = x0 / ab
            abx0 = ab * x0
            for _ in range(max_attempts):
                u1 = max(1e-15, min(1.0 - 1e-15, rng.random()))
                z = math.sqrt(2.0) * _inv_erf(2.0 * u1 - 1.0)
                alpha = (z * z) / abx0
                s = 1.0 + alpha + math.sqrt(alpha * (2.0 + alpha))
                t_cand = mu / s if rng.random() <= s / (1.0 + s) else mu * s
                if 0.0 < t_cand <= tau:
                    tau_samp = t_cand
                    accepted = True
                    break
        else:
            for _ in range(max_attempts):
                tau_samp = tau * rng.random()
                p_samp = pmax * rng.random()
                if p_samp < pt(tau_samp):
                    accepted = True
                    break
        if not accepted:
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
