"""
PySTARC Wiener process step subdivision.

When a Brownian-dynamics step would change the forces or torques too much, it
violates the assumptions of the Ermak-McCammon integrator. Rather than rejecting
such a step outright, which would introduce bias, this module refines the random
walk by inserting a midpoint and splitting the step in two.

Given the current Wiener increment W, the midpoint increment is

    W_mid = 0.5 W + sqrt(Δt/2) N,

where N is a standard normal vector with N ~ Normal(0, I). The two halves are
then W_1 = W_mid and W_2 = W - W_mid. Both halves are advanced with the halved
time step Δt/2. If either half is also too large, the procedure recurses, so the
overall scheme is a stack-based binary subdivision of the original step.
"""

from __future__ import annotations
from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class WienerStep:
    """A single pending Wiener increment together with its time step."""

    dW: np.ndarray  # The random increment, with shape (n_dof,).
    dt: float  # The time step associated with this increment.


class WienerProcess:
    """
    A Wiener process driven by a stack with midpoint insertion.

    The stack holds the pending (dW, dt) pairs. The current step is whatever sits
    on top of the stack. When a step must be undone, the current step is split
    into two halves and both halves are pushed back onto the stack.
    """

    def __init__(self, dW: np.ndarray, dt: float):
        self._stack: List[WienerStep] = [WienerStep(dW.copy(), dt)]
        self._t: float = 0.0

    @property
    def t(self) -> float:
        return self._t

    @property
    def dt(self) -> float:
        return self._stack[-1].dt

    @property
    def dW(self) -> np.ndarray:
        return self._stack[-1].dW

    @property
    def at_end(self) -> bool:
        return len(self._stack) == 0

    def step_forward(self):
        """Accept the current step by advancing time and popping the stack."""
        self._t += self._stack[-1].dt
        self._stack.pop()

    def split(self, rng: np.random.Generator):
        """
        Split the current step into two halves by inserting a midpoint.

        The midpoint increment is W_mid = 0.5 W + sqrt(Δt/4) N, following the
        standard Wiener process formula, and the second half is W_2nd = W - W_mid.
        Here W is the current increment, Δt is its time step, and N is a standard
        normal vector. Each half carries the halved time step Δt/2. The Gaussian
        scale used in the code is s = sqrt((Δt/2)/2), which equals sqrt(Δt/4).
        """
        old = self._stack[-1]
        hdt = old.dt / 2.0
        s = math.sqrt(hdt / 2.0)
        n_dof = len(old.dW)
        # The midpoint increment.
        w_mid = 0.5 * old.dW + s * rng.standard_normal(n_dof)
        # The second half.
        w_2nd = old.dW - w_mid
        self._stack.pop()
        # Push the second half first, since it runs after the first half.
        self._stack.append(WienerStep(w_2nd, hdt))
        # Push the first half last so that it runs first, on top of the stack.
        self._stack.append(WienerStep(w_mid, hdt))


def do_one_full_step(
    advance_fn: Callable[[np.ndarray, float, float], Tuple[bool, bool]],
    step_back_fn: Callable[[float, float], None],
    rng: np.random.Generator,
    dW_init: np.ndarray,
    dt0: float,
) -> float:
    """
    Execute one full Brownian-dynamics step with automatic Wiener subdivision.

    The argument advance_fn is a callable advance_fn(dW, t, dt) that advances the
    system state by one Brownian-dynamics step and returns the pair (is_done,
    must_backstep). It sets is_done to True when the trajectory has ended, and
    must_backstep to True when the step was too large. The argument step_back_fn
    is a callable step_back_fn(t, dt) that restores the system to its pre-step
    state. The argument rng is the random number generator, dW_init is the initial
    Wiener increment of shape (n_dof,), and dt0 is the nominal time step.

    The function returns final_dt, the time step that was actually used.
    """
    process = WienerProcess(dW_init.copy(), dt0)
    final_dt = dt0
    while not process.at_end:
        dt = process.dt
        t = process.t
        is_done, must_backstep = advance_fn(process.dW, t, dt)
        if not is_done:
            if must_backstep:
                step_back_fn(t, dt)
                process.split(rng)
            else:
                final_dt = dt
                process.step_forward()
        else:
            final_dt = dt
            break

    return final_dt


def make_initial_dW(n_dof: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate the initial Wiener increment dW ~ Normal(0, sqrt(Δt) I), returned as
    a vector of shape (n_dof,). The sqrt(2) factor is not included here, since the
    integrator applies it.
    """
    return math.sqrt(dt) * rng.standard_normal(n_dof)
