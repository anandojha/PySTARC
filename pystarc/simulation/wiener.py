"""
PySTARC Wiener process step subdivision
=======================================

When a BD step would change forces/torques too much (violating the
assumptions of the Ermak-McCammon integrator), this implementation refines the
random walk by inserting a midpoint, rather than simply rejecting the
step, which would introduce bias.

Given current Wiener increment W:
  W_mid = 0.5*W + sqrt(dt/2) * N    (N ~ Normal(0,I))
  W_1   = W_mid
  W_2   = W - W_mid

Both halves W_1, W_2 are used with dt/2. If those are also too large,
the process recurses. This is a stack-based binary subdivision.
"""

from __future__ import annotations
from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import math

@dataclass
class WienerStep:
    """One pending Wiener increment with its time step."""
    dW: np.ndarray   # shape (n_dof,) - the random increment
    dt: float        # time step for this increment

class WienerProcess:
    """
    Stack-based Wiener process with midpoint insertion.
    Core algorithm.
    The stack holds pending (dW, dt) pairs. The current step is the
    top of the stack. If a backstep is needed, the current step is
    split into two halves and both pushed back.
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
        """Accept current step - advance time and pop stack."""
        self._t += self._stack[-1].dt
        self._stack.pop()

    def split(self, rng: np.random.Generator):
        """
        Split current step into two halves (midpoint insertion).
        W_mid = 0.5*W + sqrt(dt/4) * N    (standard Wiener process formula)
        W_2nd = W - W_mid

          new_cur.dt = old.dt / 2
          s = sqrt(new_cur.dt / 2)
          new_cur.w = 0.5*old.w + s*gaussian
          next.w = old.w - new_cur.w
          next.dt = new_cur.dt
        """
        old  = self._stack[-1]
        hdt  = old.dt / 2.0
        s    = math.sqrt(hdt / 2.0)
        n_dof = len(old.dW)
        # midpoint increment
        w_mid = 0.5 * old.dW + s * rng.standard_normal(n_dof)
        # second half
        w_2nd = old.dW - w_mid
        self._stack.pop()
        # push second half first (it runs after first half)
        self._stack.append(WienerStep(w_2nd, hdt))
        # push first half (runs first - top of stack)
        self._stack.append(WienerStep(w_mid, hdt))

def do_one_full_step(
        advance_fn:   Callable[[np.ndarray, float, float], Tuple[bool, bool]],
        step_back_fn: Callable[[float, float], None],
        rng:          np.random.Generator,
        dW_init:      np.ndarray,
        dt0:          float,
) -> float:
    """
    Execute one full BD step with automatic Wiener subdivision.
    Parameters
    ----------
    advance_fn   : callable(dW, t, dt) -> (is_done, must_backstep)
                   Advances the system state by one BD step.
                   Returns is_done=True if trajectory ended.
                   Returns must_backstep=True if step was too large.
    step_back_fn : callable(t, dt) -> None
                   Restore the system to its pre-step state.
    rng          : random number generator
    dW_init      : initial Wiener increment shape (n_dof,)
    dt0          : nominal time step
    Returns
    -------
    final_dt : the time step that was actually used
    """
    process = WienerProcess(dW_init.copy(), dt0)
    final_dt = dt0
    while not process.at_end:
        dt = process.dt
        t  = process.t
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

def make_initial_dW(n_dof: int, dt: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Generate initial Wiener increment: dW ~ Normal(0, sqrt(dt)*I)
    shape (n_dof,)
    Note: Does not include the sqrt(2) factor - that is in the integrator.
    """
    return math.sqrt(dt) * rng.standard_normal(n_dof)