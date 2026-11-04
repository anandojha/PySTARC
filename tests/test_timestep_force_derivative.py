"""
The force-change limit of the adaptive time step (C24).

Δt_force = 0.01 / |D ∂F0/∂r| uses the screened Coulomb force between the net
charges, F0 = -dV/dr with V = V0 exp(-r/λ)/r. The derivative must be the exact
one, -V0 exp(-r/λ)[2/r³ + 2/(r²λ) + 1/(rλ²)], as in BrownDye 2. An earlier
version had 1/λ² in place of 1/(rλ²), which overstated |∂F0/∂r| by up to a
factor of 12 for trypsin-benzamidine and made the step too small.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pystarc.simulation.gpu_batch_simulator import _radial_force_derivative

# Trypsin-benzamidine: V0 = Q_rec Q_lig/(4π ε) in kT·Å and the Debye length.
V0 = 43.06
LAM = 7.858
R = np.array([5.0, 8.0, 12.0, 16.0, 20.0, 23.0])


def _V(r):
    return V0 * np.exp(-r / LAM) / r


def _forces(r):
    return _radial_force_derivative(V0, r, LAM, np.exp(-r / LAM))


def test_force_is_minus_dV_dr():
    h = 1e-5
    fd = -(_V(R + h) - _V(R - h)) / (2 * h)
    Fr0, _ = _forces(R)
    assert Fr0 == pytest.approx(fd, rel=1e-7)


def test_derivative_matches_finite_difference():
    h = 1e-5
    Fp, _ = _forces(R + h)
    Fm, _ = _forces(R - h)
    _, Fr1 = _forces(R)
    assert Fr1 == pytest.approx((Fp - Fm) / (2 * h), rel=1e-6)


def test_derivative_closed_form():
    _, Fr1 = _forces(R)
    exact = -V0 * np.exp(-R / LAM) * (2 / R**3 + 2 / (R**2 * LAM) + 1 / (R * LAM**2))
    assert Fr1 == pytest.approx(exact, rel=1e-12)


def test_old_form_overstated_the_derivative():
    """The earlier 1/λ² term made |∂F0/∂r| 1.44-12.5 times too large at 5-23 Å."""
    Fr0, Fr1 = _forces(R)
    old = -V0 * np.exp(-R / LAM) / LAM**2 - 2.0 * Fr0 / R
    ratio = old / Fr1
    assert ratio == pytest.approx([1.44, 2.43, 4.47, 7.09, 10.07, 12.48], abs=0.01)


def test_scalar_input():
    Fr0, Fr1 = _radial_force_derivative(-1.0, 10.0, 7.0, math.exp(-10.0 / 7.0))
    assert Fr0 < 0 and Fr1 > 0  # attraction weakening with distance
