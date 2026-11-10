"""
Constants of the RPY parallel diffusion D_par(r) in the GPU step.

With hydrodynamic interactions the GPU step moves each trajectory with D_par
at the separation at the start of the step. The constants of D_par come from
the mobility D_t. D_par must then equal the radial RPY relative diffusion of
the mobility at every separation and at the temperature the mobility was built
for. k_b, p_b and the step without hydrodynamic interactions take D_t from the
same mobility. The GPU class needs CuPy. The constants are therefore taken
from GPUBatchSimulator._hi_constants on an instance built without __init__.
D_par and dD/dr are then formed in NumPy with the expressions of the step loop
in GPUBatchSimulator.run.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator

TEMPERATURES = (290.0, 298.15, 310.0)
# Equal radii, a small sphere beside a larger one in either order, and a large
# ratio of radii.
RADII = ((2.0, 2.0), (6.0, 1.0), (1.5, 9.0), (20.0, 5.0), (30.0, 3.0))
# Any direction gives the same radial projection.
U = np.array([1.0, 2.0, 2.0]) / 3.0


def _constants(mob: MobilityTensor):
    sim = object.__new__(GPUBatchSimulator)
    sim.mob = mob
    return sim._hi_constants(float(mob.relative_translational_diffusion()))


def _step_D(hi, r):
    """D_par and dD/dr at the separations r as the step loop forms them."""
    dpre, D0, Deng, a2, S, Ad, am2, p = hi
    rm1 = 1.0 / np.maximum(r, np.full_like(r, 0.01))
    rm2 = rm1 * rm1
    rm3 = rm2 * rm1
    D_far = D0 - dpre * (3.0 * rm1 - 2.0 * a2 * rm3)
    D_ov = D0 - (dpre / (8.0 * p)) * (
        8.0 * S - 6.0 * am2 * rm1 - 3.0 * r + am2 * am2 * rm3
    )
    D = np.where(r > S, D_far, np.where(r > Ad, D_ov, Deng))
    D = np.maximum(D, np.full_like(D, 1e-6))
    dD_far = dpre * (3.0 * rm2 - 6.0 * a2 * rm1 * rm3)
    dD_ov = (3.0 * dpre / (8.0 * p)) * (1.0 - am2 * rm2) ** 2
    dD = np.where(r > S, dD_far, np.where(r > Ad, dD_ov, 0.0))
    return D, dD


def _regimes(a0: float, a1: float):
    """Separations in each regime of the RPY tensor. Equal radii have no
    engulfed regime."""
    S, Ad = a0 + a1, abs(a0 - a1)
    out = {
        "far": S * np.array([1.01, 1.1, 2.0, 5.0, 30.0, 1e3]),
        "overlap": Ad + (S - Ad) * np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    }
    if Ad > 0.0:
        out["engulfed"] = Ad * np.array([0.05, 0.5, 0.95])
    return out


def _mobility_D(mob: MobilityTensor, r):
    return np.array([mob.relative_translational_diffusion(x * U) for x in r])


@pytest.mark.parametrize("T", TEMPERATURES)
@pytest.mark.parametrize("a0, a1", RADII)
def test_D_par_equals_mobility_in_every_regime(T, a0, a1):
    mob = MobilityTensor.from_radii(a0, a1, T=T)
    hi = _constants(mob)
    for regime, r in _regimes(a0, a1).items():
        D, _ = _step_D(hi, r)
        ref = _mobility_D(mob, r)
        rel = np.max(np.abs(D / ref - 1.0))
        assert rel < 1e-12, (regime, rel)


@pytest.mark.parametrize("T", TEMPERATURES)
@pytest.mark.parametrize("a0, a1", RADII)
def test_divergence_equals_slope_of_mobility(T, a0, a1):
    """dD/dr in the drift is the slope of the mobility D in every regime."""
    mob = MobilityTensor.from_radii(a0, a1, T=T)
    hi = _constants(mob)
    dpre = hi[0]
    for regime, r in _regimes(a0, a1).items():
        r = r[r < 10.0 * (a0 + a1)]
        _, dD = _step_D(hi, r)
        h = 1e-5 * r
        slope = (_mobility_D(mob, r + h) - _mobility_D(mob, r - h)) / (2.0 * h)
        # dpre / r^2 sets the size of dD/dr near contact and scales the error.
        err = np.max(np.abs(dD - slope) * r * r / dpre)
        assert err < 1e-8, (regime, err)


@pytest.mark.parametrize("T", TEMPERATURES)
def test_far_field_value_is_D_t(T):
    for a0, a1 in RADII:
        mob = MobilityTensor.from_radii(a0, a1, T=T)
        D_t = float(mob.relative_translational_diffusion())
        dpre, D0, *_ = _constants(mob)
        assert D0 == D_t
        assert abs(dpre * (1.0 / a0 + 1.0 / a1) / D_t - 1.0) < 1e-15


@pytest.mark.parametrize("T", (290.0, 310.0))
def test_constants_follow_the_temperature(T):
    """dpre, D0 and Deng scale as T like the mobility. The radius terms do not
    change. Constants fixed at one temperature fail here."""
    for a0, a1 in RADII:
        ref = _constants(MobilityTensor.from_radii(a0, a1, T=298.15))
        at_T = _constants(MobilityTensor.from_radii(a0, a1, T=T))
        for k in range(3):
            if ref[k] != 0.0:
                assert abs(at_T[k] / ref[k] - T / 298.15) < 1e-13
            else:
                assert abs(at_T[k]) < 1e-15
        assert at_T[3:] == ref[3:]


def test_D_par_follows_the_viscosity_of_the_mobility():
    mob = MobilityTensor.from_radii(6.0, 1.0, eta=6.9e-4, T=310.0)
    hi = _constants(mob)
    for regime, r in _regimes(6.0, 1.0).items():
        D, _ = _step_D(hi, r)
        rel = np.max(np.abs(D / _mobility_D(mob, r) - 1.0))
        assert rel < 1e-12, (regime, rel)


def test_radii_must_be_positive():
    mob = MobilityTensor(0.1, 0.01, 0.2, 0.02, radius1=0.0, radius2=3.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _constants(mob)
