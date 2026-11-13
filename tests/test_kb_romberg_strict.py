"""k_b and the return probability come from the Romberg integrals or the run
stops, and k_on is CONV k_b P with no Smoluchowski fallback."""

import math
from types import SimpleNamespace

import pytest
from scipy.integrate import quad

from pystarc.global_defs.constants import VACUUM_PERMITTIVITY_KBT
from pystarc.simulation.gpu_batch_simulator import GPUBatchResult, GPUBatchSimulator


def _stub(debye=7.828, sdie=78.0, q=(6.0, -1.0), b=10.0, r_esc=20.0, a=5.0, D0=0.03, hi=True):
    return SimpleNamespace(
        params=SimpleNamespace(r_start=b, r_escape=r_esc, dielectric=sdie, _kT_scale=1.0),
        _qb_factor=1.1,
        mol1=SimpleNamespace(total_charge=lambda: q[0]),
        mol2=SimpleNamespace(total_charge=lambda: q[1]),
        engine=SimpleNamespace(_debye=debye),
        mob=SimpleNamespace(relative_translational_diffusion=lambda: D0, radius1=a, radius2=a),
        _use_hi=hi,
    )


def _quad_kb(lo_r, debye, sdie, q, a, D0):
    fpe = 4 * math.pi * sdie * VACUUM_PERMITTIVITY_KBT
    ainv, a2 = 2.0 / a, a * a
    dpre = D0 / ainv

    def f(s):
        if s == 0.0:
            return 1.0 / D0
        r = 1.0 / s
        U = q[0] * q[1] * math.exp(-r / debye) / (fpe * r)
        return math.exp(U) / (dpre * (ainv - 3.0 / r + 2.0 * a2 / r**3))

    return 4 * math.pi / quad(f, 0.0, 1.0 / lo_r, epsabs=0, epsrel=1e-12)[0]


def test_k_b_and_return_probability_match_quadrature():
    s = _stub()
    p_b = GPUBatchSimulator._compute_return_prob(s)
    kb = _quad_kb(10.0, 7.828, 78.0, (6.0, -1.0), 5.0, 0.03)
    kq = _quad_kb(20.0, 7.828, 78.0, (6.0, -1.0), 5.0, 0.03)
    assert s._k_b == pytest.approx(kb, rel=1e-7)
    assert p_b == pytest.approx(kb / kq, rel=1e-7)
    assert s._k_b == pytest.approx(3.2656, abs=1e-4) and p_b == pytest.approx(0.5197, abs=1e-4)


@pytest.mark.parametrize("kw", [dict(debye=0.0), dict(debye=-5.0), dict(debye=float("nan")), dict(sdie=0.0)])
def test_unusable_screening_or_dielectric_raises(kw):
    with pytest.raises(ValueError, match="must be positive"):
        GPUBatchSimulator._compute_return_prob(_stub(**kw))


def test_escape_sphere_inside_the_trigger_raises():
    with pytest.raises(ValueError, match="must exceed 1.1 x b"):
        GPUBatchSimulator._compute_return_prob(_stub(r_esc=11.0))


def _result(n_reacted=30, n_escaped=60, n_max=10):
    return GPUBatchResult(n_trajectories=100, n_reacted=n_reacted, n_escaped=n_escaped,
                          n_max_steps=n_max, reaction_counts={}, r_start=10.0, r_escape=20.0,
                          dt=0.0, elapsed_sec=1.0, steps_per_sec=1.0)


def test_rate_is_conv_k_b_p_and_needs_a_k_b():
    r = _result()
    assert r.rate_constant(0.05, 3.0) == pytest.approx(6.022e8 * 3.0 * 0.3, rel=1e-15)
    for kb in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="k_b must be finite and positive"):
            r.rate_constant(0.05, kb)
    lo, hi = r.rate_constant_bounds(0.05, 3.0)
    assert hi == pytest.approx(6.022e8 * 3.0 * 0.4)


def test_ci_uses_z_196_at_95_percent_and_labels_other_levels():
    r = _result()
    p, n, z = 0.3, 100, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert r.reaction_probability_ci(0.95) == pytest.approx((c - m, c + m), rel=1e-15)
    assert "99.9% CI" in r.summary(0.05, 3.0, 0.999) and "58% CI" in r.summary(0.05, 3.0, 0.58)
