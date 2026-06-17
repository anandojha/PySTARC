"""Tests for low-severity robustness guards in pystarc.lib.numerical."""

import pytest

from pystarc.lib.numerical import legendre_p


def test_legendre_p_negative_degree_raises():
    """A negative degree must raise a clear ValueError rather than silently
    returning x (the previous behavior)."""
    with pytest.raises(ValueError):
        legendre_p(-1, 0.5)
    with pytest.raises(ValueError):
        legendre_p(-3, -0.25)


def test_legendre_p_healthy_path_unchanged():
    """The nonnegative-degree path must be unchanged: P0(x)=1, P1(x)=x,
    P2(x)=(3x^2-1)/2."""
    x = 0.3
    assert legendre_p(0, x) == 1.0
    assert legendre_p(1, x) == x
    assert legendre_p(2, x) == pytest.approx((3 * x * x - 1.0) / 2.0)
