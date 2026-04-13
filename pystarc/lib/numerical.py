"""
Numerical utilities for PySTARC.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np
import math

# Cubic spline


class CubicSpline:
    """
    Natural cubic spline interpolation.
    Usage::
        spline = CubicSpline(x_data, y_data)
        y = spline(x)
        dy = spline.derivative(x)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        assert len(x) == len(y) and len(x) >= 2
        self.x = x
        self.y = y
        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        n = len(self.x)
        h = np.diff(self.x)
        # Set up tridiagonal system for natural spline (m_0 = m_{n-1} = 0)
        A = np.zeros((n, n))
        b = np.zeros(n)
        A[0, 0] = 1.0
        A[n - 1, n - 1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3 * (
                (self.y[i + 1] - self.y[i]) / h[i]
                - (self.y[i] - self.y[i - 1]) / h[i - 1]
            )
        self.m = np.linalg.solve(A, b)

    def _find_interval(self, x: float) -> int:
        idx = np.searchsorted(self.x, x, side="right") - 1
        return int(np.clip(idx, 0, len(self.x) - 2))

    def __call__(self, x: float) -> float:
        i = self._find_interval(x)
        h = self.x[i + 1] - self.x[i]
        t = (x - self.x[i]) / h
        a = self.y[i]
        b = (self.y[i + 1] - self.y[i]) / h - h * (2 * self.m[i] + self.m[i + 1]) / 3
        c = self.m[i]
        d = (self.m[i + 1] - self.m[i]) / (3 * h)
        dx = x - self.x[i]
        return float(a + b * dx + c * dx**2 + d * dx**3)

    def derivative(self, x: float) -> float:
        i = self._find_interval(x)
        h = self.x[i + 1] - self.x[i]
        b = (self.y[i + 1] - self.y[i]) / h - h * (2 * self.m[i] + self.m[i + 1]) / 3
        c = self.m[i]
        d = (self.m[i + 1] - self.m[i]) / (3 * h)
        dx = x - self.x[i]
        return float(b + 2 * c * dx + 3 * d * dx**2)


# Romberg integration
def romberg_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_order: int = 12,
) -> float:
    """
    Romberg integration of f on [a, b].
    Returns integral estimate.
    """
    R = [[0.0] * (max_order + 1) for _ in range(max_order + 1)]
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))
    for i in range(1, max_order + 1):
        h /= 2.0
        n_new = 2**i
        sumval = sum(f(a + (2 * k - 1) * h) for k in range(1, n_new // 2 + 1))
        R[i][0] = 0.5 * R[i - 1][0] + h * sumval
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4**j - 1)
        if i >= 2 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i]
    return R[max_order][max_order]


# Wiener process
def wiener_step(D: float, dt: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one Wiener process increment: σ = √(2Ddt), dW ~ N(0, σ²).
    """
    sigma = math.sqrt(2.0 * D * dt)
    return sigma * rng.standard_normal(dim)


# Cartesian multipoles
def monopole_moment(charges: np.ndarray) -> float:
    """Total charge q = Σ qᵢ."""
    return float(np.sum(charges))


def dipole_moment(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Dipole moment p = Σ qᵢ rᵢ."""
    return (charges[:, None] * positions).sum(axis=0)


def quadrupole_moment(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Traceless quadrupole tensor Qᵢⱼ = Σ qₖ(3rₖᵢ rₖⱼ - δᵢⱼ r²)."""
    Q = np.zeros((3, 3))
    for q, r in zip(charges, positions):
        r2 = np.dot(r, r)
        Q += q * (3 * np.outer(r, r) - r2 * np.eye(3))
    return Q


# Legendre polynomials
def legendre_p(n: int, x: float) -> float:
    """Legendre polynomial Pₙ(x) via recurrence."""
    if n == 0:
        return 1.0
    if n == 1:
        return x
    p_prev, p_curr = 1.0, x
    for k in range(2, n + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        p_prev, p_curr = p_curr, p_next
    return p_curr


def legendre_series(coeffs: List[float], x: float) -> float:
    """Evaluate Σ cₙ Pₙ(x)."""
    return sum(c * legendre_p(n, x) for n, c in enumerate(coeffs))
