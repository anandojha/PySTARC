"""
General-purpose numerical utilities used throughout PySTARC. This module
provides a natural cubic spline interpolator, Romberg integration, a single
Wiener-process increment for Brownian dynamics, the low-order Cartesian
multipole moments of a charge distribution, and Legendre polynomials.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np
import math


class CubicSpline:
    """
    Natural cubic spline interpolation through a set of data points.

    Construct the spline by passing the sample abscissas and ordinates, for
    example spline = CubicSpline(x_data, y_data). Calling the spline at a point,
    spline(x), returns the interpolated value, and spline.derivative(x) returns
    its first derivative. The natural boundary condition sets the second
    derivative to zero at both ends.
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
        # Assemble the tridiagonal system for the second derivatives m_i, with
        # the natural boundary condition m_0 = m_{n-1} = 0.
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


def romberg_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_order: int = 12,
) -> float:
    """
    Estimate the definite integral of f over the interval [a, b] by Romberg
    integration, which applies repeated Richardson extrapolation to the
    trapezoidal rule. The routine returns once successive diagonal estimates
    agree to within tol, or after max_order refinements.
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


def wiener_step(D: float, dt: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a single Wiener-process increment for a free Brownian step. The
    displacement in each of the dim Cartesian directions is an independent
    Gaussian with zero mean and standard deviation

        σ = √(2 D Δt)

    so that dW ~ N(0, σ²). Here D is the diffusion coefficient and Δt is the
    time step.
    """
    sigma = math.sqrt(2.0 * D * dt)
    return sigma * rng.standard_normal(dim)


def monopole_moment(charges: np.ndarray) -> float:
    """Return the total charge q = Σ qᵢ of the distribution."""
    return float(np.sum(charges))


def dipole_moment(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Return the dipole moment p = Σ qᵢ rᵢ, where rᵢ is the position of charge qᵢ."""
    return (charges[:, None] * positions).sum(axis=0)


def quadrupole_moment(positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """
    Return the traceless Cartesian quadrupole tensor

        Qᵢⱼ = Σ qₖ (3 rₖᵢ rₖⱼ - δᵢⱼ r²)

    where the sum runs over charges qₖ at positions rₖ, the indices i and j label
    the Cartesian components, δᵢⱼ is the Kronecker delta, and r² is the squared
    distance of charge k from the origin.
    """
    Q = np.zeros((3, 3))
    for q, r in zip(charges, positions):
        r2 = np.dot(r, r)
        Q += q * (3 * np.outer(r, r) - r2 * np.eye(3))
    return Q


def legendre_p(n: int, x: float) -> float:
    """Evaluate the Legendre polynomial Pₙ(x) using the standard three-term recurrence."""
    if n < 0:
        raise ValueError(f"legendre_p requires a nonnegative degree n, got n={n}.")
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
    """Evaluate the Legendre series Σ cₙ Pₙ(x), where coeffs holds the coefficients cₙ."""
    return sum(c * legendre_p(n, x) for n, c in enumerate(coeffs))
