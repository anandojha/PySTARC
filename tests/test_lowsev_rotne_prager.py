"""Low-severity robustness tests for pystarc.hydrodynamics.rotne_prager.

These tests cover two input-validation fixes:

1. _hydrodynamic_center raises a clear ValueError when the total bead
   radius is non-positive (empty or all-zero input) instead of dividing
   by zero and returning NaN with a RuntimeWarning.

2. chain_diffusion_tensors validates the bead count before the heavy
   resistance computation, so empty input raises the same ValueError it
   already raised, without leaking a divide-by-zero RuntimeWarning first.

All cases run on CPU with numpy only, with no GPU, APBS, or external
binaries.
"""

import warnings

import numpy as np
import pytest

from pystarc.hydrodynamics.rotne_prager import (
    _hydrodynamic_center,
    chain_diffusion_tensors,
)


def test_hydrodynamic_center_healthy_path_unchanged():
    # Radius-weighted centroid for positive radii must be exact.
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([1.0, 3.0])
    hc = _hydrodynamic_center(positions, radii)
    expected = (1.0 * positions[0] + 3.0 * positions[1]) / 4.0
    np.testing.assert_allclose(hc, expected)


def test_hydrodynamic_center_empty_raises_without_runtime_warning():
    # Empty input must raise ValueError and must not leak a RuntimeWarning
    # from a divide-by-zero. Promoting warnings to errors catches a leak.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            _hydrodynamic_center(np.empty((0, 3)), np.empty((0,)))


def test_hydrodynamic_center_all_zero_radii_raises():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            _hydrodynamic_center(positions, radii)


def test_chain_diffusion_tensors_empty_raises_same_error_without_warning():
    # The empty-input guard now runs first, so the existing message is
    # raised without a leaked RuntimeWarning from the resistance step.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="at least one bead required"):
            chain_diffusion_tensors(np.empty((0, 3)), np.empty((0,)))
