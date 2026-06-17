"""Regression tests for the WCA energy shift in lj_pair_force.

In the WCA treatment the Lennard-Jones potential keeps only its repulsive
branch (r <= 2^(1/6) * sigma) and is shifted up by the well depth so that it is
purely repulsive: V >= 0 everywhere, V -> 0 at the cutoff, and the force is
identical to the unshifted Lennard-Jones force.
"""

import math

import numpy as np

from pystarc.forces.lj import lj_pair_force


EPSILON = 1.3
SIGMA = 2.7
FACTOR = 0.75


def _force_energy(r, use_wca, factor=1.0):
    pos_a = np.zeros(3)
    pos_b = np.array([r, 0.0, 0.0])
    return lj_pair_force(pos_a, pos_b, EPSILON, SIGMA, factor=factor, use_wca=use_wca)


def test_wca_energy_zero_at_cutoff():
    """The WCA energy is zero just inside the cutoff r_cut = 2^(1/6) σ."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    # Evaluate just inside the cutoff to stay within the WCA branch.
    _, energy = _force_energy(r_cut * (1.0 - 1e-9), use_wca=True)
    assert energy == 0.0 or abs(energy) < 1e-6


def test_wca_energy_nonnegative_inside():
    """The WCA energy stays non-negative across separations from 0.5 σ up to the cutoff."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    radii = np.linspace(0.5 * SIGMA, r_cut * (1.0 - 1e-12), 200)
    for r in radii:
        _, energy = _force_energy(r, use_wca=True)
        assert energy >= -1e-9, f"WCA energy negative at r={r}: {energy}"


def test_wca_energy_continuous_at_cutoff():
    """Approaching the cutoff from inside, the WCA energy decreases monotonically to zero and is exactly zero beyond it."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    # Approaching the cutoff from inside, the energy stays non-negative and
    # shrinks monotonically toward zero.
    deltas = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    energies = []
    for delta in deltas:
        _, energy = _force_energy(r_cut - delta, use_wca=True)
        assert energy >= -1e-9
        energies.append(energy)
    for closer, farther in zip(energies[1:], energies[:-1]):
        assert closer <= farther + 1e-12
    # The value adjacent to the cutoff is essentially zero.
    assert energies[-1] < 1e-3
    # Beyond the cutoff the energy is exactly zero (no discontinuity).
    _, e_outside = _force_energy(r_cut * 1.01, use_wca=True)
    assert e_outside == 0.0


def test_wca_force_unchanged():
    # Within the repulsive branch the WCA force matches the plain LJ force.
    """Within the repulsive branch the WCA force equals the plain Lennard-Jones force."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    for r in np.linspace(0.6 * SIGMA, r_cut * (1.0 - 1e-9), 50):
        f_plain, _ = _force_energy(r, use_wca=False, factor=FACTOR)
        f_wca, _ = _force_energy(r, use_wca=True, factor=FACTOR)
        assert np.allclose(f_wca, f_plain, rtol=1e-12, atol=1e-12)


def test_wca_energy_shift_matches_well_depth():
    # The WCA energy is the plain LJ energy plus the well depth factor*eps/4.
    """The WCA energy equals the plain Lennard-Jones energy plus the well depth factor·ε/4."""
    r_cut = 2.0 ** (1.0 / 6.0) * SIGMA
    for r in np.linspace(0.6 * SIGMA, r_cut * (1.0 - 1e-9), 50):
        _, e_plain = _force_energy(r, use_wca=False, factor=FACTOR)
        _, e_wca = _force_energy(r, use_wca=True, factor=FACTOR)
        assert math.isclose(e_wca, e_plain + FACTOR * EPSILON * 0.25, rel_tol=1e-12, abs_tol=1e-12)
