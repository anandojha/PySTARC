"""
Regression tests for the splitting branch of the weighted ensemble resampling
step.

When a bin holds fewer than the target number of trajectories, the weighted
ensemble splits trajectories by cloning until the bin again holds n_per_bin of
them. Following Huber and McCammon, each split takes the heaviest trajectory in
the bin and divides its weight evenly between the original and the clone. By
re-selecting the heaviest trajectory before every split, the cloning is spread
across the trajectories carrying the most weight, so the bin ends with the
target number of trajectories whose weights form an even distribution rather
than a degenerate geometric cascade. The total probability weight in the bin is
conserved exactly.
"""

import numpy as np

from pystarc.simulation.we_simulator import (
    WESimulator,
    WEParameters,
    WETrajectory,
)
from pystarc.transforms.quaternion import Quaternion


def _bare_simulator(n_per_bin, seed=0):
    """
    Build a WESimulator whose resampling behaviour can be exercised directly,
    without constructing the full molecular machinery, by setting only the
    fields that the resampling step reads.
    """
    sim = WESimulator.__new__(WESimulator)
    sim.params = WEParameters(n_per_bin=n_per_bin, n_bins=1)
    sim.rng = np.random.default_rng(seed)
    return sim


def _make_traj(weight, bin_idx=0):
    return WETrajectory(
        position=np.array([1.0, 0.0, 0.0]),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        weight=weight,
        bin_idx=bin_idx,
    )


def test_split_reaches_target_count_with_even_weights_power_of_two():
    """
    Splitting a single trajectory up to a power-of-two target yields exactly
    that many trajectories, all of equal weight, with the total weight
    conserved.
    """
    sim = _bare_simulator(n_per_bin=4)
    w0 = 1.0
    out = sim._resample([_make_traj(w0)])
    assert len(out) == 4
    weights = sorted(t.weight for t in out)
    assert np.allclose(weights, [w0 / 4.0] * 4)
    assert np.isclose(sum(t.weight for t in out), w0)


def test_split_eight_from_one_is_uniform():
    """
    A single trajectory split up to eight produces eight trajectories of equal
    weight, confirming the even split for a larger power-of-two target.
    """
    sim = _bare_simulator(n_per_bin=8)
    w0 = 0.4
    out = sim._resample([_make_traj(w0)])
    assert len(out) == 8
    weights = sorted(t.weight for t in out)
    assert np.allclose(weights, [w0 / 8.0] * 8)
    assert np.isclose(sum(t.weight for t in out), w0)


def test_split_non_power_of_two_is_balanced_not_geometric():
    """
    For a target that is not a power of two the split yields a balanced binary
    distribution rather than a degenerate geometric cascade. Splitting one
    trajectory up to three gives weights w/2, w/4, w/4: the spread between the
    heaviest and lightest is only a factor of two, far from the geometric
    sequence w/2, w/4, w/8 that an un-rebalanced cascade would produce.
    """
    sim = _bare_simulator(n_per_bin=3)
    w0 = 0.6
    out = sim._resample([_make_traj(w0)])
    assert len(out) == 3
    weights = sorted(t.weight for t in out)
    assert np.allclose(weights, [w0 / 4.0, w0 / 4.0, w0 / 2.0])
    assert np.isclose(sum(t.weight for t in out), w0)
    # The heaviest-to-lightest ratio is the balanced value 2, not the geometric
    # cascade value of 4 that splitting the same trajectory repeatedly would give.
    assert np.isclose(max(weights) / min(weights), 2.0)


def test_split_multiple_starting_trajectories_conserves_and_balances():
    """
    Splitting from several starting trajectories of unequal weight still
    reaches the target count, conserves the total weight, and keeps the weight
    spread bounded rather than letting one trajectory dominate.
    """
    sim = _bare_simulator(n_per_bin=6)
    start = [_make_traj(0.5), _make_traj(0.3), _make_traj(0.2)]
    total0 = sum(t.weight for t in start)
    out = sim._resample(start)
    assert len(out) == 6
    assert np.isclose(sum(t.weight for t in out), total0)
    weights = [t.weight for t in out]
    # No resulting weight exceeds the heaviest starting weight, and the spread
    # stays well below the factor that an un-rebalanced geometric cascade of
    # three additional splits on one trajectory would create.
    assert max(weights) <= 0.5 + 1e-12
    assert max(weights) / min(weights) <= 4.0


def test_split_clones_are_independent_objects():
    """
    Each clone produced by the split is a distinct trajectory object whose later
    mutation does not change any other trajectory, so the ensemble can evolve
    the copies independently.
    """
    sim = _bare_simulator(n_per_bin=4)
    out = sim._resample([_make_traj(1.0)])
    assert len({id(t) for t in out}) == 4
    out[0].position[0] = 99.0
    assert all(t.position[0] == 1.0 for t in out[1:])


def test_resample_full_run_conserves_total_weight():
    """
    Across a full resampling pass over an ensemble spanning several bins the
    total probability weight is conserved, so splitting and merging together
    leave the normalisation untouched.
    """
    sim = _bare_simulator(n_per_bin=3)
    sim.params = WEParameters(n_per_bin=3, n_bins=4)
    rng = np.random.default_rng(123)
    trajs = []
    total0 = 0.0
    # Populate bins with uneven occupancy: some under target, some over.
    for b_idx, count in enumerate([1, 3, 5, 2]):
        for _ in range(count):
            w = float(rng.uniform(0.01, 0.2))
            total0 += w
            trajs.append(_make_traj(w, bin_idx=b_idx))
    out = sim._resample(trajs)
    assert np.isclose(sum(t.weight for t in out), total0)
    # Bins that were under target are brought up to target by splitting.
    from collections import Counter

    counts = Counter(t.bin_idx for t in out)
    assert counts[0] == 3  # was 1, split up to 3
    assert counts[1] == 3  # already at target
    assert counts[3] == 3  # was 2, split up to 3
