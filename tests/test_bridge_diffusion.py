"""
The diffusion coefficient of the Brownian bridge in the GPU step.

The bridge gives each reaction pair the chance exp(-x0 x1 / (D dt)) of having
touched its cutoff between two steps. D must be the diffusion coefficient the
step moved the pair with. With hydrodynamic interactions the step uses D_par at
the separation at the start of the step and not the far field D_trans. A bridge
with D_trans counts crossings the move does not make and raises P_rxn. Without
hydrodynamic interactions the bridge is unchanged bit for bit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pystarc.simulation.gpu_batch_simulator import GPUBatchSimulator

D_TRANS = 0.25
D_ROT = np.array([0.0, 0.03, 0.11])  # D_rot × L² of three pairs
D_PAR = np.array([0.06, 0.11, 0.2, 0.25])  # D_par of four trajectories
DT = np.array([0.2, 0.05, 0.3, 1.0])


def _sim(use_hi: bool, pairs: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        _use_hi=use_hi,
        _bb_D_eff=D_TRANS + D_ROT if pairs else None,
        _bb_D_rot=D_ROT if pairs else None,
    )


def _bridge_D_dt(sim: SimpleNamespace) -> np.ndarray:
    return GPUBatchSimulator._bridge_D_dt(sim, D_PAR, DT, D_TRANS)


class TestBridgeDiffusion:
    def test_hi_takes_the_step_D_of_each_trajectory(self):
        out = _bridge_D_dt(_sim(use_hi=True))
        assert out.shape == (4, 3)
        for i in range(4):
            for k in range(3):
                assert out[i, k] == (D_PAR[i] + D_ROT[k]) * DT[i]

    def test_hi_never_takes_the_far_field_D(self):
        out = _bridge_D_dt(_sim(use_hi=True))
        far = (D_TRANS + D_ROT)[None, :] * DT[:, None]
        assert np.all(out[:3] < far[:3])
        assert np.array_equal(out[3], far[3])  # D_par equals D_trans far away

    def test_without_hi_the_bridge_is_unchanged(self):
        sim = _sim(use_hi=False)
        before_fix = sim._bb_D_eff[None, :] * DT[:, None]
        assert np.array_equal(_bridge_D_dt(sim), before_fix)

    def test_without_pair_atoms(self):
        out_hi = _bridge_D_dt(_sim(use_hi=True, pairs=False))
        assert np.array_equal(out_hi, D_PAR[:, None] * DT[:, None])
        out_free = _bridge_D_dt(_sim(use_hi=False, pairs=False))
        assert np.array_equal(out_free, D_TRANS * DT[:, None])
