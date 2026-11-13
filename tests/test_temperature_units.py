"""The far field and the WCA depth in kBT at the input temperature. The engine
test needs a GPU."""

import numpy as np
import pytest

from pystarc.forces.multipole_farfield import MultipoleExpansion
from pystarc.pipeline.desolvation_grid import coulomb_kbt

LAM, SDIE = 7.86, 78.0
R = np.array([20.0, 35.0, 60.0])


def _si(T, q):
    """Screened Coulomb energy and force of q and +1 e at R, in kBT at T."""
    U = q * coulomb_kbt(T) / SDIE * np.exp(-R / LAM) / R
    return U, U * (1.0 / R + 1.0 / LAM)


@pytest.mark.parametrize("T", [290.0, 298.15, 310.0])
def test_multipole_far_field_is_in_kbt_at_t(T):
    mp = MultipoleExpansion(np.zeros((1, 3)), np.array([-2.0]), LAM, SDIE, kT_scale=T / 298.15)
    U = [mp.potential(np.array([r, 0.0, 0.0])) for r in R]
    assert np.allclose(U, _si(T, -2.0)[0], rtol=2e-4, atol=0.0)
    pos = np.array([[1.0, 0.3, -0.2], [-0.8, 0.1, 0.5], [0.2, -1.1, 0.4]])
    q = np.array([0.6, -1.3, 0.4])
    at = lambda s: MultipoleExpansion(pos, q, LAM, SDIE, kT_scale=s).potential(np.array([12.0, -9.0, 15.0]))
    assert at(T / 298.15) == pytest.approx(at(1.0) * 298.15 / T, rel=1e-13)


@pytest.mark.parametrize("T", [290.0, 298.15, 310.0])
def test_engine_far_field_and_wca_depth_are_in_kbt_at_t(T):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1).sum()
    except Exception:
        pytest.skip("no usable CUDA device")
    from pystarc.forces.gpu_batch_engine import GPUBatchForceEngine

    eng = GPUBatchForceEngine([], [], alpha=0.0, receptor_charge=-2.0, debye_length=LAM, sdie=SDIE,
                              kT_scale=T / 298.15, rec_positions=np.zeros((1, 3)),
                              rec_radii=np.array([1.7]), lig_radii=np.array([1.5]), use_lj=True)
    pos = cp.asarray(np.stack([R, 0 * R, 0 * R], axis=1)[:, None, :])
    f, _, e = eng._yukawa_forces_gpu(pos, cp.asarray([1.0]), centroids=pos[:, 0])
    U, F = _si(T, -2.0)
    assert np.allclose(cp.asnumpy(e), U, rtol=2e-4, atol=0.0)
    assert np.allclose(cp.asnumpy(f)[:, 0], F, rtol=2e-4, atol=0.0)
    assert eng._lj_epsilon * T / 298.15 == pytest.approx(0.17, rel=1e-15)
