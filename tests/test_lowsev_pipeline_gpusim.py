"""Tests for the gpu_sim guard in pystarc.pipeline.pipeline.run.

These tests target a low-severity robustness fix. In the run function the
k_b estimate is selected with

    _k_b = getattr(gpu_sim, "_k_b", 0.0) if gpu_sim is not None else 0.0

Previously the selection was guarded by cfg.gpu while gpu_sim was only bound
inside the branch where both cfg.gpu was set and the engine backend was cupy.
If a user requested the GPU but cupy was unavailable and the engine fell back
to a CPU backend, gpu_sim was never bound and reading it raised NameError.

The fix initializes gpu_sim = None before the branch and guards the access on
gpu_sim is not None. These tests reproduce that selection logic directly so
they can run without a GPU, APBS, or any external binary. They confirm:

  1. When gpu_sim is None the selection returns 0.0 and does not raise, which
     is the Smoluchowski fallback used by the CPU path.
  2. When gpu_sim is a simulator-like object carrying _k_b the selection
     returns that value unchanged (the healthy GPU path).
  3. The module source still contains the defensive initialization and the
     None guard so the fix cannot silently regress.
"""

import ast
import inspect

import pystarc.pipeline.pipeline as pipeline


def _select_k_b(gpu_sim):
    """Reproduce the k_b selection expression from run() exactly."""
    return getattr(gpu_sim, "_k_b", 0.0) if gpu_sim is not None else 0.0


class _StubGpuSim:
    """Stand-in for GPUBatchSimulator carrying a Romberg k_b estimate."""

    def __init__(self, k_b):
        self._k_b = k_b


def test_k_b_selection_none_returns_zero_without_raising():
    # The CPU fallback case: gpu_sim is None. The previous code path could hit
    # an unbound name here, so the key assertion is that nothing is raised and
    # the Smoluchowski sentinel 0.0 is selected.
    assert _select_k_b(None) == 0.0


def test_k_b_selection_uses_attribute_on_healthy_gpu_path():
    sim = _StubGpuSim(1.2345)
    assert _select_k_b(sim) == 1.2345


def test_k_b_selection_attribute_missing_falls_back_to_zero():
    # A simulator-like object that never set _k_b still resolves to 0.0 via the
    # getattr default rather than raising AttributeError.
    class _NoKB:
        pass

    assert _select_k_b(_NoKB()) == 0.0


def test_run_source_initializes_gpu_sim_and_guards_access():
    src = inspect.getsource(pipeline.run)
    # The defensive initialization must be present before any branch uses it.
    assert "gpu_sim = None" in src
    # The access must be guarded on the object existing rather than on cfg.gpu,
    # which is what previously allowed the unbound-name failure.
    assert 'getattr(gpu_sim, "_k_b", 0.0) if gpu_sim is not None else 0.0' in src
    # Make sure the source still parses (guards against an accidental syntax
    # break introduced alongside the fix).
    ast.parse(src.strip())
