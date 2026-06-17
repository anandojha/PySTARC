"""
Tests for the unused-import cleanup in pystarc.simulation.parallel.

The cleanup removed six names that were imported but never referenced
(bd_step, bd_step_adaptive, Atom, Callable, os, sys). These tests confirm
that the module still imports cleanly, that its public API names are still
present, and that the removed names are no longer leaked as module
attributes. No production code path is exercised, so no GPU, APBS, or
external binary is needed.
"""

import importlib


def test_module_imports_cleanly():
    """The module must import without error after the cleanup."""
    mod = importlib.import_module("pystarc.simulation.parallel")
    assert mod is not None


def test_public_api_still_present():
    """The names callers depend on must remain importable."""
    from pystarc.simulation import parallel

    for name in (
        "run_parallel",
        "ParallelBackend",
        "recommended_backend",
        "auto_n_threads",
    ):
        assert hasattr(parallel, name), f"missing public name {name}"


def test_used_imports_retained():
    """Imports that the code actually references must still be bound."""
    from pystarc.simulation import parallel

    for name in ("Molecule", "MobilityTensor", "PathwaySet", "Quaternion"):
        assert hasattr(parallel, name), f"used import {name} was dropped"


def test_unused_imports_removed():
    """The six genuinely unused names must no longer be module attributes."""
    from pystarc.simulation import parallel

    for name in ("bd_step", "bd_step_adaptive", "Atom", "Callable", "os", "sys"):
        assert not hasattr(parallel, name), (
            f"unused import {name} is still bound on the module"
        )
