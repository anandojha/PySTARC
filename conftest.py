"""Pytest configuration for PySTARC tests."""

import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_sessionfinish(session, exitstatus):
    """Remove directories created by tests using package defaults.

    Some CLI commands and pipeline configs default to writing output
    in cwd-relative directories (bd_sims/, chain_bd_results/). When
    tests invoke these without overriding the path -- and especially
    if a test changes cwd -- droppings can land deep inside package
    source (e.g. pystarc/pipeline/bd_sims/). Walk the tree recursively
    to catch them, skipping vendored and build paths.

    __pycache__ is cleaned only at top-level: Python regenerates these
    on import, and recursive removal would slow subsequent sessions.
    """
    runtime_targets = {"bd_sims", "chain_bd_results"}
    skip_path_parts = {".git", "build", "dist", "pystarc.egg-info", ".github"}

    bases = {_PROJECT_ROOT.resolve(), Path.cwd().resolve()}
    for base in bases:
        if not base.is_dir():
            continue
        for root, dirs, _files in os.walk(str(base), topdown=True):
            dirs[:] = [d for d in dirs if d not in skip_path_parts]
            to_remove = [d for d in list(dirs) if d in runtime_targets]
            for d in to_remove:
                shutil.rmtree(Path(root) / d, ignore_errors=True)
                dirs.remove(d)
        pc = base / "__pycache__"
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)
