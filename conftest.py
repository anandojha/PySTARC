"""Pytest configuration for PySTARC tests."""

from pathlib import Path
import shutil
import sys
import os

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

    __pycache__ is cleaned only at the top-level: Python regenerates these
    on import, and recursive removal would slow subsequent sessions.
    """
    runtime_targets = {"bd_sims", "chain_bd_results"}
    skip_path_parts = {".git", "build", "dist", "pystarc.egg-info", ".github"}

    # Safety: only walk inside the project root. Without this guard,
    # invoking pytest from outside the project (e.g. cd ~ && pytest tests/)
    # would walk the user's home tree and rmtree any directory named
    # 'bd_sims' or 'chain_bd_results' it encountered, regardless of owner.
    project_root = _PROJECT_ROOT.resolve()
    bases = {project_root}
    cwd = Path.cwd().resolve()
    try:
        cwd.relative_to(project_root)
        bases.add(cwd)
    except ValueError:
        # cwd is outside the project; don't walk it.
        pass
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
