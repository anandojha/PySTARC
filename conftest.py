"""Pytest configuration for PySTARC tests."""
from pathlib import Path
import shutil
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

def pytest_sessionfinish(session, exitstatus):
    """Clean up artifacts after all tests complete."""
    for d in ["bd_sims", "__pycache__"]:
        if os.path.isdir(d):
            shutil.rmtree(d)
