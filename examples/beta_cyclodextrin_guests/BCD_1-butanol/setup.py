#!/usr/bin/env python3
"""Builds the inputs of this example from config.xml with pystarc.pipeline.build_inputs."""
import os
import sys

if not any(os.path.isdir(os.path.join(p, "pystarc")) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p):
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(10):
        root = os.path.abspath(os.path.join(here, *[".."] * up))
        if os.path.isdir(os.path.join(root, "pystarc", "templates")):
            sys.path.insert(0, root)
            break
from pystarc.pipeline.build_inputs import main  # noqa: E402

main()
