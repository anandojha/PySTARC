#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=${PYSTARC_ROOT:-$(pwd)}
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
if [ ! -f "$ROOT/run_pystarc.py" ]; then
  ROOT="$(python -c 'import os, pystarc; print(os.path.dirname(os.path.dirname(os.path.abspath(pystarc.__file__))))' 2>/dev/null || true)"
fi
[ -f "$ROOT/run_pystarc.py" ] || { echo "PySTARC not found. Set PYSTARC_ROOT to the PySTARC folder"; exit 1; }
export PYTHONPATH="$ROOT"
rm -rf bd_sims
python "$ROOT/run_pystarc.py" config.xml
if [ -f analytical.py ]; then python analytical.py; fi
if [ -f convergence.py ]; then python convergence.py; fi
