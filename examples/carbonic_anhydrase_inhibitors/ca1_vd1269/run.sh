#!/bin/bash
set -e
cd "$(dirname "$0")"
ROOT=${PYSTARC_ROOT:-$(pwd)}
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
[ -f "$ROOT/run_pystarc.py" ] || { echo "PySTARC not found. Set PYSTARC_ROOT to the PySTARC folder"; exit 1; }
export PYTHONPATH=$ROOT
rm -rf bd_sims
if [ -f setup.py ]; then python setup.py; fi
python "$ROOT/run_pystarc.py" input.xml
if [ -f analytical.py ]; then python analytical.py; fi
if [ -f convergence.py ]; then python convergence.py; fi
if [ -f bb_effect.py ]; then python bb_effect.py; fi
