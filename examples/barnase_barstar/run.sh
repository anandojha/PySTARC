#!/bin/bash
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
rm -rf bd_sims
if [ -f setup.py ]; then python setup.py; fi
python "$ROOT/run_pystarc.py" input.xml
if [ -f analytical.py ]; then python analytical.py; fi
if [ -f convergence.py ]; then python convergence.py; fi
