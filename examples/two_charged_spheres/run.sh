#!/bin/bash
# Run two charged spheres BD simulation, verify against analytical solution, and run convergence test.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"

echo "PySTARC two charged spheres (analytical validation)"
echo "  PySTARC root: $PYSTARC_ROOT"
echo ""
cd "$SCRIPT_DIR"
# Clean previous outputs
rm -rf bd_sims
echo "  Cleaned previous outputs"
# Run BD
echo "  Running BD simulation ..."
python "$RUNNER" input.xml
if [ $? -ne 0 ]; then
    echo "  Error: BD simulation failed"
    exit 1
fi
# Verify against analytical solution
echo ""
echo "  Verifying against analytical solution ..."
python analytical.py
# Run multi-seed convergence test
echo ""
echo "  Running convergence test (4 seeds x 10k trajectories) ..."
python convergence.py
