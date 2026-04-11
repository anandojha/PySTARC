#!/bin/bash
# Run thrombin-thrombomodulin BD simulation and Brownian bridge diagnostic.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"

echo "PySTARC thrombin-thrombomodulin simulation"
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
# Run Brownian bridge diagnostic
echo ""
echo "  Running Brownian bridge A/B test (4 seeds x 10k trajectories) ..."
python bb_effect.py
