#!/bin/bash
# Run trypsin-benzamidine BD simulation.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"

echo "PySTARC trypsin-benzamidine simulation"
echo "  PySTARC root: $PYSTARC_ROOT"
echo ""
cd "$SCRIPT_DIR"
# Clean previous outputs
rm -rf bd_sims
rm -f receptor.pqr ligand.pqr input.xml rxns.xml _full.rst7 _grid_gen.xml
echo "  Cleaned previous outputs"
# Setup
echo "  Running setup.py ..."
python setup.py
if [ $? -ne 0 ]; then
    echo "  Error: setup.py failed"
    exit 1
fi
# Run BD
echo "  Running BD simulation ..."
python "$RUNNER" input.xml
