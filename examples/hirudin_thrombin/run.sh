#!/bin/bash
# Run hirudin C-tail / thrombin chain BD simulation.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"

echo "PySTARC hirudin-thrombin chain BD simulation"
echo "  PySTARC root: $PYSTARC_ROOT"
echo ""
cd "$SCRIPT_DIR"
# Clean previous outputs (only files setup.py generates)
rm -rf bd_sims
rm -f chain.json input.xml
echo "  Cleaned previous outputs"
# Setup
echo "  Running setup.py ..."
python setup.py
if [ $? -ne 0 ]; then
    echo "  Error: setup.py failed"
    exit 1
fi
# Verify DX grids (required for chain BD; setup.py warns but does not fail)
if [ ! -f apbs_output/thrombin1.dx ] || [ ! -f apbs_output/thrombin1_born.dx ]; then
    echo ""
    echo "  Error: APBS DX grids missing in apbs_output/"
    echo "  Copy from archive (one-time):"
    echo "    mkdir -p apbs_output"
    echo "    cp /mnt/home/aojha/Downloads/pystarc_backups/PySTARC_v2_results_archive/hirudin_thrombin/apbs_output/thrombin1.dx apbs_output/"
    echo "    cp /mnt/home/aojha/Downloads/pystarc_backups/PySTARC_v2_results_archive/hirudin_thrombin/apbs_output/thrombin1_born.dx apbs_output/"
    exit 1
fi
# Run BD
echo "  Running chain BD simulation ..."
python "$RUNNER" input.xml
