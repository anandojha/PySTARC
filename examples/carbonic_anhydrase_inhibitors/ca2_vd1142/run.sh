#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"

cd "$SCRIPT_DIR"

# Clean previous outputs
rm -rf bd_sims
rm -f input.xml rxns.xml receptor.pqr ligand.pqr receptor.pdb ligand.pdb
rm -f protein.prmtop protein.rst7 ligand.prmtop ligand.rst7 *.cache *.out

# Step 1: Generate input files
python setup.py || { echo "setup.py failed"; exit 1; }

# Step 2: Run BD
python "$RUNNER" input.xml || { echo "BD failed"; exit 1; }
