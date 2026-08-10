#!/bin/bash
#SBATCH --job-name=barstar_chainbd
#SBATCH --partition=ccb
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=72:00:00
#SBATCH --output=barstar_chainbd_%j.out
#SBATCH --error=barstar_chainbd_%j.err

# Source environment BEFORE enabling strict mode --
# /etc/bashrc references unbound variables that trip 'set -u'
source ~/.bashrc
conda activate PySTARC

# Strict mode for the actual run (no -u)
set -eo pipefail

# Run in the directory this job was submitted from, and locate the PySTARC
# checkout by walking up until run_pystarc.py is found. No hardcoded paths.
BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$BASE"
ROOT="$BASE"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done

echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-not-set}"
echo "Python:    $(which python)"

# Step 1: Regenerate all inputs from setup.py
echo ""
echo "=== Running setup.py ==="
time python setup.py

# Step 2: Run the chain BD simulation
echo ""
echo "=== Running chain BD ==="
time python "$ROOT/run_pystarc.py" input.xml

echo "End: $(date)"
