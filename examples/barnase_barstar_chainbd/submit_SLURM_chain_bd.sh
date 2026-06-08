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

cd /mnt/home/aojha/ceph/PySTARC_simulations/barnase_barstar_chainbd

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
time python /mnt/home/aojha/ceph/PySTARC/run_pystarc.py input.xml

echo "End: $(date)"
