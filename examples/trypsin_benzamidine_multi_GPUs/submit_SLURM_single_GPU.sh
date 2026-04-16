#!/bin/bash
#SBATCH --job-name=PySTARC_TRP_BEN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=4:00:00

# Load CUDA and activate PySTARC conda environment
module load cuda
eval "$(conda shell.bash hook)"
conda activate PySTARC

# Paths: edit PYSTARC_ROOT to point to your PySTARC installation
PYSTARC_ROOT=$HOME/PySTARC
BASE=$PYSTARC_ROOT/examples/trypsin_benzamidine_multi_GPUs
RUN=$PYSTARC_ROOT/run_pystarc.py

cd $BASE
rm -rf bd_sims
rm -f input.xml rxns.xml receptor.pqr ligand.pqr *.cache _full.rst7 *.out*
python setup.py
python $RUN input.xml
