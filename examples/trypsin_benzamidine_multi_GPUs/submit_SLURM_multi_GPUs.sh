#!/bin/bash
#SBATCH --job-name=PySTARC_TRP_BEN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=4:00:00

# Load CUDA and activate PySTARC conda environment
module load cuda
eval "$(conda shell.bash hook)"
conda activate PySTARC

# Paths: edit PYSTARC_ROOT to point to your PySTARC installation
PYSTARC_ROOT=$HOME/PySTARC
BASE=$PYSTARC_ROOT/examples/trypsin_benzamidine_multi_GPUs
RUNNER=$PYSTARC_ROOT/run_pystarc.py
SPLITTER=$PYSTARC_ROOT/pystarc/multi_GPU/multi_GPU_runs.py
COMBINER=$PYSTARC_ROOT/pystarc/multi_GPU/combine_data.py

cd $BASE
rm -rf bd_sims
rm -f input.xml rxns.xml receptor.pqr ligand.pqr *.cache _full.rst7 *.out*
python setup.py

python $SPLITTER input.xml --n-splits 4

CUDA_VISIBLE_DEVICES=0 bash -c "cd $BASE/bd_sims/bd_1 && python $RUNNER input.xml" &
sleep 10
CUDA_VISIBLE_DEVICES=1 bash -c "cd $BASE/bd_sims/bd_2 && python $RUNNER input.xml" &
sleep 10
CUDA_VISIBLE_DEVICES=2 bash -c "cd $BASE/bd_sims/bd_3 && python $RUNNER input.xml" &
sleep 10
CUDA_VISIBLE_DEVICES=3 bash -c "cd $BASE/bd_sims/bd_4 && python $RUNNER input.xml" &
wait

python $COMBINER
