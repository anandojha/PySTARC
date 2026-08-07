#!/bin/bash
#SBATCH --job-name=bcd_pystarc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load cuda
source /mnt/ceph/users/aojha/miniforge3/etc/profile.d/conda.sh
conda activate PySTARC

BASE=/mnt/home/aojha/ceph/PySTARC_simulations/beta_cyclodextrin_guests/BCD_aspirin
PYSTARC=/mnt/home/aojha/ceph/PySTARC
RUNNER=$PYSTARC/run_pystarc.py
SPLITTER=$PYSTARC/pystarc/multi_GPU/multi_GPU_runs.py
COMBINER=$PYSTARC/pystarc/multi_GPU/combine_data.py

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
