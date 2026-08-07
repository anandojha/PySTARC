#!/bin/bash
#SBATCH --job-name=trypsin_pystarc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load cuda
source /mnt/ceph/users/aojha/miniforge3/etc/profile.d/conda.sh
conda activate PySTARC

BASE=/mnt/home/aojha/ceph/PySTARC_simulations/trypsin_benzamidine
RUN=/mnt/home/aojha/ceph/PySTARC/run_pystarc.py

cd $BASE
rm -rf bd_sims
rm -f input.xml rxns.xml receptor.pqr ligand.pqr *.cache _full.rst7 *.out*
python setup.py
python $RUN input.xml
