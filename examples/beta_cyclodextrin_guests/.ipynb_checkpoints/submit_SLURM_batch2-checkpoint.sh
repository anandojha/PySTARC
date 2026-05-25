#!/bin/bash
#SBATCH --job-name=PySTARC_BCD_batch2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=10
#SBATCH --mem=128000
#SBATCH --time=96:00:00
#SBATCH --mail-user=anandojha.002@gmail.com
#SBATCH --mail-type=BEGIN,END
module load cuda
source /mnt/ceph/users/aojha/miniforge3/etc/profile.d/conda.sh
conda activate PySTARC
BASE=/mnt/home/aojha/ceph/PySTARC_simulations/beta_cyclodextrin_guests
RUN=/mnt/home/aojha/ceph/PySTARC/run_pystarc.py

run_system() {
    local dir=$1
    local gpu=$2
    cd ${BASE}/${dir}
    rm -rf bd_sims
    rm -f slurm-*.out input.xml rxns.xml receptor.pqr ligand.pqr *.cache _full.rst7
    python setup.py
    CUDA_VISIBLE_DEVICES=$gpu python $RUN input.xml
}

run_system BCD_aspirin 0 &
run_system BCD_methyl_butyrate 1 &
run_system BCD_tertbutanol 2 &
wait
