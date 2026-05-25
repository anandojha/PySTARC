#!/bin/bash
#SBATCH --job-name=PySTARC_BCD_batch1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
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

run_system BCD_1-butanol 0 &
run_system BCD_1-naphthylethanol 1 &
run_system BCD_1-propanol 2 &
run_system BCD_2-naphthylethanol 3 &
wait
