#!/bin/bash
#SBATCH --job-name=ttk_pystarc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024000
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load cuda
source /mnt/ceph/users/aojha/miniforge3/etc/profile.d/conda.sh
conda activate PySTARC
export OE_LICENSE=/mnt/home/aojha/ceph/licenses/oe_license.txt

BASE=/mnt/ceph/users/aojha/PySTARC_simulations/ttk_inhibitors/3H9F
RUN=/mnt/home/aojha/ceph/PySTARC/run_pystarc.py

cd $BASE
rm -rf bd_sims  
rm -f input.xml rxns.xml receptor.pqr ligand.pqr receptor.pdb ligand.pdb
rm -f protein.prmtop protein.rst7 ligand.prmtop ligand.rst7 *.cache _full.rst7 sqm.out slurm-*.out
python setup.py 
python $RUN input.xml
