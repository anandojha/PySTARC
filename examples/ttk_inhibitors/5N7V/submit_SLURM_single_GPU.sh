#!/bin/bash
#SBATCH --job-name=pystarc_single
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PySTARC

BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="$BASE"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done

cd "$BASE"
rm -rf bd_sims
if [ -f setup.py ]; then
  rm -f input.xml rxns.xml receptor.pqr ligand.pqr receptor.pdb ligand.pdb
  rm -f protein.prmtop protein.rst7 ligand.prmtop ligand.rst7 *.cache _full.rst7
  python setup.py
fi
python "$ROOT/run_pystarc.py" input.xml
