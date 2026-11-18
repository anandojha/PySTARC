#!/bin/bash
#SBATCH --job-name=pystarc_multi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=24:00:00

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PySTARC

BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROOT=${PYSTARC_ROOT:-$BASE}
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
[ -f "$ROOT/run_pystarc.py" ] || { echo "PySTARC not found. Set PYSTARC_ROOT to the PySTARC folder"; exit 1; }
export PYTHONPATH=$ROOT

cd "$BASE"
rm -rf bd_sims
python "$ROOT/run_pystarc.py" config.xml --n-gpus 4
