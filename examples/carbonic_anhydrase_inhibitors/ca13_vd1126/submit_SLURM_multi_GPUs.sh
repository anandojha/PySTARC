#!/bin/bash
#SBATCH --job-name=pystarc_multi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=24:00:00

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PySTARC

BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="$BASE"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
SPLITTER="$ROOT/pystarc/multi_GPU/multi_GPU_runs.py"
COMBINER="$ROOT/pystarc/multi_GPU/combine_data.py"

cd "$BASE"
rm -rf bd_sims
if [ -f setup.py ]; then
  rm -f input.xml rxns.xml receptor.pqr ligand.pqr receptor.pdb ligand.pdb
  rm -f protein.prmtop protein.rst7 ligand.prmtop ligand.rst7 *.cache _full.rst7
  python setup.py
fi

python "$SPLITTER" input.xml --n-splits 4
for i in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((i-1)) bash -c "cd \"$BASE/bd_sims/bd_$i\" && python \"$ROOT/run_pystarc.py\" input.xml" &
  sleep 10
done
wait
python "$COMBINER"
