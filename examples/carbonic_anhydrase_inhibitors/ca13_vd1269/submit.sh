#!/bin/bash
#SBATCH --job-name=ca13_vd1269
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00
# Unified SLURM submission script (1 GPU, 1-node multi-GPU, or multi-node multi-GPUs).
# sbatch submit.sh                                   # 1 node with 4 GPUs (default)
# sbatch --nodes=1 --ntasks-per-node=1 submit.sh     # 1 node with 1 GPU
# sbatch --nodes=2 --ntasks-per-node=4 submit.sh     # 2 nodes with 4 GPUs/node (8 GPUs total)

set -eo pipefail
module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PySTARC
BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${PYSTARC_ROOT:-$BASE}"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
if [ ! -f "$ROOT/run_pystarc.py" ]; then
  ROOT="$(python -c 'import os, pystarc; print(os.path.dirname(os.path.dirname(os.path.abspath(pystarc.__file__))))' 2>/dev/null || true)"
fi
[ -f "$ROOT/run_pystarc.py" ] || { echo "PySTARC not found. Set PYSTARC_ROOT to the PySTARC folder" >&2; exit 1; }
export PYTHONPATH="$ROOT"
export PYSTARC_ROOT="$ROOT"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS"
cd "$BASE"
WORK_DIR="$(python -c 'import os, xml.etree.ElementTree as ET; w=(ET.parse("config.xml").getroot().findtext("compute/work_dir") or "bd_sims").strip(); print(os.path.abspath(w))')"
rm -rf "$WORK_DIR"
N_NODES="${SLURM_JOB_NUM_NODES:-1}"
N_TASKS="${SLURM_NTASKS:-1}"
GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-1}"
GPUS_ON_NODE="${GPUS_ON_NODE%%(*}"
GPUS_ON_NODE="${GPUS_ON_NODE##*:}"
[[ "$GPUS_ON_NODE" =~ ^[0-9]+$ ]] || GPUS_ON_NODE=1
# Single task on one node
if [ "$N_TASKS" -le 1 ] && [ "$N_NODES" -le 1 ]; then
  if [ "$GPUS_ON_NODE" -gt 1 ]; then
    python "$ROOT/run_pystarc.py" config.xml --n-gpus "$GPUS_ON_NODE"
  else
    python "$ROOT/run_pystarc.py" config.xml --n-gpus 1
  fi
  exit 0
fi
# Multiple GPU tasks across 1 or more nodes (1 GPU per SLURM task)
python "$ROOT/pystarc/multi_GPU/multi_GPU_runs.py" config.xml --n-splits "$N_TASKS"
SRUN_ARGS=(--ntasks="$N_TASKS" --kill-on-bad-exit=1)
if [ -n "${SLURM_GPUS_PER_TASK:-}" ]; then
  SRUN_ARGS+=(--gpus-per-task="${SLURM_GPUS_PER_TASK}")
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  SRUN_ARGS+=(--cpus-per-task="${SLURM_CPUS_PER_TASK}")
fi
srun "${SRUN_ARGS[@]}" \
  bash -c '
    sleep $(( ${SLURM_LOCALID:-0} * 2 ))
    T="/tmp/pystarc_${SLURM_JOB_ID:-local}_${SLURM_PROCID:-0}"
    trap "rm -rf \"$T\"" EXIT
    export NUMBA_CACHE_DIR="$T/numba" CUPY_CACHE_DIR="$T/cupy" MPLCONFIGDIR="$T/mpl"
    cd "$1/bd_$((SLURM_PROCID + 1))"
    python "$2/run_pystarc.py" input.xml --n-gpus 1
  ' _ "$WORK_DIR" "$ROOT"
python "$ROOT/pystarc/multi_GPU/combine_data.py" --bd-sims "$WORK_DIR"
