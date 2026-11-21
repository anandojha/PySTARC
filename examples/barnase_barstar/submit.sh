#!/bin/bash
#SBATCH --job-name=barnase_barstar
#SBATCH --partition=ccb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=125
#SBATCH --mem=600G
#SBATCH --constraint=rome
#SBATCH --time=7-00:00:00
# Unified SLURM submission script for multi-node CPU flexible-chain BD.
# sbatch submit.sh                   # 2 Rome nodes x 125 CPUs (250 CPUs total, default)
# sbatch --nodes=1 submit.sh         # 1 Rome node x 125 CPUs (125 CPUs total)
# sbatch --nodes=20 submit.sh        # 20 Rome nodes x 125 CPUs (2,500 CPUs total)

if ! command -v conda >/dev/null 2>&1; then
  [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
fi
if ! command -v conda >/dev/null 2>&1; then
  for _c in "${CONDA_BASE:-}" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
  done
fi
set -eo pipefail
command -v conda >/dev/null 2>&1 || { echo "conda not found. Load conda or set CONDA_BASE." >&2; exit 1; }
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "$BASE"

N_TASKS="${SLURM_NTASKS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-125}"

rm -rf bd_sims shards
python "$ROOT/run_pystarc.py" config.xml --setup

# Build APBS electrostatic and Born grids once on the head node if absent.
OMP_NUM_THREADS=16 python "$BASE/make_grids.py" bd_sims/input.xml

# Split total trajectories evenly across SLURM tasks (one worker pool of $CPUS per node).
python - "$BASE" "$BASE/bd_sims/input.xml" "$N_TASKS" "$CPUS" <<'PY'
import os
import re
import sys

base, src_xml, n_tasks_s, cpus_s = sys.argv[1:5]
n_tasks = int(n_tasks_s)
cpus = int(cpus_s)

template = open(src_xml).read()
m = re.search(r"<n_trajectories>([^<]+)</n_trajectories>", template)
if not m:
    sys.exit(f"n_trajectories not found in {src_xml}")
total_traj = int(m.group(1).strip())
per_task, rem = divmod(total_traj, n_tasks)

shards_dir = os.path.join(base, "shards")
os.makedirs(shards_dir, exist_ok=True)
print(f"Staging {total_traj} trajectories across {n_tasks} nodes x {cpus} CPUs ({n_tasks * cpus} CPUs total)")

for i in range(1, n_tasks + 1):
    n_i = per_task + (1 if i <= rem else 0)
    seed_i = i * 11111111 + 1
    shard_dir = os.path.join(shards_dir, f"shard_{i:02d}")
    os.makedirs(shard_dir, exist_ok=True)
    wdir_i = os.path.join(shard_dir, "bd_sims")
    txt = template
    txt, n1 = re.subn(r"<n_trajectories>[^<]*</n_trajectories>", f"<n_trajectories>{n_i}</n_trajectories>", txt)
    txt, n2 = re.subn(r"<seed>[^<]*</seed>", f"<seed>{seed_i}</seed>", txt)
    txt, n3 = re.subn(r"<work_dir>[^<]*</work_dir>", f"<work_dir>{wdir_i}</work_dir>", txt)
    txt, n4 = re.subn(r"<n_workers>[^<]*</n_workers>", f"<n_workers>{cpus}</n_workers>", txt)
    assert n1 == 1 and n2 == 1 and n3 == 1 and n4 == 1, (n1, n2, n3, n4)
    with open(os.path.join(shard_dir, "input.xml"), "w") as fh:
        fh.write(txt)
    print(f"  shard_{i:02d}: n_trajectories={n_i} n_workers={cpus} seed={seed_i}")
PY

srun --ntasks="$N_TASKS" --cpus-per-task="$CPUS" --kill-on-bad-exit=1 \
  bash -c '
    set -eo pipefail
    SHARD=$(printf "shard_%02d" $((SLURM_PROCID + 1)))
    T="/tmp/pystarc_${SLURM_JOB_ID:-local}_${SLURM_PROCID:-0}"
    trap "rm -rf \"$T\"" EXIT
    export NUMBA_CACHE_DIR="$T/numba" MPLCONFIGDIR="$T/mpl"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    cd "$1/shards/$SHARD"
    python "$2/run_pystarc.py" input.xml
  ' _ "$BASE" "$ROOT"

python "$BASE/combine_shards.py"
