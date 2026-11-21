#!/bin/bash
set -eo pipefail

# Stage and submit independent single-node CPU shard jobs for barnase_barstar.
BASE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PYSTARC_ROOT:-$BASE}"
while [ ! -f "$ROOT/run_pystarc.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
if [ ! -f "$ROOT/run_pystarc.py" ]; then
  ROOT="$(python -c 'import os, pystarc; print(os.path.dirname(os.path.dirname(os.path.abspath(pystarc.__file__))))' 2>/dev/null || true)"
fi
[ -f "$ROOT/run_pystarc.py" ] || { echo "PySTARC not found. Set PYSTARC_ROOT to the PySTARC folder" >&2; exit 1; }
RUNNER="$ROOT/run_pystarc.py"
CORES=${CORES:-125}
NSHARD=${NSHARD:-2}

cd "$BASE"
rm -rf bd_sims
python "$RUNNER" config.xml --setup

read -r BRAD NTRAJ < <(python - config.xml <<'PY'
import sys, xml.etree.ElementTree as ET
c = {e.tag: (e.text or "").strip()
     for s in ET.parse(sys.argv[1]).getroot() for e in s}
print(c["bd_milestone_radius"], c["n_trajectories"])
PY
)
PER_SHARD=$(( NTRAJ / NSHARD ))
REM=$(( NTRAJ % NSHARD ))

mkdir -p shards logs
echo "b $BRAD A   $NTRAJ total trajectories across $NSHARD shards x $CORES CPUs"

for i in $(seq 1 "$NSHARD"); do
    d="shards/shard_$(printf %02d "$i")"
    if [ -f "$d/bd_sims/results.json" ] || squeue -u "$USER" -h -n "chain_$i" 2>/dev/null | grep -q .; then
        echo "  skip $d (done or running)"; continue
    fi
    mkdir -p "$d"
    n_i=$PER_SHARD
    [ "$i" -le "$REM" ] && n_i=$(( n_i + 1 ))
    seed=$(( i * 11111111 + 1 ))
    python - "$BASE/bd_sims/input.xml" "$d/input.xml" "$n_i" "$seed" "$BASE/$d/bd_sims" "$CORES" <<'PY'
import re, sys
src, dst, n_i, seed, wdir, cores = sys.argv[1:7]
t = open(src).read()
t, n0 = re.subn(r"<n_trajectories>[^<]*</n_trajectories>", f"<n_trajectories>{n_i}</n_trajectories>", t)
t, n1 = re.subn(r"<seed>[^<]*</seed>", f"<seed>{seed}</seed>", t)
t, n2 = re.subn(r"<work_dir>[^<]*</work_dir>", f"<work_dir>{wdir}</work_dir>", t)
t, n3 = re.subn(r"<n_workers>[^<]*</n_workers>", f"<n_workers>{cores}</n_workers>", t)
assert n0 == 1 and n1 == 1 and n2 == 1 and n3 == 1, (n0, n1, n2, n3)
open(dst, "w").write(t)
PY
    jid=$(sbatch --parsable \
        -p ccb --constraint=rome -N 1 --ntasks-per-node=1 \
        --cpus-per-task="$CORES" --mem=600G -t 168:00:00 \
        --job-name="chain_$i" \
        --output="$BASE/logs/shard_${i}.out" \
        --error="$BASE/logs/shard_${i}.err" \
        --wrap "source ~/.bashrc; conda activate PySTARC; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$ROOT; cd $BASE/$d; python $RUNNER input.xml")
    echo "  $d  n=$n_i  seed=$seed  job=$jid"
done

echo
