#!/bin/bash
set -eo pipefail

# Run from the directory that holds this script, and locate the PySTARC
# checkout by walking up until run_pystarc.py is found. No hardcoded paths.
BASE="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$BASE"
while [ ! -f "$RUNNER/run_pystarc.py" ] && [ "$RUNNER" != "/" ]; do RUNNER="$(dirname "$RUNNER")"; done
RUNNER="$RUNNER/run_pystarc.py"
CORES=96
SHARE=(apbs_output)

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
NSHARD=${NSHARD:-25}

mkdir -p shards logs
echo "b $BRAD A   $NTRAJ trajectories x $NSHARD shards = $((NTRAJ*NSHARD))"

for i in $(seq 1 "$NSHARD"); do
    d="shards/shard_$(printf %02d "$i")"
    if [ -f "$d/bd_sims/results.json" ] || squeue -u "$USER" -h -n "chain_$i" 2>/dev/null | grep -q .; then
        echo "  skip $d (done or running)"; continue
    fi
    mkdir -p "$d"
    for f in "${SHARE[@]}"; do
        [ -e "$d/$f" ] || ln -s "$BASE/$f" "$d/$f"
    done
    seed=$(( i * 11111111 + 1 ))
    python - "$BASE/bd_sims/input.xml" "$d/input.xml" "$seed" "$BASE/$d/bd_sims" <<'PY'
import re, sys
src, dst, seed, wdir = sys.argv[1:5]
t = open(src).read()
t, n = re.subn(r"<seed>[^<]*</seed>", f"<seed>{seed}</seed>", t)
assert n == 1, f"seed not found in {src}"
t, n2 = re.subn(r"<work_dir>[^<]*</work_dir>", f"<work_dir>{wdir}</work_dir>", t)
assert n2 == 1, f"work_dir not found in {src}"
open(dst, "w").write(t)
PY
    jid=$(sbatch --parsable \
        -p ccb -N 1 --ntasks-per-node=1 \
        --cpus-per-task=$CORES --mem=600G -t 168:00:00 \
        --job-name="chain_$i" \
        --output="$BASE/logs/shard_${i}.out" \
        --error="$BASE/logs/shard_${i}.err" \
        --wrap "source ~/.bashrc; conda activate PySTARC; cd $BASE/$d; python $RUNNER input.xml")
    echo "  $d  seed $seed  job $jid"
done

echo