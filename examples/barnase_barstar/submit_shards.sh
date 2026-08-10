#!/bin/bash
set -eo pipefail

# Run from the directory that holds this script, and locate the PySTARC
# checkout by walking up until run_pystarc.py is found. No hardcoded paths.
BASE="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$BASE"
while [ ! -f "$RUNNER/run_pystarc.py" ] && [ "$RUNNER" != "/" ]; do RUNNER="$(dirname "$RUNNER")"; done
RUNNER="$RUNNER/run_pystarc.py"
CORES=96
SHARE=(chain.json reaction_pairs.json barnase.pqr apbs_output)

cd "$BASE"
[ -f input.xml ] || { echo "no input.xml, run: python setup.py"; exit 1; }

read -r BRAD NTRAJ NSHARD < <(python - config.xml <<'PY'
import sys, xml.etree.ElementTree as ET
c = {e.tag: (e.text or "").strip()
     for s in ET.parse(sys.argv[1]).getroot() for e in s}
print(c["bd_milestone_radius"], c["n_trajectories"], c["n_shards"])
PY
)

mkdir -p shards logs
echo "b $BRAD A   $NTRAJ trajectories x $NSHARD shards = $((NTRAJ*NSHARD))"

for i in $(seq 1 "$NSHARD"); do
    d="shards/shard_$(printf %02d "$i")"
    mkdir -p "$d"
    for f in "${SHARE[@]}"; do
        [ -e "$d/$f" ] || ln -s "$BASE/$f" "$d/$f"
    done
    seed=$(( i * 11111111 + 1 ))
    python - "$BASE/input.xml" "$d/input.xml" "$seed" <<'PY'
import re, sys
src, dst, seed = sys.argv[1:4]
t = open(src).read()
t, n = re.subn(r"<seed>[^<]*</seed>", f"<seed>{seed}</seed>", t)
assert n == 1, f"seed not found in {src}"
open(dst, "w").write(t)
PY
    jid=$(sbatch --parsable \
        -p ccb --constraint=genoa -N 1 --ntasks-per-node=1 \
        --cpus-per-task=$CORES --mem=600G -t 168:00:00 \
        --job-name="chain_$i" \
        --output="$BASE/logs/shard_${i}.out" \
        --error="$BASE/logs/shard_${i}.err" \
        --wrap "source ~/.bashrc; conda activate PySTARC; cd $BASE/$d; python $RUNNER input.xml")
    echo "  $d  seed $seed  job $jid"
done

echo