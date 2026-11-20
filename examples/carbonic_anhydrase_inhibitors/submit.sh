#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for d in ca13_azm ca13_vd1125 ca13_vd1126 ca13_vd1209 ca13_vd1269 ca1_vd1269 ca2_vd1142; do
  echo "Submitting $d ..."
  (cd "$DIR/$d" && sbatch submit.sh)
done
