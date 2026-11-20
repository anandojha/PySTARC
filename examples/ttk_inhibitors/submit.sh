#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for d in 2X9E 3GFW 3H9F 5LJJ 5N7V 5N84 5N93 5NAD; do
  echo "Submitting $d ..."
  (cd "$DIR/$d" && sbatch submit.sh)
done
