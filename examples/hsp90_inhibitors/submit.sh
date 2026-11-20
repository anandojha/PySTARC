#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for d in HSP90-aminopyridine HSP90-indazole_5LNZ HSP90-indazole_5OCI HSP90-quinazoline HSP90-quinazoline_6EI5 HSP90-resorcinol; do
  echo "Submitting $d ..."
  (cd "$DIR/$d" && sbatch submit.sh)
done
