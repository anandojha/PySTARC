#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for d in BCD_1-butanol BCD_1-naphthylethanol BCD_1-propanol BCD_2-naphthylethanol BCD_aspirin BCD_methyl_butyrate BCD_tertbutanol; do
  echo "Submitting $d ..."
  (cd "$DIR/$d" && sbatch submit.sh)
done
