#!/usr/bin/env bash
# Criterion tier: Tier 1 (centroid), symmetric macrocyclic host-guest cavity (N_rec_heavy < 120).
# Rule: uniform centroid capture at 8.0 A across all seven beta-cyclodextrin guests.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --prmtop "$DIR/complex.parm7" \
  --pdb "$DIR/complex.pdb" \
  --ligand APN \
  --receptor-out "$OUT/receptor.pqr" \
  --ligand-out "$OUT/ligand.pqr")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pqr" \
  --ligand "$OUT/ligand.pqr" \
  --mode centroid \
  --pair-cutoff 8.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
