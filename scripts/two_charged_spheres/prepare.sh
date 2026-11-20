#!/usr/bin/env bash
# Criterion tier: Tier 1 (centroid), analytical charged-sphere pair.
# Rule: centroid capture at 2.5 A (sum of sphere radii plus 0.5 A contact shell).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --sphere-pair 1.0:1.0,-1.0:1.0 \
  --receptor-out "$OUT/receptor.pqr" \
  --ligand-out "$OUT/ligand.pqr")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pqr" \
  --ligand "$OUT/ligand.pqr" \
  --mode centroid \
  --pair-cutoff 2.5)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
