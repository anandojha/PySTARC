#!/usr/bin/env bash
# Criterion: 3 of 6 shortest bound polar contacts (N, O, S; one per receptor residue) at 7.0 A.
# Automatic selection across the solvent-accessible protein-protein interface.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --receptor-pqr "$DIR/t.pqr" \
  --ligand-pqr "$DIR/m.pqr" \
  --receptor-out "$OUT/receptor.pqr" \
  --ligand-out "$OUT/ligand.pqr")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pqr" \
  --ligand "$OUT/ligand.pqr" \
  --atom-filter polar \
  --search-cutoff 6.0 \
  --n-pairs 6 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
