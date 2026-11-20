#!/usr/bin/env bash
# Criterion tier: Tier 2 (open-site polar), compact fragment (N_lig_heavy = 13 <= 15).
# Rule: automatic 3-of-6 shortest bound polar contacts (N, O, S; one per receptor residue) at 7.0 A.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/3CZV.pdb" \
  --chain A \
  --keep-hetatm ZN \
  --metal-his \
  --ligand AZM \
  --ligand-resid 263 \
  --deprotonate N1 \
  --formula "C4 H5 N4 O3 S2" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --atom-filter polar \
  --n-pairs 6 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
