#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/2YKJ.pdb" \
  --chain A \
  --altloc B \
  --ligand YKJ \
  --formula "C25 H18 N6 O" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 51:CB:ASN,54:CB:ASP,102:CB:ASP,107:CB:LEU,138:CB:PHE,139:CB:TYR \
  --ligand-atoms N6 \
  --pair-cutoff 8.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
