#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --split-complex "$DIR/complex.pdb" \
  --ligand UNK \
  --ligand-charge 0 \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 35:ND2:ASN,38:OD1:ASP,42:NZ:LYS,82:SD:MET,86:OD1:ASP,91:CG:LEU \
  --atom-filter all \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
