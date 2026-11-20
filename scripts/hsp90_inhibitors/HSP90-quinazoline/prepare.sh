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
  --receptor-targets 36:CB:ASN,39:CB:ASP,87:CB:ASP,92:CB:LEU,123:CB:PHE,124:CB:TYR \
  --ligand-atoms N4x \
  --pair-cutoff 8.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
