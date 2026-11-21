#!/usr/bin/env bash
# Criterion: 3 of 6 exposed sidechain contacts shared across all HSP90 complexes at 7.5 A.
# Receptor atoms ASN51:ND2, ASP54:OD1, LYS58:NZ, MET98:SD, ASP102:OD1, LEU107:CG paired with nearest bound ligand heavy atoms.
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
  --pair-cutoff 7.5)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
