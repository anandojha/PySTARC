#!/usr/bin/env bash
# Criterion: 3 of 6 exposed CB contacts in the deep ATP binding site at 8.0 A.
# Receptor residues 35, 38, 107, 109, 168, 173 paired with hinge acceptor NB1.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/1A9U.pdb" \
  --chain A \
  --resid-min 4 \
  --ligand SB2 \
  --formula "C21 H16 F N3 O S" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 35:CB:TYR,38:CB:VAL,107:CB:HIS,109:CB:MET,168:CB:ASP,173:CB:ARG \
  --ligand-atoms NB1 \
  --pair-cutoff 8.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
