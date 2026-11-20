#!/usr/bin/env bash
# Criterion tier: Tier 3 (pocket-rim), bulky kinase inhibitor in deep ATP pocket (enclosure >= 0.87, N_lig_heavy >= 26).
# Rule: 3-of-6 conserved ATP-mouth CB targets (529, 531, 606, 611, 651, 663) paired with GLY605:N hinge acceptor at 9.0 A (smallest integer clearing rigid steric exclusion across all 8 TTK complexes).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/3GFW.pdb" \
  --chain A \
  --resid-min 515 \
  --ligand S22 \
  --formula "C28 H33 N5 O4 S" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 529:CB:LYS,531:CB:ILE,606:CB:ASN,611:CB:SER,651:CB:ALA,663:CB:ILE \
  --ligand-atoms NAF \
  --pair-cutoff 9.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
