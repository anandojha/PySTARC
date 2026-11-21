#!/usr/bin/env bash
# Criterion: 3 of 6 exposed CB contacts shared across all carbonic anhydrase complexes at 7.0 A.
# Receptor residues 64, 67, 92, 131, 200, 201 paired with sulfonamide N1.
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
  --receptor-targets 64:CB:HIS,67:CB:ASN,92:CB:GLN,131:CB:PHE,200:CB:VAL,201:CB:PRO \
  --ligand-atoms N1 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
