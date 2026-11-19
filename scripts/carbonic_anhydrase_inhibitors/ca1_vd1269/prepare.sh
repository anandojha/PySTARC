#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/2NMX.pdb" \
  --chain A \
  --keep-hetatm ZN \
  --metal-his \
  --smiles "COC1=CC=C(CNC2=C(C(F)=C(F)C(F)=C2F)S([NH-])(=O)=O)C=C1" \
  --ligand-resname V17 \
  --pocket-ligand M25 \
  --pocket-ligand-resid 311 \
  --formula "C14 H11 F4 N2 O3 S" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 199:OG1:THR,94:NE2:HID,96:NE2:HID,119:ND1:HIE \
  --ligand-atoms N2 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
