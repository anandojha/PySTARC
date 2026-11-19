#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(cd "$DIR/../.." && pwd)"
OUT="${1:-$DIR/prepared}"
mkdir -p "$OUT"

STRUCT_XML="$(python "$SCRIPTS/prepare_pdb.py" \
  --pdb "$DIR/3HS4.pdb" \
  --chain A \
  --keep-hetatm ZN \
  --metal-his \
  --smiles "[NH-]S(=O)(=O)C1=C(F)C(NC2CCCCCCC2)=C(C(F)=C1F)S(=O)(=O)CCO" \
  --ligand-resname V13 \
  --pocket-ligand AZM \
  --pocket-ligand-resid 701 \
  --formula "C16 H22 F3 N2 O5 S2" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 64:CB:HIS,67:CB:ASN,92:CB:GLN,131:CB:PHE,200:CB:THR,201:CB:PRO \
  --ligand-atoms N1 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
