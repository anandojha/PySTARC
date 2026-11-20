#!/usr/bin/env bash
# Criterion tier: Tier 3 (pocket-rim), bulky inhibitor in deep enclosed pocket (enclosure >= 0.91, N_lig_heavy >= 24).
# Rule: 3-of-6 conserved mouth-ring CB targets (64, 67, 92, 131, 200, 201) paired with sulfonamide anchor N at 7.0 A.
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
  --smiles "[NH-]S(=O)(=O)C1=C(NC2CCCCCCC2)C(F)=C(SCCO)C(F)=C1F" \
  --ligand-resname V12 \
  --pocket-ligand AZM \
  --pocket-ligand-resid 263 \
  --formula "C16 H22 F3 N2 O3 S2" \
  --receptor-out "$OUT/receptor.pdb" \
  --ligand-out "$OUT/ligand.pdb")"

CRIT_XML="$(python "$SCRIPTS/find_reaction_criteria.py" \
  --receptor "$OUT/receptor.pdb" \
  --ligand "$OUT/ligand.pdb" \
  --receptor-targets 64:CB:HIS,67:CB:ASN,92:CB:GLN,131:CB:PHE,200:CB:VAL,201:CB:PRO \
  --ligand-atoms N1 \
  --pair-cutoff 7.0)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
