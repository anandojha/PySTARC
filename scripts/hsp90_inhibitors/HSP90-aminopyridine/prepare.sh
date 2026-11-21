#!/usr/bin/env bash
# Criterion tier: Tier 3 (pocket-rim), multi-scaffold inhibitor in deep enclosed pocket (enclosure >= 0.93, N_lig_heavy >= 21).
# Rule: 3-of-6 conserved outer-mouth sidechain tips (ASN51:ND2, ASP54:OD1, LYS58:NZ, MET98:SD, ASP102:OD1, LEU107:CG; sequence-aligned) paired automatically (--atom-filter all) with nearest bound ligand heavy atoms at 7.5 A.
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
  --receptor-targets 51:ND2:ASN,54:OD1:ASP,58:NZ:LYS,98:SD:MET,102:OD1:ASP,107:CG:LEU \
  --atom-filter all \
  --pair-cutoff 7.5)"

printf '%s\n\n%s\n' "$STRUCT_XML" "$CRIT_XML" | tee "$OUT/config_snippet.xml"
