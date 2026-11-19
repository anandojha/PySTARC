#!/usr/bin/env bash
set -euo pipefail

FAM_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${1:-}"
for sh in "$FAM_DIR"/*/prepare.sh; do
  name="$(basename "$(dirname "$sh")")"
  echo "=== $name ===" >&2
  if [ -n "$OUT_ROOT" ]; then
    bash "$sh" "$OUT_ROOT/$name"
  else
    bash "$sh"
  fi
done
