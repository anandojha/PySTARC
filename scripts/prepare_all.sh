#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${1:-}"
find "$SCRIPTS_DIR" -mindepth 2 -maxdepth 3 -name prepare.sh | sort | while read -r sh; do
  rel="${sh#$SCRIPTS_DIR/}"
  rel="${rel%/prepare.sh}"
  echo "=== $rel ===" >&2
  if [ -n "$OUT_ROOT" ]; then
    bash "$sh" "$OUT_ROOT/$rel"
  else
    bash "$sh"
  fi
done
