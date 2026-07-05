#!/bin/bash
# Periodic momentum discovery scan — find + track new range-bound cards.
# Self-locating: it derives the repo path from its own location, so it works
# unchanged on any Mac (just clone the repo + `make install`, then schedule it).
# Schedule it every few hours (discovery changes on the scale of days, not minutes).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${FUTMARKET_CONFIG:-$REPO/config.yaml}"
PY="$REPO/.venv/bin/futmarket"
LOG="$REPO/data/scan.log"
mkdir -p "$REPO/data"
stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

if [ ! -x "$PY" ]; then
  echo "[$(stamp)] ERROR futmarket not found at $PY — run 'make install' first" >>"$LOG"
  exit 1
fi

echo "[$(stamp)] scan start" >>"$LOG"
"$PY" --config "$CONFIG" scan-momentum >>"$LOG" 2>&1
echo "[$(stamp)] scan done (exit $?)" >>"$LOG"
