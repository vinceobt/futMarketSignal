#!/bin/bash
# Autonomous collection wrapper — run by the futmarket LaunchAgent every INTERVAL_MIN.
#
# It (1) arms the NEXT scheduled wake so the Mac wakes itself for the following
# run even while asleep, then (2) runs one polite collect-once pass. The next
# wake is armed *before* scraping so a failed/slow scrape never breaks the chain.
#
# The __PLACEHOLDER__ values are filled in by scripts/install_autonomous.sh.
set -uo pipefail

REPO="__REPO__"
INTERVAL_MIN="__INTERVAL_MIN__"
WAKE="__WAKE__"                 # "1" = arm pmset wakes (wake-through-sleep), "0" = awake-only
CONFIG="${FUTMARKET_CONFIG:-$REPO/config.yaml}"
PY="$REPO/.venv/bin/futmarket"
LOG="$REPO/data/autonomous.log"

mkdir -p "$REPO/data"
stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# 1) Arm the next wake (best-effort; never abort the run if it fails).
if [ "$WAKE" = "1" ]; then
  NEXT="$(date -v+"${INTERVAL_MIN}"M '+%m/%d/%y %H:%M:%S')"
  if sudo -n /usr/bin/pmset schedule wake "$NEXT" 2>>"$LOG"; then
    echo "[$(stamp)] armed next wake for $NEXT" >>"$LOG"
  else
    echo "[$(stamp)] WARN could not arm wake (sudoers/pmset?) — awake-only this cycle" >>"$LOG"
  fi
fi

# 2) One collection pass, then the rebound advisor (analyze + position mgmt + alerts).
echo "[$(stamp)] collect-once start" >>"$LOG"
if [ ! -x "$PY" ]; then
  echo "[$(stamp)] ERROR futmarket not found at $PY — is the venv installed?" >>"$LOG"
  exit 1
fi
"$PY" --config "$CONFIG" collect-once >>"$LOG" 2>&1
echo "[$(stamp)] collect-once done (exit $?)" >>"$LOG"

echo "[$(stamp)] advise start" >>"$LOG"
"$PY" --config "$CONFIG" advise >>"$LOG" 2>&1
echo "[$(stamp)] advise done (exit $?)" >>"$LOG"
