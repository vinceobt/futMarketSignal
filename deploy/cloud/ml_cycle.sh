#!/usr/bin/env bash
# One ML cycle on the cloud box: fresh prices -> recorded picks -> scorecard ->
# insights. The Mac version arms a pmset wake because a laptop sleeps; a server
# never sleeps, so this is just the four steps, run by a systemd timer.
set -uo pipefail
REPO=/opt/futmarket
FUT="$REPO/.venv/bin/futmarket"
LOG="$REPO/data/ml_daily.log"
cd "$REPO" || exit 1
stamp() { date "+%Y-%m-%dT%H:%M:%S%z"; }
run_step() {
  local label="$1"; shift
  if "$@" >>"$LOG" 2>&1; then
    echo "[$(stamp)] ok   $label" >>"$LOG"
  else
    echo "[$(stamp)] FAIL $label (rc=$?)" >>"$LOG"
  fi
}
echo "[$(stamp)] --- ml cycle start ---" >>"$LOG"
run_step "collect-bulk" "$FUT" collect-bulk
run_step "picks"        "$FUT" picks --limit 12 --min-sales-per-hour 5 --save
run_step "scorecard"    "$FUT" scorecard
run_step "insights"     "$FUT" insights
echo "[$(stamp)] --- ml cycle done ---" >>"$LOG"
