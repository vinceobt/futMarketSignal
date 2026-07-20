#!/bin/bash
# The ML loop, run by launchd. Placeholders are filled in by install_ml.sh.
#
# Each cycle:
#   1. arm the next pmset wake, so this keeps running while the Mac sleeps
#   2. collect-bulk   -- whole-market prices in ~4s. Frequent snapshots are what
#                        make the scorecard scorable and build the hourly dataset
#                        the intraday dump-window features need.
#   3. picks --save   -- record today's recommendations so they can be graded
#   4. scorecard      -- resolve any picks the market has now answered
#
# Steps are independent: a failure in one must not stop the others, because
# missing a price snapshot is far worse than missing a scorecard refresh.
set -uo pipefail

REPO="__REPO__"
WAKE="__WAKE__"                    # "1" = arm pmset wakes, "0" = awake-only
INTERVAL_MIN="__INTERVAL_MIN__"
PICKS_LIMIT="__PICKS_LIMIT__"
MIN_SALES="__MIN_SALES__"
LOG="$REPO/data/ml_daily.log"
FUT="$REPO/.venv/bin/futmarket"

stamp() { date "+%Y-%m-%dT%H:%M:%S%z"; }
cd "$REPO" || exit 1

# --- keep the machine coming back ------------------------------------------
if [ "$WAKE" = "1" ]; then
  NEXT="$(date -v +"${INTERVAL_MIN}"M "+%m/%d/%y %H:%M:%S" 2>/dev/null)"
  if [ -n "$NEXT" ]; then
    if sudo -n /usr/bin/pmset schedule wake "$NEXT" 2>>"$LOG"; then
      echo "[$(stamp)] armed next wake for $NEXT" >>"$LOG"
    else
      echo "[$(stamp)] WARN could not arm wake (sudoers/pmset?) — awake-only" >>"$LOG"
    fi
  fi
fi

run_step() {
  local label="$1"; shift
  if "$@" >>"$LOG" 2>&1; then
    echo "[$(stamp)] ok   $label" >>"$LOG"
  else
    echo "[$(stamp)] FAIL $label (rc=$?)" >>"$LOG"
  fi
}

echo "[$(stamp)] --- ml cycle start ---" >>"$LOG"
# Prices first: everything downstream is worthless without fresh snapshots.
run_step "collect-bulk" "$FUT" collect-bulk
run_step "picks"        "$FUT" picks --limit "$PICKS_LIMIT" \
                                     --min-sales-per-hour "$MIN_SALES" --save
run_step "scorecard"    "$FUT" scorecard
echo "[$(stamp)] --- ml cycle done ---" >>"$LOG"
