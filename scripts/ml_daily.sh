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
# New promo cards: refresh the registry ~once a day (full crawl is heavy), then
# rescore liquidity so new cards become tradeable/consultable.
# NEW CARDS, EVERY CYCLE. This is not optional housekeeping: the release trade
# buys a promo card on day 4-6 of its crash, so a card that isn't in the registry
# within a day or two can never be traded in its best window. It has already gone
# wrong once -- cards released 19-24 Jul got their first price snapshot on the
# 25th, and every one of those release windows was missed. The first pages of
# fut.gg's list are the newest cards, so this is 4 requests, not a full crawl.
run_step "new-cards" "$FUT" build-registry --max-pages 4
# The full crawl is heavy, so it stays daily; it also rescores liquidity so new
# cards become tradeable and consultable.
REG_MARK="$REPO/data/.last_registry"
if [ ! -f "$REG_MARK" ] || [ -n "$(find "$REG_MARK" -mmin +1200 2>/dev/null)" ]; then
  run_step "build-registry"  "$FUT" build-registry
  run_step "score-liquidity" "$FUT" score-liquidity
  touch "$REG_MARK"
fi
# Prices first: everything downstream is worthless without fresh snapshots.
run_step "collect-bulk" "$FUT" collect-bulk
# Real completed sales for the most tradeable cards. Two jobs at once: it keeps
# the buy bands honest, and it banks the 'futgg_sold' price series -- actual
# transactions rather than the cheapest listing, whose day-to-day return std is
# 193% against 41% for a robust daily median. Each fetch returns ~100 sales
# spanning most of a day, so a slice per cycle accumulates real history fast.
# Capped so a cycle stays well under the 2h interval.
# Stalest cards first, so each cycle advances coverage instead of re-fetching the
# same head of the list. 250 at 2s is ~8 minutes of the 2h window and ~3k cards a
# day: the whole tradeable universe inside three days, then continuous refresh.
# Sized deliberately below what trips fut.gg's limiter -- a continuous 8k sweep
# stalled at ~600 cards behind escalating backoff and banked nothing further.
run_step "sale-stats"   "$FUT" sale-stats --tiers AB --limit 250 --delay 2.0
run_step "picks"        "$FUT" picks --limit "$PICKS_LIMIT" \
                                     --min-sales-per-hour "$MIN_SALES" --save
# Sell side: ping to sell any held pick that reached its target before grading it.
run_step "sell-alerts"  "$FUT" sell-alerts
run_step "scorecard"    "$FUT" scorecard
# Rhythms sweep millions of rows, so the dashboard reads them from cache.
run_step "insights"     "$FUT" insights
# All-tier buy tips learned from the Discord pros, cached for the /ml panel (~1 min).
run_step "trader-tips"  "$FUT" trader-tips
# A short Discord summary so the owner can keep account of every run.
run_step "notify"       "$FUT" notify
echo "[$(stamp)] --- ml cycle done ---" >>"$LOG"
