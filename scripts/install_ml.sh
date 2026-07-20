#!/bin/bash
# Install the ML loop as a macOS LaunchAgent.
#
#   scripts/install_ml.sh [--interval-min N] [--no-wake]
#                         [--picks N] [--min-sales-per-hour F]
#
# Runs collect-bulk + picks --save + scorecard every N minutes (default 120).
# Frequent cycles are deliberate: prices are what the scorecard grades against,
# and intraday snapshots are what the dump-window features need. The picks
# themselves only change meaningfully once a day, but re-recording is harmless --
# pick_log is unique per (card, minute).
#
# Sleep: with wake mode on, the wrapper arms a pmset wake each cycle, so the Mac
# wakes itself, runs, and sleeps again. A powered-OFF Mac runs nothing.
set -euo pipefail

INTERVAL_MIN=120
WAKE=1
PICKS=15
MIN_SALES=3
while [ $# -gt 0 ]; do
  case "$1" in
    --interval-min)        INTERVAL_MIN="$2"; shift 2 ;;
    --picks)               PICKS="$2"; shift 2 ;;
    --min-sales-per-hour)  MIN_SALES="$2"; shift 2 ;;
    --no-wake)             WAKE=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.futmarket.ml"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
WRAPPER_TMPL="$REPO/scripts/ml_daily.sh"
WRAPPER="$REPO/data/.ml_daily.rendered.sh"
INTERVAL_SEC=$(( INTERVAL_MIN * 60 ))

if [ ! -x "$REPO/.venv/bin/futmarket" ]; then
  echo "ERROR: $REPO/.venv/bin/futmarket not found. Run 'make install' first." >&2
  exit 1
fi

echo "Installing the ML loop:"
echo "  repo:      $REPO"
echo "  every:     $INTERVAL_MIN min"
echo "  picks:     top $PICKS, min $MIN_SALES sales/hour"
echo "  wake mode: $([ "$WAKE" = 1 ] && echo 'wake Mac from sleep' || echo 'awake-only')"
echo

mkdir -p "$REPO/data" "$HOME/Library/LaunchAgents"

sed -e "s|__REPO__|$REPO|g" \
    -e "s|__WAKE__|$WAKE|g" \
    -e "s|__INTERVAL_MIN__|$INTERVAL_MIN|g" \
    -e "s|__PICKS_LIMIT__|$PICKS|g" \
    -e "s|__MIN_SALES__|$MIN_SALES|g" \
    "$WRAPPER_TMPL" > "$WRAPPER"
chmod +x "$WRAPPER"

sed -e "s|__REPO__|$REPO|g" \
    -e "s|__INTERVAL_SEC__|$INTERVAL_SEC|g" \
    "$REPO/deploy/$LABEL.plist" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed. It will run now and every $INTERVAL_MIN minutes."
echo
echo "  watch it:   tail -f $REPO/data/ml_daily.log"
echo "  results:    $REPO/.venv/bin/futmarket scorecard --list 10"
echo "  stop it:    launchctl unload $PLIST_DST"
if [ "$WAKE" = 1 ]; then
  echo
  echo "Wake-from-sleep needs passwordless pmset. If the log warns it could not"
  echo "arm a wake, install the sudoers rule:"
  echo "  sudo cp $REPO/deploy/futmarket-pmset.sudoers /etc/sudoers.d/futmarket-pmset"
fi
