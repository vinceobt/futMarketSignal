#!/bin/bash
# Install the FUT Market Desk autonomous agents (macOS).
#
#   scripts/install_autonomous.sh [--interval-min N] [--no-wake]
#                                 [--scan-interval-min N] [--no-scan]
#
# Installs TWO LaunchAgents:
#   com.futmarket.collect  — scrape + advise every N min (default 20), waking the Mac.
#   com.futmarket.scan     — momentum discovery scan every M min (default 360 = 6h).
# The scan needs no wake of its own; it rides the collector's wakes.
# --no-wake: collector runs awake-only (skips sudo/pmset). --no-scan: skip the scanner.
set -euo pipefail

INTERVAL_MIN=20
SCAN_INTERVAL_MIN=360
WAKE=1
SCAN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --interval-min) INTERVAL_MIN="$2"; shift 2 ;;
    --scan-interval-min) SCAN_INTERVAL_MIN="$2"; shift 2 ;;
    --no-wake)      WAKE=0; shift ;;
    --no-scan)      SCAN=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
UID_NUM="$(id -u)"
LABEL="com.futmarket.collect"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
WRAPPER_TMPL="$REPO/scripts/autonomous_collect.sh"      # committed template (placeholders)
WRAPPER="$REPO/data/.autonomous_collect.rendered.sh"    # rendered, what launchd runs
INTERVAL_SEC=$(( INTERVAL_MIN * 60 ))

echo "Installing autonomous collector:"
echo "  repo:      $REPO"
echo "  interval:  every $INTERVAL_MIN min ($(awk "BEGIN{printf \"%.1f\", 60/$INTERVAL_MIN}" < /dev/null)x/hour)"
echo "  wake mode: $([ "$WAKE" = 1 ] && echo 'wake Mac from sleep' || echo 'awake-only')"
echo

if [ ! -x "$REPO/.venv/bin/futmarket" ]; then
  echo "ERROR: $REPO/.venv/bin/futmarket not found. Run 'make install' first." >&2
  exit 1
fi

# Preflight: the skip-guard (poll_minutes x skip_guard_fraction) makes a pass skip
# any player collected more recently than that. If it's >= our interval, some runs
# will skip-as-fresh, so the effective source cadence is slower than requested.
GUARD_MIN="$("$REPO/.venv/bin/python" - "$INTERVAL_MIN" <<'PY' 2>/dev/null || echo "?"
import sys
from futmarket.config import load_config
c = load_config("config.yaml")
print(f"{c.poll_minutes * c.skip_guard_fraction:.0f}")
PY
)"
if [ "$GUARD_MIN" != "?" ] && [ "$GUARD_MIN" -ge "$INTERVAL_MIN" ] 2>/dev/null; then
  echo "NOTE: your skip-guard is ${GUARD_MIN} min (poll_minutes x skip_guard_fraction),"
  echo "      which is >= the ${INTERVAL_MIN}-min interval, so some runs will skip as 'fresh'"
  echo "      and the effective source cadence will be ~${GUARD_MIN} min, not ${INTERVAL_MIN}."
  echo "      (Data granularity is unaffected — fut.gg backfills full history each hit.)"
  echo "      For a true ${INTERVAL_MIN}-min cadence, lower poll_minutes or skip_guard_fraction"
  echo "      in config.yaml so poll_minutes x skip_guard_fraction < ${INTERVAL_MIN}."
  echo
fi

# 1) Render the wrapper (fill placeholders) from the committed template.
mkdir -p "$REPO/data"
sed -e "s|__REPO__|$REPO|g" \
    -e "s|__INTERVAL_MIN__|$INTERVAL_MIN|g" \
    -e "s|__WAKE__|$WAKE|g" \
    "$WRAPPER_TMPL" > "$WRAPPER"
chmod +x "$WRAPPER"
echo "✓ rendered wrapper $WRAPPER"

# 2) Scoped passwordless sudoers for pmset (only in wake mode).
if [ "$WAKE" = 1 ]; then
  SUDOERS_DST="/etc/sudoers.d/futmarket-pmset"
  TMP_SUDO="$(mktemp)"
  sed "s|__USER__|$USER_NAME|g" "$REPO/deploy/futmarket-pmset.sudoers" > "$TMP_SUDO"
  echo "Installing scoped sudoers rule (one-time sudo). It grants ONLY:"
  echo "    $USER_NAME ALL=(root) NOPASSWD: /usr/bin/pmset schedule wake *, /usr/bin/pmset schedule cancelall"
  if sudo visudo -cf "$TMP_SUDO" >/dev/null; then
    sudo install -m 0440 -o root -g wheel "$TMP_SUDO" "$SUDOERS_DST"
    rm -f "$TMP_SUDO"
    echo "✓ installed $SUDOERS_DST"
  else
    rm -f "$TMP_SUDO"
    echo "ERROR: sudoers validation failed; not installing." >&2
    exit 1
  fi
fi

# 3) Render + install the LaunchAgent plist.
mkdir -p "$HOME/Library/LaunchAgents" "$REPO/data"
sed -e "s|__REPO__|$REPO|g" \
    -e "s|__INTERVAL_SEC__|$INTERVAL_SEC|g" \
    "$REPO/deploy/com.futmarket.collect.plist" > "$PLIST_DST"
echo "✓ installed $PLIST_DST"

# 4) (Re)load the agent.
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST_DST"
launchctl enable "gui/$UID_NUM/$LABEL" 2>/dev/null || true
echo "✓ loaded LaunchAgent $LABEL"

# 5) Arm the first wake now so the chain starts even before the next login.
if [ "$WAKE" = 1 ]; then
  NEXT="$(date -v+"${INTERVAL_MIN}"M '+%m/%d/%y %H:%M:%S')"
  sudo -n /usr/bin/pmset schedule wake "$NEXT" && echo "✓ armed first wake for $NEXT"
fi

# 6) Momentum discovery scan agent (rides the collector's wakes; no wake of its own).
if [ "$SCAN" = 1 ]; then
  SCAN_LABEL="com.futmarket.scan"
  SCAN_PLIST="$HOME/Library/LaunchAgents/$SCAN_LABEL.plist"
  chmod +x "$REPO/scripts/scan_momentum.sh"
  sed -e "s|__REPO__|$REPO|g" \
      -e "s|__SCAN_INTERVAL_SEC__|$(( SCAN_INTERVAL_MIN * 60 ))|g" \
      "$REPO/deploy/com.futmarket.scan.plist" > "$SCAN_PLIST"
  launchctl bootout "gui/$UID_NUM/$SCAN_LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$UID_NUM" "$SCAN_PLIST"
  launchctl enable "gui/$UID_NUM/$SCAN_LABEL" 2>/dev/null || true
  echo "✓ loaded LaunchAgent $SCAN_LABEL (every $SCAN_INTERVAL_MIN min)"
fi

echo
echo "Done. Collecting every $INTERVAL_MIN min$( [ "$SCAN" = 1 ] && echo ", scanning every $SCAN_INTERVAL_MIN min" )."
echo "  status:  make autonomous-status"
echo "  logs:    make autonomous-log   (scan: tail -f data/scan.log)"
echo "  stop:    make autonomous-uninstall"
