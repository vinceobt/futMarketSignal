#!/bin/bash
# Remove the FUT Market Desk autonomous collector. Leaves the DB and logs intact.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UID_NUM="$(id -u)"
LABEL="com.futmarket.collect"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "Removing autonomous collector…"

# 1) Unload + remove both LaunchAgents (collector + scanner).
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null && echo "✓ unloaded $LABEL" || echo "· collector was not loaded"
rm -f "$PLIST_DST" && echo "✓ removed $PLIST_DST"
rm -f "$REPO/data/.autonomous_collect.rendered.sh"

SCAN_LABEL="com.futmarket.scan"
SCAN_PLIST="$HOME/Library/LaunchAgents/$SCAN_LABEL.plist"
launchctl bootout "gui/$UID_NUM/$SCAN_LABEL" 2>/dev/null && echo "✓ unloaded $SCAN_LABEL" || echo "· scanner was not loaded"
rm -f "$SCAN_PLIST"

# 2) Cancel our scheduled wakes (best-effort; needs the scoped sudoers, still present).
if sudo -n /usr/bin/pmset schedule cancelall 2>/dev/null; then
  echo "✓ cleared scheduled power wakes"
else
  echo "· could not run pmset (sudoers already removed?) — check: pmset -g sched"
fi

# 3) Remove the sudoers rule (interactive sudo; it's the last thing we need it for).
if [ -f /etc/sudoers.d/futmarket-pmset ]; then
  sudo rm -f /etc/sudoers.d/futmarket-pmset && echo "✓ removed /etc/sudoers.d/futmarket-pmset"
fi

echo "Done. Autonomous collection is off."
