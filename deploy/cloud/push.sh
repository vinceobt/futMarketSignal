#!/usr/bin/env bash
# Push code + database from this Mac up to the cloud box.
# Usage:  deploy/cloud/push.sh <server-ip>
# Re-run any time after changing code; it's a fast incremental sync.
set -euo pipefail
IP="${1:?usage: push.sh <server-ip>}"
KEY=~/.ssh/futmarket_deploy
RSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

$RSH root@"$IP" "mkdir -p /opt/futmarket/data"

echo "== syncing code =="
rsync -az --delete -e "$RSH" \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude 'data' \
  "$REPO/" root@"$IP":/opt/futmarket/

echo "== syncing database (large, incremental) =="
rsync -az --partial --info=progress2 -e "$RSH" \
  "$REPO/data/market.db" root@"$IP":/opt/futmarket/data/market.db

echo "done."
