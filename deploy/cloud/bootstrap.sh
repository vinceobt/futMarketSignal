#!/usr/bin/env bash
# Turn a fresh Ubuntu 24.04 box into the always-on futmarket server.
# Assumes the code + data/market.db are already at /opt/futmarket (rsynced up
# before running this). Idempotent: safe to re-run after a code change.
set -euo pipefail
REPO=/opt/futmarket
cd "$REPO"

echo "== system packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# Python + build tools, and the shared libraries a headless browser needs.
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-dev build-essential git curl ca-certificates \
  ufw fonts-liberation libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
  libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
  libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 libatspi2.0-0 \
  libgtk-3-0 libx11-xcb1 libdbus-glib-1-2

echo "== uv (fast Python package manager) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== virtualenv + app =="
[ -d .venv ] || uv venv .venv
uv pip install --python .venv/bin/python -e ".[web,ml]"
uv pip install --python .venv/bin/python patchright "camoufox[geoip]"

echo "== browser binaries =="
# camoufox = stealth Firefox used to get past fut.gg's bot check.
.venv/bin/python -m camoufox fetch || echo "WARN camoufox fetch failed — retry live"
.venv/bin/patchright install chromium || echo "WARN patchright chromium failed — retry live"

echo "== dashboard access key =="
ENV=data/.dashboard_env
if [ ! -f "$ENV" ]; then
  KEY=$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)
  echo "FUTMARKET_KEY=$KEY" > "$ENV"
  chmod 600 "$ENV"
fi

echo "== systemd services =="
chmod +x deploy/cloud/ml_cycle.sh
cp deploy/cloud/futmarket-dashboard.service /etc/systemd/system/
cp deploy/cloud/futmarket-ml.service /etc/systemd/system/
cp deploy/cloud/futmarket-ml.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now futmarket-dashboard.service
systemctl enable --now futmarket-ml.timer

echo "== firewall (SSH + dashboard only) =="
ufw allow 22/tcp   >/dev/null 2>&1 || true
ufw allow 8899/tcp >/dev/null 2>&1 || true
ufw --force enable >/dev/null 2>&1 || true

IP=$(curl -s https://api.ipify.org || echo "<server-ip>")
KEY=$(grep -oE 'FUTMARKET_KEY=.*' "$ENV" | cut -d= -f2)
echo
echo "=================================================================="
echo "  Dashboard is live (always on):"
echo "     http://$IP:8899/?key=$KEY"
echo "  ML cycle runs every 2 hours. Logs: $REPO/data/ml_daily.log"
echo "=================================================================="
