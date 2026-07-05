"""Assemble the self-contained Artifact preview: same CSS + JS as the live app,
with a data snapshot inlined (no external requests, per Artifact CSP). Writes
the body-content HTML to the given path."""
import json
import sys
from pathlib import Path

from futmarket import dashboard, db
from futmarket.config import load_config

WEB = Path("src/futmarket/web")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dashboard_preview.html")

cfg = load_config("config.yaml")
conn = db.connect(cfg.database_path)
payload = dashboard.build_payload(cfg, conn, "manual")

css = (WEB / "app.css").read_text()
js = (WEB / "app.js").read_text()
data = json.dumps(payload, separators=(",", ":"))

html = f"""<title>FUT Market Desk</title>
<div id="app"></div>
<style>
{css}
</style>
<script>
window.__FUT_DATA__ = {data};
{js}
window.FUT.render(window.__FUT_DATA__);
</script>
"""
out.write_text(html)
print(f"wrote {out} ({len(html):,} bytes, {len(payload['players'])} players)")
