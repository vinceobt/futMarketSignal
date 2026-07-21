"""A short Discord summary after each run, so the owner can keep account.

One message per cycle: what the robot just recommended, and how its past calls
are doing. Plain language, no stats jargon — the same voice as the picks output.
Sending publishes externally, so it only fires when a webhook is configured.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .. import db
from ..alerts import _card_tag, _coins
from . import scorecard

logger = logging.getLogger(__name__)


def build_run_summary(conn, *, title: str = "fc26", top: int = 6) -> str:
    """The message body: the latest cycle's top buys + the running track record."""
    latest = conn.execute(
        "SELECT MAX(picked_at) FROM pick_log WHERE title=?", (title,)).fetchone()[0]

    picks = []
    if latest:
        picks = conn.execute(
            """SELECT p.buy_low, p.buy_high, p.target_price, p.confidence,
                      p.sales_per_hour, m.name, m.rating, m.version, m.url
               FROM pick_log p LEFT JOIN card_meta m ON m.player_id = p.player_id
               WHERE p.title = ? AND p.picked_at = ?
               ORDER BY p.confidence DESC LIMIT ?""",
            (title, latest, top)).fetchall()

    when = datetime.now().astimezone().strftime("%a %d %b %H:%M")
    lines = [f"🤖 **FUT robot ran** · {when}"]

    if not picks:
        lines.append("No new buys cleared the bar this run.")
    else:
        lines.append(f"Fresh prices in. Top {len(picks)} buy(s):")
        for p in picks:
            name = p["name"] or "?"
            tag = _card_tag(p["rating"], p["version"])
            conf = f"{p['confidence']:.0%}" if p["confidence"] is not None else "?"
            sph = f" · {p['sales_per_hour']:.0f}/hr" if p["sales_per_hour"] else ""
            lines.append(
                f"🟢 **{name}**{tag} — buy {_coins(p['buy_low'])}–{_coins(p['buy_high'])}"
                f" → sell {_coins(p['target_price'])} · {conf}{sph}")
            if p["url"]:
                lines.append(f"<{p['url']}>")

    s = scorecard.summary(conn, title=title)
    if s.get("closed"):
        lines.append(
            f"📊 Record: {s['hit_target']}W-{s['hit_stop']}L · "
            f"{s['win_rate']:.0%} win (of {s['closed']} graded, {s['open']} still open)")
    else:
        lines.append(f"📊 Record: {s.get('total', 0)} picks logged, none graded yet.")

    return "\n".join(lines)


def send_run_summary(conn, webhook_url: str, *, title: str = "fc26") -> bool:
    """Compose and post the summary. Returns False (and logs) if it can't send,
    so a notification failure never brings down the collection cycle."""
    from ..alerts import DiscordAlerter
    text = build_run_summary(conn, title=title)
    try:
        DiscordAlerter(webhook_url).send(text)
        return True
    except Exception as e:  # noqa: BLE001 - notifying is best-effort
        logger.warning("discord notify failed: %s", e)
        return False
