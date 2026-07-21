"""A short Discord summary after each run, so the owner can keep account.

One message per cycle: what the robot just recommended, and how its past calls
are doing. Plain language, no stats jargon — the same voice as the picks output.
Sending publishes externally, so it only fires when a webhook is configured.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import db
from ..alerts import _card_tag, _coins
from . import scorecard

logger = logging.getLogger(__name__)

# Alert when the live price is within this fraction of the target — a touch early,
# so you have time to actually list and sell into the bounce.
NEAR_TARGET = 0.98


def sell_alerts(conn, webhook_url: str, *, source: str = "futgg", title: str = "fc26",
                tax_rate: float = 0.05, sell_slippage_pct: float = 0.0) -> int:
    """Ping Discord to SELL a held pick that has reached its target, or CUT one that
    hit its stop. Each pick alerts once (recorded via ``alerted_at``)."""
    from ..alerts import DiscordAlerter
    from ..ml.picks import STRATEGY_VERSION
    alerter = DiscordAlerter(webhook_url)
    now = datetime.now(timezone.utc)
    sale_net = (1 - tax_rate) * (1 - sell_slippage_pct / 100.0)
    sent = 0
    for pick in db.open_picks(conn, title=title):
        # only alert on the current strategy's real positions, not old paper picks
        if pick["strategy"] != STRATEGY_VERSION or pick["alerted_at"]:
            continue
        price = db.latest_price(conn, pick["player_id"], source)
        if price is None:
            continue
        target, stop = pick["target_price"], pick["stop_price"]
        if price >= target * NEAR_TARGET:
            kind, emoji, why = "SELL", "🟢", "hit its target"
        elif price <= stop:
            kind, emoji, why = "CUT", "🔴", "hit its stop"
        else:
            continue
        meta = db.card_meta_get(conn, pick["player_id"])
        name = meta["name"] if meta else pick["player_id"]
        tag = _card_tag(meta["rating"] if meta else None,
                        meta["version"] if meta else None)
        net = (price * sale_net / pick["entry_price"] - 1) * 100
        url = f"\n<{meta['url']}>" if meta and meta["url"] else ""
        msg = (f"{emoji} **{kind} {name}**{tag} now ~{_coins(price)} · "
               f"{net:+.0f}% net — {why}{url}")
        try:
            alerter.send(msg)
            db.mark_pick_alerted(conn, pick["id"], now)
            sent += 1
        except Exception as e:  # noqa: BLE001 - a failed send must not stop the rest
            logger.warning("sell alert failed for %s: %s", pick["player_id"], e)
    logger.info("sell alerts: %d sent", sent)
    return sent


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
    if s.get("graded"):
        lines.append(
            f"📊 Record: {s['hit_target']}W-{s['hit_stop']}L · {s['win_rate']:.0%} win · "
            f"{s['return_on_capital_pct']:+.1f}% on capital "
            f"({s['coins_pnl']:+,} coins over {s['graded']} trades, {s['open']} open)")
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
