"""The track record: what actually happened to every recommendation.

Until this existed we could only ever say whether a pick *looked* reasonable on
the day. That is not evidence. This closes the loop: every pick is recorded with
the price you'd have paid and the barriers it was judged against, then scored
once the market has had its say.

Scoring walks the card's prices after the pick and applies the same triple
barrier the model was trained on — target first, stop first, or neither by the
horizon — so the scorecard and the training labels can never quietly disagree.
Returns are net of EA's sell tax, because gross returns aren't money.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .. import db

logger = logging.getLogger(__name__)

TARGET, STOP, EXPIRED = "target", "stop", "expired"


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def score_pick(pick, prices: list[tuple[datetime, int]], *, tax_rate: float = 0.05,
               now: datetime | None = None) -> tuple[str, int, float] | None:
    """(status, exit_price, realized_pct) for one pick, or None if still open.

    Whichever barrier is touched first wins; if neither is touched by the
    horizon, the position is marked out at the last observed price.
    """
    now = now or datetime.now(timezone.utc)
    picked = _parse(pick["picked_at"])
    deadline = picked + timedelta(days=int(pick["horizon_days"]))
    entry = int(pick["entry_price"])
    if entry <= 0:
        return None

    after = [(t, p) for t, p in prices if t > picked and p > 0]
    for t, price in after:
        if price >= pick["target_price"]:
            return TARGET, price, (price * (1 - tax_rate) / entry - 1) * 100
        if price <= pick["stop_price"]:
            return STOP, price, (price * (1 - tax_rate) / entry - 1) * 100

    if now < deadline:
        return None                      # still running, no verdict yet
    if not after:
        return None                      # no prices seen since the pick
    last_price = after[-1][1]
    return EXPIRED, last_price, (last_price * (1 - tax_rate) / entry - 1) * 100


def score_open_picks(conn, *, title: str = "fc26", source: str = "futgg",
                     tax_rate: float = 0.05, now: datetime | None = None) -> dict:
    """Resolve every pick the market has answered. Returns counts by outcome."""
    now = now or datetime.now(timezone.utc)
    res = {"checked": 0, TARGET: 0, STOP: 0, EXPIRED: 0, "still_open": 0}
    for pick in db.open_picks(conn, title=title):
        res["checked"] += 1
        rows = db.snapshots(conn, pick["player_id"], source)
        prices = [(_parse(r["timestamp"]), int(r["price"])) for r in rows]
        outcome = score_pick(pick, prices, tax_rate=tax_rate, now=now)
        if outcome is None:
            res["still_open"] += 1
            continue
        status, exit_price, realized = outcome
        db.close_pick(conn, pick["id"], status=status, exit_price=exit_price,
                      realized_pct=realized, at=now)
        res[status] += 1
    conn.commit()
    logger.info("scorecard: %s", res)
    return res


def summary(conn, *, title: str = "fc26") -> dict:
    """The honest headline: how the recommendations have actually done."""
    rows = conn.execute(
        "SELECT status, realized_pct, confidence FROM pick_log WHERE title=?",
        (title,)).fetchall()
    closed = [r for r in rows if r["status"] != "open"]
    wins = [r for r in closed if r["status"] == TARGET]
    if not closed:
        return {"total": len(rows), "closed": 0, "open": len(rows) - len(closed)}
    realized = [r["realized_pct"] for r in closed if r["realized_pct"] is not None]
    return {
        "total": len(rows),
        "open": len(rows) - len(closed),
        "closed": len(closed),
        "hit_target": len(wins),
        "hit_stop": sum(1 for r in closed if r["status"] == STOP),
        "expired": sum(1 for r in closed if r["status"] == EXPIRED),
        "win_rate": round(len(wins) / len(closed), 4),
        "avg_return_pct": round(sum(realized) / len(realized), 3) if realized else 0.0,
        "profitable_share": round(
            sum(1 for r in realized if r > 0) / len(realized), 4) if realized else 0.0,
    }
