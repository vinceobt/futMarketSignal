"""Market rhythms, computed in SQL and cached.

The dashboard has two kinds of content. Picks, coverage and the scorecard are
single-row lookups — they are read live on every page load and are never stale.
These aggregates sweep 2.5M price rows, so they are computed on a schedule and
cached, with the timestamp surfaced so nothing pretends to be fresher than it is.

They're expressed as SQL window functions rather than pandas: the same numbers
for a fraction of the memory, cheap enough to refresh every collection cycle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .. import db

logger = logging.getLogger(__name__)

CACHE_KEY = "market_insights"
# Ignore day-to-day moves beyond this: they're data artefacts or relistings, not
# the market rhythm we're trying to read.
MAX_MOVE_PCT = 40
# SQLite's %w is 0=Sunday; we want Monday-first.
_DOW_SQL = "CAST(strftime('%w', s.timestamp) AS INTEGER)"
_DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

_RETURNS_CTE = f"""
WITH moves AS (
  SELECT s.timestamp AS ts,
         (s.price * 1.0 / LAG(s.price) OVER (PARTITION BY s.player_id
                                             ORDER BY s.timestamp) - 1) * 100 AS ret
  FROM price_snapshots s
  WHERE s.source = ? AND s.price > 0
)
"""


def _weekly(conn, source: str) -> list[dict]:
    rows = conn.execute(
        _RETURNS_CTE + f"""
        SELECT CAST(strftime('%w', ts) AS INTEGER) AS dow, AVG(ret) AS avg_ret,
               COUNT(*) AS n
        FROM moves WHERE ret IS NOT NULL AND ret BETWEEN -{MAX_MOVE_PCT} AND {MAX_MOVE_PCT}
        GROUP BY dow""", (source,)).fetchall()
    by_dow = {r["dow"]: r for r in rows}
    order = [1, 2, 3, 4, 5, 6, 0]          # Mon..Sun
    out = []
    for d in order:
        r = by_dow.get(d)
        out.append({"day": _DAY_NAMES[d],
                    "ret": round(float(r["avg_ret"]), 3) if r else 0.0,
                    "n": int(r["n"]) if r else 0})
    return out


def _hourly(conn, source: str, *, since: str, min_samples: int = 300) -> list[dict]:
    rows = conn.execute(
        _RETURNS_CTE + f"""
        SELECT CAST(strftime('%w', ts) AS INTEGER) AS dow,
               CAST(strftime('%H', ts) AS INTEGER) AS hour,
               AVG(ret) AS avg_ret, COUNT(*) AS n
        FROM moves
        WHERE ret IS NOT NULL AND ret BETWEEN -{MAX_MOVE_PCT} AND {MAX_MOVE_PCT}
          AND ts >= ? AND CAST(strftime('%H', ts) AS INTEGER) != 0
        GROUP BY dow, hour HAVING n >= ?""", (source, since, min_samples)).fetchall()
    return [{"day": _DAY_NAMES[r["dow"]], "hour": int(r["hour"]),
             "ret": round(float(r["avg_ret"]), 3), "n": int(r["n"])} for r in rows]


def _release_curve(conn, source: str, *, days: int = 14) -> list[dict]:
    """Average price path after release, indexed to each card's first price."""
    rows = conn.execute(
        """
        WITH first_seen AS (
          SELECT player_id, MIN(timestamp) AS t0 FROM price_snapshots
          WHERE source = ? AND price > 0 GROUP BY player_id
        ), base AS (
          SELECT s.player_id, s.price AS p0 FROM price_snapshots s
          JOIN first_seen f ON f.player_id = s.player_id AND f.t0 = s.timestamp
          WHERE s.source = ? AND s.price > 0
        )
        SELECT CAST(julianday(s.timestamp) - julianday(m.release_date) AS INTEGER) AS age,
               AVG(s.price * 100.0 / b.p0) AS idx, COUNT(*) AS n
        FROM price_snapshots s
        JOIN card_meta m ON m.player_id = s.player_id
        JOIN base b ON b.player_id = s.player_id
        WHERE s.source = ? AND s.price > 0 AND m.release_date IS NOT NULL
          AND m.version NOT IN ('Common', 'Rare') AND m.version != ''
          AND (s.price * 100.0 / b.p0) BETWEEN 10 AND 300
        GROUP BY age HAVING age BETWEEN 0 AND ? AND n >= 50
        ORDER BY age""", (source, source, source, days)).fetchall()
    return [{"day": int(r["age"]), "index": round(float(r["idx"]), 1),
             "n": int(r["n"])} for r in rows]


def _promo_reactions(conn, source: str, *, window: int = 3) -> list[dict]:
    """How each promo *type* moves the market: the average daily move over the few
    days from each event, grouped by type (Icon, Hero, TOTS, SBC…)."""
    import datetime as _dt
    from collections import defaultdict

    from . import promos

    rows = conn.execute(
        _RETURNS_CTE + f"""
        SELECT substr(ts, 1, 10) AS d, AVG(ret) AS ar FROM moves
        WHERE ret IS NOT NULL AND ret BETWEEN -{MAX_MOVE_PCT} AND {MAX_MOVE_PCT}
        GROUP BY d""", (source,)).fetchall()
    by_date = {r["d"]: r["ar"] for r in rows}

    acc: dict[str, list[float]] = defaultdict(list)
    for e in conn.execute("SELECT start_date, notes, event_type FROM market_events"):
        t = promos.classify(e["notes"], e["event_type"])
        try:
            base = _dt.date.fromisoformat((e["start_date"] or "")[:10])
        except ValueError:
            continue
        for k in range(window):
            v = by_date.get((base + _dt.timedelta(days=k)).isoformat())
            if v is not None:
                acc[t].append(v)

    out = [{"type": t, "n": len(vals), "avg_move": round(sum(vals) / len(vals), 3)}
           for t, vals in acc.items() if len(vals) >= 3]
    out.sort(key=lambda x: x["avg_move"])
    return out


def compute(conn, *, source: str = "futgg", intraday_since: str = "2026-07-01") -> dict:
    """Recompute every rhythm. Costs a few seconds, not a few minutes."""
    stats = {
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": source,
        "weekly": _weekly(conn, source),
        "hourly": _hourly(conn, source, since=intraday_since),
        "release_curve": _release_curve(conn, source),
        "promo_reactions": _promo_reactions(conn, source),
    }
    logger.info("insights: %d weekday, %d hourly, %d release points",
                len(stats["weekly"]), len(stats["hourly"]),
                len(stats["release_curve"]))
    return stats


def refresh(conn, *, source: str = "futgg") -> dict:
    stats = compute(conn, source=source)
    db.meta_set(conn, CACHE_KEY, json.dumps(stats))
    conn.commit()
    return stats


def load(conn) -> dict | None:
    raw = db.meta_get(conn, CACHE_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("insights cache is corrupt; recompute with `futmarket insights`")
        return None
