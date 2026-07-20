"""Real sale prices — what cards actually change hands for.

Everything upstream of this used the *lowest listing*, which is frequently a
mispriced snipe that no ordinary buyer can catch. That made our backtests
quietly optimistic: they assumed you bought at a price that mostly isn't real.

fut.gg's completed-auction feed gives the last ~100 genuine transactions per
card. From those we take:

  sold_p25 / median / p75   the true going rate, and the band to quote in a
                            recommendation ("buy 20k-22k" instead of a fake
                            exact number)
  sales_per_hour            measured trade activity -- the honest liquidity
                            signal, replacing our price-change proxy
  sold_vs_listed            how far the going rate sits above the cheapest
                            listing, i.e. the realistic entry haircut

Measured on liquid cards: sold_median runs ~1.04x the lowest listing and the
p25-p75 band spans ~20%, while top cards trade hundreds of times per hour. So
filling a buy at the going rate is easy; sniping the floor is not required.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

import httpx
import numpy as np

from .. import db
from ..collectors import history_source
from ..collectors.base import SourceError
from ..collectors.history_source import RateLimited

logger = logging.getLogger(__name__)

MIN_SALES = 5          # below this the percentiles are noise
DEFAULT_DELAY = 1.5
# The feed returns ~100 sales, which on a mid-liquidity card spans 15+ hours. If
# the price trended over that window, the median of all of them is a price that
# no longer exists: a real case quoted a 53,500 band while the card traded at
# 74,500. So the tradeable band comes from the most RECENT sales only. The rate
# still uses the full window, because a rate needs a span to be measured over.
RECENT_SALES = 10
RECENT_HOURS = 2.0


def _recent_slice(sales: list[tuple]) -> list[tuple]:
    """The sales that reflect the price you can actually trade at right now."""
    if not sales:
        return sales
    newest = max(t for t, _ in sales)
    within = [s for s in sales
              if (newest - s[0]).total_seconds() / 3600.0 <= RECENT_HOURS]
    # Prefer a time window, but never fall below a usable sample size.
    return within if len(within) >= MIN_SALES else sales[-RECENT_SALES:]


def summarise_sales(sales: list[tuple], listed: int | None) -> dict | None:
    """Activity rate over the full window; price band from recent sales only."""
    if len(sales) < MIN_SALES:
        return None
    times = [t for t, _ in sales]
    span_hours = max((max(times) - min(times)).total_seconds() / 3600.0, 0.01)

    recent = _recent_slice(sales)
    prices = np.array([p for _, p in recent], dtype=float)
    p25, median, p75 = (float(x) for x in np.percentile(prices, [25, 50, 75]))
    recent_span = max(
        (max(t for t, _ in recent) - min(t for t, _ in recent)).total_seconds() / 3600.0,
        0.01)
    return {
        "n_sales": len(sales),
        "window_hours": round(span_hours, 3),
        "sales_per_hour": round(len(sales) / span_hours, 3),
        "sold_p25": int(round(p25)),
        "sold_median": int(round(median)),
        "sold_p75": int(round(p75)),
        "band_from_sales": len(recent),
        "band_window_hours": round(recent_span, 3),
        "listed_price": int(listed) if listed else None,
        "sold_vs_listed": round(median / listed, 4) if listed else None,
    }


# You pay slightly over the cheapest listing to actually get filled -- measured at
# ~1.04x on liquid cards. This is the realistic window above the live price.
BUY_OVER_LISTED_PCT = 5.0


def buy_band(stats, *, listed_price: int | None = None,
             widen_pct: float = BUY_OVER_LISTED_PCT) -> tuple[int, int] | None:
    """The price range to actually buy in.

    Anchored to the LIVE listed price, not to completed sales. Sales describe
    what the card traded for over the past several hours, which on a moving card
    lags badly: a real case quoted 173k-195k while the card was listed at 241k.
    The listing is always current, so it is the honest anchor, and you pay a
    little over it to get filled.

    Completed sales remain the right source for *liquidity* (sales per hour) --
    they are simply the wrong source for "what will this cost me right now".
    """
    anchor = listed_price
    if anchor is None and stats is not None:
        anchor = stats.get("listed_price") or stats.get("sold_median")
    if not anchor or anchor <= 0:
        return None
    return (int(anchor), int(round(anchor * (1 + widen_pct / 100.0))))


def refresh_sale_stats(conn, *, title: str = "fc26", limit: int | None = None,
                       min_tier: tuple[str, ...] = ("A", "B"),
                       delay: float = DEFAULT_DELAY, retries: int = 6,
                       backoff_base: float = 10.0,
                       max_consecutive_failures: int = 10,
                       client: httpx.Client | None = None,
                       progress: Callable[[dict], None] | None = None) -> dict:
    """Fetch real sale data for tradeable cards and store the going-rate stats."""
    rows = conn.execute(
        f"""SELECT m.player_id, m.definition_id, m.title
            FROM card_meta m LEFT JOIN liquidity l ON l.player_id = m.player_id
            WHERE m.title = ? AND m.tradeable = 1 AND m.definition_id IS NOT NULL
              AND COALESCE(l.tier, 'Z') IN ({','.join('?' for _ in min_tier)})
            ORDER BY COALESCE(l.score, -1) DESC""",
        (title, *min_tier),
    ).fetchall()
    if limit:
        rows = rows[:limit]

    own = client is None
    client = client or httpx.Client(timeout=25.0, headers=history_source._HEADERS)
    res = {"cards": 0, "stored": 0, "thin": 0, "failed": 0}
    consecutive = 0
    try:
        for card in rows:
            game = (card["title"] or "fc26").replace("fc", "") or "26"
            detail = None
            for attempt in range(retries + 1):
                try:
                    detail = history_source.fetch_card_detail(
                        int(card["definition_id"]), game, client=client)
                    break
                except RateLimited:
                    if attempt == retries:
                        break
                    wait = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("rate-limited; backing off %.0fs", wait)
                    time.sleep(wait)
                except SourceError as e:
                    logger.warning("sales fetch failed for %s: %s", card["player_id"], e)
                    break
            res["cards"] += 1
            if detail is None:
                res["failed"] += 1
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    logger.error("circuit breaker: %d consecutive failures", consecutive)
                    break
                continue
            consecutive = 0

            stats = summarise_sales(detail["sales"], detail.get("current"))
            if stats is None:
                res["thin"] += 1
            else:
                db.upsert_sale_stats(conn, player_id=card["player_id"],
                                     title=title, **stats)
                res["stored"] += 1
            # Commit per card, not per batch: batching held the write lock for
            # ~40s at a time, long enough to lock out a concurrent training run.
            conn.commit()
            if res["cards"] % 25 == 0 and progress:
                progress(res)
            if delay:
                time.sleep(delay)
    finally:
        if own:
            client.close()
    conn.commit()
    logger.info("sale stats: %s", res)
    return res
