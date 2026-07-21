"""Liquidity scoring — rule #1: only ever act on cards you can actually sell.

Every card gets a 0-10 tradeability score and an A/B/C tier that decides how much
attention (polling budget) it earns. The score is *measured from real trade
activity* wherever we have price history — how often the price actually moves is
the truest "is this liquid" signal — and only falls back to a transparent, clearly
temporary cold-start prior for cards we haven't collected yet.

Design intent: as market-wide history accumulates, essentially every card becomes
measured and the prior fades out. We deliberately do NOT hard-code a "high rating =
liquid" rule (FUT liquidity is bimodal — top-meta AND cheap SBC fodder both sell
fast); that non-linearity is for the model to learn, not for this scorer to assume.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

from .. import db
from ..timeseries import to_series

logger = logging.getLogger(__name__)

# Activity at/above this many price *changes* per day counts as fully liquid.
ACTIVITY_FULL = 6.0
# Tier cutoffs on the 0-10 score.
TIER_A_MIN = 6.5
TIER_B_MIN = 3.5
# A card scored only from the cold-start prior can't exceed this (no evidence yet).
PROVISIONAL_CAP = TIER_B_MIN  # provisional cards stay B/C until measured

# --- real completed-sale rates (the honest signal) -------------------------
# Measured across the market: genuinely liquid cards clear 300-1,000 sales/hour,
# mid-tier cards 5-100, and thin ones 1-2. Counting how often a *listed price*
# changed was only ever a proxy for this; where real sale rates exist they win.
SALES_PER_HOUR_FULL = 20.0   # at/above this a card clears in minutes
TIER_A_MIN_SALES = 20.0
TIER_B_MIN_SALES = 2.0


def updates_per_day(rows, *, now: datetime | None = None,
                    window_days: int = 14) -> float | None:
    """Real trade activity: distinct price *changes* per day over the trailing
    window. None when there isn't enough history to measure (needs a spanning
    pair of points)."""
    s = to_series([{"timestamp": r["timestamp"], "price": r["price"]} for r in rows])
    if s.empty:
        return None
    now = now or datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=window_days)
    s = s[s.index >= cutoff]
    if len(s) < 2:
        return None
    span_days = (s.index[-1] - s.index[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return None
    changes = int((s.diff().fillna(0) != 0).sum())  # first diff is NaN->0
    return changes / span_days


def price_band_factor(price: int | None) -> float:
    """Cold-start prior only: a mild 0-1 prior on flip-ability by price band.
    Cheap/mid cards generally clear fastest; ultra-expensive clears slowest.
    Intentionally gentle — it only orders not-yet-measured cards."""
    if not price or price <= 0:
        return 0.5
    if price < 1_000:
        return 0.9
    if price < 50_000:
        return 1.0
    if price < 250_000:
        return 0.7
    if price < 1_000_000:
        return 0.45
    return 0.25


def sales_activity_norm(sales_per_hour: float) -> float:
    """Real sale rate -> 0..1. Log-scaled because the rate spans three orders of
    magnitude (1/hour to 1,000/hour) and the difference between 1 and 10 matters
    far more than between 500 and 1,000."""
    if sales_per_hour is None or sales_per_hour <= 0:
        return 0.0
    return min(1.0, math.log10(1 + sales_per_hour) / math.log10(1 + SALES_PER_HOUR_FULL))


def score_card(*, tradeable: bool, activity: float | None = None,
               price: int | None = None,
               sales_per_hour: float | None = None) -> tuple[float, str, bool]:
    """Return (score 0-10, tier A/B/C, measured?). Untradeable = hard 0/C.

    Preference order:
      1. real completed-sale rate  -- the actual "can I sell this fast" evidence
      2. price-change proxy        -- weaker; capped below tier A, since a moving
                                      listing is not proof anything traded
      3. price-band prior          -- cold start only
    """
    if not tradeable:
        return 0.0, "C", False
    band = price_band_factor(price)

    if sales_per_hour is not None:
        norm = sales_activity_norm(sales_per_hour)
        score = 10.0 * (0.85 * norm + 0.15 * band)
        # Tier off the real rate, not the blended score: what matters to a trader
        # is how quickly it actually sells.
        tier = ("A" if sales_per_hour >= TIER_A_MIN_SALES
                else "B" if sales_per_hour >= TIER_B_MIN_SALES else "C")
        return round(score, 3), tier, True

    if activity is not None:
        activity_norm = min(1.0, activity / ACTIVITY_FULL)
        score = min(TIER_A_MIN - 0.01, 10.0 * (0.75 * activity_norm + 0.25 * band))
        tier = "B" if score >= TIER_B_MIN else "C"
        return round(score, 3), tier, True

    score = min(PROVISIONAL_CAP + 1.0, 10.0 * 0.4 * band)
    tier = "B" if score >= TIER_B_MIN else "C"
    return round(score, 3), tier, False


def refresh_liquidity(conn, *, title: str = "fc26", source: str | None = None,
                      window_days: int = 14, now: datetime | None = None) -> dict:
    """Score every registry card and write the liquidity table.
    Returns per-tier counts plus how many were measured vs provisional."""
    now = now or datetime.now(timezone.utc)
    counts = {"A": 0, "B": 0, "C": 0, "measured": 0, "provisional": 0,
              "from_real_sales": 0}
    for card in db.card_registry(conn, title=title, tradeable_only=False):
        pid = card["player_id"]
        rows = db.snapshots(conn, pid, source)
        activity = updates_per_day(rows, now=now, window_days=window_days)
        price = rows[-1]["price"] if rows else None

        stats = db.sale_stats_get(conn, pid)
        sales_rate = stats["sales_per_hour"] if stats is not None else None

        score, tier, measured = score_card(
            tradeable=bool(card["tradeable"]), activity=activity, price=price,
            sales_per_hour=sales_rate)
        db.upsert_liquidity(conn, player_id=pid, score=score, tier=tier, title=title,
                            updates_per_day=activity, price=price, at=now)
        counts[tier] += 1
        counts["measured" if measured else "provisional"] += 1
        if sales_rate is not None:
            counts["from_real_sales"] += 1
    conn.commit()
    logger.info("liquidity: A=%(A)d B=%(B)d C=%(C)d (measured=%(measured)d)", counts)
    return counts
