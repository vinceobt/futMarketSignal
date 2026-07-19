"""Training-matrix assembly: one row per (card, day) with everything the model sees.

Four families of signal are joined here:
  card       how this card itself is behaving (returns, volatility, where it sits
             in its own recent range)
  cohort     how the groups it belongs to are moving, and its strength relative
             to them (see cohorts.py)
  lifecycle  where we are in the season (see lifecycle.py)
  liquidity  can it actually be sold (rule #1)

Leakage discipline: every rolling statistic is shifted by one day, so a row only
ever sees data strictly *before* its own timestamp. Labels are added separately
by the training step, and look forward only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import db
from . import cohorts, lifecycle

logger = logging.getLogger(__name__)

RETURN_HORIZONS = (1, 3, 7, 14)
RANGE_WINDOW = 30          # days defining a card's "recent range"
VOL_WINDOW = 14
MIN_HISTORY_DAYS = 20      # skip cards too new to describe


def load_daily_prices(conn, *, source: str = "futgg", title: str = "fc26",
                      min_history_days: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Daily last price per card, long format: player_id, date, price."""
    rows = conn.execute(
        """SELECT s.player_id, s.timestamp, s.price
           FROM price_snapshots s
           JOIN card_meta c ON c.player_id = s.player_id
           WHERE s.source = ? AND c.title = ?""",
        (source, title),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["player_id", "date", "price"])

    df = pd.DataFrame(rows, columns=["player_id", "timestamp", "price"])
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("ts")
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    daily = (df.groupby(["player_id", "date"], observed=True)["price"]
             .last().reset_index())

    counts = daily.groupby("player_id", observed=True)["price"].transform("size")
    daily = daily[counts >= min_history_days]
    return daily.sort_values(["player_id", "date"]).reset_index(drop=True)


def add_card_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-card behaviour. All rolling stats are shifted a day: a row never sees
    its own price inside the window it's compared against."""
    if daily.empty:
        return daily.copy()
    out = daily.sort_values(["player_id", "date"]).copy()
    grouped = out.groupby("player_id", observed=True)["price"]

    for h in RETURN_HORIZONS:
        out[f"ret_{h}d"] = (out["price"] / grouped.shift(h) - 1.0) * 100.0

    # Past-only window: describe the range up to *yesterday*.
    prev = grouped.shift(1)
    roll = prev.groupby(out["player_id"], observed=True).rolling(
        RANGE_WINDOW, min_periods=5)
    med = roll.median().reset_index(level=0, drop=True)
    std = roll.std().reset_index(level=0, drop=True)
    lo = roll.min().reset_index(level=0, drop=True)
    hi = roll.max().reset_index(level=0, drop=True)

    out["roll_median"] = med
    out["z_score"] = (out["price"] - med) / std.replace(0, np.nan)
    out["dist_to_floor_pct"] = (out["price"] / lo - 1.0) * 100.0
    out["dist_to_ceiling_pct"] = (hi / out["price"] - 1.0) * 100.0
    out["range_pct"] = (hi / lo - 1.0) * 100.0

    ret1 = out.groupby("player_id", observed=True)["ret_1d"].shift(1)
    out["vol_14d"] = (ret1.groupby(out["player_id"], observed=True)
                      .rolling(VOL_WINDOW, min_periods=5).std()
                      .reset_index(level=0, drop=True))
    out["drawdown_pct"] = (out["price"] / hi - 1.0) * 100.0
    return out


def _attributes(conn, title: str) -> pd.DataFrame:
    rows = conn.execute(
        """SELECT player_id, name, rating, position, league, nation, version
           FROM card_meta WHERE title = ?""", (title,)).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "name", "rating", "position",
                                       "league", "nation", "version"])


def _liquidity(conn, title: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT player_id, score AS liq_score, tier AS liq_tier, "
        "updates_per_day AS liq_updates_per_day FROM liquidity WHERE title = ?",
        (title,)).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "liq_score", "liq_tier",
                                       "liq_updates_per_day"])


def _explode_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (card, day, cohort it belongs to)."""
    records = frame.to_dict("records")
    keys = [cohorts.cohort_keys(r) for r in records]
    repeat = np.fromiter((len(k) for k in keys), dtype=int, count=len(keys))
    exploded = frame.loc[frame.index.repeat(repeat)].copy()
    exploded["cohort_key"] = [k for group in keys for k in group]
    return exploded


def _lifecycle_frame(conn, dates, title: str) -> pd.DataFrame:
    life = lifecycle.load(conn, title=title)
    return pd.DataFrame([{"date": d, **life.features(d)} for d in sorted(set(dates))])


def build_dataset(conn, *, source: str = "futgg", title: str = "fc26",
                  min_history_days: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Assemble the full feature matrix: one row per (card, day)."""
    daily = load_daily_prices(conn, source=source, title=title,
                              min_history_days=min_history_days)
    if daily.empty:
        logger.warning("no daily prices for source=%s title=%s", source, title)
        return daily

    frame = add_card_features(daily)
    frame = frame.merge(_attributes(conn, title), on="player_id", how="left")

    exploded = _explode_cohorts(frame)
    indices = cohorts.build_indices(
        exploded[["cohort_key", "date", "price"]].copy())
    frame = cohorts.attach_cohort_features(exploded, indices)

    frame = frame.merge(_lifecycle_frame(conn, frame["date"], title),
                        on="date", how="left")
    frame = frame.merge(_liquidity(conn, title), on="player_id", how="left")

    frame = frame.sort_values(["date", "player_id"]).reset_index(drop=True)
    logger.info("dataset built: %d rows x %d cols, %d cards, %s..%s",
                len(frame), frame.shape[1], frame["player_id"].nunique(),
                frame["date"].min(), frame["date"].max())
    return frame


FEATURE_COLUMNS = (
    [f"ret_{h}d" for h in RETURN_HORIZONS]
    + ["z_score", "dist_to_floor_pct", "dist_to_ceiling_pct", "range_pct",
       "vol_14d", "drawdown_pct"]
    + [f"cohort_ret_{h}d" for h in cohorts.INDEX_HORIZONS]
    + [f"rel_strength_{h}d" for h in cohorts.INDEX_HORIZONS]
    + ["n_cohorts", "days_since_launch", "days_to_next_promo",
       "days_since_last_promo", "days_to_next_totw", "days_since_last_totw",
       "days_to_next_sbc", "days_since_last_sbc", "active_sbc_count",
       "is_weekend_window", "rating", "liq_score", "liq_updates_per_day"]
)
