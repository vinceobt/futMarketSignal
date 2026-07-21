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
# Just enough history to compute a short return. Deliberately low: newly
# released cards ARE the trade -- a promo card crashes ~33% over four days and
# bottoms around day nine, which is precisely the release_phase feature. A 20-day
# minimum made every card in its release window invisible to the model, excluding
# the single most tradeable pattern in the game.
MIN_HISTORY_DAYS = 3


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
    daily = daily.sort_values(["player_id", "date"]).reset_index(drop=True)
    # ~8k player_ids and ~300 dates repeated across 2M rows: as categories these
    # cost a small dictionary plus an integer code, instead of 2M Python strings.
    for col in ("player_id", "date"):
        daily[col] = daily[col].astype("category")
    return daily


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
        """SELECT player_id, name, rating, position, league, nation, version,
                  release_date
           FROM card_meta WHERE title = ?""", (title,)).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "name", "rating", "position",
                                       "league", "nation", "version", "release_date"])


def add_behaviour_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The rhythms a human trader actually watches.

    Two patterns dominate this market and neither is visible from price alone:

    * The weekly cycle. Promos drop on Friday; Saturday is reliably the worst day
      (-1.86% average) and Monday the best (+1.31%) -- a ~3% swing every week.
    * The release curve. A new promo card crashes ~33% over its first four days,
      bottoms around day nine, then recovers ~12%. Where a card sits on that curve
      matters far more than its absolute price.

    So we tell the model what day of the week it is and how old the card is.
    """
    out = frame.copy()
    # `date` is stored as a category to keep 2M rows cheap; datetime arithmetic
    # needs real strings, so widen it here rather than pay for it everywhere.
    raw_dates = out["date"]
    if isinstance(raw_dates.dtype, pd.CategoricalDtype):
        raw_dates = raw_dates.astype(str)
    dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    out["day_of_week"] = dates.dt.dayofweek                    # 0 = Monday
    out["is_weekend_window"] = dates.dt.dayofweek.isin((3, 4, 5, 6)).astype(int)

    if "release_date" in out.columns:
        released = pd.to_datetime(out["release_date"], format="%Y-%m-%d",
                                  errors="coerce")
        out["days_since_card_release"] = (dates - released).dt.days
        # The crash-and-recover phase: fresh drop, the slide, the bottom, recovery.
        out["release_phase"] = pd.cut(
            out["days_since_card_release"],
            bins=[-np.inf, 0, 2, 5, 11, 21, np.inf],
            labels=[0, 1, 2, 3, 4, 5]).astype("float")
    return out


def _liquidity(conn, title: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT player_id, score AS liq_score, tier AS liq_tier, "
        "updates_per_day AS liq_updates_per_day FROM liquidity WHERE title = ?",
        (title,)).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "liq_score", "liq_tier",
                                       "liq_updates_per_day"])


def _sentiment(conn) -> pd.DataFrame:
    """Per (card, day) social buzz: how much it's being talked about and how the
    talk leans. Sparse until weeks of chatter accumulate, so missing days are
    simply 'no buzz' (0) — the model learns from it as the history fills in."""
    rows = conn.execute(
        """SELECT player_id, substr(timestamp, 1, 10) AS date,
                  SUM(mention_count) AS buzz_mentions,
                  AVG(sentiment) AS buzz_sentiment,
                  MAX(COALESCE(buzz_z, 0)) AS buzz_z
           FROM sentiment_signals GROUP BY player_id, substr(timestamp, 1, 10)"""
    ).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "date", "buzz_mentions",
                                       "buzz_sentiment", "buzz_z"])


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
    frame = add_behaviour_features(frame)

    # Price band is a cohort dimension, so derive it before grouping.
    frame["band"] = cohorts.price_band_series(frame["price"])
    frame = cohorts.add_cohort_features(frame)

    frame = frame.merge(_lifecycle_frame(conn, frame["date"], title),
                        on="date", how="left")
    frame = frame.merge(_liquidity(conn, title), on="player_id", how="left")

    # Social buzz (Reddit/YouTube/X). Sparse today; a left-join + fill means "no
    # buzz" is 0 and the model starts learning the signal as chatter accumulates.
    buzz = _sentiment(conn)
    frame = frame.merge(buzz, on=["player_id", "date"], how="left")
    for col in ("buzz_mentions", "buzz_sentiment", "buzz_z"):
        frame[col] = frame[col].fillna(0.0)

    frame = frame.sort_values(["date", "player_id"]).reset_index(drop=True)
    # float64 buys no accuracy for features whose inputs are integer coin prices.
    wide = frame.select_dtypes(include=["float64"]).columns
    frame[wide] = frame[wide].astype("float32")
    # Card attributes repeat heavily (30 leagues, ~180 versions) -- same trick.
    for col in ("position", "league", "nation", "version", "band", "liq_tier"):
        if col in frame.columns:
            frame[col] = frame[col].astype("category")
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
       "days_to_next_announce", "days_since_last_announce",
       "is_weekend_window", "rating", "liq_score", "liq_updates_per_day"]
    # the weekly rhythm and the release curve -- see add_behaviour_features
    + ["day_of_week", "days_since_card_release", "release_phase"]
    # social buzz (Reddit/YouTube/X) -- sparse today, grows over time
    + ["buzz_mentions", "buzz_sentiment", "buzz_z"]
)
