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
# released cards ARE the trade -- the `release` gate buys a promo card on day 4-6
# of its crash, which measured +11.5% net at +25pp of alpha. A 20-day minimum
# once made every card in its release window invisible, excluding the single most
# tradeable pattern in the game.
#
# Kept at 2 rather than 3 as slack against collection lag: a card released on
# Friday whose first snapshot lands a day late still has two days of history by
# day 4, and stays tradeable in its best window. Rolling stats need
# ``min_periods`` anyway, so a barely-observed card simply carries NaN features
# -- the gates require the columns they use to be non-null, so it can only be
# picked up by a strategy that doesn't need them.
MIN_HISTORY_DAYS = 2
# A snapshot this many times away from the day's median is a bad print (a discard
# price, a mispriced listing), not the market. Wide enough to keep genuine
# intraday swings -- the median card-day already ranges 14%.
OUTLIER_FACTOR = 3.0
# Horizons over which the market-wide drift is measured.
MARKET_HORIZONS = (1, 3, 7)
# fut.gg's version names for base cards. Everything else is a promo/special,
# which is what has a tradeable release curve.
BASE_VERSIONS = ("Common", "Rare")


def load_daily_prices(conn, *, source: str = "futgg", title: str = "fc26",
                      min_history_days: int = MIN_HISTORY_DAYS,
                      outlier_factor: float = OUTLIER_FACTOR) -> pd.DataFrame:
    """Daily **robust** price per card: player_id, date, price, n_snapshots.

    The day's *median* snapshot, not its last. This is not a stylistic choice --
    it is the single cheapest accuracy win available here. What we record is the
    cheapest live listing at a moment in time, and on a card trading hundreds of
    times an hour that is a jittery statistic: one stale or mispriced listing sets
    the number. Measured over the season:

        last snapshot   day-to-day return std 193%, median move 14.6%
        day's median    day-to-day return std  41%, median move  8.2%

    More than half the apparent volatility of this market was an artifact of
    sampling. Everything downstream -- the z-score that defines "the dip", the
    barriers, the labels -- was built on that noise.

    Snapshots more than ``outlier_factor``x away from the day's median in either
    direction are dropped first: those are discard-price prints and mispriced
    listings, not the market. ``n_snapshots`` records how well-observed the day
    was, so a single-print day can be treated with the suspicion it deserves.
    """
    rows = conn.execute(
        """SELECT s.player_id, s.timestamp, s.price
           FROM price_snapshots s
           JOIN card_meta c ON c.player_id = s.player_id
           WHERE s.source = ? AND c.title = ? AND s.price > 0""",
        (source, title),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["player_id", "date", "price", "n_snapshots"])

    df = pd.DataFrame(rows, columns=["player_id", "timestamp", "price"])
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values("ts")
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")

    keys = ["player_id", "date"]
    rough = df.groupby(keys, observed=True)["price"].transform("median")
    keep = (df["price"] <= rough * outlier_factor) & (df["price"] >= rough / outlier_factor)
    df = df[keep]

    daily = (df.groupby(keys, observed=True)["price"]
             .agg(price="median", n_snapshots="size").reset_index())
    daily["price"] = daily["price"].round().astype("int64")

    counts = daily.groupby("player_id", observed=True)["price"].transform("size")
    daily = daily[counts >= min_history_days]
    daily = daily.sort_values(["player_id", "date"]).reset_index(drop=True)
    # ~8k player_ids and ~300 dates repeated across 2M rows: as categories these
    # cost a small dictionary plus an integer code, instead of 2M Python strings.
    for col in ("player_id", "date"):
        daily[col] = daily[col].astype("category")
    return daily


def add_market_features(frame: pd.DataFrame, *, horizons=MARKET_HORIZONS
                        ) -> pd.DataFrame:
    """What the market as a whole is doing, and how this card compares.

    The fact this codebase never represented: **most cards do nothing, and the
    market moves as a block when it moves at all.** The median tradeable card's
    14-day price change is zero, while whole months (2025-09, 2026-02) drop 20-50%
    together. A model asked to predict absolute moves spends its capacity on that
    common drift instead of on what makes one card differ from another.

    So we hand the model the drift explicitly (``market_ret_{h}d``) and, more
    usefully, the part of a card's move that is *its own* (``excess_ret_{h}d``).
    A card that fell 4% on a day the market fell 9% is strong, and nothing in the
    old feature set could say so.
    """
    out = frame.copy()
    for h in horizons:
        own = f"ret_{h}d"
        if own not in out.columns:
            continue
        market = out.groupby("date", observed=True)[own].transform("median")
        out[f"market_ret_{h}d"] = market.astype("float32")
        out[f"excess_ret_{h}d"] = (out[own] - market).astype("float32")
    return out


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
    if "version" in out.columns:
        # A promo/special card, as opposed to a base gold. Only these have a
        # release curve worth trading: a Common or Rare is "released" the day the
        # game launches and never crashes. Measured on special cards alone, the
        # curve is a clean +7 to +10pp of alpha; pooled with base golds it
        # disappears into 6,400 cards that never had a release event.
        out["is_special"] = (~out["version"].astype(str)
                             .isin(BASE_VERSIONS)).astype("float32")
    return out


def _liquidity(conn, title: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT player_id, score AS liq_score, tier AS liq_tier, "
        "updates_per_day AS liq_updates_per_day FROM liquidity WHERE title = ?",
        (title,)).fetchall()
    return pd.DataFrame(rows, columns=["player_id", "liq_score", "liq_tier",
                                       "liq_updates_per_day"])


def _major_promo(conn, dates, title: str) -> pd.DataFrame:
    """Per date: days since / until a *major* promo (Icon/Hero/TOTS/TOTY) — the
    drops that reprice big chunks of the market. Lets the model treat those very
    differently from a routine promo, which the generic PROMO distance can't."""
    from . import promos
    rows = conn.execute(
        "SELECT start_date, notes, event_type FROM market_events WHERE title=?",
        (title,)).fetchall()
    majors = sorted({r["start_date"] for r in rows
                     if promos.is_major(r["notes"], r["event_type"])})
    md = pd.to_datetime(pd.Series(majors), errors="coerce").dropna().sort_values().tolist()
    out = []
    for d in sorted(set(dates)):
        dt = pd.to_datetime(str(d), errors="coerce")
        since = to = np.nan
        if not pd.isna(dt) and md:
            past = [x for x in md if x <= dt]
            future = [x for x in md if x >= dt]
            since = (dt - past[-1]).days if past else np.nan
            to = (future[0] - dt).days if future else np.nan
        out.append({"date": d, "days_since_major_promo": since,
                    "days_to_major_promo": to})
    return pd.DataFrame(out)


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
    # The market's own drift, before cohorts: "did this card fall less than
    # everything else" is the question the whole strategy now turns on.
    frame = add_market_features(frame)
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

    # What kind of news is in the air: distance to the big market-moving promos.
    frame = frame.merge(_major_promo(conn, frame["date"], title), on="date", how="left")

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
    + ["day_of_week", "days_since_card_release", "release_phase", "is_special"]
    # social buzz (Reddit/YouTube/X) -- sparse today, grows over time
    + ["buzz_mentions", "buzz_sentiment", "buzz_z"]
    # what kind of news: distance to the big market-moving promos
    + ["days_since_major_promo", "days_to_major_promo"]
    # the market's own drift, and how much of this card's move is its own
    + [f"market_ret_{h}d" for h in MARKET_HORIZONS]
    + [f"excess_ret_{h}d" for h in MARKET_HORIZONS]
    # how well-observed today's price is -- a one-print day is barely evidence
    + ["n_snapshots"]
)
