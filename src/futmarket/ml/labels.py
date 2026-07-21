"""Labels — what the model is asked to predict.

Two targets, matching the two heads:

  forward return    the price move N days ahead (the forecaster's target)
  triple barrier    did this card reach its own resistance before its own
                    support, within the window? (the direction head's target)

The barriers are **sized to each card**, not flat percentages. Target is the
card's recent resistance (its 30-day high); the stop sits just below its support
(its 30-day low), with room for noise and a max-loss cap. This matters: these
cards swing 17-24% in a week, so a flat 8% stop is inside the noise and books
routine dips as losses. A card that swings wide gets a wider stop.

Purging: a row is only labelled if the full horizon of future prices exists.
Rows near the end of a card's history are left unlabelled (NaN) rather than
guessed at, so the model never learns from a truncated future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Label meanings for the direction head.
HIT_TARGET = 1.0
HIT_STOP = -1.0
NEITHER = 0.0

# When a card's range isn't known yet (too little history), fall back to a modest
# fixed target rather than leaving it unlabelled.
FALLBACK_TARGET_PCT = 15.0


def card_barriers(frame: pd.DataFrame, *, stop_buffer_pct: float = 2.0,
                  stop_min_pct: float = 5.0, stop_max_pct: float = 18.0,
                  fallback_target_pct: float = FALLBACK_TARGET_PCT
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (upper_price, lower_price) from each card's own structure.

    Upper = the card's resistance (30-day high, via ``dist_to_ceiling_pct``).
    Lower = just below its support (30-day low, via ``dist_to_floor_pct``), the
    stop distance floored for noise room and capped for max loss.
    """
    price = frame["price"].to_numpy(dtype=float)
    n = len(price)

    def _col(name):
        return (frame[name].to_numpy(dtype=float) if name in frame.columns
                else np.full(n, np.nan))

    ceil_pct = _col("dist_to_ceiling_pct")
    floor_pct = _col("dist_to_floor_pct")

    target_pct = np.where(np.isfinite(ceil_pct) & (ceil_pct > 0),
                          ceil_pct, fallback_target_pct)
    upper = price * (1.0 + target_pct / 100.0)

    stop_dist = np.where(np.isfinite(floor_pct), floor_pct + stop_buffer_pct,
                         stop_min_pct)
    stop_dist = np.clip(stop_dist, stop_min_pct, stop_max_pct)
    lower = price * (1.0 - stop_dist / 100.0)
    return upper, lower


def triple_barrier(prices: np.ndarray, *, horizon: int,
                   upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Which barrier a position opened at each point would hit first.

    ``upper``/``lower`` are per-row price levels (from ``card_barriers``).
    Returns HIT_TARGET / HIT_STOP / NEITHER per row, NaN where the full horizon
    of future prices isn't available.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    if horizon < 1:
        return out
    for i in range(n):
        last = i + horizon
        if last >= n:            # not enough future -> purge, don't guess
            break
        entry = prices[i]
        if not entry or entry <= 0 or not np.isfinite(entry):
            continue
        u, l = upper[i], lower[i]
        label = NEITHER
        for j in range(i + 1, last + 1):
            price = prices[j]
            if not np.isfinite(price):
                continue
            if price >= u:
                label = HIT_TARGET
                break
            if price <= l:
                label = HIT_STOP
                break
        out[i] = label
    return out


def add_labels(frame: pd.DataFrame, *, horizon_days: int = 5,
               stop_buffer_pct: float = 2.0, stop_min_pct: float = 5.0,
               stop_max_pct: float = 18.0) -> pd.DataFrame:
    """Attach both targets to a (card, day) feature matrix.

    Adds:
      fwd_return_{h}d   forward % move (regression target)
      barrier           HIT_TARGET / HIT_STOP / NEITHER (raw 3-way outcome)
      is_profitable     1 when the resistance was reached before the support
    """
    if frame.empty:
        out = frame.copy()
        out[f"fwd_return_{horizon_days}d"] = np.nan
        out["barrier"] = np.nan
        out["is_profitable"] = np.nan
        return out

    out = frame.sort_values(["player_id", "date"]).copy()
    grouped = out.groupby("player_id", observed=True)["price"]
    future = grouped.shift(-horizon_days)
    out[f"fwd_return_{horizon_days}d"] = (future / out["price"] - 1.0) * 100.0

    upper, lower = card_barriers(out, stop_buffer_pct=stop_buffer_pct,
                                 stop_min_pct=stop_min_pct, stop_max_pct=stop_max_pct)

    barriers = np.full(len(out), np.nan)
    positions = 0
    for _, block in out.groupby("player_id", observed=True, sort=False):
        m = len(block)
        prices = block["price"].to_numpy(dtype=float)
        labels = triple_barrier(prices, horizon=horizon_days,
                                upper=upper[positions:positions + m],
                                lower=lower[positions:positions + m])
        barriers[positions:positions + m] = labels
        positions += m
    out["barrier"] = barriers
    out["is_profitable"] = np.where(np.isnan(barriers), np.nan,
                                    (barriers == HIT_TARGET).astype(float))
    return out
