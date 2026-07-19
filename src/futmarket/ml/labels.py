"""Labels — what the model is asked to predict.

Two targets, matching the two heads:

  forward return    the price move N days ahead (the forecaster's target)
  triple barrier    did this card hit a profit target before hitting a stop,
                    within the window? (the direction head's target)

The triple barrier deliberately mirrors the *existing* strategy's economics: the
upside barrier is grossed up for EA's sell tax exactly as
`strategy.target_price` does, so a "profitable" label means a trade that would
genuinely have paid out after tax — not a paper gain.

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


def barrier_multipliers(*, target_pct: float, stop_pct: float,
                        tax_rate: float) -> tuple[float, float]:
    """Price multiples for the upper/lower barriers.

    Upper is grossed up by the sell tax (you must sell higher than the naive
    target to actually net `target_pct`), matching strategy.target_price.
    """
    upper = (1.0 + target_pct / 100.0) / (1.0 - tax_rate)
    lower = 1.0 - stop_pct / 100.0
    return upper, lower


def triple_barrier(prices: np.ndarray, *, horizon: int,
                   upper_mult: float, lower_mult: float) -> np.ndarray:
    """Which barrier a position opened at each point would hit first.

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
        upper, lower = entry * upper_mult, entry * lower_mult
        label = NEITHER
        for j in range(i + 1, last + 1):
            price = prices[j]
            if not np.isfinite(price):
                continue
            if price >= upper:
                label = HIT_TARGET
                break
            if price <= lower:
                label = HIT_STOP
                break
        out[i] = label
    return out


def add_labels(frame: pd.DataFrame, *, horizon_days: int = 7,
               target_pct: float = 25.0, stop_pct: float = 8.0,
               tax_rate: float = 0.05) -> pd.DataFrame:
    """Attach both targets to a (card, day) feature matrix.

    Adds:
      fwd_return_{h}d   forward % move (regression target)
      barrier           HIT_TARGET / HIT_STOP / NEITHER (raw 3-way outcome)
      is_profitable     1 when the target barrier was hit first (binary target)
    """
    if frame.empty:
        out = frame.copy()
        out[f"fwd_return_{horizon_days}d"] = np.nan
        out["barrier"] = np.nan
        out["is_profitable"] = np.nan
        return out

    upper_mult, lower_mult = barrier_multipliers(
        target_pct=target_pct, stop_pct=stop_pct, tax_rate=tax_rate)

    out = frame.sort_values(["player_id", "date"]).copy()
    grouped = out.groupby("player_id", observed=True)["price"]
    future = grouped.shift(-horizon_days)
    out[f"fwd_return_{horizon_days}d"] = (future / out["price"] - 1.0) * 100.0

    barriers = np.full(len(out), np.nan)
    positions = 0
    for _, block in out.groupby("player_id", observed=True, sort=False):
        prices = block["price"].to_numpy(dtype=float)
        labels = triple_barrier(prices, horizon=horizon_days,
                                upper_mult=upper_mult, lower_mult=lower_mult)
        barriers[positions:positions + len(block)] = labels
        positions += len(block)
    out["barrier"] = barriers
    out["is_profitable"] = np.where(np.isnan(barriers), np.nan,
                                    (barriers == HIT_TARGET).astype(float))
    return out
