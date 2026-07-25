"""Labels — what the model is asked to predict.

The question a label asks is the question the model will answer, so getting it
wrong cannot be fixed downstream. The old label asked *"will this card's price
rise?"* — and the median tradeable card's price doesn't change at all over a
fortnight, so the honest answer is usually "no, and it doesn't matter". A model
trained on that learns the market's common drift rather than what separates one
card from another.

So the primary targets are now **market-relative and cost-aware**:

  excess_return_{h}d   the card's move minus what the whole market did over the
                       same window. "Fell 4% while everything fell 9%" is a win,
                       and nothing in the old label set could express it.
  clears_cost_{h}d     did the trade beat the market *and* clear the full round
                       trip (buy premium + EA's 5% tax + sell slippage)? This is
                       the only question whose answer is money. Break-even is
                       +7.5% gross before any premium — larger than the entire
                       5-day dip bounce, which is why that strategy could not
                       have worked however good the model got.

Labels are produced at **every candidate horizon** (3/5/7/10/14 days), because
how long to hold is part of the decision, not a constant. The dip signal's edge
grows with holding period; a fixed 5-day horizon threw most of it away.

The triple barrier is kept as a secondary target. Its barriers are **sized to
each card**: target is the card's resistance (30-day high), the stop sits just
below its support (30-day low), with noise room and a max-loss cap.

Purging: a row is only labelled if the future price actually exists at that
calendar date. Rows near the end of a card's history, and rows whose future day
was never collected, are left NaN rather than guessed at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate

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
               horizons=None, stop_buffer_pct: float = 2.0,
               stop_min_pct: float = 5.0, stop_max_pct: float = 18.0,
               tax_rate: float = 0.05, sell_slippage_pct: float = 2.0,
               buy_premium_pct: float = 0.0,
               with_barrier: bool = True) -> pd.DataFrame:
    """Attach every target to a (card, day) feature matrix.

    Adds, per horizon in ``horizons`` (default 3/5/7/10/14 plus ``horizon_days``):
      fwd_return_{h}d      forward % move
      bench_return_{h}d    what the market did over the same window
      excess_return_{h}d   fwd minus bench — the part that is this card's own
      net_return_{h}d      fwd after buy premium, EA tax and sell slippage
      clears_cost_{h}d     1 when the trade both beat the market and made money

    Plus, at ``horizon_days`` only, when ``with_barrier``:
      barrier              HIT_TARGET / HIT_STOP / NEITHER (raw 3-way outcome)
      is_profitable        1 when the resistance was reached before the support

    The barrier walk is a Python loop over every row x every future day -- ~28M
    steps on the full matrix. No head trains on it any more, so training skips
    it; it stays available for analysis and for the picks' exit levels.
    """
    wanted = tuple(dict.fromkeys((*(horizons or evaluate.HORIZONS), horizon_days)))
    if frame.empty:
        out = frame.copy()
        for h in wanted:
            for prefix in ("fwd_return", "bench_return", "excess_return",
                           "net_return", "clears_cost"):
                out[f"{prefix}_{h}d"] = np.nan
        out["barrier"] = np.nan
        out["is_profitable"] = np.nan
        return out

    out = frame.sort_values(["player_id", "date"]).copy()
    # Forward returns are matched on the calendar, not on row position: a card
    # whose collection gapped would otherwise have its "5-day" return silently
    # measured over eight.
    out = evaluate.add_forward_returns(out, horizons=wanted)
    out = evaluate.add_benchmark_returns(out, horizons=wanted)

    for h in wanted:
        gross = out[f"fwd_return_{h}d"]
        net = evaluate.net_return_pct(gross, tax_rate=tax_rate,
                                      sell_slippage_pct=sell_slippage_pct,
                                      buy_premium_pct=buy_premium_pct)
        out[f"net_return_{h}d"] = pd.Series(net, index=out.index).astype("float32")
        # Both conditions matter. Beating the market while losing coins is not a
        # trade you can make in a long-only game; making coins while lagging the
        # market means the signal added nothing.
        clears = ((out[f"net_return_{h}d"] > 0)
                  & (out[f"excess_return_{h}d"] > 0)).astype("float32")
        # An unknown future is unknown, not a failure.
        out[f"clears_cost_{h}d"] = clears.where(gross.notna())

    if not with_barrier:
        return out

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
