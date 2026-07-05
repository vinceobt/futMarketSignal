"""Rebound strategy — buy a card that *reliably bounces off a floor*, sell at a
fixed profit target.

The idea, as a trader states it: some cards swing in a range — they keep falling
to a floor and recovering. If a card has done that reliably over the past ~month
and it's near that floor now, that's a buy; once it's up your target, sell.

This module is **pure functions over a price series** (no DB, no I/O), so the
exact same analysis runs in the backtest and in the live advisor — the rule you
prove is the rule that fires. Numbers are all in StrategyParams (config-tuned).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import trailing_window


@dataclass(frozen=True)
class StrategyParams:
    window_days: int = 30          # how much history defines the "floor" and bounces
    min_bounces: int = 3           # rebounds needed to call a card a reliable bouncer
    buy_zone_pct: float = 3.0      # "near the floor" = within this % above the floor
    sell_mode: str = "target"      # "target" (fixed % profit) or "resistance" (ride to the ceiling)
    target_pct: float = 25.0       # target mode: SELL when up this % net of tax vs entry
    resistance_pctile: float = 80.0  # resistance mode: the ceiling = this percentile of window prices
    stop_pct: float = 8.0          # SELL (bail) if price falls this % below the floor; 0 = no stop
    floor_pctile: float = 20.0     # the floor = this percentile of window prices
    floor_drift_pct: float = 10.0  # reject if recent rebound-troughs are this % below earlier ones (real downtrend)
    tax_rate: float = 0.05         # EA sell tax, for the net-of-tax target math

    @classmethod
    def from_config(cls, config) -> "StrategyParams":
        return cls(window_days=config.strategy_window_days,
                   min_bounces=config.strategy_min_bounces,
                   buy_zone_pct=config.strategy_buy_zone_pct,
                   sell_mode=config.strategy_sell_mode,
                   target_pct=config.strategy_target_pct,
                   resistance_pctile=config.strategy_resistance_pctile,
                   stop_pct=config.strategy_stop_pct,
                   floor_drift_pct=config.strategy_floor_drift_pct,
                   tax_rate=config.tax_rate)


@dataclass(frozen=True)
class ReboundView:
    price: float
    floor: float | None
    ceiling: float | None
    bounces: int
    is_reliable: bool
    in_buy_zone: bool
    n_points: int
    reason: str


def _bounces_troughs_peaks(prices: list[float], buy_zone: float,
                           target_pct: float) -> tuple[int, list[float], list[float]]:
    """Find completed rebounds, the trough each bounced from, and the peak each
    bounce reached before the next dip. A rebound: price enters the buy-zone
    (≤ buy_zone), we track the low of that dip, and it counts once price later
    rises ≥ target_pct above that low. Troughs = the real support (they tell a
    stable floor from a falling knife); peaks = the real resistance it bounces up
    to. Both exclude the launch crash by construction (the crash is a one-way
    slide, never a dip-then-recover), so neither is polluted by it."""
    troughs: list[float] = []
    peaks: list[float] = []
    in_dip = False
    dip_low = None
    bounced = False
    cur_peak = None
    for p in prices:
        if p <= buy_zone:
            if bounced and cur_peak is not None:   # a dip closes the prior up-leg
                peaks.append(cur_peak)
                bounced, cur_peak = False, None
            in_dip = True
            dip_low = p if dip_low is None else min(dip_low, p)
        elif in_dip and dip_low is not None and p >= dip_low * (1.0 + target_pct / 100.0):
            troughs.append(dip_low)               # bounce completes
            in_dip, dip_low = False, None
            bounced, cur_peak = True, p
        elif bounced:
            cur_peak = max(cur_peak, p)            # track the peak of this up-leg
    if bounced and cur_peak is not None:
        peaks.append(cur_peak)
    return len(troughs), troughs, peaks


def analyze(series: pd.Series, at: pd.Timestamp, params: StrategyParams) -> ReboundView:
    """Assess a card at time `at` from its trailing `window_days` price series."""
    window = trailing_window(series, at, pd.Timedelta(days=params.window_days))
    price = float(series[series.index <= at].iloc[-1]) if len(series[series.index <= at]) else None
    if price is None:
        return ReboundView(0.0, None, None, 0, False, False, 0, "no price yet")
    if len(window) < max(8, params.min_bounces * 2):
        return ReboundView(price, None, None, 0, False, False, len(window),
                           f"not enough history yet ({len(window)} pts in {params.window_days}d)")

    prices = [float(p) for p in window.values]
    floor = float(pd.Series(prices).quantile(params.floor_pctile / 100.0))
    buy_zone = floor * (1.0 + params.buy_zone_pct / 100.0)
    bounces, troughs, peaks = _bounces_troughs_peaks(prices, buy_zone, params.target_pct)
    # Resistance = a percentile of the *actual rebound peaks*, not the raw window
    # (which the launch crash would inflate). None until we've seen a peak.
    ceiling = (float(pd.Series(peaks).quantile(params.resistance_pctile / 100.0))
               if peaks else None)

    # Falling-knife guard, done over the *rebound troughs* — not the raw window,
    # which still holds the post-launch crash. A stable bouncer keeps returning to
    # ~the same floor; a knife makes progressively lower lows. Compare the earliest
    # troughs to the latest and reject only if the support has genuinely dropped.
    floor_declining = False
    early_floor = late_floor = floor
    if len(troughs) >= 2:
        half = len(troughs) // 2
        early_floor = min(troughs[:half]) if half else troughs[0]
        late_floor = min(troughs[half:])
        floor_declining = late_floor < early_floor * (1.0 - params.floor_drift_pct / 100.0)

    in_buy_zone = price <= buy_zone
    is_reliable = bounces >= params.min_bounces and not floor_declining

    if floor_declining:
        reason = (f"support keeps dropping ({early_floor:,.0f}→{late_floor:,.0f} across "
                  f"rebounds) — a downtrend, not a stable bouncer")
    elif bounces < params.min_bounces:
        reason = f"only {bounces} clean rebound(s) in {params.window_days}d (need {params.min_bounces})"
    else:
        reason = (f"bounced off ~{floor:,.0f} {bounces}× in {params.window_days}d; "
                  f"{'in buy-zone' if in_buy_zone else 'above buy-zone'}")

    return ReboundView(price, floor, ceiling, bounces, is_reliable, in_buy_zone,
                       len(window), reason)


def should_buy(view: ReboundView) -> bool:
    return view.is_reliable and view.in_buy_zone


def target_price(entry: float, params: StrategyParams) -> float:
    """Gross price at which selling nets `target_pct` after tax."""
    return entry * (1.0 + params.target_pct / 100.0) / (1.0 - params.tax_rate)


def stop_price(floor: float, params: StrategyParams) -> float | None:
    if params.stop_pct <= 0:
        return None
    return floor * (1.0 - params.stop_pct / 100.0)


def _net_pct(price: float, entry: float, tax_rate: float) -> float:
    return (price * (1.0 - tax_rate) / entry - 1.0) * 100.0


def should_sell(price: float, entry: float, floor: float | None,
                ceiling: float | None, params: StrategyParams) -> tuple[bool, str]:
    """Exit decision. The stop applies in both modes; the take-profit trigger is
    either a fixed net-of-tax target or a return to the range's resistance/ceiling."""
    # Stop first (both modes): pattern broke below the floor.
    sp = stop_price(floor, params) if floor is not None else None
    if sp is not None and price <= sp:
        return True, f"stopped out ({_net_pct(price, entry, params.tax_rate):+.1f}% — broke below floor)"

    if params.sell_mode == "resistance":
        # Ride it up to the top of its range, but never sell at a loss.
        if ceiling is not None and price >= ceiling and price > entry:
            return True, (f"reached resistance ~{ceiling:,.0f} "
                          f"(+{_net_pct(price, entry, params.tax_rate):.1f}% net)")
        return False, ""

    # target mode (default): fixed net-of-tax profit.
    if price >= target_price(entry, params):
        return True, f"hit +{params.target_pct:.0f}% target ({_net_pct(price, entry, params.tax_rate):+.1f}% net of tax)"
    return False, ""
