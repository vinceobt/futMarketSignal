"""The decision engine — one rule that turns a price series into BUY/SELL/SKIP.

The idea, as a trader states it: some cards swing in a range — they keep falling
to a floor and recovering. If a card has bounced off a *validated* floor
repeatedly, sits near that floor now, is statistically cheap versus its own
recent range, and the range is wide enough to clear EA's 5% sell tax with real
margin — that's a BUY. Holding, it's a SELL at the net-of-tax target, back at
resistance, or through the stop. Anything the data can't support with all of
those checks is an explicit SKIP with the reasons attached.

All statistics are duration-weighted (see features.robust_stats): each sample
counts by how long its price held, so daily-then-hourly BIN history doesn't
skew the math, and the current price never sits inside its own reference range.

This module is **pure functions over a price series** (no DB, no I/O), so the
exact same decide() runs in the backtest and in the live advisor — the rule you
prove is the rule that fires. Numbers are all in StrategyParams (config-tuned).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median as _median

import numpy as np
import pandas as pd

from .features import (duration_weights, robust_cv, robust_stats, robust_z,
                       trailing_window, weighted_quantile)
from .signals import BUY, CRASHING_EVENTS, HOLD, SELL, SKIP

# Machine reason codes: every non-BUY decision names exactly which gates failed,
# so an alert / log line is auditable without rereading the series.
NO_DATA = "NO_DATA"
INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
SPAN_TOO_SHORT = "SPAN_TOO_SHORT"
STALE = "STALE"
FLAT_MARKET = "FLAT_MARKET"
TOO_VOLATILE = "TOO_VOLATILE"
NOT_NEAR_FLOOR = "NOT_NEAR_FLOOR"
FLOOR_BROKEN = "FLOOR_BROKEN"
Z_NOT_LOW = "Z_NOT_LOW"
SUPPORT_UNVALIDATED = "SUPPORT_UNVALIDATED"
TOO_FEW_BOUNCES = "TOO_FEW_BOUNCES"
FLOOR_DRIFTING = "FLOOR_DRIFTING"
MARGIN_TOO_THIN = "MARGIN_TOO_THIN"
EVENT_IMMINENT = "EVENT_IMMINENT"


@dataclass(frozen=True)
class StrategyParams:
    window_days: int = 30          # how much history defines the "floor" and bounces
    min_bounces: int = 3           # rebounds needed to call a card a reliable bouncer
    buy_zone_pct: float = 3.0      # "near the floor" = within this % above the floor
    sell_mode: str = "either"      # "target" | "resistance" | "either" (any trigger)
    target_pct: float = 25.0       # SELL when up this % net of tax vs entry
    resistance_pctile: float = 80.0  # the ceiling = this duration-weighted percentile
    stop_pct: float = 8.0          # SELL (bail) if price falls this % below the floor; 0 = no stop
    floor_pctile: float = 15.0     # the floor = this duration-weighted percentile
    floor_drift_pct: float = 10.0  # reject if dip-lows trend down more than this % per window
    tax_rate: float = 0.05         # EA sell tax, for the net-of-tax margin/target math
    # Support validation + statistical cheapness gates.
    touch_tol_pct: float = 3.0     # a "touch" = price within this % of the floor
    min_touches: int = 3           # distinct floor touches to call the support real
    z_buy: float = 0.5             # BUY needs robust z <= -z_buy (cheap vs own range)
    # Risk / worth-it gates (the SKIP conditions). Note cv_max is a chaos rail,
    # not a tight filter: a healthy rebounder's floor→ceiling range IS dispersion
    # (a 1.2k→2.8k oscillator runs CV ≈ 0.4-0.6), and the structure gates below
    # (touches/bounces/drift) do the orderly-vs-chaotic discrimination.
    cv_max: float = 0.60           # SKIP if robust CV above this (pure chaos, no structure)
    min_margin_pct: float = 8.0    # SKIP if floor→ceiling nets less than this after tax
    min_points: int = 10           # SKIP below this many weighted samples
    min_span_hours: float = 168.0  # SKIP with less than this much wall-clock history
    max_stale_hours: float = 48.0  # SKIP if the last sample is older than this
    exit_band_pct: float = 2.0     # resistance exit fires within this % below the ceiling
    event_guard_days: int = 2      # SKIP a BUY if a crashing event is this close
    max_gap_hours: float = 48.0    # cap on any one sample's duration weight

    @classmethod
    def from_config(cls, config) -> "StrategyParams":
        return cls(window_days=config.strategy_window_days,
                   min_bounces=config.strategy_min_bounces,
                   buy_zone_pct=config.strategy_buy_zone_pct,
                   sell_mode=config.strategy_sell_mode,
                   target_pct=config.strategy_target_pct,
                   resistance_pctile=config.strategy_resistance_pctile,
                   stop_pct=config.strategy_stop_pct,
                   floor_pctile=config.strategy_floor_pctile,
                   floor_drift_pct=config.strategy_floor_drift_pct,
                   tax_rate=config.tax_rate,
                   touch_tol_pct=config.strategy_touch_tol_pct,
                   min_touches=config.strategy_min_touches,
                   z_buy=config.strategy_z_buy,
                   cv_max=config.strategy_cv_max,
                   min_margin_pct=config.strategy_min_margin_pct,
                   min_points=config.strategy_min_points,
                   min_span_hours=config.strategy_min_span_hours,
                   max_stale_hours=config.strategy_max_stale_hours,
                   exit_band_pct=config.strategy_exit_band_pct,
                   event_guard_days=config.strategy_event_guard_days,
                   max_gap_hours=config.strategy_max_gap_hours)


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
    # Robust-statistics extensions (None whenever the data gates failed).
    robust_z: float | None = None
    robust_cv: float | None = None
    touches: int = 0
    floor_drifting: bool = False
    n_eff: int = 0
    span_hours: float = 0.0
    net_margin_pct: float | None = None
    buy_ready: bool = False   # every BUY gate except the event guard passed


@dataclass(frozen=True)
class Decision:
    action: str                 # BUY | SELL | HOLD | SKIP
    confidence: float
    codes: tuple[str, ...]      # machine reasons (failed gates, or the sell trigger)
    detail: str                 # human sentence built from the actual numbers
    view: ReboundView | None


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


def _count_touches(prices: list[float], floor: float, tol_pct: float) -> int:
    """Distinct visits to the floor band, with hysteresis: a touch registers at
    ≤ floor·(1+tol) and only re-arms once price clears floor·(1+2·tol), so noise
    wiggling inside the band counts once."""
    band = floor * (1.0 + tol_pct / 100.0)
    rearm = floor * (1.0 + 2.0 * tol_pct / 100.0)
    touches, armed = 0, True
    for p in prices:
        if armed and p <= band:
            touches += 1
            armed = False
        elif not armed and p >= rearm:
            armed = True
    return touches


def _dip_lows(window: pd.Series, buy_zone: float) -> list[tuple[float, float]]:
    """(days-from-window-start, low) of every excursion into the buy zone —
    every candidate support test, whether or not a full rebound completed."""
    lows: list[tuple[float, float]] = []
    t0 = window.index[0]
    in_dip, low, low_t = False, None, None
    for t, p in window.items():
        if p <= buy_zone:
            if not in_dip or p < low:
                low, low_t = float(p), t
            in_dip = True
        elif in_dip:
            lows.append(((low_t - t0).total_seconds() / 86400.0, low))
            in_dip, low = False, None
    if in_dip:
        lows.append(((low_t - t0).total_seconds() / 86400.0, low))
    return lows


def _floor_is_drifting(lows: list[tuple[float, float]], floor: float,
                       params: StrategyParams) -> bool:
    """Falling-knife test on the dip lows. With ≥3 lows: Theil–Sen (median of
    pairwise slopes — one outlier low can't fake or hide a trend); drifting when
    the fitted drop across the window exceeds floor_drift_pct of the floor.
    With exactly 2: plain early-vs-late comparison."""
    if len(lows) >= 3:
        slopes = [(l2 - l1) / (t2 - t1)
                  for i, (t1, l1) in enumerate(lows)
                  for (t2, l2) in lows[i + 1:] if t2 > t1]
        if not slopes:
            return False
        beta = _median(slopes)  # coins per day
        return beta * params.window_days <= -(params.floor_drift_pct / 100.0) * floor
    if len(lows) == 2:
        return lows[1][1] < lows[0][1] * (1.0 - params.floor_drift_pct / 100.0)
    return False


def _assess(series: pd.Series, at: pd.Timestamp,
            params: StrategyParams) -> tuple[ReboundView, list[str]]:
    """All the math: levels, gates, and the failed-gate codes for a would-be BUY
    (everything except the event guard, which needs calendar context)."""
    past = series[series.index <= at]
    if past.empty:
        return ReboundView(0.0, None, None, 0, False, False, 0, "no price yet"), [NO_DATA]
    price = float(past.iloc[-1])
    window = trailing_window(series, at, pd.Timedelta(days=params.window_days))

    # --- data-quality gates -------------------------------------------------
    codes: list[str] = []
    st = robust_stats(window, at, params.max_gap_hours)
    stale_h = (at - past.index[-1]).total_seconds() / 3600.0
    if st is None or st.n_eff < params.min_points:
        n = st.n_eff if st else 0
        return ReboundView(price, None, None, 0, False, False, len(window),
                           f"not enough history yet ({n} weighted pts in "
                           f"{params.window_days}d)", n_eff=n), [INSUFFICIENT_POINTS]
    if st.span_hours < params.min_span_hours:
        return ReboundView(price, None, None, 0, False, False, len(window),
                           f"history spans only {st.span_hours:.0f}h "
                           f"(need {params.min_span_hours:.0f}h)",
                           n_eff=st.n_eff, span_hours=st.span_hours), [SPAN_TOO_SHORT]
    if stale_h > params.max_stale_hours:
        return ReboundView(price, None, None, 0, False, False, len(window),
                           f"last price is {stale_h:.0f}h old — data too stale to act on",
                           n_eff=st.n_eff, span_hours=st.span_hours), [STALE]

    z = robust_z(price, st)
    cv = robust_cv(st)
    if z is None:  # MAD == 0: the market is a flat line, nothing to trade
        return ReboundView(price, None, None, 0, False, False, len(window),
                           "price has been flat — a dead market, no range to trade",
                           robust_cv=cv, n_eff=st.n_eff,
                           span_hours=st.span_hours), [FLAT_MARKET]
    if cv is not None and cv > params.cv_max:
        codes.append(TOO_VOLATILE)

    # --- levels ---------------------------------------------------------------
    w = duration_weights(window.index, at, params.max_gap_hours)
    values = np.asarray(window.values, dtype=float)
    floor = weighted_quantile(values, w, params.floor_pctile / 100.0)
    buy_zone = floor * (1.0 + params.buy_zone_pct / 100.0)
    prices = [float(p) for p in window.values]
    bounces, troughs, peaks = _bounces_troughs_peaks(prices, buy_zone, params.target_pct)
    # Resistance from the whole (duration-weighted) range — defined from day one —
    # tightened by the actual rebound peaks once enough exist, so the margin gate
    # never banks on a launch-crash high the card no longer reaches.
    ceiling = weighted_quantile(values, w, params.resistance_pctile / 100.0)
    if len(peaks) >= 2:
        ceiling = min(ceiling, float(np.quantile(peaks, params.resistance_pctile / 100.0)))

    touches = _count_touches(prices, floor, params.touch_tol_pct)
    drifting = _floor_is_drifting(_dip_lows(window, buy_zone), floor, params)
    net_margin = (ceiling * (1.0 - params.tax_rate) / price - 1.0) * 100.0

    in_buy_zone = price <= buy_zone
    is_reliable = (touches >= params.min_touches
                   and bounces >= params.min_bounces
                   and not drifting)

    # --- BUY gates -------------------------------------------------------------
    if not in_buy_zone:
        codes.append(NOT_NEAR_FLOOR)
    # Entry/exit symmetry: if this price would already trip the stop, the
    # support has failed — that's a knife mid-fall, not a dip to buy.
    sp = stop_price(floor, params)
    if sp is not None and price <= sp:
        codes.append(FLOOR_BROKEN)
    if z > -params.z_buy:
        codes.append(Z_NOT_LOW)
    if touches < params.min_touches:
        codes.append(SUPPORT_UNVALIDATED)
    if bounces < params.min_bounces:
        codes.append(TOO_FEW_BOUNCES)
    if drifting:
        codes.append(FLOOR_DRIFTING)
    if net_margin < params.min_margin_pct:
        codes.append(MARGIN_TOO_THIN)

    if drifting:
        reason = "support keeps dropping across dips — a downtrend, not a stable bouncer"
    elif not is_reliable:
        reason = (f"{touches} floor touch(es), {bounces} clean rebound(s) in "
                  f"{params.window_days}d (need {params.min_touches}/{params.min_bounces})")
    else:
        reason = (f"bounced off ~{floor:,.0f} {bounces}× ({touches} touches) in "
                  f"{params.window_days}d; "
                  f"{'in buy-zone' if in_buy_zone else 'above buy-zone'}; "
                  f"range nets {net_margin:+.1f}% after tax")

    view = ReboundView(price, floor, ceiling, bounces, is_reliable, in_buy_zone,
                       len(window), reason,
                       robust_z=z, robust_cv=cv, touches=touches,
                       floor_drifting=drifting, n_eff=st.n_eff,
                       span_hours=st.span_hours, net_margin_pct=net_margin,
                       buy_ready=not codes)
    return view, codes


def analyze(series: pd.Series, at: pd.Timestamp, params: StrategyParams) -> ReboundView:
    """Assess a card at time `at` from its trailing `window_days` price series."""
    view, _ = _assess(series, at, params)
    return view


_GATE_DETAIL = {
    NO_DATA: "no prices stored yet",
    INSUFFICIENT_POINTS: "too few samples to judge",
    SPAN_TOO_SHORT: "history window too short",
    STALE: "last price too old",
    FLAT_MARKET: "dead flat market",
    TOO_VOLATILE: "swings too wild to be a tradable range",
    NOT_NEAR_FLOOR: "price is above the buy-zone",
    FLOOR_BROKEN: "price has already broken below the floor (falling knife)",
    Z_NOT_LOW: "not statistically cheap vs its own range",
    SUPPORT_UNVALIDATED: "floor not tested often enough",
    TOO_FEW_BOUNCES: "too few completed rebounds",
    FLOOR_DRIFTING: "support keeps dropping (falling knife)",
    MARGIN_TOO_THIN: "floor→ceiling nets too little after the 5% tax",
    EVENT_IMMINENT: "a supply-flooding event is imminent",
}


def _buy_confidence(view: ReboundView, params: StrategyParams) -> float:
    """Half how deep the dip is (robust z, saturating at 3σ), half how fat the
    net margin is (saturating at 2× the minimum)."""
    z_part = min(1.0, abs(view.robust_z or 0.0) / 3.0)
    m_part = min(1.0, (view.net_margin_pct or 0.0) / (2.0 * params.min_margin_pct))
    return min(1.0, 0.5 * z_part + 0.5 * m_part)


def decide(series: pd.Series, at: pd.Timestamp, params: StrategyParams, *,
           entry_price: float | None = None,
           days_to_next_event: int | None = None,
           next_event_type: str | None = None) -> Decision:
    """The one rule. Flat → BUY or SKIP (with every failed gate named).
    Holding (entry_price given) → SELL on stop/target/resistance, else HOLD."""
    view, codes = _assess(series, at, params)

    if entry_price is not None:
        # Managing an open position: the exit rule runs even when the data gates
        # would refuse a fresh entry — a held position must always be manageable.
        sell, why = should_sell(view.price, entry_price, view.floor, view.ceiling, params)
        if sell:
            return Decision(SELL, 1.0, (SELL,), why, view)
        return Decision(HOLD, 0.0, (),
                        f"holding from {entry_price:,.0f}; no exit trigger yet "
                        f"(price {view.price:,.0f})", view)

    if (not codes and days_to_next_event is not None
            and days_to_next_event <= params.event_guard_days
            and next_event_type in CRASHING_EVENTS):
        codes = [EVENT_IMMINENT]
        view = replace(view, buy_ready=False,
                       reason=f"{next_event_type} {days_to_next_event}d away "
                              f"could sink it further — waiting")

    if not codes:
        return Decision(BUY, _buy_confidence(view, params), (BUY,),
                        f"price {view.price:,.0f} at a validated floor "
                        f"(~{view.floor:,.0f}, {view.touches} touches, "
                        f"{view.bounces} rebounds, z {view.robust_z:+.2f}); "
                        f"ceiling ~{view.ceiling:,.0f} nets "
                        f"{view.net_margin_pct:+.1f}% after tax", view)

    detail = "; ".join(_GATE_DETAIL.get(c, c) for c in codes)
    return Decision(SKIP, 0.0, tuple(codes), detail, view)


def should_buy(view: ReboundView) -> bool:
    return view.buy_ready


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
    """Exit decision. The stop applies in every mode; the take-profit trigger is
    a fixed net-of-tax target, a return to resistance, or (mode "either") both."""
    # Stop first (every mode): pattern broke below the floor.
    sp = stop_price(floor, params) if floor is not None else None
    if sp is not None and price <= sp:
        return True, f"stopped out ({_net_pct(price, entry, params.tax_rate):+.1f}% — broke below floor)"

    if params.sell_mode in ("resistance", "either"):
        # Ride it up to the top of its range, but never sell at a net loss.
        band = (ceiling * (1.0 - params.exit_band_pct / 100.0)
                if ceiling is not None else None)
        if (band is not None and price >= band
                and price * (1.0 - params.tax_rate) > entry):
            return True, (f"reached resistance ~{ceiling:,.0f} "
                          f"(+{_net_pct(price, entry, params.tax_rate):.1f}% net)")

    if params.sell_mode in ("target", "either"):
        if price >= target_price(entry, params):
            return True, (f"hit +{params.target_pct:.0f}% target "
                          f"({_net_pct(price, entry, params.tax_rate):+.1f}% net of tax)")

    return False, ""
