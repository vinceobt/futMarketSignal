"""The unified decide() vocabulary: BUY/SELL/HOLD/SKIP actions, the event
guard, confidence, and the alert formatting built on top of them."""

import math

import pandas as pd

from futmarket.alerts import Alert, format_digest, format_realtime
from futmarket.signals import BUY, HOLD, SELL, SKIP
from futmarket.strategy import (EVENT_IMMINENT, NOT_NEAR_FLOOR, StrategyParams,
                                decide, target_price)


def _series(prices, start="2026-05-01", freq="6h"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.Series([float(p) for p in prices], index=idx)


def _sawtooth(cycles=8, n=300, lo=40_000, hi=55_000):
    return [lo + (hi - lo) * (0.5 - 0.5 * math.cos(i / (n / cycles) * 2 * math.pi))
            for i in range(n)]


PARAMS = StrategyParams(window_days=30, min_bounces=2, buy_zone_pct=4,
                        target_pct=12, stop_pct=0)
BUY_SERIES = _series(_sawtooth())  # ends on a trough of a validated range


def test_buy_ready_series_is_a_buy():
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS)
    assert d.action == BUY
    assert 0 < d.confidence <= 1
    assert "floor" in d.detail and "after tax" in d.detail


def test_buy_suppressed_by_imminent_crashing_event():
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS,
               days_to_next_event=1, next_event_type="SBC")
    assert d.action == SKIP
    assert d.codes == (EVENT_IMMINENT,)


def test_event_far_away_does_not_suppress_buy():
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS,
               days_to_next_event=10, next_event_type="SBC")
    assert d.action == BUY


def test_non_crashing_event_does_not_suppress_buy():
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS,
               days_to_next_event=1, next_event_type="PATCH")
    assert d.action == BUY


def test_skip_names_every_failed_gate():
    # at a peak the price is neither near the floor nor statistically cheap
    prices = _sawtooth()
    s = _series(prices)
    peak_i = max(range(250, 300), key=lambda i: prices[i])
    d = decide(s, s.index[peak_i], PARAMS)
    assert d.action == SKIP
    assert NOT_NEAR_FLOOR in d.codes
    assert d.confidence == 0.0


def test_holding_without_trigger_is_hold_not_skip():
    entry = float(BUY_SERIES.iloc[-1])  # just bought at the trough
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS, entry_price=entry)
    assert d.action == HOLD
    assert "holding" in d.detail


def test_holding_sells_at_target_with_full_confidence():
    price = float(BUY_SERIES.iloc[-1])
    # entry far enough below that the current price clears the net-of-tax target
    entry = price / (target_price(1.0, PARAMS) * 1.01)
    d = decide(BUY_SERIES, BUY_SERIES.index[-1], PARAMS, entry_price=entry)
    assert d.action == SELL
    assert d.confidence == 1.0


def test_alert_formatting():
    a = Alert("salah-base", "Mohamed Salah", BUY, 0.72, "undervalued, buy")
    assert "Mohamed Salah" in format_realtime(a) and "72%" in format_realtime(a)
    digest = format_digest([a], "2026-07-05")
    assert "Mohamed Salah" in digest and "BUY" in digest
    assert "nothing actionable" in format_digest([], "2026-07-05")
