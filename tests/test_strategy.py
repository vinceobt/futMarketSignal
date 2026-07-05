"""Rebound strategy: reliable-bouncer detection, falling-knife rejection, and the
buy/sell triggers — spot-checked against constructed price series."""

import math

import pandas as pd
import pytest

from futmarket.strategy import (StrategyParams, analyze, should_buy, should_sell,
                                target_price)


def _series(prices, start="2026-05-01", freq="6h"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.Series([float(p) for p in prices], index=idx)


def _sawtooth(cycles=8, n=300, lo=40_000, hi=55_000):
    return [lo + (hi - lo) * (0.5 - 0.5 * math.cos(i / (n / cycles) * 2 * math.pi))
            for i in range(n)]


PARAMS = StrategyParams(window_days=30, min_bounces=2, buy_zone_pct=4,
                        target_pct=12, stop_pct=0)


def test_reliable_rebounder_in_zone_is_a_buy():
    s = _series(_sawtooth())
    v = analyze(s, s.index[-1], PARAMS)   # last point is a trough
    assert v.is_reliable and v.in_buy_zone
    assert v.bounces >= PARAMS.min_bounces
    assert should_buy(v)


def test_rebounder_at_peak_is_not_a_buy():
    prices = _sawtooth(n=300)
    s = _series(prices)
    # find an index near a peak (max of the last window)
    peak_i = max(range(250, 300), key=lambda i: prices[i])
    v = analyze(s, s.index[peak_i], PARAMS)
    assert v.is_reliable          # still a known bouncer…
    assert not v.in_buy_zone      # …but not near the floor now
    assert not should_buy(v)


def test_monotonic_crash_is_rejected():
    # a straight slide never rebounds → no bounces → not a buy
    s = _series([60_000 - 150 * i for i in range(300)])
    v = analyze(s, s.index[-1], PARAMS)
    assert not v.is_reliable and not should_buy(v)


def test_noisy_downtrend_lower_lows_is_rejected():
    # bounces exist, but each trough is lower than the last → falling knife
    prices, lvl = [], 5000.0
    for _ in range(12):
        prices += [lvl, lvl * 1.12, lvl * 0.95]
        lvl *= 0.85
    s = _series(prices)
    v = analyze(s, s.index[-1], PARAMS)
    assert not v.is_reliable and not should_buy(v)


def test_crash_then_oscillate_is_a_buy():
    # the user's case: launch high, crash, THEN settle into a range and sit at the floor
    prices = [10_000 + (3_000 - 10_000) * (i / 27) for i in range(28)]  # crash
    for _ in range(5):
        prices += [3_000, 2_400, 1_600, 1_200, 1_500, 2_200, 2_800, 2_500, 1_800, 1_300]
    prices.append(1_320)  # currently near the established floor
    s = _series(prices)
    v = analyze(s, s.index[-1], StrategyParams(window_days=30, min_bounces=3,
                                               buy_zone_pct=8, target_pct=12, stop_pct=0))
    assert v.is_reliable and v.in_buy_zone and should_buy(v)  # launch crash ignored


def test_sell_fires_at_net_of_tax_target():
    p = StrategyParams(sell_mode="target", target_pct=12, tax_rate=0.05, stop_pct=0)
    entry = 40_000
    tgt = target_price(entry, p)          # gross price that nets +12% after tax
    assert should_sell(tgt - 1, entry, 38_000, 55_000, p)[0] is False
    ok, why = should_sell(tgt + 1, entry, 38_000, 55_000, p)
    assert ok and "target" in why


def test_sell_resistance_mode_rides_to_ceiling():
    p = StrategyParams(sell_mode="resistance", tax_rate=0.05, stop_pct=0)
    entry, ceiling = 1_200, 2_800
    # below the ceiling → hold, even though it's already very profitable
    assert should_sell(2_500, entry, 1_100, ceiling, p)[0] is False
    ok, why = should_sell(2_850, entry, 1_100, ceiling, p)
    assert ok and "resistance" in why


def test_stop_fires_below_floor():
    p = StrategyParams(target_pct=25, tax_rate=0.05, stop_pct=8)
    entry, floor = 40_000, 39_000
    stop = floor * (1 - 0.08)
    ok, why = should_sell(stop - 1, entry, floor, 55_000, p)
    assert ok and ("stop" in why.lower() or "broke" in why.lower())


def test_thin_history_holds():
    s = _series([40_000, 41_000, 40_500])   # only 3 points
    v = analyze(s, s.index[-1], PARAMS)
    assert not v.is_reliable
    assert not should_buy(v)
