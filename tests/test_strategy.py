"""The unified engine: reliable-bouncer detection, falling-knife rejection, the
SKIP gates, and the buy/sell triggers — spot-checked against constructed series."""

import math

import pandas as pd
import pytest

from futmarket.signals import BUY, SKIP
from futmarket.strategy import (FLAT_MARKET, FLOOR_BROKEN, INSUFFICIENT_POINTS,
                                MARGIN_TOO_THIN, SPAN_TOO_SHORT, STALE,
                                TOO_FEW_BOUNCES, StrategyParams,
                                _count_touches, _floor_is_drifting, analyze,
                                decide, should_buy, should_sell, target_price)


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
    d = decide(s, s.index[-1], PARAMS)
    assert d.action == SKIP and INSUFFICIENT_POINTS in d.codes


# ---- explicit SKIP gates ----

def test_flat_market_is_skipped():
    s = _series([5_000.0] * 40)
    d = decide(s, s.index[-1], PARAMS)
    assert d.action == SKIP and FLAT_MARKET in d.codes


def test_short_span_is_skipped_even_with_many_points():
    s = _series(_sawtooth(n=48), freq="30min")   # 48 pts but only ~1 day of history
    d = decide(s, s.index[-1], PARAMS)
    assert d.action == SKIP and SPAN_TOO_SHORT in d.codes


def test_stale_data_is_skipped():
    s = _series(_sawtooth())
    d = decide(s, s.index[-1] + pd.Timedelta(hours=72), PARAMS)
    assert d.action == SKIP and STALE in d.codes


def test_thin_margin_is_skipped_even_at_the_floor():
    # a tidy range whose whole span nets < 1% after the 5% tax: never worth it
    cycle = [10_000, 10_150, 10_300, 10_450, 10_600, 10_450, 10_300, 10_150]
    s = _series(cycle * 8)   # 64 pts * 6h = 16 days
    d = decide(s, s.index[-1], PARAMS)
    assert d.action == SKIP and MARGIN_TOO_THIN in d.codes
    assert d.view.net_margin_pct < 1.0


def test_price_through_the_stop_level_is_never_a_buy():
    # a valid rebounder whose price then knifes BELOW the floor: entering here
    # would trip the stop immediately, so the engine must refuse the entry
    p = StrategyParams(window_days=30, min_bounces=2, buy_zone_pct=4,
                       target_pct=12, stop_pct=8)
    prices = _sawtooth() + [33_000]      # floor ≈ 40.8k, stop level ≈ 37.6k
    s = _series(prices)
    d = decide(s, s.index[-1], p)
    assert d.action == SKIP and FLOOR_BROKEN in d.codes


def test_current_crash_does_not_drag_the_floor_down():
    # the floor is judged on how long prices HELD, and the point at `at` holds
    # for 0s — so a fresh crash can't soften its own entry test
    prices = [5_000, 5_100, 5_200] * 13 + [3_000]
    s = _series(prices)
    d = decide(s, s.index[-1], PARAMS)
    assert d.view.floor == 5_000            # not dragged toward 3,000
    assert d.action == SKIP                 # a one-way slide, never a rebounder
    assert TOO_FEW_BOUNCES in d.codes


# ---- primitive spot-checks ----

def test_touch_hysteresis_counts_distinct_visits_once():
    # floor 100, tol 3%: touch band <=103, re-arm at >=106
    assert _count_touches([102, 104, 102, 107, 101], 100.0, 3.0) == 2
    # wiggling inside the band never re-arms: one touch
    assert _count_touches([101, 103, 102, 103, 101], 100.0, 3.0) == 1
    assert _count_touches([120, 118, 125], 100.0, 3.0) == 0


def test_theil_sen_drift_detects_stairs_but_forgives_one_outlier():
    p = StrategyParams(window_days=30, floor_drift_pct=10)
    stairs = [(0.0, 1_000.0), (5.0, 900.0), (10.0, 800.0), (15.0, 700.0)]
    assert _floor_is_drifting(stairs, 700.0, p)
    # one bad low among stable ones must NOT read as a downtrend
    outlier = [(0.0, 1_000.0), (5.0, 1_000.0), (10.0, 600.0), (15.0, 1_000.0)]
    assert not _floor_is_drifting(outlier, 1_000.0, p)


def test_sell_mode_either_takes_any_trigger_but_never_a_net_loss():
    p = StrategyParams(sell_mode="either", target_pct=25, stop_pct=8,
                       exit_band_pct=2, tax_rate=0.05)
    entry, floor = 1_000, 950
    # target: 1000*1.25/0.95 = 1315.8
    assert should_sell(1_320, entry, floor, 2_000, p)[0]
    # resistance band: ceiling 1200 -> fires from 1176, profitable net of tax
    ok, why = should_sell(1_180, entry, floor, 1_200, p)
    assert ok and "resistance" in why
    # same price but entry too high: net loss -> resistance refuses to sell
    assert should_sell(1_180, 1_150, floor, 1_200, p)[0] is False
    # stop: floor*0.92 = 874
    ok, why = should_sell(870, entry, floor, 1_200, p)
    assert ok and ("stop" in why.lower() or "broke" in why.lower())
