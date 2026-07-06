"""Backtest P&L math, spot-checked by hand (Phase 2 DoD)."""

import math

import pandas as pd
import pytest

from futmarket import backtest as B
from futmarket.features import FeatureRow
from futmarket.signals import BUY, HOLD, SELL
from futmarket.strategy import StrategyParams


def _f(ts, price, z=None, days=None, ev=None):
    """Minimal FeatureRow; only the fields the rule/backtest read need to be real."""
    return FeatureRow(
        player_id="p", source="test", timestamp=ts, price=price,
        pct_change_1h=None, pct_change_24h=None, pct_change_7d=None,
        rolling_mean_24h=None, rolling_std_24h=None, z_score=z,
        days_to_next_event=days, next_event_type=ev, is_weekend_window=0,
    )


def test_single_trade_return_net_of_tax():
    # buy 100, sell 110, 5% tax -> proceeds 104.5 -> +4.5%
    feats = [_f("t0", 100), _f("t1", 110)]
    decide = lambda f: BUY if f.timestamp == "t0" else SELL
    r = B.run_backtest(feats, decide, tax_rate=0.05)
    assert r.n_trades == 1
    assert r.trades[0].ret == pytest.approx(0.045)
    assert r.total_return == pytest.approx(0.045)
    assert r.hit_rate == 1.0


def test_losing_trade_counts_against_hit_rate():
    feats = [_f("t0", 100), _f("t1", 100)]  # flat, but 5% tax makes it a loss
    decide = lambda f: BUY if f.timestamp == "t0" else SELL
    r = B.run_backtest(feats, decide, tax_rate=0.05)
    assert r.trades[0].ret == pytest.approx(-0.05)
    assert r.hit_rate == 0.0


def test_max_drawdown_marks_to_market_while_holding():
    # buy at 100, price sinks to 60 (paper -40%), recovers to 105 before sell
    feats = [_f("t0", 100), _f("t1", 60), _f("t2", 105)]
    decide = lambda f: BUY if f.timestamp == "t0" else (SELL if f.timestamp == "t2" else HOLD)
    r = B.run_backtest(feats, decide, tax_rate=0.0)
    assert r.max_drawdown == pytest.approx(0.40, abs=1e-9)


def test_no_double_buy_and_open_position_marked_at_end():
    feats = [_f("t0", 100), _f("t1", 120)]  # buy, never sell
    r = B.run_backtest(feats, lambda f: BUY, tax_rate=0.05)
    assert r.n_trades == 0
    assert r.ending_equity == pytest.approx(1.2)  # 1 unit cash -> 0.01 units @120


def _res(total_return, max_drawdown, label="x"):
    return B.BacktestResult(label=label, n_trades=1, hit_rate=1.0,
                            avg_trade_return=total_return, total_return=total_return,
                            max_drawdown=max_drawdown, ending_equity=1 + total_return)


def test_promotion_requires_beating_every_baseline_return():
    cmp = B.Comparison(candidate=_res(0.10, 0.05),
                       baselines=[_res(0.20, 0.30), _res(-0.05, 0.40)])
    assert not cmp.promote()


def test_promotion_rejects_higher_return_bought_with_worse_drawdown():
    # beats both baselines on return, but its drawdown is 3x the worst baseline's
    cmp = B.Comparison(candidate=_res(0.50, 0.90),
                       baselines=[_res(0.20, 0.30), _res(-0.05, 0.25)])
    assert not cmp.promote(1.0)
    assert cmp.promote(4.0)  # a looser dd multiple admits it


def test_promotion_accepts_better_return_and_drawdown():
    cmp = B.Comparison(candidate=_res(0.50, 0.10),
                       baselines=[_res(0.20, 0.30), _res(-0.05, 0.25)])
    assert cmp.promote()
    assert cmp.beats_baselines  # alias with dd_multiple=1.0


def test_rebound_backtest_trades_the_unified_engine():
    # a clean wide sawtooth: decide() should buy troughs and exit on target/
    # resistance, ending positive with at least one round trip
    n, cycles, lo, hi = 300, 8, 40_000, 55_000
    prices = [lo + (hi - lo) * (0.5 - 0.5 * math.cos(i / (n / cycles) * 2 * math.pi))
              for i in range(n)]
    idx = pd.date_range("2026-05-01", periods=n, freq="6h", tz="UTC")
    series = pd.Series([float(p) for p in prices], index=idx)
    feats = [_f(ts.strftime("%Y-%m-%dT%H:%M:%SZ"), int(p))
             for ts, p in series.items()]
    params = StrategyParams(window_days=30, min_bounces=2, buy_zone_pct=4,
                            target_pct=12, stop_pct=0)
    r = B.run_rebound_backtest(feats, series, params)
    assert r.n_trades >= 1
    assert r.total_return > 0
