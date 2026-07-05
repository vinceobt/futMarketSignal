"""Backtest P&L math, spot-checked by hand (Phase 2 DoD)."""

import pytest

from futmarket import backtest as B
from futmarket.features import FeatureRow
from futmarket.signals import BUY, HOLD, SELL, SignalParams


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


def test_rule_decider_matches_live_signal_engine():
    params = SignalParams(buy_z=-1.5, sell_z=1.5, event_guard_days=2)
    decide = B.rule_decider(params)
    assert decide(_f("t", 100, z=-2.0)) == BUY
    assert decide(_f("t", 100, z=2.0)) == SELL
    assert decide(_f("t", 100, z=0.0)) == HOLD
    # imminent SBC suppresses the BUY
    assert decide(_f("t", 100, z=-2.0, days=1, ev="SBC")) == HOLD


def test_candidate_beats_baselines_on_mean_reverting_series():
    # sawtooth: cheap (z<<0) then rich (z>>0), repeated. The z-rule should
    # buy the dips and sell the peaks; buy-and-hold ends flat.
    feats = []
    for i in range(6):
        feats.append(_f(f"lo{i}", 80, z=-2.0))
        feats.append(_f(f"hi{i}", 120, z=2.0))
    params = SignalParams()
    cmp = B.evaluate_rule(feats, params, tax_rate=0.05)
    assert cmp.candidate.n_trades >= 3
    assert cmp.candidate.total_return > 0
    assert cmp.beats_baselines
