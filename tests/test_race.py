"""Race accounting: the model must trade under exactly the engine's economics."""

import numpy as np
import pandas as pd

from futmarket.ml import race


def _rows(prices, probas):
    return pd.DataFrame({
        "date": [f"2026-01-{i+1:02d}" for i in range(len(prices))],
        "price": prices, "proba": probas})


KW = dict(threshold=0.5, target_pct=25.0, stop_pct=8.0, tax_rate=0.05)


def test_no_trade_when_never_confident():
    res = race.run_model_backtest(_rows([100, 110, 120], [0.1, 0.2, 0.3]), **KW)
    assert res.n_trades == 0
    assert res.total_return == 0.0        # stayed in cash


def test_buys_on_confidence_and_sells_at_target():
    # target multiple = 1.25/0.95 = 1.3158 -> needs >= ~131.6 to exit
    res = race.run_model_backtest(_rows([100, 105, 140], [0.9, 0.1, 0.1]), **KW)
    assert res.n_trades == 1
    trade = res.trades[0]
    assert trade.buy_price == 100 and trade.sell_price == 140
    # net of 5% tax: 140*0.95/100 - 1 = +33%
    assert round(trade.ret, 6) == round(140 * 0.95 / 100 - 1, 6)


def test_sells_at_stop():
    res = race.run_model_backtest(_rows([100, 90, 200], [0.9, 0.1, 0.1]), **KW)
    assert res.n_trades == 1
    assert res.trades[0].sell_price == 90          # stop at 92 tripped first
    assert res.trades[0].ret < 0


def test_tax_is_charged_on_every_exit():
    """A round trip at an unchanged price must LOSE the tax, never break even."""
    res = race.run_model_backtest(_rows([100, 132, 132], [0.9, 0.1, 0.1]), **KW)
    assert res.n_trades == 1
    gross = 132 / 100 - 1
    assert res.trades[0].ret < gross                # tax drag applied


def test_holds_until_a_barrier_is_hit():
    prices = [100, 101, 102, 103, 104]              # never reaches either barrier
    res = race.run_model_backtest(_rows(prices, [0.9] + [0.1] * 4), **KW)
    assert res.n_trades == 0                        # still holding at the end
    assert res.ending_equity > 0                    # marked to market, not lost


def test_no_reentry_while_holding():
    """Repeated confidence must not stack positions -- single position, all-in."""
    res = race.run_model_backtest(_rows([100, 101, 140], [0.9, 0.9, 0.9]), **KW)
    assert res.n_trades == 1


def test_drawdown_tracks_paper_losses():
    res = race.run_model_backtest(_rows([100, 60, 61], [0.9, 0.1, 0.1]), **KW)
    assert res.max_drawdown > 0.3                   # felt the dip while holding


def test_nan_probability_never_buys():
    res = race.run_model_backtest(_rows([100, 140], [np.nan, np.nan]), **KW)
    assert res.n_trades == 0


def test_equity_matches_cash_after_a_closed_trade():
    res = race.run_model_backtest(_rows([100, 140, 141], [0.9, 0.1, 0.1]), **KW)
    assert res.n_trades == 1
    # after exiting, equity is pure cash and equals the trade's net proceeds
    assert round(res.ending_equity, 6) == round(140 * 0.95 / 100, 6)
