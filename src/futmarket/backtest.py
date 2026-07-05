"""Phase 2 backtesting framework.

Before any rule is trusted, replay it over a player's historical feature series
and measure what buying/selling on its signals would actually have returned —
net of EA's transfer-market tax. A rule is only worth promoting to live alerts
if it beats a naive baseline (buy-and-hold, and random timing) on the same
series with the same accounting.

Accounting model: single long position, all-in. Start with 1.0 unit of cash.
BUY (when flat) converts cash to the card at that price; SELL (when holding)
converts back to cash minus tax. Equity is marked-to-market every step so the
drawdown reflects paper losses while holding, not just realized trades.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .features import FeatureRow
from .signals import BUY, HOLD, SELL, SignalParams, evaluate
from . import strategy as strat

Decider = Callable[[FeatureRow], str]


@dataclass
class Trade:
    buy_ts: str
    buy_price: int
    sell_ts: str
    sell_price: int
    ret: float  # net-of-tax fractional return on this round trip


@dataclass
class BacktestResult:
    label: str
    n_trades: int
    hit_rate: float          # fraction of round trips that were profitable
    avg_trade_return: float  # mean net return per round trip
    total_return: float      # compounded equity change over the whole series
    max_drawdown: float      # worst peak-to-trough of the equity curve
    ending_equity: float
    trades: list[Trade] = field(default_factory=list)


def run_backtest(features: list[FeatureRow], decide: Decider,
                 tax_rate: float = 0.05, label: str = "candidate") -> BacktestResult:
    cash, units, buy_ctx = 1.0, 0.0, None
    peak, max_dd = 1.0, 0.0
    trades: list[Trade] = []

    for f in features:
        price = f.price
        action = decide(f)

        if action == BUY and units == 0.0 and cash > 0.0:
            units, cash, buy_ctx = cash / price, 0.0, (f.timestamp, price)
        elif action == SELL and units > 0.0:
            proceeds = units * price * (1.0 - tax_rate)
            ret = proceeds / (units * buy_ctx[1]) - 1.0
            trades.append(Trade(buy_ctx[0], buy_ctx[1], f.timestamp, price, ret))
            cash, units, buy_ctx = proceeds, 0.0, None

        equity = cash + units * price
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    ending_equity = cash + (units * features[-1].price if units > 0 else 0.0)
    wins = sum(1 for t in trades if t.ret > 0)
    n = len(trades)
    return BacktestResult(
        label=label,
        n_trades=n,
        hit_rate=(wins / n) if n else 0.0,
        avg_trade_return=(sum(t.ret for t in trades) / n) if n else 0.0,
        total_return=ending_equity - 1.0,
        max_drawdown=max_dd,
        ending_equity=ending_equity,
        trades=trades,
    )


def rule_decider(params: SignalParams) -> Decider:
    """The live rule, as a backtest decider — same code path as Phase 3."""
    return lambda f: evaluate(f, params).signal_type


def run_rebound_backtest(features: list[FeatureRow], series: "pd.Series",
                         params: "strat.StrategyParams",
                         label: str = "rebound") -> BacktestResult:
    """Position-aware backtest of the rebound strategy. Same equity/tax/drawdown
    accounting as run_backtest, but the SELL check needs the entry price (the
    plain Decider only sees the current row), so the loop lives here and calls
    strategy.analyze on the trailing window at each step — exactly what the live
    advisor does, so backtest and live never diverge."""
    tax_rate = params.tax_rate
    cash, units, entry, entry_ctx = 1.0, 0.0, 0.0, None
    peak, max_dd = 1.0, 0.0
    trades: list[Trade] = []

    for f in features:
        price = f.price
        at = pd.Timestamp(f.timestamp)
        view = strat.analyze(series, at, params)

        if units == 0.0 and cash > 0.0 and strat.should_buy(view):
            units, cash, entry, entry_ctx = cash / price, 0.0, float(price), (f.timestamp, price)
        elif units > 0.0:
            sell, _ = strat.should_sell(price, entry, view.floor, view.ceiling, params)
            if sell:
                proceeds = units * price * (1.0 - tax_rate)
                ret = proceeds / (units * entry_ctx[1]) - 1.0
                trades.append(Trade(entry_ctx[0], entry_ctx[1], f.timestamp, price, ret))
                cash, units, entry, entry_ctx = proceeds, 0.0, 0.0, None

        equity = cash + units * price
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    ending_equity = cash + (units * features[-1].price if units > 0 else 0.0)
    wins = sum(1 for t in trades if t.ret > 0)
    n = len(trades)
    return BacktestResult(
        label=label, n_trades=n,
        hit_rate=(wins / n) if n else 0.0,
        avg_trade_return=(sum(t.ret for t in trades) / n) if n else 0.0,
        total_return=ending_equity - 1.0, max_drawdown=max_dd,
        ending_equity=ending_equity, trades=trades)


def evaluate_rebound(features: list[FeatureRow], series: "pd.Series",
                     params: "strat.StrategyParams") -> Comparison:
    """Score the rebound strategy vs the naive baselines on the same series."""
    candidate = run_rebound_backtest(features, series, params, label="rebound")
    return Comparison(
        candidate=candidate,
        baselines=[buy_and_hold(features, params.tax_rate),
                   random_baseline(features, params.tax_rate)],
    )


def buy_and_hold(features: list[FeatureRow], tax_rate: float = 0.05) -> BacktestResult:
    """Buy at the first snapshot, sell at the last. The 'do nothing clever' bar."""
    if len(features) < 2:
        return BacktestResult("buy_and_hold", 0, 0.0, 0.0, 0.0, 0.0, 1.0)
    first, last = features[0], features[-1]
    decide = lambda f: BUY if f is first else (SELL if f is last else HOLD)
    return run_backtest(features, decide, tax_rate, label="buy_and_hold")


def random_baseline(features: list[FeatureRow], tax_rate: float = 0.05,
                    n_runs: int = 50, seed: int = 0) -> BacktestResult:
    """Random buy/sell timing, averaged over many seeds — the 'monkey with a
    coin' bar the candidate must clear. Reports the mean across runs."""
    rng = random.Random(seed)
    totals, dds, hits, avgs, ntr = [], [], [], [], []
    for _ in range(n_runs):
        def decide(f, _r=rng):
            x = _r.random()
            return BUY if x < 0.15 else (SELL if x > 0.85 else HOLD)
        res = run_backtest(features, decide, tax_rate, label="random")
        totals.append(res.total_return)
        dds.append(res.max_drawdown)
        hits.append(res.hit_rate)
        avgs.append(res.avg_trade_return)
        ntr.append(res.n_trades)

    k = len(totals)
    return BacktestResult(
        label="random(mean)",
        n_trades=round(sum(ntr) / k),
        hit_rate=sum(hits) / k,
        avg_trade_return=sum(avgs) / k,
        total_return=sum(totals) / k,
        max_drawdown=sum(dds) / k,
        ending_equity=1.0 + sum(totals) / k,
    )


@dataclass
class Comparison:
    candidate: BacktestResult
    baselines: list[BacktestResult]

    @property
    def beats_baselines(self) -> bool:
        """Promote only if the candidate's total return clears every baseline."""
        return all(self.candidate.total_return > b.total_return for b in self.baselines)


def evaluate_rule(features: list[FeatureRow], params: SignalParams,
                  tax_rate: float = 0.05) -> Comparison:
    candidate = run_backtest(features, rule_decider(params), tax_rate, label="z-score rule")
    return Comparison(
        candidate=candidate,
        baselines=[buy_and_hold(features, tax_rate),
                   random_baseline(features, tax_rate)],
    )
