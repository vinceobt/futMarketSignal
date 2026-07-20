"""The race: model vs the existing rules engine, same cards, same period, same money.

This is the gate that decides whether the model is worth anything. Precision and
lift are lab metrics; what matters is whether trading its signals would have made
more coins than the engine you already have.

Fairness rules enforced here:
  * identical accounting -- the model reuses backtest.py's equity/tax/drawdown
    model (single long position, all-in, 5% sell tax, marked to market).
  * identical window -- every strategy trades the same cards over the same dates.
  * out-of-sample only -- the model's probabilities come from walk-forward folds,
    so it never trades a day it was trained on.
  * the rules engine keeps its full lookback -- it sees price history from before
    the window, exactly as it would live; only the *trading* window is shared.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import db
from ..backtest import (BacktestResult, Trade, buy_and_hold, random_baseline,
                        run_rebound_backtest)
from ..features import compute_feature_table, to_series
from ..strategy import StrategyParams
from . import dataset, labels, train, validation

logger = logging.getLogger(__name__)

BUY_PERCENTILE = 90        # buy when the model is in the top decile of confidence
MIN_WINDOW_ROWS = 30       # a card needs this many test-window days to be raceable


def run_model_backtest(rows: pd.DataFrame, *, threshold: float,
                       target_pct: float, stop_pct: float, tax_rate: float,
                       label: str = "model") -> BacktestResult:
    """Trade one card off model probabilities, with backtest.py's accounting.

    Entry: model confidence at/above `threshold` while flat.
    Exit: the same barriers the labels were built from -- a target grossed up for
    tax, or a stop -- so the backtest honours the economics the model learnt.
    """
    cash, units, entry, entry_ctx = 1.0, 0.0, 0.0, None
    peak, max_dd = 1.0, 0.0
    trades: list[Trade] = []

    target_mult = (1.0 + target_pct / 100.0) / (1.0 - tax_rate)
    stop_mult = 1.0 - stop_pct / 100.0

    for row in rows.itertuples(index=False):
        price = float(row.price)
        if price <= 0:
            continue
        if units == 0.0 and cash > 0.0:
            proba = getattr(row, "proba", np.nan)
            if np.isfinite(proba) and proba >= threshold:
                units, cash = cash / price, 0.0
                entry, entry_ctx = price, (row.date, int(price))
        elif units > 0.0:
            if price >= entry * target_mult or price <= entry * stop_mult:
                proceeds = units * price * (1.0 - tax_rate)
                ret = proceeds / (units * entry_ctx[1]) - 1.0
                trades.append(Trade(entry_ctx[0], entry_ctx[1], row.date,
                                    int(price), ret))
                cash, units, entry, entry_ctx = proceeds, 0.0, 0.0, None

        equity = cash + units * price
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    last_price = float(rows["price"].iloc[-1]) if len(rows) else 0.0
    ending = cash + (units * last_price if units > 0 else 0.0)
    wins = sum(1 for t in trades if t.ret > 0)
    n = len(trades)
    return BacktestResult(
        label=label, n_trades=n, hit_rate=(wins / n) if n else 0.0,
        avg_trade_return=(sum(t.ret for t in trades) / n) if n else 0.0,
        total_return=ending - 1.0, max_drawdown=max_dd,
        ending_equity=ending, trades=trades)


def out_of_sample_probabilities(frame: pd.DataFrame, *, horizon: int,
                                n_splits: int = 3) -> tuple[pd.DataFrame, float]:
    """Walk-forward probabilities for test rows only, plus the buy threshold.

    Each fold trains on its past and scores its future, so no row is ever scored
    by a model that saw it. The threshold is the BUY_PERCENTILE of the *training*
    scores, chosen without looking at the test set.
    """
    data = frame.dropna(subset=["is_profitable"]).copy()
    cols = [c for c in dataset.FEATURE_COLUMNS if c in data.columns]
    data["proba"] = np.nan
    thresholds = []

    for train_idx, test_idx in validation.walk_forward_splits(
            data["date"], n_splits=n_splits, embargo_days=horizon):
        trn, tst = data.iloc[train_idx], data.iloc[test_idx]
        if trn["is_profitable"].nunique() < 2 or tst.empty:
            continue
        model = train._estimator("classifier")
        model.fit(trn[cols], trn["is_profitable"])
        thresholds.append(float(np.percentile(
            model.predict_proba(trn[cols])[:, 1], BUY_PERCENTILE)))
        data.iloc[test_idx, data.columns.get_loc("proba")] = \
            model.predict_proba(tst[cols])[:, 1]

    threshold = float(np.mean(thresholds)) if thresholds else 1.0
    scored = data.dropna(subset=["proba"])
    logger.info("out-of-sample rows scored: %d, buy threshold=%.4f",
                len(scored), threshold)
    return scored, threshold


def race(conn, *, source: str = "futgg", title: str = "fc26", horizon: int = 7,
         target_pct: float = 25.0, stop_pct: float = 8.0, tax_rate: float = 0.05,
         n_splits: int = 3, max_cards: int = 250,
         params: StrategyParams | None = None) -> dict:
    """Run every strategy over the same cards and window; return the scoreboard."""
    frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return {"error": "no data"}
    frame = labels.add_labels(frame, horizon_days=horizon, target_pct=target_pct,
                              stop_pct=stop_pct, tax_rate=tax_rate)
    if "liq_tier" in frame.columns:
        liquid = frame[frame["liq_tier"].isin(("A", "B"))]
        frame = liquid if not liquid.empty else frame

    scored, threshold = out_of_sample_probabilities(
        frame, horizon=horizon, n_splits=n_splits)
    if scored.empty:
        return {"error": "no out-of-sample rows"}

    counts = scored.groupby("player_id", observed=True).size()
    eligible = counts[counts >= MIN_WINDOW_ROWS].sort_values(ascending=False)
    card_ids = list(eligible.index[:max_cards])
    logger.info("racing %d cards", len(card_ids))

    params = params or StrategyParams(tax_rate=tax_rate, target_pct=target_pct,
                                      stop_pct=stop_pct)
    tally: dict[str, list[BacktestResult]] = {
        "model": [], "rules_engine": [], "buy_and_hold": [], "random": []}
    paired: list[tuple[float, float]] = []   # (model, rules_engine) per card
    raced = 0

    for player_id in card_ids:
        rows = (scored[scored["player_id"] == player_id]
                .sort_values("date").reset_index(drop=True))
        if len(rows) < MIN_WINDOW_ROWS:
            continue
        window_start, window_end = rows["date"].iloc[0], rows["date"].iloc[-1]

        model_result = run_model_backtest(
            rows, threshold=threshold, target_pct=target_pct,
            stop_pct=stop_pct, tax_rate=tax_rate)
        tally["model"].append(model_result)

        # The rules engine gets its full lookback but only trades the same window.
        snaps = db.snapshots(conn, player_id, source)
        series = to_series([{"timestamp": r["timestamp"], "price": r["price"]}
                            for r in snaps])
        feats = [f for f in compute_feature_table(conn, player_id, source)
                 if window_start <= f.timestamp[:10] <= window_end]
        if len(feats) >= 2:
            rules_result = run_rebound_backtest(feats, series, params,
                                                label="rules_engine")
            tally["rules_engine"].append(rules_result)
            tally["buy_and_hold"].append(buy_and_hold(feats, tax_rate))
            tally["random"].append(random_baseline(feats, tax_rate, n_runs=20))
            paired.append((model_result.total_return, rules_result.total_return))
        raced += 1

    def summarise(name: str) -> dict:
        """Robust statistics only.

        The inherited backtest accounting is all-in and compounds every trade, so
        a card traded often enough produces mathematically real but practically
        impossible totals (50 trades at +20% compounds to 9,100x). The *mean*
        compounded return is therefore meaningless and kept only as a diagnostic.
        Judge strategies on the median card, the per-trade edge, and how often
        they actually made money.
        """
        results = tally[name]
        if not results:
            return {"cards": 0}
        traded = [r for r in results if r.n_trades > 0]
        returns = np.array([r.total_return for r in results])
        return {
            "cards": len(results),
            "cards_traded": len(traded),
            "median_return_pct": round(float(np.median(returns)) * 100, 3),
            "pct_cards_profitable": round(float(np.mean(returns > 0)), 4),
            # per-trade return does not compound -> the honest measure of edge
            "mean_trade_return_pct": round(
                float(np.mean([r.avg_trade_return for r in traded])) * 100, 3) if traded else 0.0,
            "total_trades": int(sum(r.n_trades for r in results)),
            "hit_rate": round(float(np.mean([r.hit_rate for r in traded])), 4) if traded else 0.0,
            "mean_max_drawdown": round(float(np.mean([r.max_drawdown for r in results])), 4),
            "_mean_return_pct_unreliable": round(float(np.mean(returns)) * 100, 1),
        }

    scoreboard = {name: summarise(name) for name in tally}
    # Compare on the median card, which no single blow-up can distort.
    model_med = scoreboard["model"].get("median_return_pct", 0)
    beat = {name: bool(model_med > scoreboard[name].get("median_return_pct", 0))
            for name in ("rules_engine", "buy_and_hold", "random")
            if scoreboard[name].get("cards")}

    # Paired head-to-head: the fairest read, since both strategies traded the
    # same card over the same days.
    head_to_head = {}
    if paired:
        arr = np.array(paired)
        wins = int((arr[:, 0] > arr[:, 1]).sum())
        ties = int((arr[:, 0] == arr[:, 1]).sum())
        head_to_head = {
            "cards": len(paired),
            "model_wins": wins,
            "rules_wins": len(paired) - wins - ties,
            "ties": ties,
            "model_win_rate": round(wins / len(paired), 4),
        }

    return {
        "cards_raced": raced,
        "window": {"start": scored["date"].min(), "end": scored["date"].max()},
        "buy_threshold": round(threshold, 4),
        "scoreboard": scoreboard,
        "head_to_head": head_to_head,
        "model_beats": beat,
        "model_wins_all": bool(beat) and all(beat.values()),
    }
