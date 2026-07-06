"""Analytics services: backtest and signal computation, returned as plain
JSON-able dicts so both the CLI and the web job runner can use them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import watch
from .. import backtest, db, features, strategy
from ..config import Config
from ..signals import BUY, HOLD, SELL, SKIP


def evaluate_player(conn, config: Config, source: str,
                    player_id: str) -> strategy.Decision | None:
    """Run the unified engine on one player's stored series: entry price of any
    open paper position (so holders get SELL/HOLD) and upcoming-event context
    are looked up here so decide() itself stays pure. None if no data."""
    series = features.to_series(db.snapshots(conn, player_id, source))
    if series.empty:
        return None
    at = series.index[-1]
    pos = db.position_get_open(conn, player_id)
    days, ev_type = features._next_event(conn, player_id, at)
    return strategy.decide(series, at, strategy.StrategyParams.from_config(config),
                           entry_price=(pos["entry_price"] if pos else None),
                           days_to_next_event=days, next_event_type=ev_type)


def _result_dict(r) -> dict:
    return {
        "label": r.label,
        "n_trades": r.n_trades,
        "hit_rate": round(r.hit_rate, 4),
        "avg_trade_return": round(r.avg_trade_return, 4),
        "total_return": round(r.total_return, 4),
        "max_drawdown": round(r.max_drawdown, 4),
    }


def run_backtest(conn, config: Config, source: str) -> dict:
    """Backtest the unified engine vs baselines for every watchlist player."""
    params = strategy.StrategyParams.from_config(config)
    rows = []
    for entry in watch.effective_entries(conn, config):
        table = features.compute_feature_table(conn, entry.player_id, source)
        if len(table) < 2:
            continue
        series = features.to_series(db.snapshots(conn, entry.player_id, source))
        cmp = backtest.evaluate_rebound(table, series, params)
        rows.append({
            "player_id": entry.player_id,
            "name": entry.name,
            "snapshots": len(table),
            "promote": cmp.promote(config.backtest_max_dd_multiple),
            "candidate": _result_dict(cmp.candidate),
            "baselines": [_result_dict(b) for b in cmp.baselines],
        })
    return {"tax_rate": config.tax_rate, "players": rows,
            "promoted": sum(1 for r in rows if r["promote"])}


def run_signals(conn, config: Config, source: str) -> dict:
    """Evaluate + persist the current BUY/SELL/HOLD/SKIP decision for each
    player. SKIPs are stored too — the audit trail of why the engine passed."""
    now = datetime.now(timezone.utc)
    out, counts = [], {BUY: 0, SELL: 0, HOLD: 0, SKIP: 0}
    for entry in watch.effective_entries(conn, config):
        decision = evaluate_player(conn, config, source, entry.player_id)
        if decision is None:
            continue
        counts[decision.action] += 1
        reason = decision.detail
        if decision.action == SKIP and decision.codes:
            reason = f"[{','.join(decision.codes)}] {reason}"
        db.insert_signal(conn, player_id=entry.player_id, signal_type=decision.action,
                         confidence=decision.confidence, reason=reason, at=now)
        out.append({"player_id": entry.player_id, "name": entry.name,
                    "type": decision.action, "confidence": round(decision.confidence, 3),
                    "reason": reason})
    conn.commit()
    return {"buys": counts[BUY], "sells": counts[SELL], "holds": counts[HOLD],
            "skips": counts[SKIP], "signals": out}
