"""Analytics services: backtest and signal computation, returned as plain
JSON-able dicts so both the CLI and the web job runner can use them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import watch
from .. import backtest, db, features
from ..config import Config
from ..signals import BUY, HOLD, SELL, SignalParams, evaluate


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
    """Backtest the signal rule vs baselines for every watchlist player."""
    params = SignalParams.from_config(config)
    rows = []
    for entry in watch.effective_entries(conn, config):
        table = features.compute_feature_table(conn, entry.player_id, source)
        if len(table) < 2:
            continue
        cmp = backtest.evaluate_rule(table, params, config.tax_rate)
        rows.append({
            "player_id": entry.player_id,
            "name": entry.name,
            "snapshots": len(table),
            "promote": cmp.beats_baselines,
            "candidate": _result_dict(cmp.candidate),
            "baselines": [_result_dict(b) for b in cmp.baselines],
        })
    return {"tax_rate": config.tax_rate, "players": rows,
            "promoted": sum(1 for r in rows if r["promote"])}


def run_signals(conn, config: Config, source: str) -> dict:
    """Evaluate + persist the current BUY/SELL/HOLD signal for each player."""
    params = SignalParams.from_config(config)
    now = datetime.now(timezone.utc)
    out, counts = [], {BUY: 0, SELL: 0, HOLD: 0}
    for entry in watch.effective_entries(conn, config):
        table = features.compute_feature_table(conn, entry.player_id, source)
        if not table:
            continue
        sig = evaluate(table[-1], params)
        counts[sig.signal_type] += 1
        db.insert_signal(conn, player_id=entry.player_id, signal_type=sig.signal_type,
                         confidence=sig.confidence, reason=sig.reason, at=now)
        out.append({"player_id": entry.player_id, "name": entry.name,
                    "type": sig.signal_type, "confidence": round(sig.confidence, 3),
                    "reason": sig.reason})
    conn.commit()
    return {"buys": counts[BUY], "sells": counts[SELL], "holds": counts[HOLD],
            "signals": out}
