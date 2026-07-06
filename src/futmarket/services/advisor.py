"""The rebound advisor — the per-cycle brain.

After each collection pass this analyzes every watchlisted player (plus, if
configured, a few fut.gg momentum movers) with the rebound strategy, manages
one paper position per player, and fires a Discord/console alert only on the
BUY→…→SELL state transitions (never repeat spam while holding).

Decision-support only: it advises and tracks a paper position; it never trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .. import alerts as alertmod
from .. import db, features, strategy
from ..config import Config
from ..features import to_series
from ..signals import BUY, SELL
from . import watch

log = logging.getLogger("futmarket.advisor")


def _series(conn, player_id: str, source: str) -> pd.Series:
    return to_series(db.snapshots(conn, player_id, source))


def _candidates(conn, config: Config, source: str) -> list[tuple[str, str]]:
    """(player_id, name) to analyze this cycle: the watchlist, plus up to
    strategy_momentum_screen_limit fut.gg movers we already have history for.
    Movers with no stored history yet are skipped (they get scraped once a BUY
    auto-adds them to the watchlist, then analyzed next cycle)."""
    out = {e.player_id: e.name for e in watch.effective_entries(conn, config)}
    limit = config.strategy_momentum_screen_limit
    if limit > 0:
        for r in db.momentum_list(conn)[:limit]:
            out.setdefault(r["player_id"], r["name"])
    return list(out.items())


def run(conn, config: Config, source: str, *, dry_run: bool = False,
        alerter=None) -> dict:
    """Analyze, manage positions, alert. Returns a summary dict. dry_run makes it
    read-only: it prints intended actions and neither writes nor sends."""
    params = strategy.StrategyParams.from_config(config)
    if alerter is None and not dry_run:
        alerter = alertmod.get_alerter(config)
    now = datetime.now(timezone.utc)
    opened, closed, holding, screened = [], [], [], 0

    for player_id, name in _candidates(conn, config, source):
        series = _series(conn, player_id, source)
        if series.empty:
            continue
        screened += 1
        at = series.index[-1]
        pos = db.position_get_open(conn, player_id)

        if pos is None:
            days, ev_type = features._next_event(conn, player_id, at)
            decision = strategy.decide(series, at, params,
                                       days_to_next_event=days, next_event_type=ev_type)
            view = decision.view
            price = int(view.price)
            if decision.action == BUY:
                tgt = int(strategy.target_price(price, params))
                stop = strategy.stop_price(view.floor, params)
                stop = int(stop) if stop else None
                msg = alertmod.format_trade_alert("BUY", name, price, view=view, target=tgt)
                opened.append({"player_id": player_id, "name": name, "price": price,
                               "target": tgt, "msg": msg})
                if not dry_run:
                    # keep tracking it (movers may not be in the watchlist yet)
                    try:
                        watch.add(conn, config, _url_for(conn, player_id))
                    except Exception:
                        pass
                    db.position_open(conn, player_id=player_id, entry_price=price,
                                     entry_ts=now, target_price=tgt, stop_price=stop,
                                     floor_price=int(view.floor) if view.floor else None,
                                     reason=decision.detail)
                    _send(alerter, msg)
            # SKIP stays silent: no position, no alert, codes only in the logs.
        else:
            entry = pos["entry_price"]
            decision = strategy.decide(series, at, params, entry_price=entry)
            price = int(decision.view.price)
            if decision.action == SELL:
                why = decision.detail
                realized = (price * (1 - params.tax_rate) / entry - 1.0) * 100.0
                msg = alertmod.format_trade_alert("SELL", name, price,
                                                  realized_pct=realized, reason=why)
                closed.append({"player_id": player_id, "name": name, "price": price,
                               "realized_pct": round(realized, 1), "msg": msg})
                if not dry_run:
                    db.position_close(conn, pos["id"], exit_price=price, exit_ts=now,
                                      realized_pct=realized, reason=why)
                    _send(alerter, msg)
            else:
                holding.append({"player_id": player_id, "name": name, "price": price,
                                "entry": entry, "target": pos["target_price"]})

    summary = {"screened": screened, "opened": opened, "closed": closed,
               "holding": holding, "dry_run": dry_run}
    log.info("advise done screened=%d opened=%d closed=%d holding=%d dry_run=%s",
             screened, len(opened), len(closed), len(holding), dry_run)
    return summary


def _url_for(conn, player_id: str) -> str:
    """The fut.gg URL for a player: from the momentum cache if present, else
    reconstructed from the id (…-26-<card> → /players/<base>/26-<card>/)."""
    row = conn.execute("SELECT url FROM momentum WHERE player_id=?", (player_id,)).fetchone()
    if row and row["url"]:
        return row["url"]
    # id looks like "239085-erling-haaland-26-184788461" → base "/239085-erling-haaland/26-184788461/"
    parts = player_id.rsplit("-", 2)  # [base, "26", "184788461"]
    if len(parts) == 3:
        return f"https://www.fut.gg/players/{parts[0]}/{parts[1]}-{parts[2]}/"
    return f"https://www.fut.gg/players/{player_id}/"


def _send(alerter, text: str) -> None:
    try:
        alerter.send(text)
    except Exception as e:  # never let a delivery failure break the loop
        log.warning("alert delivery failed: %s", e)
