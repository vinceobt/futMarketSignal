"""Momentum discovery scan — find *new* range-bound cards to start tracking.

Separate from the every-cycle advisor (which handles buy/sell on cards already
tracked). This periodically refreshes fut.gg's momentum movers, scrapes the
recent history of any not yet on the watchlist, and auto-adds the ones that match
the rebound pattern — so the advisor then watches them for a buy at their floor.

Discovery changes on the scale of days, not minutes, so this is meant to run
infrequently (every few hours), not every cycle.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone

from .. import db, strategy
from ..config import Config, ConfigError, WatchlistEntry
from ..collectors import momentum_source
from ..collectors.base import SourceError
from ..collectors.turnstile_source import TurnstileMockSource
from ..features import to_series
from ..signals import BUY
from . import watch

log = logging.getLogger("futmarket.scanner")


def scan(conn, config: Config, source: str, *, add: bool = True,
         alerter=None, sleep=time.sleep) -> dict:
    """Refresh momentum, screen untracked movers for the rebound pattern, and
    (if add) auto-track the reliable ones. Returns a summary."""
    params = strategy.StrategyParams.from_config(config)
    limit = config.strategy_momentum_screen_limit

    movers = momentum_source.fetch_momentum(limit=config.momentum_limit)
    if movers:
        db.momentum_replace(conn, movers)
        db.meta_set(conn, "momentum_updated_at",
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        conn.commit()

    tracked = {e.player_id for e in watch.effective_entries(conn, config)}
    src = TurnstileMockSource()
    lo, hi = config.inter_player_delay_seconds

    scanned, found, added = 0, [], []
    for m in movers:
        if m.player_id in tracked or scanned >= limit:
            continue
        scanned += 1
        if scanned > 1:
            sleep(random.uniform(lo, hi))  # politeness between fetches
        try:
            quote = src.fetch_price(WatchlistEntry(player_id=m.player_id, name=m.name,
                                                   url=m.url), config.platform)
        except SourceError as exc:
            log.info("scan skip %s: %s", m.name, exc)
            continue
        for at, price in quote.history:
            db.insert_snapshot(conn, player_id=m.player_id, price=price, source=source, at=at)
        if quote.price:
            db.insert_snapshot(conn, player_id=m.player_id, price=quote.price,
                               source=source, at=quote.fetched_at)
        conn.commit()

        series = to_series(db.snapshots(conn, m.player_id, source))
        if series.empty:
            continue
        decision = strategy.decide(series, series.index[-1], params)
        view = decision.view
        if view.is_reliable:
            found.append(m.name)
            # Only auto-track cards that are actionable NOW — a full BUY decision
            # (validated floor, cheap z, real net margin) — to keep the watchlist
            # lean. A reliable card sitting above its floor is left untracked; a
            # later scan picks it up when it dips into the zone.
            if add and decision.action == BUY and db.watchlist_count(conn) < config.max_watchlist_size:
                try:
                    watch.add(conn, config, m.url)
                    added.append(m.name)
                except ConfigError:
                    pass

    if add and added and alerter is not None:
        alerter.send("📡 discovery: now tracking " + str(len(added)) +
                     " new range-bound card(s): " + ", ".join(added))
    summary = {"scanned": scanned, "reliable": found, "added": added}
    log.info("scan done scanned=%d reliable=%d added=%d", scanned, len(found), len(added))
    return summary
