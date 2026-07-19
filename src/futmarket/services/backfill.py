"""History backfill — replay the season, liquid cards first.

Walks the registry in liquidity order (tier A>B>C>unscored) and pulls each card's
daily price history since release from fut.gg, appending it to price_snapshots.
Idempotent (UNIQUE(player_id, source, minute)), so re-runs only fill gaps and
newly-released cards. A circuit breaker halts on a run of failures rather than
hammering a source that's rate-limiting us.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

from .. import db
from ..collectors import history_source
from ..collectors.base import SourceError
from .bulk_collect import BULK_SOURCE

logger = logging.getLogger(__name__)


def _game_of(title: str | None) -> str:
    return (title or "fc26").replace("fc", "") or "26"


def backfill_history(conn, *, source: str = BULK_SOURCE,
                     tiers: tuple[str, ...] | None = None,
                     limit: int | None = None, delay: float = 1.0,
                     max_consecutive_failures: int = 5,
                     client: httpx.Client | None = None,
                     progress: Callable[[dict], None] | None = None) -> dict:
    """Backfill daily history for tracked cards, liquid-first. Returns counts."""
    cards = db.cards_for_backfill(conn, tiers=tiers, limit=limit)
    own_client = client is None
    client = client or httpx.Client(timeout=25.0, headers=history_source._HEADERS)
    res = {"cards": 0, "points": 0, "inserted": 0, "failed": 0, "skipped": 0}
    consecutive = 0
    try:
        for card in cards:
            def_id = card["definition_id"]
            if def_id is None:
                res["skipped"] += 1
                continue
            try:
                points = history_source.fetch_history(
                    int(def_id), _game_of(card["title"]), client=client)
                consecutive = 0
            except SourceError as e:
                res["failed"] += 1
                consecutive += 1
                logger.warning("backfill failed for %s: %s", card["player_id"], e)
                if consecutive >= max_consecutive_failures:
                    logger.error("circuit breaker: %d consecutive failures, stopping",
                                 consecutive)
                    break
                continue
            for ts, price in points:
                res["points"] += 1
                if db.insert_snapshot(conn, player_id=card["player_id"],
                                      price=price, source=source, at=ts):
                    res["inserted"] += 1
            res["cards"] += 1
            conn.commit()
            if progress is not None and res["cards"] % 25 == 0:
                progress(res)
            if delay:
                time.sleep(delay)
    finally:
        if own_client:
            client.close()
    conn.commit()
    logger.info("backfill done: %s", res)
    return res
