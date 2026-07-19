"""History backfill — replay the season, liquid cards first.

Walks the registry in liquidity order (tier A>B>C>unscored) and pulls each card's
daily price history since release from fut.gg, appending it to price_snapshots.
Idempotent (UNIQUE(player_id, source, minute)), so re-runs only fill gaps and
newly-released cards. A circuit breaker halts on a run of failures rather than
hammering a source that's rate-limiting us.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

import httpx

from .. import db
from ..collectors import history_source
from ..collectors.base import SourceError
from ..collectors.history_source import RateLimited
from .bulk_collect import BULK_SOURCE

logger = logging.getLogger(__name__)


def _game_of(title: str | None) -> str:
    return (title or "fc26").replace("fc", "") or "26"


def _fetch_with_backoff(def_id: int, game: str, client: httpx.Client, *,
                        retries: int = 4, base: float = 5.0,
                        sleep=time.sleep) -> list[tuple]:
    """fetch_history, but ride out 429s with exponential backoff + jitter.
    Only a persistent 429 (retries exhausted) or a non-429 error propagates."""
    for attempt in range(retries + 1):
        try:
            return history_source.fetch_history(def_id, game, client=client)
        except RateLimited:
            if attempt == retries:
                raise
            wait = base * (2 ** attempt) + random.uniform(0, 1.0)
            logger.warning("rate-limited on %s; backing off %.1fs (attempt %d/%d)",
                           def_id, wait, attempt + 1, retries)
            sleep(wait)


def backfill_history(conn, *, source: str = BULK_SOURCE,
                     tiers: tuple[str, ...] | None = None,
                     limit: int | None = None, delay: float = 1.0,
                     max_consecutive_failures: int = 5,
                     skip_existing: bool = True, min_existing: int = 10,
                     client: httpx.Client | None = None,
                     progress: Callable[[dict], None] | None = None) -> dict:
    """Backfill daily history for tracked cards, liquid-first. Returns counts.

    `skip_existing` makes re-runs resumable: a card already holding >= min_existing
    snapshots for this source has been backfilled (spot-only cards have just a few),
    so we skip it rather than re-request and risk a needless 429."""
    # When resuming, `limit` should bound cards we actually *fetch*, so pull the
    # ordered list and skip-then-count rather than truncating up front.
    cards = db.cards_for_backfill(conn, tiers=tiers,
                                  limit=None if skip_existing else limit)
    own_client = client is None
    client = client or httpx.Client(timeout=25.0, headers=history_source._HEADERS)
    res = {"cards": 0, "points": 0, "inserted": 0, "failed": 0,
           "skipped": 0, "already": 0}
    consecutive = 0
    try:
        for card in cards:
            if limit is not None and res["cards"] >= limit:
                break
            def_id = card["definition_id"]
            if def_id is None:
                res["skipped"] += 1
                continue
            if skip_existing and db.snapshot_count(
                    conn, card["player_id"], source) >= min_existing:
                res["already"] += 1
                continue
            try:
                points = _fetch_with_backoff(
                    int(def_id), _game_of(card["title"]), client)
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
