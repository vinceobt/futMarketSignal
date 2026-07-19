"""Card registry service — crawl fut.gg's card list into card_meta.

The registry is the card universe every later stage draws on: the liquidity
scorer ranks it, the collector polls it (tier by tier), and the feature builder
groups it into cohorts. Rebuilt idempotently (upsert per card), so re-running
just refreshes prices/ratings and folds in newly-released cards.
"""

from __future__ import annotations

import logging
from typing import Callable

from .. import db
from ..collectors import card_list_source

logger = logging.getLogger(__name__)


def refresh_registry(conn, *, game: str = "26", max_pages: int | None = None,
                     delay: float = 0.5,
                     progress: Callable[[int, int], None] | None = None) -> dict:
    """Crawl the card list and upsert every card into card_meta.

    Returns {"seen": N, "tradeable": M}. Commits in batches so a long crawl is
    crash-safe and partially useful."""
    seen = tradeable = 0
    for row in card_list_source.iter_cards(game, max_pages=max_pages, delay=delay):
        db.upsert_card_meta(conn, row)
        seen += 1
        tradeable += int(bool(row.get("tradeable")))
        if seen % 200 == 0:
            conn.commit()
            if progress is not None:
                progress(seen, tradeable)
    conn.commit()
    logger.info("registry refresh done: %d cards (%d tradeable)", seen, tradeable)
    return {"seen": seen, "tradeable": tradeable}
