"""Market-wide price collection via the fut.gg bulk CDN files.

One pass snapshots the entire tracked market: fetch {definitionId: price} for all
~25.6k cards in a couple of requests, map each id to a registry card, and append
a price_snapshot. Idempotent (UNIQUE(player_id, source, minute)); source is
tagged 'futgg_bulk' so it's distinguishable from the per-card scraper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import db
from ..collectors import bulk_price_source

logger = logging.getLogger(__name__)

BULK_SOURCE = "futgg_bulk"


def collect_bulk(conn, *, platform: str = "console", source: str = BULK_SOURCE,
                 at: datetime | None = None,
                 prices: dict[int, int] | None = None) -> dict:
    """Append a market-wide price snapshot.

    `prices` ({definitionId: price}) can be injected (tests); otherwise fetched
    live. Only cards present in card_meta are stored. Returns counts."""
    at = at or datetime.now(timezone.utc)
    if prices is None:
        prices = bulk_price_source.fetch_bulk_prices(platform)

    # definitionId -> our player_id, from the registry.
    id_to_player: dict[int, str] = {}
    for card in db.card_registry(conn, tradeable_only=False):
        did = card["definition_id"]
        if did is not None:
            id_to_player[int(did)] = card["player_id"]

    inserted = matched = unknown = 0
    for ea_id, price in prices.items():
        player_id = id_to_player.get(int(ea_id))
        if player_id is None:
            unknown += 1
            continue
        matched += 1
        if db.insert_snapshot(conn, player_id=player_id, price=price,
                              source=source, at=at):
            inserted += 1
    conn.commit()
    result = {"fetched": len(prices), "matched": matched,
              "inserted": inserted, "unknown": unknown}
    logger.info("bulk collect: %s", result)
    return result
