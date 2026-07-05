"""Source adapter contract. One adapter per site/API, swappable via config.

Adding a real adapter later (licensed API, or a site that has granted
permission): implement PriceSource, register it in collectors/__init__.py,
add its name to config.VALID_SOURCES. The scheduler and CLI need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..config import WatchlistEntry


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class PriceQuote:
    player_id: str
    price: int
    source: str
    fetched_at: datetime
    # Optional metadata a source may enrich by reading the player's page. When a
    # source can't determine these it leaves them None and the scheduler falls
    # back to the URL-derived values on the WatchlistEntry.
    name: str | None = None
    rating: int | None = None
    version: str | None = None
    # The card's full market price history as (timestamp, price) points, oldest
    # first — everything the source could read off the player's profile in one
    # shot (daily since release, hourly for the recent day). Empty for sources
    # that only expose a single current price. The scheduler backfills all of
    # these; UNIQUE(player_id, source, minute) makes re-fetches idempotent.
    history: tuple[tuple[datetime, int], ...] = ()


class PriceSource(Protocol):
    name: str

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        """Return the current lowest BIN for a player. Raise SourceError on failure."""
        ...
