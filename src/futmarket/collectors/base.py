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


class PriceSource(Protocol):
    name: str

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        """Return the current lowest BIN for a player. Raise SourceError on failure."""
        ...
