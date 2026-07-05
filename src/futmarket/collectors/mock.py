"""Deterministic fake source: validates the whole pipeline without any network.

Prices follow a per-player random walk seeded by (player_id, hour bucket), so
repeated calls within the same hour agree, and history looks plausibly noisy.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..config import WatchlistEntry
from .base import PriceQuote


def _noise(key: str) -> float:
    """Stable hash -> [-1.0, 1.0)."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**31 - 1.0


class MockSource:
    name = "mock"

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        now = datetime.now(timezone.utc)
        # Stable per-player base in ~[500, 150000], derived from the id so the
        # pipeline has plausible prices without any rating on the watchlist.
        base = 500 + (abs(_noise(player.player_id)) * 149_500)
        hour_bucket = now.strftime("%Y-%m-%dT%H")
        drift = _noise(f"{player.player_id}:{hour_bucket}") * 0.08
        price = max(200, int(base * (1 + drift)))
        return PriceQuote(player_id=player.player_id, price=price,
                          source=self.name, fetched_at=now)
