"""FUTNext price source — the fast, batched bulk feed for the ML rebuild.

Unlike the fut.gg headless-browser scraper (one Chromium navigation per card),
this hits an open JSON endpoint that returns current lowest-BIN prices for up to
50 cards per request, no auth / no browser. It is the workhorse for polling the
whole card universe; fut.gg stays the source for *historical* backfill (the
FUTNext trend endpoints are auth-gated). See docs/futnext-price-api.md.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from ..config import WatchlistEntry
from .base import PriceQuote, SourceError

PRICES_URL = "https://enhancer-api.futnext.com/players/v2/prices"
MAX_BATCH = 50  # the endpoint's documented cap (extension maxBatchSize)

# definitionId = the numeric id in a fut.gg card URL. Prefer the card-segment id
# (".../26-184788461/" -> 184788461); fall back to the base id for base URLs.
_CARD_SEG_ID = re.compile(r"-(\d+)/?$")
_BASE_ID = re.compile(r"/players/(\d+)-")


def definition_id(url: str) -> int:
    """EA definitionId from a fut.gg player URL (the FUTNext join key)."""
    url = url.strip()
    m = _CARD_SEG_ID.search(url)
    if m:
        return int(m.group(1))
    m = _BASE_ID.search(url)
    if m:
        return int(m.group(1))
    raise SourceError(f"cannot derive definitionId from URL: {url!r}")


def _platform_param(platform: str) -> str:
    """Map our config platform to FUTNext's. There is no combined 'console'
    price — the PlayStation market ('ps') is the console market since FC25."""
    return "pc" if platform == "pc" else "ps"


class FutNextSource:
    """PriceSource adapter over the open FUTNext current-price endpoint."""

    name = "futnext"

    def __init__(self, *, timeout: float = 15.0,
                 client: httpx.Client | None = None) -> None:
        self._timeout = timeout
        self._client = client  # injectable for tests

    # -- batch API (the reason to use this source) --------------------------

    def fetch_prices(self, players: list[WatchlistEntry], platform: str,
                     *, at: datetime | None = None) -> list[PriceQuote]:
        """Current prices for many cards in as few HTTP calls as possible.
        Cards the endpoint doesn't know are silently omitted (no price)."""
        at = at or datetime.now(timezone.utc)
        by_def: dict[int, WatchlistEntry] = {}
        for p in players:
            try:
                by_def[definition_id(p.url)] = p
            except SourceError:
                continue  # skip cards we can't map rather than fail the whole batch

        quotes: list[PriceQuote] = []
        def_ids = list(by_def)
        for i in range(0, len(def_ids), MAX_BATCH):
            chunk = def_ids[i:i + MAX_BATCH]
            for row in self._get(chunk, platform):
                did = row.get("definitionId")
                price = row.get("price")
                entry = by_def.get(did)
                if entry is None or price is None:
                    continue
                quotes.append(PriceQuote(
                    player_id=entry.player_id, price=int(price),
                    source=self.name, fetched_at=at,
                    name=entry.name or None, rating=entry.rating,
                    version=entry.version or None,
                ))
        return quotes

    # -- single-card API (PriceSource protocol) -----------------------------

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        did = definition_id(player.url)
        rows = self._get([did], platform)
        by_id = {r.get("definitionId"): r for r in rows}
        row = by_id.get(did)
        if row is None or row.get("price") is None:
            raise SourceError(f"no FUTNext price for {player.player_id} (id {did})")
        return PriceQuote(
            player_id=player.player_id, price=int(row["price"]),
            source=self.name, fetched_at=datetime.now(timezone.utc),
            name=player.name or None, rating=player.rating,
            version=player.version or None,
        )

    # -- transport ----------------------------------------------------------

    def _get(self, def_ids: list[int], platform: str) -> list[dict]:
        params = {"ids": "_".join(str(d) for d in def_ids),
                  "platform": _platform_param(platform)}
        try:
            if self._client is not None:
                resp = self._client.get(PRICES_URL, params=params)
            else:
                resp = httpx.get(PRICES_URL, params=params, timeout=self._timeout)
        except httpx.HTTPError as e:
            raise SourceError(f"FUTNext request failed: {e}") from e
        if resp.status_code != 200:
            raise SourceError(f"FUTNext HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not isinstance(data, list):
            raise SourceError(f"FUTNext returned non-list payload: {data!r}")
        return data
