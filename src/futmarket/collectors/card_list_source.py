"""fut.gg card-list source — enumerates the whole card universe.

fut.gg exposes its player database as a clean paginated JSON API
(`/api/fut/players/v2/<game>/?page=N`) reachable with a plain HTTP GET — no
browser, no Turnstile. Each item carries the metadata cohorts are built from
(rating/position/league/nation/club/rarity) plus the EA `eaId` (= definitionId,
the FUTNext join key). This is the "know every card that exists" layer feeding
the card_meta registry.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import httpx

from ..config import ConfigError, parse_player_url

logger = logging.getLogger(__name__)

_BASE = "https://www.fut.gg"
_LIST_PATH = "/api/fut/players/v2/{game}/"
# Identify honestly; the endpoint is public but be a good citizen.
_UA = "fc-market-analytics/0.1 (card registry; https://www.fut.gg)"


def _name_of(obj) -> str | None:
    """league/nation/club arrive as nested objects; we keep the display name."""
    return obj.get("name") if isinstance(obj, dict) else None


def normalize(item: dict) -> dict | None:
    """Map one raw API item to a card_meta row dict, or None if unusable.

    `tradeable` is a market card with a price that isn't SBC/objective/evolution
    (those never trade on the transfer market)."""
    path = item.get("url")
    if not path:
        return None
    try:
        player_id, name_fallback = parse_player_url(_BASE + path)
    except ConfigError:
        return None

    game = str(item.get("game") or "26")
    tradeable = (bool(item.get("hasPrice"))
                 and not item.get("isSbc")
                 and not item.get("isObjective")
                 and not item.get("isEvolutionPlayerItem"))
    created = item.get("createdAt")
    release_date = created[:10] if isinstance(created, str) and len(created) >= 10 else None
    name = (item.get("commonName")
            or f"{item.get('firstName', '')} {item.get('lastName', '')}".strip()
            or name_fallback)

    return {
        "player_id": player_id,
        "definition_id": item.get("eaId"),
        "title": f"fc{game}",
        "name": name,
        "rating": item.get("overall"),
        "position": item.get("position") or "",
        "league": _name_of(item.get("league")),
        "nation": _name_of(item.get("nation")),
        "club": _name_of(item.get("club")),
        "version": item.get("rarityName") or "",
        "release_date": release_date,
        "tradeable": 1 if tradeable else 0,
        "url": _BASE + path,
    }


def fetch_page(page: int, game: str = "26", *,
               client: httpx.Client | None = None,
               timeout: float = 20.0) -> tuple[list[dict], int | None]:
    """One page of normalized card rows plus the next page number (None at end)."""
    url = _BASE + _LIST_PATH.format(game=game)
    params = {"page": page}
    if client is not None:
        resp = client.get(url, params=params)
    else:
        resp = httpx.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": _UA})
    resp.raise_for_status()
    data = resp.json()
    rows = [r for r in (normalize(it) for it in data.get("data", [])) if r]
    return rows, data.get("next")


def iter_cards(game: str = "26", *, max_pages: int | None = None,
               delay: float = 0.5,
               client: httpx.Client | None = None) -> Iterator[dict]:
    """Yield every card in the game, following the API's `next` cursor.

    `delay` throttles between pages (politeness); `max_pages` bounds the crawl
    (handy for tests / partial refreshes)."""
    page: int | None = 1
    pages_done = 0
    while page is not None:
        rows, nxt = fetch_page(page, game, client=client)
        logger.info("card list page %s: %d cards (next=%s)", page, len(rows), nxt)
        yield from rows
        pages_done += 1
        if max_pages is not None and pages_done >= max_pages:
            break
        page = nxt
        if page is not None and delay:
            time.sleep(delay)
