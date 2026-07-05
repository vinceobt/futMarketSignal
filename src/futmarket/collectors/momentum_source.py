"""fut.gg momentum scanner — market-wide mover discovery.

Unlike the per-player price scraper, this reads fut.gg's *momentum* feed: a
ranked list of special cards by how hard their price is moving right now
(`momentumPercentage`). It's the discovery layer — the market tells you what to
look at, then you Track a card and the normal per-player pipeline takes over.

Reuses the same patchright navigate + response-interception trick as
`turnstile_source.fetch_price`: drive a headless page to the momentum URL and
grab the JSON the page fetches for itself.

Note on direction: the feed carries a movement *magnitude* (`momentumPercentage`)
but no clean up/down flag (`accelerateType` is the AcceleRATE gameplay trait, not
price direction), so this is presented as "top movers", not gainers-vs-losers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from patchright.sync_api import sync_playwright

from ..config import ConfigError, parse_player_url

logger = logging.getLogger(__name__)

MOMENTUM_URL = "https://www.fut.gg/players/momentum/?onlySpecialCards=true"
_BASE = "https://www.fut.gg"


@dataclass(frozen=True)
class MomentumRow:
    player_id: str
    name: str
    url: str
    price: int
    momentum: float
    rating: int | None
    position: str
    rarity: str


def _row_from_item(item: dict) -> MomentumRow | None:
    """Turn one momentum API item into a MomentumRow, or None if it isn't a
    tradeable priced special card we can also add to a watchlist."""
    if not item.get("hasPrice") or item.get("isSbc") or item.get("isObjective"):
        return None
    if item.get("isEvolutionPlayerItem"):
        return None  # evolutions aren't bought/sold on the market
    path = item.get("url")
    if not path:
        slug, base = item.get("slug"), item.get("basePlayerSlug")
        if not (slug and base):
            return None
        path = f"/players/{base}/{slug}/"
    try:
        player_id, name_fallback = parse_player_url(_BASE + path)
    except ConfigError:
        return None
    price = item.get("currentDbPrice") or item.get("price")
    momentum = item.get("momentumPercentage")
    if price is None or momentum is None:
        return None
    name = item.get("commonName") or (
        f"{item.get('firstName', '')} {item.get('lastName', '')}".strip()) or name_fallback
    return MomentumRow(
        player_id=player_id,
        name=name,
        url=_BASE + path,
        price=int(price),
        momentum=round(float(momentum), 1),
        rating=item.get("overall"),
        position=item.get("position") or "",
        rarity=item.get("rarityName") or "",
    )


def fetch_momentum(limit: int = 30) -> list[MomentumRow]:
    """Return the top `limit` special-card movers, highest momentum first.

    Raises SourceError-style RuntimeError on total failure so a job can report it.
    """
    collected: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def handle_response(response):
            if "players/v2/momentum" in response.url:
                try:
                    data = response.json()
                    if isinstance(data.get("data"), list):
                        collected.extend(data["data"])
                        logger.info("intercepted %d momentum items from %s",
                                    len(data["data"]), response.url)
                except Exception as e:  # noqa: BLE001
                    logger.debug("momentum JSON parse error: %s", e)

        page.on("response", handle_response)
        try:
            page.goto(MOMENTUM_URL, wait_until="load")
            page.wait_for_timeout(6000)  # let the momentum XHR fire
        except Exception as e:  # noqa: BLE001
            logger.error("momentum navigation error: %s", e)
        finally:
            browser.close()

    rows = [r for r in (_row_from_item(it) for it in collected) if r is not None]
    rows.sort(key=lambda r: r.momentum, reverse=True)
    return rows[:limit]
