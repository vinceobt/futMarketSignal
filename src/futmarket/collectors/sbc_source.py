"""fut.gg SBC feed — Squad Building Challenges with their live windows.

SBCs are among the strongest market movers: when a challenge demands 84-rated
fodder, those cards get dumped and tank, then rebound when it expires. fut.gg
serves them as plain paginated JSON (`/api/fut/sbc/<game>/`), giving each SBC's
name, start (`createdAt`) and expiry (`endTime`) — exactly the windows the
lifecycle features and event study need.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://www.fut.gg"
_SBC_PATH = "/api/fut/sbc/{game}/"
_UA = "fc-market-analytics/0.1 (personal price tracker)"


def normalize(item: dict) -> dict | None:
    """One SBC -> an event dict, or None if it has no usable start date."""
    created = item.get("createdAt")
    if not isinstance(created, str) or len(created) < 10:
        return None
    end = item.get("endTime")
    name = item.get("name") or item.get("slug") or "SBC"
    category = (item.get("category") or {}).get("name")
    note = f"SBC: {name}" + (f" ({category})" if category else "")
    return {
        "event_type": "SBC",
        "player_id": None,          # market-wide: SBCs move whole fodder cohorts
        "start_date": created[:10],
        "end_date": end[:10] if isinstance(end, str) and len(end) >= 10 else None,
        "notes": note,
    }


def fetch_page(page: int, game: str = "26", *,
               client: httpx.Client | None = None,
               timeout: float = 20.0) -> tuple[list[dict], int | None]:
    url = _BASE + _SBC_PATH.format(game=game)
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


def iter_sbcs(game: str = "26", *, max_pages: int | None = None,
              delay: float = 0.4,
              client: httpx.Client | None = None) -> Iterator[dict]:
    """Yield every SBC as an event dict, following the `next` cursor."""
    page: int | None = 1
    done = 0
    while page is not None:
        rows, nxt = fetch_page(page, game, client=client)
        logger.info("sbc page %s: %d entries (next=%s)", page, len(rows), nxt)
        yield from rows
        done += 1
        if max_pages is not None and done >= max_pages:
            break
        page = nxt
        if page is not None and delay:
            time.sleep(delay)
