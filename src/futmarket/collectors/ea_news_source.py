"""EA official news — authoritative announcements for the event calendar.

EA's FC news index is a Next.js page whose article list is embedded in the
`__NEXT_DATA__` JSON blob, reachable with a plain GET and paginated by `?page=N`.
(There is also a `/_next/data/<buildId>/...json` route, but buildId changes on
every EA deploy, so parsing the page blob is the stable path.)

This complements the calendar derived from card releases:
  derived_cards -> when a promo's cards actually hit the market
  ea_news       -> the official announcement, its real name, plus non-card events
                   (title updates, Season rollovers) that card drops never reveal
Cross-checked live: EA's announcement dates match the derived card-drop dates
(Glory Hunters, Greats of the Game, Path to Glory all matched exactly).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)

NEWS_URL = "https://www.ea.com/games/ea-sports-fc/fc-26/news"
_BASE = "https://www.ea.com"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
}
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

# Title keywords -> event type. Order matters: patches before promos.
_PATCH_MARKERS = ("title update", "patch notes", "pitch notes")
_PROMO_MARKERS = ("ultimate team", "fut ", "team of the", "heroes", "icons",
                  "festival of football", "season")


def _extract_next_data(html: str) -> dict:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("EA news: __NEXT_DATA__ blob not found (page layout changed?)")
    return json.loads(m.group(1))


def classify(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _PATCH_MARKERS):
        return "PATCH"
    if any(k in t for k in _PROMO_MARKERS):
        return "PROMO"
    return "NEWS"


def normalize(item: dict) -> dict | None:
    """One EA article -> an event dict, or None without a usable date/title."""
    date = item.get("publishingDate")
    title = item.get("title")
    if not (isinstance(date, str) and len(date) >= 10 and title):
        return None
    slug = item.get("slug")
    return {
        "event_type": classify(title),
        "player_id": None,          # EA announcements are market-wide
        "start_date": date[:10],
        "end_date": None,           # announcements are points in time
        "notes": f"EA: {title}" + (f" [{slug}]" if slug else ""),
    }


def fetch_page(page: int = 1, *, client: httpx.Client | None = None,
               timeout: float = 25.0) -> tuple[list[dict], int]:
    """One page of normalized events plus the total article count."""
    params = {"page": page} if page > 1 else None
    if client is not None:
        resp = client.get(NEWS_URL, params=params)
    else:
        resp = httpx.get(NEWS_URL, params=params, timeout=timeout,
                         headers=_HEADERS, follow_redirects=True)
    resp.raise_for_status()
    blob = _extract_next_data(resp.text)
    news = blob.get("props", {}).get("pageProps", {}).get("newsDataFallback", {})
    items = news.get("items") or []
    events = [e for e in (normalize(it) for it in items) if e]
    return events, int(news.get("totalItems") or len(items))


def iter_news(*, max_pages: int | None = None, delay: float = 0.5,
              client: httpx.Client | None = None) -> Iterator[dict]:
    """Yield every EA news item as an event dict, paging until exhausted."""
    page = 1
    seen = 0
    total = None
    while True:
        events, total_items = fetch_page(page, client=client)
        total = total_items if total is None else total
        if not events:
            break
        logger.info("ea news page %d: %d items (total %s)", page, len(events), total)
        yield from events
        seen += len(events)
        page += 1
        if max_pages is not None and page > max_pages:
            break
        if total is not None and seen >= total:
            break
        if delay:
            time.sleep(delay)
