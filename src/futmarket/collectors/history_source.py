"""fut.gg per-card price history — plain HTTP, no browser.

The card price-history endpoint is gated behind a short-lived signed token, but
the signing endpoint mints one over a plain JSON POST with no Turnstile challenge
(`challengeRequired: false`), so the whole flow is two `httpx` calls:

  POST /api/fut/price-access/sign/  {"url": "/api/fut/player-prices/26/<def>/"}
    -> {"data": {"url": "...?verify=<token>", "challengeRequired": false}}
  GET  <signed url>
    -> {"data": {"history": [{date, price}], "completedAuctions": [...], ...}}

`history` is the daily price series since the card released — the "replay the
season" data. `completedAuctions` (real sold price/time) is a true liquidity
signal we can fold in later. This is far faster than the headless-browser scraper
and is the backfill workhorse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import SourceError

logger = logging.getLogger(__name__)


class RateLimited(SourceError):
    """HTTP 429 — the source is throttling us; back off and retry."""

_BASE = "https://www.fut.gg"
_SIGN_PATH = "/api/fut/price-access/sign/"
_HISTORY_PATH = "/api/fut/player-prices/{game}/{def_id}/"
_HEADERS = {
    "User-Agent": "fc-market-analytics/0.1 (personal price tracker)",
    "Referer": "https://www.fut.gg/",
    "Origin": "https://www.fut.gg",
}


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _sign(def_id: int, game: str, client: httpx.Client) -> str:
    """Return the signed relative URL for a card's history, or raise SourceError."""
    protected = _HISTORY_PATH.format(game=game, def_id=def_id)
    resp = client.post(_BASE + _SIGN_PATH, json={"url": protected}, headers=_HEADERS)
    if resp.status_code == 429:
        raise RateLimited(f"sign 429 for {def_id}")
    if resp.status_code != 200:
        raise SourceError(f"sign HTTP {resp.status_code} for {def_id}: {resp.text[:120]}")
    data = resp.json().get("data", {})
    if data.get("challengeRequired"):
        raise SourceError(f"challenge required for {def_id} (needs browser fallback)")
    signed = data.get("url")
    if not signed:
        raise SourceError(f"sign returned no url for {def_id}")
    return signed


def fetch_card_detail(def_id: int, game: str = "26", *,
                      client: httpx.Client | None = None,
                      timeout: float = 25.0) -> dict:
    """The card's full price payload: daily history plus real completed sales.

    `completedAuctions` is what the card *actually changed hands for* — far more
    honest than the lowest listing, which is often a mispriced snipe that never
    represents the going rate. Returns:
      history  [(utc_datetime, price)] oldest first
      sales    [(utc_datetime, sold_price)] most recent ~100 real transactions
      current  the current lowest listed price (may be None)
    """
    own = client is None
    client = client or httpx.Client(timeout=timeout, headers=_HEADERS)
    try:
        signed = _sign(def_id, game, client)
        resp = client.get(_BASE + signed, headers=_HEADERS)
        if resp.status_code == 429:
            raise RateLimited(f"history 429 for {def_id}")
        if resp.status_code != 200:
            raise SourceError(f"history HTTP {resp.status_code} for {def_id}")
        data = resp.json().get("data", {})

        history: list[tuple[datetime, int]] = []
        for h in data.get("history", []):
            price, date = h.get("price"), h.get("date")
            if price and date:
                history.append((_parse_ts(date), int(price)))

        sales: list[tuple[datetime, int]] = []
        for a in data.get("completedAuctions", []):
            price, date = a.get("soldPrice"), a.get("soldDate")
            if price and date:
                sales.append((_parse_ts(date), int(price)))
        sales.sort()

        current = data.get("currentPrice")
        if isinstance(current, dict):
            current = current.get("price")
        return {"history": history, "sales": sales, "current": current}
    except httpx.HTTPError as e:
        raise SourceError(f"history request failed for {def_id}: {e}") from e
    finally:
        if own:
            client.close()


def fetch_history(def_id: int, game: str = "26", *,
                  client: httpx.Client | None = None,
                  timeout: float = 25.0) -> list[tuple[datetime, int]]:
    """Daily price history since release as (utc_datetime, price), oldest first."""
    return fetch_card_detail(def_id, game, client=client, timeout=timeout)["history"]
