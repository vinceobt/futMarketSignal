"""fut.gg bulk price source — the whole market in one shot.

fut.gg ships current prices as two static files on its CDN:

  player-prices-index.v1.<hash>.json   {id0, d:[deltas], ...}   (id mapping, ~stable)
  player-prices-<plat>-dyn.v1.<hash>.json  {p:[prices], s:[...]} (rotates on update)

The index encodes a strictly-sorted list of EA definitionIds (eaId) as
`cumsum([id0] + d)`; the dynamic file's `p[i]` is the price for `ids[i]`. Together
they yield `{eaId: price}` for the entire ~25.6k-card market — verified against
FUTNext to the coin. That's ~500x cheaper than batching per-card price calls.

The dynamic file's URL carries a rotating content hash that isn't in the page
HTML (client JS builds it), so we discover both URLs by intercepting the XHRs on
a fut.gg page load (same patchright trick as momentum_source), then decode in
pure Python. `decode()` is separated out so the parsing is unit-tested offline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Which dynamic file to read per our platform. Console (PS+Xbox share a market
# since FC25) = the PS5 file; PC has its own.
_PLATFORM_FRAGMENT = {"console": "ps5", "pc": "pc"}

_LIST_URL = "https://www.fut.gg/players/"


def decode(index_json: dict, dyn_json: dict) -> dict[int, int]:
    """Reconstruct {definitionId: price} from the two CDN payloads.

    Skips entries with no positive price (0/None = extinct/unlisted)."""
    id0 = index_json["id0"]
    deltas = index_json["d"]
    prices = dyn_json["p"]

    ids = [id0]
    for delta in deltas:
        ids.append(ids[-1] + delta)
    if len(ids) != len(prices):
        logger.warning("bulk price length mismatch: %d ids vs %d prices "
                       "(zipping to the shorter)", len(ids), len(prices))

    out: dict[int, int] = {}
    for ea_id, price in zip(ids, prices):
        if price and price > 0:
            out[ea_id] = int(price)
    return out


def fetch_bulk_prices(platform: str = "console", *, timeout_ms: int = 8000) -> dict[int, int]:
    """Intercept the two CDN price files off a fut.gg page load and decode them.

    Returns {definitionId: price} for the whole market. Raises RuntimeError if the
    files never arrive."""
    from patchright.sync_api import sync_playwright  # heavy import; keep it local

    frag = _PLATFORM_FRAGMENT.get(platform, "ps5")
    captured: dict[str, dict] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def handle_response(response):
            url = response.url
            if "r2.fut.gg" not in url or "player-prices" not in url:
                return
            try:
                if "player-prices-index" in url and "index" not in captured:
                    captured["index"] = response.json()
                    logger.info("captured price index (%s)", url)
                elif f"{frag}-dyn" in url and "dyn" not in captured:
                    captured["dyn"] = response.json()
                    logger.info("captured price dyn (%s)", url)
            except Exception as e:  # noqa: BLE001
                logger.debug("bulk price JSON parse error: %s", e)

        page.on("response", handle_response)
        try:
            page.goto(_LIST_URL, wait_until="load")
            # give both XHRs time to fire
            for _ in range(6):
                if "index" in captured and "dyn" in captured:
                    break
                page.wait_for_timeout(timeout_ms // 6)
        except Exception as e:  # noqa: BLE001
            logger.error("bulk price navigation error: %s", e)
        finally:
            browser.close()

    if "index" not in captured or "dyn" not in captured:
        raise RuntimeError(
            f"bulk price files not captured (got {sorted(captured)}); "
            "fut.gg page layout or CDN scheme may have changed")
    return decode(captured["index"], captured["dyn"])
