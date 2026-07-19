"""FutNextSource: id extraction, platform mapping, batching, response mapping.

Network is stubbed with httpx.MockTransport, so these run offline.
"""

import httpx
import pytest

from futmarket.collectors.base import SourceError
from futmarket.collectors.futnext_source import (
    FutNextSource, _platform_param, definition_id)
from futmarket.config import WatchlistEntry


def _entry(pid, url, **kw):
    return WatchlistEntry(player_id=pid, name=kw.get("name", pid), url=url,
                          rating=kw.get("rating"), version=kw.get("version", ""))


# ---- pure helpers ---------------------------------------------------------

def test_definition_id_from_card_url():
    assert definition_id(
        "https://www.fut.gg/players/239085-erling-haaland/26-184788461/") == 184788461


def test_definition_id_from_base_url():
    assert definition_id("https://www.fut.gg/players/231747-kylian-mbappe/") == 231747


def test_definition_id_invalid():
    with pytest.raises(SourceError):
        definition_id("https://www.fut.gg/players/")


def test_platform_mapping():
    assert _platform_param("pc") == "pc"
    assert _platform_param("console") == "ps"   # no combined console price
    assert _platform_param("anything-else") == "ps"


# ---- transport-backed behaviour -------------------------------------------

def _source_with(prices: dict[int, int], *, calls: list | None = None) -> FutNextSource:
    """A FutNextSource whose HTTP layer returns `prices` (definitionId -> price).
    Ids not in the map are omitted, exactly like the real endpoint."""
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        ids = [int(x) for x in request.url.params["ids"].split("_")]
        body = [{"definitionId": i, "price": prices[i], "avg": prices[i],
                 "top5Cheapest": [prices[i]], "updatedAt": 1}
                for i in ids if i in prices]
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FutNextSource(client=client)


def test_fetch_prices_batch_maps_by_definition_id():
    src = _source_with({184788461: 210000, 231747: 55000})
    players = [
        _entry("haaland", "https://www.fut.gg/players/239085-haaland/26-184788461/"),
        _entry("mbappe", "https://www.fut.gg/players/231747-mbappe/"),
        _entry("unknown", "https://www.fut.gg/players/999-nobody/26-424242/"),
    ]
    quotes = {q.player_id: q for q in src.fetch_prices(players, "console")}
    assert quotes["haaland"].price == 210000
    assert quotes["mbappe"].price == 55000
    assert "unknown" not in quotes          # omitted id → no quote
    assert quotes["haaland"].source == "futnext"


def test_fetch_prices_chunks_over_50():
    calls: list = []
    prices = {i: i * 10 for i in range(1, 121)}
    src = _source_with(prices, calls=calls)
    players = [_entry(f"p{i}", f"https://www.fut.gg/players/{i}-x/26-{i}/")
               for i in range(1, 121)]
    quotes = src.fetch_prices(players, "pc")
    assert len(quotes) == 120
    assert len(calls) == 3                   # 120 ids / 50 per call -> 3 requests


def test_fetch_price_single_found():
    src = _source_with({184788461: 210000})
    q = src.fetch_price(
        _entry("haaland", "https://www.fut.gg/players/239085-haaland/26-184788461/"),
        "console")
    assert q.price == 210000 and q.player_id == "haaland"


def test_fetch_price_single_missing_raises():
    src = _source_with({})   # endpoint knows nothing
    with pytest.raises(SourceError):
        src.fetch_price(
            _entry("ghost", "https://www.fut.gg/players/1-ghost/26-1/"), "console")


def test_http_error_becomes_source_error():
    def handler(request):
        return httpx.Response(400, json={"errors": ["Invalid input"]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    src = FutNextSource(client=client)
    with pytest.raises(SourceError):
        src.fetch_price(
            _entry("x", "https://www.fut.gg/players/1-x/26-1/"), "console")


def test_registered_in_source_registry():
    from futmarket.collectors import get_source
    assert isinstance(get_source("futnext", None), FutNextSource)
