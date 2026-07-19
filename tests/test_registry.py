"""Card registry crawler: normalization, pagination, and persistence to card_meta."""

import httpx

from futmarket import db as futdb
from futmarket.collectors import card_list_source
from futmarket.services import registry


def _raw(ea_id, name, url, *, rating=84, position="ST", rarity="Gold",
         has_price=True, is_sbc=False, is_obj=False, is_evo=False,
         league="Prem", nation="England", club="Arsenal", game="26",
         created="2025-09-26T12:00:00.000000Z"):
    return {
        "eaId": ea_id, "url": url, "game": game, "overall": rating,
        "position": position, "rarityName": rarity, "commonName": name,
        "hasPrice": has_price, "isSbc": is_sbc, "isObjective": is_obj,
        "isEvolutionPlayerItem": is_evo, "createdAt": created,
        "league": {"name": league}, "nation": {"name": nation}, "club": {"name": club},
    }


def test_normalize_maps_fields():
    row = card_list_source.normalize(_raw(
        184788461, "Erling Haaland",
        "/players/239085-erling-haaland/26-184788461/",
        rating=91, position="ST", rarity="TOTW", league="Prem", nation="Norway",
        club="Man City", created="2025-10-02T09:30:00.000000Z"))
    assert row["definition_id"] == 184788461
    assert row["player_id"] == "239085-erling-haaland-26-184788461"
    assert row["title"] == "fc26"
    assert row["rating"] == 91 and row["position"] == "ST"
    assert row["league"] == "Prem" and row["nation"] == "Norway" and row["club"] == "Man City"
    assert row["version"] == "TOTW"
    assert row["release_date"] == "2025-10-02"   # date part of createdAt
    assert row["tradeable"] == 1


def test_normalize_tradeable_flag():
    base_url = "/players/1-x/26-1/"
    assert card_list_source.normalize(_raw(1, "X", base_url, is_sbc=True))["tradeable"] == 0
    assert card_list_source.normalize(_raw(1, "X", base_url, is_obj=True))["tradeable"] == 0
    assert card_list_source.normalize(_raw(1, "X", base_url, is_evo=True))["tradeable"] == 0
    assert card_list_source.normalize(_raw(1, "X", base_url, has_price=False))["tradeable"] == 0
    assert card_list_source.normalize(_raw(1, "X", base_url))["tradeable"] == 1


def test_normalize_rejects_urlless_item():
    assert card_list_source.normalize({"eaId": 1, "overall": 84}) is None


def _paged_client(pages: dict[int, dict]) -> httpx.Client:
    """MockTransport serving {page_number: json_payload}."""
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(200, json=pages[page])
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_iter_cards_follows_next_cursor():
    pages = {
        1: {"data": [_raw(1, "A", "/players/1-a/26-1/")], "next": 2,
            "currentPage": 1, "total": 3},
        2: {"data": [_raw(2, "B", "/players/2-b/26-2/")], "next": 3,
            "currentPage": 2, "total": 3},
        3: {"data": [_raw(3, "C", "/players/3-c/26-3/")], "next": None,
            "currentPage": 3, "total": 3},
    }
    client = _paged_client(pages)
    got = list(card_list_source.iter_cards(client=client, delay=0))
    assert [r["definition_id"] for r in got] == [1, 2, 3]


def test_iter_cards_respects_max_pages():
    pages = {
        1: {"data": [_raw(1, "A", "/players/1-a/26-1/")], "next": 2},
        2: {"data": [_raw(2, "B", "/players/2-b/26-2/")], "next": 3},
    }
    client = _paged_client(pages)
    got = list(card_list_source.iter_cards(client=client, delay=0, max_pages=1))
    assert [r["definition_id"] for r in got] == [1]


def test_refresh_registry_persists(conn, monkeypatch):
    pages = {
        1: {"data": [_raw(10, "Meta", "/players/10-meta/26-10/", rating=90),
                     _raw(11, "Fodder", "/players/11-fodder/26-11/", is_sbc=True)],
            "next": None},
    }
    client = _paged_client(pages)
    # iter_cards default client is None (real network); inject the mock instead.
    orig = card_list_source.iter_cards
    monkeypatch.setattr(card_list_source, "iter_cards",
                        lambda *a, **k: orig(*a, **{**k, "client": client}))

    res = registry.refresh_registry(conn, delay=0)
    assert res == {"seen": 2, "tradeable": 1}
    assert futdb.card_count(conn) == 2
    assert futdb.card_count(conn, tradeable_only=True) == 1
    meta = futdb.card_meta_get(conn, "10-meta-26-10")
    assert meta["rating"] == 90 and meta["definition_id"] == 10
