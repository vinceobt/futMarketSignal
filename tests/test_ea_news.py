"""EA official news collector: blob parsing, classification, pagination."""

import json

import httpx
import pytest

from futmarket.collectors import ea_news_source as ea


def _page_html(items, total):
    blob = {"props": {"pageProps": {"newsDataFallback": {
        "items": items, "totalItems": total}}}}
    return ('<html><body><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(blob) + "</script></body></html>")


def _article(title, date, slug="slug-x"):
    return {"title": title, "publishingDate": date, "slug": slug}


# ---- parsing / classification --------------------------------------------

def test_extract_next_data_missing_blob_raises():
    with pytest.raises(ValueError):
        ea._extract_next_data("<html>no blob here</html>")


def test_classify_promo_patch_news():
    assert ea.classify("Football Ultimate Team™ 26 - Phenoms") == "PROMO"
    assert ea.classify("Football Ultimate Team™ 26 - Team of the Season") == "PROMO"
    assert ea.classify("EA SPORTS FC 26 Title Update 5") == "PATCH"
    assert ea.classify("EA SPORTS FC 26 - 2026 Apple TV Offer") == "NEWS"


def test_normalize_maps_article():
    ev = ea.normalize(_article("Football Ultimate Team™ 26 - Glory Hunters",
                               "2026-06-26T15:00:00Z", "fc-26-glory-hunters"))
    assert ev["event_type"] == "PROMO"
    assert ev["start_date"] == "2026-06-26"
    assert ev["end_date"] is None            # announcements are points in time
    assert "Glory Hunters" in ev["notes"] and "fc-26-glory-hunters" in ev["notes"]


def test_normalize_requires_date_and_title():
    assert ea.normalize({"title": "No date"}) is None
    assert ea.normalize({"publishingDate": "2026-01-01T00:00:00Z"}) is None


# ---- pagination -----------------------------------------------------------

def _paged_client(pages: dict[int, tuple[list, int]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        items, total = pages[page]
        return httpx.Response(200, text=_page_html(items, total),
                              headers={"content-type": "text/html"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_iter_news_pages_until_total_reached():
    pages = {
        1: ([_article("Ultimate Team A", "2026-07-01T00:00:00Z")], 3),
        2: ([_article("Ultimate Team B", "2026-06-01T00:00:00Z")], 3),
        3: ([_article("Ultimate Team C", "2026-05-01T00:00:00Z")], 3),
    }
    got = list(ea.iter_news(client=_paged_client(pages), delay=0))
    assert [e["start_date"] for e in got] == ["2026-07-01", "2026-06-01", "2026-05-01"]


def test_iter_news_respects_max_pages():
    pages = {
        1: ([_article("Ultimate Team A", "2026-07-01T00:00:00Z")], 10),
        2: ([_article("Ultimate Team B", "2026-06-01T00:00:00Z")], 10),
    }
    got = list(ea.iter_news(client=_paged_client(pages), delay=0, max_pages=1))
    assert len(got) == 1


def test_iter_news_stops_on_empty_page():
    pages = {1: ([_article("Ultimate Team A", "2026-07-01T00:00:00Z")], 99),
             2: ([], 99)}
    got = list(ea.iter_news(client=_paged_client(pages), delay=0))
    assert len(got) == 1


def test_calendar_includes_news(conn):
    from futmarket import db as futdb
    from futmarket.services import calendar
    futdb.upsert_card_meta(conn, {"player_id": "p1", "version": "Winter Wildcards",
                                  "release_date": "2025-12-01"})
    for i in range(3):
        futdb.upsert_card_meta(conn, {"player_id": f"w{i}", "version": "Winter Wildcards",
                                      "release_date": "2025-12-01"})
    conn.commit()
    pages = {1: ([_article("Ultimate Team - Winter Wildcards", "2025-11-30T00:00:00Z")], 1)}
    res = calendar.build_calendar(conn, include_sbc=False, delay=0,
                                  news_client=_paged_client(pages))
    assert res["news"] == 1
    notes = [r["notes"] for r in futdb.events_list(conn)]
    assert any("EA:" in (n or "") for n in notes)
