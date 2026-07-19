"""Game calendar: promo/TOTW derivation from card releases, SBC feed, persistence."""

import httpx

from futmarket import db as futdb
from futmarket.collectors import sbc_source
from futmarket.services import calendar


def _card(conn, pid, version, release, rating=84):
    futdb.upsert_card_meta(conn, {"player_id": pid, "definition_id": abs(hash(pid)) % 10**6,
                                  "version": version, "release_date": release,
                                  "rating": rating})


# ---- derivation from card releases ---------------------------------------

def test_base_versions_are_not_events(conn):
    for i in range(5):
        _card(conn, f"c{i}", "Common", "2025-09-08")
        _card(conn, f"r{i}", "Rare", "2025-09-08")
    conn.commit()
    assert calendar.derive_card_release_events(conn) == []


def test_promo_derived_from_first_card_appearance(conn):
    _card(conn, "p1", "Festival of Football: Phenoms", "2026-07-09")
    _card(conn, "p2", "Festival of Football: Phenoms", "2026-07-09")
    _card(conn, "p3", "Festival of Football: Phenoms", "2026-07-11")
    conn.commit()
    events = calendar.derive_card_release_events(conn)
    assert len(events) == 1
    e = events[0]
    assert e["event_type"] == "PROMO"
    assert e["start_date"] == "2026-07-09"      # first appearance = launch
    assert e["end_date"] == "2026-07-11"        # last card of the drop
    assert "3 cards" in e["notes"]


def test_totw_split_into_weekly_events(conn):
    # one version name, three weekly batches -> three separate TOTW events
    for week, day in enumerate(("2025-09-17", "2025-09-24", "2025-10-01")):
        for i in range(4):
            _card(conn, f"totw{week}-{i}", "Team of the Week", day)
    conn.commit()
    events = calendar.derive_card_release_events(conn)
    assert [e["event_type"] for e in events] == ["TOTW"] * 3
    assert [e["start_date"] for e in events] == ["2025-09-17", "2025-09-24", "2025-10-01"]


def test_tiny_versions_ignored_as_noise(conn):
    _card(conn, "x1", "One Off Thing", "2026-01-01")   # only 1 card (< MIN)
    conn.commit()
    assert calendar.derive_card_release_events(conn) == []


def test_events_sorted_by_date(conn):
    for i in range(3):
        _card(conn, f"late{i}", "Late Promo", "2026-05-01")
        _card(conn, f"early{i}", "Early Promo", "2025-11-01")
    conn.commit()
    events = calendar.derive_card_release_events(conn)
    assert [e["start_date"] for e in events] == ["2025-11-01", "2026-05-01"]


# ---- launch anchor --------------------------------------------------------

def test_launch_anchored_to_earliest_release(conn):
    _card(conn, "a", "Common", "2025-09-08")
    _card(conn, "b", "Icon", "2025-09-12")
    conn.commit()
    assert calendar.set_launch_from_registry(conn) == "2025-09-08"
    assert futdb.game_launch(conn, "fc26") == "2025-09-08"


# ---- SBC feed -------------------------------------------------------------

def test_sbc_normalize():
    ev = sbc_source.normalize({
        "name": "Enzo Fernández", "createdAt": "2026-07-18T17:00:28Z",
        "endTime": "2026-07-25T17:00:01Z", "category": {"name": "Players"}})
    assert ev["event_type"] == "SBC"
    assert ev["start_date"] == "2026-07-18" and ev["end_date"] == "2026-07-25"
    assert "Enzo" in ev["notes"] and "Players" in ev["notes"]


def test_sbc_normalize_requires_start():
    assert sbc_source.normalize({"name": "No date"}) is None


def test_sbc_pagination():
    pages = {
        1: {"data": [{"name": "A", "createdAt": "2026-07-01T00:00:00Z"}], "next": 2},
        2: {"data": [{"name": "B", "createdAt": "2026-07-02T00:00:00Z"}], "next": None},
    }

    def handler(request):
        return httpx.Response(200, json=pages[int(request.url.params.get("page", "1"))])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    got = list(sbc_source.iter_sbcs(client=client, delay=0))
    assert [e["notes"] for e in got] == ["SBC: A", "SBC: B"]


# ---- persistence / idempotency -------------------------------------------

def test_build_calendar_persists_and_is_idempotent(conn):
    for i in range(4):
        _card(conn, f"p{i}", "Winter Wildcards", "2025-12-01")
    conn.commit()

    def handler(request):
        return httpx.Response(200, json={"data": [
            {"name": "Fodder SBC", "createdAt": "2026-01-05T00:00:00Z",
             "endTime": "2026-01-12T00:00:00Z"}], "next": None})
    client = httpx.Client(transport=httpx.MockTransport(handler))

    first = calendar.build_calendar(conn, sbc_client=client, delay=0,
                                    include_news=False)
    assert first["promo"] == 1 and first["sbc"] == 1 and first["launch"] == "2025-12-01"

    client2 = httpx.Client(transport=httpx.MockTransport(handler))
    calendar.build_calendar(conn, sbc_client=client2, delay=0, include_news=False)
    # rebuild replaces rather than duplicating
    assert len(futdb.events_list(conn)) == 2


def test_rebuild_preserves_manual_events(conn):
    for i in range(4):
        _card(conn, f"p{i}", "Winter Wildcards", "2025-12-01")
    futdb.insert_event(conn, event_type="PATCH", start_date="2026-02-01",
                       notes="hand-logged gameplay patch")
    conn.commit()

    calendar.build_calendar(conn, include_sbc=False, include_news=False)
    types = {r["event_type"] for r in futdb.events_list(conn)}
    assert "PATCH" in types and "PROMO" in types   # manual row survived the rebuild
