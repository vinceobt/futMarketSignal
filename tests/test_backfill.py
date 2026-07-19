"""History source (sign + fetch) and the liquid-first backfill service."""

import httpx
import pytest

from futmarket import db as futdb
from futmarket.collectors import history_source
from futmarket.collectors.base import SourceError
from futmarket.services import backfill


def _history_client(*, history_by_id: dict[int, list], challenge=False,
                    calls: list | None = None) -> httpx.Client:
    """MockTransport mimicking sign (POST) + history (GET) for given cards."""
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if request.url.path.endswith("/price-access/sign/"):
            import json
            body = json.loads(request.content.decode())
            protected = body["url"]                      # /api/fut/player-prices/26/<def>/
            def_id = int(protected.rstrip("/").split("/")[-1])
            return httpx.Response(200, json={"data": {
                "url": f"{protected}?verify=tok-{def_id}",
                "challengeRequired": challenge, "expiresIn": 120}})
        # history GET
        def_id = int(request.url.path.rstrip("/").split("/")[-1])
        hist = history_by_id.get(def_id, [])
        return httpx.Response(200, json={"data": {
            "eaId": def_id,
            "history": [{"date": d, "price": p} for d, p in hist]}})
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---- history_source -------------------------------------------------------

def test_fetch_history_signs_then_fetches():
    client = _history_client(history_by_id={5: [
        ("2026-07-03T00:00:00Z", 3683000), ("2026-07-04T00:00:00Z", 3126000)]})
    pts = history_source.fetch_history(5, "26", client=client)
    assert len(pts) == 2
    ts, price = pts[0]
    assert price == 3683000 and ts.year == 2026 and ts.month == 7 and ts.day == 3


def test_fetch_history_challenge_raises():
    client = _history_client(history_by_id={5: []}, challenge=True)
    with pytest.raises(SourceError):
        history_source.fetch_history(5, "26", client=client)


def test_fetch_history_skips_zero_prices():
    client = _history_client(history_by_id={7: [
        ("2026-07-03T00:00:00Z", 0), ("2026-07-04T00:00:00Z", 5000)]})
    pts = history_source.fetch_history(7, "26", client=client)
    assert [p for _, p in pts] == [5000]


# ---- backfill service -----------------------------------------------------

def _seed(conn):
    futdb.upsert_card_meta(conn, {"player_id": "a-liquid", "definition_id": 101, "rating": 90})
    futdb.upsert_card_meta(conn, {"player_id": "b-mid", "definition_id": 102, "rating": 84})
    futdb.upsert_card_meta(conn, {"player_id": "c-noid", "definition_id": None, "rating": 70})
    futdb.upsert_liquidity(conn, player_id="a-liquid", score=8.0, tier="A")
    futdb.upsert_liquidity(conn, player_id="b-mid", score=4.0, tier="B")
    conn.commit()


def test_backfill_inserts_and_orders_liquid_first(conn):
    _seed(conn)
    calls: list = []
    client = _history_client(history_by_id={
        101: [("2026-07-03T00:00:00Z", 1000000), ("2026-07-04T00:00:00Z", 1100000)],
        102: [("2026-07-03T00:00:00Z", 5000)],
    }, calls=calls)

    res = backfill.backfill_history(conn, client=client, delay=0)
    assert res["cards"] == 2                 # the None-def card is skipped
    assert res["skipped"] == 1
    assert res["inserted"] == 3              # 2 + 1 daily points

    # tier A card must have been signed before tier B card (liquid-first order)
    sign_order = [c for c in calls if c.url.path.endswith("/sign/")]
    first_ids = [int(__import__("json").loads(c.content.decode())["url"].rstrip("/").split("/")[-1])
                 for c in sign_order]
    assert first_ids[0] == 101 and first_ids[1] == 102

    hist = futdb.snapshots(conn, "a-liquid", "futgg")
    assert [r["price"] for r in hist] == [1000000, 1100000]


def test_backfill_is_idempotent(conn):
    _seed(conn)
    client = _history_client(history_by_id={
        101: [("2026-07-03T00:00:00Z", 1000000)], 102: [("2026-07-03T00:00:00Z", 5000)]})
    first = backfill.backfill_history(conn, client=client, delay=0)
    # rebuild a fresh mock client (the first one is consumed lazily but stateless)
    client2 = _history_client(history_by_id={
        101: [("2026-07-03T00:00:00Z", 1000000)], 102: [("2026-07-03T00:00:00Z", 5000)]})
    second = backfill.backfill_history(conn, client=client2, delay=0)
    assert first["inserted"] == 2
    assert second["inserted"] == 0 and second["points"] == 2


def test_backfill_circuit_breaker(conn):
    # every card fails to sign -> breaker stops after N consecutive failures
    for i in range(10):
        futdb.upsert_card_meta(conn, {"player_id": f"p{i}", "definition_id": 200 + i})
    conn.commit()

    def handler(request):
        return httpx.Response(500, text="boom")
    client = httpx.Client(transport=httpx.MockTransport(handler))

    res = backfill.backfill_history(conn, client=client, delay=0,
                                    max_consecutive_failures=3)
    assert res["failed"] == 3 and res["cards"] == 0   # stopped at the breaker
