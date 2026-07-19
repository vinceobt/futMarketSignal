"""Bulk price source: delta decode + market-wide persistence."""

from datetime import datetime, timezone

from futmarket import db as futdb
from futmarket.collectors import bulk_price_source
from futmarket.services import bulk_collect

UTC = timezone.utc
NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


# ---- decode (pure) --------------------------------------------------------

def test_decode_reconstructs_ids_and_prices():
    index = {"id0": 100, "d": [5, 10]}          # ids -> 100, 105, 115
    dyn = {"p": [200, 0, 300]}                   # 105 has price 0 -> dropped
    assert bulk_price_source.decode(index, dyn) == {100: 200, 115: 300}


def test_decode_matches_known_shape():
    # deltas like the real file: strictly increasing eaIds
    index = {"id0": 27, "d": [14, 10, 189]}
    dyn = {"p": [16000, 157000, 68500, 97000]}
    got = bulk_price_source.decode(index, dyn)
    assert got == {27: 16000, 41: 157000, 51: 68500, 240: 97000}


def test_decode_tolerates_length_mismatch():
    index = {"id0": 1, "d": [1, 1]}              # 3 ids
    dyn = {"p": [10, 20]}                         # only 2 prices
    assert bulk_price_source.decode(index, dyn) == {1: 10, 2: 20}


# ---- persistence ----------------------------------------------------------

def _seed_registry(conn):
    futdb.upsert_card_meta(conn, {"player_id": "haaland", "definition_id": 184788461})
    futdb.upsert_card_meta(conn, {"player_id": "mbappe", "definition_id": 231747})
    conn.commit()


def test_collect_bulk_maps_and_inserts(conn):
    _seed_registry(conn)
    prices = {184788461: 210000, 231747: 55000, 999999: 12345}  # last not in registry
    res = bulk_collect.collect_bulk(conn, prices=prices, at=NOW)
    assert res == {"fetched": 3, "matched": 2, "inserted": 2, "unknown": 1}

    hist = futdb.snapshots(conn, "haaland", "futgg_bulk")
    assert len(hist) == 1 and hist[0]["price"] == 210000


def test_collect_bulk_is_idempotent(conn):
    _seed_registry(conn)
    prices = {184788461: 210000}
    first = bulk_collect.collect_bulk(conn, prices=prices, at=NOW)
    second = bulk_collect.collect_bulk(conn, prices=prices, at=NOW)   # same minute
    assert first["inserted"] == 1
    assert second["inserted"] == 0 and second["matched"] == 1
