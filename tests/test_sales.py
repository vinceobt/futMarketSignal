"""Real-sale statistics: true price band, trade rate, and the listed-price gap."""

from datetime import datetime, timedelta, timezone

from futmarket import db as futdb
from futmarket.services import sales

UTC = timezone.utc
T0 = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _sales(prices, minutes_apart=10):
    return [(T0 + timedelta(minutes=i * minutes_apart), p)
            for i, p in enumerate(prices)]


def test_summarise_percentiles_and_rate():
    # 9 sales spanning 80 minutes
    stats = sales.summarise_sales(_sales([10, 12, 14, 16, 18, 20, 22, 24, 26]), listed=15)
    assert stats["n_sales"] == 9
    assert stats["sold_median"] == 18
    assert stats["sold_p25"] == 14 and stats["sold_p75"] == 22
    assert round(stats["window_hours"], 2) == round(80 / 60, 2)
    assert stats["sales_per_hour"] > 0


def test_summarise_requires_enough_sales():
    assert sales.summarise_sales(_sales([10, 12]), listed=11) is None


def test_sold_vs_listed_gap():
    """The going rate usually sits ABOVE the cheapest listing -- that gap is the
    realistic entry haircut a backtest must not ignore."""
    stats = sales.summarise_sales(_sales([100, 102, 104, 106, 108]), listed=100)
    assert stats["sold_median"] == 104
    assert stats["sold_vs_listed"] == 1.04


def test_summarise_handles_missing_listed():
    stats = sales.summarise_sales(_sales([10, 11, 12, 13, 14]), listed=None)
    assert stats["listed_price"] is None and stats["sold_vs_listed"] is None


def test_buy_band_is_lower_half_of_real_sales():
    stats = sales.summarise_sales(_sales([10, 12, 14, 16, 18, 20, 22, 24, 26]), listed=15)
    lo, hi = sales.buy_band(stats)
    assert (lo, hi) == (14, 18)          # p25 .. median, not a fake exact price
    assert lo < hi


def test_buy_band_widen():
    stats = sales.summarise_sales(_sales([100] * 5), listed=100)
    lo, hi = sales.buy_band(stats, widen_pct=10)
    assert lo == 100 and hi == 110


def test_buy_band_none_without_stats():
    assert sales.buy_band(None) is None


def test_persist_and_read_back(conn):
    stats = sales.summarise_sales(_sales([10, 12, 14, 16, 18]), listed=11)
    futdb.upsert_card_meta(conn, {"player_id": "p1", "name": "Test Card"})
    futdb.upsert_sale_stats(conn, player_id="p1", **stats)
    conn.commit()

    got = futdb.sale_stats_get(conn, "p1")
    assert got["sold_median"] == stats["sold_median"]
    assert got["n_sales"] == 5
    listed = futdb.sale_stats_list(conn, min_sales_per_hour=0.0)
    assert [r["player_id"] for r in listed] == ["p1"]


def test_sale_stats_upsert_replaces(conn):
    futdb.upsert_sale_stats(conn, player_id="p1", n_sales=5, sold_median=100)
    futdb.upsert_sale_stats(conn, player_id="p1", n_sales=9, sold_median=200)
    conn.commit()
    got = futdb.sale_stats_get(conn, "p1")
    assert got["n_sales"] == 9 and got["sold_median"] == 200
