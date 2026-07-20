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


def test_buy_band_anchors_to_the_live_listing():
    """Real failure this guards: a band built from completed sales quoted
    173k-195k while the card was actually listed at 241k. Sales lag; the
    listing is current, so the band must start there."""
    stats = sales.summarise_sales(_sales([10, 12, 14, 16, 18, 20, 22, 24, 26]), listed=100)
    lo, hi = sales.buy_band(stats, listed_price=100)
    assert lo == 100                      # start at the live price
    assert hi == 105                      # pay a little over it to get filled
    assert lo < hi


def test_buy_band_ignores_stale_sales_when_listing_moved():
    stats = sales.summarise_sales(_sales([50_000] * 9), listed=241_000)
    lo, hi = sales.buy_band(stats, listed_price=241_000)
    assert lo == 241_000 and hi > lo      # not the 50k the old sales suggest


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


# ---- the band must reflect prices you can trade at NOW --------------------

def test_band_tracks_a_rising_price_not_the_stale_average():
    """Real failure this guards: 100 sales over 15h on a rising card gave a
    53,500 band while the card actually traded at 74,500."""
    old = [(T0 + timedelta(minutes=i * 6), 50_000) for i in range(80)]     # ~8h cheap
    new = [(T0 + timedelta(hours=8, minutes=i * 6), 72_000) for i in range(20)]
    stats = sales.summarise_sales(old + new, listed=74_000)
    # the band follows the recent sales, not the 15-hour average
    assert stats["sold_median"] == 72_000
    assert stats["band_from_sales"] <= 25
    # and the rate still uses the whole window
    assert stats["n_sales"] == 100
    assert stats["window_hours"] > 8


def test_band_falls_back_to_last_sales_when_window_is_quiet():
    """A thin card with no sales in the recent window still gets a usable band."""
    spaced = [(T0 + timedelta(hours=i * 3), 10_000 + i * 100) for i in range(12)]
    stats = sales.summarise_sales(spaced, listed=11_000)
    assert stats is not None
    assert stats["band_from_sales"] >= sales.MIN_SALES


def test_recent_slice_prefers_the_time_window():
    dense = [(T0 + timedelta(minutes=i * 5), 100) for i in range(40)]
    recent = sales._recent_slice(dense)
    newest = max(t for t, _ in dense)
    assert all((newest - t).total_seconds() / 3600 <= sales.RECENT_HOURS
               for t, _ in recent)
