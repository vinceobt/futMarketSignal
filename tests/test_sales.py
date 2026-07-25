"""Real-sale statistics: true price band, trade rate, and the listed-price gap."""

from datetime import datetime, timedelta, timezone

from futmarket import db as futdb
from futmarket.collectors.base import SourceError
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


# ---- banking real transactions as a price series --------------------------

def test_sold_prices_are_banked_as_their_own_series(conn):
    """The listing index is one number set by whoever is currently most
    desperate; its day-to-day return std is 193%. Completed sales are what the
    market actually agreed on, so they are banked as a series in their own right
    and accumulate into a second, cleaner price history."""
    from datetime import datetime, timezone
    from futmarket.services import sales as sales_service

    hour = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
    trades = [(hour.replace(minute=m), p)
              for m, p in ((5, 10_000), (20, 10_400), (50, 40_000))]
    stored = sales_service.bank_sold_prices(conn, "c1", trades)
    conn.commit()

    assert stored == 1                      # one hour, one banked point
    rows = futdb.snapshots(conn, "c1", sales_service.SOLD_SOURCE)
    # the hourly MEDIAN, so a single panic sale can't set the going rate
    assert [r["price"] for r in rows] == [10_400]


def test_banking_the_same_sales_twice_is_idempotent(conn):
    """Sweeps overlap — the feed returns the last ~100 sales every time."""
    from datetime import datetime, timezone
    from futmarket.services import sales as sales_service

    hour = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
    trades = [(hour, 10_000), (hour.replace(hour=15), 11_000)]
    sales_service.bank_sold_prices(conn, "c1", trades)
    sales_service.bank_sold_prices(conn, "c1", trades)
    conn.commit()
    assert len(futdb.snapshots(conn, "c1", sales_service.SOLD_SOURCE)) == 2


def test_banking_ignores_junk_prices(conn):
    from datetime import datetime, timezone
    from futmarket.services import sales as sales_service

    hour = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
    assert sales_service.bank_sold_prices(conn, "c1", [(hour, 0)]) == 0
    assert sales_service.bank_sold_prices(conn, "c1", []) == 0


def test_the_stalest_cards_are_fetched_first(conn, monkeypatch):
    """A capped run must advance coverage, not re-fetch the same head of the list.

    Ordering by liquidity meant `--limit 250` fetched the same 250 cards every
    cycle forever, so sale history could never grow past the most liquid cards —
    fine for keeping a buy band fresh, useless for building a price series.
    """
    from datetime import datetime, timezone
    from futmarket.services import sales as sales_service

    for defn, (pid, score) in enumerate((("fresh", 9.0), ("stale", 8.0),
                                         ("never", 1.0)), start=1):
        futdb.upsert_card_meta(conn, {"player_id": pid, "name": pid,
                                      "definition_id": defn, "tradeable": 1})
        futdb.upsert_liquidity(conn, player_id=pid, score=score, tier="A")
    # "fresh" was just refreshed, "stale" long ago, "never" has no row at all
    for pid, when in (("fresh", "2026-07-25T12:00:00Z"), ("stale", "2026-01-01T12:00:00Z")):
        futdb.upsert_sale_stats(conn, player_id=pid, n_sales=10, sales_per_hour=5.0)
        conn.execute("UPDATE sale_stats SET computed_at=? WHERE player_id=?", (when, pid))
    conn.commit()

    seen = []
    from futmarket.collectors import history_source

    def spy(definition_id, game, client=None):
        # record which card we were asked for, in order
        row = conn.execute("SELECT player_id FROM card_meta WHERE definition_id=?",
                           (definition_id,)).fetchone()
        seen.append(row["player_id"] if row else "?")
        raise SourceError("no network in tests")

    monkeypatch.setattr(history_source, "fetch_card_detail", spy)
    sales_service.refresh_sale_stats(conn, delay=0, retries=0,
                                     max_consecutive_failures=99)
    # never-fetched first, then the oldest refresh, and only then the fresh one
    assert seen[0] == "never"
    assert seen.index("stale") < seen.index("fresh")
