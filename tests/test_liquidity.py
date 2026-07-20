"""Liquidity scoring: activity measure, cold-start prior, tiers, and persistence."""

from datetime import datetime, timedelta, timezone

from futmarket import db as futdb
from futmarket.services import liquidity

UTC = timezone.utc
NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def _rows(prices_at):
    """[(days_ago, price)] -> snapshot-like rows (ascending)."""
    out = []
    for days_ago, price in sorted(prices_at, reverse=True):
        ts = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({"timestamp": ts, "price": price})
    return out


# ---- activity measure -----------------------------------------------------

def test_updates_per_day_counts_changes():
    # points at days 3,2,1,0 -> ascending 100,100,120,120: 1 change over 3 days
    rows = _rows([(3, 100), (2, 100), (1, 120), (0, 120)])
    upd = liquidity.updates_per_day(rows, now=NOW, window_days=14)
    assert upd is not None and round(upd, 2) == round(1 / 3, 2)
    # two changes over 3 days -> ~0.67/day
    rows2 = _rows([(3, 100), (2, 110), (1, 110), (0, 130)])
    assert round(liquidity.updates_per_day(rows2, now=NOW), 2) == round(2 / 3, 2)


def test_updates_per_day_needs_two_points():
    assert liquidity.updates_per_day(_rows([(1, 100)]), now=NOW) is None
    assert liquidity.updates_per_day([], now=NOW) is None


def test_updates_per_day_windowed():
    # old point outside a 5-day window is dropped, leaving <2 in-window -> None
    rows = _rows([(30, 100), (0, 100)])
    assert liquidity.updates_per_day(rows, now=NOW, window_days=5) is None


# ---- scoring / tiers ------------------------------------------------------

def test_untradeable_is_zero_c():
    assert liquidity.score_card(tradeable=False, activity=99.0, price=50000) == (0.0, "C", False)


def test_busy_listing_scores_well_but_caps_at_b():
    """A busy *listed* price earns a good score, but tier A now requires evidence
    of real completed sales -- see test_proxy_alone_can_never_reach_tier_a."""
    score, tier, measured = liquidity.score_card(tradeable=True, activity=8.0, price=30000)
    assert measured and tier == "B" and score >= liquidity.TIER_B_MIN


def test_measured_inactive_card_is_low():
    score, tier, measured = liquidity.score_card(tradeable=True, activity=0.0, price=30000)
    assert measured and tier in ("B", "C") and score < liquidity.TIER_A_MIN


def test_provisional_capped_below_tier_a():
    # no activity data -> provisional, must never reach tier A regardless of band
    for price in (500, 20000, 200000, 5_000_000):
        score, tier, measured = liquidity.score_card(tradeable=True, activity=None, price=price)
        assert not measured and tier != "A" and score <= liquidity.PROVISIONAL_CAP + 1.0


def test_price_band_factor_monotone_ish():
    assert liquidity.price_band_factor(20000) > liquidity.price_band_factor(5_000_000)
    assert liquidity.price_band_factor(None) == 0.5


# ---- persistence ----------------------------------------------------------

def test_refresh_liquidity_writes_tiers(conn):
    futdb.upsert_card_meta(conn, {"player_id": "active", "rating": 88, "tradeable": 1})
    futdb.upsert_card_meta(conn, {"player_id": "cold", "rating": 84, "tradeable": 1})
    futdb.upsert_card_meta(conn, {"player_id": "sbc", "rating": 84, "tradeable": 0})
    conn.commit()

    # give "active" a busy price history; "cold" none
    for r in _rows([(6, 10000), (5, 10500), (4, 9800), (3, 10200), (2, 9900), (1, 10600), (0, 10000)]):
        futdb.insert_snapshot(conn, player_id="active", price=r["price"],
                              source="mock",
                              at=datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC))
    conn.commit()

    res = liquidity.refresh_liquidity(conn, source="mock", now=NOW)
    assert res["A"] + res["B"] + res["C"] == 3

    active = futdb.liquidity_get(conn, "active")
    cold = futdb.liquidity_get(conn, "cold")
    sbc = futdb.liquidity_get(conn, "sbc")
    assert active["updates_per_day"] is not None        # measured
    assert active["score"] > cold["score"]              # activity beats provisional
    assert cold["updates_per_day"] is None              # provisional
    assert sbc["tier"] == "C" and sbc["score"] == 0.0   # untradeable


# ---- real sale rates outrank the price-change proxy -----------------------

def test_real_sale_rate_sets_the_tier():
    """Tiers come from how fast a card actually sells, not a listing proxy."""
    fast, tier_fast, measured = liquidity.score_card(
        tradeable=True, sales_per_hour=300.0, price=20_000)
    assert tier_fast == "A" and measured and fast > 8

    _, tier_mid, _ = liquidity.score_card(
        tradeable=True, sales_per_hour=5.0, price=20_000)
    assert tier_mid == "B"

    _, tier_thin, _ = liquidity.score_card(
        tradeable=True, sales_per_hour=0.5, price=20_000)
    assert tier_thin == "C"


def test_proxy_alone_can_never_reach_tier_a():
    """A moving listed price is not proof anything traded."""
    score, tier, measured = liquidity.score_card(
        tradeable=True, activity=999.0, price=20_000)
    assert measured and tier != "A" and score < liquidity.TIER_A_MIN


def test_real_sales_beat_the_proxy_when_both_present():
    # proxy says busy, real sales say thin -> thin wins
    _, tier, _ = liquidity.score_card(
        tradeable=True, activity=999.0, sales_per_hour=0.2, price=20_000)
    assert tier == "C"


def test_sales_activity_is_log_scaled():
    n1 = liquidity.sales_activity_norm(1)
    n10 = liquidity.sales_activity_norm(10)
    n1000 = liquidity.sales_activity_norm(1000)
    assert n1 < n10 < 1.0 and n1000 == 1.0     # saturates, doesn't run away
    assert (n10 - n1) > (n1000 - n10)          # early gains matter most


def test_refresh_prefers_real_sales(conn):
    futdb.upsert_card_meta(conn, {"player_id": "fast", "tradeable": 1})
    futdb.upsert_card_meta(conn, {"player_id": "slow", "tradeable": 1})
    futdb.upsert_sale_stats(conn, player_id="fast", n_sales=100,
                            sales_per_hour=250.0, sold_median=20_000)
    futdb.upsert_sale_stats(conn, player_id="slow", n_sales=100,
                            sales_per_hour=0.3, sold_median=20_000)
    conn.commit()
    res = liquidity.refresh_liquidity(conn, now=NOW)
    assert res["from_real_sales"] == 2
    assert futdb.liquidity_get(conn, "fast")["tier"] == "A"
    assert futdb.liquidity_get(conn, "slow")["tier"] == "C"
