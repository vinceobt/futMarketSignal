"""Track record: recording picks and scoring them against what the market did."""

from datetime import datetime, timedelta, timezone

from futmarket import db as futdb
from futmarket.services import scorecard

UTC = timezone.utc
PICKED = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _pick(conn, *, entry=10_000, target=13_000, stop=9_200, horizon=7):
    futdb.upsert_card_meta(conn, {"player_id": "c1", "name": "Test Card"})
    futdb.insert_pick(conn, player_id="c1", entry_price=entry, target_price=target,
                      stop_price=stop, horizon_days=horizon, at=PICKED,
                      confidence=0.8)
    conn.commit()
    return futdb.open_picks(conn)[0]


def _prices(pairs):
    return [(PICKED + timedelta(days=d), p) for d, p in pairs]


# ---- scoring one pick -----------------------------------------------------

def test_target_hit_is_a_win_net_of_tax():
    pick = None

    class P(dict):
        def __getitem__(self, k): return dict.__getitem__(self, k)
    pick = P(picked_at=PICKED.strftime("%Y-%m-%dT%H:%M:%SZ"), horizon_days=7,
             entry_price=10_000, target_price=13_000, stop_price=9_200)
    status, exit_price, realized = scorecard.score_pick(
        pick, _prices([(1, 11_000), (2, 13_500)]),
        now=PICKED + timedelta(days=8))
    assert status == scorecard.TARGET and exit_price == 13_500
    # net of 5% tax, not the gross move
    assert round(realized, 4) == round((13_500 * 0.95 / 10_000 - 1) * 100, 4)


def test_stop_hit_is_a_loss(conn):
    pick = _pick(conn)
    status, exit_price, realized = scorecard.score_pick(
        pick, _prices([(1, 9_000)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.STOP and realized < 0


def test_whichever_barrier_comes_first_wins(conn):
    pick = _pick(conn)
    # dips to the stop on day 1, then rockets -- must score as stopped
    status, _, _ = scorecard.score_pick(
        pick, _prices([(1, 9_000), (2, 20_000)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.STOP


def test_expired_marks_out_at_the_last_price(conn):
    pick = _pick(conn)
    status, exit_price, realized = scorecard.score_pick(
        pick, _prices([(1, 10_100), (6, 10_500)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.EXPIRED and exit_price == 10_500


def test_still_open_before_the_horizon(conn):
    pick = _pick(conn)
    assert scorecard.score_pick(
        pick, _prices([(1, 10_100)]), now=PICKED + timedelta(days=2)) is None


def test_prices_before_the_pick_are_ignored(conn):
    """A card that hit the target *before* we picked it is not a win."""
    pick = _pick(conn)
    earlier = [(PICKED - timedelta(days=1), 20_000)]
    assert scorecard.score_pick(pick, earlier,
                                now=PICKED + timedelta(days=2)) is None


# ---- the loop and the summary --------------------------------------------

def test_score_open_picks_closes_and_summarises(conn):
    _pick(conn)
    for d, price in ((1, 10_500), (2, 13_500)):
        futdb.insert_snapshot(conn, player_id="c1", price=price, source="futgg",
                              at=PICKED + timedelta(days=d))
    conn.commit()

    res = scorecard.score_open_picks(conn, now=PICKED + timedelta(days=8))
    assert res["target"] == 1 and res["still_open"] == 0
    assert futdb.open_picks(conn) == []

    s = scorecard.summary(conn)
    assert s["closed"] == 1 and s["hit_target"] == 1
    assert s["win_rate"] == 1.0 and s["avg_return_pct"] > 0


def test_summary_with_nothing_resolved(conn):
    _pick(conn)
    s = scorecard.summary(conn)
    assert s["closed"] == 0 and s["open"] == 1


def test_same_card_not_double_logged_in_a_minute(conn):
    _pick(conn)
    again = futdb.insert_pick(conn, player_id="c1", entry_price=10_000,
                              target_price=13_000, stop_price=9_200,
                              horizon_days=7, at=PICKED)
    assert again is False
