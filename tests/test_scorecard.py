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
    assert s["closed"] == 1 and s["graded"] == 1 and s["hit_target"] == 1
    assert s["win_rate"] == 1.0
    assert s["coins_pnl"] > 0 and s["return_on_capital_pct"] > 0


def test_summary_with_nothing_resolved(conn):
    _pick(conn)
    s = scorecard.summary(conn)
    assert s["closed"] == 0 and s["open"] == 1 and s["graded"] == 0


def _closed_pick(conn, pid, *, entry, exit_price, target, at, strategy="legacy"):
    """Log a pick and immediately close it at a known exit, for summary tests."""
    futdb.upsert_card_meta(conn, {"player_id": pid, "name": pid})
    futdb.insert_pick(conn, player_id=pid, entry_price=entry, target_price=target,
                      stop_price=int(entry * 0.9), horizon_days=7, at=at,
                      strategy=strategy)
    pick = [p for p in futdb.open_picks(conn) if p["player_id"] == pid][0]
    realized = (exit_price * 0.95 / entry - 1) * 100
    futdb.close_pick(conn, pick["id"], status="target", exit_price=exit_price,
                     realized_pct=realized, at=at + timedelta(days=1))
    conn.commit()


def test_score_pick_applies_sell_slippage():
    pick = dict(picked_at=PICKED.strftime("%Y-%m-%dT%H:%M:%SZ"), horizon_days=7,
                entry_price=10_000, target_price=13_000, stop_price=9_200)
    prices = _prices([(1, 13_500)])
    _, _, r0 = scorecard.score_pick(pick, prices, sell_slippage_pct=0.0,
                                    now=PICKED + timedelta(days=8))
    _, _, r2 = scorecard.score_pick(pick, prices, sell_slippage_pct=2.0,
                                    now=PICKED + timedelta(days=8))
    assert r2 < r0                       # selling under the going rate nets less


def test_summary_judges_the_current_strategy_alone(conn):
    # a winning new-strategy pick and a losing legacy pick, both tradeable
    _closed_pick(conn, "new", entry=10_000, exit_price=13_000, target=12_000,
                 at=PICKED, strategy="dip_v1")
    _closed_pick(conn, "old", entry=10_000, exit_price=8_000, target=13_000,
                 at=PICKED + timedelta(minutes=1), strategy="legacy")
    dip = scorecard.summary(conn, strategy="dip_v1")
    legacy = scorecard.summary(conn, strategy="legacy")
    both = scorecard.summary(conn)
    assert dip["graded"] == 1 and dip["return_on_capital_pct"] > 0    # new: winning
    assert legacy["graded"] == 1 and legacy["return_on_capital_pct"] < 0  # old: losing
    assert both["graded"] == 2                                        # unfiltered = all


def test_cheap_penny_card_cannot_fake_the_record(conn):
    """A 200-coin card that 'tripled' must barely move return-on-capital, while a
    real 50k trade that lost dominates — the whole point of coin-weighting."""
    # penny card: 200 -> 700 on paper (+232% net), but only ~475 coins
    _closed_pick(conn, "penny", entry=200, exit_price=700, target=700, at=PICKED)
    # real trade: 50k -> 46k, a genuine loss of thousands of coins
    _closed_pick(conn, "real", entry=50_000, exit_price=46_000, target=65_000,
                 at=PICKED + timedelta(minutes=1))

    s = scorecard.summary(conn)
    # the sub-1000 penny card is excluded from the judged set entirely
    assert s["graded"] == 1
    # capital-weighted view reflects the real trade: a loss
    assert s["return_on_capital_pct"] < 0 and s["coins_pnl"] < 0


def test_summary_reports_zero_when_only_junk_resolved(conn):
    _closed_pick(conn, "penny", entry=300, exit_price=900, target=900, at=PICKED)
    s = scorecard.summary(conn)
    assert s["closed"] == 1 and s["graded"] == 0     # nothing tradeable to judge


def test_same_card_not_double_logged_in_a_minute(conn):
    _pick(conn)
    again = futdb.insert_pick(conn, player_id="c1", entry_price=10_000,
                              target_price=13_000, stop_price=9_200,
                              horizon_days=7, at=PICKED)
    assert again is False
