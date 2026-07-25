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
                      confidence=0.8, strategy=scorecard.CURRENT_STRATEGY)
    conn.commit()
    return futdb.open_picks(conn)[0]


def _daily(pairs):
    """The robust one-price-per-day series ``db.daily_prices`` returns."""
    return [((PICKED + timedelta(days=d)).strftime("%Y-%m-%d"), p, 1)
            for d, p in pairs]


def _fake_pick(**over):
    base = dict(picked_at=PICKED.strftime("%Y-%m-%dT%H:%M:%SZ"), horizon_days=7,
                chosen_horizon_days=7, entry_price=10_000, target_price=13_000,
                stop_price=9_200)
    base.update(over)
    return base


# ---- scoring one pick -----------------------------------------------------

def test_target_hit_is_a_win_net_of_tax():
    status, exit_price, realized = scorecard.score_pick(
        _fake_pick(), _daily([(1, 11_000), (2, 13_500)]),
        now=PICKED + timedelta(days=8))
    assert status == scorecard.TARGET and exit_price == 13_500
    # net of 5% tax, not the gross move
    assert round(realized, 4) == round((13_500 * 0.95 / 10_000 - 1) * 100, 4)


def test_stop_hit_is_a_loss(conn):
    pick = _pick(conn)
    status, exit_price, realized = scorecard.score_pick(
        pick, _daily([(1, 9_000)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.STOP and realized < 0


def test_whichever_barrier_comes_first_wins(conn):
    pick = _pick(conn)
    # dips to the stop on day 1, then rockets -- must score as stopped
    status, _, _ = scorecard.score_pick(
        pick, _daily([(1, 9_000), (2, 20_000)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.STOP


def test_expired_marks_out_at_the_last_price(conn):
    pick = _pick(conn)
    status, exit_price, realized = scorecard.score_pick(
        pick, _daily([(1, 10_100), (6, 10_500)]), now=PICKED + timedelta(days=8))
    assert status == scorecard.EXPIRED and exit_price == 10_500


def test_still_open_before_the_horizon(conn):
    pick = _pick(conn)
    assert scorecard.score_pick(
        pick, _daily([(1, 10_100)]), now=PICKED + timedelta(days=2)) is None


def test_prices_before_the_pick_are_ignored(conn):
    """A card that hit the target *before* we picked it is not a win."""
    pick = _pick(conn)
    earlier = _daily([(-1, 20_000)])
    assert scorecard.score_pick(pick, earlier,
                                now=PICKED + timedelta(days=2)) is None


def test_the_chosen_horizon_decides_when_a_trade_expires():
    """The model picks the holding period per trade; scoring must honour it and
    not the table's default, or a 14-day call gets marked out on day 7."""
    pick = _fake_pick(horizon_days=7, chosen_horizon_days=14)
    flat = _daily([(d, 10_100) for d in range(1, 13)])
    assert scorecard.score_pick(pick, flat, now=PICKED + timedelta(days=10)) is None
    status, _, _ = scorecard.score_pick(pick, flat, now=PICKED + timedelta(days=15))
    assert status == scorecard.EXPIRED


def test_intraday_jitter_cannot_stop_a_trade_out(conn):
    """The bug that broke the last strategy, from the other end.

    Scoring used to walk every raw two-hourly snapshot, so a single cheap listing
    booked a loss. Grading reads one robust price per day, so a day that *dipped*
    intraday but held up overall is not a stop.
    """
    pick = _pick(conn)
    next_day = (PICKED + timedelta(days=1)).replace(hour=0)
    for hour, price in ((2, 8_000), (8, 10_400), (14, 10_600), (20, 10_500)):
        futdb.insert_snapshot(conn, player_id="c1", price=price, source="futgg",
                              at=next_day + timedelta(hours=hour))
    conn.commit()
    daily = futdb.daily_prices(conn, "c1", "futgg")
    status = scorecard.score_pick(pick, daily, now=PICKED + timedelta(days=2))
    assert status is None            # still running, not stopped by the 8,000 print


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


def _closed_pick(conn, pid, *, entry, exit_price, target, at, strategy="legacy",
                 benchmark_pct=None):
    """Log a pick and immediately close it at a known exit, for summary tests."""
    futdb.upsert_card_meta(conn, {"player_id": pid, "name": pid})
    futdb.insert_pick(conn, player_id=pid, entry_price=entry, target_price=target,
                      stop_price=int(entry * 0.9), horizon_days=7, at=at,
                      strategy=strategy)
    pick = [p for p in futdb.open_picks(conn) if p["player_id"] == pid][0]
    realized = (exit_price * 0.95 / entry - 1) * 100
    futdb.close_pick(conn, pick["id"], status="target", exit_price=exit_price,
                     realized_pct=realized, benchmark_pct=benchmark_pct,
                     at=at + timedelta(days=1))
    conn.commit()


def test_score_pick_applies_sell_slippage():
    pick = _fake_pick()
    daily = _daily([(1, 13_500)])
    _, _, r0 = scorecard.score_pick(pick, daily, sell_slippage_pct=0.0,
                                    now=PICKED + timedelta(days=8))
    _, _, r2 = scorecard.score_pick(pick, daily, sell_slippage_pct=2.0,
                                    now=PICKED + timedelta(days=8))
    assert r2 < r0                       # selling under the going rate nets less


def test_summary_judges_the_current_strategy_alone(conn):
    # a winning current-strategy pick and a losing legacy pick, both tradeable
    _closed_pick(conn, "new", entry=10_000, exit_price=13_000, target=12_000,
                 at=PICKED, strategy=scorecard.CURRENT_STRATEGY)
    _closed_pick(conn, "old", entry=10_000, exit_price=8_000, target=13_000,
                 at=PICKED + timedelta(minutes=1), strategy="legacy")
    current = scorecard.summary(conn)
    legacy = scorecard.summary(conn, strategy="legacy")
    both = scorecard.summary(conn, strategy=None)
    assert current["graded"] == 1 and current["return_on_capital_pct"] > 0
    assert legacy["graded"] == 1 and legacy["return_on_capital_pct"] < 0
    assert both["graded"] == 2                             # unfiltered = all


def test_broken_dip_picks_are_never_in_the_headline(conn):
    """dip_v1 was graded with stops that sat above the market price. Those rows
    stay in the database for reference but must not touch the current record."""
    _closed_pick(conn, "broken", entry=10_000, exit_price=8_600, target=13_000,
                 at=PICKED, strategy="dip_v1_broken")
    assert scorecard.summary(conn)["graded"] == 0
    assert scorecard.summary(conn, strategy="dip_v1_broken")["graded"] == 1


def test_alpha_compares_the_pick_to_the_market_over_the_same_window(conn):
    """A pick that lost 3% in a fortnight when the median card lost 9% is a good
    call. Absolute return alone calls it a failure."""
    # 10,210 sold nets 10,210 x 0.95 = 9,700 on a 10,000 entry: -3%.
    _closed_pick(conn, "beat", entry=10_000, exit_price=10_210, target=13_000,
                 at=PICKED, strategy=scorecard.CURRENT_STRATEGY,
                 benchmark_pct=-9.0)
    s = scorecard.summary(conn)
    assert s["return_on_capital_pct"] < 0        # it did lose coins
    assert s["alpha_vs_market_pct"] > 0          # ...and still beat the market


def test_cheap_penny_card_cannot_fake_the_record(conn):
    """A 200-coin card that 'tripled' must barely move return-on-capital, while a
    real 50k trade that lost dominates — the whole point of coin-weighting."""
    # penny card: 200 -> 700 on paper (+232% net), but only ~475 coins
    _closed_pick(conn, "penny", entry=200, exit_price=700, target=700, at=PICKED)
    # real trade: 50k -> 46k, a genuine loss of thousands of coins
    _closed_pick(conn, "real", entry=50_000, exit_price=46_000, target=65_000,
                 at=PICKED + timedelta(minutes=1))

    s = scorecard.summary(conn, strategy=None)
    # the sub-1000 penny card is excluded from the judged set entirely
    assert s["graded"] == 1
    # capital-weighted view reflects the real trade: a loss
    assert s["return_on_capital_pct"] < 0 and s["coins_pnl"] < 0


def test_summary_reports_zero_when_only_junk_resolved(conn):
    _closed_pick(conn, "penny", entry=300, exit_price=900, target=900, at=PICKED)
    s = scorecard.summary(conn, strategy=None)
    assert s["closed"] == 1 and s["graded"] == 0     # nothing tradeable to judge


def test_same_card_not_double_logged_in_a_minute(conn):
    _pick(conn)
    again = futdb.insert_pick(conn, player_id="c1", entry_price=10_000,
                              target_price=13_000, stop_price=9_200,
                              horizon_days=7, at=PICKED)
    assert again is False


def test_one_open_position_per_card(conn):
    """The loop re-derives the same shortlist every two hours. Without this, one
    opportunity was recorded as a dozen independent trades — 92 picks over 37
    cards, some simultaneously 'open' and 'stop'."""
    _pick(conn)
    assert futdb.has_open_pick(conn, "c1") is True
    assert futdb.has_open_pick(conn, "never-picked") is False
