"""Lifecycle features: season position and event distances."""

from datetime import date

from futmarket import db as futdb
from futmarket.ml import lifecycle


EVENTS = [
    {"event_type": "PROMO", "start_date": "2026-01-10", "end_date": None},
    {"event_type": "PROMO", "start_date": "2026-02-20", "end_date": None},
    {"event_type": "TOTW", "start_date": "2026-01-14", "end_date": None},
    {"event_type": "SBC", "start_date": "2026-01-05", "end_date": "2026-01-20"},
    {"event_type": "SBC", "start_date": "2026-01-18", "end_date": "2026-01-25"},
]
LC = lifecycle.Lifecycle(launch_date="2025-09-08", events=EVENTS)


def test_days_since_launch():
    assert LC.days_since_launch("2025-09-08") == 0
    assert LC.days_since_launch(date(2025, 9, 18)) == 10


def test_days_to_next_promo():
    assert LC.days_to_next("2026-01-01", "PROMO") == 9
    assert LC.days_to_next("2026-01-10", "PROMO") == 0      # today counts as 0
    assert LC.days_to_next("2026-01-11", "PROMO") == 40     # rolls to the next one
    assert LC.days_to_next("2026-03-01", "PROMO") is None   # nothing ahead


def test_days_since_last_promo():
    assert LC.days_since_last("2026-01-15", "PROMO") == 5
    assert LC.days_since_last("2026-01-01", "PROMO") is None   # none yet
    assert LC.days_since_last("2026-02-20", "PROMO") == 0


def test_active_sbc_windows_counted():
    assert LC.active_count("2026-01-06", "SBC") == 1
    assert LC.active_count("2026-01-19", "SBC") == 2   # both windows overlap
    assert LC.active_count("2026-01-30", "SBC") == 0


def test_weekend_window_flag():
    # 2026-01-15 is a Thursday -> inside the Weekend League demand window
    assert LC.features("2026-01-15")["is_weekend_window"] == 1
    # 2026-01-13 is a Tuesday
    assert LC.features("2026-01-13")["is_weekend_window"] == 0


def test_features_shape_and_values():
    f = LC.features("2026-01-15")
    assert f["days_since_launch"] == (date(2026, 1, 15) - date(2025, 9, 8)).days
    assert f["days_to_next_promo"] == 36
    assert f["days_since_last_totw"] == 1
    assert f["active_sbc_count"] == 1
    for key in ("days_to_next_promo", "days_since_last_promo", "days_to_next_totw",
                "days_since_last_totw", "days_to_next_sbc", "days_since_last_sbc"):
        assert key in f


def test_empty_calendar_is_safe():
    empty = lifecycle.Lifecycle()
    f = empty.features("2026-01-15")
    assert f["days_since_launch"] is None
    assert f["days_to_next_promo"] is None
    assert f["active_sbc_count"] == 0


def test_bad_dates_ignored():
    lc = lifecycle.Lifecycle(launch_date="not-a-date",
                             events=[{"event_type": "PROMO", "start_date": "nonsense"}])
    assert lc.days_since_launch("2026-01-01") is None
    assert lc.days_to_next("2026-01-01", "PROMO") is None


def test_load_from_db(conn):
    futdb.set_game_launch(conn, title="fc26", launch_date="2025-09-08")
    futdb.replace_events(conn, source="derived_cards", events=[
        {"event_type": "PROMO", "start_date": "2026-01-10", "end_date": None}])
    conn.commit()
    lc = lifecycle.load(conn, title="fc26")
    assert lc.days_since_launch("2025-09-18") == 10
    assert lc.days_to_next("2026-01-01", "PROMO") == 9


# ---- EA announcements are tracked apart from the card drop ----------------

def test_announcement_is_separate_from_the_card_release(conn):
    """An EA promo article and the card drop it precedes are different moments:
    the market dips on the announcement, then recovers days later."""
    futdb.set_game_launch(conn, title="fc26", launch_date="2025-09-08")
    futdb.replace_events(conn, source="ea_news", events=[
        {"event_type": "PROMO", "start_date": "2026-01-08", "end_date": None}])
    futdb.replace_events(conn, source="derived_cards", events=[
        {"event_type": "PROMO", "start_date": "2026-01-10", "end_date": None}])
    conn.commit()

    lc = lifecycle.load(conn, title="fc26")
    f = lc.features("2026-01-09")
    assert f["days_since_last_announce"] == 1     # EA spoke yesterday
    assert f["days_to_next_promo"] == 1           # cards land tomorrow
    # the announcement must NOT also count as a card-release promo
    assert lc.days_since_last("2026-01-09", "PROMO") is None


def test_non_promo_ea_articles_are_not_announcements(conn):
    futdb.replace_events(conn, source="ea_news", events=[
        {"event_type": "PATCH", "start_date": "2026-01-08", "end_date": None}])
    conn.commit()
    lc = lifecycle.load(conn, title="fc26")
    assert lc.days_since_last("2026-01-09", "ANNOUNCE") is None
