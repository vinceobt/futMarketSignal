"""Feature-matrix assembly: card features, joins, and leakage discipline."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from futmarket import db as futdb
from futmarket.ml import dataset

UTC = timezone.utc
START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _seed_card(conn, pid, prices, *, rating=84, position="ST", league="Prem",
               nation="England", version="Gold"):
    futdb.upsert_card_meta(conn, {
        "player_id": pid, "definition_id": abs(hash(pid)) % 10**6, "rating": rating,
        "position": position, "league": league, "nation": nation, "version": version,
        "release_date": "2025-09-08"})
    for i, price in enumerate(prices):
        futdb.insert_snapshot(conn, player_id=pid, price=price, source="futgg",
                              at=START + timedelta(days=i))


def _seed_many(conn, n=6, days=40, base=10_000):
    for c in range(n):
        prices = [base + c * 100 + i * 50 for i in range(days)]
        _seed_card(conn, f"card{c}", prices, rating=84)
    conn.commit()


# ---- daily loading --------------------------------------------------------

def test_load_daily_takes_the_median_price_per_day(conn):
    """The day's median, not its last snapshot.

    What we record is the cheapest live listing at a moment in time. Taking the
    last one made day-to-day returns swing with a std of 193%; the median brings
    it to 41%. More than half this market's apparent volatility was an artifact
    of when the collector happened to run.
    """
    futdb.upsert_card_meta(conn, {"player_id": "p1", "rating": 84})
    for hour, price in ((9, 100), (13, 120), (20, 150)):
        futdb.insert_snapshot(conn, player_id="p1", price=price, source="futgg",
                              at=START.replace(hour=hour))
    conn.commit()
    daily = dataset.load_daily_prices(conn, min_history_days=1)
    assert len(daily) == 1
    assert daily["price"].iloc[0] == 120      # the median, not the 150 close
    assert daily["n_snapshots"].iloc[0] == 3


def test_load_daily_ignores_a_bad_print(conn):
    """A discard-price or mispriced listing must not become the day's price."""
    futdb.upsert_card_meta(conn, {"player_id": "p1", "rating": 84})
    for hour, price in ((9, 10_000), (13, 200), (17, 10_400), (20, 10_200)):
        futdb.insert_snapshot(conn, player_id="p1", price=price, source="futgg",
                              at=START.replace(hour=hour))
    conn.commit()
    daily = dataset.load_daily_prices(conn, min_history_days=1)
    assert daily["price"].iloc[0] == 10_200   # the 200 print is dropped


def test_market_features_separate_a_cards_move_from_the_markets(conn):
    """The fact the old feature set could not express: a card that fell 4% on a
    day everything fell 9% is strong, not weak."""
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "date": ["2026-01-02"] * 3,
        "price": [100.0, 100.0, 100.0],
        "ret_1d": [-4.0, -9.0, -14.0],
    })
    out = dataset.add_market_features(frame, horizons=(1,))
    assert out["market_ret_1d"].iloc[0] == -9.0          # the median card
    assert out["excess_ret_1d"].tolist() == [5.0, 0.0, -5.0]


def test_short_history_cards_dropped(conn):
    _seed_card(conn, "brief", [100, 110, 120])
    conn.commit()
    assert dataset.load_daily_prices(conn, min_history_days=20).empty


# ---- card features --------------------------------------------------------

def test_returns_computed():
    daily = pd.DataFrame({
        "player_id": ["p"] * 9,
        "date": [f"2026-01-0{i+1}" for i in range(9)],
        "price": [100, 110, 120, 130, 140, 150, 160, 170, 180]})
    out = dataset.add_card_features(daily)
    # day 2 vs day 1: 110/100 - 1 = +10%
    assert round(out["ret_1d"].iloc[1], 6) == 10.0
    assert np.isnan(out["ret_1d"].iloc[0])      # nothing before the first day


def test_rolling_stats_exclude_current_day():
    """The range a price is compared against must not contain that price."""
    prices = [100] * 8 + [500]        # a spike on the final day
    daily = pd.DataFrame({"player_id": ["p"] * 9,
                          "date": [f"2026-01-0{i+1}" for i in range(9)],
                          "price": prices})
    out = dataset.add_card_features(daily)
    last = out.iloc[-1]
    # window is the flat 100s only, so the spike reads as far above its range
    assert last["dist_to_floor_pct"] == 400.0
    assert last["drawdown_pct"] == 400.0      # vs a past max of 100, not itself


def test_no_feature_uses_future_rows():
    """Truncating the future must not change features computed on the past."""
    prices = [100, 105, 98, 110, 115, 120, 118, 125, 130, 140, 90, 95]
    dates = [f"2026-01-{i+1:02d}" for i in range(len(prices))]
    full = dataset.add_card_features(pd.DataFrame(
        {"player_id": ["p"] * len(prices), "date": dates, "price": prices}))
    cut = 8
    truncated = dataset.add_card_features(pd.DataFrame(
        {"player_id": ["p"] * cut, "date": dates[:cut], "price": prices[:cut]}))
    cols = ["ret_1d", "ret_7d", "z_score", "dist_to_floor_pct", "drawdown_pct"]
    pd.testing.assert_frame_equal(
        full.head(cut)[cols].reset_index(drop=True),
        truncated[cols].reset_index(drop=True))


# ---- full assembly --------------------------------------------------------

def test_build_dataset_joins_everything(conn):
    _seed_many(conn, n=6, days=40)
    futdb.set_game_launch(conn, title="fc26", launch_date="2025-09-08")
    futdb.replace_events(conn, source="derived_cards", events=[
        {"event_type": "PROMO", "start_date": "2026-01-20", "end_date": None}])
    futdb.upsert_liquidity(conn, player_id="card0", score=8.0, tier="A")
    conn.commit()

    frame = dataset.build_dataset(conn)
    assert not frame.empty
    # one row per card/day, not duplicated by cohort membership
    assert len(frame) == len(frame.drop_duplicates(["player_id", "date"]))

    for col in ("ret_1d", "z_score", "cohort_ret_1d", "rel_strength_1d",
                "days_since_launch", "days_to_next_promo", "liq_score", "n_cohorts"):
        assert col in frame.columns

    row = frame[(frame["player_id"] == "card0") & (frame["date"] == "2026-01-10")]
    assert row["days_since_launch"].iloc[0] == 124     # 2025-09-08 -> 2026-01-10
    assert row["days_to_next_promo"].iloc[0] == 10     # promo on 2026-01-20
    assert row["liq_score"].iloc[0] == 8.0
    assert row["n_cohorts"].iloc[0] > 0                # belongs to real cohorts


def test_build_dataset_empty_is_safe(conn):
    assert dataset.build_dataset(conn).empty


def test_relative_strength_signs(conn):
    """A card outperforming its cohort gets positive relative strength."""
    for c in range(5):                       # flat cohort peers
        _seed_card(conn, f"flat{c}", [10_000] * 40, rating=84)
    _seed_card(conn, "riser", [10_000 + i * 500 for i in range(40)], rating=84)
    conn.commit()

    frame = dataset.build_dataset(conn)
    riser = frame[(frame["player_id"] == "riser") & (frame["date"] == "2026-02-05")]
    assert riser["rel_strength_1d"].iloc[0] > 0


# ---- behaviour features (weekly rhythm + release curve) -------------------

def test_day_of_week_and_weekend_flag():
    frame = pd.DataFrame({"player_id": ["p"] * 3,
                          "date": ["2026-07-17", "2026-07-18", "2026-07-20"],
                          "price": [100, 100, 100]})
    out = dataset.add_behaviour_features(frame)
    # 2026-07-17 is a Friday, 18th Saturday, 20th Monday
    assert list(out["day_of_week"]) == [4, 5, 0]
    assert list(out["is_weekend_window"]) == [1, 1, 0]


def test_days_since_card_release_and_phase():
    frame = pd.DataFrame({
        "player_id": ["p"] * 4,
        "date": ["2026-07-01", "2026-07-02", "2026-07-10", "2026-08-01"],
        "price": [100] * 4,
        "release_date": ["2026-07-01"] * 4})
    out = dataset.add_behaviour_features(frame)
    assert list(out["days_since_card_release"]) == [0, 1, 9, 31]
    # phase rises as the card ages through crash -> bottom -> recovery
    phases = list(out["release_phase"])
    assert phases == sorted(phases) and phases[0] < phases[-1]


def test_behaviour_features_survive_missing_release_date():
    frame = pd.DataFrame({"player_id": ["p"], "date": ["2026-07-17"],
                          "price": [100], "release_date": [None]})
    out = dataset.add_behaviour_features(frame)
    assert out["day_of_week"].iloc[0] == 4
    assert pd.isna(out["days_since_card_release"].iloc[0])


def test_behaviour_features_handle_categorical_dates():
    """`date` is a category in the real pipeline (2M rows); datetime arithmetic
    must still work. Plain-string fixtures hid this until it hit live data."""
    frame = pd.DataFrame({
        "player_id": ["p"] * 2,
        "date": pd.Series(["2026-07-17", "2026-07-20"]).astype("category"),
        "price": [100, 100],
        "release_date": ["2026-07-10", "2026-07-10"]})
    out = dataset.add_behaviour_features(frame)
    assert list(out["day_of_week"]) == [4, 0]
    assert list(out["days_since_card_release"]) == [7, 10]


def test_special_cards_are_flagged_for_the_release_trade():
    """Only promo/special cards have a release curve worth trading — a Common is
    'released' at launch and never crashes. Pooled with 6,400 base golds the
    effect disappears."""
    frame = pd.DataFrame({
        "player_id": list("abcd"), "date": ["2026-01-02"] * 4,
        "price": [100.0] * 4, "release_date": ["2026-01-01"] * 4,
        "version": ["Common", "Rare", "Team of the Season", "Icon"],
    })
    out = dataset.add_behaviour_features(frame)
    assert out["is_special"].tolist() == [0.0, 0.0, 1.0, 1.0]
