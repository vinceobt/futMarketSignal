"""ML-warehouse schema + helpers: registry, liquidity, calendar, predictions, models."""

import sqlite3
from datetime import datetime, timezone

import pytest

from futmarket import db as futdb


UTC = timezone.utc


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_new_tables_created(conn):
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"card_meta", "liquidity", "game_calendar", "predictions",
            "model_runs"} <= tables


def test_title_column_present_on_fresh_db(conn):
    assert "title" in _cols(conn, "players")
    assert "title" in _cols(conn, "market_events")


def test_migration_adds_title_to_legacy_db(tmp_path):
    """A DB created before the rebuild (no `title` column) gets it added, defaulting
    to the current game, without losing existing rows."""
    db_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE players (player_id TEXT PRIMARY KEY, name TEXT)")
    raw.execute("CREATE TABLE market_events (event_id INTEGER PRIMARY KEY, "
                "event_type TEXT, start_date DATE)")
    raw.execute("INSERT INTO players (player_id, name) VALUES ('p1', 'Old Card')")
    raw.commit()
    raw.close()

    conn = futdb.connect(db_path)
    assert "title" in _cols(conn, "players")
    row = conn.execute("SELECT title FROM players WHERE player_id='p1'").fetchone()
    assert row["title"] == "fc26"  # existing rows back-labelled to the current game


def test_migration_is_idempotent(conn):
    # connect() already ran once via the fixture; running again must not error.
    futdb._migrate(conn)
    assert "title" in _cols(conn, "players")


def test_card_meta_upsert_and_registry(conn):
    futdb.upsert_card_meta(conn, {
        "player_id": "231747-mbappe", "definition_id": 231747, "name": "Mbappe",
        "rating": 91, "position": "ST", "league": "La Liga", "nation": "France",
        "club": "Real Madrid", "version": "gold", "tradeable": 1,
    })
    futdb.upsert_card_meta(conn, {
        "player_id": "sbc-only", "definition_id": 999, "name": "Fodder",
        "rating": 84, "position": "CB", "tradeable": 0,
    })
    conn.commit()

    got = futdb.card_meta_get(conn, "231747-mbappe")
    assert got["rating"] == 91 and got["title"] == "fc26"

    tradeable = futdb.card_registry(conn, title="fc26", tradeable_only=True)
    ids = {r["player_id"] for r in tradeable}
    assert "231747-mbappe" in ids and "sbc-only" not in ids

    assert futdb.card_count(conn) == 2
    assert futdb.card_count(conn, tradeable_only=True) == 1


def test_card_meta_upsert_updates(conn):
    futdb.upsert_card_meta(conn, {"player_id": "c1", "rating": 84})
    futdb.upsert_card_meta(conn, {"player_id": "c1", "rating": 85, "name": "Grown"})
    conn.commit()
    got = futdb.card_meta_get(conn, "c1")
    assert got["rating"] == 85 and got["name"] == "Grown"


def test_liquidity_upsert_and_tier(conn):
    futdb.upsert_liquidity(conn, player_id="a", score=9.1, tier="A", price=200000,
                           updates_per_day=12.0)
    futdb.upsert_liquidity(conn, player_id="b", score=2.0, tier="C", price=800)
    conn.commit()

    assert futdb.liquidity_get(conn, "a")["tier"] == "A"
    a_tier = futdb.liquidity_by_tier(conn, "A")
    assert [r["player_id"] for r in a_tier] == ["a"]

    # upsert overwrites in place (one row per card)
    futdb.upsert_liquidity(conn, player_id="a", score=1.0, tier="C")
    conn.commit()
    assert futdb.liquidity_get(conn, "a")["tier"] == "C"
    assert futdb.liquidity_by_tier(conn, "A") == []


def test_game_calendar(conn):
    futdb.set_game_launch(conn, title="fc26", launch_date="2025-09-26",
                          notes="EA FC 26 global launch")
    conn.commit()
    assert futdb.game_launch(conn, "fc26") == "2025-09-26"
    assert futdb.game_launch(conn, "fc27") is None


def test_predictions_insert_and_idempotency(conn):
    run_id = futdb.create_model_run(conn, kind="forecast", horizon_h=72, n_samples=1000)
    at = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
    futdb.insert_prediction(conn, subject_id="c1", level="card", kind="forecast",
                            horizon_h=72, at=at, run_id=run_id, yhat=210000,
                            yhat_lo=195000, yhat_hi=228000)
    # same key again with new values → update, not duplicate
    futdb.insert_prediction(conn, subject_id="c1", level="card", kind="forecast",
                            horizon_h=72, at=at, run_id=run_id, yhat=215000,
                            yhat_lo=200000, yhat_hi=232000)
    conn.commit()

    rows = futdb.latest_predictions(conn, level="card", kind="forecast")
    assert len(rows) == 1
    assert rows[0]["yhat"] == 215000 and rows[0]["run_id"] == run_id


def test_cohort_prediction_level(conn):
    run_id = futdb.create_model_run(conn, kind="direction", level="cohort")
    at = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
    futdb.insert_prediction(conn, subject_id="rating:84", level="cohort",
                            kind="direction", horizon_h=48, at=at, run_id=run_id,
                            proba=0.72)
    conn.commit()
    rows = futdb.latest_predictions(conn, level="cohort")
    assert rows[0]["subject_id"] == "rating:84" and rows[0]["proba"] == pytest.approx(0.72)


def test_model_runs_registry(conn):
    r1 = futdb.create_model_run(conn, kind="forecast", horizon_h=24,
                                metrics_json='{"mae": 5000}')
    r2 = futdb.create_model_run(conn, kind="forecast", horizon_h=24,
                                metrics_json='{"mae": 4200}')
    assert r2 > r1
    latest = futdb.latest_model_run(conn, kind="forecast")
    assert latest["run_id"] == r2 and latest["metrics_json"] == '{"mae": 4200}'
    assert len(futdb.model_runs_list(conn, kind="forecast")) == 2
