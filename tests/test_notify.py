"""The Discord run-summary: composed from the DB, no network in the builder."""

from datetime import datetime, timezone

from futmarket import db as futdb
from futmarket.services import notify

UTC = timezone.utc
NOW = datetime(2026, 7, 21, 9, 0, tzinfo=UTC)


def _card(conn, pid, name, **kw):
    futdb.upsert_card_meta(conn, {"player_id": pid, "name": name, "tradeable": 1,
                                  "rating": kw.get("rating", 90),
                                  "version": kw.get("version"),
                                  "url": kw.get("url")})


def test_summary_lists_latest_picks_and_record(conn):
    _card(conn, "c1", "Erling Haaland", rating=91, version="Team of the Season",
          url="https://fut.gg/x")
    futdb.insert_pick(conn, player_id="c1", entry_price=100000, target_price=130000,
                      stop_price=92000, horizon_days=5, at=NOW, confidence=0.72,
                      buy_low=100000, buy_high=105000, sales_per_hour=14.0,
                      reasons="near floor")
    conn.commit()
    text = notify.build_run_summary(conn)
    assert "Erling Haaland" in text
    assert "Team of the Season" in text          # exact card is identified
    assert "https://fut.gg/x" in text            # link so you buy the right one
    assert "FUT robot ran" in text
    assert "Record" in text


def test_summary_handles_a_quiet_run(conn):
    """No picks at all still produces a sendable, honest message."""
    conn.commit()
    text = notify.build_run_summary(conn)
    assert "No new buys" in text
    assert "none graded yet" in text
