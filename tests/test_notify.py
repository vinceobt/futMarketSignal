"""The Discord run-summary: composed from the DB, no network in the builder."""

from datetime import datetime, timezone

from futmarket import db as futdb
from futmarket.ml import picks
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


# ---- sell alerts ----------------------------------------------------------

def _hold(conn, pid, *, target, stop, price, strategy=picks.STRATEGY_VERSION):
    futdb.upsert_card_meta(conn, {"player_id": pid, "name": pid, "rating": 90})
    futdb.insert_pick(conn, player_id=pid, entry_price=10_000, target_price=target,
                      stop_price=stop, horizon_days=5, at=NOW, strategy=strategy)
    futdb.insert_snapshot(conn, player_id=pid, price=price, source="futgg", at=NOW)
    conn.commit()


def test_sell_alert_fires_once_at_target(conn, monkeypatch):
    sent = []
    monkeypatch.setattr("futmarket.alerts.DiscordAlerter.send",
                        lambda self, text: sent.append(text))
    _hold(conn, "c1", target=12_000, stop=9_200, price=12_500)
    assert notify.sell_alerts(conn, "http://wh", source="futgg") == 1
    assert len(sent) == 1 and "SELL" in sent[0]
    # already alerted -> does not fire again
    assert notify.sell_alerts(conn, "http://wh", source="futgg") == 0


def test_sell_alert_cuts_at_stop(conn, monkeypatch):
    sent = []
    monkeypatch.setattr("futmarket.alerts.DiscordAlerter.send",
                        lambda self, text: sent.append(text))
    _hold(conn, "c2", target=12_000, stop=9_200, price=9_000)
    assert notify.sell_alerts(conn, "http://wh", source="futgg") == 1
    assert "CUT" in sent[0]


def test_sell_alert_ignores_legacy_and_mid_positions(conn, monkeypatch):
    sent = []
    monkeypatch.setattr("futmarket.alerts.DiscordAlerter.send",
                        lambda self, text: sent.append(text))
    _hold(conn, "old", target=12_000, stop=9_200, price=13_000, strategy="legacy")
    _hold(conn, "mid", target=12_000, stop=9_200, price=10_500)   # between -> hold
    assert notify.sell_alerts(conn, "http://wh", source="futgg") == 0
    assert sent == []
