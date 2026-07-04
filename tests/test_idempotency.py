"""Re-running collection must never duplicate snapshots (brief §7)."""

from datetime import datetime, timedelta, timezone

from futmarket import db as futdb
from futmarket.collectors.base import PriceQuote, SourceError
from futmarket.scheduler import run_pass


class FixedSource:
    name = "fixed"

    def __init__(self, price=10_000, at=None):
        self.at = at or datetime.now(timezone.utc)
        self.price = price

    def fetch_price(self, player, platform):
        return PriceQuote(player_id=player.player_id, price=self.price,
                          source=self.name, fetched_at=self.at)


class FailingSource:
    name = "failing"

    def fetch_price(self, player, platform):
        raise SourceError("boom")


def snapshot_count(conn):
    return conn.execute("SELECT COUNT(*) AS n FROM price_snapshots").fetchone()["n"]


def test_double_pass_creates_no_duplicates(config, conn):
    source = FixedSource()
    first = run_pass(config, conn, source, sleep=lambda s: None)
    assert len(first.collected) == 3
    assert snapshot_count(conn) == 3

    second = run_pass(config, conn, source, sleep=lambda s: None)
    assert second.collected == []
    assert len(second.skipped_fresh) == 3
    assert snapshot_count(conn) == 3


def test_same_minute_insert_is_ignored(conn):
    at = datetime(2026, 7, 4, 12, 0, 15, tzinfo=timezone.utc)
    same_minute = datetime(2026, 7, 4, 12, 0, 59, tzinfo=timezone.utc)
    assert futdb.insert_snapshot(conn, player_id="p1", price=100, source="s", at=at)
    assert not futdb.insert_snapshot(conn, player_id="p1", price=105, source="s",
                                     at=same_minute)
    assert snapshot_count(conn) == 1


def test_skip_guard_releases_after_interval(config, conn):
    start = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    run_pass(config, conn, FixedSource(at=start), sleep=lambda s: None,
             now=lambda: start)

    # 46 min later (> poll interval): guard releases, new snapshots land
    later = start + timedelta(minutes=46)
    result = run_pass(config, conn, FixedSource(at=later), sleep=lambda s: None,
                      now=lambda: later)
    assert len(result.collected) == 3
    assert snapshot_count(conn) == 6


def test_circuit_breaker_aborts_pass(config, conn):
    result = run_pass(config, conn, FailingSource(), sleep=lambda s: None)
    assert result.aborted_by_breaker
    assert len(result.failed) == 3  # max_consecutive_failures
    assert snapshot_count(conn) == 0


def test_single_failure_does_not_stop_others(config, conn):
    class FlakyForP2(FixedSource):
        name = "flaky"

        def fetch_price(self, player, platform):
            if player.player_id == "p2":
                raise SourceError("timeout")
            return super().fetch_price(player, platform)

    result = run_pass(config, conn, FlakyForP2(), sleep=lambda s: None)
    assert result.failed == ["p2"]
    assert sorted(result.collected) == ["p1", "p3"]
    assert not result.aborted_by_breaker
