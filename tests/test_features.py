"""Feature math, spot-checked against hand calculation (Phase 1 DoD)."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from futmarket import db as futdb
from futmarket import features as F


def _series(pairs):
    """pairs of (ISO-Z timestamp, price) -> pandas Series like to_series expects."""
    return F.to_series([{"timestamp": t, "price": p} for t, p in pairs])


def test_pct_change_1h_hand_calc():
    s = _series([("2026-07-01T10:00:00Z", 100_000),
                 ("2026-07-01T11:00:00Z", 110_000)])
    at = pd.Timestamp("2026-07-01T11:00:00Z")
    assert F.pct_change(s, 110_000, at, F.H1) == pytest.approx(10.0)


def test_pct_change_none_when_no_comparable_point():
    s = _series([("2026-07-01T11:00:00Z", 110_000)])
    at = pd.Timestamp("2026-07-01T11:00:00Z")
    assert F.pct_change(s, 110_000, at, F.H1) is None       # nothing 1h back
    assert F.pct_change(s, 110_000, at, F.D7) is None       # nothing 7d back


def test_rolling_mean_std_zscore_hand_calc():
    # trailing-24h window holds exactly [90k, 100k, 110k]; mean 100k, sample std 10k
    s = _series([("2026-07-01T12:00:00Z", 90_000),
                 ("2026-07-01T13:00:00Z", 100_000),
                 ("2026-07-01T14:00:00Z", 110_000)])
    at = pd.Timestamp("2026-07-01T14:00:00Z")
    window = F.trailing_window(s, at, F.H24)
    mean = float(window.mean())
    std = float(window.std(ddof=1))
    assert mean == pytest.approx(100_000)
    assert std == pytest.approx(10_000)
    assert F.zscore(110_000, mean, std) == pytest.approx(1.0)


def test_rolling_window_excludes_points_older_than_24h():
    s = _series([("2026-07-01T13:00:00Z", 50_000),   # 25h before -> excluded
                 ("2026-07-02T10:00:00Z", 100_000),
                 ("2026-07-02T14:00:00Z", 120_000)])
    at = pd.Timestamp("2026-07-02T14:00:00Z")
    window = F.trailing_window(s, at, F.H24)
    assert list(window.values) == [100_000, 120_000]


def test_zscore_none_when_std_zero_or_missing():
    assert F.zscore(100, 100, 0) is None
    assert F.zscore(100, None, 5) is None
    assert F.zscore(100, 100, None) is None


def test_price_asof_respects_staleness():
    s = _series([("2026-07-01T00:00:00Z", 500)])
    # target 30h later, 24h staleness cap -> too old
    assert F.price_asof(s, pd.Timestamp("2026-07-02T06:00:00Z"), F.H24) is None
    assert F.price_asof(s, pd.Timestamp("2026-07-01T12:00:00Z"), F.H24) == 500


def test_is_weekend_window():
    # 2026-07-02 is a Thursday (in window); 2026-07-01 a Wednesday (not)
    assert F.is_weekend_window(pd.Timestamp("2026-07-02T09:00:00Z")) == 1
    assert F.is_weekend_window(pd.Timestamp("2026-07-01T09:00:00Z")) == 0


def test_compute_feature_table_end_to_end(config, conn):
    base = [("2026-07-01T12:00:00Z", 90_000),
            ("2026-07-01T13:00:00Z", 100_000),
            ("2026-07-01T14:00:00Z", 110_000)]
    for t, p in base:
        at = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        futdb.insert_snapshot(conn, player_id="p1", price=p, source="mock", at=at)
    futdb.insert_event(conn, event_type="SBC", start_date="2026-07-04",
                       player_id="p1", notes="test")
    conn.commit()

    table = F.compute_feature_table(conn, "p1", "mock")
    latest = table[-1]
    assert latest.price == 110_000
    assert latest.z_score == pytest.approx(1.0)
    assert latest.pct_change_1h == pytest.approx(10.0)
    assert latest.days_to_next_event == 3          # 07-01 -> 07-04
    assert latest.next_event_type == "SBC"


def test_build_and_store_is_idempotent(config, conn):
    at = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    pid = config.watchlist[0].player_id
    futdb.insert_snapshot(conn, player_id=pid, price=100_000, source="mock", at=at)
    conn.commit()
    from futmarket import features
    n1 = features.build_and_store(conn, config, "mock")
    count1 = conn.execute("SELECT COUNT(*) AS n FROM features").fetchone()["n"]
    n2 = features.build_and_store(conn, config, "mock")
    count2 = conn.execute("SELECT COUNT(*) AS n FROM features").fetchone()["n"]
    assert n1 == n2 and count1 == count2  # rebuild overwrites, never duplicates
