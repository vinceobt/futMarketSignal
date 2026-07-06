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


# ---- robust, duration-weighted statistics ----

def test_weighted_quantile_hand_calc():
    # values 1,2,3 with weights 1,1,2 (total 4): the step CDF hits
    # 0.5*4=2 at value 2 and 0.75*4=3 at value 3
    assert F.weighted_quantile([1, 2, 3], [1, 1, 2], 0.5) == 2
    assert F.weighted_quantile([1, 2, 3], [1, 1, 2], 0.75) == 3
    assert F.weighted_quantile([1, 2, 3], [1, 1, 2], 0.10) == 1
    assert F.weighted_quantile([1, 2, 3], [0, 0, 0], 0.5) is None


def test_duration_weights_hold_until_next_sample():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-01T00:00:00Z"),
                            pd.Timestamp("2026-07-01T01:00:00Z"),
                            pd.Timestamp("2026-07-01T03:00:00Z")])
    w = F.duration_weights(idx, pd.Timestamp("2026-07-01T04:00:00Z"))
    assert list(w) == [3600.0, 7200.0, 3600.0]
    # the sample exactly at `at` holds for 0s -> weight 0 (self-exclusion)
    w = F.duration_weights(idx, idx[-1])
    assert w[-1] == 0.0
    # a week-long stale plateau is capped at max_gap_hours
    idx2 = pd.DatetimeIndex([pd.Timestamp("2026-07-01T00:00:00Z"),
                             pd.Timestamp("2026-07-08T00:00:00Z")])
    w2 = F.duration_weights(idx2, idx2[-1], max_gap_hours=48.0)
    assert w2[0] == 48.0 * 3600.0


def test_robust_stats_invariant_to_sampling_density():
    # the same price path — one day at 100 then one day at 200 — sampled
    # sparsely (2 points) and densely (hourly) must yield identical stats
    at = pd.Timestamp("2026-07-03T00:00:00Z")
    sparse = _series([("2026-07-01T00:00:00Z", 100), ("2026-07-02T00:00:00Z", 200)])
    dense_pairs = ([(f"2026-07-01T{h:02d}:00:00Z", 100) for h in range(24)]
                   + [(f"2026-07-02T{h:02d}:00:00Z", 200) for h in range(24)])
    dense = _series(dense_pairs)
    st_s = F.robust_stats(sparse, at)
    st_d = F.robust_stats(dense, at)
    assert st_s.median == st_d.median
    assert st_s.mad == st_d.mad


def test_robust_z_and_cv():
    at = pd.Timestamp("2026-07-03T00:00:00Z")
    s = _series([("2026-07-01T00:00:00Z", 100), ("2026-07-02T00:00:00Z", 200)])
    st = F.robust_stats(s, at)
    # step-CDF weighted median of {100(24h), 200(24h)} = 100; MAD likewise
    assert st.median == 100 and st.mad == 0
    assert F.robust_z(150, st) is None      # flat MAD -> no z
    flat = _series([("2026-07-01T00:00:00Z", 100), ("2026-07-02T00:00:00Z", 100)])
    st_flat = F.robust_stats(flat, at)
    assert F.robust_cv(st_flat) == 0.0


def test_robust_median_excludes_the_current_point():
    # 2 held samples at 100, then a crash to 50 exactly at `at`: the crash
    # carries no holding time yet, so the reference stats ignore it
    s = _series([("2026-07-01T00:00:00Z", 100), ("2026-07-01T12:00:00Z", 100),
                 ("2026-07-02T00:00:00Z", 50)])
    st = F.robust_stats(s, pd.Timestamp("2026-07-02T00:00:00Z"))
    assert st.median == 100
    assert st.n_eff == 2


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
