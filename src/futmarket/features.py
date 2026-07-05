"""Phase 1 feature engine.

Turns raw price_snapshots into signal-ready features that reflect *why* FUT
prices move, not just raw momentum:

  pct_change_1h / 24h / 7d   short/medium/long momentum
  rolling_mean_24h, _std_24h current price's recent norm and dispersion
  z_score                    how many std devs the current price is from norm
  days_to_next_event         distance to the next TOTW/SBC/PROMO for this card
  next_event_type            type of that event
  is_weekend_window          Thu-Sun UTC (Weekend League demand)

The per-series helpers are pure functions over a pandas Series so the maths is
unit-testable against hand calculation (Phase 1 definition of done).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from . import db

H1 = pd.Timedelta(hours=1)
H24 = pd.Timedelta(hours=24)
D7 = pd.Timedelta(days=7)
WEEKEND_DAYS = {3, 4, 5, 6}  # Thu=3 .. Sun=6 (Python weekday())


@dataclass(frozen=True)
class FeatureRow:
    player_id: str
    source: str
    timestamp: str
    price: int
    pct_change_1h: float | None
    pct_change_24h: float | None
    pct_change_7d: float | None
    rolling_mean_24h: float | None
    rolling_std_24h: float | None
    z_score: float | None
    days_to_next_event: int | None
    next_event_type: str | None
    is_weekend_window: int


def to_series(rows) -> pd.Series:
    """rows of {timestamp, price} -> ascending float Series indexed by UTC time.
    Collapses any duplicate timestamps to their last value."""
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([r["timestamp"] for r in rows], utc=True)
    s = pd.Series([float(r["price"]) for r in rows], index=idx).sort_index()
    return s[~s.index.duplicated(keep="last")]


def price_asof(s: pd.Series, target: pd.Timestamp,
               max_stale: pd.Timedelta) -> float | None:
    """Last price at or before `target`, provided it is no older than max_stale."""
    sub = s[s.index <= target]
    if sub.empty:
        return None
    ts = sub.index[-1]
    if target - ts > max_stale:
        return None
    return float(sub.iloc[-1])


def pct_change(s: pd.Series, price: float, at: pd.Timestamp,
               horizon: pd.Timedelta) -> float | None:
    """Percent change vs the price one `horizon` ago (as-of, within one horizon
    of staleness). None when there is no comparable earlier point."""
    past = price_asof(s, at - horizon, max_stale=horizon)
    if past is None or past == 0:
        return None
    return (price / past - 1.0) * 100.0


def trailing_window(s: pd.Series, at: pd.Timestamp,
                    window: pd.Timedelta) -> pd.Series:
    """Points in (at - window, at]."""
    return s[(s.index > at - window) & (s.index <= at)]


def zscore(price: float, mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or std == 0:
        return None
    return (price - mean) / std


def is_weekend_window(at: pd.Timestamp | datetime) -> int:
    return 1 if at.weekday() in WEEKEND_DAYS else 0


def _next_event(conn, player_id: str, at: pd.Timestamp) -> tuple[int | None, str | None]:
    on_date = at.date().isoformat()
    events = db.upcoming_events(conn, player_id, on_date)
    if not events:
        return None, None
    nearest = events[0]
    delta = (pd.Timestamp(nearest["start_date"]).date() - at.date()).days
    return delta, nearest["event_type"]


def compute_feature_table(conn, player_id: str, source: str) -> list[FeatureRow]:
    """One FeatureRow per snapshot of (player_id, source), oldest first."""
    rows = db.snapshots(conn, player_id, source)
    s = to_series(rows)
    out: list[FeatureRow] = []
    for at, price in s.items():
        window = trailing_window(s, at, H24)
        mean = float(window.mean()) if len(window) >= 1 else None
        std = float(window.std(ddof=1)) if len(window) >= 2 else None
        days, ev_type = _next_event(conn, player_id, at)
        out.append(FeatureRow(
            player_id=player_id,
            source=source,
            timestamp=at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            price=int(price),
            pct_change_1h=pct_change(s, price, at, H1),
            pct_change_24h=pct_change(s, price, at, H24),
            pct_change_7d=pct_change(s, price, at, D7),
            rolling_mean_24h=mean,
            rolling_std_24h=std,
            z_score=zscore(price, mean, std),
            days_to_next_event=days,
            next_event_type=ev_type,
            is_weekend_window=is_weekend_window(at),
        ))
    return out


def build_and_store(conn, config, source: str) -> int:
    """Compute + persist features for every watchlist player.
    Returns the number of feature rows written."""
    from .services import watch
    written = 0
    for entry in watch.effective_entries(conn, config):
        for fr in compute_feature_table(conn, entry.player_id, source):
            db.upsert_features(conn, asdict(fr))
            written += 1
    conn.commit()
    return written
