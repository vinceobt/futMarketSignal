"""SQLite storage. price_snapshots is append-only; history is never overwritten."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
  player_id     TEXT PRIMARY KEY,
  name          TEXT,
  rating        INTEGER,
  position      TEXT,
  version       TEXT,
  platform      TEXT
);

CREATE TABLE IF NOT EXISTS price_snapshots (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id     TEXT REFERENCES players(player_id),
  timestamp     DATETIME NOT NULL,
  price         INTEGER NOT NULL,
  source        TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_unique
  ON price_snapshots(player_id, source, timestamp);

CREATE TABLE IF NOT EXISTS market_events (
  event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type    TEXT NOT NULL,
  player_id     TEXT NULL,
  start_date    DATE NOT NULL,
  end_date      DATE,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS signals (
  signal_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id     TEXT REFERENCES players(player_id),
  timestamp     DATETIME NOT NULL,
  signal_type   TEXT NOT NULL,
  confidence    REAL,
  reason        TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def bucket_timestamp(dt: datetime) -> str:
    """UTC ISO-8601 truncated to the minute — the idempotency grain."""
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def upsert_player(conn: sqlite3.Connection, *, player_id: str, name: str,
                  rating: int, position: str, version: str, platform: str) -> None:
    conn.execute(
        """INSERT INTO players (player_id, name, rating, position, version, platform)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(player_id) DO UPDATE SET
             name=excluded.name, rating=excluded.rating, position=excluded.position,
             version=excluded.version, platform=excluded.platform""",
        (player_id, name, rating, position, version, platform),
    )


def insert_snapshot(conn: sqlite3.Connection, *, player_id: str, price: int,
                    source: str, at: datetime) -> bool:
    """Append a snapshot. Returns False if this (player, source, minute) already exists."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO price_snapshots (player_id, timestamp, price, source) "
        "VALUES (?, ?, ?, ?)",
        (player_id, bucket_timestamp(at), int(price), source),
    )
    return cur.rowcount == 1


def latest_snapshot_time(conn: sqlite3.Connection, player_id: str,
                         source: str) -> datetime | None:
    row = conn.execute(
        "SELECT MAX(timestamp) AS ts FROM price_snapshots WHERE player_id=? AND source=?",
        (player_id, source),
    ).fetchone()
    if row is None or row["ts"] is None:
        return None
    return datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def history(conn: sqlite3.Connection, player_id: str, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT timestamp, price, source FROM price_snapshots "
        "WHERE player_id=? ORDER BY timestamp DESC LIMIT ?",
        (player_id, limit),
    ).fetchall()
