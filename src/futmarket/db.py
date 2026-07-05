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

-- Phase 1 output: one row per (player, source, snapshot). Derived, so it is
-- safe to rebuild from price_snapshots at any time.
CREATE TABLE IF NOT EXISTS features (
  player_id            TEXT NOT NULL,
  source               TEXT NOT NULL,
  timestamp            DATETIME NOT NULL,
  price                INTEGER NOT NULL,
  pct_change_1h        REAL,
  pct_change_24h       REAL,
  pct_change_7d        REAL,
  rolling_mean_24h     REAL,
  rolling_std_24h      REAL,
  z_score              REAL,
  days_to_next_event   INTEGER,
  next_event_type      TEXT,
  is_weekend_window    INTEGER NOT NULL,
  PRIMARY KEY (player_id, source, timestamp)
);

-- The tracked player list (the "watchlist"). App-managed data, kept out of the
-- settings file so add/remove is a transactional row op, not a text rewrite.
-- Seeded once from config.yaml on first run (see services/watch.py).
CREATE TABLE IF NOT EXISTS watchlist (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,   -- preserves insertion order
  player_id  TEXT NOT NULL UNIQUE,
  name       TEXT NOT NULL,
  url        TEXT NOT NULL,
  added_at   DATETIME NOT NULL
);

-- Small key/value store for app-level flags (e.g. the one-time watchlist seed).
CREATE TABLE IF NOT EXISTS app_meta (
  key    TEXT PRIMARY KEY,
  value  TEXT
);

-- Cached snapshot of fut.gg's market-wide momentum scanner (top movers). A heavy
-- fetch, so it's refreshed on demand and the whole table is replaced each time.
CREATE TABLE IF NOT EXISTS momentum (
  rank       INTEGER PRIMARY KEY,   -- 1 = biggest mover
  player_id  TEXT NOT NULL,
  name       TEXT NOT NULL,
  url        TEXT NOT NULL,
  price      INTEGER NOT NULL,
  momentum   REAL NOT NULL,
  rating     INTEGER,
  position   TEXT,
  rarity     TEXT
);

-- Paper positions opened by the rebound advisor. One open row per player at a
-- time; this both de-dupes alerts (no repeat BUY while holding) and remembers the
-- entry price so the SELL target can be computed. Never a real trade.
CREATE TABLE IF NOT EXISTS positions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id     TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'open',   -- open | closed
  entry_price   INTEGER NOT NULL,
  entry_ts      DATETIME NOT NULL,
  target_price  INTEGER NOT NULL,
  stop_price    INTEGER,
  floor_price   INTEGER,
  exit_price    INTEGER,
  exit_ts       DATETIME,
  realized_pct  REAL,
  reason        TEXT
);

-- Background jobs triggered from the dashboard (scrape passes, backtests, …).
-- The web layer enqueues rows here; the worker thread runs them and updates
-- status/progress/result so the UI can poll.
CREATE TABLE IF NOT EXISTS jobs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  type          TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|failed|cancelled
  detail        TEXT,                            -- short human status line
  progress      INTEGER NOT NULL DEFAULT 0,
  total         INTEGER NOT NULL DEFAULT 0,
  result_json   TEXT,
  log           TEXT NOT NULL DEFAULT '',
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL,
  finished_at   DATETIME
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # WAL lets the web process read while the worker thread writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn


# ---- jobs -----------------------------------------------------------------

def create_job(conn: sqlite3.Connection, *, job_type: str, total: int = 0,
               detail: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO jobs (type, status, detail, total, created_at) "
        "VALUES (?, 'queued', ?, ?, ?)",
        (job_type, detail, total, bucket_timestamp(datetime.now(timezone.utc))),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_job(conn: sqlite3.Connection, job_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
    conn.commit()


def append_job_log(conn: sqlite3.Connection, job_id: int, line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    conn.execute("UPDATE jobs SET log = log || ? WHERE id=?",
                 (f"[{stamp}] {line}\n", job_id))
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def job_cancel_requested(conn: sqlite3.Connection, job_id: int) -> bool:
    row = conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def request_job_cancel(conn: sqlite3.Connection, job_id: int) -> None:
    conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
    conn.commit()


def list_jobs(conn: sqlite3.Connection, limit: int = 30) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, type, status, detail, progress, total, created_at, finished_at "
        "FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


# ---- app_meta ----

def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


# ---- watchlist ----

def watchlist_add(conn: sqlite3.Connection, *, player_id: str, name: str,
                  url: str, at: datetime) -> bool:
    """Insert a tracked player. Returns False if the player_id is already there."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO watchlist (player_id, name, url, added_at) "
        "VALUES (?, ?, ?, ?)",
        (player_id, name, url, at.astimezone(timezone.utc).isoformat()))
    return cur.rowcount == 1


def watchlist_remove(conn: sqlite3.Connection, player_id: str) -> bool:
    cur = conn.execute("DELETE FROM watchlist WHERE player_id=?", (player_id,))
    return cur.rowcount == 1


def watchlist_entries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT player_id, name, url FROM watchlist ORDER BY seq").fetchall()


def watchlist_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM watchlist").fetchone()["n"]


# ---- momentum cache ----

def momentum_replace(conn: sqlite3.Connection, rows: list) -> None:
    """Replace the cached momentum snapshot with a fresh ranked list of
    MomentumRow-like objects (rank follows list order)."""
    conn.execute("DELETE FROM momentum")
    conn.executemany(
        "INSERT INTO momentum (rank, player_id, name, url, price, momentum, "
        "rating, position, rarity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i + 1, r.player_id, r.name, r.url, r.price, r.momentum,
          r.rating, r.position, r.rarity) for i, r in enumerate(rows)],
    )
    conn.commit()


def momentum_list(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT rank, player_id, name, url, price, momentum, rating, position, "
        "rarity FROM momentum ORDER BY rank").fetchall()


# ---- paper positions (rebound advisor) ----

def position_open(conn: sqlite3.Connection, *, player_id: str, entry_price: int,
                  entry_ts: datetime, target_price: int, stop_price: int | None,
                  floor_price: int | None, reason: str) -> int:
    cur = conn.execute(
        "INSERT INTO positions (player_id, status, entry_price, entry_ts, "
        "target_price, stop_price, floor_price, reason) "
        "VALUES (?, 'open', ?, ?, ?, ?, ?, ?)",
        (player_id, int(entry_price), bucket_timestamp(entry_ts), int(target_price),
         stop_price, floor_price, reason))
    conn.commit()
    return int(cur.lastrowid)


def position_close(conn: sqlite3.Connection, position_id: int, *, exit_price: int,
                   exit_ts: datetime, realized_pct: float, reason: str) -> None:
    conn.execute(
        "UPDATE positions SET status='closed', exit_price=?, exit_ts=?, "
        "realized_pct=?, reason=? WHERE id=?",
        (int(exit_price), bucket_timestamp(exit_ts), realized_pct, reason, position_id))
    conn.commit()


def position_get_open(conn: sqlite3.Connection, player_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM positions WHERE player_id=? AND status='open' "
        "ORDER BY id DESC LIMIT 1", (player_id,)).fetchone()


def positions_list(conn: sqlite3.Connection, status: str | None = None,
                   limit: int = 100) -> list[sqlite3.Row]:
    if status:
        return conn.execute("SELECT * FROM positions WHERE status=? "
                            "ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
    return conn.execute("SELECT * FROM positions ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()


def bucket_timestamp(dt: datetime) -> str:
    """UTC ISO-8601 truncated to the minute — the idempotency grain."""
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def upsert_player(conn: sqlite3.Connection, *, player_id: str, name: str,
                  rating: int | None, position: str, version: str, platform: str) -> None:
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


def snapshots(conn: sqlite3.Connection, player_id: str,
              source: str | None = None) -> list[sqlite3.Row]:
    """Full ascending price series for a player, optionally pinned to one source."""
    if source is None:
        return conn.execute(
            "SELECT timestamp, price, source FROM price_snapshots "
            "WHERE player_id=? ORDER BY timestamp ASC",
            (player_id,),
        ).fetchall()
    return conn.execute(
        "SELECT timestamp, price, source FROM price_snapshots "
        "WHERE player_id=? AND source=? ORDER BY timestamp ASC",
        (player_id, source),
    ).fetchall()


def insert_event(conn: sqlite3.Connection, *, event_type: str, start_date: str,
                 end_date: str | None = None, player_id: str | None = None,
                 notes: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO market_events (event_type, player_id, start_date, end_date, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_type, player_id, start_date, end_date, notes),
    )
    return cur.lastrowid


def upcoming_events(conn: sqlite3.Connection, player_id: str,
                    on_date: str) -> list[sqlite3.Row]:
    """Events on/after on_date that apply to this player (its own or market-wide),
    nearest first."""
    return conn.execute(
        "SELECT event_type, start_date, end_date FROM market_events "
        "WHERE (player_id=? OR player_id IS NULL) AND start_date >= ? "
        "ORDER BY start_date ASC",
        (player_id, on_date),
    ).fetchall()


def insert_signal(conn: sqlite3.Connection, *, player_id: str, signal_type: str,
                  confidence: float, reason: str, at: datetime) -> int:
    cur = conn.execute(
        "INSERT INTO signals (player_id, timestamp, signal_type, confidence, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (player_id, bucket_timestamp(at), signal_type, confidence, reason),
    )
    return cur.lastrowid


def latest_signals(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT player_id, timestamp, signal_type, confidence, reason FROM signals "
        "ORDER BY timestamp DESC, signal_id DESC LIMIT ?", (limit,),
    ).fetchall()


def upsert_features(conn: sqlite3.Connection, row: dict) -> None:
    cols = ["player_id", "source", "timestamp", "price", "pct_change_1h",
            "pct_change_24h", "pct_change_7d", "rolling_mean_24h", "rolling_std_24h",
            "z_score", "days_to_next_event",
            "next_event_type", "is_weekend_window"]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in
                        ("player_id", "source", "timestamp"))
    conn.execute(
        f"INSERT INTO features ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(player_id, source, timestamp) DO UPDATE SET {updates}",
        tuple(row[c] for c in cols),
    )
