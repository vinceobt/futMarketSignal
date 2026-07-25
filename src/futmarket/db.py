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
  signal_type   TEXT NOT NULL,   -- BUY | SELL | HOLD (holding, no exit) | SKIP (gates failed)
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

-- ===========================================================================
-- ML warehouse (the "trading guru" rebuild). See the approved plan:
-- continuously-learning model over the whole card market. These tables are
-- additive; the collector/advisor above keep working unchanged.
-- ===========================================================================

-- The full card universe (the "registry"): every card that exists in a game
-- title, with the metadata cohorts are built from (rating/position/league/…).
-- definition_id is the EA id (numeric part of a fut.gg URL) and the join key to
-- FUTNext prices. player_id matches players.player_id where the two overlap.
CREATE TABLE IF NOT EXISTS card_meta (
  player_id     TEXT PRIMARY KEY,
  definition_id INTEGER,
  title         TEXT NOT NULL DEFAULT 'fc26',
  name          TEXT,
  rating        INTEGER,
  position      TEXT,
  league        TEXT,
  nation        TEXT,
  club          TEXT,
  version       TEXT,                        -- rarity / promo type (gold, TOTW, TOTS, …)
  price_band    TEXT,                        -- coarse tier bucket (filled by the liquidity pass)
  release_date  DATE,
  tradeable     INTEGER NOT NULL DEFAULT 1,  -- 0 = untradeable / SBC-only / extinct
  url           TEXT,
  updated_at    DATETIME
);
CREATE INDEX IF NOT EXISTS idx_card_meta_defid  ON card_meta(definition_id);
CREATE INDEX IF NOT EXISTS idx_card_meta_cohort ON card_meta(title, rating, position);

-- Per-card tradeability. Higher score = easier to sell; tier A/B/C drives how
-- often the collector polls the card. Rebuilt from price activity + metadata.
CREATE TABLE IF NOT EXISTS liquidity (
  player_id       TEXT PRIMARY KEY,
  title           TEXT NOT NULL DEFAULT 'fc26',
  score           REAL NOT NULL,
  tier            TEXT NOT NULL,             -- A | B | C
  updates_per_day REAL,                      -- price-change frequency proxy
  price           INTEGER,                   -- last price (for banding)
  spread_pct      REAL,                      -- top-5 dispersion where available
  computed_at     DATETIME NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_liquidity_tier ON liquidity(title, tier);

-- What the crowd is saying about a card. Volume is the real signal: a player
-- being discussed ten times more than usual is information regardless of whether
-- the talk reads positive. Sentiment is a crude bullish/bearish lean from trader
-- vocabulary ("invest", "dump"), which in this domain beats a general-purpose
-- model that would misread "crash" inside a card name.
CREATE TABLE IF NOT EXISTS sentiment_signals (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id     TEXT,                    -- NULL = market-wide chatter
  platform      TEXT NOT NULL,           -- reddit | youtube | x
  timestamp     DATETIME NOT NULL,
  mention_count INTEGER NOT NULL,
  sentiment     REAL,                    -- -1 bearish .. +1 bullish
  buzz_z        REAL,                    -- how anomalous vs this card's own baseline
  raw_ref       TEXT
);
CREATE INDEX IF NOT EXISTS idx_sentiment_player ON sentiment_signals(player_id, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sentiment_unique
  ON sentiment_signals(player_id, platform, timestamp);

-- What a card ACTUALLY changed hands for, from fut.gg's completed-auction feed.
-- The lowest listing is often a mispriced snipe nobody can realistically catch;
-- these percentiles are the true going rate, and (p25, p75) is the band a buy
-- recommendation should quote ("buy 20k-22k") instead of a false exact price.
-- sales_per_hour is measured trade activity -- a far better liquidity signal
-- than counting how often a listed price changed.
CREATE TABLE IF NOT EXISTS sale_stats (
  player_id       TEXT PRIMARY KEY,
  title           TEXT NOT NULL DEFAULT 'fc26',
  n_sales         INTEGER NOT NULL,
  window_hours    REAL,
  band_from_sales INTEGER,       -- how many recent sales the band came from
  band_window_hours REAL,        -- how recent those sales are
  sales_per_hour  REAL,
  sold_p25        INTEGER,
  sold_median     INTEGER,
  sold_p75        INTEGER,
  listed_price    INTEGER,        -- lowest listing at fetch time, for comparison
  sold_vs_listed  REAL,           -- sold_median / listed_price
  computed_at     DATETIME NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sale_stats_activity ON sale_stats(title, sales_per_hour);

-- Per-title lifecycle anchors (launch date, …). Individual promos/SBCs/TOTW live
-- in market_events (now title-scoped) — this is just the calendar's "day zero".
CREATE TABLE IF NOT EXISTS game_calendar (
  title        TEXT PRIMARY KEY,
  launch_date  DATE NOT NULL,
  notes        TEXT
);

-- Model predictions: one row per (subject, timestamp, model kind, horizon).
-- subject_id is a player_id for card models or a cohort key ("rating:84") for
-- cohort models. Derived + rebuildable.
CREATE TABLE IF NOT EXISTS predictions (
  subject_id  TEXT NOT NULL,
  level       TEXT NOT NULL,                 -- card | cohort
  title       TEXT NOT NULL DEFAULT 'fc26',
  timestamp   DATETIME NOT NULL,
  run_id      INTEGER NOT NULL,
  kind        TEXT NOT NULL,                 -- forecast | direction
  horizon_h   INTEGER NOT NULL,
  yhat        REAL,                          -- forecast point (p50) / expected return
  yhat_lo     REAL,                          -- p10 band
  yhat_hi     REAL,                          -- p90 band
  proba       REAL,                          -- direction classifier probability
  PRIMARY KEY (subject_id, level, kind, horizon_h, timestamp)
);

-- Every recommendation the app has ever made, with what actually happened to it.
-- This is the track record: without it we only ever know whether a pick looked
-- reasonable, never whether it made coins. Scored against the same barriers the
-- model was trained on, so the scorecard and the labels agree.
CREATE TABLE IF NOT EXISTS pick_log (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id      TEXT NOT NULL,
  title          TEXT NOT NULL DEFAULT 'fc26',
  picked_at      DATETIME NOT NULL,
  run_id         INTEGER,
  confidence     REAL,
  entry_price    INTEGER NOT NULL,     -- what you would realistically have paid
  buy_low        INTEGER,
  buy_high       INTEGER,
  target_price   INTEGER NOT NULL,
  stop_price     INTEGER NOT NULL,
  horizon_days   INTEGER NOT NULL,
  sales_per_hour REAL,
  reasons        TEXT,
  status         TEXT NOT NULL DEFAULT 'open',  -- open | target | stop | expired
  exit_price     INTEGER,
  scored_at      DATETIME,
  realized_pct   REAL,
  strategy       TEXT NOT NULL DEFAULT 'legacy', -- which strategy made this pick
  alerted_at     DATETIME,                       -- when a "sell now" alert was sent
  UNIQUE(player_id, picked_at)
);
CREATE INDEX IF NOT EXISTS idx_pick_log_status ON pick_log(status, picked_at);

-- The model registry / training-run log: the "getting smarter over time" record.
-- Each retrain writes a row with its walk-forward metrics + artifact path.
CREATE TABLE IF NOT EXISTS model_runs (
  run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  kind              TEXT NOT NULL,           -- forecast | direction
  level             TEXT NOT NULL DEFAULT 'card',
  title             TEXT NOT NULL DEFAULT 'fc26',
  horizon_h         INTEGER,
  trained_at        DATETIME NOT NULL,
  n_samples         INTEGER,
  metrics_json      TEXT,
  artifact_path     TEXT,
  feature_list_json TEXT,
  git_sha           TEXT
);
"""


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent migrations for DBs created before the ML rebuild.
    New tables are handled by CREATE TABLE IF NOT EXISTS; only column additions
    to pre-existing tables need an explicit ALTER."""
    # Title dimension: the game a row belongs to. Defaults to the current game so
    # existing FC26 data is labelled correctly. Next year's data lands as 'fc27'
    # in the same warehouse — the learning loop never restarts.
    for table in ("players", "market_events"):
        if "title" not in _column_names(conn, table):
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN title TEXT NOT NULL DEFAULT 'fc26'")
    # Where an event came from, so *derived* calendar rows can be rebuilt without
    # touching events a human logged via `futmarket log-event`.
    for col, decl in (("band_from_sales", "INTEGER"), ("band_window_hours", "REAL")):
        if col not in _column_names(conn, "sale_stats"):
            conn.execute(f"ALTER TABLE sale_stats ADD COLUMN {col} {decl}")
    if "source" not in _column_names(conn, "market_events"):
        conn.execute(
            "ALTER TABLE market_events ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
    # Which strategy made a pick. Existing rows are the old rules/+25%-8% engine, so
    # they default to 'legacy' and the honest track record can judge the current
    # strategy on its own without legacy losses dragging the number down.
    if "strategy" not in _column_names(conn, "pick_log"):
        conn.execute(
            "ALTER TABLE pick_log ADD COLUMN strategy TEXT NOT NULL DEFAULT 'legacy'")
        # One-time backfill: the dip strategy uses a 5-day horizon, the old engine
        # used 7. Tag the already-saved dip picks correctly so the first batch isn't
        # blended into legacy.
        conn.execute("UPDATE pick_log SET strategy='dip_v1' WHERE horizon_days=5")
    # When a "sell now" alert was sent for a held pick, so it only fires once.
    if "alerted_at" not in _column_names(conn, "pick_log"):
        conn.execute("ALTER TABLE pick_log ADD COLUMN alerted_at DATETIME")
    # What the market did over the same window, and how long the trade was given.
    # Without a benchmark a track record cannot distinguish a good call in a bad
    # month from a bad call, and in this market the median tradeable card doesn't
    # move at all -- the round trip is the whole obstacle.
    for col, decl in (("benchmark_pct", "REAL"),
                      ("chosen_horizon_days", "INTEGER"),
                      ("market_price", "INTEGER")):
        if col not in _column_names(conn, "pick_log"):
            conn.execute(f"ALTER TABLE pick_log ADD COLUMN {col} {decl}")
    # The dip_v1 picks were graded under a broken convention: the stop was
    # derived from a marked-up entry but compared against the raw price series,
    # so it landed on average 0.76% ABOVE the market price at pick time and 90%
    # of trades stopped out before they began. Those rows are kept for reference
    # but must never appear in a headline -- re-tag them once, by name.
    if conn.execute("SELECT 1 FROM pick_log WHERE strategy='dip_v1' LIMIT 1").fetchone():
        conn.execute("UPDATE pick_log SET strategy='dip_v1_broken' "
                     "WHERE strategy='dip_v1'")
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # WAL lets the web process read while the worker thread writes.
    conn.execute("PRAGMA journal_mode=WAL")
    # Long jobs (a training run, a bulk fetch) legitimately hold the write lock
    # for a while. 5s was short enough that an 11-minute training run could do all
    # its work and then fail on the final one-row insert, losing everything.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.executescript(SCHEMA)
    _migrate(conn)
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


def snapshot_count(conn: sqlite3.Connection, player_id: str, source: str) -> int:
    """How many snapshots a card has for a source — used to tell a backfilled
    card (deep history) from a spot-only one (a handful of points)."""
    return conn.execute(
        "SELECT COUNT(*) AS n FROM price_snapshots WHERE player_id=? AND source=?",
        (player_id, source)).fetchone()["n"]


def history(conn: sqlite3.Connection, player_id: str, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT timestamp, price, source FROM price_snapshots "
        "WHERE player_id=? ORDER BY timestamp DESC LIMIT ?",
        (player_id, limit),
    ).fetchall()


DAILY_OUTLIER_FACTOR = 3.0


def daily_prices(conn: sqlite3.Connection, player_id: str,
                 source: str = "futgg",
                 outlier_factor: float = DAILY_OUTLIER_FACTOR
                 ) -> list[tuple[str, int, int]]:
    """One robust price per day — (date, price, n_snapshots), ascending.

    Grading a trade against every raw tick is what broke the old track record.
    The recorded price is the cheapest live listing at a moment in time, and the
    median card-day *ranges 14.3%* -- so a stop placed anywhere near the price
    gets touched by sampling jitter alone, not by the market. Worse, the training
    labels were computed on daily prices while scoring walked intraday ticks, so
    the model was graded on a harder question than it was ever taught. Both ends
    now read this same series, built the same way as
    ``ml.dataset.load_daily_prices``: drop the bad prints, then take the median.

    (The aggregation is done here rather than in SQL because SQLite has no median
    and the mean is exactly what we are trying to get away from.)
    """
    from statistics import median

    rows = conn.execute(
        """SELECT substr(timestamp, 1, 10) AS date, price
           FROM price_snapshots
           WHERE player_id=? AND source=? AND price>0
           ORDER BY timestamp""",
        (player_id, source)).fetchall()

    by_day: dict[str, list[int]] = {}
    for row in rows:
        by_day.setdefault(row["date"], []).append(int(row["price"]))

    out = []
    for date in sorted(by_day):
        prices = by_day[date]
        rough = median(prices)
        kept = [p for p in prices
                if rough / outlier_factor <= p <= rough * outlier_factor]
        out.append((date, int(round(median(kept or prices))), len(prices)))
    return out


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


def replace_events(conn: sqlite3.Connection, *, source: str,
                   events: list[dict], title: str = "fc26") -> int:
    """Swap in a freshly derived event set for one source, leaving other sources
    (notably hand-logged 'manual' events) untouched. Returns rows written."""
    conn.execute("DELETE FROM market_events WHERE source=? AND title=?",
                 (source, title))
    conn.executemany(
        "INSERT INTO market_events (event_type, player_id, start_date, end_date, "
        "notes, title, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(e["event_type"], e.get("player_id"), e["start_date"], e.get("end_date"),
          e.get("notes"), title, source) for e in events],
    )
    conn.commit()
    return len(events)


def events_list(conn: sqlite3.Connection, *, title: str | None = None,
                event_type: str | None = None,
                limit: int | None = None) -> list[sqlite3.Row]:
    clauses, params = [], []
    if title is not None:
        clauses.append("title=?")
        params.append(title)
    if event_type is not None:
        clauses.append("event_type=?")
        params.append(event_type)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM market_events {where} ORDER BY start_date DESC, event_id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, tuple(params)).fetchall()


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


# ===========================================================================
# ML warehouse helpers (card registry, liquidity, calendar, predictions, models)
# ===========================================================================

# ---- card_meta (the registry / card universe) ----

_CARD_META_COLS = ["player_id", "definition_id", "title", "name", "rating",
                   "position", "league", "nation", "club", "version",
                   "price_band", "release_date", "tradeable", "url", "updated_at"]


def upsert_card_meta(conn: sqlite3.Connection, row: dict) -> None:
    """Insert/update one card in the registry. Missing keys default to NULL
    (tradeable defaults to 1, updated_at to now)."""
    data = {c: row.get(c) for c in _CARD_META_COLS}
    if data["title"] is None:
        data["title"] = "fc26"
    if data["tradeable"] is None:
        data["tradeable"] = 1
    if data["updated_at"] is None:
        data["updated_at"] = bucket_timestamp(datetime.now(timezone.utc))
    placeholders = ", ".join("?" for _ in _CARD_META_COLS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in _CARD_META_COLS if c != "player_id")
    conn.execute(
        f"INSERT INTO card_meta ({', '.join(_CARD_META_COLS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(player_id) DO UPDATE SET {updates}",
        tuple(data[c] for c in _CARD_META_COLS),
    )


def card_meta_get(conn: sqlite3.Connection, player_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM card_meta WHERE player_id=?", (player_id,)).fetchone()


def card_registry(conn: sqlite3.Connection, *, title: str | None = None,
                  tradeable_only: bool = True,
                  limit: int | None = None) -> list[sqlite3.Row]:
    """The tracked card universe, optionally scoped to one title / tradeable only."""
    clauses, params = [], []
    if title is not None:
        clauses.append("title=?")
        params.append(title)
    if tradeable_only:
        clauses.append("tradeable=1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM card_meta {where} ORDER BY rating DESC, player_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, tuple(params)).fetchall()


def cards_for_backfill(conn: sqlite3.Connection, *, title: str | None = None,
                       tradeable_only: bool = True,
                       tiers: tuple[str, ...] | None = None,
                       order: str = "liquidity",
                       limit: int | None = None) -> list[sqlite3.Row]:
    """Registry cards in backfill order.

    order='liquidity' walks sellable cards first (tier A>B>C>unscored, then score,
    then rating) — right when you want tradeable coverage soonest.
    order='oldest' walks earliest-released cards first, which maximises history
    length per fetch and is the cure for a training set skewed to recent cards:
    the model should learn from the long-lived unglamorous cards too, even though
    it only ever *trades* liquid ones.
    """
    clauses, params = [], []
    if title is not None:
        clauses.append("c.title=?")
        params.append(title)
    if tradeable_only:
        clauses.append("c.tradeable=1")
    if tiers is not None:
        clauses.append(f"l.tier IN ({','.join('?' for _ in tiers)})")
        params.extend(tiers)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    if order == "oldest":
        # Earliest releases first: longest price histories, and the long-lived
        # cards that a recency-skewed training set is missing.
        order_by = ("c.release_date IS NULL, c.release_date ASC, "
                    "c.rating DESC, c.player_id")
    else:
        order_by = ("CASE COALESCE(l.tier, 'Z') "
                    "WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 ELSE 3 END, "
                    "COALESCE(l.score, -1.0) DESC, c.rating DESC, c.player_id")
    sql = f"""
        SELECT c.player_id, c.definition_id, c.name, c.title, c.rating,
               c.release_date,
               COALESCE(l.tier, 'Z') AS tier, COALESCE(l.score, -1.0) AS score
        FROM card_meta c
        LEFT JOIN liquidity l ON l.player_id = c.player_id
        {where}
        ORDER BY {order_by}
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, tuple(params)).fetchall()


def card_count(conn: sqlite3.Connection, *, title: str | None = None,
               tradeable_only: bool = False) -> int:
    clauses, params = [], []
    if title is not None:
        clauses.append("title=?")
        params.append(title)
    if tradeable_only:
        clauses.append("tradeable=1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(
        f"SELECT COUNT(*) AS n FROM card_meta {where}", tuple(params)).fetchone()["n"]


# ---- liquidity (tradeability score + tier) ----

def upsert_liquidity(conn: sqlite3.Connection, *, player_id: str, score: float,
                     tier: str, title: str = "fc26",
                     updates_per_day: float | None = None,
                     price: int | None = None, spread_pct: float | None = None,
                     at: datetime | None = None) -> None:
    at = at or datetime.now(timezone.utc)
    conn.execute(
        """INSERT INTO liquidity (player_id, title, score, tier, updates_per_day,
                                  price, spread_pct, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(player_id) DO UPDATE SET
             title=excluded.title, score=excluded.score, tier=excluded.tier,
             updates_per_day=excluded.updates_per_day, price=excluded.price,
             spread_pct=excluded.spread_pct, computed_at=excluded.computed_at""",
        (player_id, title, float(score), tier, updates_per_day, price, spread_pct,
         bucket_timestamp(at)),
    )


def liquidity_get(conn: sqlite3.Connection, player_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM liquidity WHERE player_id=?", (player_id,)).fetchone()


def liquidity_by_tier(conn: sqlite3.Connection, tier: str,
                      title: str = "fc26") -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM liquidity WHERE tier=? AND title=? ORDER BY score DESC",
        (tier, title)).fetchall()


# ---- sentiment_signals (what the crowd is saying) ----

def insert_sentiment(conn: sqlite3.Connection, *, player_id: str, platform: str,
                     at: datetime, mention_count: int,
                     sentiment: float | None = None,
                     buzz_z: float | None = None,
                     raw_ref: str | None = None) -> None:
    """One player's buzz on one platform at one moment. Upserts on the minute so
    a re-run doesn't inflate the counts."""
    conn.execute(
        """INSERT INTO sentiment_signals (player_id, platform, timestamp,
             mention_count, sentiment, buzz_z, raw_ref)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(player_id, platform, timestamp) DO UPDATE SET
             mention_count=excluded.mention_count, sentiment=excluded.sentiment,
             buzz_z=excluded.buzz_z""",
        (player_id, platform, bucket_timestamp(at), int(mention_count),
         sentiment, buzz_z, raw_ref))


def sentiment_for(conn: sqlite3.Connection, player_id: str,
                  limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sentiment_signals WHERE player_id=? "
        "ORDER BY timestamp DESC LIMIT ?", (player_id, limit)).fetchall()


# ---- pick_log (the track record) ----

def insert_pick(conn: sqlite3.Connection, *, player_id: str, entry_price: int,
                target_price: int, stop_price: int, horizon_days: int,
                at: datetime, title: str = "fc26", run_id: int | None = None,
                confidence: float | None = None, buy_low: int | None = None,
                buy_high: int | None = None, sales_per_hour: float | None = None,
                reasons: str | None = None, strategy: str = "legacy",
                market_price: int | None = None,
                chosen_horizon_days: int | None = None) -> bool:
    """Record a recommendation. False if this card was already picked this minute."""
    cur = conn.execute(
        """INSERT OR IGNORE INTO pick_log (player_id, title, picked_at, run_id,
             confidence, entry_price, buy_low, buy_high, target_price, stop_price,
             horizon_days, sales_per_hour, reasons, strategy, market_price,
             chosen_horizon_days)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (player_id, title, bucket_timestamp(at), run_id, confidence,
         int(entry_price), buy_low, buy_high, int(target_price), int(stop_price),
         int(horizon_days), sales_per_hour, reasons, strategy, market_price,
         chosen_horizon_days or int(horizon_days)),
    )
    return cur.rowcount == 1


def open_picks(conn: sqlite3.Connection, *, title: str = "fc26") -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM pick_log WHERE status='open' AND title=? ORDER BY picked_at",
        (title,)).fetchall()


def has_open_pick(conn: sqlite3.Connection, player_id: str, *,
                  title: str = "fc26", strategy: str | None = None) -> bool:
    """Is this card already held?

    The loop runs every two hours and re-derives the same shortlist each time, so
    without this a single opportunity was recorded as a dozen separate trades:
    92 picks over 37 distinct cards, the same card simultaneously 'open' and
    'stop'. One position per card is both the honest accounting and the real
    trade -- you only buy it once.
    """
    sql = "SELECT 1 FROM pick_log WHERE player_id=? AND title=? AND status='open'"
    params: list = [player_id, title]
    if strategy is not None:
        sql += " AND strategy=?"
        params.append(strategy)
    return conn.execute(sql + " LIMIT 1", tuple(params)).fetchone() is not None


def latest_price(conn: sqlite3.Connection, player_id: str,
                 source: str = "futgg") -> int | None:
    """The most recent live price for a card, or None if never seen."""
    row = conn.execute(
        "SELECT price FROM price_snapshots WHERE player_id=? AND source=? AND price>0 "
        "ORDER BY timestamp DESC LIMIT 1", (player_id, source)).fetchone()
    return int(row["price"]) if row else None


def mark_pick_alerted(conn: sqlite3.Connection, pick_id: int,
                      at: datetime | None = None) -> None:
    conn.execute("UPDATE pick_log SET alerted_at=? WHERE id=?",
                 (bucket_timestamp(at or datetime.now(timezone.utc)), pick_id))
    conn.commit()


def close_pick(conn: sqlite3.Connection, pick_id: int, *, status: str,
               exit_price: int, realized_pct: float,
               benchmark_pct: float | None = None,
               at: datetime | None = None) -> None:
    conn.execute(
        "UPDATE pick_log SET status=?, exit_price=?, realized_pct=?, "
        "benchmark_pct=?, scored_at=? WHERE id=?",
        (status, int(exit_price), realized_pct, benchmark_pct,
         bucket_timestamp(at or datetime.now(timezone.utc)), pick_id))


def picks_log(conn: sqlite3.Connection, *, title: str = "fc26",
              status: str | None = None, limit: int = 100) -> list[sqlite3.Row]:
    clauses, params = ["title=?"], [title]
    if status:
        clauses.append("status=?")
        params.append(status)
    params.append(limit)
    return conn.execute(
        f"SELECT * FROM pick_log WHERE {' AND '.join(clauses)} "
        f"ORDER BY picked_at DESC LIMIT ?", tuple(params)).fetchall()


# ---- sale_stats (what cards really sold for) ----

def upsert_sale_stats(conn: sqlite3.Connection, *, player_id: str, n_sales: int,
                      title: str = "fc26", window_hours: float | None = None,
                      band_from_sales: int | None = None,
                      band_window_hours: float | None = None,
                      sales_per_hour: float | None = None,
                      sold_p25: int | None = None, sold_median: int | None = None,
                      sold_p75: int | None = None, listed_price: int | None = None,
                      sold_vs_listed: float | None = None,
                      at: datetime | None = None) -> None:
    at = at or datetime.now(timezone.utc)
    conn.execute(
        """INSERT INTO sale_stats (player_id, title, n_sales, window_hours,
              band_from_sales, band_window_hours,
              sales_per_hour, sold_p25, sold_median, sold_p75, listed_price,
              sold_vs_listed, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(player_id) DO UPDATE SET
             title=excluded.title, n_sales=excluded.n_sales,
             window_hours=excluded.window_hours,
             band_from_sales=excluded.band_from_sales,
             band_window_hours=excluded.band_window_hours,
             sales_per_hour=excluded.sales_per_hour,
             sold_p25=excluded.sold_p25, sold_median=excluded.sold_median,
             sold_p75=excluded.sold_p75, listed_price=excluded.listed_price,
             sold_vs_listed=excluded.sold_vs_listed, computed_at=excluded.computed_at""",
        (player_id, title, int(n_sales), window_hours, band_from_sales,
         band_window_hours, sales_per_hour, sold_p25,
         sold_median, sold_p75, listed_price, sold_vs_listed, bucket_timestamp(at)),
    )


def sale_stats_get(conn: sqlite3.Connection, player_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sale_stats WHERE player_id=?",
                        (player_id,)).fetchone()


def sale_stats_list(conn: sqlite3.Connection, *, title: str = "fc26",
                    min_sales_per_hour: float | None = None,
                    limit: int | None = None) -> list[sqlite3.Row]:
    clauses, params = ["title=?"], [title]
    if min_sales_per_hour is not None:
        clauses.append("sales_per_hour >= ?")
        params.append(min_sales_per_hour)
    sql = (f"SELECT * FROM sale_stats WHERE {' AND '.join(clauses)} "
           "ORDER BY sales_per_hour DESC")
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, tuple(params)).fetchall()


# ---- game_calendar (per-title lifecycle anchor) ----

def set_game_launch(conn: sqlite3.Connection, *, title: str, launch_date: str,
                    notes: str | None = None) -> None:
    conn.execute(
        """INSERT INTO game_calendar (title, launch_date, notes) VALUES (?, ?, ?)
           ON CONFLICT(title) DO UPDATE SET
             launch_date=excluded.launch_date, notes=excluded.notes""",
        (title, launch_date, notes),
    )


def game_launch(conn: sqlite3.Connection, title: str) -> str | None:
    row = conn.execute("SELECT launch_date FROM game_calendar WHERE title=?",
                       (title,)).fetchone()
    return row["launch_date"] if row else None


# ---- predictions ----

def insert_prediction(conn: sqlite3.Connection, *, subject_id: str, level: str,
                      kind: str, horizon_h: int, at: datetime, run_id: int,
                      title: str = "fc26", yhat: float | None = None,
                      yhat_lo: float | None = None, yhat_hi: float | None = None,
                      proba: float | None = None) -> None:
    conn.execute(
        """INSERT INTO predictions (subject_id, level, title, timestamp, run_id,
                                    kind, horizon_h, yhat, yhat_lo, yhat_hi, proba)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(subject_id, level, kind, horizon_h, timestamp) DO UPDATE SET
             run_id=excluded.run_id, title=excluded.title, yhat=excluded.yhat,
             yhat_lo=excluded.yhat_lo, yhat_hi=excluded.yhat_hi, proba=excluded.proba""",
        (subject_id, level, title, bucket_timestamp(at), run_id, kind, int(horizon_h),
         yhat, yhat_lo, yhat_hi, proba),
    )


def latest_predictions(conn: sqlite3.Connection, *, level: str = "card",
                       kind: str | None = None, limit: int = 100) -> list[sqlite3.Row]:
    clauses, params = ["level=?"], [level]
    if kind is not None:
        clauses.append("kind=?")
        params.append(kind)
    params.append(limit)
    return conn.execute(
        f"SELECT * FROM predictions WHERE {' AND '.join(clauses)} "
        f"ORDER BY timestamp DESC LIMIT ?", tuple(params)).fetchall()


# ---- model_runs (the training-run registry) ----

def create_model_run(conn: sqlite3.Connection, *, kind: str, level: str = "card",
                     title: str = "fc26", horizon_h: int | None = None,
                     n_samples: int | None = None, metrics_json: str | None = None,
                     artifact_path: str | None = None,
                     feature_list_json: str | None = None,
                     git_sha: str | None = None,
                     at: datetime | None = None) -> int:
    at = at or datetime.now(timezone.utc)
    cur = conn.execute(
        """INSERT INTO model_runs (kind, level, title, horizon_h, trained_at,
             n_samples, metrics_json, artifact_path, feature_list_json, git_sha)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kind, level, title, horizon_h, bucket_timestamp(at), n_samples,
         metrics_json, artifact_path, feature_list_json, git_sha),
    )
    conn.commit()
    return int(cur.lastrowid)


def latest_model_run(conn: sqlite3.Connection, *, kind: str, level: str = "card",
                     title: str = "fc26") -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM model_runs WHERE kind=? AND level=? AND title=? "
        "ORDER BY run_id DESC LIMIT 1", (kind, level, title)).fetchone()


def model_runs_list(conn: sqlite3.Connection, *, kind: str | None = None,
                    limit: int = 50) -> list[sqlite3.Row]:
    if kind is not None:
        return conn.execute(
            "SELECT * FROM model_runs WHERE kind=? ORDER BY run_id DESC LIMIT ?",
            (kind, limit)).fetchall()
    return conn.execute(
        "SELECT * FROM model_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
