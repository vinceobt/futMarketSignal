"""Watchlist management, DB-backed.

The tracked player list lives in the database (the `watchlist` table), not in
config.yaml — so add/remove is a transactional row op that can't corrupt the
settings file or race with a running scrape. config.yaml's `watchlist:` is used
only as a **one-time seed** on first run; after that the DB is authoritative.

Everything that iterates "the watchlist" at runtime calls `effective_entries`.
The dashboard and the `futmarket watchlist` CLI both go through add/remove/import.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .. import db
from ..config import Config, ConfigError, WatchlistEntry, parse_player_url

_MIGRATED = "watchlist_seeded"


def _ensure_seeded(conn, config: Config) -> None:
    """One-time import of config.yaml's watchlist into the DB. Guarded by a
    marker so an intentionally-empty list (user removed everyone) is never
    re-seeded on the next call."""
    if db.meta_get(conn, _MIGRATED):
        return
    now = datetime.now(timezone.utc)
    for e in config.watchlist:
        db.watchlist_add(conn, player_id=e.player_id, name=e.name, url=e.url, at=now)
    db.meta_set(conn, _MIGRATED, "1")
    conn.commit()


def effective_entries(conn, config: Config) -> tuple[WatchlistEntry, ...]:
    """The authoritative watchlist as WatchlistEntry objects, in insertion order.
    Seeds from config.yaml on first run."""
    _ensure_seeded(conn, config)
    return tuple(
        WatchlistEntry(player_id=r["player_id"], name=r["name"], url=r["url"])
        for r in db.watchlist_entries(conn)
    )


def list_entries(conn, config: Config) -> list[dict]:
    _ensure_seeded(conn, config)
    return [{"player_id": r["player_id"], "name": r["name"], "url": r["url"]}
            for r in db.watchlist_entries(conn)]


def add(conn, config: Config, url: str) -> dict:
    """Validate + track a fut.gg player URL. Raises ConfigError on a bad URL,
    a duplicate, or a full watchlist."""
    _ensure_seeded(conn, config)
    url = url.strip()
    player_id, name = parse_player_url(url)  # raises ConfigError if not a fut.gg URL
    if any(r["player_id"] == player_id for r in db.watchlist_entries(conn)):
        raise ConfigError(f"already tracking {name} ({player_id})")
    if db.watchlist_count(conn) >= config.max_watchlist_size:
        raise ConfigError(f"watchlist is full (cap {config.max_watchlist_size})")
    db.watchlist_add(conn, player_id=player_id, name=name, url=url,
                     at=datetime.now(timezone.utc))
    conn.commit()
    return {"player_id": player_id, "name": name, "url": url}


def remove(conn, config: Config, player_id: str) -> None:
    _ensure_seeded(conn, config)
    if not db.watchlist_remove(conn, player_id):
        raise ConfigError(f"not in watchlist: {player_id}")
    conn.commit()


def import_urls(conn, config: Config, text: str) -> dict:
    """Bulk-add from whitespace/newline-separated URLs (tolerant, like the config
    parser). Returns {added, skipped, errors} so callers can report a summary."""
    _ensure_seeded(conn, config)
    added, skipped, errors = [], [], []
    for token in text.split():
        try:
            added.append(add(conn, config, token))
        except ConfigError as e:
            msg = str(e)
            (skipped if "already tracking" in msg else errors).append({"url": token, "error": msg})
    return {"added": added, "skipped": skipped, "errors": errors}
