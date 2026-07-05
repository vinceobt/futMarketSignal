"""Collection passes and the polling loop.

Politeness rules live here: sequential fetches, jittered inter-player delay,
skip-guard against re-collecting fresh players, and a circuit breaker that
pauses a failing source instead of hammering it.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import db
from .collectors import get_source
from .collectors.base import PriceSource, SourceError
from .config import Config

log = logging.getLogger("futmarket.collector")


@dataclass
class PassResult:
    collected: list[str] = field(default_factory=list)
    skipped_fresh: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    aborted_by_breaker: bool = False
    stopped: bool = False


def run_pass(config: Config, conn, source: PriceSource,
             sleep=time.sleep, now=None, should_stop=None,
             only_player_id=None, on_progress=None) -> PassResult:
    """One polite sequential pass over the watchlist. Never raises for a single
    player's failure; trips the breaker after max_consecutive_failures in a row.

    should_stop:     optional zero-arg callable; checked before each player so a
                     UI/worker can cancel cleanly between (never mid-) fetches.
    only_player_id:  restrict the pass to a single watchlist player (used by the
                     "scrape one" / add-link-then-enrich flow).
    on_progress:     optional callback(done:int, total:int, player) for live UI.
    """
    result = PassResult()
    guard = timedelta(minutes=config.poll_minutes * config.skip_guard_fraction)
    consecutive_failures = 0
    lo, hi = config.inter_player_delay_seconds

    from .services import watch
    players = [p for p in watch.effective_entries(conn, config)
               if only_player_id is None or p.player_id == only_player_id]
    total = len(players)
    for i, player in enumerate(players):
        if should_stop is not None and should_stop():
            log.info("pass stopped by request after %d/%d players", i, total)
            result.stopped = True
            break
        if on_progress is not None:
            on_progress(i, total, player)
        current = now() if now else datetime.now(timezone.utc)

        last = db.latest_snapshot_time(conn, player.player_id, source.name)
        if last is not None and current - last < guard:
            log.info("skip player=%s reason=fresh last=%s", player.player_id, last.isoformat())
            result.skipped_fresh.append(player.player_id)
            continue

        if i > 0:
            sleep(random.uniform(lo, hi))

        try:
            quote = source.fetch_price(player, config.platform)
        except SourceError as exc:
            consecutive_failures += 1
            log.warning("fail player=%s source=%s error=%s (%d consecutive)",
                        player.player_id, source.name, exc, consecutive_failures)
            result.failed.append(player.player_id)
            if consecutive_failures >= config.max_consecutive_failures:
                log.error("circuit breaker tripped for source=%s after %d failures; "
                          "cooling down %d min", source.name, consecutive_failures,
                          config.cooldown_minutes)
                result.aborted_by_breaker = True
                break
            continue

        consecutive_failures = 0
        # Prefer metadata the source enriched from the page; fall back to the
        # values derived from the watchlist URL.
        db.upsert_player(conn, player_id=player.player_id,
                         name=quote.name or player.name,
                         rating=quote.rating if quote.rating is not None else player.rating,
                         position=player.position,
                         version=quote.version or player.version,
                         platform=config.platform)
        # Backfill the card's full market history (idempotent — existing points
        # are ignored), then record the live price at the moment of fetch.
        new_points = 0
        for at, price in quote.history:
            if db.insert_snapshot(conn, player_id=quote.player_id, price=price,
                                  source=quote.source, at=at):
                new_points += 1
        if db.insert_snapshot(conn, player_id=quote.player_id, price=quote.price,
                              source=quote.source, at=quote.fetched_at):
            new_points += 1
        conn.commit()
        if new_points:
            log.info("collect player=%s new_points=%d latest=%d source=%s",
                     player.player_id, new_points, quote.price, quote.source)
            result.collected.append(player.player_id)
        else:
            log.info("skip player=%s reason=nothing-new", player.player_id)
            result.skipped_fresh.append(player.player_id)

    log.info("pass done collected=%d skipped=%d failed=%d breaker=%s",
             len(result.collected), len(result.skipped_fresh),
             len(result.failed), result.aborted_by_breaker)
    return result


def run_forever(config: Config) -> None:
    """Blocking polling loop. A tripped breaker just delays the next pass;
    nothing here ever kills the process on a source failure."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    source = get_source(config.source, config)
    scheduler = BlockingScheduler(timezone="UTC")
    state = {"cooldown_until": None}

    def job():
        until = state["cooldown_until"]
        if until and datetime.now(timezone.utc) < until:
            log.info("pass skipped: source cooling down until %s", until.isoformat())
            return
        state["cooldown_until"] = None
        # SQLite connections are thread-bound and APScheduler jobs run in a
        # worker thread, so each pass opens its own connection.
        conn = db.connect(config.database_path)
        try:
            outcome = run_pass(config, conn, source)
        finally:
            conn.close()
        if outcome.aborted_by_breaker:
            state["cooldown_until"] = (datetime.now(timezone.utc)
                                       + timedelta(minutes=config.cooldown_minutes))

    from .services import watch
    conn0 = db.connect(config.database_path)
    try:
        n_watch = len(watch.effective_entries(conn0, config))  # also seeds on first run
    finally:
        conn0.close()
    scheduler.add_job(job, "interval", minutes=config.poll_minutes,
                      jitter=120, next_run_time=datetime.now(timezone.utc))
    log.info("scheduler started source=%s interval=%dmin watchlist=%d",
             config.source, config.poll_minutes, n_watch)
    scheduler.start()
