"""Background job runner + continuous-scraper controller.

The web layer never runs work inside a request (scrapes are minutes-long,
browser-driven). It enqueues a job here; a single daemon worker thread runs it,
updating the `jobs` table so the UI can poll status/progress/logs. Cancellation
is cooperative: a scrape checks the job's cancel flag between players.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone

from .. import db, features
from ..collectors import get_source
from ..config import load_config
from ..scheduler import run_pass
from . import analytics, watch

log = logging.getLogger("futmarket.jobs")

JOB_TYPES = {"collect", "collect_one", "backtest", "signals", "build_features",
             "momentum", "advise"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobRunner:
    """Owns one daemon worker thread and its own SQLite connection (SQLite
    connections are thread-bound, so the worker must not share the web app's)."""

    def __init__(self, config_path):
        self.config_path = str(config_path)
        self.db_path = load_config(self.config_path).database_path
        self._q: queue.Queue = queue.Queue()
        self._current_job_id: int | None = None
        self._conn = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-worker")
        self._thread.start()

    # ---- public API (called from the web thread) ----
    def submit(self, job_type: str, **params) -> int:
        if job_type not in JOB_TYPES:
            raise ValueError(f"unknown job type: {job_type}")
        conn = db.connect(self.db_path)
        try:
            job_id = db.create_job(conn, job_type=job_type,
                                   detail=params.get("detail", "queued"))
        finally:
            conn.close()
        self._q.put((job_id, job_type, params))
        return job_id

    @property
    def current_job_id(self) -> int | None:
        return self._current_job_id

    # ---- worker thread ----
    def _loop(self):
        self._conn = db.connect(self.db_path)
        conn = self._conn
        while True:
            job_id, job_type, params = self._q.get()
            self._current_job_id = job_id
            try:
                db.update_job(conn, job_id, status="running", detail="running…")
                handler = getattr(self, f"_run_{job_type}")
                handler(conn, job_id, params)
                row = db.get_job(conn, job_id)
                if row and row["status"] == "running":  # handler didn't set terminal
                    # preserve the handler's detail (result summary); only mark done
                    db.update_job(conn, job_id, status="done", finished_at=_now())
            except Exception as exc:  # never let one job kill the worker
                log.exception("job %s failed", job_id)
                db.append_job_log(conn, job_id, f"ERROR: {exc}")
                db.update_job(conn, job_id, status="failed",
                              detail=str(exc)[:200], finished_at=_now())
            finally:
                self._current_job_id = None
                self._q.task_done()

    def _cfg(self):
        return load_config(self.config_path)

    # ---- handlers ----
    def _run_collect(self, conn, job_id, params, only_player_id=None):
        cfg = self._cfg()
        source = get_source(cfg.source, cfg)
        total = 1 if only_player_id else len(watch.effective_entries(conn, cfg))
        db.update_job(conn, job_id, total=total)
        db.append_job_log(conn, job_id, f"scrape start: source={cfg.source} players={total}")

        def should_stop():
            return db.job_cancel_requested(conn, job_id)

        def on_progress(done, tot, player):
            db.update_job(conn, job_id, progress=done, total=tot,
                          detail=f"scraping {player.name} ({done + 1}/{tot})")
            db.append_job_log(conn, job_id, f"fetching {player.name}")

        result = run_pass(cfg, conn, source, should_stop=should_stop,
                          only_player_id=only_player_id, on_progress=on_progress)
        db.update_job(conn, job_id, progress=total,
                      result_json=json.dumps({
                          "collected": result.collected,
                          "skipped": result.skipped_fresh,
                          "failed": result.failed,
                          "breaker": result.aborted_by_breaker,
                          "stopped": result.stopped,
                      }))
        if result.stopped:
            db.update_job(conn, job_id, status="cancelled", detail="stopped by user",
                          finished_at=_now())
            db.append_job_log(conn, job_id, "cancelled")
        else:
            msg = (f"done: collected={len(result.collected)} "
                   f"skipped={len(result.skipped_fresh)} failed={len(result.failed)}")
            db.append_job_log(conn, job_id, msg)
            db.update_job(conn, job_id, detail=msg)
        # features are cheap to rebuild and keep the dashboard current
        if result.collected:
            features.build_and_store(conn, cfg, cfg.source)

    def _run_collect_one(self, conn, job_id, params):
        self._run_collect(conn, job_id, params, only_player_id=params["player_id"])

    def _run_backtest(self, conn, job_id, params):
        cfg = self._cfg()
        db.append_job_log(conn, job_id, "running backtest…")
        res = analytics.run_backtest(conn, cfg, cfg.source)
        db.update_job(conn, job_id, result_json=json.dumps(res),
                      detail=f"{res['promoted']}/{len(res['players'])} rules promoted")

    def _run_signals(self, conn, job_id, params):
        cfg = self._cfg()
        db.append_job_log(conn, job_id, "evaluating signals…")
        res = analytics.run_signals(conn, cfg, cfg.source)
        db.update_job(conn, job_id, result_json=json.dumps(res),
                      detail=f"BUY {res['buys']} · SELL {res['sells']} · "
                             f"HOLD {res['holds']} · SKIP {res['skips']}")

    def _run_build_features(self, conn, job_id, params):
        cfg = self._cfg()
        n = features.build_and_store(conn, cfg, cfg.source)
        db.update_job(conn, job_id, detail=f"built {n} feature rows")

    def _run_advise(self, conn, job_id, params):
        from . import advisor
        cfg = self._cfg()
        db.append_job_log(conn, job_id, "running rebound advisor…")
        res = advisor.run(conn, cfg, cfg.source)
        db.update_job(conn, job_id, result_json=json.dumps(res),
                      detail=f"opened {len(res['opened'])} · closed {len(res['closed'])} "
                             f"· holding {len(res['holding'])}")

    def _run_momentum(self, conn, job_id, params):
        from ..collectors import momentum_source
        cfg = self._cfg()
        db.append_job_log(conn, job_id, "fetching fut.gg momentum scanner…")
        rows = momentum_source.fetch_momentum(limit=cfg.momentum_limit)
        if not rows:
            raise RuntimeError("no momentum data returned (fut.gg unreachable?)")
        db.momentum_replace(conn, rows)
        db.meta_set(conn, "momentum_updated_at", _now())
        conn.commit()
        db.update_job(conn, job_id, detail=f"{len(rows)} top movers")


class ScraperController:
    """Continuous ('run forever') mode, on/off from the UI. A thread submits a
    collect job every poll_minutes; stopping also cancels the in-flight scrape."""

    def __init__(self, runner: JobRunner):
        self.runner = runner
        self._on = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._on:
            return
        self._on = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scraper-loop")
        self._thread.start()

    def stop(self) -> None:
        self._on = False
        self._stop.set()
        # cancel a scrape currently in flight
        jid = self.runner.current_job_id
        if jid is not None:
            conn = db.connect(self.runner.db_path)
            try:
                db.request_job_cancel(conn, jid)
            finally:
                conn.close()

    def status(self) -> dict:
        cfg = self.runner._cfg()
        return {"running": self._on, "poll_minutes": cfg.poll_minutes,
                "current_job_id": self.runner.current_job_id}

    def _loop(self):
        poll_seconds = max(60, self.runner._cfg().poll_minutes * 60)
        while not self._stop.is_set():
            self.runner.submit("collect", detail="scheduled scrape")
            self._stop.wait(poll_seconds)
