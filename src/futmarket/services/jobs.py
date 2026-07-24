"""A tiny background job runner for the dashboard controls.

One worker thread runs one job at a time (so a bulk fetch and a retrain never
fight over the write lock), recording status in the `jobs` table so the page can
poll it. Each job gets its own SQLite connection because it runs off-thread.

Actions map a short name to a function that does the work and returns a one-line
result. Failures are caught and recorded, never crash the server.
"""

from __future__ import annotations

import logging
import queue
import threading

from .. import db
from ..config import Config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- the actions

def _collect_bulk(conn, config: Config) -> str:
    from .bulk_collect import collect_bulk
    r = collect_bulk(conn, platform=config.platform)
    return f"{r['inserted']:,} new snapshots ({r['matched']:,} cards priced)"


def _run_picks(conn, config: Config) -> str:
    from ..ml import picks
    found = picks.generate(
        conn, source=config.source, tax_rate=config.tax_rate,
        max_price=config.max_price, entry_z_max=config.entry_z_max,
        entry_floor_max_pct=config.entry_floor_max_pct,
        stop_buffer_pct=config.stop_buffer_pct, stop_min_pct=config.stop_min_pct,
        stop_max_pct=config.stop_max_pct, min_reward_risk=config.min_reward_risk)
    n = picks.save(conn, found)
    return f"{len(found)} candidates, {n} newly recorded"


def _grade(conn, config: Config) -> str:
    from .scorecard import score_open_picks
    r = score_open_picks(conn, source=config.source, tax_rate=config.tax_rate,
                         sell_slippage_pct=config.sell_slippage_pct)
    return (f"{r['checked']} checked · {r['target']} hit target · "
            f"{r['stop']} stopped · {r['still_open']} still open")


def _insights(conn, config: Config) -> str:
    from ..ml import insights
    s = insights.refresh(conn, source=config.source)
    return f"rhythms recomputed ({len(s['weekly'])} weekday points)"


def _advise_refresh(conn, config: Config) -> str:
    from ..ml import advise
    frame = advise.get_scored(conn, source=config.source, rebuild=True)
    return f"rescored {len(frame):,} cards for the consult"


def _train(conn, config: Config) -> str:
    from ..ml import train
    r = train.train(conn, source=config.source, horizon=config.horizon_days,
                    stop_buffer_pct=config.stop_buffer_pct,
                    stop_min_pct=config.stop_min_pct, stop_max_pct=config.stop_max_pct)
    if "error" in r:
        return r["error"]
    d = r.get("direction", {})
    return (f"trained on {r['rows']:,} rows · "
            f"top-pick precision {d.get('precision_at_top_decile', 0) * 100:.0f}%")


def _trader_tips(conn, config: Config) -> str:
    from ..ml.trader_clone import advise
    r = advise(conn, cache=True, quiet=True)
    n = sum(len(v) for v in r["tiers"].values())
    return f"trader tips refreshed ({n} across {len(r['tiers'])} tiers)"


ACTIONS = {
    "collect-bulk": _collect_bulk,
    "picks": _run_picks,
    "scorecard": _grade,
    "insights": _insights,
    "advise-refresh": _advise_refresh,
    "trader-tips": _trader_tips,
    "train": _train,
}


# ----------------------------------------------------------------- the runner

class JobRunner:
    """A single-worker queue. `submit` enqueues; the worker runs them in order."""

    def __init__(self, config: Config):
        self.config = config
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, action: str) -> int:
        if action not in ACTIONS:
            raise KeyError(action)
        conn = db.connect(self.config.database_path)
        try:
            job_id = db.create_job(conn, job_type=action, detail="queued")
        finally:
            conn.close()
        self._q.put((job_id, action))
        return job_id

    def _loop(self) -> None:
        while True:
            job_id, action = self._q.get()
            conn = db.connect(self.config.database_path)
            try:
                db.update_job(conn, job_id, status="running", detail="working…")
                detail = ACTIONS[action](conn, self.config)
                db.update_job(conn, job_id, status="done", detail=detail or "done")
            except Exception as e:  # noqa: BLE001 - a bad job must not kill the worker
                logger.exception("job %s failed", action)
                db.update_job(conn, job_id, status="error", detail=str(e)[:200])
            finally:
                conn.close()
                self._q.task_done()
