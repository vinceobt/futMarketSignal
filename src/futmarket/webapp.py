"""FastAPI app for the local dashboard.

Serves the single-page UI plus a JSON API over the pipeline. Read endpoints
render whatever the DB holds; write endpoints enqueue background jobs (scrapes,
backtests) on a worker thread and let the UI poll their status. The app never
runs a scrape inside a request. Launch with `futmarket dashboard`.
"""

import json
from pathlib import Path

from . import dashboard, db, security
from .config import ConfigError, load_config
from .services import watch
from .services.jobs import JobRunner, ScraperController

WEB = Path(__file__).parent / "web"


def _default_source(conn, config) -> str:
    row = conn.execute(
        "SELECT COUNT(*) as n FROM price_snapshots WHERE source = ?",
        (config.source,)).fetchone()
    if row and row["n"] > 0:
        return config.source
    row = conn.execute(
        "SELECT source, COUNT(*) AS n FROM price_snapshots "
        "GROUP BY source ORDER BY n DESC LIMIT 1").fetchone()
    return row["source"] if row else config.source


def create_app(config_path, source: str | None = None, access_key: str | None = None):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    config_path = str(config_path)
    config = load_config(config_path)
    runner = JobRunner(config_path)
    scraper = ScraperController(runner)

    key, generated = (access_key, False) if access_key else security.resolve_key()
    sec = security.Security(key)

    app = FastAPI(title="FUT Market Desk", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    def _conn():
        return db.connect(config.database_path)

    # ---- hardening middleware: headers, CSRF/origin, rate limit, auth ----
    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        is_api = path.startswith("/api/")
        # 1. CSRF: reject a *present, mismatched* Origin on state changes.
        if request.method in security._WRITE_METHODS:
            if not security.same_origin(request.headers.get("origin"),
                                        request.headers.get("host")):
                return JSONResponse({"detail": "cross-site request blocked"}, status_code=403)
            client = request.client.host if request.client else "?"
            if not sec.rate_ok(client):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        # 2. Auth: every /api/* except the public ones needs a live session.
        if is_api and path not in security._PUBLIC_API:
            if not sec.valid_session(request.cookies.get(security.SESSION_COOKIE)):
                return JSONResponse({"detail": "authentication required"}, status_code=401)
        response = await call_next(request)
        for k, v in security.SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    def _asset_version(name: str) -> str:
        """A cache-busting token that changes whenever the asset file changes, so
        the browser can never serve a stale bundle (no hard-reload needed)."""
        try:
            return str(int((WEB / name).stat().st_mtime))
        except OSError:
            return "0"

    @app.get("/")
    def index():
        html = (WEB / "index.html").read_text()
        for asset in ("app.css", "app.js", "boot.js"):
            html = html.replace(f"/static/{asset}",
                                f"/static/{asset}?v={_asset_version(asset)}")
        return HTMLResponse(html)

    @app.get("/api/health")
    def health():
        # Public liveness probe — intentionally reveals nothing sensitive.
        return {"ok": True}

    # ---- auth ----
    @app.post("/api/login")
    def login(request: Request, payload: dict = Body(default={})):
        client = request.client.host if request.client else "?"
        if not sec.rate_ok("login:" + client):
            raise HTTPException(status_code=429, detail="too many attempts")
        if not sec.check_key((payload or {}).get("key", "")):
            raise HTTPException(status_code=401, detail="invalid access key")
        token = sec.new_session()
        resp = JSONResponse({"ok": True})
        resp.set_cookie(security.SESSION_COOKIE, token, httponly=True,
                        samesite="strict", path="/")
        return resp

    @app.post("/api/logout")
    def logout(request: Request):
        sec.drop_session(request.cookies.get(security.SESSION_COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(security.SESSION_COOKIE, path="/")
        return resp

    @app.on_event("shutdown")
    def _shutdown():
        # Stop continuous scraping cleanly so a Ctrl-C doesn't leave a loop
        # thread submitting jobs after the server is gone.
        scraper.stop()

    # ---- read ----
    @app.get("/api/data")
    def data():
        conn = _conn()
        try:
            src = source or _default_source(conn, config)
            return JSONResponse(dashboard.build_payload(config, conn, src))
        finally:
            conn.close()

    @app.get("/api/watchlist")
    def watchlist():
        conn = _conn()
        try:
            out = []
            for e in watch.list_entries(conn, config):
                meta = conn.execute(
                    "SELECT p.name, p.rating, p.version, "
                    "  (SELECT COUNT(*) FROM price_snapshots s WHERE s.player_id=p.player_id) AS snaps "
                    "FROM players p WHERE p.player_id=?", (e["player_id"],)).fetchone()
                out.append({
                    **e,
                    "rating": meta["rating"] if meta else None,
                    "version": (meta["version"] if meta else None) or "",
                    "snapshots": meta["snaps"] if meta else 0,
                })
            return {"players": out}
        finally:
            conn.close()

    # ---- watchlist mutations ----
    @app.post("/api/watchlist")
    def add_player(payload: dict = Body(default={})):
        url = (payload or {}).get("url", "")
        conn = _conn()
        try:
            added = watch.add(conn, config, url)
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            conn.close()
        # immediately scrape just this player so its price/rating populate
        job_id = runner.submit("collect_one", player_id=added["player_id"],
                               detail=f"enrich {added['name']}")
        return {"added": added, "job_id": job_id}

    @app.delete("/api/watchlist/{player_id}")
    def remove_player(player_id: str):
        conn = _conn()
        try:
            watch.remove(conn, config, player_id)
        except ConfigError as e:
            raise HTTPException(status_code=404, detail=str(e))
        finally:
            conn.close()
        return {"removed": player_id}

    # ---- rebound advisor: paper positions ----
    @app.get("/api/positions")
    def positions():
        conn = _conn()
        try:
            rows = [dict(r) for r in db.positions_list(conn)]
            # attach the latest known price for open positions (unrealized %)
            src = source or _default_source(conn, config)
            for r in rows:
                if r["status"] == "open":
                    last = conn.execute(
                        "SELECT price FROM price_snapshots WHERE player_id=? AND source=? "
                        "ORDER BY timestamp DESC LIMIT 1", (r["player_id"], src)).fetchone()
                    cur = last["price"] if last else None
                    r["current_price"] = cur
                    r["unrealized_pct"] = (round((cur * (1 - config.tax_rate) / r["entry_price"] - 1) * 100, 1)
                                           if cur else None)
            return {"positions": rows}
        finally:
            conn.close()

    @app.post("/api/advise")
    def advise():
        return {"job_id": runner.submit("advise", detail="run advisor")}

    # ---- momentum scanner ----
    @app.get("/api/momentum")
    def momentum():
        conn = _conn()
        try:
            return {"updated_at": db.meta_get(conn, "momentum_updated_at"),
                    "players": [dict(r) for r in db.momentum_list(conn)]}
        finally:
            conn.close()

    @app.post("/api/momentum/refresh")
    def momentum_refresh():
        return {"job_id": runner.submit("momentum", detail="fetch momentum")}

    # ---- execution controls (enqueue jobs) ----
    @app.post("/api/collect")
    def collect(payload: dict = Body(default={})):
        pid = (payload or {}).get("player_id")
        if pid:
            return {"job_id": runner.submit("collect_one", player_id=pid,
                                            detail="scrape one")}
        return {"job_id": runner.submit("collect", detail="scrape all")}

    @app.post("/api/backtest")
    def backtest():
        return {"job_id": runner.submit("backtest")}

    @app.post("/api/signals")
    def signals():
        return {"job_id": runner.submit("signals")}

    @app.post("/api/build-features")
    def build_features():
        return {"job_id": runner.submit("build_features")}

    # ---- jobs ----
    @app.get("/api/jobs")
    def jobs():
        conn = _conn()
        try:
            return {"jobs": [dict(r) for r in db.list_jobs(conn)],
                    "current": runner.current_job_id}
        finally:
            conn.close()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int):
        conn = _conn()
        try:
            row = db.get_job(conn, job_id)
            if not row:
                raise HTTPException(status_code=404, detail="no such job")
            out = dict(row)
            if out.get("result_json"):
                out["result"] = json.loads(out["result_json"])
            out.pop("result_json", None)
            return out
        finally:
            conn.close()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        conn = _conn()
        try:
            db.request_job_cancel(conn, job_id)
            return {"cancelling": job_id}
        finally:
            conn.close()

    # ---- scraper engine (continuous mode) ----
    @app.post("/api/scraper/start")
    def scraper_start():
        scraper.start()
        return scraper.status()

    @app.post("/api/scraper/stop")
    def scraper_stop():
        scraper.stop()
        return scraper.status()

    @app.get("/api/scraper/status")
    def scraper_status():
        return scraper.status()

    app.state.runner = runner
    app.state.scraper = scraper
    app.state.access_key = key
    app.state.key_generated = generated
    return app
