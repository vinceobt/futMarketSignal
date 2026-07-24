"""FastAPI app for the live ML dashboard.

A read-only view of the model's picks, liquidity coverage, track record, and the
market rhythms it learned — rendered fresh from the DB on every request. Launch
with `futmarket dashboard`.

The page is open (no login): it's read-only and meant for the owner's own
machine/LAN. It still ships with hardening headers.
"""

import hmac

from . import db, security
from .config import load_config

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def create_app(config_path, source: str | None = None, access_key: str | None = None):
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

    from .services.jobs import ACTIONS, JobRunner

    config = load_config(config_path)
    key, generated = (access_key, False) if access_key else security.resolve_key()
    runner = JobRunner(config)

    app = FastAPI(title="FUT Market Desk", docs_url=None, redoc_url=None)

    def _conn():
        return db.connect(config.database_path)

    def _guard_action(request: Request):
        """Actions run code, so they're guarded. Same-origin always (blocks CSRF);
        from off-box they also need the access key. Returns a response to send on
        rejection, or None when allowed."""
        if not security.same_origin(request.headers.get("origin"),
                                    request.headers.get("host")):
            return JSONResponse({"error": "cross-site request blocked"}, status_code=403)
        client = request.client.host if request.client else ""
        if client not in _LOOPBACK:
            supplied = request.query_params.get("key") or request.headers.get("x-fm-key")
            if not (key and supplied and hmac.compare_digest(supplied, key)):
                return JSONResponse(
                    {"error": "actions need the access key when off this machine"},
                    status_code=401)
        return None

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        for k, v in security.SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.get("/")
    def index(request: Request):
        # The dashboard lives at /ml; keep any ?key= so the bookmark works.
        q = request.url.query
        return RedirectResponse(url="/ml" + (f"?{q}" if q else ""))

    @app.get("/ml")
    def ml_dashboard():
        # Open by design: a read-only dashboard on the owner's own machine/LAN.
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            return HTMLResponse(ml_dash.render(conn, source=source or config.source))
        finally:
            conn.close()

    @app.get("/app.js")
    def app_js():
        from .ml import dashboard as ml_dash
        return Response(ml_dash.APP_JS, media_type="application/javascript")

    @app.get("/api/health")
    def health():
        # Public liveness probe — reveals nothing sensitive.
        return {"ok": True}

    # ---- consult + fragments (reads, open) ----
    @app.get("/api/advise")
    def api_advise(q: str = "", version: str = ""):
        from .ml import advise, dashboard as ml_dash
        if not q.strip():
            return {"html": ""}
        conn = _conn()
        try:
            frame = advise.get_scored(conn, source=source or config.source)
            matches = advise.find_cards(frame, q, version=version or None)
            items = [(row.to_dict(), advise.card_read(row, tax_rate=config.tax_rate))
                     for _, row in matches.iterrows()]
            return {"html": ml_dash.render_card_reads(items)}
        finally:
            conn.close()

    @app.get("/api/advise/group")
    def api_advise_group(dim: str, value: str):
        from .ml import advise, dashboard as ml_dash
        conn = _conn()
        try:
            frame = advise.get_scored(conn, source=source or config.source)
            read = advise.cohort_read(frame, dim=dim, value=value,
                                      tax_rate=config.tax_rate)
            return {"html": ml_dash.render_group_read(read)}
        finally:
            conn.close()

    @app.get("/api/scorecard")
    def api_scorecard():
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            return {"html": ml_dash.record_fragment(conn)}
        finally:
            conn.close()

    @app.get("/api/picks")
    def api_picks():
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            return {"html": ml_dash.picks_fragment(conn)}
        finally:
            conn.close()

    @app.get("/api/holding")
    def api_holding():
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            return {"html": ml_dash.holding_fragment(conn, source=source or config.source)}
        finally:
            conn.close()

    @app.get("/api/trader-tips")
    def api_trader_tips():
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            return {"html": ml_dash.trader_tips_fragment(conn)}
        finally:
            conn.close()

    # ---- controls (actions run code, so guarded) ----
    @app.post("/api/run/{action}")
    def api_run(action: str, request: Request):
        blocked = _guard_action(request)
        if blocked is not None:
            return blocked
        if action not in ACTIONS:
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=404)
        return {"job_id": runner.submit(action)}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: int):
        conn = _conn()
        try:
            row = db.get_job(conn, job_id)
            if row is None:
                return JSONResponse({"error": "no such job"}, status_code=404)
            return {"id": row["id"], "type": row["type"], "status": row["status"],
                    "detail": row["detail"]}
        finally:
            conn.close()

    app.state.access_key = key
    app.state.key_generated = generated
    return app
