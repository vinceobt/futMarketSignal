"""FastAPI app for the live ML dashboard.

A read-only view of the model's picks, liquidity coverage, track record, and the
market rhythms it learned — rendered fresh from the DB on every request. Launch
with `futmarket dashboard`.

When an access key is configured it gates the page (supplied as `?key=…`, then
remembered in a cookie), so the dashboard can be exposed on the network safely.
"""

from __future__ import annotations

import hmac

from . import db, security
from .config import load_config


def create_app(config_path, source: str | None = None, access_key: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    config = load_config(config_path)
    key, generated = (access_key, False) if access_key else security.resolve_key()

    app = FastAPI(title="FUT Market Desk", docs_url=None, redoc_url=None)

    def _conn():
        return db.connect(config.database_path)

    def _authorized(request: Request) -> bool:
        # No key configured -> open (localhost dev). Otherwise the key must be
        # supplied as ?key=… or carried in the cookie set on first load.
        if not key:
            return True
        supplied = request.query_params.get("key") or request.cookies.get("fm_key")
        return bool(supplied) and hmac.compare_digest(supplied, key)

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
    def ml_dashboard(request: Request):
        if not _authorized(request):
            return HTMLResponse(
                "<h1>Unauthorized</h1><p>Append <code>?key=…</code> to the URL.</p>",
                status_code=401)
        from .ml import dashboard as ml_dash
        conn = _conn()
        try:
            html = ml_dash.render(conn, source=source or config.source)
        finally:
            conn.close()
        resp = HTMLResponse(html)
        # Remember the key so in-page navigation doesn't need it re-appended.
        if key and request.query_params.get("key"):
            resp.set_cookie("fm_key", key, httponly=True, samesite="strict", path="/")
        return resp

    @app.get("/api/health")
    def health():
        # Public liveness probe — reveals nothing sensitive.
        return {"ok": True}

    app.state.access_key = key
    app.state.key_generated = generated
    return app
