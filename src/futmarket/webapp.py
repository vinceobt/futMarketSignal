"""FastAPI app for the live ML dashboard.

A read-only view of the model's picks, liquidity coverage, track record, and the
market rhythms it learned — rendered fresh from the DB on every request. Launch
with `futmarket dashboard`.

The page is open (no login): it's read-only and meant for the owner's own
machine/LAN. It still ships with hardening headers.
"""

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

    @app.get("/api/health")
    def health():
        # Public liveness probe — reveals nothing sensitive.
        return {"ok": True}

    app.state.access_key = key
    app.state.key_generated = generated
    return app
