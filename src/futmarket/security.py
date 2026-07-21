"""Access control + hardening for the dashboard.

The dashboard exposes buttons that run code on the host (scrape, delete, run
jobs), so the API must not be openly triggerable. This module provides:

  * a single **access key** (from $FUTMARKET_KEY, or generated per-run),
  * cookie **sessions** minted after a key login,
  * path-based **auth** guarding every /api/* route except login + health,
  * **CSRF/origin** checks on state-changing requests,
  * a lightweight in-process **rate limiter** on writes,
  * baseline **security headers** (incl. a strict self-only CSP).

It's deliberately dependency-free and sized for a personal/local tool.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

log = logging.getLogger("futmarket.security")

SESSION_COOKIE = "fut_session"
# /api paths reachable without a session (the login itself + a liveness probe).
_PUBLIC_API = {"/api/login", "/api/health"}
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# rate limit: this many write requests per window (seconds), per client IP.
_RATE_MAX = 60
_RATE_WINDOW = 60.0


def resolve_key() -> tuple[str, bool]:
    """Return (access_key, is_generated). A fixed key comes from FUTMARKET_KEY;
    otherwise we mint a fresh one for this run and report it so the CLI can
    print it for the user to copy."""
    env = os.environ.get("FUTMARKET_KEY", "").strip()
    if env:
        return env, False
    return secrets.token_urlsafe(24), True


class Security:
    """Holds the key + live sessions and answers the middleware's questions."""

    def __init__(self, access_key: str):
        self.access_key = access_key
        self._sessions: set[str] = set()
        self._hits: dict[str, deque] = defaultdict(deque)

    # ---- auth ----
    def check_key(self, candidate: str) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate, self.access_key)

    def new_session(self) -> str:
        token = secrets.token_urlsafe(24)
        self._sessions.add(token)
        return token

    def valid_session(self, token: str | None) -> bool:
        return bool(token) and token in self._sessions

    def drop_session(self, token: str | None) -> None:
        self._sessions.discard(token or "")

    # ---- rate limiting (write requests only) ----
    def rate_ok(self, client: str) -> bool:
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > _RATE_WINDOW:
            q.popleft()
        if len(q) >= _RATE_MAX:
            return False
        q.append(now)
        return True


def same_origin(origin: str | None, host_header: str | None) -> bool:
    """True when a request's Origin matches the host it was sent to. Requests
    with no Origin (same-origin GETs, curl, server-to-server) are allowed; a
    *present, mismatched* Origin is the cross-site (CSRF) case we reject."""
    if not origin:
        return True
    try:
        return urlparse(origin).netloc == (host_header or "")
    except ValueError:
        return False


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # The UI is fully self-hosted (no external scripts/styles/fonts/images).
    # The live ML dashboard ships its CSS in an inline <style> block, so styles
    # allow 'unsafe-inline'; scripts stay locked to 'self' (the real XSS guard).
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}
