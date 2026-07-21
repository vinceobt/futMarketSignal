"""Access key + baseline security headers for the dashboard.

The dashboard is read-only, but when exposed on the network it's still gated by a
single access key (from $FUTMARKET_KEY, or generated per run) and served with a
strict set of hardening headers. Dependency-free, sized for a personal tool.
"""

from __future__ import annotations

import os
import secrets
from urllib.parse import urlparse


def same_origin(origin: str | None, host_header: str | None) -> bool:
    """True when a request's Origin matches the host it was sent to. Requests with
    no Origin (same-origin GETs, curl, server-to-server) are allowed; a *present,
    mismatched* Origin is the cross-site (CSRF) case we reject on state changes."""
    if not origin:
        return True
    try:
        return urlparse(origin).netloc == (host_header or "")
    except ValueError:
        return False


def resolve_key() -> tuple[str, bool]:
    """Return (access_key, is_generated). A fixed key comes from FUTMARKET_KEY;
    otherwise we mint a fresh one for this run and report it so the CLI can print
    it for the user to copy."""
    env = os.environ.get("FUTMARKET_KEY", "").strip()
    if env:
        return env, False
    return secrets.token_urlsafe(24), True


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
