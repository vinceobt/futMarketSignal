"""Credential loading, kept out of the tracked config file.

config.yaml is committed, so anything secret in it would be published.
Credentials belong in a gitignored .env, or in the real environment, and
nowhere else.

No dependency: .env here is a handful of KEY=value lines, which is all we need.
Real environment variables always win, so a server can set them properly without
a file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE = ".env"
_loaded = False


def load_env(path: str | Path = ENV_FILE, *, override: bool = False) -> int:
    """Read KEY=value lines into the environment. Returns how many were set.

    Values may be quoted; `#` starts a comment; blank lines are skipped. Existing
    environment variables are left alone unless `override` is set, so deployment
    config beats a stray local file.
    """
    global _loaded
    p = Path(path)
    if not p.exists():
        _loaded = True
        return 0
    count = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
            count += 1
    _loaded = True
    logger.debug("loaded %d values from %s", count, p)
    return count


def get(name: str, default: str | None = None) -> str | None:
    """Read a credential, loading the .env file once on first use."""
    if not _loaded:
        load_env()
    return os.environ.get(name, default)


def require(*names: str) -> dict[str, str]:
    """Fetch credentials or explain precisely what is missing and where to get it.

    Raising with the setup steps beats a downstream 401 that says nothing.
    """
    missing, found = [], {}
    for n in names:
        v = get(n)
        if v:
            found[n] = v
        else:
            missing.append(n)
    if missing:
        raise MissingCredentials(missing)
    return found


class MissingCredentials(RuntimeError):
    HELP = {
        "REDDIT_CLIENT_ID": "reddit.com/prefs/apps -> create app -> type 'script'",
        "REDDIT_CLIENT_SECRET": "the secret shown beside that same app",
        "YOUTUBE_API_KEY": "console.cloud.google.com -> enable YouTube Data API v3 -> create an API key",
        "X_SESSION_FILE": "path to the saved X login cookies (see `futmarket x-login`)",
    }

    def __init__(self, missing: list[str]):
        lines = [f"missing credentials: {', '.join(missing)}", "",
                 "Add them to a .env file in the project root (it is gitignored):"]
        for m in missing:
            lines.append(f"  {m}=...")
            if m in self.HELP:
                lines.append(f"      get it from: {self.HELP[m]}")
        lines += ["", "All of these are free. Nothing here requires a paid plan."]
        super().__init__("\n".join(lines))
        self.missing = missing
