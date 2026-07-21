"""X/Twitter leaker feed, driven by a saved login.

Deliberately narrow. It opens a real browser once so you can log in by hand, saves
the cookies, and thereafter only *reads* the profiles you list. It never posts,
likes, follows, or touches anything else, and it paces itself like a person
reading rather than a script hoovering.

Use a burner account. Automated reading is against X's terms, and the realistic
downside is a suspension — which should land on a throwaway, not the account you
actually use.

This is the most fragile collector in the project: X changes its markup often, so
expect the selectors below to need attention. It fails loudly rather than
silently returning nothing, so a break is visible instead of quietly starving the
model of data.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import SourceError
from .social_sources import Post

logger = logging.getLogger(__name__)

DEFAULT_SESSION = ".x_session.json"
LOGIN_URL = "https://x.com/login"
# The cookie X sets once you're actually authenticated.
AUTH_COOKIE = "auth_token"

# Markup selectors. First entry is current; the fallbacks buy time when X shifts.
_TWEET = 'article[data-testid="tweet"]'
_TWEET_TEXT = '[data-testid="tweetText"]'


def save_session(session_file: str | Path = DEFAULT_SESSION, *,
                 timeout_s: int = 300) -> Path:
    """Open a browser, wait for you to log in, then save the cookies.

    Run once per account. Nothing is typed for you — you log in yourself, so no
    password ever passes through this code or gets stored anywhere.
    """
    from patchright.sync_api import sync_playwright

    path = Path(session_file)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # visible: you drive it
        context = browser.new_context()
        page = context.new_page()
        page.goto(LOGIN_URL, wait_until="load")
        print("\n  A browser window has opened.")
        print("  Log in with your BURNER account, then come back here.")
        print(f"  Waiting up to {timeout_s // 60} minutes...\n")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if any(c["name"] == AUTH_COOKIE for c in context.cookies()):
                # settle briefly so the session is fully established
                page.wait_for_timeout(2500)
                context.storage_state(path=str(path))
                browser.close()
                logger.info("saved X session to %s", path)
                return path
            page.wait_for_timeout(1500)

        browser.close()
        raise SourceError("timed out waiting for login — nothing was saved")


def build_session_from_cookies(auth_token: str, ct0: str = "",
                               session_file: str | Path = DEFAULT_SESSION) -> Path:
    """Build a session file from cookies copied out of a browser you're already
    logged into.

    This is the reliable path when X blocks the automated login window: you never
    log in *through* our browser, you just hand it the proof-of-login cookies from
    a tab where you're already signed in. `auth_token` is the session; `ct0` is
    the CSRF token X expects alongside it.
    """
    auth_token = (auth_token or "").strip()
    ct0 = (ct0 or "").strip()
    if not auth_token:
        raise SourceError(
            "auth_token is required — in your logged-in X tab open DevTools "
            "-> Application -> Cookies -> https://x.com and copy the auth_token value")

    expires = time.time() + 365 * 86400          # X refreshes it on use anyway
    cookies = []
    # X still sets cookies on both domains; write both so the read works whichever
    # host a profile URL resolves to.
    for domain in (".x.com", ".twitter.com"):
        cookies.append({"name": AUTH_COOKIE, "value": auth_token, "domain": domain,
                        "path": "/", "expires": expires, "httpOnly": True,
                        "secure": True, "sameSite": "None"})
        if ct0:
            cookies.append({"name": "ct0", "value": ct0, "domain": domain,
                            "path": "/", "expires": expires, "httpOnly": False,
                            "secure": True, "sameSite": "Lax"})

    path = Path(session_file)
    path.write_text(json.dumps({"cookies": cookies, "origins": []}, indent=2))
    logger.info("wrote X session from pasted cookies to %s", path)
    return path


def _has_session(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def fetch_creator_posts(handles, *, session_file: str | Path = DEFAULT_SESSION,
                        per_handle: int = 15, delay_range=(4.0, 9.0),
                        headless: bool = True) -> list[Post]:
    """Read recent posts from each handle. Read-only, paced like a human."""
    from patchright.sync_api import sync_playwright

    path = Path(session_file)
    if not _has_session(path):
        raise SourceError(
            f"no saved X session at {path} — run `futmarket x-login` first")

    posts: list[Post] = []
    now = datetime.now(timezone.utc)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=str(path))
        page = context.new_page()
        try:
            for handle in handles:
                handle = handle.lstrip("@")
                try:
                    page.goto(f"https://x.com/{handle}", wait_until="domcontentloaded")
                    page.wait_for_timeout(3500)
                    page.wait_for_selector(_TWEET, timeout=12000)
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not read @%s: %s", handle, e)
                    continue

                seen = 0
                for node in page.query_selector_all(_TWEET)[:per_handle]:
                    body = node.query_selector(_TWEET_TEXT)
                    if body is None:
                        continue
                    text = (body.inner_text() or "").strip()
                    if not text:
                        continue
                    posts.append(Post(platform="x", text=text, created_at=now,
                                      url=f"https://x.com/{handle}"))
                    seen += 1
                logger.info("@%s: %d posts", handle, seen)
                # Read at human pace; bursts are what gets a session flagged.
                time.sleep(random.uniform(*delay_range))
        finally:
            browser.close()

    if not posts:
        raise SourceError(
            "read 0 posts — the session may have expired (re-run `futmarket "
            "x-login`) or X changed its markup")
    return posts
