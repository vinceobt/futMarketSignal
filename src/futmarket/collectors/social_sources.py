"""Where the crowd talks: Reddit and YouTube.

Both moved behind credentials — Reddit's public .json endpoints now return 403,
and YouTube blocks handle resolution — so these use the official free APIs. Free,
rate-limited generously, and ToS-compliant, which is the whole point of preferring
them over scraping.

What we take is deliberately narrow: post titles and timestamps. That's enough to
count how often a player is being talked about and whether the talk is bullish,
which is the signal; nothing else is worth the storage or the intrusion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .. import secrets
from .base import SourceError

logger = logging.getLogger(__name__)

USER_AGENT = "fc-market-analytics/0.1 (personal FUT price research)"

# Subreddits where cards are actually discussed as tradeable assets.
DEFAULT_SUBREDDITS = ("fut", "EASportsFC", "FIFA")
# Search terms for the trading side of FUT YouTube.
DEFAULT_YT_QUERIES = ("FC 26 trading tips", "FC 26 invest", "FUT 26 market crash")


@dataclass(frozen=True)
class Post:
    """One thing someone said, reduced to what we need."""
    platform: str
    text: str
    created_at: datetime
    score: int = 0          # upvotes / views — how far it travelled
    url: str | None = None


def _ts(epoch) -> datetime:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc)


# ------------------------------------------------------------------ reddit

def reddit_token(client: httpx.Client | None = None) -> str:
    """Application-only OAuth token. No user account, no password."""
    creds = secrets.require("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
    own = client is None
    client = client or httpx.Client(timeout=20.0)
    try:
        resp = client.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(creds["REDDIT_CLIENT_ID"], creds["REDDIT_CLIENT_SECRET"]),
            headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            raise SourceError(f"reddit auth failed ({resp.status_code}): "
                              f"{resp.text[:120]}")
        token = resp.json().get("access_token")
        if not token:
            raise SourceError("reddit auth returned no token")
        return token
    finally:
        if own:
            client.close()


def fetch_reddit(subreddits=DEFAULT_SUBREDDITS, *, limit: int = 100,
                 listing: str = "new", token: str | None = None,
                 client: httpx.Client | None = None) -> list[Post]:
    own = client is None
    client = client or httpx.Client(timeout=25.0)
    try:
        token = token or reddit_token(client)
        headers = {"Authorization": f"bearer {token}", "User-Agent": USER_AGENT}
        posts: list[Post] = []
        for sub in subreddits:
            resp = client.get(f"https://oauth.reddit.com/r/{sub}/{listing}",
                              params={"limit": limit}, headers=headers)
            if resp.status_code != 200:
                logger.warning("reddit r/%s -> %s", sub, resp.status_code)
                continue
            for child in resp.json().get("data", {}).get("children", []):
                d = child.get("data", {})
                title = d.get("title")
                if not title:
                    continue
                posts.append(Post(
                    platform="reddit",
                    text=f"{title} {d.get('selftext', '')[:280]}",
                    created_at=_ts(d.get("created_utc", 0)),
                    score=int(d.get("score") or 0),
                    url=f"https://reddit.com{d.get('permalink', '')}"))
        logger.info("reddit: %d posts from %d subs", len(posts), len(subreddits))
        return posts
    finally:
        if own:
            client.close()


# ----------------------------------------------------------------- youtube

_YT_SEARCH = "https://www.googleapis.com/youtube/v3/search"


def fetch_youtube(queries=DEFAULT_YT_QUERIES, *, per_query: int = 25,
                  days: int = 3, client: httpx.Client | None = None) -> list[Post]:
    """Recent trading videos. Titles carry the call ("BUY THESE NOW")."""
    key = secrets.require("YOUTUBE_API_KEY")["YOUTUBE_API_KEY"]
    own = client is None
    client = client or httpx.Client(timeout=25.0)
    after = datetime.now(timezone.utc).timestamp() - days * 86400
    published_after = datetime.fromtimestamp(after, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    try:
        posts: list[Post] = []
        for q in queries:
            resp = client.get(_YT_SEARCH, params={
                "key": key, "q": q, "part": "snippet", "type": "video",
                "order": "date", "maxResults": per_query,
                "publishedAfter": published_after})
            if resp.status_code != 200:
                logger.warning("youtube %r -> %s: %s", q, resp.status_code,
                               resp.text[:120])
                continue
            for item in resp.json().get("items", []):
                sn = item.get("snippet", {})
                title = sn.get("title")
                if not title:
                    continue
                published = sn.get("publishedAt", "")
                try:
                    when = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except ValueError:
                    continue
                vid = item.get("id", {}).get("videoId")
                posts.append(Post(
                    platform="youtube",
                    text=f"{title} {sn.get('description', '')[:280]}",
                    created_at=when,
                    url=f"https://youtu.be/{vid}" if vid else None))
        logger.info("youtube: %d videos across %d queries", len(posts), len(queries))
        return posts
    finally:
        if own:
            client.close()


# --------------------------------------------------------------- sentiment

# Traders speak plainly, so keywords carry most of the signal a general-purpose
# sentiment model would find -- and they don't misread "crash" in a card name.
_BULLISH = re.compile(
    r"\b(invest|investing|buy|buying|snipe|sniping|profit|rising|moon|"
    r"undervalued|cheap|bargain|hold|gonna rise|going up)\b", re.I)
_BEARISH = re.compile(
    r"\b(dump|dumping|sell|selling|crash|crashing|tank|tanking|falling|"
    r"drop|dropping|overpriced|avoid|losing)\b", re.I)


def lean(text: str) -> float:
    """-1 bearish .. +1 bullish, 0 when it's neither or evenly both."""
    up = len(_BULLISH.findall(text))
    down = len(_BEARISH.findall(text))
    if not (up or down):
        return 0.0
    return round((up - down) / (up + down), 3)
