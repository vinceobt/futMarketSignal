"""Turning chatter into a per-card buzz signal.

Two jobs: work out which player a post is talking about, and turn that into
something the model can use.

Matching is the hard part and it is deliberately conservative. "Sheriff says
Icons are coming" names no player; "Rodri" matches one; "Rodriguez" matches many
and is therefore worth nothing. A wrong match is worse than no match — it teaches
the model that chatter moves a card it never mentioned — so ambiguous surnames
are dropped rather than guessed at.

The output is buzz *volume* per player-day, plus a crude bullish/bearish lean.
Volume is the real signal: a card being talked about ten times more than usual is
information regardless of whether the talk sounds positive.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone

from .. import db
from ..collectors import social_sources

logger = logging.getLogger(__name__)

# A surname shorter than this matches too much ("Son", "Neto", "Alex").
MIN_NAME_LEN = 5
# A surname shared by more than this many distinct players can't identify anyone.
MAX_PLAYERS_PER_NAME = 3
# Words that look like names but aren't, in this context.
_STOPWORDS = {
    "team", "week", "season", "squad", "build", "chemistry", "objective",
    "evolution", "market", "trading", "invest", "coins", "packs", "player",
    "chance", "future", "united", "madrid", "league", "premier", "champions",
}


def _normalise(text: str) -> str:
    """Strip accents so 'Félix' in a card name matches 'Felix' in a post."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c)).lower()


def build_name_index(conn, *, title: str = "fc26") -> dict[str, list[str]]:
    """Searchable name -> the player_ids it could mean.

    Hype attaches to a *player*, not one of his seven cards, so every card of a
    matched player shares the signal.
    """
    rows = conn.execute(
        "SELECT player_id, name FROM card_meta WHERE title=? AND name IS NOT NULL "
        "AND tradeable=1", (title,)).fetchall()

    by_name: dict[str, set[str]] = defaultdict(set)
    distinct_players: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        full = _normalise(r["name"]).strip()
        if not full:
            continue
        # the base player, so "Joao Felix" counts once however many cards he has
        base = full
        keys = {full}
        parts = [p for p in re.split(r"[\s'-]+", full) if p]
        if parts:
            keys.add(parts[-1])            # surname alone: how people usually write
        for k in keys:
            if len(k) < MIN_NAME_LEN or k in _STOPWORDS:
                continue
            by_name[k].add(r["player_id"])
            distinct_players[k].add(base)

    index = {}
    for name, ids in by_name.items():
        if len(distinct_players[name]) > MAX_PLAYERS_PER_NAME:
            continue                       # too ambiguous to mean anything
        index[name] = sorted(ids)
    logger.info("name index: %d searchable names", len(index))
    return index


def find_mentions(text: str, index: dict[str, list[str]]) -> set[str]:
    """player_ids mentioned in this text. Whole words only."""
    hay = _normalise(text)
    hits: set[str] = set()
    for name, ids in index.items():
        if name in hay and re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", hay):
            hits.update(ids)
    return hits


def collect(conn, *, title: str = "fc26", reddit: bool = True,
            youtube: bool = True, posts=None, now=None) -> dict:
    """Gather posts, attribute them to cards, and store the day's buzz."""
    now = now or datetime.now(timezone.utc)
    if posts is None:
        posts = []
        if reddit:
            try:
                posts += social_sources.fetch_reddit()
            except Exception as e:  # noqa: BLE001 - one platform must not kill the run
                logger.warning("reddit skipped: %s", e)
        if youtube:
            try:
                posts += social_sources.fetch_youtube()
            except Exception as e:  # noqa: BLE001
                logger.warning("youtube skipped: %s", e)

    index = build_name_index(conn, title=title)
    tally: dict[tuple[str, str], dict] = {}
    matched_posts = 0
    for p in posts:
        ids = find_mentions(p.text, index)
        if not ids:
            continue
        matched_posts += 1
        sentiment = social_sources.lean(p.text)
        for pid in ids:
            key = (pid, p.platform)
            slot = tally.setdefault(key, {"n": 0, "sent": 0.0})
            slot["n"] += 1
            slot["sent"] += sentiment

    written = 0
    for (pid, platform), slot in tally.items():
        db.insert_sentiment(conn, player_id=pid, platform=platform, at=now,
                            mention_count=slot["n"],
                            sentiment=round(slot["sent"] / slot["n"], 3))
        written += 1
    conn.commit()
    res = {"posts": len(posts), "matched": matched_posts,
           "cards": written, "names_indexed": len(index)}
    logger.info("social: %s", res)
    return res


def buzz_table(conn, *, title: str = "fc26", limit: int = 20) -> list:
    """Who is being talked about most right now."""
    return conn.execute(
        """SELECT s.player_id, m.name, m.rating, m.version,
                  SUM(s.mention_count) AS mentions, AVG(s.sentiment) AS sentiment
           FROM sentiment_signals s
           LEFT JOIN card_meta m ON m.player_id = s.player_id
           WHERE s.timestamp >= datetime('now', '-2 days')
           GROUP BY s.player_id ORDER BY mentions DESC, sentiment DESC
           LIMIT ?""", (limit,)).fetchall()
