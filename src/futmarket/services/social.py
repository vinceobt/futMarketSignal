"""Turning chatter into a per-card buzz signal.

Two jobs: work out which player a post is talking about, and turn that into
something the model can use.

Matching is the hard part and it is deliberately conservative. "Sheriff says
Icons are coming" names no player; "Rodri" matches one. A shared surname is
resolved by prominence: bare "Messi" means the superstar Lionel, not his obscure
namesake Rayane, and bare "Simon" — where no one player stands out — is dropped
rather than smeared across a dozen cards. A wrong match is worse than no match: it
teaches the model that chatter moved a card it never mentioned.

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
# When several players share a surname, attribute it to the top one only if it
# out-ranks the runner-up by this much on the prominence score (peak rating plus a
# liquidity nudge). "Messi" clears it (Lionel 97 over Rayane 89); "Simon" doesn't
# (a crowd of 74–90s with no standout), so it's dropped.
PROMINENCE_MARGIN = 6.0
# The leaker/trader accounts worth reading. Leak accounts break promo news early;
# trading accounts drive hype on specific cards.
X_CREATORS = ("futSheriff", "FutPoliceLeaks", "Fut_scoreboard", "fifa22_info",
              "fifa_romania", "EASFCDirect", "futagentt", "AsyFutTrader",
              "Futdonk", "TradingEi", "FUT_Accountant")
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


def _person(player_id: str) -> str:
    """The stable player behind a card. fut.gg encodes it as the id's leading
    segment, so all of Lionel Messi's cards share '158023' while Rayane Messi is
    '75421' — the right key for 'hype attaches to a person, not a card'."""
    return player_id.split("-", 1)[0]


def _resolve(persons: dict[str, dict]) -> set[str] | None:
    """Which person(s) a name attributes to, or None to drop it.

    One candidate -> that person. Several -> the prominence leader, but only if it
    clears the field by PROMINENCE_MARGIN; an evenly-matched crowd names nobody.
    """
    if len(persons) == 1:
        return set(persons)
    ranked = sorted(persons.values(), key=lambda p: p["prom"], reverse=True)
    if ranked[0]["prom"] - ranked[1]["prom"] < PROMINENCE_MARGIN:
        return None                        # no standout — too ambiguous to mean anyone
    top = max(persons.items(), key=lambda kv: kv[1]["prom"])[0]
    return {top}


def build_name_index(conn, *, title: str = "fc26") -> dict[str, list[str]]:
    """Searchable name -> the card ids it means.

    Hype attaches to a *player*, not one of his seven cards, so every card of a
    matched player shares the signal. A surname shared by several players is
    resolved to the prominent one (or dropped) via `_resolve`.
    """
    rows = conn.execute(
        """SELECT m.player_id, m.name, COALESCE(m.rating, 0) AS rating,
                  COALESCE(l.score, 0) AS liq
           FROM card_meta m
           LEFT JOIN liquidity l ON l.player_id = m.player_id AND l.title = m.title
           WHERE m.title=? AND m.name IS NOT NULL AND m.tradeable=1""",
        (title,)).fetchall()

    # name -> person -> {"cards": {card_id, ...}, "prom": best-card prominence}
    by_name: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        full = _normalise(r["name"]).strip()
        if not full:
            continue
        pid = _person(r["player_id"])
        # rating carries fame; liquidity nudges toward the card people actually trade
        prom = r["rating"] + r["liq"] / 2.0
        keys = {full}
        parts = [p for p in re.split(r"[\s'-]+", full) if p]
        if parts:
            keys.add(parts[-1])            # surname alone: how people usually write
        for k in keys:
            if len(k) < MIN_NAME_LEN or k in _STOPWORDS:
                continue
            slot = by_name[k].setdefault(pid, {"cards": set(), "prom": 0.0})
            slot["cards"].add(r["player_id"])
            slot["prom"] = max(slot["prom"], prom)

    index = {}
    for name, persons in by_name.items():
        chosen = _resolve(persons)
        if not chosen:
            continue
        cards: set[str] = set()
        for pid in chosen:
            cards |= persons[pid]["cards"]
        index[name] = sorted(cards)
    logger.info("name index: %d searchable names", len(index))
    return index


def find_mentions(text: str, index: dict[str, list[str]]) -> set[str]:
    """card ids mentioned in this text. Whole words only.

    Longest names first, and a match is blanked out once found, so 'Rayane Messi'
    is consumed as a whole and doesn't also fire the bare 'Messi' -> Lionel rule.
    """
    hay = _normalise(text)
    hits: set[str] = set()
    for name in sorted(index, key=len, reverse=True):
        m = re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", hay)
        if m:
            hits.update(index[name])
            hay = f"{hay[:m.start()]}{' ' * (m.end() - m.start())}{hay[m.end():]}"
    return hits


def collect(conn, *, title: str = "fc26", reddit: bool = True,
            youtube: bool = True, x: bool = False, posts=None,
            now=None, x_handles=None) -> dict:
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
        if x:
            try:
                from ..collectors import x_source
                posts += x_source.fetch_creator_posts(x_handles or X_CREATORS)
            except Exception as e:  # noqa: BLE001
                logger.warning("x skipped: %s", e)

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


def buzz_table(conn, *, title: str = "fc26", limit: int = 20,
               days: int = 2) -> list:
    """Who is being talked about most right now.

    Grouped by player, not card. Hype attaches to a person, and storing it
    against each of their cards is right for the model but reads as the same name
    ten times in a row for a human.
    """
    return conn.execute(
        """SELECT m.name AS name,
                  MAX(s.mention_count) AS mentions,
                  AVG(s.sentiment)     AS sentiment,
                  COUNT(DISTINCT s.player_id) AS cards,
                  GROUP_CONCAT(DISTINCT s.platform) AS platforms
           FROM sentiment_signals s
           JOIN card_meta m ON m.player_id = s.player_id
           WHERE s.timestamp >= datetime('now', ?) AND m.name IS NOT NULL
           GROUP BY m.name
           ORDER BY mentions DESC, sentiment DESC
           LIMIT ?""", (f"-{days} days", limit)).fetchall()
