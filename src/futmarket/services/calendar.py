"""Game calendar — the lifecycle backbone.

Two derived timelines, both grounded in real data rather than hand-typed dates:

  PROMO/TOTW  from the card registry itself. A promo *is* its card drop, so the
              first time a rarity (version) appears in card_meta is that promo's
              launch date. Team of the Week recurs weekly under one version name,
              so it's split into one event per release day.
  SBC         from fut.gg's SBC feed, which carries real start + expiry windows.

Both are written with a `source` tag so they can be rebuilt idempotently without
disturbing events a human logged via `futmarket log-event`.
"""

from __future__ import annotations

import logging

from .. import db
from ..collectors import sbc_source

logger = logging.getLogger(__name__)

DERIVED_SOURCE = "derived_cards"
SBC_SOURCE = "futgg_sbc"

# Base-rarity cards ship with the game; their release is the launch, not a promo.
BASE_VERSIONS = {"common", "rare", "bronze", "silver", "gold",
                 "common gold", "rare gold", "common silver", "rare silver",
                 "common bronze", "rare bronze"}
# Versions that recur in weekly batches under a single name.
WEEKLY_VERSIONS = ("team of the week",)
# Ignore one-off/no-cohort versions with fewer cards than this (noise).
MIN_CARDS_PER_PROMO = 3


def _is_weekly(version: str) -> bool:
    v = version.lower()
    return any(w in v for w in WEEKLY_VERSIONS)


def derive_card_release_events(conn, *, title: str = "fc26") -> list[dict]:
    """Promo/TOTW events inferred from when each version's cards first appeared."""
    rows = conn.execute(
        """SELECT version, release_date, COUNT(*) AS n
           FROM card_meta
           WHERE title=? AND release_date IS NOT NULL AND version != ''
           GROUP BY version, release_date
           ORDER BY version, release_date""",
        (title,),
    ).fetchall()

    by_version: dict[str, list[tuple[str, int]]] = {}
    for r in rows:
        by_version.setdefault(r["version"], []).append((r["release_date"], r["n"]))

    events: list[dict] = []
    for version, days in by_version.items():
        if version.strip().lower() in BASE_VERSIONS:
            continue  # launch-day base cards, not an event
        if _is_weekly(version):
            # one event per weekly batch (a single "TOTW" blob would be useless)
            for day, n in days:
                if n >= MIN_CARDS_PER_PROMO:
                    events.append({"event_type": "TOTW", "player_id": None,
                                   "start_date": day, "end_date": None,
                                   "notes": f"{version} ({day})"})
            continue
        total = sum(n for _, n in days)
        if total < MIN_CARDS_PER_PROMO:
            continue
        start = days[0][0]
        end = days[-1][0]
        events.append({"event_type": "PROMO", "player_id": None,
                       "start_date": start,
                       "end_date": end if end != start else None,
                       "notes": f"{version} ({total} cards)"})
    events.sort(key=lambda e: e["start_date"])
    return events


def set_launch_from_registry(conn, *, title: str = "fc26") -> str | None:
    """Anchor the lifecycle: the game's launch = the earliest card release date."""
    row = conn.execute(
        "SELECT MIN(release_date) AS d FROM card_meta WHERE title=? AND release_date IS NOT NULL",
        (title,)).fetchone()
    launch = row["d"] if row else None
    if launch:
        db.set_game_launch(conn, title=title, launch_date=launch,
                           notes="derived: earliest card release in registry")
        conn.commit()
    return launch


def build_calendar(conn, *, title: str = "fc26", game: str = "26",
                   include_sbc: bool = True, sbc_max_pages: int | None = None,
                   sbc_client=None, delay: float = 0.4) -> dict:
    """Rebuild the derived calendar: launch anchor + promo/TOTW + SBC windows."""
    launch = set_launch_from_registry(conn, title=title)

    card_events = derive_card_release_events(conn, title=title)
    db.replace_events(conn, source=DERIVED_SOURCE, events=card_events, title=title)

    sbc_events: list[dict] = []
    if include_sbc:
        sbc_events = list(sbc_source.iter_sbcs(game, max_pages=sbc_max_pages,
                                               delay=delay, client=sbc_client))
        db.replace_events(conn, source=SBC_SOURCE, events=sbc_events, title=title)

    res = {
        "launch": launch,
        "promo": sum(1 for e in card_events if e["event_type"] == "PROMO"),
        "totw": sum(1 for e in card_events if e["event_type"] == "TOTW"),
        "sbc": len(sbc_events),
    }
    logger.info("calendar built: %s", res)
    return res
