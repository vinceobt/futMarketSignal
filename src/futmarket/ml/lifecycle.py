"""Lifecycle-relative features — where we are in the game's season.

This is the forward-transfer engine. Describing a card by *calendar date* teaches
the model nothing reusable ("2026-03-14" never recurs). Describing it by its
position in the season — 45 days after launch, 3 days before a promo, mid-TOTS —
means a pattern learned in FC26 still applies in FC27, because every game repeats
the same rhythm: launch -> promo cadence -> endgame.

Pure lookups over the calendar built in Phase 2 (game_calendar + market_events).
No future leakage: every value is computed from the event timeline as it stands,
and "days to next event" is knowable at the time (promos are announced ahead).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime

# Event types we build lifecycle distances for. ANNOUNCE is EA's own promo
# announcement, kept separate from the card drop it precedes: measured across 41
# announcements, the market drifts up beforehand, falls ~0.5-0.7pp below a normal
# day on the announcement and the day after, then recovers to ~0.8pp above by day
# six. Folding that into the card-release PROMO signal would blur two different
# moments -- the news, and the supply that follows it.
TRACKED_TYPES = ("PROMO", "TOTW", "SBC", "ANNOUNCE")
# Events from this source are announcements rather than card releases.
ANNOUNCEMENT_SOURCE = "ea_news"


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


class Lifecycle:
    """Indexed view of one title's calendar, answering per-timestamp questions.

    Built once per dataset run and queried per row, so the lookups are bisects
    rather than repeated scans over the event table.
    """

    def __init__(self, *, launch_date=None, events=()):
        self.launch = _as_date(launch_date)
        self._starts: dict[str, list[date]] = {}
        self._windows: dict[str, list[tuple[date, date]]] = {}
        for e in events:
            etype = e["event_type"]
            start = _as_date(e["start_date"])
            if start is None:
                continue
            self._starts.setdefault(etype, []).append(start)
            end = _as_date(e.get("end_date"))
            if end is not None and end >= start:
                self._windows.setdefault(etype, []).append((start, end))
        for starts in self._starts.values():
            starts.sort()

    # -- individual lookups -------------------------------------------------

    def days_since_launch(self, at) -> int | None:
        d = _as_date(at)
        if d is None or self.launch is None:
            return None
        return (d - self.launch).days

    def days_to_next(self, at, event_type: str) -> int | None:
        """Days until the next event of this type (0 = today). None if none ahead."""
        d = _as_date(at)
        starts = self._starts.get(event_type)
        if d is None or not starts:
            return None
        i = bisect_left(starts, d)
        return (starts[i] - d).days if i < len(starts) else None

    def days_since_last(self, at, event_type: str) -> int | None:
        """Days since the most recent event of this type at/before `at`."""
        d = _as_date(at)
        starts = self._starts.get(event_type)
        if d is None or not starts:
            return None
        i = bisect_right(starts, d) - 1
        return (d - starts[i]).days if i >= 0 else None

    def active_count(self, at, event_type: str) -> int:
        """How many windowed events of this type are live on `at` (SBCs mostly)."""
        d = _as_date(at)
        if d is None:
            return 0
        return sum(1 for s, e in self._windows.get(event_type, []) if s <= d <= e)

    # -- the feature row ----------------------------------------------------

    def features(self, at) -> dict:
        """All lifecycle features for one timestamp, ready to join to a sample."""
        row: dict[str, int | None] = {"days_since_launch": self.days_since_launch(at)}
        for etype in TRACKED_TYPES:
            key = etype.lower()
            row[f"days_to_next_{key}"] = self.days_to_next(at, etype)
            row[f"days_since_last_{key}"] = self.days_since_last(at, etype)
        row["active_sbc_count"] = self.active_count(at, "SBC")
        d = _as_date(at)
        # Thu-Sun: the Weekend League demand window that lifts meta cards.
        row["is_weekend_window"] = int(d.weekday() in (3, 4, 5, 6)) if d else 0
        return row


def load(conn, *, title: str = "fc26") -> Lifecycle:
    """Build a Lifecycle from the stored calendar for one title."""
    from .. import db
    launch = db.game_launch(conn, title)
    events = []
    for r in db.events_list(conn, title=title):
        etype = r["event_type"]
        # An EA promo article is the announcement; the card drop is tracked
        # separately by the release-derived PROMO events.
        if _source_of(r) == ANNOUNCEMENT_SOURCE and etype == "PROMO":
            etype = "ANNOUNCE"
        events.append({"event_type": etype, "start_date": r["start_date"],
                       "end_date": r["end_date"]})
    return Lifecycle(launch_date=launch, events=events)


def _source_of(row) -> str | None:
    try:
        return row["source"]
    except (KeyError, IndexError):
        return None
