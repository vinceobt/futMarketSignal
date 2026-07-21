"""Classify a calendar event into a normalised promo *type* from its note.

The lifecycle features already tell the model *when* a promo lands; this reads
*what kind* it is (Icon, Hero, Team of the Season, …), so the market's very
different reactions to different promos become learnable and measurable instead
of all being lumped together as "a promo".

Deliberately a small keyword map — FUT promo names are stable and human-readable
in the calendar notes. The `MAJOR_TYPES` are the big value-movers whose arrival
reprices large parts of the market.
"""

from __future__ import annotations

import re

# Order matters: the first pattern that matches wins (most specific first).
_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("toty", r"team of the year|\btoty\b"),
    ("tots", r"team of the season|\btots\b"),
    ("icon", r"\bicon"),
    ("hero", r"\bhero"),
    ("foundations", r"foundation"),
    ("festival", r"festival|\bsummer\b"),
    ("rttf", r"road to the|\brttf\b|\brttk\b"),
    ("flashback", r"flashback"),
    ("wildcards", r"wildcard"),
    ("futties", r"futties"),
    ("rulebreakers", r"rulebreaker"),
    ("totw", r"team of the week|\btotw\b"),
    ("sbc_player", r"sbc:.*\(player"),
    ("sbc_upgrade", r"sbc:.*\(upgrade"),
]

# Promo types that reprice big chunks of the market when they drop.
MAJOR_TYPES = {"icon", "hero", "tots", "toty"}


def classify(notes: str | None, event_type: str | None = None) -> str:
    """Normalised promo type from a calendar note, e.g. 'icon', 'tots', 'sbc_player'.
    Falls back to the broad event_type (lowered) then 'other'."""
    text = (notes or "").lower()
    for name, pattern in _TYPE_PATTERNS:
        if re.search(pattern, text):
            return name
    return (event_type or "other").lower()


def is_major(notes: str | None, event_type: str | None = None) -> bool:
    return classify(notes, event_type) in MAJOR_TYPES
