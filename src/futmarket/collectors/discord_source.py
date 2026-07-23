"""Discord trader-group exports -> a clean, queryable corpus.

One-time bulk exports (DiscordChatExporter JSON) of paid FUT trading groups are
mined for buy/sell tips. This module does the cheap, no-cost half:

  1. normalize   every message from every channel -> one flat SQLite table,
                 losing nothing (all years kept, not just FC 26).
  2. flag        mark which messages are *candidate tips* worth reading closely.

The flag is deliberately generous (recall over precision): the only cost of a
false positive is that the downstream reader skips a dud, whereas a dropped tip
is gone. So we cast a wide net -- price figures, action words, promo/version
tags, fut.gg links, or an attached card image all qualify.

Storage is a SEPARATE db (data/discord.db), never the live market.db, to avoid
the lock contention that bites when the 2-hourly loop is writing (trap #9).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

# --- the FC 26 season: the only window we have prices to grade tips against ---
FC26_START = "2025-09-08"

SCHEMA = """
CREATE TABLE IF NOT EXISTS discord_messages (
  msg_id          TEXT PRIMARY KEY,
  group_name      TEXT NOT NULL,     -- Discord server (TEDDY127 / ASY FUT TRADE)
  channel         TEXT NOT NULL,
  author          TEXT NOT NULL,
  timestamp       DATETIME NOT NULL, -- ISO UTC
  ts_date         DATE NOT NULL,     -- YYYY-MM-DD, for fast season filtering
  content         TEXT NOT NULL,
  n_attachments   INTEGER NOT NULL DEFAULT 0,
  has_card_image  INTEGER NOT NULL DEFAULT 0,
  reactions       INTEGER NOT NULL DEFAULT 0,
  reply_to        TEXT,
  is_candidate    INTEGER NOT NULL DEFAULT 0,  -- worth reading closely?
  in_fc26         INTEGER NOT NULL DEFAULT 0,  -- within the gradeable season?
  flag_reasons    TEXT                          -- why it was flagged (comma sep)
);
CREATE INDEX IF NOT EXISTS idx_dm_date ON discord_messages(ts_date);
CREATE INDEX IF NOT EXISTS idx_dm_cand ON discord_messages(is_candidate, in_fc26);
CREATE INDEX IF NOT EXISTS idx_dm_author ON discord_messages(author);
"""

# A price/coin figure: "24k", "15.750", "700c", "<16k", "under 19k", "around 20"
RE_PRICE = re.compile(
    r"(?<![a-z])(?:\d[\d,\.]*\s*[kc]\b|<\s*\d|(?:under|below|around|about|sub)\s+\d)",
    re.I,
)
# Trader action verbs -- the vocabulary of an actual call.
RE_ACTION = re.compile(
    r"\b(buy|buying|bought|snipe|sniped|sell|selling|sold|sell now|list|listing|"
    r"invest|investment|investments|flip|flipping|grab|grabbed|pick(?:ed)?\s*up|"
    r"load up|stock up|dump|dumping|hold|holding|target|cook(?:ed)?|"
    r"fodder|good buy|safe buy|profit)\b",
    re.I,
)
# Card / promo / version tags -- these name the *kind* of card, aiding both
# "is this a call" and later disambiguation of which card version is meant.
RE_TAG = re.compile(
    r"\b(tots|totw|toty|tott|motm|potm|pots|if\b|informs?|icon|hero|"
    r"fb\b|flashback|future stars|futures?|rttk|rttf|rulebreaker|ucl|uel|"
    r"evo|evolution|unbreakable|thunderstruck|winter wildcard|futties|"
    r"sbc|rare gold|special)\b",
    re.I,
)
RE_FUTGG = re.compile(r"fut\.gg|futbin|futwiz", re.I)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _has_card_image(msg: dict) -> bool:
    for a in msg.get("attachments", []):
        name = (a.get("fileName") or a.get("url") or "").lower()
        if name.endswith(IMG_EXT):
            return True
    for e in msg.get("embeds", []):
        if RE_FUTGG.search(json.dumps(e)):
            return True
    return False


def _flag(content: str, has_img: bool) -> list[str]:
    """Return the reasons this message looks like a tip (empty = chit-chat)."""
    reasons = []
    if RE_PRICE.search(content):
        reasons.append("price")
    if RE_ACTION.search(content):
        reasons.append("action")
    if RE_TAG.search(content):
        reasons.append("tag")
    if RE_FUTGG.search(content):
        reasons.append("link")
    if has_img:
        reasons.append("card_image")
    return reasons


def normalize(export_dir: Path, db_path: Path) -> dict:
    """Load every exported channel JSON into discord.db, flagging candidates.

    Idempotent: re-running replaces rows by msg_id, so it is safe to re-run
    after pulling more channels.
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    files = sorted(export_dir.glob("*.json"))
    rows = []
    for f in files:
        data = json.loads(f.read_text())
        group = data.get("guild", {}).get("name", "?")
        channel = data.get("channel", {}).get("name", f.stem)
        for m in data.get("messages", []):
            content = m.get("content", "") or ""
            has_img = _has_card_image(m)
            reasons = _flag(content, has_img)
            ts = m["timestamp"]
            rows.append((
                m["id"], group, channel, m["author"]["name"], ts, ts[:10],
                content, len(m.get("attachments", [])), int(has_img),
                sum(r.get("count", 0) for r in m.get("reactions", [])),
                (m.get("reference") or {}).get("messageId"),
                int(bool(reasons)), int(ts[:10] >= FC26_START),
                ",".join(reasons) or None,
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO discord_messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    def q(sql):
        return conn.execute(sql).fetchone()[0]

    stats = {
        "files": len(files),
        "total": q("SELECT COUNT(*) FROM discord_messages"),
        "candidates": q("SELECT COUNT(*) FROM discord_messages WHERE is_candidate=1"),
        "fc26_candidates": q(
            "SELECT COUNT(*) FROM discord_messages WHERE is_candidate=1 AND in_fc26=1"
        ),
        "fc26_total": q("SELECT COUNT(*) FROM discord_messages WHERE in_fc26=1"),
    }
    conn.close()
    return stats


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3]
    stats = normalize(root / "data" / "discord", root / "data" / "discord.db")
    print(stats)
