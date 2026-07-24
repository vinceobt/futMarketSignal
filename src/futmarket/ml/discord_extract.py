"""Turn freeform trader chatter into structured buy/sell calls, using Gemini.

Stage 3 of the Discord pipeline (after normalize + flag in
collectors/discord_source.py). Each candidate message is read once and reduced
to zero or more structured calls:

    {card, version, action, price, price_kind, condition, confidence}

Regex can't read "wait for him to drop about 20K and buy" or "Merino may get
cooked if the Enzo SBC is cheap" -- that's why this is an LLM pass. The task is
easy extraction, so it runs on Gemini Flash (free tier), called over plain
httpx like every other source in this project -- no heavy SDK.

Results land in data/discord.db (its own db; never the live market.db).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]

# Models to rotate through. Each (key, model) pair has its own free-tier quota,
# so cycling models multiplies throughput per account. flash-latest is the most
# accurate free option; lite is lighter and its quota is separate.
MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
# Free-tier Flash. Good enough for this extraction; swap for a stronger model
# only if the sample shows it fumbling the judgement calls.
DEFAULT_MODEL = "gemini-flash-latest"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Gemini structured-output schema (OpenAPI subset; inlined, no $ref).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_call": {"type": "boolean"},
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card": {"type": "string"},
                    "version": {"type": "string", "nullable": True},
                    "action": {"type": "string",
                               "enum": ["buy", "sell", "watch", "avoid", "hold"]},
                    "price": {"type": "integer", "nullable": True},
                    "price_kind": {"type": "string",
                                   "enum": ["buy_below", "target", "current",
                                            "drop_to", "sell_at", "range", "unknown"]},
                    "condition": {"type": "string", "nullable": True},
                    "confidence": {"type": "number"},
                },
                "required": ["card", "action", "price_kind", "confidence"],
            },
        },
    },
    "required": ["is_call", "calls"],
}

SYSTEM = """You read messages from professional EA FC (FIFA) Ultimate Team \
trading groups and extract actionable trading calls about specific player cards.

These traders write in shorthand. Decode it:
- Prices: "16k" = 16000 coins; "700c" or a bare "700" = 700 coins; "15.750" = \
15750; "350c" = 350. Always output price as whole coins.
- "<16k", "under 16k", "sub 16k", "below 16k" = buy BELOW that price (price_kind=buy_below).
- "wait for him to drop to/about 20k", "let it fall to X" = drop_to.
- "target 400k", "sell at 26k", "flip for 21k" = target / sell_at.
- A quoted live price with no buy/sell framing = current.
- "Ghana = Kudus FB <135k" style lines are buy lists: one call per player.
- Version/promo tags identify the exact card: TOTS, TOTW, TOTY, Icon, Hero, \
FB (Flashback/Future), IF/Inform, MOTM, POTM, Evo, Unbreakable, Gold, Rare Gold, \
RTTK, UCL, Futties. Put them in `version`.
- Actions: buy (also grab/snipe/pick up/invest/good buy), sell (also dump/list/\
cook/RIP/extinct=sell now), watch (eyeing it, no trigger yet), avoid (warning off), \
hold (keep what you own).
- If the card is only referred to as "him"/"her"/"this one" with no name in THIS \
message, still record the call but set card to the pronoun used (it gets resolved \
later from the surrounding messages).

Judgement:
- is_call=true only if there is a real, actionable tip about a specific card.
- "investments are tricky this week", "market will crash Friday", questions, and \
general hype are NOT calls -> is_call=false, calls=[].
- If unsure a line is a real call, include it with low confidence rather than dropping it.
- Never invent a price or a card that isn't there."""


def _load_key() -> str:
    return load_keys()[0]


def load_keys() -> list[str]:
    """Every GOOGLE_API_KEY* value in .env (GOOGLE_API_KEY, GOOGLE_API_KEY_2, ...)."""
    keys, seen = [], set()
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"^\s*GOOGLE_API_KEY[A-Z0-9_]*\s*=\s*(.+?)\s*$", line)
        if m and m.group(1) and m.group(1) not in seen:
            seen.add(m.group(1))
            keys.append(m.group(1))
    if not keys:
        raise RuntimeError("no GOOGLE_API_KEY* found in .env")
    return keys


def _prompt(channel: str, content: str) -> str:
    return f"[channel: {channel}]\n{content}"


def extract_one(client: httpx.Client, key: str, msg: dict,
                model: str = DEFAULT_MODEL) -> dict:
    """Read one message. `msg` needs `channel` and `content`. Returns the parsed dict."""
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": _prompt(msg["channel"], msg["content"])}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0,
        },
    }
    r = client.post(ENDPOINT.format(model=model), params={"key": key}, json=body,
                    timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def run_sample(db_path: Path, n: int = 15, model: str = DEFAULT_MODEL,
               delay: float = 1.0) -> list[dict]:
    """Extract a diverse sample and return (message, extraction) pairs to eyeball."""
    key = _load_key()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT channel, author, content FROM discord_messages "
        "WHERE is_candidate=1 AND in_fc26=1 AND length(content) BETWEEN 25 AND 400 "
        "ORDER BY RANDOM() LIMIT ?", (n,),
    ).fetchall()
    conn.close()

    out = []
    with httpx.Client() as client:
        for r in rows:
            m = dict(r)
            try:
                out.append({"msg": m, "ext": extract_one(client, key, m, model)})
            except Exception as e:
                out.append({"msg": m, "error": f"{type(e).__name__}: {e}"})
            time.sleep(delay)  # free-tier rate-limit courtesy
    return out


# --------------------------------------------------------------------------
# The full, resumable run: read every FC-26 candidate through a rotating pool
# of (key, model) lanes, pacing gently and parking any lane that hits a limit.
# --------------------------------------------------------------------------

TABLES = """
CREATE TABLE IF NOT EXISTS discord_extractions (
  msg_id        TEXT PRIMARY KEY,
  is_call       INTEGER,
  n_calls       INTEGER,
  model         TEXT,
  error         TEXT,            -- set if the read failed after retries
  raw_json      TEXT,
  extracted_at  TEXT
);
CREATE TABLE IF NOT EXISTS discord_calls (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  msg_id        TEXT NOT NULL,
  card          TEXT,
  version       TEXT,
  action        TEXT,
  price         INTEGER,
  price_kind    TEXT,
  condition     TEXT,
  confidence    REAL,
  group_name    TEXT,
  channel       TEXT,
  author        TEXT,
  timestamp     TEXT             -- when the call was posted (the tradeable moment)
);
CREATE INDEX IF NOT EXISTS idx_calls_msg ON discord_calls(msg_id);
CREATE INDEX IF NOT EXISTS idx_calls_author ON discord_calls(author);
"""


class Lane:
    """One (key, model) quota bucket, with its own cooldown and pacing."""

    def __init__(self, key: str, kidx: int, model: str, per_call_delay: float):
        self.key, self.kidx, self.model = key, kidx, model
        self.per_call_delay = per_call_delay
        self.ready_at = 0.0    # monotonic time this lane may next be used
        self.fails = 0         # consecutive 429s -> longer cooldowns
        self.dead = False      # daily quota exhausted; parked for this run
        self.done = 0

    @property
    def name(self) -> str:
        return f"key{self.kidx}/{self.model.replace('gemini-', '')}"

    def hit_limit(self):
        self.fails += 1
        # 60s, 2m, 5m, 10m ... then give up on this lane for the run.
        backoff = [60, 120, 300, 600][min(self.fails - 1, 3)]
        self.ready_at = time.monotonic() + backoff
        if self.fails >= 5:
            self.dead = True

    def ok(self):
        self.fails = 0
        self.done += 1
        self.ready_at = time.monotonic() + self.per_call_delay


def _acquire(lanes: list[Lane]) -> Lane | None:
    """Return a usable lane, sleeping until one frees up. None if all dead."""
    while True:
        live = [ln for ln in lanes if not ln.dead]
        if not live:
            return None
        now = time.monotonic()
        ready = [ln for ln in live if ln.ready_at <= now]
        if ready:
            return min(ready, key=lambda ln: ln.done)  # spread the load
        time.sleep(min(ln.ready_at for ln in live) - now + 0.05)


def _store(conn, row: dict, ext: dict | None, model: str, error: str | None):
    calls = (ext or {}).get("calls", []) if ext else []
    conn.execute(
        "INSERT OR REPLACE INTO discord_extractions VALUES (?,?,?,?,?,?,?)",
        (row["msg_id"], int(bool(ext and ext.get("is_call"))), len(calls),
         model, error, json.dumps(ext) if ext else None,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.execute("DELETE FROM discord_calls WHERE msg_id=?", (row["msg_id"],))
    for c in calls:
        conn.execute(
            "INSERT INTO discord_calls (msg_id,card,version,action,price,price_kind,"
            "condition,confidence,group_name,channel,author,timestamp) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["msg_id"], c.get("card"), c.get("version"), c.get("action"),
             c.get("price"), c.get("price_kind"), c.get("condition"),
             c.get("confidence"), row["group_name"], row["channel"],
             row["author"], row["timestamp"]),
    )
    conn.commit()


def run_full(db_path: Path, per_call_delay: float = 6.0, log_every: int = 25):
    """Process all un-read FC-26 candidates. Resumable; safe to re-run."""
    keys = load_keys()
    lanes = [Lane(k, i + 1, m, per_call_delay)
             for i, k in enumerate(keys) for m in MODELS]
    conn = sqlite3.connect(db_path)
    conn.executescript(TABLES)
    conn.row_factory = sqlite3.Row

    pending = conn.execute(
        "SELECT msg_id, group_name, channel, author, timestamp, content "
        "FROM discord_messages WHERE is_candidate=1 AND in_fc26=1 "
        "AND msg_id NOT IN (SELECT msg_id FROM discord_extractions) "
        "ORDER BY timestamp"
    ).fetchall()

    total = len(pending)
    print(f"pool: {len(keys)} key(s) x {len(MODELS)} model(s) = {len(lanes)} lanes | "
          f"{total} messages to read", flush=True)
    done = calls = 0
    with httpx.Client() as client:
        for row in pending:
            r = dict(row)
            while True:
                lane = _acquire(lanes)
                if lane is None:
                    print(f"all lanes hit their daily cap. {done}/{total} done this "
                          f"run; re-run to continue tomorrow.", flush=True)
                    conn.close()
                    return
                try:
                    ext = extract_one(client, lane.key, r, model=lane.model)
                    _store(conn, r, ext, lane.model, None)
                    lane.ok()
                    done += 1
                    calls += len(ext.get("calls", []))
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        lane.hit_limit()
                        continue  # same message, next lane
                    _store(conn, r, None, lane.model, f"http{e.response.status_code}")
                    lane.ok()
                    done += 1
                    break
                except Exception as e:  # noqa: BLE001 - network/parse; retry once via loop
                    lane.fails += 1
                    lane.ready_at = time.monotonic() + 10
                    if lane.fails >= 5:
                        _store(conn, r, None, lane.model, f"{type(e).__name__}")
                        lane.dead = True
                        done += 1
                        break
            if done % log_every == 0:
                live = sum(1 for ln in lanes if not ln.dead)
                print(f"  {done}/{total}  ({calls} calls)  live lanes: {live}/{len(lanes)}",
                      flush=True)
    print(f"DONE: read {done} messages, extracted {calls} calls.", flush=True)
    conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        delay = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
        run_full(ROOT / "data" / "discord.db", per_call_delay=delay)
    else:  # sample mode: `python -m ... [n]`
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
        print(json.dumps(run_sample(ROOT / "data" / "discord.db", n=n),
                         indent=2, default=str))
