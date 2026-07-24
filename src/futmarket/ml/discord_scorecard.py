"""Grade Discord traders' calls against what prices actually did.

Stage 5 of the pipeline (after extract). For every named BUY call we:
  1. resolve the card -- name + promo tag + PRICE-ANCHORING (the call quotes a
     price, so among the seven Joao Felixes we pick the one that was actually
     trading near that price on that day),
  2. honour the call's intent: a "buy below X" only counts if the market actually
     reached X -- and then you enter THERE, not at the (higher) price when it was
     posted. A plain "buy now" enters at the market price at the call.
  3. score the move 3/7 days later, net of the 5% EA sell tax,
  4. aggregate per caller -> the honest, measure-from-price scoreboard.

Calls live in data/discord.db; prices in data/market.db (read-only here). Only
real `futgg` snapshots are used -- the `turnstile_mock` source is ignored.
"""

from __future__ import annotations

import re
import sqlite3
import statistics
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TAX = 0.05
MIN_TOKEN = 4              # non-surname tokens must be this long to index (first names)
MIN_SURNAME = 3
FILL_WINDOW = 3           # days a "buy below X" has to actually reach X
PRONOUNS = {"him", "her", "it", "he", "she", "they", "them", "this", "that",
            "these", "those", "this one", "the card", "card", "guy", "man", "one"}


def _norm(s: str) -> str:
    """Lowercase, fold accents (Konaté->konate), keep letters+spaces."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()


def _to_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def build_index(m: sqlite3.Connection) -> dict[str, list[dict]]:
    """name / surname / first-name -> candidate cards from card_meta."""
    idx: dict[str, list[dict]] = {}
    for pid, name, rating, version, tradeable in m.execute(
        "SELECT player_id, name, rating, version, COALESCE(tradeable,1) "
        "FROM card_meta"
    ):
        card = {"pid": pid, "name": name, "rating": rating or 0,
                "version": version or "", "tradeable": tradeable}
        n = _norm(name)
        parts = n.split()
        keys = {n}
        for i, t in enumerate(parts):
            last = i == len(parts) - 1
            if (last and len(t) >= MIN_SURNAME) or (not last and len(t) >= MIN_TOKEN):
                keys.add(t)
        for k in keys:
            idx.setdefault(k, []).append(card)
    return idx


def series(m, pid, start, end):
    """Sorted [(datetime, price)] of real futgg snapshots in [start, end]."""
    rows = m.execute(
        "SELECT timestamp, price FROM price_snapshots "
        "WHERE player_id=? AND source='futgg' AND timestamp BETWEEN ? AND ?",
        (pid, start.isoformat(), end.isoformat()),
    ).fetchall()
    return sorted((_to_utc(t), p) for t, p in rows)


def _near(ser, when):
    return min(ser, key=lambda r: abs((r[0] - when).total_seconds()))[1] if ser else None


VERSION_HINTS = {
    "tots": "team of the season", "totw": "team of the week", "toty": "team of the year",
    "icon": "icon", "hero": "hero", "fb": "flashback", "if": "team of the week",
    "motm": "man of the match", "potm": "potm", "evo": "evolution",
    "unbreakable": "unbreakable", "futties": "futties", "rttk": "road to",
    "ucl": "champions", "gold": "rare", "silver": "silver", "rare": "rare",
}


def resolve(m, idx, call):
    name = _norm(call["card"])
    if not name or name in PRONOUNS or len(name) < MIN_SURNAME:
        return None
    cands = idx.get(name)
    if not cands:                       # gather across the name's tokens
        seen, cands = set(), []
        for t in (t for t in name.split() if len(t) >= MIN_SURNAME):
            for c in idx.get(t, []):
                if c["pid"] not in seen:
                    seen.add(c["pid"]); cands.append(c)
    if not cands:
        return None
    when = _to_utc(call["timestamp"])
    price, vtag = call.get("price"), (call.get("version") or "").lower()
    if price:                           # price-anchored
        scored = []
        for c in cands:
            p = _near(series(m, c["pid"], when - timedelta(days=3), when + timedelta(days=3)), when)
            if p is None:
                continue
            ratio = max(p, price) / max(1, min(p, price))
            vbonus = 0.85 if (vtag and VERSION_HINTS.get(vtag, "~~") in c["version"].lower()) else 1.0
            scored.append((ratio * vbonus, c["pid"]))
        if not scored:
            return None
        best, pid = min(scored)
        return pid if best <= 2.2 else None
    return max(cands, key=lambda c: (c["tradeable"], c["rating"]))["pid"]


def _entry(ser, when, call):
    """(entry_price, fill_time) honouring the call's intent, or (None, None)."""
    price, kind = call.get("price"), call.get("price_kind")
    if price and kind in ("buy_below", "drop_to"):
        # limit buy: fill at the target only if the market actually reaches it
        for dt, p in ser:
            if dt >= when and p <= price * 1.02:
                return price, dt
        return None, None               # never triggered -> not a fill
    p = _near([r for r in ser if r[0] >= when - timedelta(days=2)], when)
    return (p, when) if p else (None, None)


def grade(db=ROOT / "data" / "discord.db", market=ROOT / "data" / "market.db",
          horizons=(3, 7)):
    d = sqlite3.connect(db); d.row_factory = sqlite3.Row
    m = sqlite3.connect(market)
    idx = build_index(m)
    calls = d.execute(
        "SELECT author, card, version, price, price_kind, timestamp "
        "FROM discord_calls WHERE action='buy' ORDER BY timestamp"
    ).fetchall()

    graded, resolved, filled = [], 0, 0
    for row in calls:
        call = dict(row)
        pid = resolve(m, idx, call)
        if not pid:
            continue
        resolved += 1
        when = _to_utc(call["timestamp"])
        ser = series(m, pid, when - timedelta(days=2), when + timedelta(days=max(horizons) + 2))
        entry, fill_t = _entry(ser, when, call)
        if not entry:
            continue
        filled += 1
        rec = {"author": call["author"], "card": call["card"], "entry": entry}
        for h in horizons:
            ex = _near(ser, fill_t + timedelta(days=h))
            rec[f"r{h}"] = (ex * (1 - TAX) - entry) / entry if ex else None
        graded.append(rec)

    print(f"buy calls: {len(calls)} | resolved: {resolved} "
          f"({100*resolved/max(1,len(calls)):.0f}%) | filled & graded: {filled}")
    return graded


def scoreboard(graded, horizon=3, min_calls=8):
    key = f"r{horizon}"
    by: dict[str, list[float]] = {}
    for g in graded:
        if g.get(key) is not None:
            by.setdefault(g["author"], []).append(g[key])
    rows = [{"caller": a, "buys": len(rs), "median": statistics.median(rs),
             "mean": statistics.fmean(rs), "win": sum(x > 0 for x in rs) / len(rs)}
            for a, rs in by.items() if len(rs) >= min_calls]
    rows.sort(key=lambda r: r["median"], reverse=True)
    return rows


if __name__ == "__main__":
    g = grade()
    all_r = [x["r3"] for x in g if x.get("r3") is not None]
    print(f"\n=== per-caller scoreboard @ +3 days, net of {int(TAX*100)}% tax ===")
    print(f"{'caller':16} {'buys':>5} {'median':>8} {'mean':>8} {'win%':>6}")
    for r in scoreboard(g, horizon=3, min_calls=8):
        print(f"{r['caller']:16} {r['buys']:>5} {r['median']*100:>7.1f}% "
              f"{r['mean']*100:>7.1f}% {r['win']*100:>5.0f}%")
    if all_r:
        print(f"\noverall median @3d: {statistics.median(all_r)*100:+.1f}%  "
              f"across {len(all_r)} filled buys")
