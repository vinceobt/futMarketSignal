"""Populate the DB with realistic *synthetic* history for the whole watchlist so
the dashboard can be previewed end-to-end. This is DEMO tooling — it invents
prices; it is not a data source. Run:  python scripts/seed_demo.py

Deterministic (fixed seed) so the preview is stable across runs.
"""
import math
import random
from datetime import datetime, timedelta, timezone

from futmarket import db
from futmarket.config import load_config

DAYS = 10
STEP_H = 1                      # hourly snapshots for the tracked players
END = datetime(2026, 7, 5, 0, 0, tzinfo=timezone.utc)
START = END - timedelta(days=DAYS)

# A rotating set of price "regimes" (shape, base price) assigned to the watchlist
# in order, so signals & backtests vary across the board regardless of which
# players (URLs) the user is tracking.
REGIME_CYCLE = [
    ("crash",     1_420_000),  # promo dump -> SELL/BUY interplay
    ("oscillate",    92_000),  # mean-reverting -> rule trades well
    ("uptrend",      64_000),  # steady rise -> HOLD/buy-and-hold wins
    ("dip",          58_000),  # sharp dip + recovery -> BUY
    ("oscillate",    71_000),
    ("uptrend",      88_000),
    ("dip",          41_000),
    ("crash",        55_000),
]

rng = random.Random(26)

# A sharp move in the final hours for a couple of players so the *current* signal
# varies: negative -> dislocated low -> BUY; positive -> overextended -> SELL.
TAIL_H = 6
SHOCK_CYCLE = [-0.16, 0.0, 0.0, 0.0, 0.15, 0.10, 0.0, 0.0]


def price_at(shape: str, base: float, p: float, h: int) -> int:
    noise = rng.uniform(-0.012, 0.012)
    if shape == "uptrend":
        v = base * (1 + 0.28 * p) + base * 0.02 * math.sin(h / 7.0)
    elif shape == "crash":
        # stable, then a sharp drop in the last third (a promo/SBC flood)
        drop = 0.0 if p < 0.62 else -0.34 * ((p - 0.62) / 0.38)
        v = base * (1 + drop) + base * 0.015 * math.sin(h / 9.0)
    elif shape == "dip":
        # gentle drift, deep V dip near 80% of the window, then recover
        dip = -0.30 * math.exp(-((p - 0.8) ** 2) / (2 * 0.0016))
        v = base * (1 + 0.06 * p + dip) + base * 0.015 * math.sin(h / 6.0)
    else:  # oscillate — clean, higher-amplitude mean reversion
        v = base * (1 + 0.19 * math.sin(p * math.pi * 5)) + base * 0.015 * math.sin(h / 5.0)
    return max(200, int(v * (1 + noise)))


def main():
    cfg = load_config("config.yaml")
    conn = db.connect(cfg.database_path)
    conn.execute("DELETE FROM price_snapshots")
    conn.execute("DELETE FROM market_events")
    conn.execute("DELETE FROM signals")
    conn.execute("DELETE FROM features")

    n_steps = DAYS * 24 // STEP_H
    entries = list(cfg.watchlist)

    for idx, entry in enumerate(entries):
        pid = entry.player_id
        shape, base = REGIME_CYCLE[idx % len(REGIME_CYCLE)]
        shock = SHOCK_CYCLE[idx % len(SHOCK_CYCLE)]
        db.upsert_player(conn, player_id=pid, name=entry.name, rating=entry.rating,
                         position=entry.position, version=entry.version, platform="console")
        for i in range(n_steps + 1):
            h = i * STEP_H
            t = START + timedelta(hours=h)
            price = price_at(shape, base, h / (n_steps * STEP_H), h)
            if shock and i > n_steps - TAIL_H:
                price = int(price * (1 + shock * (i - (n_steps - TAIL_H)) / TAIL_H))
            db.insert_snapshot(conn, player_id=pid, price=max(200, price),
                               source="manual", at=t)

    # A market-wide event to light up the event markers + signal logic.
    db.insert_event(conn, event_type="PROMO", start_date="2026-07-10",
                    notes="Festival of Football — market-wide")
    if entries:
        db.insert_event(conn, event_type="SBC", start_date="2026-07-07",
                        player_id=entries[0].player_id, notes="expected SBC requirement")
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    print(f"seeded {n} snapshots for {len(entries)} players, "
          f"{START.date()}..{END.date()}")


if __name__ == "__main__":
    main()
