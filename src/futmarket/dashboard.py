"""Assemble everything the (simplified) dashboard needs into one JSON payload.

Pure read layer over the pipeline already built: db -> features -> signals.
The FastAPI app serves this at /api/data; the Artifact preview bakes the same
shape in. No market logic lives here — it only shapes stored data for display.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import db, features
from .config import Config
from .services import watch
from .signals import BUY, HOLD, SELL, SignalParams, evaluate


def build_payload(config: Config, conn, source: str) -> dict:
    params = SignalParams.from_config(config)

    # rating/name are enriched by the collector into the players table (the
    # URL-only watchlist entry carries no rating); prefer those stored values.
    stored = {
        r["player_id"]: r
        for r in conn.execute("SELECT player_id, name, rating FROM players").fetchall()
    }

    players, buys, sells, holds = [], 0, 0, 0
    for entry in watch.effective_entries(conn, config):
        table = features.compute_feature_table(conn, entry.player_id, source)
        if not table:
            continue
        latest = table[-1]
        sig = evaluate(latest, params)
        buys += sig.signal_type == BUY
        sells += sig.signal_type == SELL
        holds += sig.signal_type == HOLD

        row = stored.get(entry.player_id)
        players.append({
            "id": entry.player_id,
            "name": (row["name"] if row and row["name"] else entry.name),
            "rating": (row["rating"] if row else entry.rating),
            "price": latest.price,
            "pct_change_24h": (round(latest.pct_change_24h, 2)
                               if latest.pct_change_24h is not None else None),
            "signal": {"type": sig.signal_type, "reason": sig.reason},
            "series": [{"t": f.timestamp, "price": f.price} for f in table],
        })

    snap_total = conn.execute("SELECT COUNT(*) AS n FROM price_snapshots").fetchone()["n"]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "platform": config.platform,
        "summary": {
            "tracked": len(players),
            "snapshots": snap_total,
            "buys": buys,
            "sells": sells,
            "holds": holds,
        },
        "players": players,
    }
