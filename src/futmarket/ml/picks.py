"""The output: which cards to buy right now, at what price, and why.

Everything else in this package is machinery. This is the part a person uses.

For each tracked card we take today's features, score them with the trained
direction model, keep only cards that are genuinely sellable, and turn the raw
numbers into a plain-English case: what price to pay, and the specific reasons
the model liked it (near its floor, promo landing soon, Monday recovery window,
day nine of the post-release slump, its cohort already moving).

The buy price is a *band* taken from real completed sales, not the lowest
listing. The lowest listing is usually a mispriced snipe nobody can catch; the
band is what the card genuinely changes hands for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import db
from ..services import sales as sales_service
from . import dataset

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.30        # below this the model isn't saying anything useful
DEFAULT_LIMIT = 20
# Percentage returns are wildly misleading on near-discard cards: a 200-coin card
# going to 263 is "+31%" but earns 63 coins and can't be bought in any volume.
# Judge a pick on the coins it could actually make, not the percentage.
MIN_PRICE = 5_000
MIN_PROFIT_COINS = 2_000


@dataclass
class Pick:
    player_id: str
    name: str
    rating: int | None
    version: str
    confidence: float
    price_now: int
    buy_low: int | None
    buy_high: int | None
    sell_target: int
    stop: int
    liquidity_tier: str | None
    sales_per_hour: float | None
    reasons: list[str] = field(default_factory=list)


# --- turning numbers into a human case ------------------------------------

def _reasons(row: pd.Series) -> list[str]:
    """The specific, checkable reasons this card looks interesting today."""
    out: list[str] = []

    floor = row.get("dist_to_floor_pct")
    if pd.notna(floor):
        if floor < 0:
            # Below the 30-day low is a falling knife, not a bargain. Say so.
            out.append(f"WARNING: {abs(floor):.0f}% BELOW its 30-day low — still falling")
        elif floor <= 15:
            out.append(f"near its floor ({floor:.0f}% above the 30-day low)")
    z = row.get("z_score")
    if pd.notna(z) and z <= -1:
        out.append(f"unusually cheap for itself ({z:.1f} sigma below normal)")

    dow = row.get("day_of_week")
    if pd.notna(dow):
        if dow in (6, 0):
            out.append("Sunday/Monday trough — supply dries up from here")
        elif dow == 3:
            out.append("Thursday rewards dump — supply flooding in")
        elif dow == 4:
            out.append("Friday promo + Weekend League — heavy supply")

    age = row.get("days_since_card_release")
    if pd.notna(age):
        if 5 <= age <= 12:
            out.append(f"day {int(age)} after release — the usual bottom")
        elif age <= 2:
            out.append(f"only {int(age)}d old — release crash still running")

    promo = row.get("days_to_next_promo")
    if pd.notna(promo) and promo <= 3:
        out.append(f"promo in {int(promo)}d — expect supply")
    # (live-SBC count is omitted: 50-60 are always running, so it appears on every
    #  card and discriminates nothing.)

    coh = row.get("cohort_ret_7d")
    rel = row.get("rel_strength_7d")
    if pd.notna(coh) and 3 <= abs(coh) <= 60:
        direction = "rising" if coh > 0 else "falling"
        out.append(f"its group is {direction} ({coh:+.0f}% this week)")
    # Beyond ~50% the relative figure is arithmetic noise off a tiny price, not
    # a real dislocation, so it isn't worth quoting as a reason.
    if pd.notna(rel) and -50 <= rel <= -5:
        out.append(f"lagging its group by {abs(rel):.0f}% — room to catch up")

    return out or ["model pattern match (no single standout reason)"]


def _fill_missing_bands(conn, player_ids: list[str], *, title: str = "fc26",
                        delay: float = 1.0) -> int:
    """Fetch real sale data for shortlisted cards that don't have it yet.

    A pick without a price band is half a recommendation -- it can only quote the
    lowest listing, which is the very number we decided not to trust. The
    shortlist is short, so fetching on demand is cheap.
    """
    import time
    from ..collectors import history_source
    from ..collectors.base import SourceError
    from ..services import sales as sales_service

    fetched = 0
    for pid in player_ids:
        if db.sale_stats_get(conn, pid) is not None:
            continue
        meta = db.card_meta_get(conn, pid)
        if meta is None or meta["definition_id"] is None:
            continue
        game = (meta["title"] or "fc26").replace("fc", "") or "26"
        try:
            detail = history_source.fetch_card_detail(int(meta["definition_id"]), game)
        except SourceError as e:
            logger.warning("could not fetch sales for %s: %s", pid, e)
            continue
        stats = sales_service.summarise_sales(detail["sales"], detail.get("current"))
        if stats:
            db.upsert_sale_stats(conn, player_id=pid, title=title, **stats)
            conn.commit()
            fetched += 1
        if delay:
            time.sleep(delay)
    return fetched


def _load_latest_model(conn, *, kind: str = "direction", title: str = "fc26"):
    import joblib
    run = db.latest_model_run(conn, kind=kind, title=title)
    if run is None or not run["artifact_path"]:
        return None, None
    try:
        artifact = joblib.load(run["artifact_path"])
    except OSError as e:
        logger.warning("could not load model artifact: %s", e)
        return None, None
    return artifact, run


def generate(conn, *, source: str = "futgg", title: str = "fc26",
             target_pct: float = 25.0, stop_pct: float = 8.0,
             tax_rate: float = 0.05, min_confidence: float = MIN_CONFIDENCE,
             limit: int = DEFAULT_LIMIT, min_price: int = MIN_PRICE,
             min_profit_coins: int = MIN_PROFIT_COINS, skip_falling: bool = True,
             fetch_missing_bands: bool = True, min_sales_per_hour: float = 0.0,
             liquid_tiers: tuple[str, ...] = ("A", "B")) -> list[Pick]:
    """Today's ranked buy candidates, worth actually trading."""
    artifact, run = _load_latest_model(conn, title=title)
    if artifact is None:
        raise RuntimeError("no trained model — run `futmarket train` first")

    frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return []

    # latest row per card only: we're deciding about today, not replaying history
    latest = (frame.sort_values("date").groupby("player_id", observed=True)
              .tail(1).copy())
    if "liq_tier" in latest.columns:
        latest = latest[latest["liq_tier"].isin(liquid_tiers)]
    # Drop near-discard cards: their percentage moves are arithmetic noise and
    # the coins involved aren't worth a trade.
    latest = latest[latest["price"] >= min_price]
    target_mult_pre = (1.0 + target_pct / 100.0) / (1.0 - tax_rate)
    latest = latest[latest["price"] * (target_mult_pre - 1.0) >= min_profit_coins]
    # Never recommend a card trading below its own 30-day low: that is a falling
    # knife, not a discount. (The original rules engine gated on this and it was
    # right to.) Cheapness alone is not a reason to buy.
    if skip_falling and "dist_to_floor_pct" in latest.columns:
        latest = latest[latest["dist_to_floor_pct"].fillna(0) >= 0]
    if latest.empty:
        return []

    cols = artifact["features"]
    missing = [c for c in cols if c not in latest.columns]
    for c in missing:
        latest[c] = np.nan
    latest["confidence"] = artifact["model"].predict_proba(latest[cols])[:, 1]

    candidates = latest[latest["confidence"] >= min_confidence]
    candidates = candidates.sort_values("confidence", ascending=False).head(limit * 4)

    if fetch_missing_bands:
        # Fetch a wider shortlist than we'll show, so the sales filter below has
        # real rates to judge and isn't just dropping cards for missing data.
        _fill_missing_bands(conn, candidates["player_id"].tolist()[:limit * 3],
                            title=title)

    if min_sales_per_hour > 0:
        # Rule #1 in its truest form: confidence is worthless if you can't get out.
        keep = []
        for pid in candidates["player_id"]:
            stats = db.sale_stats_get(conn, pid)
            keep.append(stats is not None and (stats["sales_per_hour"] or 0)
                        >= min_sales_per_hour)
        candidates = candidates[pd.Series(keep, index=candidates.index)]

    target_mult = (1.0 + target_pct / 100.0) / (1.0 - tax_rate)
    picks: list[Pick] = []
    for row in candidates.itertuples(index=False):
        r = pd.Series(row._asdict())
        price = int(r["price"])
        stats = db.sale_stats_get(conn, r["player_id"])
        band = None
        if stats is not None:
            band = sales_service.buy_band({
                "sold_p25": stats["sold_p25"], "sold_median": stats["sold_median"]})
        # Target and stop must be anchored to the price you would actually PAY.
        # Deriving them from the current listing while quoting a band from
        # completed sales put them on different scales -- it produced sell targets
        # below the buy price.
        entry = band[1] if band else price
        picks.append(Pick(
            player_id=r["player_id"],
            name=str(r.get("name") or r["player_id"]),
            rating=int(r["rating"]) if pd.notna(r.get("rating")) else None,
            version=str(r.get("version") or ""),
            confidence=float(r["confidence"]),
            price_now=price,
            buy_low=band[0] if band else None,
            buy_high=band[1] if band else None,
            sell_target=int(round(entry * target_mult)),
            stop=int(round(entry * (1 - stop_pct / 100.0))),
            liquidity_tier=r.get("liq_tier"),
            sales_per_hour=(float(stats["sales_per_hour"])
                            if stats is not None and stats["sales_per_hour"] else None),
            reasons=_reasons(r),
        ))
    logger.info("generated %d picks from run %s", len(picks),
                run["run_id"] if run is not None else "?")
    return picks


def save(conn, picks: list[Pick], *, title: str = "fc26",
         horizon_h: int = 168) -> int:
    """Record today's picks so we can score them honestly later."""
    from datetime import datetime, timezone
    run = db.latest_model_run(conn, kind="direction", title=title)
    run_id = run["run_id"] if run is not None else 0
    now = datetime.now(timezone.utc)
    for p in picks:
        db.insert_prediction(conn, subject_id=p.player_id, level="card",
                             kind="direction", horizon_h=horizon_h, at=now,
                             run_id=run_id, title=title, proba=p.confidence)
    conn.commit()
    return len(picks)
