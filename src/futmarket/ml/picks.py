"""The output: which cards to buy right now, at what price, and why.

Everything else in this package is machinery. This is the part a person uses.

The trade, from measured edge: buy a **cheap/mid card on the dip** — oversold and
near its own 30-day floor — and sell into its resistance. In backtest that returns
+5-11% on capital net of tax; expensive icons are priced efficiently and lose, so
they're excluded. We take today's features, keep only cards that pass the dip gate
and are genuinely sellable, score them with the trained model, size each trade to
the card's own range, and turn it into a plain-English case.

The buy price is a *band* taken from real completed sales, not the lowest listing.
The lowest listing is usually a mispriced snipe nobody can catch; the band is what
the card genuinely changes hands for.
"""

from __future__ import annotations

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
MIN_PRICE = 1_000            # ignore near-discard noise
MAX_PRICE = 40_000           # ignore efficiently-priced cards (icons) that lose after tax
ENTRY_Z_MAX = -0.5           # only buy when this far below normal (the dip)
ENTRY_FLOOR_MAX_PCT = 5.0    # only buy within this % of the card's own 30-day low
STOP_BUFFER_PCT = 2.0
STOP_MIN_PCT = 5.0
STOP_MAX_PCT = 18.0
MIN_REWARD_RISK = 1.0
FALLBACK_TARGET_PCT = 15.0   # when a card's resistance isn't known yet


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
    reward_risk: float
    liquidity_tier: str | None
    sales_per_hour: float | None
    url: str | None = None          # exact card -- names repeat across versions
    reasons: list[str] = field(default_factory=list)


def _barriers(entry: int, ceil_pct, floor_pct, *, tax_rate: float,
              stop_buffer_pct: float = STOP_BUFFER_PCT,
              stop_min_pct: float = STOP_MIN_PCT, stop_max_pct: float = STOP_MAX_PCT
              ) -> tuple[int, int, float]:
    """(sell_target, stop, reward:risk) sized to the card's own structure.

    Target = the card's resistance (its 30-day high). Stop = just below its
    support (its 30-day low), floored for noise and capped for max loss. Reward
    and risk are both net of tax, so the ratio is the real one.
    """
    target_pct = float(ceil_pct) if pd.notna(ceil_pct) and ceil_pct > 0 else FALLBACK_TARGET_PCT
    stop_dist = (float(floor_pct) + stop_buffer_pct) if pd.notna(floor_pct) else stop_min_pct
    stop_dist = min(max(stop_dist, stop_min_pct), stop_max_pct)
    target = int(round(entry * (1 + target_pct / 100.0)))
    stop = int(round(entry * (1 - stop_dist / 100.0)))
    net_reward = target * (1 - tax_rate) / entry - 1.0
    rr = net_reward / (stop_dist / 100.0) if stop_dist > 0 else 0.0
    return target, stop, round(rr, 2)


# --- turning numbers into a human case ------------------------------------

def _reasons(row: pd.Series) -> list[str]:
    """The specific, checkable reasons this card looks interesting today."""
    out: list[str] = []

    # The core of the trade: it's on the dip (oversold, near its own floor).
    z = row.get("z_score")
    if pd.notna(z) and z <= -0.5:
        out.append(f"on the dip — {abs(z):.1f} sigma below its own normal price")
    floor = row.get("dist_to_floor_pct")
    if pd.notna(floor):
        if floor < 0:
            out.append(f"WARNING: {abs(floor):.0f}% BELOW its 30-day low — still falling")
        elif floor <= 5:
            out.append(f"right at its floor ({floor:.0f}% above the 30-day low)")
    ceil = row.get("dist_to_ceiling_pct")
    if pd.notna(ceil) and ceil >= 8:
        out.append(f"room to bounce — resistance is {ceil:.0f}% up")

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

    ann = row.get("days_since_last_announce")
    if pd.notna(ann) and ann <= 1:
        out.append(f"EA announced a promo {int(ann)}d ago — the usual dip")
    elif pd.notna(ann) and 4 <= ann <= 7:
        out.append(f"{int(ann)}d after an EA announcement — the usual rebound")

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
                        delay: float = 1.0, max_age_minutes: float = 60.0) -> int:
    """Fetch real sale data for shortlisted cards that lack it OR have gone stale.

    A pick without a band can only quote the lowest listing -- the very number we
    decided not to trust. A pick with a *stale* band is worse: it quotes a
    confident price that has since moved. Prices here can run 30%+ in a few
    hours, so anything older than an hour is refetched. The shortlist is short,
    so doing it on demand is cheap.
    """
    import time
    from datetime import datetime, timezone
    from ..collectors import history_source
    from ..collectors.base import SourceError
    from ..services import sales as sales_service

    now = datetime.now(timezone.utc)
    fetched = 0
    for pid in player_ids:
        existing = db.sale_stats_get(conn, pid)
        if existing is not None:
            try:
                age_min = (now - datetime.strptime(
                    existing["computed_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)).total_seconds() / 60.0
            except (TypeError, ValueError):
                age_min = float("inf")
            if age_min <= max_age_minutes:
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
             tax_rate: float = 0.05, min_confidence: float = MIN_CONFIDENCE,
             limit: int = DEFAULT_LIMIT, min_price: int = MIN_PRICE,
             max_price: int = MAX_PRICE, entry_z_max: float = ENTRY_Z_MAX,
             entry_floor_max_pct: float = ENTRY_FLOOR_MAX_PCT,
             stop_buffer_pct: float = STOP_BUFFER_PCT, stop_min_pct: float = STOP_MIN_PCT,
             stop_max_pct: float = STOP_MAX_PCT, min_reward_risk: float = MIN_REWARD_RISK,
             fetch_missing_bands: bool = True, min_sales_per_hour: float = 0.0,
             liquid_tiers: tuple[str, ...] = ("A", "B")) -> list[Pick]:
    """Today's ranked buy candidates: cheap/mid cards on the dip, worth trading."""
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
    # The tradeable universe: the edge lives in cheap/mid cards; expensive icons
    # are priced efficiently and lose after tax.
    latest = latest[(latest["price"] >= min_price) & (latest["price"] <= max_price)]
    # The entry gate — this IS the edge: only buy on the dip, oversold and near
    # the card's own floor. Cheapness alone loses money; the dip is what pays.
    if "z_score" in latest.columns:
        latest = latest[latest["z_score"] <= entry_z_max]
    if "dist_to_floor_pct" in latest.columns:
        floor = latest["dist_to_floor_pct"]
        latest = latest[(floor >= 0) & (floor <= entry_floor_max_pct)]
    if latest.empty:
        return []

    cols = artifact["features"]
    for c in [c for c in cols if c not in latest.columns]:
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

    picks: list[Pick] = []
    for row in candidates.itertuples(index=False):
        r = pd.Series(row._asdict())
        price = int(r["price"])
        stats = db.sale_stats_get(conn, r["player_id"])
        meta = db.card_meta_get(conn, r["player_id"])
        # The listed price refreshed moments ago is the only figure that is
        # genuinely current; sales lag it on anything that's moving.
        listed = stats["listed_price"] if stats is not None else None
        band = sales_service.buy_band(
            {"listed_price": listed, "sold_median": stats["sold_median"]}
            if stats is not None else None,
            listed_price=listed or price)
        # Barriers are anchored to the price you would actually PAY, and sized to
        # the card's own range (resistance target, support stop).
        entry = band[1] if band else price
        target, stop, rr = _barriers(
            entry, r.get("dist_to_ceiling_pct"), r.get("dist_to_floor_pct"),
            tax_rate=tax_rate, stop_buffer_pct=stop_buffer_pct,
            stop_min_pct=stop_min_pct, stop_max_pct=stop_max_pct)
        if rr < min_reward_risk:
            continue                     # upside not worth the downside — skip
        reasons = _reasons(r)
        reasons.append(f"risking {round((1 - stop/entry) * 100)}% to make "
                       f"{round((target * (1 - tax_rate) / entry - 1) * 100)}% "
                       f"(reward:risk {rr:g})")
        picks.append(Pick(
            player_id=r["player_id"],
            name=str(r.get("name") or r["player_id"]),
            rating=int(r["rating"]) if pd.notna(r.get("rating")) else None,
            version=str(r.get("version") or ""),
            confidence=float(r["confidence"]),
            price_now=price,
            buy_low=band[0] if band else None,
            buy_high=band[1] if band else None,
            sell_target=target,
            stop=stop,
            reward_risk=rr,
            liquidity_tier=r.get("liq_tier"),
            sales_per_hour=(float(stats["sales_per_hour"])
                            if stats is not None and stats["sales_per_hour"] else None),
            url=(meta["url"] if meta is not None else None),
            reasons=reasons,
        ))
        if len(picks) >= limit:
            break
    logger.info("generated %d picks from run %s", len(picks),
                run["run_id"] if run is not None else "?")
    return picks


def save(conn, picks: list[Pick], *, title: str = "fc26",
         horizon_days: int = 5) -> int:
    """Record today's picks so the market can grade them later.

    Stores the price you'd realistically have paid and the barriers it will be
    judged against, so scoring never depends on re-deriving them afterwards.
    """
    from datetime import datetime, timezone
    run = db.latest_model_run(conn, kind="direction", title=title)
    run_id = run["run_id"] if run is not None else None
    now = datetime.now(timezone.utc)
    saved = 0
    for p in picks:
        entry = p.buy_high or p.price_now
        if db.insert_pick(conn, player_id=p.player_id, title=title, at=now,
                          run_id=run_id, confidence=p.confidence,
                          entry_price=entry, buy_low=p.buy_low,
                          buy_high=p.buy_high, target_price=p.sell_target,
                          stop_price=p.stop, horizon_days=horizon_days,
                          sales_per_hour=p.sales_per_hour,
                          reasons="; ".join(p.reasons)):
            saved += 1
    conn.commit()
    return saved
