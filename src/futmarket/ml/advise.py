"""Consult the trained model about any card or group.

`picks` produces a filtered buy-list. This is the other door: you name a card
(Mbappe's gold) or a group (84-rated, Premier League) and the model gives its
read — buy now, wait for a dip, or avoid — grounded in everything it has learned
about that card's behaviour, its cohort, the weekly rhythm, the release curve,
and the timing around EA news. Nothing is filtered out: expensive icons and
illiquid cards get an honest read too (with the honest caveats).

Scoring the whole market takes a moment, so the scored feature frame is cached to
parquet and reused; it refreshes on its own once stale.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import db
from . import dataset, picks

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/advise_features.pkl")
CACHE_STAMP_KEY = "advise_features_at"
CACHE_MAX_AGE_HOURS = 6.0
EXPENSIVE_PRICE = 40_000        # above this the edge is thin after tax (be honest)


def _norm(text: str) -> str:
    """Accent- and case-insensitive, for matching 'felix' to 'Félix'."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text))
                   if not unicodedata.combining(c)).lower().strip()


# --------------------------------------------------------------- scored frame

CONSULT_HORIZON = 14        # the hold the consult's verdict assumes


def _build_scored(conn, *, source: str, title: str,
                  horizon: int = CONSULT_HORIZON) -> pd.DataFrame:
    """Latest features per card, scored for "will this beat the market and pay?".

    ``confidence`` here is the clears-cost probability at the default holding
    period, so a consult and a pick are answering the same question. It used to
    be the old direction head's probability of touching a target first, which is
    not the same thing and not the thing that pays.
    """
    frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return frame
    latest = (frame.sort_values("date").groupby("player_id", observed=True)
              .tail(1).copy())
    artifact, _ = picks._load_latest_model(conn, kind="clears", title=title)
    if artifact is None:
        raise RuntimeError("no trained model — run `futmarket train` first")
    horizon = horizon if horizon in artifact["models"] else next(
        iter(artifact["models"]), None)
    if horizon is None:
        raise RuntimeError("trained model has no usable horizon — retrain")
    model = artifact["models"][horizon]
    for c in [c for c in artifact["features"] if c not in latest.columns]:
        latest[c] = np.nan
    # Exactly the columns this model was fitted on: a feature that was constant
    # at fit time was dropped, and the estimator will not accept it back.
    cols = (artifact.get("feature_sets") or {}).get(horizon) or artifact["features"]
    latest["confidence"] = model.predict_proba(latest[cols])[:, 1]
    return latest


def get_scored(conn, *, source: str = "futgg", title: str = "fc26",
               max_age_hours: float = CACHE_MAX_AGE_HOURS,
               rebuild: bool = False) -> pd.DataFrame:
    """The scored frame, from cache when fresh, else rebuilt and cached."""
    if not rebuild and CACHE_PATH.exists():
        stamp = db.meta_get(conn, CACHE_STAMP_KEY)
        if stamp:
            try:
                age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
                         ).total_seconds() / 3600.0
                if age_h <= max_age_hours:
                    return pd.read_pickle(CACHE_PATH)
            except (ValueError, OSError):
                pass
    frame = _build_scored(conn, source=source, title=title)
    if not frame.empty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(CACHE_PATH)
        db.meta_set(conn, CACHE_STAMP_KEY, datetime.now(timezone.utc).isoformat())
        conn.commit()
    return frame


# ----------------------------------------------------------------- card read

def card_read(row: pd.Series, *, tax_rate: float = 0.05) -> dict:
    """Turn one card's scored features into a plain verdict + the trade if it's one."""
    price = int(row["price"])
    floor = row.get("dist_to_floor_pct")
    ceil = row.get("dist_to_ceiling_pct")
    z = row.get("z_score")
    conf = float(row.get("confidence") or 0.0)
    expensive = price > EXPENSIVE_PRICE

    # Two thresholds, deliberately. A shallow dip is worth *describing* -- it is
    # what the card is doing -- but only a deep one is worth buying: measured
    # over the season, z <= -0.5 nets -2.85% after costs and z <= -1.5 nets
    # +4.14%. Saying "on a dip" and saying "buy" are not the same claim.
    on_dip = (pd.notna(z) and z <= -0.5) and (pd.notna(floor) and 0 <= floor <= 5)
    deep_dip = ((pd.notna(z) and z <= picks.ENTRY_Z_MAX)
                and (pd.notna(floor) and 0 <= floor <= picks.ENTRY_FLOOR_MAX_PCT))
    # A consult has no buy band, so the price you'd pay is the market price.
    target, stop, rr = picks._barriers(price, price, ceil, floor, tax_rate=tax_rate)

    if pd.notna(floor) and floor < 0:
        verdict = "AVOID"
        headline = (f"trading {abs(floor):.0f}% below its own 30-day low — a "
                    f"falling knife, not a dip. Wait for it to stop dropping.")
    elif deep_dip and conf >= 0.5 and rr >= picks.MIN_REWARD_RISK and not expensive:
        verdict = "BUY"
        headline = (f"deep on the dip and the model likes it ({conf:.0%} it "
                    f"beats the market) — buy near {price:,}, sell into "
                    f"~{target:,}, give it about two weeks.")
    elif on_dip and not deep_dip:
        verdict = "WATCH"
        headline = (f"on a dip, but a shallow one — that trade loses money after "
                    f"the 7.4% round trip. Wait for a proper dislocation.")
    elif on_dip and expensive:
        verdict = "WATCH"
        headline = (f"on a dip, but at {price:,} it's an expensive card — icons "
                    f"are priced efficiently and the 5% tax eats the edge.")
    elif on_dip:
        verdict = "WATCH"
        headline = (f"on a dip but the model is only {conf:.0%} — not a confident buy.")
    else:
        floor_price = int(price / (1 + floor / 100)) if pd.notna(floor) else None
        wait = f" — wait for a pull-back toward ~{floor_price:,}" if floor_price else ""
        verdict = "WAIT"
        headline = f"not on a dip right now{wait}. Buying it here is chasing."

    # Keep the reasons coherent with the verdict: don't call it "on the dip" when
    # we're not treating it as one, and don't dangle "room to bounce" on a knife.
    reasons = picks._reasons(row)
    if not on_dip:
        reasons = [r for r in reasons if not r.startswith("on the dip")]
    if verdict == "AVOID":
        reasons = [r for r in reasons if "room to bounce" not in r]
    reasons = reasons or ["nothing standout — the model has no strong read here"]

    return {"verdict": verdict, "headline": headline, "price": price,
            "confidence": conf, "target": target, "stop": stop, "reward_risk": rr,
            "expensive": expensive, "on_dip": on_dip, "reasons": reasons}


def find_cards(frame: pd.DataFrame, query: str, *, version: str | None = None,
               limit: int = 8) -> pd.DataFrame:
    """Cards whose name matches the query (optionally a version keyword like 'gold').

    Most-traded / highest-rated first, so the version people mean surfaces on top.
    """
    if frame.empty or "name" not in frame.columns:
        return frame
    q = _norm(query)
    names = frame["name"].fillna("").map(_norm)
    hits = frame[names.str.contains(q, regex=False)].copy()
    if version:
        v = _norm(version)
        # "gold" means a base card: Rare / Common in fut.gg's version naming
        wanted = {"gold": ("rare", "common", "gold")}.get(v, (v,))
        vnorm = hits["version"].fillna("").map(_norm)
        hits = hits[vnorm.apply(lambda s: any(w in s for w in wanted))]
    sort_cols = [c for c in ("liq_score", "rating", "price") if c in hits.columns]
    return hits.sort_values(sort_cols, ascending=False).head(limit)


# --------------------------------------------------------------- cohort read

COHORT_LABELS = {"rating": "rated", "league": "", "position": "", "nation": "",
                 "version": ""}


def cohort_read(frame: pd.DataFrame, *, dim: str, value, tax_rate: float = 0.05) -> dict:
    """A group's read: how the cohort is moving and where the opportunities are."""
    if dim not in frame.columns:
        return {"error": f"unknown group dimension {dim!r}"}
    col = frame[dim].astype(str)
    sub = frame[col == str(value)].copy()
    if sub.empty:
        return {"error": f"no cards found for {dim}={value!r}"}

    n = len(sub)
    z = sub.get("z_score")
    floor = sub.get("dist_to_floor_pct")
    on_dip = ((z <= -0.5) & (floor.between(0, 5))) if z is not None and floor is not None \
        else pd.Series(False, index=sub.index)
    cohort_move = sub.get("cohort_ret_7d")
    group_move = float(cohort_move.dropna().median()) if cohort_move is not None and \
        cohort_move.notna().any() else None

    # the best individual opportunities inside the group
    picks_in = sub[on_dip].sort_values("confidence", ascending=False).head(5) \
        if "confidence" in sub.columns else sub.head(0)
    top = [{"name": r["name"], "version": r.get("version"), "price": int(r["price"]),
            "confidence": float(r.get("confidence") or 0),
            "url": r.get("url")} for _, r in picks_in.iterrows()]

    return {"dim": dim, "value": value, "n": n, "group_move_7d": group_move,
            "share_on_dip": round(float(on_dip.mean()) * 100, 0),
            "opportunities": top}
