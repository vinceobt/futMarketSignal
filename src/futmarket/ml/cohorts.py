"""Cohorts — cards don't move alone.

A card is a member of several groups at once (its rating, position, league,
nation, promo type, price band), and those groups move together: an 84-rated SBC
drags every 84 with it; a promo lifts its whole card set. This module builds
per-group daily price *indices* (the "sector" a card trades in) and the card's
strength *relative* to its groups.

We deliberately don't hard-code which grouping matters for which event — the
model discovers that. We just supply the memberships and the indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cohort dimensions we build. Each maps a card_meta column to a key prefix.
COHORT_DIMS = {
    "rating": "rating",
    "position": "position",
    "league": "league",
    "nation": "nation",
    "version": "version",
}
# A cohort needs at least this many member cards on a day to be meaningful.
MIN_COHORT_MEMBERS = 5
# Return horizons (days) computed for each cohort index.
INDEX_HORIZONS = (1, 7)

_PRICE_BANDS = [
    (1_000, "band:micro"),
    (15_000, "band:low"),
    (75_000, "band:mid"),
    (400_000, "band:high"),
    (float("inf"), "band:elite"),
]


def price_band(price) -> str | None:
    """Coarse price tier — cards clear very differently by price bracket."""
    if price is None or (isinstance(price, float) and np.isnan(price)) or price <= 0:
        return None
    for ceiling, label in _PRICE_BANDS:
        if price < ceiling:
            return label
    return None


def cohort_keys(card: dict) -> list[str]:
    """Every group this card belongs to, as stable string keys."""
    keys: list[str] = []
    for column, prefix in COHORT_DIMS.items():
        value = card.get(column)
        if value is None or value == "":
            continue
        keys.append(f"{prefix}:{value}")
    band = price_band(card.get("price"))
    if band:
        keys.append(band)
    return keys


def build_indices(daily: pd.DataFrame, *, min_members: int = MIN_COHORT_MEMBERS,
                  horizons=INDEX_HORIZONS) -> pd.DataFrame:
    """Daily median price per cohort, plus that cohort's returns.

    `daily` needs columns: date, cohort_key, price. Returns one row per
    (cohort_key, date) with `cohort_median`, `cohort_members` and
    `cohort_ret_{h}d` for each horizon.
    """
    if daily.empty:
        return pd.DataFrame(columns=["cohort_key", "date", "cohort_median",
                                     "cohort_members"] +
                                    [f"cohort_ret_{h}d" for h in horizons])

    grouped = (daily.groupby(["cohort_key", "date"], observed=True)["price"]
               .agg(cohort_median="median", cohort_members="size")
               .reset_index())
    grouped = grouped[grouped["cohort_members"] >= min_members]
    grouped = grouped.sort_values(["cohort_key", "date"])

    for h in horizons:
        prev = grouped.groupby("cohort_key", observed=True)["cohort_median"].shift(h)
        grouped[f"cohort_ret_{h}d"] = (grouped["cohort_median"] / prev - 1.0) * 100.0
    return grouped


def attach_cohort_features(samples: pd.DataFrame, indices: pd.DataFrame,
                           *, horizons=INDEX_HORIZONS) -> pd.DataFrame:
    """Join each sample to its cohorts and summarise them.

    A card belongs to several cohorts, so per sample we average its cohorts'
    returns (the "sector move") and derive relative strength: how the card moved
    versus the groups it trades in. `samples` needs date, player_id, cohort_key,
    and the card's own `ret_{h}d` columns.
    """
    if samples.empty or indices.empty:
        out = samples.drop(columns=["cohort_key"], errors="ignore").copy()
        for h in horizons:
            out[f"cohort_ret_{h}d"] = np.nan
            out[f"rel_strength_{h}d"] = np.nan
        out["n_cohorts"] = 0
        return out

    merged = samples.merge(indices, on=["cohort_key", "date"], how="left")
    agg = {f"cohort_ret_{h}d": "mean" for h in horizons}
    agg["cohort_key"] = "size"
    rolled = (merged.groupby(["player_id", "date"], observed=True)
              .agg(**{k: (k, v) for k, v in agg.items()})
              .rename(columns={"cohort_key": "n_cohorts"})
              .reset_index())

    base = (samples.drop(columns=["cohort_key"])
            .drop_duplicates(subset=["player_id", "date"]))
    out = base.merge(rolled, on=["player_id", "date"], how="left")
    for h in horizons:
        own = out.get(f"ret_{h}d")
        if own is not None:
            out[f"rel_strength_{h}d"] = own - out[f"cohort_ret_{h}d"]
    return out
