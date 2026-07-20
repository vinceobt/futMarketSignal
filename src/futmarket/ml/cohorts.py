"""Cohorts — cards don't move alone.

A card belongs to several groups at once (rating, position, league, nation,
promo type, price band), and those groups move together: an 84-rated SBC drags
every 84 with it; a promo lifts its whole card set. We give the model each
group's move and the card's strength *relative* to its groups.

Which grouping matters for which event is deliberately not hard-coded — the
model discovers that. We only supply the memberships and the moves.

Implementation note: this is computed one dimension at a time with a groupby and
a merge. The obvious approach — explode every row into one row per cohort — turns
2M rows into 12M and cost 7GB of RAM; this does the same arithmetic in a fraction
of that, which is the difference between running on a laptop and needing a server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Card attributes that define a cohort.
COHORT_DIMS = ("rating", "position", "league", "nation", "version", "band")
# A cohort needs at least this many member cards on a day to mean anything.
MIN_COHORT_MEMBERS = 5
# Horizons (days) over which each cohort's move is measured.
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


def price_band_series(prices: pd.Series) -> pd.Series:
    """Vectorised price_band for a whole column."""
    edges = [0] + [c for c, _ in _PRICE_BANDS[:-1]] + [np.inf]
    labels = [lbl for _, lbl in _PRICE_BANDS]
    banded = pd.cut(prices, bins=edges, labels=labels, right=False)
    return banded.astype("object").where(prices > 0)


def cohort_keys(card: dict) -> list[str]:
    """Every group this card belongs to, as stable string keys."""
    keys: list[str] = []
    for column in COHORT_DIMS:
        if column == "band":
            continue
        value = card.get(column)
        if value is None or value == "":
            continue
        keys.append(f"{column}:{value}")
    band = price_band(card.get("price"))
    if band:
        keys.append(band)
    return keys


def _dimension_moves(frame: pd.DataFrame, column: str, *, min_members: int,
                     horizons) -> pd.DataFrame | None:
    """Each (value, date) group's median price and its move over each horizon."""
    key = frame[column]
    usable = key.notna() & (key.astype(str) != "")
    if not usable.any():
        return None

    idx = (frame.loc[usable].groupby([column, "date"], observed=True)["price"]
           .agg(cohort_median="median", members="size").reset_index())
    idx = idx[idx["members"] >= min_members]
    if idx.empty:
        return None

    idx = idx.sort_values([column, "date"])
    grouped = idx.groupby(column, observed=True)["cohort_median"]
    for h in horizons:
        idx[f"_r{h}"] = (grouped.pct_change(h) * 100.0).astype("float32")
    return idx[[column, "date", *(f"_r{h}" for h in horizons)]]


def add_cohort_features(frame: pd.DataFrame, *, dims=COHORT_DIMS,
                        min_members: int = MIN_COHORT_MEMBERS,
                        horizons=INDEX_HORIZONS) -> pd.DataFrame:
    """Attach each card's cohort moves and its strength relative to them.

    A card sits in several cohorts, so `cohort_ret_{h}d` is the average move of
    the groups it belongs to (the "sector move"), and `rel_strength_{h}d` is how
    far the card beat or lagged that.
    """
    n = len(frame)
    out = frame
    if n == 0:
        for h in horizons:
            out[f"cohort_ret_{h}d"] = np.nan
            out[f"rel_strength_{h}d"] = np.nan
        out["n_cohorts"] = 0
        return out

    totals = {h: np.zeros(n, dtype="float32") for h in horizons}
    counts = {h: np.zeros(n, dtype="float32") for h in horizons}
    membership = np.zeros(n, dtype="int16")

    for column in dims:
        if column not in frame.columns:
            continue
        moves = _dimension_moves(frame, column, min_members=min_members,
                                 horizons=horizons)
        if moves is None:
            continue
        # Merge on the key pair only -- never widen the frame itself.
        joined = frame[[column, "date"]].merge(moves, on=[column, "date"], how="left")
        in_cohort = np.zeros(n, dtype=bool)
        for h in horizons:
            values = joined[f"_r{h}"].to_numpy(dtype="float32", na_value=np.nan)
            present = ~np.isnan(values)
            totals[h][present] += values[present]
            counts[h][present] += 1.0
            in_cohort |= present
        membership += in_cohort.astype("int16")

    for h in horizons:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts[h] > 0, totals[h] / counts[h], np.nan)
        out[f"cohort_ret_{h}d"] = mean.astype("float32")
        own = out.get(f"ret_{h}d")
        if own is not None:
            out[f"rel_strength_{h}d"] = (
                own.to_numpy(dtype="float32", na_value=np.nan) - mean).astype("float32")
    out["n_cohorts"] = membership
    return out
