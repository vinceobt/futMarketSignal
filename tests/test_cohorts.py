"""Cohort features: memberships, group indices, relative strength."""

import numpy as np
import pandas as pd

from futmarket.ml import cohorts


# ---- memberships ----------------------------------------------------------

def test_cohort_keys_covers_dimensions():
    keys = cohorts.cohort_keys({
        "rating": 84, "position": "ST", "league": "Premier League",
        "nation": "England", "version": "TOTW", "price": 20_000})
    assert "rating:84" in keys and "position:ST" in keys
    assert "league:Premier League" in keys and "nation:England" in keys
    assert "version:TOTW" in keys and "band:mid" in keys   # 20k -> mid (15k-75k)


def test_cohort_keys_skips_missing():
    keys = cohorts.cohort_keys({"rating": 84, "position": "", "league": None})
    assert keys == ["rating:84"]


def test_price_bands():
    assert cohorts.price_band(500) == "band:micro"        # < 1k
    assert cohorts.price_band(8_000) == "band:low"        # 1k-15k
    assert cohorts.price_band(20_000) == "band:mid"       # 15k-75k
    assert cohorts.price_band(100_000) == "band:high"     # 75k-400k
    assert cohorts.price_band(5_000_000) == "band:elite"  # 400k+
    assert cohorts.price_band(None) is None
    assert cohorts.price_band(0) is None


# ---- indices --------------------------------------------------------------

def _daily(rows):
    return pd.DataFrame(rows, columns=["cohort_key", "date", "price"])


def test_build_indices_medians_and_returns():
    rows = []
    # 5 members so the cohort clears MIN_COHORT_MEMBERS; price doubles day 2
    for day, price in (("2026-01-01", 100), ("2026-01-02", 200)):
        for i in range(5):
            rows.append(("rating:84", day, price))
    idx = cohorts.build_indices(_daily(rows))
    assert list(idx["cohort_median"]) == [100, 200]
    assert idx["cohort_members"].tolist() == [5, 5]
    # day-2 return vs day-1 = +100%
    assert idx.loc[idx["date"] == "2026-01-02", "cohort_ret_1d"].iloc[0] == 100.0


def test_build_indices_drops_thin_cohorts():
    rows = [("rating:99", "2026-01-01", 100)] * 2      # only 2 members
    idx = cohorts.build_indices(_daily(rows))
    assert idx.empty


def test_build_indices_empty_input():
    idx = cohorts.build_indices(pd.DataFrame(columns=["cohort_key", "date", "price"]))
    assert idx.empty and "cohort_ret_1d" in idx.columns


# ---- attaching to samples -------------------------------------------------

def test_attach_cohort_features_relative_strength():
    # one card in one cohort; card rose 10%, its cohort rose 4% -> rel +6
    samples = pd.DataFrame([
        {"player_id": "p1", "date": "2026-01-02", "cohort_key": "rating:84", "ret_1d": 10.0,
         "ret_7d": 0.0},
    ])
    indices = pd.DataFrame([
        {"cohort_key": "rating:84", "date": "2026-01-02", "cohort_median": 100.0,
         "cohort_members": 9, "cohort_ret_1d": 4.0, "cohort_ret_7d": 0.0},
    ])
    out = cohorts.attach_cohort_features(samples, indices)
    assert len(out) == 1
    assert out["cohort_ret_1d"].iloc[0] == 4.0
    assert out["rel_strength_1d"].iloc[0] == 6.0
    assert out["n_cohorts"].iloc[0] == 1


def test_attach_averages_multiple_cohorts():
    # card belongs to two cohorts moving +2% and +6% -> sector move averages +4%
    samples = pd.DataFrame([
        {"player_id": "p1", "date": "2026-01-02", "cohort_key": "rating:84", "ret_1d": 10.0},
        {"player_id": "p1", "date": "2026-01-02", "cohort_key": "position:ST", "ret_1d": 10.0},
    ])
    indices = pd.DataFrame([
        {"cohort_key": "rating:84", "date": "2026-01-02", "cohort_median": 100.0,
         "cohort_members": 9, "cohort_ret_1d": 2.0},
        {"cohort_key": "position:ST", "date": "2026-01-02", "cohort_median": 100.0,
         "cohort_members": 9, "cohort_ret_1d": 6.0},
    ])
    out = cohorts.attach_cohort_features(samples, indices, horizons=(1,))
    assert len(out) == 1                       # collapsed back to one row per card/day
    assert out["cohort_ret_1d"].iloc[0] == 4.0
    assert out["rel_strength_1d"].iloc[0] == 6.0
    assert out["n_cohorts"].iloc[0] == 2


def test_attach_handles_empty_indices():
    samples = pd.DataFrame([
        {"player_id": "p1", "date": "2026-01-02", "cohort_key": "rating:84", "ret_1d": 5.0}])
    out = cohorts.attach_cohort_features(samples, pd.DataFrame(), horizons=(1,))
    assert len(out) == 1 and np.isnan(out["cohort_ret_1d"].iloc[0])
