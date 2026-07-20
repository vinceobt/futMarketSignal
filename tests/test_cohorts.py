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


# ---- cohort moves and relative strength ----------------------------------

def _frame(rows):
    return pd.DataFrame(rows)


def _members(value, date, price, n=6, dim="rating", start=0):
    """n cards sharing a cohort on one day, so it clears MIN_COHORT_MEMBERS."""
    return [{"player_id": f"{dim}{value}-{start+i}", "date": date, "price": price,
             dim: value} for i in range(n)]


def test_cohort_move_is_the_group_median_change():
    rows = _members("84", "2026-01-01", 100) + _members("84", "2026-01-02", 200)
    out = cohorts.add_cohort_features(_frame(rows), dims=("rating",), horizons=(1,))
    day2 = out[out["date"] == "2026-01-02"]
    assert day2["cohort_ret_1d"].iloc[0] == 100.0     # 100 -> 200
    assert day2["n_cohorts"].iloc[0] == 1


def test_thin_cohorts_are_ignored():
    rows = _members("99", "2026-01-01", 100, n=2) + _members("99", "2026-01-02", 200, n=2)
    out = cohorts.add_cohort_features(_frame(rows), dims=("rating",), horizons=(1,))
    assert out["cohort_ret_1d"].isna().all()
    assert (out["n_cohorts"] == 0).all()


def test_relative_strength_is_own_move_minus_the_group():
    rows = _members("84", "2026-01-01", 100) + _members("84", "2026-01-02", 200)
    frame = _frame(rows)
    frame["ret_1d"] = 0.0
    frame.loc[frame["date"] == "2026-01-02", "ret_1d"] = 150.0   # beat the group
    out = cohorts.add_cohort_features(frame, dims=("rating",), horizons=(1,))
    day2 = out[out["date"] == "2026-01-02"]
    assert day2["rel_strength_1d"].iloc[0] == 50.0    # 150 own - 100 cohort


def test_multiple_dimensions_are_averaged():
    """A card in two cohorts moving +2% and +6% sees a +4% sector move.

    The two cohorts are kept disjoint apart from the test card itself, so each
    group's median reflects only its own dimension.
    """
    rows = []
    for day, rating_price, st_price in (("2026-01-01", 100, 100),
                                        ("2026-01-02", 102, 106)):
        # rating-84 peers: all goalkeepers, so they don't join the ST cohort
        rows += [{"player_id": f"r{i}", "date": day, "price": rating_price,
                  "rating": "84", "position": "GK"} for i in range(6)]
        # ST peers: all rated 99, so they don't join the rating-84 cohort
        rows += [{"player_id": f"s{i}", "date": day, "price": st_price,
                  "rating": "99", "position": "ST"} for i in range(6)]
        # the card under test belongs to both
        rows.append({"player_id": "both", "date": day, "price": rating_price,
                     "rating": "84", "position": "ST"})

    out = cohorts.add_cohort_features(_frame(rows), dims=("rating", "position"),
                                      horizons=(1,))
    card = out[(out["player_id"] == "both") & (out["date"] == "2026-01-02")]
    assert card["n_cohorts"].iloc[0] == 2
    # rating cohort +2%, position cohort +6% -> mean +4%
    assert round(float(card["cohort_ret_1d"].iloc[0]), 3) == 4.0


def test_empty_frame_is_safe():
    out = cohorts.add_cohort_features(pd.DataFrame(
        {"player_id": [], "date": [], "price": [], "rating": []}), horizons=(1,))
    assert "cohort_ret_1d" in out.columns and len(out) == 0


def test_price_band_series_matches_scalar():
    prices = pd.Series([500, 8_000, 20_000, 100_000, 5_000_000, 0])
    banded = list(cohorts.price_band_series(prices))
    assert banded[:5] == [cohorts.price_band(p) for p in prices[:5]]
    assert pd.isna(banded[5])
