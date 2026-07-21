"""Consult: the plain verdict for a card, matching, and the group read."""

import pandas as pd

from futmarket.ml import advise


def _row(**kw):
    base = dict(price=10_000, dist_to_floor_pct=2.0, dist_to_ceiling_pct=20.0,
                z_score=-1.0, confidence=0.7, name="Test Card", rating=84,
                version="Rare")
    base.update(kw)
    return pd.Series(base)


def test_buy_when_cheap_on_the_dip_and_confident():
    r = advise.card_read(_row())
    assert r["verdict"] == "BUY"
    assert r["target"] > r["price"] > r["stop"]


def test_wait_when_not_on_a_dip():
    # well above its floor -> not a dip -> tell them the price to wait for
    r = advise.card_read(_row(z_score=0.2, dist_to_floor_pct=30.0))
    assert r["verdict"] == "WAIT"
    assert "wait" in r["headline"].lower()


def test_avoid_a_falling_knife():
    r = advise.card_read(_row(dist_to_floor_pct=-9.0))
    assert r["verdict"] == "AVOID"
    assert not any("room to bounce" in x for x in r["reasons"])   # coherent reasons


def test_expensive_card_is_watch_not_buy():
    """On a dip but expensive -> the tax eats the edge, so not a confident buy."""
    r = advise.card_read(_row(price=500_000))
    assert r["verdict"] == "WATCH" and r["expensive"]


def test_reasons_never_claim_on_the_dip_when_waiting():
    r = advise.card_read(_row(z_score=-1.0, dist_to_floor_pct=30.0))  # oversold but not at floor
    assert r["verdict"] == "WAIT"
    assert not any(x.startswith("on the dip") for x in r["reasons"])


# ---- matching --------------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows)


def test_find_cards_matches_accent_insensitively():
    frame = _frame([
        {"name": "Kylian Mbappé", "version": "Rare", "rating": 91, "price": 200000,
         "liq_score": 5.0},
        {"name": "Erling Haaland", "version": "Rare", "rating": 91, "price": 180000,
         "liq_score": 5.0}])
    out = advise.find_cards(frame, "mbappe")
    assert len(out) == 1 and out.iloc[0]["name"] == "Kylian Mbappé"


def test_find_cards_version_gold_means_base_card():
    frame = _frame([
        {"name": "Kylian Mbappé", "version": "Rare", "rating": 91, "price": 200000,
         "liq_score": 5.0},
        {"name": "Kylian Mbappé", "version": "Team of the Year", "rating": 96,
         "price": 2000000, "liq_score": 5.0}])
    out = advise.find_cards(frame, "mbappe", version="gold")
    assert len(out) == 1 and out.iloc[0]["version"] == "Rare"


# ---- group read ------------------------------------------------------------

def test_cohort_read_summarises_the_group():
    frame = _frame([
        {"rating": "84", "z_score": -1.0, "dist_to_floor_pct": 2.0,
         "cohort_ret_7d": 5.0, "confidence": 0.6, "price": 800, "name": "A",
         "version": "Rare", "url": None},
        {"rating": "84", "z_score": 0.5, "dist_to_floor_pct": 40.0,
         "cohort_ret_7d": 5.0, "confidence": 0.1, "price": 900, "name": "B",
         "version": "Rare", "url": None},
        {"rating": "86", "z_score": -1.0, "dist_to_floor_pct": 1.0,
         "cohort_ret_7d": 9.0, "confidence": 0.9, "price": 1500, "name": "C",
         "version": "Rare", "url": None}])
    r = advise.cohort_read(frame, dim="rating", value="84")
    assert r["n"] == 2                       # only the two 84s
    assert r["group_move_7d"] == 5.0
    assert r["opportunities"] and r["opportunities"][0]["name"] == "A"  # the dip one
