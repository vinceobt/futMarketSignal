"""Labels: per-card triple-barrier economics, purging, and forward returns."""

import numpy as np
import pandas as pd

from futmarket.ml import labels

UP = np.full(6, 130.0)      # a resistance level for the whole path
DN = np.full(6, 92.0)       # a support level for the whole path


def test_triple_barrier_hits_target():
    prices = np.array([100, 105, 140, 90, 90, 90], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper=UP, lower=DN)
    assert out[0] == labels.HIT_TARGET      # 140 >= 130 before any stop


def test_triple_barrier_hits_stop_first():
    prices = np.array([100, 80, 200, 200, 200, 200], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper=UP, lower=DN)
    assert out[0] == labels.HIT_STOP        # 80 <= 92 happened first


def test_triple_barrier_neither():
    prices = np.array([100, 101, 99, 102, 100, 100], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper=UP, lower=DN)
    assert out[0] == labels.NEITHER


def test_triple_barrier_purges_incomplete_horizon():
    """Rows without a full horizon of future must be NaN, never guessed as 0."""
    prices = np.array([100, 101, 102, 103, 104], dtype=float)
    up, dn = np.full(5, 130.0), np.full(5, 92.0)
    out = labels.triple_barrier(prices, horizon=3, upper=up, lower=dn)
    assert np.isfinite(out[0]) and np.isfinite(out[1])
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])


def test_triple_barrier_order_matters():
    """Stop before target on the same path -> stopped, not profitable."""
    dips_first = np.array([100, 91, 200], dtype=float)
    pops_first = np.array([100, 200, 91], dtype=float)
    up, dn = np.full(3, 130.0), np.full(3, 92.0)
    assert labels.triple_barrier(dips_first, horizon=2, upper=up, lower=dn)[0] == labels.HIT_STOP
    assert labels.triple_barrier(pops_first, horizon=2, upper=up, lower=dn)[0] == labels.HIT_TARGET


# ---- barriers sized to the card (the fix for the money leak) ---------------

def test_card_barriers_size_the_stop_to_the_cards_range():
    """The bug that lost money: a flat 8% stop on a wide-range card. A card that
    swings wide must get a WIDER stop than a tight one."""
    frame = pd.DataFrame({
        "price": [100.0, 100.0],
        "dist_to_floor_pct": [3.0, 15.0],       # tight vs wide
        "dist_to_ceiling_pct": [20.0, 20.0],
    })
    upper, lower = labels.card_barriers(frame, stop_buffer_pct=2.0,
                                        stop_min_pct=5.0, stop_max_pct=18.0)
    assert round(upper[0]) == 120 and round(upper[1]) == 120     # target = resistance
    assert lower[1] < lower[0]                                   # wide card, more room
    assert round(100 - lower[0]) == 5                            # tight card floored at 5%
    assert round(100 - lower[1]) == 17                           # wide card: 15 + 2 buffer


def test_card_barriers_cap_the_max_loss():
    frame = pd.DataFrame({"price": [100.0], "dist_to_floor_pct": [40.0],
                          "dist_to_ceiling_pct": [20.0]})
    _, lower = labels.card_barriers(frame, stop_max_pct=18.0)
    assert round(100 - lower[0]) == 18          # 40+2 capped at 18


# ---- add_labels -----------------------------------------------------------

def _frame(prices, pid="p", ceil=25.0, floor=10.0):
    return pd.DataFrame({
        "player_id": [pid] * len(prices),
        "date": [f"2026-01-{i+1:02d}" for i in range(len(prices))],
        "price": prices,
        "dist_to_ceiling_pct": [ceil] * len(prices),
        "dist_to_floor_pct": [floor] * len(prices)})


def test_add_labels_forward_return():
    out = labels.add_labels(_frame([100, 110, 120, 130, 140, 150]), horizon_days=2)
    assert round(out["fwd_return_2d"].iloc[0], 6) == 20.0    # day1->day3
    assert np.isnan(out["fwd_return_2d"].iloc[-1])           # no future -> NaN


def test_add_labels_binary_profitable():
    # ceil=25 -> target 125; rises to 200 -> target hit
    out = labels.add_labels(_frame([100, 105, 200, 200, 200, 200]), horizon_days=3)
    assert out["is_profitable"].iloc[0] == 1.0
    assert out["barrier"].iloc[0] == labels.HIT_TARGET


def test_add_labels_stopped_is_not_profitable():
    # floor=10 -> stop at 100*(1-0.12)=88; drops to 80 -> stopped
    out = labels.add_labels(_frame([100, 80, 80, 80, 80, 80]), horizon_days=3)
    assert out["barrier"].iloc[0] == labels.HIT_STOP
    assert out["is_profitable"].iloc[0] == 0.0


def test_add_labels_per_card_isolation():
    """One card's prices must never leak into another's labels."""
    frame = pd.concat([_frame([100, 100, 100, 100, 100], "a"),
                       _frame([100, 500, 500, 500, 500], "b")], ignore_index=True)
    out = labels.add_labels(frame, horizon_days=2)
    a = out[out.player_id == "a"]
    b = out[out.player_id == "b"]
    assert a["barrier"].iloc[0] == labels.NEITHER    # flat card stays flat
    assert b["barrier"].iloc[0] == labels.HIT_TARGET


def test_add_labels_empty_frame():
    out = labels.add_labels(pd.DataFrame(columns=["player_id", "date", "price"]))
    assert out.empty and "is_profitable" in out.columns
