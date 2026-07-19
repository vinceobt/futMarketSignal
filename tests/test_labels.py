"""Labels: triple-barrier economics, purging, and forward returns."""

import numpy as np
import pandas as pd

from futmarket.ml import labels


def test_barrier_multipliers_gross_up_for_tax():
    up, dn = labels.barrier_multipliers(target_pct=25.0, stop_pct=8.0, tax_rate=0.05)
    # must sell above 1.25x to net 25% after a 5% tax
    assert round(up, 6) == round(1.25 / 0.95, 6)
    assert up > 1.25
    assert dn == 0.92


def test_triple_barrier_hits_target():
    prices = np.array([100, 105, 140, 90, 90, 90], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper_mult=1.3, lower_mult=0.92)
    assert out[0] == labels.HIT_TARGET      # 140 >= 130 before any stop


def test_triple_barrier_hits_stop_first():
    prices = np.array([100, 80, 200, 200, 200], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper_mult=1.3, lower_mult=0.92)
    assert out[0] == labels.HIT_STOP        # 80 <= 92 happened first


def test_triple_barrier_neither():
    prices = np.array([100, 101, 99, 102, 100], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper_mult=1.3, lower_mult=0.92)
    assert out[0] == labels.NEITHER


def test_triple_barrier_purges_incomplete_horizon():
    """Rows without a full horizon of future must be NaN, never guessed as 0."""
    prices = np.array([100, 101, 102, 103, 104], dtype=float)
    out = labels.triple_barrier(prices, horizon=3, upper_mult=1.3, lower_mult=0.92)
    assert np.isfinite(out[0]) and np.isfinite(out[1])
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])


def test_triple_barrier_order_matters():
    """Stop before target on the same path -> stopped, not profitable."""
    dips_first = np.array([100, 91, 200], dtype=float)
    pops_first = np.array([100, 200, 91], dtype=float)
    kw = dict(horizon=2, upper_mult=1.3, lower_mult=0.92)
    assert labels.triple_barrier(dips_first, **kw)[0] == labels.HIT_STOP
    assert labels.triple_barrier(pops_first, **kw)[0] == labels.HIT_TARGET


def _frame(prices, pid="p"):
    return pd.DataFrame({
        "player_id": [pid] * len(prices),
        "date": [f"2026-01-{i+1:02d}" for i in range(len(prices))],
        "price": prices})


def test_add_labels_forward_return():
    out = labels.add_labels(_frame([100, 110, 120, 130, 140, 150]), horizon_days=2)
    # day 1 -> day 3: 120/100 - 1 = +20%
    assert round(out["fwd_return_2d"].iloc[0], 6) == 20.0
    # last `horizon` rows have no future -> NaN
    assert np.isnan(out["fwd_return_2d"].iloc[-1])


def test_add_labels_binary_profitable():
    # rises hard -> target hit; is_profitable should be 1 early on
    out = labels.add_labels(_frame([100, 105, 200, 200, 200, 200]),
                            horizon_days=3, target_pct=25.0, stop_pct=8.0)
    assert out["is_profitable"].iloc[0] == 1.0
    assert out["barrier"].iloc[0] == labels.HIT_TARGET


def test_add_labels_stopped_is_not_profitable():
    out = labels.add_labels(_frame([100, 80, 80, 80, 80, 80]),
                            horizon_days=3, target_pct=25.0, stop_pct=8.0)
    assert out["barrier"].iloc[0] == labels.HIT_STOP
    assert out["is_profitable"].iloc[0] == 0.0


def test_add_labels_per_card_isolation():
    """One card's prices must never leak into another's labels."""
    frame = pd.concat([_frame([100, 100, 100, 100, 100], "a"),
                       _frame([100, 500, 500, 500, 500], "b")], ignore_index=True)
    out = labels.add_labels(frame, horizon_days=2, target_pct=25.0, stop_pct=8.0)
    a = out[out.player_id == "a"]
    b = out[out.player_id == "b"]
    assert a["barrier"].iloc[0] == labels.NEITHER    # flat card stays flat
    assert b["barrier"].iloc[0] == labels.HIT_TARGET


def test_add_labels_empty_frame():
    out = labels.add_labels(pd.DataFrame(columns=["player_id", "date", "price"]))
    assert out.empty and "is_profitable" in out.columns
