"""The honest scoreboard: costs, calendar-matched returns, and alpha by month."""

import numpy as np
import pandas as pd
import pytest

from futmarket.ml import evaluate


# ---- costs ----------------------------------------------------------------

def test_break_even_is_the_number_that_decides_everything():
    """5% tax + 2% slippage means a trade must gain 7.5% just to stand still.

    This is larger than the entire five-day dip bounce the old strategy traded
    (~4.7% gross), which is why it could not have made money however good the
    model got.
    """
    cost = evaluate.round_trip_cost_pct(tax_rate=0.05, sell_slippage_pct=2.0)
    assert 7.0 < cost < 8.0
    # ...and by construction, a move of exactly that size nets zero.
    assert abs(evaluate.net_return_pct(cost, tax_rate=0.05,
                                       sell_slippage_pct=2.0)) < 1e-9


def test_paying_over_the_listing_raises_the_bar():
    patient = evaluate.round_trip_cost_pct(buy_premium_pct=0.0)
    hasty = evaluate.round_trip_cost_pct(buy_premium_pct=5.0)
    assert hasty > patient + 4


def test_a_flat_card_still_loses_money():
    """The trap behind every optimistic number here: doing nothing is not free."""
    assert evaluate.net_return_pct(0.0, tax_rate=0.05, sell_slippage_pct=2.0) < -6


# ---- forward returns ------------------------------------------------------

def _series(pid, dates, prices):
    return pd.DataFrame({"player_id": pid, "date": dates, "price": prices})


def test_forward_return_is_matched_on_the_calendar_not_the_row():
    """A gap in collection must produce NaN, not a mislabelled trade.

    Counting rows instead of days would measure this card's '2-day' return over
    six actual days and quietly teach the model the wrong thing.
    """
    frame = _series("a", ["2026-01-01", "2026-01-02", "2026-01-07"],
                    [100.0, 110.0, 200.0])
    out = evaluate.add_forward_returns(frame, horizons=(2,))
    # 01-01 + 2 days = 01-03, which was never collected -> honest gap
    assert np.isnan(out["fwd_return_2d"].iloc[0])
    # 01-05 doesn't exist either
    assert np.isnan(out["fwd_return_2d"].iloc[1])


def test_forward_return_uses_the_matching_day():
    frame = _series("a", ["2026-01-01", "2026-01-02", "2026-01-03"],
                    [100.0, 150.0, 120.0])
    out = evaluate.add_forward_returns(frame, horizons=(2,))
    assert out["fwd_return_2d"].iloc[0] == pytest.approx(20.0)


def test_one_card_never_borrows_anothers_future():
    frame = pd.concat([
        _series("a", ["2026-01-01", "2026-01-02"], [100.0, 100.0]),
        _series("b", ["2026-01-01", "2026-01-02"], [100.0, 500.0]),
    ], ignore_index=True)
    out = evaluate.add_forward_returns(frame, horizons=(1,))
    a = out[out.player_id == "a"]["fwd_return_1d"].iloc[0]
    assert a == pytest.approx(0.0)


# ---- benchmark and excess -------------------------------------------------

def test_excess_return_is_zero_when_a_card_moves_with_the_market():
    """The label the whole rebuild turns on. A card that did exactly what
    everything else did has told us nothing, and must score zero."""
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "date": ["2026-01-01"] * 3,
        "price": [100.0, 100.0, 100.0],
        "fwd_return_5d": [-9.0, -9.0, -9.0],
    })
    out = evaluate.add_benchmark_returns(frame, horizons=(5,))
    assert list(out["excess_return_5d"]) == [0.0, 0.0, 0.0]


def test_falling_less_than_the_market_is_positive_excess():
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "date": ["2026-01-01"] * 3,
        "price": [100.0] * 3,
        "fwd_return_5d": [-4.0, -9.0, -14.0],
    })
    out = evaluate.add_benchmark_returns(frame, horizons=(5,))
    assert out["excess_return_5d"].iloc[0] == pytest.approx(5.0)
    assert out["excess_return_5d"].iloc[2] == pytest.approx(-5.0)


# ---- gates and the backtest ----------------------------------------------

def test_gate_selects_only_a_real_dislocation():
    frame = pd.DataFrame({"z_score": [-0.6, -1.6, -2.5, 0.5],
                          "dist_to_floor_pct": [2.0, 2.0, 1.0, 2.0]})
    shallow = evaluate.gate_mask(frame, "dip_v1").sum()
    deep = evaluate.gate_mask(frame, "relval_v1").sum()
    assert shallow == 3 and deep == 2


def test_gate_ignores_rows_with_no_z_score():
    frame = pd.DataFrame({"z_score": [np.nan, -2.0],
                          "dist_to_floor_pct": [1.0, 1.0]})
    assert evaluate.gate_mask(frame, "relval_v1").tolist() == [False, True]


def _universe(n_months=4, per_month=90, seed=0):
    """A market where gated cards genuinely bounce and the rest go nowhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, n_months + 1):
        for i in range(per_month):
            gated = i < per_month // 3
            rows.append({
                "player_id": f"c{i}",
                "date": f"2026-0{m}-15",
                "price": 10_000.0,
                "liq_tier": "A",
                "z_score": -2.0 if gated else 0.5,
                "dist_to_floor_pct": 1.0 if gated else 40.0,
                # gated cards recover well past the round trip; the rest drift
                "fwd_return_14d": (25.0 if gated else 0.0) + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def test_backtest_reports_alpha_against_the_same_month_market():
    res = evaluate.backtest(_universe(), horizon=14, gate="relval_v1")
    assert res["n"] > 0
    assert res["median_net_pct"] > 0                 # clears the round trip
    assert res["universe_median_net_pct"] < 0        # a flat market still loses
    assert res["median_alpha_pp"] > 15
    # An edge must show up repeatedly, not in one exceptional month.
    assert res["months_positive"] == res["months"] == 4


def test_backtest_refuses_to_pool_thin_months():
    """A month with a handful of trades has a meaningless median and must not
    contribute to the headline."""
    res = evaluate.backtest(_universe(n_months=2, per_month=30), horizon=14,
                            gate="relval_v1", min_month_rows=25)
    assert res["months"] == 0          # 10 gated trades/month is too few


def test_backtest_excludes_untradeable_cards():
    """Tier C returns flattered the old backtest: +17.3% vs +1.6% on tier A. The
    thinner the card, the more its 'price' is one stale listing."""
    frame = _universe()
    frame.loc[frame.index[:len(frame) // 2], "liq_tier"] = "C"
    res = evaluate.backtest(frame, horizon=14, gate="relval_v1",
                            tiers=("A", "B"))
    assert all(t["tier"] == "A" for t in res["by_tier"])


def test_backtest_says_so_when_a_gate_finds_nothing():
    frame = _universe()
    frame["z_score"] = 1.0
    res = evaluate.backtest(frame, horizon=14, gate="relval_v1")
    assert res["n"] == 0 and "note" in res


# ---- payoff profile -------------------------------------------------------

def _long_universe(days=160, per_day=12, win_share=0.4, seed=1):
    """A gated population spread over enough calendar to split walk-forward."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-10-01")
    rows = []
    for d in range(days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for i in range(per_day):
            gated = i < per_day // 2
            wins = gated and rng.random() < win_share
            rows.append({
                "player_id": f"c{i}", "date": date, "price": 10_000.0,
                "liq_tier": "A",
                "z_score": -2.0 if gated else 0.5,
                "dist_to_floor_pct": 1.0 if gated else 40.0,
                "fwd_return_14d": (30.0 if wins else -8.0) + rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


def test_payoff_profile_separates_the_wins_from_the_losses():
    """`picks` needs both halves of the expected value, and only one of them
    comes from a model: the regression head that was meant to predict magnitude
    scores worse than assuming a card moves with the market, so the size of the
    win and the loss is taken from the gate's measured history instead."""
    profile = evaluate.payoff_profile(_long_universe(), gate="relval_v1",
                                      horizons=(14,))
    assert 14 in profile
    p = profile[14]
    assert p["win_net"] > 0 > p["loss_net"]
    assert 0.2 < p["base_rate"] < 0.7
    assert p["n"] > 0


def test_payoff_profile_is_measured_out_of_sample_by_default():
    """Pooling the whole season put the 14-day base rate at 58% while
    walk-forward said 40% — using the in-sample figure would inflate every
    expected value by ~7pp and quietly reintroduce the optimism this module
    exists to remove."""
    frame = _long_universe()
    oos = evaluate.payoff_profile(frame, gate="relval_v1", horizons=(14,))[14]
    hist = evaluate.payoff_profile(frame, gate="relval_v1", horizons=(14,),
                                   out_of_sample=False)[14]
    assert oos["out_of_sample"] is True and hist["out_of_sample"] is False
    # the honest number carries its optimistic twin for comparison
    assert oos["in_sample_base_rate"] == pytest.approx(hist["base_rate"])
    assert oos["n"] < hist["n"]          # test folds only


def test_payoff_profile_skips_a_horizon_with_no_variation():
    """If every gated trade cleared (or none did), there is no loss to size and
    no expected value to compute — better to omit the horizon than invent one."""
    frame = _long_universe()
    frame["fwd_return_14d"] = 25.0             # everything wins
    assert evaluate.payoff_profile(frame, gate="relval_v1", horizons=(14,)) == {}


def test_the_gate_is_defined_in_exactly_one_place():
    """Picks derive their thresholds from GATES rather than restating them.

    A second copy of `-1.5` in picks.py would be free to drift away from the
    number the backtest measured, and we would go on quoting a track record for
    a rule we no longer trade — the same class of mistake as grading against a
    different price series than we trained on.
    """
    from futmarket.ml import picks
    assert picks.ENTRY_Z_MAX == evaluate.GATES["relval_v1"]["z_score"][1]
    assert (picks.ENTRY_FLOOR_MAX_PCT
            == evaluate.GATES["relval_v1"]["dist_to_floor_pct"][1])
    assert ((picks.RELEASE_AGE_MIN, picks.RELEASE_AGE_MAX)
            == evaluate.GATES["release"]["days_since_card_release"])
