"""Picks: the reasons a card is surfaced, and the buy/sell numbers attached."""

import pandas as pd
import pytest

from futmarket import db as futdb
from futmarket.ml import evaluate, picks


def _row(**kw):
    base = dict(dist_to_floor_pct=50.0, z_score=0.0, day_of_week=1,
                days_since_card_release=100, days_to_next_promo=30,
                active_sbc_count=0, cohort_ret_7d=0.0, rel_strength_7d=0.0)
    base.update(kw)
    return pd.Series(base)


def test_reason_near_floor():
    out = picks._reasons(_row(dist_to_floor_pct=3.0))
    assert any("floor" in r for r in out)


def test_reason_on_the_dip():
    out = picks._reasons(_row(z_score=-2.0))
    assert any("on the dip" in r and "sigma" in r for r in out)


def test_reason_room_to_bounce():
    out = picks._reasons(_row(dist_to_ceiling_pct=20.0))
    assert any("room to bounce" in r for r in out)


def test_reason_weekly_supply_cycle():
    """The reward cycle the market actually runs on."""
    assert any("trough" in r for r in picks._reasons(_row(day_of_week=6)))    # Sun
    assert any("rewards dump" in r for r in picks._reasons(_row(day_of_week=3)))  # Thu
    assert any("Weekend League" in r for r in picks._reasons(_row(day_of_week=4)))  # Fri


def test_reason_release_curve():
    """The measured bottom is day 4-6, not the day-9 folklore: held three weeks
    it returns +11.5% net at +25pp alpha, better than day 7-9 at every horizon."""
    best = picks._reasons(_row(days_since_card_release=5))
    assert any("measured" in r and "bottom" in r for r in best)
    later = picks._reasons(_row(days_since_card_release=9))
    assert any("past the best window" in r for r in later)
    fresh = picks._reasons(_row(days_since_card_release=1))
    assert any("release crash" in r for r in fresh)


def test_reason_incoming_promo():
    assert any("promo in" in r for r in picks._reasons(_row(days_to_next_promo=2)))


def test_live_sbc_count_is_not_quoted():
    """50-60 SBCs run at all times, so the count appears on every card and
    discriminates nothing -- it is deliberately not a reason."""
    assert not any("SBC" in r for r in picks._reasons(_row(active_sbc_count=62)))


def test_reason_cohort_and_lag():
    out = picks._reasons(_row(cohort_ret_7d=6.0, rel_strength_7d=-9.0))
    assert any("group is rising" in r for r in out)
    assert any("lagging its group" in r for r in out)


def test_reasons_never_empty():
    out = picks._reasons(_row())
    assert out and isinstance(out[0], str)


def test_reasons_survive_missing_values():
    out = picks._reasons(pd.Series({}))
    assert out == ["model pattern match (no single standout reason)"]


def test_generate_requires_a_trained_model(conn):
    with pytest.raises(RuntimeError, match="no trained model"):
        picks.generate(conn)


def test_falling_knife_is_flagged_not_praised():
    """Below the 30-day low means still falling -- a warning, never a buy reason."""
    out = picks._reasons(_row(dist_to_floor_pct=-40.0))
    assert any("WARNING" in r and "still falling" in r for r in out)
    assert not any("near its floor" in r for r in out)


def test_extreme_relative_strength_is_not_quoted():
    """A -92% 'lag' is arithmetic noise off a tiny price, not a dislocation."""
    out = picks._reasons(_row(rel_strength_7d=-92.0))
    assert not any("lagging its group" in r for r in out)
    assert any("lagging its group" in r for r in picks._reasons(_row(rel_strength_7d=-12.0)))


def test_barriers_target_resistance_stop_support():
    """Target sits at the card's resistance, stop below its support, and the
    reward:risk is computed net of tax."""
    target, stop, rr = picks._barriers(
        20_000, 20_000, ceil_pct=20.0, floor_pct=4.0, tax_rate=0.05,
        stop_min_pct=5.0, stop_buffer_pct=2.0)
    assert target == 24_000                     # +20% to resistance
    assert stop == round(20_000 * (1 - 0.06))   # floor 4% + 2% buffer
    assert target > 20_000 > stop and rr > 1.0


def test_barriers_reward_risk_reflects_a_thin_ceiling():
    """A card near its ceiling has little upside -> low reward:risk (gets skipped)."""
    _, _, rr = picks._barriers(20_000, 20_000, ceil_pct=3.0, floor_pct=4.0,
                               tax_rate=0.05, stop_min_pct=5.0)
    assert rr < 1.0


def test_stop_is_never_above_the_market_price():
    """**The bug that broke the last strategy.**

    Barriers used to be derived from the marked-up entry while the scorer
    compared them against the raw market series, so a "5% stop" landed on
    average 0.76% ABOVE the live price and 53 of 59 trades stopped out before
    they began. The stop must be a level on the series that grades it, whatever
    premium the entry carries.
    """
    market = 20_000
    for premium in (1.0, 1.05, 1.25, 1.5):      # including the pathological ones
        entry = int(market * premium)
        target, stop, _ = picks._barriers(
            market, entry, ceil_pct=20.0, floor_pct=4.0, tax_rate=0.05)
        assert stop < market, f"stop {stop} not below market {market} at {premium}x"
        assert target > market


def test_stop_sits_outside_the_cards_daily_noise():
    """The median card-day ranges 14.3%. A 5% stop is a coin-flip on jitter and
    truncates exactly the recovery the trade exists to capture."""
    _, stop, _ = picks._barriers(20_000, 20_000, ceil_pct=25.0, floor_pct=1.0,
                                 tax_rate=0.05)
    assert (1 - stop / 20_000) * 100 >= picks.STOP_MIN_PCT >= 12.0


# ---- choosing how long to hold, and whether to trade at all ----------------

# What the gate has historically paid at each horizon. Wins grow with the hold;
# losses do too, which is what makes the choice non-trivial.
PAYOFFS = {
    3:  {"win_net": 12.0, "loss_net": -10.0, "base_rate": 0.37},
    5:  {"win_net": 16.0, "loss_net": -12.0, "base_rate": 0.36},
    7:  {"win_net": 20.0, "loss_net": -14.0, "base_rate": 0.35},
    10: {"win_net": 28.0, "loss_net": -16.0, "base_rate": 0.42},
    14: {"win_net": 35.0, "loss_net": -18.0, "base_rate": 0.40},
}


def _plan(clears, *, payoffs=None, **kw):
    probs = (clears if isinstance(clears, dict)
             else {h: clears for h in PAYOFFS})
    return picks._choose_horizon(probs, payoffs or PAYOFFS, **kw)


def test_a_coin_flip_is_not_a_trade():
    """The answer the old code could never give. At the gate's own base rate the
    expected value is negative, so the honest recommendation is to buy nothing."""
    assert _plan(0.40) is None


def test_a_confident_card_becomes_a_trade():
    plan = _plan(0.80)
    assert plan is not None
    horizon, expected, edge, prob = plan
    assert horizon in PAYOFFS
    assert expected > picks.MIN_EXPECTED_NET_PCT
    assert edge > 0                       # better than picking blind from the gate
    assert prob == pytest.approx(0.80)


def test_the_shortest_horizon_that_pays_wins():
    """Coins tied up in one card are coins not working in another, so the choice
    is best expected return *per day held*, not the biggest headline number."""
    flat = {3: {"win_net": 30.0, "loss_net": -5.0, "base_rate": 0.4},
            14: {"win_net": 32.0, "loss_net": -5.0, "base_rate": 0.4}}
    horizon, _, _, _ = _plan(0.9, payoffs=flat)
    assert horizon == 3          # nearly the same payoff, a fifth of the wait


def test_a_much_bigger_payoff_justifies_a_longer_hold():
    steep = {3: {"win_net": 8.0, "loss_net": -6.0, "base_rate": 0.4},
             14: {"win_net": 90.0, "loss_net": -6.0, "base_rate": 0.4}}
    horizon, _, _, _ = _plan(0.9, payoffs=steep)
    assert horizon == 14


def test_a_horizon_the_model_is_unsure_about_is_skipped():
    """Per-horizon probabilities differ, and only the confident ones qualify."""
    probs = {h: 0.2 for h in PAYOFFS}
    probs[10] = 0.85
    horizon, _, _, _ = _plan(probs)
    assert horizon == 10


def test_the_magnitude_head_is_deliberately_not_consulted():
    """Measured: the excess-return regressor scores WORSE than assuming a card
    moves with the market (-12% to -19% skill). Expected value must come from the
    classifier, which works, plus the gate's measured history -- never from a
    predicted magnitude we have no evidence for."""
    import inspect
    args = inspect.signature(picks._choose_horizon).parameters
    assert "excess" not in args and "market_drift" not in args


def test_per_day_ranking_prefers_the_quicker_trade():
    """The default: coins freed early can go into the next trade, so a shorter
    hold earning almost as much wins."""
    quick = {3: {"win_net": 30.0, "loss_net": -10.0, "base_rate": 0.37},
             14: {"win_net": 34.0, "loss_net": -10.0, "base_rate": 0.40}}
    assert _plan(0.80, payoffs=quick)[0] == 3


def test_total_ranking_takes_the_bigger_trade_when_the_days_are_free():
    """The weekly-cycle trade can only be opened on a Saturday or Sunday, so
    coming out on Tuesday rather than Thursday frees coins with nothing to do.
    Charging the trade for those days would pick the worse exit."""
    quick = {3: {"win_net": 30.0, "loss_net": -10.0, "base_rate": 0.37},
             14: {"win_net": 34.0, "loss_net": -10.0, "base_rate": 0.40}}
    assert _plan(0.80, payoffs=quick, rank_by="total")[0] == 14


def test_the_weekend_trade_does_not_exist_midweek(conn, monkeypatch):
    """A gate that names a buy day must say so on the other days, not return an
    empty list that reads exactly like a broken pipeline (trap #17)."""
    import datetime as dt

    class Wednesday(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 5, 12, 0, tzinfo=tz)      # a Wednesday

    monkeypatch.setattr(picks, "datetime", Wednesday)
    # No model is loaded and no dataset is built: the day check comes first, so
    # this returning cleanly *is* the assertion.
    assert picks.generate(conn, strategy="weekend_v1") == []


def test_the_weekend_trade_is_tier_a_only():
    """Its edge inverts on tier B (-4.7% against tier A's +4.2%), so the tier
    restriction lives with the gate and every consumer reads the same one."""
    assert evaluate.gate_tiers("weekend_v1") == ("A",)
    assert evaluate.gate_tiers("relval_v1") == evaluate.TRADEABLE_TIERS


def test_target_is_capped_at_what_the_gate_has_actually_paid():
    """A real pick quoted a sell 141% up with reward:risk 5.1 — that was the
    card's 30-day high, i.e. the price it used to be before it crashed, not a
    fortnight's trade. The target is capped at the measured winning payoff."""
    # resistance is 141% up, but a winning trade at this horizon nets ~27%
    target, _, rr = picks._barriers(
        12_000, 12_600, ceil_pct=141.0, floor_pct=2.0, tax_rate=0.05,
        sell_slippage_pct=2.0, payoff_target_pct=36.0)
    assert target == round(12_000 * 1.36)
    assert rr < 2.0                      # a believable ratio, not 5.1


def test_resistance_still_caps_from_the_other_side():
    """No sense targeting through a level the card keeps failing at."""
    target, _, _ = picks._barriers(
        10_000, 10_000, ceil_pct=8.0, floor_pct=2.0, tax_rate=0.05,
        payoff_target_pct=36.0)
    assert target == round(10_000 * 1.08)


# ---- two strategies, judged separately ------------------------------------

def test_strategies_name_gates_that_actually_exist():
    """Picks and the backtest must share one gate definition, or what we trade
    and what we claim to have measured quietly drift apart."""
    from futmarket.ml import evaluate
    for name, spec in picks.STRATEGIES.items():
        assert spec["gate"] in evaluate.GATES, f"{name} names an unknown gate"
        assert spec["horizons"], f"{name} has no holding periods"


def test_the_release_trade_is_allowed_below_the_floor():
    """Trap #7 says below the 30-day low is a falling knife — true for a dip, and
    deliberately wrong for a six-day-old promo card, which makes a new low every
    day it exists. Its protection is the time stop and its age instead."""
    assert picks.STRATEGIES["release_v1"]["allow_below_floor"] is True
    assert not picks.STRATEGIES["relval_v1"].get("allow_below_floor")


def test_generate_rejects_an_unknown_strategy(conn):
    with pytest.raises(ValueError, match="unknown strategy"):
        picks.generate(conn, strategy="nonsense")


def test_a_pick_carries_the_strategy_that_made_it(conn):
    """The two trades pay differently, so the record must never blend them."""
    p = picks.Pick(player_id="c1", name="x", rating=84, version="TOTS",
                   confidence=0.7, price_now=10_000, buy_low=10_000,
                   buy_high=10_500, sell_target=13_000, stop=8_500,
                   reward_risk=1.2, liquidity_tier="A", sales_per_hour=50.0,
                   strategy="release_v1")
    futdb.upsert_card_meta(conn, {"player_id": "c1", "name": "x"})
    picks.save(conn, [p])
    assert futdb.open_picks(conn)[0]["strategy"] == "release_v1"
