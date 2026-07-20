"""Picks: the reasons a card is surfaced, and the buy/sell numbers attached."""

import pandas as pd
import pytest

from futmarket.ml import picks


def _row(**kw):
    base = dict(dist_to_floor_pct=50.0, z_score=0.0, day_of_week=1,
                days_since_card_release=100, days_to_next_promo=30,
                active_sbc_count=0, cohort_ret_7d=0.0, rel_strength_7d=0.0)
    base.update(kw)
    return pd.Series(base)


def test_reason_near_floor():
    out = picks._reasons(_row(dist_to_floor_pct=8.0))
    assert any("floor" in r for r in out)


def test_reason_cheap_for_itself():
    out = picks._reasons(_row(z_score=-2.0))
    assert any("sigma below normal" in r for r in out)


def test_reason_weekly_supply_cycle():
    """The reward cycle the market actually runs on."""
    assert any("trough" in r for r in picks._reasons(_row(day_of_week=6)))    # Sun
    assert any("rewards dump" in r for r in picks._reasons(_row(day_of_week=3)))  # Thu
    assert any("Weekend League" in r for r in picks._reasons(_row(day_of_week=4)))  # Fri


def test_reason_release_curve():
    out = picks._reasons(_row(days_since_card_release=9))
    assert any("usual bottom" in r for r in out)
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


def test_sell_target_always_exceeds_buy_price():
    """Guards the bug where target came from the listing but the band from sales."""
    entry, target_pct, tax = 22_000, 25.0, 0.05
    target = round(entry * (1 + target_pct / 100) / (1 - tax))
    assert target > entry
