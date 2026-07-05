"""Signal rule decisions + human-readable reason strings (Phase 3 DoD)."""

from futmarket.alerts import Alert, format_digest, format_realtime
from futmarket.features import FeatureRow
from futmarket.signals import BUY, HOLD, SELL, SignalParams, evaluate


def _f(price, z=None, mean=None, days=None, ev=None, mom=None):
    return FeatureRow(
        player_id="p", source="test", timestamp="t", price=price,
        pct_change_1h=None, pct_change_24h=mom, pct_change_7d=None,
        rolling_mean_24h=mean, rolling_std_24h=None, z_score=z,
        days_to_next_event=days, next_event_type=ev, is_weekend_window=0,
    )


# momentum guard on (5% 24h) for the confirmation tests
PARAMS = SignalParams(buy_z=-1.5, sell_z=1.5, event_guard_days=2, momentum_guard_pct=5.0)


def test_buy_when_cheap_and_no_event():
    s = evaluate(_f(80_000, z=-1.8, mean=92_000), PARAMS)
    assert s.signal_type == BUY
    assert "below its 24h average" in s.reason
    assert "92,000" in s.reason and "80,000" in s.reason  # real feature values
    assert 0 < s.confidence <= 1


def test_buy_suppressed_by_imminent_crashing_event():
    s = evaluate(_f(80_000, z=-1.8, mean=92_000, days=1, ev="SBC"), PARAMS)
    assert s.signal_type == HOLD
    assert "SBC" in s.reason and "1d" in s.reason


def test_event_far_away_does_not_suppress_buy():
    s = evaluate(_f(80_000, z=-1.8, mean=92_000, days=10, ev="SBC"), PARAMS)
    assert s.signal_type == BUY


def test_sell_when_overextended():
    s = evaluate(_f(120_000, z=2.1, mean=100_000), PARAMS)
    assert s.signal_type == SELL
    assert "above its 24h average" in s.reason


# ---- momentum confirmation (falling-knife / let-it-run) ----

def test_cheap_but_still_falling_is_held_not_bought():
    # z says undervalued, but the price is still crashing 12% in 24h -> falling knife
    s = evaluate(_f(80_000, z=-1.8, mean=92_000, mom=-12.0), PARAMS)
    assert s.signal_type == HOLD
    assert "still falling" in s.reason


def test_cheap_and_drop_stabilized_is_bought():
    # z undervalued and the drop has flattened (only -1% in 24h) -> dip confirmed
    s = evaluate(_f(80_000, z=-1.8, mean=92_000, mom=-1.0), PARAMS)
    assert s.signal_type == BUY
    assert "stabilized" in s.reason


def test_overextended_but_still_climbing_is_held_not_sold():
    # z says dear, but it's still ripping +9% in 24h -> let it run
    s = evaluate(_f(120_000, z=2.1, mean=100_000, mom=9.0), PARAMS)
    assert s.signal_type == HOLD
    assert "still climbing" in s.reason


def test_momentum_guard_off_reverts_to_pure_zscore():
    off = SignalParams(buy_z=-1.5, sell_z=1.5, event_guard_days=2, momentum_guard_pct=0)
    # even mid-crash, guard-off buys purely on z-score
    assert evaluate(_f(80_000, z=-1.8, mean=92_000, mom=-12.0), off).signal_type == BUY


def test_hold_when_in_normal_range():
    assert evaluate(_f(100_000, z=0.3, mean=99_000), PARAMS).signal_type == HOLD


def test_hold_with_reason_when_no_history():
    s = evaluate(_f(100_000, z=None), PARAMS)
    assert s.signal_type == HOLD
    assert s.confidence == 0.0
    assert "insufficient" in s.reason.lower()


def test_confidence_scales_with_z_magnitude():
    weak = evaluate(_f(80_000, z=-1.6, mean=90_000), PARAMS).confidence
    strong = evaluate(_f(70_000, z=-3.0, mean=90_000), PARAMS).confidence
    assert strong > weak
    assert strong == 1.0  # saturates at ~3 sigma


def test_alert_formatting():
    a = Alert("salah-base", "Mohamed Salah", BUY, 0.72, "undervalued, buy")
    assert "Mohamed Salah" in format_realtime(a) and "72%" in format_realtime(a)
    digest = format_digest([a], "2026-07-05")
    assert "Mohamed Salah" in digest and "BUY" in digest
    assert "nothing actionable" in format_digest([], "2026-07-05")
