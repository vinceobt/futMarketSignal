"""Phase 3 signal engine.

Interpretable, rule-based BUY/SELL/HOLD decisions with a plain-language reason
built from the actual feature values. This same `evaluate()` is what the
Phase 2 backtester replays over history, so a rule you trust in a backtest is
literally the rule that goes live — no separate, drifting implementation.

ML is deliberately not here: it only earns a place once these rules beat a
naive baseline in the backtest harness, and it must be scored by that same
harness before replacing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import FeatureRow

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"

# Events that typically flood supply / crash a card's price in the short term.
CRASHING_EVENTS = {"SBC", "PROMO", "TOTW"}


@dataclass(frozen=True)
class SignalParams:
    buy_z: float = -1.5
    sell_z: float = 1.5
    event_guard_days: int = 2
    # Momentum confirmation: a BUY is blocked while the price is still falling
    # faster than this (24h %), and a SELL is held while it is still climbing
    # faster than this. 0 disables the guard (pure mean-reversion) — and 0 is the
    # default because it must beat the plain rule in YOUR backtest before you
    # trust it (on coarse/daily history it can over-block dip-buys).
    momentum_guard_pct: float = 0.0

    @classmethod
    def from_config(cls, config) -> "SignalParams":
        return cls(buy_z=config.buy_z, sell_z=config.sell_z,
                   event_guard_days=config.event_guard_days,
                   momentum_guard_pct=config.momentum_guard_pct)


@dataclass(frozen=True)
class Signal:
    signal_type: str
    confidence: float
    reason: str


def _confidence(z: float) -> float:
    """|z| mapped into [0, 1]; ~3σ saturates to full confidence."""
    return max(0.0, min(1.0, abs(z) / 3.0))


def _imminent_crash(f: FeatureRow, guard_days: int) -> bool:
    return (f.days_to_next_event is not None
            and f.days_to_next_event <= guard_days
            and (f.next_event_type in CRASHING_EVENTS))


def _still_falling(f: FeatureRow, guard_pct: float) -> bool:
    """Price dropping faster than the guard over the last 24h — a falling knife.
    Uses pct_change_24h (the same field the backtester replays, so the live and
    backtested rules never diverge)."""
    m = f.pct_change_24h
    return guard_pct > 0 and m is not None and m <= -guard_pct


def _still_climbing(f: FeatureRow, guard_pct: float) -> bool:
    m = f.pct_change_24h
    return guard_pct > 0 and m is not None and m >= guard_pct


def evaluate(f: FeatureRow, params: SignalParams) -> Signal:
    """Decide on a single player from its latest feature row.

    Two layers: mean-reversion (z-score: is the price cheap or dear?) gated by a
    momentum check (is it still moving that way?). The momentum gate blocks
    buying a card that is still crashing and holds a sell while it is still
    ripping — the "don't catch a falling knife / let winners run" discipline."""
    if f.z_score is None:
        return Signal(HOLD, 0.0,
                      "insufficient recent history (need ≥2 snapshots in the "
                      "last 24h) to judge whether the price is high or low")

    mean = f.rolling_mean_24h
    norm = f"{mean:,.0f}" if mean is not None else "?"
    z = f.z_score
    m = f.pct_change_24h

    if z <= params.buy_z:
        if _imminent_crash(f, params.event_guard_days):
            return Signal(
                HOLD, _confidence(z) * 0.5,
                f"price {f.price:,} is {abs(z):.1f}σ below its 24h average "
                f"({norm}), but a {f.next_event_type} is {f.days_to_next_event}d "
                f"away that could sink it further — waiting")
        if _still_falling(f, params.momentum_guard_pct):
            return Signal(
                HOLD, _confidence(z) * 0.5,
                f"price {f.price:,} is {abs(z):.1f}σ below its 24h average "
                f"({norm}), but it's still falling ({m:.1f}% in 24h) — "
                f"waiting for the drop to stop before buying")
        confirm = (f"and the drop has stabilized ({m:+.1f}% 24h)"
                   if m is not None else "and no crashing event is pending")
        return Signal(
            BUY, _confidence(z),
            f"price {f.price:,} is {abs(z):.1f}σ below its 24h average "
            f"({norm}) {confirm} — undervalued, buy")

    if z >= params.sell_z:
        if _still_climbing(f, params.momentum_guard_pct):
            return Signal(
                HOLD, _confidence(z) * 0.5,
                f"price {f.price:,} is {z:.1f}σ above its 24h average ({norm}), "
                f"but it's still climbing ({m:+.1f}% in 24h) — letting it run")
        return Signal(
            SELL, _confidence(z),
            f"price {f.price:,} is {z:.1f}σ above its 24h average ({norm}) "
            f"— overextended, take profit")

    return Signal(
        HOLD, _confidence(z),
        f"price {f.price:,} is within normal range ({z:+.1f}σ of its 24h "
        f"average {norm}) — no edge")
