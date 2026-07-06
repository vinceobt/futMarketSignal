"""Signal vocabulary for the unified decision engine in strategy.py.

The engine itself — decide(), which turns a price series into an explicit
BUY / SELL / HOLD / SKIP with machine reason codes — lives in strategy.py.
This module only holds the shared action constants (imported across the CLI,
services, backtester, and dashboard) and the event types that crash prices.
"""

from __future__ import annotations

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"   # holding an open position with no exit trigger yet
SKIP = "SKIP"   # flat, and at least one entry gate failed (codes say which)

# Events that typically flood supply / crash a card's price in the short term.
CRASHING_EVENTS = {"SBC", "PROMO", "TOTW"}
