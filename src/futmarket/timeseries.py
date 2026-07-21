"""Small price-series helpers shared across services.

A price history is a list of ``{"timestamp", "price"}`` rows; turning it into a
clean, ascending pandas Series is a step several callers need, so it lives here
rather than being duplicated.
"""

from __future__ import annotations

import pandas as pd


def to_series(rows) -> pd.Series:
    """rows of {timestamp, price} -> ascending float Series indexed by UTC time.
    Collapses any duplicate timestamps to their last value."""
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([r["timestamp"] for r in rows], utc=True)
    s = pd.Series([float(r["price"]) for r in rows], index=idx).sort_index()
    return s[~s.index.duplicated(keep="last")]
