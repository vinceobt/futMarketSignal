"""Walk-forward validation with an embargo — the honest way to test this.

Random k-fold would be catastrophic here: it lets the model train on next week
and test on last week, and on correlated cards from the same day. Scores would
look superb and the model would lose money live.

So we split strictly by time (train on the past, test on the future) and insert
an **embargo** gap between them. The embargo matters because labels look forward:
a row dated D carries the outcome up to D+horizon, so without a gap the tail of
the training set already "knows" the start of the test period. The embargo must
therefore be at least the label horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_splits(dates, *, n_splits: int = 4, embargo_days: int = 7,
                        min_train_days: int = 30):
    """Yield (train_idx, test_idx) pairs, expanding the training window forward.

    `dates` is a date-like series aligned to the rows being split. Test blocks
    are consecutive, equal-width spans of the *calendar*, each preceded by an
    embargo gap that belongs to neither side.
    """
    series = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    unique_days = np.array(sorted(series.dt.normalize().unique()))
    if len(unique_days) < min_train_days + n_splits:
        return

    embargo = pd.Timedelta(days=embargo_days)
    first_test = unique_days[min_train_days]
    last_day = unique_days[-1]
    span = (last_day - first_test) / max(n_splits, 1)
    if span <= pd.Timedelta(0):
        return

    normalized = series.dt.normalize()
    for i in range(n_splits):
        test_start = first_test + span * i
        test_end = first_test + span * (i + 1)
        train_mask = normalized < (test_start - embargo)
        test_mask = (normalized >= test_start) & (normalized < test_end)
        if i == n_splits - 1:                      # final fold takes the tail
            test_mask = normalized >= test_start
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def describe_splits(dates, **kwargs) -> list[dict]:
    """Summarise the folds — handy for logging what was actually validated."""
    series = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    out = []
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(dates, **kwargs)):
        out.append({
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_end": str(series.iloc[train_idx].max().date()),
            "test_start": str(series.iloc[test_idx].min().date()),
            "test_end": str(series.iloc[test_idx].max().date()),
        })
    return out
