"""Walk-forward validation: time ordering and the embargo gap."""

import numpy as np
import pandas as pd

from futmarket.ml import validation


def _dates(n_days, cards=3):
    """n_days of calendar, `cards` rows per day (mimics the real matrix)."""
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.Series(np.repeat(days, cards))


def test_train_always_precedes_test():
    dates = _dates(120)
    for train_idx, test_idx in validation.walk_forward_splits(
            dates, n_splits=4, embargo_days=7, min_train_days=30):
        assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min()


def test_embargo_gap_is_respected():
    dates = _dates(120)
    embargo = 7
    for train_idx, test_idx in validation.walk_forward_splits(
            dates, n_splits=4, embargo_days=embargo, min_train_days=30):
        gap = (dates.iloc[test_idx].min() - dates.iloc[train_idx].max()).days
        assert gap > embargo, f"embargo violated: only {gap}d gap"


def test_no_row_appears_in_both_sides():
    dates = _dates(120)
    for train_idx, test_idx in validation.walk_forward_splits(dates, n_splits=4):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_training_window_expands():
    dates = _dates(160)
    sizes = [len(tr) for tr, _ in validation.walk_forward_splits(
        dates, n_splits=4, min_train_days=30)]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]


def test_folds_cover_distinct_test_periods():
    dates = _dates(160)
    starts = [dates.iloc[te].min() for _, te in
              validation.walk_forward_splits(dates, n_splits=4, min_train_days=30)]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_too_little_history_yields_nothing():
    assert list(validation.walk_forward_splits(_dates(10), min_train_days=30)) == []


def test_describe_splits_shape():
    info = validation.describe_splits(_dates(160), n_splits=3, min_train_days=30)
    assert len(info) == 3
    for fold in info:
        assert fold["n_train"] > 0 and fold["n_test"] > 0
        assert fold["train_end"] < fold["test_start"]
