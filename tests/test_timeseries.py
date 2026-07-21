"""to_series: raw price rows -> a clean, ascending, de-duplicated Series."""

import pandas as pd

from futmarket.timeseries import to_series


def test_empty_rows_give_empty_series():
    s = to_series([])
    assert len(s) == 0


def test_sorts_and_casts_to_float():
    s = to_series([
        {"timestamp": "2026-01-02T00:00:00Z", "price": "200"},
        {"timestamp": "2026-01-01T00:00:00Z", "price": 100},
    ])
    assert list(s) == [100.0, 200.0]                 # ascending by time
    assert s.index.is_monotonic_increasing
    assert str(s.dtype) == "float64"


def test_duplicate_timestamps_keep_last():
    s = to_series([
        {"timestamp": "2026-01-01T00:00:00Z", "price": 100},
        {"timestamp": "2026-01-01T00:00:00Z", "price": 150},
    ])
    assert len(s) == 1 and s.iloc[0] == 150.0
