"""Training: robustness of the fit/predict contract, and the honest baselines."""

import numpy as np
import pandas as pd
import pytest

from futmarket.ml import evaluate, train


def _frame(days=150, cards=40, seed=0, constant_col=True):
    """A small but realistic (card, day) matrix with labels already attached."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days)
    rows = []
    for c in range(cards):
        price = 10_000.0
        for d in dates:
            price *= 1 + rng.normal(0, 0.03)
            rows.append({
                "player_id": f"c{c}", "date": d.strftime("%Y-%m-%d"),
                "price": price, "liq_tier": "A", "rating": 84,
                "z_score": rng.normal(0, 1),
                "dist_to_floor_pct": abs(rng.normal(5, 4)),
                "dist_to_ceiling_pct": abs(rng.normal(20, 8)),
                "ret_1d": rng.normal(0, 4), "ret_7d": rng.normal(0, 10),
                "vol_14d": 4.0, "liq_score": 0.9, "day_of_week": d.dayofweek,
                "days_since_card_release": rng.integers(0, 60),
                # The column that broke a real training run: constant in the
                # slice, which scikit-learn's binner cannot fit.
                "is_special": 0.0 if constant_col else float(c % 2),
            })
    frame = pd.DataFrame(rows)
    for h in (1, 3, 7):
        frame[f"market_ret_{h}d"] = frame.groupby("date")["ret_1d"].transform("median")
    return frame


def test_a_constant_feature_does_not_crash_training(tmp_path, conn):
    """The real failure: the first walk-forward fold is the start of the season,
    when every liquid card is a base gold and `is_special` is 0 for all of them.
    scikit-learn's binner then dies inside numpy with "window shape cannot be
    larger than input array shape", which nobody would connect to their data.
    """
    res = train.train(conn, frame=_frame(), horizon=14, horizons=(7, 14),
                      n_splits=2, model_dir=tmp_path, record_predictions=True)
    assert "error" not in res
    assert res["runs"], "no models were registered"


def test_a_fitted_model_can_always_be_used_to_predict(tmp_path, conn):
    """The follow-on bug: dropping a constant column at fit time means the
    estimator rejects it coming back, so every prediction path must use the
    columns its own horizon was fitted on — not the full feature list."""
    import joblib

    res = train.train(conn, frame=_frame(), horizon=14, horizons=(7, 14),
                      n_splits=2, model_dir=tmp_path, record_predictions=False)
    latest = _frame(days=3, cards=5, seed=9)

    for kind in ("excess", "clears"):
        run = res["runs"].get(kind)
        if run is None:
            continue
        artifact = joblib.load(run["artifact"])
        assert artifact["feature_sets"], f"{kind} recorded no per-horizon columns"
        for h, model in artifact["models"].items():
            cols = artifact["feature_sets"][h]
            assert "is_special" not in cols          # constant, so dropped
            for c in [c for c in cols if c not in latest.columns]:
                latest[c] = np.nan
            model.predict(latest[cols])              # must not raise


def test_predictions_are_stored_for_calibration(tmp_path, conn):
    """The `predictions` table sat empty for months, so there was never any way
    to ask "what did the model say, and was it right?"."""
    from futmarket import db as futdb

    train.train(conn, frame=_frame(), horizon=14, horizons=(14,), n_splits=2,
                model_dir=tmp_path, record_predictions=True)
    assert futdb.latest_predictions(conn, kind="excess")


def test_every_traded_gate_gets_its_own_payoff_profile(tmp_path, conn):
    """A deep dip and a release crash pay very differently; one blended profile
    would price both wrong."""
    res = train.train(conn, frame=_frame(constant_col=False), horizon=14,
                      horizons=(7, 14), n_splits=2, model_dir=tmp_path,
                      record_predictions=False)
    assert set(res["payoffs"]) == set(train.TRADED_GATES)


def test_the_forecaster_baseline_is_the_market_not_zero():
    """"Assume no change" was nearly free to beat — the old head managed 0.42%
    over it and shipped. The honest null is "this card moves with the market"."""
    frame = _frame(days=120, cards=20)
    frame = evaluate.add_forward_returns(frame, horizons=(14,))
    frame = evaluate.add_benchmark_returns(frame, horizons=(14,))
    res = train.evaluate_excess(frame, horizon=14, n_splits=2)
    # the baseline is the mean absolute EXCESS return, i.e. error of predicting 0
    expected = frame["excess_return_14d"].abs().mean()
    assert res["baseline_mae"] == pytest.approx(expected, rel=0.25)
