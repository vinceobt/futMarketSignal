"""Training — the never-ending learning loop.

Two heads, both histogram gradient boosting (scikit-learn):

  forecaster   quantile regression at p10/p50/p90 -> a forward-return estimate
               with an honest confidence band, not a false point prediction.
  direction    binary classifier on the triple-barrier label -> "would a position
               opened here have hit its profit target before its stop?", trained
               on liquid cards only (rule #1: never learn from cards you can't sell).

Every run is scored by walk-forward validation with an embargo and compared to a
dumb baseline. That comparison is the gate: a model that can't beat "assume no
change" (forecaster) or "guess the base rate" (direction) is not worth trusting,
and the run is recorded as such rather than quietly shipped.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import db
from . import dataset, labels, validation

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
QUANTILES = (0.1, 0.5, 0.9)
LIQUID_TIERS = ("A", "B")
DEFAULT_HORIZON = 7
MIN_TRAIN_ROWS = 200


def _estimator(kind: str, **kwargs):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    common = dict(max_iter=300, learning_rate=0.06, max_depth=None,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    common.update(kwargs)
    if kind == "classifier":
        return HistGradientBoostingClassifier(**common)
    return HistGradientBoostingRegressor(**common)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _features(frame: pd.DataFrame) -> list[str]:
    return [c for c in dataset.FEATURE_COLUMNS if c in frame.columns]


def _pinball(y_true, y_pred, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def evaluate_forecaster(frame: pd.DataFrame, *, horizon: int,
                        n_splits: int = 4) -> dict:
    """Walk-forward MAE + band coverage vs a no-change baseline."""
    target = f"fwd_return_{horizon}d"
    cols = _features(frame)
    data = frame.dropna(subset=[target])
    if len(data) < MIN_TRAIN_ROWS:
        return {"folds": 0, "note": "insufficient labelled rows"}

    maes, base_maes, coverages, pinballs = [], [], [], []
    for train_idx, test_idx in validation.walk_forward_splits(
            data["date"], n_splits=n_splits, embargo_days=horizon):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        y_test = test[target].to_numpy()
        preds = {}
        for q in QUANTILES:
            model = _estimator("regressor", loss="quantile", quantile=q)
            model.fit(train[cols], train[target])
            preds[q] = model.predict(test[cols])
        maes.append(float(np.mean(np.abs(y_test - preds[0.5]))))
        # baseline: predict "no change" -- the honest null hypothesis
        base_maes.append(float(np.mean(np.abs(y_test))))
        coverages.append(float(np.mean((y_test >= preds[0.1]) & (y_test <= preds[0.9]))))
        pinballs.append(float(np.mean([_pinball(y_test, preds[q], q) for q in QUANTILES])))

    if not maes:
        return {"folds": 0, "note": "no usable folds"}
    mae, base = float(np.mean(maes)), float(np.mean(base_maes))
    return {
        "folds": len(maes),
        "mae": round(mae, 4),
        "baseline_mae": round(base, 4),
        "skill_vs_baseline_pct": round((1 - mae / base) * 100, 2) if base else None,
        "band_coverage_p10_p90": round(float(np.mean(coverages)), 4),
        "pinball": round(float(np.mean(pinballs)), 4),
        "beat_baseline": bool(mae < base),
    }


def evaluate_direction(frame: pd.DataFrame, *, horizon: int,
                       n_splits: int = 4, top_decile: float = 0.1) -> dict:
    """Walk-forward PR-AUC + precision@top-decile vs the base rate."""
    from sklearn.metrics import average_precision_score

    cols = _features(frame)
    data = frame.dropna(subset=["is_profitable"])
    if len(data) < MIN_TRAIN_ROWS:
        return {"folds": 0, "note": "insufficient labelled rows"}

    aps, bases, precisions = [], [], []
    for train_idx, test_idx in validation.walk_forward_splits(
            data["date"], n_splits=n_splits, embargo_days=horizon):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        if train["is_profitable"].nunique() < 2 or test["is_profitable"].nunique() < 2:
            continue
        model = _estimator("classifier")
        model.fit(train[cols], train["is_profitable"])
        proba = model.predict_proba(test[cols])[:, 1]
        y_test = test["is_profitable"].to_numpy()
        aps.append(float(average_precision_score(y_test, proba)))
        bases.append(float(np.mean(y_test)))          # base rate = guess prevalence
        k = max(1, int(len(proba) * top_decile))
        top = np.argsort(proba)[-k:]
        precisions.append(float(np.mean(y_test[top])))

    if not aps:
        return {"folds": 0, "note": "no usable folds"}
    ap, base = float(np.mean(aps)), float(np.mean(bases))
    return {
        "folds": len(aps),
        "avg_precision": round(ap, 4),
        "base_rate": round(base, 4),
        "lift_vs_base_rate": round(ap / base, 3) if base else None,
        "precision_at_top_decile": round(float(np.mean(precisions)), 4),
        "beat_baseline": bool(ap > base),
    }


def _fit_final(frame: pd.DataFrame, *, kind: str, horizon: int):
    """Refit on all labelled data — the artifact that actually gets used."""
    cols = _features(frame)
    if kind == "direction":
        data = frame.dropna(subset=["is_profitable"])
        model = _estimator("classifier")
        model.fit(data[cols], data["is_profitable"])
        return {"model": model, "features": cols}
    target = f"fwd_return_{horizon}d"
    data = frame.dropna(subset=[target])
    models = {}
    for q in QUANTILES:
        m = _estimator("regressor", loss="quantile", quantile=q)
        m.fit(data[cols], data[target])
        models[q] = m
    return {"models": models, "features": cols}


def _liquid_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "liq_tier" not in frame.columns:
        return frame
    liquid = frame[frame["liq_tier"].isin(LIQUID_TIERS)]
    return liquid if not liquid.empty else frame


def train(conn, *, source: str = "futgg", title: str = "fc26",
          horizon: int = DEFAULT_HORIZON, target_pct: float = 25.0,
          stop_pct: float = 8.0, tax_rate: float = 0.05, n_splits: int = 4,
          model_dir: Path | None = None, frame: pd.DataFrame | None = None) -> dict:
    """Build features, label, validate walk-forward, fit, and register the run."""
    import joblib

    model_dir = Path(model_dir or MODEL_DIR)
    if frame is None:
        frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return {"error": "no data — run backfill-history first"}

    frame = labels.add_labels(frame, horizon_days=horizon, target_pct=target_pct,
                              stop_pct=stop_pct, tax_rate=tax_rate)
    forecast_frame = frame
    direction_frame = _liquid_only(frame)   # never learn from unsellable cards

    results = {
        "horizon_days": horizon,
        "rows": int(len(frame)),
        "cards": int(frame["player_id"].nunique()),
        "forecast": evaluate_forecaster(forecast_frame, horizon=horizon,
                                        n_splits=n_splits),
        "direction": evaluate_direction(direction_frame, horizon=horizon,
                                        n_splits=n_splits),
        "runs": {},
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_sha()

    for kind, source_frame in (("forecast", forecast_frame),
                               ("direction", direction_frame)):
        metrics = results[kind]
        if not metrics.get("folds"):
            continue
        artifact = _fit_final(source_frame, kind=kind, horizon=horizon)
        path = model_dir / f"{kind}_{horizon}d_{stamp}.joblib"
        joblib.dump(artifact, path)
        run_id = db.create_model_run(
            conn, kind=kind, level="card", title=title, horizon_h=horizon * 24,
            n_samples=int(len(source_frame)), metrics_json=json.dumps(metrics),
            artifact_path=str(path),
            feature_list_json=json.dumps(_features(source_frame)), git_sha=sha)
        results["runs"][kind] = {"run_id": run_id, "artifact": str(path)}

    logger.info("training done: %s", {k: results[k] for k in ("forecast", "direction")})
    return results
