"""Training — the never-ending learning loop.

Two heads, both histogram gradient boosting (scikit-learn), and both trained at
**every candidate holding period** (3/5/7/10/14 days) because how long to hold is
part of the decision:

  excess    regression on ``excess_return_{h}d`` -- how much this card will beat
            the market by. Not its absolute move: the median tradeable card
            doesn't move at all over a fortnight, so a model predicting absolute
            returns spends its capacity learning that nothing happens.
  clears    binary on ``clears_cost_{h}d`` -- "would a position opened here have
            beaten the market *and* cleared the full round trip?" Trained on
            liquid cards only (rule #1: never learn from cards you can't sell).

**Validation is on the traded population.** This is the change that matters most.
The old run reported metrics over all 771k rows and pronounced itself healthy at
2.4x lift -- while the strategy, which only ever trades the few hundred rows a day
that pass the entry gate, went 0-for-26 on its highest-confidence picks. A model
can be excellent on the average card and worse than useless on the narrow slice
it is actually asked about. Every run is now scored twice: once overall, once on
the gated slice, and it is the gated number that decides whether it ships.

The baseline is likewise honest. "Beat the base rate" is nearly free. The
forecaster must beat *"assume this card moves exactly with the market"* -- which
is what the old head could not do: it managed 0.42% skill over "assume no
change", i.e. none at all.
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
from . import dataset, evaluate, labels, validation

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
LIQUID_TIERS = ("A", "B")
DEFAULT_HORIZON = 14          # the horizon the barriers and the scorecard default to
# 21 days is in here for the release trade, whose edge keeps growing that far out.
HORIZONS = (*evaluate.HORIZONS, 21)
MIN_TRAIN_ROWS = 200
# The gates the strategies actually trade. Validation on anything wider flatters:
# the model is only ever asked about cards that pass one of these.
TRADED_GATES = ("relval_v1", "release")
# The gate whose precision is the headline in the training report.
TRADED_GATE = TRADED_GATES[0]


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


def _usable_features(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop columns that are constant in this slice.

    scikit-learn's histogram binner cannot fit a feature with a single distinct
    value -- it fails deep inside numpy with "window shape cannot be larger than
    input array shape", which is not a message anyone will connect to their data.
    And it happens for real: the first walk-forward fold is the start of the
    season, when every liquid card is a base gold and ``is_special`` is 0 for all
    of them. Fold-by-fold, so a column that is useless early can still be used
    later once it varies.
    """
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def evaluate_excess(frame: pd.DataFrame, *, horizon: int, n_splits: int = 4) -> dict:
    """Walk-forward MAE against "this card moves with the market", overall and
    on the slice the strategy actually trades."""
    target = f"excess_return_{horizon}d"
    cols = _features(frame)
    data = frame.dropna(subset=[target])
    if len(data) < MIN_TRAIN_ROWS:
        return {"folds": 0, "note": "insufficient labelled rows"}

    maes, base_maes, gated_maes, gated_base = [], [], [], []
    for train_idx, test_idx in validation.walk_forward_splits(
            data["date"], n_splits=n_splits, embargo_days=horizon):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        fold_cols = _usable_features(train, cols)
        model = _estimator("regressor")
        model.fit(train[fold_cols], train[target])
        pred = model.predict(test[fold_cols])
        y = test[target].to_numpy()
        maes.append(float(np.mean(np.abs(y - pred))))
        # The null hypothesis worth beating: no view at all -- this card does
        # whatever the market does, so its excess return is zero.
        base_maes.append(float(np.mean(np.abs(y))))

        gate = evaluate.gate_mask(test, TRADED_GATE).to_numpy()
        if gate.sum() >= 20:
            gated_maes.append(float(np.mean(np.abs(y[gate] - pred[gate]))))
            gated_base.append(float(np.mean(np.abs(y[gate]))))

    if not maes:
        return {"folds": 0, "note": "no usable folds"}
    mae, base = float(np.mean(maes)), float(np.mean(base_maes))
    out = {
        "folds": len(maes),
        "mae": round(mae, 4),
        "baseline_mae": round(base, 4),
        "skill_vs_market_pct": round((1 - mae / base) * 100, 2) if base else None,
        "beat_baseline": bool(mae < base),
    }
    if gated_maes:
        g_mae, g_base = float(np.mean(gated_maes)), float(np.mean(gated_base))
        out.update({
            "gated_mae": round(g_mae, 4),
            "gated_baseline_mae": round(g_base, 4),
            "gated_skill_pct": round((1 - g_mae / g_base) * 100, 2) if g_base else None,
            "gated_folds": len(gated_maes),
        })
    return out


def evaluate_clears(frame: pd.DataFrame, *, horizon: int, n_splits: int = 4,
                    top_decile: float = 0.1) -> dict:
    """Walk-forward PR-AUC and precision — reported on the traded slice too.

    ``gated_precision`` is the number to read: of the cards this model would
    actually have recommended, what share went on to beat the market and clear
    costs. ``gated_base_rate`` is what you'd have got picking at random from the
    same gate, so the difference is what the model contributes. The old metrics
    could not see this and reported a healthy model that was inverted in
    production.
    """
    from sklearn.metrics import average_precision_score

    target = f"clears_cost_{horizon}d"
    cols = _features(frame)
    data = frame.dropna(subset=[target])
    if len(data) < MIN_TRAIN_ROWS:
        return {"folds": 0, "note": "insufficient labelled rows"}

    aps, bases, precisions = [], [], []
    per_gate: dict[str, dict[str, list]] = {
        g: {"prec": [], "base": []} for g in TRADED_GATES}
    for train_idx, test_idx in validation.walk_forward_splits(
            data["date"], n_splits=n_splits, embargo_days=horizon):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        fold_cols = _usable_features(train, cols)
        model = _estimator("classifier")
        model.fit(train[fold_cols], train[target])
        proba = model.predict_proba(test[fold_cols])[:, 1]
        y = test[target].to_numpy()
        aps.append(float(average_precision_score(y, proba)))
        bases.append(float(np.mean(y)))
        k = max(1, int(len(proba) * top_decile))
        precisions.append(float(np.mean(y[np.argsort(proba)[-k:]])))

        # The only slices that matter: cards an entry gate would have offered,
        # judged at the probability the strategy actually buys at. Each gate is
        # scored separately -- the model can be useful on one trade and useless
        # on the other, and a blended number would hide that.
        for name in TRADED_GATES:
            gate = evaluate.gate_mask(test, name).to_numpy()
            picked = gate & (proba >= evaluate.MIN_CLEARS_PROB)
            if picked.sum() >= 20 and gate.sum() >= 20:
                per_gate[name]["prec"].append(float(np.mean(y[picked])))
                per_gate[name]["base"].append(float(np.mean(y[gate])))

    if not aps:
        return {"folds": 0, "note": "no usable folds"}
    ap, base = float(np.mean(aps)), float(np.mean(bases))
    out = {
        "folds": len(aps),
        "avg_precision": round(ap, 4),
        "base_rate": round(base, 4),
        "lift_vs_base_rate": round(ap / base, 3) if base else None,
        "precision_at_top_decile": round(float(np.mean(precisions)), 4),
        "beat_baseline": bool(ap > base),
        "gates": {},
    }
    for name, acc in per_gate.items():
        if not acc["prec"]:
            continue
        g_prec, g_base = float(np.mean(acc["prec"])), float(np.mean(acc["base"]))
        out["gates"][name] = {
            "precision": round(g_prec, 4),
            "base_rate": round(g_base, 4),
            "lift": round(g_prec / g_base, 3) if g_base else None,
            "folds": len(acc["prec"]),
            # What decides whether this run is worth shipping: the model must add
            # something on the cards it will actually be asked about.
            "beats_baseline": bool(g_prec > g_base),
        }
    # Back-compat headline for callers that want a single number.
    headline = out["gates"].get(TRADED_GATE)
    if headline:
        out.update({"gated_precision": headline["precision"],
                    "gated_base_rate": headline["base_rate"],
                    "gated_lift": headline["lift"],
                    "gated_folds": headline["folds"],
                    "beats_gate_baseline": headline["beats_baseline"]})
    return out


def _fit_final(frame: pd.DataFrame, *, kind: str, horizons) -> dict:
    """Refit on all labelled data — the artifacts that actually get used."""
    all_cols = _features(frame)
    models, feature_sets = {}, {}
    for h in horizons:
        target = (f"excess_return_{h}d" if kind == "excess" else f"clears_cost_{h}d")
        data = frame.dropna(subset=[target])
        if len(data) < MIN_TRAIN_ROWS:
            continue
        if kind == "clears":
            if data[target].nunique() < 2:
                continue
            model = _estimator("classifier")
        else:
            model = _estimator("regressor")
        cols = _usable_features(data, all_cols)
        model.fit(data[cols], data[target])
        models[h] = model
        # Per horizon, because a column can be constant for one label and not
        # another. Prediction must use exactly the columns the model saw.
        feature_sets[h] = cols
    return {"models": models, "features": all_cols, "feature_sets": feature_sets,
            "kind": kind, "horizons": sorted(models)}


def _liquid_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "liq_tier" not in frame.columns:
        return frame
    liquid = frame[frame["liq_tier"].isin(LIQUID_TIERS)]
    return liquid if not liquid.empty else frame


def _record_predictions(conn, frame: pd.DataFrame, artifact: dict, *,
                        run_id: int, horizon: int, title: str) -> int:
    """Store today's excess-return predictions so calibration can be checked.

    The `predictions` table existed for months and held zero rows, which meant
    there was never any way to ask "what did the model say, and was it right?"
    other than through the handful of cards that became picks.
    """
    model = artifact["models"].get(horizon)
    if model is None or frame.empty:
        return 0
    latest = (frame.sort_values("date").groupby("player_id", observed=True)
              .tail(1).copy())
    for c in [c for c in artifact["features"] if c not in latest.columns]:
        latest[c] = np.nan
    # Exactly the columns this horizon's model was fitted on — constant features
    # were dropped at fit time and the estimator rejects them coming back.
    cols = (artifact.get("feature_sets") or {}).get(horizon) or artifact["features"]
    predicted = model.predict(latest[cols])
    now = datetime.now(timezone.utc)
    saved = 0
    for pid, value in zip(latest["player_id"].astype(str), predicted):
        db.insert_prediction(conn, subject_id=pid, level="card", title=title,
                             kind="excess", horizon_h=horizon * 24, at=now,
                             run_id=run_id, yhat=float(value))
        saved += 1
    conn.commit()
    return saved


def train(conn, *, source: str = "futgg", title: str = "fc26",
          horizon: int = DEFAULT_HORIZON, horizons=HORIZONS,
          stop_buffer_pct: float = 3.0, stop_min_pct: float = 15.0,
          stop_max_pct: float = 30.0, tax_rate: float = 0.05,
          sell_slippage_pct: float = 2.0, buy_premium_pct: float = 0.0,
          n_splits: int = 4, model_dir: Path | None = None,
          frame: pd.DataFrame | None = None,
          record_predictions: bool = True) -> dict:
    """Build features, label at every horizon, validate walk-forward, fit, register."""
    import joblib

    model_dir = Path(model_dir or MODEL_DIR)
    if frame is None:
        frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return {"error": "no data — run backfill-history first"}

    frame = labels.add_labels(frame, horizon_days=horizon, horizons=horizons,
                              stop_buffer_pct=stop_buffer_pct,
                              stop_min_pct=stop_min_pct, stop_max_pct=stop_max_pct,
                              tax_rate=tax_rate, sell_slippage_pct=sell_slippage_pct,
                              buy_premium_pct=buy_premium_pct,
                              # No head trains on the triple barrier any more, and
                              # walking it is ~28M Python steps on the full matrix.
                              with_barrier=False)
    # Never learn from cards you can't sell -- for either head. The old code
    # trained the forecaster on everything, including tier-C cards whose "price"
    # is one stale listing, which is where most of its apparent skill came from.
    liquid = _liquid_only(frame)

    results = {
        "horizons": list(horizons),
        "primary_horizon": horizon,
        "rows": int(len(frame)),
        "liquid_rows": int(len(liquid)),
        "cards": int(frame["player_id"].nunique()),
        "excess": {h: evaluate_excess(liquid, horizon=h, n_splits=n_splits)
                   for h in horizons},
        "clears": {h: evaluate_clears(liquid, horizon=h, n_splits=n_splits)
                   for h in horizons},
        "runs": {},
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_sha()

    # What each gated trade has actually paid when it worked and cost when it
    # didn't. Shipped inside the classifier artifact because `picks` needs both
    # halves of the expected value and only one of them comes from a model --
    # see picks._choose_horizon for why the regressor's magnitudes are not used.
    # One profile per gate: a deep dip and a release crash pay very differently,
    # and blending them would price both wrong.
    payoffs = {
        gate: evaluate.payoff_profile(
            liquid, gate=gate, horizons=horizons, tax_rate=tax_rate,
            sell_slippage_pct=sell_slippage_pct, buy_premium_pct=buy_premium_pct,
            n_splits=n_splits)
        for gate in TRADED_GATES
    }
    results["payoffs"] = payoffs

    for kind in ("excess", "clears"):
        artifact = _fit_final(liquid, kind=kind, horizons=horizons)
        if not artifact["models"]:
            logger.warning("no %s models could be fitted", kind)
            continue
        if kind == "clears":
            artifact["payoffs"] = payoffs
            artifact["gates"] = list(TRADED_GATES)
        path = model_dir / f"{kind}_{stamp}.joblib"
        joblib.dump(artifact, path)
        run_id = db.create_model_run(
            conn, kind=kind, level="card", title=title, horizon_h=horizon * 24,
            n_samples=int(len(liquid)),
            metrics_json=json.dumps(results[kind], default=str),
            artifact_path=str(path),
            feature_list_json=json.dumps(artifact["features"]), git_sha=sha)
        results["runs"][kind] = {"run_id": run_id, "artifact": str(path),
                                 "horizons": artifact["horizons"]}
        if kind == "excess" and record_predictions:
            results["predictions_saved"] = _record_predictions(
                conn, liquid, artifact, run_id=run_id, horizon=horizon, title=title)

    logger.info("training done: excess=%s clears=%s",
                results["excess"].get(horizon), results["clears"].get(horizon))
    return results
