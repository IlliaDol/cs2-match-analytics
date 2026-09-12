"""M11 §1 — training artifact producer (python -m cs2analytics.models.train).

Fits the M7 LR pipeline on TRAIN only, evaluates on TEST, saves model + metadata
+ final Elo ratings to artifacts/, and logs one MLflow run per invocation.
Deterministic: fixed seeds; features.json sorted keys -> byte-identical re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from cs2analytics.evaluation.calibration import expected_calibration_error
from cs2analytics.evaluation.metrics import accuracy_from_probs, brier_score, log_loss
from cs2analytics.features.elo import run_elo_backtest
from cs2analytics.features.matrix import build_feature_matrix
from cs2analytics.models.logistic import RANDOM_STATE, fit_logistic, predict_proba

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"
FEATURES_PATH = REPO / "outputs" / "features_v1.parquet"
SERIES_PATH = REPO / "outputs" / "series_clean.csv"

MODEL_VERSION = "lr-2026-09-11"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    fs = build_feature_matrix(FEATURES_PATH)
    X_tr, y_tr = fs.X[fs.train_mask], fs.y[fs.train_mask]
    X_te, y_te = fs.X[fs.test_mask], fs.y[fs.test_mask]

    model = fit_logistic(X_tr, y_tr)
    p_te = predict_proba(model, X_te)
    metrics = {
        "logloss": log_loss(p_te, y_te),
        "brier": brier_score(p_te, y_te),
        "acc": accuracy_from_probs(p_te, y_te),
        "ece": expected_calibration_error(y_te, p_te),
        "n_test": int(len(y_te)),
        "n_train": int(len(y_tr)),
    }
    print(
        f"model={MODEL_VERSION} | logloss={metrics['logloss']:.4f} brier={metrics['brier']:.4f} "
        f"acc={metrics['acc']:.4f} ece={metrics['ece']:.4f} n_test={metrics['n_test']}"
    )

    joblib.dump(model, ARTIFACTS / "model.pkl")

    features_meta = {
        "feature_names": fs.feature_names,
        "cutoff": str(fs.cutoff),
        "model_version": MODEL_VERSION,
        "metrics": metrics,
        "random_state": RANDOM_STATE,
    }
    (ARTIFACTS / "features.json").write_text(
        json.dumps(features_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    elo_ratings = _elo_ratings_json()
    (ARTIFACTS / "elo_ratings.json").write_text(
        json.dumps(elo_ratings, indent=2, sort_keys=True), encoding="utf-8"
    )

    _mlflow_log(model, metrics, fs)
    print(f"artifacts written to {ARTIFACTS}")


def _elo_ratings_json() -> dict[str, float]:
    import pandas as pd

    series = pd.read_csv(SERIES_PATH)
    series["datetime"] = pd.to_datetime(series["datetime"], utc=True, format="ISO8601")
    series = series.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    bt = run_elo_backtest(series, k=32.0)
    ratings: dict[str, float] = {}
    # walk once more applying post-match updates to land on final ratings
    for row in bt.itertuples(index=False):
        ra, rb = row.elo_t1_pre, row.elo_t2_pre
        s = row.result
        ratings[row.team1] = ra + 32.0 * (s - 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0)))
        ratings[row.team2] = rb + 32.0 * ((1.0 - s) - 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0)))
    return {team: round(r, 4) for team, r in ratings.items()}


def _mlflow_log(model, metrics: dict, fs) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{REPO / 'mlflow.db'}")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params(
                {
                    "model_type": "logistic_regression",
                    "cutoff": str(fs.cutoff),
                    "n_train": metrics["n_train"],
                    "random_state": RANDOM_STATE,
                }
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(model, artifact_path="model")
    except Exception as exc:
        print(f"[train] mlflow logging skipped ({exc})")


if __name__ == "__main__":
    main()
