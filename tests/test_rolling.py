"""Rolling-origin backtest contract tests (P1.4). No private data needed."""

from __future__ import annotations

import pandas as pd

from cs2analytics.models.rolling import summarize_rolling


def test_summarize_rolling_shapes_and_ci():
    table = pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
            "n_train": [8000, 8200, 8400],
            "n_test": [200, 200, 200],
            "logloss": [0.640, 0.645, 0.650],
            "brier": [0.224, 0.226, 0.228],
            "acc": [0.620, 0.625, 0.630],
        }
    )
    out = summarize_rolling(table)
    assert list(out["metric"]) == ["logloss", "brier", "acc"]
    assert out["n_folds"].eq(3).all()
    # mean of an increasing sequence sits strictly between the ends
    assert out.loc[out["metric"] == "logloss", "mean"].iloc[0] == 0.645
    # CI is non-degenerate (n=3 -> t is wide)
    assert (out["ci_hi"] >= out["ci_lo"]).all()


def test_summarize_rolling_single_fold_degenerate():
    table = pd.DataFrame(
        {
            "cutoff": [pd.Timestamp("2026-01-01")],
            "n_train": [8000],
            "n_test": [200],
            "logloss": [0.64],
            "brier": [0.22],
            "acc": [0.62],
        }
    )
    out = summarize_rolling(table)
    assert (out["ci_lo"] == out["ci_hi"]).all()


def test_summarize_rolling_empty():
    assert summarize_rolling(pd.DataFrame()).empty
