"""M7 tests — feature matrix, models, calibration. Data-gated (CI skips)."""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
FEATURES = REPO / "outputs" / "features_v1.parquet"
PRIMARY = REPO / "data" / "raw" / "cs2_all_tiers_games.csv"

pytest.importorskip("sklearn")

# M7 modules are the human's build; skip the whole module until they exist.
try:
    import cs2analytics.evaluation.calibration  # noqa: F401
    import cs2analytics.features.matrix  # noqa: F401

    _M7_READY = True
    _M7_REASON = ""
except ModuleNotFoundError as _e:
    _M7_READY = False
    _M7_REASON = f"M7 modules not implemented yet ({_e.name}) — the human's build"

pytestmark = [pytest.mark.skipif(not _M7_READY, reason=_M7_REASON)]

_needs_data = pytest.mark.skipif(
    not PRIMARY.exists(), reason="real Kaggle data not available (CI: git-ignored)"
)
_needs_features = pytest.mark.skipif(
    not FEATURES.exists(), reason="outputs/features_v1.parquet missing — run M6 §4"
)


@pytest.fixture(scope="module")
def fs():
    from cs2analytics.features.matrix import build_feature_matrix

    return build_feature_matrix(FEATURES)


# --- feature matrix contract ---------------------------------------------------


@_needs_data
@_needs_features
def test_matrix_no_nan(fs):
    assert not np.isnan(fs.X).any()


@_needs_data
@_needs_features
def test_time_split_respected(fs):
    train_dates = fs.dates[fs.split == "train"]
    test_dates = fs.dates[fs.split == "test"]
    assert train_dates.max() < fs.cutoff
    assert test_dates.min() >= fs.cutoff
    assert len(train_dates) > 500
    assert len(test_dates) > 200


@_needs_data
@_needs_features
def test_leakage_guard_raises():
    """A forbidden post-outcome column must raise, not silently train."""
    from cs2analytics.features.matrix import LeakageError, build_feature_matrix

    with pytest.raises(LeakageError):
        build_feature_matrix(FEATURES, extra_features=["games_played"])


# --- calibration module ----------------------------------------------------------


def test_ece_perfectly_calibrated_is_small():
    """Simulate data that IS calibrated: y ~ Bernoulli(p); ECE should be tiny."""
    from cs2analytics.evaluation.calibration import expected_calibration_error

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p, bins=10) < 0.02


def test_ece_miscalibrated_is_large():
    from cs2analytics.evaluation.calibration import expected_calibration_error

    y = np.ones(1000)
    p = np.full(1000, 0.5)
    assert expected_calibration_error(y, p, bins=10) > 0.4


def test_reliability_table_bins_sum_to_n():
    from cs2analytics.evaluation.calibration import reliability_table

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(size=1000)
    table = reliability_table(y, p, bins=10)
    assert table["n"].sum() == 1000
    assert {"bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"} <= set(table.columns)


def test_brier_decomposition_identity():
    from cs2analytics.evaluation.calibration import brier_decomposition, brier_score

    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 5000)
    p = rng.uniform(0.05, 0.95, 5000)
    d = brier_decomposition(y, p, bins=10)
    recon = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert abs(recon - brier_score(p, y)) < 0.01


# --- artifacts ------------------------------------------------------------------


@_needs_data
def test_money_chart_exists():
    fig = REPO / "outputs" / "fig_calibration.png"
    if not fig.exists():
        pytest.skip("outputs/fig_calibration.png not produced yet (M7 §4)")
    assert fig.stat().st_size > 30_000


@_needs_data
def test_leakage_demo_table():
    path = REPO / "outputs" / "m7_leakage_demo.csv"
    if not path.exists():
        pytest.skip("outputs/m7_leakage_demo.csv not produced yet (M7 §4)")
    import pandas as pd

    df = pd.read_csv(path)
    assert {"model", "split_mode", "logloss", "brier", "acc"} <= set(df.columns)
    train_modes = set(df["split_mode"])
    assert {"time", "random"} <= train_modes
    # the point of the demo: random split looks BETTER (lower logloss)
    for model in df["model"].unique():
        sub = df[df["model"] == model].set_index("split_mode")["logloss"]
        if {"time", "random"} <= set(sub.index):
            assert sub["random"] <= sub["time"] + 1e-9, (
                f"{model}: random split should not be worse than time split"
            )


@_needs_data
def test_model_comparison_table_complete():
    path = REPO / "outputs" / "m7_model_comparison.csv"
    if not path.exists():
        pytest.skip("outputs/m7_model_comparison.csv not produced yet (M7 §4)")
    import pandas as pd

    df = pd.read_csv(path)
    assert {"model", "logloss", "brier", "acc", "ece"} <= set(df.columns)
    assert {"constant_0.5", "elo_k32", "lr", "gbm"} <= set(df["model"])
    assert df["logloss"].max() < 0.6932
    lr = float(df.loc[df["model"] == "lr", "logloss"].iloc[0])
    elo = float(df.loc[df["model"] == "elo_k32", "logloss"].iloc[0])
    assert lr <= elo + 0.01, "LR should be at least competitive with Elo"
