"""M4 data-gated tests: walk-forward backtest + Bayesian ratings artifacts.

These need the real Kaggle data (git-ignored) so CI skips them; run locally.
"""

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
PRIMARY = REPO / "data" / "raw" / "cs2_all_tiers_games.csv"

_needs_data = pytest.mark.skipif(
    not PRIMARY.exists(), reason="real Kaggle data not available (CI: git-ignored)"
)


@pytest.fixture(scope="module")
def series():
    return pd.read_csv(REPO / "outputs" / "series_clean.csv")


@pytest.fixture(scope="module")
def backtest_results():
    """The human's report notebook must export this table."""
    path = REPO / "outputs" / "m4_backtest_results.csv"
    if not path.exists():
        pytest.fail("outputs/m4_backtest_results.csv missing — run m4_backtest_report.ipynb")
    return pd.read_csv(path)


@_needs_data
def test_backtest_results_table_complete(backtest_results):
    required = {"model", "split", "logloss", "brier", "acc"}
    assert required <= set(backtest_results.columns)
    assert {"elo_k32", "constant_0.5"} <= set(backtest_results["model"])
    assert "test" in set(backtest_results["split"])


@_needs_data
def test_elo_beats_constant_baseline(backtest_results):
    test = backtest_results[backtest_results["split"] == "test"]
    elo = test[test["model"] == "elo_k32"]["logloss"].iloc[0]
    const = test[test["model"] == "constant_0.5"]["logloss"].iloc[0]
    assert const == pytest.approx(0.6931, abs=0.001)
    assert elo < const, "Elo must beat the constant-0.5 baseline on logloss"


@_needs_data
def test_elo_test_logloss_in_plausible_band(backtest_results):
    """DATA-informed sanity: Elo on test should land ~0.66-0.69 logloss.
    Below 0.60 = leakage; above 0.72 = implementation bug."""
    test = backtest_results[backtest_results["split"] == "test"]
    elo = test[test["model"] == "elo_k32"]["logloss"].iloc[0]
    assert 0.60 <= elo <= 0.72, f"elo_k32 test logloss {elo:.3f} outside plausible band"


@_needs_data
def test_bayesian_ratings_artifact():
    if not PRIMARY.exists():
        pytest.skip("real Kaggle data not available (CI: git-ignored)")
    path = REPO / "outputs" / "bayesian_ratings.csv"
    if not path.exists():
        pytest.skip("outputs/bayesian_ratings.csv not produced yet — PyMC notebook pending (M4 §4)")
    br = pd.read_csv(path)
    assert {"team", "posterior_mean", "hdi_3", "hdi_97"} <= set(br.columns)
    assert len(br) >= 50
    assert (br["hdi_3"] < br["posterior_mean"]).all()
    assert (br["posterior_mean"] < br["hdi_97"]).all()
