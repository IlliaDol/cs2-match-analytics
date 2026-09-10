"""M13 tests — forecasting outputs + DiD artifacts. Skips until produced."""

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
OUT = REPO / "outputs"
NB_DID = REPO / "notebooks" / "07_break_effect_did.ipynb"


def test_rating_forecast_contract():
    path = OUT / "rating_forecast.csv"
    if not path.exists():
        pytest.skip("outputs/rating_forecast.csv not produced yet (M13 §1)")
    df = pd.read_csv(path)
    assert {"team", "month", "actual", "fitted", "forecast", "lo80", "hi80"} <= set(df.columns)
    assert df["team"].nunique() >= 10
    assert (df["lo80"] < df["forecast"]).all()
    assert (df["forecast"] < df["hi80"]).all()


def test_forecast_metrics_include_naive_baseline():
    path = OUT / "rating_forecast_metrics.csv"
    if not path.exists():
        pytest.skip("outputs/rating_forecast_metrics.csv not produced yet (M13 §1.5)")
    df = pd.read_csv(path)
    assert {"team", "mape", "n_months"} <= set(df.columns)
    assert (df["mape"] > 0).all()
    assert "naive" in set(df["team"].str.lower()), "the random-walk baseline must be reported"


def test_forecast_figure():
    fig = OUT / "fig_rating_forecast.png"
    if not fig.exists():
        pytest.skip("outputs/fig_rating_forecast.png not produced yet (M13 §1.4)")
    assert fig.stat().st_size > 20_000


def test_did_results_contract():
    path = OUT / "m13_did_results.csv"
    if not path.exists():
        pytest.skip("outputs/m13_did_results.csv not produced yet (M13 §2)")
    df = pd.read_csv(path)
    assert {"specification", "estimate", "se", "ci_lo", "ci_hi", "n"} <= set(df.columns)
    assert len(df) >= 2
    assert (df["ci_lo"] <= df["estimate"]).all()
    assert (df["estimate"] <= df["ci_hi"]).all()
    assert (df["n"] > 100).all()


def test_did_notebook_states_assumption():
    if not NB_DID.exists():
        pytest.skip("notebooks/07_break_effect_did.ipynb not produced yet (M13 §2)")
    import json

    raw = json.loads(NB_DID.read_text(encoding="utf-8"))
    text = " ".join("".join(c.get("source", [])) for c in raw.get("cells", [])).lower()
    assert "parallel trends" in text, "the identifying assumption must be stated explicitly"
