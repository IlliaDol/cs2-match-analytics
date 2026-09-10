"""Module 3 (R inference) — mechanical verification of R outputs.

Checks the artifacts that r/01_wrangle.R, r/02_inference.R, r/03_plots.R must produce
per docs/SPEC_M3_r_inference.md. Run from repo root: pytest -q
"""

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
OUT = REPO / "outputs"
RSCRIPT = Path("C:/Program Files/R/R-4.6.1/bin/Rscript.exe")


def _run(script: str) -> None:
    """Run an R script from the repo root; fail loudly on R errors."""
    import subprocess

    if not RSCRIPT.exists():
        pytest.fail(f"Rscript not found at {RSCRIPT}")
    res = subprocess.run(
        [str(RSCRIPT), f"r/{script}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert res.returncode == 0, f"r/{script} failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}"


@pytest.fixture(scope="module")
def r_pipeline():
    """Run the full R pipeline once; yield the wrangled table."""
    _run("01_wrangle.R")
    _run("02_inference.R")
    return pd.read_csv(REPO / "outputs" / "series_clean.csv")


# --- 01_wrangle.R contract ----------------------------------------------------


def test_series_clean_exists_with_expected_columns(r_pipeline):
    required = {
        "match_id",
        "datetime",
        "team1",
        "team2",
        "winner",
        "t1_series_score",
        "t2_series_score",
        "games_played",
        "bestOf",
        "tier",
        "total_maps",
        "margin",
        "is_bo1",
        "swept",
    }
    assert required <= set(r_pipeline.columns)


def test_series_clean_row_count(r_pipeline):
    assert len(r_pipeline) == 9922


def test_series_clean_derived_columns_consistent(r_pipeline):
    assert (
        r_pipeline["total_maps"] == r_pipeline["t1_series_score"] + r_pipeline["t2_series_score"]
    ).all()
    assert (
        r_pipeline["margin"]
        == (r_pipeline["t1_series_score"] - r_pipeline["t2_series_score"]).abs()
    ).all()
    assert (r_pipeline["is_bo1"] == (r_pipeline["games_played"] == 1)).all()
    assert r_pipeline["total_maps"].notna().all()
    assert r_pipeline["margin"].notna().all()


def test_bo1_share_matches_data_md_quirk(r_pipeline):
    """DATA.md Quirk 2: Bo1 share ~20.4% (2020/9922)."""
    share = r_pipeline["is_bo1"].mean()
    assert 0.18 < share < 0.23, f"Bo1 share {share:.3f} outside expected band"


# --- 02_inference.R contract --------------------------------------------------


def test_inference_results_exist_with_three_tests():
    res = pd.read_csv(REPO / "outputs" / "inference_results.csv")
    assert {"test", "statistic", "p_value", "effect_size", "interpretation"} <= set(res.columns)
    assert len(res) == 3
    assert res["p_value"].between(0, 1).all()
    assert res["interpretation"].notna().all()
    assert (res["interpretation"].str.len() > 10).all()


def test_anova_p_value_plausible():
    res = pd.read_csv(REPO / "outputs" / "inference_results.csv")
    anova = res[res["test"].str.contains("anova", case=False, na=False)]
    assert len(anova) == 1
    # sanity: with n≈9900 the F test on 3 groups is almost surely significant
    assert float(anova["p_value"].iloc[0]) < 0.05


# --- 03_plots.R contract ------------------------------------------------------


def test_figures_render(tmp_path=None):
    _run("03_plots.R")
    for fig in ("fig_winshare_top15.png", "fig_margin_by_tier.png"):
        p = OUT / fig
        assert p.exists(), f"missing {fig}"
        assert p.stat().st_size > 20_000, f"{fig} suspiciously small ({p.stat().st_size} B)"
