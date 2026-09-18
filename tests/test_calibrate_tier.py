"""Per-tier calibrator contract tests (D.1 / v1.1). Synthetic only, no private data."""

from __future__ import annotations

import numpy as np
import pytest

from cs2analytics.models.calibrate_tier import PerTierCalibrator


def _overconfident(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    # model is over-confident: pushes everything toward the extremes
    p_model = np.clip(true_p + np.where(true_p > 0.5, 0.2, -0.2), 0, 1)
    return p_model, y


def _well_calibrated(n: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    p_model = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p_model).astype(float)
    return p_model, y


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_overconfident_tier_gets_recalibrated(method: str):
    p_model, y = _overconfident()
    tiers = np.array(["tier3"] * len(y))

    cal = PerTierCalibrator(method=method).fit(p_model, y, tiers)
    fixed = cal.transform(p_model, tiers)

    # after recalibration the mean prediction tracks the observed rate
    assert abs(fixed.mean() - y.mean()) < abs(p_model.mean() - y.mean())


def test_unknown_tier_passes_through():
    cal = PerTierCalibrator(min_n=1).fit(
        np.array([0.3, 0.7]), np.array([0, 1]), np.array(["t1", "t1"])
    )
    out = cal.transform(np.array([0.42, 0.55]), np.array(["tier_unknown", "tier_unknown"]))
    # no map for that tier -> values pass through
    assert np.allclose(out, [0.42, 0.55])


def test_too_few_train_rows_is_skipped_not_overfit():
    p_model, y = _overconfident(n=40, seed=1)
    tiers = np.array(["tier3"] * len(y))

    cal = PerTierCalibrator(method="platt", min_n=100).fit(p_model, y, tiers)

    assert cal.maps == {}
    assert cal.skipped == [("tier3", 40, "too-few-rows")]
    # with no map the transform is the identity
    assert np.allclose(cal.transform(p_model, tiers), p_model)


def test_single_class_tier_is_skipped():
    p_model, y = _overconfident(n=500, seed=2)
    y = np.ones_like(y)                      # every series won: no map is learnable

    cal = PerTierCalibrator(method="platt", min_n=10).fit(p_model, y, np.array(["t1"] * len(y)))

    assert cal.maps == {} and cal.skipped == [("t1", 500, "single-class")]


def test_selection_calibrates_only_the_miscalibrated_tier():
    """`select="overconfident"` must use TRAIN evidence only: the over-confident
    tier gets a map, the well-calibrated one is left alone."""
    p_bad, y_bad = _overconfident(n=1500, seed=3)
    p_ok, y_ok = _well_calibrated(n=1500, seed=4)
    p = np.concatenate([p_bad, p_ok])
    y = np.concatenate([y_bad, y_ok])
    tiers = np.array(["tier3"] * len(y_bad) + ["tier1"] * len(y_ok))

    cal = PerTierCalibrator(select="overconfident", min_n=100).fit(p, y, tiers)

    assert sorted(cal.maps) == ["tier3"]
    assert cal.skipped == [("tier1", 1500, "well-calibrated-on-train")]
    fixed = cal.transform(p, tiers)
    # the well-calibrated tier is untouched, the over-confident one is corrected
    assert np.allclose(fixed[len(y_bad):], p_ok)
    assert abs(fixed[: len(y_bad)].mean() - y_bad.mean()) < abs(p_bad.mean() - y_bad.mean())


def test_invalid_method_or_select_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        PerTierCalibrator(method="beta")
    with pytest.raises(ValueError, match="select must be one of"):
        PerTierCalibrator(select="sometimes")
