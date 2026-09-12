"""Per-regime calibration contract tests (P0.1 improvement).

These do not need the private data: they pin the group-splitting behaviour of
`calibration_by_group` on synthetic labels so CI stays green without data/raw.
"""

from __future__ import annotations

import numpy as np

from cs2analytics.evaluation.calibration import calibration_by_group


def test_groups_split_and_small_groups_dropped():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.3, 0.7, n)
    # 120 "a", 60 "b", 20 "c" (below the min_samples=30 cutoff)
    groups = np.array(["a"] * 120 + ["b"] * 60 + ["c"] * 20)

    out = calibration_by_group(y, p, groups)

    assert set(out["group"]) == {"a", "b"}, "groups with <30 samples must be dropped"
    assert int(out.loc[out["group"] == "a", "n"].iloc[0]) == 120
    assert int(out.loc[out["group"] == "b", "n"].iloc[0]) == 60
    assert out["logloss"].notna().all()
    assert out["ece"].between(0.0, 1.0).all()


def test_tuple_labels_joined():
    rng = np.random.default_rng(1)
    n = 100
    y = rng.integers(0, 2, n)
    p = np.full(n, 0.5)
    groups = [("bo1", "t1"), ("bo3", "t1")] * 50

    out = calibration_by_group(y, p, groups)

    assert set(out["group"]) == {"bo1/t1", "bo3/t1"}
    # mean_pred of an all-0.5 forecast is exactly 0.5 while obs_rate is ~y.mean()
    assert np.allclose(out["mean_pred"], 0.5)
