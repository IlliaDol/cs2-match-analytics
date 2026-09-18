"""Fit per-tier calibration on train, evaluate on test — writes
outputs/m11_tier_calibration.csv (one block of rows per method). From repo root:

    .venv/Scripts/python.exe scripts/build_tier_calibration.py

Methods compared on the same test split:
    none      the uncalibrated model (baseline; every row carries the same
              ece_before/logloss_before, so a method row is self-contained)
    isotonic  per-tier isotonic — the variant docs/DECISIONS.md rejects
    platt     per-tier Platt scaling — the v1.1 candidate

The verdict is reported in README (per-tier table + conclusion) and recorded in
docs/DECISIONS.md; nothing here is enabled in serving by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cs2analytics.evaluation.calibration import expected_calibration_error
from cs2analytics.evaluation.metrics import log_loss
from cs2analytics.features.matrix import build_feature_matrix
from cs2analytics.models.calibrate_tier import PerTierCalibrator
from cs2analytics.models.logistic import fit_logistic, predict_proba

REPO = Path(__file__).resolve().parents[1]

ROSTER_FEATURES = ["roster_stability_diff", "standin_diff"]
METHODS = (
    ("none", {}),
    ("isotonic", {"method": "isotonic"}),
    ("platt", {"method": "platt"}),
    ("platt-sel", {"method": "platt", "select": "overconfident"}),
)


def main() -> None:
    fs = build_feature_matrix(
        REPO / "outputs" / "features_v1.parquet", extra_features=ROSTER_FEATURES
    )
    # tier labels per row live in the feature store
    fv = pd.read_parquet(REPO / "outputs" / "features_v1.parquet")
    fv["datetime"] = pd.to_datetime(fv["datetime"], utc=True)

    model = fit_logistic(fs.X[fs.train_mask], fs.y[fs.train_mask])
    p_tr = predict_proba(model, fs.X[fs.train_mask])
    p_te = predict_proba(model, fs.X[fs.test_mask])
    y_tr, y_te = fs.y[fs.train_mask], fs.y[fs.test_mask]
    tier_te = fv.loc[fs.test_mask, "tier"].to_numpy()
    tier_tr = fv.loc[fs.train_mask, "tier"].to_numpy()
    tiers = sorted(set(tier_te))

    rows = []
    for method, kwargs in METHODS:
        if method == "none":
            p_after = p_te
        else:
            cal = PerTierCalibrator(**kwargs).fit(p_tr, y_tr, tier_tr)
            p_after = cal.transform(p_te, tier_te)
            extra = f" — skipped {cal.skipped}" if cal.skipped else ""
            print(f"{method}: maps for {sorted(cal.maps)}{extra}")
        for tier in list(tiers) + ["ALL"]:
            mask = np.ones_like(tier_te, dtype=bool) if tier == "ALL" else (tier_te == tier)
            rows.append(
                {
                    "method": method,
                    "tier": tier,
                    "n": int(mask.sum()),
                    "ece_before": expected_calibration_error(y_te[mask], p_te[mask]),
                    "ece_after": expected_calibration_error(y_te[mask], p_after[mask]),
                    "logloss_before": log_loss(p_te[mask], y_te[mask]),
                    "logloss_after": log_loss(p_after[mask], y_te[mask]),
                }
            )
    out = pd.DataFrame(rows)
    dest = REPO / "outputs" / "m11_tier_calibration.csv"
    out.to_csv(dest, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
