"""M7 §3 — calibration: reliability table, ECE, Brier decomposition.

Identity (bin-based Murphy decomposition): brier ≈ reliability − resolution + uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cs2analytics.evaluation.metrics import (
    brier_score,  # noqa: F401 (re-exported for the contract test)
)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Per-bin [bin_lo, bin_hi, n, mean_pred, obs_rate] over probability bins in [0, 1]."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # clip p==1.0 into the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_pred": float(p[mask].mean()) if n else np.nan,
                "obs_rate": float(y[mask].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |obs_rate_b - mean_pred_b| over non-empty bins."""
    table = reliability_table(y, p, bins=bins)
    filled = table[table["n"] > 0]
    n_total = int(filled["n"].sum())
    weights = filled["n"] / n_total
    return float((weights * (filled["obs_rate"] - filled["mean_pred"]).abs()).sum())


def brier_decomposition(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Murphy bin decomposition: reliability - resolution + uncertainty.

    uncertainty = y_bar(1 - y_bar); per bin: (n_b/N) (mean_pred_b - obs_b)^2 is
    the reliability term and (n_b/N) (obs_rate_b - y_bar)^2 the resolution term.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    table = reliability_table(y, p, bins=bins)
    y_bar = float(y.mean())
    n_total = int(table["n"].sum())

    rel = res = 0.0
    for row in table.itertuples(index=False):
        if row.n == 0:
            continue
        w = row.n / n_total
        rel += w * (row.mean_pred - row.obs_rate) ** 2
        res += w * (row.obs_rate - y_bar) ** 2
    uncertainty = y_bar * (1.0 - y_bar)
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(uncertainty),
    }
