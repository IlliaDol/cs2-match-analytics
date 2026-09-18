"""Shared inference: turn (team1, team2, best_of, context) into a prediction.

This is the SINGLE source of truth for feature-row assembly + symmetrization,
used by both `serve/app.py` (FastAPI) and `serve/dashboard.py` (Streamlit) so
they cannot drift apart. The trained model is `lr+roster`; its feature order is
read from `artifacts/features.json` at load time, never hardcoded.
"""

from __future__ import annotations

import numpy as np


def build_feature_row(
    feature_names: list[str],
    elo_diff: float,
    best_of: int,
    *,
    form5_diff: float | None = None,
    rest_days_diff: float | None = None,
    h2h_t1_win_share: float | None = None,
    roster_stability_diff: float | None = None,
    standin_diff: float | None = None,
) -> np.ndarray:
    """Assemble a single feature row in the model's trained order.

    `feature_names` comes from artifacts/features.json — the manifest is the
    contract. Caller-supplied context wins; everything else gets its
    training-neutral default (0.0 for diffs, 0.5 for a share, 0.0 for tier
    one-hots = tier unknown at prediction time).
    """
    values: dict[str, float] = {
        "elo_diff": float(elo_diff),
        "is_bo1": 1.0 if best_of == 1 else 0.0,
        "form5_diff": 0.0,
        "rest_days_diff": 0.0,
        "h2h_t1_win_share": 0.5,
        "roster_stability_diff": 0.0,
        "standin_diff": 0.0,
        "bo1_elo": 0.0,
    }
    for name, val in (
        ("form5_diff", form5_diff),
        ("rest_days_diff", rest_days_diff),
        ("h2h_t1_win_share", h2h_t1_win_share),
        ("roster_stability_diff", roster_stability_diff),
        ("standin_diff", standin_diff),
    ):
        if val is not None:
            values[name] = float(val)

    row = []
    for name in feature_names:
        if name.startswith("tier_"):
            row.append(0.0)  # tier unknown at prediction time
        elif name in values:
            row.append(values[name])
        else:
            raise KeyError(
                f"trained feature {name!r} has no assembly rule in serve/inference.py — "
                "add one; serving a silent 0.0 for an unknown feature is how A.2 happened"
            )
    return np.array([row], dtype=float)


#: features that are ODD under team swap. h2h flips around 0.5 (p -> 1-p);
#: the diffs and bo1_elo flip around 0 (p -> -p).
_ODD_FEATURES = {
    "elo_diff",
    "form5_diff",
    "rest_days_diff",
    "roster_stability_diff",
    "standin_diff",
    "bo1_elo",
}
_ODD_AROUND_HALF = {"h2h_t1_win_share"}


def predict_symmetrized(model, meta: dict, elo_t1: float, elo_t2: float, X: np.ndarray) -> dict:
    """Symmetrized P(team1 wins), immune to the scaler's column-order artifact.

    p = (p_raw + (1 - p_swap)) / 2 where p_swap is the model run with every
    team-oriented feature sign-flipped — see DECISIONS.md "M11: /predict
    symmetrizes". Flipping ALL oriented features (not just elo_diff) keeps the
    law exact even when the caller supplies asymmetric context.
    """
    swap_idx = [i for i, name in enumerate(meta["feature_names"]) if name in _ODD_FEATURES]
    half_idx = [i for i, name in enumerate(meta["feature_names"]) if name in _ODD_AROUND_HALF]
    X_swap = X.copy()
    for i in swap_idx:
        X_swap[0, i] *= -1.0
    for i in half_idx:
        X_swap[0, i] = 1.0 - X_swap[0, i]
    p_raw = float(model.predict_proba(X)[0, 1])
    p_swap = float(model.predict_proba(X_swap)[0, 1])
    p = (p_raw + (1.0 - p_swap)) / 2.0
    return {
        "p_team1": p,
        "elo_t1": elo_t1,
        "elo_t2": elo_t2,
    }
