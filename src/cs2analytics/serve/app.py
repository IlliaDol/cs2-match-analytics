"""M11 §2 — FastAPI service: GET /health + POST /predict.

Loads artifacts at first request (module-level lazy cache). Unknown teams get
Elo 1500 (documented fallback) and are listed in `unknown_team`. `best_of`
maps to the `is_bo1` feature exactly as in training (Bo1 iff best_of == 1).
Numeric features other than elo_diff/is_bo1 are set to training-neutral values
(form5_diff=0, rest_days_diff=0, h2h=0.5) so the endpoint is a pure rating gap
+ format model unless a caller supplies more context.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"

app = FastAPI(title="CS2 match analytics API", version="1.0")


class PredictBody(BaseModel):
    team1: str
    team2: str
    best_of: int


@lru_cache(maxsize=1)
def _artifacts() -> dict:
    model = joblib.load(ARTIFACTS / "model.pkl")
    meta = json_load(ARTIFACTS / "features.json")
    elo = json_load(ARTIFACTS / "elo_ratings.json")
    return {"model": model, "meta": meta, "elo": elo}


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _elo_of(name: str, elo: dict[str, float]) -> tuple[float, bool]:
    """Elo rating + whether the team is unknown (documented 1500 fallback)."""
    if name in elo:
        return elo[name], False
    # case-insensitive retry (team-name casing is a known data trap)
    for k, v in elo.items():
        if k.lower() == name.lower():
            return v, False
    return 1500.0, True


@app.get("/health")
def health() -> dict:
    meta = _artifacts()["meta"]
    return {"status": "ok", "model_version": meta["model_version"]}


@app.post("/predict")
def predict(body: PredictBody) -> dict:
    if body.best_of not in (1, 3, 5):
        raise HTTPException(status_code=422, detail="best_of must be one of 1, 3, 5")

    art = _artifacts()
    meta = art["meta"]
    elo = art["elo"]
    elo_t1, unk1 = _elo_of(body.team1, elo)
    elo_t2, unk2 = _elo_of(body.team2, elo)

    unknown = [n for n, u in ((body.team1, unk1), (body.team2, unk2)) if u]

    # assemble the feature row in the EXACT trained order
    row = {}
    for name in meta["feature_names"]:
        if name == "elo_diff":
            row[name] = elo_t1 - elo_t2
        elif name == "is_bo1":
            row[name] = 1.0 if body.best_of == 1 else 0.0
        elif name == "form5_diff":
            row[name] = 0.0
        elif name == "rest_days_diff":
            row[name] = 0.0
        elif name.startswith("tier_"):
            row[name] = 0.0  # tier unknown at prediction time -> all zeros
        else:
            row[name] = 0.0
    X = np.array([[row[name] for name in meta["feature_names"]]])

    # Symmetrize over both orientations: p(a,b) + p(b,a) must equal 1 by
    # definition. The raw pipeline is NOT odd in elo_diff (the scaler's
    # training mean is ~+23.6 because team1 wins 55% of rows — a seeding
    # artifact, see EDA surprise 3), so serving the raw output would encode
    # column order into every prediction. Averaging the model over (1,2) and
    # (2,1) removes the artifact while keeping the learned magnitudes.
    X_swap = X.copy()
    X_swap[0, meta["feature_names"].index("elo_diff")] *= -1.0
    p_raw = float(art["model"].predict_proba(X)[0, 1])
    p_swap = float(art["model"].predict_proba(X_swap)[0, 1])
    p = (p_raw + (1.0 - p_swap)) / 2.0
    return {
        "p_team1": p,
        "model_version": meta["model_version"],
        "elo_t1": elo_t1,
        "elo_t2": elo_t2,
        "n_train": int(meta["metrics"]["n_train"]),
        "unknown_team": unknown,
    }
