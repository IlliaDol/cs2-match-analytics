"""M11 §2 — FastAPI service: GET /health + POST /predict.

Loads artifacts at first request (module-level lazy cache). Unknown teams get
Elo 1500 (documented fallback) and are listed in `unknown_team`. `best_of`
maps to the `is_bo1` feature exactly as in training (Bo1 iff best_of == 1).

Feature assembly + symmetrization live in `serve/inference.py` — the single
source of truth shared with the Streamlit dashboard. The trained model is
`lr+roster`; callers MAY supply form/rest/h2h/roster context and the endpoint
uses their values, otherwise it applies training-neutral defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cs2analytics.serve.inference import build_feature_row, predict_symmetrized

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "artifacts"

app = FastAPI(title="CS2 match analytics API", version="1.1")


class PredictBody(BaseModel):
    team1: str
    team2: str
    best_of: int
    # optional pre-match context; default None -> training-neutral value
    form5_diff: float | None = None
    rest_days_diff: float | None = None
    h2h_t1_win_share: float | None = None
    roster_stability_diff: float | None = None
    standin_diff: float | None = Field(default=None, ge=-1.0, le=1.0)


@lru_cache(maxsize=1)
def _artifacts() -> dict[str, Any]:
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

    X = build_feature_row(
        meta["feature_names"],
        elo_t1 - elo_t2,
        body.best_of,
        form5_diff=body.form5_diff,
        rest_days_diff=body.rest_days_diff,
        h2h_t1_win_share=body.h2h_t1_win_share,
        roster_stability_diff=body.roster_stability_diff,
        standin_diff=body.standin_diff,
    )

    result = predict_symmetrized(art["model"], meta, elo_t1, elo_t2, X)
    return {
        "p_team1": result["p_team1"],
        "model_version": meta["model_version"],
        "elo_t1": result["elo_t1"],
        "elo_t2": result["elo_t2"],
        "n_train": int(meta["metrics"]["n_train"]),
        "unknown_team": unknown,
    }
