"""M11 tests — PSI monitor (CI-runs) + FastAPI contract (skips without artifacts)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

REPO = Path(__file__).parent.parent
ARTIFACTS = REPO / "artifacts" / "model.pkl"

try:
    from cs2analytics.serve.monitor import psi

    _M11_READY = True
    _M11_REASON = ""
except ModuleNotFoundError as _e:
    _M11_READY = False
    _M11_REASON = f"M11 monitor not implemented yet ({_e.name}) — the human's build"

pytestmark = [pytest.mark.skipif(not _M11_READY, reason=_M11_REASON)]


# --- PSI: pure function, runs in CI -------------------------------------------


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_distribution_flags_drift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 10_000)
    shifted = rng.normal(2, 1, 10_000)  # 2 sigma shift = obvious drift
    assert psi(base, shifted) > 0.25


def test_psi_near_shift_below_threshold():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 20_000)
    near = rng.normal(0.05, 1, 20_000)
    assert psi(base, near) < 0.1


# --- API contract (needs trained artifacts) ------------------------------------


@pytest.fixture()
def client():
    if not ARTIFACTS.exists():
        pytest.skip("artifacts/model.pkl missing — run cs2analytics.models.train (M11 §1)")
    from cs2analytics.serve.app import app
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"]


def test_predict_probability_range(client):
    r = client.post("/predict", json={"team1": "NAVI", "team2": "FaZe", "best_of": 3})
    assert r.status_code == 200
    p = r.json()["p_team1"]
    assert 0.0 < p < 1.0


def test_predict_symmetry(client):
    a = client.post("/predict", json={"team1": "NAVI", "team2": "FaZe", "best_of": 3}).json()
    b = client.post("/predict", json={"team1": "FaZe", "team2": "NAVI", "best_of": 3}).json()
    assert abs(a["p_team1"] + b["p_team1"] - 1.0) < 0.02


def test_predict_rejects_bad_best_of(client):
    r = client.post("/predict", json={"team1": "NAVI", "team2": "FaZe", "best_of": 2})
    assert r.status_code == 422


def test_predict_unknown_team_falls_back(client):
    r = client.post(
        "/predict", json={"team1": "Definitely Not A Team", "team2": "NAVI", "best_of": 3}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["elo_t1"] == pytest.approx(1500.0)
    assert "Definitely Not A Team" in body.get("unknown_team", [])
