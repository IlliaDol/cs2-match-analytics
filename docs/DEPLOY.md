# Deploying the API (P2.7/A.4)

Two serve surfaces exist:
- `cs2analytics.serve.app` — FastAPI (`/health`, `/predict`), the M11 model service.
- `cs2analytics.serve.dashboard` — Streamlit demo (M12), local-first.

Both load `artifacts/{model.pkl,features.json,elo_ratings.json}` at request time.
The trained model is **lr+roster** (`MODEL_VERSION = lr-roster-2026-09-12`) — the
same model the README headline reports (logloss 0.6377). `python -m
cs2analytics.models.train` regenerates all three artifacts.

## Why the Dockerfile needs a build-arg

`artifacts/` is git-ignored (trained models don't belong in git), so a fresh
clone has an EMPTY `artifacts/`. The image therefore provisions the model at
BUILD time from a GitHub release asset tarball:

```
docker build \
  --build-arg ARTIFACTS_URL="https://api.github.com/repos/<you>/<repo>/releases/assets/<id>" \
  --build-arg ARTIFACTS_TOKEN="<fine-grained PAT with that asset's read scope>" \
  -t cs2-api .
```

If `ARTIFACTS_URL` is unset the build FAILS with an explicit error — a silent
model-less image is exactly the bug class this repo refuses to ship. The tarball
must contain `model.pkl`, `features.json`, `elo_ratings.json` at its root
(`tar -czf artifacts.tgz -C artifacts .`).

## Render (Docker service)

1. Push this repo (Dockerfile + render.yaml are committed).
2. render.com → New → **Blueprint** → select the repo. Render reads
   `render.yaml`'s `services:` block (`type: web`, `env: docker`,
   `healthCheckPath: /health`) and injects `$PORT`.
3. Under the service's **Environment** tab add the two build-arg values as
   `ARTIFACTS_URL` / `ARTIFACTS_TOKEN` secret files OR set them in a small
   wrapper workflow — Render Blueprint secrets apply at RUNTIME; build args
   need the "Build Hook"/Docker command override (free tier limitation: set
   them via `render.yaml`'s `dockerCommand`/predeploy job if you go that far).
4. Free tier caveat: spins down after ~15 min idle; first request cold-starts.

## Hugging Face Space (Streamlit, simplest demo)

1. huggingface.co → New Space → SDK **Docker**.
2. Upload the repo (or connect via git); set Space secrets `ARTIFACTS_URL` /
   `ARTIFACTS_TOKEN` and reference them in the Space's Dockerfile build.
3. Start command: `uvicorn ... --port 7860` for the API or
   `streamlit run src/cs2analytics/serve/dashboard.py` for the dashboard.

## What runs where after A.1–A.6

- `python -m cs2analytics.models.train` → artifacts for **lr+roster** (0.6377).
- `serve/app.py` and `serve/dashboard.py` share one prediction path
  (`serve/inference.py`) and both report `model_version` from
  `features.json` — the live demo is the paper's model, verifiably.
- `/predict` now accepts optional `form5_diff`, `rest_days_diff`,
  `h2h_t1_win_share`, `roster_stability_diff`, `standin_diff`; omit them for
  training-neutral defaults.
