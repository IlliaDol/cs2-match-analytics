# Deploying the API + dashboard (P2.7) — what you must decide

The repo has two serve surfaces:
- `cs2analytics.serve.app` — FastAPI (`/health`, `/predict`), M11.
- `cs2analytics.serve.dashboard` — Streamlit demo, M12.

Both load only `artifacts/` (model.pkl + features.json + elo_ratings.json).
They never ship the match CSVs.

## Option A — Render (recommended, no Docker install needed locally)

1. Push this repo (the new `render.yaml` and `Dockerfile` are included).
2. render.com → New → **Blueprint** → point at the repo.
3. Set the service as a **Docker** service (it reads `Dockerfile`), not a native
   service, or `render.yaml` will be ignored.
4. Add a **free web service**; Render injects `$PORT`. That's all the config.

Free tier caveat: spins down after ~15 min idle; first request is a cold start.
Fine for a portfolio link, not for real traffic.

## Option B — Hugging Face Space (Streamlit only, simplest for the demo)

1. huggingface.co → New Space → SDK **Docker** (or Gradio/Streamlit template).
2. Upload: `src/`, `artifacts/`, `requirements.txt` (see below), `app.py` that
   calls `streamlit run src/cs2analytics/serve/dashboard.py`.
3. The Space serves `streamlit` from the container start command.

## The one real blocker

Both hosts need a container image built. **Docker is not installed on this
Windows machine** (`docker` not found), so I cannot build/push the image from
here. The repo now ships a `Dockerfile`; the build happens on Render/HF's side
when you connect the repo. You need:
- a Render or HF account (free),
- the artifacts committed or uploaded (they're git-ignored — either generate
  them in the CI job and download from there, or upload to the host directly).

## requirements.txt for HF Space (if not using Dockerfile)

```
fastapi
uvicorn
joblib
pydantic
pandas>=2.2
numpy>=1.26
streamlit
scikit-learn
```
