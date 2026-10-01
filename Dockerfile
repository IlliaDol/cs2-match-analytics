FROM python:3.12-slim

WORKDIR /app

# build-time args for artifact provisioning (see docs/DEPLOY.md)
# ARTIFACTS_URL  — URL of a GitHub release asset tarball containing
#                  artifacts/{model.pkl,features.json,elo_ratings.json}
# ARTIFACTS_TOKEN— optional bearer token for that asset (private release)
ARG ARTIFACTS_URL=""
ARG ARTIFACTS_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

RUN pip install --no-cache-dir -e . fastapi uvicorn joblib pydantic

# Provision a REAL model. Without this the image serves a model-less API that
# throws on first /health. Kept as one explicit step so a missing URL fails the
# build loudly instead of producing a silent model-less image.
RUN if [ -n "$ARTIFACTS_URL" ]; then \
      mkdir -p artifacts && \
      curl -fsSL ${ARTIFACTS_TOKEN:+-H "Authorization: Bearer $ARTIFACTS_TOKEN"} "$ARTIFACTS_URL" -o artifacts.tgz && \
      tar -xzf artifacts.tgz -C artifacts && \
      rm artifacts.tgz && \
      ls -la artifacts/; \
    else \
      echo "ERROR: ARTIFACTS_URL build-arg not set — the image would ship without model.pkl" >&2; \
      exit 1; \
    fi

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn cs2analytics.serve.app:app --host 0.0.0.0 --port ${PORT}"]
