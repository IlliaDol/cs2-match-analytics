FROM python:3.12-slim

WORKDIR /app

# system deps: none beyond git for hatchling (editables)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY artifacts ./artifacts
COPY README.md ./

RUN pip install --no-cache-dir -e . fastapi uvicorn joblib pydantic

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn cs2analytics.serve.app:app --host 0.0.0.0 --port ${PORT}"]
