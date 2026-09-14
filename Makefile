# Resolve Rscript portably: PATH first, then the known local install.
# Kept in one place so README/SAUCE_INDEX never hardcode a Windows-only path.
RSCRIPT ?= $(shell command -v Rscript 2>/dev/null || echo "C:/Program Files/R/R-4.6.1/bin/Rscript.exe")

.PHONY: r-m3 r-all features train serve test lint

r-m3:
	"$(RSCRIPT)" r/01_wrangle.R
	"$(RSCRIPT)" r/02_inference.R
	"$(RSCRIPT)" r/03_plots.R

r-all: r-m3
	"$(RSCRIPT)" r/04_rating_forecast.R

features:
	python scripts/build_interim.py
	python scripts/build_features_v1.py

train:
	python -m cs2analytics.models.train

serve:
	uvicorn cs2analytics.serve.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
