"""M7 §2 — GBM model wrappers (kept separate for the module map).

`fit_gbm` / `predict_proba` re-exported here so `models/gbm.py` exists as the
spec's named module; the shared implementation lives in logistic.py to keep one
source of truth for the pipeline definitions.
"""

from __future__ import annotations

from cs2analytics.models.logistic import RANDOM_STATE, fit_gbm, fit_gbm_isotonic, predict_proba

__all__ = ["RANDOM_STATE", "fit_gbm", "fit_gbm_isotonic", "predict_proba"]