"""Evaluation metrics for probabilistic match predictions.

M4 module: exact-value contract tests live in tests/test_elo.py.
All functions take iterables of probabilities and binary outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _to_arrays(p: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.asarray(list(p), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"p and y must have the same shape, got {p_arr.shape} vs {y_arr.shape}")
    return p_arr, y_arr


def log_loss(p: Iterable[float], y: Iterable[float]) -> float:
    """Mean binary log loss: -(y ln p + (1-y) ln(1-p)).

    Probabilities are clipped to [1e-15, 1-1e-15] so confident mistakes stay
    large but finite (ln(0) would be -inf).
    """
    p_arr, y_arr = _to_arrays(p, y)
    p_clipped = np.clip(p_arr, 1e-15, 1.0 - 1e-15)
    losses = -(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped))
    return float(np.mean(losses))


def brier_score(p: Iterable[float], y: Iterable[float]) -> float:
    """Mean squared error of probability forecasts: mean (p - y)^2."""
    p_arr, y_arr = _to_arrays(p, y)
    return float(np.mean((p_arr - y_arr) ** 2))


def accuracy_from_probs(p: Iterable[float], y: Iterable[float], threshold: float = 0.5) -> float:
    """Share of correct class predictions when thresholding p at `threshold`."""
    p_arr, y_arr = _to_arrays(p, y)
    return float(np.mean((p_arr >= threshold).astype(float) == y_arr))
