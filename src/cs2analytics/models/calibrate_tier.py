"""Per-tier calibration (D.1 / v1.1): isotonic vs Platt, fit on TRAIN per tier.

The per-regime table (notebook 03) showed that aggregate ECE hides a tier-3
problem: ECE 0.126 vs 0.02-0.03 elsewhere — the model is over-confident exactly
where the data is noisiest. `scripts/build_tier_calibration.py` measures three
refits on the test split; `docs/DECISIONS.md` and the README table hold the
verdict:

* **isotonic, per tier** — tier-3 ECE 0.0788 -> 0.0616 but tier-3 logloss
  0.622 -> 0.752, and tier-1 worse on both. With a few hundred rows per tier the
  map fits bin edges, not signal (same failure mode as M7's GBM+isotonic
  negative). **Rejected.**
* **Platt, per tier** (one logistic parameter on logit(p)) — much less flexible,
  so it can fix the ECE without paying logloss: tier-3 ECE 0.0788 -> 0.0669 with
  logloss 0.6220 -> 0.6199, tier-2 better on both — but tier-1 ECE gets worse
  (0.0202 -> 0.0326), so the *aggregate* ECE rises (0.0185 -> 0.0233).
* **Platt + tier selection** (`select="overconfident"`, tested) — meant to spare the
  well-calibrated tiers, but the *relative* bar cannot discriminate on this data:
  the pooled TRAIN ECE is 0.0066 while every per-tier TRAIN ECE is larger (tier1
  0.0124, tier2 0.0220, tier3 0.0325), because the tiers' miscalibrations partly
  cancel in the mixture. The rule therefore selects all three tiers and degenerates
  to plain Platt — kept as a tested knob (`select="all"` by default) rather than
  tuned until the aggregate looked better.

Leakage law: every map is fit on the TRAINING split only, per tier, and the
selection uses train statistics only. At predict time the tier must be supplied;
unknown tiers pass through uncalibrated, mirroring the API's behaviour of zeroing
tier one-hots. Skipped tiers are reported in `.skipped` as (tier, train rows,
reason) instead of being given a map that cannot generalise.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cs2analytics.evaluation.calibration import expected_calibration_error

_METHODS = ("isotonic", "platt")
_SELECT = ("all", "overconfident")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


class _Platt:
    """Platt scaling: a one-feature logistic regression on logit(p).

    Deliberately almost unregularised (`C=1e6`): the fit is two parameters, and
    the point of the method is to *not* fit bin edges.
    """

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> _Platt:
        self.lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class PerTierCalibrator:
    """One calibration map per tier, fit on (p_model, y) pairs from TRAIN.

    Args:
        method: ``"isotonic"`` or ``"platt"``.
        min_n: minimum training rows for a tier to be calibrated at all.
        select: ``"all"`` calibrates every eligible tier; ``"overconfident"``
            calibrates only tiers whose training ECE is worse than
            ``ece_factor`` x the overall training ECE.
        ece_factor: the selection threshold used by ``select="overconfident"``.
    """

    def __init__(
        self,
        method: str = "platt",
        min_n: int = 60,
        select: str = "all",
        ece_factor: float = 1.25,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if select not in _SELECT:
            raise ValueError(f"select must be one of {_SELECT}, got {select!r}")
        self.method = method
        self.min_n = int(min_n)
        self.select = select
        self.ece_factor = float(ece_factor)
        self.maps: dict[str, object] = {}
        self.skipped: list[tuple[str, int, str]] = []

    def fit(self, p: np.ndarray, y: np.ndarray, tier_labels: np.ndarray) -> PerTierCalibrator:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        labels = np.asarray(tier_labels)
        self.maps, self.skipped = {}, []
        overall_ece = (
            expected_calibration_error(y, p) if self.select == "overconfident" else None
        )
        for tier in sorted(set(labels.tolist())):
            mask = labels == tier
            n = int(mask.sum())
            if n < self.min_n:
                self.skipped.append((str(tier), n, "too-few-rows"))
                continue
            if len(set(y[mask].tolist())) < 2:
                self.skipped.append((str(tier), n, "single-class"))
                continue
            if overall_ece is not None:
                tier_ece = expected_calibration_error(y[mask], p[mask])
                if tier_ece <= self.ece_factor * overall_ece:
                    self.skipped.append((str(tier), n, "well-calibrated-on-train"))
                    continue
            if self.method == "isotonic":
                fitted: object = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.0, y_max=1.0
                ).fit(p[mask], y[mask])
            else:
                fitted = _Platt().fit(p[mask], y[mask])
            self.maps[str(tier)] = fitted
        return self

    def transform(self, p: np.ndarray, tier_labels: np.ndarray) -> np.ndarray:
        out = np.asarray(p, dtype=float).copy()
        labels = np.asarray(tier_labels)
        for tier, fitted in self.maps.items():
            mask = labels == tier
            if mask.any():
                out[mask] = fitted.predict(p[mask])          # type: ignore[attr-defined]
        return np.clip(out, 0.0, 1.0)
