"""Transparent baseline estimators for the roster causal study.

These functions intentionally avoid hiding assumptions behind a modelling
framework. They return tables/dictionaries suitable for a report and are meant
to be paired with pre-trend, overlap, placebo, and sensitivity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cell_mean(panel: pd.DataFrame, treated: int, post: int, outcome: str) -> float:
    values = pd.to_numeric(
        panel.loc[(panel["treated"] == treated) & (panel["post"] == post), outcome],
        errors="coerce",
    ).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def did_estimate(panel: pd.DataFrame, outcome: str = "win") -> dict[str, float | int]:
    """Return the four cell means and the 2x2 DiD estimate.

    ``estimate = (treated_post - treated_pre) - (control_post - control_pre)``.
    This is a descriptive baseline and has no automatic standard-error claim.
    """
    required = {"treated", "post", outcome}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing DiD columns: {missing}")
    values = {
        "treated_pre": _cell_mean(panel, 1, 0, outcome),
        "treated_post": _cell_mean(panel, 1, 1, outcome),
        "control_pre": _cell_mean(panel, 0, 0, outcome),
        "control_post": _cell_mean(panel, 0, 1, outcome),
    }
    estimate = (values["treated_post"] - values["treated_pre"]) - (
        values["control_post"] - values["control_pre"]
    )
    counts = panel.groupby(["treated", "post"], dropna=False)[outcome].count()
    return {
        **values,
        "treated_change": values["treated_post"] - values["treated_pre"],
        "control_change": values["control_post"] - values["control_pre"],
        "estimate": float(estimate),
        "n_treated": int(counts.get((1, 0), 0) + counts.get((1, 1), 0)),
        "n_control": int(counts.get((0, 0), 0) + counts.get((0, 1), 0)),
    }


def event_study_table(panel: pd.DataFrame, outcome: str = "win") -> pd.DataFrame:
    """Summarize treated/control means and differences by relative match index."""
    required = {"relative_match", "treated", outcome}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing event-study columns: {missing}")
    grouped = (
        panel.groupby(["relative_match", "treated"], as_index=False)[outcome]
        .agg(mean="mean", n="count")
        .pivot(index="relative_match", columns="treated", values=["mean", "n"])
        .sort_index()
    )
    grouped.columns = [
        f"{'treated' if treatment else 'control'}_{metric}"
        for metric, treatment in grouped.columns
    ]
    grouped = grouped.reset_index()
    if "treated_mean" not in grouped:
        grouped["treated_mean"] = np.nan
    if "control_mean" not in grouped:
        grouped["control_mean"] = np.nan
    grouped["difference"] = grouped["treated_mean"] - grouped["control_mean"]
    return grouped


def bootstrap_did(
    panel: pd.DataFrame,
    outcome: str = "win",
    *,
    cluster: str = "event_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Cluster-bootstrap the DiD by event, preserving within-event rows.

    This is the uncertainty estimate used by the report. It is appropriate only
    when ``cluster`` identifies independent matched events; it is not a cure for
    selection bias or failed parallel trends.
    """
    if cluster not in panel:
        raise ValueError(f"bootstrap requires cluster column {cluster!r}")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    clusters = panel[cluster].dropna().unique()
    if len(clusters) < 2:
        raise ValueError("bootstrap requires at least two independent clusters")
    grouped = (
        panel.groupby([cluster, "treated", "post"], dropna=False)[outcome]
        .agg(["sum", "count"])
        .reset_index()
    )
    cells = {
        (row[cluster], int(row["treated"]), int(row["post"])): (
            float(row["sum"]),
            int(row["count"]),
        )
        for _, row in grouped.iterrows()
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        totals: dict[tuple[int, int], list[float]] = {}
        for value in sampled:
            for treated in (0, 1):
                for post in (0, 1):
                    total, count = cells.get((value, treated, post), (0.0, 0))
                    cell = totals.setdefault((treated, post), [0.0, 0.0])
                    cell[0] += total
                    cell[1] += count
        means = {
            cell: (total / count if count else np.nan)
            for cell, (total, count) in totals.items()
        }
        estimate = (means[(1, 1)] - means[(1, 0)]) - (
            means[(0, 1)] - means[(0, 0)]
        )
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        raise ValueError("bootstrap produced no complete four-cell draws")
    values = np.asarray(estimates, dtype=float)
    return {
        "estimate": float(did_estimate(panel, outcome)["estimate"]),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_boot": int(len(estimates)),
        "n_clusters": int(len(clusters)),
        "seed": int(seed),
    }


def balance_table(
    events: pd.DataFrame,
    covariates: list[str],
    treatment_column: str = "treated",
) -> pd.DataFrame:
    """Compute standardized mean differences for pre-treatment covariates."""
    if treatment_column not in events:
        raise ValueError(f"missing treatment column {treatment_column!r}")
    rows: list[dict[str, float | str | int]] = []
    treated = events[treatment_column] == 1
    control = events[treatment_column] == 0
    for column in covariates:
        if column not in events:
            raise ValueError(f"missing balance covariate {column!r}")
        a = pd.to_numeric(events.loc[treated, column], errors="coerce").dropna()
        b = pd.to_numeric(events.loc[control, column], errors="coerce").dropna()
        if a.empty or b.empty:
            rows.append(
                {
                    "covariate": column,
                    "treated_n": len(a),
                    "control_n": len(b),
                    "smd": np.nan,
                }
            )
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        mean_difference = float(a.mean() - b.mean())
        smd = (
            0.0
            if pooled == 0 and mean_difference == 0
            else (float("inf") if pooled == 0 else mean_difference / pooled)
        )
        rows.append(
            {
                "covariate": column,
                "treated_n": len(a),
                "control_n": len(b),
                "treated_mean": float(a.mean()),
                "control_mean": float(b.mean()),
                "smd": smd,
            }
        )
    return pd.DataFrame(rows)
