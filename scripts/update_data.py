"""Live-refresh path (v1.1 3): fetch new raw data, rebuild the derived tables,
and report what actually changed.

Why this exists: without it the repo is a snapshot — "the data stops at the
download date" was a real limitation. This script closes that gap without faking
anything. It either

* fetches the CSVs named by ``DATA_SOURCE_URL`` (the same zip + bearer-token
  convention the nightly CI job uses — see `docs/CI_DATA_JOB.md`), or
* uses the local ``data/raw/*.csv`` when no URL is configured,

then rebuilds series -> feature store -> monthly Elo and writes
``outputs/update_report.json`` describing the drift between the old and new
feature store. It never invents data: with no URL and no local CSVs it raises.

Run from the repo root:

    .venv/Scripts/python.exe scripts/update_data.py                  # local data
    DATA_SOURCE_URL=... .venv/Scripts/python.exe scripts/update_data.py    # fetch new CSVs

The R step (`make r-m3`) is deliberately NOT run here — R has its own make target
(see README "Reproducibility"), and this script stays Python-only.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw"
FEATURES = REPO / "outputs" / "features_v1.parquet"
REPORT = REPO / "outputs" / "update_report.json"

# Python steps of the pipeline, in order. Each must be runnable from the repo root.
REBUILD_STEPS = ("build_interim.py", "build_features_v1.py", "build_elo_monthly.py")

# Columns summarised in the drift report (intersected with what the store has).
DIGEST_COLUMNS = ("elo_diff", "form5_diff", "rest_days_diff", "h2h_t1_win_share")

# A mean shift larger than this (in standard deviations of the old data) is flagged.
SHIFT_FLAG_SIGMA = 0.25


def raw_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """The non-empty CSVs currently in data/raw (sorted, stable order)."""
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.glob("*.csv") if p.stat().st_size > 0)


def refresh_raw(
    raw_dir: Path = RAW_DIR,
    url: str | None = None,
    token: str | None = None,
    timeout: int = 120,
) -> list[Path]:
    """Put raw CSVs in place: download+extract when `url` is given, else use local.

    Raises FileNotFoundError when neither route yields data — this script must fail
    loudly rather than let a stale dataset look refreshed.
    """
    if url:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            blob = response.read()
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = [n for n in archive.namelist()
                     if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            for name in names:
                (raw_dir / Path(name).name).write_bytes(archive.read(name))
    files = raw_csvs(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"no raw CSVs in {raw_dir} and no DATA_SOURCE_URL set — see docs/CI_DATA_JOB.md"
        )
    return files


def digest(features_path: Path = FEATURES) -> dict:
    """Cheap summary of a feature store: rows, latest date, per-column mean/std."""
    if not features_path.exists():
        return {"exists": False}
    frame = pd.read_parquet(features_path)
    out: dict = {"exists": True, "rows": int(len(frame))}
    if "datetime" in frame.columns:
        dates = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        out["latest"] = dates.max().date().isoformat() if dates.notna().any() else None
    stats = {}
    for column in DIGEST_COLUMNS:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(series):
                stats[column] = {"mean": float(series.mean()), "std": float(series.std(ddof=0))}
    out["stats"] = stats
    return out


def drift_report(before: dict, after: dict, sigma: float = SHIFT_FLAG_SIGMA) -> dict:
    """Compare two digests. Flags row loss, a stale latest date, and mean shifts."""
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows_before": before.get("rows"),
        "rows_after": after.get("rows"),
        "rows_delta": (after.get("rows") or 0) - (before.get("rows") or 0)
        if before.get("exists") and after.get("exists") else None,
        "latest_before": before.get("latest"),
        "latest_after": after.get("latest"),
        "mean_shift": {},
        "flagged": [],
    }
    for column, old in (before.get("stats") or {}).items():
        new = (after.get("stats") or {}).get(column)
        if not new:
            continue
        shift = new["mean"] - old["mean"]
        report["mean_shift"][column] = round(shift, 6)
        scale = old["std"] or 1.0
        if abs(shift) > sigma * scale:
            report["flagged"].append(f"{column}: mean moved {shift:+.4f} (> {sigma:g} sigma)")
    if report["rows_delta"] is not None and report["rows_delta"] < 0:
        report["flagged"].append(f"rows fell by {-report['rows_delta']}")
    if before.get("latest") and after.get("latest") and after["latest"] <= before["latest"]:
        report["flagged"].append(f"latest date did not advance ({after['latest']})")
    return report


def rebuild(repo: Path = REPO, steps: tuple[str, ...] = REBUILD_STEPS) -> list[str]:
    """Run the Python pipeline steps in order; returns the steps that ran."""
    ran = []
    for step in steps:
        script = repo / "scripts" / step
        if not script.exists():
            raise FileNotFoundError(f"pipeline step missing: {script}")
        subprocess.run([sys.executable, str(script)], cwd=repo, check=True)
        ran.append(step)
    return ran


def main() -> None:
    url = os.environ.get("DATA_SOURCE_URL") or None
    token = os.environ.get("DATA_SOURCE_TOKEN") or None
    before = digest()

    files = refresh_raw(url=url, token=token)
    print(f"raw CSVs       : {len(files)} in {RAW_DIR}"
          + ("  (downloaded from DATA_SOURCE_URL)" if url else "  (local, DATA_SOURCE_URL unset)"))

    ran = rebuild()
    print(f"rebuilt        : {', '.join(ran)}")

    after = digest()
    report = drift_report(before, after)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    delta = report["rows_delta"]
    print(f"rows           : {report['rows_before']} -> {report['rows_after']}"
          + (f" ({delta:+d})" if delta is not None else " (delta n/a)"))
    print(f"latest date    : {report['latest_before']} -> {report['latest_after']}")
    print(f"wrote          : {REPORT.relative_to(REPO)}"
          + (f"  — flagged: {report['flagged']}" if report["flagged"] else "  — no drift flags"))


if __name__ == "__main__":
    main()
