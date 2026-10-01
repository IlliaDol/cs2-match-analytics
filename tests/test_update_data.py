"""Live-refresh contract tests (v1.1 3). Local tmp dirs only — no network, no real data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from update_data import digest, drift_report, raw_csvs, refresh_raw  # noqa: E402


def test_raw_csvs_ignores_empty_and_missing_dirs(tmp_path: Path):
    assert raw_csvs(tmp_path / "nope") == []
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "real.csv").write_text("a,b\n1,2\n")
    assert [p.name for p in raw_csvs(tmp_path)] == ["real.csv"]


def test_refresh_raw_raises_without_url_or_local_data(tmp_path: Path):
    """It must fail loudly instead of letting a stale dataset look refreshed."""
    with pytest.raises(FileNotFoundError, match="no raw CSVs"):
        refresh_raw(raw_dir=tmp_path, url=None)
    assert list(tmp_path.glob("*")) == []          # nothing was fabricated


def test_refresh_raw_uses_local_csvs(tmp_path: Path):
    (tmp_path / "cs2_all_tiers_games.csv").write_text("a,b\n1,2\n")
    files = refresh_raw(raw_dir=tmp_path, url=None)
    assert [p.name for p in files] == ["cs2_all_tiers_games.csv"]


def _store(path: Path, rows: int, latest: str, mean: float, std: float = 1.0) -> None:
    frame = pd.DataFrame(
        {
            "datetime": [latest] * rows,
            "elo_diff": [mean] * rows,
            "form5_diff": [0.0] * rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    # a constant column has std 0 in pandas' ddof=1; the digest uses ddof=0, so that is 0 too —
    # shift flaging then falls back to a 1.0 scale (see SHIFT_FLAG_SIGMA handling).


def test_digest_reads_rows_latest_and_stats(tmp_path: Path):
    store = tmp_path / "features.parquet"
    _store(store, rows=10, latest="2026-05-01", mean=12.0)
    d = digest(store)
    assert d["exists"] and d["rows"] == 10 and d["latest"] == "2026-05-01"
    assert d["stats"]["elo_diff"]["mean"] == pytest.approx(12.0)


def test_digest_of_missing_store(tmp_path: Path):
    assert digest(tmp_path / "absent.parquet") == {"exists": False}


def test_drift_report_flags_row_loss_and_stale_latest(tmp_path: Path):
    before_store, after_store = tmp_path / "b.parquet", tmp_path / "a.parquet"
    _store(before_store, rows=100, latest="2026-05-01", mean=10.0, std=2.0)
    _store(after_store, rows=90, latest="2026-05-01", mean=10.2, std=2.0)

    report = drift_report(digest(before_store), digest(after_store))

    assert report["rows_delta"] == -10
    assert any("rows fell by 10" in f for f in report["flagged"])
    assert any("did not advance" in f for f in report["flagged"])


def test_drift_report_flags_a_refresh_that_adds_nothing(tmp_path: Path):
    """Same data in, same data out: the one thing worth saying is that the latest
    date did not advance — a refresh that brought no new series."""
    store = tmp_path / "same.parquet"
    _store(store, rows=50, latest="2026-06-01", mean=5.0, std=1.0)
    report = drift_report(digest(store), digest(store))
    assert report["rows_delta"] == 0
    assert report["flagged"] == ["latest date did not advance (2026-06-01)"]


def test_drift_report_quiet_when_new_data_advances(tmp_path: Path):
    before_store, after_store = tmp_path / "b.parquet", tmp_path / "a.parquet"
    _store(before_store, rows=50, latest="2026-06-01", mean=5.0, std=1.0)
    _store(after_store, rows=64, latest="2026-07-01", mean=5.05, std=1.0)
    report = drift_report(digest(before_store), digest(after_store))
    assert report["rows_delta"] == 14
    assert report["flagged"] == []
