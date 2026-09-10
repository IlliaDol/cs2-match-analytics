"""M9 artifact tests — Spark notebook + timing table. Skips until artifacts exist."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
NB = REPO / "notebooks" / "05_spark_features.ipynb"
TIMING = REPO / "outputs" / "m9_spark_vs_pandas.csv"


def test_spark_notebook_present_and_valid():
    if not NB.exists():
        pytest.skip("notebooks/05_spark_features.ipynb not produced yet (M9 §1)")
    raw = json.loads(NB.read_text(encoding="utf-8"))
    assert raw.get("cells"), "notebook has no cells"
    text = " ".join("".join(c.get("source", [])) for c in raw["cells"])
    assert "SparkSession" in text, "notebook must create a SparkSession"
    assert "partitionBy" in text, "notebook must use a partitioned window"
    assert "kafka" in text.lower() or "stream" in text.lower(), (
        "notebook must discuss batch vs streaming (M9 §1.5)"
    )


def test_timing_table_contract():
    if not TIMING.exists():
        pytest.skip("outputs/m9_spark_vs_pandas.csv not produced yet (M9 §1.4)")
    import pandas as pd

    df = pd.read_csv(TIMING)
    assert {"scale_rows", "pandas_sec", "spark_sec", "speedup"} <= set(df.columns)
    assert len(df) >= 3
    assert df["scale_rows"].is_monotonic_increasing
    assert (df[["pandas_sec", "spark_sec"]] > 0).all().all()
    ratio = df["pandas_sec"] / df["spark_sec"]
    assert (abs(ratio - df["speedup"]) < 1e-6).all(), "speedup must equal pandas_sec/spark_sec"
