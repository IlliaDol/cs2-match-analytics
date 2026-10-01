import os
from pathlib import Path

import pandas as pd
import pytest

# Keep the numpy / PyMC stack single-threaded. On a memory-tight machine a
# multi-threaded OpenBLAS fails to allocate its buffers ("OpenBLAS error:
# Memory allocation still failed after 10 retries, giving up."), which kills the
# PyMC worker inside test_model_fitting and surfaces as BrokenProcessPool or
# EOFError instead of a model failure. One thread keeps the suite green at
# ~1.5 GB free RAM (verified 2026-09-14: 1 failed -> 137 passed under the same
# pressure). setdefault() lets an explicit environment value win.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture()
def toy_matches_path() -> Path:
    return Path(__file__).parent.parent / "data" / "raw" / "toy_matches.csv"


@pytest.fixture()
def toy_matches(toy_matches_path) -> pd.DataFrame:
    return pd.read_csv(toy_matches_path)
