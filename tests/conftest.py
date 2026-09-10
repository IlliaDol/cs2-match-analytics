from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def toy_matches_path() -> Path:
    return Path(__file__).parent.parent / "data" / "raw" / "toy_matches.csv"


@pytest.fixture()
def toy_matches(toy_matches_path) -> pd.DataFrame:
    return pd.read_csv(toy_matches_path)
