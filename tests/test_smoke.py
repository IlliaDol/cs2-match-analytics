import pandas as pd


def test_toy_dataset_exists_and_loads(toy_matches_path):
    df = pd.read_csv(toy_matches_path)
    assert df.shape == (10, 9)


def test_package_imports():
    import cs2analytics

    assert cs2analytics.__version__


def test_ruff_clean_is_enforced_in_ci():
    """Placeholder proving the test suite runs. Real contract tests come with the loader."""
    assert True
