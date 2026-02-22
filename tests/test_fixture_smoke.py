from pathlib import Path

import pandas as pd

from lib.normalize import normalize_dataset


def test_fixture_load_and_normalize():
    fixture = Path("tests/fixtures/testdata.csv")
    df = pd.read_csv(fixture)
    out = normalize_dataset(df)
    assert not out.missing_required
    assert {"Scenario", "Site ID", "Product Group"}.issubset(out.df.columns)
