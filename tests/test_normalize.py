import pandas as pd

from lib.normalize import normalize_dataset


def test_normalization_whitespace_and_typo_columns():
    df = pd.DataFrame(
        {
            "Home Terminal ": ["A"],
            "Monthyl Volume (30 day b)": [300.0],
            "Scenario": ["Today"],
            "Site ID": [1],
            "Product Group": ["Regular"],
            "New Terminal": ["B"],
            "New TCN": ["T1"],
            "Freight Cost (30 days)": [1.0],
            "Product Cost (30 days)": [2.0],
            "Total Cost (30 days)": [3.0],
        }
    )
    out = normalize_dataset(df)
    assert "Home Terminal" in out.df.columns
    assert "Monthly Volume (bbl 30 day)" in out.df.columns


def test_required_columns_enforcement():
    df = pd.DataFrame({"Scenario": ["Today"], "Site ID": [1]})
    out = normalize_dataset(df)
    assert "Product Group" in out.missing_required
