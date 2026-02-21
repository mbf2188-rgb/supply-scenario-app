import pandas as pd

from lib.constants import ANNUALIZATION_FACTOR
from lib.metrics import changed_sites_only, delta_vs_baseline
from lib.normalize import derive_volumes


def test_unit_conversions_42_gal_and_30_day_month():
    df = pd.DataFrame({"Daily Volume (gal)": [84.0]})
    out, note = derive_volumes(df)
    assert out["Daily Volume (bbl)"].iloc[0] == 2.0
    assert out["Monthly Volume (gal 30 day)"].iloc[0] == 2520.0
    assert out["Monthly Volume (bbl 30 day)"].iloc[0] == 60.0
    assert "Derived" in note


def test_changed_sites_only_logic():
    df = pd.DataFrame(
        [
            {"Scenario": "X", "Site ID": "1", "Product Group": "Regular", "New TCN": "A"},
            {"Scenario": "Y", "Site ID": "1", "Product Group": "Regular", "New TCN": "B"},
            {"Scenario": "X", "Site ID": "2", "Product Group": "Regular", "New TCN": "C"},
            {"Scenario": "Y", "Site ID": "2", "Product Group": "Regular", "New TCN": "C"},
        ]
    )
    out = changed_sites_only(df, "X", "Y")
    assert set(out["Site ID"]) == {"1"}


def test_delta_vs_baseline_logic():
    df = pd.DataFrame(
        [
            {"Scenario": "Today", "Site ID": "1", "Product Group": "Regular", "Total Cost (30 days)": 100.0},
            {"Scenario": "Future", "Site ID": "1", "Product Group": "Regular", "Total Cost (30 days)": 70.0},
        ]
    )
    out = delta_vs_baseline(df, "Today", "Future")
    assert out["delta_30"].iloc[0] == -30.0
    assert out["delta_1y"].iloc[0] == -30.0 * ANNUALIZATION_FACTOR
