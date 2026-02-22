import pandas as pd

from lib.constants import ANNUALIZATION_FACTOR
from lib.metrics import changed_sites_only, delta_vs_baseline, impacted_sites_compare, impacted_sites_overview
from lib.normalize import derive_volumes


def test_unit_conversions_42_gal_and_30_day_month():
    df = pd.DataFrame({"Daily Volume (gal)": [84.0]})
    out, note = derive_volumes(df)
    assert out["Daily Volume (bbl)"].iloc[0] == 2.0
    assert out["Monthly Volume (gal 30 day)"].iloc[0] == 2520.0
    assert out["Monthly Volume (bbl 30 day)"].iloc[0] == 60.0
    assert "Derived" in note


def test_changed_sites_only_logic_terminal_diff():
    df = pd.DataFrame(
        [
            {"Scenario": "Today", "Site ID": "1", "Product Group": "Regular", "New Terminal": "A"},
            {"Scenario": "Future", "Site ID": "1", "Product Group": "Regular", "New Terminal": "B"},
            {"Scenario": "Today", "Site ID": "2", "Product Group": "Regular", "New Terminal": "C"},
            {"Scenario": "Future", "Site ID": "2", "Product Group": "Regular", "New Terminal": "C"},
        ]
    )
    out = changed_sites_only(df, "Today", "Future")
    assert set(out["Site ID"]) == {"1"}


def test_impacted_sites_unique_site_count_compare_and_overview():
    compare_df = pd.DataFrame(
        [
            {"Scenario": "Today", "Site ID": "10", "Product Group": "Regular", "New Terminal": "T1"},
            {"Scenario": "Future", "Site ID": "10", "Product Group": "Regular", "New Terminal": "T2"},
            {"Scenario": "Today", "Site ID": "10", "Product Group": "Diesel", "New Terminal": "T1"},
            {"Scenario": "Future", "Site ID": "10", "Product Group": "Diesel", "New Terminal": "T3"},
        ]
    )
    assert impacted_sites_compare(compare_df, "Today", "Future") == 1

    overview_df = pd.DataFrame(
        [
            {"Site ID": "20", "Product Group": "Regular", "Home Terminal": "H1", "New Terminal": "N1"},
            {"Site ID": "20", "Product Group": "Diesel", "Home Terminal": "H1", "New Terminal": "N2"},
        ]
    )
    assert impacted_sites_overview(overview_df) == 1


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
