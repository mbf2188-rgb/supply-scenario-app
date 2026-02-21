from __future__ import annotations

import pandas as pd

from lib.constants import ANNUALIZATION_FACTOR


def impacted_sites(df: pd.DataFrame) -> int:
    if {"Home Terminal", "New Terminal", "Site ID"}.issubset(df.columns):
        return int(df.loc[df["Home Terminal"] != df["New Terminal"], "Site ID"].nunique())
    return 0


def totals(df: pd.DataFrame) -> dict:
    freight30 = float(df.get("Freight Cost (30 days)", pd.Series(dtype=float)).fillna(0).sum())
    supply30 = float(df.get("Product Cost (30 days)", pd.Series(dtype=float)).fillna(0).sum())
    total30 = float(df.get("Total Cost (30 days)", pd.Series(dtype=float)).fillna(0).sum())
    return {
        "freight_30": freight30,
        "supply_30": supply30,
        "total_30": total30,
        "freight_1y": freight30 * ANNUALIZATION_FACTOR,
        "supply_1y": supply30 * ANNUALIZATION_FACTOR,
        "total_1y": total30 * ANNUALIZATION_FACTOR,
    }


def delta_vs_baseline(df: pd.DataFrame, baseline: str, selected: str) -> pd.DataFrame:
    gcols = ["Site ID", "Product Group"]
    sel = (
        df[df["Scenario"] == selected]
        .groupby(gcols, as_index=False)["Total Cost (30 days)"]
        .sum()
        .rename(columns={"Total Cost (30 days)": "selected_total_30"})
    )
    base = (
        df[df["Scenario"] == baseline]
        .groupby(gcols, as_index=False)["Total Cost (30 days)"]
        .sum()
        .rename(columns={"Total Cost (30 days)": "baseline_total_30"})
    )
    out = sel.merge(base, how="outer", on=gcols).fillna(0)
    out["delta_30"] = out["selected_total_30"] - out["baseline_total_30"]
    out["delta_1y"] = out["delta_30"] * ANNUALIZATION_FACTOR
    return out.sort_values("delta_30")


def changed_sites_only(df: pd.DataFrame, scenario_x: str, scenario_y: str) -> pd.DataFrame:
    cols = ["Site ID", "Product Group", "New TCN"]
    left = df[df["Scenario"] == scenario_x][cols].rename(columns={"New TCN": "x_tcn"})
    right = df[df["Scenario"] == scenario_y][cols].rename(columns={"New TCN": "y_tcn"})
    merged = left.merge(right, on=["Site ID", "Product Group"], how="inner")
    changed_keys = merged[merged["x_tcn"] != merged["y_tcn"]][["Site ID", "Product Group"]].drop_duplicates()
    return df.merge(changed_keys, on=["Site ID", "Product Group"], how="inner")


def terminal_shift_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Home Terminal", "New Terminal", "Monthly Volume (bbl 30 day)"}.issubset(df.columns):
        return pd.DataFrame()
    return pd.pivot_table(
        df,
        index="Home Terminal",
        columns="New Terminal",
        values="Monthly Volume (bbl 30 day)",
        aggfunc="sum",
        fill_value=0,
    )


def volume_by_terminal_product(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    if not {"New Terminal", "Product Group", volume_col}.issubset(df.columns):
        return pd.DataFrame()
    return pd.pivot_table(
        df,
        index="New Terminal",
        columns="Product Group",
        values=volume_col,
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
