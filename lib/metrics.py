from __future__ import annotations

import pandas as pd

from lib.constants import ANNUALIZATION_FACTOR


def impacted_sites_overview(df: pd.DataFrame) -> int:
    if {"Home Terminal", "New Terminal", "Site ID"}.issubset(df.columns):
        impacted = df[df["Home Terminal"].astype(str) != df["New Terminal"].astype(str)]["Site ID"]
        return int(impacted.astype(str).nunique())
    return 0


def impacted_sites_compare(df: pd.DataFrame, baseline: str, new_scenario: str) -> int:
    if not {"Scenario", "Site ID", "Product Group", "New Terminal"}.issubset(df.columns):
        return 0

    cols = ["Site ID", "Product Group", "New Terminal"]
    left = df[df["Scenario"].astype(str) == baseline][cols].rename(columns={"New Terminal": "base_terminal"})
    right = df[df["Scenario"].astype(str) == new_scenario][cols].rename(columns={"New Terminal": "new_terminal"})
    merged = left.merge(right, on=["Site ID", "Product Group"], how="inner")
    changed_site_ids = merged[merged["base_terminal"].astype(str) != merged["new_terminal"].astype(str)]["Site ID"]
    return int(changed_site_ids.astype(str).nunique())


def changed_sites_only(df: pd.DataFrame, baseline: str, new_scenario: str) -> pd.DataFrame:
    if not {"Scenario", "Site ID", "Product Group", "New Terminal"}.issubset(df.columns):
        return df.iloc[0:0].copy()

    cols = ["Site ID", "Product Group", "New Terminal"]
    left = df[df["Scenario"].astype(str) == baseline][cols].rename(columns={"New Terminal": "base_terminal"})
    right = df[df["Scenario"].astype(str) == new_scenario][cols].rename(columns={"New Terminal": "new_terminal"})
    merged = left.merge(right, on=["Site ID", "Product Group"], how="inner")
    changed = merged[merged["base_terminal"].astype(str) != merged["new_terminal"].astype(str)]
    changed_sites = changed["Site ID"].astype(str).drop_duplicates()
    return df[df["Site ID"].astype(str).isin(changed_sites)].copy()


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
        df[df["Scenario"].astype(str) == selected]
        .groupby(gcols, as_index=False)["Total Cost (30 days)"]
        .sum()
        .rename(columns={"Total Cost (30 days)": "selected_total_30"})
    )
    base = (
        df[df["Scenario"].astype(str) == baseline]
        .groupby(gcols, as_index=False)["Total Cost (30 days)"]
        .sum()
        .rename(columns={"Total Cost (30 days)": "baseline_total_30"})
    )
    out = sel.merge(base, how="outer", on=gcols).fillna(0)
    out["delta_30"] = out["selected_total_30"] - out["baseline_total_30"]
    out["delta_1y"] = out["delta_30"] * ANNUALIZATION_FACTOR
    return out.sort_values("delta_30")


def delta_totals(df: pd.DataFrame, baseline: str, new_scenario: str) -> dict:
    base = totals(df[df["Scenario"].astype(str) == baseline])
    new = totals(df[df["Scenario"].astype(str) == new_scenario])
    return {
        "delta_total_30": new["total_30"] - base["total_30"],
        "delta_total_1y": new["total_1y"] - base["total_1y"],
        "delta_freight_30": new["freight_30"] - base["freight_30"],
        "delta_freight_1y": new["freight_1y"] - base["freight_1y"],
        "delta_supply_30": new["supply_30"] - base["supply_30"],
        "delta_supply_1y": new["supply_1y"] - base["supply_1y"],
    }


def terminal_shift_matrix(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    if not {"Home Terminal", "New Terminal", volume_col}.issubset(df.columns):
        return pd.DataFrame()
    return pd.pivot_table(
        df,
        index="Home Terminal",
        columns="New Terminal",
        values=volume_col,
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


def volume_by_product_tcn(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    if not {"New TCN", "Product Group", volume_col}.issubset(df.columns):
        return pd.DataFrame()
    return pd.pivot_table(
        df,
        index="Product Group",
        columns="New TCN",
        values=volume_col,
        aggfunc="sum",
        fill_value=0,
    ).reset_index()


def delta_by_group(df: pd.DataFrame, baseline: str, new_scenario: str, group_cols: list[str], value_col: str) -> pd.DataFrame:
    required = {"Scenario", *group_cols, value_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    base = (
        df[df["Scenario"].astype(str) == baseline]
        .groupby(group_cols, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "baseline_value"})
    )
    new = (
        df[df["Scenario"].astype(str) == new_scenario]
        .groupby(group_cols, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "new_value"})
    )
    out = base.merge(new, on=group_cols, how="outer").fillna(0)
    out["delta_30"] = out["new_value"] - out["baseline_value"]
    out["delta_1y"] = out["delta_30"] * ANNUALIZATION_FACTOR
    return out.sort_values("delta_30", ascending=False)
