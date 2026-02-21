import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Supply + Freight Scenario Viewer")

st.title("Supply + Freight Scenario Viewer")

uploaded_file = st.file_uploader("Upload scenario file (.csv or .xlsx)", type=["csv", "xlsx"])

def _load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        df0 = pd.read_csv(file)
    else:
        df0 = pd.read_excel(file, sheet_name=0, engine="openpyxl")
    df0.columns = [str(c).strip() for c in df0.columns]
    return df0

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_multiselect(label: str, series: pd.Series, key: str):
    opts = sorted([x for x in series.dropna().unique().tolist()])
    return st.sidebar.multiselect(label, options=opts, default=[], key=key)

if not uploaded_file:
    st.info("Upload a CSV or XLSX to begin.")
    st.stop()

df = _load_df(uploaded_file)

st.success(f"Loaded {len(df):,} rows")

# --- Column mapping / aliases (keeps your map working even if the file uses Lat/Lon etc.)
lat_col = _first_existing(df, ["Latitude", "Lat", "LAT"])
lon_col = _first_existing(df, ["Longitude", "Lon", "Lng", "Long", "LON"])

# --- Required columns (MVP)
required = [
    "Scenario",
    "Site ID",
    "Product Group",
    "Home Terminal",
    "New Terminal",
    "Home TCN",
    "New TCN",
    "Total Cost (30 days)",
]
missing = [c for c in required if c not in df.columns]

# Schema warning banner (continue when possible)
if missing:
    st.error(f"Schema warning: Missing required columns: {missing}")

unexpected = [c for c in df.columns if c not in (set(required) | {"Brand", "Supplier", "Freight Cost (30 days)", "Product Cost (30 days)", lat_col, lon_col})]
if unexpected:
    st.warning(f"Schema note: Unexpected columns present (not an error): {unexpected[:50]}{' ...' if len(unexpected) > 50 else ''}")

# --- Sidebar global controls
st.sidebar.header("Global Filters")

search_q = st.sidebar.text_input("Global search (filters everything)", value="", key="global_search").strip()

if "Scenario" in df.columns:
    scenarios = sorted(df["Scenario"].dropna().unique().tolist())
else:
    scenarios = []

scenario = st.sidebar.selectbox("Scenario (applies to all views for now)", scenarios, index=0 if scenarios else None)

# Optional columns
has_brand = "Brand" in df.columns
has_supplier = "Supplier" in df.columns

# Multiselect filters (minimum set required by your spec)
sel_home_terminal = _safe_multiselect("Old Terminal (Home Terminal)", df["Home Terminal"] if "Home Terminal" in df.columns else pd.Series([], dtype=object), "f_home_terminal")
sel_new_terminal  = _safe_multiselect("New Terminal", df["New Terminal"] if "New Terminal" in df.columns else pd.Series([], dtype=object), "f_new_terminal")
sel_home_tcn      = _safe_multiselect("Old TCN (Home TCN)", df["Home TCN"] if "Home TCN" in df.columns else pd.Series([], dtype=object), "f_home_tcn")
sel_new_tcn       = _safe_multiselect("New TCN", df["New TCN"] if "New TCN" in df.columns else pd.Series([], dtype=object), "f_new_tcn")
sel_site_id       = _safe_multiselect("Site ID", df["Site ID"] if "Site ID" in df.columns else pd.Series([], dtype=object), "f_site_id")
sel_prod_group    = _safe_multiselect("Product Group", df["Product Group"] if "Product Group" in df.columns else pd.Series([], dtype=object), "f_prod_group")

sel_brand = []
if has_brand:
    sel_brand = _safe_multiselect("Brand", df["Brand"], "f_brand")

sel_supplier = []
if has_supplier:
    sel_supplier = _safe_multiselect("Supplier", df["Supplier"], "f_supplier")

# --- Apply scenario filter
df_f = df.copy()
if scenario and "Scenario" in df_f.columns:
    df_f = df_f[df_f["Scenario"] == scenario]

# --- Apply multiselect filters
def apply_in(df_in: pd.DataFrame, col: str, selected: list):
    if selected and col in df_in.columns:
        return df_in[df_in[col].isin(selected)]
    return df_in

df_f = apply_in(df_f, "Home Terminal", sel_home_terminal)
df_f = apply_in(df_f, "New Terminal", sel_new_terminal)
df_f = apply_in(df_f, "Home TCN", sel_home_tcn)
df_f = apply_in(df_f, "New TCN", sel_new_tcn)
df_f = apply_in(df_f, "Site ID", sel_site_id)
df_f = apply_in(df_f, "Product Group", sel_prod_group)
if has_brand:
    df_f = apply_in(df_f, "Brand", sel_brand)
if has_supplier:
    df_f = apply_in(df_f, "Supplier", sel_supplier)

# --- Apply global search (across key fields)
if search_q:
    search_cols = [c for c in ["Site ID", "Product Group", "Brand", "Supplier", "Home Terminal", "New Terminal", "Home TCN", "New TCN", "Scenario"] if c in df_f.columns]
    blob = df_f[search_cols].astype(str).agg(" | ".join, axis=1)
    df_f = df_f[blob.str.contains(search_q, case=False, na=False)]

st.caption(f"Filtered rows: {len(df_f):,}")

# --- Compute KPIs (30d + 1y)
kpi_total_30 = df_f["Total Cost (30 days)"].sum() if "Total Cost (30 days)" in df_f.columns else np.nan
kpi_total_1y = kpi_total_30 * (365/30) if np.isfinite(kpi_total_30) else np.nan

kpi_freight_30 = df_f["Freight Cost (30 days)"].sum() if "Freight Cost (30 days)" in df_f.columns else None
kpi_supply_30  = df_f["Product Cost (30 days)"].sum() if "Product Cost (30 days)" in df_f.columns else None

impacted_sites = None
if "Home Terminal" in df_f.columns and "New Terminal" in df_f.columns and "Site ID" in df_f.columns:
    impacted_sites = df_f.loc[df_f["Home Terminal"] != df_f["New Terminal"], "Site ID"].nunique()

# --- KPI strip (expanded)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Cost (30d)", f"${kpi_total_30:,.0f}" if np.isfinite(kpi_total_30) else "N/A")
c2.metric("Total Cost (1y)", f"${kpi_total_1y:,.0f}" if np.isfinite(kpi_total_1y) else "N/A")
c3.metric("Freight (30d)", f"${kpi_freight_30:,.0f}" if isinstance(kpi_freight_30, (int, float, np.floating)) else "N/A")
c4.metric("Supply (30d)", f"${kpi_supply_30:,.0f}" if isinstance(kpi_supply_30, (int, float, np.floating)) else "N/A")
c5.metric("Impacted Sites", f"{impacted_sites:,}" if isinstance(impacted_sites, int) else "N/A")

# --- Map (bigger markers + stronger color contrast)
st.subheader("Map (colored by New Terminal)")

if lat_col and lon_col and (lat_col in df_f.columns) and (lon_col in df_f.columns):
    df_map = df_f.dropna(subset=[lat_col, lon_col]).copy()

    if len(df_map) == 0:
        st.info("No mappable rows after filtering (missing lat/lon).")
    else:
        # Auto-center to filtered extent
        center_lat = float(df_map[lat_col].mean())
        center_lon = float(df_map[lon_col].mean())

        # Larger markers; Plotly Express uses marker size in px
        fig = px.scatter_map(
            df_map,
            lat=lat_col,
            lon=lon_col,
            color="New Terminal" if "New Terminal" in df_map.columns else None,
            hover_name="Site ID" if "Site ID" in df_map.columns else None,
            hover_data=[c for c in ["Product Group", "Brand", "Home Terminal", "New Terminal", "Home TCN", "New TCN"] if c in df_map.columns],
            zoom=4,
            center={"lat": center_lat, "lon": center_lon},
        )
        fig.update_traces(marker={"size": 14, "opacity": 0.9})  # ~2x bigger than default-feel
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

        st.plotly_chart(fig, width="stretch")
else:
    st.info("Map disabled: Latitude/Longitude columns not found (accepted: Latitude/Lat and Longitude/Lon/Lng/Long).")

# --- Data table (filtered)
st.subheader("Filtered Data (preview)")
st.dataframe(df_f, width="stretch", height=520)
