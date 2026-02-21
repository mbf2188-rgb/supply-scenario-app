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

def apply_in(df_in: pd.DataFrame, col: str, selected: list):
    if selected and col in df_in.columns:
        return df_in[df_in[col].isin(selected)]
    return df_in

if not uploaded_file:
    st.info("Upload a CSV or XLSX to begin.")
    st.stop()

df = _load_df(uploaded_file)
st.success(f"Loaded {len(df):,} rows")

# Column aliasing (map stays working if names vary)
lat_col = _first_existing(df, ["Latitude", "Lat", "LAT"])
lon_col = _first_existing(df, ["Longitude", "Lon", "Lng", "Long", "LON"])

# --- Required columns (per your spec)
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

if missing:
    st.error(f"Schema warning: Missing required columns: {missing}")

# Put “unexpected columns” behind an expander to avoid noisy banners
with st.expander("Schema details", expanded=False):
    st.write("Required columns:", required)
    st.write("Detected columns:", list(df.columns))
    if lat_col and lon_col:
        st.write(f"Detected lat/lon columns: {lat_col}, {lon_col}")
    else:
        st.write("Detected lat/lon columns: NOT FOUND (accepted: Latitude/Lat and Longitude/Lon/Lng/Long)")

# --- Sidebar global controls
st.sidebar.header("Global Controls")

search_q = st.sidebar.text_input("Global search (filters everything)", value="", key="global_search").strip()

scenarios = sorted(df["Scenario"].dropna().unique().tolist()) if "Scenario" in df.columns else []
scenario = st.sidebar.selectbox("Scenario", scenarios, index=0 if scenarios else None)

# REQUIRED filters (per your latest instruction)
# Remove old/home terminal filters.
# Use ONLY New Terminal and New TCN, renamed to Assigned.
st.sidebar.subheader("Filters")

sel_assigned_terminal = _safe_multiselect(
    "Assigned Terminal",
    df["New Terminal"] if "New Terminal" in df.columns else pd.Series([], dtype=object),
    "f_assigned_terminal",
)
sel_assigned_tcn = _safe_multiselect(
    "Assigned TCN",
    df["New TCN"] if "New TCN" in df.columns else pd.Series([], dtype=object),
    "f_assigned_tcn",
)

# Keep other useful minimum filters from your spec
sel_site_id    = _safe_multiselect("Site ID", df["Site ID"] if "Site ID" in df.columns else pd.Series([], dtype=object), "f_site_id")
sel_prod_group = _safe_multiselect("Product Group", df["Product Group"] if "Product Group" in df.columns else pd.Series([], dtype=object), "f_prod_group")

has_brand = "Brand" in df.columns
has_supplier = "Supplier" in df.columns
sel_brand = _safe_multiselect("Brand", df["Brand"], "f_brand") if has_brand else []
sel_supplier = _safe_multiselect("Supplier", df["Supplier"], "f_supplier") if has_supplier else []

# --- Apply scenario filter first
df_f = df.copy()
if scenario and "Scenario" in df_f.columns:
    df_f = df_f[df_f["Scenario"] == scenario]

# --- Apply filters
df_f = apply_in(df_f, "New Terminal", sel_assigned_terminal)
df_f = apply_in(df_f, "New TCN", sel_assigned_tcn)
df_f = apply_in(df_f, "Site ID", sel_site_id)
df_f = apply_in(df_f, "Product Group", sel_prod_group)
if has_brand:
    df_f = apply_in(df_f, "Brand", sel_brand)
if has_supplier:
    df_f = apply_in(df_f, "Supplier", sel_supplier)

# --- Apply global search across key fields
if search_q:
    search_cols = [c for c in ["Site ID", "Product Group", "Brand", "Supplier", "Home Terminal", "New Terminal", "Home TCN", "New TCN", "Scenario"] if c in df_f.columns]
    blob = df_f[search_cols].astype(str).agg(" | ".join, axis=1)
    df_f = df_f[blob.str.contains(search_q, case=False, na=False)]

st.caption(f"Filtered rows: {len(df_f):,}")

# --- KPIs (30d + 1y). Freight/Supply only if columns exist.
kpi_total_30 = df_f["Total Cost (30 days)"].sum() if "Total Cost (30 days)" in df_f.columns else np.nan
kpi_total_1y = kpi_total_30 * (365 / 30) if np.isfinite(kpi_total_30) else np.nan

kpi_freight_30 = df_f["Freight Cost (30 days)"].sum() if "Freight Cost (30 days)" in df_f.columns else None
kpi_supply_30  = df_f["Product Cost (30 days)"].sum() if "Product Cost (30 days)" in df_f.columns else None

impacted_sites = None
if "Home Terminal" in df_f.columns and "New Terminal" in df_f.columns and "Site ID" in df_f.columns:
    impacted_sites = df_f.loc[df_f["Home Terminal"] != df_f["New Terminal"], "Site ID"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Cost (30d)", f"${kpi_total_30:,.0f}" if np.isfinite(kpi_total_30) else "N/A")
c2.metric("Total Cost (1y)", f"${kpi_total_1y:,.0f}" if np.isfinite(kpi_total_1y) else "N/A")
c3.metric("Freight (30d)", f"${kpi_freight_30:,.0f}" if isinstance(kpi_freight_30, (int, float, np.floating)) else "N/A")
c4.metric("Supply (30d)", f"${kpi_supply_30:,.0f}" if isinstance(kpi_supply_30, (int, float, np.floating)) else "N/A")
c5.metric("Impacted Sites", f"{impacted_sites:,}" if isinstance(impacted_sites, int) else "N/A")

# --- Map: one marker per site, colored by Assigned Terminal (New Terminal)
st.subheader("Map (sites colored by Assigned Terminal)")

if lat_col and lon_col and (lat_col in df_f.columns) and (lon_col in df_f.columns):
    df_map_raw = df_f.dropna(subset=[lat_col, lon_col]).copy()

    if len(df_map_raw) == 0:
        st.info("No mappable rows after filtering (missing lat/lon).")
    else:
        # Aggregate to one marker per Site ID
        # If a site has multiple assigned terminals across product groups, label as "Multiple"
        def _assigned_terminal_for_site(s: pd.Series) -> str:
            vals = s.dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                return vals[0]
            if len(vals) == 0:
                return ""
            return "Multiple"

        agg_dict = {
            lat_col: "first",
            lon_col: "first",
            "New Terminal": _assigned_terminal_for_site if "New Terminal" in df_map_raw.columns else "first",
        }
        if "Brand" in df_map_raw.columns:
            agg_dict["Brand"] = "first"

        df_sites = df_map_raw.groupby("Site ID", as_index=False).agg(agg_dict)

        center_lat = float(df_sites[lat_col].mean())
        center_lon = float(df_sites[lon_col].mean())

        fig = px.scatter_map(
            df_sites,
            lat=lat_col,
            lon=lon_col,
            color="New Terminal" if "New Terminal" in df_sites.columns else None,
            hover_name="Site ID",
            hover_data=[c for c in ["Brand", "New Terminal"] if c in df_sites.columns],
            zoom=4,
            center={"lat": center_lat, "lon": center_lon},
        )
        # Bigger markers, stronger visibility
        fig.update_traces(marker={"size": 16, "opacity": 0.95})
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})

        st.plotly_chart(fig, width="stretch")

        # Site details panel (reliable). No lat/lon shown.
        st.sidebar.subheader("Site Details")
        site_options = sorted(df_map_raw["Site ID"].dropna().unique().tolist())
        site_pick = st.sidebar.selectbox("Select a site", options=[""] + site_options, index=0, key="site_pick")

        if site_pick:
            site_rows = df_map_raw[df_map_raw["Site ID"] == site_pick].copy()

            # Build a clean “card” with required fields (no lat/lon)
            card_fields = []
            for field in ["Site ID", "Brand", "Home Terminal", "New Terminal", "Supplier", "Home TCN", "New TCN"]:
                if field in site_rows.columns:
                    val = site_rows[field].dropna().astype(str).unique().tolist()
                    if len(val) == 1:
                        card_fields.append((field, val[0]))
                    elif len(val) > 1:
                        card_fields.append((field, "Multiple"))

            st.sidebar.markdown("### Details")
            for k, v in card_fields:
                label = k
                # Rename displayed labels per your request
                if k == "New Terminal":
                    label = "Assigned Terminal"
                if k == "New TCN":
                    label = "Assigned TCN"
                st.sidebar.write(f"**{label}:** {v}")

            st.sidebar.markdown("### Product Breakdown")
            cols = [c for c in ["Product Group", "Total Cost (30 days)", "Freight Cost (30 days)", "Product Cost (30 days)"] if c in site_rows.columns]
            # show per product group rows
            if cols:
                breakdown = site_rows[cols].copy()
                st.sidebar.dataframe(breakdown, width="stretch", height=220)
else:
    st.info("Map disabled: Latitude/Longitude columns not found (accepted: Latitude/Lat and Longitude/Lon/Lng/Long).")

# --- Data table (filtered) — remove lat/lon from display
st.subheader("Filtered Data")
df_display = df_f.drop(columns=[c for c in [lat_col, lon_col] if c], errors="ignore")
st.dataframe(df_display, width="stretch", height=520)
