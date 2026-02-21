# app.py
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Scenario Viewer", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
REQUIRED_BASE_COLS = ["Scenario", "Site ID", "Product Group", "Assigned Terminal", "Assigned TCN", "Lat", "Lon"]

def _norm_col(c: str) -> str:
    return str(c).strip()

def _coerce_str(s):
    return s.astype(str).str.strip()

def _safe_contains(series: pd.Series, needle: str) -> pd.Series:
    if not needle:
        return pd.Series(True, index=series.index)
    return series.fillna("").astype(str).str.contains(needle, case=False, regex=False)

def _mapbox_zoom_from_bounds(lat_min, lat_max, lon_min, lon_max) -> float:
    lat_span = max(1e-6, float(lat_max - lat_min))
    lon_span = max(1e-6, float(lon_max - lon_min))
    span = max(lat_span, lon_span)
    zoom = np.log2(360.0 / span) - 1.5
    return float(np.clip(zoom, 1.0, 15.5))

def _apply_product_offsets(df_in: pd.DataFrame, lat_col="Lat", lon_col="Lon", group_col="Product Group") -> pd.DataFrame:
    df = df_in.copy()
    codes = pd.Categorical(df[group_col]).codes.astype(float)
    r = 0.0006
    ang = (codes % 12) * (2.0 * np.pi / 12.0)
    df["lat_plot"] = df[lat_col].astype(float) + r * np.sin(ang)
    df["lon_plot"] = df[lon_col].astype(float) + r * np.cos(ang)
    return df

@st.cache_data(show_spinner=False)
def read_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        # Handle common CSV issues (BOM, delimiters)
        raw = file_bytes
        try:
            return pd.read_csv(io.BytesIO(raw), engine="python")
        except Exception:
            # fallback with utf-8-sig
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    raise ValueError("Unsupported file type (upload .csv or .xlsx)")

def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize core columns
    df["Scenario"] = _coerce_str(df["Scenario"])
    df["Site ID"] = _coerce_str(df["Site ID"])
    df["Product Group"] = _coerce_str(df["Product Group"])
    df["Assigned Terminal"] = _coerce_str(df["Assigned Terminal"])
    df["Assigned TCN"] = _coerce_str(df["Assigned TCN"])

    # Optional columns
    for opt in ["Brand", "Supplier", "Home Terminal", "Home TCN", "New Supplier", "Old Supplier"]:
        if opt in df.columns:
            df[opt] = df[opt].astype(str).str.strip()

    # Numeric lat/lon
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")

    # Helpful derived flags
    if "Home Terminal" in df.columns:
        df["Impacted"] = (df["Home Terminal"].astype(str).str.strip() != df["Assigned Terminal"].astype(str).str.strip())
    else:
        df["Impacted"] = False

    return df

def build_search_blob(df: pd.DataFrame) -> pd.Series:
    cols = ["Scenario", "Site ID", "Product Group", "Assigned Terminal", "Assigned TCN"]
    for c in ["Brand", "Supplier", "Home Terminal"]:
        if c in df.columns:
            cols.append(c)
    blob = df[cols].fillna("").astype(str).agg(" | ".join, axis=1)
    return blob

def apply_filters(
    df: pd.DataFrame,
    scenario: str,
    terminals: list[str],
    tcns: list[str],
    site_ids: list[str],
    products: list[str],
    brands: list[str] | None,
    suppliers: list[str] | None,
    global_search: str,
) -> pd.DataFrame:
    d = df[df["Scenario"] == scenario].copy()

    if terminals:
        d = d[d["Assigned Terminal"].isin(terminals)]
    if tcns:
        d = d[d["Assigned TCN"].isin(tcns)]
    if site_ids:
        d = d[d["Site ID"].isin(site_ids)]
    if products:
        d = d[d["Product Group"].isin(products)]
    if brands is not None and "Brand" in d.columns and brands:
        d = d[d["Brand"].isin(brands)]
    if suppliers is not None and "Supplier" in d.columns and suppliers:
        d = d[d["Supplier"].isin(suppliers)]

    if global_search:
        blob = build_search_blob(d)
        d = d[_safe_contains(blob, global_search)]

    return d

def kpis(df_scenario: pd.DataFrame, df_filtered: pd.DataFrame) -> dict:
    rows = int(len(df_filtered))
    sites = int(df_filtered["Site ID"].nunique()) if not df_filtered.empty else 0
    impacted_sites = int(df_filtered.loc[df_filtered["Impacted"], "Site ID"].nunique()) if "Impacted" in df_filtered.columns else 0

    # Baseline for scenario-level totals (useful for context)
    scen_rows = int(len(df_scenario))
    scen_sites = int(df_scenario["Site ID"].nunique())
    scen_impacted = int(df_scenario.loc[df_scenario["Impacted"], "Site ID"].nunique()) if "Impacted" in df_scenario.columns else 0

    return {
        "rows": rows,
        "sites": sites,
        "impacted_sites": impacted_sites,
        "scenario_rows": scen_rows,
        "scenario_sites": scen_sites,
        "scenario_impacted_sites": scen_impacted,
    }

# -----------------------------
# UI
# -----------------------------
st.title("Fuel Supply Optimization Scenario Viewer")

uploaded = st.file_uploader("Upload a .csv or .xlsx with a Scenario column", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.stop()

try:
    raw_df = read_uploaded(uploaded.getvalue(), uploaded.name)
    df = validate_and_prepare(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Scenario selection
scenarios = sorted(df["Scenario"].dropna().unique().tolist())
scenario = st.selectbox("Scenario", scenarios, index=0)

df_scenario = df[df["Scenario"] == scenario].copy()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Global search
    global_search = st.text_input("Global search", value="", placeholder="Search across key fields...")

    # Options based on scenario subset for speed
    opt_terminals = sorted(df_scenario["Assigned Terminal"].dropna().unique().tolist())
    opt_tcns = sorted(df_scenario["Assigned TCN"].dropna().unique().tolist())
    opt_sites = sorted(df_scenario["Site ID"].dropna().unique().tolist())
    opt_products = sorted(df_scenario["Product Group"].dropna().unique().tolist())

    terminals = st.multiselect("Assigned Terminal", opt_terminals, default=[])
    tcns = st.multiselect("Assigned TCN", opt_tcns, default=[])
    site_ids = st.multiselect("Site ID", opt_sites, default=[])
    products = st.multiselect("Product Group", opt_products, default=[])

    brands = None
    suppliers = None

    if "Brand" in df_scenario.columns:
        opt_brands = sorted(df_scenario["Brand"].dropna().astype(str).unique().tolist())
        brands = st.multiselect("Brand", opt_brands, default=[])

    if "Supplier" in df_scenario.columns:
        opt_suppliers = sorted(df_scenario["Supplier"].dropna().astype(str).unique().tolist())
        suppliers = st.multiselect("Supplier", opt_suppliers, default=[])

    st.divider()
    st.subheader("Map options")
    one_marker_per_site = st.toggle("One marker per Site ID", value=True)
    offsets_enabled = st.toggle("Tiny offsets by Product Group", value=False, disabled=one_marker_per_site)

# Apply filters
df_filtered = apply_filters(
    df=df,
    scenario=scenario,
    terminals=terminals,
    tcns=tcns,
    site_ids=site_ids,
    products=products,
    brands=brands,
    suppliers=suppliers,
    global_search=global_search.strip(),
)

# KPIs (only rows/sites/impacted shown; include scenario totals as subtle reference)
k = kpis(df_scenario, df_filtered)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows (filtered)", f"{k['rows']:,}")
c2.metric("Sites (filtered)", f"{k['sites']:,}")
c3.metric("Impacted Sites (filtered)", f"{k['impacted_sites']:,}")
c4.metric("Rows (scenario)", f"{k['scenario_rows']:,}")
c5.metric("Sites (scenario)", f"{k['scenario_sites']:,}")
c6.metric("Impacted Sites (scenario)", f"{k['scenario_impacted_sites']:,}")

st.divider()

# -----------------------------
# Map + Details layout
# -----------------------------
left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.subheader("Map")

    df_map_src = df_filtered.copy()
    df_map_src["Lat"] = pd.to_numeric(df_map_src["Lat"], errors="coerce")
    df_map_src["Lon"] = pd.to_numeric(df_map_src["Lon"], errors="coerce")
    df_map_src = df_map_src.dropna(subset=["Lat", "Lon"])

    if df_map_src.empty:
        st.info("No mappable rows after filters.")
        clicked = []
    else:
        if one_marker_per_site:
            agg = {
                "Lat": "first",
                "Lon": "first",
                "Assigned Terminal": "first",
                "Assigned TCN": "first",
            }
            for opt in ["Brand", "Supplier", "Home Terminal"]:
                if opt in df_map_src.columns:
                    agg[opt] = "first"

            df_plot = df_map_src.groupby("Site ID", as_index=False).agg(agg)

            prod_summary = (
                df_map_src.groupby("Site ID")["Product Group"]
                .apply(lambda s: ", ".join(sorted(pd.unique(s.dropna().astype(str)))))
                .rename("Products")
                .reset_index()
            )
            df_plot = df_plot.merge(prod_summary, on="Site ID", how="left")

            df_plot["lat_plot"] = df_plot["Lat"].astype(float)
            df_plot["lon_plot"] = df_plot["Lon"].astype(float)

            custom_cols = ["Site ID", "Products", "Assigned Terminal", "Assigned TCN"]
            for opt in ["Brand", "Supplier", "Home Terminal"]:
                if opt in df_plot.columns:
                    custom_cols.append(opt)

            hovertemplate = (
                "<b>Site:</b> %{customdata[0]}<br>"
                "<b>Products:</b> %{customdata[1]}<br>"
                "<b>Assigned Terminal:</b> %{customdata[2]}<br>"
                "<b>Assigned TCN:</b> %{customdata[3]}<br>"
            )
            idx = 4
            for opt_label in ["Brand", "Supplier", "Home Terminal"]:
                if opt_label in df_plot.columns:
                    hovertemplate += f"<b>{opt_label}:</b> %{{customdata[{idx}]}}<br>"
                    idx += 1
            hovertemplate += "<extra></extra>"
        else:
            df_plot = df_map_src.copy()
            if offsets_enabled:
                df_plot = _apply_product_offsets(df_plot, lat_col="Lat", lon_col="Lon", group_col="Product Group")
            else:
                df_plot["lat_plot"] = df_plot["Lat"].astype(float)
                df_plot["lon_plot"] = df_plot["Lon"].astype(float)

            custom_cols = ["Site ID", "Product Group", "Assigned Terminal", "Assigned TCN"]
            for opt in ["Brand", "Supplier", "Home Terminal"]:
                if opt in df_plot.columns:
                    custom_cols.append(opt)

            hovertemplate = (
                "<b>Site:</b> %{customdata[0]}<br>"
                "<b>Product:</b> %{customdata[1]}<br>"
                "<b>Assigned Terminal:</b> %{customdata[2]}<br>"
                "<b>Assigned TCN:</b> %{customdata[3]}<br>"
            )
            idx = 4
            for opt_label in ["Brand", "Supplier", "Home Terminal"]:
                if opt_label in df_plot.columns:
                    hovertemplate += f"<b>{opt_label}:</b> %{{customdata[{idx}]}}<br>"
                    idx += 1
            hovertemplate += "<extra></extra>"

        lat_min, lat_max = df_plot["lat_plot"].min(), df_plot["lat_plot"].max()
        lon_min, lon_max = df_plot["lon_plot"].min(), df_plot["lon_plot"].max()
        center = {"lat": float((lat_min + lat_max) / 2.0), "lon": float((lon_min + lon_max) / 2.0)}
        zoom = _mapbox_zoom_from_bounds(lat_min, lat_max, lon_min, lon_max)

        fig = px.scatter_mapbox(
            df_plot,
            lat="lat_plot",
            lon="lon_plot",
            color="Assigned Terminal",
            custom_data=custom_cols,
            zoom=zoom,
            center=center,
            height=720,
        )

        fig.update_layout(
            mapbox=dict(style="open-street-map"),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                x=0.01,
                y=0.01,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.75)",
                borderwidth=0,
                orientation="v",
            ),
        )

        fig.update_traces(
            hovertemplate=hovertemplate,
            marker=dict(size=10),
        )

        plotly_config = {"displayModeBar": False, "scrollZoom": True}

        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=720,
            override_width="100%",
            config=plotly_config,
            key="map_click",
        )

        if clicked and "customdata" in clicked[0] and clicked[0]["customdata"]:
            st.session_state["selected_site_id"] = str(clicked[0]["customdata"][0])

with right:
    st.subheader("Details")

    selected_site = st.session_state.get("selected_site_id", "")
    if selected_site:
        st.caption(f"Selected Site ID: {selected_site}")
        df_details = df_filtered[df_filtered["Site ID"].astype(str) == selected_site].copy()
        # Hide lat/lon in details table
        hide_cols = [c for c in ["Lat", "Lon"] if c in df_details.columns]
        st.dataframe(df_details.drop(columns=hide_cols), use_container_width=True, hide_index=True)
    else:
        st.info("Click a site on the map to see details.")

st.divider()

# -----------------------------
# Filtered table (lat/lon hidden)
# -----------------------------
st.subheader("Filtered Data")
df_table = df_filtered.copy()
hide_cols = [c for c in ["Lat", "Lon"] if c in df_table.columns]
st.dataframe(df_table.drop(columns=hide_cols), use_container_width=True, hide_index=True)
