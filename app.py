import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_plotly_events import plotly_events

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


def zoom_from_bounds(lat_min, lat_max, lon_min, lon_max) -> float:
    lat_span = max(1e-9, float(lat_max - lat_min))
    lon_span = max(1e-9, float(lon_max - lon_min))
    span = max(lat_span, lon_span)
    z = np.log2(360.0 / span)
    return float(np.clip(z, 1.0, 14.5))


if not uploaded_file:
    st.info("Upload a CSV or XLSX to begin.")
    st.stop()

df = _load_df(uploaded_file)
st.success(f"Loaded {len(df):,} rows")

# Column aliasing (map stays working if names vary)
lat_col = _first_existing(df, ["Latitude", "Lat", "LAT"])
lon_col = _first_existing(df, ["Longitude", "Lon", "Lng", "Long", "LON"])

# Required columns
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

with st.expander("Schema details", expanded=False):
    st.write("Required columns:", required)
    st.write("Detected columns:", list(df.columns))
    if lat_col and lon_col:
        st.write(f"Detected lat/lon columns: {lat_col}, {lon_col}")
    else:
        st.write("Detected lat/lon columns: NOT FOUND (accepted: Latitude/Lat and Longitude/Lon/Lng/Long)")

# --- Sidebar controls
st.sidebar.header("Global Controls")

search_q = st.sidebar.text_input("Global search (filters everything)", value="", key="global_search").strip()

scenarios = sorted(df["Scenario"].dropna().unique().tolist()) if "Scenario" in df.columns else []
scenario = st.sidebar.selectbox("Scenario", scenarios, index=0 if scenarios else None)

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
sel_site_id = _safe_multiselect(
    "Site ID",
    df["Site ID"] if "Site ID" in df.columns else pd.Series([], dtype=object),
    "f_site_id",
)
sel_prod_group = _safe_multiselect(
    "Product Group",
    df["Product Group"] if "Product Group" in df.columns else pd.Series([], dtype=object),
    "f_prod_group",
)

has_brand = "Brand" in df.columns
has_supplier = "Supplier" in df.columns
sel_brand = _safe_multiselect("Brand", df["Brand"], "f_brand") if has_brand else []
sel_supplier = _safe_multiselect("Supplier", df["Supplier"], "f_supplier") if has_supplier else []

# optional: separate markers by product group (slight offsets)
st.sidebar.subheader("Map Options")
separate_by_product = st.sidebar.toggle(
    "Separate markers by Product Group (tiny offsets)",
    value=False,
    help="Slightly offsets per product group for overlapping lat/lon.",
)

# --- Filter pipeline
df_f = df.copy()
if scenario and "Scenario" in df_f.columns:
    df_f = df_f[df_f["Scenario"] == scenario]

df_f = apply_in(df_f, "New Terminal", sel_assigned_terminal)
df_f = apply_in(df_f, "New TCN", sel_assigned_tcn)
df_f = apply_in(df_f, "Site ID", sel_site_id)
df_f = apply_in(df_f, "Product Group", sel_prod_group)
if has_brand:
    df_f = apply_in(df_f, "Brand", sel_brand)
if has_supplier:
    df_f = apply_in(df_f, "Supplier", sel_supplier)

if search_q:
    search_cols = [
        c
        for c in ["Site ID", "Product Group", "Brand", "Supplier", "Home Terminal", "New Terminal", "Home TCN", "New TCN", "Scenario"]
        if c in df_f.columns
    ]
    blob = df_f[search_cols].astype(str).agg(" | ".join, axis=1)
    df_f = df_f[blob.str.contains(search_q, case=False, na=False)]

st.caption(f"Filtered rows: {len(df_f):,}")

# --- Single-scenario page: remove cost KPIs (keep only assignment-impact context)
impacted_sites = None
if "Home Terminal" in df_f.columns and "New Terminal" in df_f.columns and "Site ID" in df_f.columns:
    impacted_sites = df_f.loc[df_f["Home Terminal"] != df_f["New Terminal"], "Site ID"].nunique()

site_count = df_f["Site ID"].nunique() if "Site ID" in df_f.columns else None
row_count = len(df_f)

k1, k2, k3 = st.columns(3)
k1.metric("Rows", f"{row_count:,}")
k2.metric("Sites", f"{site_count:,}" if isinstance(site_count, (int, np.integer)) else "N/A")
k3.metric("Impacted Sites", f"{impacted_sites:,}" if isinstance(impacted_sites, (int, np.integer)) else "N/A")

# --- Layout: map (left) + details (right)
st.subheader("Map (sites colored by Assigned Terminal)")

left, right = st.columns([2.2, 1.0], gap="large")

selected_site_id = st.session_state.get("selected_site_id", "")

with left:
    if lat_col and lon_col and (lat_col in df_f.columns) and (lon_col in df_f.columns):
        df_map_raw = df_f.dropna(subset=[lat_col, lon_col]).copy()
        if len(df_map_raw) == 0:
            st.info("No mappable rows after filtering (missing lat/lon).")
        else:
            # Ensure numeric lat/lon
            df_map_raw[lat_col] = pd.to_numeric(df_map_raw[lat_col], errors="coerce")
            df_map_raw[lon_col] = pd.to_numeric(df_map_raw[lon_col], errors="coerce")
            df_map_raw = df_map_raw.dropna(subset=[lat_col, lon_col])

            if len(df_map_raw) == 0:
                st.info("No mappable rows after filtering (invalid lat/lon).")
            else:
                # Auto-center + auto-zoom based on filtered extents (original lat/lon bounds)
                lat_min = float(df_map_raw[lat_col].min())
                lat_max = float(df_map_raw[lat_col].max())
                lon_min = float(df_map_raw[lon_col].min())
                lon_max = float(df_map_raw[lon_col].max())
                center_lat = float(df_map_raw[lat_col].mean())
                center_lon = float(df_map_raw[lon_col].mean())
                z = zoom_from_bounds(lat_min, lat_max, lon_min, lon_max)

                # Build map points:
                # - Default: one marker per site
                # - Optional: slight offsets per product group
                df_plot = None

                if separate_by_product and "Product Group" in df_map_raw.columns:
                    offsets = {
                        "Regular": (0.00025, 0.00025),
                        "Premium": (0.00025, -0.00025),
                        "Diesel": (-0.00025, 0.00025),
                    }

                    def _off_lat(pg):
                        return offsets.get(str(pg), (0.0, 0.0))[0]

                    def _off_lon(pg):
                        return offsets.get(str(pg), (0.0, 0.0))[1]

                    df_pts = df_map_raw.copy()
                    df_pts["_lat_plot"] = df_pts[lat_col].astype(float) + df_pts["Product Group"].map(_off_lat).astype(float)
                    df_pts["_lon_plot"] = df_pts[lon_col].astype(float) + df_pts["Product Group"].map(_off_lon).astype(float)
                    df_plot = df_pts

                    color_col = "New Terminal" if "New Terminal" in df_plot.columns else None

                    # FIX: use scatter_mapbox (stable geo tiles) + mapbox style
                    fig = px.scatter_mapbox(
                        df_plot,
                        lat="_lat_plot",
                        lon="_lon_plot",
                        color=color_col,
                        hover_name=None,
                        zoom=4,  # overridden below
                        height=620,
                    )

                    custom = np.stack(
                        [
                            df_plot["Site ID"].astype(str) if "Site ID" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["Product Group"].astype(str) if "Product Group" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["Brand"].astype(str) if "Brand" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["New Terminal"].astype(str) if "New Terminal" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["New TCN"].astype(str) if "New TCN" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        ],
                        axis=1,
                    )

                    fig.update_traces(
                        customdata=custom,
                        hovertemplate=(
                            "<b>Site ID:</b> %{customdata[0]}<br>"
                            "<b>Product:</b> %{customdata[1]}<br>"
                            "<b>Brand:</b> %{customdata[2]}<br>"
                            "<b>Assigned Terminal:</b> %{customdata[3]}<br>"
                            "<b>Assigned TCN:</b> %{customdata[4]}<extra></extra>"
                        ),
                    )

                else:
                    # One marker per site (aggregate)
                    def _assigned_terminal_for_site(s: pd.Series) -> str:
                        vals = s.dropna().astype(str).unique().tolist()
                        if len(vals) == 1:
                            return vals[0]
                        if len(vals) == 0:
                            return ""
                        return "Multiple"

                    agg = {
                        lat_col: "first",
                        lon_col: "first",
                        "New Terminal": _assigned_terminal_for_site if "New Terminal" in df_map_raw.columns else "first",
                    }
                    if "Brand" in df_map_raw.columns:
                        agg["Brand"] = "first"
                    if "New TCN" in df_map_raw.columns:
                        agg["New TCN"] = "first"

                    df_sites = df_map_raw.groupby("Site ID", as_index=False).agg(agg)
                    df_sites["_lat_plot"] = df_sites[lat_col].astype(float)
                    df_sites["_lon_plot"] = df_sites[lon_col].astype(float)
                    df_plot = df_sites

                    fig = px.scatter_mapbox(
                        df_plot,
                        lat="_lat_plot",
                        lon="_lon_plot",
                        color="New Terminal" if "New Terminal" in df_plot.columns else None,
                        hover_name=None,
                        zoom=4,  # overridden below
                        height=620,
                    )

                    custom = np.stack(
                        [
                            df_plot["Site ID"].astype(str),
                            df_plot["Brand"].astype(str) if "Brand" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["New Terminal"].astype(str) if "New Terminal" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                            df_plot["New TCN"].astype(str) if "New TCN" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        ],
                        axis=1,
                    )

                    fig.update_traces(
                        customdata=custom,
                        hovertemplate=(
                            "<b>Site ID:</b> %{customdata[0]}<br>"
                            "<b>Brand:</b> %{customdata[1]}<br>"
                            "<b>Assigned Terminal:</b> %{customdata[2]}<br>"
                            "<b>Assigned TCN:</b> %{customdata[3]}<extra></extra>"
                        ),
                    )

                # Layout: mapbox tiles + auto center/zoom + legend placement
                fig.update_layout(
                    margin={"l": 0, "r": 0, "t": 0, "b": 0},
                    mapbox=dict(
                        style="open-street-map",
                        center={"lat": center_lat, "lon": center_lon},
                        zoom=z,
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.01,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                fig.update_traces(marker={"size": 16, "opacity": 0.95})

                # Render + click capture (disable modebar to avoid overlap)
                click_data = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=620,
                    key="map_events",
                    config={"displayModeBar": False, "scrollZoom": True},
                )

                # Map-click -> details panel (use returned customdata for stability)
                if click_data and isinstance(click_data, list) and len(click_data) > 0:
                    pt = click_data[0]
                    cd = pt.get("customdata", None)
                    if isinstance(cd, (list, tuple)) and len(cd) >= 1:
                        site_id_clicked = str(cd[0])
                        st.session_state["selected_site_id"] = site_id_clicked
                        selected_site_id = site_id_clicked
    else:
        st.info("Map disabled: Latitude/Longitude columns not found (accepted: Latitude/Lat and Longitude/Lon/Lng/Long).")

with right:
    st.markdown("### Site details")
    if selected_site_id:
        site_rows = df_f[df_f["Site ID"] == selected_site_id].copy()

        def _uniq_or_multiple(col: str) -> str:
            if col not in site_rows.columns:
                return ""
            vals = site_rows[col].dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                return vals[0]
            if len(vals) == 0:
                return ""
            return "Multiple"

        st.write(f"**Site ID:** {selected_site_id}")
        if "Brand" in site_rows.columns:
            st.write(f"**Brand:** {_uniq_or_multiple('Brand')}")
        st.write(f"**Assigned Terminal:** {_uniq_or_multiple('New Terminal')}")
        st.write(f"**Assigned TCN:** {_uniq_or_multiple('New TCN')}")
        if "Supplier" in site_rows.columns:
            st.write(f"**Supplier:** {_uniq_or_multiple('Supplier')}")
        if "Home Terminal" in site_rows.columns:
            st.write(f"**Home Terminal:** {_uniq_or_multiple('Home Terminal')}")
        if "Home TCN" in site_rows.columns:
            st.write(f"**Home TCN:** {_uniq_or_multiple('Home TCN')}")

        st.markdown("#### Product breakdown")
        cols = [c for c in ["Product Group", "Total Cost (30 days)", "Freight Cost (30 days)", "Product Cost (30 days)"] if c in site_rows.columns]
        if cols:
            st.dataframe(site_rows[cols], width="stretch", height=260)
        else:
            st.write("No cost columns available for breakdown.")
    else:
        st.write("Click a site on the map to view details here.")

# Filtered data table (hide lat/lon columns)
st.subheader("Filtered Data")
df_display = df_f.drop(columns=[c for c in [lat_col, lon_col] if c], errors="ignore")
st.dataframe(df_display, width="stretch", height=520)
