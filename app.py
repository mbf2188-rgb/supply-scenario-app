import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="Supply + Freight Scenario Viewer")
st.title("Supply + Freight Scenario Viewer")

uploaded_file = st.file_uploader("Upload scenario file (.csv or .xlsx)", type=["csv", "xlsx"])

# -----------------------------
# Fix focus:
# - Robust lat/lon parsing (strings, commas, whitespace, swapped columns)
# - Enforce valid lat/lon ranges (prevents Plotly silently rendering nothing)
# - Prefer px.scatter_map (MapLibre) when available; fallback to px.scatter_mapbox
# - Keep click-to-details, no lat/lon in hover, auto center/zoom, stable in Streamlit Cloud
# -----------------------------

@st.cache_data(show_spinner=False)
def _load_df_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        try:
            df0 = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df0 = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", engine="python")
    else:
        df0 = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, engine="openpyxl")

    df0.columns = [str(c).strip() for c in df0.columns]
    return df0


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_multiselect(label: str, series: pd.Series, key: str):
    if series is None or series.empty:
        return []
    opts = sorted([x for x in series.dropna().unique().tolist()])
    return st.sidebar.multiselect(label, options=opts, default=[], key=key)


def zoom_from_bounds(lat_min, lat_max, lon_min, lon_max) -> float:
    lat_span = max(1e-9, float(lat_max - lat_min))
    lon_span = max(1e-9, float(lon_max - lon_min))
    span = max(lat_span, lon_span)
    z = np.log2(360.0 / span)
    return float(np.clip(z, 1.0, 14.5))


def _assigned_terminal_for_site(series: pd.Series) -> str:
    vals = series.dropna().astype(str).unique().tolist()
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 0:
        return ""
    return "Multiple"


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # Handles: " 32.9 ", "32,9000", "(-96.8)" etc.
    if s.dtype.kind in "if":
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip()
    x = x.str.replace("\u2212", "-", regex=False)  # unicode minus
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (96.8) -> -96.8
    x = x.str.replace(r"[^0-9\.\-\+eE]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def _coerce_and_validate_latlon(df_in: pd.DataFrame, lat_col: str, lon_col: str) -> tuple[pd.DataFrame, dict]:
    df = df_in.copy()
    lat = _clean_numeric_series(df[lat_col])
    lon = _clean_numeric_series(df[lon_col])

    info = {
        "raw_rows": int(len(df)),
        "lat_nan": int(lat.isna().sum()),
        "lon_nan": int(lon.isna().sum()),
    }

    # Detect swapped columns if most "lat" values look like longitudes and vice-versa
    lat_in_range = lat.between(-90, 90, inclusive="both")
    lon_in_range = lon.between(-180, 180, inclusive="both")

    # Swapped heuristic: many lat out of range but would be ok as lon, and many lon out of range but would be ok as lat
    swapped = (
        (lat_in_range.mean() < 0.35 and lon_in_range.mean() < 0.35)
        and (_clean_numeric_series(df[lat_col]).between(-180, 180, inclusive="both").mean() > 0.80)
        and (_clean_numeric_series(df[lon_col]).between(-90, 90, inclusive="both").mean() > 0.80)
    )

    # Alternate (more common) swapped case: lat mostly outside [-90,90] but lon mostly outside [-180,180] is rare;
    # instead check: lat mostly outside [-90,90] while lon mostly inside [-90,90]
    if not swapped:
        swapped = (lat_in_range.mean() < 0.35) and (_clean_numeric_series(df[lon_col]).between(-90, 90, inclusive="both").mean() > 0.80)

    info["swapped_detected"] = bool(swapped)

    if swapped:
        lat, lon = lon, lat

    # Enforce ranges (invalid coords can lead to empty-looking maps)
    valid = lat.between(-90, 90, inclusive="both") & lon.between(-180, 180, inclusive="both")
    info["invalid_range"] = int((~valid & ~(lat.isna() | lon.isna())).sum())

    df["_lat_base"] = lat
    df["_lon_base"] = lon
    df = df.dropna(subset=["_lat_base", "_lon_base"])
    df = df.loc[valid.loc[df.index]].copy()

    info["valid_rows"] = int(len(df))
    if len(df) > 0:
        info["lat_min"] = float(df["_lat_base"].min())
        info["lat_max"] = float(df["_lat_base"].max())
        info["lon_min"] = float(df["_lon_base"].min())
        info["lon_max"] = float(df["_lon_base"].max())
    return df, info


@st.cache_data(show_spinner=False)
def _build_sites(df_map_raw: pd.DataFrame, site_col: str) -> pd.DataFrame:
    # One marker per site (uses _lat_base/_lon_base)
    agg = {"_lat_base": "first", "_lon_base": "first"}
    if "New Terminal" in df_map_raw.columns:
        agg["New Terminal"] = _assigned_terminal_for_site
    if "Brand" in df_map_raw.columns:
        agg["Brand"] = "first"
    if "New TCN" in df_map_raw.columns:
        agg["New TCN"] = "first"

    df_sites = df_map_raw.groupby(site_col, as_index=False).agg(agg)
    df_sites["_lat_plot"] = pd.to_numeric(df_sites["_lat_base"], errors="coerce")
    df_sites["_lon_plot"] = pd.to_numeric(df_sites["_lon_base"], errors="coerce")
    df_sites = df_sites.dropna(subset=["_lat_plot", "_lon_plot"])
    return df_sites


def _build_pts(df_map_raw: pd.DataFrame) -> pd.DataFrame:
    # Multiple markers (tiny offsets by Product Group)
    df_pts = df_map_raw.copy()

    offsets = {
        "Regular": (0.00025, 0.00025),
        "Premium": (0.00025, -0.00025),
        "Diesel": (-0.00025, 0.00025),
    }

    if "Product Group" in df_pts.columns:
        pg = df_pts["Product Group"].astype(str)
        off = pg.map(lambda x: offsets.get(x, (0.0, 0.0)))
        off_lat = off.map(lambda t: t[0]).astype(float)
        off_lon = off.map(lambda t: t[1]).astype(float)
    else:
        off_lat = 0.0
        off_lon = 0.0

    base_lat = pd.to_numeric(df_pts["_lat_base"], errors="coerce")
    base_lon = pd.to_numeric(df_pts["_lon_base"], errors="coerce")
    df_pts["_lat_plot"] = base_lat + off_lat
    df_pts["_lon_plot"] = base_lon + off_lon
    df_pts = df_pts.dropna(subset=["_lat_plot", "_lon_plot"])
    return df_pts


def _prep_color(df_plot: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    # Keep points visible even when category is missing
    df_plot = df_plot.copy()
    if "New Terminal" in df_plot.columns:
        df_plot["New Terminal"] = df_plot["New Terminal"].fillna("Unassigned").astype(str)
        # Prevent empty strings becoming "invisible categories" in some legend states
        df_plot.loc[df_plot["New Terminal"].str.strip().eq(""), "New Terminal"] = "Unassigned"
        return df_plot, "New Terminal"
    return df_plot, None


def _map_center_zoom_from_plot(df_plot: pd.DataFrame) -> tuple[float, float, float]:
    lat_min = float(df_plot["_lat_plot"].min())
    lat_max = float(df_plot["_lat_plot"].max())
    lon_min = float(df_plot["_lon_plot"].min())
    lon_max = float(df_plot["_lon_plot"].max())
    center_lat = float((lat_min + lat_max) / 2.0)
    center_lon = float((lon_min + lon_max) / 2.0)
    z = zoom_from_bounds(lat_min, lat_max, lon_min, lon_max)
    return center_lat, center_lon, z


def _make_map_figure(df_plot: pd.DataFrame, color_arg: str | None, center_lat: float, center_lon: float, z: float, custom, hovertemplate):
    # Prefer MapLibre (px.scatter_map) if available; fallback to Mapbox.
    use_maplibre = hasattr(px, "scatter_map")
    if use_maplibre:
        fig = px.scatter_map(
            df_plot,
            lat="_lat_plot",
            lon="_lon_plot",
            color=color_arg,
            hover_name=None,
            height=620,
            zoom=z,
            center={"lat": center_lat, "lon": center_lon},
        )
        # Hide modebar in Streamlit reliably (layout + Streamlit config)
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    else:
        fig = px.scatter_mapbox(
            df_plot,
            lat="_lat_plot",
            lon="_lon_plot",
            color=color_arg,
            hover_name=None,
            height=620,
            zoom=4,  # overridden below
        )
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
            modebar_remove=[
                "zoom", "pan", "select", "lasso2d",
                "zoomIn", "zoomOut", "autoScale", "resetScale",
            ],
        )

    fig.update_traces(
        mode="markers",
        marker={"size": 14, "opacity": 0.95},
        customdata=custom,
        hovertemplate=hovertemplate,
    )
    return fig


if not uploaded_file:
    st.info("Upload a CSV or XLSX to begin.")
    st.stop()

df = _load_df_bytes(uploaded_file.getvalue(), uploaded_file.name)
st.success(f"Loaded {len(df):,} rows")

lat_col = _first_existing(df, ["Latitude", "Lat", "LAT"])
lon_col = _first_existing(df, ["Longitude", "Lon", "Lng", "Long", "LON"])

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

st.sidebar.subheader("Map Options")
separate_by_product = st.sidebar.toggle(
    "Separate markers by Product Group (tiny offsets)",
    value=False,
    help="Slightly offsets per product group for overlapping lat/lon.",
)
show_map_debug = st.sidebar.toggle(
    "Show map debug counters",
    value=False,
    help="Shows why points might be filtered out (NaN/invalid range/swapped detection).",
)

# --- Filter pipeline (mask-based)
df_f = df
mask = pd.Series(True, index=df_f.index)

if scenario and "Scenario" in df_f.columns:
    mask &= (df_f["Scenario"] == scenario)

if sel_assigned_terminal and "New Terminal" in df_f.columns:
    mask &= df_f["New Terminal"].isin(sel_assigned_terminal)

if sel_assigned_tcn and "New TCN" in df_f.columns:
    mask &= df_f["New TCN"].isin(sel_assigned_tcn)

if sel_site_id and "Site ID" in df_f.columns:
    mask &= df_f["Site ID"].isin(sel_site_id)

if sel_prod_group and "Product Group" in df_f.columns:
    mask &= df_f["Product Group"].isin(sel_prod_group)

if has_brand and sel_brand:
    mask &= df_f["Brand"].isin(sel_brand)

if has_supplier and sel_supplier:
    mask &= df_f["Supplier"].isin(sel_supplier)

df_f = df_f.loc[mask].copy()

if search_q:
    search_cols = [
        c for c in [
            "Site ID", "Product Group", "Brand", "Supplier",
            "Home Terminal", "New Terminal", "Home TCN", "New TCN", "Scenario"
        ] if c in df_f.columns
    ]
    if search_cols:
        blob = df_f[search_cols].astype(str).agg(" | ".join, axis=1)
        df_f = df_f.loc[blob.str.contains(search_q, case=False, na=False)].copy()

st.caption(f"Filtered rows: {len(df_f):,}")

# --- KPIs
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
        # Robust lat/lon coercion + range validation + swap detection
        df_map_raw, ll_info = _coerce_and_validate_latlon(df_f, lat_col, lon_col)

        if show_map_debug:
            st.caption(
                "Map debug — "
                f"raw={ll_info.get('raw_rows', 0):,}, "
                f"lat_nan={ll_info.get('lat_nan', 0):,}, "
                f"lon_nan={ll_info.get('lon_nan', 0):,}, "
                f"invalid_range={ll_info.get('invalid_range', 0):,}, "
                f"swapped={ll_info.get('swapped_detected', False)}, "
                f"valid={ll_info.get('valid_rows', 0):,}"
            )

        if len(df_map_raw) == 0:
            st.info("No mappable rows after parsing/validation of lat/lon.")
        else:
            # Build plotted dataframe
            if separate_by_product and "Product Group" in df_map_raw.columns:
                df_plot = _build_pts(df_map_raw)
                custom = np.column_stack(
                    [
                        df_plot["Site ID"].astype(str) if "Site ID" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["Product Group"].astype(str) if "Product Group" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["Brand"].astype(str) if "Brand" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["New Terminal"].astype(str) if "New Terminal" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["New TCN"].astype(str) if "New TCN" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                    ]
                )
                hovertemplate = (
                    "<b>Site ID:</b> %{customdata[0]}<br>"
                    "<b>Product:</b> %{customdata[1]}<br>"
                    "<b>Brand:</b> %{customdata[2]}<br>"
                    "<b>Assigned Terminal:</b> %{customdata[3]}<br>"
                    "<b>Assigned TCN:</b> %{customdata[4]}<extra></extra>"
                )
            else:
                df_plot = _build_sites(df_map_raw, "Site ID")
                custom = np.column_stack(
                    [
                        df_plot["Site ID"].astype(str),
                        df_plot["Brand"].astype(str) if "Brand" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["New Terminal"].astype(str) if "New Terminal" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                        df_plot["New TCN"].astype(str) if "New TCN" in df_plot.columns else np.array([""] * len(df_plot), dtype=object),
                    ]
                )
                hovertemplate = (
                    "<b>Site ID:</b> %{customdata[0]}<br>"
                    "<b>Brand:</b> %{customdata[1]}<br>"
                    "<b>Assigned Terminal:</b> %{customdata[2]}<br>"
                    "<b>Assigned TCN:</b> %{customdata[3]}<extra></extra>"
                )

            if df_plot.empty:
                st.info("No plottable points after building the plot dataframe.")
            else:
                # Ensure color column never null/blank (prevents Plotly category issues)
                df_plot, color_arg = _prep_color(df_plot)

                # Compute center/zoom from plotted df
                center_lat, center_lon, z = _map_center_zoom_from_plot(df_plot)

                fig = _make_map_figure(df_plot, color_arg, center_lat, center_lon, z, custom, hovertemplate)

                st.caption(f"Plotted points: {len(df_plot):,}")

                click_data = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=620,
                    key="map_events",
                )

                # Click -> details
                if click_data and isinstance(click_data, list) and len(click_data) > 0:
                    pt = click_data[0]
                    cd = pt.get("customdata", None)

                    if isinstance(cd, (list, tuple)) and len(cd) >= 1 and str(cd[0]).strip() != "":
                        site_id_clicked = str(cd[0])
                    else:
                        idx = pt.get("pointIndex", pt.get("pointNumber", None))
                        site_id_clicked = ""
                        if idx is not None:
                            idx = int(idx)
                            if 0 <= idx < len(df_plot):
                                site_id_clicked = str(df_plot.reset_index(drop=True).iloc[idx].get("Site ID", ""))

                    if site_id_clicked:
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
