from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from lib.constants import PRODUCT_GROUPS, terminal_color


OFFSETS = {
    "Regular": (0.00020, 0.00020),
    "Premium": (0.00020, -0.00020),
    "Diesel": (-0.00020, 0.00020),
}


def _zoom(df: pd.DataFrame) -> tuple[float, float, float]:
    lat_min, lat_max = float(df["_lat_plot"].min()), float(df["_lat_plot"].max())
    lon_min, lon_max = float(df["_lon_plot"].min()), float(df["_lon_plot"].max())
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    span = max(1e-9, lat_max - lat_min, lon_max - lon_min)
    zoom = float(np.clip(np.log2(360.0 / span), 1.0, 14.5))
    return center_lat, center_lon, zoom


def build_map_df(df: pd.DataFrame, product_offset: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out = out.dropna(subset=["Latitude", "Longitude"])
    out = out[out["Latitude"].between(-90, 90) & out["Longitude"].between(-180, 180)]
    if product_offset and "Product Group" in out.columns:
        off = out["Product Group"].astype(str).map(lambda v: OFFSETS.get(v, (0.0, 0.0)))
        out["_lat_plot"] = out["Latitude"] + off.map(lambda t: t[0]).astype(float)
        out["_lon_plot"] = out["Longitude"] + off.map(lambda t: t[1]).astype(float)
    else:
        out["_lat_plot"] = out["Latitude"]
        out["_lon_plot"] = out["Longitude"]
    out["_terminal"] = out.get("New Terminal", "Unassigned").fillna("Unassigned").astype(str)
    out["_tcn"] = out.get("New TCN", "").fillna("").astype(str)
    return out


def build_map_figure(df_plot: pd.DataFrame):
    if df_plot.empty:
        return None

    center_lat, center_lon, zoom = _zoom(df_plot)
    terminals = sorted(df_plot["_terminal"].unique().tolist())
    color_map = {t: terminal_color(t) for t in terminals}

    fig = px.scatter_mapbox(
        df_plot,
        lat="_lat_plot",
        lon="_lon_plot",
        color="_terminal",
        color_discrete_map=color_map,
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom,
        height=680,
    )
    fig.update_layout(
        mapbox={"style": "open-street-map", "center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h"},
        uirevision="map-stable",
    )

    fig.update_traces(
        customdata=np.column_stack([
            df_plot["Site ID"].astype(str),
            df_plot.get("Product Group", "").astype(str),
            df_plot["_terminal"].astype(str),
            df_plot["_tcn"].astype(str),
        ]),
        hovertemplate=(
            "<b>Site:</b> %{customdata[0]}"
            "<br><b>Product:</b> %{customdata[1]}"
            "<br><b>New Terminal:</b> %{customdata[2]}"
            "<br><b>New TCN:</b> %{customdata[3]}<extra></extra>"
        ),
        marker={"size": 11, "opacity": 0.95},
    )
    return fig


def product_groups_available(df: pd.DataFrame) -> list[str]:
    if "Product Group" not in df.columns:
        return PRODUCT_GROUPS
    return sorted(df["Product Group"].dropna().astype(str).unique().tolist())
