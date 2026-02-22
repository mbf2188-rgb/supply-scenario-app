"""Apply a targeted hotfix to legacy single-file Streamlit app.py.

This script is for repositories that still have only app.py + requirements.txt
and do NOT have lib/maps.py.

Changes applied:
1) Adds a visible build hash caption under the app title.
2) Forces map rendering path to px.scatter_mapbox with explicit
   open-street-map style and center/zoom layout updates.
"""

from pathlib import Path


def apply_hotfix() -> None:
    app = Path("app.py")
    if not app.exists():
        raise SystemExit("app.py not found in current directory")

    text = app.read_text()

    # Detect legacy layout quickly.
    is_legacy = "Global Controls" in text and "Assigned Terminal" in text
    if not is_legacy:
        raise SystemExit(
            "This does not look like the legacy single-file app layout. "
            "Abort to avoid patching the wrong file."
        )

    # 1) Add subprocess import + build hash helper + caption.
    if "import subprocess" not in text:
        text = text.replace("import io\n", "import io\nimport subprocess\n", 1)

    if "def _git_short_sha()" not in text:
        marker = "st.set_page_config(layout=\"wide\", page_title=\"Supply + Freight Scenario Viewer\")\nst.title(\"Supply + Freight Scenario Viewer\")\n"
        replacement = (
            "st.set_page_config(layout=\"wide\", page_title=\"Supply + Freight Scenario Viewer\")\n"
            "st.title(\"Supply + Freight Scenario Viewer\")\n"
            "\n"
            "def _git_short_sha() -> str:\n"
            "    try:\n"
            "        return subprocess.check_output([\"git\", \"rev-parse\", \"--short\", \"HEAD\"], text=True).strip()\n"
            "    except Exception:\n"
            "        return \"unknown\"\n"
            "\n"
            "st.caption(f\"Build: {_git_short_sha()}\")\n"
        )
        if marker in text:
            text = text.replace(marker, replacement, 1)

    # 2) Force _make_map_figure to use stable mapbox OSM path.
    fn_start = text.find("def _make_map_figure(")
    if fn_start == -1:
        raise SystemExit("Could not find _make_map_figure function")
    fn_end = text.find("\n\nif not uploaded_file:")
    if fn_end == -1:
        raise SystemExit("Could not find end of _make_map_figure block")

    new_fn = '''def _make_map_figure(df_plot: pd.DataFrame, color_arg: str | None, center_lat: float, center_lon: float, z: float, custom, hovertemplate):
    # Stable rendering path for Streamlit Cloud: tokenless OSM + explicit center/zoom.
    fig = px.scatter_mapbox(
        df_plot,
        lat="_lat_plot",
        lon="_lon_plot",
        color=color_arg,
        hover_name=None,
        height=620,
        zoom=z,
        center={"lat": center_lat, "lon": center_lon},
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        mapbox={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": z,
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
        ),
        uirevision="map-stable",
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
'''

    text = text[:fn_start] + new_fn + text[fn_end:]

    app.write_text(text)
    print("Patched legacy app.py successfully")


if __name__ == "__main__":
    apply_hotfix()
