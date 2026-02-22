from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

from lib.export import build_html_report, export_product_group_images, fig_to_html
from lib.io import load_uploaded_file
from lib.maps import build_map_df, build_map_figure, product_groups_available
from lib.metrics import changed_sites_only, delta_vs_baseline, impacted_sites, terminal_shift_matrix, totals, volume_by_terminal_product
from lib.normalize import normalize_dataset
from lib.ui import apply_theme, choose_scenario_defaults, explorer_table, global_filter, paginated_table


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    st.set_page_config(layout="wide", page_title="Supply + Freight Scenario Viewer")
    st.title("Supply + Freight Scenario Viewer")
    st.caption(f"Build: {_git_short_sha()}")

    uploaded_file = st.file_uploader("Upload scenario file (.csv or .xlsx)", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Upload a CSV or XLSX file to begin.")
        st.stop()

    night_mode = st.sidebar.toggle("Night mode", value=False)
    apply_theme(night_mode)

    try:
        raw = load_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    norm = normalize_dataset(raw)
    if norm.missing_required:
        st.error(f"Missing required canonical fields: {norm.missing_required}")
        st.stop()

    df = norm.df
    st.success(f"Loaded {len(df):,} rows")
    st.caption(norm.volume_note)

    with st.expander("Schema details", expanded=False):
        st.write("Original to canonical mapping:", norm.column_map)
        st.write("Columns:", list(df.columns))

    search = st.sidebar.text_input("Global free-text search", value="").strip()
    filtered = global_filter(df, search)

    scenarios = sorted(filtered["Scenario"].dropna().astype(str).unique().tolist())
    default_baseline, default_display = choose_scenario_defaults(scenarios)
    baseline = st.sidebar.selectbox("Baseline scenario", scenarios, index=scenarios.index(default_baseline) if default_baseline in scenarios else 0)
    display = st.sidebar.selectbox("Scenario to display", scenarios, index=scenarios.index(default_display) if default_display in scenarios else 0)

    scenario_filter = st.sidebar.multiselect("Scenario filter", scenarios, default=[])
    if scenario_filter:
        filtered = filtered[filtered["Scenario"].astype(str).isin(scenario_filter)]

    units = st.sidebar.selectbox("Volume units", ["bbl/day", "bbl/month", "gal/day", "gal/month"], index=0)

    for label in ["New Terminal", "New TCN", "Site ID", "Product Group", "Brand", "Supplier"]:
        if label in filtered.columns:
            choices = sorted(filtered[label].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(label, choices, default=[])
            if selected:
                filtered = filtered[filtered[label].astype(str).isin(selected)]

    if units == "bbl/day":
        volume_col = "Daily Volume (bbl)"
    elif units == "bbl/month":
        volume_col = "Monthly Volume (bbl 30 day)"
    elif units == "gal/day":
        volume_col = "Daily Volume (gal)"
    else:
        volume_col = "Monthly Volume (gal 30 day)"

    if volume_col not in filtered.columns:
        st.error(f"Selected units require volume column: {volume_col}")
        st.stop()

    selected_df = filtered[filtered["Scenario"].astype(str) == display].copy()

    k1, k2, k3 = st.columns(3)
    sel_tot = totals(selected_df)
    with k1:
        st.metric("Sites", int(selected_df["Site ID"].nunique()))
    with k2:
        st.metric("Impacted Sites", impacted_sites(selected_df))
    with k3:
        st.metric("Total Cost (1 year)", f"${sel_tot['total_1y']:,.0f}")

    overview_tab, compare_tab, explorer_tab = st.tabs(["Overview", "Compare", "Explorer"])

    with overview_tab:
        delta = delta_vs_baseline(filtered, baseline, display)
        wins = delta.nsmallest(10, "delta_30")
        losses = delta.nlargest(10, "delta_30")

        left, right = st.columns(2)
        with left:
            st.subheader("Top wins vs baseline (30-day delta)")
            st.dataframe(wins[["Site ID", "Product Group", "delta_30", "delta_1y"]], use_container_width=True)
        with right:
            st.subheader("Top losses vs baseline (30-day delta)")
            st.dataframe(losses[["Site ID", "Product Group", "delta_30", "delta_1y"]], use_container_width=True)

        shift = terminal_shift_matrix(selected_df)
        st.subheader("Terminal shift matrix (Home → New)")
        st.dataframe(shift, use_container_width=True)

        ranked = selected_df.groupby("New Terminal", as_index=False)[volume_col].sum().sort_values(volume_col, ascending=False)
        st.subheader(f"Volume ranked by New Terminal ({units})")
        st.dataframe(ranked, use_container_width=True)

    with compare_tab:
        sx = st.selectbox("Scenario X", scenarios, index=scenarios.index(display) if display in scenarios else 0)
        sy = st.selectbox("Scenario Y", scenarios, index=scenarios.index(baseline) if baseline in scenarios else 0)
        changed_only = st.checkbox("Changed sites only (New TCN differs)", value=True)

        cmp_df = filtered[filtered["Scenario"].astype(str).isin([sx, sy])].copy()
        if changed_only:
            cmp_df = changed_sites_only(cmp_df, sx, sy)

        tx, ty = totals(cmp_df[cmp_df["Scenario"].astype(str) == sx]), totals(cmp_df[cmp_df["Scenario"].astype(str) == sy])
        metric_df = pd.DataFrame([{"Scenario": sx, **tx}, {"Scenario": sy, **ty}])
        st.subheader("Scenario totals (30-day and 1-year)")
        st.dataframe(metric_df, use_container_width=True)

        matrix = volume_by_terminal_product(cmp_df, volume_col)
        st.subheader(f"Volume matrix: New Terminal × Product Group ({units})")
        st.dataframe(matrix, use_container_width=True)

    with explorer_tab:
        pg_options = product_groups_available(selected_df)
        selected_groups = st.multiselect("Map filter: Product Group", options=pg_options, default=pg_options)
        map_df = selected_df[selected_df["Product Group"].astype(str).isin(selected_groups)] if selected_groups else selected_df.iloc[0:0]

        map_df = build_map_df(map_df, product_offset=True)
        fig = build_map_figure(map_df)
        selected_site = ""
        if fig is not None:
            clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=620, key="map_click")
            if clicks:
                cdata = clicks[0].get("customdata")
                if isinstance(cdata, (list, tuple)) and cdata:
                    selected_site = str(cdata[0])

        st.subheader("Decision table")
        ex = explorer_table(selected_df)
        if selected_site:
            st.caption(f"Selected site: {selected_site}")
            ex = ex[ex["Site ID"].astype(str) == selected_site]
        paginated_table(ex, key="explorer")

    st.subheader("Export")
    export_images = st.checkbox("Also export product-group map images (Regular/Premium/Diesel)", value=False)
    if st.button("Export HTML report"):
        out_dir = Path("exports")
        out_dir.mkdir(exist_ok=True)

        delta = delta_vs_baseline(filtered, baseline, display)
        overview_fig = px.bar(
            delta.sort_values("delta_30").tail(20),
            x="Site ID",
            y="delta_30",
            color="Product Group",
            title=f"Top deltas: {display} - {baseline} (30-day)",
        )
        compare_fig = px.bar(pd.DataFrame([{"Scenario": sx, **tx}, {"Scenario": sy, **ty}]).melt(id_vars=["Scenario"], value_vars=["freight_30", "supply_30", "total_30"]), x="Scenario", y="value", color="variable")
        map_fig = build_map_figure(build_map_df(selected_df, product_offset=True))

        html_path = out_dir / "supply_scenario_report.html"
        build_html_report(
            output_path=html_path,
            title="Supply + Freight Scenario Report",
            overview_fig_html=fig_to_html(overview_fig),
            compare_fig_html=fig_to_html(compare_fig),
            map_fig_html=fig_to_html(map_fig),
            overview_table=delta.head(100),
            compare_table=pd.DataFrame([{"Scenario": sx, **tx}, {"Scenario": sy, **ty}]),
            explorer_table=explorer_table(selected_df).head(500),
        )

        if export_images:
            try:
                images = export_product_group_images(selected_df, lambda x: build_map_figure(build_map_df(x, product_offset=True)), out_dir / "maps")
                st.success(f"Export complete: {html_path} and {len(images)} map images")
            except Exception as exc:
                st.warning(f"HTML exported ({html_path}), but PNG export failed: {exc}")
        else:
            st.success(f"Export complete: {html_path}")

        st.download_button("Download report HTML", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")


if __name__ == "__main__":
    main()
