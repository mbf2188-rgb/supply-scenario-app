from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.constants import ANNUALIZATION_FACTOR, EXPLORER_COLUMNS
from lib.export import build_html_report, export_product_group_images, fig_to_html
from lib.io import load_uploaded_file
from lib.maps import build_map_df, build_map_figure
from lib.metrics import (
    changed_sites_only,
    delta_by_group,
    delta_totals,
    impacted_sites_compare,
    impacted_sites_overview,
    terminal_shift_matrix,
    volume_by_product_tcn,
    volume_by_terminal_product,
)
from lib.normalize import normalize_dataset
from lib.ui import apply_theme, choose_scenario_defaults, explorer_table, global_filter, paginated_table, safe_dataframe


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _volume_column(df: pd.DataFrame, units: str) -> tuple[pd.DataFrame, str, str]:
    out = df.copy()
    if units == "bbl/day":
        return out, "Daily Volume (bbl)", ""
    if units == "bbl/month":
        return out, "Monthly Volume (bbl 30 day)", ""
    if units == "gal/day":
        return out, "Daily Volume (gal)", ""
    if units == "gal/month":
        return out, "Monthly Volume (gal 30 day)", ""

    if units == "bbl/yr":
        if "Daily Volume (bbl)" in out.columns:
            out["__volume_year"] = pd.to_numeric(out["Daily Volume (bbl)"], errors="coerce") * 365
            return out, "__volume_year", "Derived bbl/yr from Daily Volume (bbl) × 365"
        out["__volume_year"] = pd.to_numeric(out["Monthly Volume (bbl 30 day)"], errors="coerce") * (365 / 30)
        return out, "__volume_year", "Derived bbl/yr from Monthly Volume (bbl 30 day) × (365/30)"

    if "Daily Volume (gal)" in out.columns:
        out["__volume_year"] = pd.to_numeric(out["Daily Volume (gal)"], errors="coerce") * 365
        return out, "__volume_year", "Derived gal/yr from Daily Volume (gal) × 365"
    out["__volume_year"] = pd.to_numeric(out["Monthly Volume (gal 30 day)"], errors="coerce") * (365 / 30)
    return out, "__volume_year", "Derived gal/yr from Monthly Volume (gal 30 day) × (365/30)"


def _default_new_scenario(scenarios: list[str], baseline: str) -> str:
    for s in scenarios:
        if s != baseline:
            return s
    return baseline


def _format_explorer(df: pd.DataFrame) -> pd.DataFrame:
    out = explorer_table(df).copy()
    rate_cols = ["Primary Freight Rate", "New Freight Rate", "Primary Supply Rate", "New Supply Rate"]
    cost_cols = ["Freight Cost (30 days)", "Product Cost (30 days)", "Total Cost (30 days)"]

    for col in rate_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    for col in cost_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    out = out.rename(
        columns={
            "Primary Freight Rate": "Primary Freight Rate (¢/gal)",
            "New Freight Rate": "New Freight Rate (¢/gal)",
            "Primary Supply Rate": "Primary Supply Rate (¢/gal)",
            "New Supply Rate": "New Supply Rate (¢/gal)",
        }
    )
    return out


def main() -> None:
    st.set_page_config(layout="wide", page_title="Supply & Freight Scenario Viewer")
    st.title("Supply & Freight Scenario Viewer")
    st.caption(f"Build: {_git_short_sha()}")

    uploaded_file = st.file_uploader("Upload scenario file (.csv or .xlsx)", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("Upload a CSV or XLSX file to begin.")
        st.stop()

    dark_mode = st.sidebar.toggle("Dark mode", value=False)
    apply_theme(dark_mode)

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

    # Global controls
    search = st.sidebar.text_input("Global free-text search", value="").strip()
    filtered = global_filter(df, search)

    units = st.sidebar.selectbox("Global volume units", ["bbl/day", "bbl/month", "bbl/yr", "gal/day", "gal/month", "gal/yr"], index=0)
    filtered, volume_col, volume_note = _volume_column(filtered, units)
    if volume_note:
        st.sidebar.caption(volume_note)

    global_filters = ["New Terminal", "New TCN", "Site ID", "Product Group", "Brand", "Assigned Carrier"]
    for label in global_filters:
        if label in filtered.columns:
            options = sorted(filtered[label].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(label, options=options, default=[])
            if selected:
                filtered = filtered[filtered[label].astype(str).isin(selected)]

    scenarios = sorted(filtered["Scenario"].dropna().astype(str).unique().tolist())
    if not scenarios:
        st.warning("No scenarios available after current filters.")
        st.stop()

    default_primary, default_secondary = choose_scenario_defaults(scenarios)

    # Export controls near top
    e1, e2 = st.columns([1, 2])
    with e1:
        export_images = st.checkbox("Export map images (Regular/Premium/Diesel)", value=False)
        export_clicked = st.button("Export HTML report")
    with e2:
        st.caption("Export reflects current tab selections and global filters.")

    compare_tab, overview_tab, explorer_tab = st.tabs(["Compare Scenarios", "Scenario Overview", "Data Explorer"])

    # Shared objects for export
    compare_fig = None
    overview_fig = None
    map_fig = None
    compare_table = pd.DataFrame()
    overview_table = pd.DataFrame()

    with compare_tab:
        baseline = st.selectbox("Baseline scenario", scenarios, index=scenarios.index(default_primary), key="cmp_base")
        default_new = _default_new_scenario(scenarios, baseline)
        new_scenario = st.selectbox("New scenario", scenarios, index=scenarios.index(default_new), key="cmp_new")
        changed_only = st.checkbox("Changed sites only (New Terminal differs)", value=True)

        compare_df = filtered[filtered["Scenario"].astype(str).isin([baseline, new_scenario])].copy()
        if changed_only:
            compare_df = changed_sites_only(compare_df, baseline, new_scenario)

        deltas = delta_totals(compare_df, baseline, new_scenario)
        impacted = impacted_sites_compare(filtered, baseline, new_scenario)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Impacted Sites", impacted)
        k2.metric("Δ Total Cost (30-day)", f"${deltas['delta_total_30']:,.0f}")
        k3.metric("Δ Total Cost (1-year)", f"${deltas['delta_total_1y']:,.0f}")
        k4.metric("Δ Freight Cost (30-day)", f"${deltas['delta_freight_30']:,.0f}")
        k5.metric("Δ Supply Cost (30-day)", f"${deltas['delta_supply_30']:,.0f}")

        delta_terminal_pg = delta_by_group(compare_df, baseline, new_scenario, ["New Terminal", "Product Group"], "Total Cost (30 days)")
        st.subheader("Delta cost by New Terminal × Product Group")
        safe_dataframe(delta_terminal_pg)

        delta_brand = delta_by_group(compare_df, baseline, new_scenario, ["Brand"], "Total Cost (30 days)")
        st.subheader("Delta cost by Brand")
        safe_dataframe(delta_brand)

        if "Assigned Carrier" in compare_df.columns:
            delta_carrier = delta_by_group(compare_df, baseline, new_scenario, ["Assigned Carrier"], "Total Cost (30 days)")
            st.subheader("Delta cost by Assigned Carrier")
            safe_dataframe(delta_carrier)

        delta_tcn = delta_by_group(compare_df, baseline, new_scenario, ["New TCN"], "Total Cost (30 days)")
        st.subheader("Delta cost by New TCN")
        safe_dataframe(delta_tcn)

        delta_volume = delta_by_group(compare_df, baseline, new_scenario, ["New Terminal", "Product Group"], volume_col)
        delta_volume = delta_volume.rename(columns={"delta_30": f"delta_{units}"})
        st.subheader(f"Delta volume by New Terminal × Product Group ({units})")
        safe_dataframe(delta_volume)

        compare_table = delta_terminal_pg.copy()
        if not compare_table.empty:
            compare_fig = px.bar(compare_table.head(20), x="New Terminal", y="delta_30", color="Product Group", title="Δ Cost (30-day)")

    with overview_tab:
        scenario_overview = st.selectbox("Scenario", scenarios, index=scenarios.index(default_primary), key="ov_scenario")
        ov_df = filtered[filtered["Scenario"].astype(str) == scenario_overview].copy()

        st.metric("Impacted Sites", impacted_sites_overview(ov_df))

        vol_terminal = volume_by_terminal_product(ov_df, volume_col)
        st.subheader(f"Volume by Product Group × New Terminal ({units})")
        safe_dataframe(vol_terminal)

        vol_tcn = volume_by_product_tcn(ov_df, volume_col)
        st.subheader(f"Volume by Product Group × New TCN ({units})")
        safe_dataframe(vol_tcn)

        shift = terminal_shift_matrix(ov_df, volume_col)
        st.subheader(f"Home Terminal → New Terminal shift ({units})")
        safe_dataframe(shift)

        map_df = build_map_df(ov_df, product_offset=True)
        map_fig = build_map_figure(map_df)
        if map_fig is not None:
            st.subheader("Map")
            st.plotly_chart(
                map_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                },
            )

        overview_table = vol_terminal.copy()
        if not vol_terminal.empty:
            val_cols = [c for c in vol_terminal.columns if c != "New Terminal"]
            if val_cols:
                plot_df = vol_terminal.melt(id_vars=["New Terminal"], value_vars=val_cols, var_name="Product Group", value_name="Volume")
                overview_fig = px.bar(plot_df, x="New Terminal", y="Volume", color="Product Group", title=f"Volume by Terminal ({units})")

    with explorer_tab:
        st.subheader("Data Explorer")
        display_df = _format_explorer(filtered)
        if set(EXPLORER_COLUMNS).issubset(explorer_table(filtered).columns):
            paginated_table(display_df, key="explorer")
        else:
            safe_dataframe(display_df)

    if export_clicked:
        out_dir = Path("exports")
        out_dir.mkdir(exist_ok=True)
        html_path = out_dir / "supply_scenario_report.html"
        build_html_report(
            output_path=html_path,
            title="Supply & Freight Scenario Report",
            overview_fig_html=fig_to_html(overview_fig),
            compare_fig_html=fig_to_html(compare_fig),
            map_fig_html=fig_to_html(map_fig),
            overview_table=overview_table.head(500),
            compare_table=compare_table.head(500),
            explorer_table=_format_explorer(filtered).head(1000),
        )

        if export_images:
            try:
                images = export_product_group_images(
                    filtered,
                    lambda x: build_map_figure(build_map_df(x, product_offset=True)),
                    out_dir / "maps",
                )
                st.success(f"Export complete: {html_path} and {len(images)} map images")
            except Exception as exc:
                st.warning(f"HTML exported ({html_path}), but PNG export failed: {exc}")
        else:
            st.success(f"Export complete: {html_path}")

        st.download_button("Download report HTML", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")


if __name__ == "__main__":
    main()
