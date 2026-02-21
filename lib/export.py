from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.io as pio
from jinja2 import Template


def build_html_report(
    output_path: Path,
    title: str,
    overview_fig_html: str,
    compare_fig_html: str,
    map_fig_html: str,
    overview_table: pd.DataFrame,
    compare_table: pd.DataFrame,
    explorer_table: pd.DataFrame,
) -> Path:
    template = Template(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 16px; }
h2 { margin-top: 28px; }
.table { border-collapse: collapse; width: 100%; margin-top: 8px; }
.table th, .table td { border: 1px solid #ddd; padding: 6px; font-size: 12px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<h2>Overview</h2>
{{ overview_fig_html }}
{{ overview_table_html }}
<h2>Compare</h2>
{{ compare_fig_html }}
{{ compare_table_html }}
<h2>Explorer</h2>
{{ map_fig_html }}
{{ explorer_table_html }}
</body></html>
"""
    )
    html = template.render(
        title=title,
        overview_fig_html=overview_fig_html,
        compare_fig_html=compare_fig_html,
        map_fig_html=map_fig_html,
        overview_table_html=overview_table.to_html(index=False, classes="table"),
        compare_table_html=compare_table.to_html(index=False, classes="table"),
        explorer_table_html=explorer_table.to_html(index=False, classes="table"),
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def fig_to_html(fig) -> str:
    if fig is None:
        return "<p>No figure available.</p>"
    return pio.to_html(fig, include_plotlyjs=True, full_html=False)


def export_product_group_images(df: pd.DataFrame, build_fig_func, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for group in ["Regular", "Premium", "Diesel"]:
        subset = df[df["Product Group"].astype(str) == group]
        if subset.empty:
            continue
        fig = build_fig_func(subset)
        if fig is None:
            continue
        p = out_dir / f"map_{group.lower()}.png"
        fig.write_image(str(p), width=1400, height=900)
        created.append(p)
    return created
