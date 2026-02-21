from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.constants import EXPLORER_COLUMNS


def apply_theme(night_mode: bool) -> None:
    if not night_mode:
        return
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def choose_scenario_defaults(scenarios: list[str]) -> tuple[str, str]:
    sorted_scenarios = sorted(scenarios)
    default = "Today" if "Today" in sorted_scenarios else (sorted_scenarios[0] if sorted_scenarios else "")
    return default, default


def global_filter(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if not text:
        return df
    needle = text.lower()
    mask = df.astype(str).apply(lambda c: c.str.lower().str.contains(needle, na=False)).any(axis=1)
    return df[mask]


def safe_dataframe(df: pd.DataFrame, *, height: int | None = None) -> None:
    """Render dataframe with backward-compatible width behavior across Streamlit versions."""
    try:
        st.dataframe(df, use_container_width=True, height=height)
    except TypeError:
        # Older Streamlit builds may fail on width-related kwargs.
        if height is None:
            st.dataframe(df)
        else:
            st.dataframe(df, height=height)


def paginated_table(df: pd.DataFrame, key: str) -> None:
    size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1, key=f"{key}_size")
    pages = max(1, (len(df) + size - 1) // size)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"{key}_page")
    start = (page - 1) * size
    safe_dataframe(df.iloc[start : start + size], height=460)


def explorer_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Monthly volume"] = out.get("Monthly Volume (bbl 30 day)", 0)
    for col in EXPLORER_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[EXPLORER_COLUMNS]
