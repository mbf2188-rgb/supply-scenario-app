from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.constants import EXPLORER_COLUMNS


def apply_theme() -> None:
    """Apply a consistent 7-Eleven inspired style using Streamlit's active base theme."""
    base = st.get_option("theme.base") or "light"
    dark_mode = str(base).lower() == "dark"

    bg = "#0b1220" if dark_mode else "#f4f7fb"
    fg = "#e5e7eb" if dark_mode else "#111827"
    sidebar_bg = "#0f1a30" if dark_mode else "#1f2937"  # always dark for filter contrast
    sidebar_fg = "#f9fafb"
    card_bg = "#111b31" if dark_mode else "#ffffff"
    border = "#1f3a63" if dark_mode else "#c7d2e3"
    accent_red = "#d71920"
    accent_green = "#0a8f3d"
    accent_orange = "#ff7a00"
def apply_theme(dark_mode: bool) -> None:
    bg = "#0c1730" if dark_mode else "#f7f9fc"
    fg = "#e5e7eb" if dark_mode else "#111827"
    card_bg = "#111f3d" if dark_mode else "#ffffff"
    border = "#1f2f55" if dark_mode else "#d1d5db"
    accent = "#e11d48"  # subtle 7-Eleven red accent
    green = "#16a34a"  # subtle 7-Eleven green accent
    tab_bg = "#0f1b36" if dark_mode else "#eef2ff"

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {fg}; }}
        [data-testid="stHeader"] {{ border-top: 3px solid {accent_red}; }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {sidebar_bg} 0%, #111827 100%);
            color: {sidebar_fg};
            border-right: 1px solid #26344f;
        }}
        [data-testid="stSidebar"] * {{ color: {sidebar_fg} !important; }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stTextInput label {{
            font-size: 0.98rem !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #0b1220 !important;
            color: #f9fafb !important;
            border: 1px solid #334155 !important;
            font-size: 1rem !important;
        }}

        [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stDataFrame"] {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab-list"] button {{
            background: {card_bg};
            border: 1px solid {border};
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            color: {fg};
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            border-top: 3px solid {accent_green} !important;
            box-shadow: inset 0 -3px 0 {accent_orange};
        }}

        .stButton > button {{ border: 1px solid {accent_green}; }}
        .stButton > button:hover {{ border-color: {accent_orange}; color: {accent_orange}; }}
        [data-testid="stSidebar"] {{ background: {card_bg}; border-right: 1px solid {border}; }}
        [data-testid="stMetric"] {{ background: {card_bg}; border: 1px solid {border}; border-radius: 10px; padding: 10px; }}
        [data-testid="stExpander"] {{ background: {card_bg}; border: 1px solid {border}; border-radius: 10px; }}
        [data-testid="stDataFrame"] {{ background: {card_bg}; border: 1px solid {border}; border-radius: 10px; }}
        .stTabs [data-baseweb="tab-list"] button {{ background: {tab_bg}; border-radius: 8px 8px 0 0; }}
        .stTabs [aria-selected="true"] {{ border-bottom: 3px solid {accent} !important; }}
        .stButton > button {{ border: 1px solid {border}; }}
        .stButton > button:hover {{ border-color: {green}; }}
        h1, h2, h3 {{ color: {fg}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def choose_scenario_defaults(scenarios: list[str]) -> tuple[str, str]:
    sorted_scenarios = sorted(scenarios)
    default = "Today" if "Today" in sorted_scenarios else (sorted_scenarios[0] if sorted_scenarios else "")
    second = next((s for s in sorted_scenarios if s != default), default)
    return default, second


def global_filter(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if not text:
        return df
    needle = text.lower()
    mask = df.astype(str).apply(lambda c: c.str.lower().str.contains(needle, na=False)).any(axis=1)
    return df[mask]


def safe_dataframe(df: pd.DataFrame, *, height: int | None = None) -> None:
    try:
        if height is None:
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True, height=height)
    except Exception:
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
    for col in EXPLORER_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    rate_cols = ["Primary Freight Rate", "New Freight Rate", "Primary Supply Rate", "New Supply Rate"]
    for col in rate_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out[EXPLORER_COLUMNS]
