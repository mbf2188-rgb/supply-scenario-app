from __future__ import annotations

import io

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        if name.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", sheet_name=0)
    except Exception as exc:
        raise ValueError(f"Unable to parse {filename}: {exc}") from exc
    raise ValueError("Unsupported file format. Please upload CSV or XLSX.")
