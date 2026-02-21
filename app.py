import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Supply + Freight Scenario Viewer")

uploaded_file = st.file_uploader("Upload scenario file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:

    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [c.strip() for c in df.columns]

    st.success(f"Loaded {len(df)} rows")

    # Required column check
    required = [
        "Scenario",
        "Site ID",
        "Product Group",
        "Home Terminal",
        "New Terminal",
        "Total Cost (30 days)"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success("Schema validation passed")

        scenarios = df["Scenario"].unique()
        scenario = st.selectbox("Select Scenario", scenarios)

        df_s = df[df["Scenario"] == scenario]

        total_30 = df_s["Total Cost (30 days)"].sum()
        total_1y = total_30 * (365/30)

        col1, col2 = st.columns(2)
        col1.metric("Total Cost (30d)", f"${total_30:,.0f}")
        col2.metric("Total Cost (1y)", f"${total_1y:,.0f}")

        if "Latitude" in df.columns and "Longitude" in df.columns:
            fig = px.scatter_mapbox(
                df_s,
                lat="Latitude",
                lon="Longitude",
                color="New Terminal",
                hover_name="Site ID",
                mapbox_style="open-street-map",
                zoom=4
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_s.head(100))
