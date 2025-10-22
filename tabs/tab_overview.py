import streamlit as st
import pandas as pd
import utils
import plotly.graph_objects as go

def render_overview():
    st.title("📊 Market Overview")
    st.caption("AI-driven multi-asset overview — predictions, probabilities, and accuracy for all tracked instruments.")

    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()))
    st.divider()

    st.info("Fetching and analyzing market data... please wait ⏳")
    results_df = utils.summarize_assets()

    if results_df.empty:
        st.error("No assets could be analyzed. Please check your internet connection or data source.")
        return

    st.subheader("📈 Model Output Summary")
    st.dataframe(results_df, width="stretch")

    # Plot confidence bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["Asset"],
        y=results_df["Confidence"],
        name="Confidence %",
        marker_color="steelblue"
    ))
    fig.add_trace(go.Bar(
        x=results_df["Asset"],
        y=results_df["Accuracy"],
        name="Accuracy %",
        marker_color="lightgreen"
    ))

    fig.update_layout(
        title="Confidence vs Accuracy by Asset",
        barmode="group",
        xaxis_title="Asset",
        yaxis_title="Percentage (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.success("✅ Overview analysis complete.")