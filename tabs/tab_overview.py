import streamlit as st
import plotly.graph_objects as go
import utils
import pandas as pd

def render_overview():
    """Dashboard Overview — asset summary and market snapshot."""
    st.title("📊 Market Overview")
    st.caption("AI-driven multi-asset summary with prediction accuracy and trade recommendations.")

    # Risk selector (affects TP/SL levels)
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()))

    results_df = utils.summarize_assets()

    if results_df.empty:
        st.warning("⚠️ No assets could be analyzed. Check your internet connection or data source.")
        return

    st.subheader("Asset Summary")
    st.dataframe(
        results_df.style.format({
            "Probability": "{:.2%}",
            "Accuracy": "{:.2f}%",
            "Take Profit": "{:.2f}",
            "Stop Loss": "{:.2f}"
        })
        .background_gradient(cmap="RdYlGn", subset=["Accuracy"])
    )

    # Display key insights
    avg_acc = results_df["Accuracy"].mean()
    buys = (results_df["Prediction"] == "BUY").sum()
    sells = (results_df["Prediction"] == "SELL").sum()
    holds = (results_df["Prediction"] == "HOLD").sum()

    st.markdown("### 🔍 Summary Insights")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Model Accuracy", f"{avg_acc:.2f}%")
    col2.metric("Buy Signals", buys)
    col3.metric("Sell Signals", sells)
    col4.metric("Hold Signals", holds)

    # Accuracy distribution chart
    st.markdown("### 🎯 Model Accuracy by Asset")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["Asset"],
        y=results_df["Accuracy"],
        marker_color="green"
    ))
    fig.update_layout(
        xaxis_title="Asset",
        yaxis_title="Model Accuracy (%)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "✅ **Interpretation Guide:**\n\n"
        "- **Prediction** shows AI’s suggested trade action.\n"
        "- **Probability** indicates confidence in that signal.\n"
        "- **Accuracy** reflects backtested historical precision.\n"
        "- **TP/SL** give dynamic take-profit and stop-loss levels based on risk level."
    )