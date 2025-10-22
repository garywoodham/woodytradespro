import streamlit as st
import utils
import plotly.express as px

def render_overview():
    st.title("📊 Market Overview")
    st.caption("AI-driven multi-asset summary with prediction accuracy, probability, and performance insights.")

    # Risk selector (affects TP/SL levels)
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()))

    # Fetch and summarize results
    results_df = utils.summarize_assets()

    if results_df.empty:
        st.warning("No assets could be analyzed. Check your internet connection or data source.")
        return

    # Display data table
    st.dataframe(results_df, use_container_width=True)

    # Visualization: Confidence per asset
    fig_conf = px.bar(
        results_df,
        x="Asset",
        y="Confidence",
        color="Prediction",
        title="📈 Prediction Confidence by Asset",
        text="Confidence",
        height=450
    )
    st.plotly_chart(fig_conf, width="stretch")

    # Visualization: Accuracy
    fig_acc = px.bar(
        results_df,
        x="Asset",
        y="Accuracy",
        color="Prediction",
        title="🎯 Model Accuracy by Asset",
        text="Accuracy",
        height=450
    )
    st.plotly_chart(fig_acc, width="stretch")

    # Overall average accuracy
    avg_acc = results_df["Accuracy"].mean()
    st.metric("📊 Overall Model Accuracy", f"{avg_acc:.2f}%")