import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback
from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
    asset_prediction_single,
)

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="WoodyTrades Pro",
    page_icon="💹",
    layout="wide"
)

st.title("💹 WoodyTrades Pro — Smart v6.3")
st.markdown("### Adaptive ML & Sentiment-based Forecasting Dashboard")

# --------------------------------------------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

interval_key = st.sidebar.selectbox(
    "Interval",
    options=["15m", "1h", "4h", "1d", "1wk"],
    index=1,
    key="interval_select"
)

risk = st.sidebar.selectbox(
    "Risk Level",
    options=["Low", "Medium", "High"],
    index=1,
    key="risk_select"
)

tabs = st.tabs(["📊 Market Summary", "📈 Asset Analysis", "🧪 Backtest"])

# --------------------------------------------------------------------------------------
# TAB 1 — MARKET SUMMARY
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📊 Market Overview")

    @st.cache_data(show_spinner=True)
    def load_summary(interval_key, risk):
        return summarize_assets(interval_key, risk, use_cache=True)

    try:
        df_summary = load_summary(interval_key, risk)
        if isinstance(df_summary, pd.DataFrame) and not df_summary.empty:
            st.dataframe(df_summary, width="stretch")
        else:
            st.warning("⚠️ No summary data available. Please retry.")
    except Exception as e:
        st.error(f"Error loading summary: {e}")
        st.text(traceback.format_exc())

# --------------------------------------------------------------------------------------
# TAB 2 — ASSET ANALYSIS
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("📈 Asset Prediction and Analysis")

    asset = st.selectbox(
        "Select Asset",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="asset_select"
    )

    if st.button("🔍 Analyze Asset"):
        with st.spinner(f"Analyzing {asset}..."):
            try:
                pred = asset_prediction_single(asset, interval_key, risk)
                if pred and isinstance(pred, dict):
                    st.write("### Signal Summary")
                    st.json(pred)

                    symbol, df = load_asset_with_indicators(asset, interval_key)
                    if not df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df["Open"], high=df["High"],
                            low=df["Low"], close=df["Close"],
                            name="Price"
                        ))
                        fig.update_layout(
                            title=f"{asset} ({symbol}) — {interval_key} Chart",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            width=1200, height=600,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, config={"responsive": True})
                    else:
                        st.warning("⚠️ No chart data available.")
                else:
                    st.warning("⚠️ Could not generate prediction.")
            except Exception as e:
                st.error(f"Error analyzing asset: {e}")
                st.text(traceback.format_exc())

# --------------------------------------------------------------------------------------
# TAB 3 — BACKTEST
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🧪 Backtest Results")

    asset_bt = st.selectbox(
        "Select Asset for Backtest",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="bt_asset_select"
    )

    if st.button("▶ Run Backtest"):
        with st.spinner(f"Running backtest for {asset_bt}..."):
            try:
                df_bt, stats = asset_prediction_and_backtest(asset_bt, interval_key, risk)
                if isinstance(df_bt, pd.DataFrame) and not df_bt.empty:
                    st.write("### Backtest Metrics")
                    st.json(stats)

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=df_bt.index,
                        y=df_bt["Close"],
                        mode="lines",
                        name="Price"
                    ))
                    fig_bt.update_layout(
                        title=f"{asset_bt} ({interval_key}) — Backtest Price Curve",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        width=1200, height=600,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_bt, config={"responsive": True})
                else:
                    st.warning("⚠️ No backtest data available.")
            except Exception as e:
                st.error(f"Error during backtest: {e}")
                st.text(traceback.format_exc())

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 WoodyTrades Pro — AI-enhanced Forecasting Engine (Smart v6.3)")