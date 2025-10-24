# app.py — WoodyTradesPro Smart v2 (Full + Clean, fixed duplicate selectbox IDs)
# ---------------------------------------------------------------------------

import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"  # prevent reload spam

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback
import warnings

from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
)

warnings.filterwarnings("ignore", message="Please replace `use_container_width`")

# ---------------------------------------------------------------------------
# APP CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WoodyTradesPro Forecast", layout="wide")
st.title("📈 WoodyTradesPro Forecast (Smart v2)")

tabs = st.tabs(["📊 Market Summary", "🎯 Predictions", "📈 Chart View", "🔍 Backtest Results"])

# ---------------------------------------------------------------------------
# TAB 1 — MARKET SUMMARY
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📊 Multi-Asset Market Summary")

    col1, col2 = st.columns(2)
    interval_key = col1.selectbox(
        "Interval", ["15m", "1h", "4h", "1d", "1wk"], index=1, key="summary_interval"
    )
    risk = col2.selectbox(
        "Risk Level", ["Low", "Medium", "High"], index=1, key="summary_risk"
    )

    st.info(f"Fetching latest market data for interval **{interval_key}**, risk profile **{risk}**...")

    @st.cache_data(ttl=3600)
    def load_summary(interval_key, risk):
        return summarize_assets(interval_key, risk, use_cache=True)

    try:
        df_summary = load_summary(interval_key, risk)
        if df_summary.empty:
            st.warning("⚠️ No market data could be loaded.")
        else:
            st.dataframe(df_summary, width="stretch")
    except Exception as e:
        st.error(f"Error loading summary: {e}")
        st.text(traceback.format_exc())

# ---------------------------------------------------------------------------
# TAB 2 — PREDICTIONS
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🎯 Asset Predictions and Backtest")

    asset = st.selectbox(
        "Select Asset",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="pred_asset",
    )
    interval_key = st.selectbox(
        "Interval", ["15m", "1h", "4h", "1d", "1wk"], index=1, key="pred_interval"
    )
    risk = st.selectbox(
        "Risk Level", ["Low", "Medium", "High"], index=1, key="pred_risk"
    )

    st.info(f"Analyzing {asset} at {interval_key} timeframe with {risk} risk...")

    try:
        result, df = asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True)
        if not result:
            st.warning("⚠️ Could not generate prediction.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Signal", result["side"])
            col2.metric("Probability", f"{result['probability']}%")
            col3.metric("Win Rate", f"{result['win_rate']:.1f}%")
            col4.metric("Trades", result["n_trades"])

            st.markdown(f"""
            **Current Price:** {result['price']:.2f}  
            **Take Profit:** {result['tp']}  
            **Stop Loss:** {result['sl']}  
            **Sentiment:** {result['sentiment']:.2f}  
            **Market Regime:** {result['regime']}  
            **ML Probability (Up):** {result['ml_prob']:.2f}
            """)

            if result["trades"]:
                st.subheader("📜 Trade Log")
                st.dataframe(pd.DataFrame(result["trades"]), width="stretch")
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        st.text(traceback.format_exc())

# ---------------------------------------------------------------------------
# TAB 3 — CHART VIEW
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📈 Interactive Chart View")

    asset = st.selectbox(
        "Asset",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="chart_asset",
    )
    interval_key = st.selectbox(
        "Interval", ["15m", "1h", "4h", "1d", "1wk"], index=1, key="chart_interval"
    )

    try:
        symbol, df = load_asset_with_indicators(asset, interval_key, use_cache=True)
        if df.empty:
            st.warning("No data available for this asset.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                               low=df["Low"], close=df["Close"], name="Price")
            )
            if "ema20" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20"))
            if "ema50" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50"))
            fig.update_layout(title=f"{asset} ({symbol}) — {interval_key}", height=600)
            st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Chart failed to load: {e}")
        st.text(traceback.format_exc())

# ---------------------------------------------------------------------------
# TAB 4 — BACKTEST RESULTS
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🔍 Backtest Results Summary")

    asset = st.selectbox(
        "Select Asset for Backtest",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="bt_asset",
    )
    interval_key = st.selectbox(
        "Interval", ["15m", "1h", "4h", "1d", "1wk"], index=1, key="bt_interval"
    )
    risk = st.selectbox(
        "Risk Level", ["Low", "Medium", "High"], index=1, key="bt_risk"
    )

    try:
        result, df = asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True)
        if not result:
            st.warning("⚠️ Could not run backtest.")
        else:
            st.metric("Backtest Win Rate", f"{result['win_rate']:.1f}%")
            st.metric("Total Return", f"{result['backtest_return_pct']:.2f}%")
            if result["trades"]:
                st.subheader("📄 Detailed Trade History")
                st.dataframe(pd.DataFrame(result["trades"]), width="stretch")
    except Exception as e:
        st.error(f"Error in backtest: {e}")
        st.text(traceback.format_exc())

# ---------------------------------------------------------------------------
# END OF APP
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 WoodyTradesPro | Smart v2 Forecast System")