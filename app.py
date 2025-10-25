# ======================================================================================
# app.py — Smart v7 dashboard (Market Summary + Asset Analysis + Backtest)
# ======================================================================================

import os
import traceback
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --------------------------------------------------------------------------------------
# ENVIRONMENT / WATCHER FIX
# --------------------------------------------------------------------------------------
# Prevent inotify watch-limit crash
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# Streamlit deprecation suppressions
st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_page_config(page_title="Woody Trades Pro - Smart v7", layout="wide")

# --------------------------------------------------------------------------------------
# IMPORT UTILS
# --------------------------------------------------------------------------------------
from utils import (
    summarize_assets,
    asset_prediction_single,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
)

# --------------------------------------------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

interval_key = st.sidebar.selectbox(
    "Select timeframe",
    options=["15m", "1h", "4h", "1d"],
    index=1,
    key="interval_selectbox",
)

risk_level = st.sidebar.selectbox(
    "Select risk level",
    options=["Low", "Medium", "High"],
    index=1,
    key="risk_selectbox",
)

refresh_button = st.sidebar.button("🔄 Refresh Data", key="refresh_button")

# --------------------------------------------------------------------------------------
# MAIN TITLE
# --------------------------------------------------------------------------------------
st.title("📈 Woody Trades Pro — Smart v7 AI Market Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Interval: {interval_key} | Risk: {risk_level}")

# --------------------------------------------------------------------------------------
# TAB SETUP
# --------------------------------------------------------------------------------------
tabs = st.tabs([
    "🌍 Market Summary",
    "🔍 Asset Analysis",
    "📊 Backtest & Performance"
])

# ======================================================================================
# 🌍 TAB 1 — MARKET SUMMARY
# ======================================================================================
with tabs[0]:
    st.subheader("🌍 Market Summary (Smart v7)")

    try:
        with st.spinner("Fetching and analyzing market data (smart v7)..."):
            df_summary = summarize_assets(interval_key, risk_level, use_cache=not refresh_button)

        if df_summary is not None and not df_summary.empty:
            st.dataframe(
                df_summary,
                width='stretch',
                hide_index=True,
                use_container_width=False,
            )

            # Plot Probability vs Sentiment
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_summary["Sentiment"],
                y=df_summary["Probability"],
                mode="markers+text",
                text=df_summary["Asset"],
                textposition="top center",
            ))
            fig.update_layout(
                title="Market Sentiment vs Probability",
                xaxis_title="Sentiment",
                yaxis_title="Probability",
                template="plotly_dark",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("No market summary data available.")

    except Exception as e:
        st.error(f"Error loading summary: {e}")
        st.code(traceback.format_exc())

# ======================================================================================
# 🔍 TAB 2 — ASSET ANALYSIS
# ======================================================================================
with tabs[1]:
    st.subheader("🔍 Asset Analysis")

    asset = st.selectbox(
        "Choose Asset:",
        list(df_summary["Asset"]) if 'df_summary' in locals() and not df_summary.empty else
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="asset_selectbox",
    )

    if asset:
        try:
            with st.spinner(f"Analyzing {asset} ..."):
                symbol, df = load_asset_with_indicators(asset, interval_key)
                result = asset_prediction_single(asset, interval_key, risk_level)

            if isinstance(result, dict):
                st.markdown(f"### {asset} ({symbol}) — Current Signal")

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Side", result.get("side", "Hold"))
                c2.metric("Probability", f"{result.get('probability', 0.0):.2f}")
                c3.metric("Sentiment", f"{result.get('sentiment', 0.0):.2f}")
                c4.metric("TP", f"{result.get('tp', 0.0):,.2f}")
                c5.metric("SL", f"{result.get('sl', 0.0):,.2f}")
                c6.metric("RR", f"{result.get('rr', 0.0):.2f}")

                # Chart
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Price",
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df["ema20"],
                        line=dict(width=1.5),
                        name="EMA20",
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df["ema50"],
                        line=dict(width=1.5),
                        name="EMA50",
                    ))
                    fig.update_layout(
                        title=f"{asset} Price & EMA Overview",
                        template="plotly_dark",
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.warning("No chart data available.")

        except Exception as e:
            st.error(f"Error analyzing asset: {e}")
            st.code(traceback.format_exc())

# ======================================================================================
# 📊 TAB 3 — BACKTEST
# ======================================================================================
with tabs[2]:
    st.subheader("📊 Strategy Backtest & Performance")

    asset_bt = st.selectbox(
        "Select asset for backtest:",
        list(df_summary["Asset"]) if 'df_summary' in locals() and not df_summary.empty else
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
        key="asset_backtest_selectbox",
    )

    if asset_bt:
        try:
            with st.spinner(f"Running backtest for {asset_bt}..."):
                df_bt, stats = asset_prediction_and_backtest(asset_bt, interval_key, risk_level)

            if stats:
                st.markdown("### 📈 Backtest Results")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Win Rate", f"{stats.get('winrate', 0.0):.1f}%")
                col2.metric("Trades", f"{stats.get('trades', 0)}")
                col3.metric("Return", f"{stats.get('return', 0.0):.2f}%")
                col4.metric("MaxDD", f"{stats.get('maxdd', 0.0):.2f}%")
                col5.metric("Sharpe-like", f"{stats.get('sharpe', 0.0):.2f}")

            # Draw price + EMA chart
            if df_bt is not None and not df_bt.empty:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Candlestick(
                    x=df_bt.index,
                    open=df_bt["Open"],
                    high=df_bt["High"],
                    low=df_bt["Low"],
                    close=df_bt["Close"],
                    name="Price",
                ))
                if "ema20" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt["ema20"], name="EMA20"))
                if "ema50" in df_bt.columns:
                    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt["ema50"], name="EMA50"))
                fig_bt.update_layout(
                    title=f"{asset_bt} Price + EMA Backtest View",
                    template="plotly_dark",
                    height=600,
                )
                st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("No backtest data available.")

        except Exception as e:
            st.error(f"Error during backtest: {e}")
            st.code(traceback.format_exc())

# ======================================================================================
# FOOTER
# ======================================================================================
st.markdown("---")
st.caption("© 2025 Woody Trades Pro | Smart v7 AI system — all analytics, no guarantees.")