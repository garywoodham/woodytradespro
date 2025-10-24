# app.py — WoodyTrades Pro Dashboard (stable build for current utils.py)
# --------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import traceback

from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
    ASSET_SYMBOLS,
)

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="WoodyTrades Pro", layout="wide")
st.title("📈 WoodyTrades Pro — Multi-Asset Signal Dashboard")

# --------------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

interval_key = st.sidebar.selectbox(
    "Select Interval",
    options=["15m", "1h", "4h", "1d", "1wk"],
    index=1,
)

risk = st.sidebar.selectbox(
    "Select Risk Level",
    options=["Low", "Medium", "High"],
    index=1,
)

use_cache = st.sidebar.checkbox("Use cached data", value=True)

selected_asset = st.sidebar.selectbox("Select Asset", list(ASSET_SYMBOLS.keys()), index=0)

# --------------------------------------------------------------------------------------
# SAFETY WRAPPER — so Streamlit never goes blank
# --------------------------------------------------------------------------------------
def safe_section(fn):
    try:
        fn()
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.exception(traceback.format_exc())

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Overview", "📊 Asset Detail", "🧪 Scenarios"])

# --------------------------------------------------------------------------------------
# TAB 1 — MARKET OVERVIEW
# --------------------------------------------------------------------------------------
with tab1:
    def render_overview():
        st.subheader("🌍 Market Overview")
        st.caption("Aggregated signals and backtest metrics for all tracked assets.")
        df_summary = summarize_assets(interval_key=interval_key, risk=risk, use_cache=use_cache)

        if df_summary is None or df_summary.empty:
            st.warning("⚠️ No data available. Try disabling cache or check your internet connection.")
            return

        st.dataframe(df_summary, use_container_width=True)

    safe_section(render_overview)

# --------------------------------------------------------------------------------------
# TAB 2 — ASSET DETAIL
# --------------------------------------------------------------------------------------
with tab2:
    def render_asset_detail():
        st.subheader(f"📊 Detailed Analysis — {selected_asset}")
        pred, df = asset_prediction_and_backtest(selected_asset, interval_key, risk, use_cache=use_cache)

        if pred is None or df is None or df.empty:
            st.warning("⚠️ Not enough data to compute indicators or signals.")
            return

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal", pred["side"])
        col2.metric("Probability", f"{pred['probability']}%")
        col3.metric("Win Rate", f"{pred.get('win_rate', 0):.1f}%")
        col4.metric("Backtest Return", f"{pred.get('backtest_return_pct', 0):.2f}%")

        st.write("---")

        # Price chart
        st.line_chart(df["Close"], use_container_width=True)
        st.caption(f"Closing Price — {selected_asset} ({interval_key})")

        # Trade history table
        trades = pred.get("trades", [])
        if trades:
            st.subheader("Trade History")
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No completed trades to display yet.")

    safe_section(render_asset_detail)

# --------------------------------------------------------------------------------------
# TAB 3 — SCENARIO TESTING
# --------------------------------------------------------------------------------------
with tab3:
    def render_scenarios():
        st.subheader("🧪 Scenario Simulation")
        st.caption("Re-evaluate the selected asset with different intervals and risk levels.")

        alt_interval = st.selectbox("Interval to Test", ["15m", "1h", "4h", "1d", "1wk"], index=1)
        alt_risk = st.selectbox("Risk Level to Test", ["Low", "Medium", "High"], index=1)

        pred_alt, df_alt = asset_prediction_and_backtest(selected_asset, alt_interval, alt_risk, use_cache=use_cache)

        if pred_alt is None or df_alt is None or df_alt.empty:
            st.warning("⚠️ Could not generate scenario data. Try another interval or disable cache.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Signal", pred_alt["side"])
        col2.metric("Probability", f"{pred_alt['probability']}%")
        col3.metric("ATR", f"{pred_alt.get('atr', 0):.2f}")

        st.write("---")
        st.line_chart(df_alt["Close"], use_container_width=True)
        st.caption(f"Scenario price curve — {selected_asset} ({alt_interval})")

    safe_section(render_scenarios)

# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.write("---")
st.caption("© 2025 WoodyTrades Pro — Forecast Project (Streamlit build)")

# --------------------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------------------