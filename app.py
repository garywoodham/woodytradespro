# app.py - Woody Trades Pro Dashboard
# Smart v7.9 Calibrated Confidence + EV/PF + Dynamic Filter UI

import os
import traceback
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
    ASSET_SYMBOLS,
    INTERVALS,
)

# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"
st.set_page_config(page_title="Woody Trades Pro - Smart v7.9", layout="wide")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Settings")

interval_key = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=list(INTERVALS.keys()).index("1h") if "1h" in INTERVALS else 0,
    help="Candle timeframe used for signals and backtest window.",
)

risk = st.sidebar.selectbox(
    "Risk Profile",
    options=["Low", "Medium", "High"],
    index=1,
    help="Controls TP/SL distance (ATR multiples).",
)

filter_level = st.sidebar.selectbox(
    "Signal Filtering Level",
    options=["Loose", "Balanced", "Strict"],
    index=1,
    help="Adjusts how strict the trade filter is. Loose = more signals, Strict = fewer but higher confidence.",
)

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    help="Used in the Detailed / Chart sections below.",
)

st.sidebar.caption("Smart v7.9 • Calibrated ML • EV/PF • Adaptive TP/SL • Dynamic Filter")

# ---------------------------------------------------------------------
# Cached calls
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_summary(interval_key: str, risk: str, filter_level: str) -> pd.DataFrame:
    return summarize_assets(interval_key, risk, use_cache=True, filter_level=filter_level)

@st.cache_data(show_spinner=False)
def load_prediction_and_chart(asset: str, interval_key: str, risk: str, filter_level: str):
    return asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True, filter_level=filter_level)

@st.cache_data(show_spinner=False)
def load_price_df(asset: str, interval_key: str, filter_level: str):
    return load_asset_with_indicators(asset, interval_key, use_cache=True, filter_level=filter_level)

# ---------------------------------------------------------------------
# MARKET SUMMARY
# ---------------------------------------------------------------------
st.header("📊 Market Summary")

try:
    df_summary = load_summary(interval_key, risk, filter_level)
    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        # Visual enhancements
        st.dataframe(
            df_summary[
                [
                    "Asset", "Signal", "Probability", "WinRate",
                    "EV%", "ProfitFactor", "Trades", "Return%",
                    "MaxDD%", "SharpeLike", "Stale"
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# DETAILED VIEW
# ---------------------------------------------------------------------
st.header("🔍 Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(asset_choice, interval_key, risk, filter_level)

    if not pred_block:
        st.warning("No prediction data available for this asset.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Signal", pred_block.get("side", "Hold"))
        colB.metric("Confidence (Raw)", f"{round(pred_block.get('probability_raw', 0)*100,2)}%")
        colC.metric("Confidence (Calibrated)", f"{round(pred_block.get('probability_calibrated', 0)*100,2)}%")
        colD.metric("Win Rate", f"{round(pred_block.get('win_rate', 0),2)}%")

        colE, colF, colG, colH = st.columns(4)
        colE.metric("TP", f"{pred_block.get('tp', 0):.4f}" if pred_block.get("tp") else "—")
        colF.metric("SL", f"{pred_block.get('sl', 0):.4f}" if pred_block.get("sl") else "—")
        colG.metric("R/R", f"{pred_block.get('rr', 0):.2f}" if pred_block.get("rr") else "—")
        colH.metric("EV%", f"{pred_block.get('ev_pct', 0):.3f}")

        colI, colJ, colK, colL = st.columns(4)
        colI.metric("Profit Factor", f"{pred_block.get('profit_factor', 0):.2f}")
        colJ.metric("Trades", f"{pred_block.get('trades', 0)}")
        colK.metric("Return%", f"{pred_block.get('backtest_return_pct', 0):.2f}")
        stale_flag = "⚠️ STALE" if pred_block.get("stale") else "✅ Fresh"
        colL.metric("Data Status", stale_flag)

    # Chart section
    if isinstance(df_asset_ind, pd.DataFrame) and not df_asset_ind.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_asset_ind.index,
            open=df_asset_ind["Open"],
            high=df_asset_ind["High"],
            low=df_asset_ind["Low"],
            close=df_asset_ind["Close"],
            name="Price",
        ))
        if "ema20" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(x=df_asset_ind.index, y=df_asset_ind["ema20"], mode="lines", name="EMA20"))
        if "ema50" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(x=df_asset_ind.index, y=df_asset_ind["ema50"], mode="lines", name="EMA50"))

        # Add buy/sell markers if available
        if "buy_times" in df_asset_ind.attrs and "buy_prices" in df_asset_ind.attrs:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.attrs["buy_times"],
                y=df_asset_ind.attrs["buy_prices"],
                mode="markers",
                name="Buy",
                marker_symbol="triangle-up",
                marker_color="green",
                marker_size=8,
            ))
        if "sell_times" in df_asset_ind.attrs and "sell_prices" in df_asset_ind.attrs:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.attrs["sell_times"],
                y=df_asset_ind.attrs["sell_prices"],
                mode="markers",
                name="Sell",
                marker_symbol="triangle-down",
                marker_color="red",
                marker_size=8,
            ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=450,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset.")

except Exception as e:
    st.error(f"Error in detailed view: {e}")
    st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# DEBUG / RAW DATA
# ---------------------------------------------------------------------
st.header("🧰 Debug / Raw Data")
with st.expander("Show model input / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, sig_pts = load_price_df(asset_choice, interval_key, filter_level)
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(df_asset_full.tail(200), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())

st.caption("Smart v7.9 • Calibrated ML • EV/PF • Adaptive TP/SL • Dynamic Filter • ATR / ADX regimes")