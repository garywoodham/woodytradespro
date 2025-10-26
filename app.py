# app.py – Woody Trades Pro dashboard (smart v7.8.1)
# Streamlit UI layer for utils.py (regime adaptive + dynamic filtering)

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
# Hardening: prevent Streamlit Cloud file watcher overflow
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

st.set_page_config(
    page_title="Woody Trades Pro – Smart v7.8.1",
    layout="wide"
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.title("Settings")

interval_key = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=list(INTERVALS.keys()).index("1h") if "1h" in INTERVALS else 0,
    key="sidebar_interval",
    help="Candle timeframe used for signals, TP/SL, and backtest window.",
)

risk = st.sidebar.selectbox(
    "Risk Profile",
    options=["Low", "Medium", "High"],
    index=["Low", "Medium", "High"].index("Medium"),
    key="sidebar_risk",
    help="Controls TP/SL distance (ATR multiples).",
)

# 🆕 Dynamic filtering level control
filter_level = st.sidebar.selectbox(
    "Signal Filtering Level",
    options=["Loose", "Balanced", "Strict"],
    index=1,
    key="sidebar_filter",
    help=(
        "Controls how selective the signal engine is:\n"
        "• Loose → many trades, permissive thresholds\n"
        "• Balanced → default (recommended)\n"
        "• Strict → fewer, higher-confidence trades"
    ),
)

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
    help="Used in the Detailed / Scenarios sections below.",
)

st.sidebar.caption("v7.8.1 engine • regime-adaptive • ATR TP/SL • ML blend • filter control")

# ---------------------------------------------------------------------
# Cached loaders – include filter_level in keys
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
# 1) MARKET SUMMARY
# ---------------------------------------------------------------------

st.header("📊 Market Summary")

try:
    df_summary = load_summary(interval_key, risk, filter_level)
    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        st.dataframe(df_summary, width="stretch", hide_index=True)
except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# 2) DETAILED VIEW
# ---------------------------------------------------------------------

st.header("🔍 Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(asset_choice, interval_key, risk, filter_level)

    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        colA.metric("Signal", pred_block.get("side") or pred_block.get("signal") or "Hold")
        prob_val = pred_block.get("probability") or pred_block.get("prob") or 0.0
        colB.metric("Confidence", f"{round(prob_val*100,2) if prob_val<=1 else prob_val:.2f}%")
        wr = pred_block.get("win_rate") or pred_block.get("winrate") or 0.0
        colC.metric("Win Rate (backtest)", f"{wr:.2f}%")
        tr = pred_block.get("trades") or 0
        colD.metric("Trades (backtest)", str(tr))

        colE, colF, colG, colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("TP", f"{tp_val:.4f}" if tp_val else "—")
        colF.metric("SL", f"{sl_val:.4f}" if sl_val else "—")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "—")

        sent_val = pred_block.get("sentiment") or pred_block.get("Sentiment")
        colH.metric("Sentiment", f"{sent_val:.2f}" if sent_val is not None else "—")

    else:
        st.warning("No prediction block available for this asset.")

    # ---------------- Candlestick chart + markers ----------------
    if isinstance(df_asset_ind, pd.DataFrame) and not df_asset_ind.empty:
        from plotly.subplots import make_subplots

        symbol_for_asset, df_ind, sig_pts = load_price_df(asset_choice, interval_key, filter_level)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_ind.index,
            open=df_ind["Open"], high=df_ind["High"],
            low=df_ind["Low"], close=df_ind["Close"],
            name="Price",
        ))

        # EMAs
        if "ema20" in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema20"], mode="lines", name="EMA20"))
        if "ema50" in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema50"], mode="lines", name="EMA50"))

        # 🔹 Buy / Sell markers
        if sig_pts.get("buy_times"):
            fig.add_trace(go.Scatter(
                x=sig_pts["buy_times"],
                y=sig_pts["buy_prices"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=8),
                name="Buy Signal",
            ))
        if sig_pts.get("sell_times"):
            fig.add_trace(go.Scatter(
                x=sig_pts["sell_times"],
                y=sig_pts["sell_prices"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=8),
                name="Sell Signal",
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
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# 3) DEBUG / RAW DATA
# ---------------------------------------------------------------------

st.header("🛠 Debug / Raw Data")

with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, _ = load_price_df(asset_choice, interval_key, filter_level)
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(df_asset_full.tail(200), width="stretch")
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())

st.caption("Engine smart v7.8.1 • regime-adaptive • ATR TP/SL • ML blend • filter control • signal markers")