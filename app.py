# app_v7_2.py - Woody Trades Pro dashboard (smart v7.2 fixed)
# Streamlit UI layer for utils_v7_2.py
# Adds candlestick Buy/Sell markers using rule-engine signals

import os
import traceback
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils_v7_2 import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
    ASSET_SYMBOLS,
    INTERVALS,
)

# ---------------------------------------------------------------------
# Hardening: Streamlit Cloud watcher sometimes explodes with inotify.
# We reduce watcher aggressiveness in containers with low limits.
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# General Streamlit config
st.set_page_config(
    page_title="Woody Trades Pro - Smart v7.2",
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
    options=list({"Low","Medium","High"}),
    index=["Low","Medium","High"].index("Medium"),
    key="sidebar_risk",
    help="Controls TP/SL distance (ATR multiples).",
)

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
    help="Used in the Detailed / Scenarios sections below.",
)

st.sidebar.caption("v7.2 engine: blended signal, ATR TP/SL, relaxed backtest, weekend-safe, marker overlay")


# ---------------------------------------------------------------------
# Cached loaders inside app layer
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_summary(interval_key: str, risk: str) -> pd.DataFrame:
    return summarize_assets(interval_key, risk, use_cache=True)

@st.cache_data(show_spinner=False)
def load_prediction_and_chart(asset: str, interval_key: str, risk: str):
    return asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True)

@st.cache_data(show_spinner=False)
def load_price_df(asset: str, interval_key: str):
    # NEW: returns symbol, df_with_indicators, signal_points
    return load_asset_with_indicators(asset, interval_key, use_cache=True)


# ---------------------------------------------------------------------
# 1) MARKET SUMMARY TAB / SECTION
# ---------------------------------------------------------------------

st.header("ð Market Summary")

try:
    df_summary = load_summary(interval_key, risk)
    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        st.dataframe(
            df_summary,
            width="stretch",
            hide_index=True,
        )

except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())


# ---------------------------------------------------------------------
# 2) DETAILED VIEW FOR ONE ASSET
# ---------------------------------------------------------------------

st.header("ð Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(asset_choice, interval_key, risk)

    # Left metrics panel
    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        # Signal side
        colA.metric(
            label="Signal",
            value=pred_block.get("side") or pred_block.get("signal") or "Hold",
            help="Buy / Sell / Hold from rule engine + overrides",
        )

        # Probability / Confidence
        prob_val = pred_block.get("probability") or pred_block.get("prob") or 0.0
        colB.metric(
            label="Confidence",
            value=f"{round(prob_val*100,2) if prob_val<=1 else prob_val:.2f}%",
            help="Blended rule+ML conviction, clipped 5%-95%.",
        )

        # Win rate
        wr = pred_block.get("win_rate") or pred_block.get("winrate") or 0.0
        colC.metric(
            label="Win Rate (backtest)",
            value=f"{wr:.2f}%",
            help="Relaxed horizon backtest win %, weekend-safe synthetic trade if quiet.",
        )

        # Trades
        tr = pred_block.get("trades") or 0
        colD.metric(
            label="Trades (backtest)",
            value=str(tr),
            help="Number of simulated trades in the relaxed backtest.",
        )

        # Second row with TP/SL/RR etc.
        colE, colF, colG, colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("TP", f"{tp_val:.4f}" if tp_val else "â")
        colF.metric("SL", f"{sl_val:.4f}" if sl_val else "â")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "â")

        sent_val = pred_block.get("sentiment", None)
        if sent_val is None:
            sent_val = pred_block.get("Sentiment", None)
        colH.metric(
            "Sentiment",
            f"{sent_val:.2f}" if sent_val is not None else "â",
            help="Stub sentiment: higher = more bullish tone.",
        )

    else:
        st.warning("No prediction block available for this asset.")

    # Price chart w/ Buy/Sell markers
    symbol_for_asset, df_asset_full, sig_pts = load_price_df(asset_choice, interval_key)

    if isinstance(df_asset_full, pd.DataFrame) and not df_asset_full.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_asset_full.index,
            open=df_asset_full["Open"],
            high=df_asset_full["High"],
            low=df_asset_full["Low"],
            close=df_asset_full["Close"],
            name="Price",
        ))
        # overlay EMAs if available
        if "ema20" in df_asset_full.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_full.index,
                y=df_asset_full["ema20"],
                mode="lines",
                name="EMA20",
            ))
        if "ema50" in df_asset_full.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_full.index,
                y=df_asset_full["ema50"],
                mode="lines",
                name="EMA50",
            ))

        # overlay Buy markers
        if sig_pts["buy_times"]:
            fig.add_trace(go.Scatter(
                x=sig_pts["buy_times"],
                y=sig_pts["buy_prices"],
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green",
                ),
            ))

        # overlay Sell markers
        if sig_pts["sell_times"]:
            fig.add_trace(go.Scatter(
                x=sig_pts["sell_times"],
                y=sig_pts["sell_prices"],
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="red",
                ),
            ))

        fig.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            height=400,
            xaxis_title=f"{asset_choice} ({symbol_for_asset})",
            yaxis_title="Price",
        )

        st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())


# ---------------------------------------------------------------------
# 3) RAW DATA / DEBUG
# ---------------------------------------------------------------------

st.header("ð  Debug / Raw Data")

with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, sig_pts = load_price_df(asset_choice, interval_key)
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(
            df_asset_full.tail(200),
            width="stretch",
        )
        st.write("Signal markers (most recent 10):")
        marker_preview = pd.DataFrame({
            "type": (["BUY"] * len(sig_pts["buy_times"])) + (["SELL"] * len(sig_pts["sell_times"])),
            "time": sig_pts["buy_times"] + sig_pts["sell_times"],
            "price": sig_pts["buy_prices"] + sig_pts["sell_prices"],
        }).sort_values("time").tail(10)
        st.dataframe(marker_preview, hide_index=True)
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())


st.caption("Engine smart v7.2 â¢ blended rule+ML â¢ relaxed backtest â¢ ATR TP/SL â¢ weekend-safe â¢ sentiment stub â¢ chart markers")