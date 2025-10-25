# app_v7_3.py - Woody Trades Pro dashboard (Smart v7.3)
# Streamlit UI layer for utils.py (Smart v7.3 weekend-safe + chart markers)

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

# -----------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------

os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

st.set_page_config(
    page_title="Woody Trades Pro - Smart v7.3",
    layout="wide"
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

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
    index=1,
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

st.sidebar.caption("v7.3 engine â¢ blended rule+ML â¢ ATR TP/SL â¢ relaxed backtest â¢ weekend-safe â¢ chart markers")

# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_summary(interval_key: str, risk: str) -> pd.DataFrame:
    return summarize_assets(interval_key, risk, use_cache=True)

@st.cache_data(show_spinner=False)
def load_prediction_and_chart(asset: str, interval_key: str, risk: str):
    return asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True)

@st.cache_data(show_spinner=False)
def load_price_df(asset: str, interval_key: str):
    return load_asset_with_indicators(asset, interval_key, use_cache=True)

# -----------------------------------------------------------------------------
# 1) MARKET SUMMARY SECTION
# -----------------------------------------------------------------------------

st.header("ð Market Summary")

try:
    df_summary = load_summary(interval_key, risk)
    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        st.dataframe(df_summary, width="stretch", hide_index=True)
except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())

# -----------------------------------------------------------------------------
# 2) DETAILED VIEW SECTION
# -----------------------------------------------------------------------------

st.header("ð Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(asset_choice, interval_key, risk)

    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        colA.metric("Signal", pred_block.get("side") or pred_block.get("signal") or "Hold",
                    help="Buy / Sell / Hold from rule engine + overrides")

        prob_val = pred_block.get("probability") or pred_block.get("prob") or 0.0
        colB.metric("Confidence", f"{round(prob_val*100,2) if prob_val<=1 else prob_val:.2f}%",
                    help="Blended rule+ML conviction, clipped 5%-95%.")

        wr = pred_block.get("win_rate") or pred_block.get("winrate") or 0.0
        colC.metric("Win Rate (backtest)", f"{wr:.2f}%",
                    help="Relaxed horizon backtest win %, weekend-safe synthetic trade if quiet.")

        tr = pred_block.get("trades") or 0
        colD.metric("Trades (backtest)", str(tr),
                    help="Number of simulated trades in the relaxed backtest.")

        colE, colF, colG, colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("TP", f"{tp_val:.4f}" if tp_val else "â")
        colF.metric("SL", f"{sl_val:.4f}" if sl_val else "â")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "â")

        sent_val = pred_block.get("sentiment") or pred_block.get("Sentiment")
        colH.metric("Sentiment", f"{sent_val:.2f}" if sent_val is not None else "â",
                    help="Stub sentiment: higher = more bullish tone.")
    else:
        st.warning("No prediction block available for this asset.")

    # -----------------------------------------------------------------
    # Candlestick chart + EMA overlays + Buy/Sell markers
    # -----------------------------------------------------------------
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

        if "ema20" in df_asset_full.columns:
            fig.add_trace(go.Scatter(x=df_asset_full.index, y=df_asset_full["ema20"],
                                     mode="lines", name="EMA20"))
        if "ema50" in df_asset_full.columns:
            fig.add_trace(go.Scatter(x=df_asset_full.index, y=df_asset_full["ema50"],
                                     mode="lines", name="EMA50"))

        # Add Buy markers
        if sig_pts.get("buy_times"):
            fig.add_trace(go.Scatter(
                x=sig_pts["buy_times"], y=sig_pts["buy_prices"],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=10, color="green")
            ))
        # Add Sell markers
        if sig_pts.get("sell_times"):
            fig.add_trace(go.Scatter(
                x=sig_pts["sell_times"], y=sig_pts["sell_prices"],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=10, color="red")
            ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            xaxis_title=f"{asset_choice} ({symbol_for_asset})",
            yaxis_title="Price",
        )
        st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())

# -----------------------------------------------------------------------------
# 3) DEBUG / RAW DATA SECTION
# -----------------------------------------------------------------------------

st.header("ð  Debug / Raw Data")

with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, sig_pts = load_price_df(asset_choice, interval_key)
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(df_asset_full.tail(200), width="stretch")
        if sig_pts:
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

st.caption("Engine Smart v7.3 â¢ blended rule+ML â¢ relaxed backtest â¢ ATR TP/SL â¢ weekend-safe â¢ sentiment stub â¢ chart markers")