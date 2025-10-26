# app.py - Woody Trades Pro dashboard
# Smart v8.0 UI (Performance Mode default)
#
# - Keeps original layout/metrics
# - Adds sidebar engine controls:
#     • Weekend Mode toggle (hard skip last 48h if stale)
#     • TP/SL Adaptivity ("Off", "Normal", "Aggressive")
#     • Confidence Filter ("Loose", "Balanced", "Strict")
#     • Calibration Bias toggle
#     • Forced Trades toggle
# - Passes those controls through to utils
# - Shows them in captions for transparency
# - Adds Buy/Sell marker overlays to candlestick chart
#
# NOTE: This expects utils.py v8.0 with matching signatures.
#
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
# Hardening: Streamlit Cloud watcher sometimes explodes with inotify.
# We reduce watcher aggressiveness in containers with low limits.
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# General Streamlit config
st.set_page_config(
    page_title="Woody Trades Pro - Smart v8.0",
    layout="wide"
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.title("Settings")

# timeframe
interval_key = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=list(INTERVALS.keys()).index("1h") if "1h" in INTERVALS else 0,
    key="sidebar_interval",
    help="Candle timeframe used for signals, TP/SL, and backtest window.",
)

# risk profile
risk = st.sidebar.selectbox(
    "Risk Profile",
    options=list({"Low","Medium","High"}),
    index=["Low","Medium","High"].index("Medium"),
    key="sidebar_risk",
    help="Controls base TP/SL ATR multiples.",
)

# main asset focus for the detail panel
asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
    help="Used in the Detailed / Scenarios sections below.",
)

st.sidebar.caption("v8.0 engine: weekend-aware, adaptive TP/SL, HTF bias, calibrated ML")

# ---- Engine Control Block ----
st.sidebar.subheader("Engine Controls")

weekend_mode = st.sidebar.checkbox(
    "Weekend Mode (skip last 48h if stale)",
    value=True,
    help="If enabled, stale last-48h data is hard skipped for closed markets to avoid fake weekend losses."
)

tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Adaptivity",
    options=["Off", "Normal", "Aggressive"],
    index=1,
    help="Controls ATR scaling range for targets/stops.\nOff = fixed; Normal = 0.8–1.4x; Aggressive = 0.5–2.0x."
)

filter_level = st.sidebar.selectbox(
    "Confidence Filter",
    options=["Loose", "Balanced", "Strict"],
    index=1,
    help="Affects ADX / volatility gating and min confidence required to take trades."
)

calibration_enabled = st.sidebar.checkbox(
    "Calibration Bias (per-asset learning)",
    value=True,
    help="If ON, the model nudges confidence using each asset's historical win rate memory."
)

forced_trades_enabled = st.sidebar.checkbox(
    "Forced Trades (test mode)",
    value=False,
    help="If ON, allow low-confidence 'forced' trades in backtest to keep stats populated.\nFor production keep OFF."
)

# ---------------------------------------------------------------------
# Cached loaders (must include the engine knobs in their signatures)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_summary(
    interval_key: str,
    risk: str,
    weekend_mode: bool,
    tp_sl_mode: str,
    filter_level: str,
    calibration_enabled: bool,
    forced_trades_enabled: bool,
) -> pd.DataFrame:
    return summarize_assets(
        interval_key,
        risk,
        use_cache=True,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        forced_trades_enabled=forced_trades_enabled,
        tp_sl_mode=tp_sl_mode,
    )


@st.cache_data(show_spinner=False)
def load_prediction_and_chart(
    asset: str,
    interval_key: str,
    risk: str,
    weekend_mode: bool,
    tp_sl_mode: str,
    filter_level: str,
    calibration_enabled: bool,
    forced_trades_enabled: bool,
):
    return asset_prediction_and_backtest(
        asset,
        interval_key,
        risk,
        use_cache=True,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        forced_trades_enabled=forced_trades_enabled,
        tp_sl_mode=tp_sl_mode,
    )


@st.cache_data(show_spinner=False)
def load_price_df(
    asset: str,
    interval_key: str,
    weekend_mode: bool,
    filter_level: str,
    calibration_enabled: bool,
):
    # We also want signal markers for buy/sell to plot on the candle chart.
    # The updated utils.load_asset_with_indicators now returns:
    #   (symbol, df_with_ind, sig_points_dict)
    symbol, df_ind, sig_points = load_asset_with_indicators(
        asset,
        interval_key,
        use_cache=True,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
    )
    return symbol, df_ind, sig_points


# ---------------------------------------------------------------------
# 1) MARKET SUMMARY SECTION
# ---------------------------------------------------------------------

st.header("📊 Market Summary")

try:
    df_summary = load_summary(
        interval_key,
        risk,
        weekend_mode,
        tp_sl_mode,
        filter_level,
        calibration_enabled,
        forced_trades_enabled,
    )

    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        # Show which engine settings are active right now (for clarity / A-B testing)
        st.caption(
            f"Engine v8.0 • Weekend={'ON' if weekend_mode else 'OFF'} • TP/SL={tp_sl_mode} • "
            f"Filter={filter_level} • Calib={'ON' if calibration_enabled else 'OFF'} • "
            f"Forced={'ON' if forced_trades_enabled else 'OFF'}"
        )

        # Show raw dashboard table
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

st.header("🔍 Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(
        asset_choice,
        interval_key,
        risk,
        weekend_mode,
        tp_sl_mode,
        filter_level,
        calibration_enabled,
        forced_trades_enabled,
    )

    # Top metrics row
    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        # Signal side
        colA.metric(
            label="Signal",
            value=pred_block.get("side") or pred_block.get("signal") or "Hold",
            help="Buy / Sell / Hold from blended rule+regime+HTF logic.",
        )

        # Confidence
        prob_val = pred_block.get("probability")
        if prob_val is None:
            prob_val = pred_block.get("probability_calibrated")
        if prob_val is None:
            prob_val = pred_block.get("probability_raw")
        if prob_val is None:
            prob_val = 0.0

        colB.metric(
            label="Confidence",
            value=f"{(prob_val*100 if prob_val<=1 else prob_val):.2f}%",
            help="Blended rule+ML conviction (calibrated if enabled).",
        )

        # Win rate (%)
        wr = pred_block.get("win_rate") or pred_block.get("winrate") or 0.0
        wr_std = pred_block.get("win_rate_std") or pred_block.get("winrate_std") or 0.0
        colC.metric(
            label="Win Rate (backtest)",
            value=f"{wr:.2f}%",
            help=f"Mean win% across ensemble seeds. ±{wr_std:.2f} stdev.",
        )

        # Trades
        tr = pred_block.get("trades") or 0
        colD.metric(
            label="Trades (backtest)",
            value=str(tr),
            help="Avg trades simulated across ensemble runs.",
        )

        # Second row with TP/SL/RR etc.
        colE, colF, colG, colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("TP", f"{tp_val:.4f}" if tp_val else "—")
        colF.metric("SL", f"{sl_val:.4f}" if sl_val else "—")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "—")

        sent_val = pred_block.get("sentiment", None)
        if sent_val is None:
            sent_val = pred_block.get("Sentiment", None)
        colH.metric(
            "Sentiment",
            f"{sent_val:.2f}" if sent_val is not None else "—",
            help="Stubbed sentiment: higher = more bullish tone.",
        )

        # Third row with EV%, PF, stale flag
        colI, colJ, colK, colL = st.columns(4)
        pf_val = pred_block.get("profit_factor", "—")
        ev_val = pred_block.get("ev_pct", "—")
        stale_flag = pred_block.get("stale", False)
        colI.metric(
            "Profit Factor",
            f"{pf_val}" if isinstance(pf_val, str) else f"{pf_val:.2f}",
            help="TP total / SL total. >1 means more reward than risk historically.",
        )
        colJ.metric(
            "EV per Trade",
            f"{ev_val:.4f}%" if not isinstance(ev_val, str) else "—",
            help="Average % gain (or loss) per trade in backtest ensemble.",
        )
        colK.metric(
            "Stale Market?",
            "Yes" if stale_flag else "No",
            help="Yes means recent candles were older than normal (weekend / closed market).",
        )

        rrisk = pred_block.get("interval", interval_key)
        colL.metric(
            "Interval",
            rrisk,
            help="Timeframe used for this prediction / backtest.",
        )

    else:
        st.warning("No prediction block available for this asset.")

    # Price chart with EMA overlays and Buy/Sell markers
    symbol_for_asset, df_asset_full, sig_points = load_price_df(
        asset_choice,
        interval_key,
        weekend_mode,
        filter_level,
        calibration_enabled,
    )

    if isinstance(df_asset_full, pd.DataFrame) and not df_asset_full.empty:
        fig = go.Figure()

        # Candles
        fig.add_trace(go.Candlestick(
            x=df_asset_full.index,
            open=df_asset_full["Open"],
            high=df_asset_full["High"],
            low=df_asset_full["Low"],
            close=df_asset_full["Close"],
            name="Price",
        ))

        # EMA20 / EMA50
        if "ema20" in df_asset_full.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_full.index,
                y=df_asset_full["ema20"],
                mode="lines",
                name="EMA20",
                line=dict(width=1.2),
            ))
        if "ema50" in df_asset_full.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_full.index,
                y=df_asset_full["ema50"],
                mode="lines",
                name="EMA50",
                line=dict(width=1.2),
            ))

        # Buy markers
        if sig_points and len(sig_points.get("buy_times", [])) > 0:
            fig.add_trace(go.Scatter(
                x=sig_points["buy_times"],
                y=sig_points["buy_prices"],
                mode="markers",
                name="Buy signal",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green",
                    line=dict(width=1, color="black"),
                ),
            ))

        # Sell markers
        if sig_points and len(sig_points.get("sell_times", [])) > 0:
            fig.add_trace(go.Scatter(
                x=sig_points["sell_times"],
                y=sig_points["sell_prices"],
                mode="markers",
                name="Sell signal",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="red",
                    line=dict(width=1, color="black"),
                ),
            ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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

st.header("🛠 Debug / Raw Data")

with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(
            df_asset_full.tail(200),
            width="stretch",
        )
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())


st.caption("Engine smart v8.0 • weekend-aware skip • adaptive TP/SL • HTF bias • calibrated ML • ensemble backtest (PF / EV%)")
