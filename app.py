# app.py - Woody Trades Pro dashboard
# Smart v8.2 UI
#
# v8.2 highlights:
# - Engine now uses 5-fold cross-validated ML confidence (more stable)
# - Structure Overlay (Buy Dips / Breakouts / Both) still included
# - Weekend Mode, Adaptive TP/SL, Calibration Bias, Confidence Filter, etc.
# - Candlestick chart overlays signals, dip buys, breakouts, support/resistance
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

os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

st.set_page_config(
    page_title="Woody Trades Pro - Smart v8.2",
    layout="wide"
)

warnings.filterwarnings("ignore")

# Sidebar controls
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
    help="Controls base TP/SL ATR multiples.",
)

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
    help="Used in the Detailed / Scenarios sections below.",
)

st.sidebar.caption("v8.2: 5-fold CV ML, structure confluence, adaptive TP/SL, calibration, PF/EV backtest")

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
    help="Controls ATR scaling range for targets/stops.\nOff = fixed; Normal = 0.8â1.4x; Aggressive = 0.5â2.0x."
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

structure_mode = st.sidebar.selectbox(
    "Structure Overlay",
    options=["Off", "Buy Dips", "Breakouts", "Both"],
    index=0,
    help="Require structure confluence (support dips / breakouts). Fewer trades but higher quality."
)

# Cached loaders
@st.cache_data(show_spinner=False)
def load_summary(
    interval_key: str,
    risk: str,
    weekend_mode: bool,
    tp_sl_mode: str,
    filter_level: str,
    calibration_enabled: bool,
    forced_trades_enabled: bool,
    structure_mode: str,
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
        structure_mode=structure_mode,
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
    structure_mode: str,
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
        structure_mode=structure_mode,
    )

@st.cache_data(show_spinner=False)
def load_price_df(
    asset: str,
    interval_key: str,
    weekend_mode: bool,
    filter_level: str,
    calibration_enabled: bool,
    structure_mode: str,
):
    symbol, df_ind, sig_points = load_asset_with_indicators(
        asset,
        interval_key,
        use_cache=True,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        structure_mode=structure_mode,
    )
    return symbol, df_ind, sig_points

# 1) MARKET SUMMARY
st.header("ð Market Summary")

try:
    df_summary = load_summary(
        interval_key,
        risk,
        weekend_mode,
        tp_sl_mode,
        filter_level,
        calibration_enabled,
        forced_trades_enabled,
        structure_mode,
    )

    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        st.caption(
            f"Engine v8.2 â¢ Weekend={'ON' if weekend_mode else 'OFF'} â¢ TP/SL={tp_sl_mode} â¢ "
            f"Filter={filter_level} â¢ Calib={'ON' if calibration_enabled else 'OFF'} â¢ "
            f"Forced={'ON' if forced_trades_enabled else 'OFF'} â¢ Structure={structure_mode} â¢ "
            f"ML=5-fold CV"
        )

        st.dataframe(
            df_summary,
            width="stretch",
            hide_index=True,
        )

except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())

# 2) DETAILED VIEW
st.header("ð Detailed View")

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
        structure_mode,
    )

    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        colA.metric(
            label="Signal",
            value=pred_block.get("side") or pred_block.get("signal") or "Hold",
            help="Buy / Sell / Hold after HTF bias + structure confluence.",
        )

        prob_val = (
            pred_block.get("probability")
            or pred_block.get("probability_calibrated")
            or pred_block.get("probability_raw")
            or 0.0
        )
        colB.metric(
            label="Confidence",
            value=f"{(prob_val*100 if prob_val<=1 else prob_val):.2f}%",
            help="Blended rule+ML conviction (calibrated if enabled). Uses 5-fold CV ML.",
        )

        wr = pred_block.get("win_rate") or pred_block.get("winrate") or 0.0
        wr_std = pred_block.get("win_rate_std") or pred_block.get("winrate_std") or 0.0
        colC.metric(
            label="Win Rate (backtest)",
            value=f"{wr:.2f}%",
            help=f"Mean win% across ensemble seeds. Â±{wr_std:.2f} stdev.",
        )

        tr = pred_block.get("trades") or 0
        colD.metric(
            label="Trades (backtest)",
            value=str(tr),
            help="Avg trades simulated across ensemble runs.",
        )

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
            help="Stubbed sentiment: higher = more bullish tone.",
        )

        colI, colJ, colK, colL = st.columns(4)
        pf_val = pred_block.get("profit_factor", "â")
        ev_val = pred_block.get("ev_pct", "â")
        stale_flag = pred_block.get("stale", False)
        colI.metric(
            "Profit Factor",
            f"{pf_val}" if isinstance(pf_val, str) else f"{pf_val:.2f}",
            help="TP total / SL total. >1 means more reward than risk historically.",
        )
        colJ.metric(
            "EV per Trade",
            f"{ev_val:.4f}%" if not isinstance(ev_val, str) else "â",
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

    # Price chart
    symbol_for_asset, df_asset_full, sig_points = load_price_df(
        asset_choice,
        interval_key,
        weekend_mode,
        filter_level,
        calibration_enabled,
        structure_mode,
    )

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

        if sig_points and len(sig_points.get("time_index", [])) > 0:
            ti = sig_points["time_index"]
            sup = sig_points["support_series"]
            res = sig_points["resistance_series"]

            fig.add_trace(go.Scatter(
                x=ti,
                y=sup,
                mode="lines",
                name="Support",
                line=dict(width=1, dash="dot", color="lightgray"),
            ))
            fig.add_trace(go.Scatter(
                x=ti,
                y=res,
                mode="lines",
                name="Resistance",
                line=dict(width=1, dash="dot", color="darkgray"),
            ))

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

        if sig_points and len(sig_points.get("dip_buy_times", [])) > 0:
            fig.add_trace(go.Scatter(
                x=sig_points["dip_buy_times"],
                y=sig_points["dip_buy_prices"],
                mode="markers",
                name="Dip Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color="blue",
                    line=dict(width=1, color="black"),
                ),
            ))

        if sig_points and len(sig_points.get("bull_breakout_times", [])) > 0:
            fig.add_trace(go.Scatter(
                x=sig_points["bull_breakout_times"],
                y=sig_points["bull_breakout_prices"],
                mode="markers",
                name="Bull Breakout",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="orange",
                    line=dict(width=1, color="black"),
                ),
            ))

        if sig_points and len(sig_points.get("bear_breakdown_times", [])) > 0:
            fig.add_trace(go.Scatter(
                x=sig_points["bear_breakdown_times"],
                y=sig_points["bear_breakdown_prices"],
                mode="markers",
                name="Bear Breakdown",
                marker=dict(
                    symbol="x",
                    size=9,
                    color="purple",
                    line=dict(width=1, color="black"),
                ),
            ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10),
            ),
        )

        st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())

# 3) RAW DATA / DEBUG
st.header("ð  Debug / Raw Data")

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

st.caption(
    "Engine Smart v8.2 • 5-Fold ML Cross-Validation • "
    "Structure Overlay • Adaptive TP/SL • Weekend-Aware • "
    "Calibrated ML • Profit Factor / EV% Backtest"
)