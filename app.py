# app.py v8.3.1 (lazy-load safe version for Streamlit Cloud)
# ---------------------------------------------------------------------------------
# Keeps ALL dashboard behavior:
# - Timeframe / Risk / TP-SL adaptivity
# - Strategy Mode (structure gating)
# - Confidence Filter (Loose / Balanced / Strict)
# - Calibration toggle
# - Weekend stale flagging
# - Engine Depth (Accuracy vs Performance)
# - Backtest stats: PF, EV/trade, WinRate ± std
# - Candlestick chart with overlays + markers
#
# Key change for Streamlit Cloud:
#   No heavy work at the top-level except import & layout config. The heavy stuff
#   only runs when Streamlit calls our cached functions.

import traceback
import warnings

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

warnings.filterwarnings("ignore")

# Streamlit page config
st.set_page_config(
    page_title="Woody Trades Pro - Smart v8.3.1",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
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
    index=["Low", "Medium", "High"].index("Medium"),
    key="sidebar_risk",
    help="Controls baseline TP/SL ATR multiples.",
)

tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Adaptivity",
    options=["Normal", "Aggressive", "Off"],
    index=0,
    key="sidebar_tp_sl_mode",
    help="Adaptive scaling of TP/SL distances using ADX & volatility regime.",
)

structure_mode = st.sidebar.selectbox(
    "Structure Overlay (Strategy Mode)",
    options=[
        "Off",
        "Buy Dips",
        "Breakouts",
        "Both (Dips + Breakouts)",
        "Mean Reversion",
        "Trend Continuation",
        "Volatility Expansion",
        "Range Reversal",
        "Swing Structure",
    ],
    index=0,
    key="sidebar_structure_mode",
    help="Gate trades so we ONLY allow setups that match this style.",
)

filter_level = st.sidebar.selectbox(
    "Confidence Filter",
    options=["Loose", "Balanced", "Strict"],
    index=1,
    key="sidebar_filter_level",
    help="Loose = more trades allowed. Strict = fewer, higher-confidence trades.",
)

calibration_enabled = st.sidebar.checkbox(
    "Calibration Bias Enabled",
    value=True,
    key="sidebar_calibration_enabled",
    help="Reinforces/penalizes signals based on rolling historical performance.",
)

weekend_mode = st.sidebar.checkbox(
    "Skip Stale (Weekend Mode)",
    value=True,
    key="sidebar_weekend_mode",
    help="Flags symbols with stale last candle (e.g. weekend, market closed). BTC exempt.",
)

engine_depth = st.sidebar.selectbox(
    "Engine Depth",
    options=["Accuracy", "Performance"],
    index=0,
    key="sidebar_engine_depth",
    help=(
        "Accuracy = require confidence and structure, do NOT force trades.\n"
        "Performance = allow low-conf 'forced' trades to increase backtest sample size."
    ),
)

forced_trades_enabled = (engine_depth == "Performance")

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
)

st.sidebar.caption(
    "Smart v8.3.1 • Structure Modes • 5-Fold CV ML • Adaptive TP/SL • PF/EV backtest"
)


# -----------------------------------------------------------------------------
# CACHING LAYERS
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_summary_cached(
    interval_key,
    risk,
    tp_sl_mode,
    structure_mode,
    forced_trades_enabled,
    filter_level,
    calibration_enabled,
    weekend_mode,
):
    return summarize_assets(
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )


@st.cache_data(show_spinner=False)
def load_prediction_and_chart_cached(
    asset,
    interval_key,
    risk,
    tp_sl_mode,
    structure_mode,
    forced_trades_enabled,
    filter_level,
    calibration_enabled,
    weekend_mode,
):
    return asset_prediction_and_backtest(
        asset=asset,
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )


@st.cache_data(show_spinner=False)
def load_price_df_cached(
    asset,
    interval_key,
    structure_mode,
    forced_trades_enabled,
    filter_level,
    calibration_enabled,
):
    return load_asset_with_indicators(
        asset=asset,
        interval_key=interval_key,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
    )


# -----------------------------------------------------------------------------
# 1) MARKET SUMMARY
# -----------------------------------------------------------------------------

st.header("📊 Market Summary")

try:
    df_summary = load_summary_cached(
        interval_key,
        risk,
        tp_sl_mode,
        structure_mode,
        forced_trades_enabled,
        filter_level,
        calibration_enabled,
        weekend_mode,
    )

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


# -----------------------------------------------------------------------------
# 2) DETAILED VIEW
# -----------------------------------------------------------------------------

st.header("🔍 Detailed View")

try:
    pred_block, df_asset_ind, pts = load_prediction_and_chart_cached(
        asset_choice,
        interval_key,
        risk,
        tp_sl_mode,
        structure_mode,
        forced_trades_enabled,
        filter_level,
        calibration_enabled,
        weekend_mode,
    )

    # Metric row 1
    colA, colB, colC, colD = st.columns(4)

    if pred_block:
        # current signal
        colA.metric(
            label="Signal",
            value=pred_block.get("side", "Hold"),
            help="Buy / Sell / Hold AFTER structure-mode gating and confidence filter.",
        )

        # model confidence
        prob_val = pred_block.get("probability", 0.0)
        if prob_val <= 1:
            conf_str = f"{prob_val * 100:.2f}%"
        else:
            conf_str = f"{prob_val:.2f}%"
        colB.metric(
            label="Confidence",
            value=conf_str,
            help="Blended rule+ML conviction (5-fold CV); clipped 5%-95%.",
        )

        # win rate (± std)
        wr_val = pred_block.get("win_rate", 0.0)
        wr_std_val = pred_block.get("win_rate_std", 0.0)
        colC.metric(
            label="Win Rate (backtest)",
            value=f"{wr_val:.2f}% ±{wr_std_val:.2f}",
            help="Relaxed horizon backtest hit-rate and variability.",
        )

        # trades taken
        tr_val = pred_block.get("trades", 0)
        colD.metric(
            label="Trades (backtest)",
            value=str(tr_val),
            help="Number of simulated trades under current filters.",
        )

        # Metric row 2: TP / SL / RR / Sentiment
        colE, colF, colG, colH = st.columns(4)

        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")

        colE.metric(
            "TP",
            f"{tp_val:.4f}" if tp_val else "—",
            help="Adaptive Take Profit from ATR/ADX/vol regime."
        )
        colF.metric(
            "SL",
            f"{sl_val:.4f}" if sl_val else "—",
            help="Adaptive Stop Loss from ATR/ADX/vol regime."
        )
        colG.metric(
            "R/R",
            f"{rr_val:.2f}" if rr_val else "—",
            help="Reward/Risk ratio implied by TP and SL."
        )

        sent_val = pred_block.get("sentiment", "—")
        colH.metric(
            "Sentiment",
            f"{sent_val}",
            help="Stub sentiment: higher = more bullish tone.",
        )

        # Metric row 3: PF / EV / Return / Drawdown
        colI, colJ, colK, colL = st.columns(4)

        colI.metric(
            "Profit Factor",
            f"{pred_block.get('profit_factor', '—')}",
            help="Sum wins / sum losses in backtest. >1 is typically good."
        )
        colJ.metric(
            "EV% / Trade",
            f"{pred_block.get('ev_per_trade', '—')}",
            help="Average PnL per trade in %, including losers."
        )
        colK.metric(
            "Total Return (bt)",
            f"{pred_block.get('backtest_return_pct', '—')}%",
            help="Naive compounding of +/-1% outcomes."
        )
        colL.metric(
            "Max DD (bt)",
            f"{pred_block.get('maxdd', '—')}%",
            help="Worst peak-to-trough drawdown in the simulated path.",
        )

    else:
        st.warning("No prediction block available for this asset.")

    # Price chart + overlays
    if isinstance(df_asset_ind, pd.DataFrame) and not df_asset_ind.empty:
        fig = go.Figure()

        # Candle
        fig.add_trace(go.Candlestick(
            x=df_asset_ind.index,
            open=df_asset_ind["Open"],
            high=df_asset_ind["High"],
            low=df_asset_ind["Low"],
            close=df_asset_ind["Close"],
            name="Price",
        ))

        # EMA20/EMA50
        if "ema20" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.index,
                y=df_asset_ind["ema20"],
                mode="lines",
                name="EMA20",
                line=dict(width=1),
            ))
        if "ema50" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.index,
                y=df_asset_ind["ema50"],
                mode="lines",
                name="EMA50",
                line=dict(width=1),
            ))

        # Support / Resistance bands
        if len(pts.get("sup_x", [])) > 0:
            fig.add_trace(go.Scatter(
                x=pts["sup_x"],
                y=pts["sup_y"],
                mode="lines",
                name="Support",
                line=dict(width=1, dash="dot"),
            ))
        if len(pts.get("res_x", [])) > 0:
            fig.add_trace(go.Scatter(
                x=pts["res_x"],
                y=pts["res_y"],
                mode="lines",
                name="Resistance",
                line=dict(width=1, dash="dot"),
            ))

        # If we're specifically in Volatility Expansion mode, overlay BB channels
        if structure_mode == "Volatility Expansion":
            fig.add_trace(go.Scatter(
                x=pts["bb_upper_x"],
                y=pts["bb_upper_y"],
                mode="lines",
                name="BB Upper",
                line=dict(width=1, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=pts["bb_lower_x"],
                y=pts["bb_lower_y"],
                mode="lines",
                name="BB Lower",
                line=dict(width=1, dash="dash"),
            ))

        # If we're specifically in Range Reversal mode, overlay range bounds
        if structure_mode == "Range Reversal":
            fig.add_trace(go.Scatter(
                x=pts["range_hi_x"],
                y=pts["range_hi_y"],
                mode="lines",
                name="Range High",
                line=dict(width=1, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=pts["range_lo_x"],
                y=pts["range_lo_y"],
                mode="lines",
                name="Range Low",
                line=dict(width=1, dash="dot"),
            ))

        # If we're specifically in Swing Structure mode, overlay swing pivots
        if structure_mode == "Swing Structure":
            fig.add_trace(go.Scatter(
                x=pts["swing_high_x"],
                y=pts["swing_high_y"],
                mode="markers",
                name="Swing High",
                marker=dict(symbol="triangle-down", size=8),
            ))
            fig.add_trace(go.Scatter(
                x=pts["swing_low_x"],
                y=pts["swing_low_y"],
                mode="markers",
                name="Swing Low",
                marker=dict(symbol="triangle-up", size=8),
            ))

        # Core buy/sell markers
        fig.add_trace(go.Scatter(
            x=pts["buy_x"],
            y=pts["buy_y"],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", size=9),
        ))
        fig.add_trace(go.Scatter(
            x=pts["sell_x"],
            y=pts["sell_y"],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", size=9),
        ))

        # Structure markers for context
        fig.add_trace(go.Scatter(
            x=pts["dip_x"], y=pts["dip_y"],
            mode="markers",
            name="Dip Buy",
            marker=dict(symbol="triangle-up", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["bo_long_x"], y=pts["bo_long_y"],
            mode="markers",
            name="Bull Breakout",
            marker=dict(symbol="diamond", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["bo_short_x"], y=pts["bo_short_y"],
            mode="markers",
            name="Bear Breakdown",
            marker=dict(symbol="x", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["mr_long_x"], y=pts["mr_long_y"],
            mode="markers",
            name="MeanRev Long",
            marker=dict(symbol="circle", size=6),
        ))
        fig.add_trace(go.Scatter(
            x=pts["mr_short_x"], y=pts["mr_short_y"],
            mode="markers",
            name="MeanRev Short",
            marker=dict(symbol="circle-open", size=6),
        ))
        fig.add_trace(go.Scatter(
            x=pts["tc_long_x"], y=pts["tc_long_y"],
            mode="markers",
            name="TrendCont Long",
            marker=dict(symbol="square", size=6),
        ))
        fig.add_trace(go.Scatter(
            x=pts["tc_short_x"], y=pts["tc_short_y"],
            mode="markers",
            name="TrendCont Short",
            marker=dict(symbol="square-open", size=6),
        ))
        fig.add_trace(go.Scatter(
            x=pts["ve_long_x"], y=pts["ve_long_y"],
            mode="markers",
            name="VolExp Long",
            marker=dict(symbol="star", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["ve_short_x"], y=pts["ve_short_y"],
            mode="markers",
            name="VolExp Short",
            marker=dict(symbol="star-open", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["rr_long_x"], y=pts["rr_long_y"],
            mode="markers",
            name="RangeRev Long",
            marker=dict(symbol="diamond-open", size=7),
        ))
        fig.add_trace(go.Scatter(
            x=pts["rr_short_x"], y=pts["rr_short_y"],
            mode="markers",
            name="RangeRev Short",
            marker=dict(symbol="x-open", size=7),
        ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=500,
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())


# -----------------------------------------------------------------------------
# 3) RAW DATA / DEBUG
# -----------------------------------------------------------------------------

st.header("🛠 Debug / Raw Data")

with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, pts_debug = load_price_df_cached(
            asset_choice,
            interval_key,
            structure_mode,
            forced_trades_enabled,
            filter_level,
            calibration_enabled,
        )

        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(
            df_asset_full.tail(200),
            width="stretch",
        )

    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())

st.caption(
    f"Engine Smart v8.3.1 • Strategy Mode: {structure_mode} • Depth: {engine_depth} • "
    "5-Fold CV ML • Adaptive TP/SL • PF/EV backtest • Weekend-aware"
)