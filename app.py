# app.py v8.3 - Woody Trades Pro Dashboard
# - Strategy Mode selector (9 modes)
# - Engine Depth (Accuracy vs Performance)
# - Uses utils v8.3
# - Plots candles + EMAs + support/resistance + strategy markers
# - Conditional overlays (Bollinger, Range, Swing pivots)

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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Woody Trades Pro - Smart v8.3",
    layout="wide"
)

st.sidebar.title("Settings")

# timeframe
interval_key = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=list(INTERVALS.keys()).index("1h") if "1h" in INTERVALS else 0,
    key="sidebar_interval",
)

# risk profile
risk = st.sidebar.selectbox(
    "Risk Profile",
    options=["Low","Medium","High"],
    index=["Low","Medium","High"].index("Medium"),
    key="sidebar_risk",
    help="ATR-based TP/SL distance scaling baseline."
)

# TP/SL adaptivity
tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Adaptivity",
    options=["Normal","Aggressive","Off"],
    index=0,
    key="sidebar_tp_sl_mode",
    help="Adaptive scaling of TP/SL based on trend strength & volatility."
)

# Structure / Strategy overlay modes
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
    help="Filter trades by style. Only trades matching this confluence are allowed."
)

# Confidence filter strictness
filter_level = st.sidebar.selectbox(
    "Confidence Filter",
    options=["Loose","Balanced","Strict"],
    index=1,
    key="sidebar_filter_level",
    help="Loose = more trades, Strict = fewer/more selective."
)

# Calibration bias
calibration_enabled = st.sidebar.checkbox(
    "Calibration Bias Enabled",
    value=True,
    help="Bias thresholds based on rolling per-asset historical performance."
)

# Weekend/stale cutoff
weekend_mode = st.sidebar.checkbox(
    "Skip stale market (weekend mode)",
    value=True,
    help="If data is older than ~48h for that timeframe (except BTC), mark stale."
)

# Engine depth toggle
engine_depth = st.sidebar.selectbox(
    "Engine Depth",
    options=["Accuracy","Performance"],
    index=0,
    help="Accuracy = higher quality stats (no forced trades).\nPerformance = allow forced trades to keep sample size up."
)

forced_trades_enabled = (engine_depth == "Performance")

# focus asset for detail
asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    key="sidebar_asset",
)

st.sidebar.caption(
    "Smart v8.3 â¢ Strategy Modes â¢ 5-Fold CV ML â¢ Adaptive TP/SL â¢ PF/EV Backtest"
)

# Cache wrappers
@st.cache_data(show_spinner=False)
def load_summary_cached(interval_key, risk, tp_sl_mode,
                        structure_mode, forced_trades_enabled,
                        filter_level, calibration_enabled, weekend_mode):
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
def load_prediction_and_chart_cached(asset, interval_key, risk, tp_sl_mode,
                                     structure_mode, forced_trades_enabled,
                                     filter_level, calibration_enabled, weekend_mode):
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
def load_price_df_cached(asset, interval_key, structure_mode,
                         forced_trades_enabled, filter_level, calibration_enabled):
    return load_asset_with_indicators(
        asset=asset,
        interval_key=interval_key,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
    )

# 1) MARKET SUMMARY
st.header("ð Market Summary")
try:
    df_summary = load_summary_cached(
        interval_key, risk, tp_sl_mode,
        structure_mode, forced_trades_enabled,
        filter_level, calibration_enabled, weekend_mode
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

# 2) DETAILED VIEW
st.header("ð Detailed View")
try:
    pred_block, df_asset_ind, pts = load_prediction_and_chart_cached(
        asset_choice, interval_key, risk, tp_sl_mode,
        structure_mode, forced_trades_enabled,
        filter_level, calibration_enabled, weekend_mode
    )

    # metrics
    colA, colB, colC, colD = st.columns(4)
    if pred_block:
        # core metrics row 1
        colA.metric(
            label="Signal",
            value=pred_block.get("side","Hold"),
            help="Buy / Sell / Hold after structure_mode gating"
        )
        prob_val = pred_block.get("probability",0.0)
        colB.metric(
            label="Confidence",
            value=f"{prob_val*100:.2f}%" if prob_val<=1 else f"{prob_val:.2f}%",
            help="Blended rule+ML confidence (5-fold CV) clipped 5%-95%."
        )
        colC.metric(
            label="Win Rate (backtest)",
            value=f"{pred_block.get('win_rate',0):.2f}% Â±{pred_block.get('win_rate_std',0):.2f}",
            help="Relaxed horizon backtest hit-rate plus variability."
        )
        colD.metric(
            label="Trades (backtest)",
            value=str(pred_block.get("trades",0)),
            help="Simulated number of qualifying trades under current filters."
        )

        # row 2 metrics
        colE,colF,colG,colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("TP", f"{tp_val:.4f}" if tp_val else "â")
        colF.metric("SL", f"{sl_val:.4f}" if sl_val else "â")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "â")
        colH.metric(
            "Sentiment",
            f"{pred_block.get('sentiment','â')}",
            help="Stub sentiment for asset bias."
        )

        # row 3 metrics
        colI,colJ,colK,colL = st.columns(4)
        colI.metric(
            "Profit Factor",
            f"{pred_block.get('profit_factor','â')}",
            help="Sum wins / sum losses in the backtest."
        )
        colJ.metric(
            "EV% / Trade",
            f"{pred_block.get('ev_per_trade','â')}",
            help="Avg PnL per trade in %, incl losers."
        )
        colK.metric(
            "Total Return (bt)",
            f"{pred_block.get('backtest_return_pct','â')}%",
            help="Naive compounding of +1%/-1% exits."
        )
        colL.metric(
            "Max DD (bt)",
            f"{pred_block.get('maxdd','â')}%",
            help="Largest drawdown in backtest path."
        )

    else:
        st.warning("No prediction block available for this asset.")

    # Chart
    if isinstance(df_asset_ind,pd.DataFrame) and not df_asset_ind.empty:
        fig = go.Figure()

        # Candles
        fig.add_trace(go.Candlestick(
            x=df_asset_ind.index,
            open=df_asset_ind["Open"],
            high=df_asset_ind["High"],
            low=df_asset_ind["Low"],
            close=df_asset_ind["Close"],
            name="Price",
        ))

        # EMA overlays
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

        # Support / resistance lines
        if "sup_x" in pts and len(pts["sup_x"])>0:
            fig.add_trace(go.Scatter(
                x=pts["sup_x"],
                y=pts["sup_y"],
                mode="lines",
                name="Support",
                line=dict(width=1,dash="dot")
            ))
        if "res_x" in pts and len(pts["res_x"])>0:
            fig.add_trace(go.Scatter(
                x=pts["res_x"],
                y=pts["res_y"],
                mode="lines",
                name="Resistance",
                line=dict(width=1,dash="dot")
            ))

        # Bollinger only if relevant mode (Volatility Expansion)
        if structure_mode == "Volatility Expansion":
            fig.add_trace(go.Scatter(
                x=pts["bb_upper_x"],
                y=pts["bb_upper_y"],
                mode="lines",
                name="BB Upper",
                line=dict(width=1,dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=pts["bb_lower_x"],
                y=pts["bb_lower_y"],
                mode="lines",
                name="BB Lower",
                line=dict(width=1,dash="dash")
            ))

        # Range channel if Range Reversal
        if structure_mode == "Range Reversal":
            fig.add_trace(go.Scatter(
                x=pts["range_hi_x"],
                y=pts["range_hi_y"],
                mode="lines",
                name="Range High",
                line=dict(width=1,dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=pts["range_lo_x"],
                y=pts["range_lo_y"],
                mode="lines",
                name="Range Low",
                line=dict(width=1,dash="dot")
            ))

        # Swing pivots if Swing Structure
        if structure_mode == "Swing Structure":
            fig.add_trace(go.Scatter(
                x=pts["swing_high_x"],
                y=pts["swing_high_y"],
                mode="markers",
                name="Swing High",
                marker=dict(symbol="triangle-down",size=8)
            ))
            fig.add_trace(go.Scatter(
                x=pts["swing_low_x"],
                y=pts["swing_low_y"],
                mode="markers",
                name="Swing Low",
                marker=dict(symbol="triangle-up",size=8)
            ))

        # Core buy/sell after gating
        fig.add_trace(go.Scatter(
            x=pts["buy_x"],
            y=pts["buy_y"],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up",size=9)
        ))
        fig.add_trace(go.Scatter(
            x=pts["sell_x"],
            y=pts["sell_y"],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down",size=9)
        ))

        # Style flavour markers
        fig.add_trace(go.Scatter(
            x=pts["dip_x"], y=pts["dip_y"],
            mode="markers",
            name="Dip Buy",
            marker=dict(symbol="triangle-up",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["bo_long_x"], y=pts["bo_long_y"],
            mode="markers",
            name="Bull Breakout",
            marker=dict(symbol="diamond",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["bo_short_x"], y=pts["bo_short_y"],
            mode="markers",
            name="Bear Breakdown",
            marker=dict(symbol="x",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["mr_long_x"], y=pts["mr_long_y"],
            mode="markers",
            name="MeanRev Long",
            marker=dict(symbol="circle",size=6)
        ))
        fig.add_trace(go.Scatter(
            x=pts["mr_short_x"], y=pts["mr_short_y"],
            mode="markers",
            name="MeanRev Short",
            marker=dict(symbol="circle-open",size=6)
        ))
        fig.add_trace(go.Scatter(
            x=pts["tc_long_x"], y=pts["tc_long_y"],
            mode="markers",
            name="TrendCont Long",
            marker=dict(symbol="square",size=6)
        ))
        fig.add_trace(go.Scatter(
            x=pts["tc_short_x"], y=pts["tc_short_y"],
            mode="markers",
            name="TrendCont Short",
            marker=dict(symbol="square-open",size=6)
        ))
        fig.add_trace(go.Scatter(
            x=pts["ve_long_x"], y=pts["ve_long_y"],
            mode="markers",
            name="VolExp Long",
            marker=dict(symbol="star",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["ve_short_x"], y=pts["ve_short_y"],
            mode="markers",
            name="VolExp Short",
            marker=dict(symbol="star-open",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["rr_long_x"], y=pts["rr_long_y"],
            mode="markers",
            name="RangeRev Long",
            marker=dict(symbol="diamond-open",size=7)
        ))
        fig.add_trace(go.Scatter(
            x=pts["rr_short_x"], y=pts["rr_short_y"],
            mode="markers",
            name="RangeRev Short",
            marker=dict(symbol="x-open",size=7)
        ))

        fig.update_layout(
            margin=dict(l=10,r=10,t=30,b=10),
            height=500,
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No chart data available for this asset/timeframe.")

except Exception as e:
    st.error(f"Error loading detail view: {e}")
    st.code(traceback.format_exc())

# 3) RAW DATA / DEBUG
st.header("ð  Debug / Raw Data")
with st.expander("Show model input data / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, pts_debug = load_price_df_cached(
            asset_choice, interval_key, structure_mode,
            forced_trades_enabled, filter_level, calibration_enabled
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
    f"Engine Smart v8.3 â¢ Strategy Mode: {structure_mode} â¢ Depth: {engine_depth} â¢ "
    "5-Fold CV ML â¢ Adaptive TP/SL â¢ PF/EV backtest â¢ Weekend-aware"
)
