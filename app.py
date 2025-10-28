# app.py
# ---------------------------------------------------------------------------------
# Streamlit frontend for WoodyTradesPro / Forecast
#
# - Sidebar controls feed into utils.summarize_assets() etc.
# - No blocking cache on get_summary() to avoid spinner freeze on cold start.
# - Uses utils v8.3.8 fast-fail fetch logic.
#
# ---------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import utils  # must be the full utils.py we just built


st.set_page_config(
    page_title="WoodyTrades Pro",
    page_icon="📊",
    layout="wide",
)

# =========================
# Sidebar Controls
# =========================

st.sidebar.header("Controls")

interval_key = st.sidebar.selectbox(
    "Interval",
    list(utils.INTERVALS.keys()),
    index=list(utils.INTERVALS.keys()).index("1h") if "1h" in utils.INTERVALS else 0,
    help="Data timeframe for signals & backtests."
)

risk = st.sidebar.selectbox(
    "Risk Profile",
    list(utils.RISK_MULT.keys()),
    index=list(utils.RISK_MULT.keys()).index("Medium") if "Medium" in utils.RISK_MULT else 0,
    help="Affects TP/SL distance scaling."
)

tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Scaling",
    list(utils._TP_SL_PROFILES.keys()),
    index=list(utils._TP_SL_PROFILES.keys()).index("Normal") if "Normal" in utils._TP_SL_PROFILES else 0,
    help="Adaptive take-profit / stop-loss expansion logic."
)

structure_mode = st.sidebar.selectbox(
    "Strategy Mode",
    [
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
    help="Filters trades by price action structure. 'Off' = no structure filter."
)

filter_level = st.sidebar.selectbox(
    "Filter Strictness",
    ["Loose", "Balanced", "Strict"],
    index=1,
    help="Controls confidence threshold, ADX min, etc."
)

forced_trades = st.sidebar.checkbox(
    "Performance Mode (force marginal trades)",
    value=False,
    help="Adds very low-confidence trades to keep stats alive in backtest."
)

calibration_enabled = st.sidebar.checkbox(
    "Calibration Memory",
    value=True,
    help="Learns recent per-symbol winrates and adapts thresholds."
)

weekend_mode = st.sidebar.checkbox(
    "Weekend/Stale Aware",
    value=True,
    help="Don't panic if markets are closed and candles are stale."
)

asset_choice = st.sidebar.selectbox(
    "Asset (Detailed / Raw tabs)",
    list(utils.ASSET_SYMBOLS.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("WoodyTrades Pro · Forecast Engine")


# =========================
# Helper functions for plotting
# =========================

def plot_candles_with_overlays(df_ind: pd.DataFrame, pts: dict, title: str):
    """
    Candlestick chart + overlays from utils.generate_signal_points().
    """
    if df_ind is None or df_ind.empty:
        st.warning("No data to plot.")
        return

    # We'll assume df_ind index is datetime-like (from fetch_data -> DataFrame.index)
    ts = df_ind.index

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=ts,
        open=df_ind["Open"],
        high=df_ind["High"],
        low=df_ind["Low"],
        close=df_ind["Close"],
        name="Price",
        showlegend=True,
    ))

    # Bollinger bands
    if "bb_upper" in df_ind.columns and "bb_lower" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["bb_upper"],
            mode="lines",
            name="BB Upper",
            line=dict(width=1),
        ))
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["bb_lower"],
            mode="lines",
            name="BB Lower",
            line=dict(width=1),
        ))

    # Range high/low
    if "range_high" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["range_high"],
            mode="lines",
            name="Range High",
            line=dict(width=1, dash="dot"),
        ))
    if "range_low" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["range_low"],
            mode="lines",
            name="Range Low",
            line=dict(width=1, dash="dot"),
        ))

    # Support / Resistance (last known)
    if "support_level" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["support_level"],
            mode="lines",
            name="Support",
            line=dict(width=1, dash="dash"),
        ))
    if "resistance_level" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=ts,
            y=df_ind["resistance_level"],
            mode="lines",
            name="Resistance",
            line=dict(width=1, dash="dash"),
        ))

    # Buy / Sell markers
    fig.add_trace(go.Scatter(
        x=pts["buy_x"],
        y=pts["buy_y"],
        mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", size=10, color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=pts["sell_x"],
        y=pts["sell_y"],
        mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", size=10, color="red"),
    ))

    # Dip buy markers
    fig.add_trace(go.Scatter(
        x=pts["dip_x"],
        y=pts["dip_y"],
        mode="markers",
        name="Dip Buy",
        marker=dict(symbol="circle", size=6, color="blue"),
    ))

    # Breakout / breakdown
    fig.add_trace(go.Scatter(
        x=pts["bo_long_x"],
        y=pts["bo_long_y"],
        mode="markers",
        name="Bull Breakout",
        marker=dict(symbol="star", size=8, color="gold"),
    ))
    fig.add_trace(go.Scatter(
        x=pts["bo_short_x"],
        y=pts["bo_short_y"],
        mode="markers",
        name="Bear Breakdown",
        marker=dict(symbol="star-triangle-down", size=8, color="purple"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# Page Layout
# =========================

st.title("📊 WoodyTrades Pro — Smart Strategy Modes Edition")

tabs = st.tabs(["Dashboard", "Detailed View", "Raw Data / Indicators"])


# =========================
# TAB 1: DASHBOARD
# =========================
with tabs[0]:
    st.subheader("Summary Dashboard")

    st.caption(
        "Confidence, RR, PF, WinRate, and PnL stats are calculated using your "
        "chosen risk mode, TP/SL scaling, structure filter, and strictness."
    )

    with st.spinner("Fetching and analyzing assets..."):
        # IMPORTANT: We are NOT using st.cache_data here anymore.
        df_summary = utils.summarize_assets(
            interval_key=interval_key,
            risk=risk,
            tp_sl_mode=tp_sl_mode,
            structure_mode=structure_mode,
            forced_trades_enabled=forced_trades,
            filter_level=filter_level,
            calibration_enabled=calibration_enabled,
            weekend_mode=weekend_mode,
        )

    if df_summary is None or df_summary.empty:
        st.error("No data available (rate-limited across all symbols?).")
    else:
        # Show table with nicer column order / formatting
        col_order = [
            "Asset", "Symbol", "Interval", "Price", "Signal", "Probability",
            "TP", "SL", "RR",
            "Trades", "WinRate", "WinRateStd",
            "Return%", "PF", "EV%/Trade",
            "MaxDD%", "SharpeLike",
            "Sentiment", "Stale",
        ]
        have_cols = [c for c in col_order if c in df_summary.columns]
        pretty_df = df_summary[have_cols].copy()

        # rounding for UI
        for c in ["Price", "TP", "SL"]:
            if c in pretty_df.columns:
                pretty_df[c] = pretty_df[c].astype(float).round(3)
        for c in ["RR", "Probability", "WinRate", "Return%", "PF", "EV%/Trade",
                  "MaxDD%", "SharpeLike", "Sentiment", "WinRateStd"]:
            if c in pretty_df.columns:
                pretty_df[c] = pretty_df[c].astype(float).round(2)

        st.dataframe(pretty_df, use_container_width=True)


# =========================
# TAB 2: DETAILED VIEW
# =========================
with tabs[1]:
    st.subheader(f"Detailed View: {asset_choice}")

    block, df_ind, pts = utils.asset_prediction_and_backtest(
        asset=asset_choice,
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )

    if block is None or df_ind is None or df_ind.empty:
        st.warning("No data for this asset (rate-limited or empty).")
    else:
        # Metrics panel
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Signal", f"{block['side']}", f"{block['probability']*100:.1f}%")
        colB.metric("Price", f"{block['price']:.4f}")
        if block["tp"] is not None and block["sl"] is not None:
            colC.metric("TP / SL", f"{block['tp']:.4f} / {block['sl']:.4f}")
        else:
            colC.metric("TP / SL", "—")
        if block["rr"] is not None:
            colD.metric("R/R", f"{block['rr']:.2f}")
        else:
            colD.metric("R/R", "—")

        st.markdown(
            f"""
            **Sentiment:** {block['sentiment']:.2f}  
            **ADX:** {block.get('adx', 0):.2f}  
            **ATR:** {block.get('atr', 0):.4f}  
            **Stale:** {block['stale']}  
            """
        )

        st.markdown("**Backtest / Quality Stats (10-bar horizon):**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Win Rate", f"{block['win_rate']:.2f}%")
        col2.metric("PF", f"{block['profit_factor']}")
        col3.metric("EV%/Trade", f"{block['ev_per_trade']:.2f}%")
        col4.metric("Return%", f"{block['backtest_return_pct']:.2f}%")
        col5.metric("MaxDD%", f"{block['maxdd']:.2f}%")
        col6.metric("SharpeLike", f"{block['sharpe']:.2f}")

        st.caption(
            f"Trades={block['trades']}, WinRateStd={block['win_rate_std']:.2f}%, "
            "forced trades may inflate this if enabled."
        )

        st.markdown("---")

        # Candles + overlays plot
        plot_candles_with_overlays(
            df_ind,
            pts,
            title=f"{asset_choice} ({block['interval']}) — {block['side']} {block['probability']*100:.1f}%"
        )


# =========================
# TAB 3: RAW DATA / INDICATORS
# =========================
with tabs[2]:
    st.subheader(f"Raw Data / Indicators: {asset_choice}")

    symbol, df_ind2, pts2 = utils.load_asset_with_indicators(
        asset=asset_choice,
        interval_key=interval_key,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
    )

    if df_ind2 is None or df_ind2.empty:
        st.warning("No indicator data (rate-limited or empty).")
    else:
        # show tail of indicators for debugging
        tail_cols = [
            "Close", "ema20", "ema50", "RSI", "rsi_low_band", "rsi_high_band",
            "macd", "macd_signal", "macd_hist", "macd_slope",
            "adx", "atr", "atr_pct",
            "bb_upper", "bb_lower", "bb_width",
            "support_level", "resistance_level",
            "range_low", "range_high",
            "swing_low_series", "swing_high_series",
            "dip_buy_flag", "bull_breakout_flag", "bear_breakdown_flag",
            "mr_long_flag", "mr_short_flag",
            "trend_cont_long_flag", "trend_cont_short_flag",
            "volexp_long_flag", "volexp_short_flag",
            "range_rev_long_flag", "range_rev_short_flag",
            "swing_long_flag", "swing_short_flag",
        ]

        have_cols = [c for c in tail_cols if c in df_ind2.columns]
        debug_tail = df_ind2[have_cols].tail(50).copy()

        st.caption(f"Symbol: {symbol}")
        st.dataframe(debug_tail, use_container_width=True)

        st.markdown("---")
        st.caption("Chart Preview (raw tab)")
        plot_candles_with_overlays(
            df_ind2,
            pts2,
            title=f"{asset_choice} raw view ({interval_key})"
        )