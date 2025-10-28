"""
app.py — WoodyTradesPro / Forecast Dashboard
Version 8.3.11 (Offline Cache + Progress Edition)

Key changes:
- "Refresh All Data" button to bulk download & cache data (yfinance)
- Daily auto-refresh safety hook
- Analysis runs OFFLINE from cache for speed + rate-limit immunity
- Live progress bar during analysis using summarize_assets(progress_callback=...)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import time
import utils

st.set_page_config(
    page_title="WoodyTrades Pro — Smart Strategy Modes Edition",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Sidebar: Data cache management
# -----------------------------------------------------------------------------
st.sidebar.header("📦 Data Cache Control")

# We'll maintain a once-per-day auto-refresh to avoid stale data.
today = dt.date.today()
if "last_refresh_date" not in st.session_state:
    st.session_state["last_refresh_date"] = today

auto_refresh_needed = (st.session_state["last_refresh_date"] != today)

# Manual refresh button
if st.sidebar.button("🔄 Refresh All Data (1h cache)"):
    st.sidebar.info("Refreshing data for all assets — please wait...")

    prog_txt = st.sidebar.empty()
    prog_bar = st.sidebar.progress(0.0)

    def sidebar_progress(cur, total, asset_name, symbol):
        pct = cur / total
        prog_txt.info(f"Downloading {asset_name} ({symbol})... {int(pct*100)}%")
        prog_bar.progress(pct)
        time.sleep(0.05)

    utils.refresh_all_data(interval_key="1h", progress_hook=sidebar_progress)
    st.session_state["last_refresh_date"] = today
    st.sidebar.success("✅ Cache updated")
    st.sidebar.write("You can now run analysis offline (fast).")
    st.stop()

# Auto-refresh once per day if wanted
if auto_refresh_needed:
    st.sidebar.warning("⚠ Data not refreshed today.")
    st.sidebar.caption("Tip: tap 'Refresh All Data' for latest candles.")
else:
    st.sidebar.caption(f"Cache last refreshed: {st.session_state['last_refresh_date']}")

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar: Analysis configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Analysis Configuration")

interval_key = st.sidebar.selectbox(
    "Interval",
    list(utils.INTERVALS.keys()),
    index=2,
    help="Data timeframe for signals & backtests. Cache currently refreshes 1h.",
)

risk = st.sidebar.selectbox("Risk Profile", list(utils.RISK_MULT.keys()), index=1)

tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Scaling",
    list(utils._TP_SL_PROFILES.keys()),
    index=1,
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
)

filter_level = st.sidebar.selectbox(
    "Filter Level",
    ["Loose", "Balanced", "Strict"],
    index=1,
)

forced_trades = st.sidebar.checkbox("Force Trades (avoid empty backtests)", False)
calibration_enabled = st.sidebar.checkbox("Calibration Memory Enabled", True)
weekend_mode = st.sidebar.checkbox("Allow Weekend/Closed Markets", True)

st.sidebar.markdown("---")
st.sidebar.caption("WoodyTradesPro v8.3.11 — Offline Optimized")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("📊 WoodyTrades Pro — Smart Strategy Modes Edition")


# -----------------------------------------------------------------------------
# Summary Section (offline analysis from cached data)
# -----------------------------------------------------------------------------

status_placeholder = st.empty()
progress_bar = st.progress(0.0)

def progress_callback(current, total, asset_name, symbol):
    progress = current / total
    status_placeholder.info(
        f"Analysing {asset_name} ({symbol}) from cache... {int(progress*100)}%"
    )
    progress_bar.progress(progress)
    time.sleep(0.05)

with st.spinner("Computing signals and performance metrics from cached data..."):
    try:
        # offline=True means: do NOT hit Yahoo. Use cached OHLC only.
        df_summary = utils.summarize_assets(
            interval_key=interval_key,
            risk=risk,
            tp_sl_mode=tp_sl_mode,
            structure_mode=structure_mode,
            forced_trades_enabled=forced_trades,
            filter_level=filter_level,
            calibration_enabled=calibration_enabled,
            weekend_mode=weekend_mode,
            progress_callback=progress_callback,
            offline=True,  # <-- CRITICAL: analyse purely from cache
        )
    except Exception as e:
        st.error(f"Error during summary analysis: {e}")
        df_summary = pd.DataFrame()

progress_bar.progress(1.0)
status_placeholder.success("✅ All assets analysed (offline cache)")

# -----------------------------------------------------------------------------
# Summary Table
# -----------------------------------------------------------------------------
if df_summary.empty:
    st.warning("No cached data available for any asset yet. Hit 'Refresh All Data' in the sidebar.")
else:
    st.subheader("📈 Summary Overview")
    st.dataframe(
        df_summary[
            [
                "Asset",
                "Signal",
                "Probability",
                "WinRate",
                "Return%",
                "PF",
                "EV%/Trade",
                "SharpeLike",
                "Stale",
            ]
        ].style.format(
            {
                "Probability": "{:.2f}",
                "WinRate": "{:.2f}",
                "Return%": "{:.2f}",
                "PF": "{:.2f}",
                "EV%/Trade": "{:.2f}",
                "SharpeLike": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=480,
    )

# -----------------------------------------------------------------------------
# Detailed View
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🔍 Detailed View")

asset_choice = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)

if st.button("Run Detailed Analysis (offline)"):
    with st.spinner(f"Running detailed analysis for {asset_choice} from cached data..."):
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
            offline=True,  # <-- we stay offline here as well
        )

        if block is None or df_ind.empty:
            st.error("Analysis failed or no cached data available for that asset.")
        else:
            st.subheader(
                f"🪙 {asset_choice} — {block['side']} "
                f"({block['probability']*100:.1f}%)"
            )
            st.caption(
                f"Last price: {block['price']:.2f} | "
                f"ATR: {block['atr']:.4f if block['atr'] is not None else 'n/a'} | "
                f"ADX: {block['adx']:.2f if block['adx'] is not None else 'n/a'}"
            )

            if block["tp"] is not None and block["sl"] is not None and block["rr"] is not None:
                st.markdown(
                    f"**TP:** {block['tp']:.2f} | "
                    f"**SL:** {block['sl']:.2f} | "
                    f"**RR:** {block['rr']:.2f}"
                )
            else:
                st.markdown("No active TP/SL because signal is Hold.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Win Rate", f"{block['win_rate']:.2f}%")
            col2.metric("Profit Factor", f"{block['profit_factor']:.2f}")
            col3.metric("Trades", f"{block['trades']}")

            # Plot price + overlays
            fig = go.Figure()

            # Candles
            fig.add_trace(
                go.Candlestick(
                    x=df_ind.index,
                    open=df_ind["Open"],
                    high=df_ind["High"],
                    low=df_ind["Low"],
                    close=df_ind["Close"],
                    name="Price",
                )
            )

            # EMA20 / EMA50 overlays (if present)
            if "ema20" in df_ind.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_ind.index,
                        y=df_ind["ema20"],
                        name="EMA20",
                        line=dict(width=1),
                    )
                )
            if "ema50" in df_ind.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_ind.index,
                        y=df_ind["ema50"],
                        name="EMA50",
                        line=dict(width=1),
                    )
                )

            # Buy markers
            if pts["buy_x"]:
                fig.add_trace(
                    go.Scatter(
                        x=pts["buy_x"],
                        y=pts["buy_y"],
                        mode="markers",
                        name="Buy",
                        marker=dict(color="green", size=8),
                    )
                )

            # Sell markers
            if pts["sell_x"]:
                fig.add_trace(
                    go.Scatter(
                        x=pts["sell_x"],
                        y=pts["sell_y"],
                        mode="markers",
                        name="Sell",
                        marker=dict(color="red", size=8),
                    )
                )

            fig.update_layout(
                height=600,
                template="plotly_white",
                title=f"{asset_choice} ({interval_key}) — {block['side']} Signal",
                xaxis_rangeslider_visible=False,
            )

            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("WoodyTradesPro © Forecast Project 2025 — Offline-Optimized Edition")