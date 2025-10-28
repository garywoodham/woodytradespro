"""
app_v8_3_12.py
WoodyTradesPro / Forecast Dashboard — Persistent Cache Edition

This app:
- Lets you refresh + cache all assets once using yfinance (with retry).
- Stores candles in ./data_cache next to the code (persistent).
- Then runs all analytics in offline mode (no more freeze / rate limit).
- Shows progress so you can tell it's still running, not dead.

It preserves:
- Interval choice
- Risk profile
- TP/SL scaling
- Structure mode
- Filter level
- Detailed View with TP/SL, RR, PF, WinRate, Sharpe-like, etc.
"""
import time
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils as utils  # rename to utils after you copy over

st.set_page_config(
    page_title="WoodyTradesPro — Smart Strategy Modes",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Sidebar: Cache + configuration
# -----------------------------------------------------------------------------
st.sidebar.header("📦 Data Cache Control")

today = dt.date.today()
if "last_refresh_date" not in st.session_state:
    st.session_state["last_refresh_date"] = None

needs_refresh = st.session_state["last_refresh_date"] != today

def sidebar_progress_hook(cur, total, asset_name, symbol, status_msg):
    st.sidebar.write(f"{cur}/{total} {asset_name} ({symbol}) {status_msg}")
    st.sidebar.progress(cur / total)

if st.sidebar.button("🔄 Refresh All Data (1h)"):
    st.sidebar.info("Refreshing all assets from Yahoo Finance…")
    utils.refresh_all_data(interval_key="1h", progress_hook=sidebar_progress_hook)
    st.session_state["last_refresh_date"] = today
    st.sidebar.success("✅ Cache updated and saved to ./data_cache")
    st.stop()

if needs_refresh:
    st.sidebar.warning("⚠ Data not refreshed today.\nClick 'Refresh All Data' to update.")
else:
    st.sidebar.caption(f"Cache last refreshed: {st.session_state['last_refresh_date']}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Analysis Configuration")

interval_key = st.sidebar.selectbox(
    "Interval",
    list(utils.INTERVALS.keys()),
    index=2,
    help="Backtest / signal timeframe. Cache 'Refresh All Data' currently targets 1h by default.",
)

risk = st.sidebar.selectbox(
    "Risk Profile",
    list(utils.RISK_MULT.keys()),
    index=1,
)

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

forced_trades = st.sidebar.checkbox("Force Trades (avoid 0-trade backtests)", False)
calibration_enabled = st.sidebar.checkbox("Calibration Memory Enabled", True)
weekend_mode = st.sidebar.checkbox("Allow Weekend/Closed Markets", True)

st.sidebar.caption("WoodyTradesPro v8.3.12 — Persistent Cache Edition")
st.sidebar.markdown("---")


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("📊 WoodyTradesPro — Smart Strategy Modes Edition")


# -----------------------------------------------------------------------------
# Summary Section
# -----------------------------------------------------------------------------
status_placeholder = st.empty()
progress_bar = st.progress(0.0)

def progress_callback(current, total, asset_name, symbol):
    pct = current / total
    status_placeholder.info(
        f"Analysing {asset_name} ({symbol}) from cache… {int(pct*100)}%"
    )
    progress_bar.progress(pct)
    time.sleep(0.05)

with st.spinner("Computing signals and performance metrics from cached data…"):
    try:
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
            offline=True,  # <-- IMPORTANT: stay offline for live UI
        )
    except Exception as e:
        st.error(f"Summary analysis error: {e}")
        df_summary = pd.DataFrame()

progress_bar.progress(1.0)
status_placeholder.success("✅ All assets analysed from cache")

if df_summary.empty:
    st.warning("No cached data yet. Use 'Refresh All Data' in the sidebar first.")
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
    with st.spinner(f"Running detailed analysis for {asset_choice} from cache…"):
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
            offline=True,  # offline analysis
        )

        if block is None or df_ind.empty:
            st.error("No cached data for that asset yet. Refresh first.")
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
st.caption("WoodyTradesPro © Forecast Project 2025 — Persistent Cache Edition")