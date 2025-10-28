"""
app.py — WoodyTradesPro / Forecast Dashboard
Version 8.3.10 (Progress Edition)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils
import time

st.set_page_config(page_title="WoodyTrades Pro — Smart Strategy Modes Edition", layout="wide")

# ---------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

interval_key = st.sidebar.selectbox(
    "Interval",
    list(utils.INTERVALS.keys()),
    index=2,
    help="Data timeframe for signals & backtests.",
)

risk = st.sidebar.selectbox("Risk Profile", list(utils.RISK_MULT.keys()), index=1)
tp_sl_mode = st.sidebar.selectbox("TP/SL Scaling", list(utils._TP_SL_PROFILES.keys()), index=1)
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
filter_level = st.sidebar.selectbox("Filter Level", ["Loose", "Balanced", "Strict"], index=1)
forced_trades = st.sidebar.checkbox("Force Trades (avoid empty backtests)", False)
calibration_enabled = st.sidebar.checkbox("Calibration Memory Enabled", True)
weekend_mode = st.sidebar.checkbox("Allow Weekend/Closed Markets", True)

st.sidebar.markdown("---")
st.sidebar.caption("WoodyTradesPro v8.3.10")

# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------
st.title("📊 WoodyTrades Pro — Smart Strategy Modes Edition")

# ---------------------------------------------------------------------
# Summary Progress Bar + Fetching
# ---------------------------------------------------------------------

status_placeholder = st.empty()
progress_bar = st.progress(0.0)

# progress callback function
def progress_callback(current, total, asset_name, symbol):
    progress = current / total
    status_placeholder.info(f"Fetching and analysing {asset_name} ({symbol})... {int(progress*100)}%")
    progress_bar.progress(progress)
    time.sleep(0.05)  # allow Streamlit UI refresh

with st.spinner("Fetching and analysing assets..."):
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
        )
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        df_summary = pd.DataFrame()

progress_bar.progress(1.0)
status_placeholder.success("✅ All assets analysed successfully!")

# ---------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------

if df_summary.empty:
    st.warning("No data available or all assets failed to load.")
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

# ---------------------------------------------------------------------
# Detailed View
# ---------------------------------------------------------------------

st.markdown("---")
st.header("🔍 Detailed View")

asset_choice = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)

if st.button("Run Detailed Analysis"):
    with st.spinner(f"Running detailed analysis for {asset_choice}..."):
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

        if block is None or df_ind.empty:
            st.error("Analysis failed or no data available.")
        else:
            st.subheader(f"🪙 {asset_choice} — {block['side']} ({block['probability']*100:.1f}%)")
            st.caption(f"Last price: {block['price']:.2f} | ATR: {block['atr']:.4f} | ADX: {block['adx']:.2f}")

            st.markdown(f"**TP:** {block['tp']:.2f} | **SL:** {block['sl']:.2f} | **RR:** {block['rr']:.2f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Win Rate", f"{block['win_rate']:.2f}%")
            col2.metric("Profit Factor", f"{block['profit_factor']:.2f}")
            col3.metric("Trades", f"{block['trades']}")

            # plot
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_ind.index,
                open=df_ind["Open"],
                high=df_ind["High"],
                low=df_ind["Low"],
                close=df_ind["Close"],
                name="Price",
            ))

            # overlays: EMA20 / EMA50
            if "ema20" in df_ind.columns:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema20"], name="EMA20", line=dict(width=1)))
            if "ema50" in df_ind.columns:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema50"], name="EMA50", line=dict(width=1)))

            # buy/sell markers
            if pts["buy_x"]:
                fig.add_trace(go.Scatter(x=pts["buy_x"], y=pts["buy_y"], mode="markers", name="Buy", marker=dict(color="green", size=8)))
            if pts["sell_x"]:
                fig.add_trace(go.Scatter(x=pts["sell_x"], y=pts["sell_y"], mode="markers", name="Sell", marker=dict(color="red", size=8)))

            fig.update_layout(
                height=600,
                template="plotly_white",
                title=f"{asset_choice} ({interval_key}) — {block['side']} Signal",
                xaxis_rangeslider_visible=False,
            )

            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("WoodyTradesPro © Forecast Project 2025 — All rights reserved.")