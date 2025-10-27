"""
app.py v8.3.2 (Local Import Safe)
Streamlit front-end for WoodyTradesPro Smart Strategy Modes engine.

Ensures we always load the local utils.py v8.3.2 (not a pip package called utils).
Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import importlib.util, sys, pathlib

# ---------------------------------------------------------------------
# Force import local utils.py regardless of environment conflicts
# ---------------------------------------------------------------------
UTILS_PATH = pathlib.Path(__file__).parent / "utils.py"
spec = importlib.util.spec_from_file_location("utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)

st.write("✅ Loaded utils from:", utils.__file__)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="WoodyTradesPro", page_icon="📈", layout="wide")
st.title("📈 WoodyTradesPro Smart Strategy Dashboard")

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Controls")

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
filter_level = st.sidebar.selectbox("Confidence Filter", ["Loose", "Balanced", "Strict"], index=1)
forced_trades_enabled = st.sidebar.checkbox("Force trades (for stats)", value=False)
calibration_enabled = st.sidebar.checkbox("Calibration Memory Enabled", value=True)
weekend_mode = st.sidebar.checkbox("Weekend Mode (keep stale)", value=True)
st.sidebar.divider()

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

# ---------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------
st.subheader("📊 Market Summary")

@st.cache_data(ttl=300, show_spinner="Loading market summary...")
def _load_summary():
    return utils.summarize_assets(
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )

summary_df = _load_summary()

if summary_df.empty:
    st.warning("No data available. Yahoo may be throttling — try again shortly.")
    st.stop()

# highlight stale rows
def _row_style(row):
    if row.get("Stale"):
        return ["background-color: #ffefef"] * len(row)
    return [""] * len(row)

st.dataframe(
    summary_df.style.apply(_row_style, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------
# Detailed Asset View
# ---------------------------------------------------------------------
st.subheader("🔍 Detailed Asset View")

asset_choice = st.selectbox("Select asset for details:", list(utils.ASSET_SYMBOLS.keys()))

@st.cache_data(ttl=300, show_spinner="Loading asset details...")
def _load_asset(asset_choice):
    return utils.asset_prediction_and_backtest(
        asset_choice,
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )

block, df_ind, pts = _load_asset(asset_choice)

if block is None or df_ind.empty:
    st.warning("No data returned for this asset.")
    st.stop()

# --- Metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", f"{block['price']:.2f}")
m2.metric("Signal", block["side"])
m3.metric("Confidence", f"{block['probability']*100:.1f}%")
m4.metric("Win Rate", f"{block['win_rate']:.1f}%")
m5.metric("PF", f"{block['profit_factor']:.2f}")
m6.metric("Sharpe-like", f"{block['sharpe']:.2f}")

if block.get("stale"):
    st.info("ℹ️ Market appears stale (weekend or closed session).")

# --- Candlestick chart with overlays
st.subheader("Price Chart & Structure Overlays")

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df_ind.index,
        open=df_ind["Open"],
        high=df_ind["High"],
        low=df_ind["Low"],
        close=df_ind["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )
)

# Moving averages
if "ema20" in df_ind.columns:
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema20"], name="EMA20", line=dict(width=1)))
if "ema50" in df_ind.columns:
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema50"], name="EMA50", line=dict(width=1)))

# Support/resistance overlays
for key, color in [
    ("sup_y", "rgba(0,200,0,0.3)"),
    ("res_y", "rgba(200,0,0,0.3)"),
]:
    if pts.get(f"{key}", []):
        fig.add_trace(
            go.Scatter(
                x=pts[f"{key.replace('_y','_x')}"],
                y=pts[key],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=key.replace("_y", "").capitalize(),
            )
        )

# Buy/sell markers
fig.add_trace(go.Scatter(
    x=pts["buy_x"], y=pts["buy_y"], mode="markers",
    marker=dict(symbol="triangle-up", color="green", size=8),
    name="Buy"
))
fig.add_trace(go.Scatter(
    x=pts["sell_x"], y=pts["sell_y"], mode="markers",
    marker=dict(symbol="triangle-down", color="red", size=8),
    name="Sell"
))

fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Indicator chart
st.subheader("Indicators Snapshot")
cols = [c for c in ["RSI", "adx", "atr_pct", "macd_hist"] if c in df_ind.columns]
if cols:
    st.line_chart(df_ind[cols].tail(300))

# --- Raw data / debug
with st.expander("Raw indicator dataframe (tail 50)"):
    st.dataframe(df_ind.tail(50), use_container_width=True, hide_index=True)

with st.expander("Engine output block"):
    st.json(block, expanded=False)

st.caption(
    "✅ Local import ensures the correct utils.py v8.3.2 is used.\n"
    "✅ Cached data & randomized backoff prevent Yahoo rate-limit issues.\n"
    "✅ 'Stale' rows = market closed / weekend (handled safely)."
)