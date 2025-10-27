# =============================================================================
# WoodyTrades Pro — Smart Strategy Modes Edition (v8.3.2)
# Streamlit Frontend (Safe Local Import)
# =============================================================================

import sys, importlib.util, pathlib

# --- Force-load local utils.py (bypass any package "utils") ---
UTILS_PATH = pathlib.Path(__file__).parent / "utils.py"
spec = importlib.util.spec_from_file_location("utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)

# --- Confirm correct file loaded ---
import streamlit as st
st.set_page_config(page_title="WoodyTrades Pro Dashboard", layout="wide")
st.write("✅ Loaded utils from:", utils.__file__)

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# Sidebar Configuration
# =============================================================================
st.sidebar.header("⚙️ Configuration")

# --- Interval selector ---
interval_key = st.sidebar.selectbox(
    "Interval",
    list(utils.INTERVALS.keys()) if hasattr(utils, "INTERVALS") else ["1h", "4h", "1d", "1wk"],
    index=2,
    help="Data timeframe for signals & backtests.",
)

# --- Risk profile ---
risk = st.sidebar.selectbox(
    "Risk Profile",
    list(utils.RISK_MULT.keys()) if hasattr(utils, "RISK_MULT") else ["Low", "Medium", "High"],
    index=1,
)

# --- TP/SL scaling ---
tp_sl_mode = st.sidebar.selectbox(
    "TP/SL Scaling",
    list(utils._TP_SL_PROFILES.keys()) if hasattr(utils, "_TP_SL_PROFILES") else ["Normal"],
    index=0,
)

# --- Strategy mode ---
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

# --- Filter strictness ---
filter_level = st.sidebar.selectbox(
    "Filter Strictness",
    ["Loose", "Balanced", "Strict"],
    index=1,
)

# --- Toggles ---
forced_trades = st.sidebar.checkbox("Force Trades (for stats only)", value=False)
calibration_enabled = st.sidebar.checkbox("Calibration Memory", value=True)
weekend_mode = st.sidebar.checkbox("Weekend/Stale Safe Mode", value=True)
show_detailed_view = st.sidebar.checkbox("Show Detailed View", value=True)

st.sidebar.divider()
st.sidebar.caption("v8.3.2 — Smart Strategy Modes Edition")

# =============================================================================
# Cached Summary Fetch (avoid Yahoo rate-limits)
# =============================================================================
@st.cache_data(ttl=600)
def get_summary(interval_key, risk, tp_sl_mode, structure_mode,
                forced_trades, filter_level, calibration_enabled, weekend_mode):
    return utils.summarize_assets(
        interval_key=interval_key,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        weekend_mode=weekend_mode,
    )

# =============================================================================
# Summary Table
# =============================================================================
st.title("📊 WoodyTrades Pro — Smart Strategy Modes Edition")

with st.spinner("Fetching and analyzing assets..."):
    df_summary = get_summary(
        interval_key, risk, tp_sl_mode, structure_mode,
        forced_trades, filter_level, calibration_enabled, weekend_mode
    )

if df_summary.empty:
    st.error("No data available — possibly rate-limited. Try again in a few minutes.")
    st.stop()

st.subheader("📈 Summary Overview")
st.dataframe(df_summary, use_container_width=True)

# =============================================================================
# Detailed View
# =============================================================================
if show_detailed_view:
    st.subheader("🔍 Detailed View")

    asset_list = list(utils.ASSET_SYMBOLS.keys())
    selected_asset = st.selectbox("Select Asset", asset_list, index=0)

    with st.spinner(f"Analyzing {selected_asset}..."):
        pred_block, df_ind, pts = utils.asset_prediction_and_backtest(
            asset=selected_asset,
            interval_key=interval_key,
            risk=risk,
            tp_sl_mode=tp_sl_mode,
            structure_mode=structure_mode,
            forced_trades_enabled=forced_trades,
            filter_level=filter_level,
            calibration_enabled=calibration_enabled,
            weekend_mode=weekend_mode,
        )

    if not pred_block:
        st.warning("No prediction available — possibly rate-limited or insufficient data.")
        st.stop()

    # --- Metrics ---
    cols = st.columns(3)
    cols[0].metric("Signal", pred_block["side"])
    cols[1].metric("Confidence", f"{pred_block['probability']*100:.1f}%")
    cols[2].metric("Price", f"{pred_block['price']:.2f}")

    cols = st.columns(3)
    cols[0].metric("TP", f"{pred_block['tp']:.2f}" if pred_block["tp"] else "–")
    cols[1].metric("SL", f"{pred_block['sl']:.2f}" if pred_block["sl"] else "–")
    cols[2].metric("R/R", f"{pred_block['rr']:.2f}" if pred_block["rr"] else "–")

    cols = st.columns(3)
    cols[0].metric("Win Rate", f"{pred_block['win_rate']:.1f}%")
    cols[1].metric("PF", f"{pred_block['profit_factor']:.2f}")
    cols[2].metric("Sharpe-Like", f"{pred_block['sharpe']:.2f}")

    st.write(f"Sentiment: {pred_block['sentiment']:.2f} | "
             f"ADX: {pred_block['adx']:.2f} | "
             f"ATR: {pred_block['atr']:.4f} | "
             f"Stale: {pred_block['stale']}")

    # --- Chart ---
    if not df_ind.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_ind.index,
            open=df_ind["Open"], high=df_ind["High"],
            low=df_ind["Low"], close=df_ind["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=pts["buy_x"], y=pts["buy_y"], mode="markers",
            marker_symbol="triangle-up", marker_color="green", name="Buy Signals"
        ))
        fig.add_trace(go.Scatter(
            x=pts["sell_x"], y=pts["sell_y"], mode="markers",
            marker_symbol="triangle-down", marker_color="red", name="Sell Signals"
        ))

        fig.update_layout(
            height=600,
            title=f"{selected_asset} — {interval_key} Chart",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

st.success("✅ Dashboard ready.")