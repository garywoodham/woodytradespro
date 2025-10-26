# app.py - Woody Trades Pro Smart v7.9.2
# Calibrated Confidence • EV/PF • TP/SL Highlights • Historical Trade Overlay

import os
import traceback
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
    ASSET_SYMBOLS,
    INTERVALS,
    RISK_MULT,
    _get_higher_tf_bias_for_asset,   # internal utils (used locally here)
    _compute_tp_sl_regime_dynamic,   # internal utils (used for plotting)
    CONF_EXEC_THRESHOLD,             # note: updated dynamically in utils when called
    FORCED_TRADE_PROB,
    FORCED_CONF_MIN,
)

# We also call some private-ish helpers from utils logic patterns:
# We'll replicate lightweight versions of:
#   - signal computation w/ HTF bias
#   - conf-adaptive horizon
# Without re-importing the entire backtest, to avoid circular execution issues.

from utils import (
    add_indicators,
    _compute_signal_row_with_higher_tf,
)

# ---------------------------------------------------------------------
# Streamlit setup / stability
# ---------------------------------------------------------------------
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"
st.set_page_config(page_title="Woody Trades Pro - Smart v7.9.2", layout="wide")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Settings")

interval_key = st.sidebar.selectbox(
    "Timeframe",
    options=list(INTERVALS.keys()),
    index=list(INTERVALS.keys()).index("1h") if "1h" in INTERVALS else 0,
    help="Candle timeframe used for signals and backtest window.",
)

risk = st.sidebar.selectbox(
    "Risk Profile",
    options=["Low", "Medium", "High"],
    index=1,
    help="Controls TP/SL distance (ATR multiples).",
)

filter_level = st.sidebar.selectbox(
    "Signal Filtering Level",
    options=["Loose", "Balanced", "Strict"],
    index=1,
    help="Loose = more trades, Strict = fewer but higher conviction.",
)

asset_choice = st.sidebar.selectbox(
    "Focus Asset",
    options=list(ASSET_SYMBOLS.keys()),
    index=0,
    help="Used in the Detailed / Chart sections below.",
)

show_hist_trades = st.sidebar.checkbox(
    "Show historical TP/SL markers on chart",
    value=True,
    help="Overlays historical trade entries, TP/SL levels, and outcomes.",
)

st.sidebar.caption(
    "Smart v7.9.2 • Calibrated ML • EV/PF • TP/SL • Dynamic Filter • Trade Overlays"
)

# ---------------------------------------------------------------------
# Cached calls to utils
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_summary(interval_key: str, risk: str, filter_level: str) -> pd.DataFrame:
    return summarize_assets(interval_key, risk, use_cache=True, filter_level=filter_level)

@st.cache_data(show_spinner=False)
def load_prediction_and_chart(asset: str, interval_key: str, risk: str, filter_level: str):
    return asset_prediction_and_backtest(asset, interval_key, risk, use_cache=True, filter_level=filter_level)

@st.cache_data(show_spinner=False)
def load_price_df(asset: str, interval_key: str, filter_level: str):
    # utils.load_asset_with_indicators returns (symbol, df_ind, sig_pts)
    return load_asset_with_indicators(asset, interval_key, use_cache=True, filter_level=filter_level)

# ---------------------------------------------------------------------
# Helper (local): reconstruct historical trades for plotting
# ---------------------------------------------------------------------
def reconstruct_trade_history_for_chart(
    df_ind: pd.DataFrame,
    symbol: str,
    interval_key: str,
    risk: str,
    max_lookback: int = 250,
):
    """
    We replay simplified trade logic across df_ind to produce:
    - entries (timestamp, price, side, conf)
    - TP/SL levels
    - exit outcome (TP or SL), exit time, exit price
    This mirrors utils._backtest_once(), but we keep it chart-focused.

    Returns dict with:
      entries: list of dicts {
          'time': ts,
          'side': "Buy"/"Sell",
          'entry_price': float,
          'tp': float,
          'sl': float,
          'conf': float,
          'exit_time': ts or None,
          'exit_price': float or None,
          'outcome': "TP"/"SL"/None,
          'rr_local': float,
      }
    """
    out = {"entries": []}
    if df_ind is None or df_ind.empty or len(df_ind) < 40:
        return out

    # limit lookback for performance/clarity
    dfw = df_ind.copy()
    if len(dfw) > max_lookback:
        dfw = dfw.iloc[-max_lookback:].copy()

    # get higher timeframe bias once (approx; same shortcut as utils)
    try:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    except Exception:
        higher_bias = 0

    # We'll reuse CONF_EXEC_THRESHOLD from utils (already tuned by filter_level when summary was built)
    # We won't recompute filter_level tuning here to avoid recursion.

    # choose a horizon baseline similar to utils (10 candles base)
    base_horizon = 10

    for i in range(20, len(dfw) - base_horizon):
        prev_row = dfw.iloc[i - 1]
        cur_row  = dfw.iloc[i]

        # same signal logic as utils
        side_local, conf_local = _compute_signal_row_with_higher_tf(prev_row, cur_row, higher_bias)
        side = side_local

        # relaxed gating like utils.backtest_once
        trend_reg = int(cur_row.get("trend_regime", 0))
        range_reg = int(cur_row.get("range_regime", 0))
        bullish_trend = trend_reg and (cur_row.get("ema20", 0) > cur_row.get("ema50", 0))
        bearish_trend = trend_reg and (cur_row.get("ema20", 0) < cur_row.get("ema50", 0))
        regime_allows_relaxed = (
            (side == "Buy"  and bullish_trend) or
            (side == "Sell" and bearish_trend) or
            (range_reg and side in ["Buy", "Sell"])
        )

        # apply gating / forced entry logic
        if side == "Hold":
            if (conf_local > FORCED_CONF_MIN) and (np.random.rand() < FORCED_TRADE_PROB):
                side = np.random.choice(["Buy", "Sell"])
            else:
                continue
        else:
            if conf_local < CONF_EXEC_THRESHOLD and not regime_allows_relaxed:
                continue

        # at this point we consider we "took the trade"
        entry_px = float(cur_row["Close"])
        atr_now = float(cur_row.get("atr", entry_px * 0.005))

        # compute TP/SL using dynamic regime TP/SL
        tp_lvl, sl_lvl = _compute_tp_sl_regime_dynamic(entry_px, atr_now, side, risk, cur_row)

        # approximate RR for hover
        if side == "Buy":
            reward_dist = max(tp_lvl - entry_px, 1e-12)
            risk_dist   = max(entry_px - sl_lvl, 1e-12)
        else:
            reward_dist = max(entry_px - tp_lvl, 1e-12)
            risk_dist   = max(sl_lvl - entry_px, 1e-12)
        rr_local = reward_dist / risk_dist if risk_dist != 0 else 1.0

        # adaptive horizon same style as utils
        dyn_horizon = int(base_horizon * (0.8 + conf_local * 0.6))
        if dyn_horizon < 1:
            dyn_horizon = 1
        if dyn_horizon > base_horizon * 2:
            dyn_horizon = base_horizon * 2

        exit_time = None
        exit_price = None
        outcome = None

        # walk forward to see TP or SL hit first
        for j in range(1, dyn_horizon + 1):
            if i + j >= len(dfw):
                break
            nxt = dfw.iloc[i + j]
            nxt_px = float(nxt["Close"])

            if side == "Buy":
                if nxt_px >= tp_lvl:
                    exit_time = dfw.index[i + j]
                    exit_price = nxt_px
                    outcome = "TP"
                    break
                elif nxt_px <= sl_lvl:
                    exit_time = dfw.index[i + j]
                    exit_price = nxt_px
                    outcome = "SL"
                    break
            else:  # Sell
                if nxt_px <= tp_lvl:
                    exit_time = dfw.index[i + j]
                    exit_price = nxt_px
                    outcome = "TP"
                    break
                elif nxt_px >= sl_lvl:
                    exit_time = dfw.index[i + j]
                    exit_price = nxt_px
                    outcome = "SL"
                    break

        out["entries"].append({
            "time": dfw.index[i],
            "side": side,
            "conf": float(conf_local),
            "entry_price": entry_px,
            "tp": float(tp_lvl),
            "sl": float(sl_lvl),
            "rr_local": float(rr_local),
            "exit_time": exit_time,
            "exit_price": float(exit_price) if exit_price is not None else None,
            "outcome": outcome,
        })

    return out


# ---------------------------------------------------------------------
# Market Summary
# ---------------------------------------------------------------------
st.header("📊 Market Summary")

try:
    df_summary = load_summary(interval_key, risk, filter_level)
    if df_summary is None or df_summary.empty:
        st.warning("No summary data available.")
    else:
        # format + highlighting for summary table

        def highlight_cells(val, col):
            # Signal highlighting
            if col == "Signal":
                if val == "Buy":
                    return "background-color: rgba(0,255,0,0.15)"
                elif val == "Sell":
                    return "background-color: rgba(255,0,0,0.15)"
                else:
                    return "background-color: rgba(128,128,128,0.08)"

            # EV% highlighting
            if col == "EV%":
                if val > 0.5:
                    return "background-color: rgba(0,255,0,0.1)"
                elif val < 0:
                    return "background-color: rgba(255,0,0,0.1)"

            # ProfitFactor highlighting
            if col == "ProfitFactor":
                if val > 1.2:
                    return "background-color: rgba(0,200,0,0.08)"

            return ""

        def highlight_table(df):
            return df.style.apply(
                lambda col: [highlight_cells(v, col.name) for v in col],
                axis=0
            )

        df_display = df_summary[
            [
                "Asset",
                "Signal",
                "Probability",
                "WinRate",
                "EV%",
                "ProfitFactor",
                "TP",
                "SL",
                "Trades",
                "Return%",
                "MaxDD%",
                "SharpeLike",
                "Stale",
            ]
        ].copy()

        st.dataframe(
            highlight_table(df_display),
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:
    st.error(f"Error loading summary: {e}")
    st.code(traceback.format_exc())


# ---------------------------------------------------------------------
# Detailed View
# ---------------------------------------------------------------------
st.header("🔍 Detailed View")

try:
    pred_block, df_asset_ind = load_prediction_and_chart(asset_choice, interval_key, risk, filter_level)

    if not pred_block:
        st.warning("No prediction data available for this asset.")
    else:
        # Row 1: Signal + confidences
        colA, colB, colC, colD = st.columns(4)
        colA.metric(
            "Signal",
            pred_block.get("side", "Hold"),
            help="Current directional bias after regime logic + higher timeframe confirmation."
        )
        colB.metric(
            "Confidence (Raw)",
            f"{round(pred_block.get('probability_raw', 0)*100,2)}%",
            help="Blended rule+ML before calibration scaling."
        )
        colC.metric(
            "Confidence (Calibrated)",
            f"{round(pred_block.get('probability_calibrated', 0)*100,2)}%",
            help="Final calibrated conviction (asset-specific reliability applied)."
        )
        colD.metric(
            "Win Rate (Backtest)",
            f"{round(pred_block.get('win_rate', 0),2)}%",
            help="Average hit rate from the relaxed multi-seed backtest."
        )

        # Row 2: TP/SL, RR, EV%
        colE, colF, colG, colH = st.columns(4)
        tp_val = pred_block.get("tp")
        sl_val = pred_block.get("sl")
        rr_val = pred_block.get("rr")
        colE.metric("Take Profit", f"{tp_val:.4f}" if tp_val else "—")
        colF.metric("Stop Loss", f"{sl_val:.4f}" if sl_val else "—")
        colG.metric("R/R", f"{rr_val:.2f}" if rr_val else "—")
        colH.metric(
            "EV% / Trade",
            f"{pred_block.get('ev_pct', 0):.3f}",
            help="Expected value per trade (%) from simulation."
        )

        # Row 3: PF, Trades, Return%, Stale
        colI, colJ, colK, colL = st.columns(4)
        colI.metric(
            "Profit Factor",
            f"{pred_block.get('profit_factor', 0):.2f}",
            help="Total gains / total losses. >1 = net profitable."
        )
        colJ.metric(
            "Trades (BT)",
            f"{pred_block.get('trades', 0)}",
            help="Avg trades per symbol in the backtest ensemble."
        )
        colK.metric(
            "BT Return%",
            f"{pred_block.get('backtest_return_pct', 0):.2f}",
            help="Total balance change (%) in the sim."
        )
        stale_flag = "⚠️ STALE" if pred_block.get("stale") else "✅ Fresh"
        colL.metric(
            "Data Status",
            stale_flag,
            help="'STALE' often means weekend / market closed / no recent candles."
        )

    # -----------------------------------------------------------------
    # Chart section (candles + EMAs + optional historical trade overlay)
    # -----------------------------------------------------------------
    if isinstance(df_asset_ind, pd.DataFrame) and not df_asset_ind.empty:
        # We need symbol to reconstruct trade marks:
        focus_symbol = ASSET_SYMBOLS[asset_choice]

        # Build trade overlay (entry/exit/TP/SL/etc.)
        trade_overlay = reconstruct_trade_history_for_chart(
            df_asset_ind,
            symbol=focus_symbol,
            interval_key=interval_key,
            risk=risk,
            max_lookback=250,
        )

        fig = go.Figure()

        # Candle trace
        fig.add_trace(go.Candlestick(
            x=df_asset_ind.index,
            open=df_asset_ind["Open"],
            high=df_asset_ind["High"],
            low=df_asset_ind["Low"],
            close=df_asset_ind["Close"],
            name="Price",
        ))

        # EMA20 / EMA50 overlays
        if "ema20" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.index,
                y=df_asset_ind["ema20"],
                mode="lines",
                name="EMA20",
                line=dict(width=1.2)
            ))
        if "ema50" in df_asset_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_asset_ind.index,
                y=df_asset_ind["ema50"],
                mode="lines",
                name="EMA50",
                line=dict(width=1.2, dash="dot")
            ))

        # Historical trade overlay (if enabled)
        if show_hist_trades and trade_overlay["entries"]:
            entry_times      = []
            entry_prices     = []
            entry_sides      = []
            entry_conf       = []
            tp_lines_x_start = []
            tp_lines_x_end   = []
            tp_lines_y       = []
            sl_lines_x_start = []
            sl_lines_x_end   = []
            sl_lines_y       = []
            tp_hit_times     = []
            tp_hit_prices    = []
            sl_hit_times     = []
            sl_hit_prices    = []

            for trade in trade_overlay["entries"]:
                t0 = trade["time"]
                ep = trade["entry_price"]
                side = trade["side"]
                conf = trade["conf"]
                tpv = trade["tp"]
                slv = trade["sl"]
                rr  = trade["rr_local"]
                t_exit = trade["exit_time"]
                p_exit = trade["exit_price"]
                outcome = trade["outcome"]

                # entry marker
                entry_times.append(t0)
                entry_prices.append(ep)
                entry_sides.append(side)
                entry_conf.append(conf)

                # TP line (dashed green-ish)
                tp_lines_x_start.append(t0)
                tp_lines_x_end.append(t0)
                tp_lines_y.append(tpv)

                # SL line (dashed red-ish)
                sl_lines_x_start.append(t0)
                sl_lines_x_end.append(t0)
                sl_lines_y.append(slv)

                # exit marker if known hit
                if outcome == "TP" and t_exit is not None and p_exit is not None:
                    tp_hit_times.append(t_exit)
                    tp_hit_prices.append(p_exit)
                elif outcome == "SL" and t_exit is not None and p_exit is not None:
                    sl_hit_times.append(t_exit)
                    sl_hit_prices.append(p_exit)

            # plot entry points (blue dots, hover shows side/conf/RR)
            fig.add_trace(go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers",
                name="Entry",
                marker=dict(
                    symbol="circle",
                    size=7,
                    color="rgba(0,0,255,0.6)",
                    line=dict(width=1, color="rgba(0,0,80,0.8)")
                ),
                text=[
                    f"{s} @ {round(p,4)}"
                    f"<br>Conf={round(c*100,2)}%"
                    for s, p, c in zip(entry_sides, entry_prices, entry_conf)
                ],
                hovertemplate="%{text}<extra></extra>",
            ))

            # TP horizontal markers (as short line segments)
            fig.add_trace(go.Scatter(
                x=tp_lines_x_start + tp_lines_x_end,
                y=tp_lines_y + tp_lines_y,
                mode="lines",
                name="TP level",
                line=dict(color="rgba(0,200,0,0.4)", width=1, dash="dash"),
                hovertemplate="TP %{y:.4f}<extra></extra>",
            ))

            # SL horizontal markers
            fig.add_trace(go.Scatter(
                x=sl_lines_x_start + sl_lines_x_end,
                y=sl_lines_y + sl_lines_y,
                mode="lines",
                name="SL level",
                line=dict(color="rgba(200,0,0,0.4)", width=1, dash="dash"),
                hovertemplate="SL %{y:.4f}<extra></extra>",
            ))

            # TP hit markers
            if tp_hit_times:
                fig.add_trace(go.Scatter(
                    x=tp_hit_times,
                    y=tp_hit_prices,
                    mode="markers",
                    name="TP hit",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="rgba(0,255,0,0.7)",
                        line=dict(width=1, color="rgba(0,100,0,0.9)")
                    ),
                    hovertemplate="✅ TP hit @ %{y:.4f}<extra></extra>",
                ))

            # SL hit markers
            if sl_hit_times:
                fig.add_trace(go.Scatter(
                    x=sl_hit_times,
                    y=sl_hit_prices,
                    mode="markers",
                    name="SL hit",
                    marker=dict(
                        symbol="x",
                        size=9,
                        color="rgba(255,0,0,0.8)",
                        line=dict(width=1, color="rgba(80,0,0,1)")
                    ),
                    hovertemplate="❌ SL hit @ %{y:.4f}<extra></extra>",
                ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=500,
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("No chart data available for this asset.")

except Exception as e:
    st.error(f"Error in detailed view: {e}")
    st.code(traceback.format_exc())

# ---------------------------------------------------------------------
# Debug / Raw
# ---------------------------------------------------------------------
st.header("🧰 Debug / Raw Data")
with st.expander("Show model input / indicators / backtest inputs"):
    try:
        symbol_for_asset, df_asset_full, sig_pts = load_price_df(asset_choice, interval_key, filter_level)
        st.write(f"Symbol: {symbol_for_asset}")
        st.dataframe(df_asset_full.tail(200), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        st.code(traceback.format_exc())

st.caption(
    "Smart v7.9.2 • Calibrated ML • EV/PF • TP/SL • Dynamic Filter • Adaptive Horizon • "
    "HTF Bias • ATR/ADX Regimes • Trade Overlay"
)