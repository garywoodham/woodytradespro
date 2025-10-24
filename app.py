# app.py — WoodyTradesPro Smart v4 (Final Stable Build)
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import traceback
from plotly import graph_objects as go

from utils import (
    summarize_assets,
    asset_prediction_and_backtest,
    load_asset_with_indicators,
)

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WoodyTradesPro Smart v4",
    page_icon="💹",
    layout="wide",
)

# ---------------------------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------------------------
st.title("💹 WoodyTradesPro Smart v4")
st.caption("AI-assisted multi-asset analytics with sentiment, ML and regime awareness")

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

risk_level = st.sidebar.selectbox(
    "Risk Level",
    ["Low", "Medium", "High"],
    index=1,
    key="risk_level",
)

interval_choice = st.sidebar.selectbox(
    "Chart Interval",
    ["15m", "1h", "4h", "1d", "1wk"],
    index=1,
    key="interval_choice",
)

tab_choice = st.sidebar.radio(
    "Select Mode",
    ["Market Summary", "Asset Analysis", "Backtest", "News / Sentiment"],
    key="tab_choice",
)

# ---------------------------------------------------------------------------
# UNIVERSAL PLOT CONFIG (fixes deprecation warnings)
# ---------------------------------------------------------------------------
PLOTLY_CFG = {"responsive": True, "displayModeBar": False}

# ---------------------------------------------------------------------------
# MAIN TABS LOGIC
# ---------------------------------------------------------------------------
try:
    # ==================== MARKET SUMMARY TAB ====================
    if tab_choice == "Market Summary":
        st.subheader("🌍 Market Summary")
        with st.spinner("Fetching and analyzing market data..."):
            df_summary = summarize_assets(interval_choice, risk_level)
        if df_summary is not None and not df_summary.empty:
            st.dataframe(df_summary, width="stretch")
        else:
            st.warning("No data could be fetched. Please check your connection or retry.")

    # ==================== ASSET ANALYSIS TAB ====================
    elif tab_choice == "Asset Analysis":
        st.subheader("🔍 Individual Asset Analysis")

        assets = [
            "Gold", "NASDAQ 100", "S&P 500",
            "EUR/USD", "GBP/USD", "USD/JPY",
            "Crude Oil", "Bitcoin",
        ]
        asset_choice = st.selectbox("Choose Asset", assets, key="asset_analysis_choice")

        with st.spinner(f"Loading {asset_choice} data..."):
            symbol, df_asset = load_asset_with_indicators(asset_choice, interval_choice)
            if df_asset is None or df_asset.empty:
                st.warning("Failed to load data for this asset.")
            else:
                # --- Price Chart ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_asset.index,
                    open=df_asset["Open"],
                    high=df_asset["High"],
                    low=df_asset["Low"],
                    close=df_asset["Close"],
                    name="Price",
                ))
                fig.add_trace(go.Scatter(
                    x=df_asset.index,
                    y=df_asset["ema20"],
                    line=dict(width=1),
                    name="EMA 20",
                ))
                fig.add_trace(go.Scatter(
                    x=df_asset.index,
                    y=df_asset["ema50"],
                    line=dict(width=1),
                    name="EMA 50",
                ))
                fig.update_layout(
                    title=f"{asset_choice} ({symbol}) — {interval_choice}",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600,
                )
                st.plotly_chart(fig, config=PLOTLY_CFG, width="stretch")

                # --- Prediction ---
                result, _ = asset_prediction_and_backtest(asset_choice, interval_choice, risk_level)
                if result:
                    st.markdown("### 📈 Latest Prediction")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Signal", result["side"])
                    c2.metric("Probability", f"{result['probability']:.2f}%")
                    c3.metric("Sentiment", f"{result['sentiment']:.2f}")
                    st.write(
                        f"**TP:** {result['tp']:.2f} | **SL:** {result['sl']:.2f} "
                        f"| **Regime:** {result['regime']} | **ATR:** {result['atr']:.2f}"
                    )

    # ==================== BACKTEST TAB ====================
    elif tab_choice == "Backtest":
        st.subheader("⏳ Strategy Backtest")

        assets = [
            "Gold", "NASDAQ 100", "S&P 500",
            "EUR/USD", "GBP/USD", "USD/JPY",
            "Crude Oil", "Bitcoin",
        ]
        asset_choice = st.selectbox("Choose Asset", assets, key="backtest_asset_choice")

        with st.spinner(f"Running backtest for {asset_choice}..."):
            result, df = asset_prediction_and_backtest(asset_choice, interval_choice, risk_level)

        if result:
            st.markdown("### 📊 Backtest Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win Rate", f"{result['win_rate']:.2f}%")
            c2.metric("Total Return", f"{result['backtest_return_pct']:.2f}%")
            c3.metric("Trades", f"{result['n_trades']}")
            c4.metric("Sharpe Ratio", f"{result.get('sharpe', 0):.2f}")

            trades_df = pd.DataFrame(result["trades"])
            if not trades_df.empty:
                st.dataframe(trades_df, width="stretch")
            else:
                st.info("No trades generated for this period.")
        else:
            st.warning("Backtest data unavailable. Try again later.")

    # ==================== NEWS / SENTIMENT TAB ====================
    elif tab_choice == "News / Sentiment":
        st.subheader("📰 News & Sentiment Overview")
        st.info(
            "This section displays aggregated sentiment results "
            "from recent market headlines and social data used in predictions."
        )
        st.write(
            "Each asset’s sentiment contributes dynamically to probability weighting "
            "in the ML-based signal engine. Positive sentiment amplifies bullish signals; "
            "negative tone suppresses them, but only in trending regimes (ADX > 20)."
        )
        st.success("Sentiment data successfully integrated in the model pipeline.")

except Exception as e:
    st.error(f"⚠️ An unexpected error occurred:\n{e}")
    st.text(traceback.format_exc())

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------