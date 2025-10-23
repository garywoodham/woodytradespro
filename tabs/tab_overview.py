import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import (
    fetch_data,
    add_indicators,
    ASSET_SYMBOLS,
    calculate_model_performance,  # new util: returns win_rate, avg_return, tp/sl
    get_model_signal  # new util: current model signal + confidence
)


def render_overview():
    st.title("📊 Trading Overview Dashboard")

    # === FILTERS ===
    col1, col2 = st.columns([2, 1])
    with col1:
        interval = st.selectbox(
            "Select Timeframe",
            ["15m", "1h", "4h", "1d", "1w"],
            index=1,
        )
    with col2:
        refresh = st.button("🔄 Refresh Data")

    # === SUMMARY TABLE ===
    st.subheader("📈 Market Summary by Asset")

    summary_data = []

    for asset, symbol in ASSET_SYMBOLS.items():
        try:
            df = fetch_data(symbol, interval)
            if df.empty:
                continue

            df = add_indicators(df)
            signal, confidence, tp, sl = get_model_signal(df)

            perf = calculate_model_performance(df)
            win_rate = perf.get("win_rate", 0)
            avg_return = perf.get("avg_return", 0)

            latest_close = df["Close"].iloc[-1]

            summary_data.append({
                "Asset": asset,
                "Signal": signal,
                "Confidence %": round(confidence, 2),
                "Current Price": round(latest_close, 2),
                "TP": round(tp, 2),
                "SL": round(sl, 2),
                "Win Rate %": round(win_rate, 2),
                "Avg Return %": round(avg_return, 2)
            })
        except Exception as e:
            st.write(f"⚠️ Skipped {asset}: {e}")

    if not summary_data:
        st.warning("No market data available at the moment.")
        return

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # === INDIVIDUAL ASSET VIEW ===
    st.subheader("📉 Detailed Asset Analysis")

    col_chart, col_card = st.columns([3, 1])

    with col_chart:
        asset = st.selectbox("Choose Asset for Detail", list(ASSET_SYMBOLS.keys()))
        symbol = ASSET_SYMBOLS[asset]
        df = fetch_data(symbol, interval)
        df = add_indicators(df)
        if df.empty:
            st.warning(f"No data for {asset}")
            return

        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

        # --- Modelled signals ---
        signal, confidence, tp, sl = get_model_signal(df)
        buy_points = df[df["MACD"] > df["Signal"]]
        sell_points = df[df["MACD"] < df["Signal"]]

        fig.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points["Close"],
                mode="markers",
                marker=dict(color="green", size=8, symbol="triangle-up"),
                name="Buy",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points["Close"],
                mode="markers",
                marker=dict(color="red", size=8, symbol="triangle-down"),
                name="Sell",
            )
        )

        fig.update_layout(
            title=f"{asset} Historical Model Signals ({interval})",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # === CURRENT RECOMMENDATION CARD ===
    with col_card:
        st.markdown("### 🧭 Current Recommendation")

        perf = calculate_model_performance(df)
        win_rate = perf.get("win_rate", 0)
        avg_return = perf.get("avg_return", 0)

        st.markdown(
            f"""
            **Signal:** {signal}  
            **Confidence:** {confidence:.2f}%  
            **TP:** {tp:.2f}  
            **SL:** {sl:.2f}  
            **Win Rate:** {win_rate:.2f}%  
            **Avg Return:** {avg_return:.2f}%  
            **Interval:** {interval}  
            """
        )

        if signal == "Buy":
            st.success("✅ The model indicates a **Buy opportunity**.")
        elif signal == "Sell":
            st.error("⚠️ The model indicates a **Sell signal**.")
        else:
            st.info("⏸ The model suggests to **Hold / Wait** for confirmation.")

    # === (Optional) PERFORMANCE CHART ===
    st.subheader("📊 Strategy Equity Curve")

    perf = calculate_model_performance(df)
    if "equity_curve" in perf:
        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=perf["equity_curve"].index,
                y=perf["equity_curve"].values,
                mode="lines",
                name="Strategy Equity",
            )
        )
        fig_perf.update_layout(
            title=f"{asset} Model Performance ({interval})",
            yaxis_title="Normalized Equity",
            height=400,
        )
        st.plotly_chart(fig_perf, use_container_width=True, config={"displaylogo": False})