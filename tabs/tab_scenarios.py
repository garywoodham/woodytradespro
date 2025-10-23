import streamlit as st
import plotly.graph_objects as go
from utils import fetch_data, add_indicators, ASSET_SYMBOLS


def render_scenarios():
    st.title("🎯 Market Scenarios and Predictions")

    # --- Asset & interval selection ---
    asset = st.selectbox(
        "Select Asset",
        [
            "Gold",
            "NASDAQ 100",
            "S&P 500",
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "Crude Oil",
            "Bitcoin",
        ],
    )

    interval = st.selectbox(
        "Select Interval",
        ["15m", "1h", "4h", "1d", "1w"],
        index=1,
    )

    symbol = ASSET_SYMBOLS.get(asset)
    st.info(f"Fetching and analyzing **{asset} ({symbol})**... please wait ⏳")

    # --- Fetch data ---
    df = fetch_data(symbol, interval)
    df = add_indicators(df)

    if df.empty:
        st.error(f"⚠️ No data available for {asset}. Please try again later.")
        return

    # --- Helper to get safe column names ---
    def get_col(df, name):
        cols = {c.lower(): c for c in df.columns}
        return cols.get(name.lower())

    # --- PRICE CHART ---
    fig_price = go.Figure()

    open_col = get_col(df, "Open")
    high_col = get_col(df, "High")
    low_col = get_col(df, "Low")
    close_col = get_col(df, "Close")

    if open_col and high_col and low_col and close_col:
        fig_price.add_trace(
            go.Candlestick(
                x=df.index,
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="Price",
            )
        )

    ema20_col = get_col(df, "EMA_20")
    ema50_col = get_col(df, "EMA_50")

    if ema20_col:
        fig_price.add_trace(
            go.Scatter(x=df.index, y=df[ema20_col], mode="lines", name="EMA 20")
        )
    if ema50_col:
        fig_price.add_trace(
            go.Scatter(x=df.index, y=df[ema50_col], mode="lines", name="EMA 50")
        )

    fig_price.update_layout(
        title=f"{asset} Price & Trend Analysis ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
    )
    st.plotly_chart(fig_price, use_container_width=True, config={"displaylogo": False})

    # --- RSI Chart ---
    rsi_col = get_col(df, "RSI")
    if rsi_col:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df[rsi_col], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
        fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
        fig_rsi.update_layout(
            title="RSI (Relative Strength Index)",
            height=300,
            yaxis_title="RSI Value",
        )
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("ℹ️ RSI data not available for this timeframe.")

    # --- MACD Chart ---
    macd_col = get_col(df, "MACD")
    signal_col = get_col(df, "Signal")

    if macd_col and signal_col:
        fig_macd = go.Figure()
        fig_macd.add_trace(
            go.Scatter(x=df.index, y=df[macd_col], mode="lines", name="MACD")
        )
        fig_macd.add_trace(
            go.Scatter(x=df.index, y=df[signal_col], mode="lines", name="Signal")
        )
        fig_macd.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            height=300,
            yaxis_title="MACD Value",
        )
        st.plotly_chart(fig_macd, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("ℹ️ MACD data not available for this timeframe.")

    # --- Scenario summary ---
    st.subheader("📊 Market Scenario Summary")
    st.markdown(
        f"""
        - **Asset:** {asset} ({symbol})  
        - **Interval:** {interval}  
        - **Current Trend:** {'📈 Uptrend' if df[get_col(df, 'Close')].iloc[-1] > df[get_col(df, 'Close')].iloc[-5] else '📉 Downtrend'}  
        - **RSI Level:** {round(df[rsi_col].iloc[-1], 2) if rsi_col else 'N/A'}  
        - **MACD Signal:** {'Bullish' if macd_col and df[macd_col].iloc[-1] > df[signal_col].iloc[-1] else 'Bearish' if macd_col else 'N/A'}
        """
    )

    # --- Simulated win rate & returns ---
    st.subheader("💰 Strategy Performance Simulation")

    close = df[get_col(df, "Close")].dropna()

    if macd_col and signal_col and len(close) > 50:
        df["signal"] = 0
        df.loc[df[macd_col] > df[signal_col], "signal"] = 1  # Buy
        df.loc[df[macd_col] < df[signal_col], "signal"] = -1  # Sell

        # shift to simulate signal before next price change
        df["future_return"] = close.pct_change().shift(-1)
        df["strategy_return"] = df["signal"] * df["future_return"]

        total_return = (df["strategy_return"] + 1).prod() - 1
        win_rate = (df["strategy_return"] > 0).mean() * 100

        st.write(
            f"""
            **Simulated Results (MACD-based strategy):**  
            - ✅ **Win Rate:** {win_rate:.2f}%  
            - 💹 **Cumulative Return:** {total_return * 100:.2f}%  
            - 📈 **Holding Period:** {len(df)} data points  
            """
        )

        # Chart: cumulative performance
        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=df.index,
                y=(1 + df["strategy_return"].fillna(0)).cumprod(),
                mode="lines",
                name="Strategy Equity Curve",
            )
        )
        fig_perf.add_trace(
            go.Scatter(
                x=df.index,
                y=(close / close.iloc[0]),
                mode="lines",
                name="Buy & Hold",
            )
        )
        fig_perf.update_layout(
            title="Simulated Performance vs Buy & Hold",
            yaxis_title="Equity Growth (normalized)",
            height=400,
        )
        st.plotly_chart(fig_perf, use_container_width=True, config={"displaylogo": False})

    else:
        st.info("ℹ️ Not enough MACD data to simulate performance for this timeframe.")