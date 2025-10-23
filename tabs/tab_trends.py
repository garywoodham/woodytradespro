import streamlit as st
import plotly.graph_objects as go
from utils import fetch_data, add_indicators

def render_trends():
    st.title("📈 Market Trends and Technical Analysis")

    asset = st.selectbox(
        "Select Asset",
        ["Gold", "NASDAQ 100", "S&P 500", "EUR/USD", "GBP/USD", "USD/JPY", "Crude Oil", "Bitcoin"],
    )

    interval = st.selectbox("Select Interval", ["15m", "1h", "4h", "1d", "1w"])
    st.write(f"🔍 Showing {asset} — {interval} data")

    symbol_map = {
        "Gold": "GC=F",
        "NASDAQ 100": "^NDX",
        "S&P 500": "^GSPC",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "Crude Oil": "CL=F",
        "Bitcoin": "BTC-USD",
    }
    symbol = symbol_map[asset]

    with st.spinner(f"Fetching {asset} ({symbol})..."):
        df = fetch_data(symbol, interval=interval)
        df = add_indicators(df)

    if df.empty:
        st.warning(f"⚠️ No data available for {asset}.")
        return

    # Handle capitalization differences safely
    def col(name):
        if name in df.columns:
            return name
        elif name.upper() in df.columns:
            return name.upper()
        elif name.lower() in df.columns:
            return name.lower()
        return None

    # --- PRICE + EMA ---
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index,
        open=df[col("Open")],
        high=df[col("High")],
        low=df[col("Low")],
        close=df[col("Close")],
        name="Price"
    ))

    if col("EMA_20"):
        fig_price.add_trace(go.Scatter(x=df.index, y=df[col("EMA_20")],
                                       mode="lines", name="EMA 20"))
    if col("EMA_50"):
        fig_price.add_trace(go.Scatter(x=df.index, y=df[col("EMA_50")],
                                       mode="lines", name="EMA 50"))

    fig_price.update_layout(
        title=f"{asset} Price & EMA Trends ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- RSI ---
    if col("RSI"):
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df[col("RSI")], mode="lines", name="RSI"
        ))
        fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
        fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
        fig_rsi.update_layout(
            title="RSI (Relative Strength Index)",
            height=300,
            yaxis_title="RSI Value"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.info("ℹ️ RSI data not available for this timeframe.")

    # --- MACD ---
    if col("MACD") and col("Signal"):
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df[col("MACD")], mode="lines", name="MACD"
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df[col("Signal")], mode="lines", name="Signal"
        ))
        fig_macd.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            height=300,
            yaxis_title="MACD Value"
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("ℹ️ MACD data not available for this timeframe.")