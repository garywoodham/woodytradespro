import streamlit as st
import utils
import plotly.graph_objects as go

def render_trends():
    st.title("📉 Market Technical Trends")
    st.caption("Track RSI, MACD, and volatility to gauge market momentum and potential reversals.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()))

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval=interval)

    if df.empty:
        st.warning(f"No data available for {asset}")
        return

    st.subheader(f"Technical Indicators — {asset}")

    # --- Price Chart ---
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
    fig_price.update_layout(title=f"{asset} Price", height=400, template="plotly_white")
    st.plotly_chart(fig_price, width="stretch", config={"displayModeBar": False})

    # --- RSI ---
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI", line=dict(color="orange")))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
    fig_rsi.update_layout(title="RSI (Relative Strength Index)", height=300, template="plotly_white")
    st.plotly_chart(fig_rsi, width="stretch", config={"displayModeBar": False})

    # --- MACD ---
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color="blue")))
    fig_macd.update_layout(title="MACD Indicator", height=300, template="plotly_white")
    st.plotly_chart(fig_macd, width="stretch", config={"displayModeBar": False})

    # --- Volatility ---
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df["volatility"], mode="lines", name="Volatility", line=dict(color="purple")))
    fig_vol.update_layout(title="Rolling Volatility", height=300, template="plotly_white")
    st.plotly_chart(fig_vol, width="stretch", config={"displayModeBar": False})

    st.caption("Indicators auto-update based on interval and asset selection.")