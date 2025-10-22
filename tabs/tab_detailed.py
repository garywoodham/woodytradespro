import streamlit as st
import utils
import plotly.graph_objects as go
import numpy as np

def render_detailed():
    st.title("📈 Detailed Asset Analysis")
    st.caption("Interactive charting, AI signals, and entry/exit target levels for each selected market.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()))
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()))

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval=interval)

    if df.empty:
        st.warning(f"No data available for {asset}")
        return

    prediction = utils.train_and_predict(df, horizon=interval, risk=risk)
    if not prediction:
        st.warning("Could not generate prediction for this asset.")
        return

    st.divider()
    st.subheader(f"🔍 {asset} ({interval}) — Signal Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction["prediction"])
    with col2:
        st.metric("Confidence", f"{prediction['probability']*100:.2f}%")
    with col3:
        st.metric("Model Accuracy", f"{prediction['accuracy']*100:.2f}%")

    st.write(f"**Take Profit:** {prediction['tp']:.2f}")
    st.write(f"**Stop Loss:** {prediction['sl']:.2f}")

    # --- Price Chart with Overlay ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    # Overlay TP and SL lines
    last_close = df["Close"].iloc[-1]
    fig.add_hline(y=prediction["tp"], line=dict(color="green", dash="dash"), annotation_text="TP", annotation_position="bottom right")
    fig.add_hline(y=prediction["sl"], line=dict(color="red", dash="dash"), annotation_text="SL", annotation_position="top right")

    fig.update_layout(
        title=f"{asset} Price Chart ({interval})",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_white",
        height=600,
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.divider()
    st.caption("Note: The TP/SL lines reflect the current risk multiplier setting.")