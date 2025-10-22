import streamlit as st
import plotly.graph_objects as go
import utils

def render_detailed():
    st.title("📈 Detailed Analysis")
    st.caption("Combines AI predictions, overlays, and sentiment-driven entry/exit targeting.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()))
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()))

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval=interval)

    if df.empty:
        st.warning(f"No data available for {asset}")
        return

    pred = utils.train_and_predict(df, horizon=interval, risk=risk)
    if not pred:
        st.error(f"Unable to generate prediction for {asset}")
        return

    st.subheader(f"{asset} | {pred['prediction']} Signal ({pred['probability']*100:.1f}% Confidence)")
    st.write(f"🎯 Accuracy: **{pred['accuracy']*100:.2f}%**")

    # Plot candlestick with TP/SL
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=asset
    )])

    fig.add_hline(y=pred["tp"], line=dict(color="green", width=2, dash="dash"), name="Take Profit")
    fig.add_hline(y=pred["sl"], line=dict(color="red", width=2, dash="dash"), name="Stop Loss")

    fig.update_layout(
        title=f"{asset} Price Chart ({interval})",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, width="stretch")

    st.info(f"💡 Suggested Trade: **{pred['prediction']}** | TP: {pred['tp']:.2f} | SL: {pred['sl']:.2f}")