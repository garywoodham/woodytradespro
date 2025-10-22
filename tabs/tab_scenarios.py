import streamlit as st
import utils
import plotly.graph_objects as go

def render_scenarios():
    st.title("🧩 Scenario Testing")
    st.caption("Simulate entry/exit and backtest results for the selected asset.")

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
        st.error(f"Unable to create scenario for {asset}")
        return

    # Simulated trade outcome visualization
    st.subheader(f"Scenario: {asset} ({pred['prediction']})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"))
    fig.add_hline(y=pred["tp"], line=dict(color="green", width=2, dash="dash"), name="Take Profit")
    fig.add_hline(y=pred["sl"], line=dict(color="red", width=2, dash="dash"), name="Stop Loss")

    fig.update_layout(
        title=f"Simulated Trade Path ({interval})",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, width="stretch")

    st.metric("Prediction", pred["prediction"])
    st.metric("Confidence", f"{pred['probability']*100:.1f}%")
    st.metric("Model Accuracy", f"{pred['accuracy']*100:.2f}%")