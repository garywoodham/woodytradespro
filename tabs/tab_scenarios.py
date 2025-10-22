import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils
import time

def render_scenarios():
    st.title("📊 AI + Theory Trading Scenarios")
    st.caption("Simulate outcomes under multiple risk levels with AI and trading methodology overlays.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)
    interval = st.selectbox("Select Timeframe", list(utils.INTERVALS.keys()), index=1)
    symbol = utils.ASSET_SYMBOLS[asset]

    st.info(f"Fetching and analyzing **{asset} ({symbol})**... please wait ⏳")
    df = utils.fetch_data(symbol, utils.INTERVALS[interval]["interval"], utils.INTERVALS[interval]["period"])

    if df.empty:
        st.error(f"No data available for {asset}. Please try again later.")
        return
    st.success(f"✅ Loaded {len(df)} candles for {asset}.")

    results = []
    progress = st.progress(0)
    for i, risk in enumerate(utils.RISK_MULT.keys(), 1):
        progress.progress(i / len(utils.RISK_MULT))
        try:
            pred = utils.train_and_predict(df, horizon=interval, risk=risk)
            back = utils.backtest_signals(df, pred)
            if pred:
                results.append({
                    "Risk": risk,
                    "Prediction": pred["prediction"],
                    "Confidence": round(pred["probability"] * 100, 2),
                    "Accuracy": round(pred["accuracy"] * 100, 2),
                    "Win Rate": round(back["winrate"] * 100, 2),
                    "Total Return": round(back["total_return"] * 100, 2),
                    "TP": round(pred["tp"], 4),
                    "SL": round(pred["sl"], 4)
                })
        except Exception as e:
            st.warning(f"⚠️ Error processing {risk}: {e}")
        time.sleep(0.5)

    progress.progress(1.0)
    if not results:
        st.error("No scenarios produced valid results.")
        return

    st.subheader("📈 Scenario Performance Overview")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.divider()
    st.subheader("📊 Trade Signal Visualization")

    chosen = st.radio("Select Risk Level to Visualize", list(utils.RISK_MULT.keys()), index=1)
    pred = utils.train_and_predict(df, horizon=interval, risk=chosen)
    back = utils.backtest_signals(df, pred)

    if not pred:
        st.warning("No valid prediction for visualization.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price", line=dict(width=1.8)))

    # Mark signals
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for i in range(1, len(df)):
        if pred["prediction"] == "buy" and df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            buy_x.append(df.index[i])
            buy_y.append(df["Close"].iloc[i])
        elif pred["prediction"] == "sell" and df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            sell_x.append(df.index[i])
            sell_y.append(df["Close"].iloc[i])

    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="Buy Signal",
                                 marker=dict(color="green", size=8, symbol="triangle-up")))
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="Sell Signal",
                                 marker=dict(color="red", size=8, symbol="triangle-down")))

    eq = back["equity_curve"]
    if not eq.empty:
        fig.add_trace(go.Scatter(x=eq.index, y=df["Close"].iloc[0] * (eq / eq.iloc[0]),
                                 mode="lines", name="Equity (Simulated)",
                                 line=dict(dash="dot", width=1.2, color="royalblue"), opacity=0.6))

    fig.update_layout(
        title=f"{asset} — {pred['prediction'].upper()} ({chosen} Risk)",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"""
    ### 📊 Strategy Summary
    **Prediction:** {pred['prediction']}  
    **Confidence:** {pred['probability'] * 100:.2f}%  
    **Accuracy:** {pred['accuracy'] * 100:.2f}%  
    **Historical Win Rate:** {back['winrate'] * 100:.2f}%  
    **Estimated Return:** {back['total_return'] * 100:.2f}%  
    **TP:** {pred['tp']:.4f} | **SL:** {pred['sl']:.4f}  
    **Candles Tested:** {len(df):,}  
    """)