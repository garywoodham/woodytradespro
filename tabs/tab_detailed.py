import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils

def render_detailed():
    st.title("📈 Detailed Asset Analysis — AI + Performance Simulation")
    st.caption("Combines AI predictions, trading methodology overlays, and simulated trade equity performance.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()), index=1)
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()), index=1)

    symbol = utils.ASSET_SYMBOLS[asset]
    st.info(f"Fetching detailed data for **{asset}**…")
    df = utils.fetch_data(symbol, interval)
    if df.empty:
        st.warning("No data found for this asset.")
        return

    # --- Indicators ---
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = utils.compute_rsi(df["Close"], 14)
    df["ATR"] = utils.compute_atr(df, 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = utils.compute_macd(df["Close"])

    # --- Predictions ---
    X, clf, pred = utils.train_and_predict(df, interval, risk)
    if X is None or pred is None:
        st.warning("Prediction unavailable for this timeframe.")
        return

    signal = pred["signal"]
    prob = pred["prob"]
    acc = pred["accuracy"]
    risk_label = pred["risk"]

    st.subheader("🎯 Model Prediction Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Signal", signal)
    c2.metric("Probability", f"{prob*100:.2f}%")
    c3.metric("Accuracy", f"{acc*100:.2f}%")
    c4.metric("Risk Level", risk_label)

    # --- Entry/Exit logic ---
    entries, exits, confirmed_buys, confirmed_sells = [], [], [], []
    for i in range(2, len(df)):
        ma_cross_up = df["MA50"].iloc[i] > df["MA200"].iloc[i] and df["MA50"].iloc[i-1] <= df["MA200"].iloc[i-1]
        ma_cross_dn = df["MA50"].iloc[i] < df["MA200"].iloc[i] and df["MA50"].iloc[i-1] >= df["MA200"].iloc[i-1]
        rsi = df["RSI"].iloc[i]
        macd_up = df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]
        macd_dn = df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]

        if ma_cross_up and rsi < 70 and macd_up:
            entries.append(df.index[i])
        elif ma_cross_dn and rsi > 30 and macd_dn:
            exits.append(df.index[i])

    # --- AI confirmation overlay ---
    if signal == "BUY":
        confirmed_buys = [d for d in entries[-5:] if prob > 0.55]
    elif signal == "SELL":
        confirmed_sells = [d for d in exits[-5:] if prob > 0.55]

    # --- Simulated performance ---
    trades = []
    trade_dates = []
    equity = [10000]  # Starting balance $10k

    for buy in confirmed_buys:
        sell_candidates = [e for e in exits if e > buy]
        if not sell_candidates:
            continue
        sell = sell_candidates[0]
        buy_price = df.loc[buy, "Close"]
        sell_price = df.loc[sell, "Close"]
        ret = (sell_price - buy_price) / buy_price
        trades.append(ret)
        trade_dates.append(sell)
        equity.append(equity[-1] * (1 + ret))

    total_return = (equity[-1] - equity[0]) / equity[0] * 100 if len(equity) > 1 else 0
    win_rate = np.mean([t > 0 for t in trades]) * 100 if trades else 0
    profit_factor = (
        np.sum([t for t in trades if t > 0]) / abs(np.sum([t for t in trades if t < 0])) if any(t < 0 for t in trades) else np.inf
    )
    ai_score = (acc * 100 * (win_rate / 100) * (profit_factor if profit_factor < 10 else 10)) / 10

    st.subheader("📊 Simulated AI Trade Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{total_return:.2f}%")
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Profit Factor", f"{profit_factor:.2f}")
    c4.metric("AI Trade Score", f"{ai_score:.1f}")

    # --- Main Candlestick Chart ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], line=dict(color="orange", width=1.5), name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], line=dict(color="purple", width=1.5), name="MA200"))

    fig.add_trace(go.Scatter(
        x=entries, y=df.loc[entries, "Close"], mode="markers",
        marker=dict(color="lime", size=9, symbol="triangle-up"), name="Tech Buy"
    ))
    fig.add_trace(go.Scatter(
        x=exits, y=df.loc[exits, "Close"], mode="markers",
        marker=dict(color="red", size=9, symbol="triangle-down"), name="Tech Sell"
    ))

    if confirmed_buys:
        fig.add_trace(go.Scatter(
            x=confirmed_buys, y=df.loc[confirmed_buys, "Close"], mode="markers+text",
            marker=dict(color="deepskyblue", size=13, symbol="star"),
            text=["AI Confirmed BUY"] * len(confirmed_buys),
            textposition="top center", name="AI Confirmed BUY"
        ))
    if confirmed_sells:
        fig.add_trace(go.Scatter(
            x=confirmed_sells, y=df.loc[confirmed_sells, "Close"], mode="markers+text",
            marker=dict(color="magenta", size=13, symbol="star"),
            text=["AI Confirmed SELL"] * len(confirmed_sells),
            textposition="bottom center", name="AI Confirmed SELL"
        ))

    fig.update_layout(
        title=f"{asset} — AI + Methodology Overlay & Equity Simulation",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_dark", height=650,
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        font=dict(color="#e6e6e6"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Equity Curve ---
    if len(equity) > 1:
        st.subheader("💰 Equity Curve (Cumulative Portfolio Value)")
        eq_df = pd.DataFrame({"Date": trade_dates, "Equity": equity[1:]})
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=eq_df["Date"], y=eq_df["Equity"],
                                    line=dict(color="deepskyblue", width=2),
                                    name="Equity Curve"))
        eq_fig.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
            font=dict(color="#e6e6e6"),
            xaxis_title="Trade Date", yaxis_title="Equity ($)"
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    # --- Indicators Below ---
    with st.expander("📊 Technical Indicators"):
        st.line_chart(df[["RSI"]], height=150)
        st.line_chart(df[["MACD", "MACD_Signal"]], height=150)

    st.caption(
        "🧩 *Blue stars = AI Confirmed BUY; Magenta stars = AI Confirmed SELL.*\n"
        "💰 *Equity Curve shows cumulative value from confirmed AI trades.*\n"
        "AI Trade Score combines accuracy, win rate, and profit factor."
    )