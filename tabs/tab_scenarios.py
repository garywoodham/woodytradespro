import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils


def render_scenarios():
    st.title("🧠 Strategy Scenarios & Backtesting")
    st.caption("Simulate and evaluate trading signals with AI accuracy and strategy win rate.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()), index=1)
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()), index=1)

    symbol = utils.ASSET_SYMBOLS[asset]

    st.info(f"Fetching data and running scenario simulation for **{asset}**...")
    df = utils.fetch_data(symbol, interval)
    if df.empty:
        st.warning(f"No data available for {asset}.")
        return

    X, clf, pred = utils.train_and_predict(df, interval, risk)
    if X is None or clf is None or X.empty or pred is None:
        st.warning("Unable to build scenario simulation. Try another interval or asset.")
        return

    # Clean and clip
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X_clean[utils.FEATURES] = X_clean[utils.FEATURES].clip(-1e6, 1e6)

    preds, probs, timestamps = [], [], []
    for i in range(-min(200, len(X_clean)), 0):
        try:
            row = X_clean.iloc[[i]][utils.FEATURES]
            p = clf.predict_proba(row)[0]
            classes = clf.classes_.tolist()
            p_buy = p[classes.index(1)] if 1 in classes else 0
            p_sell = p[classes.index(-1)] if -1 in classes else 0
            if p_buy > max(0.5, p_sell + 0.1):
                signal = "BUY"
                prob = p_buy
            elif p_sell > max(0.5, p_buy + 0.1):
                signal = "SELL"
                prob = p_sell
            else:
                signal = "HOLD"
                prob = max(p_buy, p_sell)
            preds.append(signal)
            probs.append(prob)
            timestamps.append(X_clean.index[i])
        except Exception:
            continue

    hist_df = pd.DataFrame({"Timestamp": timestamps, "Signal": preds, "Prob": probs})
    hist_df["Prob (%)"] = (hist_df["Prob"] * 100).round(2)
    hist_df = hist_df.drop(columns=["Prob"])

    st.subheader(f"📊 Historical Scenario Predictions — {asset}")
    st.dataframe(
        hist_df.tail(100).sort_values(by="Timestamp", ascending=False),
        use_container_width=True,
    )

    # Run backtest and show results
    bt = utils.backtest_signals(X_clean)
    equity = bt["equity_curve"]
    winrate = bt.get("winrate", 0)
    total_return = bt.get("total_return", 0)
    num_trades = bt.get("num_trades", 0)
    accuracy = pred.get("accuracy", 0)

    st.subheader("💰 Strategy Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Win Rate", f"{winrate*100:.2f}%")
    col3.metric("Total Return", f"{total_return*100:.2f}%")
    col4.metric("Trades Executed", num_trades)

    # Interpretive insights
    st.markdown("---")
    if winrate > accuracy:
        st.success("🎯 Strategy execution is outperforming model predictions — TP/SL rules are effective!")
    elif accuracy > winrate:
        st.warning("⚠️ Model predictions are strong, but strategy execution (TP/SL) might be too tight.")
    else:
        st.info("Model and trading strategy are performing equally — consider tuning risk levels.")

    # Plot equity curve
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            line=dict(width=2, color="#00cc96"),
            name="Equity Curve",
        )
    )
    fig.update_layout(
        title="📈 Equity Curve (Cumulative Strategy Performance)",
        xaxis_title="Time",
        yaxis_title="Portfolio Value",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#e6e6e6"),
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "🧩 *Model Accuracy measures how often signals were correct; "
        "Win Rate measures how many trades were profitable.*"
    )