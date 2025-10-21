import streamlit as st
import pandas as pd
import numpy as np
import utils

def render_scenarios():
    st.header("🧪 Scenarios")
    st.caption("Simulate threshold-based entries and see hypothetical returns. Export CSV.")

    asset = st.selectbox("Asset", list(utils.ASSET_SYMBOLS.keys()))
    timeframe = st.selectbox("Timeframe", list(utils.INTERVALS.keys()), index=1)
    prob_th = st.slider("Min probability to act", 0.0, 1.0, 0.6, 0.01)
    risk = st.sidebar.selectbox("Risk (TP/SL)", list(utils.RISK_MULT.keys()), index=1)

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, timeframe)
    if df.empty:
        st.warning("No data loaded.")
        return

    # Build ML frame and fit once
    X, clf, pred = utils.train_and_predict(df, timeframe, risk=risk)
    if X is None or X.empty:
        st.warning("Not enough data to simulate.")
        return

    # For each bar, get probability of BUY/SELL and decide action if prob >= threshold
    proba = []
    classes = clf.classes_.tolist()
    for i in range(len(X)):
        p = clf.predict_proba(X.iloc[[i]][utils.FEATURES])[0]
        p_buy = p[classes.index(1)] if 1 in classes else 0.0
        p_sell= p[classes.index(-1)] if -1 in classes else 0.0
        proba.append((p_buy, p_sell))
    X["P_BUY"], X["P_SELL"] = zip(*proba)

    def decide(row):
        if row["P_BUY"] >= prob_th and row["P_BUY"] > row["P_SELL"]:
            return 1
        if row["P_SELL"] >= prob_th and row["P_SELL"] > row["P_BUY"]:
            return -1
        return 0

    X["Action"] = X.apply(decide, axis=1)
    X["NextRet"] = X["Close"].pct_change().shift(-1).fillna(0)
    X["StratRet"] = X["NextRet"] * X["Action"].shift().fillna(0)
    eq = (1 + X["StratRet"]).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    trades = int((X["Action"].diff().abs() > 0).sum())
    winrate = float((X["StratRet"] > 0).sum() / max(1,(X["StratRet"] != 0).sum()))

    c1,c2,c3 = st.columns(3)
    c1.metric("Total return", f"{total*100:.2f}%")
    c2.metric("Win rate", f"{winrate*100:.1f}%")
    c3.metric("Trades", f"{trades}")

    st.line_chart(eq.rename("Equity"))

    csv = X.reset_index().rename(columns={"index":"Date"}).to_csv(index=False)
    st.download_button("⬇️ Export Scenario (CSV)", csv, file_name=f"{symbol}_{timeframe}_scenario.csv", mime="text/csv")

    with st.expander("Notes"):
        st.markdown("""
- **Action** triggers only when the model's **probability ≥ threshold**.
- This is a simplified simulation (no fees/slippage).
- Use the **Detailed** tab for TP/SL visualization on candles.
""")