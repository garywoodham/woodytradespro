import streamlit as st
import pandas as pd
import numpy as np
import utils

def render_detailed():
    st.header("🔎 Detailed")
    st.caption("Pick an asset & timeframe. Candlesticks + ML + backtest + CSV export.")

    c1, c2 = st.columns(2)
    asset = c1.selectbox("Asset", list(utils.ASSET_SYMBOLS.keys()))
    timeframe = c2.selectbox("Timeframe", list(utils.INTERVALS.keys()), index=1)
    risk = st.sidebar.selectbox("Risk (TP/SL)", list(utils.RISK_MULT.keys()), index=1)

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, timeframe)
    if df.empty:
        st.warning(f"No data for {asset}.")
        return

    last = float(df["Close"].iloc[-1])
    change = float(df["Close"].pct_change().iloc[-1]*100) if len(df)>1 else 0.0
    m1,m2,m3 = st.columns(3)
    m1.metric("Last", f"{last:.2f}")
    m2.metric("Change", f"{change:.2f}%")
    m3.metric("Bars", len(df))

    X, clf, pred = utils.train_and_predict(df, timeframe, risk=risk)
    buys = X.index[X["Y"]==1] if X is not None and not X.empty else pd.Index([])
    sells= X.index[X["Y"]==-1] if X is not None and not X.empty else pd.Index([])

    fig = utils.make_candles(
        df, title=f"{asset} · {timeframe}",
        max_points=500,
        buys=buys, sells=sells,
        sl=pred.get("sl"), tp=pred.get("tp")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Signal", pred["signal"])
    c2.metric("Prob", f"{pred['prob']*100:.1f}%")
    c3.metric("TP", f"{pred['tp']:.2f}" if pred.get("tp") else "—")
    c4.metric("SL", f"{pred['sl']:.2f}" if pred.get("sl") else "—")

    # Backtest & export
    if X is not None and not X.empty:
        bt = utils.backtest_signals(X, "Y")
        st.subheader("Backtest (model-label baseline)")
        d1,d2,d3 = st.columns(3)
        d1.metric("Total return", f"{bt['total_return']*100:.2f}%")
        d2.metric("Win rate", f"{bt['winrate']*100:.1f}%")
        d3.metric("Trades", f"{bt['num_trades']}")

        csv = X.reset_index().rename(columns={"index":"Date"}).to_csv(index=False)
        st.download_button("⬇️ Export ML frame (CSV)", csv, file_name=f"{symbol}_{timeframe}_mlframe.csv", mime="text/csv")

    with st.expander("How to read this chart"):
        st.markdown("""
- **Candlesticks** show OHLC price.
- **Triangles** mark historical model labels (Buy ▲ / Sell ▼) for visual audit.
- **TP/SL lines** are derived from ATR × risk multiple (change risk in sidebar).
- **Prediction box** shows the current recommended action, probability, and TP/SL.
- **Backtest** is a quick baseline on the model’s label sequence (not a perfect sim).
""")