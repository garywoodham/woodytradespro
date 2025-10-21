import streamlit as st
import pandas as pd
import numpy as np
import utils

def render_trends():
    st.header("📈 Trends")
    st.caption("Multi-asset comparison by timeframe, with ML prediction & TP/SL.")

    risk = st.sidebar.selectbox("Risk level (for TP/SL)", list(utils.RISK_MULT.keys()), index=1)
    timeframe = st.selectbox("Select timeframe", list(utils.INTERVALS.keys()), index=1)

    rows = []
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        df = utils.fetch_data(symbol, timeframe)
        if df.empty:
            rows.append({"Asset":asset,"Change (%)":np.nan,"Signal":"—","Prob":np.nan,"TP":np.nan,"SL":np.nan})
            continue

        chg = float((df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100) if len(df)>1 else 0.0
        X, clf, pred = utils.train_and_predict(df, timeframe, risk=risk)
        rows.append({
            "Asset": asset,
            "Change (%)": chg,
            "Signal": pred["signal"],
            "Prob": pred["prob"],
            "TP": utils.guard_float(pred["tp"]),
            "SL": utils.guard_float(pred["sl"])
        })

    perf = pd.DataFrame(rows)
    st.dataframe(
        perf.style.format({"Change (%)":"{:.2f}","Prob":"{:.2f}","TP":"{:.2f}","SL":"{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("—")
    asset = st.selectbox("Show chart for", list(utils.ASSET_SYMBOLS.keys()))
    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, timeframe)
    X, clf, pred = utils.train_and_predict(df, timeframe, risk=risk)

    # Build simple buy/sell markers using model's classification label on full set
    buys, sells = pd.Index([]), pd.Index([])
    if X is not None and not X.empty and "Y" in X.columns:
        buys = X.index[X["Y"] == 1]
        sells = X.index[X["Y"] == -1]

    fig = utils.make_candles(
        df, title=f"{asset} · {timeframe}",
        max_points=500,
        buys=buys, sells=sells,
        sl=pred.get("sl"), tp=pred.get("tp")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"**Prediction:** {pred['signal']}  |  **Prob:** {pred['prob']*100:.1f}%  |  "
        f"**TP/SL:** {pred.get('tp') and f'{pred['tp']:.2f}'} / {pred.get('sl') and f'{pred['sl']:.2f}'}  |  "
        f"**Risk:** {pred['risk']}"
    )