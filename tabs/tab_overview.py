import streamlit as st
import pandas as pd
import numpy as np
import utils

def render_overview():
    st.header("📊 Overview")
    st.caption("Live prices • ML signal • Probability • Overall model accuracy")

    risk = st.sidebar.selectbox("Risk level (for TP/SL)", list(utils.RISK_MULT.keys()), index=1)

    rows = []
    accs = []
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        df = utils.fetch_data(symbol, "1h")
        X, clf, pred = utils.train_and_predict(df, "1h", risk=risk)

        last_close = float(df["Close"].iloc[-1]) if not df.empty else np.nan
        chg = float(df["Close"].pct_change().iloc[-1]*100) if len(df)>1 else 0.0
        rows.append({
            "Asset": asset,
            "Last": last_close,
            "Change (%)": chg,
            "Signal": pred["signal"],
            "Prob": pred["prob"],
            "TP": utils.guard_float(pred["tp"]),
            "SL": utils.guard_float(pred["sl"]),
        })
        if pred["accuracy"] is not None:
            accs.append(pred["accuracy"])

    df_sum = pd.DataFrame(rows)
    st.dataframe(
        df_sum.style.format({"Last":"{:.2f}","Change (%)":"{:.2f}","Prob":"{:.2f}","TP":"{:.2f}","SL":"{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    overall_acc = float(np.mean(accs)) if accs else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked assets", len(utils.ASSET_SYMBOLS))
    c2.metric("Overall model accuracy (CV)", f"{overall_acc*100:.1f}%" if overall_acc else "—")
    c3.metric("Theme", "Dark")

    st.markdown("—")
    st.caption("Tip: Switch risk in sidebar to adjust TP/SL multiples.")