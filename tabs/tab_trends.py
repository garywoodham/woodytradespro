import streamlit as st
import pandas as pd
import utils

def render_trends(interval, period, risk, show_help):
    st.title("Trends & Performance")
    if show_help:
        st.info("This tab compares performance across assets and shows current recommendation/TP/SL.")

    rows = []
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        df = utils.fetch_data(symbol, interval, period)
        if df is None or df.empty:
            continue
        change = float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)
        pred = utils.predict_next(df, risk=risk, news_headlines=[asset+" outlook"])
        rows.append({
            "Asset": asset,
            "Symbol": symbol,
            "Change (%)": change,
            "Signal": pred["signal"],
            "Prob (%)": round(pred["probability"]*100,1),
            "TP": round(pred["tp"], 2) if pred["tp"] else None,
            "SL": round(pred["sl"], 2) if pred["sl"] else None,
            "Sentiment": pred["sentiment"]
        })

    if not rows:
        st.warning("No data to show.")
        return

    perf = pd.DataFrame(rows).sort_values("Change (%)", ascending=False)
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.markdown("#### Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Assets", str(len(perf)))
    c2.metric("Avg Change", f"{perf['Change (%)'].mean():.2f}%")
    c3.metric("BUY Count", str((perf["Signal"]=="BUY").sum()))
    c4.metric("SELL Count", str((perf["Signal"]=="SELL").sum()))
