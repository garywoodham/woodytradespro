import streamlit as st
import pandas as pd
import utils

def render_trends():
    st.title("📊 Market Trends")
    st.caption("Trend strength, sentiment direction, and trading opportunities based on recent model predictions.")

    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()), index=1)
    interval = st.sidebar.selectbox("Select Interval", list(utils.INTERVALS.keys()), index=1)

    trend_data = []
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        try:
            df = utils.fetch_data(symbol, interval)
            if df.empty:
                continue

            X, clf, pred = utils.train_and_predict(df, interval, risk)
            if pred is None:
                continue

            trend = df["Close"].pct_change(10).iloc[-1] * 100
            trend_data.append({
                "Asset": asset,
                "Signal": pred["signal"],
                "Trend (%)": f"{trend:.2f}",
                "Probability": f"{pred['prob']*100:.2f}%",
                "Risk": pred["risk"],
                "TP": f"{pred['tp']:.2f}" if pred["tp"] else "—",
                "SL": f"{pred['sl']:.2f}" if pred["sl"] else "—",
                "Accuracy": f"{pred['accuracy']*100:.2f}%" if pred["accuracy"] else "—"
            })
        except Exception as e:
            st.error(f"Error for {asset}: {e}")

    if not trend_data:
        st.warning("No trend data available right now.")
        return

    df_trend = pd.DataFrame(trend_data)
    st.dataframe(df_trend.style
        .background_gradient(subset=["Trend (%)"], cmap="RdYlGn")
        .apply(lambda x: ["background-color:#00ff80" if v == "BUY"
                          else "background-color:#ff6666" if v == "SELL"
                          else "" for v in x], subset=["Signal"]),
        use_container_width=True)

    st.info("✅ Tip: Green indicates strong positive trends; red shows weakening momentum.")