import streamlit as st
import pandas as pd
import utils

def render_overview(interval, period, risk, show_help):
    st.title("Overview")
    if show_help:
        st.info("This tab summarizes all assets with live prediction, probability, and mini charts.")

    cols = st.columns(3)
    i = 0
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        df = utils.fetch_data(symbol, interval, period)
        with cols[i % 3]:
            st.markdown(f"### {asset}")
            if df is None or df.empty:
                st.warning("No data.")
                i += 1
                continue

            last_close = float(df["Close"].iloc[-1])
            change_pct = float(df["Close"].pct_change().iloc[-1] * 100)
            headlines = [f"{asset} sees momentum as {symbol} traders eye breakout"]
            pred = utils.predict_next(df, risk=risk, news_headlines=headlines)

            c1,c2,c3 = st.columns(3)
            c1.metric("Last", f"{last_close:.2f}")
            c2.metric("Δ %", f"{change_pct:.2f}%")
            c3.metric("Signal", pred["signal"], delta=f"{pred['probability']*100:.0f}% {pred['sentiment']}")

            sigs = utils.generate_signals(df)
            fig = utils.plot_candles(df.tail(300), title=f"{asset} · {interval}", signals_df=sigs.tail(300), latest=pred)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"TP: **{pred['tp']:.2f}** • SL: **{pred['sl']:.2f}** • RSI: {pred['rsi']:.1f} • EMA12/26: {pred['ema12']:.2f}/{pred['ema26']:.2f}")
        i += 1
