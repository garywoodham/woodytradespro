import streamlit as st
import pandas as pd
import utils

def render_scenarios(interval, period, risk, show_help):
    st.title("Scenarios / What-If")
    asset = st.selectbox("Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)
    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval, period)
    if df is None or df.empty:
        st.warning("No data available.")
        return

    st.caption("This runs a simple strategy (EMA cross + RSI filter) with your selected risk to show hypothetical trades and results.")
    res = utils.backtest(df, risk=risk)

    pred = utils.predict_next(df, risk=risk, news_headlines=[asset+" scenario"])
    st.plotly_chart(utils.plot_candles(df.tail(500), title=f"{asset} · {interval}", signals_df=utils.generate_signals(df).tail(500), latest=pred), use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Return (%)", f"{res['total_return']:.2f}%")
    c2.metric("Win Rate", f"{res['win_rate']:.1f}%")
    c3.metric("Trades", f"{len(res['trades'])}")

    if res["trades"]:
        trade_df = pd.DataFrame(res["trades"])
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        csv = trade_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades CSV", csv, file_name=f"scenarios_{symbol}.csv", mime="text/csv")
