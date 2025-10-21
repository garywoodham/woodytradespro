import streamlit as st
import pandas as pd
import utils

def render_detailed(interval, period, risk, show_help):
    st.title("Detailed View")
    asset = st.selectbox("Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)
    symbol = utils.ASSET_SYMBOLS[asset]
    if show_help:
        st.info("Use the selectors to choose timeframe and asset. Candlestick shows BUY/SELL markers and TP/SL lines.")

    df = utils.fetch_data(symbol, interval, period)
    if df is None or df.empty:
        st.warning("No data available.")
        return

    last = float(df['Close'].iloc[-1])
    delta_pct = float(df['Close'].pct_change().iloc[-1] * 100)
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close", f"{last:.2f}")
    col2.metric("Δ %", f"{delta_pct:.2f}%")
    col3.metric("Bars", f"{len(df):,}")

    headlines = [f"{asset} technical setup improves"]
    pred = utils.predict_next(df, risk=risk, news_headlines=headlines)
    sigs = utils.generate_signals(df)
    st.subheader(f"{asset} · {interval}")
    st.plotly_chart(utils.plot_candles(df.tail(500), title=f"{asset} · {interval}", signals_df=sigs.tail(500), latest=pred), use_container_width=True)

    st.markdown("#### Backtest (EMA x RSI)")
    result = utils.backtest(df, risk=risk)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Return (%)", f"{result['total_return']:.2f}%")
    c2.metric("Win Rate", f"{result['win_rate']:.1f}%")
    c3.metric("Trades", f"{len(result['trades'])}")

    if result["trades"]:
        trade_df = pd.DataFrame(result["trades"])
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        csv = trade_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades CSV", csv, file_name=f"backtest_{symbol}.csv", mime="text/csv")
    else:
        st.info("No completed trades in window.")
