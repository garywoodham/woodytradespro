import streamlit as st
import utils

def render_detailed():
    st.title("🔍 Detailed Asset Analysis")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()), index=1)
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()), index=1)

    symbol = utils.ASSET_SYMBOLS[asset]
    st.info(f"Fetching and analyzing {asset} data...")

    try:
        df = utils.fetch_data(symbol, interval)
        if df.empty:
            st.warning(f"No data found for {asset}.")
            return

        X, clf, pred = utils.train_and_predict(df, interval, risk)
        if pred is None:
            st.warning(f"Model could not generate a prediction for {asset}.")
            return

        st.subheader(f"📈 {asset} — {pred['signal']} Signal")
        st.markdown(f"""
        **Probability:** {pred['prob']*100:.2f}%  
        **Risk:** {pred['risk']}  
        **TP:** {pred['tp']:.2f if pred['tp'] else '—'}  
        **SL:** {pred['sl']:.2f if pred['sl'] else '—'}  
        **Model Accuracy:** {pred['accuracy']*100:.2f if pred['accuracy'] else 0:.2f}%  
        """)

        fig = utils.make_candles(df, f"{asset} Detailed Chart", tp=pred["tp"], sl=pred["sl"])
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in detailed analysis: {e}")