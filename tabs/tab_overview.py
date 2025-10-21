# ─────────────────────────────────────────────────────────────────────────────
# tab_overview.py — Dashboard summary tab for WoodyTrades Pro
# Shows per-asset summary: prediction, TP/SL, probability, model accuracy
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import utils

def render_overview():
    st.title("📈 Market Overview")
    st.caption("Live multi-asset predictions with machine learning and sentiment-enhanced indicators.")

    # Sidebar risk selection
    risk = st.sidebar.radio("Select Risk Level", list(utils.RISK_MULT.keys()), index=1)

    # Display loading message
    st.info("Fetching live market data and running model predictions...")

    # Summary table container
    results = []
    charts = []

    for asset, symbol in utils.ASSET_SYMBOLS.items():
        try:
            df = utils.fetch_data(symbol, "1h")
            if df.empty:
                st.warning(f"No data available for {asset}")
                continue

            X, clf, pred = utils.train_and_predict(df, "1h", risk=risk)
            if pred is None:
                st.warning(f"Model could not generate prediction for {asset}.")
                continue

            # Build summary
            results.append({
                "Asset": asset,
                "Signal": pred["signal"],
                "Probability": f"{pred['prob']*100:.2f}%",
                "Risk": pred["risk"],
                "TP": f"{pred['tp']:.2f}" if pred["tp"] else "—",
                "SL": f"{pred['sl']:.2f}" if pred["sl"] else "—",
                "Accuracy": f"{pred['accuracy']*100:.2f}%" if pred["accuracy"] else "—",
            })

            # Create candlestick chart for each asset
            fig = utils.make_candles(
                df,
                title=f"{asset} — {pred['signal']} | Prob: {pred['prob']*100:.2f}% | Acc: {pred['accuracy']*100:.1f}%",
                tp=pred["tp"], sl=pred["sl"]
            )
            charts.append((asset, fig))

        except Exception as e:
            st.error(f"⚠️ Error processing {asset}: {e}")

    # Display summary dataframe
    if results:
        df_summary = pd.DataFrame(results)
        df_summary = df_summary.sort_values(by="Asset").reset_index(drop=True)

        # Compute overall average accuracy
        valid_acc = [float(r["Accuracy"].replace('%','')) for r in results if r["Accuracy"] != "—"]
        avg_acc = sum(valid_acc)/len(valid_acc) if valid_acc else 0

        st.subheader("📊 Prediction Summary")
        st.dataframe(df_summary, use_container_width=True)
        st.success(f"📈 **Overall Model Accuracy:** {avg_acc:.2f}%")

        # Display each chart in expanders
        for asset, fig in charts:
            with st.expander(f"🕹 {asset} — Click to view chart"):
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No assets could be analyzed at this time. Please check your internet connection or try again later.")