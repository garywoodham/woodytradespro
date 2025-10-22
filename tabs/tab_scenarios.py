import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils
import numpy as np
import time

# ───────────────────────────────
# Render Scenarios Tab
# ───────────────────────────────
def render_scenarios():
    st.title("📊 Trade Scenarios & Model Simulation")
    st.caption("Simulate trading outcomes for each asset under different risk levels.")

    # Select asset and interval
    asset_name = st.selectbox("Choose Asset", list(utils.ASSET_SYMBOLS.keys()), index=0)
    interval_choice = st.selectbox("Select Timeframe", ["15m", "1h", "1d"], index=1)
    asset_symbol = utils.ASSET_SYMBOLS[asset_name]
    risk_levels = list(utils.RISK_MULT.keys())

    st.info(f"Fetching and analyzing **{asset_name} ({asset_symbol})**... please wait ⏳")
    df = utils.fetch_data(asset_symbol, utils.INTERVALS[interval_choice]["interval"], utils.INTERVALS[interval_choice]["period"])

    if df.empty:
        st.error(f"No data available for {asset_name}. Please try again later.")
        return

    st.success(f"Data loaded successfully for {asset_name} ({len(df)} candles).")

    # ───────────────────────────────
    # Run model predictions for each risk level
    # ───────────────────────────────
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, risk in enumerate(risk_levels, 1):
        status.markdown(f"🔍 Evaluating **{risk} Risk Strategy** ({i}/{len(risk_levels)})...")
        progress.progress(i / len(risk_levels))

        try:
            pred = utils.train_and_predict(df, horizon=interval_choice, risk=risk)
            backtest = utils.backtest_signals(df, pred)

            if pred:
                results.append({
                    "Risk": risk,
                    "Prediction": pred["prediction"],
                    "Confidence": round(pred["probability"] * 100, 2),
                    "Accuracy": round(pred["accuracy"] * 100, 2),
                    "Win Rate": round(backtest["winrate"] * 100, 2),
                    "Total Return": round(backtest["total_return"] * 100, 2),
                    "TP": round(pred["tp"], 4),
                    "SL": round(pred["sl"], 4)
                })
        except Exception as e:
            st.warning(f"⚠️ Error processing {risk} risk: {e}")

        time.sleep(0.8)

    progress.progress(1.0)
    status.markdown("✅ Completed scenario simulations.")

    if not results:
        st.error("Unable to simulate scenarios for this asset.")
        return

    results_df = pd.DataFrame(results)
    st.subheader("📈 Scenario Comparison")
    st.dataframe(results_df, use_container_width=True)

    # ───────────────────────────────
    # Interactive Trade Visualization for Current Risk (default Medium)
    # ───────────────────────────────
    st.divider()
    st.subheader("🎯 Model Trade Visualization")

    selected_risk = st.radio("Select Risk Level to Visualize", risk_levels, index=1)
    selected_pred = utils.train_and_predict(df, horizon=interval_choice, risk=selected_risk)
    selected_backtest = utils.backtest_signals(df, selected_pred)

    if selected_pred and not df.empty:
        # Build price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(width=1.8, color="black")
        ))

        # Identify Buy/Sell points
        signals = []
        for i in range(1, len(df)):
            if selected_pred["prediction"].lower() == "buy" and df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                signals.append((df.index[i], df["Close"].iloc[i], "Buy"))
            elif selected_pred["prediction"].lower() == "sell" and df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                signals.append((df.index[i], df["Close"].iloc[i], "Sell"))

        if signals:
            buy_points = [s for s in signals if s[2] == "Buy"]
            sell_points = [s for s in signals if s[2] == "Sell"]

            if buy_points:
                fig.add_trace(go.Scatter(
                    x=[s[0] for s in buy_points],
                    y=[s[1] for s in buy_points],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(color="green", size=8, symbol="triangle-up")
                ))

            if sell_points:
                fig.add_trace(go.Scatter(
                    x=[s[0] for s in sell_points],
                    y=[s[1] for s in sell_points],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(color="red", size=8, symbol="triangle-down")
                ))

        # Add equity curve overlay
        equity = selected_backtest["equity_curve"]
        if not equity.empty:
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=df["Close"].iloc[0] * (1 + equity - 1),
                mode="lines",
                name="Equity (Simulated)",
                line=dict(dash="dot", width=1.2, color="royalblue"),
                opacity=0.6
            ))

        # Layout updates
        fig.update_layout(
            title=f"{asset_name} — {selected_pred['prediction']} Signal Overview ({selected_risk} Risk)",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, width="stretch")

        # ───────────────────────────────
        # Summary of Impact
        # ───────────────────────────────
        winrate = selected_backtest["winrate"] * 100
        total_return = selected_backtest["total_return"] * 100
        confidence = selected_pred["probability"] * 100

        st.markdown(f"""
        ### 📊 Strategy Summary
        **Prediction:** {selected_pred['prediction']}  
        **Model Confidence:** {confidence:.2f}%  
        **Historical Win Rate:** {winrate:.2f}%  
        **Estimated Total Return:** {total_return:.2f}%  
        **TP:** {selected_pred['tp']:.4f} | **SL:** {selected_pred['sl']:.4f}  
        **Data Points Used:** {len(df):,}  

        _If this model's recommendations were followed historically, the simulated equity curve (dotted blue line) shows the performance relative to price._
        """)
    else:
        st.warning("No sufficient data available to visualize model predictions.")