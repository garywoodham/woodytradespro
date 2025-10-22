import streamlit as st
import utils
import pandas as pd
import plotly.graph_objects as go

def render_scenarios():
    st.title("🎯 Scenario Analysis")
    st.caption(
        "Simulate and compare trade performance across different **risk levels**, "
        "including Take Profit (TP), Stop Loss (SL), Accuracy, and Win Rate."
    )

    # --- User input ---
    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()))

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval=interval)

    if df.empty:
        st.warning(f"No data available for {asset}")
        return

    st.divider()
    st.info("Running backtests across all risk levels... please wait ⏳")

    scenario_results = []

    for risk in utils.RISK_MULT.keys():
        try:
            pred = utils.train_and_predict(df, horizon=interval, risk=risk)
            bt = utils.backtest_signals(df, pred)  # backtest for winrate and performance

            if not pred or bt is None:
                continue

            scenario_results.append({
                "Risk": risk,
                "Prediction": pred["prediction"],
                "Confidence": f"{pred['probability']*100:.2f}%",
                "Accuracy": f"{pred['accuracy']*100:.2f}%",
                "Win Rate": f"{bt.get('winrate', 0)*100:.2f}%",
                "TP": round(pred["tp"], 2),
                "SL": round(pred["sl"], 2),
                "Total Return": f"{bt.get('total_return', 0)*100:.2f}%"
            })
        except Exception as e:
            st.warning(f"⚠️ Error processing {risk} risk: {e}")

    if not scenario_results:
        st.error("Unable to simulate scenarios for this asset.")
        return

    # --- Dataframe view ---
    scenario_df = pd.DataFrame(scenario_results)
    st.subheader(f"📊 {asset} Scenario Outcomes")
    st.dataframe(scenario_df, width="stretch")

    # --- Visualization: TP/SL comparison ---
    fig_tp_sl = go.Figure()
    for s in scenario_results:
        fig_tp_sl.add_trace(go.Bar(
            x=["Take Profit", "Stop Loss"],
            y=[s["TP"], s["SL"]],
            name=f"{s['Risk']} Risk"
        ))
    fig_tp_sl.update_layout(
        title=f"{asset} — TP/SL Levels by Risk",
        barmode="group",
        height=450,
        xaxis_title="Level Type",
        yaxis_title="Price"
    )
    st.plotly_chart(fig_tp_sl, width="stretch", config={"displayModeBar": False})

    # --- Visualization: Accuracy vs Win Rate ---
    fig_acc_wr = go.Figure()
    fig_acc_wr.add_trace(go.Bar(
        x=[s["Risk"] for s in scenario_results],
        y=[float(s["Accuracy"].replace('%', '')) for s in scenario_results],
        name="Accuracy %",
        marker_color="lightgreen"
    ))
    fig_acc_wr.add_trace(go.Bar(
        x=[s["Risk"] for s in scenario_results],
        y=[float(s["Win Rate"].replace('%', '')) for s in scenario_results],
        name="Win Rate %",
        marker_color="steelblue"
    ))
    fig_acc_wr.update_layout(
        title=f"{asset} — Accuracy vs Win Rate by Risk",
        barmode="group",
        height=450,
        xaxis_title="Risk Level",
        yaxis_title="Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_acc_wr, width="stretch", config={"displayModeBar": False})

    st.caption(
        "📘 *Accuracy* measures prediction correctness, while *Win Rate* reflects successful trade outcomes "
        "from the backtest simulation."
    )
    import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import utils

# Assuming you already have `df` (asset data) and `pred` (model prediction)

if df is not None and not df.empty and pred:
    st.subheader("📊 Model Trade Visualization")

    # Recompute or reuse backtest results
    backtest = utils.backtest_signals(df, pred)
    equity = backtest["equity_curve"]
    winrate = backtest["winrate"]
    total_return = backtest["total_return"]

    # Build base price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Price",
        line=dict(width=1.8)
    ))

    # Find Buy/Sell points for visual markers
    signals = []
    for i in range(1, len(df)):
        if pred["prediction"].lower() == "buy" and df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            signals.append((df.index[i], df["Close"].iloc[i], "Buy"))
        elif pred["prediction"].lower() == "sell" and df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
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

    # Optional: overlay equity curve
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=df["Close"].iloc[0] * (1 + equity - 1),  # scaled equity overlay
        mode="lines",
        name="Equity (simulated)",
        line=dict(dash="dot", width=1.3, color="royalblue"),
        opacity=0.5
    ))

    fig.update_layout(
        title=f"{asset_name} — {pred['prediction']} Recommendation Chart",
        yaxis_title="Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width="stretch")

    # Summarize impact if followed
    st.markdown(f"""
    **📈 Trade Simulation Summary**
    - **Prediction:** {pred['prediction']}
    - **Model Confidence:** {pred['probability'] * 100:.2f}%
    - **Historical Win Rate:** {winrate * 100:.2f}%
    - **Estimated Total Return:** {total_return * 100:.2f}%
    - **Number of Trades Simulated:** {len(df)}

    _If this strategy had been followed historically, 
    the equity curve above shows the growth trajectory relative to price._ ⚙️
    """)
else:
    st.warning("No data available to visualize trades.")
    
    
    
    
    