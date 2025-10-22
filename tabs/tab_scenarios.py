import streamlit as st
import utils
import pandas as pd
import plotly.graph_objects as go

def render_scenarios():
    st.title("🎯 Scenario Analysis")
    st.caption("Simulate outcomes across Low, Medium, and High risk settings to optimize position sizing and exits.")

    asset = st.selectbox("Select Asset", list(utils.ASSET_SYMBOLS.keys()))
    interval = st.selectbox("Select Interval", list(utils.INTERVALS.keys()))

    symbol = utils.ASSET_SYMBOLS[asset]
    df = utils.fetch_data(symbol, interval=interval)

    if df.empty:
        st.warning(f"No data available for {asset}")
        return

    scenario_results = []
    for risk in utils.RISK_MULT.keys():
        pred = utils.train_and_predict(df, horizon=interval, risk=risk)
        if not pred:
            continue
        scenario_results.append({
            "Risk": risk,
            "Prediction": pred["prediction"],
            "Confidence": f"{pred['probability']*100:.2f}%",
            "Accuracy": f"{pred['accuracy']*100:.2f}%",
            "TP": round(pred["tp"], 2),
            "SL": round(pred["sl"], 2)
        })

    if not scenario_results:
        st.warning("Unable to simulate scenarios for this asset.")
        return

    scenario_df = pd.DataFrame(scenario_results)
    st.subheader(f"📊 {asset} Scenario Outcomes")
    st.dataframe(scenario_df, width="stretch")

    # Chart comparison
    fig = go.Figure()
    for scenario in scenario_results:
        fig.add_trace(go.Bar(
            x=["Take Profit", "Stop Loss"],
            y=[scenario["TP"], scenario["SL"]],
            name=f"{scenario['Risk']} Risk"
        ))

    fig.update_layout(
        title=f"{asset} TP/SL Comparison by Risk Level",
        barmode="group",
        height=500,
        xaxis_title="Level Type",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.caption("Use this view to visually compare how different risk multipliers affect trade target levels.")