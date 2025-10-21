import streamlit as st

def render_help():
    st.title("ℹ️ Help & Interpretation Guide")

    st.markdown("""
    ### Understanding the Dashboard
    - **Signal** → Model recommendation (`BUY`, `SELL`, or `HOLD`)
    - **Probability** → Model confidence for its signal
    - **TP / SL** → Suggested take-profit and stop-loss based on volatility and selected risk
    - **Accuracy** → Historical validation accuracy of the model on similar data
    - **Trend (%)** → 10-period price momentum percentage

    ### Chart Annotations
    - **Candles:** OHLC price structure
    - **Green triangle:** Buy signal
    - **Red triangle:** Sell signal
    - **Dotted lines:** Target price and stop loss zones

    ### Tips for Use
    - Always consider multiple timeframes before trading.
    - High-risk mode widens TP/SL for volatile assets.
    - Sentiment data comes from Yahoo Finance news when available.

    ### Example Workflow
    1. Go to **Overview** → get quick recommendations.
    2. Visit **Trends** → confirm trend alignment.
    3. Dive into **Detailed** → view signals and price structure.
    """)