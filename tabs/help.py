import streamlit as st

def render_help():
    st.header("❓ Help")
    st.markdown("""
**What’s included (free & Streamlit-safe):**
- Technical indicators (RSI, MACD, EMA, Bollinger width, ATR, Volatility)
- RandomForest ML classification for BUY/SELL/HOLD
- Optional news sentiment (Yahoo news via yfinance, VADER analyzer)
- Multi-timeframe (15m / 1h / 1d)
- Candlesticks with historical labels, probability, TP/SL
- Backtest baseline and scenario simulator
- CSV export

**Tips**
- If charts look truncated, we cap to 500 bars for speed. Zoom with Plotly tools.
- Change **Risk** in the sidebar to tighten/loosen TP/SL.
- On Streamlit Cloud: no TensorFlow is used (for compatibility).

**Disclaimers**
- This is not financial advice. Backtests are illustrative only.
- Data and news may be delayed; always verify before trading.
""")