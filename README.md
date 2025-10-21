# WoodyTrades Pro (Live Only, Dark Mode)

Live Streamlit dashboard for multi-asset analytics (Gold, US100, S&P500, etc.).

- Live prices via Yahoo Finance
- Heuristic signals (EMA crossover + RSI) + free news sentiment
- Risk-aware TP/SL and backtests
- Candlestick with BUY/SELL markers
- CSV export
- Works on Streamlit Cloud (Python 3.13, no TensorFlow)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
