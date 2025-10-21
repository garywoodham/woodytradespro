# WoodyTrades Pro (flat)

Dark-themed Streamlit dashboard for multi-asset analytics with:
- Candlesticks + historical ML labels
- BUY/SELL/HOLD predictions with probability
- TP/SL via ATR × risk multiple (Low/Medium/High)
- Multi-timeframe (15m/1h/1d)
- Backtest baseline & Scenario simulation
- Optional news sentiment (VADER on Yahoo Finance titles)
- CSV exports

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py