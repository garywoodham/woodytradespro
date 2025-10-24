# utils.py — WoodyTradesPro Smart v2 (Final Stable + Flatten Fix)
# ---------------------------------------------------------------------------

import os
import time
import warnings
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Logging / Warning suppression
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Could not infer format")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ASSET_SYMBOLS = {
    "Gold": "GC=F",
    "NASDAQ 100": "^NDX",
    "S&P 500": "^GSPC",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
}

INTERVALS = {"15m": "15m", "1h": "60m", "4h": "4h", "1d": "1d", "1wk": "1wk"}
PERIODS = {"15m": "7d", "1h": "2mo", "4h": "6mo", "1d": "1y", "1wk": "5y"}

# ---------------------------------------------------------------------------
# Helper Logger
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Data Fetch + Flatten
# ---------------------------------------------------------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are 1D, flattening any array-like values."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for col in df.columns:
        vals = np.asarray(df[col]).squeeze()
        if vals.ndim != 1:
            vals = vals.reshape(-1)
        df[col] = vals
    return df

def fetch_data(symbol: str, interval_key: str = "1h", use_cache=True) -> pd.DataFrame:
    """Fetch price data safely with caching and flattened data."""
    interval = INTERVALS.get(interval_key, "60m")
    period = PERIODS.get(interval_key, "2mo")
    cache_path = os.path.join(DATA_DIR, f"{symbol.replace('=','_').replace('^','')}_{interval}.csv")

    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = _flatten_columns(df)
            if not df.empty:
                return df
        except Exception:
            pass

    _log(f"⏳ Fetching {symbol} [{interval}] for {period}...")
    for attempt in range(4):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            if not raw.empty and len(raw) > 50:
                raw = _flatten_columns(raw)
                raw.to_csv(cache_path)
                _log(f"✅ {symbol}: fetched {len(raw)} rows.")
                return raw
        except Exception as e:
            _log(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
            time.sleep(1 + attempt)

    _log(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, RSI, MACD, ATR indicators with forced 1-D flattening."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # 🔧 Force-flatten all numeric columns before indicator computation
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            vals = np.asarray(df[col]).squeeze()
            if vals.ndim != 1:
                vals = vals.reshape(-1)
            df[col] = vals

    # --- Technical indicators ---
    df["ema20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["signal"] = macd.macd_signal()
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14)
    df["atr"] = atr.average_true_range()

    return df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------------
# Signal Computation
# ---------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    side, prob = "Hold", 0.5
    if row["ema20"] > row["ema50"] and row["rsi"] < 70 and row["macd"] > row["signal"]:
        side, prob = "Buy", 0.7
    elif row["ema20"] < row["ema50"] and row["rsi"] > 30 and row["macd"] < row["signal"]:
        side, prob = "Sell", 0.7
    return side, prob

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)
    if atr is None or np.isnan(atr):
        atr = price * 0.005
    if side == "Buy":
        return price + mult * atr, price - mult * atr
    elif side == "Sell":
        return price - mult * atr, price + mult * atr
    else:
        return price + mult * atr, price - mult * atr

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------
def fetch_sentiment(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", [])
        if not news:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n.get("title", "")).get("compound", 0.0) for n in news[:5]]
        return float(np.mean(scores))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Latest Prediction
# ---------------------------------------------------------------------------
def latest_prediction(df: pd.DataFrame, symbol: str = "", risk: str = "Medium") -> Optional[Dict[str, object]]:
    if df is None or df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    row_prev, row = df.iloc[-2], df.iloc[-1]
    side, prob = compute_signal_row(row_prev, row)
    atr_val = float(row["atr"]) if pd.notna(row["atr"]) else float(df["atr"].tail(14).mean())
    price = float(row["Close"])
    tp, sl = compute_tp_sl(price, atr_val, "Buy" if side == "Hold" else side, risk)
    sentiment_score = fetch_sentiment(symbol)
    regime_label = "Bull" if row["ema20"] > row["ema50"] else "Bear"
    return {
        "side": side,
        "prob": prob,
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment_score,
        "regime": regime_label,
        "ml_prob": prob,
    }

# ---------------------------------------------------------------------------
# ML + Backtesting
# ---------------------------------------------------------------------------
def train_ml_model(df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    if df is None or len(df) < 100:
        return None
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X = df[["ema20", "ema50", "rsi", "macd", "signal", "atr"]].dropna()
    y = df.loc[X.index, "target"]
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

def backtest_signals(df: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    if df is None or len(df) < 80:
        return {"win_rate": 0, "total_return_pct": 0, "n_trades": 0, "trades": []}

    trades, balance, wins = [], 1.0, 0
    for i in range(60, len(df) - 1):
        prev, row = df.iloc[i - 1], df.iloc[i]
        side, _ = compute_signal_row(prev, row)
        if side == "Hold":
            continue
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        tp, sl = compute_tp_sl(row["Close"], atr, side, risk)
        next_close = float(df["Close"].iloc[i + 1])
        profit = (next_close - row["Close"]) / row["Close"] if side == "Buy" else (row["Close"] - next_close) / row["Close"]
        wins += int(profit > 0)
        balance *= 1 + profit
        trades.append({"index": i, "side": side, "profit_pct": profit * 100})
    win_rate = 100 * wins / len(trades) if trades else 0
    total_return = (balance - 1) * 100
    return {"win_rate": win_rate, "total_return_pct": total_return, "n_trades": len(trades), "trades": trades}

# ---------------------------------------------------------------------------
# Analysis + Summary
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str, interval_key: str = "1h", risk: str = "Medium", use_cache=True) -> Optional[Dict[str, object]]:
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None
    df = add_indicators(df)
    pred = latest_prediction(df, symbol, risk)
    bt = backtest_signals(df, risk)
    return {
        "symbol": symbol,
        "interval": interval_key,
        "last_price": float(df["Close"].iloc[-1]),
        "signal": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": round(pred["tp"], 2) if pred["tp"] else None,
        "sl": round(pred["sl"], 2) if pred["sl"] else None,
        "win_rate": bt["win_rate"],
        "return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
    }

def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache=True) -> pd.DataFrame:
    _log("Fetching and analyzing market data (smart v2)...")
    rows = []
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset} ({symbol})...")
        res = analyze_asset(symbol, interval_key, risk, use_cache)
        if res:
            rows.append({
                "Asset": asset,
                "Symbol": symbol,
                "Interval": interval_key,
                "Price": res["last_price"],
                "Signal": res["signal"],
                "Probability_%": res["probability"],
                "TP": res["tp"],
                "SL": res["sl"],
                "WinRate_%": res["win_rate"],
                "Return_%": res["return_pct"],
                "Trades": res["n_trades"],
                "Sentiment": res["sentiment"],
                "Regime": res["regime"],
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
def load_asset_with_indicators(asset: str, interval_key: str, use_cache=True) -> Tuple[str, pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)
    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key, use_cache)
    return symbol, add_indicators(df)

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache=True):
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None, pd.DataFrame()
    df = add_indicators(df)
    pred = latest_prediction(df, symbol, risk)
    bt = backtest_signals(df, risk)
    return {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": pred["prob"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "ml_prob": pred["ml_prob"],
        "trades": bt["trades"],
    }, df

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------