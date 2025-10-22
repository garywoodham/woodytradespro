# ============================================================
# WoodyTradesPro Utilities (FINAL VERSION - CLEANED + CACHED)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

ASSET_SYMBOLS = {
    "Gold": "GC=F",
    "NASDAQ 100": "^NDX",
    "S&P 500": "^GSPC",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD"
}

# Risk multipliers for TP/SL calculation
RISK_MULT = {
    "Low": 1.2,
    "Medium": 1.5,
    "High": 2.0
}

# Interval → period mapping
INTERVALS = {
    "15m": {"period": "5d"},
    "30m": {"period": "7d"},
    "1h": {"period": "14d"},
    "1d": {"period": "6mo"},
    "1wk": {"period": "1y"}
}

# Feature columns used for model training
FEATURES = [
    "Return", "MA_10", "MA_50", "RSI",
    "MACD", "Signal_Line", "ATR", "Momentum", "Sentiment"
]

# ============================================================
# FETCH MARKET DATA
# ============================================================

@st.cache_data(show_spinner=False, ttl=1800)  # cache for 30 minutes
def fetch_data(symbol: str, interval: str = "1h", period: str = None, max_retries: int = 3) -> pd.DataFrame:
    """Download price data with retry, fallback, and caching."""
    if period is None:
        period = INTERVALS.get(interval, {"period": "1mo"})["period"]

    for attempt in range(max_retries):
        try:
            print(f"📊 Fetching {symbol} [{interval}] for {period}...")
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True  # prevents yfinance FutureWarning
            )

            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")

            # Flatten multi-index if exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna().copy()
            df["Return"] = df["Close"].pct_change()
            df["MA_10"] = df["Close"].rolling(10).mean()
            df["MA_50"] = df["Close"].rolling(50).mean()
            df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
            macd = MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["Signal_Line"] = macd.macd_signal()
            df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            df["Momentum"] = df["Close"].diff(4)

            # Replace infinities and NaNs
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            # Simple sentiment estimation based on returns
            df["Sentiment"] = np.where(df["Return"] > 0, 1, 0)

            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            time.sleep(2 ** attempt)

    return pd.DataFrame()

# ============================================================
# MODEL TRAINING + PREDICTION
# ============================================================

def train_and_predict(df: pd.DataFrame, horizon: str = "1h", risk: str = "Medium"):
    """Train a random forest model, return prediction + TP/SL."""
    if df.empty or len(df) < 50:
        return None

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df["Y"] = np.where(df["Return"].shift(-1) > 0, 1, 0)

    if df["Y"].nunique() < 2:
        return None

    X = df[FEATURES]
    y = df["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))

    last = X.iloc[[-1]]
    prob = clf.predict_proba(last)[0]
    pred = "BUY" if prob[1] > 0.55 else "SELL" if prob[1] < 0.45 else "HOLD"

    close = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    risk_mult = RISK_MULT.get(risk, 1.5)
    tp = close + (atr * risk_mult if pred == "BUY" else -atr * risk_mult)
    sl = close - (atr * risk_mult if pred == "BUY" else -atr * risk_mult)

    return {
        "prediction": pred,
        "probability": float(prob[1]),
        "accuracy": float(acc),
        "tp": float(tp),
        "sl": float(sl)
    }

# ============================================================
# SUMMARY FOR OVERVIEW TAB
# ============================================================

def summarize_assets():
    """Fetch, train, and summarize across all assets."""
    results = []
    for asset, symbol in ASSET_SYMBOLS.items():
        try:
            df = fetch_data(symbol, interval="1h")
            if df.empty:
                print(f"⚠️ No data for {asset}")
                continue

            pred = train_and_predict(df, "1h", "Medium")
            if pred is None:
                print(f"⚠️ No prediction for {asset}")
                continue

            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 2),
                "Accuracy": round(pred["accuracy"] * 100, 2),
                "Take Profit": round(pred["tp"], 2),
                "Stop Loss": round(pred["sl"], 2)
            })

        except Exception as e:
            print(f"⚠️ Error processing {asset}: {e}")
            continue

    if not results:
        print("❌ No assets could be analyzed.")
        return pd.DataFrame()

    return pd.DataFrame(results)