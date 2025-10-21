import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==============================
# 🔧 CONFIGURATION
# ==============================

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

INTERVALS = {
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1w": "1wk"
}

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5
}

FEATURES = [
    "Return", "Volatility", "RSI", "MA_Diff", "ATR",
    "Momentum", "MACD", "MACD_Signal", "Sentiment"
]

analyzer = SentimentIntensityAnalyzer()

# ==============================
# 📈 DATA FETCHING (CACHED)
# ==============================

@st.cache_data(ttl=3600)
def fetch_data(symbol, interval="1h", period="90d"):
    """Fetch market data from Yahoo Finance safely and cache for 1h."""
    try:
        if interval in ["1m", "5m", "15m", "30m", "1h"]:
            period = "60d"

        df = yf.download(symbol, interval=interval, period=period, progress=False, prepost=False)

        if df.empty:
            df = yf.download(symbol, interval=interval, period="30d", progress=False, prepost=False)

        if df.empty:
            return pd.DataFrame()

        df = df.dropna().copy()
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(10).std()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()
        df["MA_Diff"] = df["MA50"] - df["MA200"]
        df["RSI"] = compute_rsi(df["Close"], 14)
        df["ATR"] = compute_atr(df, 14)
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df["MACD"], df["MACD_Signal"], _ = compute_macd(df["Close"])
        df["Sentiment"] = get_sentiment_score(symbol)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()

        return df

    except Exception as e:
        print(f"[ERROR] fetch_data: {e}")
        return pd.DataFrame()

# ==============================
# 📊 TECHNICAL INDICATORS
# ==============================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# ==============================
# 🗞️ SENTIMENT FETCH (Yahoo Finance Headlines)
# ==============================

def get_sentiment_score(symbol):
    """Fetch sentiment from Yahoo Finance headlines using VADER."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            return 0.0

        soup = BeautifulSoup(r.text, "html.parser")
        headlines = [h.get_text() for h in soup.find_all("h3")][:10]

        if not headlines:
            return 0.0

        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        sentiment = np.mean(scores)
        return round(sentiment, 3)

    except Exception as e:
        print(f"[ERROR] Sentiment fetch failed for {symbol}: {e}")
        return 0.0

# ==============================
# 🤖 MODEL TRAINING & PREDICTION
# ==============================

def train_and_predict(df, interval="1h", risk="Medium"):
    """Train RandomForest model, adjust with sentiment weighting."""
    try:
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()

        df["Y"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
        X = df[FEATURES]
        y = df["Y"]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=X.index)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, shuffle=False)

        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            class_weight="balanced_subsample"
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)

        last_prob = probs[-1]
        sentiment = df["Sentiment"].iloc[-1]

        # Sentiment-weighted adjustment
        adjusted_prob = last_prob + (sentiment * 0.15)
        adjusted_prob = max(0, min(1, adjusted_prob))  # Clamp to [0,1]

        signal = (
            "BUY" if adjusted_prob > 0.55 else
            "SELL" if adjusted_prob < 0.45 else
            "HOLD"
        )

        result = {
            "signal": signal,
            "prob": round(adjusted_prob, 3),
            "accuracy": round(acc, 3),
            "sentiment": round(sentiment, 3),
            "risk": risk
        }

        return X_scaled, clf, result

    except Exception as e:
        print(f"[ERROR] train_and_predict: {e}")
        return None, None, None