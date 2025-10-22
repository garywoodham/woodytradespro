import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

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
    "15m": {"period": "5d"},
    "1h": {"period": "1mo"},
    "1d": {"period": "6mo"}
}

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5
}

FEATURES = [
    "return", "volatility", "rsi", "macd", "sentiment"
]

# For caching and sentiment analysis
CACHE_DATA = {}
analyzer = SentimentIntensityAnalyzer()

# ----------------------------------------------------------------------
# FETCH MARKET DATA (with retry + cache fallback)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_data(symbol: str, interval: str = "1h", period: str = None, max_retries: int = 4) -> pd.DataFrame:
    """Fetch market data from Yahoo Finance with retry, backoff, and cache fallback."""
    if period is None:
        period = INTERVALS.get(interval, {"period": "1mo"})["period"]

    for attempt in range(max_retries):
        try:
            print(f"📊 Fetching {symbol} [{interval}] for {period} (Attempt {attempt+1})...")
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
                timeout=30
            )

            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")

            df.dropna(inplace=True)
            df["return"] = df["Close"].pct_change()
            df["volatility"] = df["return"].rolling(10).std()
            df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["macd"] = ta.trend.MACD(df["Close"]).macd()

            # Sentiment proxy (you can extend this with news or social feed)
            df["sentiment"] = np.where(df["return"] > 0, 1, -1)

            df.dropna(inplace=True)
            CACHE_DATA[symbol] = df  # ✅ Save to cache memory
            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            wait = random.uniform(2.5, 5.5) * (attempt + 1)
            print(f"⏳ Waiting {wait:.1f}s before retry...")
            time.sleep(wait)

    # fallback if all attempts fail
    if symbol in CACHE_DATA:
        print(f"⚠️ Using cached data for {symbol} (last successful fetch).")
        return CACHE_DATA[symbol]

    print(f"🚫 All attempts failed for {symbol}, returning empty DataFrame.")
    return pd.DataFrame()

# ----------------------------------------------------------------------
# MODEL TRAINING AND PREDICTION
# ----------------------------------------------------------------------

def train_and_predict(df: pd.DataFrame, horizon: str = "1h", risk: str = "Medium") -> dict:
    """Train RandomForest model and predict direction with TP/SL."""
    try:
        df = df.copy()
        df["Y"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df.dropna(inplace=True)

        X = df[FEATURES]
        y = df["Y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        clf = RandomForestClassifier(n_estimators=120, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        prob = clf.predict_proba(X.iloc[[-1]])[0]
        pred = "Buy" if prob[1] > 0.55 else "Sell"

        last_price = df["Close"].iloc[-1]
        vol = df["volatility"].iloc[-1] or 0.002
        mult = RISK_MULT.get(risk, 1.0)

        if pred == "Buy":
            tp = last_price * (1 + vol * 10 * mult)
            sl = last_price * (1 - vol * 5 * mult)
        else:
            tp = last_price * (1 - vol * 10 * mult)
            sl = last_price * (1 + vol * 5 * mult)

        return {
            "prediction": pred,
            "probability": float(prob[1]),
            "accuracy": float(acc),
            "tp": float(tp),
            "sl": float(sl)
        }

    except Exception as e:
        print(f"⚠️ Error in training/prediction: {e}")
        return {}

# ----------------------------------------------------------------------
# MULTI-ASSET SUMMARY
# ----------------------------------------------------------------------

def summarize_assets() -> pd.DataFrame:
    """Loop through all assets, predict, and summarize results."""
    results = []

    for asset, symbol in ASSET_SYMBOLS.items():
        df = fetch_data(symbol, interval="1h")
        if df.empty:
            print(f"No data available for {asset}")
            continue

        pred = train_and_predict(df)
        if not pred:
            print(f"Error processing {asset}")
            continue

        results.append({
            "Asset": asset,
            "Prediction": pred["prediction"],
            "Confidence": round(pred["probability"] * 100, 2),
            "Accuracy": round(pred["accuracy"] * 100, 2),
            "TP": round(pred["tp"], 2),
            "SL": round(pred["sl"], 2)
        })

    if len(results) == 0:
        print("🚫 No assets could be analyzed. Check data source or connection.")
        return pd.DataFrame()

    return pd.DataFrame(results)

# ----------------------------------------------------------------------
# HELPER: CLEAN DATAFRAME
# ----------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove inf and NaN values."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df