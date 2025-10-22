import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

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

RISK_MULT = {"Low": 1.2, "Medium": 1.5, "High": 2.0}

INTERVALS = {
    "15m": {"period": "5d"},
    "30m": {"period": "7d"},
    "1h": {"period": "14d"},
    "1d": {"period": "6mo"},
    "1wk": {"period": "1y"}
}

FEATURES = [
    "Return", "MA_10", "MA_50", "RSI",
    "MACD", "Signal_Line", "ATR", "Momentum", "Sentiment"
]

# ============================================================
# SENTIMENT ANALYSIS (HEADLINES)
# ============================================================

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(symbol: str) -> float:
    """Fetch sentiment score from recent Yahoo headlines (if available)."""
    try:
        from bs4 import BeautifulSoup
        import requests

        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h.get_text() for h in soup.find_all("h3")[:5]]
        if not headlines:
            return 0.0
        sentiment = np.mean([analyzer.polarity_scores(h)["compound"] for h in headlines])
        return sentiment
    except Exception:
        return 0.0

# ============================================================
# FETCH & FEATURE ENGINEERING
# ============================================================

def fetch_data(symbol: str, interval: str = "1h", period: str = None) -> pd.DataFrame:
    """Download and preprocess market data safely."""
    if not period:
        period = INTERVALS.get(interval, {"period": "30d"})["period"]

    print(f"📊 Fetching {symbol} [{interval}] for {period}...")

    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            raise ValueError("Empty data")

        # Add core indicators (without .values.flatten())
        df["Return"] = df["Close"].pct_change()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["Signal_Line"] = macd.macd_signal()

        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()

        df["Momentum"] = ta.momentum.ROCIndicator(df["Close"]).roc()

        df["Sentiment"] = get_sentiment_score(symbol)

        df.dropna(inplace=True)

        if len(df) < 50:
            raise ValueError("Too few data points")

        return df

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ============================================================
# MODEL TRAINING & PREDICTION
# ============================================================

def train_and_predict(df: pd.DataFrame, horizon: str = "1h", risk: str = "Medium"):
    """Train a model and produce next-step signal."""
    try:
        df["Y"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df.dropna(inplace=True)

        X = df[FEATURES]
        y = df["Y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        latest_features = X.iloc[[-1]]
        pred_prob = clf.predict_proba(latest_features)[0][1]
        signal = "BUY" if pred_prob > 0.55 else "SELL" if pred_prob < 0.45 else "HOLD"

        atr = df["ATR"].iloc[-1] if not df["ATR"].isna().all() else 0.0
        mult = RISK_MULT.get(risk, 1.5)
        tp = df["Close"].iloc[-1] + atr * mult if signal == "BUY" else df["Close"].iloc[-1] - atr * mult
        sl = df["Close"].iloc[-1] - atr * mult if signal == "BUY" else df["Close"].iloc[-1] + atr * mult

        return {
            "model": clf,
            "accuracy": acc,
            "prediction": signal,
            "probability": round(pred_prob, 3),
            "tp": round(tp, 2),
            "sl": round(sl, 2)
        }

    except Exception as e:
        print(f"⚠️ Error in train_and_predict: {e}")
        return None

# ============================================================
# BACKTESTING
# ============================================================

def backtest_signals(df: pd.DataFrame):
    """Backtest simple strategy on signal predictions."""
    try:
        df["Signal"] = np.where(df["RSI"] < 30, "BUY",
                         np.where(df["RSI"] > 70, "SELL", "HOLD"))
        df["Next_Close"] = df["Close"].shift(-1)
        df["Return"] = np.where(
            df["Signal"] == "BUY",
            df["Next_Close"] / df["Close"] - 1,
            np.where(df["Signal"] == "SELL",
                     df["Close"] / df["Next_Close"] - 1, 0)
        )

        df["Equity"] = (1 + df["Return"]).cumprod()
        winrate = (df["Return"] > 0).sum() / len(df)
        total_return = df["Equity"].iloc[-1] - 1

        return {
            "equity_curve": df["Equity"],
            "winrate": round(winrate * 100, 2),
            "total_return": round(total_return * 100, 2)
        }
    except Exception as e:
        print(f"⚠️ Error in backtest_signals: {e}")
        return {"equity_curve": pd.Series(dtype=float), "winrate": 0, "total_return": 0}

# ============================================================
# SUMMARY
# ============================================================

def summarize_assets():
    """Fetch and summarize all assets with predictions."""
    results = []
    for name, symbol in ASSET_SYMBOLS.items():
        df = fetch_data(symbol)
        if df.empty:
            print(f"No data available for {name}")
            continue
        pred = train_and_predict(df)
        if not pred:
            continue
        results.append({
            "Asset": name,
            "Symbol": symbol,
            "Prediction": pred["prediction"],
            "Probability": pred["probability"],
            "Accuracy": round(pred["accuracy"] * 100, 2),
            "Take Profit": pred["tp"],
            "Stop Loss": pred["sl"]
        })
    return pd.DataFrame(results)