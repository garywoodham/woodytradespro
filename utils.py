import yfinance as yf
import pandas as pd
import numpy as np
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
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
    "15m": "Last 60 days",
    "1h": "Last 7 days",
    "1d": "Last 6 months",
    "1wk": "Last 2 years"
}

FEATURES = [
    "Return", "MA_10", "MA_50", "RSI", "MACD", "Signal_Line",
    "ATR", "Momentum", "Sentiment"
]

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5
}


# ==============================
# DATA FETCHING
# ==============================
def fetch_data(symbol, interval="1h", period=None):
    """
    Robust Yahoo Finance fetcher that auto-falls back to valid intervals.
    Works for indices, forex, crypto, and commodities.
    """
    try:
        # --- Step 1: Set safe default period ---
        if period is None:
            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                period = "7d"  # Intraday limited to ~7 days now
            else:
                period = "6mo"

        print(f"📊 Trying {symbol} [{interval}] ({period})...")
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # --- Step 2: Fallbacks ---
        if df.empty and interval != "1d":
            print(f"⚠️ {symbol}: No {interval} data → fallback to 1d")
            df = yf.download(symbol, period="6mo", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            df.attrs["data_source"] = "1d"
        else:
            df.attrs["data_source"] = interval

        if df.empty:
            print(f"⚠️ {symbol}: No 1d data → fallback to 1wk")
            df = yf.download(symbol, period="2y", interval="1wk",
                             auto_adjust=True, progress=False, threads=False)
            df.attrs["data_source"] = "1wk"

        if df.empty:
            print(f"❌ {symbol}: No data at any interval.")
            return pd.DataFrame()

        # --- Step 3: Feature Engineering ---
        df.dropna(inplace=True)
        df["Return"] = df["Close"].pct_change()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi().values.flatten()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd().values.flatten()
        df["Signal_Line"] = macd.macd_signal().values.flatten()

        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range().values.flatten()

        df["Momentum"] = ta.momentum.ROCIndicator(df["Close"]).roc().values.flatten()

        # --- Step 4: Add sentiment (placeholder random) ---
        df["Sentiment"] = np.random.uniform(-1, 1, len(df))

        df.dropna(inplace=True)
        print(f"✅ {symbol}: Loaded {len(df)} rows ({df.attrs['data_source']})")
        return df

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()


# ==============================
# STATUS MESSAGE
# ==============================
def get_data_status_message(symbol, df):
    """Readable message for Streamlit UI."""
    if df.empty:
        return f"❌ No data available for {symbol}"
    src = df.attrs.get("data_source", "unknown")
    return f"✅ Loaded {symbol} ({src}, {len(df)} candles)"


# ==============================
# MODEL TRAINING & PREDICTION
# ==============================
def train_and_predict(df, horizon="1h", risk="Medium"):
    """Train ML model and return predictions + metrics."""
    df = df.copy()
    df["Y"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = df["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    winrate = (preds == y_test).mean()

    latest = X.iloc[-1:].values
    next_pred = clf.predict(latest)[0]
    prob = clf.predict_proba(latest)[0][int(next_pred)]

    signal = "BUY" if next_pred == 1 else "SELL"
    current_price = df["Close"].iloc[-1]

    tp = current_price * (1 + 0.02 * RISK_MULT[risk]) if signal == "BUY" else current_price * (1 - 0.02 * RISK_MULT[risk])
    sl = current_price * (1 - 0.01 * RISK_MULT[risk]) if signal == "BUY" else current_price * (1 + 0.01 * RISK_MULT[risk])

    prediction = {
        "signal": signal,
        "prob": round(float(prob), 3),
        "accuracy": round(acc, 3),
        "winrate": round(winrate, 3),
        "tp": round(float(tp), 3),
        "sl": round(float(sl), 3)
    }

    return X, clf, prediction


# ==============================
# BACKTEST SIMULATION
# ==============================
def backtest_signals(df, signal_col="Y"):
    """Simple backtest for signals."""
    try:
        df = df.copy()
        df["Position"] = np.where(df[signal_col] == 1, 1, -1)
        df["Strategy_Return"] = df["Return"] * df["Position"].shift(1)
        df["Equity_Curve"] = (1 + df["Strategy_Return"]).cumprod()

        total_return = df["Equity_Curve"].iloc[-1] - 1
        winrate = (df["Strategy_Return"] > 0).mean()

        return {
            "equity_curve": df["Equity_Curve"],
            "total_return": round(float(total_return), 3),
            "winrate": round(float(winrate), 3)
        }

    except Exception as e:
        print(f"⚠️ Backtest error: {e}")
        return {"equity_curve": pd.Series(dtype=float), "total_return": 0, "winrate": 0}