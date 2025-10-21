import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Risk Multiplier Constants
# ---------------------------
RISK_MULT = {
    "low": (0.5, 0.3),
    "medium": (1.0, 0.7),
    "high": (1.5, 1.0)
}

# ---------------------------
# Global Config
# ---------------------------
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

FEATURES = [
    "Return", "MA_10", "MA_50", "RSI", "MACD", "Signal_Line", "ATR", "Momentum", "Sentiment"
]

# ---------------------------
# Data Fetching
# ---------------------------
def fetch_data(symbol, interval="1h", period=None):
    """Download data safely from Yahoo Finance with interval-aware limits."""
    try:
        if period is None:
            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                period = "30d"  # Yahoo limit for intraday
            else:
                period = "1y"

        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            raise ValueError("No data returned from Yahoo")

        # --- Feature Engineering ---
        df.dropna(inplace=True)
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

        # Sentiment (synthetic for now — could be replaced by real feed)
        analyzer = SentimentIntensityAnalyzer()
        df["Sentiment"] = np.random.uniform(-1, 1, len(df))  # placeholder random sentiment

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------
# ML Training + Prediction
# ---------------------------
def train_and_predict(df, interval="1h", risk="medium"):
    """Train predictive model and make signal."""
    try:
        if df.empty or len(df) < 50:
            return None, None, {"signal": "HOLD", "prob": 0, "tp": 0, "sl": 0}

        df["Y"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df.dropna(inplace=True)

        X = df[FEATURES]
        y = df["Y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X.tail(1))[0][1]
        signal = "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "HOLD"

        # Take Profit / Stop Loss logic
        atr = df["ATR"].iloc[-1]
        if risk == "low":
            tp_mult, sl_mult = 0.5, 0.3
        elif risk == "high":
            tp_mult, sl_mult = 1.5, 1.0
        else:
            tp_mult, sl_mult = 1.0, 0.7

        close = df["Close"].iloc[-1]
        tp = close + atr * tp_mult if signal == "BUY" else close - atr * tp_mult
        sl = close - atr * sl_mult if signal == "BUY" else close + atr * sl_mult

        return X, clf, {"signal": signal, "prob": round(prob * 100, 2), "tp": tp, "sl": sl}

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, {"signal": "HOLD", "prob": 0, "tp": 0, "sl": 0}

# ---------------------------
# Backtesting Simulation
# ---------------------------
def backtest_signals(df, threshold_buy=0.55, threshold_sell=0.45):
    """Simulate performance using predicted probabilities."""
    try:
        if df.empty:
            return {"equity_curve": [], "winrate": 0, "total_return": 0}

        df["pred_prob"] = np.random.uniform(0, 1, len(df))  # placeholder prediction
        df["signal"] = np.where(df["pred_prob"] > threshold_buy, 1,
                         np.where(df["pred_prob"] < threshold_sell, -1, 0))

        df["strategy_return"] = df["signal"].shift(1) * df["Return"]
        df["equity_curve"] = (1 + df["strategy_return"]).cumprod()

        wins = len(df[df["strategy_return"] > 0])
        total = len(df[df["signal"] != 0])
        winrate = round((wins / total) * 100, 2) if total > 0 else 0
        total_return = round((df["equity_curve"].iloc[-1] - 1) * 100, 2)

        return {
            "equity_curve": df["equity_curve"],
            "winrate": winrate,
            "total_return": total_return
        }

    except Exception as e:
        print(f"⚠️ Backtest error: {e}")
        return {"equity_curve": [], "winrate": 0, "total_return": 0}

# ---------------------------
# Helper: Overall Accuracy
# ---------------------------
def calculate_model_accuracy(df, clf):
    """Compute model accuracy for display."""
    try:
        if df.empty or clf is None:
            return 0.0

        df["Y"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        X = df[FEATURES]
        y = df["Y"]
        acc = clf.score(X, y)
        return round(acc * 100, 2)
    except Exception:
        return 0.0