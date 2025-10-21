import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta  # Technical Analysis library

# -------------------------------
# 🔧 Configuration
# -------------------------------
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
    "Return", "MA_10", "MA_50", "RSI", "MACD", "Signal_Line", "ATR",
    "Momentum", "Sentiment"
]

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5
}

sentiment_analyzer = SentimentIntensityAnalyzer()


# -------------------------------
# 📥 Data Fetching
# -------------------------------
def fetch_data(symbol, interval="1h", period="60d"):
    """Download data safely from Yahoo Finance."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        df["Return"] = df["Close"].pct_change()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["Signal_Line"] = macd.macd_signal()
        df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
        df["Momentum"] = ta.momentum.ROCIndicator(df["Close"]).roc()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return pd.DataFrame()


# -------------------------------
# 🧠 Sentiment Integration
# -------------------------------
def get_sentiment(asset):
    """Estimate sentiment based on asset name (placeholder for live news sentiment)."""
    try:
        text = asset.lower()
        sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]
        return sentiment_score
    except Exception:
        return 0


# -------------------------------
# 📈 Train and Predict
# -------------------------------
def train_and_predict(df, interval="1h", risk="Medium"):
    """Train ML model (Random Forest) and generate adaptive predictions with TP/SL."""
    try:
        if df.empty:
            return None, None, {
                "signal": "NO DATA",
                "prob": 0,
                "accuracy": 0,
                "sentiment": 0,
                "risk": risk,
                "tp": 0,
                "sl": 0
            }

        df["Y"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
        df.dropna(inplace=True)

        sentiment = get_sentiment("market")

        X = df[["Return", "MA_10", "MA_50", "RSI", "MACD", "Signal_Line", "ATR", "Momentum"]].copy()
        X["Sentiment"] = sentiment

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=FEATURES)

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = df["Y"].iloc[:split], df["Y"].iloc[split:]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        last = X.iloc[[-1]]
        prob = clf.predict_proba(last)[0][1]
        adjusted_prob = prob + (sentiment * 0.05)

        signal = (
            "BUY" if adjusted_prob > 0.55 else
            "SELL" if adjusted_prob < 0.45 else
            "HOLD"
        )

        atr = df["ATR"].iloc[-1]
        price = df["Close"].iloc[-1]
        risk_mult = RISK_MULT.get(risk, 1.0)

        # --- Adaptive Take-Profit / Stop-Loss based on volatility regime ---
        volatility_mean = df["ATR"].rolling(50).mean().iloc[-1]
        volatility_ratio = atr / volatility_mean if volatility_mean > 0 else 1

        if volatility_ratio < 0.8:
            tp_mult, sl_mult = 1.5, 0.8
        elif volatility_ratio > 1.5:
            tp_mult, sl_mult = 2.5, 1.2
        else:
            tp_mult, sl_mult = 2.0, 1.0

        tp_mult *= risk_mult
        sl_mult *= risk_mult

        if signal == "BUY":
            tp = price + atr * tp_mult
            sl = price - atr * sl_mult
        elif signal == "SELL":
            tp = price - atr * tp_mult
            sl = price + atr * sl_mult
        else:
            tp, sl = price, price

        result = {
            "signal": signal,
            "prob": round(adjusted_prob, 3),
            "accuracy": round(acc, 3),
            "sentiment": round(sentiment, 3),
            "risk": risk,
            "tp": round(tp, 2),
            "sl": round(sl, 2)
        }

        return X, clf, result

    except Exception as e:
        print(f"⚠️ train_and_predict() error: {e}")
        return None, None, {
            "signal": "ERROR",
            "prob": 0,
            "accuracy": 0,
            "sentiment": 0,
            "risk": risk,
            "tp": 0,
            "sl": 0
        }


# -------------------------------
# 📊 Backtesting
# -------------------------------
def backtest_signals(df, initial_balance=10000, fee=0.001):
    """Simulate trades using model's BUY/SELL signals."""
    try:
        df = df.copy().dropna()
        df["Signal"] = np.where(df["Return"].shift(-1) > 0, 1, -1)

        balance = initial_balance
        position = 0
        equity_curve = []
        trades = []

        for i in range(1, len(df)):
            signal = df["Signal"].iloc[i]

            if signal == 1 and position == 0:
                position = balance / df["Close"].iloc[i]
                balance = 0
                trades.append(("BUY", df.index[i], df["Close"].iloc[i]))

            elif signal == -1 and position > 0:
                balance = position * df["Close"].iloc[i] * (1 - fee)
                position = 0
                trades.append(("SELL", df.index[i], df["Close"].iloc[i]))

            equity_curve.append(balance + position * df["Close"].iloc[i])

        if position > 0:
            balance = position * df["Close"].iloc[-1]
            position = 0

        total_return = ((equity_curve[-1] - initial_balance) / initial_balance) * 100 if equity_curve else 0
        wins = sum(1 for t in trades if t[0] == "SELL")
        winrate = (wins / len(trades)) * 100 if trades else 0

        result = {
            "total_return": round(total_return, 2),
            "winrate": round(winrate, 2),
            "trades": trades,
            "equity_curve": pd.Series(equity_curve, index=df.index[-len(equity_curve):])
        }

        return result

    except Exception as e:
        print(f"[ERROR] backtest_signals: {e}")
        return {"total_return": 0, "winrate": 0, "equity_curve": pd.Series()}