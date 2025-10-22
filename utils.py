import yfinance as yf
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ───────────────────────────────
# Global Config
# ───────────────────────────────
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

INTERVALS = {
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "1d": {"interval": "1d", "period": "6mo"},
}

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5,
}

FEATURES = ["rsi", "macd", "bb_width", "returns"]

# ───────────────────────────────
# Fetching Market Data
# ───────────────────────────────
def fetch_data(symbol, interval="1h", period="1mo", max_retries=3):
    """
    Download and preprocess market data with retry and timeout logic.
    """
    print(f"📊 Fetching {symbol} [{interval}] for {period}...")
    df = pd.DataFrame()

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=False,
                timeout=10  # prevent hang
            )

            if df.empty:
                raise ValueError("Empty data returned")

            # Fix for 2D array data bug (yfinance 0.2.66+)
            for c in df.columns:
                if isinstance(df[c].iloc[0], (list, np.ndarray)):
                    df[c] = df[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

            # Indicators
            df["rsi"] = RSIIndicator(df["Close"]).rsi()
            macd = MACD(df["Close"])
            df["macd"] = macd.macd()
            bb = BollingerBands(df["Close"])
            df["bb_width"] = bb.bollinger_wband()
            df["returns"] = df["Close"].pct_change()
            df["volatility"] = df["returns"].rolling(20).std()
            df.dropna(inplace=True)

            print(f"✅ Success: {symbol} ({len(df)} rows)")
            return df

        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed for {symbol}: {e}")
            if attempt < max_retries:
                wait = 2 + np.random.uniform(0.5, 2.5)
                print(f"⏳ Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"🚫 Skipping {symbol} after {max_retries} failed attempts.")
                return pd.DataFrame()

    return df


# ───────────────────────────────
# Train Model + Predict
# ───────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    """Train RandomForest and predict direction, accuracy, TP, SL."""
    if df.empty:
        return None

    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = df["target"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prob = clf.predict_proba([X.iloc[-1]])[0][1]

        direction = "Buy" if prob > 0.5 else "Sell"
        last_price = df["Close"].iloc[-1]
        mult = RISK_MULT.get(risk, 1.0)

        tp = last_price * (1 + 0.01 * mult) if direction == "Buy" else last_price * (1 - 0.01 * mult)
        sl = last_price * (1 - 0.005 * mult) if direction == "Buy" else last_price * (1 + 0.005 * mult)

        return {
            "prediction": direction,
            "probability": float(prob),
            "accuracy": float(acc),
            "tp": float(tp),
            "sl": float(sl),
        }

    except Exception as e:
        print(f"Model error: {e}")
        return None


# ───────────────────────────────
# Summary for Dashboard
# ───────────────────────────────
def summarize_assets():
    """Fetch all assets and return summary DataFrame."""
    results = []
    for asset, symbol in ASSET_SYMBOLS.items():
        try:
            df = fetch_data(symbol, "1h", "1mo")
            if df.empty:
                print(f"No data available for {asset}")
                continue

            pred = train_and_predict(df, "1h", "Medium")
            if not pred:
                continue

            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 2),
                "Accuracy": round(pred["accuracy"] * 100, 2),
                "TP": round(pred["tp"], 2),
                "SL": round(pred["sl"], 2),
            })
        except Exception as e:
            print(f"Error processing {asset}: {e}")

    return pd.DataFrame(results)


# ───────────────────────────────
# Backtesting (for Scenarios)
# ───────────────────────────────
def backtest_signals(df, pred):
    """Estimate Win Rate & Total Return via simple simulated trades."""
    if df is None or df.empty or not isinstance(pred, dict):
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    if "Close" not in df.columns:
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    df = df.copy()
    close = df["Close"].values
    tp = pred.get("tp")
    sl = pred.get("sl")
    direction = pred.get("prediction", "").lower()

    equity = [1.0]
    wins = 0
    losses = 0

    for i in range(1, len(close)):
        prev = close[i - 1]
        price = close[i]

        if direction == "buy":
            if price >= tp:
                r = (tp - prev) / prev
                wins += 1
            elif price <= sl:
                r = (sl - prev) / prev
                losses += 1
            else:
                r = (price - prev) / prev
        elif direction == "sell":
            if price <= tp:
                r = (prev - tp) / prev
                wins += 1
            elif price >= sl:
                r = (prev - sl) / prev
                losses += 1
            else:
                r = (prev - price) / prev
        else:
            r = 0

        equity.append(equity[-1] * (1 + r))

    total_trades = max(wins + losses, 1)
    winrate = wins / total_trades
    total_return = equity[-1] - 1.0

    return {
        "winrate": float(winrate),
        "total_return": float(total_return),
        "equity_curve": pd.Series(equity, index=df.index[: len(equity)]),
    }