import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
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

RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.5}
FEATURES = ["rsi", "macd", "bb_width", "returns"]

# ───────────────────────────────
# Fetch Data  (Robust, Non-Freezing)
# ───────────────────────────────
def fetch_data(symbol, interval="1h", period="1mo", max_retries=4):
    """Robust fetch with retries, array flattening, and Yahoo Finance fallbacks."""
    print(f"📊 Fetching {symbol} [{interval}] for {period} …")
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
                timeout=10,
            )

            # sometimes yfinance returns multi-level columns (tuple keys)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            # Flatten 2D/ndarray columns
            for c in df.columns:
                if len(df[c]) > 0 and isinstance(df[c].iloc[0], (list, np.ndarray)):
                    df[c] = df[c].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

            if df.empty or "Close" not in df.columns:
                raise ValueError("Empty or invalid data")

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
                print(f"⏳ Retrying in {wait:.1f}s…")
                time.sleep(wait)
            else:
                print(f"🚫 Skipping {symbol} after {max_retries} failed attempts.")
                return pd.DataFrame()

    return pd.DataFrame()


# ───────────────────────────────
# ML Model
# ───────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    if df.empty:
        return None
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X, y = df[FEATURES], df["target"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        prob = clf.predict_proba([X.iloc[-1]])[0][1]
        direction = "Buy" if prob > 0.5 else "Sell"

        last = df["Close"].iloc[-1]
        mult = RISK_MULT.get(risk, 1.0)
        tp = last * (1 + 0.01 * mult) if direction == "Buy" else last * (1 - 0.01 * mult)
        sl = last * (1 - 0.005 * mult) if direction == "Buy" else last * (1 + 0.005 * mult)

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
# Overview Summary (with progress)
# ───────────────────────────────
def summarize_assets():
    results = []
    total = len(ASSET_SYMBOLS)
    progress = st.progress(0)
    status = st.empty()

    for i, (asset, symbol) in enumerate(ASSET_SYMBOLS.items(), 1):
        status.markdown(f"🔍 **Analyzing {asset}** ({i}/{total}) …")
        progress.progress(i / total)

        df = fetch_data(symbol, "1h", "1mo")
        if df.empty:
            print(f"No data available for {asset}")
            continue

        pred = train_and_predict(df, "1h", "Medium")
        if pred:
            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 2),
                "Accuracy": round(pred["accuracy"] * 100, 2),
                "TP": round(pred["tp"], 2),
                "SL": round(pred["sl"], 2),
            })
        else:
            print(f"Prediction failed for {asset}")

        time.sleep(np.random.uniform(0.5, 1.5))  # slight delay avoids Yahoo block

    progress.progress(1.0)
    status.markdown("✅ **All assets processed.**")

    return pd.DataFrame(results)


# ───────────────────────────────
# Backtesting (Win Rate / Return)
# ───────────────────────────────
def backtest_signals(df, pred):
    if df is None or df.empty or not isinstance(pred, dict):
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    if "Close" not in df.columns:
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    df = df.copy()
    close = df["Close"].values
    tp, sl = pred.get("tp"), pred.get("sl")
    direction = pred.get("prediction", "").lower()

    equity = [1.0]
    wins = losses = 0

    for i in range(1, len(close)):
        prev, price = close[i - 1], close[i]

        if direction == "buy":
            if price >= tp:
                r, wins = (tp - prev) / prev, wins + 1
            elif price <= sl:
                r, losses = (sl - prev) / prev, losses + 1
            else:
                r = (price - prev) / prev
        elif direction == "sell":
            if price <= tp:
                r, wins = (prev - tp) / prev, wins + 1
            elif price >= sl:
                r, losses = (prev - sl) / prev, losses + 1
            else:
                r = (prev - price) / prev
        else:
            r = 0

        equity.append(equity[-1] * (1 + r))

    trades = max(wins + losses, 1)
    return {
        "winrate": float(wins / trades),
        "total_return": float(equity[-1] - 1.0),
        "equity_curve": pd.Series(equity, index=df.index[: len(equity)]),
    }