import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import time
import random
import requests


# ───────────────────────────────
# CONFIGURATION
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

RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.8}


# ───────────────────────────────
# SAFE FETCH WRAPPER (Resilient + Fallback + Curl + Cache)
# ───────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=3, delay=3):
    """Fetch market data with multi-layer fallback and 1D flatten fix."""

    def _flatten(df):
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x)
            )
        return df

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            if not df.empty and len(df) > 10:
                df = _flatten(df)
                df["Return"] = df["Close"].pct_change()
                print(f"✅ {symbol}: fetched {len(df)} rows.")
                return df
            else:
                print(f"⚠️ Attempt {attempt}: empty for {symbol}")
        except Exception as e:
            print(f"❌ Attempt {attempt} failed for {symbol}: {e}")
        time.sleep(delay + random.random())

    # Daily fallback
    try:
        print(f"🔁 Trying daily fallback for {symbol}...")
        df = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        if not df.empty:
            df = _flatten(df)
            df["Return"] = df["Close"].pct_change()
            print(f"✅ Daily fallback succeeded for {symbol}")
            return df
    except Exception as e:
        print(f"🚫 Daily fallback failed for {symbol}: {e}")

    # Curl-based backup
    try:
        print(f"🛰 Curl backup for {symbol}...")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            js = r.json()["chart"]["result"][0]
            df = pd.DataFrame(js["indicators"]["quote"][0])
            df["Date"] = pd.to_datetime(js["timestamp"], unit="s")
            df.set_index("Date", inplace=True)
            df["Return"] = df["close"].pct_change()
            print(f"✅ Curl backup succeeded for {symbol}")
            return df
    except Exception as e:
        print(f"🚫 Curl backup failed for {symbol}: {e}")

    print(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()


# ───────────────────────────────
# ADD TECHNICAL INDICATORS
# ───────────────────────────────
def add_indicators(df):
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()

    bb = BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    return df.dropna()


# ───────────────────────────────
# TRADING THEORY OVERLAY
# ───────────────────────────────
def apply_trading_theory(pred, df):
    latest = df.iloc[-1]
    ema_trend = latest["EMA_20"] > latest["EMA_50"]
    rsi_ok = 40 < latest["RSI"] < 70
    macd_conf = latest["MACD"] > latest["Signal_Line"]
    bb_breakout = latest["Close"] > latest["BB_High"] or latest["Close"] < latest["BB_Low"]
    score = sum([ema_trend, rsi_ok, macd_conf, bb_breakout])

    if pred == "buy" and score >= 2:
        return "buy", 0.95
    elif pred == "sell" and not ema_trend and score >= 2:
        return "sell", 0.95
    else:
        return "neutral", 0.6


# ───────────────────────────────
# MODEL TRAINING & PREDICTION
# ───────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if len(df) < 60:
        return None

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    features = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X[:-1], y[:-1])

    latest = X.iloc[-1:].values
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]
    raw_pred = "buy" if pred == 1 else "sell"

    adjusted_pred, conf_adj = apply_trading_theory(raw_pred, df)
    conf = min(1.0, prob * conf_adj)

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    price = df["Close"].iloc[-1]

    tp = price + (atr * 1.5 * mult if adjusted_pred == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if adjusted_pred == "buy" else -atr * 1.0 * mult)

    return {
        "prediction": adjusted_pred,
        "probability": conf,
        "accuracy": model.score(X, y),
        "tp": tp,
        "sl": sl,
    }


# ───────────────────────────────
# BACKTESTING ENGINE
# ───────────────────────────────
def backtest_signals(df, pred):
    if df is None or df.empty or pred is None:
        return {"winrate": 0, "total_return": 0, "equity_curve": pd.Series(dtype=float)}

    df = df.copy()
    sig = 1 if pred["prediction"] == "buy" else -1 if pred["prediction"] == "sell" else 0
    df["Signal"] = sig
    df["Strat_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strat_Return"]).cumprod()

    winrate = (df["Strat_Return"] > 0).sum() / max(1, len(df))
    total_ret = df["Equity"].iloc[-1] - 1

    return {"winrate": winrate, "total_return": total_ret, "equity_curve": df["Equity"]}


# ───────────────────────────────
# MULTI-ASSET SUMMARY (Progress + Visibility + Accuracy Fix)
# ───────────────────────────────
def summarize_assets():
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(ASSET_SYMBOLS)
    processed = 0
    status_text = ""

    for asset, symbol in ASSET_SYMBOLS.items():
        processed += 1
        status_text += f"⏳ Fetching **{asset}** ({symbol})...\n"
        status.markdown(status_text)
        progress.progress(processed / total)

        df = fetch_data(symbol, "1h", "1mo")
        if df.empty:
            status_text += f"⚠️ No data for **{asset}**, skipped.\n"
            status.markdown(status_text)
            continue

        try:
            pred = train_and_predict(df)
            if not pred:
                status_text += f"⚠️ Could not generate prediction for **{asset}**.\n"
                status.markdown(status_text)
                continue

            back = backtest_signals(df, pred)
            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 2),
                "Accuracy": round(pred["accuracy"] * 100, 2),
                "Win Rate": round(back["winrate"] * 100, 2),
                "Return": round(back["total_return"] * 100, 2),
            })
            status_text += f"✅ **{asset}** analyzed successfully.\n"
            status.markdown(status_text)

        except Exception as e:
            status_text += f"❌ Error analyzing **{asset}**: {e}\n"
            status.markdown(status_text)

        time.sleep(0.5)

    if not results:
        status_text += "\n🚫 No valid data fetched — showing placeholder."
        status.markdown(status_text)
        return pd.DataFrame([
            {"Asset": "No Data", "Prediction": "neutral", "Confidence": 0.0, "Accuracy": 0.0, "Win Rate": 0.0, "Return": 0.0}
        ])

    status_text += "\n🎉 Analysis complete!"
    status.markdown(status_text)
    progress.progress(1.0)
    return pd.DataFrame(results)