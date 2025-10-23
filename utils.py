import os
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# ───────────────────────────────────────────────
# GLOBAL CONSTANTS
# ───────────────────────────────────────────────

RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.5}

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
    "1h": {"interval": "1h", "period": "1mo"},
    "4h": {"interval": "4h", "period": "3mo"},
    "1d": {"interval": "1d", "period": "6mo"},
}

FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal", "EMA_Cross"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# DATA NORMALIZATION
# ───────────────────────────────────────────────
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")

# ───────────────────────────────────────────────
# DATA FETCHING (local caching + mirror backup)
# ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=4, delay=2.0, force_refresh=False):
    """Fetch market data with local file caching, Yahoo retries, and mirror fallback."""
    fname = f"{symbol.replace('=', '_').replace('^', '')}_{interval}.csv"
    fpath = os.path.join(DATA_DIR, fname)

    # 1️⃣ Load from cache (if <24h old)
    if not force_refresh and os.path.exists(fpath):
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
        if datetime.now() - mtime < timedelta(hours=24):
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if not df.empty:
                    print(f"📁 Loaded cached {symbol} ({len(df)} rows)")
                    return _normalize_ohlcv(df)
            except Exception as e:
                print(f"⚠️ Failed to read cache for {symbol}: {e}")

    # 2️⃣ Attempt Yahoo Finance download
    def _attempt(_period, _interval):
        data = yf.download(
            symbol, period=_period, interval=_interval,
            progress=False, threads=False, auto_adjust=True
        )
        return _normalize_ohlcv(data)

    df = pd.DataFrame()
    for attempt in range(1, retries + 1):
        try:
            df = _attempt(period, interval)
            if not df.empty and len(df) >= 40:
                print(f"✅ {symbol}: fetched {len(df)} rows ({interval}, {period})")
                break
            else:
                print(f"⚠️ {symbol}: got {len(df)} rows; retrying …")
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait = 10 + random.uniform(0, 5)
                print(f"⏳ Rate limited for {symbol}. Waiting {wait:.1f}s …")
                time.sleep(wait)
            else:
                print(f"⚠️ Attempt {attempt} for {symbol} failed: {e}")
        time.sleep(delay + random.random())

    # 3️⃣ Fallback daily
    if df.empty:
        try:
            df = _attempt("3mo", "1d")
            if not df.empty:
                print(f"🔁 Daily fallback succeeded for {symbol}")
        except Exception as e:
            print(f"🚫 Daily fallback failed for {symbol}: {e}")

    # 4️⃣ Mirror backup
    if df.empty:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                js = r.json()["chart"]["result"][0]
                q = js["indicators"]["quote"][0]
                tmp = pd.DataFrame(q)
                tmp["Date"] = pd.to_datetime(js["timestamp"], unit="s")
                tmp.set_index("Date", inplace=True)
                df = _normalize_ohlcv(tmp)
                if not df.empty:
                    print(f"✅ Mirror fetch succeeded for {symbol}")
        except Exception as e:
            print(f"🚫 Mirror fetch failed for {symbol}: {e}")

    # 5️⃣ Save to disk
    if not df.empty:
        try:
            df.to_csv(fpath)
            print(f"💾 Cached {symbol} → {fpath}")
        except Exception as e:
            print(f"⚠️ Could not save cache for {symbol}: {e}")
    else:
        print(f"🚫 All fetch attempts failed for {symbol}")

    return df

# ───────────────────────────────────────────────
# TECHNICAL INDICATORS
# ───────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    df["EMA_Cross"] = np.where(df["EMA_20"] > df["EMA_50"], 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

# ───────────────────────────────────────────────
# THEORY-BASED ADJUSTMENT
# ───────────────────────────────────────────────
def apply_trading_theory(prediction, df):
    """Overlay RSI and trend logic to adjust prediction confidence."""
    if df.empty:
        return prediction, 1.0
    try:
        trend_bias = "buy" if df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1] else "sell"
        conf = 1.15 if prediction == trend_bias else 0.85

        rsi = df["RSI"].iloc[-1]
        if (rsi > 70 and prediction == "buy") or (rsi < 30 and prediction == "sell"):
            conf *= 0.7

        return prediction, conf
    except Exception:
        return prediction, 1.0

# ───────────────────────────────────────────────
# MODEL TRAINING & PREDICTION
# ───────────────────────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if df.empty or len(df) < 50:
        return None

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    feature_cols = [f for f in FEATURES if f in df.columns]
    if not feature_cols:
        feature_cols = [f.lower() for f in FEATURES if f.lower() in df.columns]
    if not feature_cols:
        print("⚠️ No valid feature columns.")
        return None

    X = df[feature_cols].dropna()
    y = df.loc[X.index, "Target"]
    if len(X) < 40:
        return None

    try:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X[:-1], y[:-1])
        acc = float(clf.score(X[:-1], y[:-1]))
        pred_cls = clf.predict(X.iloc[[-1]])[0]
        prob = float(clf.predict_proba(X.iloc[[-1]])[0][pred_cls])
        raw_pred = "buy" if pred_cls == 1 else "sell"
    except Exception as e:
        print(f"⚠️ Model failed: {e}")
        return None

    adj_pred, conf_adj = apply_trading_theory(raw_pred, df)
    conf = min(1.0, prob * conf_adj)

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    tp = price + (atr * 1.5 * mult if adj_pred == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if adj_pred == "buy" else -atr * 1.0 * mult)

    return {
        "prediction": adj_pred,
        "probability": conf,
        "accuracy": round(acc, 3),
        "tp": float(tp),
        "sl": float(sl),
    }

# ───────────────────────────────────────────────
# BACKTEST
# ───────────────────────────────────────────────
def backtest_signals(df, model_output):
    if df.empty or not model_output:
        return 0.0
    df["Signal"] = np.where(df["EMA_20"] > df["EMA_50"], 1, 0)
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]
    return round(((df["Strategy"] + 1).prod() - 1) * 100, 2)

# ───────────────────────────────────────────────
# SUMMARY ACROSS ALL ASSETS
# ───────────────────────────────────────────────
def summarize_assets(force_refresh=False):
    results = []
    for asset, symbol in ASSET_SYMBOLS.items():
        print(f"⏳ Fetching {asset} ({symbol})...")
        df = fetch_data(symbol, interval="1h", period="1mo", force_refresh=force_refresh)
        if df.empty:
            print(f"⚠️ No data for {asset}, skipped.")
            continue
        try:
            pred = train_and_predict(df)
            if not pred:
                print(f"⚠️ Could not predict {asset}.")
                continue
            back = backtest_signals(df, pred)
            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 1),
                "Accuracy": round(pred["accuracy"] * 100, 1),
                "TP": round(pred["tp"], 2),
                "SL": round(pred["sl"], 2),
                "BacktestReturn": back,
            })
        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")
            continue

    if not results:
        return pd.DataFrame()
    df_results = pd.DataFrame(results)
    print("✅ Summary complete.")
    return df_results