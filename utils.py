import os
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier

# ───────────────────────────────────────────────
# GLOBAL SETTINGS
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
# INTERNAL HELPERS
# ───────────────────────────────────────────────
def _normalize(df):
    """
    Normalize and clean an OHLCV dataframe.
    Handles tuple MultiIndex columns from yfinance safely.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    # Capitalize single-level names safely
    df.columns = [str(c).capitalize() for c in df.columns]

    # Replace infinities and fill missing
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Keep only valid price columns
    keep = ["Open", "High", "Low", "Close", "Volume"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(how="any")

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _mirror_fetch(symbol):
    """
    Secondary API to retrieve market data if Yahoo blocks direct calls.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            js = r.json()["chart"]["result"][0]
            q = js["indicators"]["quote"][0]
            df = pd.DataFrame(q)
            df["Date"] = pd.to_datetime(js["timestamp"], unit="s")
            df.set_index("Date", inplace=True)
            return _normalize(df)
    except Exception as e:
        print(f"🚫 Mirror fetch failed for {symbol}: {e}")
    return pd.DataFrame()

# ───────────────────────────────────────────────
# DATA FETCHING (CACHED)
# ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=4, force_refresh=False):
    """
    Fetch market data using yfinance with retry, caching, and mirror fallback.
    """
    fname = f"{symbol.replace('^','').replace('=','_')}_{interval}.csv"
    fpath = os.path.join(DATA_DIR, fname)

    # 1️⃣ Cached version if recent
    if not force_refresh and os.path.exists(fpath):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(fpath))
        if age < timedelta(hours=24):
            try:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if not df.empty:
                    st.write(f"📁 Loaded cached {symbol} ({len(df)} rows)")
                    return _normalize(df)
            except Exception as e:
                st.warning(f"⚠️ Cache read failed for {symbol}: {e}")

    # 2️⃣ Attempt Yahoo direct fetch
    df = pd.DataFrame()
    for attempt in range(1, retries + 1):
        try:
            st.write(f"⏳ Attempt {attempt}: Fetching {symbol} from Yahoo...")
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            df = _normalize(data)

            # ✅ FIXED LOGIC: only retry if empty or too small
            if df.empty or len(df) < 40:
                st.write(f"⚠️ {symbol}: got {len(df)} rows, retrying...")
            else:
                st.write(f"✅ {symbol}: fetched {len(df)} rows.")
                break

        except Exception as e:
            if "Too Many Requests" in str(e):
                wait = 10 + random.uniform(1, 5)
                st.warning(f"🕒 Rate limited for {symbol}. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                st.warning(f"⚠️ {symbol}: fetch error {e}")
        time.sleep(1 + random.random())

    # 3️⃣ Mirror fallback
    if df.empty:
        st.info(f"🪞 Attempting mirror fetch for {symbol}...")
        df = _mirror_fetch(symbol)
        if not df.empty:
            st.success(f"✅ Mirror fetch succeeded for {symbol}.")
        else:
            st.error(f"🚫 All fetch attempts failed for {symbol}.")

    # 4️⃣ Save to cache
    if not df.empty:
        try:
            df.to_csv(fpath)
            st.write(f"💾 Cached {symbol} data → {fpath}")
        except Exception as e:
            st.warning(f"⚠️ Could not save cache: {e}")
    return df

# ───────────────────────────────────────────────
# INDICATORS
# ───────────────────────────────────────────────
def add_indicators(df):
    """Add EMA, RSI, MACD, and crossover logic."""
    if df.empty:
        return df

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
# TRADING THEORY OVERLAY
# ───────────────────────────────────────────────
def apply_trading_theory(pred, df):
    """
    Adjust model prediction confidence using EMA trend + RSI bias.
    """
    if df.empty:
        return pred, 1.0
    try:
        trend_bias = "buy" if df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1] else "sell"
        conf = 1.15 if pred == trend_bias else 0.85
        rsi = df["RSI"].iloc[-1]
        if (rsi > 70 and pred == "buy") or (rsi < 30 and pred == "sell"):
            conf *= 0.7
        return pred, conf
    except Exception:
        return pred, 1.0

# ───────────────────────────────────────────────
# MODEL TRAINING & PREDICTION
# ───────────────────────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    """
    Train Random Forest model to predict next move and compute TP/SL.
    """
    df = add_indicators(df)
    if len(df) < 50:
        return None

    # Target = next candle up or down
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    feats = [f for f in FEATURES if f in df.columns]
    if not feats:
        return None

    X = df[feats].dropna()
    y = df.loc[X.index, "Target"]

    if len(X) < 40:
        return None

    try:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X[:-1], y[:-1])
        acc = clf.score(X[:-1], y[:-1])
        cls = clf.predict(X.iloc[[-1]])[0]
        prob = clf.predict_proba(X.iloc[[-1]])[0][cls]
        pred = "buy" if cls == 1 else "sell"
    except Exception as e:
        st.warning(f"⚠️ Model failed: {e}")
        return None

    # Apply trading theory adjustments
    pred, conf_adj = apply_trading_theory(pred, df)
    conf = min(1.0, prob * conf_adj)

    # Compute ATR-like target zones
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    tp = price + (atr * 1.5 * mult if pred == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if pred == "buy" else -atr * 1.0 * mult)

    return {
        "prediction": pred,
        "probability": conf,
        "accuracy": round(acc, 3),
        "tp": float(tp),
        "sl": float(sl),
    }

# ───────────────────────────────────────────────
# BACKTESTING
# ───────────────────────────────────────────────
def backtest_signals(df, model_output):
    """
    Simple backtest based on EMA cross and model direction.
    """
    if df.empty or not model_output:
        return 0.0

    df = df.copy()
    df["Signal"] = np.where(df["EMA_20"] > df["EMA_50"], 1, 0)
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]

    return round(((df["Strategy"] + 1).prod() - 1) * 100, 2)

# ───────────────────────────────────────────────
# SUMMARY FUNCTION
# ───────────────────────────────────────────────
def summarize_assets(force_refresh=False):
    """
    Fetch, analyze, predict, and summarize all tracked assets.
    """
    results = []
    st.info("Fetching and analyzing market data... please wait ⏳")

    for asset, symbol in ASSET_SYMBOLS.items():
        st.write(f"⏳ Fetching **{asset} ({symbol})**...")
        df = fetch_data(symbol, interval="1h", period="1mo", force_refresh=force_refresh)
        if df.empty:
            st.warning(f"⚠️ No data for {asset}, skipped.")
            continue

        try:
            pred = train_and_predict(df)
            if not pred:
                st.warning(f"⚠️ Could not predict {asset}.")
                continue

            ret = backtest_signals(df, pred)
            results.append({
                "Asset": asset,
                "Prediction": pred["prediction"],
                "Confidence": round(pred["probability"] * 100, 1),
                "Accuracy": round(pred["accuracy"] * 100, 1),
                "TP": round(pred["tp"], 2),
                "SL": round(pred["sl"], 2),
                "BacktestReturn": ret,
            })

        except Exception as e:
            st.error(f"❌ Error processing {asset}: {e}")
            continue

    if not results:
        st.error("🚫 No assets could be analyzed. Please check your internet connection or data source.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["LastUpdated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df