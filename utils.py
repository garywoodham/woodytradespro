import time
import random
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
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


# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def _to_1d_series(x, index=None, name=None):
    """Coerce x into a 1-D float Series."""
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
    else:
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        s = pd.Series(arr, index=index)
    if s.dtype == "object":
        s = s.map(lambda v: v[0] if isinstance(v, (list, np.ndarray, tuple)) else v)
    s = pd.to_numeric(s, errors="coerce")
    s.name = name
    if index is not None:
        s.index = index
    return s.astype(float)


def _normalize_ohlcv(df):
    """Normalize Yahoo Finance data into clean 1-D numeric columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(lv) for lv in col if lv]).strip() for col in df.columns]

    cols = [c.lower() for c in df.columns]
    out = pd.DataFrame(index=df.index)

    def _pick(col):
        if col in df.columns:
            return df[col]
        if col.lower() in cols:
            return df[df.columns[cols.index(col.lower())]]
        return np.nan

    out["Open"] = _to_1d_series(_pick("Open"), index=df.index, name="Open")
    out["High"] = _to_1d_series(_pick("High"), index=df.index, name="High")
    out["Low"] = _to_1d_series(_pick("Low"), index=df.index, name="Low")
    out["Close"] = _to_1d_series(_pick("Close"), index=df.index, name="Close")
    out["Volume"] = _to_1d_series(_pick("Volume"), index=df.index, name="Volume").fillna(0)

    out.dropna(subset=["Close"], inplace=True)
    out["Return"] = out["Close"].pct_change()
    return out


# ─────────────────────────────
# DATA FETCHING (with retry + fallback)
# ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=3, delay=3):
    def _attempt(_period, _interval):
        df = yf.download(symbol, period=_period, interval=_interval,
                         progress=False, threads=False, auto_adjust=True)
        return _normalize_ohlcv(df)

    for attempt in range(1, retries + 1):
        try:
            df = _attempt(period, interval)
            if not df.empty and len(df) > 20:
                print(f"✅ {symbol}: fetched {len(df)} rows ({interval}, {period}).")
                return df
        except Exception as e:
            msg = str(e)
            if "RateLimit" in msg or "Too Many Requests" in msg:
                wait = random.uniform(5, 10) * attempt
                print(f"⏳ Rate-limited for {symbol}, cooling {wait:.1f}s ...")
                time.sleep(wait)
            else:
                print(f"⚠️ Attempt {attempt} failed for {symbol}: {e}")
                time.sleep(delay + random.random())

    # fallback to daily
    try:
        df = _attempt("3mo", "1d")
        if not df.empty:
            print(f"🔁 Fallback daily worked for {symbol}")
            return df
    except Exception as e:
        print(f"🚫 Daily fallback failed for {symbol}: {e}")

    # mirror backup
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            js = r.json()["chart"]["result"][0]
            q = js["indicators"]["quote"][0]
            tmp = pd.DataFrame(q)
            tmp["Date"] = pd.to_datetime(js["timestamp"], unit="s")
            tmp.set_index("Date", inplace=True)
            df = _normalize_ohlcv(tmp)
            if not df.empty:
                print(f"✅ Mirror fetch succeeded for {symbol}")
                return df
    except Exception as e:
        print(f"🚫 Mirror fetch failed for {symbol}: {e}")

    print(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()


# ─────────────────────────────
# TECHNICAL INDICATORS  (with warm-up fix)
# ─────────────────────────────
def add_indicators(df):
    df = _normalize_ohlcv(df)
    if df.empty:
        return df.copy()

    close = _to_1d_series(df["Close"], index=df.index, name="Close")

    try:
        df["EMA_20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA_50"] = EMAIndicator(close, 50).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()
        macd = MACD(close)
        df["MACD"] = macd.macd()
        df["Signal_Line"] = macd.macd_signal()
        bb = BollingerBands(close)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
    except Exception as e:
        print(f"⚠️ Indicator computation failed: {e}")
        return pd.DataFrame()

    # The previous fix: trim warm-up, fill, recalc return
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.iloc[50:]  # drop warm-up rows
    df["Return"] = df["Close"].pct_change().fillna(0)
    return df


# ─────────────────────────────
# TRADING THEORY
# ─────────────────────────────
def apply_trading_theory(pred_label, df):
    latest = df.iloc[-1]
    ema_trend = latest["EMA_20"] > latest["EMA_50"]
    rsi_ok = 40 < latest["RSI"] < 70
    macd_conf = latest["MACD"] > latest["Signal_Line"]
    bb_breakout = latest["Close"] > latest["BB_High"] or latest["Close"] < latest["BB_Low"]
    score = sum([ema_trend, rsi_ok, macd_conf, bb_breakout])
    if pred_label == "buy" and score >= 2:
        return "buy", 0.95
    elif pred_label == "sell" and not ema_trend and score >= 2:
        return "sell", 0.95
    else:
        return "neutral", 0.6


# ─────────────────────────────
# MODEL TRAINING & PREDICTION (with fallback)
# ─────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if df.empty or len(df) < 20:  # reduced threshold
        return None

    df = df.fillna(method="ffill").fillna(method="bfill")
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, "Target"]

    # fallback if too small
    if len(X) < 20:
        latest = df.iloc[-1]
        trend = "buy" if latest["EMA_20"] > latest["EMA_50"] else "sell"
        atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        price = df["Close"].iloc[-1]
        mult = RISK_MULT.get(risk, 1.0)
        tp = price + (atr * 1.5 * mult if trend == "buy" else -atr * 1.5 * mult)
        sl = price - (atr * 1.0 * mult if trend == "buy" else -atr * 1.0 * mult)
        return {
            "prediction": trend,
            "probability": 0.65,
            "accuracy": 0.0,
            "tp": tp,
            "sl": sl,
            "model": None,
            "features": FEATURES,
            "X": X,
            "df": df,
        }

    try:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X[:-1], y[:-1])
        latest = X.iloc[-1:].values
        pred_cls = clf.predict(latest)[0]
        prob = clf.predict_proba(latest)[0][pred_cls]
        raw_pred = "buy" if pred_cls == 1 else "sell"
    except Exception as e:
        print(f"⚠️ ML training failed: {e}")
        raw_pred, prob = "neutral", 0.5

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
        "accuracy": clf.score(X, y) if "clf" in locals() else 0.0,
        "tp": tp,
        "sl": sl,
        "model": clf if "clf" in locals() else None,
        "features": FEATURES,
        "X": X,
        "df": df,
    }


# ─────────────────────────────
# BACKTESTING
# ─────────────────────────────
def backtest_signals(df, pred):
    if df is None or df.empty or pred is None:
        return {"winrate": 0, "total_return": 0, "equity_curve": pd.Series(dtype=float)}
    sig = 1 if pred["prediction"] == "buy" else -1 if pred["prediction"] == "sell" else 0
    df["Signal"] = sig
    df["Strat_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strat_Return"]).cumprod()
    winrate = (df["Strat_Return"] > 0).sum() / max(1, len(df))
    total_ret = df["Equity"].iloc[-1] - 1
    return {"winrate": winrate, "total_return": total_ret, "equity_curve": df["Equity"]}


# ─────────────────────────────
# MULTI-ASSET SUMMARY (with progress logs)
# ─────────────────────────────
def summarize_assets():
    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(ASSET_SYMBOLS)
    log = ""
    for i, (asset, symbol) in enumerate(ASSET_SYMBOLS.items(), start=1):
        log += f"⏳ Fetching **{asset}** ({symbol})...\n"
        status.markdown(log)
        progress.progress(i / total)
        df = fetch_data(symbol, "1h", "1mo")
        if df.empty:
            log += f"⚠️ No data for **{asset}**, skipped.\n"
            status.markdown(log)
            continue
        try:
            pred = train_and_predict(df)
            if not pred:
                log += f"⚠️ Could not predict **{asset}**.\n"
                status.markdown(log)
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
            log += f"✅ **{asset}** analyzed successfully.\n"
            status.markdown(log)
        except Exception as e:
            log += f"❌ Error analyzing **{asset}**: {e}\n"
            status.markdown(log)
        time.sleep(0.3)

    if not results:
        log += "\n🚫 No valid data fetched — showing placeholder."
        status.markdown(log)
        return pd.DataFrame([{
            "Asset": "No Data",
            "Prediction": "neutral",
            "Confidence": 0.0,
            "Accuracy": 0.0,
            "Win Rate": 0.0,
            "Return": 0.0,
        }])

    log += "\n🎉 Analysis complete!"
    status.markdown(log)
    progress.progress(1.0)
    return pd.DataFrame(results)