# utils.py — stable version (all fixes retained + compat updates)

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

# ─────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────
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

# Keep both "interval" and "yf_interval" keys for backward compatibility
INTERVALS = {
    "15m": {"period": "5d", "interval": "15m", "yf_interval": "15m"},
    "1h":  {"period": "1mo", "interval": "1h", "yf_interval": "1h"},
    "1d":  {"period": "1y", "interval": "1d", "yf_interval": "1d"},
    "1wk": {"period": "5y", "interval": "1wk", "yf_interval": "1wk"},
}

RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.8}
FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _to_1d_series(x, index=None, name=None):
    """Coerce input to clean 1D numeric Series (fixes 'arg must be 1-d')."""
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
        s = s.map(lambda v: v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)
    s = pd.to_numeric(s, errors="coerce")
    if index is not None:
        s.index = index
    s.name = name
    return s.astype(float)


def _normalize_ohlcv(df):
    """Normalize YF dataframe (even MultiIndex) into consistent OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(lv) for lv in col if lv]).strip() for col in df.columns]

    cols_lower = [c.lower() for c in df.columns]
    out = pd.DataFrame(index=df.index)

    def _pick(name):
        if name in df.columns:
            return df[name]
        lname = name.lower()
        if lname in cols_lower:
            return df[df.columns[cols_lower.index(lname)]]
        return np.nan

    out["Open"]   = _to_1d_series(_pick("Open"), index=df.index, name="Open")
    out["High"]   = _to_1d_series(_pick("High"), index=df.index, name="High")
    out["Low"]    = _to_1d_series(_pick("Low"), index=df.index, name="Low")
    out["Close"]  = _to_1d_series(_pick("Close"), index=df.index, name="Close")
    out["Volume"] = _to_1d_series(_pick("Volume"), index=df.index, name="Volume").fillna(0)
    out.dropna(subset=["Close"], inplace=True)
    out["Return"] = out["Close"].pct_change()
    return out


# ─────────────────────────────────────────
# FETCH DATA — with retry and mirror fallback
# ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=4, delay=2.0):
    def _attempt(_period, _interval):
        df_raw = yf.download(symbol, period=_period, interval=_interval,
                             progress=False, threads=False, auto_adjust=True)
        return _normalize_ohlcv(df_raw)

    for attempt in range(1, retries + 1):
        try:
            df = _attempt(period, interval)
            if not df.empty and len(df) >= 50:
                print(f"✅ {symbol}: fetched {len(df)} rows ({interval}, {period})")
                return df
            else:
                print(f"⚠️ {symbol}: got {len(df)} rows; retrying …")
        except Exception as e:
            print(f"⚠️ Attempt {attempt} for {symbol} failed: {e}")
        time.sleep(delay + random.random())

    # fallback daily
    try:
        df = _attempt("3mo", "1d")
        if not df.empty:
            print(f"🔁 Daily fallback succeeded for {symbol}")
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

    print(f"🚫 All fetch attempts failed for {symbol}")
    return pd.DataFrame()


# ─────────────────────────────────────────
# INDICATORS — includes lowercase copies for tab compatibility
# ─────────────────────────────────────────
def add_indicators(df):
    df = _normalize_ohlcv(df)
    if df.empty:
        return df

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

    # Modern fill replacement (was .fillna(method='...'))
    df = df.ffill().bfill()
    df = df.iloc[50:]  # drop warm-up
    df["Return"] = close.pct_change().fillna(0)

    # create lowercase aliases for tabs expecting them
    for col in df.columns:
        if col.isupper():
            df[col.lower()] = df[col]

    return df


# ─────────────────────────────────────────
# TRADING THEORY
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# MODEL TRAINING & PREDICTION
# ─────────────────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if df.empty or len(df) < 20:
        return None

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    X = df[FEATURES].dropna()
    y = df.loc[X.index, "Target"]

    if len(X) < 20:
        latest = df.iloc[-1]
        trend = "buy" if latest["EMA_20"] > latest["EMA_50"] else "sell"
        atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        price = df["Close"].iloc[-1]
        mult = RISK_MULT.get(risk, 1.0)
        tp = price + (atr * 1.5 * mult if trend == "buy" else -atr * 1.5 * mult)
        sl = price - (atr * 1.0 * mult if trend == "buy" else -atr * 1.0 * mult)
        return {"prediction": trend, "probability": 0.65, "accuracy": 0.0,
                "tp": tp, "sl": sl, "model": None, "features": FEATURES, "X": X, "df": df}

    try:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X[:-1], y[:-1])
        latest_vec = X.iloc[[-1]].values
        pred_cls = clf.predict(latest_vec)[0]
        prob = float(clf.predict_proba(latest_vec)[0][pred_cls])
        raw_pred = "buy" if pred_cls == 1 else "sell"
    except Exception as e:
        print(f"⚠️ ML training failed: {e}")
        raw_pred, prob, clf = "neutral", 0.5, None

    adjusted_pred, conf_adj = apply_trading_theory(raw_pred, df)
    conf = min(1.0, prob * conf_adj)

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    tp = price + (atr * 1.5 * mult if adjusted_pred == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if adjusted_pred == "buy" else -atr * 1.0 * mult)

    return {"prediction": adjusted_pred, "probability": conf,
            "accuracy": float(clf.score(X, y)) if clf is not None else 0.0,
            "tp": float(tp), "sl": float(sl),
            "model": clf, "features": FEATURES, "X": X, "df": df}


# ─────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────
def backtest_signals(df, pred):
    if df is None or df.empty or pred is None:
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}
    sig = 1 if pred["prediction"] == "buy" else -1 if pred["prediction"] == "sell" else 0
    work = df.copy()
    work["Signal"] = sig
    work["Strat_Return"] = work["Signal"].shift(1) * work["Return"]
    work["Equity"] = (1 + work["Strat_Return"].fillna(0)).cumprod()
    wins = (work["Strat_Return"] > 0).sum()
    winrate = wins / max(1, len(work))
    total_ret = work["Equity"].iloc[-1] - 1
    return {"winrate": float(winrate), "total_return": float(total_ret), "equity_curve": work["Equity"]}


# ─────────────────────────────────────────
# SUMMARIZE MULTIPLE ASSETS
# ─────────────────────────────────────────
def summarize_assets(interval_key="1h"):
    if interval_key not in INTERVALS:
        interval_key = "1h"
    info = INTERVALS[interval_key]
    results = []
    progress = st.progress(0)
    status = st.empty()
    log = ""
    total = len(ASSET_SYMBOLS)

    for i, (asset, symbol) in enumerate(ASSET_SYMBOLS.items(), start=1):
        log += f"⏳ Fetching **{asset}** ({symbol})...\n"
        status.markdown(log)
        df = fetch_data(symbol, interval=info["yf_interval"], period=info["period"])
        if df.empty:
            log += f"⚠️ No data for **{asset}**, skipped.\n"
            status.markdown(log)
            progress.progress(i / total)
            continue
        try:
            pred = train_and_predict(df)
            if not pred:
                log += f"⚠️ Could not predict **{asset}**.\n"
            else:
                back = backtest_signals(df, pred)
                results.append({
                    "Asset": asset,
                    "Prediction": pred["prediction"],
                    "Confidence": round(pred["probability"] * 100, 2),
                    "Accuracy": round(pred["accuracy"] * 100, 2),
                    "Win Rate": round(back["winrate"] * 100, 2),
                    "Return": round(back["total_return"] * 100, 2),
                })
                log += f"✅ {asset} analyzed successfully.\n"
        except Exception as e:
            log += f"❌ Error analyzing **{asset}**: {e}\n"
        status.markdown(log)
        progress.progress(i / total)
        time.sleep(0.3)

    if not results:
        log += "\n🚫 No valid data fetched — showing placeholder."
        status.markdown(log)
        return pd.DataFrame([{
            "Asset": "No Data", "Prediction": "neutral", "Confidence": 0.0,
            "Accuracy": 0.0, "Win Rate": 0.0, "Return": 0.0,
        }])

    log += "\n🎉 Analysis complete!"
    status.markdown(log)
    progress.progress(1.0)
    return pd.DataFrame(results)