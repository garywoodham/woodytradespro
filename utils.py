# utils.py  — WoodyTradesPro (ALL fixes kept)

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
# PUBLIC CONSTANTS (used across tabs)
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

# Mapping used by tabs to show available granularities
INTERVALS = {
    "15m": {"period": "5d", "yf_interval": "15m"},
    "1h":  {"period": "1mo", "yf_interval": "1h"},
    "1d":  {"period": "1y", "yf_interval": "1d"},
    "1wk": {"period": "5y", "yf_interval": "1wk"},
}

# Risk tuning for TP/SL
RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.8}

# Technical feature set
FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]


# ─────────────────────────────────────────
# INTERNAL HELPERS — 1-D & NaN fixes
# ─────────────────────────────────────────
def _to_1d_series(x, index=None, name=None):
    """Coerce input to a clean 1-D float Series (fix for 'arg must be 1-d')."""
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
    else:
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        s = pd.Series(arr, index=index)
    # flatten object cells like [value]
    if s.dtype == "object":
        s = s.map(lambda v: v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)
    s = pd.to_numeric(s, errors="coerce")
    if index is not None:
        s.index = index
    s.name = name
    return s.astype(float)


def _normalize_ohlcv(df):
    """Normalize YF output (also from MultiIndex) to flat OHLCV with numeric 1-D."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
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

    out["Open"]   = _to_1d_series(_pick("Open"),   index=df.index, name="Open")
    out["High"]   = _to_1d_series(_pick("High"),   index=df.index, name="High")
    out["Low"]    = _to_1d_series(_pick("Low"),    index=df.index, name="Low")
    out["Close"]  = _to_1d_series(_pick("Close"),  index=df.index, name="Close")
    out["Volume"] = _to_1d_series(_pick("Volume"), index=df.index, name="Volume").fillna(0)

    out = out.dropna(subset=["Close"])
    out["Return"] = out["Close"].pct_change()
    return out


# ─────────────────────────────────────────
# DATA FETCHING — retry, daily fallback, mirror
# (includes visible step logs via print; tabs show them)
# ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=4, base_delay=3.0):
    """Fetch from yfinance with robust fallbacks. Keeps earlier working behavior."""
    def _attempt(_period, _interval):
        df_raw = yf.download(
            symbol,
            period=_period,
            interval=_interval,
            progress=False,
            threads=False,
            auto_adjust=True,  # yfinance default changed; set explicit
        )
        return _normalize_ohlcv(df_raw)

    # Primary attempts
    for attempt in range(1, retries + 1):
        try:
            df = _attempt(period, interval)
            if not df.empty and len(df) >= 60:
                print(f"✅ {symbol}: fetched {len(df)} rows ({interval}, {period}).")
                return df
            else:
                print(f"⚠️ {symbol}: got {len(df)} rows; retrying …")
        except Exception as e:
            msg = str(e)
            if "RateLimit" in msg or "Too Many Requests" in msg:
                cool = random.uniform(5, 10) * attempt
                print(f"⏳ Rate-limited {symbol}; cooling {cool:.1f}s …")
                time.sleep(cool)
            else:
                print(f"⚠️ Attempt {attempt} for {symbol} failed: {e}")

        time.sleep(base_delay + random.random())

    # Daily fallback (often succeeds when intraday blocked)
    try:
        df_daily = _attempt("3mo", "1d")
        if not df_daily.empty:
            print(f"🔁 Daily fallback succeeded for {symbol} (3mo, 1d).")
            return df_daily
    except Exception as e:
        print(f"🚫 Daily fallback failed for {symbol}: {e}")

    # Mirror backup using Yahoo chart API
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200:
            js = r.json()["chart"]["result"][0]
            q = js["indicators"]["quote"][0]
            tmp = pd.DataFrame(q)
            tmp["Date"] = pd.to_datetime(js["timestamp"], unit="s")
            tmp.set_index("Date", inplace=True)
            df = _normalize_ohlcv(tmp)
            if not df.empty:
                print(f"✅ Mirror fetch succeeded for {symbol}.")
                return df
        else:
            print(f"⚠️ Mirror HTTP {r.status_code} for {symbol}")
    except Exception as e:
        print(f"🚫 Mirror fetch failed for {symbol}: {e}")

    print(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()


# ─────────────────────────────────────────
# INDICATORS — with warm-up trim (the fix that worked before)
# ─────────────────────────────────────────
def add_indicators(df):
    df = _normalize_ohlcv(df)
    if df.empty:
        return df

    close = _to_1d_series(df["Close"], index=df.index, name="Close")
    high  = _to_1d_series(df["High"],  index=df.index, name="High")
    low   = _to_1d_series(df["Low"],   index=df.index, name="Low")

    try:
        df["EMA_20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA_50"] = EMAIndicator(close, 50).ema_indicator()
        df["RSI"] = RSIIndicator(close, 14).rsi()
        macd = MACD(close)
        df["MACD"] = macd.macd()
        df["Signal_Line"] = macd.macd_signal()
        bb = BollingerBands(close)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"]  = bb.bollinger_lband()
    except Exception as e:
        # This was the source of 'arg must be 1-d' previously.
        print(f"⚠️ Indicator computation failed: {e}")
        return pd.DataFrame()

    # Warm-up trim + fill (THIS IS THE PRIOR FIX)
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.iloc[50:]                  # drop unstable warm-up rows
    df["Return"] = close.pct_change().fillna(0)
    return df


# ─────────────────────────────────────────
# TRADING THEORY LAYER (signal sanity)
# ─────────────────────────────────────────
def apply_trading_theory(pred_label, df):
    latest = df.iloc[-1]
    ema_trend  = latest["EMA_20"] > latest["EMA_50"]
    rsi_ok     = 40 < latest["RSI"] < 70
    macd_conf  = latest["MACD"] > latest["Signal_Line"]
    bb_break   = (latest["Close"] > latest["BB_High"]) or (latest["Close"] < latest["BB_Low"])
    votes = sum([ema_trend, rsi_ok, macd_conf, bb_break])

    if pred_label == "buy" and votes >= 2:
        return "buy", 0.95
    if pred_label == "sell" and (not ema_trend) and votes >= 2:
        return "sell", 0.95
    return "neutral", 0.6


# ─────────────────────────────────────────
# MODEL — with fallback when data is thin
# ─────────────────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if df.empty or len(df) < 20:
        return None

    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    X = df[FEATURES].dropna()
    y = df.loc[X.index, "Target"]

    # Fallback if not enough rows to train
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

    # Train simple, robust classifier
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

    # Trading-theory adjustment
    adj_label, conf_adj = apply_trading_theory(raw_pred, df)
    conf = min(1.0, prob * conf_adj)

    # TP/SL via simple ATR proxy
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    tp = price + (atr * 1.5 * mult if adj_label == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if adj_label == "buy" else -atr * 1.0 * mult)

    return {
        "prediction": adj_label,
        "probability": conf,
        "accuracy": float(clf.score(X, y)) if clf is not None else 0.0,
        "tp": float(tp),
        "sl": float(sl),
        "model": clf,
        "features": FEATURES,
        "X": X,
        "df": df,
    }


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
# MULTI-ASSET SUMMARY — with visible progress log
# ─────────────────────────────────────────
def summarize_assets(interval_key="1h"):
    """Runs fetch → train → backtest across ASSET_SYMBOLS and returns DataFrame."""
    if interval_key not in INTERVALS:
        interval_key = "1h"
    yf_interval = INTERVALS[interval_key]["yf_interval"]
    yf_period   = INTERVALS[interval_key]["period"]

    results = []
    progress = st.progress(0)
    status = st.empty()
    log = ""
    total = len(ASSET_SYMBOLS)

    for i, (asset, symbol) in enumerate(ASSET_SYMBOLS.items(), start=1):
        log += f"⏳ Fetching **{asset}** ({symbol})... "
        status.markdown(log)
        df = fetch_data(symbol, interval=yf_interval, period=yf_period)
        if df.empty:
            log += f"⚠️ No data for **{asset}**, skipped.\n"
            status.markdown(log)
            progress.progress(i / total)
            continue

        try:
            pred = train_and_predict(df, horizon=interval_key, risk="Medium")
            if not pred:
                log += f"⚠️ Could not predict **{asset}**.\n"
                status.markdown(log)
                progress.progress(i / total)
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
            log += "✅ Done.\n"
        except Exception as e:
            log += f"❌ Error analyzing **{asset}**: {e}\n"

        status.markdown(log)
        progress.progress(i / total)
        time.sleep(0.25)

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