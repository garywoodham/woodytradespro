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

# ──────────────────────────────────────────────────────────────────────────────
# GLOBALS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

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
    "15m": {"interval": "15m", "period": "5d"},
    "1h":  {"interval": "1h",  "period": "1mo"},
    "4h":  {"interval": "4h",  "period": "3mo"},
    "1d":  {"interval": "1d",  "period": "6mo"},
    "1w":  {"interval": "1wk", "period": "2y"},
}

FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal", "EMA_Cross"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_ndarray_like(x):
    """Return a 1-D numpy array when x is (n,1) or Pandas Series; otherwise pass through."""
    if isinstance(x, pd.Series):
        return x.values
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    return x

def _coerce_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a standard OHLCV dataframe with columns:
    Open, High, Low, Close, Volume (capitalized).
    Handles MultiIndex and common Yahoo variants (Adj Close).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # If MultiIndex columns, take the first level name
    if isinstance(df.columns, pd.MultiIndex):
        # Try to pick level 0 names
        df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns]

    # Lower/strip then map to canonical names
    mapping = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["open", "o"]:
            mapping[c] = "Open"
        elif lc in ["high", "h"]:
            mapping[c] = "High"
        elif lc in ["low", "l"]:
            mapping[c] = "Low"
        elif lc in ["close", "c", "price", "last"]:
            mapping[c] = "Close"
        elif lc in ["adj close", "adj_close", "adjusted close"]:
            mapping[c] = "Adj Close"
        elif lc in ["volume", "vol", "v"]:
            mapping[c] = "Volume"
        else:
            mapping[c] = c  # keep fallback

    df = df.rename(columns=mapping)

    # If Close missing but Adj Close exists, use Adj Close as Close
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Keep only the standard set if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not keep:
        # Sometimes yahoo "quote" endpoint returns open/high/low/close keys already lowercase
        # Try again without filtering
        keep = list(df.columns)

    df = df[keep].copy()

    # Convert any (n,1) columns to 1-D arrays and coerce numeric
    for c in df.columns:
        df[c] = _flatten_ndarray_like(df[c])
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no close
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        except Exception:
            pass

    return df

def _normalize(raw):
    """Normalize any raw Yahoo result into standard OHLCV DataFrame."""
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="Close")
    df = pd.DataFrame(raw).copy()
    df = _coerce_ohlc_columns(df)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# MIRROR FETCH FALLBACK
# ──────────────────────────────────────────────────────────────────────────────

def _mirror_fetch(symbol, range_="1mo", interval_="1d"):
    """
    Simple mirror using Yahoo chart JSON. Not perfect, but helps when yfinance is rate-limited.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval_}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        js = r.json()["chart"]["result"][0]
        q = js["indicators"]["quote"][0]
        out = pd.DataFrame(q)
        out["Date"] = pd.to_datetime(js["timestamp"], unit="s")
        out.set_index("Date", inplace=True)
        return _normalize(out)
    except Exception as e:
        print(f"🚫 Mirror fetch failed for {symbol}: {e}")
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# FETCH DATA (CACHE + RETRIES + MIRROR)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, interval: str = "1h", period: str = "1mo",
               retries: int = 4, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch Yahoo data robustly; accept MultiIndex/2D; cache to disk; mirror fallback.
    """
    fname = f"{symbol.replace('^','').replace('=','_').replace('-','_')}_{interval}.csv"
    fpath = os.path.join(DATA_DIR, fname)

    # Use cache on disk if fresh (<48h) and not forcing refresh
    if not force_refresh and os.path.exists(fpath):
        try:
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(fpath))
            if age < timedelta(hours=48):
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                df = _normalize(df)
                if not df.empty:
                    st.write(f"📁 Loaded cached {symbol} {interval} ({len(df)} rows, {age.seconds//3600}h old)")
                    return df
        except Exception as e:
            st.warning(f"⚠️ Cache read failed for {symbol}: {e}")

    df = pd.DataFrame()
    for attempt in range(1, retries + 1):
        try:
            st.write(f"⏳ Attempt {attempt}: Fetching {symbol} from Yahoo…")
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            df = _normalize(data)

            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= 50:
                st.write(f"✅ {symbol}: fetched {len(df)} rows.")
                break
            else:
                st.warning(f"⚠️ {symbol}: insufficient/invalid shape ({type(df)} with {len(df)} rows), retrying…")
        except Exception as e:
            msg = str(e)
            if "Too Many Requests" in msg or "rate limit" in msg.lower():
                wait = 10 + random.uniform(2, 6)
                st.warning(f"🕒 Rate limit for {symbol}. Sleeping {wait:.1f}s…")
                time.sleep(wait)
            else:
                st.warning(f"⚠️ {symbol}: fetch error {e}")
        time.sleep(0.8 + random.random())

    if df.empty or len(df) < 50:
        st.info(f"🪞 Attempting mirror fetch for {symbol}…")
        # map our intrvl to a mirror-friendly one
        mirror_interval = "1d" if interval in ["1h", "4h", "15m"] else ("1wk" if interval == "1w" else interval)
        mirror_range = "1mo" if mirror_interval == "1d" else ("2y" if mirror_interval == "1wk" else "6mo")
        df = _mirror_fetch(symbol, range_=mirror_range, interval_=mirror_interval)
        if not df.empty:
            st.success(f"✅ Mirror fetch succeeded for {symbol}.")
        else:
            st.error(f"🚫 Mirror fetch failed for {symbol}.")

    # Save to disk cache if we have anything
    if not df.empty:
        try:
            df.to_csv(fpath)
            st.write(f"💾 Cached {symbol} data → {fpath}")
        except Exception as e:
            st.warning(f"⚠️ Could not write cache for {symbol}: {e}")

    return df

# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "Close" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    # ensure 1-D numeric
    out["Close"] = pd.to_numeric(_flatten_ndarray_like(out["Close"]), errors="coerce")
    out["High"]  = pd.to_numeric(_flatten_ndarray_like(out.get("High", out["Close"])), errors="coerce")
    out["Low"]   = pd.to_numeric(_flatten_ndarray_like(out.get("Low", out["Close"])), errors="coerce")

    try:
        out["EMA_20"] = EMAIndicator(out["Close"], 20).ema_indicator()
        out["EMA_50"] = EMAIndicator(out["Close"], 50).ema_indicator()
        out["RSI"]    = RSIIndicator(out["Close"], 14).rsi()
        macd          = MACD(out["Close"])
        out["MACD"]   = macd.macd()
        out["Signal"] = macd.macd_signal()
        out["EMA_Cross"] = np.where(out["EMA_20"] > out["EMA_50"], 1, 0)
    except Exception as e:
        st.warning(f"⚠️ Indicator calc failed: {e}")
        return pd.DataFrame()

    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# ──────────────────────────────────────────────────────────────────────────────
# TRADING THEORY OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

def apply_trading_theory(pred: str, df: pd.DataFrame):
    if df is None or df.empty:
        return pred, 1.0
    try:
        trend = "buy" if df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1] else "sell"
        conf = 1.15 if pred == trend else 0.85
        rsi = float(df["RSI"].iloc[-1])
        if (rsi > 70 and pred == "buy") or (rsi < 30 and pred == "sell"):
            conf *= 0.7
        return pred, conf
    except Exception:
        return pred, 1.0

# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING & PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def train_and_predict(df: pd.DataFrame, horizon: str = "1h", risk: str = "Medium"):
    # Indicators
    Xdf = add_indicators(df)
    if Xdf.empty:
        st.write("🔎 DEBUG: indicators empty (likely column shape/labels).")
        return None

    # Target: next-bar up/down
    try:
        Xdf["Target"] = np.where(Xdf["Close"].shift(-1) > Xdf["Close"], 1, 0)
    except Exception as e:
        st.write(f"🔎 DEBUG: Target build failed: {e}")
        return None

    feats = [f for f in FEATURES if f in Xdf.columns]
    if len(feats) < 4:  # need enough signals
        st.write(f"🔎 DEBUG: Not enough features present: {feats}")
        return None

    X = Xdf[feats].dropna()
    if "Target" not in Xdf.columns:
        st.write("🔎 DEBUG: Target column missing after features.")
        return None
    y = Xdf.loc[X.index, "Target"]

    if len(X) < 50:
        st.write(f"🔎 DEBUG: Too few samples for training: {len(X)}")
        return None

    try:
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X[:-1], y[:-1])
        acc = float(clf.score(X[:-1], y[:-1]))
        cls = int(clf.predict(X.iloc[[-1]])[0])
        proba = float(clf.predict_proba(X.iloc[[-1]])[0][cls])
        pred = "buy" if cls == 1 else "sell"
    except Exception as e:
        st.write(f"🔎 DEBUG: Model failure: {e}")
        return None

    pred, adj = apply_trading_theory(pred, Xdf)
    conf = min(1.0, proba * adj)

    # TP/SL via ATR-lite
    atr = (Xdf["High"] - Xdf["Low"]).rolling(14).mean().iloc[-1]
    price = Xdf["Close"].iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)

    if pred == "buy":
        tp = price + atr * 1.5 * mult
        sl = price - atr * 1.0 * mult
    else:
        tp = price - atr * 1.5 * mult
        sl = price + atr * 1.0 * mult

    return {
        "prediction": pred,
        "probability": conf,
        "accuracy": round(acc, 3),
        "tp": float(tp),
        "sl": float(sl),
    }

# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTING (quick, simple)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_signals(df: pd.DataFrame, model_output: dict):
    if df is None or df.empty:
        return 0.0
    Xdf = add_indicators(df)
    if Xdf.empty:
        return 0.0
    Xdf["SignalRule"] = np.where(Xdf["EMA_20"] > Xdf["EMA_50"], 1, 0)
    Xdf["Return"] = Xdf["Close"].pct_change()
    Xdf["Strategy"] = Xdf["SignalRule"].shift(1) * Xdf["Return"]
    total = (Xdf["Strategy"] + 1).prod() - 1
    return round(total * 100, 2)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION ACROSS ASSETS
# ──────────────────────────────────────────────────────────────────────────────

def summarize_assets(force_refresh: bool = False) -> pd.DataFrame:
    st.info("Fetching and analyzing market data... please wait ⏳")
    results = []
    for asset, symbol in ASSET_SYMBOLS.items():
        st.write(f"⏳ Fetching **{asset} ({symbol})**…")
        df = fetch_data(symbol, interval="1h", period="1mo", force_refresh=force_refresh)
        if df.empty:
            st.warning(f"⚠️ No data for {asset}, skipped.")
            continue

        pred = train_and_predict(df, horizon="1h", risk="Medium")
        if not pred:
            st.warning(f"⚠️ Could not predict {asset}.")
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

    if not results:
        st.error("🚫 No assets could be analyzed. Please check your internet connection or data source.")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["LastUpdated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out