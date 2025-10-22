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

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
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

# expected price columns
_CORE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_CORE_COLS_LOWER = [c.lower() for c in _CORE_COLS]


# ─────────────────────────────────────────────────────────
# HELPER: force any input into a 1-D float Series (never 2-D)
# ─────────────────────────────────────────────────────────
def _to_1d_series(x, index=None, name=None):
    """
    Coerce x into a 1-D float pandas Series with the provided index.
    Handles:
      - 2D arrays shaped (n,1)
      - Series of lists/arrays -> takes first element
      - object dtype columns with array-like cells
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        # if single column, use it; else reduce via first column
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.iloc[:, 0]
    else:
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        elif arr.ndim > 1:
            # flatten last axis if weird shape
            arr = arr.reshape(arr.shape[0], -1)[:, 0]
        s = pd.Series(arr, index=index)

    # If series has array-like cells, extract first element
    if s.dtype == "object":
        def _first(v):
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else np.nan
            return v
        s = s.map(_first)

    # numeric coerce
    s = pd.to_numeric(s, errors="coerce")
    # ensure index if not provided
    if index is not None and not s.index.equals(index):
        s.index = index
    s.name = name if name is not None else getattr(s, "name", None)
    return s.astype(float)


# ─────────────────────────────────────────────────────────
# HELPER: normalize entire OHLCV dataframe to clean 1-D floats
# ─────────────────────────────────────────────────────────
def _normalize_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex columns (can happen on some yfinance versions), collapse
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(lv) for lv in col if lv is not None]).strip() for col in df.columns]

    # Standardize column names (prefer proper case, fallback to lower)
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # Build new frame with expected columns if possible
    out = pd.DataFrame(index=df.index)

    for want in _CORE_COLS:
        if want in df.columns:
            out[want] = _to_1d_series(df[want], index=df.index, name=want)
        elif want.lower() in lower_map:
            src = lower_map[want.lower()]
            out[want] = _to_1d_series(df[src], index=df.index, name=want)
        else:
            # if missing High/Low we can synthesize from Close as a last resort
            if want in ("High", "Low") and ("Close" in df.columns or "close" in lower_map):
                src = "Close" if "Close" in df.columns else lower_map["close"]
                base = _to_1d_series(df[src], index=df.index, name=want)
                if want == "High":
                    out[want] = base * 1.0005
                else:
                    out[want] = base * 0.9995
            elif want == "Adj Close" and ("Adj Close" in df.columns or "adj close" in lower_map):
                src = "Adj Close" if "Adj Close" in df.columns else lower_map["adj close"]
                out[want] = _to_1d_series(df[src], index=df.index, name=want)
            elif want == "Close" and ("close" in lower_map):
                src = lower_map["close"]
                out[want] = _to_1d_series(df[src], index=df.index, name=want)
            else:
                # create empty column of NaN to keep shape; will be dropped if unusable
                out[want] = np.nan

    # Drop rows with no Close
    out["Close"] = _to_1d_series(out["Close"], index=out.index, name="Close")
    out = out.dropna(subset=["Close"])

    # Ensure strictly increasing, unique index
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Basic return
    out["Return"] = out["Close"].pct_change()

    # Volume if missing: fill with zeros
    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype(float)

    return out


# ─────────────────────────────────────────────────────────
# FETCH DATA (Resilient + Flatten Fix + Cache)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol, interval="1h", period="1mo", retries=3, delay=3):
    """
    Fetch market data with retries and multiple fallbacks, then normalize
    so all columns are 1-D float Series. This *prevents* the
    'arg must be a list, tuple, 1-d array, or Series' and 'Data must be 1-D' errors.
    """
    def _attempt_fetch(_period, _interval):
        df = yf.download(
            symbol,
            period=_period,
            interval=_interval,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        return _normalize_ohlcv(df)

    # Primary attempts
    for attempt in range(1, retries + 1):
        try:
            df = _attempt_fetch(period, interval)
            if not df.empty and len(df) > 20:
                print(f"✅ {symbol}: fetched {len(df)} rows ({interval}, {period}).")
                return df
            else:
                print(f"⚠️ Attempt {attempt}: empty for {symbol} ({interval}, {period})")
        except Exception as e:
            print(f"❌ Attempt {attempt} failed for {symbol}: {e}")
        time.sleep(delay + random.random())

    # Fallback 1: same interval, shorter period
    try:
        alt_period = "5d" if interval in ("1m", "2m", "5m", "15m") else "1mo"
        df = _attempt_fetch(alt_period, interval)
        if not df.empty:
            print(f"🔁 Fallback period worked for {symbol}: {interval}, {alt_period}")
            return df
    except Exception as e:
        print(f"🚫 Fallback period failed for {symbol}: {e}")

    # Fallback 2: daily bars
    try:
        df = _attempt_fetch("3mo", "1d")
        if not df.empty:
            print(f"🔁 Fallback to daily worked for {symbol}.")
            return df
    except Exception as e:
        print(f"🚫 Fallback daily failed for {symbol}: {e}")

    # Fallback 3: raw chart endpoint (daily)
    try:
        print(f"🛰 Curl backup for {symbol}...")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            js = r.json()["chart"]["result"][0]
            quote = js["indicators"]["quote"][0]
            tmp = pd.DataFrame({
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            })
            tmp["Date"] = pd.to_datetime(js["timestamp"], unit="s")
            tmp.set_index("Date", inplace=True)
            df = _normalize_ohlcv(tmp)
            if not df.empty:
                print(f"✅ Curl backup succeeded for {symbol}")
                return df
    except Exception as e:
        print(f"🚫 Curl backup failed for {symbol}: {e}")

    print(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# INDICATORS (guarded with 1-D inputs)
# ─────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = _normalize_ohlcv(df).copy()
    if df.empty:
        return pd.DataFrame()

    # strict 1-D inputs
    close = _to_1d_series(df["Close"], index=df.index, name="Close")
    high = _to_1d_series(df.get("High", close * 1.0005), index=df.index, name="High")
    low = _to_1d_series(df.get("Low", close * 0.9995), index=df.index, name="Low")

    # indicators
    try:
        df["EMA_20"] = _to_1d_series(EMAIndicator(close, window=20).ema_indicator(), index=df.index, name="EMA_20")
        df["EMA_50"] = _to_1d_series(EMAIndicator(close, window=50).ema_indicator(), index=df.index, name="EMA_50")
        df["RSI"] = _to_1d_series(RSIIndicator(close, window=14).rsi(), index=df.index, name="RSI")
        macd = MACD(close)
        df["MACD"] = _to_1d_series(macd.macd(), index=df.index, name="MACD")
        df["Signal_Line"] = _to_1d_series(macd.macd_signal(), index=df.index, name="Signal_Line")
        bb = BollingerBands(close)
        df["BB_High"] = _to_1d_series(bb.bollinger_hband(), index=df.index, name="BB_High")
        df["BB_Low"] = _to_1d_series(bb.bollinger_lband(), index=df.index, name="BB_Low")
    except Exception as e:
        # If TA fails for any reason, return empty to skip this asset gracefully
        print(f"⚠️ Indicator computation failed: {e}")
        return pd.DataFrame()

    # ensure finiteness
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ─────────────────────────────────────────────────────────
# TRADING THEORY OVERLAY (unchanged behavior)
# ─────────────────────────────────────────────────────────
def apply_trading_theory(pred_label, df):
    latest = df.iloc[-1]
    ema_trend = latest["EMA_20"] > latest["EMA_50"]
    rsi_ok = 40 < latest["RSI"] < 70
    macd_conf = latest["MACD"] > latest["Signal_Line"]
    bb_breakout = (latest["Close"] > latest["BB_High"]) or (latest["Close"] < latest["BB_Low"])
    score = sum([ema_trend, rsi_ok, macd_conf, bb_breakout])

    if pred_label == "buy" and score >= 2:
        return "buy", 0.95
    elif pred_label == "sell" and (not ema_trend) and score >= 2:
        return "sell", 0.95
    else:
        return "neutral", 0.6


# ─────────────────────────────────────────────────────────
# MODEL TRAINING & PREDICTION (keeps accuracy)
# ─────────────────────────────────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if df is None or df.empty or len(df) < 60:
        return None

    # Target: next-bar up/down
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    FEATURES = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    # align y with X index
    y = df.loc[X.index, "Target"]

    if len(X) < 30:
        return None

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X.iloc[:-1], y.iloc[:-1])

    latest_row = X.iloc[-1].values.reshape(1, -1)
    pred_cls = clf.predict(latest_row)[0]
    pred_proba = clf.predict_proba(latest_row)[0][pred_cls]
    raw_pred = "buy" if pred_cls == 1 else "sell"

    # Overlay with trading theory
    adjusted_pred, conf_adj = apply_trading_theory(raw_pred, df)
    confidence = float(np.clip(pred_proba * conf_adj, 0.0, 1.0))

    # TP/SL using ATR-ish proxy
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    atr = float(atr) if np.isfinite(atr) and atr > 0 else float(df["Close"].pct_change().rolling(14).std().iloc[-1] * df["Close"].iloc[-1])
    mult = float(RISK_MULT.get(risk, 1.0))
    price = float(df["Close"].iloc[-1])

    if adjusted_pred == "buy":
        tp = price + atr * 1.5 * mult
        sl = price - atr * 1.0 * mult
    elif adjusted_pred == "sell":
        tp = price - atr * 1.5 * mult
        sl = price + atr * 1.0 * mult
    else:
        tp = price
        sl = price

    accuracy = float(clf.score(X, y))

    return {
        "prediction": adjusted_pred,
        "probability": confidence,
        "accuracy": accuracy,
        "tp": float(tp),
        "sl": float(sl),
        "model": clf,
        "features": FEATURES,
        "X": X,  # for scenarios tab
        "df": df,  # enriched with indicators
    }


# ─────────────────────────────────────────────────────────
# BACKTESTING (kept as before)
# ─────────────────────────────────────────────────────────
def backtest_signals(df, pred_dict):
    """
    Simple constant-position backtest from last prediction:
      - if 'buy' -> long next bars
      - if 'sell' -> short next bars
    """
    if df is None or df.empty or pred_dict is None:
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    df = _normalize_ohlcv(df).copy()
    if df.empty or "Return" not in df.columns:
        return {"winrate": 0.0, "total_return": 0.0, "equity_curve": pd.Series(dtype=float)}

    sig = 1 if pred_dict.get("prediction") == "buy" else -1 if pred_dict.get("prediction") == "sell" else 0
    if sig == 0:
        eq = (1 + df["Return"].fillna(0)).cumprod()
        return {"winrate": 0.5, "total_return": eq.iloc[-1] - 1, "equity_curve": eq}

    df["Signal"] = sig
    df["Strat_Return"] = df["Signal"].shift(1).fillna(0) * df["Return"].fillna(0)
    df["Equity"] = (1 + df["Strat_Return"]).cumprod()

    wins = (df["Strat_Return"] > 0).sum()
    total = max(1, (df["Strat_Return"].notna()).sum())
    winrate = wins / total
    total_ret = df["Equity"].iloc[-1] - 1

    return {"winrate": float(winrate), "total_return": float(total_ret), "equity_curve": df["Equity"]}


# ─────────────────────────────────────────────────────────
# SUMMARY (with progress & detailed log)
# ─────────────────────────────────────────────────────────
def summarize_assets():
    results = []
    progress = st.progress(0.0)
    status = st.empty()
    total = len(ASSET_SYMBOLS)
    done = 0
    log = ""

    for asset, symbol in ASSET_SYMBOLS.items():
        done += 1
        log += f"⏳ Fetching **{asset}** ({symbol})...\n"
        status.markdown(log)
        progress.progress(min(1.0, done / total))

        df = fetch_data(symbol, "1h", "1mo")
        if df.empty:
            log += f"⚠️ No data for **{asset}**, skipped.\n"
            status.markdown(log)
            continue

        try:
            pred = train_and_predict(df)
            if not pred:
                log += f"⚠️ Could not generate prediction for **{asset}**.\n"
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

        time.sleep(0.2)

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