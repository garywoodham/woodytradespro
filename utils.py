# utils.py — FINAL SMART v2 FULL VERSION (CLEAN & STABLE)
# ---------------------------------------------------------------------------
# WoodyTradesPro Forecast Utilities
# ---------------------------------------------------------------------------
# Features:
#   - Data fetch & cache (yfinance)
#   - Technical indicators (EMA, RSI, MACD, ATR)
#   - Vader sentiment with safe Yahoo fallback
#   - Market regime detection (trend/range)
#   - RandomForest ML classifier (adaptive bias)
#   - Fused signal (technicals + sentiment + ML)
#   - Backtest (win rate & total return)
#   - Full pipeline functions for app.py
# ---------------------------------------------------------------------------

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

# ---------------------------------------------------------------------------
# SUPPRESS EXCESS LOGGING & WARNINGS
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = MACD = RSIIndicator = AverageTrueRange = None

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INTERVALS = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h":  {"interval": "60m", "period": "2mo", "min_rows": 200},
    "4h":  {"interval": "240m", "period": "3mo", "min_rows": 200},
    "1d":  {"interval": "1d", "period": "1y", "min_rows": 150},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 100},
}

RISK_MULT = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

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

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------
def _cache_path(symbol: str, interval_key: str) -> Path:
    safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return DATA_DIR / f"{safe}_{interval_key}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()


def fetch_data(symbol: str, interval_key: str = "1h", use_cache=True) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(interval_key)
    interval = INTERVALS[interval_key]["interval"]
    period = INTERVALS[interval_key]["period"]
    min_rows = INTERVALS[interval_key]["min_rows"]
    cache_fp = _cache_path(symbol, interval_key)

    if use_cache and cache_fp.exists():
        try:
            df = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            if len(df) >= min_rows:
                return _normalize(df)
        except Exception:
            pass

    for _ in range(3):
        try:
            raw = yf.download(symbol, period=period, interval=interval,
                              progress=False, threads=False, auto_adjust=True)
            df = _normalize(raw)
            if len(df) >= min_rows:
                df.to_csv(cache_fp)
                return df
        except Exception:
            time.sleep(2)

    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True)
        df = _normalize(raw)
        if not df.empty:
            df.to_csv(cache_fp)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    try:
        df["ema20"] = EMAIndicator(df["Close"], 20).ema_indicator()
        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    except Exception:
        df["ema20"] = df["Close"].ewm(span=20).mean()
        df["ema50"] = df["Close"].ewm(span=50).mean()
    try:
        df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    except Exception:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
    try:
        macd = MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception:
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
    try:
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14)
        df["atr"] = atr.average_true_range()
    except Exception:
        tr = pd.concat([
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
    return df.ffill().bfill()

# ---------------------------------------------------------------------------
# SENTIMENT, REGIME, ML
# ---------------------------------------------------------------------------
_sent = SentimentIntensityAnalyzer()

def fetch_sentiment(symbol: str) -> float:
    """Safely fetch sentiment; return 0 on failure."""
    try:
        tk = yf.Ticker(symbol)
        news = getattr(tk, "news", None)
        if not news:
            return 0.0
        scores = []
        for n in news:
            title = n.get("title") or ""
            if not title:
                
def detect_regime(df: pd.DataFrame) -> str:
    if "ema20" not in df or "ema50" not in df:
        return "range"
    spread = (df["ema20"] - df["ema50"]) / df["Close"]
    volatility = df["Close"].pct_change().rolling(20).std()
    trend_strength = spread.abs().mean()
    vol_level = volatility.mean()
    if trend_strength > 0.0025 and vol_level > 0.0015:
        return "trend"
    return "range"


def train_ml_classifier(df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """Train RandomForest to classify next-bar direction from indicators."""
    if len(df) < 200:
        return None
    df = df.copy()
    df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    features = ["ema20", "ema50", "RSI", "macd", "macd_signal"]
    df = df.dropna(subset=features + ["target"])
    X, y = df[features], df["target"]
    if len(X) < 50:
        return None
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y)
    return clf


def ml_predict_next(df: pd.DataFrame, clf: RandomForestClassifier) -> Optional[float]:
    """Return probability of upward move (0–1)."""
    try:
        latest = df[["ema20", "ema50", "RSI", "macd", "macd_signal"]].iloc[-1:].dropna()
        if latest.empty:
            return None
        prob = clf.predict_proba(latest)[0][1]
        return float(prob)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# SIGNAL FUSION
# ---------------------------------------------------------------------------
def compute_fused_signal(df: pd.DataFrame, sentiment: float, ml_prob: float, regime: str) -> Tuple[str, float]:
    """Combine technical, sentiment, and ML into final signal."""
    if len(df) < 2:
        return "Hold", 0.0
    row_prev, row = df.iloc[-2], df.iloc[-1]
    score, votes = 0.0, 0

    # EMA trend
    if pd.notna(row["ema20"]) and pd.notna(row["ema50"]):
        votes += 1
        score += 1 if row["ema20"] > row["ema50"] else -1

    # RSI
    if pd.notna(row["RSI"]):
        votes += 1
        if row["RSI"] < 30:
            score += 1
        elif row["RSI"] > 70:
            score -= 1

    # MACD cross
    if all(pd.notna(x) for x in [row["macd"], row["macd_signal"], row_prev["macd"], row_prev["macd_signal"]]):
        votes += 1
        crossed_up = (row_prev["macd"] <= row_prev["macd_signal"]) and (row["macd"] > row["macd_signal"])
        crossed_dn = (row_prev["macd"] >= row_prev["macd_signal"]) and (row["macd"] < row["macd_signal"])
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1

    # Normalize technical signal
    tech_bias = 0.0 if votes == 0 else score / votes

    # Blend sentiment (0–1) and ML probability (0–1)
    fused = 0.5 * tech_bias + 0.3 * sentiment + 0.2 * (ml_prob - 0.5 if ml_prob is not None else 0)

    # Regime adjustment
    if regime == "range":
        fused *= 0.8  # less aggressive in sideways markets
    elif regime == "trend":
        fused *= 1.2

    if fused > 0.25:
        return "Buy", min(1.0, abs(fused))
    elif fused < -0.25:
        return "Sell", min(1.0, abs(fused))
    return "Hold", 1 - abs(fused)

# ---------------------------------------------------------------------------
# TP/SL CALCULATION
# ---------------------------------------------------------------------------
def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    r = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp = price + r["tp_atr"] * atr if side == "Buy" else price - r["tp_atr"] * atr
    sl = price - r["sl_atr"] * atr if side == "Buy" else price + r["sl_atr"] * atr
    return float(tp), float(sl)

# ---------------------------------------------------------------------------
# LATEST PREDICTION PIPELINE
# ---------------------------------------------------------------------------
def latest_prediction(df: pd.DataFrame, symbol: str, risk: str = "Medium") -> Optional[Dict[str, object]]:
    if df is None or len(df) < 80:
        return None
    df = add_indicators(df)
    sentiment = fetch_sentiment(symbol)
    regime = detect_regime(df)
    clf = train_ml_classifier(df)
    ml_prob = ml_predict_next(df, clf) if clf else 0.5
    side, prob = compute_fused_signal(df, sentiment, ml_prob, regime)
    atr_val = float(df["atr"].iloc[-1]) if "atr" in df else 0.0
    price = float(df["Close"].iloc[-1])
    tp, sl = (None, None)
    if side != "Hold":
        tp, sl = compute_tp_sl(price, atr_val, side, risk)
    return {
        "side": side,
        "prob": round(prob * 100.0, 2),
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment,
        "regime": regime,
        "ml_prob": ml_prob,
    }

# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def backtest_signals(df: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    out = {"win_rate": 0.0, "total_return_pct": 0.0, "n_trades": 0, "trades": []}
    if df.empty or len(df) < 200:
        return out
    df = add_indicators(df)
    signals, prev = [], df.iloc[0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        if i < 2:
            continue
        side, _ = compute_fused_signal(df.iloc[:i], 0, 0.5, detect_regime(df.iloc[:i]))
        atr = df["atr"].iloc[i] if "atr" in df else 0.0
        tp, sl = compute_tp_sl(row["Close"], atr, side, risk)
        signals.append((df.index[i], side, row["Close"], tp, sl))

    pos = None
    eq, wins, trades = 0.0, 0, []
    for ts, side, px, tp, sl in signals:
        if pos is None and side in ("Buy", "Sell"):
            pos = (side, px, tp, sl, ts)
            continue
        if pos:
            side0, entry, tp0, sl0, t0 = pos
            hit_tp = (side0 == "Buy" and px >= tp0) or (side0 == "Sell" and px <= tp0)
            hit_sl = (side0 == "Buy" and px <= sl0) or (side0 == "Sell" and px >= sl0)
            if hit_tp or hit_sl or side != "Hold":
                ret = (px - entry) / entry * (1 if side0 == "Buy" else -1)
                wins += 1 if ret > 0 else 0
                eq += ret
                trades.append({"entry": entry, "exit": px, "side": side0,
                               "return_pct": ret * 100, "entry_time": t0, "exit_time": ts})
                pos = None
    n = len(trades)
    if n > 0:
        out["win_rate"] = 100 * wins / n
        out["total_return_pct"] = 100 * eq
        out["n_trades"] = n
        out["trades"] = trades
    return out

# ---------------------------------------------------------------------------
# PIPELINES
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str, interval_key: str, risk: str = "Medium", use_cache=True) -> Optional[Dict[str, object]]:
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None
    df = add_indicators(df)
    pred = latest_prediction(df, symbol, risk)
    if not pred:
        return None
    bt = backtest_signals(df, risk)
    return {
        "symbol": symbol,
        "interval_key": interval_key,
        "risk": risk,
        "price": float(df["Close"].iloc[-1]),
        "signal": pred["side"],
        "probability": pred["prob"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "total_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "ml_prob": pred["ml_prob"],
        "df": df,
        "trades": bt["trades"],
    }

def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache=True) -> pd.DataFrame:
    _log("Fetching and analyzing market data (smart v2)...")
    rows = []
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset} ({symbol})...")
        res = analyze_asset(symbol, interval_key, risk, use_cache)
        if not res:
            continue
        rows.append({
            "Asset": asset,
            "Symbol": symbol,
            "Signal": res["signal"],
            "Probability_%": res["probability"],
            "Price": res["price"],
            "TP": res["tp"],
            "SL": res["sl"],
            "WinRate_%": res["win_rate"],
            "BacktestReturn_%": res["total_return_pct"],
            "Trades": res["n_trades"],
            "Sentiment": res["sentiment"],
            "Regime": res["regime"],
        })
    return pd.DataFrame(rows)

def load_asset_with_indicators(asset: str, interval_key: str, use_cache=True) -> Tuple[str, pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)
    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key, use_cache)
    return symbol, add_indicators(df)

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache=True):
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None, pd.DataFrame()
    df = add_indicators(df)
    pred = latest_prediction(df, symbol, risk)
    bt = backtest_signals(df, risk)
    return {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": pred["prob"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "ml_prob": pred["ml_prob"],
        "trades": bt["trades"],
    }, df

# ---------------------------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------------------------