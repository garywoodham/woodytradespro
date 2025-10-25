# ======================================================================================
# utils.py
# Smart v7 Full Runtime (Historical + ML + Sentiment + Adaptive Backtest)
#
# This file is intentionally verbose.
# It merges:
#  - data caching + normalization
#  - indicator pipelines
#  - signal logic (rule-based & ML hybrid)
#  - sentiment engine w/ fallback
#  - TP/SL, RR, risk scaling
#  - adaptive fusion model with dynamic thresholds
#  - rolling backtest with win rate, return, max drawdown, sharpe-like ratio
#  - asset summary table builder
#  - Streamlit-friendly helper functions
#  - debug/status utilities
#
# Absolutely nothing is intentionally dropped:
# - We still compute and expose probability, sentiment, TP/SL, RR, win rate, return, drawdown.
# - We compute indicators like EMA20/50/100, RSI, MACD, ATR, ADX-like proxy, Bollinger width,
#   relative ATR, regime duration, etc.
# - We produce short-horizon ML classification to predict 3-bar direction.
# - Sentiment has multi-source fallback and synthetic mode.
# - "Hold" entries still get populated outputs (TP/SL etc.).
#
# Notes:
#  - This module does NOT import from app.py (no circular import).
#  - Streamlit is imported only for caching decorators and logging convenience.
#  - All functions are pure / stable enough to be used across tabs.
# ======================================================================================

from __future__ import annotations

import os
import time
import math
import json
import traceback
import random
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

import yfinance as yf
import requests
import xml.etree.ElementTree as ET

# optional: scikit-learn for ML model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# for app caches (safe if running under Streamlit)
try:
    import streamlit as st
except Exception:
    st = None  # allows running utils standalone from CLI

# --------------------------------------------------------------------------------------
# GLOBAL CONSTANTS / CONFIG
# --------------------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# timeframes -> yfinance interval + historical lookback period + required rows
INTERVAL_CONFIG: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m",  "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "60m",  "period": "2mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",   "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk",  "period": "5y",  "min_rows": 150},
}

# risk multipliers (the "old" model)
RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

# asset universe
ASSET_SYMBOLS: Dict[str, str] = {
    "Gold": "GC=F",
    "NASDAQ 100": "^NDX",
    "S&P 500": "^GSPC",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
}

# Analyzer instances
_VADER = SentimentIntensityAnalyzer()

# ML model cache keyed by (symbol, interval_key, regime_label)
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}

# Fallback base thresholds for adaptive entry
_BASE_PROB_THRESHOLD = 0.6
_BASE_MIN_RR = 1.2

# ======================================================================================
# UTILITY / LOGGING
# ======================================================================================

def _log(msg: str) -> None:
    """Safe print to console (works in Streamlit logs)."""
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def _now_ts() -> float:
    return time.time()

# ======================================================================================
# DATA FETCHING / CACHING
# ======================================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    """Generate stable CSV cache path for a (symbol, interval_key) pair."""
    safe = (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return DATA_DIR / f"{safe}_{interval_key}.csv"

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Unify yfinance output into a clean OHLCV DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # rename weird lowercase columns, etc.
    rename_map = {}
    for c in df.columns:
        if c.lower() == "adj close":
            rename_map[c] = "Adj Close"
        elif c.lower() == "close":
            rename_map[c] = "Close"
        elif c.lower() == "open":
            rename_map[c] = "Open"
        elif c.lower() == "high":
            rename_map[c] = "High"
        elif c.lower() == "low":
            rename_map[c] = "Low"
        elif c.lower() == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep_cols:
        return pd.DataFrame()
    df = df[keep_cols].copy()

    # force datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # flatten any 2D columns
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        # numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)
    return df

def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Primary fetch using yf.download."""
    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            threads=False,
        )
        return _normalize_ohlcv(raw)
    except Exception as e:
        _log(f"⚠️ {symbol}: yf.download error {e}")
        return pd.DataFrame()

def _yahoo_mirror_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Secondary fetch via Ticker.history (fallback)."""
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        raw2 = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        return _normalize_ohlcv(raw2)
    except Exception as e:
        _log(f"⚠️ {symbol}: mirror fetch error {e}")
        return pd.DataFrame()

def fetch_data(
    symbol: str,
    interval_key: str = "1h",
    use_cache: bool = True,
    max_retries: int = 4,
    backoff_range: Tuple[float, float] = (2.0, 6.0),
) -> pd.DataFrame:
    """
    Unified robust fetch:
     1. Attempt cache if adequate
     2. Attempt yf.download with retries
     3. Attempt Ticker.history fallback
     4. Cache success result
    """
    if interval_key not in INTERVAL_CONFIG:
        raise KeyError(f"Unknown interval_key={interval_key}, valid={list(INTERVAL_CONFIG.keys())}")

    interval = str(INTERVAL_CONFIG[interval_key]["interval"])
    period = str(INTERVAL_CONFIG[interval_key]["period"])
    min_rows = int(INTERVAL_CONFIG[interval_key]["min_rows"])

    cache_fp = _cache_path(symbol, interval_key)

    # cache path attempt
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ Cache {symbol} is short ({len(cached)}/{min_rows}).")
        except Exception as e:
            _log(f"⚠️ Cache read fail for {symbol}: {e}")

    # live fetch attempts
    _log(f"⏳ Fetching {symbol} [{interval}] ...")
    for attempt in range(1, max_retries + 1):
        df_live = _yahoo_try_download(symbol, interval, period)
        if not df_live.empty and len(df_live) >= min_rows:
            _log(f"✅ {symbol}: fetched {len(df_live)} rows.")
            try:
                df_live.to_csv(cache_fp)
                _log(f"💾 Cached → {cache_fp}")
            except Exception as e:
                _log(f"⚠️ Cache write fail for {symbol}: {e}")
            return df_live

        got = len(df_live) if isinstance(df_live, pd.DataFrame) else "N/A"
        _log(f"⚠️ Retry {attempt} failed for {symbol} ({got} rows).")
        low, high = backoff_range
        time.sleep(np.random.uniform(low, high))

    # fallback
    _log(f"🪞 Mirror fetch for {symbol}...")
    df_m = _yahoo_mirror_history(symbol, interval, period)
    if not df_m.empty and len(df_m) >= min_rows:
        _log(f"✅ Mirror fetch success for {symbol}.")
        try:
            df_m.to_csv(cache_fp)
            _log(f"💾 Cached mirror → {cache_fp}")
        except Exception as e:
            _log(f"⚠️ Cache write fail for {symbol}: {e}")
        return df_m

    _log(f"🚫 All fetch attempts failed for {symbol}. Returning empty frame.")
    return pd.DataFrame()

# ======================================================================================
# TECHNICAL INDICATORS / FEATURE ENGINEERING
# ======================================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Lightweight ADX-ish proxy.
    We'll approximate directional movement velocity vs ATR.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_local = _atr_series(high, low, close, window)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean() / atr_local
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean() / atr_local
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    return adx

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds full indicator set:
      ema20, ema50, ema100
      RSI
      MACD, MACD signal, MACD hist
      ATR (14)
      ATR relative (atr/close)
      Bollinger %B, bandwidth
      Rate of change (ROC)
      ADX proxy
      stretch from ema20 measured in ATRs
      trend_age / bars since flip
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ensure numeric columns
    for col in ["Close", "High", "Low", "Open"]:
        if col not in df.columns:
            return pd.DataFrame()
        arr = df[col].values
        if getattr(arr, "ndim", 1) > 1:
            df[col] = pd.Series(arr.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- EMAs
    df["ema20"] = _ema(df["Close"], 20)
    df["ema50"] = _ema(df["Close"], 50)
    df["ema100"] = _ema(df["Close"], 100)

    # -- RSI
    df["RSI"] = _rsi_series(df["Close"], 14)
    df["rsi"] = df["RSI"]  # backward compat older code

    # -- MACD
    ema12 = _ema(df["Close"], 12)
    ema26 = _ema(df["Close"], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # -- ATR
    df["atr"] = _atr_series(df["High"], df["Low"], df["Close"], 14)
    df["atr_rel"] = df["atr"] / df["Close"]

    # -- Bollinger
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_percent_b"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)

    # -- Rate of change (momentum short-term)
    df["roc_close"] = df["Close"].pct_change(5)

    # -- ADX proxy
    df["adx"] = _adx_proxy(df["High"], df["Low"], df["Close"], 14)

    # stretch above/below EMA20 measured in ATR
    df["close_above_ema20_atr"] = (df["Close"] - df["ema20"]) / df["atr"].replace(0, np.nan)

    # trend age calculation
    # We'll consider an uptrend if ema20>ema50, else downtrend
    trend_flag = (df["ema20"] > df["ema50"]).astype(int)  # 1 bull, 0 bear
    age = []
    current_age = 0
    last_flag = None
    for val in trend_flag:
        if last_flag is None:
            current_age = 1
        else:
            if val == last_flag:
                current_age += 1
            else:
                current_age = 1
        age.append(current_age)
        last_flag = val
    df["trend_age"] = age

    # This ensures all inf -> NaN -> filled
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df

# ======================================================================================
# SIGNAL ENGINE
# ======================================================================================

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Heuristic ensemble:
      - EMA20 vs EMA50 alignment + MACD cross => directional bias
      - RSI extremes confirm mean-revert edges
      - We also require some volatility/structure (atr_rel, adx)

    Returns:
      - side: "Buy", "Sell", or "Hold"
      - base_conf: 0..1
    """
    side = "Hold"
    score = 0.0
    votes = 0

    # Trend bias via EMAs
    if (not math.isnan(row["ema20"])) and (not math.isnan(row["ema50"])):
        votes += 1
        if row["ema20"] > row["ema50"]:
            score += 1
        elif row["ema20"] < row["ema50"]:
            score -= 1

    # RSI extremes (mean reversion tilt)
    if not math.isnan(row["RSI"]):
        votes += 1
        if row["RSI"] < 30:
            score += 1
        elif row["RSI"] > 70:
            score -= 1

    # MACD cross
    if (
        (not math.isnan(row["macd"])) and
        (not math.isnan(row["macd_signal"])) and
        (not math.isnan(row_prev.get("macd", np.nan))) and
        (not math.isnan(row_prev.get("macd_signal", np.nan)))
    ):
        votes += 1
        crossed_up = (row_prev["macd"] <= row_prev["macd_signal"]) and (row["macd"] > row["macd_signal"])
        crossed_dn = (row_prev["macd"] >= row_prev["macd_signal"]) and (row["macd"] < row["macd_signal"])
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1

    # "Volatility / structure present" gate
    adx_ok = _safe_float(row.get("adx"), 0.0) > 12
    atr_rel_ok = _safe_float(row.get("atr_rel"), 0.0) >= 0.6

    # base confidence
    base_conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)

    # decision
    if score > 0.67 * votes and adx_ok and atr_rel_ok:
        side = "Buy"
    elif score < -0.67 * votes and adx_ok and atr_rel_ok:
        side = "Sell"
    else:
        side = "Hold"

    return side, base_conf

def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    Overextension filter:
    - if price is stretched too far from EMA20 in ATR terms for too long,
      we avoid chasing continuation in that same direction.
    """
    stretch = _safe_float(row.get("close_above_ema20_atr"), 0.0)
    trend_age = _safe_float(row.get("trend_age"), 0.0)
    if side == "Buy" and stretch > 2 and trend_age > 30:
        return True
    if side == "Sell" and stretch < -2 and trend_age > 30:
        return True
    return False

# ======================================================================================
# TP/SL + RISK LOGIC
# ======================================================================================

def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Older "RISK_MULT" ATR-based model.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])

    # fallback atr
    if atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        atr = price * 0.005

    if side == "Buy":
        tp = price + tp_k * atr
        sl = price - sl_k * atr
    elif side == "Sell":
        tp = price - tp_k * atr
        sl = price + sl_k * atr
    else:
        # neutral/hold fallback
        tp = price * 1.005
        sl = price * 0.995

    return float(tp), float(sl)

def compute_tp_sl(close_price: float, atr_val: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Wrapper to keep backward compatibility with versions that call compute_tp_sl directly.
    """
    return _compute_tp_sl(close_price, atr_val, side, risk)

def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Reward:Risk ratio for the planned trade.
    """
    if side == "Sell":
        reward = price - tp
        risk_amt = sl - price
    else:  # Buy or Hold fallback math
        reward = tp - price
        risk_amt = price - sl
    if risk_amt <= 0:
        return 0.0
    return float(reward / risk_amt)

# ======================================================================================
# SENTIMENT ENGINE
# ======================================================================================

@lru_cache(maxsize=64)
def _fetch_sentiment(symbol: str) -> float:
    """
    Multi-layer sentiment logic:
      1. yfinance Ticker.news headlines via VADER
      2. Google News RSS fallback
      3. Synthetic price-derived "sentiment" if no news
    Returns float in [-1, 1].
    """
    scores: List[float] = []

    # 1. Yahoo Finance news
    try:
        tk = yf.Ticker(symbol)
        news_items = tk.news or []
        for n in news_items[:8]:
            title = n.get("title", "")
            if not title:
                continue
            compound = _VADER.polarity_scores(title)["compound"]
            scores.append(compound)
    except Exception:
        pass

    # 2. Google RSS fallback
    if not scores:
        try:
            rss_url = f"https://news.google.com/rss/search?q={symbol}+finance"
            xml_raw = requests.get(rss_url, timeout=5).text
            root = ET.fromstring(xml_raw)
            for item in root.findall(".//item")[:8]:
                headline = item.findtext("title", "") or ""
                if headline:
                    compound = _VADER.polarity_scores(headline)["compound"]
                    scores.append(compound)
        except Exception:
            pass

    # 3. synthetic fallback = slope/momentum-based
    if not scores:
        tmp = fetch_data(symbol, "1h", use_cache=True)
        if not tmp.empty and len(tmp) >= 20:
            closes = tmp["Close"].iloc[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            mom = tmp["Close"].pct_change().tail(14).mean()
            guess = np.tanh((slope * 1000 + mom * 50) / 2.0)
            scores.append(float(guess))

    if not scores:
        return 0.0

    # exponential smoothing to weight recent more
    alpha = 0.3
    smoothed = scores[0]
    for s in scores[1:]:
        smoothed = alpha * s + (1 - alpha) * smoothed

    smoothed = float(np.clip(smoothed, -1.0, 1.0))
    return smoothed

# ======================================================================================
# ML FEATURE EXTRACTION / TRAINING / INFERENCE
# ======================================================================================

def _extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core ML feature set. Must match what we train/predict on.
    """
    feat_cols = [
        "ema20",
        "ema50",
        "ema100",
        "ema_gap",
        "RSI",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "atr_rel",
        "bb_percent_b",
        "bb_bandwidth",
        "adx",
        "roc_close",
        "close_above_ema20_atr",
        "trend_age",
    ]
    # guarantee existence
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df[feat_cols].fillna(0.0)

def _label_regime(df: pd.DataFrame) -> str:
    """
    regime label, e.g. bull or bear, based on ema20 > ema50
    """
    if df.empty:
        return "neutral"
    last_gap = df["ema20"].iloc[-1] - df["ema50"].iloc[-1]
    if last_gap >= 0:
        return "bull"
    else:
        return "bear"

def _train_ml_model(symbol: str, interval_key: str, df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train a regime-specific model:
    - target: will price be higher 3 bars ahead?
    - we re-train per regime (bull/bear) and cache model in _MODEL_CACHE
    """
    if df.empty or len(df) < 120:
        return None

    regime_lbl = _label_regime(df)
    cache_key = (symbol, interval_key, regime_lbl)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Build classification target
    df_local = df.copy()
    df_local["target_up3"] = (df_local["Close"].shift(-3) > df_local["Close"]).astype(int)

    feats = _extract_ml_features(df_local)
    y = df_local["target_up3"]

    if y.nunique() < 2:
        return None

    X_train, _, y_train, _ = train_test_split(feats, y, test_size=0.3, shuffle=False)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    _MODEL_CACHE[cache_key] = clf
    return clf

def _ml_predict_prob(model: Optional[RandomForestClassifier], row_feat: pd.Series) -> Optional[float]:
    """
    Predict up-move probability for single row.
    """
    if model is None:
        return None
    try:
        inp = row_feat.to_frame().T  # shape (1, n_features)
        proba = model.predict_proba(inp)[0][1]
        return float(proba)
    except Exception:
        return None

# ======================================================================================
# PROBABILITY FUSION / DYNAMIC THRESHOLDS
# ======================================================================================

def _fuse_prob(
    rule_prob: float,
    ml_prob: Optional[float],
    sentiment_score: float,
    adx_val: float,
    atr_pct: float,
    recent_winrate: float,
) -> float:
    """
    Weighted fusion:
      - rule_prob from compute_signal_row
      - ml_prob from _ml_predict_prob
      - sentiment (scaled)
      - regime stability via adx/atr
      - adapt weighting by recent win rate (backtest performance)
    """
    ml_weight = np.clip(recent_winrate / 100.0, 0.25, 0.75)
    if ml_prob is None:
        fused = rule_prob
    else:
        fused = ml_weight * ml_prob + (1 - ml_weight) * rule_prob

    # penalize high chaos (atr_pct big => reduce confidence)
    fused *= math.exp(-min(atr_pct * 5.0, 2.5))

    # sentiment only helps if ADX strong
    if adx_val >= 20:
        fused *= np.clip(1 + 0.2 * sentiment_score, 0.7, 1.4)

    fused = float(np.clip(fused, 0.0, 1.0))
    return fused

def _dynamic_cutoff(recent_winrate: float) -> float:
    """
    If strategy has been winning, we can tolerate slightly lower threshold.
    If it's losing, demand higher threshold.
    """
    raw = 0.55 + (0.65 - recent_winrate / 200.0)
    return float(np.clip(raw, 0.50, 0.75))

def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    Adjust entry requirements based on regime:
    - strong trending? lower prob ok but require better RR
    - choppy/slow? demand higher prob but allow lower RR
    """
    adx_val = _safe_float(row.get("adx"), 0.0)
    atr_rel = _safe_float(row.get("atr_rel"), 1.0)

    prob_thresh = _BASE_PROB_THRESHOLD
    rr_thresh = _BASE_MIN_RR

    if adx_val > 25 and atr_rel >= 1.0:
        prob_thresh -= 0.03
        rr_thresh += 0.2
    elif adx_val < 15 or atr_rel < 0.8:
        prob_thresh += 0.05
        rr_thresh -= 0.1

    prob_thresh = max(0.5, min(0.9, prob_thresh))
    rr_thresh = max(1.0, min(2.5, rr_thresh))

    return prob_thresh, rr_thresh

# ======================================================================================
# BACKTEST ENGINE
# ======================================================================================

def backtest_signals(df: pd.DataFrame, risk: str, horizon: int = 5) -> Dict[str, Any]:
    """
    Rolling pseudo-backtest:
      - iterate through bars, compute signal
      - simulate TP/SL hits within horizon bars
      - track wins, losses, running balance
      - compute winrate, total return %, max drawdown %, sharpe-like metric
    """
    out = {
        "winrate": 0.0,
        "trades": 0,
        "return": 0.0,
        "maxdd": 0.0,
        "sharpe": 0.0,
    }

    if df.empty or len(df) < 100:
        return out

    balance = 1.0
    peak = 1.0
    wins = 0
    losses = 0
    trades = 0
    drawdowns = []

    for i in range(60, len(df) - horizon):
        prev = df.iloc[i-1]
        row = df.iloc[i]

        side, base_conf = compute_signal_row(prev, row)
        if side == "Hold" or _is_exhausted(row, side):
            continue

        price = float(row["Close"])
        atr_here = float(row["atr"]) if "atr" in row and not math.isnan(row["atr"]) else price * 0.005
        tp, sl = _compute_tp_sl(price, atr_here, side, risk)

        # forward simulate horizon bars
        executed = False
        for j in range(1, horizon+1):
            nxt = df.iloc[i+j]
            nxt_px = float(nxt["Close"])

            # check hit conditions
            if side == "Buy":
                if nxt_px >= tp:
                    balance *= 1.01
                    wins += 1
                    executed = True
                    break
                if nxt_px <= sl:
                    balance *= 0.99
                    losses += 1
                    executed = True
                    break
            else:  # Sell
                if nxt_px <= tp:
                    balance *= 1.01
                    wins += 1
                    executed = True
                    break
                if nxt_px >= sl:
                    balance *= 0.99
                    losses += 1
                    executed = True
                    break

        if executed:
            trades += 1
            peak = max(peak, balance)
            dd = (peak - balance) / peak if peak > 0 else 0
            drawdowns.append(dd)

    total_ret = (balance - 1.0) * 100.0
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    maxdd = (max(drawdowns) * 100.0) if drawdowns else 0.0
    sharpe_like = (winrate / (maxdd + 1)) if maxdd > 0 else winrate

    out["winrate"] = round(winrate, 2)
    out["trades"] = trades
    out["return"] = round(total_ret, 2)
    out["maxdd"] = round(maxdd, 2)
    out["sharpe"] = round(sharpe_like, 2)
    return out

# ======================================================================================
# CORE PREDICTION PIPELINE
# ======================================================================================

def _latest_prediction(symbol: str, interval_key: str, risk: str, recent_winrate: float = 50.0, use_cache: bool = True) -> Dict[str, Any]:
    """
    Full "brain":
      - fetch & indicators
      - compute rule signal
      - compute sentiment
      - train/score ML model on 3-bar move
      - fuse probabilities (rule+ml+sentiment)
      - adapt thresholds
      - compute TP/SL, RR
      - produce final side, prob, tp/sl, rr, sentiment
      - handle Hold gracefully
    """

    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    if df_raw.empty or len(df_raw) < 60:
        return {"symbol": symbol, "side": "Hold", "probability": 0.5, "sentiment": 0.0,
                "tp": None, "sl": None, "rr": None}

    df = add_indicators(df_raw)
    if df.empty or len(df) < 60:
        return {"symbol": symbol, "side": "Hold", "probability": 0.5, "sentiment": 0.0,
                "tp": None, "sl": None, "rr": None}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # rule signal
    rule_side, rule_conf = compute_signal_row(prev, last)

    # sentiment
    sent_score = _fetch_sentiment(symbol)

    # ML model
    model = _train_ml_model(symbol, interval_key, df)
    feats_all = _extract_ml_features(df)
    row_feat = feats_all.iloc[-1]
    ml_prob = _ml_predict_prob(model, row_feat)

    # fused prob
    atr_pct = _safe_float(last["atr_rel"], 0.0)
    fused_prob = _fuse_prob(
        rule_prob=rule_conf,
        ml_prob=ml_prob,
        sentiment_score=sent_score,
        adx_val=_safe_float(last["adx"], 0.0),
        atr_pct=atr_pct,
        recent_winrate=recent_winrate,
    )

    # thresholds
    prob_thresh_regime, rr_thresh_regime = _adaptive_thresholds(last)
    prob_thresh_recent = _dynamic_cutoff(recent_winrate)
    # final probability threshold is the stricter of both
    live_prob_thresh = max(prob_thresh_regime, prob_thresh_recent)

    # compute TP/SL/RR
    price_now = float(last["Close"])
    atr_now = _safe_float(last["atr"], price_now * 0.005)
    tp, sl = _compute_tp_sl(price_now, atr_now, rule_side, risk)
    rr = _calc_rr(price_now, tp, sl, rule_side)

    exhausted = _is_exhausted(last, rule_side)
    if exhausted:
        fused_prob *= 0.85

    # final side decision
    final_side = "Hold"
    if rule_side != "Hold":
        if (fused_prob >= live_prob_thresh) and (rr >= rr_thresh_regime):
            final_side = rule_side

    # ensure "Hold" still shows tp/sl/rr and not zeros
    if final_side == "Hold":
        if tp is None or sl is None:
            tp = price_now * 1.005
            sl = price_now * 0.995
            rr = 1.0
        if sent_score == 0.0:
            # fallback to simple recent momentum if sentiment source was empty
            sent_score = float(df["Close"].pct_change().tail(10).mean())

    return {
        "symbol": symbol,
        "side": final_side,
        "probability": float(round(fused_prob, 3)),
        "sentiment": float(round(sent_score, 3)),
        "tp": float(round(tp, 4)),
        "sl": float(round(sl, 4)),
        "rr": float(round(rr, 3)),
    }

# ======================================================================================
# WRAPPERS FOR DASHBOARD / TABS
# ======================================================================================

def analyze_asset(symbol: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    High-level single-asset analysis pipeline:
      - fetch data
      - compute indicators
      - run backtest to get winrate etc
      - run latest_prediction using that winrate as context
    Returns dict that the app can display directly.
    """
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    back = backtest_signals(df_ind, risk)

    pred = _latest_prediction(
        symbol=symbol,
        interval_key=interval_key,
        risk=risk,
        recent_winrate=back["winrate"],
        use_cache=use_cache,
    )

    out = {
        "symbol": symbol,
        "side": pred.get("side", "Hold"),
        "probability": pred.get("probability", 0.0),
        "sentiment": pred.get("sentiment", 0.0),
        "tp": pred.get("tp", None),
        "sl": pred.get("sl", None),
        "rr": pred.get("rr", None),
        "winrate": back.get("winrate", 0.0),
        "trades": back.get("trades", 0),
        "return": back.get("return", 0.0),
        "maxdd": back.get("maxdd", 0.0),
        "sharpe": back.get("sharpe", 0.0),
    }
    return out

def summarize_assets(interval_key: str, risk: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Iterates over ASSET_SYMBOLS and builds a summary dataframe
    for the Market Summary tab.
    """
    rows = []
    _log("Fetching and analyzing market data (smart v7)...")
    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        try:
            res = analyze_asset(symbol, interval_key, risk, use_cache=use_cache)
            rows.append({
                "Asset": asset_name,
                "Side": res.get("side", "Hold"),
                "Probability": res.get("probability", 0.0),
                "Sentiment": res.get("sentiment", 0.0),
                "TP": res.get("tp"),
                "SL": res.get("sl"),
                "RR": res.get("rr"),
                "WinRate": res.get("winrate", 0.0),
                "Trades": res.get("trades", 0),
                "Return%": res.get("return", 0.0),
                "MaxDD%": res.get("maxdd", 0.0),
                "SharpeLike": res.get("sharpe", 0.0),
            })
        except Exception as e:
            _log(f"❌ Error analyzing {asset_name}: {e}")
            traceback.print_exc()

    if not rows:
        return pd.DataFrame()

    df_sum = pd.DataFrame(rows)

    # Ensure numeric cleaning
    num_cols = ["Probability", "Sentiment", "WinRate", "Trades", "Return%", "MaxDD%", "SharpeLike"]
    for c in num_cols:
        df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce").fillna(0.0)

    # sort by Probability desc
    df_sum = df_sum.sort_values("Probability", ascending=False).reset_index(drop=True)
    return df_sum

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For the Backtest tab / Scenario tab:
      - get data + indicators
      - run backtest
      - return df (for chart) and stats dict
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)

    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    back = backtest_signals(df_ind, risk)

    return df_ind, back

def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    For per-asset charting in Streamlit tabs.
    Returns (symbol, df_with_indicators).
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind

def asset_prediction_single(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    For 'Asset Analysis' tab: just produce latest signal dict.
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    back = backtest_signals(df_ind, risk)
    pred = _latest_prediction(symbol, interval_key, risk, recent_winrate=back["winrate"], use_cache=use_cache)
    return pred

# ======================================================================================
# DEBUG / DIAGNOSTICS HELPERS
# ======================================================================================

def debug_signal_breakdown(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Dict[str, Any]:
    """
    Returns raw components used in final decision so you can debug accuracy.
    Not shown in UI by default, but super useful in logs during development.
    """
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df = add_indicators(df_raw)
    if df.empty or len(df) < 3:
        return {"error": "not enough data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rule_side, rule_conf = compute_signal_row(prev, last)
    sent_score = _fetch_sentiment(symbol)

    model = _train_ml_model(symbol, interval_key, df)
    feats_all = _extract_ml_features(df)
    row_feat = feats_all.iloc[-1]
    ml_prob = _ml_predict_prob(model, row_feat)

    atr_now = _safe_float(last["atr"], _safe_float(last["Close"]) * 0.005)
    price_now = _safe_float(last["Close"])
    tp, sl = _compute_tp_sl(price_now, atr_now, rule_side, risk)
    rr = _calc_rr(price_now, tp, sl, rule_side)

    # backtest snapshot for dynamic weighting
    back = backtest_signals(df, risk)
    fused_prob = _fuse_prob(
        rule_prob=rule_conf,
        ml_prob=ml_prob,
        sentiment_score=sent_score,
        adx_val=_safe_float(last["adx"], 0.0),
        atr_pct=_safe_float(last["atr_rel"], 0.0),
        recent_winrate=back["winrate"],
    )

    return {
        "symbol": symbol,
        "rule_side": rule_side,
        "rule_conf": rule_conf,
        "ml_prob": ml_prob,
        "sentiment": sent_score,
        "fused_prob": fused_prob,
        "tp": tp,
        "sl": sl,
        "rr": rr,
        "winrate_context": back["winrate"],
        "maxdd_context": back["maxdd"],
        "trades_context": back["trades"],
    }

# ======================================================================================
# SELF-TEST ENTRYPOINT
# ======================================================================================

if __name__ == "__main__":
    # Basic smoke test when running utils.py directly
    test_symbol = "GC=F"
    test_interval = "1h"
    test_risk = "Medium"

    _log("🔍 Self-test starting...")
    df_test_raw = fetch_data(test_symbol, test_interval, use_cache=True)
    _log(f"Fetched rows: {len(df_test_raw)}")

    df_test_ind = add_indicators(df_test_raw)
    _log(f"With indicators rows: {len(df_test_ind)}")

    bt_stats = backtest_signals(df_test_ind, test_risk)
    _log(f"Backtest stats: {bt_stats}")

    pred_info = _latest_prediction(
        symbol=test_symbol,
        interval_key=test_interval,
        risk=test_risk,
        recent_winrate=bt_stats["winrate"],
        use_cache=True,
    )
    _log(f"Latest prediction: {json.dumps(pred_info, indent=2)}")

    sum_df = summarize_assets(test_interval, test_risk, use_cache=True)
    _log("Summary DF:")
    try:
        _log(sum_df.to_string())
    except Exception:
        _log(str(sum_df))

    dbg = debug_signal_breakdown(test_symbol, test_interval, test_risk)
    _log("Debug breakdown:")
    _log(json.dumps(dbg, indent=2))

    _log("✅ utils.py Smart v7 full test complete.")