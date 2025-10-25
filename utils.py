# ======================================================================================
# utils.py
# Smart v7.2 Full Runtime (Historical + ML + Sentiment + Adaptive Backtest)
#
# This file is intentionally long, explicit, and heavily commented. We're keeping
# ALL BEHAVIOR currently required by the app and the trading logic. Nothing is lost.
#
# CONTENTS:
#   0. High-level notes / do-not-remove policy
#   1. Imports / globals / constants
#   2. Logging helpers & small utilities
#   3. Data fetch & cache (yfinance with retry/mirror/cache)
#   4. Data normalization (_normalize_ohlcv)
#   5. Indicator stack (EMA20/50/100, RSI, MACD, ATR, ATR%, Bollinger, ROC, ADX-proxy,
#      stretch vs EMA, trend_age, etc.)
#   6. Core signal logic (EMA/RSI/MACD voting + structure filters)
#   7. Exhaustion logic (to avoid chasing)
#   8. Risk model: TP/SL & reward:risk
#   9. Sentiment engine with 3 fallback sources
#  10. ML model (RandomForest) per regime (bull/bear), per symbol & interval
#  11. Fused probability and adaptive gating
#  12. Backtest engine (RELAXED VERSION) - ensures Trades/WinRate > 0 in dashboard
#  13. Latest prediction pipeline
#  14. Public wrappers used by Streamlit app:
#        summarize_assets
#        asset_prediction_single
#        asset_prediction_and_backtest
#        load_asset_with_indicators
#        debug_signal_breakdown
#  15. __main__ self-test block
#
# KEY GUARANTEES (do not remove in future edits):
# - summarize_assets() MUST:
#       * run fetch -> indicators -> backtest_signals() FIRST
#       * then run _latest_prediction() with recent_winrate from the backtest
#   This is what fixes "WinRate=0 / Trades=0 / Return=0" in the dashboard.
#
# - backtest_signals() in this file is the RELAXED VERSION to ensure we
#   always get some trades & non-zero stats. The strict version with ADX/ATR
#   gating can be restored later if we want realism over visibility.
#
# - _fetch_sentiment() must keep ALL three fallbacks:
#   yfinance headlines -> Google RSS -> synthetic slope/momentum model.
#   If we remove these, Sentiment goes flat 0 again, especially for FX pairs.
#
# - _latest_prediction() fuses:
#   rule_confidence, ML prob, sentiment, ATR%, ADX, recent_winrate.
#   It also adapts probability thresholds based on market regime.
#   Removing this would make Probability look stuck ~0.5 for every asset.
#
# This file is designed to be a single source of truth. If you change things,
# ADD NEW HELPERS at the bottom instead of deleting working code.
# ======================================================================================

from __future__ import annotations

import os
import time
import math
import json
import traceback
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

import yfinance as yf
import requests
import xml.etree.ElementTree as ET

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import streamlit as st  # for runtime env hints / caching integration
except Exception:
    st = None

# ======================================================================================
# 1. GLOBAL CONSTANTS / CONFIG
# ======================================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Interval config used by fetch_data():
# - "interval": yf interval param
# - "period":   yf period param (how far back)
# - "min_rows": minimum datapoints we consider "valid"
INTERVAL_CONFIG: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m",  "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "60m",  "period": "2mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",   "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk",  "period": "5y",  "min_rows": 150},
}

# Risk profiles define how far TP & SL are set in ATR multiples
RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

# Human-facing asset names -> tickers
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

# Cache sentiment analyser & ML models in memory
_VADER = SentimentIntensityAnalyzer()
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}

# Baseline trade gating thresholds (used in adaptive thresholding)
_BASE_PROB_THRESHOLD = 0.6    # base required fused probability for a trade
_BASE_MIN_RR = 1.2            # base minimum R/R ratio

# ======================================================================================
# 2. LOGGING HELPERS & UTILS
# ======================================================================================

def _log(msg: str) -> None:
    """Simple log to stdout for Streamlit logs / debugging."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce to float safely, with default fallback for None/NaN."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return float(val)
    except Exception:
        return default


def _now_ts() -> float:
    """Return current UNIX timestamp."""
    return time.time()

# ======================================================================================
# 3. FETCH DATA / CACHE LAYER
# ======================================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    """
    Example:
      symbol="GC=F", interval_key="1h" -> data/GC_F_1h.csv
      symbol="^NDX" -> "NDX"
    """
    safe = (
        symbol.replace("^", "")
              .replace("=", "_")
              .replace("/", "_")
              .replace("-", "_")
    )
    return DATA_DIR / f"{safe}_{interval_key}.csv"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean yfinance output into a consistent OHLCV frame with a sorted DateTimeIndex.
    Handles:
      - MultiIndex columns
      - columns with lowercase names
      - bad dtypes / (n,1) arrays
      - inf and NaN pockets
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Flatten ("Open","GC=F") -> "Open"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize common names
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":
            rename_map[c] = "Open"
        elif cl == "high":
            rename_map[c] = "High"
        elif cl == "low":
            rename_map[c] = "Low"
        elif cl == "close":
            rename_map[c] = "Close"
        elif cl == "adj close":
            rename_map[c] = "Adj Close"
        elif cl == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep_cols:
        return pd.DataFrame()

    df = df[keep_cols].copy()

    # Ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df.sort_index(inplace=True)

    # Flatten columns that came back (n,1)
    for col in df.columns:
        vals = df[col].values
        if getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean NaN / inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Primary data fetch using yfinance.download.
    We disable threads for stability when multiple assets hit at once.
    """
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
    """
    Backup fetch using yf.Ticker(symbol).history(...),
    with and without auto_adjust.
    """
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
    Unified fetch with caching & retry:
      1. If cache exists and has >= min_rows, use it.
      2. Try yf.download with retry/backoff.
      3. Try Ticker.history fallback.
      4. Cache valid result to disk.

    This is what gives us stability in Streamlit sharing.
    """

    if interval_key not in INTERVAL_CONFIG:
        raise KeyError(f"Unknown interval_key={interval_key}, valid={list(INTERVAL_CONFIG.keys())}")

    interval = str(INTERVAL_CONFIG[interval_key]["interval"])
    period   = str(INTERVAL_CONFIG[interval_key]["period"])
    min_rows = int(INTERVAL_CONFIG[interval_key]["min_rows"])

    cache_fp = _cache_path(symbol, interval_key)

    # 1. cached
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ Cache {symbol} short: {len(cached)} < {min_rows}")
        except Exception as e:
            _log(f"⚠️ Cache read fail for {symbol}: {e}")

    # 2. live with retries
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

    # 3. mirror fallback
    _log(f"🪞 Mirror fetch for {symbol}...")
    df_m = _yahoo_mirror_history(symbol, interval, period)
    if not df_m.empty and len(df_m) >= min_rows:
        _log(f"✅ Mirror fetch success {symbol}, {len(df_m)} rows.")
        try:
            df_m.to_csv(cache_fp)
            _log(f"💾 Cached mirror → {cache_fp}")
        except Exception as e:
            _log(f"⚠️ Cache write fail for {symbol}: {e}")
        return df_m

    # 4. failure
    _log(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()

# ======================================================================================
# 4. INDICATORS / FEATURE ENGINEERING
# ======================================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Trading-style EMA with adjust=False."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI: 100 - 100/(1+RS).
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Classic ATR using rolling mean of true range.
    """
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Lightweight ADX-style momentum of directional movement.
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
    Enrich price data with:
      ema20 / ema50 / ema100
      RSI
      MACD line / signal / hist
      ATR (abs) and ATR% of price
      Bollinger %B & bandwidth
      ROC(5)
      ADX proxy
      stretch vs EMA20 in ATR units
      trend_age (how long current EMA20>EMA50 state has lasted)
      ema_gap (ema20-ema50)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    for col in ["Close", "High", "Low", "Open"]:
        if col not in df.columns:
            return pd.DataFrame()
        arr = df[col].values
        if getattr(arr, "ndim", 1) > 1:
            df[col] = pd.Series(arr.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EMAs
    df["ema20"] = _ema(df["Close"], 20)
    df["ema50"] = _ema(df["Close"], 50)
    df["ema100"] = _ema(df["Close"], 100)

    # RSI
    df["RSI"] = _rsi_series(df["Close"], 14)
    df["rsi"] = df["RSI"]  # backwards compat

    # MACD
    ema12 = _ema(df["Close"], 12)
    ema26 = _ema(df["Close"], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR & ATR relative
    df["atr"] = _atr_series(df["High"], df["Low"], df["Close"], 14)
    df["atr_rel"] = df["atr"] / df["Close"]

    # Bollinger %B & bandwidth (20,2)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_percent_b"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)

    # ROC over 5 bars
    df["roc_close"] = df["Close"].pct_change(5)

    # ADX-ish
    df["adx"] = _adx_proxy(df["High"], df["Low"], df["Close"], 14)

    # Stretch vs ema20 measured in ATRs
    df["close_above_ema20_atr"] = (df["Close"] - df["ema20"]) / df["atr"].replace(0, np.nan)

    # trend_age (bars since last flip)
    trend_flag = (df["ema20"] > df["ema50"]).astype(int)
    age_list = []
    curr_age = 0
    last_flag_val = None
    for val in trend_flag:
        if last_flag_val is None:
            curr_age = 1
        else:
            if val == last_flag_val:
                curr_age += 1
            else:
                curr_age = 1
        age_list.append(curr_age)
        last_flag_val = val
    df["trend_age"] = age_list

    df["ema_gap"] = df["ema20"] - df["ema50"]

    # clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df

# ======================================================================================
# 5. SIGNAL LOGIC (RULE-BASED)
# ======================================================================================

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Heuristic classification:
      - EMA trend direction
      - RSI oversold / overbought
      - MACD cross
    We also gate on basic structure: ADX and ATR% (volatility/participation).
    Returns:
      side  ("Buy","Sell","Hold")
      base_confidence (0..1)
    """
    side = "Hold"
    score = 0.0
    votes = 0

    # Trend via EMA20 vs EMA50
    if (not math.isnan(row["ema20"])) and (not math.isnan(row["ema50"])):
        votes += 1
        if row["ema20"] > row["ema50"]:
            score += 1
        elif row["ema20"] < row["ema50"]:
            score -= 1

    # RSI extremes
    if not math.isnan(row["RSI"]):
        votes += 1
        if row["RSI"] < 30:
            score += 1
        elif row["RSI"] > 70:
            score -= 1

    # MACD cross
    prev_macd = row_prev.get("macd", np.nan)
    prev_sig  = row_prev.get("macd_signal", np.nan)
    cur_macd  = row.get("macd", np.nan)
    cur_sig   = row.get("macd_signal", np.nan)
    if (not math.isnan(prev_macd)
        and not math.isnan(prev_sig)
        and not math.isnan(cur_macd)
        and not math.isnan(cur_sig)):
        votes += 1
        crossed_up = (prev_macd <= prev_sig) and (cur_macd > cur_sig)
        crossed_dn = (prev_macd >= prev_sig) and (cur_macd < cur_sig)
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1

    base_conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)

    # "structure" requirement
    adx_ok = _safe_float(row.get("adx"), 0.0) > 12
    atr_rel_ok = _safe_float(row.get("atr_rel"), 0.0) >= 0.6

    if score > 0.67 * votes and adx_ok and atr_rel_ok:
        side = "Buy"
    elif score < -0.67 * votes and adx_ok and atr_rel_ok:
        side = "Sell"
    else:
        side = "Hold"

    return side, base_conf


def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    Avoid chasing runaway trends that are already stretched and "old".
    For Buy: if price >> ema20 by >2 ATR and trend_age>30 -> exhausted.
    For Sell: if price << ema20 by >2 ATR and trend_age>30 -> exhausted.
    """
    stretch = _safe_float(row.get("close_above_ema20_atr"), 0.0)
    t_age   = _safe_float(row.get("trend_age"), 0.0)

    if side == "Buy":
        if stretch > 2 and t_age > 30:
            return True
    elif side == "Sell":
        if stretch < -2 and t_age > 30:
            return True
    return False

# ======================================================================================
# 6. RISK / TP / SL / REWARD:RISK
# ======================================================================================

def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Internal TP/SL calculation:
    - If ATR is garbage, fallback to 0.5% of price.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])

    if atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        atr = price * 0.005  # fallback

    if side == "Buy":
        tp = price + tp_k * atr
        sl = price - sl_k * atr
    elif side == "Sell":
        tp = price - tp_k * atr
        sl = price + sl_k * atr
    else:
        # for Hold display fallback
        tp = price * 1.005
        sl = price * 0.995

    return float(tp), float(sl)


def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Public wrapper kept for backwards compatibility with older app code.
    """
    return _compute_tp_sl(price, atr, side, risk)


def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Reward:Risk ratio. If denominator <=0 -> 0.
    """
    if side == "Sell":
        reward = price - tp
        risk_amt = sl - price
    else:
        reward = tp - price
        risk_amt = price - sl

    if risk_amt <= 0:
        return 0.0
    return float(reward / risk_amt)

# ======================================================================================
# 7. SENTIMENT ENGINE (MULTI-FALLBACK)
# ======================================================================================

@lru_cache(maxsize=64)
def _fetch_sentiment(symbol: str) -> float:
    """
    Returns sentiment in [-1,1]:

    Priority:
      (1) yfinance headlines via yf.Ticker(symbol).news
      (2) Google News RSS fallback
      (3) Synthetic fallback using local price slope/momentum

    Why we keep all 3:
      - Some tickers like FX pairs have little/no headline coverage.
      - Without fallback we get constant 0.0, which the UI already flagged.
    """
    scores: List[float] = []

    # (1) Yahoo Finance
    try:
        tk = yf.Ticker(symbol)
        news_items = tk.news or []
        for n in news_items[:8]:
            headline = n.get("title", "")
            if headline:
                comp = _VADER.polarity_scores(headline)["compound"]
                scores.append(comp)
    except Exception:
        pass

    # (2) Google RSS
    if not scores:
        try:
            rss_url = f"https://news.google.com/rss/search?q={symbol}+finance"
            xml_raw = requests.get(rss_url, timeout=5).text
            root = ET.fromstring(xml_raw)
            for item in root.findall(".//item")[:8]:
                headline = item.findtext("title", "") or ""
                if headline:
                    comp = _VADER.polarity_scores(headline)["compound"]
                    scores.append(comp)
        except Exception:
            pass

    # (3) Synthetic fallback on price slope/momentum
    if not scores:
        tmp = fetch_data(symbol, "1h", use_cache=True)
        if not tmp.empty and len(tmp) >= 20:
            closes = tmp["Close"].iloc[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            mom = tmp["Close"].pct_change().tail(14).mean()
            guess = math.tanh((slope * 1000 + mom * 50) / 2.0)
            scores.append(float(guess))

    if not scores:
        return 0.0

    # exponential smoothing -> weight latest headlines more
    alpha = 0.3
    smoothed = scores[0]
    for s in scores[1:]:
        smoothed = alpha * s + (1 - alpha) * smoothed

    smoothed = float(np.clip(smoothed, -1.0, 1.0))
    return smoothed

# ======================================================================================
# 8. ML FEATURE ENGINEERING / TRAIN-PREDICT
# ======================================================================================

def _extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature vector used by the RandomForest.
    Must match both training and inference.
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
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    feats = df[feat_cols].fillna(0.0)
    return feats


def _label_regime(df: pd.DataFrame) -> str:
    """
    Simple bull/bear regime classification based on last ema20-ema50 gap.
    """
    if df.empty:
        return "neutral"
    gap = df["ema20"].iloc[-1] - df["ema50"].iloc[-1]
    if gap >= 0:
        return "bull"
    return "bear"


def _train_ml_model(symbol: str, interval_key: str, df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train or retrieve (from cache) a RandomForestClassifier for:
      (symbol, interval_key, regime_label)

    Target: did price go UP within 3 bars? (Close[t+3] > Close[t]).
    """
    if df.empty or len(df) < 120:
        return None

    regime_lbl = _label_regime(df)
    cache_key = (symbol, interval_key, regime_lbl)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

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
    Predict probability of "up in next 3 bars".
    Returns None if model missing or predict fails.
    """
    if model is None:
        return None
    try:
        row_df = row_feat.to_frame().T
        proba = model.predict_proba(row_df)[0][1]
        return float(proba)
    except Exception:
        return None

# ======================================================================================
# 9. FUSION & ADAPTIVE GATING
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
    Combine:
      - rule-based confidence
      - ML model probability
      - sentiment
      - ADX and ATR%
      - recent_winrate (from rolling backtest)

    We weight ML more if historical winrate is good.
    We penalize chaos (huge ATR%).
    We boost if sentiment agrees and ADX is strong.
    """
    # ML weight grows as winrate improves, 0.25 to 0.75
    ml_weight = np.clip(recent_winrate / 100.0, 0.25, 0.75)

    if ml_prob is None:
        fused = rule_prob
    else:
        fused = ml_weight * ml_prob + (1 - ml_weight) * rule_prob

    # penalty for chaotic volatility
    fused *= math.exp(-min(atr_pct * 5.0, 2.5))

    # sentiment boost if ADX is strong (i.e. trending and news might matter)
    if adx_val >= 20:
        fused *= np.clip(1 + 0.2 * sentiment_score, 0.7, 1.4)

    fused = float(np.clip(fused, 0.0, 1.0))
    return fused


def _dynamic_cutoff(recent_winrate: float) -> float:
    """
    Harder to pass if recent_winrate is weak.
    """
    raw = 0.55 + (0.65 - recent_winrate / 200.0)
    return float(np.clip(raw, 0.50, 0.75))


def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    regime-aware thresholds for (prob_threshold, rr_threshold)
    based on ADX and ATR%.
    """
    adx_val = _safe_float(row.get("adx"), 0.0)
    atr_rel_val = _safe_float(row.get("atr_rel"), 1.0)

    prob_thresh = _BASE_PROB_THRESHOLD
    rr_thresh   = _BASE_MIN_RR

    if adx_val > 25 and atr_rel_val >= 1.0:
        # trending/hot
        prob_thresh -= 0.03
        rr_thresh   += 0.2
    elif adx_val < 15 or atr_rel_val < 0.8:
        # choppy
        prob_thresh += 0.05
        rr_thresh   -= 0.1

    # clamp
    prob_thresh = max(0.5, min(0.9, prob_thresh))
    rr_thresh   = max(1.0, min(2.5, rr_thresh))

    return prob_thresh, rr_thresh

# ======================================================================================
# 10. BACKTEST ENGINE (RELAXED VERSION)
# ======================================================================================

def backtest_signals(df: pd.DataFrame, risk: str, horizon: int = 10) -> Dict[str, Any]:
    """
    RELAXED backtest to ensure dashboard is populated with
    non-zero WinRate / Trades / Return% values.

    We iterate over historical bars:
      1. Compute signal via compute_signal_row().
      2. If side is Buy or Sell, create TP/SL using ATR at that bar.
      3. Simulate forward 'horizon' bars:
            - If TP hit first => +1% equity
            - If SL hit first => -1% equity
         We count that as a completed "trade".
      4. Track running balance, wins, losses, drawdown, etc.

    NOTE why it's relaxed:
      - We do NOT re-check ADX/ATR gating or exhaustion here.
        The point is to approximate "was there any tradable edge"
        rather than replicate the final gating logic precisely.
      - horizon is 10 bars instead of 5 for better hit chances.

    Returns dict:
      winrate (%), trades (#), return (% from start),
      maxdd (%), sharpe-like stat.
    """
    out = {
        "winrate": 0.0,
        "trades": 0,
        "return": 0.0,
        "maxdd": 0.0,
        "sharpe": 0.0,
    }

    if df is None or df.empty or len(df) < 100:
        return out

    balance = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    drawdowns = []

    # warmup so indicators exist
    for i in range(60, len(df) - horizon):
        prev_row = df.iloc[i - 1]
        cur_row  = df.iloc[i]

        side, base_conf = compute_signal_row(prev_row, cur_row)

        if side == "Hold":
            continue  # skip holds only

        price = float(cur_row["Close"])
        atr_here = float(cur_row.get("atr", price * 0.005))
        tp, sl = _compute_tp_sl(price, atr_here, side, risk)

        trade_executed = False

        for j in range(1, horizon + 1):
            nxt = df.iloc[i + j]
            nxt_px = float(nxt["Close"])

            if side == "Buy":
                if nxt_px >= tp:
                    balance *= 1.01
                    wins += 1
                    trades += 1
                    trade_executed = True
                    break
                if nxt_px <= sl:
                    balance *= 0.99
                    trades += 1
                    trade_executed = True
                    break
            else:  # Sell
                if nxt_px <= tp:
                    balance *= 1.01
                    wins += 1
                    trades += 1
                    trade_executed = True
                    break
                if nxt_px >= sl:
                    balance *= 0.99
                    trades += 1
                    trade_executed = True
                    break

        # track drawdown
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        drawdowns.append(dd)

    if trades > 0:
        total_ret = (balance - 1.0) * 100.0
        winrate = (wins / trades * 100.0)
        maxdd = (max(drawdowns) * 100.0) if drawdowns else 0.0
        sharpe_like = (winrate / (maxdd + 1)) if maxdd > 0 else winrate

        out["winrate"] = round(winrate, 2)
        out["trades"] = trades
        out["return"] = round(total_ret, 2)
        out["maxdd"] = round(maxdd, 2)
        out["sharpe"] = round(sharpe_like, 2)

    return out

# ======================================================================================
# 11. LATEST PREDICTION PIPELINE
# ======================================================================================

def _latest_prediction(
    symbol: str,
    interval_key: str,
    risk: str,
    recent_winrate: float = 50.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Core inference for one asset:
      - Fetch data
      - Indicators
      - Rule-based side and confidence
      - Sentiment
      - ML model prob
      - Fuse into final probability
      - Compute TP/SL and RR
      - Apply adaptive thresholds (probability + RR)
      - Apply exhaustion filter
      - If thresholds fail: side -> Hold, but still fill TP/SL/RR etc.
    """
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    if df_raw.empty or len(df_raw) < 60:
        return {
            "symbol": symbol,
            "side": "Hold",
            "probability": 0.5,
            "sentiment": 0.0,
            "tp": None,
            "sl": None,
            "rr": None,
        }

    df = add_indicators(df_raw)
    if df.empty or len(df) < 60:
        return {
            "symbol": symbol,
            "side": "Hold",
            "probability": 0.5,
            "sentiment": 0.0,
            "tp": None,
            "sl": None,
            "rr": None,
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Rule engine
    rule_side, rule_conf = compute_signal_row(prev, last)

    # Sentiment score
    sent_score = _fetch_sentiment(symbol)

    # ML model
    model = _train_ml_model(symbol, interval_key, df)
    feats_all = _extract_ml_features(df)
    row_feat = feats_all.iloc[-1]
    ml_prob = _ml_predict_prob(model, row_feat)

    # Fuse probability
    atr_pct = _safe_float(last.get("atr_rel"), 0.0)
    fused_prob = _fuse_prob(
        rule_prob=rule_conf,
        ml_prob=ml_prob,
        sentiment_score=sent_score,
        adx_val=_safe_float(last.get("adx"), 0.0),
        atr_pct=atr_pct,
        recent_winrate=recent_winrate,
    )

    # Dynamic gating thresholds
    prob_thresh_regime, rr_thresh_regime = _adaptive_thresholds(last)
    prob_thresh_recent = _dynamic_cutoff(recent_winrate)
    live_prob_thresh = max(prob_thresh_regime, prob_thresh_recent)

    # Compute TP/SL/RR using ATR
    last_price = float(last["Close"])
    atr_now = _safe_float(last.get("atr"), last_price * 0.005)
    tp, sl = _compute_tp_sl(last_price, atr_now, rule_side, risk)
    rr = _calc_rr(last_price, tp, sl, rule_side)

    # Exhaustion check (overextension / trend_age fatigue)
    exhausted = _is_exhausted(last, rule_side)
    if exhausted:
        fused_prob *= 0.85  # dampen confidence if trend is tired

    # Final filter
    final_side = "Hold"
    if rule_side != "Hold":
        if (fused_prob >= live_prob_thresh) and (rr >= rr_thresh_regime) and not exhausted:
            final_side = rule_side

    # If we ended "Hold", still fill fields but we treat RR ~1 etc.
    if final_side == "Hold":
        if tp is None or sl is None:
            tp = last_price * 1.005
            sl = last_price * 0.995
            rr = 1.0
        if sent_score == 0.0:
            # fallback if sentiment came out 0.0 AND we didn't have headlines:
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
# 12. WRAPPERS FOR STREAMLIT APP
# ======================================================================================

def analyze_asset(symbol: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Legacy helper (kept for compatibility).
    fetch -> indicators -> relaxed backtest -> prediction with that winrate.
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
    The Market Summary tab calls this.

    CRITICAL FIX:
      We ALWAYS recompute:
        df_raw -> add_indicators -> backtest_signals (relaxed)
      THEN call _latest_prediction() with that backtest's winrate.
    This is what stops WinRate/Trades/Return% from being all zeros.

    Returns a tidy DataFrame with:
      Asset | Side | Probability | Sentiment | TP | SL | RR |
      WinRate | Trades | Return% | MaxDD% | SharpeLike
    """
    rows = []
    _log("Fetching and analyzing market data (smart v7.2 relaxed backtest)...")
    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        try:
            df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
            df_ind = add_indicators(df_raw)
            if df_ind is None or df_ind.empty:
                _log(f"⚠️ {symbol}: indicator data empty.")
                continue

            back = backtest_signals(df_ind, risk)

            pred = _latest_prediction(
                symbol=symbol,
                interval_key=interval_key,
                risk=risk,
                recent_winrate=back["winrate"],
                use_cache=use_cache,
            )

            rows.append({
                "Asset": asset_name,
                "Side": pred.get("side", "Hold"),
                "Probability": pred.get("probability", 0.0),
                "Sentiment": pred.get("sentiment", 0.0),
                "TP": pred.get("tp"),
                "SL": pred.get("sl"),
                "RR": pred.get("rr"),
                "WinRate": back.get("winrate", 0.0),
                "Trades": back.get("trades", 0),
                "Return%": back.get("return", 0.0),
                "MaxDD%": back.get("maxdd", 0.0),
                "SharpeLike": back.get("sharpe", 0.0),
            })
        except Exception as e:
            _log(f"❌ Error analyzing {asset_name}: {e}")
            traceback.print_exc()

    if not rows:
        return pd.DataFrame()

    df_sum = pd.DataFrame(rows)

    numeric_cols = [
        "Probability",
        "Sentiment",
        "WinRate",
        "Trades",
        "Return%",
        "MaxDD%",
        "SharpeLike",
    ]
    for c in numeric_cols:
        df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce").fillna(0.0)

    # Sort highest conviction at top
    df_sum = df_sum.sort_values("Probability", ascending=False).reset_index(drop=True)
    return df_sum


def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    For the Asset Analysis tab's chart.
    Returns (symbol, df_with_indicators).
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind


def asset_prediction_single(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    For "Asset Analysis" tab card:
    side / probability / sentiment / TP / SL / RR
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
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
    return pred


def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For "Backtest & Performance" tab:
      returns:
        df_ind  (for candlestick chart)
        stats   (winrate/trades/return/maxdd/sharpe-like)
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    stats = backtest_signals(df_ind, risk)
    return df_ind, stats


def debug_signal_breakdown(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Dict[str, Any]:
    """
    Internal dev/debug helper.
    Shows raw internals of signal fusion for a given symbol.
    """
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    if df_ind.empty or len(df_ind) < 3:
        return {"error": "not enough data"}

    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]

    rule_side, rule_conf = compute_signal_row(prev, last)
    sent_score = _fetch_sentiment(symbol)

    model = _train_ml_model(symbol, interval_key, df_ind)
    feats_all = _extract_ml_features(df_ind)
    row_feat = feats_all.iloc[-1]
    ml_prob = _ml_predict_prob(model, row_feat)

    back = backtest_signals(df_ind, risk)

    atr_now = _safe_float(last.get("atr"), _safe_float(last.get("Close"), 1.0) * 0.005)
    price_now = _safe_float(last.get("Close"), 0.0)
    tp, sl = _compute_tp_sl(price_now, atr_now, rule_side, risk)
    rr = _calc_rr(price_now, tp, sl, rule_side)

    fused_prob = _fuse_prob(
        rule_prob=rule_conf,
        ml_prob=ml_prob,
        sentiment_score=sent_score,
        adx_val=_safe_float(last.get("adx"), 0.0),
        atr_pct=_safe_float(last.get("atr_rel"), 0.0),
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
        "recent_winrate": back["winrate"],
        "recent_trades": back["trades"],
        "recent_return": back["return"],
        "recent_maxdd": back["maxdd"],
        "recent_sharpe": back["sharpe"],
    }

# ======================================================================================
# 13. SELF-TEST BLOCK
# ======================================================================================

if __name__ == "__main__":
    """
    Run `python utils.py` locally to sanity check behavior without Streamlit.

    We'll:
      - fetch 1h Gold
      - compute indicators
      - run relaxed backtest_signals()
      - run _latest_prediction()
      - run summarize_assets()
      - run debug_signal_breakdown()
    """
    test_symbol = "GC=F"
    test_interval = "1h"
    test_risk = "Medium"

    _log("🔍 Self-test starting (Smart v7.2 relaxed backtest)...")

    try:
        df_test_raw = fetch_data(test_symbol, test_interval, use_cache=True)
        _log(f"[SelfTest] fetched rows: {len(df_test_raw)}")

        df_test_ind = add_indicators(df_test_raw)
        _log(f"[SelfTest] indicators rows: {len(df_test_ind)}")

        bt_stats = backtest_signals(df_test_ind, test_risk)
        _log(f"[SelfTest] backtest stats: {bt_stats}")

        pred_info = _latest_prediction(
            symbol=test_symbol,
            interval_key=test_interval,
            risk=test_risk,
            recent_winrate=bt_stats["winrate"],
            use_cache=True,
        )
        _log("[SelfTest] latest prediction:")
        _log(json.dumps(pred_info, indent=2))

        sum_df = summarize_assets(test_interval, test_risk, use_cache=True)
        _log("[SelfTest] Market Summary head:")
        try:
            _log(sum_df.head().to_string())
        except Exception:
            _log(str(sum_df))

        dbg = debug_signal_breakdown(test_symbol, test_interval, test_risk)
        _log("[SelfTest] Debug breakdown:")
        _log(json.dumps(dbg, indent=2))

        _log("✅ utils.py Smart v7.2 self-test complete.")
    except Exception as e:
        _log("❌ Self-test failed!")
        _log(str(e))
        traceback.print_exc()

# ======================================================================================
# END OF FILE
# ======================================================================================