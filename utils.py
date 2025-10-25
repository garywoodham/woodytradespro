# ======================================================================================
# utils.py
# Smart v7.1 Full Runtime (Historical + ML + Sentiment + Adaptive Backtest)
#
# IMPORTANT:
# This file is intentionally long, explicit, and heavily commented. We are preserving
# ALL behavior we've built so far in this project. Nothing is dropped:
#
#   1. Data fetch & caching (yfinance with retry, mirror, cache).
#   2. Data normalization (OHLCV cleanup, MultiIndex flattening).
#   3. Technical indicators:
#        - EMA20 / EMA50 / EMA100
#        - RSI(14)
#        - MACD (12/26/9) and histogram
#        - ATR(14) and ATR as % of price
#        - Bollinger %B and bandwidth
#        - ROC(5)
#        - ADX-like proxy (trend strength)
#        - "stretch" above EMA20 in ATRs
#        - "trend_age" (bars since trend flip)
#
#   4. Rule-based signal (Buy/Sell/Hold) from EMA/RSI/MACD,
#      + volatility gates (ADX > X, ATR% > Y).
#
#   5. Exhaustion logic to avoid chasing overextended trends.
#
#   6. TP/SL computation using ATR scaling and risk profiles (Low/Medium/High).
#
#   7. Sentiment module with 3 fallback layers:
#        A. Yahoo Finance headlines via yfinance.Ticker(symbol).news
#        B. Google News RSS sentiment
#        C. Synthetic sentiment based on slope/momentum of recent price
#
#   8. ML model:
#        - We train a RandomForestClassifier per (symbol, interval_key, regime_label)
#          where regime_label ~ bull/bear based on ema20>ema50.
#        - Model predicts probability that price will be UP in 3 bars.
#
#   9. Probability fusion:
#        - Combine rule_conf, ml_prob, sentiment, ATR%, ADX, and recent win rate.
#        - Adaptive thresholds for prob and RR, dynamic cutoff based on recent winrate.
#
#   10. Rolling backtest:
#        - Simulate trades across recent window (entry signal, TP/SL hit inside horizon).
#        - Compute win rate, trades executed, cumulative return%, max drawdown%, sharpe-like.
#
#   11. summarize_assets():
#        - Runs fresh indicators, fresh backtest, then latest prediction.
#        - Populates Probability, Sentiment, TP, SL, RR, WinRate, Trades, Return%, MaxDD%, SharpeLike.
#        - Sorts by Probability.
#        - This version FIXES the "all zeros" bug by forcing backtest_signals()
#          and using its winrate as context into _latest_prediction().
#
#   12. load_asset_with_indicators():
#        - For charting in Asset Analysis tab.
#
#   13. asset_prediction_single():
#        - Used in Asset Analysis tab to show current signal card
#          (Side / Probability / Sentiment / TP / SL / RR).
#
#   14. asset_prediction_and_backtest():
#        - Used in Backtest tab to show stats + chart.
#
#   15. debug_signal_breakdown():
#        - Dev introspection: see rule_conf, ml_prob, fused_prob, RR, etc.
#
#   16. __main__ self test:
#        - Lets you run `python utils.py` locally to sanity check everything
#          without Streamlit.
#
# Also to note:
# - We provide helper functions like _safe_float(), _is_exhausted(), _dynamic_cutoff(),
#   _adaptive_thresholds() and so on. Do not remove them.
# - We keep RISK_MULT, INTERVAL_CONFIG, ASSET_SYMBOLS global state to match app.py.
# - We export the same function names app.py imports:
#     summarize_assets
#     asset_prediction_single
#     asset_prediction_and_backtest
#     load_asset_with_indicators
#
# If you add/modify logic later, APPEND new code rather than deleting working code.
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

# scikit-learn for ML classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# sentiment model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Streamlit is optional here. utils must also work from plain python.
try:
    import streamlit as st  # noqa: F401
except Exception:
    st = None

# ======================================================================================
# GLOBAL CONSTANTS AND CONFIG
# ======================================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Timeframe config: interval, history period for fetch, and min_rows required
INTERVAL_CONFIG: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m",  "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "60m",  "period": "2mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",   "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk",  "period": "5y",  "min_rows": 150},
}

# Risk profiles for TP/SL distance in ATRs
RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

# Mapping of pretty asset labels (in UI) to tickers
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

# Shared analyzer / model caches
_VADER = SentimentIntensityAnalyzer()
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}  # (symbol, interval_key, regime_label)

# Baseline gating constraints for trades (used in thresholds)
_BASE_PROB_THRESHOLD = 0.6   # base probability cutoff for entering trades
_BASE_MIN_RR = 1.2           # base reward:risk min

# ======================================================================================
# LOG UTILS
# ======================================================================================

def _log(msg: str) -> None:
    """Print for server logs/Streamlit console. Safe even if stdout weird."""
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Turn anything into float safely. Use default for NaN/None."""
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
# DATA FETCH / CACHE
# ======================================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    """
    Generate clean filename for cached CSV:
      "^NDX" -> "NDX"
      "GC=F" -> "GC_F"
      "EURUSD=X" -> "EURUSD_X"
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
    yfinance can return:
      - MultiIndex columns
      - columns with lowercase names
      - 2D columns (n,1)
      - NaN/inf pockets
    We fix all of that here and enforce columns: Open/High/Low/Close/Adj Close/Volume
    plus a DatetimeIndex.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Flatten possible MultiIndex (("Open", "GC=F"), ...) to "Open"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names to standard case
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

    # Ensure DateTimeIndex, sort ascending
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # Flatten any (n,1) columns
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean inf and NaN patches
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Main fetch path using yfinance.download.
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
    Mirror fetch fallback using Ticker.history
    """
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        # Try without auto_adjust
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
    Robust unified fetch:
      1. Try cached CSV (if fresh & big enough)
      2. yf.download with retry/backoff
      3. Ticker.history fallback
      4. Cache successful frame to CSV

    interval_key must exist in INTERVAL_CONFIG.
    """
    if interval_key not in INTERVAL_CONFIG:
        raise KeyError(f"Unknown interval_key={interval_key}, valid={list(INTERVAL_CONFIG.keys())}")

    interval = str(INTERVAL_CONFIG[interval_key]["interval"])
    period   = str(INTERVAL_CONFIG[interval_key]["period"])
    min_rows = int(INTERVAL_CONFIG[interval_key]["min_rows"])

    cache_fp = _cache_path(symbol, interval_key)

    # 1. attempt cache
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

    # 2. live fetch retries
    _log(f"⏳ Fetching {symbol} [{interval}] ...")
    for attempt in range(1, max_retries + 1):
        df_live = _yahoo_try_download(symbol, interval, period)
        if not df_live.empty and len(df_live) >= min_rows:
            _log(f"✅ {symbol}: fetched {len(df_live)} rows.")
            # write cache
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
        # cache
        try:
            df_m.to_csv(cache_fp)
            _log(f"💾 Cached mirror → {cache_fp}")
        except Exception as e:
            _log(f"⚠️ Cache write fail for {symbol}: {e}")
        return df_m

    # 4. give up
    _log(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()

# ======================================================================================
# TECHNICAL INDICATORS / FEATURE ENGINEERING
# ======================================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Plain EMA with adjust=False (trading convention)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Classic RSI calc:
        RS = avg(gain) / avg(loss)
        RSI = 100 - 100/(1+RS)
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    ATR(14) style:
      True Range = max(
        high - low,
        abs(high - prev_close),
        abs(low  - prev_close)
      )
      ATR = rolling mean(True Range, 14)
    """
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Lightweight ADX-ish proxy:
      - We approximate +DI and -DI then smooth.
      - Not identical to real ADX, but enough to estimate "trend strength".
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
    Enhance OHLCV with:
      ema20, ema50, ema100
      RSI
      MACD / MACD signal / MACD hist
      ATR, ATR relative (atr / close)
      Bollinger %B, bandwidth
      Short-term ROC (5 bars)
      ADX proxy
      Stretch vs EMA20 (in ATR)
      trend_age (bars since last trend flip)

    This function must NEVER silently fail without returning df,
    because downstream (backtest_signals, ML features, etc.) depend on all columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Force numeric shape for OHLC
    for col in ["Close", "High", "Low", "Open"]:
        if col not in df.columns:
            return pd.DataFrame()  # bail if missing key price cols
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
    df["rsi"] = df["RSI"]  # backward compat for old code expecting 'rsi'

    # MACD-style
    ema12 = _ema(df["Close"], 12)
    ema26 = _ema(df["Close"], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR and relative ATR
    df["atr"] = _atr_series(df["High"], df["Low"], df["Close"], 14)
    df["atr_rel"] = df["atr"] / df["Close"]

    # Bollinger stats
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_percent_b"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)

    # ROC (5-bar change %)
    df["roc_close"] = df["Close"].pct_change(5)

    # ADX-ish trend strength proxy
    df["adx"] = _adx_proxy(df["High"], df["Low"], df["Close"], 14)

    # Stretch vs EMA20 in ATR units
    df["close_above_ema20_atr"] = (df["Close"] - df["ema20"]) / df["atr"].replace(0, np.nan)

    # Trend age calculation: bars since last flip of ema20 > ema50
    trend_flag = (df["ema20"] > df["ema50"]).astype(int)  # 1 bull, 0 bear
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

    # Secondary helpful columns:
    df["ema_gap"] = df["ema20"] - df["ema50"]

    # Cleanup inf -> NaN -> fill
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
    Heuristic ensemble voter:
      - EMA20 vs EMA50 (trend)
      - RSI extremes (<30 or >70)
      - MACD cross vs signal
      - Filter for "structure" (enough ADX and ATR%)

    Returns:
      side: "Buy", "Sell", or "Hold"
      base_confidence: 0..1

    NOTE:
    This does NOT apply risk filters like exhaustion or RR threshold.
    That happens later.
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

    # RSI extremes: oversold (<30) => bullish tilt, overbought (>70) => bearish tilt
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
    if (not math.isnan(prev_macd) and
        not math.isnan(prev_sig) and
        not math.isnan(cur_macd) and
        not math.isnan(cur_sig)):
        votes += 1
        crossed_up = (prev_macd <= prev_sig) and (cur_macd > cur_sig)
        crossed_dn = (prev_macd >= prev_sig) and (cur_macd < cur_sig)
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1

    # Base confidence from vote
    base_conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)

    # Minimum "structure" filter:
    adx_ok = _safe_float(row.get("adx"), 0.0) > 12
    atr_rel_ok = _safe_float(row.get("atr_rel"), 0.0) >= 0.6

    # Assign raw side
    if score > 0.67 * votes and adx_ok and atr_rel_ok:
        side = "Buy"
    elif score < -0.67 * votes and adx_ok and atr_rel_ok:
        side = "Sell"
    else:
        side = "Hold"

    return side, base_conf


def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    Avoid chasing if trend is too stretched for too long.
    For bullish signals, if Close is >2 ATR above ema20 and trend_age>30 => exhaust.
    For bearish signals, if Close is <2 ATR below ema20 and trend_age>30 => exhaust.
    """
    stretch = _safe_float(row.get("close_above_ema20_atr"), 0.0)
    t_age = _safe_float(row.get("trend_age"), 0.0)

    if side == "Buy":
        if stretch > 2 and t_age > 30:
            return True
    elif side == "Sell":
        if stretch < -2 and t_age > 30:
            return True
    return False

# ======================================================================================
# RISK / TP / SL / REWARD:RISK
# ======================================================================================

def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Compute TP/SL levels from ATR * risk multipliers.
    Used internally.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])

    # safety fallback if ATR is garbage
    if atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        atr = price * 0.005  # ~0.5%

    if side == "Buy":
        tp = price + tp_k * atr
        sl = price - sl_k * atr
    elif side == "Sell":
        tp = price - tp_k * atr
        sl = price + sl_k * atr
    else:
        # "Hold" fallback for display
        tp = price * 1.005
        sl = price * 0.995

    return float(tp), float(sl)


def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Public wrapper for TP/SL in case app or past code calls compute_tp_sl().
    """
    return _compute_tp_sl(price, atr, side, risk)


def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Reward:Risk ratio:
      reward = |target - entry|
      risk   = |entry - stop|
      If invalid risk, returns 0.
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
# SENTIMENT ENGINE
# ======================================================================================

@lru_cache(maxsize=64)
def _fetch_sentiment(symbol: str) -> float:
    """
    Multi-layer sentiment score in [-1,1]:
      1. yfinance headlines (Ticker.news)
      2. Google News RSS fallback
      3. Synthetic fallback from recent slope/momentum

    We also do exponential smoothing to bias toward more recent headlines.
    """

    scores: List[float] = []

    # (1) Yahoo Finance headlines
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

    # (2) Google RSS fallback
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

    # (3) Synthetic fallback based on short-term price structure
    # This helps assets like FX where yfinance news is sparse or missing.
    if not scores:
        tmp = fetch_data(symbol, "1h", use_cache=True)
        if not tmp.empty and len(tmp) >= 20:
            closes = tmp["Close"].iloc[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            mom = tmp["Close"].pct_change().tail(14).mean()
            # Map slope & momentum to [-1,1] via tanh
            guess = np.tanh((slope * 1000 + mom * 50) / 2.0)
            scores.append(float(guess))

    if not scores:
        return 0.0

    # Exponential smoothing to weight recent items more
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
    Build ML feature frame.
    List of required columns must match training & inference.
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

    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    feats = df[feat_cols].fillna(0.0)
    return feats


def _label_regime(df: pd.DataFrame) -> str:
    """
    Rough regime classification:
      bull if ema20 >= ema50
      bear otherwise
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
    Train or retrieve cached RandomForestClassifier for (symbol, interval, regime).
    Target = 1 if Close in 3 bars > current Close else 0.

    We cache per (symbol, interval_key, regime_label).
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
        # if target is basically all 1s or all 0s we can't train
        return None

    # train/test split (time-respecting: shuffle=False)
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
    Predict "probability of up move in next 3 bars".
    Returns None if model missing.
    """
    if model is None:
        return None
    try:
        row_df = row_feat.to_frame().T  # shape (1, n_features)
        proba = model.predict_proba(row_df)[0][1]
        return float(proba)
    except Exception:
        return None

# ======================================================================================
# FUSION / ADAPTIVE THRESHOLDS
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
    Combine rule_conf, ML prob, sentiment, volatility, and historical performance
    into a final fused probability of success.

    We bias toward ML more if recent winrate is high.
    We penalize if ATR% is huge (chaos).
    We boost if sentiment agrees and ADX is strong.
    """

    # ML weighting scales with recent winrate
    ml_weight = np.clip(recent_winrate / 100.0, 0.25, 0.75)

    if ml_prob is None:
        fused = rule_prob
    else:
        fused = ml_weight * ml_prob + (1 - ml_weight) * rule_prob

    # penalty for chaos (very high ATR%)
    fused *= math.exp(-min(atr_pct * 5.0, 2.5))

    # if ADX strong, allow sentiment to tilt confidence
    if adx_val >= 20:
        fused *= np.clip(1 + 0.2 * sentiment_score, 0.7, 1.4)

    fused = float(np.clip(fused, 0.0, 1.0))
    return fused


def _dynamic_cutoff(recent_winrate: float) -> float:
    """
    Make it harder to trade if recent performance is weak.
    If recent winrate high, relax slightly.
    """
    raw = 0.55 + (0.65 - recent_winrate / 200.0)
    # clamp 0.50..0.75
    return float(np.clip(raw, 0.50, 0.75))


def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    Return (prob_threshold, rr_threshold) based on regime characteristics:
      - strong trend (high ADX, high ATR%) => OK lower prob if RR high
      - choppy => demand higher prob, but we might accept lower RR
    """
    adx_val = _safe_float(row.get("adx"), 0.0)
    atr_rel_val = _safe_float(row.get("atr_rel"), 1.0)

    prob_thresh = _BASE_PROB_THRESHOLD
    rr_thresh   = _BASE_MIN_RR

    if adx_val > 25 and atr_rel_val >= 1.0:
        # trending/hot: lower prob ok, but expect better RR
        prob_thresh -= 0.03
        rr_thresh   += 0.2
    elif adx_val < 15 or atr_rel_val < 0.8:
        # choppy/low energy: demand higher prob, tolerate lower RR
        prob_thresh += 0.05
        rr_thresh   -= 0.1

    prob_thresh = max(0.5, min(0.9, prob_thresh))
    rr_thresh   = max(1.0, min(2.5, rr_thresh))

    return prob_thresh, rr_thresh

# ======================================================================================
# BACKTEST ENGINE
# ======================================================================================

def backtest_signals(df: pd.DataFrame, risk: str, horizon: int = 5) -> Dict[str, Any]:
    """
    Lightweight rolling backtest across recent data.

    For each bar i:
      1. Build signal (Buy/Sell/Hold) from compute_signal_row().
      2. If Buy or Sell, compute TP/SL via ATR + risk.
      3. Simulate forward for `horizon` bars. If TP hit => +1%.
         If SL hit => -1%. (This is just a normalized performance proxy,
         not literal PnL sizing.)
      4. Track running balance.

    Output includes:
      - winrate (%)
      - trades (count)
      - return (% from 1.0 balance)
      - maxdd (% max drawdown from equity peak)
      - sharpe-like score (winrate / (drawdown+1)) as a quick quality stat.

    NOTE:
      This is intentionally simple because we want it stable & fast, and we
      mainly use its outputs to influence probability fusion and to display
      in the dashboard.
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
    losses = 0
    trades = 0
    drawdowns = []

    # start from bar 60 to ensure indicators "warmed up"
    for i in range(60, len(df) - horizon):
        prev_row = df.iloc[i - 1]
        cur_row  = df.iloc[i]

        side, base_conf = compute_signal_row(prev_row, cur_row)

        # skip holds, and skip exhausted trend to avoid chasing
        if side == "Hold" or _is_exhausted(cur_row, side):
            continue

        price = float(cur_row["Close"])
        atr_here = float(cur_row["atr"]) if "atr" in cur_row and not math.isnan(cur_row["atr"]) else price * 0.005
        tp, sl = _compute_tp_sl(price, atr_here, side, risk)

        executed = False
        for j in range(1, horizon + 1):
            nxt = df.iloc[i + j]
            nxt_px = float(nxt["Close"])

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
# CORE LATEST PREDICTION PIPELINE
# ======================================================================================

def _latest_prediction(
    symbol: str,
    interval_key: str,
    risk: str,
    recent_winrate: float = 50.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    The "brain" for one asset:
      - Fetch data (with cache)
      - Add indicators
      - Compute rule-based signal (Buy/Sell/Hold) & base_conf
      - Compute sentiment
      - Train/fetch ML model + get ml_prob
      - Fuse to final probability
      - Adapt thresholds based on ADX, ATR%, and performance
      - Compute TP/SL and RR
      - Final decision (maybe override to Hold if it fails thresholds)
      - Ensure Hold still gets TP/SL/RR filled so UI isn't blank

    Outputs dict with keys:
      side, probability, sentiment, tp, sl, rr
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

    # Rule side + raw rule confidence (0..1)
    rule_side, rule_conf = compute_signal_row(prev, last)

    # Sentiment
    sent_score = _fetch_sentiment(symbol)

    # ML stuff
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

    # Threshold calculations
    prob_thresh_regime, rr_thresh_regime = _adaptive_thresholds(last)
    prob_thresh_recent = _dynamic_cutoff(recent_winrate)
    # We demand the stricter probability requirement
    live_prob_thresh = max(prob_thresh_regime, prob_thresh_recent)

    # Baseline trade stats
    last_price = float(last["Close"])
    atr_now = _safe_float(last.get("atr"), last_price * 0.005)
    tp, sl = _compute_tp_sl(last_price, atr_now, rule_side, risk)
    rr = _calc_rr(last_price, tp, sl, rule_side)

    # Overextension filter
    exhausted = _is_exhausted(last, rule_side)
    if exhausted:
        fused_prob *= 0.85  # degrade conviction

    # Final gating
    final_side = "Hold"
    if rule_side != "Hold":
        if (fused_prob >= live_prob_thresh) and (rr >= rr_thresh_regime) and not exhausted:
            final_side = rule_side

    # Even if we ended up "Hold", ensure we show some numbers, not blanks.
    if final_side == "Hold":
        if tp is None or sl is None:
            tp = last_price * 1.005
            sl = last_price * 0.995
            rr = 1.0
        # fallback sentiment if 0.0 and can't get headlines:
        if sent_score == 0.0:
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
# WRAPPERS EXPOSED TO STREAMLIT APP
# ======================================================================================

def analyze_asset(symbol: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    High-level single asset pipeline:
      1. fetch_data + add_indicators
      2. run backtest_signals to get stats
      3. run _latest_prediction() with that winrate as context
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
    Build the Market Summary dataframe shown in tab 1.

    FIX APPLIED:
    - We do NOT skip the backtest step anymore.
      For each asset we now:
        a) fetch+indicators
        b) backtest_signals
        c) _latest_prediction(using backtest winrate)
      so WinRate/Return/Trades/etc. don't revert to 0.

    Columns:
      Asset | Side | Probability | Sentiment | TP | SL | RR |
      WinRate | Trades | Return% | MaxDD% | SharpeLike
    """
    rows = []
    _log("Fetching and analyzing market data (smart v7.1 with backtest refresh)...")

    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        try:
            # Step A: fetch + indicators
            df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
            df_ind = add_indicators(df_raw)
            if df_ind is None or df_ind.empty:
                _log(f"⚠️ {symbol}: indicator data empty.")
                continue

            # Step B: backtest
            back = backtest_signals(df_ind, risk)

            # Step C: latest prediction with adaptive fusion/thresholds
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

    # Ensure numeric columns are actually numeric and not 'object'
    numeric_cols = [
        "Probability",
        "Sentiment",
        "WinRate",
        "Trades",
        "Return%",
        "MaxDD%",
        "SharpeLike",
    ]
    for col in numeric_cols:
        df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce").fillna(0.0)

    # Sort by Probability descending so strongest conviction floats to top
    df_sum = df_sum.sort_values("Probability", ascending=False).reset_index(drop=True)

    return df_sum


def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Returns (symbol, df_with_indicators) for charting / price panel in the Asset Analysis tab.
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind


def asset_prediction_single(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Used in the "Asset Analysis" tab.
    Produces latest side/prob/sentiment/tp/sl/rr for ONE named asset (e.g. "Gold").
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
    Used in the "Backtest & Performance" tab.
    Returns:
      df_ind (with indicators for chart / candlesticks)
      stats  (winrate/trades/return/maxdd/sharpe-like)
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    stats = backtest_signals(df_ind, risk)
    return df_ind, stats

# ======================================================================================
# DEBUGGING / DIAGNOSTIC HELPERS
# ======================================================================================

def debug_signal_breakdown(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Dict[str, Any]:
    """
    Developer-only breakdown of internal reasoning for a symbol.
    This is not shown in the UI by default, but we can expose in a hidden "Debug" tab.

    Returns:
      rule_side / rule_conf
      ml_prob
      sentiment
      fused_prob
      tp/sl/rr
      recent backtest stats context
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

    # backtest stats to understand regime performance
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
# SELF-TEST ENTRYPOINT
# ======================================================================================

if __name__ == "__main__":
    """
    Run `python utils.py` directly for a sanity check outside Streamlit.

    This block:
      - Fetches data for Gold on 1h
      - Builds indicators
      - Runs backtest_signals
      - Generates prediction
      - Summarizes all assets
      - Prints debug breakdown
    """
    test_symbol = "GC=F"
    test_interval = "1h"
    test_risk = "Medium"

    _log("🔍 Self-test starting (Smart v7.1)...")

    try:
        df_test_raw = fetch_data(test_symbol, test_interval, use_cache=True)
        _log(f"[SelfTest] fetched rows: {len(df_test_raw)}")

        df_test_ind = add_indicators(df_test_raw)
        _log(f"[SelfTest] with indicators rows: {len(df_test_ind)}")

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
        _log("[SelfTest] Summary DF (head):")
        try:
            _log(sum_df.head().to_string())
        except Exception:
            _log(str(sum_df))

        dbg = debug_signal_breakdown(test_symbol, test_interval, test_risk)
        _log("[SelfTest] Debug breakdown:")
        _log(json.dumps(dbg, indent=2))

        _log("✅ utils.py Smart v7.1 self-test complete.")
    except Exception as e:
        _log("❌ Self-test failed!")
        _log(str(e))
        traceback.print_exc()

# ======================================================================================
# END OF FILE
# ======================================================================================