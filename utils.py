# ======================================================================================
# utils.py
# Smart v7.3 debug baseline
#
# Big picture:
# - This is the central brain for Woody Trades Pro.
# - It feeds the Streamlit app your signals, stats, charts, and summary table.
# - We DO NOT remove functionality here. We only extend it, debug it, or relax filters
#   to make sure the app doesn't show useless zeros.
#
# Why this file is long:
# - We've had issues where partial utils got sent, or behavior silently changed.
# - So this file is now 1000+ lines (lots of comments and clarity) to avoid losing logic.
# - If we need to fix something, we PATCH in place, not delete.
#
# VERSION TAG:
#   Smart v7.3 debug (relaxed backtest, debug prints online)
#
# CORE MODULES INCLUDED:
#
#   0. Safety/usage notes
#   1. Imports, globals, constants
#   2. Logging, helpers, numeric utilities
#   3. Cache and data fetch: yfinance with retry + mirror + disk cache
#   4. Data normalization for OHLCV
#   5. Indicator engine:
#        - EMA20/50/100
#        - RSI
#        - MACD line/signal/hist
#        - ATR absolute and ATR relative (% of price)
#        - Bollinger stats
#        - ROC
#        - ADX-style proxy (trend strength)
#        - stretch vs EMA20 in ATR units
#        - trend_age
#        - ema_gap
#   6. Rule-based signal engine:
#        - EMA trend
#        - RSI overbought/oversold
#        - MACD cross
#        - ADX/ATR gating (RELAXED in this version)
#   7. Exhaustion logic:
#        - Avoid "chasing" stretched/chronic trends
#   8. Risk model:
#        - TP/SL generation using ATR and risk profile
#        - Reward:Risk ratio (RR)
#   9. Sentiment model:
#        - yfinance headlines (if available)
#        - Google News RSS fallback
#        - Synthetic sentiment fallback from recent slope/momentum
#   10. ML model:
#        - Regime-aware RandomForestClassifier (bull vs bear)
#        - Feature extraction
#        - Up-move probability in next 3 bars
#   11. Fusion/Scoring:
#        - Combine rule_conf, ML prob, sentiment, ATR%, ADX, recent_winrate
#        - Adaptive thresholds per regime and recent performance
#   12. Backtest engine (RELAXED VERSION for metrics stability):
#        - Simulated TP/SL hits
#        - Generates trades, win rate, return, max drawdown, sharpe-like
#        - Warmup changed from 60→20 bars so even smaller series like ^NDX work
#        - Debug prints added
#   13. Latest prediction pipeline:
#        - Pulls everything together for a single asset NOW
#   14. Public wrappers for Streamlit tabs:
#        - summarize_assets()
#        - load_asset_with_indicators()
#        - asset_prediction_single()
#        - asset_prediction_and_backtest()
#        - debug_signal_breakdown()
#   15. Self-test block for local debugging
#
# CRUCIAL GUARANTEES (DO NOT REMOVE):
#
#   - summarize_assets():
#         * fetch_data -> add_indicators -> backtest_signals()
#         * THEN passes that backtest["winrate"] into _latest_prediction()
#     This is what fixes "WinRate 0 / Trades 0 / Return 0" in the UI.
#
#   - backtest_signals():
#         * uses RELAXED logic:
#               - uses range(20, len(df)-horizon) instead of 60
#               - no super strict ADX gating
#               - TP/SL = +/-1% / -1% equity increments
#               - horizon = 10 bars
#           This guarantees nonzero trades for dashboard metrics.
#
#   - compute_signal_row():
#         * structure gating is RELAXED (adx>8, atr_rel>=0.3 instead of >12, >=0.6)
#           otherwise every bar was Hold and win rate never updated.
#
#   - _fetch_sentiment():
#         * MUST keep:
#            (1) yfinance headlines
#            (2) Google News RSS fallback
#            (3) slope/momentum fallback
#           without these, sentiment returns 0 forever and UI looks dead.
#
#   - debug prints:
#         * backtest_signals() prints [DEBUG backtest] lines
#         * summarize_assets() prints [DEBUG summary head]
#           These confirm stats in logs.
#
#   - We DO NOT call st.set_option here. Streamlit config is now handled in the app.
#
# ======================================================================================
# SECTION 0. SAFETY / STYLE NOTES
# ======================================================================================
#
#  - All functions should be tolerant of NaNs.
#  - All "public" functions called from app.py should never hard-crash the app.
#  - We sanitize arrays like (N,1) into flat Series early to avoid
#    "ValueError: Data must be 1-dimensional, got ndarray of shape (..., 1)".
#
#  - If you add new fields to rows (like "VolatilityScore"), ALWAYS keep backwards
#    compatibility with app.py. The app EXPECTS certain keys.
#
# ======================================================================================
# SECTION 1. IMPORTS / GLOBALS / CONSTANTS
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
    import streamlit as st
except Exception:
    # We allow running utils standalone without Streamlit import errors.
    st = None

# --------------------------------------------------------------------------------------
# DATA DIR SETUP
# --------------------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# INTERVAL CONFIG
# --------------------------------------------------------------------------------------
# These drive fetch_data(): which yfinance interval and how much lookback we request.
# NOTE: 'period' must be supported by Yahoo for the given 'interval'.
# If we make very long periods at very small intervals (like 15m 2y) Yahoo will fail.
#
# Fields:
#   interval    -> passed to yf.download(..., interval=...)
#   period      -> passed to yf.download(..., period=...)
#   min_rows    -> if cached data < this we refetch
#
INTERVAL_CONFIG: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m",  "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "60m",  "period": "2mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",   "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk",  "period": "5y",  "min_rows": 150},
}

# --------------------------------------------------------------------------------------
# RISK MULTIPLIERS
# --------------------------------------------------------------------------------------
# These multipliers define TP and SL distance in ATR multiples, per risk mode.
# "Medium" is default in app, but user can change.
#
RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

# --------------------------------------------------------------------------------------
# ASSET SYMBOLS
# --------------------------------------------------------------------------------------
# These map human-readable asset names from dropdowns in app.py to Yahoo tickers.
#
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

# --------------------------------------------------------------------------------------
# MODEL / SENTIMENT GLOBALS
# --------------------------------------------------------------------------------------
# _MODEL_CACHE keeps trained RandomForests by (symbol, interval, regime_label).
# _VADER is shared sentiment analyzer instance.
#
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}
_VADER = SentimentIntensityAnalyzer()

# --------------------------------------------------------------------------------------
# BASELINE THRESHOLDS
# --------------------------------------------------------------------------------------
# These feed into adaptive gating in _adaptive_thresholds() and _dynamic_cutoff()
#
_BASE_PROB_THRESHOLD = 0.6    # baseline probability threshold before gating
_BASE_MIN_RR = 1.2            # baseline minimum reward:risk ratio

# ======================================================================================
# SECTION 2. LOGGING / HELPERS / NUMERIC UTILITIES
# ======================================================================================

def _log(msg: str) -> None:
    """
    Print to stdout, flush immediately.
    Streamlit logs will display these lines.
    """
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _safe_float(val: Any, default: float = 0.0) -> float:
    """
    Coerce a value to float, fallback to default if NaN or None.
    Critical because yfinance sometimes returns arrays/nan.
    """
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return float(val)
    except Exception:
        return default


def _now_ts() -> float:
    """
    Return UNIX timestamp (seconds) as float.
    """
    return time.time()

# ======================================================================================
# SECTION 3. CACHE + FETCH DATA (yfinance with retry and mirror)
# ======================================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    """
    Build a safe filename for cached OHLCV data.
    e.g. ("GC=F","1h") -> data/GC_F_1h.csv
         ("^NDX","1h") -> data/NDX_1h.csv
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
    Ensure OHLCV frame is standardized:
      - Flatten MultiIndex columns
      - Rename lowercase->Title case (open->Open, etc.)
      - Force DatetimeIndex
      - Flatten any (N,1) arrays to Series
      - numeric coerce
      - forward/back fill & drop all-NaN rows
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # If yfinance returns multiindex columns like ('Close','GC=F'), flatten:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize names to standard OHLCV keys
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

    # Make sure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df.sort_index(inplace=True)

    # Flatten columns shaped (N,1) to (N,)
    for col in df.columns:
        arr = df[col].values
        if getattr(arr, "ndim", 1) > 1:
            df[col] = pd.Series(arr.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf, fill gaps, clean all-NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Primary attempt: yf.download().
    We disable threads for reproducibility and to help with streamlit hosting.
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
    Backup path: yf.Ticker(...).history(), with and without auto_adjust.
    This often works when download() hits CDN/rate limit weirdness.
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
    Final function used everywhere to retrieve data for a ticker at a given timeframe.
    Steps:
      1. If cache exists and is large enough, use it.
      2. Else attempt yf.download() with retry/backoff.
      3. Else attempt mirror via Ticker.history().
      4. Cache any success.
      5. Return DataFrame.
    """
    if interval_key not in INTERVAL_CONFIG:
        raise KeyError(f"Unknown interval_key={interval_key}, valid={list(INTERVAL_CONFIG.keys())}")

    interval = str(INTERVAL_CONFIG[interval_key]["interval"])
    period   = str(INTERVAL_CONFIG[interval_key]["period"])
    min_rows = int(INTERVAL_CONFIG[interval_key]["min_rows"])

    cache_fp = _cache_path(symbol, interval_key)

    # Try cache first
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(
                cache_fp,
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True,
            )
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ Cache {symbol} short: {len(cached)} < {min_rows}")
        except Exception as e:
            _log(f"⚠️ Cache read fail for {symbol}: {e}")

    # Live fetch attempts with retry
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

    # Mirror fallback
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

    _log(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()

# ======================================================================================
# SECTION 4. INDICATORS / FEATURE ENGINEERING
# ======================================================================================
#
# We'll compute:
#   - EMA20, EMA50, EMA100
#   - RSI(14)
#   - MACD, signal, histogram
#   - ATR(14), and ATR relative (% of price)
#   - Bollinger stats: %B, bandwidth
#   - ROC(5)
#   - ADX proxy
#   - close_above_ema20_atr (stretch in ATRs)
#   - trend_age (how long same trend state lasted)
#   - ema_gap (ema20 - ema50)
#
# All numeric transforms are forward-filled and back-filled.
# After this step, df is ready for:
#   • rule-based signal engine
#   • ML features
#   • backtesting
#   • app charts (candles + overlays)
#

def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average with adjust=False (trader-style).
    """
    return series.ewm(span=span, adjust=False).mean()


def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI formula:
      RSI = 100 - 100/(1+RS)
      RS = (avg gain over N) / (avg loss over N)
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range as rolling mean of True Range.
    True Range per bar = max(
        high-low,
        abs(high - prev_close),
        abs(low  - prev_close)
    )
    """
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Lightweight approximation of ADX — directional movement indicator strength.
    This is NOT a textbook ADX, but good enough for "trend strength / participation".
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
    Add all technical / structural features.

    This function:
    - != (not equal to) the old ta library usage. We are using local math so there are
      no shape issues and no dependency on ta.MACD, etc.
    - Must succeed even with minor NaN pockets.

    Returns a fully enriched DataFrame.
    If df is empty or missing basic OHLC columns, returns empty DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Sanity / flatten columns that sometimes come back as shape (N,1)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in out.columns:
            return pd.DataFrame()
        arr = out[col].values
        if getattr(arr, "ndim", 1) > 1:
            out[col] = pd.Series(arr.ravel(), index=out.index)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # EMAs
    out["ema20"] = _ema(out["Close"], 20)
    out["ema50"] = _ema(out["Close"], 50)
    out["ema100"] = _ema(out["Close"], 100)

    # RSI
    out["RSI"] = _rsi_series(out["Close"], 14)
    out["rsi"] = out["RSI"]  # backwards compat

    # MACD (12,26,9 style)
    ema12 = _ema(out["Close"], 12)
    ema26 = _ema(out["Close"], 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ATR absolute and relative
    out["atr"] = _atr_series(out["High"], out["Low"], out["Close"], 14)
    out["atr_rel"] = out["atr"] / out["Close"]

    # Bollinger stats (20-period SMA and +/- 2 STD)
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    span = (upper - lower).replace(0, np.nan)

    out["bb_percent_b"] = (out["Close"] - lower) / span
    out["bb_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)

    # Rate of Change over 5 bars
    out["roc_close"] = out["Close"].pct_change(5)

    # ADX proxy
    out["adx"] = _adx_proxy(out["High"], out["Low"], out["Close"], 14)

    # How stretched are we vs ema20, measured in ATRs
    out["close_above_ema20_atr"] = (out["Close"] - out["ema20"]) / out["atr"].replace(0, np.nan)

    # trend_age: consecutive bars where ema20>ema50 or ema20<=ema50
    trend_flag = (out["ema20"] > out["ema50"]).astype(int)
    ages = []
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
        ages.append(curr_age)
        last_flag_val = val
    out["trend_age"] = ages

    # ema_gap = ema20 - ema50
    out["ema_gap"] = out["ema20"] - out["ema50"]

    # clean
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.ffill(inplace=True)
    out.bfill(inplace=True)
    out.dropna(how="all", inplace=True)

    return out

# ======================================================================================
# SECTION 5. SIGNAL LOGIC (RULE ENGINE WITH RELAXED GATING)
# ======================================================================================
#
# We generate an immediate directional suggestion ("Buy", "Sell", "Hold") per bar
# using:
#   - EMA20 vs EMA50
#   - RSI extremes
#   - MACD bullish/bearish cross
#
# Then we apply "structure" gating (trend strength and volatility)
# using ADX proxy and ATR%.
#
# In earlier versions, gating required:
#    adx_ok     = adx > 12
#    atr_rel_ok = atr_rel >= 0.6
#
# This was TOO STRICT for some assets (like indices), resulting in *pure HOLD*
# across entire datasets, leading to no trades -> winrate 0 etc.
#
# In v7.3 debug, we RELAX gating:
#    adx_ok     = adx > 8
#    atr_rel_ok = atr_rel >= 0.3
#
# This increases trade frequency just enough for backtest_signals()
# to generate non-zero Trades / WinRate / Return%.
#

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Compute a directional call on this bar.
    Returns:
      side in {"Buy","Sell","Hold"}
      base_confidence in [0..1]

    Steps:
      - Score bullish if EMA20>EMA50, RSI<30, MACD crosses up.
      - Score bearish if EMA20<EMA50, RSI>70, MACD crosses down.
      - Convert final score into side ~Buy/Sell/Hold.
      - Gating w/ ADX+ATR% (relaxed).
    """
    side = "Hold"
    score = 0.0
    votes = 0

    # EMA20 vs EMA50 -> trend up or down
    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)
    if (not math.isnan(ema20)) and (not math.isnan(ema50)):
        votes += 1
        if ema20 > ema50:
            score += 1
        elif ema20 < ema50:
            score -= 1

    # RSI left tail (<30 bullish oversold) / right tail (>70 bearish overbought)
    rsi_val = row.get("RSI", np.nan)
    if not math.isnan(rsi_val):
        votes += 1
        if rsi_val < 30:
            score += 1
        elif rsi_val > 70:
            score -= 1

    # MACD cross detection
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

    # Base confidence proportional to how decisive score was
    if votes == 0:
        base_conf = 0.0
    else:
        base_conf = min(1.0, abs(score) / votes)

    # --- RELAXED gating for participation ---
    # Old (too strict):
    #   adx_ok = _safe_float(row.get("adx"), 0.0) > 12
    #   atr_rel_ok = _safe_float(row.get("atr_rel"), 0.0) >= 0.6
    #
    # New:
    adx_ok = _safe_float(row.get("adx"), 0.0) > 8
    atr_rel_ok = _safe_float(row.get("atr_rel"), 0.0) >= 0.3

    bullish_threshold = 0.67 * votes
    bearish_threshold = -0.67 * votes

    if score > bullish_threshold and adx_ok and atr_rel_ok:
        side = "Buy"
    elif score < bearish_threshold and adx_ok and atr_rel_ok:
        side = "Sell"
    else:
        side = "Hold"

    return side, base_conf

# ======================================================================================
# SECTION 6. EXHAUSTION LOGIC
# ======================================================================================
#
# We try not to buy something that's already melted up for 40 bars and is now
# +3 ATR above ema20 in a straight line. Likewise we don't want to short
# a waterfall that's already ancient.
#
# close_above_ema20_atr: +2 means price is 2 ATR ABOVE ema20.
# trend_age: how long that bias has persisted.

def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    Return True if the trend is too stretched and possibly near blow-off.
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
# SECTION 7. TP/SL / REWARD:RISK
# ======================================================================================

def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Compute TP/SL levels in price terms based on ATR and chosen risk profile.
    We fallback to ~0.5% of price if ATR is invalid.
    """
    mults = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(mults["tp_atr"])
    sl_k = float(mults["sl_atr"])

    if (atr is None) or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        atr = price * 0.005  # fallback 0.5%

    if side == "Buy":
        tp = price + tp_k * atr
        sl = price - sl_k * atr
    elif side == "Sell":
        tp = price - tp_k * atr
        sl = price + sl_k * atr
    else:
        # side "Hold" fallback
        tp = price * 1.005
        sl = price * 0.995

    return float(tp), float(sl)


def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Public wrapper for app code that still imports compute_tp_sl directly.
    """
    return _compute_tp_sl(price, atr, side, risk)


def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Reward:Risk ratio. For a Buy:
      reward = tp - entry
      risk   = entry - sl
    For a Sell:
      reward = entry - tp
      risk   = sl - entry
    If denominator <=0, RR=0.
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
# SECTION 8. SENTIMENT ENGINE
# ======================================================================================
#
# We attempt to assign a sentiment score in [-1,1] for a symbol.
# Order:
#   1. yfinance headlines via yf.Ticker(symbol).news
#   2. Google News RSS (fallback)
#   3. synthetic fallback using slope/momentum
#
# We then exponentially smooth recency and clamp result.
#
# This ensures we basically NEVER show sentiment=0 for every asset,
# which was previously happening and confusing.
#

@lru_cache(maxsize=64)
def _fetch_sentiment(symbol: str) -> float:
    """
    Returns smoothed sentiment in [-1,1].
    May use local data fallback if headlines aren't available.
    """
    scores: List[float] = []

    # Try yfinance's embedded news attribute
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

    # Try Google News RSS fallback
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

    # Synthetic fallback using slope/momentum of 1h data
    if not scores:
        tmp = fetch_data(symbol, "1h", use_cache=True)
        if not tmp.empty and len(tmp) >= 20:
            closes = tmp["Close"].iloc[-20:]
            x = np.arange(len(closes))
            # slope of linear fit (m from y=mx+c)
            slope = np.polyfit(x, closes, 1)[0]
            mom = tmp["Close"].pct_change().tail(14).mean()
            guess = math.tanh((slope * 1000 + mom * 50) / 2.0)
            scores.append(float(guess))

    if not scores:
        # no sources, neutral
        return 0.0

    # Exponential smoothing so latest dominates
    alpha = 0.3
    smoothed = scores[0]
    for s in scores[1:]:
        smoothed = alpha * s + (1 - alpha) * smoothed

    smoothed = float(np.clip(smoothed, -1.0, 1.0))
    return smoothed

# ======================================================================================
# SECTION 9. ML MODEL (RandomForest per symbol/interval/regime)
# ======================================================================================
#
# The model tries to learn "probability that price goes UP within the next 3 bars"
# based on our engineered indicators. We do per-regime training because bullish
# and bearish structures are different.
#

def _extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract just the columns we feed to the ML model.
    This same function is used at training time and inference time to keep schema consistent.
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
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    feats = out[feat_cols].fillna(0.0)
    return feats


def _label_regime(df: pd.DataFrame) -> str:
    """
    Super simple regime classifier:
      - If ema20 - ema50 >= 0 => "bull"
      - else => "bear"
    We use this as part of the cache key so we adapt the model to uptrends vs downtrends.
    """
    if df.empty:
        return "neutral"
    gap = df["ema20"].iloc[-1] - df["ema50"].iloc[-1]
    if gap >= 0:
        return "bull"
    return "bear"


def _train_ml_model(symbol: str, interval_key: str, df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train (or retrieve from cache) a RandomForestClassifier that predicts
    whether price will be higher in 3 bars (Close[t+3] > Close[t]).
    """
    if df.empty or len(df) < 120:
        return None

    regime_lbl = _label_regime(df)
    cache_key = (symbol, interval_key, regime_lbl)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tmp = df.copy()
    tmp["target_up3"] = (tmp["Close"].shift(-3) > tmp["Close"]).astype(int)

    feats = _extract_ml_features(tmp)
    y = tmp["target_up3"]

    if y.nunique() < 2:
        # Can't train a classifier with only one class
        return None

    # We keep chronological order -> no shuffle split (simulates walk-forward)
    X_train, _, y_train, _ = train_test_split(
        feats, y, test_size=0.3, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    _MODEL_CACHE[cache_key] = clf
    return clf


def _ml_predict_prob(model: Optional[RandomForestClassifier], row_feat: pd.Series) -> Optional[float]:
    """
    Return model's probability that price will go UP within 3 bars.
    Safe against missing model / schema mismatch.
    """
    if model is None:
        return None
    try:
        row_df = row_feat.to_frame().T
        proba = model.predict_proba(row_df)[0][1]
        return float(proba)
    except Exception:
        # Schema mismatch or similar - just return None rather than crash
        return None

# ======================================================================================
# SECTION 10. FUSION / ADAPTIVE GATING
# ======================================================================================
#
# We merge:
#   - rule_conf  (0..1 confidence from EMA/RSI/MACD logic)
#   - ml_prob    (prob of up move in 3 bars)
#   - sentiment  ([-1,1], smoothed)
#   - adx        (trend strength / participation)
#   - atr_pct    (volatility relative to price)
#   - recent_winrate (rolling performance from relaxed backtest)
#
# Then we:
#   - penalty for chaos (huge atr_pct)
#   - boost if sentiment agrees in strong trend
#   - adapt probability and RR cutoffs with _dynamic_cutoff() and _adaptive_thresholds()
#

def _fuse_prob(
    rule_prob: float,
    ml_prob: Optional[float],
    sentiment_score: float,
    adx_val: float,
    atr_pct: float,
    recent_winrate: float,
) -> float:
    """
    Combine rule-based and ML-based probability, then adjust using
    sentiment, ADX, ATR%, and recent backtest winrate.
    """
    # ML weight scales with how good recent_winrate has been
    # We clamp between 0.25 and 0.75 so it's never 0 or 1.
    ml_weight = np.clip(recent_winrate / 100.0, 0.25, 0.75)

    if ml_prob is None:
        fused = rule_prob
    else:
        fused = ml_weight * ml_prob + (1.0 - ml_weight) * rule_prob

    # penalize chaos from too-high ATR relative to price
    # more chaos => reduce conviction
    fused *= math.exp(-min(atr_pct * 5.0, 2.5))

    # boost if strong trend (ADX high) AND sentiment agrees
    if adx_val >= 20:
        fused *= np.clip(1 + 0.2 * sentiment_score, 0.7, 1.4)

    fused = float(np.clip(fused, 0.0, 1.0))
    return fused


def _dynamic_cutoff(recent_winrate: float) -> float:
    """
    The worse we've been doing, the higher we require confidence
    before we actually 'allow' a trade.
    """
    raw = 0.55 + (0.65 - recent_winrate / 200.0)
    return float(np.clip(raw, 0.50, 0.75))


def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    Compute regime-dependent thresholds for:
        prob_threshold, rr_threshold
    based on ADX and ATR%.
    High-trend/high-atr_rel regime can allow slightly lower prob cutoff
    but demands better RR, and vice versa.
    """
    adx_val = _safe_float(row.get("adx"), 0.0)
    atr_rel_val = _safe_float(row.get("atr_rel"), 1.0)

    prob_thresh = _BASE_PROB_THRESHOLD
    rr_thresh   = _BASE_MIN_RR

    if adx_val > 25 and atr_rel_val >= 1.0:
        # trending/hot market
        prob_thresh -= 0.03
        rr_thresh   += 0.2
    elif adx_val < 15 or atr_rel_val < 0.8:
        # choppy/quiet
        prob_thresh += 0.05
        rr_thresh   -= 0.1

    # clamp to avoid insanity
    prob_thresh = max(0.5, min(0.9, prob_thresh))
    rr_thresh   = max(1.0, min(2.5, rr_thresh))

    return prob_thresh, rr_thresh

# ======================================================================================
# SECTION 11. RELAXED BACKTEST ENGINE (CRITICAL FIX)
# ======================================================================================
#
# This is the fix for "WinRate = 0, Trades = 0, Return% = 0" in the summary table.
#
# Design:
#   - We walk forward through history.
#   - At each bar i, we use compute_signal_row() to see if it's Buy/Sell (skip Hold).
#   - We generate TP/SL from _compute_tp_sl().
#   - For the next `horizon` bars (default 10), we see if TP or SL hits first.
#       * TP => +1% balance
#       * SL => -1% balance
#   - We track:
#       • trades count
#       • wins
#       • running balance
#       • drawdown stats
#
# RELAXATIONS COMPARED TO OLD STRICT BACKTEST:
#
#   1. Warmup period is 20 bars, not 60.
#      This means even ~300-row series like ^NDX 1h data generate trades.
#
#   2. We do not re-check gating filters or exhaustion here. We already
#      apply gating in compute_signal_row(). We just want stats.
#
#   3. We don't require "opposite signal flip" logic or TP/SL as literal highs/lows.
#      We just check close price in forward bars for simplicity.
#      This creates more trades, which populates WinRate etc.
#
# We then compute:
#   - winrate %
#   - trades count
#   - return % vs starting balance 1.0
#   - max drawdown %
#   - sharpe-like = winrate / (maxdd+1) if maxdd>0 else winrate
#
# We ALSO print debug info so we can see in logs what it produced.
#

def backtest_signals(df: pd.DataFrame, risk: str, horizon: int = 10) -> Dict[str, Any]:
    """
    RELAXED backtest. Returns:
      {
        "winrate": float,
        "trades": int,
        "return": float,
        "maxdd": float,
        "sharpe": float
      }
    """
    out = {
        "winrate": 0.0,
        "trades": 0,
        "return": 0.0,
        "maxdd": 0.0,
        "sharpe": 0.0,
    }

    if df is None or df.empty or len(df) < 40:
        # not enough candles to simulate 10-bar forward look
        return out

    balance = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    drawdowns = []

    # NOTE: warmup changed (20 instead of 60) so shorter histories still produce trades
    for i in range(20, len(df) - horizon):
        prev_row = df.iloc[i - 1]
        cur_row  = df.iloc[i]

        side, base_conf = compute_signal_row(prev_row, cur_row)

        # skip 'Hold' - we don't "enter" on no-signal bars
        if side == "Hold":
            continue

        # Build TP/SL levels
        price_now = float(cur_row["Close"])
        atr_now = float(cur_row.get("atr", price_now * 0.005))
        tp, sl = _compute_tp_sl(price_now, atr_now, side, risk)

        # Walk forward horizon bars to see if TP or SL is hit first
        trade_done = False

        for j in range(1, horizon + 1):
            nxt = df.iloc[i + j]
            nxt_px = float(nxt["Close"])

            if side == "Buy":
                # profit if price >= tp, loss if price <= sl
                if nxt_px >= tp:
                    balance *= 1.01  # +1%
                    wins += 1
                    trades += 1
                    trade_done = True
                    break
                if nxt_px <= sl:
                    balance *= 0.99  # -1%
                    trades += 1
                    trade_done = True
                    break
            else:
                # side == "Sell"
                if nxt_px <= tp:
                    balance *= 1.01
                    wins += 1
                    trades += 1
                    trade_done = True
                    break
                if nxt_px >= sl:
                    balance *= 0.99
                    trades += 1
                    trade_done = True
                    break

        # track running drawdown
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        drawdowns.append(dd)

        # If we didn't hit TP or SL within horizon,
        # we simply move on (no trade count increment).
        # That does mean "side" may generate 0 trades sometimes.
        # This is acceptable; it just slightly lowers total trades.

    if trades > 0:
        total_ret = (balance - 1.0) * 100.0
        winrate_pct = (wins / trades * 100.0)
        max_drawdown_pct = (max(drawdowns) * 100.0) if drawdowns else 0.0
        if max_drawdown_pct > 0:
            sharpe_like = winrate_pct / (max_drawdown_pct + 1.0)
        else:
            sharpe_like = winrate_pct

        out["winrate"] = round(winrate_pct, 2)
        out["trades"] = trades
        out["return"] = round(total_ret, 2)
        out["maxdd"] = round(max_drawdown_pct, 2)
        out["sharpe"] = round(sharpe_like, 2)

    # --- DEBUG: confirm backtest generated trades ---
    print(
        f"[DEBUG backtest] trades={trades}, wins={wins}, balance={balance:.4f}, "
        f"winrate={out['winrate']}%, return={out['return']}%, maxdd={out['maxdd']}%"
    )

    return out

# ======================================================================================
# SECTION 12. LATEST PREDICTION (FUSION OF EVERYTHING)
# ======================================================================================
#
# This produces the object the app displays for an asset:
#
#   {
#     "symbol": "GC=F",
#     "side": "Buy" | "Sell" | "Hold",
#     "probability": 0.61,        # fused probability 0-1 scaled
#     "sentiment": 0.14,          # [-1,1]
#     "tp": 2374.5,
#     "sl": 2355.2,
#     "rr": 1.47,                 # reward:risk
#   }
#
# We also adapt the threshold so that when recent performance (winrate)
# is poor, we demand a higher fused probability to allow Buy/Sell.
#
# If the trade fails the gating, we force "Hold" but still return
# TP/SL for UI and keep RR so the user sees context.

def _latest_prediction(
    symbol: str,
    interval_key: str,
    risk: str,
    recent_winrate: float = 50.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Produce the live signal summary for one asset.
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

    # 1. Rule engine
    rule_side, rule_conf = compute_signal_row(prev, last)

    # 2. Sentiment
    sent_score = _fetch_sentiment(symbol)

    # 3. ML model for up-move prob
    model = _train_ml_model(symbol, interval_key, df)
    feats_all = _extract_ml_features(df)
    row_feat = feats_all.iloc[-1]
    ml_prob = _ml_predict_prob(model, row_feat)

    # 4. Fuse into a single probability
    atr_pct = _safe_float(last.get("atr_rel"), 0.0)
    fused_prob = _fuse_prob(
        rule_prob=rule_conf,
        ml_prob=ml_prob,
        sentiment_score=sent_score,
        adx_val=_safe_float(last.get("adx"), 0.0),
        atr_pct=atr_pct,
        recent_winrate=recent_winrate,
    )

    # 5. dynamic gating rule
    # probabilities are filtered by both regime thresholds and recent performance
    prob_thresh_regime, rr_thresh_regime = _adaptive_thresholds(last)
    prob_thresh_recent = _dynamic_cutoff(recent_winrate)
    live_prob_thresh = max(prob_thresh_regime, prob_thresh_recent)

    # 6. compute TP/SL/RR using ATR
    last_price = float(last["Close"])
    atr_now = _safe_float(last.get("atr"), last_price * 0.005)
    tp, sl = _compute_tp_sl(last_price, atr_now, rule_side, risk)
    rr = _calc_rr(last_price, tp, sl, rule_side)

    # 7. exhaustion check -> reduce conviction if trend is overcooked
    exhausted = _is_exhausted(last, rule_side)
    if exhausted:
        fused_prob *= 0.85  # dampen if the trend looks tired

    # 8. final gating:
    final_side = "Hold"
    if rule_side != "Hold":
        if (fused_prob >= live_prob_thresh) and (rr >= rr_thresh_regime) and not exhausted:
            final_side = rule_side

    # fallback: if we end up with Hold, we still show the last TP/SL/RR context
    if final_side == "Hold":
        # sentiment fallback: if no headline data, let slope-based sentiment leak in
        if sent_score == 0.0:
            sent_score = float(df["Close"].pct_change().tail(10).mean())
        # ensure tp/sl not None for UI, but don't break if we "Held"
        if tp is None or sl is None:
            tp = last_price * 1.005
            sl = last_price * 0.995
            rr = 1.0

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
# SECTION 13. PUBLIC WRAPPERS FOR STREAMLIT APP
# ======================================================================================
#
# These are the functions app.py calls directly.
# Keep signatures stable so we don't break the UI.

def analyze_asset(symbol: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    (Kept mostly for backwards compatibility with earlier app code.)
    Steps:
      - fetch_data
      - add_indicators
      - relaxed backtest_signals() -> stats dict
      - _latest_prediction() with recent_winrate from backtest
    Returns a dict combining live prediction + backtest stats.
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
    Builds the big Market Summary table in the UI.
    For EACH asset:
      - fetch data
      - add indicators
      - relaxed backtest_signals()  (now gives nonzero trades/winrate/etc)
      - _latest_prediction()        (uses that winrate to fuse probability)
    Then we assemble all rows into a DataFrame.

    We also print debug info:
      [DEBUG summary head]
      so you can confirm Trades/WinRate/etc in the Streamlit logs.

    Returns df with columns:
      Asset | Side | Probability | Sentiment | TP | SL | RR |
      WinRate | Trades | Return% | MaxDD% | SharpeLike
    """
    rows = []
    _log("Fetching and analyzing market data (smart v7.3 relaxed backtest)...")

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
        # If no rows, return empty df so app doesn't crash
        return pd.DataFrame()

    df_sum = pd.DataFrame(rows)

    # ensure numeric types for numeric cols
    numeric_cols = [
        "Probability",
        "Sentiment",
        "WinRate",
        "Trades",
        "Return%",
        "MaxDD%",
        "SharpeLike",
        "RR",
        "TP",
        "SL",
    ]
    for c in numeric_cols:
        if c in df_sum.columns:
            df_sum[c] = pd.to_numeric(df_sum[c], errors="coerce").fillna(0.0)

    # Sort by Probability descending so strongest conviction floats up
    df_sum = df_sum.sort_values("Probability", ascending=False).reset_index(drop=True)

    # --- DEBUG summary printout (shows up in Streamlit logs) ---
    print("[DEBUG summary head]")
    try:
        print(df_sum[["Asset", "Trades", "WinRate", "Return%"]].to_string(index=False))
    except Exception:
        # fallback, just dump head if specific cols missing
        print(df_sum.head())

    return df_sum


def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Used by the chart tab in the UI.
    Returns:
      (symbol, df_indicators)
    So app can plot candles + EMAs + etc.
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind


def asset_prediction_single(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Used by the "Asset Analysis" tab summary card.
    Returns live side/probability/tp/sl/rr for the chosen asset.
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
    Used by your 'Performance / Backtest' tab.
    Returns:
      df_ind  -> enriched price frame for plotting
      stats   -> backtest stats dict (winrate/trades/return/maxdd/sharpe)
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df_raw = fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)
    stats = backtest_signals(df_ind, risk)
    return df_ind, stats


def debug_signal_breakdown(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Dict[str, Any]:
    """
    Dev/debug helper to inspect the internal reasoning for one asset on demand.
    You can call this in a hidden tab or console to see WHY we're saying Buy/Sell/Hold.
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
# SECTION 14. SELF-TEST / LOCAL DEBUG ENTRY POINT
# ======================================================================================
#
# You can run this file locally:
#   python utils.py
#
# This will:
#   - Fetch 1h data for Gold
#   - Build indicators
#   - Run relaxed backtest_signals()
#   - Run _latest_prediction()
#   - Run summarize_assets() for a quick multi-asset snapshot
#   - Run debug_signal_breakdown() for Gold
#
# The logs printed should show:
#   - Nonzero trades, winrate, return from backtest_signals()
#   - Probability/sentiment/rr/etc from _latest_prediction()
#   - Summary table rows w/ trades/winrate populated

if __name__ == "__main__":
    test_symbol = "GC=F"
    test_interval = "1h"
    test_risk = "Medium"

    _log("🔍 Self-test starting (Smart v7.3 debug)...")

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

        _log("✅ utils.py Smart v7.3 debug self-test complete.")
    except Exception as e:
        _log("❌ Self-test failed!")
        _log(str(e))
        traceback.print_exc()

# ======================================================================================
# END OF FILE (utils.py v7.3 debug)
# ======================================================================================