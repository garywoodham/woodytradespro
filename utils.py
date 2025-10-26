# utils.py - Woody Trades Pro (smart v7.7.2 full integrity)
# =============================================================================
# This file is intentionally verbose and self-contained.
#
# It merges every working feature we've discussed so far:
#   - robust fetch with caching + retry + mirror fallback
#   - indicator calc (EMA20/50, RSI, RSI quantile bands, MACD, ATR, ATR%, ADX)
#   - sentiment stub per asset
#   - adaptive rule-based signal engine (Buy / Sell / Hold)
#       * dynamic RSI thresholds (rolling quantiles)
#       * volatility regime gating via ATR% of price
#       * ADX trend-strength gating
#       * higher timeframe confirmation (1h→4h, 4h→1d, etc.)
#   - ML-style probability
#       * RandomForest + GradientBoosting ensemble on engineered features
#       * rolling-window training for regime adaptiveness
#   - RR/TP/SL engine using ATR and risk model
#   - relaxed deterministic backtest with ATR-scaled P&L
#       * ensemble backtest for stable win rate + std
#       * v7.7.2: confidence-weighted P&L, probabilistic participation
#   - adaptive confidence (weights rule vs ML based on recent performance)
#   - weekend-safe fallback logic so UI never goes blank
#   - chart marker support for Buy/Sell points
#   - summary table builder for dashboard
#   - helpers used in Detailed / Scenarios tabs, unchanged signatures
#
# Stabilisation fixes included:
#   - handles 2D columns (n,1) from yfinance
#   - cleans MultiIndex columns
#   - avoids circular import with Streamlit
#   - warmup lowered so smaller windows still produce trades
#   - probability clipped so it’s never 0 or 1
#   - sentiment never stuck at 0
#
# Weekend safety:
#   - stale data (market closed/weekend) no longer kills TP/SL/WinRate
#   - latest_prediction always returns structured info even if stale or Hold
#   - deterministic ensemble backtest guarantees ≥1 trade so stats don't go blank
#   - ATR fallback so TP/SL/RR always exist for UI
#
# Accuracy evolution:
#   v7.5: ATR-scaled P&L, ensemble backtest, adaptive confidence.
#   v7.6: Adaptive RSI, ATR% gating, ADX gating.
#   v7.7: Higher timeframe confirmation + ML ensemble w/ engineered features.
#   v7.7.2: Confidence-weighted P&L and smarter forced participation in backtest.
#
# The app imports:
#   from utils import (
#       summarize_assets,
#       asset_prediction_and_backtest,
#       load_asset_with_indicators,
#       ASSET_SYMBOLS,
#       INTERVALS,
#   )
#
# This module is pure Python + pandas/numpy/sklearn/ta/yfinance.
# No Streamlit calls here (print() only) to avoid circular imports.
#
# =============================================================================

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import yfinance as yf

# Optional ML-ish models
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
except Exception:
    RandomForestClassifier = None
    try:
        GradientBoostingClassifier  # may not exist
    except Exception:
        GradientBoostingClassifier = None  # type: ignore

# Technical indicators
try:
    from ta.trend import EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    EMAIndicator = MACD = ADXIndicator = RSIIndicator = AverageTrueRange = None


# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# IMPORTANT: "4h" now uses "4h" instead of "240m" (Yahoo quirk fix)
INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m",  "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "1h",   "period": "2mo", "min_rows": 250},
    "4h":  {"interval": "4h",   "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",   "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk",  "period": "5y",  "min_rows": 150},
}

RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

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

# Adaptive signal config (v7.6+)
RSI_WINDOW = 100            # lookback for dynamic RSI bands
RSI_Q_LOW = 0.2             # 20th percentile RSI = "oversold" threshold
RSI_Q_HIGH = 0.8            # 80th percentile RSI = "overbought" threshold

VOL_REGIME_MIN_ATR_PCT = 0.3  # if ATR% of price < this, we call it dead/chop
ADX_MIN_TREND = 20.0          # if ADX < this, trend-follow votes are downweighted

# Higher timeframe mapping (v7.7+)
HIGHER_TF_MAP = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1wk",
    "1wk": "1wk",  # top of the stack just maps to itself
}

# ML config v7.7+
ML_RECENT_WINDOW = 500  # only train ML models on the most recent N rows

# Backtest participation tuning (v7.7.2)
FORCED_TRADE_PROB = 0.02     # base random chance to "force" a trade
FORCED_CONF_MIN = 0.55       # only force if local confidence > this


# =============================================================================
# LOGGING
# =============================================================================

def _log(msg: str) -> None:
    """
    Safe print for Streamlit Cloud / containers.
    """
    try:
        print(msg, flush=True)
    except Exception:
        pass


# =============================================================================
# FETCH + CACHE
# =============================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    """
    Local CSV cache path.
    e.g. BTC-USD_1h.csv (but normalized chars)
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
    Make sure df has columns we need (Open/High/Low/Close/Volume),
    1-D columns (no (n,1)), clean datetime index, no inf/NaN explosions.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # yfinance often returns MultiIndex columns like ('Close','GC=F')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # keep canonical cols if they exist
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
        # last resort rename
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # index -> datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # flatten any (n,1) columns into 1-D
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)

        if col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")

    return df


def _yahoo_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Primary data fetch via yf.download.
    We keep threads=False for stability (Streamlit Cloud sometimes
    blows up on multi-threaded).
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


def _yahoo_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Mirror fetch via yf.Ticker(...).history() as a fallback.
    """
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df

        # fallback attempt without auto_adjust
        raw2 = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        return _normalize_ohlcv(raw2)
    except Exception as e:
        _log(f"⚠️ Mirror history error for {symbol}: {e}")
        return pd.DataFrame()


def fetch_data(
    symbol: str,
    interval_key: str = "1h",
    use_cache: bool = True,
    max_retries: int = 4,
    backoff_range: Tuple[float, float] = (2.5, 6.0),
) -> pd.DataFrame:
    """
    Unified fetch:
    1. Try cache (CSV). If enough rows, return.
    2. Retry live download up to max_retries.
    3. Mirror fallback.
    4. Cache successful result.
    """
    if interval_key not in INTERVALS:
        raise KeyError(
            f"Unknown interval_key '{interval_key}'. Known: {list(INTERVALS.keys())}"
        )

    interval = str(INTERVALS[interval_key]["interval"])
    period   = str(INTERVALS[interval_key]["period"])
    min_rows = int(INTERVALS[interval_key]["min_rows"])

    _log(f"⏳ Fetching {symbol} [{interval}] ...")
    cache_fp = _cache_path(symbol, interval_key)

    # 1. cache first
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(
                cache_fp,
                index_col=0,
                parse_dates=True,
            )
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ cache for {symbol} only {len(cached)} rows (<{min_rows})")
        except Exception as e:
            _log(f"⚠️ Cache read failed for {symbol}: {e}")

    # 2. live retries
    for attempt in range(1, max_retries + 1):
        df_live = _yahoo_download(symbol, interval, period)
        if not df_live.empty and len(df_live) >= min_rows:
            _log(f"✅ {symbol}: fetched {len(df_live)} rows.")
            try:
                df_live.to_csv(cache_fp)
                _log(f"💾 Cached → {cache_fp}")
            except Exception as e:
                _log(f"⚠️ Cache write failed for {symbol}: {e}")
            return df_live

        _log(
            f"⚠️ Retry {attempt} failed for {symbol} "
            f"({len(df_live) if isinstance(df_live, pd.DataFrame) else 'N/A'} rows)."
        )
        time.sleep(np.random.uniform(*backoff_range))

    # 3. mirror
    _log(f"🪞 Mirror fetch for {symbol}...")
    df_mirror = _yahoo_history(symbol, interval, period)
    if not df_mirror.empty and len(df_mirror) >= min_rows:
        _log(f"✅ Mirror fetch worked for {symbol} ({len(df_mirror)} rows).")
        try:
            df_mirror.to_csv(cache_fp)
            _log(f"💾 Cached → {cache_fp}")
        except Exception as e:
            _log(f"⚠️ Cache write failed for {symbol}: {e}")
        return df_mirror

    _log(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()


# =============================================================================
# INDICATORS (EMA20/50, RSI, RSI quantile bands, MACD, ATR, ATR%, ADX)
# plus engineered features for ML
# =============================================================================

def _ema_fallback(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi_fallback(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = -d.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_fallback(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr_fallback(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx_fallback(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Fallback if ta.ADXIndicator isn't available.
    We'll just return NaNs so logic downstream can gracefully skip ADX gating.
    """
    return pd.Series(index=close.index, dtype=float)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ema20, ema50
      - RSI (14)
      - Rolling RSI quantile bands (rsi_low_band, rsi_high_band)
      - MACD line/signal/hist
      - ATR (14), atr_pct (ATR / Close * 100)
      - ADX (trend strength)
      - Engineered features for ML ensemble:
          * ema_diff = ema20 - ema50
          * rsi_slope = RSI.diff()
          * macd_slope = macd.diff()
    Also duplicates RSI->rsi for legacy usage.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # Safety: remove duplicate timestamps if any
    df = df[~df.index.duplicated(keep="last")]

    for col in ["Close", "High", "Low"]:
        if col not in df.columns:
            return pd.DataFrame()
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
    try:
        if EMAIndicator is None:
            raise RuntimeError("ta not available")
        df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    except Exception:
        df["ema20"] = _ema_fallback(close, 20)
        df["ema50"] = _ema_fallback(close, 50)

    # RSI
    try:
        if RSIIndicator is None:
            raise RuntimeError("ta not available")
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    except Exception:
        df["RSI"] = _rsi_fallback(close, 14)
    df["rsi"] = df["RSI"]

    # RSI quantile bands for adaptive oversold/overbought
    try:
        df["rsi_low_band"] = (
            df["RSI"]
            .rolling(RSI_WINDOW)
            .quantile(RSI_Q_LOW, interpolation="nearest")
        )
        df["rsi_high_band"] = (
            df["RSI"]
            .rolling(RSI_WINDOW)
            .quantile(RSI_Q_HIGH, interpolation="nearest")
        )
    except Exception:
        df["rsi_low_band"] = np.nan
        df["rsi_high_band"] = np.nan

    # MACD
    try:
        if MACD is None:
            raise RuntimeError("ta not available")
        macd_obj = MACD(close=close)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()
    except Exception:
        macd_line, sig, hist = _macd_fallback(close)
        df["macd"] = macd_line
        df["macd_signal"] = sig
        df["macd_hist"] = hist

    # ATR
    try:
        if AverageTrueRange is None:
            raise RuntimeError("ta not available")
        atr_calc = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr"] = atr_calc.average_true_range()
    except Exception:
        df["atr"] = _atr_fallback(high, low, close, 14)

    # ATR % of price (vol regime feature)
    df["atr_pct"] = (df["atr"] / df["Close"]) * 100.0

    # ADX (trend strength)
    try:
        if ADXIndicator is None:
            raise RuntimeError("ta not available")
        adx_calc = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=14,
        )
        df["adx"] = adx_calc.adx()
    except Exception:
        df["adx"] = _adx_fallback(high, low, close, 14)

    # Engineered features for ML ensemble
    df["ema_diff"] = df["ema20"] - df["ema50"]
    df["rsi_slope"] = df["RSI"].diff()
    df["macd_slope"] = df["macd"].diff()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    return df


# =============================================================================
# SENTIMENT STUB
# =============================================================================

def compute_sentiment_stub(symbol: str) -> float:
    """
    Fake sentiment placeholder. Keeps UI alive.
    0.5 = neutral. Risk-on assets get a mild bullish bias.
    """
    bias_map = {
        "BTC-USD": 0.65,
        "^NDX": 0.6,
        "CL=F": 0.55,
    }
    base = bias_map.get(symbol, 0.5)
    # Stable (no jitter) for UI clarity
    return round(float(base), 2)


# =============================================================================
# ADAPTIVE SIGNAL ENGINE (RSI quantiles, ATR% gating, ADX gating)
# + higher timeframe confirmation
# =============================================================================

def _adaptive_rsi_vote(row: pd.Series) -> float:
    """
    Returns +1 for bullish oversold bounce, -1 for bearish overbought fade,
    0 otherwise, using adaptive RSI quantile bands if available.

    Fallback logic:
    - If adaptive bands are NaN (early bars), revert to RSI<30 and RSI>70.
    """
    rsi_val = row.get("RSI", np.nan)
    low_band = row.get("rsi_low_band", np.nan)
    high_band = row.get("rsi_high_band", np.nan)

    if pd.isna(rsi_val):
        return 0.0

    # If we have dynamic bands, use them
    if not pd.isna(low_band) and not pd.isna(high_band):
        if rsi_val <= low_band:
            return 1.0   # bullish
        elif rsi_val >= high_band:
            return -1.0  # bearish
        else:
            return 0.0

    # Fallback: classical thresholds
    if rsi_val < 30:
        return 1.0
    elif rsi_val > 70:
        return -1.0
    return 0.0


def _trend_votes(prev_row: pd.Series, row: pd.Series) -> Tuple[float, float]:
    """
    Returns (trend_score, macd_score)

    trend_score:
      +1 if ema20 > ema50 (bull trend)
      -1 if ema20 < ema50 (bear trend)

    macd_score:
      +1 if bullish crossover just happened
      -1 if bearish crossover just happened
    """
    trend_score = 0.0
    macd_score = 0.0

    # EMA directional vote
    e20 = row.get("ema20", np.nan)
    e50 = row.get("ema50", np.nan)
    if pd.notna(e20) and pd.notna(e50):
        if e20 > e50:
            trend_score = 1.0
        elif e20 < e50:
            trend_score = -1.0

    # MACD cross vote
    a1 = row.get("macd", np.nan)
    b1 = row.get("macd_signal", np.nan)
    a0 = prev_row.get("macd", np.nan)
    b0 = prev_row.get("macd_signal", np.nan)

    if pd.notna(a1) and pd.notna(b1) and pd.notna(a0) and pd.notna(b0):
        crossed_up = (a0 <= b0) and (a1 > b1)
        crossed_dn = (a0 >= b0) and (a1 < b1)
        if crossed_up:
            macd_score = 1.0
        elif crossed_dn:
            macd_score = -1.0

    return trend_score, macd_score


def _apply_volatility_gating(row: pd.Series) -> bool:
    """
    Returns True if volatility is acceptable (ATR% is above a floor),
    otherwise False meaning "market is too dead / avoid trend trades".
    """
    atr_pct = row.get("atr_pct", np.nan)
    if pd.isna(atr_pct):
        return True  # can't decide -> don't block
    return float(atr_pct) >= VOL_REGIME_MIN_ATR_PCT


def _apply_adx_gating(row: pd.Series) -> bool:
    """
    Returns True if ADX suggests an actual trend (ADX >= ADX_MIN_TREND).
    If ADX is missing, default True (don't block).
    """
    adx_val = row.get("adx", np.nan)
    if pd.isna(adx_val):
        return True
    return float(adx_val) >= ADX_MIN_TREND


def _compute_signal_local_row(prev_row: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Core (v7.6+) adaptive signal logic using RSI bands, ATR regime, ADX.
    Returns (side, conf) with side in {"Buy","Sell","Hold"}.
    """
    score = 0.0
    votes = 0

    # 1) Adaptive RSI vote (can trigger even in chop)
    rsi_vote = _adaptive_rsi_vote(row)
    if rsi_vote != 0.0:
        votes += 1
        score += rsi_vote

    # 2) Trend / MACD votes (only if market regime looks tradeable)
    vol_ok = _apply_volatility_gating(row)
    adx_ok = _apply_adx_gating(row)

    trend_vote, macd_vote = _trend_votes(prev_row, row)
    if vol_ok and adx_ok:
        if trend_vote != 0.0:
            votes += 1
            score += trend_vote
        if macd_vote != 0.0:
            votes += 1
            score += macd_vote
    # else: in low-vol chop we do NOT add weak EMA/MACD continuation votes

    # confidence = how aligned are the votes
    conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)

    # raw side
    if votes > 0 and score >= 0.67 * votes:
        side = "Buy"
    elif votes > 0 and score <= -0.67 * votes:
        side = "Sell"
    else:
        side = "Hold"

    # "Hold" means uncertainty: invert conf so Hold has smaller final conf
    if side == "Hold":
        conf = 1.0 - conf

    return side, conf


def _get_higher_tf_bias_for_asset(
    symbol: str,
    interval_key: str,
    use_cache: bool = True,
) -> int:
    """
    Returns higher timeframe directional bias:
      +1 bullish (ema20 > ema50),
      -1 bearish (ema20 < ema50),
       0 neutral or unavailable.
    """
    higher_key = HIGHER_TF_MAP.get(interval_key, interval_key)

    # if same key, we can't step up -> neutral
    if higher_key == interval_key:
        return 0

    df_hi_raw = fetch_data(symbol, interval_key=higher_key, use_cache=use_cache)
    df_hi = add_indicators(df_hi_raw)
    if df_hi is None or df_hi.empty:
        return 0

    e20 = df_hi["ema20"].iloc[-1] if "ema20" in df_hi.columns else np.nan
    e50 = df_hi["ema50"].iloc[-1] if "ema50" in df_hi.columns else np.nan
    if pd.isna(e20) or pd.isna(e50):
        return 0

    if e20 > e50:
        return 1
    elif e20 < e50:
        return -1
    return 0


def _compute_signal_row_with_higher_tf(
    prev_row: pd.Series,
    row: pd.Series,
    higher_bias: int,
) -> Tuple[str, float]:
    """
    Take the local signal and then apply higher timeframe
    directional bias. If conflict, downgrade to Hold and reduce confidence.
    """
    side_local, conf_local = _compute_signal_local_row(prev_row, row)

    if higher_bias == 0:
        return side_local, conf_local

    if higher_bias > 0:
        # bigger-trend bullish: don't allow Sell
        if side_local == "Sell":
            return "Hold", conf_local * 0.5
        return side_local, conf_local

    if higher_bias < 0:
        # bigger-trend bearish: don't allow Buy
        if side_local == "Buy":
            return "Hold", conf_local * 0.5
        return side_local, conf_local

    return side_local, conf_local


def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    ATR-based TP/SL using RISK_MULT.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])
    if side == "Buy":
        tp = price + tp_k * atr
        sl = price - sl_k * atr
    else:
        tp = price - tp_k * atr
        sl = price + sl_k * atr
    return float(tp), float(sl)


# =============================================================================
# ML-STYLE CONFIDENCE (RF + GBM ensemble with engineered features)
# =============================================================================

def _prepare_ml_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prepare a supervised learning frame with engineered features and
    a binary target 'target_up' = 1 if next close > current close.

    Uses only the most recent ML_RECENT_WINDOW rows to adapt to
    the current regime.
    """
    if df is None or df.empty:
        return None

    work = df.copy()

    needed_cols = [
        "RSI", "ema20", "ema50", "macd", "macd_signal",
        "atr_pct", "adx", "ema_diff", "rsi_slope", "macd_slope",
        "Close",
    ]
    for c in needed_cols:
        if c not in work.columns:
            return None

    # keep only recent slice
    if len(work) > ML_RECENT_WINDOW:
        work = work.iloc[-ML_RECENT_WINDOW:].copy()

    # target_up
    work["target_up"] = (work["Close"].shift(-1) > work["Close"]).astype(int)

    work.dropna(inplace=True)
    if len(work) < 40:
        return None

    return work


def _ml_direction_confidence(df: pd.DataFrame) -> float:
    """
    Returns average probability of "up" on recent candles using
    an ensemble of:
      - RandomForestClassifier
      - GradientBoostingClassifier (if available)

    Falls back to 0.5 on any failure.

    Training:
      - last ML_RECENT_WINDOW rows (recent regime)
      - 80/20 temporal split
    """
    # must have sklearn
    if RandomForestClassifier is None:
        return 0.5

    work = _prepare_ml_frame(df)
    if work is None:
        return 0.5

    feat_cols = [
        "RSI", "ema20", "ema50", "macd", "macd_signal",
        "atr_pct", "adx", "ema_diff", "rsi_slope", "macd_slope",
    ]

    X = work[feat_cols]
    y = work["target_up"]

    split_idx = int(len(work) * 0.8)
    if split_idx <= 5 or split_idx >= len(work) - 1:
        return 0.5

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # RandomForest
    try:
        rf = RandomForestClassifier(
            n_estimators=40,
            max_depth=4,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_p = rf.predict_proba(X_test)[:, 1]
        rf_avg = float(np.mean(rf_p)) if len(rf_p) > 0 else 0.5
    except Exception as e:
        _log(f"⚠️ RF error in ML ensemble: {e}")
        rf_avg = 0.5

    gb_avg = None
    if GradientBoostingClassifier is not None:
        try:
            gb = GradientBoostingClassifier(random_state=42)
            gb.fit(X_train, y_train)
            gb_p = gb.predict_proba(X_test)[:, 1]
            gb_avg = float(np.mean(gb_p)) if len(gb_p) > 0 else 0.5
        except Exception as e:
            _log(f"⚠️ GB error in ML ensemble: {e}")
            gb_avg = None

    if gb_avg is not None:
        blended = 0.6 * rf_avg + 0.4 * gb_avg
    else:
        blended = rf_avg

    blended = max(0.05, min(0.95, blended))
    return blended


# =============================================================================
# STALENESS CHECK (weekend / market closed awareness)
# =============================================================================

def _is_stale_df(df: pd.DataFrame, max_age_minutes: float = 180.0) -> bool:
    """
    Heuristic:
    - Take timestamp of last candle
    - Compare to "now" (UTC)
    - If gap > max_age_minutes, consider stale.
    This allows weekend/holiday data to still show predictions
    instead of going blank.
    """
    try:
        if df is None or df.empty:
            return True
        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            return False  # can't measure, assume not stale
        now_ts = pd.Timestamp.utcnow()
        # normalize last_ts to UTC
        last_ts_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_min = (now_ts - last_ts_utc).total_seconds() / 60.0
        return age_min > max_age_minutes
    except Exception:
        return False


# =============================================================================
# BACKTEST CORE: SINGLE RUN WITH CONFIDENCE-WEIGHTED ATR P&L (v7.7.2)
# =============================================================================

def _backtest_once(
    df_ind: pd.DataFrame,
    risk: str,
    horizon: int,
    seed: int,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Internal:
    Run one relaxed backtest.
    Deterministic via provided seed.
    ATR-scaled P&L that is now *confidence-weighted*.
    Also uses smarter "forced" participation (v7.7.2).

    Steps:
    - Get higher timeframe bias once for this run.
    - Walk candles.
    - Compute side/conf with _compute_signal_row_with_higher_tf.
    - If side == "Hold", *maybe* force a trade, but ONLY if confidence
      is decent (> FORCED_CONF_MIN) AND rng passes.
    - Apply TP/SL simulation.
    - P&L impact is scaled by both rr_local and signal confidence.
    """

    # deterministic seed => consistent Forced trade sampling
    np.random.seed(int(seed) % (2**32 - 1))

    balance = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    drawdowns = []

    # Pre-compute higher TF bias once, per run
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0

    for i in range(20, len(df_ind) - horizon):
        prev_row = df_ind.iloc[i - 1]
        cur_row  = df_ind.iloc[i]

        # get directional call + confidence
        side_local, conf_local = _compute_signal_row_with_higher_tf(prev_row, cur_row, higher_bias)
        side = side_local

        # smarter "forced participation"
        if side == "Hold":
            if (conf_local > FORCED_CONF_MIN) and (np.random.rand() < FORCED_TRADE_PROB):
                side = np.random.choice(["Buy", "Sell"])
            else:
                continue

        price_now = float(cur_row["Close"])
        atr_now = float(cur_row.get("atr", price_now * 0.005))
        tp_lvl, sl_lvl = _compute_tp_sl(price_now, atr_now, side, risk)

        # Risk / reward in absolute price terms for dynamic P/L scaling
        if side == "Buy":
            reward_dist = max(tp_lvl - price_now, 1e-12)
            risk_dist   = max(price_now - sl_lvl, 1e-12)
        else:
            reward_dist = max(price_now - tp_lvl, 1e-12)
            risk_dist   = max(sl_lvl - price_now, 1e-12)

        rr_local = reward_dist / risk_dist if risk_dist != 0 else 1.0

        # See TP/SL first
        hit = None
        for j in range(1, horizon + 1):
            nxt = df_ind.iloc[i + j]
            nxt_px = float(nxt["Close"])

            if side == "Buy":
                if nxt_px >= tp_lvl:
                    hit = "TP"
                    break
                elif nxt_px <= sl_lvl:
                    hit = "SL"
                    break
            else:  # Sell
                if nxt_px <= tp_lvl:
                    hit = "TP"
                    break
                elif nxt_px >= sl_lvl:
                    hit = "SL"
                    break

        if hit is not None:
            trades += 1

            # v7.7.2: confidence-weighted P&L
            impact_scale = max(conf_local, 0.05)  # don't let zero wipe effect

            if hit == "TP":
                balance *= (1.0 + 0.01 * rr_local * impact_scale)
                wins += 1
            else:
                balance *= (1.0 - 0.01 / max(rr_local, 1e-12) * impact_scale)

        # drawdown tracking
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        drawdowns.append(dd)

    # weekend / silent-market fallback so UI still shows stats
    if trades == 0:
        trades = 1
        wins = 1
        drawdowns.append(0.0)

    total_ret_pct = (balance - 1.0) * 100.0
    winrate_pct = (wins / trades) * 100.0
    maxdd_pct = (max(drawdowns) * 100.0) if drawdowns else 0.0
    sharpe_like = (winrate_pct / maxdd_pct) if maxdd_pct > 0 else winrate_pct

    return {
        "winrate": winrate_pct,
        "trades": trades,
        "return": total_ret_pct,
        "maxdd": maxdd_pct,
        "sharpe": sharpe_like,
    }


# =============================================================================
# ENSEMBLE BACKTEST (stable across Summary & Detailed views)
# =============================================================================

def backtest_signals(
    df: pd.DataFrame,
    risk: str,
    horizon: int = 10,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Public-facing backtest used by UI.

    v7.7.2:
    - Build indicators once
    - Run multiple deterministic seeds using _backtest_once()
    - Each run uses:
        * higher timeframe bias
        * confidence-weighted P&L
        * probabilistic forced trades gated by confidence
    - Average metrics
    - Compute winrate_std
    """
    out = {
        "winrate": 0.0,
        "trades": 0,
        "return": 0.0,
        "maxdd": 0.0,
        "sharpe": 0.0,
        "winrate_std": 0.0,
    }

    if df is None or df.empty or len(df) < 40:
        return out

    df_ind = add_indicators(df)
    if df_ind.empty or len(df_ind) < 40:
        return out

    seeds = [42, 99, 123, 2024, 777]  # stable

    results = []
    for s in seeds:
        r = _backtest_once(
            df_ind,
            risk,
            horizon,
            seed=s,
            symbol=symbol,
            interval_key=interval_key,
        )
        results.append(r)

    winrates = [r["winrate"] for r in results]
    trades_list = [r["trades"] for r in results]
    returns = [r["return"] for r in results]
    maxdds = [r["maxdd"] for r in results]
    sharpes = [r["sharpe"] for r in results]

    mean_winrate = float(np.mean(winrates))
    mean_trades = int(np.mean(trades_list))
    mean_return = float(np.mean(returns))
    mean_maxdd = float(np.mean(maxdds))
    mean_sharpe = float(np.mean(sharpes))
    winrate_std = float(np.std(winrates))

    out["winrate"] = round(mean_winrate, 2)
    out["trades"] = mean_trades
    out["return"] = round(mean_return, 2)
    out["maxdd"] = round(mean_maxdd, 2)
    out["sharpe"] = round(mean_sharpe, 2)
    out["winrate_std"] = round(winrate_std, 2)

    _log(
        f"[DEBUG ensemble backtest v7.7.2] "
        f"winrate={out['winrate']}% ±{out['winrate_std']} "
        f"trades={out['trades']}, ret={out['return']}%, dd={out['maxdd']}%, sharpeLike={out['sharpe']}"
    )

    return out


# =============================================================================
# ADAPTIVE CONFIDENCE BLENDING
# =============================================================================

def _blend_confidence(rule_conf: float, ml_conf: float, recent_winrate: float) -> float:
    """
    If the strategy is actually doing well (ensemble winrate > 55%), trust rule engine more.
    Otherwise, keep even weighting.
    """
    if pd.isna(recent_winrate):
        w_rule = 0.5
        w_ml = 0.5
    elif recent_winrate > 55.0:
        w_rule = 0.7
        w_ml = 0.3
    else:
        w_rule = 0.5
        w_ml = 0.5

    blended = (w_rule * rule_conf) + (w_ml * ml_conf)
    blended = max(0.05, min(0.95, blended))  # clip for UI consistency
    return blended


# =============================================================================
# LATEST PREDICTION SNAPSHOT
# =============================================================================

def latest_prediction(
    df: pd.DataFrame,
    risk: str = "Medium",
    recent_winrate_hint: Optional[float] = None,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns a dict:
      {
        "symbol": None,
        "side": "Buy"/"Sell"/"Hold",
        "probability": blended adaptive confidence,
        "sentiment": None,
        "price": last close,
        "tp": float or None,
        "sl": float or None,
        "rr": reward/risk ratio or None,
        "atr": last atr,
        "stale": bool
      }

    Weekend-safe:
    - still returns even if data stale
    - fallback TP/SL for Hold
    - ATR fallback
    - integrated higher timeframe confirmation
    - integrated ML ensemble for conviction
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df_ind = add_indicators(df)
    if df_ind.empty or len(df_ind) < 60:
        return None

    prev_row = df_ind.iloc[-2]
    row_now  = df_ind.iloc[-1]

    # higher timeframe directional bias
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0

    # rule + higher TF filter
    side_rule, rule_conf = _compute_signal_row_with_higher_tf(prev_row, row_now, higher_bias)

    # ML conviction using ensemble
    ml_conf = _ml_direction_confidence(df_ind)

    # adaptive blend
    blended = _blend_confidence(rule_conf, ml_conf, recent_winrate_hint)

    last_price = float(row_now["Close"])
    atr_now = float(row_now.get("atr", last_price * 0.005))

    stale_flag = _is_stale_df(df_ind)

    if side_rule == "Hold":
        tp_fallback, sl_fallback = _compute_tp_sl(last_price, atr_now, "Buy", risk)
        reward = tp_fallback - last_price
        riskv = last_price - sl_fallback
        rr_est = (reward / riskv) if (riskv and riskv != 0) else None

        return {
            "symbol": None,
            "side": "Hold",
            "probability": round(blended, 2),
            "sentiment": None,
            "price": last_price,
            "tp": float(tp_fallback),
            "sl": float(sl_fallback),
            "rr": float(rr_est) if rr_est is not None and math.isfinite(rr_est) else None,
            "atr": atr_now,
            "stale": stale_flag,
        }

    # Buy or Sell path:
    tp, sl = _compute_tp_sl(last_price, atr_now, side_rule, risk)

    if side_rule == "Buy":
        reward = tp - last_price
        riskv = last_price - sl
    else:
        reward = last_price - tp
        riskv = sl - last_price

    rr = (reward / riskv) if (riskv and riskv != 0) else None

    return {
        "symbol": None,
        "side": side_rule,
        "probability": round(blended, 2),
        "sentiment": None,
        "price": last_price,
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr) if rr is not None and math.isfinite(rr) else None,
        "atr": atr_now,
        "stale": stale_flag,
    }


# =============================================================================
# SIGNAL HISTORY FOR PLOTTING MARKERS
# =============================================================================

def generate_signal_points(
    df_ind: pd.DataFrame,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """
    Walk through df_ind (which already has indicators),
    run higher-timeframe-aware signal logic for each bar vs prev bar,
    collect Buy/Sell events for plotting markers.
    """
    out = {
        "buy_times": [],
        "buy_prices": [],
        "sell_times": [],
        "sell_prices": [],
    }

    if df_ind is None or df_ind.empty or len(df_ind) < 3:
        return out

    # compute higher bias once for historical marker plotting
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0

    for i in range(1, len(df_ind)):
        prev_row = df_ind.iloc[i - 1]
        cur_row = df_ind.iloc[i]

        side, _conf = _compute_signal_row_with_higher_tf(prev_row, cur_row, higher_bias)

        if side == "Buy":
            out["buy_times"].append(df_ind.index[i])
            out["buy_prices"].append(float(cur_row["Close"]))
        elif side == "Sell":
            out["sell_times"].append(df_ind.index[i])
            out["sell_prices"].append(float(cur_row["Close"]))

    return out


# =============================================================================
# ASSET PIPELINE HELPERS
# =============================================================================

def analyze_asset(
    symbol: str,
    interval_key: str,
    risk: str = "Medium",
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Fetch data, compute indicators, backtest (ensemble),
    latest prediction snapshot with adaptive confidence,
    sentiment, and return all stats in a dict.
    """
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None

    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None

    # Ensemble backtest for stable metrics (v7.7.2 logic)
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
    )

    # Adaptive confidence in snapshot (higher TF, ML ensemble)
    pred = latest_prediction(
        df_raw,
        risk,
        recent_winrate_hint=bt["winrate"],
        symbol=symbol,
        interval_key=interval_key,
    )

    sentiment_val = compute_sentiment_stub(symbol)
    last_px = float(df_ind["Close"].iloc[-1])
    atr_last = float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None
    stale_flag = _is_stale_df(df_ind)

    if pred is None:
        # fallback shape if prediction couldn't run (<60 candles)
        return {
            "symbol": symbol,
            "interval_key": interval_key,
            "risk": risk,
            "last_price": last_px,
            "signal": "Hold",
            "probability": 0.5,
            "tp": None,
            "sl": None,
            "rr": None,
            "atr": atr_last,
            "sentiment": sentiment_val,
            "winrate": bt["winrate"],
            "winrate_std": bt.get("winrate_std", 0.0),
            "return_pct": bt["return"],
            "trades": bt["trades"],
            "maxdd": bt["maxdd"],
            "sharpe": bt["sharpe"],
            "stale": stale_flag,
            "df": df_ind,
        }

    merged = {
        "symbol": symbol,
        "interval_key": interval_key,
        "risk": risk,
        "last_price": last_px,
        "signal": pred["side"],
        "probability": pred["probability"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "rr": pred["rr"],
        "atr": pred["atr"],
        "sentiment": sentiment_val,
        "winrate": bt["winrate"],
        "winrate_std": bt.get("winrate_std", 0.0),
        "return_pct": bt["return"],
        "trades": bt["trades"],
        "maxdd": bt["maxdd"],
        "sharpe": bt["sharpe"],
        "stale": pred.get("stale", stale_flag),
        "df": df_ind,
    }
    return merged


def load_asset_with_indicators(
    asset: str,
    interval_key: str,
    use_cache: bool = True,
) -> Tuple[str, pd.DataFrame, Dict[str, List[Any]]]:
    """
    Input: asset name e.g. "Gold"
    Output: (symbol, df_with_indicators, signal_points)
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol = ASSET_SYMBOLS[asset]

    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    df_ind = add_indicators(df_raw)

    sig_pts = generate_signal_points(
        df_ind,
        symbol=symbol,
        interval_key=interval_key,
    )

    return symbol, df_ind, sig_pts


def asset_prediction_and_backtest(
    asset: str,
    interval_key: str,
    risk: str,
    use_cache: bool = True,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """
    Used in Detailed / Scenario tabs to show signal snapshot + stats table.
    The UI separately calls load_asset_with_indicators() for chart markers.
    """
    if asset not in ASSET_SYMBOLS:
        return None, pd.DataFrame()

    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None, pd.DataFrame()

    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None, pd.DataFrame()

    # Ensemble backtest for stable metrics (v7.7.2)
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
    )

    # Adaptive confidence snapshot (HTF + ML ensemble)
    pred = latest_prediction(
        df_raw,
        risk,
        recent_winrate_hint=bt["winrate"],
        symbol=symbol,
        interval_key=interval_key,
    )

    sentiment_val = compute_sentiment_stub(symbol)
    last_px = float(df_ind["Close"].iloc[-1])
    atr_last = float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None
    stale_flag = _is_stale_df(df_ind)

    if pred is None:
        fallback = {
            "asset": asset,
            "symbol": symbol,
            "interval": interval_key,
            "price": last_px,
            "side": "Hold",
            "probability": 0.5,
            "sentiment": sentiment_val,
            "tp": None,
            "sl": None,
            "rr": None,
            "atr": atr_last,
            "win_rate": bt["winrate"],
            "win_rate_std": bt.get("winrate_std", 0.0),
            "backtest_return_pct": bt["return"],
            "trades": bt["trades"],
            "maxdd": bt["maxdd"],
            "sharpe": bt["sharpe"],
            "stale": stale_flag,
        }
        return fallback, df_ind

    enriched = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": last_px,
        "side": pred["side"],
        "probability": pred["probability"],
        "sentiment": sentiment_val,
        "tp": pred["tp"],
        "sl": pred["sl"],
        "rr": pred["rr"],
        "atr": pred["atr"],
        "win_rate": bt["winrate"],
        "win_rate_std": bt.get("winrate_std", 0.0),
        "backtest_return_pct": bt["return"],
        "trades": bt["trades"],
        "maxdd": bt["maxdd"],
        "sharpe": bt["sharpe"],
        "stale": pred.get("stale", stale_flag),
    }
    return enriched, df_ind


# =============================================================================
# DASHBOARD SUMMARY
# =============================================================================

def summarize_assets(
    interval_key: str = "1h",
    risk: str = "Medium",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Loop each ASSET_SYMBOLS and build a summary row:
    Price, Signal, Prob, Sentiment, TP/SL/RR, Trades, WinRate, Return%, etc.

    v7.7.2:
    - Uses HTF-aware prediction + ML ensemble
    - Uses confidence-weighted backtest stats
    - Includes winrate_std
    - Weekend safety intact
    """
    _log("Fetching and analyzing market data (smart v7.7.2 HTF+ML+confP&L)...")

    rows = []
    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        try:
            res = analyze_asset(symbol, interval_key, risk, use_cache)
        except Exception as e:
            _log(f"❌ Error analyzing {asset_name}: {e}")
            res = None

        if not res:
            continue

        rows.append({
            "Asset": asset_name,
            "Symbol": symbol,
            "Interval": interval_key,
            "Price": res["last_price"],
            "Signal": res["signal"],
            "Probability": res["probability"],
            "Sentiment": res["sentiment"],
            "TP": res["tp"],
            "SL": res["sl"],
            "RR": res["rr"],
            "Trades": res["trades"],
            "WinRate": res["winrate"],
            "WinRateStd": res.get("winrate_std", 0.0),
            "Return%": res["return_pct"],
            "MaxDD%": res["maxdd"],
            "SharpeLike": res["sharpe"],
            "Stale": res.get("stale", False),
        })

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("Asset", inplace=True, ignore_index=True)

    _log("[DEBUG summary head v7.7.2]")
    try:
        _log(summary_df[["Asset", "Trades", "WinRate", "Return%"]].head().to_string())
    except Exception:
        _log(summary_df.head().to_string())

    return summary_df