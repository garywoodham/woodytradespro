# utils.py - Woody Trades Pro (smart v7.8.1 Regime Adaptive + Dynamic Filtering)
# =============================================================================
# This module extends v7.8 with dynamic filtering levels.
#
# ALL PRIOR FUNCTIONALITY FROM v7.8 IS PRESERVED:
#   - robust fetch with caching + retry + mirror fallback
#   - indicator calc (EMA20/50, RSI, RSI quantile bands, MACD, ATR, atr_pct, ADX,
#     engineered features like ema_diff, macd_slope, etc.)
#   - regime classification (trend_regime, range_regime, ema50_slope)
#   - regime-aware signal engine (trend continuation vs mean reversion in chop)
#   - higher timeframe confirmation (multi-timeframe bias)
#   - regime-aware TP/SL (trend lets winners run, range scalps quickly)
#   - ML ensemble (RandomForest + GradientBoosting) with regime features
#   - adaptive confidence blending (rule vs ML, weighted by recent winrate)
#   - ensemble backtest, confidence-weighted P&L, winrate_std
#   - stale/weekend safety
#   - sentiment stub
#   - signal markers for chart Buy/Sell points
#
# NEW IN v7.8.1:
#   - User-controlled "Signal Filtering Level": Loose / Balanced / Strict
#   - This feeds dynamic thresholds into:
#         ADX_MIN_TREND
#         VOL_REGIME_MIN_ATR_PCT
#         CONF_EXEC_THRESHOLD
#     which affect regime detection, trade gating, and confidence requirements.
#
# We DO NOT remove or rename any public functions exposed to the Streamlit app:
#   summarize_assets
#   analyze_asset
#   asset_prediction_and_backtest
#   load_asset_with_indicators
#   ASSET_SYMBOLS
#   INTERVALS
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

# ML models
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
except Exception:
    RandomForestClassifier = None
    try:
        GradientBoostingClassifier  # might not exist in env
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
# CONFIG / CONSTANTS (these may be dynamically adjusted per filter_level)
# =============================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Yahoo Finance interval note: "4h" must be "4h" (not "240m")
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

# --- Adaptive signal config (core logic from v7.8) ---
RSI_WINDOW = 100            # lookback for dynamic RSI bands
RSI_Q_LOW = 0.2             # 20th percentile RSI => adaptive oversold
RSI_Q_HIGH = 0.8            # 80th percentile RSI => adaptive overbought

# These 3 thresholds are dynamically overridden per filter_level
VOL_REGIME_MIN_ATR_PCT = 0.3   # if ATR% < this => "dead / chop" for trend following
ADX_MIN_TREND = 20.0           # ADX threshold suggesting directional strength
CONF_EXEC_THRESHOLD = 0.6      # min confidence to allow trade (unless regime-aligned)

# Higher timeframe mapping for directional bias
HIGHER_TF_MAP = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1wk",
    "1wk": "1wk",
}

# ML config
ML_RECENT_WINDOW = 500  # train ML only on most recent N rows (to adapt to regime)

# Backtest tuning (these we KEEP from v7.8)
FORCED_TRADE_PROB = 0.02     # base chance to "force" a trade when "Hold"
FORCED_CONF_MIN = 0.55       # only force if confidence above this


# =============================================================================
# LOGGING
# =============================================================================

def _log(msg: str) -> None:
    """
    Safe print for container / Streamlit Cloud.
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
    Ensure we have canonical OHLCV columns, 1-D data, datetime index, etc.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
        # Fallback: capitalise to guess standard cols
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # Flatten 2D columns and coerce numeric
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        if col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")
    df = df[~df.index.duplicated(keep="last")]

    return df


def _yahoo_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Primary fetch via yf.download.
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
    Mirror fetch via yf.Ticker(...).history() fallback.
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
    Unified fetch with cache, retry, mirror, then recache.
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

    # 1. Try cache first
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ Cache for {symbol} only {len(cached)} rows (<{min_rows})")
        except Exception as e:
            _log(f"⚠️ Cache read failed for {symbol}: {e}")

    # 2. Live retries
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

    # 3. Mirror fallback
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
# INDICATORS + REGIME FLAGS
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
    # If ADXIndicator isn't available, fill NaN so gating remains permissive.
    return pd.Series(index=close.index, dtype=float)


def _compute_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ema50_slope: slope of ema50 over ~5 bars
      - trend_regime: 1 if trending (ema20/ema50 aligned, ADX high, slope not tiny)
      - range_regime: 1 if choppy/mean-reverting (low ADX and low ATR%)
    NOTE: ADX_MIN_TREND and VOL_REGIME_MIN_ATR_PCT may be overridden dynamically.
    """
    if df.empty:
        df["ema50_slope"] = np.nan
        df["trend_regime"] = 0
        df["range_regime"] = 0
        return df

    # ema50_slope
    if "ema50" in df.columns:
        df["ema50_slope"] = df["ema50"].diff(5)
    else:
        df["ema50_slope"] = np.nan

    ema20 = df.get("ema20", pd.Series(index=df.index, dtype=float))
    ema50 = df.get("ema50", pd.Series(index=df.index, dtype=float))
    adx   = df.get("adx",   pd.Series(index=df.index, dtype=float))
    atrp  = df.get("atr_pct", pd.Series(index=df.index, dtype=float))

    slope_ok   = df["ema50_slope"].abs() > 1e-6
    ema_align  = (ema20 > ema50) | (ema20 < ema50)
    strong_adx = adx >= ADX_MIN_TREND
    df["trend_regime"] = (
        ema_align.astype(int)
        * strong_adx.astype(int)
        * slope_ok.astype(int)
    ).astype(int)

    weak_adx = adx < ADX_MIN_TREND
    low_vol  = atrp < VOL_REGIME_MIN_ATR_PCT
    df["range_regime"] = (
        weak_adx.astype(int)
        * low_vol.astype(int)
    ).astype(int)

    # If both light up, prefer trend_regime
    both = (df["trend_regime"] == 1) & (df["range_regime"] == 1)
    if both.any():
        df.loc[both, "range_regime"] = 0

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators + regime flags.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
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
    low  = df["Low"]

    # EMA20 / EMA50
    try:
        if EMAIndicator is None:
            raise RuntimeError("ta not available")
        df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    except Exception:
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # RSI
    try:
        if RSIIndicator is None:
            raise RuntimeError("ta not available")
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    except Exception:
        df["RSI"] = _rsi_fallback(close, 14)
    df["rsi"] = df["RSI"]

    # RSI quantile bands (adaptive oversold/overbought)
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

    # ATR % of price
    df["atr_pct"] = (df["atr"] / df["Close"]) * 100.0

    # ADX
    try:
        if ADXIndicator is None:
            raise RuntimeError("ta not available")
        adx_calc = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx_calc.adx()
    except Exception:
        df["adx"] = _adx_fallback(high, low, close, 14)

    # Engineered features
    df["ema_diff"] = df["ema20"] - df["ema50"]
    df["rsi_slope"] = df["RSI"].diff()
    df["macd_slope"] = df["macd"].diff()

    # Regime flags (uses the global thresholds that may have just been updated)
    df = _compute_regime_flags(df)

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
    Stable placeholder sentiment.
    """
    bias_map = {
        "BTC-USD": 0.65,
        "^NDX": 0.6,
        "CL=F": 0.55,
    }
    base = bias_map.get(symbol, 0.5)
    return round(float(base), 2)


# =============================================================================
# SIGNAL ENGINE (regime-aware continuation vs mean reversion)
# + higher timeframe confirmation
# =============================================================================

def _adaptive_rsi_vote(row: pd.Series) -> float:
    """
    +1 if oversold bounce, -1 if overbought fade, 0 otherwise.
    Uses dynamic RSI bands if available, else classic 30/70.
    """
    rsi_val = row.get("RSI", np.nan)
    low_band = row.get("rsi_low_band", np.nan)
    high_band = row.get("rsi_high_band", np.nan)

    if pd.isna(rsi_val):
        return 0.0

    if not pd.isna(low_band) and not pd.isna(high_band):
        if rsi_val <= low_band:
            return 1.0
        elif rsi_val >= high_band:
            return -1.0
        else:
            return 0.0

    if rsi_val < 30:
        return 1.0
    elif rsi_val > 70:
        return -1.0
    return 0.0


def _trend_votes(prev_row: pd.Series, row: pd.Series) -> Tuple[float, float]:
    """
    Returns (trend_score, macd_score):
      trend_score: +1 bull if ema20>ema50, -1 bear if ema20<ema50
      macd_score: +1 bullish cross, -1 bearish cross
    """
    trend_score = 0.0
    macd_score = 0.0

    e20 = row.get("ema20", np.nan)
    e50 = row.get("ema50", np.nan)
    if pd.notna(e20) and pd.notna(e50):
        if e20 > e50:
            trend_score = 1.0
        elif e20 < e50:
            trend_score = -1.0

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
    True if ATR% looks tradeable for trend continuation.
    Uses dynamically set VOL_REGIME_MIN_ATR_PCT.
    """
    atr_pct_val = row.get("atr_pct", np.nan)
    if pd.isna(atr_pct_val):
        return True  # can't decide => don't block
    return float(atr_pct_val) >= VOL_REGIME_MIN_ATR_PCT


def _apply_adx_gating(row: pd.Series) -> bool:
    """
    True if ADX suggests a trend. Uses dynamically set ADX_MIN_TREND.
    """
    adx_val = row.get("adx", np.nan)
    if pd.isna(adx_val):
        return True
    return float(adx_val) >= ADX_MIN_TREND


def _compute_signal_regime_row(prev_row: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Regime-aware logic (v7.8 core, unchanged in 7.8.1).
    Picks between:
      - continuation logic in trend regime
      - RSI mean reversion in range regime
    and blends if uncertain.
    """
    trend_reg = int(row.get("trend_regime", 0))
    range_reg = int(row.get("range_regime", 0))

    # --- continuation logic (trend follow)
    cont_score = 0.0
    cont_votes = 0
    vol_ok = _apply_volatility_gating(row)
    adx_ok = _apply_adx_gating(row)
    tr_vote, macd_vote = _trend_votes(prev_row, row)

    if vol_ok and adx_ok:
        if tr_vote != 0.0:
            cont_votes += 1
            cont_score += tr_vote
        if macd_vote != 0.0:
            cont_votes += 1
            cont_score += macd_vote

    if cont_votes > 0:
        cont_conf_raw = min(1.0, abs(cont_score) / cont_votes)
        if cont_score >= 0.67 * cont_votes:
            cont_side = "Buy"
        elif cont_score <= -0.67 * cont_votes:
            cont_side = "Sell"
        else:
            cont_side = "Hold"
    else:
        cont_conf_raw = 0.0
        cont_side = "Hold"

    if cont_side == "Hold":
        cont_conf = 1.0 - cont_conf_raw
    else:
        cont_conf = cont_conf_raw

    # --- mean reversion logic (RSI fade)
    rev_score = 0.0
    rev_votes = 0
    rsi_vote = _adaptive_rsi_vote(row)
    if rsi_vote != 0.0:
        rev_votes += 1
        rev_score += rsi_vote

    if rev_votes > 0:
        rev_conf_raw = min(1.0, abs(rev_score) / rev_votes)
        if rev_score > 0:
            rev_side = "Buy"
        elif rev_score < 0:
            rev_side = "Sell"
        else:
            rev_side = "Hold"
    else:
        rev_conf_raw = 0.0
        rev_side = "Hold"

    if rev_side == "Hold":
        rev_conf = 1.0 - rev_conf_raw
    else:
        rev_conf = rev_conf_raw

    # Decide
    if trend_reg and not range_reg:
        final_side = cont_side
        final_conf = cont_conf
        # allow RSI fade only weakly
        if final_side == "Hold" and rev_side != "Hold":
            final_side = rev_side
            final_conf = rev_conf * 0.6
    elif range_reg and not trend_reg:
        final_side = rev_side
        final_conf = rev_conf
        # allow trend-follow only weakly
        if final_side == "Hold" and cont_side != "Hold":
            final_side = cont_side
            final_conf = cont_conf * 0.6
    else:
        # blend / ambiguous
        if cont_side != "Hold" and rev_side != "Hold":
            if cont_side == rev_side:
                final_side = cont_side
                final_conf = min(1.0, 0.5 * (cont_conf + rev_conf) + 0.25)
            else:
                final_side = "Hold"
                final_conf = 0.5 * abs(cont_conf - rev_conf)
        elif cont_side != "Hold":
            final_side = cont_side
            final_conf = cont_conf
        elif rev_side != "Hold":
            final_side = rev_side
            final_conf = rev_conf
        else:
            final_side = "Hold"
            final_conf = 0.5

    return final_side, final_conf


def _get_higher_tf_bias_for_asset(
    symbol: str,
    interval_key: str,
    use_cache: bool = True,
) -> int:
    """
    Higher timeframe ema20 vs ema50 bias:
      +1 = bullish, -1 = bearish, 0 = neutral.
    """
    higher_key = HIGHER_TF_MAP.get(interval_key, interval_key)
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
    Apply regime-aware local logic, then suppress countertrend vs higher TF.
    """
    side_local, conf_local = _compute_signal_regime_row(prev_row, row)

    if higher_bias == 0:
        return side_local, conf_local

    if higher_bias > 0:
        # bullish bias: suppress Sell
        if side_local == "Sell":
            return "Hold", conf_local * 0.5
        return side_local, conf_local

    if higher_bias < 0:
        # bearish bias: suppress Buy
        if side_local == "Buy":
            return "Hold", conf_local * 0.5
        return side_local, conf_local

    return side_local, conf_local


# =============================================================================
# TP/SL LOGIC
# =============================================================================

def _compute_tp_sl_base(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Backward compatible baseline SL/TP (kept for Hold fallback).
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


def _compute_tp_sl_regime(
    price: float,
    atr: float,
    side: str,
    risk: str,
    row: pd.Series,
) -> Tuple[float, float]:
    """
    Regime-aware asymmetric TP/SL from v7.8:
      - Trend regime: let winners run more, cut losers faster.
      - Range regime: scalp, take profit faster, tight stop.
    """
    trend_reg = int(row.get("trend_regime", 0))
    range_reg = int(row.get("range_regime", 0))

    base = RISK_MULT.get(risk, RISK_MULT["Medium"])
    base_tp = float(base["tp_atr"])
    base_sl = float(base["sl_atr"])

    tp_mult = base_tp
    sl_mult = base_sl

    if trend_reg and not range_reg:
        tp_mult = base_tp * 1.2
        sl_mult = base_sl * 0.9
    elif range_reg and not trend_reg:
        tp_mult = base_tp * 0.8
        sl_mult = base_sl * 0.8

    if side == "Buy":
        tp = price + tp_mult * atr
        sl = price - sl_mult * atr
    else:
        tp = price - tp_mult * atr
        sl = price + sl_mult * atr

    return float(tp), float(sl)


# =============================================================================
# ML ENSEMBLE CONFIDENCE (with regime features)
# =============================================================================

def _prepare_ml_frame(
    df: pd.DataFrame,
    recent_winrate_hint: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """
    Create supervised frame for ML with regime + performance hint.
    """
    if df is None or df.empty:
        return None

    work = df.copy()

    hint_val = float(recent_winrate_hint) if recent_winrate_hint is not None else 50.0
    work["recent_winrate_hint"] = hint_val

    needed_cols = [
        "RSI", "ema20", "ema50", "ema_diff", "ema50_slope",
        "macd", "macd_signal", "macd_slope",
        "atr_pct", "adx",
        "trend_regime", "range_regime",
        "rsi_slope",
        "recent_winrate_hint",
        "Close",
    ]
    for c in needed_cols:
        if c not in work.columns:
            return None

    if len(work) > ML_RECENT_WINDOW:
        work = work.iloc[-ML_RECENT_WINDOW:].copy()

    work["target_up"] = (work["Close"].shift(-1) > work["Close"]).astype(int)
    work.dropna(inplace=True)
    if len(work) < 40:
        return None

    return work


def _ml_direction_confidence(
    df: pd.DataFrame,
    recent_winrate_hint: Optional[float] = None,
) -> float:
    """
    Probability of next bar closing up.
    RF + (optionally) GB, on regime-aware features.
    """
    if RandomForestClassifier is None:
        return 0.5

    work = _prepare_ml_frame(df, recent_winrate_hint)
    if work is None:
        return 0.5

    feat_cols = [
        "RSI", "ema20", "ema50", "ema_diff", "ema50_slope",
        "macd", "macd_signal", "macd_slope",
        "atr_pct", "adx",
        "trend_regime", "range_regime",
        "rsi_slope",
        "recent_winrate_hint",
    ]

    X = work[feat_cols]
    y = work["target_up"]

    split_idx = int(len(work) * 0.8)
    if split_idx <= 5 or split_idx >= len(work) - 1:
        return 0.5

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # RF
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

    blended = 0.6 * rf_avg + 0.4 * gb_avg if gb_avg is not None else rf_avg
    return max(0.05, min(0.95, blended))


# =============================================================================
# STALENESS CHECK
# =============================================================================

def _is_stale_df(df: pd.DataFrame, max_age_minutes: float = 180.0) -> bool:
    """
    If latest candle is older than ~3h UTC, mark stale (weekend/etc).
    """
    try:
        if df is None or df.empty:
            return True
        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            return False
        now_ts = pd.Timestamp.utcnow()
        last_ts_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_min = (now_ts - last_ts_utc).total_seconds() / 60.0
        return age_min > max_age_minutes
    except Exception:
        return False


# =============================================================================
# BACKTEST CORE (single run)
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
    Single deterministic run of relaxed backtest.

    Uses:
      - regime-aware + HTF-screened signals
      - CONF_EXEC_THRESHOLD (dynamic per filter_level)
      - FORCED_TRADE_PROB and FORCED_CONF_MIN for fallback
      - Regime-aware TP/SL
      - confidence-weighted ATR P&L
    """
    np.random.seed(int(seed) % (2**32 - 1))

    balance = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    drawdowns = []

    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0

    for i in range(20, len(df_ind) - horizon):
        prev_row = df_ind.iloc[i - 1]
        cur_row  = df_ind.iloc[i]

        side_local, conf_local = _compute_signal_row_with_higher_tf(prev_row, cur_row, higher_bias)
        side = side_local

        # regime check for possible relaxed allowance:
        trend_reg = int(cur_row.get("trend_regime", 0))
        range_reg = int(cur_row.get("range_regime", 0))
        bullish_trend = trend_reg and (cur_row.get("ema20", 0) > cur_row.get("ema50", 0))
        bearish_trend = trend_reg and (cur_row.get("ema20", 0) < cur_row.get("ema50", 0))

        regime_allows_relaxed = (
            (side == "Buy"  and bullish_trend) or
            (side == "Sell" and bearish_trend) or
            (range_reg and side in ["Buy", "Sell"])
        )

        # gating:
        if side == "Hold":
            if (conf_local > FORCED_CONF_MIN) and (np.random.rand() < FORCED_TRADE_PROB):
                side = np.random.choice(["Buy", "Sell"])
            else:
                continue
        else:
            if conf_local < CONF_EXEC_THRESHOLD and not regime_allows_relaxed:
                continue

        price_now = float(cur_row["Close"])
        atr_now = float(cur_row.get("atr", price_now * 0.005))

        # regime-aware TP/SL
        tp_lvl, sl_lvl = _compute_tp_sl_regime(price_now, atr_now, side, risk, cur_row)

        # derive RR for P&L scaling
        if side == "Buy":
            reward_dist = max(tp_lvl - price_now, 1e-12)
            risk_dist   = max(price_now - sl_lvl, 1e-12)
        else:
            reward_dist = max(price_now - tp_lvl, 1e-12)
            risk_dist   = max(sl_lvl - price_now, 1e-12)
        rr_local = reward_dist / risk_dist if risk_dist != 0 else 1.0

        # forward-sim horizon
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
            else:
                if nxt_px <= tp_lvl:
                    hit = "TP"
                    break
                elif nxt_px >= sl_lvl:
                    hit = "SL"
                    break

        if hit is not None:
            trades += 1
            impact_scale = max(conf_local, 0.05)
            if hit == "TP":
                balance *= (1.0 + 0.01 * rr_local * impact_scale)
                wins += 1
            else:
                balance *= (1.0 - 0.01 / max(rr_local, 1e-12) * impact_scale)

        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        drawdowns.append(dd)

    if trades == 0:
        # weekend safety so UI still has stats
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
# ENSEMBLE BACKTEST (public)
# =============================================================================

def backtest_signals(
    df: pd.DataFrame,
    risk: str,
    horizon: int = 10,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
    filter_level: str = "Balanced",
) -> Dict[str, Any]:
    """
    Public-facing backtest used by Streamlit.

    v7.8.1:
    - Dynamically set thresholds based on filter_level BEFORE computing indicators
      and running simulations.
        Loose:
          ADX_MIN_TREND = 15.0
          VOL_REGIME_MIN_ATR_PCT = 0.2
          CONF_EXEC_THRESHOLD = 0.5
        Balanced:
          ADX_MIN_TREND = 20.0
          VOL_REGIME_MIN_ATR_PCT = 0.3
          CONF_EXEC_THRESHOLD = 0.6
        Strict:
          ADX_MIN_TREND = 25.0
          VOL_REGIME_MIN_ATR_PCT = 0.4
          CONF_EXEC_THRESHOLD = 0.7

    - Runs deterministic seeds for ensemble stats
    - Returns average WinRate, Trades, Return%, MaxDD%, SharpeLike, WinRateStd
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

    # --- Dynamic behavior tuning ---
    global ADX_MIN_TREND, VOL_REGIME_MIN_ATR_PCT, CONF_EXEC_THRESHOLD

    if filter_level == "Loose":
        ADX_MIN_TREND = 15.0
        VOL_REGIME_MIN_ATR_PCT = 0.2
        CONF_EXEC_THRESHOLD = 0.5
    elif filter_level == "Strict":
        ADX_MIN_TREND = 25.0
        VOL_REGIME_MIN_ATR_PCT = 0.4
        CONF_EXEC_THRESHOLD = 0.7
    else:
        # Balanced (default)
        ADX_MIN_TREND = 20.0
        VOL_REGIME_MIN_ATR_PCT = 0.3
        CONF_EXEC_THRESHOLD = 0.6

    # After thresholds are set, compute indicators under those thresholds.
    df_ind = add_indicators(df)
    if df_ind.empty or len(df_ind) < 40:
        return out

    seeds = [42, 99, 123, 2024, 777]
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
        f"[DEBUG ensemble backtest v7.8.1 {filter_level}] "
        f"winrate={out['winrate']}% ±{out['winrate_std']} "
        f"trades={out['trades']}, ret={out['return']}%, "
        f"dd={out['maxdd']}%, sharpeLike={out['sharpe']}"
    )

    return out


# =============================================================================
# ADAPTIVE CONFIDENCE BLENDING
# =============================================================================

def _blend_confidence(rule_conf: float, ml_conf: float, recent_winrate: float) -> float:
    """
    Weighted blend of rule_conf vs ml_conf.
    If recent winrate is strong (>55%), bias more toward rules.
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
    blended = max(0.05, min(0.95, blended))
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
    Builds the live "signal block" for Detailed View.
    Weekend-safe and regime-aware TP/SL.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df_ind = add_indicators(df)
    if df_ind.empty or len(df_ind) < 60:
        return None

    prev_row = df_ind.iloc[-2]
    row_now  = df_ind.iloc[-1]

    # higher timeframe bias
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0

    # regime-aware + higher timeframe filter
    side_rule, rule_conf = _compute_signal_row_with_higher_tf(prev_row, row_now, higher_bias)

    # ML conviction, regime-aware
    ml_conf = _ml_direction_confidence(df_ind, recent_winrate_hint)

    # adaptive blend
    blended = _blend_confidence(rule_conf, ml_conf, recent_winrate_hint if recent_winrate_hint is not None else np.nan)

    last_price = float(row_now["Close"])
    atr_now = float(row_now.get("atr", last_price * 0.005))
    stale_flag = _is_stale_df(df_ind)

    if side_rule == "Hold":
        tp_fallback, sl_fallback = _compute_tp_sl_regime(last_price, atr_now, "Buy", risk, row_now)
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

    tp, sl = _compute_tp_sl_regime(last_price, atr_now, side_rule, risk, row_now)

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
# SIGNAL HISTORY FOR CHART MARKERS
# =============================================================================

def generate_signal_points(
    df_ind: pd.DataFrame,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """
    Historical Buy/Sell timestamps & prices for plotting markers.
    Uses regime-aware + HTF-filtered signal.
    """
    out = {
        "buy_times": [],
        "buy_prices": [],
        "sell_times": [],
        "sell_prices": [],
    }

    if df_ind is None or df_ind.empty or len(df_ind) < 3:
        return out

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
    filter_level: str = "Balanced",
) -> Optional[Dict[str, Any]]:
    """
    Fetch data, compute indicators (after dynamic thresholds),
    run ensemble backtest, compute latest_prediction snapshot,
    package all fields for dashboard.
    """
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None

    # backtest_signals will set thresholds based on filter_level,
    # then call add_indicators(df_raw) with those thresholds active.
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
        filter_level=filter_level,
    )

    # We need df_ind consistent with the same thresholds we just used.
    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None

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
        # fallback if <60 candles etc.
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
    filter_level: str = "Balanced",
) -> Tuple[str, pd.DataFrame, Dict[str, List[Any]]]:
    """
    For the chart tab:
    returns (symbol, df_ind, signal_points)
    Note: df_ind uses current dynamic thresholds.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol = ASSET_SYMBOLS[asset]

    # Ensure thresholds are aligned with user's filter_level before indicators:
    global ADX_MIN_TREND, VOL_REGIME_MIN_ATR_PCT, CONF_EXEC_THRESHOLD
    if filter_level == "Loose":
        ADX_MIN_TREND = 15.0
        VOL_REGIME_MIN_ATR_PCT = 0.2
        CONF_EXEC_THRESHOLD = 0.5
    elif filter_level == "Strict":
        ADX_MIN_TREND = 25.0
        VOL_REGIME_MIN_ATR_PCT = 0.4
        CONF_EXEC_THRESHOLD = 0.7
    else:
        ADX_MIN_TREND = 20.0
        VOL_REGIME_MIN_ATR_PCT = 0.3
        CONF_EXEC_THRESHOLD = 0.6

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
    filter_level: str = "Balanced",
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """
    Used in Detailed / Scenario tabs to show per-asset stats.
    Matches the same dynamic filter_level.
    """
    if asset not in ASSET_SYMBOLS:
        return None, pd.DataFrame()

    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None, pd.DataFrame()

    # Run backtest with thresholds set for filter_level
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
        filter_level=filter_level,
    )

    # Indicators under the same thresholds:
    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None, pd.DataFrame()

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
    filter_level: str = "Balanced",
) -> pd.DataFrame:
    """
    Build summary table for Market Summary tab.
    Now also reflects the active filter_level.
    """
    _log(f"Fetching and analyzing market data "
         f"(smart v7.8.1 Regime Adaptive • {filter_level} filter)...")

    rows = []
    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        try:
            res = analyze_asset(
                symbol,
                interval_key,
                risk,
                use_cache,
                filter_level=filter_level,
            )
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

    _log("[DEBUG summary head v7.8.1]")
    try:
        _log(summary_df[["Asset", "Trades", "WinRate", "Return%"]].head().to_string())
    except Exception:
        _log(summary_df.head().to_string())

    return summary_df