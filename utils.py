# utils.py — Woody Trades Pro Smart v7.9.3
# Weekend-Aware Stability Patch • Calibration Cache • Adaptive TP/SL • ML Confidence
# Fully compatible with app v7.9.2+

from __future__ import annotations

import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
except Exception:
    RandomForestClassifier = None
    try:
        GradientBoostingClassifier
    except Exception:
        GradientBoostingClassifier = None  # type: ignore

try:
    from ta.trend import EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    EMAIndicator = MACD = ADXIndicator = RSIIndicator = AverageTrueRange = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CALIBRATION_PATH = DATA_DIR / "calibration_cache.json"
_CALIBRATION_HISTORY_LEN = 5

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

RSI_WINDOW = 100
RSI_Q_LOW = 0.2
RSI_Q_HIGH = 0.8

VOL_REGIME_MIN_ATR_PCT = 0.3
ADX_MIN_TREND = 20.0
CONF_EXEC_THRESHOLD = 0.6

HIGHER_TF_MAP = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1wk",
    "1wk": "1wk",
}

ML_RECENT_WINDOW = 500

FORCED_TRADE_PROB = 0.02
FORCED_CONF_MIN = 0.55

ATR_FLOOR_MULT = 0.0005

STALE_MULTIPLIER_HOURS = 48.0

def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _load_calibration() -> Dict[str, Any]:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        _log(f"⚠️ calibration load error: {e}")
        return {}

def _save_calibration(calib: Dict[str, Any]) -> None:
    try:
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(calib, f)
    except Exception as e:
        _log(f"⚠️ calibration save error: {e}")

def _record_calibration(calib: Dict[str, Any], symbol: str, winrate: float) -> Dict[str, Any]:
    now_str = pd.Timestamp.utcnow().isoformat() + "Z"
    entry = calib.get(symbol, {"winrates": [], "last_ts": now_str})
    hist = entry.get("winrates", [])
    if not isinstance(hist, list):
        hist = []
    hist.append(float(winrate))
    hist = hist[-_CALIBRATION_HISTORY_LEN:]
    entry["winrates"] = hist
    entry["last_ts"] = now_str
    calib[symbol] = entry
    return calib

def _get_symbol_winrate_stats(calib: Dict[str, Any], symbol: str) -> Dict[str, float]:
    entry = calib.get(symbol, {})
    hist = entry.get("winrates", [])
    if not hist:
        return {"avg_wr": 50.0, "bias": 0.0}
    avg_wr = float(np.mean(hist))
    bias = (avg_wr - 50.0) / 100.0
    bias = max(-0.1, min(0.1, bias))
    return {"avg_wr": avg_wr, "bias": bias}

def _compute_calibration_bias_for_symbol(calib: Dict[str, Any], symbol: str) -> Dict[str, float]:
    stats = _get_symbol_winrate_stats(calib, symbol)
    bias = stats["bias"]
    conf_mult = 1.0 - bias
    adx_mult = 1.0 - bias
    conf_mult = max(0.8, min(1.2, conf_mult))
    adx_mult  = max(0.8, min(1.2, adx_mult))
    return {
        "conf_mult": conf_mult,
        "adx_mult": adx_mult,
        "avg_wr": stats["avg_wr"],
    }

def _apply_filter_level_and_calibration(
    filter_level: str,
    calib_bias: Dict[str, float],
) -> None:
    global ADX_MIN_TREND, VOL_REGIME_MIN_ATR_PCT, CONF_EXEC_THRESHOLD
    if filter_level == "Loose":
        base_adx = 15.0
        base_atr = 0.2
        base_conf = 0.5
    elif filter_level == "Strict":
        base_adx = 25.0
        base_atr = 0.4
        base_conf = 0.7
    else:
        base_adx = 20.0
        base_atr = 0.3
        base_conf = 0.6
    conf_mult = calib_bias.get("conf_mult", 1.0)
    adx_mult  = calib_bias.get("adx_mult", 1.0)
    ADX_MIN_TREND = base_adx * adx_mult
    VOL_REGIME_MIN_ATR_PCT = base_atr
    CONF_EXEC_THRESHOLD = base_conf * conf_mult
    ADX_MIN_TREND = max(10.0, min(35.0, ADX_MIN_TREND))
    CONF_EXEC_THRESHOLD = max(0.4, min(0.8, CONF_EXEC_THRESHOLD))
    VOL_REGIME_MIN_ATR_PCT = max(0.1, min(1.0, VOL_REGIME_MIN_ATR_PCT))
    _log(
        f"[CALIBRATION] {filter_level} => "
        f"ADX_MIN_TREND={ADX_MIN_TREND:.2f}, "
        f"VOL_REGIME_MIN_ATR_PCT={VOL_REGIME_MIN_ATR_PCT:.2f}, "
        f"CONF_EXEC_THRESHOLD={CONF_EXEC_THRESHOLD:.2f}"
    )

def _cache_path(symbol: str, interval_key: str) -> Path:
    safe = (
        symbol.replace("^", "")
              .replace("=", "_")
              .replace("/", "_")
              .replace("-", "_")
    )
    return DATA_DIR / f"{safe}_{interval_key}.csv"

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
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
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval,
                         auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        raw2 = tk.history(period=period, interval=interval,
                          auto_adjust=False, prepost=False)
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
    if interval_key not in INTERVALS:
        raise KeyError(
            f"Unknown interval_key '{interval_key}'. Known: {list(INTERVALS.keys())}"
        )
    interval = str(INTERVALS[interval_key]["interval"])
    period   = str(INTERVALS[interval_key]["period"])
    min_rows = int(INTERVALS[interval_key]["min_rows"])
    _log(f"⏳ Fetching {symbol} [{interval}] ...")
    cache_fp = _cache_path(symbol, interval_key)
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

def _interval_hours(interval_key: str) -> float:
    if interval_key == "15m":
        return 0.25
    if interval_key == "1h":
        return 1.0
    if interval_key == "4h":
        return 4.0
    if interval_key == "1d":
        return 24.0
    if interval_key == "1wk":
        return 24.0 * 7.0
    return 1.0

def _compute_stale_flag(df: pd.DataFrame, symbol: str, interval_key: str) -> bool:
    if df is None or df.empty:
        return True
    if symbol == "BTC-USD":
        return False
    try:
        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            return False
        now_utc = pd.Timestamp.utcnow()
        last_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_hours = (now_utc - last_utc).total_seconds() / 3600.0
        limit_hours = STALE_MULTIPLIER_HOURS * _interval_hours(interval_key)
        return age_hours > limit_hours
    except Exception:
        return False

def _is_stale_df(df: pd.DataFrame, max_age_minutes: float = 180.0) -> bool:
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
    return pd.Series(index=close.index, dtype=float)

def _compute_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["ema50_slope"] = np.nan
        df["trend_regime"] = 0
        df["range_regime"] = 0
        return df
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
    both = (df["trend_regime"] == 1) & (df["range_regime"] == 1)
    if both.any():
        df.loc[both, "range_regime"] = 0
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    try:
        if EMAIndicator is None:
            raise RuntimeError("ta not available")
        df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    except Exception:
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
    try:
        if RSIIndicator is None:
            raise RuntimeError("ta not available")
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    except Exception:
        df["RSI"] = _rsi_fallback(close, 14)
    df["rsi"] = df["RSI"]
    try:
        df["rsi_low_band"] = (
            df["RSI"].rolling(RSI_WINDOW).quantile(RSI_Q_LOW, interpolation="nearest")
        )
        df["rsi_high_band"] = (
            df["RSI"].rolling(RSI_WINDOW).quantile(RSI_Q_HIGH, interpolation="nearest")
        )
    except Exception:
        df["rsi_low_band"] = np.nan
        df["rsi_high_band"] = np.nan
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
    try:
        if AverageTrueRange is None:
            raise RuntimeError("ta not available")
        atr_calc = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr"] = atr_calc.average_true_range()
    except Exception:
        df["atr"] = _atr_fallback(high, low, close, 14)
    df["atr_pct"] = (df["atr"] / df["Close"]) * 100.0
    try:
        if ADXIndicator is None:
            raise RuntimeError("ta not available")
        adx_calc = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx_calc.adx()
    except Exception:
        df["adx"] = _adx_fallback(high, low, close, 14)
    df["ema_diff"] = df["ema20"] - df["ema50"]
    df["rsi_slope"] = df["RSI"].diff()
    df["macd_slope"] = df["macd"].diff()
    df = _compute_regime_flags(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)
    return df

def compute_sentiment_stub(symbol: str) -> float:
    bias_map = {
        "BTC-USD": 0.65,
        "^NDX": 0.6,
        "CL=F": 0.55,
    }
    base = bias_map.get(symbol, 0.5)
    return round(float(base), 2)

def _adaptive_rsi_vote(row: pd.Series) -> float:
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
    atr_pct_val = row.get("atr_pct", np.nan)
    if pd.isna(atr_pct_val):
        return True
    return float(atr_pct_val) >= VOL_REGIME_MIN_ATR_PCT

def _apply_adx_gating(row: pd.Series) -> bool:
    adx_val = row.get("adx", np.nan)
    if pd.isna(adx_val):
        return True
    return float(adx_val) >= ADX_MIN_TREND

def _atr_floor_ok(row: pd.Series) -> bool:
    price_now = float(row.get("Close", np.nan))
    atr_now = float(row.get("atr", np.nan))
    if np.isnan(price_now) or np.isnan(atr_now):
        return True
    floor_val = price_now * ATR_FLOOR_MULT
    return atr_now >= floor_val

def _compute_signal_regime_row(prev_row: pd.Series, row: pd.Series) -> Tuple[str, float]:
    if not _atr_floor_ok(row):
        return "Hold", 0.5
    trend_reg = int(row.get("trend_regime", 0))
    range_reg = int(row.get("range_regime", 0))
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
    cont_conf = cont_conf_raw if cont_side != "Hold" else (1.0 - cont_conf_raw)
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
    rev_conf = rev_conf_raw if rev_side != "Hold" else (1.0 - rev_conf_raw)
    if trend_reg and not range_reg:
        final_side = cont_side
        final_conf = cont_conf
        if final_side == "Hold" and rev_side != "Hold":
            final_side = rev_side
            final_conf = rev_conf * 0.6
    elif range_reg and not trend_reg:
        final_side = rev_side
        final_conf = rev_conf
        if final_side == "Hold" and cont_side != "Hold":
            final_side = cont_side
            final_conf = cont_conf * 0.6
    else:
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
    side_local, conf_local = _compute_signal_regime_row(prev_row, row)
    if higher_bias == 0:
        return side_local, conf_local
    if higher_bias > 0:
        if side_local == "Sell":
            return "Hold", conf_local * 0.5
        return side_local, conf_local
    if higher_bias < 0:
        if side_local == "Buy":
            return "Hold", conf_local * 0.5
        return side_local, conf_local
    return side_local, conf_local

def _compute_tp_sl_regime_dynamic(
    price: float,
    atr: float,
    side: str,
    risk: str,
    row: pd.Series,
) -> Tuple[float, float]:
    base = RISK_MULT.get(risk, RISK_MULT["Medium"])
    base_tp = float(base["tp_atr"])
    base_sl = float(base["sl_atr"])
    adx_now = float(row.get("adx", 20.0))
    atrp_now = float(row.get("atr_pct", 0.3))
    ema50_slope = float(row.get("ema50_slope", 0.0))
    trend_strength = (adx_now / 25.0) + (abs(ema50_slope) * 10.0)
    trend_factor = max(0.5, min(2.0, trend_strength))
    if atrp_now <= 0:
        atrp_now = 1e-6
    vol_factor = 0.3 / atrp_now
    vol_factor = max(0.5, min(2.0, vol_factor))
    tp_mult = base_tp * trend_factor
    sl_mult = base_sl * vol_factor
    if side == "Buy":
        tp = price + tp_mult * atr
        sl = price - sl_mult * atr
    else:
        tp = price - tp_mult * atr
        sl = price + sl_mult * atr
    return float(tp), float(sl)

def _prepare_ml_frame(
    df: pd.DataFrame,
    recent_winrate_hint: Optional[float] = None,
) -> Optional[pd.DataFrame]:
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

def _raw_ml_confidence_unscaled(
    df: pd.DataFrame,
    recent_winrate_hint: Optional[float] = None,
) -> float:
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
    blended = max(0.05, min(0.95, blended))
    return blended

def _apply_ml_calibration_to_prob(
    raw_prob: float,
    calib: Dict[str, Any],
    symbol: Optional[str],
) -> float:
    if symbol is None:
        return raw_prob
    stats = _get_symbol_winrate_stats(calib, symbol)
    bias = stats["bias"]
    scale = 1.0 + bias
    prob_adj = 0.5 + (raw_prob - 0.5) * scale
    prob_adj = max(0.05, min(0.95, prob_adj))
    return prob_adj

def _ml_direction_confidence_calibrated(
    df: pd.DataFrame,
    recent_winrate_hint: Optional[float],
    calib: Dict[str, Any],
    symbol: Optional[str],
) -> float:
    raw_ml = _raw_ml_confidence_unscaled(df, recent_winrate_hint)
    cal_ml = _apply_ml_calibration_to_prob(raw_ml, calib, symbol)
    return cal_ml

def _blend_confidence_calibrated(
    rule_conf: float,
    ml_conf: float,
    recent_winrate: float,
    calib: Dict[str, Any],
    symbol: Optional[str],
) -> Tuple[float, float]:
    if pd.isna(recent_winrate):
        w_rule = 0.6
        w_ml = 0.4
    elif recent_winrate > 55.0:
        w_rule = 0.7
        w_ml = 0.3
    elif recent_winrate < 45.0:
        w_rule = 0.7
        w_ml = 0.3
    else:
        w_rule = 0.6
        w_ml = 0.4
    if symbol is not None:
        stats = _get_symbol_winrate_stats(calib, symbol)
        avg_wr = stats["avg_wr"]
        if avg_wr > 55.0:
            w_ml *= 1.1
            w_rule *= 0.9
        elif avg_wr < 45.0:
            w_ml *= 0.8
            w_rule *= 1.2
    total_w = w_rule + w_ml
    if total_w <= 0:
        w_rule, w_ml = 0.5, 0.5
        total_w = 1.0
    w_rule /= total_w
    w_ml   /= total_w
    blended_raw = (w_rule * rule_conf) + (w_ml * ml_conf)
    blended_raw = max(0.05, min(0.95, blended_raw))
    blended_cal = _apply_ml_calibration_to_prob(blended_raw, calib, symbol)
    return blended_raw, blended_cal

def _backtest_once(
    df_ind: pd.DataFrame,
    risk: str,
    horizon: int,
    seed: int,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
    stale_mode: bool = False,
) -> Dict[str, Any]:
    np.random.seed(int(seed) % (2**32 - 1))
    balance = 1.0
    peak = 1.0
    wins = 0
    trades = 0
    drawdowns = []
    trade_returns = []
    win_amounts = []
    loss_amounts = []
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0
    local_forced_prob = 0.0 if stale_mode else FORCED_TRADE_PROB
    for i in range(20, len(df_ind) - horizon):
        prev_row = df_ind.iloc[i - 1]
        cur_row  = df_ind.iloc[i]
        side_local, conf_local = _compute_signal_row_with_higher_tf(prev_row, cur_row, higher_bias)
        side = side_local
        trend_reg = int(cur_row.get("trend_regime", 0))
        range_reg = int(cur_row.get("range_regime", 0))
        bullish_trend = trend_reg and (cur_row.get("ema20", 0) > cur_row.get("ema50", 0))
        bearish_trend = trend_reg and (cur_row.get("ema20", 0) < cur_row.get("ema50", 0))
        regime_allows_relaxed = (
            (side == "Buy"  and bullish_trend) or
            (side == "Sell" and bearish_trend) or
            (range_reg and side in ["Buy", "Sell"])
        )
        if side == "Hold":
            if (conf_local > FORCED_CONF_MIN) and (np.random.rand() < local_forced_prob):
                side = np.random.choice(["Buy", "Sell"])
            else:
                continue
        else:
            if conf_local < CONF_EXEC_THRESHOLD and not regime_allows_relaxed:
                continue
        price_now = float(cur_row["Close"])
        atr_now = float(cur_row.get("atr", price_now * 0.005))
        tp_lvl, sl_lvl = _compute_tp_sl_regime_dynamic(price_now, atr_now, side, risk, cur_row)
        if side == "Buy":
            reward_dist = max(tp_lvl - price_now, 1e-12)
            risk_dist   = max(price_now - sl_lvl, 1e-12)
        else:
            reward_dist = max(price_now - tp_lvl, 1e-12)
            risk_dist   = max(sl_lvl - price_now, 1e-12)
        rr_local = reward_dist / risk_dist if risk_dist != 0 else 1.0
        dyn_horizon = int(horizon * (0.8 + conf_local * 0.6))
        if dyn_horizon < 1:
            dyn_horizon = 1
        if dyn_horizon > horizon * 2:
            dyn_horizon = horizon * 2
        hit = None
        for j in range(1, dyn_horizon + 1):
            if i + j >= len(df_ind):
                break
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
                gain = 0.01 * rr_local * impact_scale
                balance *= (1.0 + gain)
                wins += 1
                trade_returns.append(gain)
                win_amounts.append(gain)
            else:
                loss = 0.01 / max(rr_local, 1e-12) * impact_scale
                balance *= (1.0 - loss)
                trade_returns.append(-loss)
                loss_amounts.append(loss)
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        drawdowns.append(dd)
    if trades == 0:
        trades = 1
        wins = 1
        drawdowns.append(0.0)
        trade_returns.append(0.01)
        win_amounts.append(0.01)
    total_ret_pct = (balance - 1.0) * 100.0
    winrate_pct = (wins / trades) * 100.0
    maxdd_pct = (max(drawdowns) * 100.0) if drawdowns else 0.0
    sharpe_like = (winrate_pct / maxdd_pct) if maxdd_pct > 0 else winrate_pct
    total_gain = sum([x for x in trade_returns if x > 0])
    total_loss = -sum([x for x in trade_returns if x < 0])
    if total_loss <= 1e-12:
        profit_factor = float("inf") if total_gain > 0 else 0.0
    else:
        profit_factor = total_gain / total_loss
    if len(trade_returns) > 0:
        avg_trade_ret = float(np.mean(trade_returns))
        ev_pct = avg_trade_ret * 100.0
    else:
        avg_trade_ret = 0.0
        ev_pct = 0.0
        profit_factor = 0.0
    return {
        "winrate": winrate_pct,
        "trades": trades,
        "return": total_ret_pct,
        "maxdd": maxdd_pct,
        "sharpe": sharpe_like,
        "profit_factor": profit_factor,
        "ev_pct": ev_pct,
    }

def backtest_signals(
    df: pd.DataFrame,
    risk: str,
    horizon: int = 10,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
    filter_level: str = "Balanced",
) -> Dict[str, Any]:
    out = {
        "winrate": 0.0,
        "trades": 0,
        "return": 0.0,
        "maxdd": 0.0,
        "sharpe": 0.0,
        "winrate_std": 0.0,
        "profit_factor": 0.0,
        "ev_pct": 0.0,
        "stale": False,
    }
    if df is None or df.empty or len(df) < 40:
        return out
    calib = _load_calibration()
    calib_bias = _compute_calibration_bias_for_symbol(calib, symbol if symbol else "UNKNOWN")
    _apply_filter_level_and_calibration(filter_level, calib_bias)
    stale_flag = _compute_stale_flag(df, symbol if symbol else "UNKNOWN", interval_key if interval_key else "1h")
    out["stale"] = bool(stale_flag)
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
            stale_mode=stale_flag,
        )
        results.append(r)
    winrates = [r["winrate"] for r in results]
    trades_list = [r["trades"] for r in results]
    returns = [r["return"] for r in results]
    maxdds = [r["maxdd"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    pfs = [r["profit_factor"] for r in results]
    evs = [r["ev_pct"] for r in results]
    mean_winrate = float(np.mean(winrates))
    mean_trades = int(np.mean(trades_list))
    mean_return = float(np.mean(returns))
    mean_maxdd = float(np.mean(maxdds))
    mean_sharpe = float(np.mean(sharpes))
    mean_pf = float(np.mean(pfs))
    mean_ev = float(np.mean(evs))
    winrate_std = float(np.std(winrates))
    out["winrate"] = round(mean_winrate, 2)
    out["trades"] = mean_trades
    out["return"] = round(mean_return, 2)
    out["maxdd"] = round(mean_maxdd, 2)
    out["sharpe"] = round(mean_sharpe, 2)
    out["winrate_std"] = round(winrate_std, 2)
    out["profit_factor"] = round(mean_pf, 2)
    out["ev_pct"] = round(mean_ev, 4)
    _log(
        f"[DEBUG ensemble backtest v7.9.3 {filter_level}] "
        f"WR={out['winrate']}% ±{out['winrate_std']} "
        f"EV/trade={out['ev_pct']}%, PF={out['profit_factor']}, "
        f"trades={out['trades']}, ret={out['return']}%, "
        f"dd={out['maxdd']}%, sharpeLike={out['sharpe']}, "
        f"stale={stale_flag}"
    )
    if symbol and not stale_flag:
        calib = _record_calibration(calib, symbol, out["winrate"])
        _save_calibration(calib)
    return out

def latest_prediction(
    df: pd.DataFrame,
    risk: str = "Medium",
    recent_winrate_hint: Optional[float] = None,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if df is None or df.empty or len(df) < 60:
        return None
    df_ind = add_indicators(df)
    if df_ind.empty or len(df_ind) < 60:
        return None
    prev_row = df_ind.iloc[-2]
    row_now  = df_ind.iloc[-1]
    if symbol is not None and interval_key is not None:
        higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True)
    else:
        higher_bias = 0
    side_rule, rule_conf = _compute_signal_row_with_higher_tf(prev_row, row_now, higher_bias)
    calib = _load_calibration()
    ml_conf_cal = _ml_direction_confidence_calibrated(
        df_ind,
        recent_winrate_hint,
        calib,
        symbol,
    )
    blended_raw, blended_cal = _blend_confidence_calibrated(
        rule_conf,
        ml_conf_cal,
        recent_winrate_hint if recent_winrate_hint is not None else np.nan,
        calib,
        symbol,
    )
    last_price = float(row_now["Close"])
    atr_now = float(row_now.get("atr", last_price * 0.005))
    stale_flag_legacy = _is_stale_df(df_ind)
    stale_flag_market = _compute_stale_flag(df_ind, symbol if symbol else "UNKNOWN", interval_key if interval_key else "1h")
    stale_flag = bool(stale_flag_legacy or stale_flag_market)
    if side_rule == "Hold":
        tp_fallback, sl_fallback = _compute_tp_sl_regime_dynamic(
            last_price, atr_now, "Buy", risk, row_now
        )
        reward = tp_fallback - last_price
        riskv = last_price - sl_fallback
        rr_est = (reward / riskv) if (riskv and riskv != 0) else None
        return {
            "symbol": None,
            "side": "Hold",
            "probability_raw": round(blended_raw, 2),
            "probability_calibrated": round(blended_cal, 2),
            "probability": round(blended_cal, 2),
            "sentiment": None,
            "price": last_price,
            "tp": float(tp_fallback),
            "sl": float(sl_fallback),
            "rr": float(rr_est) if rr_est is not None and math.isfinite(rr_est) else None,
            "atr": atr_now,
            "stale": stale_flag,
        }
    tp, sl = _compute_tp_sl_regime_dynamic(
        last_price, atr_now, side_rule, risk, row_now
    )
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
        "probability_raw": round(blended_raw, 2),
        "probability_calibrated": round(blended_cal, 2),
        "probability": round(blended_cal, 2),
        "sentiment": None,
        "price": last_price,
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr) if rr is not None and math.isfinite(rr) else None,
        "atr": atr_now,
        "stale": stale_flag,
    }

def generate_signal_points(
    df_ind: pd.DataFrame,
    symbol: Optional[str] = None,
    interval_key: Optional[str] = None,
) -> Dict[str, List[Any]]:
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

def analyze_asset(
    symbol: str,
    interval_key: str,
    risk: str = "Medium",
    use_cache: bool = True,
    filter_level: str = "Balanced",
) -> Optional[Dict[str, Any]]:
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
        filter_level=filter_level,
    )
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
    stale_flag = bt.get("stale", False)
    if not stale_flag:
        stale_flag = _compute_stale_flag(df_ind, symbol, interval_key)
    if pred is None:
        return {
            "symbol": symbol,
            "interval_key": interval_key,
            "risk": risk,
            "last_price": last_px,
            "signal": "Hold",
            "probability": 0.5,
            "probability_raw": 0.5,
            "probability_calibrated": 0.5,
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
            "profit_factor": bt.get("profit_factor", 0.0),
            "ev_pct": bt.get("ev_pct", 0.0),
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
        "probability_raw": pred["probability_raw"],
        "probability_calibrated": pred["probability_calibrated"],
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
        "profit_factor": bt.get("profit_factor", 0.0),
        "ev_pct": bt.get("ev_pct", 0.0),
        "stale": stale_flag,
        "df": df_ind,
    }
    return merged

def load_asset_with_indicators(
    asset: str,
    interval_key: str,
    use_cache: bool = True,
    filter_level: str = "Balanced",
) -> Tuple[str, pd.DataFrame, Dict[str, List[Any]]]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol = ASSET_SYMBOLS[asset]
    calib = _load_calibration()
    calib_bias = _compute_calibration_bias_for_symbol(calib, symbol)
    _apply_filter_level_and_calibration(filter_level, calib_bias)
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
    if asset not in ASSET_SYMBOLS:
        return None, pd.DataFrame()
    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None, pd.DataFrame()
    bt = backtest_signals(
        df_raw,
        risk,
        horizon=10,
        symbol=symbol,
        interval_key=interval_key,
        filter_level=filter_level,
    )
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
    stale_flag = bt.get("stale", False)
    if not stale_flag:
        stale_flag = _compute_stale_flag(df_ind, symbol, interval_key)
    if pred is None:
        fallback = {
            "asset": asset,
            "symbol": symbol,
            "interval": interval_key,
            "price": last_px,
            "side": "Hold",
            "probability": 0.5,
            "probability_raw": 0.5,
            "probability_calibrated": 0.5,
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
            "profit_factor": bt.get("profit_factor", 0.0),
            "ev_pct": bt.get("ev_pct", 0.0),
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
        "probability_raw": pred["probability_raw"],
        "probability_calibrated": pred["probability_calibrated"],
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
        "profit_factor": bt.get("profit_factor", 0.0),
        "ev_pct": bt.get("ev_pct", 0.0),
        "stale": stale_flag,
    }
    return enriched, df_ind

def summarize_assets(
    interval_key: str = "1h",
    risk: str = "Medium",
    use_cache: bool = True,
    filter_level: str = "Balanced",
) -> pd.DataFrame:
    _log(
        f"Fetching and analyzing market data "
        f"(smart v7.9.3 • {filter_level} filter • calibrated prob • EV/PF • stale-aware)..."
    )
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
            "Probability": res.get("probability", 0.5),
            "ProbabilityRaw": res.get("probability_raw", 0.5),
            "ProbabilityCal": res.get("probability_calibrated", 0.5),
            "Sentiment": res["sentiment"],
            "TP": res["tp"],
            "SL": res["sl"],
            "RR": res["rr"],
            "Trades": res["trades"],
            "WinRate": res["winrate"],
            "WinRateStd": res.get("winrate_std", 0.0),
            "EV%": res.get("ev_pct", 0.0),
            "ProfitFactor": res.get("profit_factor", 0.0),
            "Return%": res["return_pct"],
            "MaxDD%": res["maxdd"],
            "SharpeLike": res["sharpe"],
            "Stale": res.get("stale", False),
        })
    if not rows:
        return pd.DataFrame()
    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("Asset", inplace=True, ignore_index=True)
    _log("[DEBUG summary head v7.9.3]")
    try:
        _log(summary_df[
            ["Asset", "Trades", "WinRate", "EV%", "ProfitFactor",
             "Return%", "Probability", "Stale"]
        ].head().to_string())
    except Exception:
        _log(summary_df.head().to_string())
    return summary_df
