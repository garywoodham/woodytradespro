# Woody Trades Pro – Smart v8.2
# ---------------------------------------------------------------------------------
# v8.2 builds on v8.1 (structure confluence, adaptive TP/SL, calibration, weekend skip,
# PF / EV%, support/resistance levels, dip/breakout markers, ensemble backtest).
#
# NEW IN v8.2:
#   - Cross-validated ML confidence (5-fold KFold CV)
#     * replaces single 80/20 split
#     * uses full history for out-of-fold prediction quality
#     * more stable probability for live confidence + backtest weighting
#
# Nothing from v8.1 is removed.
#
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import KFold
except Exception:
    RandomForestClassifier = None
    try:
        GradientBoostingClassifier
    except Exception:
        GradientBoostingClassifier = None  # type: ignore
    KFold = None  # type: ignore

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

# ---------------- core tunables ----------------
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
CV_FOLDS = 5  # 5-fold cross validation

FORCED_TRADE_PROB = 0.02
FORCED_CONF_MIN = 0.55

ATR_FLOOR_MULT = 0.0005
STALE_MULTIPLIER_HOURS = 48.0

_TP_SL_PROFILES = {
    "Off":        {"trend_min": 1.0, "trend_max": 1.0, "vol_min": 1.0, "vol_max": 1.0},
    "Normal":     {"trend_min": 0.8, "trend_max": 1.4, "vol_min": 0.8, "vol_max": 1.4},
    "Aggressive": {"trend_min": 0.5, "trend_max": 2.0, "vol_min": 0.5, "vol_max": 2.0},
}

# structure params
STRUCT_WINDOW = 20
SUP_PROX_PCT = 0.003
RES_PROX_PCT = 0.003
RSI_DIP_THRESHOLD = 40.0

# =============================================================================
# LOG / CALIBRATION
# =============================================================================

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
    except Exception:
        return {}

def _save_calibration(calib: Dict[str, Any]) -> None:
    try:
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(calib, f)
    except Exception:
        pass

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

    ADX_MIN_TREND          = max(10.0, min(35.0, base_adx * adx_mult))
    VOL_REGIME_MIN_ATR_PCT = max(0.1, min(1.0, base_atr))
    CONF_EXEC_THRESHOLD    = max(0.4,  min(0.8,  base_conf * conf_mult))

def _compute_calibration_bias_for_symbol(
    calib: Dict[str, Any],
    symbol: str,
    calibration_enabled: bool,
) -> Dict[str, float]:
    if not calibration_enabled:
        return {"conf_mult": 1.0, "adx_mult": 1.0, "avg_wr": 50.0}
    stats = _get_symbol_winrate_stats(calib, symbol)
    bias = stats["bias"]
    conf_mult = 1.0 - bias
    adx_mult  = 1.0 - bias
    conf_mult = max(0.8, min(1.2, conf_mult))
    adx_mult  = max(0.8, min(1.2, adx_mult))
    return {"conf_mult": conf_mult, "adx_mult": adx_mult, "avg_wr": stats["avg_wr"]}

# =============================================================================
# FETCH + NORMALIZE
# =============================================================================

def _cache_path(symbol: str, interval_key: str) -> Path:
    safe = symbol.replace("^","").replace("=","_").replace("/","_").replace("-","_")
    return DATA_DIR / f"{safe}_{interval_key}.csv"

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if not keep:
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals,"ndim",1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        if col in ["Open","High","Low","Close","Adj Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")
    df = df[~df.index.duplicated(keep="last")]
    return df

def _yahoo_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        return _normalize_ohlcv(raw)
    except Exception:
        return pd.DataFrame()

def _yahoo_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        raw2 = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        return _normalize_ohlcv(raw2)
    except Exception:
        return pd.DataFrame()

def fetch_data(symbol: str, interval_key: str="1h", use_cache: bool=True,
               max_retries: int=4, backoff_range: Tuple[float,float]=(2.5,6.0)) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(f"Unknown interval_key '{interval_key}'. Known: {list(INTERVALS.keys())}")
    interval = str(INTERVALS[interval_key]["interval"])
    period   = str(INTERVALS[interval_key]["period"])
    min_rows = int(INTERVALS[interval_key]["min_rows"])

    cache_fp = _cache_path(symbol, interval_key)
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                return cached
        except Exception:
            pass

    for attempt in range(1, max_retries+1):
        df_live = _yahoo_download(symbol, interval, period)
        if not df_live.empty and len(df_live) >= min_rows:
            try: df_live.to_csv(cache_fp)
            except Exception: pass
            return df_live
        time.sleep(np.random.uniform(*backoff_range))

    df_mirror = _yahoo_history(symbol, interval, period)
    if not df_mirror.empty and len(df_mirror) >= min_rows:
        try: df_mirror.to_csv(cache_fp)
        except Exception: pass
        return df_mirror

    return pd.DataFrame()

# =============================================================================
# STALE / WEEKEND HANDLING
# =============================================================================

def _interval_hours(interval_key: str) -> float:
    return {"15m":0.25,"1h":1.0,"4h":4.0,"1d":24.0,"1wk":168.0}.get(interval_key,1.0)

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

def _hard_skip_if_stale(df: pd.DataFrame, symbol: str, interval_key: str, weekend_mode: bool) -> Tuple[pd.DataFrame,bool]:
    stale_flag = _compute_stale_flag(df, symbol, interval_key)
    if not stale_flag or not weekend_mode:
        return df, stale_flag
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=48)
    df_use = df[df.index <= cutoff].copy()
    if df_use.empty:
        return df, stale_flag
    return df_use, stale_flag

def _is_stale_df_simple(df: pd.DataFrame, max_age_minutes: float=180.0) -> bool:
    try:
        if df is None or df.empty:
            return True
        last_ts = df.index[-1]
        if not isinstance(last_ts, pd.Timestamp):
            return False
        now_ts = pd.Timestamp.utcnow()
        last_ts_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_min = (now_ts - last_ts_utc).total_seconds()/60.0
        return age_min > max_age_minutes
    except Exception:
        return False

# =============================================================================
# INDICATORS, REGIME, STRUCTURE
# =============================================================================

def _rsi_fallback(close: pd.Series, window: int=14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(window).mean()
    loss = -d.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0,np.nan)
    return 100 - (100/(1+rs))

def _macd_fallback(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12-ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def _atr_fallback(high: pd.Series, low: pd.Series, close: pd.Series, window: int=14) -> pd.Series:
    tr1 = (high-low).abs()
    tr2 = (high-close.shift(1)).abs()
    tr3 = (low-close.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _adx_fallback(high: pd.Series, low: pd.Series, close: pd.Series, window: int=14) -> pd.Series:
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
    df["trend_regime"] = (ema_align.astype(int) * strong_adx.astype(int) * slope_ok.astype(int)).astype(int)

    weak_adx = adx < ADX_MIN_TREND
    low_vol  = atrp < VOL_REGIME_MIN_ATR_PCT
    df["range_regime"] = (weak_adx.astype(int) * low_vol.astype(int)).astype(int)

    both = (df["trend_regime"]==1) & (df["range_regime"]==1)
    if both.any():
        df.loc[both,"range_regime"] = 0

    return df

def _rolling_support_resistance(df: pd.DataFrame, window: int=STRUCT_WINDOW) -> Tuple[pd.Series,pd.Series]:
    sup = df["Low"].rolling(window, min_periods=3).min()
    res = df["High"].rolling(window, min_periods=3).max()
    return sup, res

def _detect_dip_buy_row(row: pd.Series) -> bool:
    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)
    rsi_val = row.get("RSI", np.nan)
    low_band = row.get("rsi_low_band", np.nan)
    px  = row.get("Close", np.nan)
    sup = row.get("support_level", np.nan)
    if any(pd.isna(x) for x in [ema20,ema50,rsi_val,px,sup]): return False
    if ema20 <= ema50: return False
    oversold = False
    if not pd.isna(low_band):
        oversold = rsi_val <= low_band
    else:
        oversold = rsi_val <= RSI_DIP_THRESHOLD
    if not oversold: return False
    if sup <= 0: return False
    dist = abs(px-sup)/sup
    if dist > SUP_PROX_PCT: return False
    return True

def _detect_breakout_row(row: pd.Series) -> Tuple[bool,bool]:
    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)
    px  = row.get("Close", np.nan)
    res = row.get("resistance_level", np.nan)
    sup = row.get("support_level", np.nan)
    macd_slope = row.get("macd_slope", np.nan)
    if any(pd.isna(x) for x in [ema20,ema50,px,res,sup,macd_slope]):
        return (False,False)

    bull = False
    bear = False
    if (px > res) and (ema20 > ema50) and (macd_slope > 0):
        bull = True
    if (px < sup) and (ema20 < ema50) and (macd_slope < 0):
        bear = True
    return (bull,bear)

def _add_structure_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["support_level"] = np.nan
        df["resistance_level"] = np.nan
        df["dip_buy_flag"] = 0
        df["bull_breakout_flag"] = 0
        df["bear_breakdown_flag"] = 0
        return df
    sup,res = _rolling_support_resistance(df, STRUCT_WINDOW)
    df["support_level"] = sup
    df["resistance_level"] = res
    dips = []
    bulls = []
    bears = []
    for _,row in df.iterrows():
        dips.append(1 if _detect_dip_buy_row(row) else 0)
        b_up,b_dn = _detect_breakout_row(row)
        bulls.append(1 if b_up else 0)
        bears.append(1 if b_dn else 0)
    df["dip_buy_flag"] = dips
    df["bull_breakout_flag"] = bulls
    df["bear_breakdown_flag"] = bears
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    for col in ["Close","High","Low"]:
        if col not in df.columns:
            return pd.DataFrame()
        vals = df[col].values
        if isinstance(vals,np.ndarray) and getattr(vals,"ndim",1)>1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]; high=df["High"]; low=df["Low"]

    try:
        if EMAIndicator is None: raise RuntimeError
        df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    except Exception:
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()

    try:
        if RSIIndicator is None: raise RuntimeError
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    except Exception:
        df["RSI"] = _rsi_fallback(close,14)
    df["rsi"] = df["RSI"]

    try:
        df["rsi_low_band"]  = df["RSI"].rolling(RSI_WINDOW).quantile(RSI_Q_LOW, interpolation="nearest")
        df["rsi_high_band"] = df["RSI"].rolling(RSI_WINDOW).quantile(RSI_Q_HIGH,interpolation="nearest")
    except Exception:
        df["rsi_low_band"]  = np.nan
        df["rsi_high_band"] = np.nan

    try:
        if MACD is None: raise RuntimeError
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
        if AverageTrueRange is None: raise RuntimeError
        atr_calc = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr"] = atr_calc.average_true_range()
    except Exception:
        df["atr"] = _atr_fallback(high, low, close,14)
    df["atr_pct"] = (df["atr"]/df["Close"])*100.0

    try:
        if ADXIndicator is None: raise RuntimeError
        adx_calc = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx_calc.adx()
    except Exception:
        df["adx"] = _adx_fallback(high, low, close,14)

    df["ema_diff"] = df["ema20"]-df["ema50"]
    df["rsi_slope"] = df["RSI"].diff()
    df["macd_slope"] = df["macd"].diff()

    df = _compute_regime_flags(df)

    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(how="all", inplace=True)

    df = _add_structure_signals(df)
    return df

# =============================================================================
# STRUCTURE FILTER
# =============================================================================

def _structure_filter(base_side: str, row: pd.Series, mode: str) -> Tuple[str,Dict[str,bool]]:
    dip_ok        = bool(row.get("dip_buy_flag",0))
    bull_break_ok = bool(row.get("bull_breakout_flag",0))
    bear_break_ok = bool(row.get("bear_breakdown_flag",0))

    tags = {"dip":False,"bull_breakout":False,"bear_breakdown":False}

    if mode == "Off":
        if base_side=="Buy" and dip_ok: tags["dip"]=True
        if base_side=="Buy" and bull_break_ok: tags["bull_breakout"]=True
        if base_side=="Sell" and bear_break_ok: tags["bear_breakdown"]=True
        return base_side, tags

    if base_side == "Hold":
        return "Hold", tags

    if base_side == "Buy":
        if mode == "Buy Dips":
            allow = dip_ok; tags["dip"]=dip_ok
        elif mode == "Breakouts":
            allow = bull_break_ok; tags["bull_breakout"]=bull_break_ok
        elif mode == "Both":
            allow = dip_ok or bull_break_ok
            tags["dip"]=dip_ok; tags["bull_breakout"]=bull_break_ok
        else:
            allow = True
        return ("Buy" if allow else "Hold"), tags

    if base_side == "Sell":
        if mode == "Buy Dips":
            allow = False
        elif mode == "Breakouts":
            allow = bear_break_ok; tags["bear_breakdown"]=bear_break_ok
        elif mode == "Both":
            allow = bear_break_ok; tags["bear_breakdown"]=bear_break_ok
        else:
            allow = True
        return ("Sell" if allow else "Hold"), tags

    return "Hold", tags

# =============================================================================
# RULE ENGINE + HTF BIAS
# =============================================================================

def _adaptive_rsi_vote(row: pd.Series) -> float:
    rsi_val = row.get("RSI", np.nan)
    low_band = row.get("rsi_low_band", np.nan)
    high_band = row.get("rsi_high_band", np.nan)
    if pd.isna(rsi_val): return 0.0
    if not pd.isna(low_band) and not pd.isna(high_band):
        if rsi_val <= low_band: return 1.0
        elif rsi_val >= high_band: return -1.0
        else: return 0.0
    if rsi_val < 30: return 1.0
    elif rsi_val > 70: return -1.0
    return 0.0

def _trend_votes(prev_row: pd.Series, row: pd.Series) -> Tuple[float,float]:
    trend_score=0.0; macd_score=0.0
    e20=row.get("ema20",np.nan); e50=row.get("ema50",np.nan)
    if pd.notna(e20) and pd.notna(e50):
        if e20>e50: trend_score=1.0
        elif e20<e50: trend_score=-1.0
    a1=row.get("macd",np.nan); b1=row.get("macd_signal",np.nan)
    a0=prev_row.get("macd",np.nan); b0=prev_row.get("macd_signal",np.nan)
    if pd.notna(a1) and pd.notna(b1) and pd.notna(a0) and pd.notna(b0):
        crossed_up=(a0<=b0) and (a1>b1)
        crossed_dn=(a0>=b0) and (a1<b1)
        if crossed_up: macd_score=1.0
        elif crossed_dn: macd_score=-1.0
    return trend_score,macd_score

def _apply_volatility_gating(row: pd.Series) -> bool:
    atr_pct_val=row.get("atr_pct",np.nan)
    if pd.isna(atr_pct_val): return True
    return float(atr_pct_val)>=VOL_REGIME_MIN_ATR_PCT

def _apply_adx_gating(row: pd.Series) -> bool:
    adx_val=row.get("adx",np.nan)
    if pd.isna(adx_val): return True
    return float(adx_val)>=ADX_MIN_TREND

def _atr_floor_ok(row: pd.Series) -> bool:
    price_now=float(row.get("Close",np.nan))
    atr_now=float(row.get("atr",np.nan))
    if np.isnan(price_now) or np.isnan(atr_now): return True
    floor_val=price_now*ATR_FLOOR_MULT
    return atr_now>=floor_val

def _compute_signal_regime_row(prev_row: pd.Series, row: pd.Series) -> Tuple[str,float]:
    if not _atr_floor_ok(row):
        return "Hold",0.5

    trend_reg=int(row.get("trend_regime",0))
    range_reg=int(row.get("range_regime",0))

    cont_score=0.0; cont_votes=0
    vol_ok=_apply_volatility_gating(row)
    adx_ok=_apply_adx_gating(row)
    tr_vote,macd_vote=_trend_votes(prev_row,row)
    if vol_ok and adx_ok:
        if tr_vote!=0.0:
            cont_votes+=1; cont_score+=tr_vote
        if macd_vote!=0.0:
            cont_votes+=1; cont_score+=macd_vote
    if cont_votes>0:
        cont_conf_raw=min(1.0,abs(cont_score)/cont_votes)
        if cont_score>=0.67*cont_votes: cont_side="Buy"
        elif cont_score<=-0.67*cont_votes: cont_side="Sell"
        else: cont_side="Hold"
    else:
        cont_conf_raw=0.0; cont_side="Hold"
    cont_conf=cont_conf_raw if cont_side!="Hold" else (1.0-cont_conf_raw)

    rev_score=0.0; rev_votes=0
    rsi_vote=_adaptive_rsi_vote(row)
    if rsi_vote!=0.0:
        rev_votes+=1; rev_score+=rsi_vote
    if rev_votes>0:
        rev_conf_raw=min(1.0,abs(rev_score)/rev_votes)
        if   rev_score>0: rev_side="Buy"
        elif rev_score<0: rev_side="Sell"
        else: rev_side="Hold"
    else:
        rev_conf_raw=0.0; rev_side="Hold"
    rev_conf=rev_conf_raw if rev_side!="Hold" else (1.0-rev_conf_raw)

    if trend_reg and not range_reg:
        final_side=cont_side; final_conf=cont_conf
        if final_side=="Hold" and rev_side!="Hold":
            final_side=rev_side; final_conf=rev_conf*0.6
    elif range_reg and not trend_reg:
        final_side=rev_side; final_conf=rev_conf
        if final_side=="Hold" and cont_side!="Hold":
            final_side=cont_side; final_conf=cont_conf*0.6
    else:
        if cont_side!="Hold" and rev_side!="Hold":
            if cont_side==rev_side:
                final_side=cont_side
                final_conf=min(1.0,0.5*(cont_conf+rev_conf)+0.25)
            else:
                final_side="Hold"
                final_conf=0.5*abs(cont_conf-rev_conf)
        elif cont_side!="Hold":
            final_side=cont_side; final_conf=cont_conf
        elif rev_side!="Hold":
            final_side=rev_side; final_conf=rev_conf
        else:
            final_side="Hold"; final_conf=0.5

    return final_side,final_conf

def _get_higher_tf_bias_for_asset(symbol: str, interval_key: str, use_cache: bool=True) -> int:
    higher_key = HIGHER_TF_MAP.get(interval_key, interval_key)
    if higher_key == interval_key:
        return 0
    df_hi_raw = fetch_data(symbol, interval_key=higher_key, use_cache=use_cache)
    df_hi = add_indicators(df_hi_raw)
    if df_hi is None or df_hi.empty:
        return 0
    e20 = df_hi["ema20"].iloc[-1] if "ema20" in df_hi.columns else np.nan
    e50 = df_hi["ema50"].iloc[-1] if "ema50" in df_hi.columns else np.nan
    if pd.isna(e20) or pd.isna(e50): return 0
    if e20>e50: return 1
    elif e20<e50: return -1
    return 0

def _compute_signal_row_with_higher_tf(prev_row: pd.Series, row: pd.Series,
                                       higher_bias: int, structure_mode: str) -> Tuple[str,float,Dict[str,bool]]:
    side_local,conf_local=_compute_signal_regime_row(prev_row,row)
    if higher_bias>0 and side_local=="Sell":
        side_local="Hold"; conf_local*=0.5
    elif higher_bias<0 and side_local=="Buy":
        side_local="Hold"; conf_local*=0.5

    filtered_side,tags=_structure_filter(side_local,row,structure_mode)
    if filtered_side=="Hold" and side_local!="Hold":
        conf_local*=0.4
    return filtered_side,conf_local,tags

# =============================================================================
# TP/SL dynamic
# =============================================================================

def _compute_tp_sl_regime_dynamic(price: float, atr: float, side: str, risk: str,
                                  row: pd.Series, tp_sl_mode: str) -> Tuple[float,float]:
    base=RISK_MULT.get(risk,RISK_MULT["Medium"])
    base_tp=float(base["tp_atr"])
    base_sl=float(base["sl_atr"])

    adx_now=float(row.get("adx",20.0))
    atrp_now=float(row.get("atr_pct",0.3))
    ema50_slope=float(row.get("ema50_slope",0.0))

    trend_strength=(adx_now/25.0)+(abs(ema50_slope)*10.0)
    raw_trend_factor=max(0.01,min(3.0,trend_strength))

    if atrp_now<=0: atrp_now=1e-6
    raw_vol_factor=0.3/atrp_now
    raw_vol_factor=max(0.01,min(3.0,raw_vol_factor))

    prof=_TP_SL_PROFILES.get(tp_sl_mode,_TP_SL_PROFILES["Normal"])
    trend_factor=min(max(raw_trend_factor,prof["trend_min"]),prof["trend_max"])
    vol_factor  =min(max(raw_vol_factor, prof["vol_min"]),  prof["vol_max"])

    tp_mult=base_tp*trend_factor
    sl_mult=base_sl*vol_factor

    if side=="Buy":
        tp=price+tp_mult*atr
        sl=price-sl_mult*atr
    else:
        tp=price-tp_mult*atr
        sl=price+sl_mult*atr
    return float(tp),float(sl)

# =============================================================================
# ML + calibration (v8.2: KFold CV confidence)
# =============================================================================

def _prepare_ml_frame(df: pd.DataFrame, recent_winrate_hint: Optional[float]=None) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    work=df.copy()
    hint_val=float(recent_winrate_hint) if recent_winrate_hint is not None else 50.0
    work["recent_winrate_hint"]=hint_val

    needed_cols=[
        "RSI","ema20","ema50","ema_diff","ema50_slope",
        "macd","macd_signal","macd_slope",
        "atr_pct","adx",
        "trend_regime","range_regime",
        "rsi_slope",
        "recent_winrate_hint",
        "Close",
    ]
    for c in needed_cols:
        if c not in work.columns:
            return None

    if len(work)>ML_RECENT_WINDOW:
        work=work.iloc[-ML_RECENT_WINDOW:].copy()

    work["target_up"]=(work["Close"].shift(-1)>work["Close"]).astype(int)
    work.dropna(inplace=True)

    if len(work)<40:
        return None
    return work

def _raw_ml_confidence_unscaled(df: pd.DataFrame,
                                recent_winrate_hint: Optional[float]=None) -> float:
    """
    v8.2: cross-validated ML confidence using KFold (CV_FOLDS).
    We average out-of-fold probabilities.
    """
    if RandomForestClassifier is None or KFold is None:
        return 0.5

    work=_prepare_ml_frame(df, recent_winrate_hint)
    if work is None:
        return 0.5

    feat_cols=[
        "RSI","ema20","ema50","ema_diff","ema50_slope",
        "macd","macd_signal","macd_slope",
        "atr_pct","adx",
        "trend_regime","range_regime",
        "rsi_slope",
        "recent_winrate_hint",
    ]
    X=work[feat_cols].values
    y=work["target_up"].values

    kf=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    rf_probs=[]
    gb_probs=[]

    for train_idx,test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # RF
        try:
            rf=RandomForestClassifier(n_estimators=40,max_depth=4,random_state=42)
            rf.fit(X_train,y_train)
            rf_fold=rf.predict_proba(X_test)[:,1]
            rf_probs.extend(rf_fold.tolist())
        except Exception:
            pass

        # GB
        if GradientBoostingClassifier is not None:
            try:
                gb=GradientBoostingClassifier(random_state=42)
                gb.fit(X_train,y_train)
                gb_fold=gb.predict_proba(X_test)[:,1]
                gb_probs.extend(gb_fold.tolist())
            except Exception:
                pass

    if len(rf_probs)==0 and len(gb_probs)==0:
        return 0.5

    rf_avg = np.mean(rf_probs) if len(rf_probs)>0 else 0.5
    gb_avg = np.mean(gb_probs) if len(gb_probs)>0 else rf_avg

    blended = 0.6*rf_avg + 0.4*gb_avg
    blended = max(0.05,min(0.95,blended))
    return float(blended)

def _apply_ml_calibration_to_prob(raw_prob: float,
                                  calib: Dict[str,Any],
                                  symbol: Optional[str],
                                  calibration_enabled: bool) -> float:
    raw_prob=max(0.05,min(0.95,raw_prob))
    if not calibration_enabled or symbol is None:
        return raw_prob
    stats=_get_symbol_winrate_stats(calib,symbol)
    bias=stats["bias"]
    scale=1.0+bias
    scale=max(0.85,min(1.15,scale))
    prob_adj=0.5+(raw_prob-0.5)*scale
    prob_adj=max(0.05,min(0.95,prob_adj))
    return prob_adj

def _ml_direction_confidence_calibrated(df: pd.DataFrame,
                                        recent_winrate_hint: Optional[float],
                                        calib: Dict[str,Any],
                                        symbol: Optional[str],
                                        calibration_enabled: bool) -> float:
    raw_ml=_raw_ml_confidence_unscaled(df, recent_winrate_hint)
    cal_ml=_apply_ml_calibration_to_prob(raw_ml, calib, symbol, calibration_enabled)
    return cal_ml

def _blend_confidence_calibrated(rule_conf: float,
                                 ml_conf: float,
                                 recent_winrate: float,
                                 calib: Dict[str,Any],
                                 symbol: Optional[str],
                                 calibration_enabled: bool) -> Tuple[float,float]:
    if pd.isna(recent_winrate):
        w_rule=0.6; w_ml=0.4
    elif recent_winrate>55.0:
        w_rule=0.7; w_ml=0.3
    elif recent_winrate<45.0:
        w_rule=0.7; w_ml=0.3
    else:
        w_rule=0.6; w_ml=0.4

    if calibration_enabled and symbol is not None:
        stats=_get_symbol_winrate_stats(calib,symbol)
        avg_wr=stats["avg_wr"]
        if avg_wr>55.0:
            w_ml*=1.1; w_rule*=0.9
        elif avg_wr<45.0:
            w_ml*=0.8; w_rule*=1.2

    total_w=w_rule+w_ml
    if total_w<=0:
        w_rule,w_ml=0.5,0.5; total_w=1.0
    w_rule/=total_w; w_ml/=total_w

    blended_raw=(w_rule*rule_conf)+(w_ml*ml_conf)
    blended_raw=max(0.05,min(0.95,blended_raw))

    blended_cal=_apply_ml_calibration_to_prob(
        blended_raw, calib, symbol, calibration_enabled
    )
    return blended_raw, blended_cal

# =============================================================================
# BACKTEST (same as v8.1)
# =============================================================================

def _backtest_once(df_ind: pd.DataFrame, risk: str, horizon: int, seed: int,
                   symbol: Optional[str]=None, interval_key: Optional[str]=None,
                   forced_trades_enabled: bool=True, stale_mode: bool=False,
                   tp_sl_mode: str="Normal", structure_mode: str="Off") -> Dict[str,Any]:
    np.random.seed(int(seed)%(2**32-1))
    balance=1.0; peak=1.0; wins=0; trades=0
    drawdowns=[]; trade_returns=[]

    local_forced_prob = 0.0 if stale_mode else (FORCED_TRADE_PROB if forced_trades_enabled else 0.0)
    higher_bias = _get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True) if (symbol and interval_key) else 0

    for i in range(20, len(df_ind)-horizon):
        prev_row=df_ind.iloc[i-1]
        cur_row =df_ind.iloc[i]

        side_local, conf_local, _tags = _compute_signal_row_with_higher_tf(
            prev_row, cur_row, higher_bias, structure_mode
        )
        side=side_local

        trend_reg=int(cur_row.get("trend_regime",0))
        range_reg=int(cur_row.get("range_regime",0))
        bullish_trend = trend_reg and (cur_row.get("ema20",0)>cur_row.get("ema50",0))
        bearish_trend = trend_reg and (cur_row.get("ema20",0)<cur_row.get("ema50",0))
        regime_allows_relaxed = (
            (side=="Buy" and bullish_trend) or
            (side=="Sell" and bearish_trend) or
            (range_reg and side in ["Buy","Sell"])
        )

        if side=="Hold":
            if (conf_local>FORCED_CONF_MIN) and (np.random.rand()<local_forced_prob):
                side=np.random.choice(["Buy","Sell"])
            else:
                continue
        else:
            if conf_local < CONF_EXEC_THRESHOLD and not regime_allows_relaxed:
                continue

        price_now=float(cur_row["Close"])
        atr_now=float(cur_row.get("atr", price_now*0.005))
        tp_lvl, sl_lvl = _compute_tp_sl_regime_dynamic(
            price_now, atr_now, side, risk, cur_row, tp_sl_mode
        )

        if side=="Buy":
            reward_dist=max(tp_lvl-price_now,1e-12)
            risk_dist  =max(price_now-sl_lvl,1e-12)
        else:
            reward_dist=max(price_now-tp_lvl,1e-12)
            risk_dist  =max(sl_lvl-price_now,1e-12)
        rr_local = reward_dist/risk_dist if risk_dist!=0 else 1.0

        dyn_horizon=int(horizon*(0.8+conf_local*0.6))
        dyn_horizon=max(dyn_horizon,1)
        dyn_horizon=min(dyn_horizon,horizon*2)

        hit=None
        for j in range(1,dyn_horizon+1):
            if i+j>=len(df_ind): break
            nxt=df_ind.iloc[i+j]
            nxt_px=float(nxt["Close"])
            if side=="Buy":
                if nxt_px>=tp_lvl: hit="TP"; break
                elif nxt_px<=sl_lvl: hit="SL"; break
            else:
                if nxt_px<=tp_lvl: hit="TP"; break
                elif nxt_px>=sl_lvl: hit="SL"; break

        if hit is not None:
            trades+=1
            impact_scale=max(conf_local,0.05)
            if hit=="TP":
                gain=0.01*rr_local*impact_scale
                balance*=(1.0+gain)
                wins+=1
                trade_returns.append(gain)
            else:
                loss=0.01/max(rr_local,1e-12)*impact_scale
                balance*=(1.0-loss)
                trade_returns.append(-loss)

        peak=max(peak,balance)
        dd=(peak-balance)/peak if peak>0 else 0
        drawdowns.append(dd)

    if trades==0:
        trades=1; wins=1
        drawdowns.append(0.0)
        trade_returns.append(0.01)

    total_ret_pct=(balance-1.0)*100.0
    winrate_pct =(wins/trades)*100.0
    maxdd_pct   =(max(drawdowns)*100.0) if drawdowns else 0.0
    sharpe_like =(winrate_pct/maxdd_pct) if maxdd_pct>0 else winrate_pct

    total_gain=sum([x for x in trade_returns if x>0])
    total_loss=-sum([x for x in trade_returns if x<0])
    if total_loss<=1e-12:
        profit_factor=float("inf") if total_gain>0 else 0.0
    else:
        profit_factor=total_gain/total_loss
    ev_pct=float(np.mean(trade_returns))*100.0 if len(trade_returns)>0 else 0.0

    return {
        "winrate": winrate_pct,
        "trades": trades,
        "return": total_ret_pct,
        "maxdd": maxdd_pct,
        "sharpe": sharpe_like,
        "profit_factor": profit_factor,
        "ev_pct": ev_pct,
    }

def backtest_signals(df: pd.DataFrame, risk: str, horizon: int=10,
                     symbol: Optional[str]=None, interval_key: Optional[str]=None,
                     filter_level: str="Balanced", weekend_mode: bool=True,
                     calibration_enabled: bool=True,
                     forced_trades_enabled: bool=False,
                     tp_sl_mode: str="Normal",
                     structure_mode: str="Off") -> Dict[str,Any]:
    out={
        "winrate":0.0,"trades":0,"return":0.0,"maxdd":0.0,"sharpe":0.0,
        "winrate_std":0.0,"profit_factor":0.0,"ev_pct":0.0,"stale":False,
    }
    if df is None or df.empty or len(df)<40:
        return out

    calib=_load_calibration()
    calib_bias=_compute_calibration_bias_for_symbol(
        calib, symbol if symbol else "UNKNOWN", calibration_enabled
    )
    _apply_filter_level_and_calibration(filter_level, calib_bias)

    df_use, stale_flag = _hard_skip_if_stale(
        df, symbol if symbol else "UNKNOWN",
        interval_key if interval_key else "1h",
        weekend_mode
    )
    out["stale"]=bool(stale_flag)

    df_ind=add_indicators(df_use)
    if df_ind.empty or len(df_ind)<40:
        return out

    seeds=[42,99,123,2024,777]
    results=[
        _backtest_once(
            df_ind, risk, horizon, seed=s,
            symbol=symbol, interval_key=interval_key,
            forced_trades_enabled=forced_trades_enabled,
            stale_mode=stale_flag,
            tp_sl_mode=tp_sl_mode,
            structure_mode=structure_mode,
        ) for s in seeds
    ]

    winrates=[r["winrate"] for r in results]
    trades_list=[r["trades"] for r in results]
    returns=[r["return"] for r in results]
    maxdds=[r["maxdd"] for r in results]
    sharpes=[r["sharpe"] for r in results]
    pfs=[r["profit_factor"] for r in results]
    evs=[r["ev_pct"] for r in results]

    mean_winrate=float(np.mean(winrates))
    mean_trades=int(np.mean(trades_list))
    mean_return=float(np.mean(returns))
    mean_maxdd=float(np.mean(maxdds))
    mean_sharpe=float(np.mean(sharpes))
    mean_pf=float(np.mean(pfs))
    mean_ev=float(np.mean(evs))
    winrate_std=float(np.std(winrates))

    out["winrate"]=round(mean_winrate,2)
    out["trades"]=mean_trades
    out["return"]=round(mean_return,2)
    out["maxdd"]=round(mean_maxdd,2)
    out["sharpe"]=round(mean_sharpe,2)
    out["winrate_std"]=round(winrate_std,2)
    out["profit_factor"]=round(mean_pf,2)
    out["ev_pct"]=round(mean_ev,4)

    if symbol and (not stale_flag) and calibration_enabled:
        calib=_record_calibration(calib, symbol, out["winrate"])
        _save_calibration(calib)

    return out

# =============================================================================
# SNAPSHOT / PREDICTION
# =============================================================================

def latest_prediction(df: pd.DataFrame, risk: str="Medium",
                      recent_winrate_hint: Optional[float]=None,
                      symbol: Optional[str]=None,
                      interval_key: Optional[str]=None,
                      weekend_mode: bool=True,
                      calibration_enabled: bool=True,
                      tp_sl_mode: str="Normal",
                      structure_mode: str="Off") -> Optional[Dict[str,Any]]:
    if df is None or df.empty or len(df)<60:
        return None

    df_use, stale_flag_market = _hard_skip_if_stale(
        df, symbol if symbol else "UNKNOWN",
        interval_key if interval_key else "1h",
        weekend_mode
    )
    df_ind=add_indicators(df_use)
    if df_ind.empty or len(df_ind)<60:
        return None

    prev_row=df_ind.iloc[-2]
    row_now =df_ind.iloc[-1]

    higher_bias=_get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True) if (symbol and interval_key) else 0
    side_rule, rule_conf, _tags = _compute_signal_row_with_higher_tf(
        prev_row, row_now, higher_bias, structure_mode
    )

    calib=_load_calibration()
    ml_conf_cal=_ml_direction_confidence_calibrated(
        df_ind, recent_winrate_hint, calib, symbol, calibration_enabled
    )

    blended_raw, blended_cal=_blend_confidence_calibrated(
        rule_conf, ml_conf_cal,
        recent_winrate_hint if recent_winrate_hint is not None else np.nan,
        calib, symbol, calibration_enabled
    )

    last_price=float(row_now["Close"])
    atr_now=float(row_now.get("atr", last_price*0.005))

    stale_flag_legacy=_is_stale_df_simple(df_ind)
    final_stale_flag=bool(stale_flag_legacy or stale_flag_market)

    if side_rule=="Hold":
        tp_fallback, sl_fallback=_compute_tp_sl_regime_dynamic(
            last_price, atr_now, "Buy", risk, row_now, tp_sl_mode
        )
        reward=tp_fallback-last_price
        riskv =last_price-sl_fallback
        rr_est=(reward/riskv) if (riskv and riskv!=0) else None

        return {
            "symbol": None,
            "side": "Hold",
            "probability_raw": round(blended_raw,2),
            "probability_calibrated": round(blended_cal,2),
            "probability": round(blended_cal,2),
            "sentiment": None,
            "price": last_price,
            "tp": float(tp_fallback),
            "sl": float(sl_fallback),
            "rr": float(rr_est) if rr_est is not None and math.isfinite(rr_est) else None,
            "atr": atr_now,
            "stale": final_stale_flag,
        }

    tp, sl = _compute_tp_sl_regime_dynamic(
        last_price, atr_now, side_rule, risk, row_now, tp_sl_mode
    )

    if side_rule=="Buy":
        reward=tp-last_price
        riskv =last_price-sl
    else:
        reward=last_price-tp
        riskv =sl-last_price
    rr=(reward/riskv) if (riskv and riskv!=0) else None

    return {
        "symbol": None,
        "side": side_rule,
        "probability_raw": round(blended_raw,2),
        "probability_calibrated": round(blended_cal,2),
        "probability": round(blended_cal,2),
        "sentiment": None,
        "price": last_price,
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr) if rr is not None and math.isfinite(rr) else None,
        "atr": atr_now,
        "stale": final_stale_flag,
    }

# =============================================================================
# MARKERS FOR CHART
# =============================================================================

def generate_signal_points(df_ind: pd.DataFrame,
                           symbol: Optional[str]=None,
                           interval_key: Optional[str]=None,
                           weekend_mode: bool=True,
                           structure_mode: str="Off") -> Dict[str,List[Any]]:
    out={
        "buy_times": [],"buy_prices": [],
        "sell_times": [],"sell_prices": [],
        "dip_buy_times": [],"dip_buy_prices": [],
        "bull_breakout_times": [],"bull_breakout_prices": [],
        "bear_breakdown_times": [],"bear_breakdown_prices": [],
        "support_series": [],"resistance_series": [],
        "time_index": [],
    }
    if df_ind is None or df_ind.empty or len(df_ind)<3:
        return out

    df_trim,_=_hard_skip_if_stale(
        df_ind, symbol if symbol else "UNKNOWN",
        interval_key if interval_key else "1h",
        weekend_mode
    )
    if df_trim.empty or len(df_trim)<3:
        return out

    higher_bias=_get_higher_tf_bias_for_asset(symbol, interval_key, use_cache=True) if (symbol and interval_key) else 0

    buy_t=[]; buy_p=[]
    sell_t=[]; sell_p=[]
    dip_t=[]; dip_p=[]
    bull_t=[]; bull_p=[]
    bear_t=[]; bear_p=[]

    for i in range(1,len(df_trim)):
        prev_row=df_trim.iloc[i-1]
        cur_row =df_trim.iloc[i]

        side,_conf,tags=_compute_signal_row_with_higher_tf(
            prev_row, cur_row, higher_bias, structure_mode
        )
        px=float(cur_row["Close"]); ts=df_trim.index[i]

        if side=="Buy":
            buy_t.append(ts); buy_p.append(px)
        elif side=="Sell":
            sell_t.append(ts); sell_p.append(px)

        if tags.get("dip",False):
            dip_t.append(ts); dip_p.append(px)
        if tags.get("bull_breakout",False):
            bull_t.append(ts); bull_p.append(px)
        if tags.get("bear_breakdown",False):
            bear_t.append(ts); bear_p.append(px)

    out["buy_times"]=buy_t; out["buy_prices"]=buy_p
    out["sell_times"]=sell_t; out["sell_prices"]=sell_p
    out["dip_buy_times"]=dip_t; out["dip_buy_prices"]=dip_p
    out["bull_breakout_times"]=bull_t; out["bull_breakout_prices"]=bull_p
    out["bear_breakdown_times"]=bear_t; out["bear_breakdown_prices"]=bear_p

    out["time_index"]=list(df_trim.index)
    out["support_series"]=list(df_trim["support_level"])
    out["resistance_series"]=list(df_trim["resistance_level"])

    return out

# =============================================================================
# SENTIMENT
# =============================================================================

def compute_sentiment_stub(symbol: str) -> float:
    bias_map={"BTC-USD":0.65,"^NDX":0.6,"CL=F":0.55}
    base=bias_map.get(symbol,0.5)
    return round(float(base),2)

# =============================================================================
# PUBLIC PIPELINE HELPERS
# =============================================================================

def analyze_asset(symbol: str, interval_key: str, risk: str="Medium", use_cache: bool=True,
                  filter_level: str="Balanced", weekend_mode: bool=True,
                  calibration_enabled: bool=True,
                  forced_trades_enabled: bool=False,
                  tp_sl_mode: str="Normal",
                  structure_mode: str="Off") -> Optional[Dict[str,Any]]:
    df_raw=fetch_data(symbol, interval_key, use_cache=use_cache)
    if df_raw.empty: return None

    bt=backtest_signals(
        df_raw, risk, horizon=10,
        symbol=symbol, interval_key=interval_key,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        forced_trades_enabled=forced_trades_enabled,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
    )

    df_ind_full=add_indicators(df_raw)
    if df_ind_full.empty: return None

    pred=latest_prediction(
        df_raw, risk,
        recent_winrate_hint=bt["winrate"],
        symbol=symbol, interval_key=interval_key,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
    )

    sentiment_val=compute_sentiment_stub(symbol)
    last_px=float(df_ind_full["Close"].iloc[-1])
    atr_last=float(df_ind_full["atr"].iloc[-1]) if "atr" in df_ind_full.columns else None

    df_tmp,sf_tmp=_hard_skip_if_stale(df_raw, symbol, interval_key, weekend_mode)
    stale_flag=bt.get("stale",False) or sf_tmp

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
            "winrate_std": bt.get("winrate_std",0.0),
            "return_pct": bt["return"],
            "trades": bt["trades"],
            "maxdd": bt["maxdd"],
            "sharpe": bt["sharpe"],
            "profit_factor": bt.get("profit_factor",0.0),
            "ev_pct": bt.get("ev_pct",0.0),
            "stale": stale_flag,
            "df": df_ind_full,
        }

    return {
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
        "winrate_std": bt.get("winrate_std",0.0),
        "return_pct": bt["return"],
        "trades": bt["trades"],
        "maxdd": bt["maxdd"],
        "sharpe": bt["sharpe"],
        "profit_factor": bt.get("profit_factor",0.0),
        "ev_pct": bt.get("ev_pct",0.0),
        "stale": stale_flag,
        "df": df_ind_full,
    }

def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool=True,
                               filter_level: str="Balanced",
                               weekend_mode: bool=True,
                               calibration_enabled: bool=True,
                               structure_mode: str="Off") -> Tuple[str,pd.DataFrame,Dict[str,List[Any]]]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol=ASSET_SYMBOLS[asset]

    calib=_load_calibration()
    calib_bias=_compute_calibration_bias_for_symbol(
        calib, symbol, calibration_enabled
    )
    _apply_filter_level_and_calibration(filter_level, calib_bias)

    df_raw=fetch_data(symbol, interval_key, use_cache=use_cache)
    df_ind=add_indicators(df_raw)

    sig_pts=generate_signal_points(
        df_ind, symbol=symbol, interval_key=interval_key,
        weekend_mode=weekend_mode, structure_mode=structure_mode,
    )
    return symbol, df_ind, sig_pts

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str,
                                  use_cache: bool=True,
                                  filter_level: str="Balanced",
                                  weekend_mode: bool=True,
                                  calibration_enabled: bool=True,
                                  forced_trades_enabled: bool=False,
                                  tp_sl_mode: str="Normal",
                                  structure_mode: str="Off") -> Tuple[Optional[Dict[str,Any]],pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        return None,pd.DataFrame()
    symbol=ASSET_SYMBOLS[asset]

    df_raw=fetch_data(symbol, interval_key, use_cache=use_cache)
    if df_raw.empty:
        return None,pd.DataFrame()

    bt=backtest_signals(
        df_raw, risk, horizon=10,
        symbol=symbol, interval_key=interval_key,
        filter_level=filter_level,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        forced_trades_enabled=forced_trades_enabled,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
    )

    df_ind=add_indicators(df_raw)
    if df_ind.empty:
        return None,pd.DataFrame()

    pred=latest_prediction(
        df_raw, risk,
        recent_winrate_hint=bt["winrate"],
        symbol=symbol, interval_key=interval_key,
        weekend_mode=weekend_mode,
        calibration_enabled=calibration_enabled,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
    )

    sentiment_val=compute_sentiment_stub(symbol)
    last_px=float(df_ind["Close"].iloc[-1])
    atr_last=float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None

    _,sf_tmp=_hard_skip_if_stale(df_raw, symbol, interval_key, weekend_mode)
    stale_flag=bt.get("stale",False) or sf_tmp

    if pred is None:
        fallback={
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
            "win_rate_std": bt.get("winrate_std",0.0),
            "backtest_return_pct": bt["return"],
            "trades": bt["trades"],
            "maxdd": bt["maxdd"],
            "sharpe": bt["sharpe"],
            "profit_factor": bt.get("profit_factor",0.0),
            "ev_pct": bt.get("ev_pct",0.0),
            "stale": stale_flag,
        }
        return fallback, df_ind

    enriched={
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
        "win_rate_std": bt.get("winrate_std",0.0),
        "backtest_return_pct": bt["return"],
        "trades": bt["trades"],
        "maxdd": bt["maxdd"],
        "sharpe": bt["sharpe"],
        "profit_factor": bt.get("profit_factor",0.0),
        "ev_pct": bt.get("ev_pct",0.0),
        "stale": stale_flag,
    }
    return enriched, df_ind

def summarize_assets(interval_key: str="1h", risk: str="Medium", use_cache: bool=True,
                      filter_level: str="Balanced", weekend_mode: bool=True,
                      calibration_enabled: bool=True,
                      forced_trades_enabled: bool=False,
                      tp_sl_mode: str="Normal",
                      structure_mode: str="Off") -> pd.DataFrame:
    rows=[]
    for asset_name,symbol in ASSET_SYMBOLS.items():
        try:
            res=analyze_asset(
                symbol, interval_key, risk, use_cache,
                filter_level=filter_level,
                weekend_mode=weekend_mode,
                calibration_enabled=calibration_enabled,
                forced_trades_enabled=forced_trades_enabled,
                tp_sl_mode=tp_sl_mode,
                structure_mode=structure_mode,
            )
        except Exception:
            res=None
        if not res: continue

        rows.append({
            "Asset": asset_name,
            "Symbol": symbol,
            "Interval": interval_key,
            "Price": res["last_price"],
            "Signal": res["signal"],
            "Probability": res.get("probability",0.5),
            "ProbabilityRaw": res.get("probability_raw",0.5),
            "ProbabilityCal": res.get("probability_calibrated",0.5),
            "Sentiment": res["sentiment"],
            "TP": res["tp"],
            "SL": res["sl"],
            "RR": res["rr"],
            "Trades": res["trades"],
            "WinRate": res["winrate"],
            "WinRateStd": res.get("winrate_std",0.0),
            "EV%": res.get("ev_pct",0.0),
            "ProfitFactor": res.get("profit_factor",0.0),
            "Return%": res["return_pct"],
            "MaxDD%": res["maxdd"],
            "SharpeLike": res["sharpe"],
            "Stale": res.get("stale",False),
        })
    if not rows:
        return pd.DataFrame()
    summary_df=pd.DataFrame(rows)
    summary_df.sort_values("Asset", inplace=True, ignore_index=True)
    return summary_df