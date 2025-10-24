# utils.py - FINAL MERGED VERSION (v1 + v2)
# ---------------------------------------------------------------------------
# Includes:
# - Robust data fetch, caching, indicators
# - v1: basic technical signal engine (EMA, RSI, MACD, ATR)
# - v1: basic backtest and pipelines
# - v2: sentiment, market regime detection, ML classifier (RandomForest)
# - v2: fused adaptive signal + improved backtest + summary
# ---------------------------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

# ML / Sentiment
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Technical indicators
try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = MACD = RSIIndicator = AverageTrueRange = None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h":  {"interval": "60m", "period": "2mo", "min_rows": 200},
    "4h":  {"interval": "240m", "period": "3mo", "min_rows": 200},
    "1d":  {"interval": "1d", "period": "1y", "min_rows": 150},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 100},
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

# ---------------------------------------------------------------------------
# HELPERS
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
    safe_sym = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return DATA_DIR / f"{safe_sym}_{interval_key}.csv"


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
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="all")
    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        return _normalize_ohlcv(raw)
    except Exception as e:
        _log(f"Fetch error for {symbol}: {e}")
        return pd.DataFrame()


def _yahoo_mirror_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        raw2 = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        return _normalize_ohlcv(raw2)
    except Exception as e:
        _log(f"Mirror fetch error for {symbol}: {e}")
        return pd.DataFrame()


def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True,
               max_retries: int = 4, backoff_range: Tuple[float, float] = (3.5, 12.5)) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(f"Unknown interval_key {interval_key}")
    interval = INTERVALS[interval_key]["interval"]
    period = INTERVALS[interval_key]["period"]
    min_rows = INTERVALS[interval_key]["min_rows"]
    cache_fp = _cache_path(symbol, interval_key)
    _log(f"Fetching {symbol} [{interval}] for {period}...")
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"Using cached {symbol} ({len(cached)} rows).")
                return cached
        except Exception as e:
            _log(f"Cache read failed: {e}")
    for attempt in range(1, max_retries + 1):
        df = _yahoo_try_download(symbol, interval, period)
        if not df.empty and len(df) >= min_rows:
            _log(f"{symbol}: fetched {len(df)} rows.")
            try:
                df.to_csv(cache_fp)
            except Exception:
                pass
            return df
        _log(f"Retry {attempt} failed for {symbol} ({len(df)} rows).")
        time.sleep(np.random.uniform(*backoff_range))
    _log(f"Trying mirror fetch for {symbol}...")
    df = _yahoo_mirror_history(symbol, interval, period)
    if not df.empty and len(df) >= min_rows:
        try:
            df.to_csv(cache_fp)
        except Exception:
            pass
        return df
    _log(f"All fetch attempts failed for {symbol}.")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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
    df["rsi"] = df["RSI"]
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
        tr1 = (df["High"] - df["Low"]).abs()
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"] - df["Close"].shift(1)).abs()
        df["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ---------------------------------------------------------------------------
# SIGNAL ENGINE (v1)
# ---------------------------------------------------------------------------

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    score, votes = 0.0, 0
    if pd.notna(row.get("ema20")) and pd.notna(row.get("ema50")):
        votes += 1
        score += 1 if row["ema20"] > row["ema50"] else -1
    if pd.notna(row.get("RSI")):
        votes += 1
        if row["RSI"] < 40:
            score += 1
        elif row["RSI"] > 60:
            score -= 1
    if (pd.notna(row.get("macd")) and pd.notna(row.get("macd_signal")) and
        pd.notna(row_prev.get("macd")) and pd.notna(row_prev.get("macd_signal"))):
        votes += 1
        crossed_up = row_prev["macd"] <= row_prev["macd_signal"] and row["macd"] > row["macd_signal"]
        crossed_dn = row_prev["macd"] >= row_prev["macd_signal"] and row["macd"] < row["macd_signal"]
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1
    conf = 0 if votes == 0 else abs(score) / votes
    if score > 0:
        return "Buy", conf
    elif score < 0:
        return "Sell", conf
    return "Hold", 0.2

# ---------------------------------------------------------------------------
# CONTINUES BELOW...