# utils_fetch.py  -- drop into your utils module (replace your existing fetch_data)

import yfinance as yf
import pandas as pd
import time
import os
import json
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(".yf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# map intervals to max allowed period to avoid Yahoo errors
INTERVAL_MAX_PERIOD = {
    "1m": "7d",
    "2m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "90d",  # usually OK
    "90m": "90d",
    "1h": "90d",
    "1d": "max",   # daily can be long
    "1wk": "max",
    "1mo": "max"
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yf.download when group_by or multiple tickers used."""
    if isinstance(df.columns, pd.MultiIndex):
        # if df came from yf.download with single ticker, columns might be ('Adj Close', '')
        df.columns = ["_".join([str(i) for i in col if i and col is not None]).strip("_") for col in df.columns.values]
        # attempt to rename common patterns back to standard
        rename_map = {}
        for c in df.columns:
            low = c.lower()
            if "open" in low:
                rename_map[c] = "Open"
            elif "high" in low:
                rename_map[c] = "High"
            elif "low" in low:
                rename_map[c] = "Low"
            elif "close" in low:
                rename_map[c] = "Close"
            elif "volume" in low:
                rename_map[c] = "Volume"
        df = df.rename(columns=rename_map)
    return df

def _cache_path(symbol: str, interval: str, period: str):
    name = f"{symbol.replace('/','_')}_{interval}_{period}.parquet"
    return CACHE_DIR / name

def fetch_data(symbol: str, interval: str = "1h", period: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Robust fetch wrapper for yfinance with retries, period checking, caching, and fallbacks.
    - symbol: ticker symbol for yfinance (e.g. 'GC=F')
    - interval: '1m','15m','1h','1d', etc
    - period: optional override like '60d','90d','max'. If None we pick a safe default.
    """
    # choose safe default period based on interval
    if period is None:
        maxp = INTERVAL_MAX_PERIOD.get(interval, "60d")
        period = maxp if maxp != "max" else "5y"

    cache_file = _cache_path(symbol, interval, period)
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception:
            pass

    # attempt several retries with backoff
    last_exc = None
    for attempt in range(4):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, prepost=False, threads=False)
            if df is None or df.empty:
                raise ValueError("No data returned")
            df = _normalize_columns(df)

            # ensure required columns exist
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(set(df.columns)):
                # try a daily fallback (sometimes intraday not available)
                if interval not in ("1d", "1wk", "1mo"):
                    df2 = yf.download(symbol, period="5y", interval="1d", progress=False, prepost=False, threads=False)
                    df2 = _normalize_columns(df2)
                    if df2 is not None and not df2.empty:
                        df = df2
                    else:
                        raise ValueError("Data missing core columns")
                else:
                    raise ValueError("Data missing core columns")

            # compute simple indicators used by app so downstream code doesn't fail
            if "Close" in df.columns:
                df["MA_10"] = df["Close"].rolling(10, min_periods=1).mean()
                df["MA_50"] = df["Close"].rolling(50, min_periods=1).mean()

            # persist cache
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

            return df

        except Exception as e:
            last_exc = e
            wait = (2 ** attempt) * 0.5
            time.sleep(wait)
            continue

    # all retries failed: raise a helpful error
    raise RuntimeError(f"Failed to fetch {symbol} (interval={interval}, period={period}): {last_exc}")