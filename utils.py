# utils.py  —  full module
# --------------------------------------------------------------------------------------
# Robust data fetch, caching, indicators, signal engine (BUY/SELL/HOLD),
# backtesting (win rate & total return), asset summarization for all tabs.
# Safe against yfinance API quirks and pandas shape issues.

from __future__ import annotations

import os
import time
import math
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import yfinance as yf

# Technical indicators (ta)
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# --------------------------------------------------------------------------------------
# CONSTANTS & CONFIG
# --------------------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Restored all intervals (including 15m and 1wk) with sensible default periods and row targets
INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h":  {"interval": "60m", "period": "1mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "3mo", "min_rows": 250},
    "1d":  {"interval": "1d", "period": "1y", "min_rows": 200},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 150},
}

# Risk multipliers for TP/SL (percent of ATR)
RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

# Asset map (kept as-is and widely used across tabs)
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
# HELPERS: Safe prints for Streamlit console visibility (tabs show what's happening)
# --------------------------------------------------------------------------------------

def _log(msg: str) -> None:
    # Print-only logging that is always visible in Streamlit terminal logs
    try:
        print(msg, flush=True)
    except Exception:
        pass

# --------------------------------------------------------------------------------------
# DATA FETCH & CACHE
# --------------------------------------------------------------------------------------

def _cache_path(symbol: str, interval_key: str) -> Path:
    safe_sym = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return DATA_DIR / f"{safe_sym}_{interval_key}.csv"

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we return a clean OHLCV DataFrame with DatetimeIndex and 1-D columns.
    Fixes for:
      - MultiIndex columns from yfinance
      - ndarray shape (n,1) -> Series
      - Mixed dtypes & NaNs
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # If MultiIndex (columns like ('Open','GC=F')) -> take level 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only known OHLCV columns if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
        # some indices come as 'close' lowercase from alternative sources
        # standardize common variants
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    df = df[keep].copy()

    # Make sure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # Ensure each column is 1-D numeric where needed
    for col in df.columns:
        val = df[col].values
        if isinstance(val, np.ndarray) and getattr(val, "ndim", 1) > 1:
            df[col] = pd.Series(val.ravel(), index=df.index)
        # force numeric for prices/volume except where it's clearly not numeric
        if col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with all-NaN and forward/back fill small gaps
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")
    return df

def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Primary yahoo fetch using yf.download (robust to params). Avoids threads for stability.
    """
    try:
        # yfinance can be finicky: disable threads and progress, let auto_adjust default be True (new default)
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            threads=False,
        )
        return _normalize_ohlcv(raw)
    except Exception as e:
        _log(f"⚠️ {symbol}: fetch error {e}")
        return pd.DataFrame()

def _yahoo_mirror_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Mirror attempt using Ticker.history with a couple of parameter variants,
    which often succeeds after rate-limit / CDN quirks.
    """
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        df = _normalize_ohlcv(raw)
        if not df.empty:
            return df
        # variant: auto_adjust False
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
    backoff_range: Tuple[float, float] = (3.5, 12.5),
) -> pd.DataFrame:
    """
    Unified fetch that:
      1) uses cache if available and fresh enough to meet min_rows
      2) tries Yahoo (retries) then mirror
      3) writes CSV cache when successful
    """
    if interval_key not in INTERVALS:
        raise KeyError(f"Unknown interval_key '{interval_key}'. Known: {list(INTERVALS.keys())}")

    interval = str(INTERVALS[interval_key]["interval"])
    period   = str(INTERVALS[interval_key]["period"])
    min_rows = int(INTERVALS[interval_key]["min_rows"])

    _log(f"⏳ Fetching {symbol} [{interval}] for {period}...")

    cache_fp = _cache_path(symbol, interval_key)

    # 1) Try cache
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"✅ Using cached {symbol} ({len(cached)} rows).")
                return cached
            else:
                _log(f"ℹ️ Cache exists for {symbol} but only {len(cached)} rows; needs {min_rows}.")
        except Exception as e:
            _log(f"⚠️ Cache read failed for {symbol}: {e}")

    # 2) Live fetch with retry backoff
    for attempt in range(1, max_retries + 1):
        _log(f"⏳ Attempt {attempt}: Fetching {symbol} from Yahoo...")
        df = _yahoo_try_download(symbol, interval, period)
        if not df.empty and len(df) >= min_rows:
            _log(f"✅ {symbol}: fetched {len(df)} rows.")
            try:
                df.to_csv(cache_fp)
                _log(f"💾 Cached {symbol} data → {cache_fp}")
            except Exception as e:
                _log(f"⚠️ Cache write failed for {symbol}: {e}")
            return df

        got = len(df) if isinstance(df, pd.DataFrame) else "N/A"
        _log(f"⚠️ {symbol}: invalid or insufficient data ({type(df)} with {got} rows), retrying...")
        # backoff
        low, high = backoff_range
        time.sleep(np.random.uniform(low, high))

    # 3) Mirror fallback
    _log(f"🪞 Attempting mirror fetch for {symbol}...")
    df = _yahoo_mirror_history(symbol, interval, period)
    if not df.empty and len(df) >= min_rows:
        _log(f"✅ Mirror fetch succeeded for {symbol}.")
        try:
            df.to_csv(cache_fp)
            _log(f"💾 Cached {symbol} data → {cache_fp}")
        except Exception as e:
            _log(f"⚠️ Cache write failed for {symbol}: {e}")
        return df

    _log(f"🚫 All attempts failed for {symbol}, returning empty DataFrame.")
    return pd.DataFrame()

# --------------------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA20, EMA50, RSI(14), MACD(12,26,9), ATR(14).
    Exposes both 'RSI' and 'rsi' to be compatible with Trends tab.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Guarantee close/high/low are 1-D and numeric
    for col in ["Close", "High", "Low"]:
        if col not in df.columns:
            return pd.DataFrame()
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EMA
    try:
        df["ema20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    except Exception:
        # fallbacks if ta hiccups
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI
    try:
        rsi_series = RSIIndicator(close=df["Close"], window=14).rsi()
    except Exception:
        # naive RSI fallback
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))

    df["RSI"] = rsi_series
    df["rsi"] = df["RSI"]  # ensure Trends tab never KeyError('rsi')

    # MACD
    try:
        macd = MACD(close=df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    except Exception:
        # fallback simple MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR
    try:
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["atr"] = atr.average_true_range()
    except Exception:
        tr1 = (df["High"] - df["Low"]).abs()
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    # Clean edges
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# --------------------------------------------------------------------------------------
# SIGNAL ENGINE (BUY/SELL/HOLD) + TP/SL
# --------------------------------------------------------------------------------------

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Rule-based ensemble:
      - EMA cross: ema20 above ema50 => bullish bias, below => bearish bias
      - RSI: <30 oversold (bullish), >70 overbought (bearish)
      - MACD cross up/down supports trend
    Result: ('Buy' | 'Sell' | 'Hold', probability 0..1)
    """
    score = 0.0
    votes = 0

    # EMA trend
    if pd.notna(row["ema20"]) and pd.notna(row["ema50"]):
        votes += 1
        if row["ema20"] > row["ema50"]:
            score += 1.0
        elif row["ema20"] < row["ema50"]:
            score -= 1.0

    # RSI context
    if pd.notna(row["RSI"]):
        votes += 1
        if row["RSI"] < 30:
            score += 1.0
        elif row["RSI"] > 70:
            score -= 1.0

    # MACD cross
    if pd.notna(row["macd"]) and pd.notna(row["macd_signal"]) and pd.notna(row_prev["macd"]) and pd.notna(row_prev["macd_signal"]):
        votes += 1
        crossed_up = (row_prev["macd"] <= row_prev["macd_signal"]) and (row["macd"] > row["macd_signal"])
        crossed_dn = (row_prev["macd"] >= row_prev["macd_signal"]) and (row["macd"] < row["macd_signal"])
        if crossed_up:
            score += 1.0
        elif crossed_dn:
            score -= 1.0

    # Normalize confidence
    conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)

    # Decision with HOLD buffer (don’t force trades)
    if score >= 0.67 * votes:
        return "Buy", conf
    elif score <= -0.67 * votes:
        return "Sell", conf
    else:
        return "Hold", 1.0 - conf  # low conviction -> Hold

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Price-based TP/SL using ATR and risk multipliers.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])
    tp = price + tp_k * atr if side == "Buy" else price - tp_k * atr
    sl = price - sl_k * atr if side == "Buy" else price + sl_k * atr
    return float(tp), float(sl)

def latest_prediction(df: pd.DataFrame, risk: str = "Medium") -> Optional[Dict[str, object]]:
    """
    Compute the latest signal/TP/SL based on the last 2 rows (needs indicators).
    """
    if df is None or df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    if df.empty or len(df) < 60:
        return None

    row_prev = df.iloc[-2]
    row = df.iloc[-1]
    side, prob = compute_signal_row(row_prev, row)

    if side == "Hold":
        return {
            "side": "Hold",
            "prob": prob,
            "price": float(row["Close"]),
            "tp": None,
            "sl": None,
            "atr": float(row["atr"]) if pd.notna(row["atr"]) else None,
        }

    atr_val = float(row["atr"]) if pd.notna(row["atr"]) else float(df["atr"].iloc[-14:].mean())
    tp, sl = compute_tp_sl(price=float(row["Close"]), atr=atr_val, side=side, risk=risk)

    return {
        "side": side,
        "prob": prob,
        "price": float(row["Close"]),
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
    }

# --------------------------------------------------------------------------------------
# BACKTEST
# --------------------------------------------------------------------------------------

def backtest_signals(
    df: pd.DataFrame,
    risk: str = "Medium",
    hold_allowed: bool = True,
) -> Dict[str, object]:
    """
    Simple sequential backtest:
      - Enter when signal flips from Hold/flat to Buy or Sell
      - Exit at TP or SL (simulated on future bars) or when opposite signal appears
      - Reports win rate, total return (%), #trades and trade list
    """
    out = {
        "win_rate": None,
        "total_return_pct": None,
        "n_trades": 0,
        "trades": [],
    }

    if df is None or df.empty or len(df) < 120:
        return out

    df = add_indicators(df)
    if df.empty:
        return out

    # Build signals per-bar
    signals: List[Tuple[pd.Timestamp, str, float, float, float]] = []  # (ts, side, conf, tp, sl)
    prev = df.iloc[0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        side, conf = compute_signal_row(prev, row)
        if side == "Hold" and hold_allowed:
            signals.append((row.name, "Hold", conf, np.nan, np.nan))
        else:
            atr_here = row["atr"] if pd.notna(row["atr"]) else df["atr"].iloc[max(0, i-14):i+1].mean()
            tp, sl = compute_tp_sl(row["Close"], atr_here, side, risk)
            signals.append((row.name, side, conf, tp, sl))
        prev = row

    # Walk forward: enter on Buy/Sell when flat or switching direction
    position = None  # None | ("Buy"|"Sell", entry_px, tp, sl, entry_time)
    eq_curve = 0.0
    wins = 0
    trades = []

    # Using next bars to detect TP/SL hits
    for idx in range(1, len(df)):
        ts, side, conf, tp, sl = signals[idx]
        price = float(df["Close"].iloc[idx])

        if position is None:
            if side in ("Buy", "Sell"):
                position = (side, price, tp, sl, ts)
        else:
            pos_side, entry_px, pos_tp, pos_sl, entry_ts = position
            high = float(df["High"].iloc[idx])
            low = float(df["Low"].iloc[idx])

            exit_reason = None
            exit_px = price

            if pos_side == "Buy":
                # hit TP/SL?
                if not np.isnan(pos_tp) and high >= pos_tp:
                    exit_reason = "TP"
                    exit_px = pos_tp
                elif not np.isnan(pos_sl) and low <= pos_sl:
                    exit_reason = "SL"
                    exit_px = pos_sl
                elif side == "Sell":  # opposite signal
                    exit_reason = "Flip"
            else:  # Sell
                if not np.isnan(pos_tp) and low <= pos_tp:
                    exit_reason = "TP"
                    exit_px = pos_tp
                elif not np.isnan(pos_sl) and high >= pos_sl:
                    exit_reason = "SL"
                    exit_px = pos_sl
                elif side == "Buy":
                    exit_reason = "Flip"

            if exit_reason:
                ret = (exit_px - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
                eq_curve += ret
                wins += 1 if ret > 0 else 0
                trades.append({
                    "entry_time": entry_ts,
                    "exit_time": ts,
                    "side": pos_side,
                    "entry": entry_px,
                    "exit": exit_px,
                    "reason": exit_reason,
                    "return_pct": ret * 100.0,
                })
                position = None

                # If flip, immediately enter new
                if exit_reason == "Flip" and side in ("Buy", "Sell"):
                    position = (side, price, tp, sl, ts)

    n = len(trades)
    out["n_trades"] = n
    if n > 0:
        out["win_rate"] = 100.0 * (wins / n)
        out["total_return_pct"] = 100.0 * eq_curve
    else:
        out["win_rate"] = 0.0
        out["total_return_pct"] = 0.0
    out["trades"] = trades
    return out

# --------------------------------------------------------------------------------------
# PIPELINES FOR TABS
# --------------------------------------------------------------------------------------

def analyze_asset(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Optional[Dict[str, object]]:
    """
    Full pipeline for a single asset:
      - fetch (cache+retry+mirror)
      - indicators
      - latest signal & TP/SL
      - backtest stats (win rate, total return)
    """
    df = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df.empty:
        return None

    # Indicators
    df = add_indicators(df)
    if df.empty:
        return None

    # Latest prediction
    pred = latest_prediction(df, risk=risk)
    if pred is None:
        return None

    # Backtest (on the fetched window)
    bt = backtest_signals(df, risk=risk, hold_allowed=True)

    out = {
        "symbol": symbol,
        "interval_key": interval_key,
        "risk": risk,
        "last_price": float(df["Close"].iloc[-1]),
        "signal": pred["side"],
        "probability": float(round(pred["prob"] * 100.0, 2)),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],
        "win_rate": bt.get("win_rate", None),
        "total_return_pct": bt.get("total_return_pct", None),
        "n_trades": bt.get("n_trades", 0),
        "df": df,               # for charts in tabs
        "trades": bt["trades"], # for scenario tab table/chart
    }
    return out

def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> pd.DataFrame:
    """
    Iterate across ASSET_SYMBOLS and return a tidy summary DataFrame used in Overview/Insights/Summary tabs.
    Also prints progress lines exactly like you asked (kept behavior).
    """
    rows = []
    _log("Fetching and analyzing market data... please wait ⏳")
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"⏳ Fetching {asset} ({symbol})...")
        try:
            res = analyze_asset(symbol, interval_key=interval_key, risk=risk, use_cache=use_cache)
            if res is None:
                _log(f"⚠️ Could not analyze {asset}.")
                continue

            rows.append({
                "Asset": asset,
                "Symbol": symbol,
                "Interval": interval_key,
                "Price": res["last_price"],
                "Signal": res["signal"],
                "Probability_%": res["probability"],
                "TP": res["tp"],
                "SL": res["sl"],
                "WinRate_%": res["win_rate"],
                "BacktestReturn_%": res["total_return_pct"],
                "Trades": res["n_trades"],
            })
        except Exception as e:
            _log(f"❌ Error analyzing {asset}: {e}")

    if not rows:
        _log("No assets could be analyzed. Please check your internet connection or data source.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Order columns nicely
    cols = [
        "Asset", "Symbol", "Interval", "Price",
        "Signal", "Probability_%", "TP", "SL",
        "WinRate_%", "BacktestReturn_%", "Trades"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df

# --------------------------------------------------------------------------------------
# Utilities for tabs to pull a single asset (used by Detailed/Trends/Scenarios)
# --------------------------------------------------------------------------------------

def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Given display asset name (e.g., 'Gold'), fetch symbol and return (symbol, df_with_indicators)
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    df = add_indicators(df)
    return symbol, df

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
    """
    Helper for Scenarios/Trends tabs: returns (prediction dict, df_with_indicators) for selected asset.
    """
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()

    df = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df.empty:
        return None, pd.DataFrame()
    df = add_indicators(df)
    if df.empty:
        return None, pd.DataFrame()

    pred = latest_prediction(df, risk=risk)
    bt = backtest_signals(df, risk=risk, hold_allowed=True)

    if pred is None:
        return None, df

    pred_out = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": float(round(pred["prob"] * 100.0, 2)),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],
        "win_rate": bt.get("win_rate", None),
        "backtest_return_pct": bt.get("total_return_pct", None),
        "n_trades": bt.get("n_trades", 0),
        "trades": bt.get("trades", []),
    }
    return pred_out, df

# --------------------------------------------------------------------------------------
# LEGACY COMPATIBILITY: calculate_model_performance (for tab_overview)
# --------------------------------------------------------------------------------------

def calculate_model_performance(df: pd.DataFrame) -> dict:
    """
    Backward-compatible helper for Overview tab.
    Computes basic performance stats (win rate, avg return, equity curve)
    based on MACD signal-following logic.
    """
    df = add_indicators(df)
    out = {"win_rate": 0.0, "avg_return": 0.0, "equity_curve": pd.Series(dtype=float)}
    if df is None or df.empty:
        return out

    try:
        df["sig"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
        df["fut_ret"] = df["Close"].pct_change().shift(-1)
        df["strat_ret"] = df["sig"] * df["fut_ret"]
        r = df["strat_ret"].dropna()
        if not r.empty:
            out["win_rate"] = float((r > 0).mean() * 100)
            out["avg_return"] = float(r.mean() * 100)
            out["equity_curve"] = (1 + r.fillna(0)).cumprod()
    except Exception as e:
        print(f"⚠️ calculate_model_performance failed: {e}")
    return out

# --------------------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------------------