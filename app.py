# utils.py — complete and corrected
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
try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = MACD = RSIIndicator = AverageTrueRange = None

# --------------------------------------------------------------------------------------
# CONSTANTS & CONFIG
# --------------------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h":  {"interval": "60m", "period": "1mo", "min_rows": 300},
    "4h":  {"interval": "240m", "period": "3mo", "min_rows": 250},
    "1d":  {"interval": "1d", "period": "1y", "min_rows": 200},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 150},
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

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------

def _log(msg: str) -> None:
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
        val = df[col].values
        if isinstance(val, np.ndarray) and getattr(val, "ndim", 1) > 1:
            df[col] = pd.Series(val.ravel(), index=df.index)
        if col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="all")
    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        return _normalize_ohlcv(raw)
    except Exception as e:
        _log(f"⚠️ {symbol}: fetch error {e}")
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
        _log(f"⚠️ Mirror history error for {symbol}: {e}")
        return pd.DataFrame()


def fetch_data(
    symbol: str,
    interval_key: str = "1h",
    use_cache: bool = True,
    max_retries: int = 4,
    backoff_range: Tuple[float, float] = (3.5, 12.5),
) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(f"Unknown interval_key '{interval_key}'. Known: {list(INTERVALS.keys())}")

    interval = str(INTERVALS[interval_key]["interval"])
    period = str(INTERVALS[interval_key]["period"])
    min_rows = int(INTERVALS[interval_key]["min_rows"])

    _log(f"⏳ Fetching {symbol} [{interval}] for {period}...")
    cache_fp = _cache_path(symbol, interval_key)

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
        _log(f"⚠️ {symbol}: invalid or insufficient data ({got} rows), retrying...")
        time.sleep(np.random.uniform(*backoff_range))

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
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

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
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI
    try:
        rsi_series = RSIIndicator(close=df["Close"], window=14).rsi()
    except Exception:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
    df["RSI"] = rsi_series
    df["rsi"] = df["RSI"]

    # MACD
    try:
        macd = MACD(close=df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    except Exception:
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# --------------------------------------------------------------------------------------
# SIGNAL ENGINE
# --------------------------------------------------------------------------------------

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    score = 0.0
    votes = 0

    if pd.notna(row["ema20"]) and pd.notna(row["ema50"]):
        votes += 1
        if row["ema20"] > row["ema50"]:
            score += 1.0
        elif row["ema20"] < row["ema50"]:
            score -= 1.0

    if pd.notna(row["RSI"]):
        votes += 1
        if row["RSI"] < 30:
            score += 1.0
        elif row["RSI"] > 70:
            score -= 1.0

    if (
        pd.notna(row["macd"])
        and pd.notna(row["macd_signal"])
        and pd.notna(row_prev["macd"])
        and pd.notna(row_prev["macd_signal"])
    ):
        votes += 1
        crossed_up = (row_prev["macd"] <= row_prev["macd_signal"]) and (row["macd"] > row["macd_signal"])
        crossed_dn = (row_prev["macd"] >= row_prev["macd_signal"]) and (row["macd"] < row["macd_signal"])
        if crossed_up:
            score += 1.0
        elif crossed_dn:
            score -= 1.0

    conf = 0.0 if votes == 0 else min(1.0, abs(score) / votes)
    if score >= 0.67 * votes:
        return "Buy", conf
    elif score <= -0.67 * votes:
        return "Sell", conf
    else:
        return "Hold", 1.0 - conf


def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    tp_k = float(m["tp_atr"])
    sl_k = float(m["sl_atr"])
    tp = price + tp_k * atr if side == "Buy" else price - tp_k * atr
    sl = price - sl_k * atr if side == "Buy" else price + sl_k * atr
    return float(tp), float(sl)


def latest_prediction(df: pd.DataFrame, risk: str = "Medium") -> Optional[Dict[str, object]]:
    if df is None or df.empty or len(df) < 60:
        return None
    df = add_indicators(df)
    if df.empty or len(df) < 60:
        return None

    row_prev = df.iloc[-2]
    row = df.iloc[-1]
    side, prob = compute_signal_row(row_prev, row)

    if side == "Hold":
        return {"side": "Hold", "prob": prob, "price": float(row["Close"]),
                "tp": None, "sl": None, "atr": float(row["atr"]) if pd.notna(row["atr"]) else None}

    atr_val = float(row["atr"]) if pd.notna(row["atr"]) else float(df["atr"].iloc[-14:].mean())
    tp, sl = compute_tp_sl(float(row["Close"]), atr_val, side, risk)

    return {"side": side, "prob": prob, "price": float(row["Close"]),
            "tp": tp, "sl": sl, "atr": atr_val}

# --------------------------------------------------------------------------------------
# BACKTEST (FIXED)
# --------------------------------------------------------------------------------------

def backtest_signals(df: pd.DataFrame, risk: str = "Medium", hold_allowed: bool = True) -> Dict[str, object]:
    out = {"win_rate": None, "total_return_pct": None, "n_trades": 0, "trades": []}

    if df is None or df.empty or len(df) < 120:
        return out

    df = add_indicators(df)
    if df.empty:
        return out

    signals: List[Tuple[pd.Timestamp, str, float, float, float]] = []
    prev = df.iloc[0]
    for i in range(1, len(df)):
        row = df.iloc[i]
        side, conf = compute_signal_row(prev, row)
        if side == "Hold" and hold_allowed:
            signals.append((row.name, "Hold", conf, np.nan, np.nan))
        else:
            atr_here = row["atr"] if pd.notna(row["atr"]) else df["atr"].iloc[max(0, i - 14):i + 1].mean()
            tp, sl = compute_tp_sl(row["Close"], atr_here, side, risk)
            signals.append((row.name, side, conf, tp, sl))
        prev = row

    position = None
    eq_curve, wins = 0.0, 0
    trades: List[Dict[str, object]] = []

    for idx in range(len(signals)):
        ts, side, conf, tp, sl = signals[idx]
        candle_idx = idx + 1
        if candle_idx >= len(df):
            break

        price = float(df["Close"].iloc[candle_idx])
        high = float(df["High"].iloc[candle_idx])
        low = float(df["Low"].iloc[candle_idx])

        if position is None:
            if side in ("Buy", "Sell"):
                position = (side, price, tp, sl, ts)
            continue

        pos_side, entry_px, pos_tp, pos_sl, entry_ts = position
        exit_reason, exit_px = None, price

        if pos_side == "Buy":
            if not np.isnan(pos_tp) and high >= pos_tp:
                exit_reason, exit_px = "TP", pos_tp
            elif not np.isnan(pos_sl) and low <= pos_sl:
                exit_reason, exit_px = "SL", pos_sl
            elif side == "Sell":
                exit_reason = "Flip"
        else:
            if not np.isnan(pos_tp) and low <= pos_tp:
                exit_reason, exit_px = "TP", pos_tp
            elif not np.isnan(pos_sl) and high >= pos_sl:
                exit_reason, exit_px = "SL", pos_sl
            elif side == "Buy":
                exit_reason = "Flip"

        if exit_reason:
            ret = (exit_px - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
            eq_curve += ret
            wins += 1 if ret > 0 else 0
            trades.append({
                "entry_time": entry_ts, "exit_time": ts,
                "side": pos_side, "entry": entry_px, "exit": exit_px,
                "reason": exit_reason, "return_pct": ret * 100.0,
            })
            position = None
            if exit_reason == "Flip" and side in ("Buy", "Sell"):
                position = (side, price, tp, sl, ts)

    n = len(trades)
    out["n_trades"] = n
    out["win_rate"] = 100.0 * (wins / n) if n > 0 else 0.0
    out["total_return_pct"] = 100.0 * eq_curve if n > 0 else 0.0
    out["trades"] = trades
    return out

# --------------------------------------------------------------------------------------
# PIPELINES
# --------------------------------------------------------------------------------------

def analyze_asset(symbol: str, interval_key: str, risk: str = "Medium", use_cache: bool = True) -> Optional[Dict[str, object]]:
    df = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    if df.empty:
        return None

    df = add_indicators(df)
    if df.empty:
        return None

    pred = latest_prediction(df, risk=risk)
    if pred is None:
        return None

    bt = backtest_signals(df, risk=risk, hold_allowed=True)

    return {
        "symbol": symbol, "interval_key": interval_key, "risk": risk,
        "last_price": float(df["Close"].iloc[-1]),
        "signal": pred["side"], "probability": round(pred["prob"] * 100.0, 2),
        "tp": pred["tp"], "sl": pred["sl"], "atr": pred["atr"],
        "win_rate": bt.get("win_rate"), "total_return_pct": bt.get("total_return_pct"),
        "n_trades": bt.get("n_trades"), "df": df, "trades": bt["trades"],
    }


def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> pd.DataFrame:
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
                "Asset": asset, "Symbol": symbol, "Interval": interval_key,
                "Price": res["last_price"], "Signal": res["signal"],
                "Probability_%": res["probability"], "TP": res["tp"], "SL": res["sl"],
                "WinRate_%": res["win_rate"], "BacktestReturn_%": res["total_return_pct"],
                "Trades": res["n_trades"],
            })
        except Exception as e:
            _log(f"❌ Error analyzing {asset}: {e}")

    if not rows:
        _log("No assets could be analyzed. Check your connection or data source.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = ["Asset", "Symbol", "Interval", "Price", "Signal",
            "Probability_%", "TP", "SL", "WinRate_%", "BacktestReturn_%", "Trades"]
    return df[[c for c in cols if c in df.columns]]


def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key=interval_key, use_cache=use_cache)
    df = add_indicators(df)
    return symbol, df


def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
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
        "asset": asset, "symbol": symbol, "interval": interval_key,
        "price": float(df["Close"].iloc[-1]), "side": pred["side"],
        "probability": round(pred["prob"] * 100.0, 2),
        "tp": pred["tp"], "sl": pred["sl"], "atr": pred["atr"],
        "win_rate": bt.get("win_rate"), "backtest_return_pct": bt.get("total_return_pct"),
        "n_trades": bt.get("n_trades"), "trades": bt.get("trades", []),
    }
    return pred_out, df

# --------------------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------------------