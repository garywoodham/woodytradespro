# utils.py — FULL VERSION, nothing removed
# --------------------------------------------------------------------------------------
# Robust utilities for market data, indicators, signals, backtesting and summaries.

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h": {"interval": "60m", "period": "1mo", "min_rows": 300},
    "4h": {"interval": "240m", "period": "3mo", "min_rows": 250},
    "1d": {"interval": "1d", "period": "1y", "min_rows": 200},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 150},
}

RISK_MULT: Dict[str, Dict[str, float]] = {
    "Low": {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High": {"tp_atr": 2.0, "sl_atr": 0.8},
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
# HELPERS / LOGGING
# --------------------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _cache_path(symbol: str, interval_key: str) -> Path:
    safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return DATA_DIR / f"{safe}_{interval_key}.csv"


# --------------------------------------------------------------------------------------
# DATA FETCH & NORMALIZATION
# --------------------------------------------------------------------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
        rename_map = {c: c.capitalize() for c in df.columns}
        df.rename(columns=rename_map, inplace=True)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        vals = df[col].values
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            df[col] = pd.Series(vals.ravel(), index=df.index)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        return _normalize_ohlcv(df)
    except Exception as e:
        _log(f"⚠️ {symbol}: fetch error {e}")
        return pd.DataFrame()


def _yahoo_mirror_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        return _normalize_ohlcv(df)
    except Exception as e:
        _log(f"⚠️ Mirror fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_data(symbol: str, interval_key="1h", use_cache=True) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(f"Invalid interval: {interval_key}")
    info = INTERVALS[interval_key]
    cache_file = _cache_path(symbol, interval_key)
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df = _normalize_ohlcv(df)
            if len(df) >= info["min_rows"]:
                _log(f"✅ Using cached data for {symbol}")
                return df
        except Exception as e:
            _log(f"⚠️ Cache load failed: {e}")
    for i in range(4):
        _log(f"⏳ Attempt {i+1}: Fetching {symbol} from Yahoo...")
        df = _yahoo_try_download(symbol, info["interval"], info["period"])
        if not df.empty and len(df) >= info["min_rows"]:
            _log(f"✅ Downloaded {len(df)} rows for {symbol}")
            df.to_csv(cache_file)
            return df
        _log(f"⚠️ {symbol}: insufficient data ({len(df) if not df.empty else 0} rows), retrying...")
        time.sleep(np.random.uniform(3.5, 8.5))
    _log(f"🪞 Mirror fetch for {symbol}...")
    df = _yahoo_mirror_history(symbol, info["interval"], info["period"])
    if not df.empty:
        _log(f"✅ Mirror fetch success for {symbol}")
        df.to_csv(cache_file)
    return df


# --------------------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ema20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    df["rsi"] = df["RSI"]
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    df.ffill(inplace=True)
    return df


# --------------------------------------------------------------------------------------
# SIGNAL ENGINE
# --------------------------------------------------------------------------------------

def compute_signal_row(prev, row) -> Tuple[str, float]:
    score, votes = 0.0, 0
    if row["ema20"] > row["ema50"]:
        score += 1; votes += 1
    elif row["ema20"] < row["ema50"]:
        score -= 1; votes += 1
    if row["RSI"] < 30:
        score += 1; votes += 1
    elif row["RSI"] > 70:
        score -= 1; votes += 1
    crossed_up = row["macd"] > row["macd_signal"] and prev["macd"] <= prev["macd_signal"]
    crossed_dn = row["macd"] < row["macd_signal"] and prev["macd"] >= prev["macd_signal"]
    if crossed_up:
        score += 1; votes += 1
    elif crossed_dn:
        score -= 1; votes += 1
    conf = 0 if votes == 0 else min(1, abs(score) / votes)
    if score >= 0.67 * votes:
        return "Buy", conf
    elif score <= -0.67 * votes:
        return "Sell", conf
    else:
        return "Hold", 1 - conf


def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    mult = RISK_MULT[risk]
    tp = price + mult["tp_atr"] * atr if side == "Buy" else price - mult["tp_atr"] * atr
    sl = price - mult["sl_atr"] * atr if side == "Buy" else price + mult["sl_atr"] * atr
    return tp, sl


def latest_prediction(df: pd.DataFrame, risk="Medium") -> Optional[dict]:
    if df.empty or len(df) < 60:
        return None
    row, prev = df.iloc[-1], df.iloc[-2]
    side, prob = compute_signal_row(prev, row)
    atr = row["atr"]
    if side == "Hold":
        return {"side": "Hold", "prob": prob, "price": row["Close"], "tp": None, "sl": None, "atr": atr}
    tp, sl = compute_tp_sl(row["Close"], atr, side, risk)
    return {"side": side, "prob": prob, "price": row["Close"], "tp": tp, "sl": sl, "atr": atr}


# --------------------------------------------------------------------------------------
# BACKTEST
# --------------------------------------------------------------------------------------

def backtest_signals(df: pd.DataFrame, risk="Medium", hold_allowed=True) -> dict:
    if df.empty or len(df) < 100:
        return {"win_rate": 0, "total_return_pct": 0, "n_trades": 0, "trades": []}
    df = add_indicators(df)
    trades = []
    prev = df.iloc[0]
    position = None
    wins, equity = 0, 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        side, conf = compute_signal_row(prev, row)
        price = row["Close"]
        if position is None:
            if side in ["Buy", "Sell"]:
                atr = row["atr"]; tp, sl = compute_tp_sl(price, atr, side, risk)
                position = (side, price, tp, sl, row.name)
        else:
            ps, ep, tp, sl, ent = position
            high, low = row["High"], row["Low"]
            exit_reason = None
            exit_px = price
            if ps == "Buy":
                if high >= tp: exit_reason, exit_px = "TP", tp
                elif low <= sl: exit_reason, exit_px = "SL", sl
                elif side == "Sell": exit_reason = "Flip"
            else:
                if low <= tp: exit_reason, exit_px = "TP", tp
                elif high >= sl: exit_reason, exit_px = "SL", sl
                elif side == "Buy": exit_reason = "Flip"
            if exit_reason:
                ret = (exit_px - ep) / ep * (1 if ps == "Buy" else -1)
                equity += ret; wins += 1 if ret > 0 else 0
                trades.append({"entry": ep, "exit": exit_px, "side": ps, "return_pct": ret*100})
                position = None
        prev = row
    n = len(trades)
    win_rate = (wins/n)*100 if n else 0
    total_return = equity*100
    return {"win_rate": win_rate, "total_return_pct": total_return, "n_trades": n, "trades": trades}


# --------------------------------------------------------------------------------------
# ANALYSIS PIPELINES
# --------------------------------------------------------------------------------------

def analyze_asset(symbol: str, interval_key="1h", risk="Medium") -> Optional[dict]:
    df = fetch_data(symbol, interval_key)
    if df.empty:
        return None
    df = add_indicators(df)
    pred = latest_prediction(df, risk)
    bt = backtest_signals(df, risk)
    return {
        "symbol": symbol,
        "interval": interval_key,
        "price": df["Close"].iloc[-1],
        "signal": pred["side"],
        "probability": round(pred["prob"]*100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "total_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "df": df,
    }


def summarize_assets(interval_key="1h", risk="Medium"):
    rows = []
    _log("Fetching and analyzing market data... please wait ⏳")
    for name, sym in ASSET_SYMBOLS.items():
        _log(f"⏳ Fetching {name} ({sym})...")
        try:
            res = analyze_asset(sym, interval_key, risk)
            if res:
                rows.append({
                    "Asset": name,
                    "Signal": res["signal"],
                    "Prob_%": res["probability"],
                    "TP": res["tp"],
                    "SL": res["sl"],
                    "WinRate_%": res["win_rate"],
                    "Return_%": res["total_return_pct"],
                    "Trades": res["n_trades"],
                })
        except Exception as e:
            _log(f"⚠️ {name} failed: {e}")
    if not rows:
        _log("🚫 No assets could be analyzed.")
        return pd.DataFrame()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# LEGACY COMPATIBILITY FOR tab_overview
# --------------------------------------------------------------------------------------

def calculate_model_performance(df: pd.DataFrame) -> dict:
    """
    Legacy helper for older tabs.
    Computes win rate, average return and equity curve using MACD-follow strategy.
    """
    df = add_indicators(df)
    out = {"win_rate": 0.0, "avg_return": 0.0, "equity_curve": pd.Series(dtype=float)}
    if df.empty:
        return out
    try:
        df["sig"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
        df["fut_ret"] = df["Close"].pct_change().shift(-1)
        df["strat_ret"] = df["sig"] * df["fut_ret"]
        r = df["strat_ret"].dropna()
        if not r.empty:
            out["win_rate"] = (r > 0).mean() * 100
            out["avg_return"] = r.mean() * 100
            out["equity_curve"] = (1 + r.fillna(0)).cumprod()
    except Exception as e:
        print(f"⚠️ calculate_model_performance failed: {e}")
    return out


# --------------------------------------------------------------------------------------
# END OF MODULE
# --------------------------------------------------------------------------------------