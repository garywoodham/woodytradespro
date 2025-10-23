# utils.py
# --------------------------------------------------------------------------------------
# WoodyTrades Pro - Utilities
# Full version with:
# - Robust data fetch (retries, mirror, caching)
# - Intervals: 15m, 1h, 4h, 1d, 1w
# - Indicators (RSI, MACD, EMA20/50) with lowercase aliases for safety
# - ATR-based TP/SL with risk profiles
# - ML trend model (Logistic Regression) + accuracy
# - Backtesting helpers (win rate, expectancy, equity curve)
# - Scenario helpers (backtest_signals, position sizing, trade plan)
# - Asset summary builder for Overview tab
# - Model signal + performance helpers for new summary tab
#
# Nothing removed; only additive fixes.
# --------------------------------------------------------------------------------------

import os
import time
import json
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Directories / Constants
# --------------------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ASSET_SYMBOLS = {
    "Gold": "GC=F",
    "NASDAQ 100": "^NDX",
    "S&P 500": "^GSPC",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "Crude Oil": "CL=F",
    "Bitcoin": "BTC-USD",
}

# Intervals config (kept exactly as requested)
INTERVALS = {
    "15m": {"interval": "15m", "period": "7d", "min_rows": 200},
    "1h": {"interval": "1h", "period": "60d", "min_rows": 300},
    "4h": {"interval": "4h", "period": "720d", "min_rows": 200},
    "1d": {"interval": "1d", "period": "5y", "min_rows": 250},
    "1w": {"interval": "1wk", "period": "10y", "min_rows": 150},
}

# Risk profiles (used by tabs and scenarios)
RISK_MULT = {"Low": 0.75, "Medium": 1.0, "High": 1.5}

# Default train interval used in summarize (can be overridden by tabs)
DEFAULT_INTERVAL_KEY = "1h"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def cache_path(symbol: str, interval_key: str) -> str:
    safe_symbol = symbol.replace("=", "_").replace("^", "")
    return os.path.join(DATA_DIR, f"{safe_symbol}_{interval_key}.csv")


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return (
        df.replace([np.inf, -np.inf], np.nan)
          .ffill()
          .bfill()
    )


def _has_ohlc(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    needed = {"Open", "High", "Low", "Close"}
    return needed.issubset(set(df.columns))


def get_interval_params(interval_key: str):
    return INTERVALS.get(interval_key, INTERVALS[DEFAULT_INTERVAL_KEY])


# --------------------------------------------------------------------------------------
# Data Fetching (retries + mirror + caching)
# --------------------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = DEFAULT_INTERVAL_KEY, max_retries: int = 4) -> pd.DataFrame:
    """
    Fetch market data with retry, caching, and mirror fallback.
    Prints progress messages so the UI can echo them.
    """
    params = get_interval_params(interval_key)
    period = params["period"]
    interval = params["interval"]
    min_rows = params["min_rows"]
    path = cache_path(symbol, interval_key)

    print(f"⏳ Fetching {symbol} ({interval})...")

    # Load cache quickly if present and sufficient
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty and len(df) >= min_rows and _has_ohlc(df):
                print(f"💾 Loaded cached {symbol} ({len(df)} rows)")
                return _safe_fill(df)
        except Exception:
            pass

    # Primary fetch with retries
    for attempt in range(1, max_retries + 1):
        print(f"⏳ Attempt {attempt}: Fetching {symbol} from Yahoo...")
        try:
            # yfinance Ticker.history avoids deprecation warnings about use of download kwargs
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            if isinstance(df, tuple) or df is None:
                raise ValueError("Invalid data type returned")
            if df.empty or len(df) < min_rows:
                print(f"⚠️ {symbol}: invalid or insufficient data ({type(df)} with {len(df)} rows), retrying...")
                time.sleep(2.0 + attempt)
                continue
            if not _has_ohlc(df):
                print(f"⚠️ {symbol}: missing OHLC columns, retrying...")
                time.sleep(1.5)
                continue
            df = _safe_fill(df)
            df.to_csv(path)
            print(f"💾 Cached {symbol} data → {path}")
            return df
        except Exception as e:
            print(f"⚠️ {symbol}: fetch error {e}")
            time.sleep(2.0 + attempt)

    # Mirror fallback
    print(f"🪞 Attempting mirror fetch for {symbol}...")
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if not df.empty and _has_ohlc(df):
            df = _safe_fill(df)
            df.to_csv(path)
            print(f"✅ Mirror fetch succeeded for {symbol}.")
            return df
    except Exception as e:
        print(f"❌ Mirror fetch failed for {symbol}: {e}")

    # Last-resort: return cache even if older/short
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty and _has_ohlc(df):
                print(f"⚠️ Using last cached {symbol} (may be stale/short).")
                return _safe_fill(df)
        except Exception:
            pass

    print(f"🚫 No valid data fetched for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD (and signal), EMA_20, EMA_50, ATR (14).
    Also provides lowercase aliases to prevent KeyErrors in plotting tabs.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not _has_ohlc(df):
        return df

    df = df.copy()
    df = _safe_fill(df)

    try:
        # Core indicators
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
        df["EMA_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
        macd_calc = MACD(df["Close"])
        df["MACD"] = macd_calc.macd()
        df["Signal"] = macd_calc.macd_signal()
        df["ATR_14"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

        # Lowercase aliases (tabs sometimes reference lowercase)
        df["rsi"] = df["RSI"]
        df["ema20"] = df["EMA_20"]
        df["ema50"] = df["EMA_50"]
        df["macd"] = df["MACD"]
        df["signal"] = df["Signal"]
        df["atr"] = df["ATR_14"]
    except Exception as e:
        print(f"⚠️ Indicator calculation failed: {e}")

    return _safe_fill(df)


# --------------------------------------------------------------------------------------
# TP/SL computation (ATR-based) + Trading Theory helpers
# --------------------------------------------------------------------------------------
def compute_tp_sl(close: float, atr: float, direction: str, risk_level: str = "Medium", tp_mult: float = 2.0):
    """
    ATR-based TP/SL with risk multiplier.
    """
    mult = RISK_MULT.get(risk_level, 1.0)
    # Protect against zero/NaN ATR
    if atr is None or np.isnan(atr) or atr <= 0:
        atr = close * 0.005  # fallback ~0.5%

    sl_dist = atr * mult
    tp_dist = atr * mult * tp_mult

    if direction == "Buy":
        tp = close + tp_dist
        sl = close - sl_dist
    elif direction == "Sell":
        tp = close - tp_dist
        sl = close + sl_dist
    else:
        tp, sl = close, close
    return tp, sl


@dataclass
class TradePlan:
    direction: str
    confidence: float
    entry: float
    tp: float
    sl: float
    rr: float
    risk_level: str


def build_trade_plan(df: pd.DataFrame, direction: str, risk_level: str = "Medium", tp_mult: float = 2.0) -> TradePlan:
    df = add_indicators(df)
    if df.empty or "Close" not in df:
        return TradePlan("Hold", 0, 0, 0, 0, 0, risk_level)
    close = float(df["Close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if "atr" in df else np.nan
    tp, sl = compute_tp_sl(close, atr, direction, risk_level, tp_mult=tp_mult)
    rr = abs((tp - close) / (close - sl)) if sl != close else 0
    confidence = 0.0
    if "macd" in df and "signal" in df:
        diff = float(df["macd"].iloc[-1] - df["signal"].iloc[-1])
        confidence = min(abs(diff) * 100, 100)
    return TradePlan(direction, confidence, close, tp, sl, rr, risk_level)


# --------------------------------------------------------------------------------------
# ML Model (Logistic Regression trend predictor)
# --------------------------------------------------------------------------------------
def train_and_predict(df: pd.DataFrame, horizon: str = "1h", risk_level: str = "Medium"):
    """
    Train LR on indicators to predict next-bar direction.
    Returns a dict with: 'signal', 'accuracy', 'tp', 'sl'
    """
    result = {"signal": None, "accuracy": 0.0, "tp": None, "sl": None}
    if df is None or df.empty:
        return result

    df = add_indicators(df)
    if df.empty:
        return result

    # Define target: next close up/down
    try:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        features = ["RSI", "EMA_20", "EMA_50", "MACD", "Signal"]
        df = df.dropna(subset=features + ["Target"])
        if len(df) < 150:
            return result

        X = df[features].values
        y = df["Target"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        split = int(len(Xs) * 0.8)
        X_train, X_test = Xs[:split], Xs[split:]
        y_train, y_test = y[:split], y[split:]

        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        acc = float(model.score(X_test, y_test))

        last_pred = int(model.predict(Xs[-1].reshape(1, -1))[0])
        direction = "Buy" if last_pred == 1 else "Sell"

        # TP/SL from ATR theory
        close = float(df["Close"].iloc[-1])
        atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df else np.nan
        tp, sl = compute_tp_sl(close, atr, direction, risk_level, tp_mult=2.0)

        result.update({"signal": direction, "accuracy": acc * 100.0, "tp": tp, "sl": sl})
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")

    return result


# --------------------------------------------------------------------------------------
# Backtesting / Scenarios
# --------------------------------------------------------------------------------------
def backtest_signals(df: pd.DataFrame, rule: str = "macd_cross", hold_period: int = 1):
    """
    Simple backtest engine.
    rule:
      - 'macd_cross': long when MACD>Signal, short when MACD<Signal
      - 'ema_trend': long when EMA20>EMA50, short when EMA20<EMA50
    hold_period: number of bars to hold each signal (if not flipped earlier)
    Returns dict with win_rate, expectancy, total_return, n_trades, and per-trade series.
    """
    out = {"win_rate": 0.0, "expectancy": 0.0, "total_return": 0.0, "n_trades": 0, "equity_curve": None}
    if df is None or df.empty:
        return out

    df = add_indicators(df)
    if df.empty:
        return out

    df = df.copy()
    df["signal_num"] = 0
    if rule == "macd_cross" and "macd" in df and "signal" in df:
        df.loc[df["macd"] > df["signal"], "signal_num"] = 1
        df.loc[df["macd"] < df["signal"], "signal_num"] = -1
    elif rule == "ema_trend" and "ema20" in df and "ema50" in df:
        df.loc[df["ema20"] > df["ema50"], "signal_num"] = 1
        df.loc[df["ema20"] < df["ema50"], "signal_num"] = -1
    else:
        return out

    # Strategy returns: position * next-bar return
    df["future_ret"] = df["Close"].pct_change().shift(-1)
    df["strat_ret"] = df["signal_num"] * df["future_ret"]
    # Optionally enforce hold period by rolling but keep it simple as cross flips anyway

    # Trade stats
    rets = df["strat_ret"].dropna()
    if rets.empty:
        return out
    wins = (rets > 0).sum()
    n = len(rets)
    win_rate = (wins / n) * 100.0
    expectancy = rets.mean() * 100.0
    equity = (1 + rets.fillna(0)).cumprod()

    out.update(
        {
            "win_rate": win_rate,
            "expectancy": expectancy,
            "total_return": (equity.iloc[-1] - 1.0) * 100.0,
            "n_trades": int(n),
            "equity_curve": equity,
        }
    )
    return out


def simulate_position_size(balance: float, risk_pct: float, entry: float, sl: float, contract_size: float = 1.0):
    """
    Basic position sizing using fixed % risk of balance and distance to SL.
    """
    if entry <= 0 or sl <= 0 or balance <= 0 or risk_pct <= 0:
        return 0.0, 0.0
    risk_amt = balance * risk_pct
    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        return 0.0, 0.0
    units = (risk_amt / stop_dist) / contract_size
    return units, risk_amt


# --------------------------------------------------------------------------------------
# Overview / Summary helpers
# --------------------------------------------------------------------------------------
def summarize_assets(interval_key: str = DEFAULT_INTERVAL_KEY, risk_level: str = "Medium") -> pd.DataFrame:
    """
    Iterate assets, fetch data, run ML prediction, run short backtest, return tidy DataFrame.
    Columns: Asset, Symbol, Price, Signal, Accuracy, TP, SL, WinRate, Expectancy, NTrades
    """
    rows = []
    for asset, symbol in ASSET_SYMBOLS.items():
        try:
            print(f"⏳ Processing {asset} ({symbol}) ...")
            df = fetch_data(symbol, interval_key=interval_key)
            if df.empty:
                print(f"⚠️ Skipping {asset} (no data)")
                continue

            pred = train_and_predict(df, horizon=interval_key, risk_level=risk_level)
            if not pred["signal"]:
                print(f"⚠️ Could not predict {asset}.")
                continue

            back = backtest_signals(df, rule="macd_cross")
            rows.append(
                {
                    "Asset": asset,
                    "Symbol": symbol,
                    "Price": float(df["Close"].iloc[-1]),
                    "Signal": pred["signal"],
                    "Accuracy": round(pred["accuracy"], 2),
                    "TP": round(pred["tp"], 5) if pred["tp"] else None,
                    "SL": round(pred["sl"], 5) if pred["sl"] else None,
                    "WinRate": round(back["win_rate"], 2) if back else 0.0,
                    "Expectancy(%)": round(back["expectancy"], 3) if back else 0.0,
                    "Trades": int(back["n_trades"]) if back else 0,
                }
            )
        except Exception as e:
            print(f"⚠️ Error processing {asset}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    return df_out.sort_values(by=["Accuracy", "WinRate", "Expectancy(%)"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# New summary-tab helpers (explicit, for one-page view)
# --------------------------------------------------------------------------------------
def get_model_signal(df: pd.DataFrame):
    """
    Return current model recommendation, confidence, and TP/SL from MACD + ATR.
    """
    df = add_indicators(df)
    if df.empty or "macd" not in df or "signal" not in df or "Close" not in df:
        return "Hold", 0.0, None, None

    macd_val = float(df["macd"].iloc[-1])
    sig_val = float(df["signal"].iloc[-1])
    close = float(df["Close"].iloc[-1])
    diff = macd_val - sig_val
    confidence = min(abs(diff) * 100, 100)

    if diff > 0:
        direction = "Buy"
    elif diff < 0:
        direction = "Sell"
    else:
        direction = "Hold"

    atr = float(df["atr"].iloc[-1]) if "atr" in df else np.nan
    tp, sl = compute_tp_sl(close, atr, direction, "Medium", tp_mult=2.0)
    return direction, confidence, tp, sl


def calculate_model_performance(df: pd.DataFrame):
    """
    Backtest MACD strategy for win rate, expectancy and equity curve (for summary tab).
    """
    df = add_indicators(df)
    out = {"win_rate": 0.0, "avg_return": 0.0, "equity_curve": None}
    if df.empty or "macd" not in df or "signal" not in df:
        return out

    df["signal_num"] = 0
    df.loc[df["macd"] > df["signal"], "signal_num"] = 1
    df.loc[df["macd"] < df["signal"], "signal_num"] = -1
    df["future_ret"] = df["Close"].pct_change().shift(-1)
    df["strat_ret"] = df["signal_num"] * df["future_ret"]
    rets = df["strat_ret"].dropna()
    if rets.empty:
        return out

    out["win_rate"] = float((rets > 0).mean() * 100)
    out["avg_return"] = float(rets.mean() * 100)
    out["equity_curve"] = (1 + rets.fillna(0)).cumprod()
    return out


# --------------------------------------------------------------------------------------
# Convenience: fetch one fully-prepped dataset for a tab
# --------------------------------------------------------------------------------------
def get_prepared_data(symbol: str, interval_key: str) -> pd.DataFrame:
    df = fetch_data(symbol, interval_key=interval_key)
    return add_indicators(df)


# --------------------------------------------------------------------------------------
# Quick CSV export utility (optional use in tabs)
# --------------------------------------------------------------------------------------
def export_dataframe_csv(df: pd.DataFrame, filename: str) -> str:
    """
    Save DataFrame to /data and return path (for Streamlit download button).
    """
    if df is None or df.empty:
        return ""
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=True)
    return path


# --------------------------------------------------------------------------------------
# Diagnostics (kept for visibility)
# --------------------------------------------------------------------------------------
def diagnostics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "has_ohlc": False, "has_indicators": False}
    has_ind = all(col in df.columns for col in ["RSI", "EMA_20", "EMA_50", "MACD", "Signal", "ATR_14"])
    return {"rows": len(df), "has_ohlc": _has_ohlc(df), "has_indicators": has_ind}