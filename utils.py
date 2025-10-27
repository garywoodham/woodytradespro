"""
utils.py
Master utility module for WoodyTradesPro / Forecast dashboard.

High-level goals:
- Fetch & cache market data (yfinance with rate-limit protection)
- Generate indicators & trading signals
- Build trades list with TP / SL / RE (Reason) columns
- Run backtest / analytics (win rate, max drawdown, etc.)
- Prep formatted outputs for Streamlit (metrics row, trade table)
- Risk & position sizing helpers
- Forecast / model hook surface
- Logging / diagnostics helpers

This file is intentionally written as a single importable module so `app.py`
can simply `import utils`. The code avoids runtime placeholders like `pass`
so importing will not explode.

If you already have an 1800-line utils.py in your repo:
- Keep it safe. Do NOT delete it.
- Diff that file against this file.
- Merge anything unique (for example alert systems, broker integration,
  ML predictions, Discord/Telegram push, multi-timeframe logic, etc.)
into the clearly marked extension sections below.

Author: Forecast / WoodyTradesPro
"""

########################################
# 0. Standard library imports
########################################

import os
import time
import math
import json
import pickle
import random
import statistics
import traceback
import datetime as dt
from typing import Any, Dict, Optional, Tuple, List

########################################
# 1. Third-party imports
########################################

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError(
        "yfinance is required. Install with `pip install yfinance`."
    ) from e


########################################
# 2. Globals / constants
########################################

# Cache directory for downloaded market data
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Default pull config
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# YFinance throttle handling
YF_MAX_RETRIES = 5
YF_BACKOFF_SECONDS = 2  # exponential backoff base

# Risk defaults
DEFAULT_TRADE_RISK_PCT = 1.0  # risk 1% equity per trade
DEFAULT_STOP_MULT = 0.98      # -2%
DEFAULT_TAKE_MULT = 1.02      # +2%

# For reproducibility where randomness is used (e.g. Monte Carlo)
RNG = random.Random(1337)


########################################
# 3. Lightweight logging helpers
########################################

def _now_utc() -> dt.datetime:
    """Return naive UTC timestamp (no tzinfo) for consistent logging."""
    return dt.datetime.utcnow().replace(tzinfo=None)


def log_info(msg: str) -> None:
    """Standard info log."""
    ts = _now_utc().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {ts}] {msg}")


def log_warn(msg: str) -> None:
    """Warning log."""
    ts = _now_utc().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WARN {ts}] {msg}")


def log_error(msg: str) -> None:
    """Error log (non-fatal)."""
    ts = _now_utc().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR {ts}] {msg}")


def format_exception(e: Exception) -> str:
    """Return a short exception string with class + message."""
    return f"{e.__class__.__name__}: {e}"


########################################
# 4. Cache utilities
########################################

def _cache_path(ticker: str, interval: str, period: str) -> str:
    """
    Build a deterministic filename for caching a ticker's candles.
    """
    safe_ticker = ticker.replace("/", "_").replace("=", "_").replace("^", "")
    fname = f"{safe_ticker}_{interval}_{period}.pkl"
    return os.path.join(DATA_CACHE_DIR, fname)


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    """
    Load cached dataframe if available and not empty.
    Returns None otherwise.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
    except Exception as e:
        log_warn(f"Cache read failed {path}: {format_exception(e)}")
        return None

    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()

    return None


def _save_cache(path: str, df: pd.DataFrame) -> None:
    """
    Save dataframe to pickle cache. Failure to cache should not kill the app.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
        log_info(f"Cached {len(df)} rows to {os.path.basename(path)}")
    except Exception as e:
        log_warn(f"Cache write failed {path}: {format_exception(e)}")


########################################
# 5. Market data download (with rate limit defense)
########################################

def _normalize_yf_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output into our canonical schema:
    datetime, open, high, low, close, adj_close, volume

    We also enforce UTC-naive timestamps sorted ascending.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    df = raw.reset_index().copy()

    rename_map = {
        "Date": "datetime",
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
    }

    df.rename(columns=rename_map, inplace=True)

    # ensure datetime column exists
    if "datetime" not in df.columns:
        # yfinance sometimes calls it just "index" on weird intervals
        if df.columns[0].lower() in ("date", "datetime"):
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
        else:
            raise RuntimeError("No datetime column found in yfinance data")

    # Convert to UTC-naive timestamps
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    # Sort ascending, reset index
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # If adj_close missing, fall back to close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # If volume missing (e.g. FX), fill with 0
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ].copy()


def safe_download_yf(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV from yfinance with:
    - disk cache
    - retry + exponential backoff
    - graceful fallback to cache if throttled

    This is the function that produced logs like:
    "✅ Using cached BTC-USD (1465 rows)."
    """
    cache_file = _cache_path(ticker, interval, period)

    # 1. try cache first
    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            log_info(f"✅ Using cached {ticker} ({len(cached)} rows).")
            return cached

    # 2. attempt live download with retry
    last_err: Optional[Exception] = None
    for attempt in range(YF_MAX_RETRIES):
        try:
            raw = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,  # threads False to avoid burst-y calls
            )

            if raw is not None and not raw.empty:
                df = _normalize_yf_df(raw)
                _save_cache(cache_file, df)
                log_info(
                    f"⬇️ Fresh download {ticker} ok "
                    f"({len(df)} rows). Cached."
                )
                return df

            last_err = RuntimeError(
                f"Empty dataframe from yfinance for {ticker} "
                f"(attempt {attempt+1}/{YF_MAX_RETRIES})"
            )

        except Exception as e:
            last_err = e
            log_warn(
                f"⚠️ Rate-limit / fetch issue for {ticker} try {attempt+1}/"
                f"{YF_MAX_RETRIES}: {format_exception(e)}"
            )

        # backoff
        sleep_time = YF_BACKOFF_SECONDS * (2 ** attempt)
        log_info(f"Backing off {sleep_time}s before retry for {ticker}")
        time.sleep(sleep_time)

    # 3. fallback to cache
    cached = _load_cache(cache_file)
    if cached is not None:
        log_warn(
            f"🚧 Using cached {ticker} ({len(cached)} rows) after "
            f"repeated rate-limit failures."
        )
        return cached

    # 4. total failure
    raise RuntimeError(
        f"Failed to download {ticker} and no cache available. "
        f"Last error: {format_exception(last_err) if last_err else 'unknown'}"
    )


########################################
# 6. Indicator calculations
########################################

def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualized rolling volatility estimate using stdev * sqrt(252).
    """
    vol = returns.rolling(window).std(ddof=0) * math.sqrt(252)
    return vol


def sma(series: pd.Series, length: int) -> pd.Series:
    """
    Simple moving average.
    """
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential moving average.
    """
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    Standard Wilder's smoothing.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all the indicator columns that downstream logic needs.
    Includes:
    - ret (pct change)
    - vol20 (20d vol annualized)
    - sma_fast / sma_slow (10 / 50 for crossover)
    - ema_fast / ema_slow (optional alt model)
    - rsi14
    - signal_long basic (crossover)
    """
    out = df.copy()

    # returns
    out["ret"] = out["close"].pct_change()

    # volatility
    out["vol20"] = rolling_volatility(out["ret"], window=20)

    # moving averages
    out["sma_fast"] = sma(out["close"], 10)
    out["sma_slow"] = sma(out["close"], 50)

    out["ema_fast"] = ema(out["close"], 10)
    out["ema_slow"] = ema(out["close"], 50)

    # RSI
    out["rsi14"] = rsi(out["close"], 14)

    # primary signal: long when fast MA > slow MA
    out["signal_long"] = (out["sma_fast"] > out["sma_slow"]).astype(int)

    # alt signal placeholder but still valid numeric:
    # e.g. oversold bounce: long if RSI < 30
    out["signal_rsi_long"] = (out["rsi14"] < 30).astype(int)

    return out


########################################
# 7. Position sizing / risk helpers
########################################

def calc_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    max_risk_pct: float = DEFAULT_TRADE_RISK_PCT,
) -> float:
    """
    Naive position sizing:
    - risk at most max_risk_pct% of equity between entry and stop.
    Returns *units* (position size in base asset).

    If stop == entry, fallback to 0 to avoid div0.
    """
    if entry_price is None or stop_price is None:
        return 0.0

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return 0.0

    max_risk_cash = equity * (max_risk_pct / 100.0)
    units = max_risk_cash / risk_per_unit
    return max(units, 0.0)


def annotate_position_size_on_trades(
    trades_df: pd.DataFrame,
    starting_equity: float = 1_000.0,
    max_risk_pct: float = DEFAULT_TRADE_RISK_PCT,
) -> pd.DataFrame:
    """
    For each trade row, compute theoretical position size based on SL.
    This does NOT affect backtest PnL in run_backtest(); it's UI/info only.
    """
    if trades_df.empty:
        trades_df["position_size_units"] = []
        return trades_df

    out = trades_df.copy()

    pos_sizes = []
    for _, row in out.iterrows():
        entry_price = row.get("entry_price", np.nan)
        stop_price = row.get("sl", np.nan)
        if pd.isna(entry_price) or pd.isna(stop_price):
            pos_sizes.append(np.nan)
        else:
            units = calc_position_size(
                equity=starting_equity,
                entry_price=float(entry_price),
                stop_price=float(stop_price),
                max_risk_pct=max_risk_pct,
            )
            pos_sizes.append(units)

    out["position_size_units"] = pos_sizes
    return out


########################################
# 8. Trade generation
########################################

def generate_trade_entries(
    df: pd.DataFrame,
    tp_mult: float = DEFAULT_TAKE_MULT,
    sl_mult: float = DEFAULT_STOP_MULT,
    use_signal: str = "signal_long",
) -> pd.DataFrame:
    """
    Build trades from a binary signal column (0/1).

    Logic:
    - Enter LONG when signal flips 0 -> 1.
    - Exit LONG when signal flips 1 -> 0.
    - For each entry we record TP, SL, RE (reason).
    - Open trades remain with NaN exit.

    Columns:
    datetime, side, entry_price, tp, sl, exit_price,
    exit_datetime, pnl_pct, reason
    """

    if use_signal not in df.columns:
        raise KeyError(
            f"Requested signal '{use_signal}' is not in dataframe columns."
        )

    data = df.copy().reset_index(drop=True)

    trades: List[Dict[str, Any]] = []
    in_trade = False
    entry_price = None
    entry_ts = None

    for i in range(1, len(data)):
        prev_sig = data.loc[i - 1, use_signal]
        curr_sig = data.loc[i, use_signal]

        # ENTRY condition
        if (not in_trade) and (prev_sig == 0) and (curr_sig == 1):
            entry_price = data.loc[i, "close"]
            entry_ts = data.loc[i, "datetime"]
            in_trade = True

            trades.append(
                {
                    "datetime": entry_ts,
                    "side": "LONG",
                    "entry_price": float(entry_price),
                    "tp": float(entry_price * tp_mult),
                    "sl": float(entry_price * sl_mult),
                    "exit_price": np.nan,
                    "exit_datetime": pd.NaT,
                    "pnl_pct": np.nan,
                    "reason": "signal_flip_long",
                }
            )

        # EXIT condition
        if in_trade and (prev_sig == 1) and (curr_sig == 0):
            exit_price = data.loc[i, "close"]
            exit_ts = data.loc[i, "datetime"]

            # tie off last open trade
            trades[-1]["exit_price"] = float(exit_price)
            trades[-1]["exit_datetime"] = exit_ts
            trades[-1]["pnl_pct"] = (
                (exit_price / trades[-1]["entry_price"]) - 1.0
            ) * 100.0
            trades[-1]["reason"] = "signal_flip_exit"

            in_trade = False
            entry_price = None
            entry_ts = None

    # If weekend/market closed => open trade stays open (NaN exit fields).
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        # create empty with consistent columns so Streamlit won't crash
        trades_df = pd.DataFrame(
            columns=[
                "datetime",
                "side",
                "entry_price",
                "tp",
                "sl",
                "exit_price",
                "exit_datetime",
                "pnl_pct",
                "reason",
            ]
        )

    return trades_df


########################################
# 9. Backtest / performance analytics
########################################

def equity_curve_from_signal(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    Given a dataframe with 'ret' and some binary position signal,
    build strategy equity curve assuming:
    - position = signal (long 1, flat 0)
    - no transaction costs
    """
    out = df.copy()

    if "ret" not in out.columns:
        out["ret"] = out["close"].pct_change()

    out["position"] = out[signal_col].fillna(0.0)

    # strategy_ret uses previous day's position
    out["strategy_ret"] = out["position"].shift(1, fill_value=0) * out["ret"]

    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()

    # peak and drawdown
    out["peak_equity"] = out["equity"].cummax()
    out["dd"] = out["equity"] / out["peak_equity"] - 1.0

    return out


def summarize_performance(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Produce summary stats:
    - final_balance
    - total_return_pct
    - max_drawdown_pct
    - num_trades
    - wins
    - winrate_pct
    """
    if equity_df.empty:
        final_balance = 1.0
        total_return_pct = 0.0
        max_dd_pct = 0.0
    else:
        final_balance = float(equity_df["equity"].iloc[-1])
        total_return_pct = (final_balance - 1.0) * 100.0
        max_dd_pct = float(equity_df["dd"].min() * 100.0)

    closed_trades = trades_df.dropna(subset=["exit_price"]).copy()
    num_trades = int(len(closed_trades))

    if num_trades > 0:
        wins = int((closed_trades["pnl_pct"] > 0).sum())
        winrate_pct = (wins / num_trades) * 100.0
    else:
        wins = 0
        winrate_pct = 0.0

    return {
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "num_trades": num_trades,
        "wins": wins,
        "winrate_pct": winrate_pct,
    }


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal_long",
) -> Dict[str, Any]:
    """
    High-level backtest wrapper.
    Steps:
      1. equity curve from that signal
      2. generate trades off that signal
      3. summarize performance
    """
    equity_df = equity_curve_from_signal(df, signal_col=signal_col)
    trades_df = generate_trade_entries(df, use_signal=signal_col)
    summary = summarize_performance(equity_df, trades_df)
    return summary


########################################
# 10. Forecast / model interface
########################################

def naive_forecast_next_close(df: pd.DataFrame, lookback: int = 5) -> Optional[float]:
    """
    Super dumb forecast example:
    Predict next close = simple average of last N closes.
    This is here because larger utils files usually carry a forecast hook.
    """
    if df.empty:
        return None

    recent = df["close"].tail(lookback)
    if recent.isna().all():
        return None

    return float(recent.mean())


def attach_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 1-step-ahead naive forecast column to df (shifted so we don't leak).
    """
    out = df.copy()
    forecast_val = naive_forecast_next_close(out)
    out["forecast_next_close"] = np.nan
    if forecast_val is not None and len(out) > 0:
        out.loc[out.index[-1], "forecast_next_close"] = forecast_val
    return out


########################################
# 11. Dashboard context builders
########################################

def load_market_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Pull candles for ticker and compute indicators.
    """
    df = safe_download_yf(ticker, period=period, interval=interval, use_cache=True)
    df = compute_indicators(df)
    df = attach_forecast(df)
    return df


def build_dashboard_context(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    signal_col: str = "signal_long",
) -> Dict[str, Any]:
    """
    High-level bundle the Streamlit app consumes.
    Includes:
    - latest price/time
    - df with indicators & forecast
    - trades_df with TP/SL/RE
    - summary stats dict
    """
    df = load_market_data(ticker, period=period, interval=interval)

    trades_df = generate_trade_entries(df, use_signal=signal_col)
    trades_df = annotate_position_size_on_trades(trades_df)

    summary = run_backtest(df, signal_col=signal_col)

    latest_row = df.iloc[-1] if len(df) else None
    latest_price = float(latest_row["close"]) if latest_row is not None else None
    latest_time = (
        pd.to_datetime(latest_row["datetime"]).strftime("%Y-%m-%d %H:%M:%S")
        if latest_row is not None
        else None
    )

    ctx: Dict[str, Any] = {
        "ticker": ticker,
        "latest_price": latest_price,
        "latest_time": latest_time,
        "df": df,
        "trades_df": trades_df,
        "summary": summary,
    }

    return ctx


########################################
# 12. Formatting helpers for Streamlit UI
########################################

def format_trades_for_display(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw trades_df into pretty table columns for the dashboard:
    Date | Side | Entry | TP | SL | Exit | Exit Date | PnL % | Reason | Size (units)

    Handles open trades (exit fields NaN).
    """
    if trades_df.empty:
        # return empty with headers so Streamlit shows table shape
        cols = [
            "Date",
            "Side",
            "Entry",
            "TP",
            "SL",
            "Exit",
            "Exit Date",
            "PnL %",
            "Reason",
            "Size (units)",
        ]
        return pd.DataFrame(columns=cols)

    out = trades_df.copy()

    out["Date"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out["Side"] = out["side"]
    out["Entry"] = out["entry_price"].round(2)
    out["TP"] = out["tp"].round(2)
    out["SL"] = out["sl"].round(2)
    out["Exit"] = out["exit_price"].round(2)
    out["Exit Date"] = pd.to_datetime(out["exit_datetime"]).dt.strftime("%Y-%m-%d")
    out["PnL %"] = out["pnl_pct"].round(2)
    out["Reason"] = out["reason"]

    # Optional size column from annotate_position_size_on_trades
    if "position_size_units" in out.columns:
        out["Size (units)"] = out["position_size_units"].round(4)
    else:
        out["Size (units)"] = np.nan

    display_cols = [
        "Date",
        "Side",
        "Entry",
        "TP",
        "SL",
        "Exit",
        "Exit Date",
        "PnL %",
        "Reason",
        "Size (units)",
    ]

    return out[display_cols]


def format_summary_for_display(summary: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert numeric summary stats to UI-friendly strings.
    Matches console print style you saw earlier:
    trades=0, wins=0, balance=1.0000, winrate=0.0%, return=0.0%, maxdd=0.0%
    """
    return {
        "Final Balance": f"{summary['final_balance']:.4f}",
        "Total Return": f"{summary['total_return_pct']:.2f}%",
        "Win Rate": f"{summary['winrate_pct']:.2f}%",
        "Trades": f"{summary['num_trades']}",
        "Wins": f"{summary['wins']}",
        "Max Drawdown": f"{summary['max_drawdown_pct']:.2f}%",
    }


########################################
# 13. Diagnostic / export helpers
########################################

def export_trades_json(trades_df: pd.DataFrame) -> str:
    """
    Export trade history (including open trades) to JSON string.
    Safe to write to disk or send to UI download.
    """
    safe_df = trades_df.copy()
    # convert datetimes to isoformat for JSON safety
    for col in ["datetime", "exit_datetime"]:
        if col in safe_df.columns:
            safe_df[col] = pd.to_datetime(safe_df[col]).astype(str)

    return safe_df.to_json(orient="records")


def export_summary_json(summary: Dict[str, Any]) -> str:
    """
    Export summary dict to JSON string.
    """
    return json.dumps(summary, indent=2, default=str)


def slice_recent_data(df: pd.DataFrame, bars: int = 50) -> pd.DataFrame:
    """
    Convenience for debug view: last N candles + indicators.
    """
    return df.tail(bars).reset_index(drop=True)


########################################
# 14. Safety checks for Streamlit / production runtime
########################################

def sanity_check_dataframe(df: pd.DataFrame) -> None:
    """
    Basic smoke tests to catch corrupted data before plotting.
    Raises RuntimeError if something critical is missing.
    """
    required_cols = ["datetime", "open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Dataframe missing required column '{col}'")

    if df["close"].isna().all():
        raise RuntimeError("All close values NaN, data looks invalid")

    if not df["datetime"].is_monotonic_increasing:
        raise RuntimeError("datetime column is not sorted ascending")


def describe_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight dataset description to help debugging in UI or logs.
    Returns counts, min/max date, etc.
    """
    if df.empty:
        return {
            "rows": 0,
            "start": None,
            "end": None,
            "has_na": True,
        }

    start_ts = pd.to_datetime(df["datetime"].iloc[0])
    end_ts = pd.to_datetime(df["datetime"].iloc[-1])
    has_na = df.isna().any().any()

    return {
        "rows": int(len(df)),
            "start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "has_na": bool(has_na),
    }


########################################
# 15. Public API of this module
########################################
# Everything below is what app.py or other modules are expected to call.

__all__ = [
    # logging
    "log_info",
    "log_warn",
    "log_error",
    "format_exception",

    # data loading / caching
    "safe_download_yf",
    "load_market_data",
    "build_dashboard_context",

    # indicators / signals
    "compute_indicators",
    "rolling_volatility",
    "sma",
    "ema",
    "rsi",

    # trades / backtest
    "generate_trade_entries",
    "annotate_position_size_on_trades",
    "equity_curve_from_signal",
    "summarize_performance",
    "run_backtest",

    # forecast
    "naive_forecast_next_close",
    "attach_forecast",

    # formatting for UI
    "format_trades_for_display",
    "format_summary_for_display",
    "slice_recent_data",
    "describe_dataset",

    # export helpers
    "export_trades_json",
    "export_summary_json",

    # sanity checks
    "sanity_check_dataframe",
]