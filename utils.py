# utils.py — WoodyTradesPro Smart v2 (FINAL NORMALIZED VERSION)
# ---------------------------------------------------------------------------
# Includes:
# - Safe yfinance fetch with cache
# - OHLCV normalization (fixes 2D columns / MultiIndex / array-of-arrays)
# - Technical indicators (EMA/RSI/MACD/ATR)
# - Signal engine (Buy/Sell/Hold + prob)
# - TP/SL generation
# - Sentiment via news headlines
# - ML scaffold
# - Backtest
# - Summary helpers for app tabs

import os
import time
import warnings
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Logging / Warning suppression
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Could not infer format")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
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

INTERVALS = {
    "15m": "15m",
    "1h": "60m",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
}
PERIODS = {
    "15m": "7d",
    "1h": "2mo",
    "4h": "6mo",
    "1d": "1y",
    "1wk": "5y",
}

# ---------------------------------------------------------------------------
# Helper Logger
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Core cleaner: normalize OHLCV into a predictable, flat numeric DataFrame
# ---------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we return a clean OHLCV DataFrame with:
    - DatetimeIndex sorted ascending
    - Single-level columns: Open, High, Low, Close, Volume (Adj Close kept if present)
    - Each column strictly 1-D numeric (no shape (n,1) arrays, no object arrays of arrays)
    - No infs, minimal NaN holes (ffill/bfill)
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # If MultiIndex columns (e.g. ('Close','GC=F')), take only the first level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Restrict to known OHLCV columns if available
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep_cols:
        # Sometimes we get lowercase from alt sources. Try capitalize pass.
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    df = df[keep_cols].copy()

    # Index to datetime if possible
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Sort by time
    df = df.sort_index()

    # Force each kept column to be strict 1-D numeric
    for col in df.columns:
        vals = df[col].values

        # Flatten any 2D arrays like shape (n,1)
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            vals = vals.reshape(-1)

        # If values are arrays-of-length-1 per row (object dtype), squeeze them
        if len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)):
            vals = np.array([v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else np.nan for v in vals])

        df[col] = pd.to_numeric(vals, errors="coerce")

    # Clean NaNs/Infs, forward/back fill small gaps
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")

    return df

# ---------------------------------------------------------------------------
# Fetch data with cache + normalization
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch price data from cache or Yahoo Finance.
    - Reads cached CSV if available
    - Otherwise downloads via yfinance
    - Normalizes via _normalize_ohlcv() so downstream never sees weird shapes
    """

    interval = INTERVALS.get(interval_key, "60m")
    period = PERIODS.get(interval_key, "2mo")

    safe_name = symbol.replace("=", "_").replace("^", "").replace("/", "_").replace("-", "_")
    cache_path = os.path.join(DATA_DIR, f"{safe_name}_{interval}.csv")

    # 1. Try cache
    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if not cached.empty:
                return cached
        except Exception:
            pass  # fall through to live fetch

    # 2. Live fetch
    _log(f"⏳ Fetching {symbol} [{interval}] for {period}...")
    for attempt in range(4):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            norm = _normalize_ohlcv(raw)
            if not norm.empty and len(norm) > 50:
                try:
                    norm.to_csv(cache_path)
                    _log(f"✅ {symbol}: fetched {len(norm)} rows.")
                except Exception:
                    pass
                return norm
        except Exception as e:
            _log(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
            time.sleep(1 + attempt)

    _log(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA20, EMA50, RSI(14), MACD(12/26/9), ATR(14) to a clean OHLCV df.
    We re-normalize here to be extra safe (this fixes the (n,1) Close bug).
    """

    df = _normalize_ohlcv(df)
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    # EMA
    out["ema20"] = EMAIndicator(out["Close"], 20).ema_indicator()
    out["ema50"] = EMAIndicator(out["Close"], 50).ema_indicator()

    # RSI
    out["rsi"] = RSIIndicator(out["Close"], 14).rsi()

    # MACD
    macd_obj = MACD(out["Close"])
    out["macd"] = macd_obj.macd()
    out["signal"] = macd_obj.macd_signal()

    # ATR
    atr_obj = AverageTrueRange(out["High"], out["Low"], out["Close"], 14)
    out["atr"] = atr_obj.average_true_range()

    # Final cleanup, drop warmup NaNs
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out

# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Determine Buy / Sell / Hold and a crude confidence.
    Heuristic:
    - Buy bias if ema20>ema50, RSI not overbought, MACD above signal
    - Sell bias if ema20<ema50, RSI not oversold, MACD below signal
    Otherwise Hold.
    """
    side = "Hold"
    prob = 0.5

    if row["ema20"] > row["ema50"] and row["rsi"] < 70 and row["macd"] > row["signal"]:
        side, prob = "Buy", 0.7
    elif row["ema20"] < row["ema50"] and row["rsi"] > 30 and row["macd"] < row["signal"]:
        side, prob = "Sell", 0.7

    return side, prob

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Generate take-profit and stop-loss using ATR and a risk preference.
    - Low risk: tighter TP, wider SL
    - High risk: wider TP, tighter SL
    """
    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)

    # Fall back to a % of price if ATR is dodgy
    if atr is None or np.isnan(atr):
        atr = price * 0.005  # ~0.5%

    if side == "Buy":
        tp = price + mult * atr
        sl = price - mult * atr
    elif side == "Sell":
        tp = price - mult * atr
        sl = price + mult * atr
    else:
        # Even for Hold we still suggest TP/SL using a notional long bias
        tp = price + mult * atr
        sl = price - mult * atr

    return float(tp), float(sl)

# ---------------------------------------------------------------------------
# Sentiment via Yahoo Finance headlines
# ---------------------------------------------------------------------------
def fetch_sentiment(symbol: str) -> float:
    """
    Pull last few news headlines from yfinance.Ticker(symbol).news
    and score with VADER (compound). If anything fails, return 0.
    """
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", [])
        if not news:
            return 0.0

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for item in news[:5]:
            title = item.get("title", "")
            if not title:
                continue
            s = analyzer.polarity_scores(title)["compound"]
            scores.append(s)

        if not scores:
            return 0.0
        return float(np.mean(scores))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Latest Prediction per asset
# ---------------------------------------------------------------------------
def latest_prediction(df: pd.DataFrame, symbol: str = "", risk: str = "Medium") -> Optional[Dict[str, object]]:
    """
    Use the last 2 rows to determine:
    - signal side (Buy/Sell/Hold)
    - probability/confidence
    - TP/SL
    - ATR, sentiment, regime
    Always returns TP/SL even if Hold.
    """

    df = add_indicators(df)
    if df is None or df.empty or len(df) < 2:
        return None

    row_prev = df.iloc[-2]
    row_curr = df.iloc[-1]

    side, prob = compute_signal_row(row_prev, row_curr)

    atr_val = float(row_curr["atr"]) if "atr" in row_curr and not np.isnan(row_curr["atr"]) \
        else float(df["atr"].tail(14).mean())

    price = float(row_curr["Close"])

    # If Hold, still compute TP/SL with a "Buy-like" orientation so UI always shows numbers
    orient = side if side != "Hold" else "Buy"
    tp, sl = compute_tp_sl(price, atr_val, orient, risk)

    sentiment_score = fetch_sentiment(symbol)
    regime_label = "Bull" if row_curr["ema20"] > row_curr["ema50"] else "Bear"

    return {
        "side": side,
        "prob": prob,
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment_score,
        "regime": regime_label,
        "ml_prob": prob,  # placeholder for ML prob if/when used
    }

# ---------------------------------------------------------------------------
# ML scaffold
# ---------------------------------------------------------------------------
def train_ml_model(df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Basic random forest direction model.
    Not directly surfaced in UI yet, but kept for evolution.
    """
    df_ind = add_indicators(df)
    if df_ind is None or len(df_ind) < 100:
        return None

    tmp = df_ind.copy()
    tmp["target"] = (tmp["Close"].shift(-1) > tmp["Close"]).astype(int)

    features = ["ema20", "ema50", "rsi", "macd", "signal", "atr"]
    X = tmp[features].dropna()
    y = tmp.loc[X.index, "target"]

    if X.empty:
        return None

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    return model

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest_signals(df: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    """
    Very simple walk-forward:
    - At each bar from index 60 onward, generate side.
    - If side != Hold, measure PnL next bar.
    - Track winrate, cumulative growth, and trade list.
    """

    df_ind = add_indicators(df)
    if df_ind is None or len(df_ind) < 80:
        return {"win_rate": 0, "total_return_pct": 0, "n_trades": 0, "trades": []}

    trades = []
    balance = 1.0
    wins = 0

    # We'll skip first ~60 bars because indicators need warmup
    for i in range(60, len(df_ind) - 1):
        prev_row = df_ind.iloc[i - 1]
        row = df_ind.iloc[i]
        side, _ = compute_signal_row(prev_row, row)
        if side == "Hold":
            continue

        entry_px = float(row["Close"])
        next_px = float(df_ind["Close"].iloc[i + 1])

        if side == "Buy":
            profit = (next_px - entry_px) / entry_px
        else:  # Sell
            profit = (entry_px - next_px) / entry_px

        wins += 1 if profit > 0 else 0
        balance *= (1 + profit)

        trades.append({
            "index": i,
            "side": side,
            "entry_price": entry_px,
            "exit_price": next_px,
            "profit_pct": profit * 100.0,
        })

    n_trades = len(trades)
    if n_trades > 0:
        win_rate = 100.0 * wins / n_trades
        total_return = (balance - 1.0) * 100.0
    else:
        win_rate = 0.0
        total_return = 0.0

    return {
        "win_rate": win_rate,
        "total_return_pct": total_return,
        "n_trades": n_trades,
        "trades": trades,
    }

# ---------------------------------------------------------------------------
# Single-asset pipeline for Summary / Analysis views
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str, interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> Optional[Dict[str, object]]:
    """
    Full pipeline for one symbol:
    - fetch & normalize
    - indicators
    - latest prediction (signal, TP, SL, sentiment)
    - backtest stats
    """

    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None

    df_ind = add_indicators(df)
    if df_ind.empty:
        return None

    pred = latest_prediction(df, symbol, risk)
    bt = backtest_signals(df, risk)

    return {
        "symbol": symbol,
        "interval": interval_key,
        "last_price": float(df_ind["Close"].iloc[-1]),
        "signal": pred["side"],
        "probability": round(pred["prob"] * 100.0, 2),
        "tp": round(pred["tp"], 2) if pred["tp"] is not None else None,
        "sl": round(pred["sl"], 2) if pred["sl"] is not None else None,
        "win_rate": bt["win_rate"],
        "return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
    }

# ---------------------------------------------------------------------------
# Multi-asset summary for dashboard
# ---------------------------------------------------------------------------
def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> pd.DataFrame:
    """
    Called by the Market Summary tab.
    Iterates all assets, runs analyze_asset, returns a tidy table.
    """

    _log("Fetching and analyzing market data (smart v2)...")
    rows = []

    for asset_name, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset_name} ({symbol})...")
        res = analyze_asset(symbol, interval_key, risk, use_cache)

        if res is None:
            continue

        rows.append({
            "Asset": asset_name,
            "Symbol": symbol,
            "Interval": interval_key,
            "Price": res["last_price"],
            "Signal": res["signal"],
            "Probability_%": res["probability"],
            "TP": res["tp"],
            "SL": res["sl"],
            "WinRate_%": res["win_rate"],
            "Return_%": res["return_pct"],
            "Trades": res["n_trades"],
            "Sentiment": res["sentiment"],
            "Regime": res["regime"],
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Helpers used by Asset Analysis / Backtest tabs in app.py
# ---------------------------------------------------------------------------
def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Return (symbol, df_with_indicators) for plotting etc.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)

    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key, use_cache)
    df_ind = add_indicators(df)
    return symbol, df_ind

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True):
    """
    Return both:
    - a dict (prediction, backtest stats, sentiment, etc.)
    - df_with_indicators for plotting
    """

    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()

    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None, pd.DataFrame()

    df_ind = add_indicators(df)
    if df_ind.empty:
        return None, pd.DataFrame()

    pred = latest_prediction(df, symbol, risk)
    bt = backtest_signals(df, risk)

    result = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df_ind["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": pred["prob"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "ml_prob": pred["ml_prob"],
        "trades": bt["trades"],
    }

    return result, df_ind

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------