# utils.py — WoodyTradesPro Smart v3 (Full, Quiet, Backward Compatible)
# ---------------------------------------------------------------------------
# This module provides:
# - Safe, normalized OHLCV fetching with cache
# - Technical indicators (EMA/RSI/MACD/ATR/Bollinger/ADX/ROC/etc.)
# - Sentiment scoring from yfinance headlines
# - Rule-based signal engine (Buy/Sell/Hold)
# - ML model for directional probability
# - Blended probability: rule + ML + sentiment
# - TP/SL suggestions based on ATR and risk profile
# - Backtesting with Sharpe and drawdown
# - Summary helpers for Streamlit tabs
#
# IMPORTANT:
# - Keeps all functions expected by app.py:
#       summarize_assets
#       analyze_asset
#       load_asset_with_indicators
#       asset_prediction_and_backtest
#
# - Keeps normalize logic to avoid (n,1) and MultiIndex column bugs.

import os
import time
import warnings
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Logging / Warning suppression (quiet mode)
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Could not infer format")

# ---------------------------------------------------------------------------
# Globals / config
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

# in-memory cache for trained ML to avoid retraining every call
_MODEL_CACHE: Dict[Tuple[str, str], RandomForestClassifier] = {}

# ---------------------------------------------------------------------------
# Helper Logger (quiet but still prints critical events)
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Core cleaner: normalize OHLCV into predictable, flat numeric DataFrame
# ---------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure clean OHLCV:
    - DatetimeIndex ascending
    - Flatten MultiIndex columns
    - Keep only Open, High, Low, Close, Adj Close, Volume
    - Force each to 1-D numeric (no ndarray-of-length-1 cells)
    - Fill small gaps
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # If yfinance returns MultiIndex columns like ('Close','GC=F'), flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # standard finance columns, some assets may miss Volume, etc.
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep_cols:
        # fallback: try capitalizing columns
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    df = df[keep_cols].copy()

    # make index datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # sort
    df = df.sort_index()

    # flatten weird shapes, convert to numeric
    for col in df.columns:
        vals = df[col].values

        # collapse (n,1) arrays
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            vals = vals.reshape(-1)

        # collapse arrays-of-arrays per cell
        if len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)):
            new_vals = []
            for v in vals:
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    new_vals.append(v[0])
                else:
                    new_vals.append(np.nan)
            vals = np.array(new_vals)

        df[col] = pd.to_numeric(vals, errors="coerce")

    # clean inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")

    return df

# ---------------------------------------------------------------------------
# Fetch data with cache + normalization
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch price data for a symbol:
    1. Try cached CSV
    2. Else fetch from yfinance
    3. Normalize shape so downstream math never explodes
    """

    interval = INTERVALS.get(interval_key, "60m")
    period = PERIODS.get(interval_key, "2mo")

    safe_name = (
        symbol.replace("=", "_")
        .replace("^", "")
        .replace("/", "_")
        .replace("-", "_")
    )
    cache_path = os.path.join(DATA_DIR, f"{safe_name}_{interval}.csv")

    # 1. try cache
    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if not cached.empty:
                return cached
        except Exception:
            pass

    # 2. live fetch
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
            _log(f"⚠️ Fetch attempt {attempt + 1} failed for {symbol}: {e}")
            time.sleep(1 + attempt)

    _log(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# Feature Engineering / Indicators
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a full indicator panel:
    - ema20 / ema50
    - rsi14
    - macd, macd signal
    - atr14
    - bollinger bands (20,2): %B and bandwidth
    - adx14
    - roc_close: % change over 10 periods
    - roc_vol: % change in volume over 10 periods
    - ema_gap: (ema20 - ema50) / ema50
    Returns cleaned df with these columns appended, NaN warmup rows dropped.
    """

    base = _normalize_ohlcv(df)
    if base.empty:
        return pd.DataFrame()

    out = base.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    # EMA
    out["ema20"] = EMAIndicator(close, 20).ema_indicator()
    out["ema50"] = EMAIndicator(close, 50).ema_indicator()

    # RSI
    out["rsi"] = RSIIndicator(close, 14).rsi()

    # MACD
    macd_obj = MACD(close)
    out["macd"] = macd_obj.macd()
    out["signal"] = macd_obj.macd_signal()
    out["macd_hist"] = macd_obj.macd_diff()

    # ATR
    atr_obj = AverageTrueRange(high, low, close, 14)
    out["atr"] = atr_obj.average_true_range()

    # Bollinger
    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    # %B = (Close - lower) / (upper - lower)
    out["bb_percent_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    # bandwidth = (upper - lower)/mid
    out["bb_bandwidth"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    # ADX (trend strength)
    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
    out["adx"] = adx_obj.adx()

    # Rate of Change (momentum)
    out["roc_close"] = close.pct_change(10)
    if "Volume" in out.columns:
        out["roc_vol"] = out["Volume"].pct_change(10)
    else:
        out["roc_vol"] = np.nan

    # EMA gap (regime strength)
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    # Cleanup
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)

    # reset_index so downstream backtest indexing stays consistent
    out = out.reset_index(drop=True)
    return out

# ---------------------------------------------------------------------------
# Rule-Based Signal Engine
# ---------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Rule-based directional call + confidence.
    Uses classic signals:
      - EMA regime
      - RSI bounds
      - MACD alignment
      - ADX trend strength (new)
      - %B extremes (new)
    """
    side = "Hold"
    prob = 0.5

    bull = (
        row["ema20"] > row["ema50"] and
        row["macd"] > row["signal"] and
        row["rsi"] < 70 and
        row["bb_percent_b"] < 1.1 and
        row["adx"] > 15  # needs some trend strength
    )

    bear = (
        row["ema20"] < row["ema50"] and
        row["macd"] < row["signal"] and
        row["rsi"] > 30 and
        row["bb_percent_b"] > -0.1 and
        row["adx"] > 15
    )

    if bull:
        side, prob = "Buy", 0.7
    elif bear:
        side, prob = "Sell", 0.7
    else:
        side, prob = "Hold", 0.5

    # Slight momentum nudge: if roc_close is strongly positive and bull, boost confidence
    if side == "Buy" and row.get("roc_close", 0) > 0.01:
        prob = min(0.9, prob + 0.1)
    if side == "Sell" and row.get("roc_close", 0) < -0.01:
        prob = min(0.9, prob + 0.1)

    return side, prob

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Generate take-profit and stop-loss using ATR and a risk preference.
    Risk multipliers:
      Low: tigher TP/wider SL-ish (so smaller size, more protection)
      Medium: balanced
      High: looser SL / bigger TP
    """

    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)

    # ATR fallback if missing
    if atr is None or np.isnan(atr):
        atr = price * 0.005  # ~0.5%

    if side == "Buy":
        tp = price + mult * atr
        sl = price - mult * atr
    elif side == "Sell":
        tp = price - mult * atr
        sl = price + mult * atr
    else:
        # We still give a notional long-style tp/sl to show ranges in UI
        tp = price + mult * atr
        sl = price - mult * atr

    return float(tp), float(sl)

# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
def fetch_sentiment(symbol: str) -> float:
    """
    Uses yfinance.Ticker(symbol).news -> headline 'title' fields.
    VADER compound scores averaged.
    If we can't get news, return 0.
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
            score = analyzer.polarity_scores(title)["compound"]
            scores.append(score)

        if not scores:
            return 0.0
        return float(np.mean(scores))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# ML model training and probability prediction
# ---------------------------------------------------------------------------
def _prepare_ml_frame(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Build supervised data:
      target = 1 if next close > current close else 0
    """
    d = df_ind.copy()
    d["target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    return d

def _get_ml_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    ML feature set. These must all exist in df_ind.
    """
    feats = [
        "ema20", "ema50", "ema_gap",
        "rsi",
        "macd", "signal", "macd_hist",
        "atr",
        "bb_percent_b", "bb_bandwidth",
        "adx",
        "roc_close", "roc_vol",
    ]
    return df_ind[feats].copy()

def train_ml_model(symbol: str, interval_key: str, df_ind: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train (or reuse cached) RandomForest on engineered features.
    Cache key is (symbol, interval_key).
    """
    cache_key = (symbol, interval_key)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if df_ind is None or df_ind.empty or len(df_ind) < 120:
        return None

    supervised = _prepare_ml_frame(df_ind)
    X = _get_ml_features(supervised).dropna()
    y = supervised.loc[X.index, "target"]

    if X.empty or y.nunique() < 2:
        return None

    # quick train/test split so we avoid pure overfit
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    if X_train.empty or y_train.nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced",  # handle imbalance a bit
    )
    model.fit(X_train, y_train)

    _MODEL_CACHE[cache_key] = model
    return model

def predict_ml_probability(model: Optional[RandomForestClassifier], row: pd.Series) -> Optional[float]:
    """
    Given a trained model and the latest feature row,
    return probability that next bar goes UP.
    """
    if model is None:
        return None
    try:
        X_last = row.to_frame().T  # shape (1, n_features)
        proba_up = model.predict_proba(X_last)[0][1]
        return float(proba_up)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Probability fusion (rule + ML + sentiment)
# ---------------------------------------------------------------------------
def fuse_probabilities(rule_prob: float, ml_prob: Optional[float], sentiment: float) -> float:
    """
    Blend:
      - rule_prob (0..1)
      - ml_prob (0..1 or None)
      - sentiment (-1..1 from VADER)
    We'll:
      - Map sentiment to a [0.9,1.1] weight band
      - Weighted average rule+ml (60/40 if ml exists)
    """
    # sentiment scaling: mild boost if bullish (>0), mild suppress if bearish (<0)
    sent_weight = 1.0 + 0.2 * np.clip(sentiment, -1, 1)  # between ~0.8 and ~1.2
    sent_weight = float(max(0.5, min(1.5, sent_weight)))

    base_prob = rule_prob
    if ml_prob is not None:
        base_prob = 0.6 * rule_prob + 0.4 * ml_prob

    final_prob = base_prob * sent_weight

    # clamp final to [0,1]
    if final_prob < 0:
        final_prob = 0.0
    if final_prob > 1:
        final_prob = 1.0

    return float(final_prob)

# ---------------------------------------------------------------------------
# Latest Prediction (side, TP/SL, sentiment, regime, blended probability)
# ---------------------------------------------------------------------------
def latest_prediction(df_raw: pd.DataFrame, symbol: str = "", risk: str = "Medium", interval_key: str = "1h") -> Optional[Dict[str, object]]:
    """
    Take raw df:
      - add_indicators
      - run rule-based signal on last 2 rows
      - train/reuse ML model and get ML probability
      - blend with sentiment
      - compute TP/SL using ATR
    """
    df_ind = add_indicators(df_raw)
    if df_ind is None or df_ind.empty or len(df_ind) < 2:
        return None

    # last 2 rows for rule-based decision
    row_prev = df_ind.iloc[-2]
    row_curr = df_ind.iloc[-1]

    side, rule_prob = compute_signal_row(row_prev, row_curr)

    # ML prob_up (prob next candle up)
    model = train_ml_model(symbol, interval_key, df_ind)
    feat_row = _get_ml_features(df_ind).iloc[-1]
    ml_prob = predict_ml_probability(model, feat_row)

    # sentiment score (-1..1 approx)
    sentiment_score = fetch_sentiment(symbol)

    # fuse them
    final_prob = fuse_probabilities(rule_prob, ml_prob, sentiment_score)

    # compute TP/SL using ATR and risk settings
    atr_val = float(row_curr["atr"]) if not np.isnan(row_curr["atr"]) else float(df_ind["atr"].tail(14).mean())
    price = float(row_curr["Close"])

    orient = side if side != "Hold" else "Buy"
    tp, sl = compute_tp_sl(price, atr_val, orient, risk)

    # regime label
    regime_label = "Bull" if row_curr["ema20"] > row_curr["ema50"] else "Bear"

    return {
        "side": side,
        "prob": final_prob,         # blended final prob in 0..1
        "rule_prob": rule_prob,     # raw rule prob
        "ml_prob": ml_prob,         # raw ML prob_up or None
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment_score,
        "regime": regime_label,
    }

# ---------------------------------------------------------------------------
# Backtest utilities
# ---------------------------------------------------------------------------
def _compute_drawdown_stats(equity_curve: List[float]) -> Tuple[float, float]:
    """
    Given equity curve (cumulative balance over trades),
    return (max_drawdown_pct, sharpe_like)
    We'll compute:
      - max drawdown in %
      - simple Sharpe proxy = mean(pct_change)/std(pct_change) * sqrt(N)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0

    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak  # negative or zero
    max_dd_pct = float(dd.min() * 100.0)

    # daily-ish returns between steps
    rets = np.diff(eq) / eq[:-1]
    if np.std(rets) > 0:
        sharpe_like = float(np.mean(rets) / np.std(rets) * np.sqrt(len(rets)))
    else:
        sharpe_like = 0.0

    return max_dd_pct, sharpe_like

def backtest_signals(df_raw: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    """
    Walk-forward backtest:
      - for each bar i (>60), get rule-based side at i using i-1 and i
      - enter at close[i], exit at close[i+1]
      - accumulate balance
    Adds:
      - max drawdown %
      - sharpe-like ratio
    """
    df_ind = add_indicators(df_raw)
    if df_ind is None or len(df_ind) < 80:
        return {
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "n_trades": 0,
            "trades": [],
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
        }

    trades = []
    balance = 1.0
    wins = 0
    equity_curve = [balance]

    for i in range(60, len(df_ind) - 1):
        prev_row = df_ind.iloc[i - 1]
        row = df_ind.iloc[i]

        side, _ = compute_signal_row(prev_row, row)
        if side == "Hold":
            equity_curve.append(balance)
            continue

        entry_px = float(row["Close"])
        exit_px = float(df_ind["Close"].iloc[i + 1])

        if side == "Buy":
            profit = (exit_px - entry_px) / entry_px
        else:  # Sell
            profit = (entry_px - exit_px) / entry_px

        if profit > 0:
            wins += 1

        balance *= (1 + profit)
        equity_curve.append(balance)

        trades.append({
            "index": i,
            "side": side,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "profit_pct": profit * 100.0,
        })

    n_trades = len(trades)
    win_rate = 100.0 * wins / n_trades if n_trades > 0 else 0.0
    total_return_pct = (balance - 1.0) * 100.0

    max_dd_pct, sharpe_like = _compute_drawdown_stats(equity_curve)

    return {
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "n_trades": n_trades,
        "trades": trades,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_like": sharpe_like,
    }

# ---------------------------------------------------------------------------
# Single-asset pipeline for tabs
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str, interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> Optional[Dict[str, object]]:
    """
    Full pipeline per symbol:
      - fetch data
      - run indicators
      - get prediction (fused prob, TP/SL, sentiment)
      - run backtest (win rate, return%, draws, etc.)
    Output fields are used in Market Summary table.
    """
    df_raw = fetch_data(symbol, interval_key, use_cache)
    if df_raw.empty:
        return None

    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None

    pred = latest_prediction(df_raw, symbol, risk, interval_key)
    bt = backtest_signals(df_raw, risk)

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
        "max_drawdown_pct": bt["max_drawdown_pct"],
        "sharpe_like": bt["sharpe_like"],
    }

# ---------------------------------------------------------------------------
# Multi-asset summary for dashboard table
# ---------------------------------------------------------------------------
def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> pd.DataFrame:
    """
    Called by Market Summary tab in app.py.
    Loops through all ASSET_SYMBOLS and builds a summary row for each.
    """
    _log("Fetching and analyzing market data (smart v3)...")
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
            "MaxDD_%": res["max_drawdown_pct"],
            "SharpeLike": res["sharpe_like"],
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Helpers for Asset Analysis tab & Backtest tab
# ---------------------------------------------------------------------------
def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Returns (symbol, df_with_indicators) for chart plotting in the Asset Analysis tab.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)

    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key, use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind

def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True):
    """
    Returns:
      result dict with:
        - side, prob, tp/sl, sentiment, regime
        - win_rate, backtest return, drawdown, sharpe, n_trades
        - ml_prob and rule_prob for debugging if needed
      plus df_ind for plotting candles + indicators
    """
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()

    df_raw = fetch_data(symbol, interval_key, use_cache)
    if df_raw.empty:
        return None, pd.DataFrame()

    df_ind = add_indicators(df_raw)
    if df_ind.empty:
        return None, pd.DataFrame()

    pred = latest_prediction(df_raw, symbol, risk, interval_key)
    bt = backtest_signals(df_raw, risk)

    result = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df_ind["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": pred["prob"],  # blended final prob (0..1)
        "rule_prob": pred["rule_prob"],
        "ml_prob": pred["ml_prob"],
        "tp": pred["tp"],
        "sl": pred["sl"],
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "max_drawdown_pct": bt["max_drawdown_pct"],
        "sharpe_like": bt["sharpe_like"],
        "trades": bt["trades"],
    }

    return result, df_ind

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------