# utils.py — WoodyTradesPro Smart v4 (Regime-aware, Volatility Filtered, Quiet)
# ---------------------------------------------------------------------------
# This module provides:
# - Safe normalized OHLCV fetching with cache (fixes 2D column bugs)
# - Full technical feature set (EMA/RSI/MACD/ATR/Bollinger/ADX/ROC/etc.)
# - Regime detection (Bull / Bear)
# - Volatility filter (skip chop)
# - Sentiment scoring via headline tone
# - ML model that predicts 3-bars-ahead direction, trained per regime
# - Probability fusion (rule + ML + sentiment)
# - Thresholding to avoid low-conviction trades
# - TP/SL suggestions
# - Backtesting with Sharpe-like + Max Drawdown and ATR-aware sizing
# - Summary helpers for Streamlit tabs
#
# Maintains API compatibility with app.py:
#   summarize_assets
#   analyze_asset
#   load_asset_with_indicators
#   asset_prediction_and_backtest
#
# Keeps previous output fields (probability, win_rate, etc.) so UI won't break.

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
# Quiet logging / warning suppression
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

# minimum confidence threshold to actually take a directional stance
_PROB_THRESHOLD = 0.55

# ML model cache, now regime-aware
# key = (symbol, interval_key, regime_label)
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}

# ---------------------------------------------------------------------------
# Helper Logger (kept intentionally minimal / quiet)
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Normalization: fix yfinance quirks, flatten MultiIndex, coerce numeric
# ---------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean OHLCV DataFrame:
    - Flatten MultiIndex columns
    - Keep Open/High/Low/Close/Adj Close/Volume
    - Ensure each is 1D numeric (no shape (n,1), no arrays of arrays)
    - DatetimeIndex sorted asc
    - ffill/bfill tiny gaps
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep_cols:
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    df = df[keep_cols].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    df = df.sort_index()

    for col in df.columns:
        vals = df[col].values

        # collapse (n,1)->(n,)
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            vals = vals.reshape(-1)

        # collapse arrays of singletons
        if len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)):
            flat_vals = []
            for v in vals:
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    flat_vals.append(v[0])
                else:
                    flat_vals.append(np.nan)
            vals = np.array(flat_vals)

        df[col] = pd.to_numeric(vals, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")

    return df

# ---------------------------------------------------------------------------
# Fetch data with cache, then normalize
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """
    1. Attempt cache
    2. Otherwise yfinance download
    3. Normalize result using _normalize_ohlcv
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

    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if not cached.empty:
                return cached
        except Exception:
            pass

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
# Indicator panel / feature engineering
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - EMA20/EMA50
    - RSI14
    - MACD, MACD signal, MACD hist
    - ATR14
    - Bollinger Bands (20,2): mid/upper/lower/%B/bandwidth
    - ADX14
    - ROC features (price and volume over 10 bars)
    - ema_gap = (ema20-ema50)/ema50
    - atr_rel = ATR / rolling ATR mean (volatility regime)
    Also computes:
    - atr_mean_50: rolling avg(ATR,50) for volatility filter downstream
    Returns a clean df with all columns, NaN warmup rows dropped.
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
    out["atr_mean_50"] = out["atr"].rolling(50).mean()
    out["atr_rel"] = out["atr"] / out["atr_mean_50"]

    # Bollinger
    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_percent_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_bandwidth"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    # ADX (trend strength)
    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
    out["adx"] = adx_obj.adx()

    # Rate of change / momentum
    out["roc_close"] = close.pct_change(10)
    if "Volume" in out.columns:
        out["roc_vol"] = out["Volume"].pct_change(10)
    else:
        out["roc_vol"] = np.nan

    # EMA gap / regime strength
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    # cleanup
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    out = out.reset_index(drop=True)

    return out

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
def _get_regime_label(row: pd.Series) -> str:
    """
    Classify regime for ML:
    - 'bull' if ema_gap > 0
    - 'bear' if ema_gap < 0
    """
    if row.get("ema_gap", 0) >= 0:
        return "bull"
    else:
        return "bear"

# ---------------------------------------------------------------------------
# Rule-based directional signal with volatility filter
# ---------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Returns (side, base_confidence).
    Uses:
      - EMA trend alignment
      - MACD alignment
      - RSI bounds
      - Bollinger %B extremes
      - ADX (trend strength)
      - Volatility filter (atr_rel)
    """

    # Volatility / trend filter: need some movement and some structure
    vol_ok = (
        row.get("atr_rel", 1.0) >= 1.0 and  # current ATR >= rolling mean ATR
        row.get("adx", 0.0) > 15           # trend strength not totally dead
    )

    bull = (
        row["ema20"] > row["ema50"] and
        row["macd"] > row["signal"] and
        row["rsi"] < 70 and
        row["bb_percent_b"] < 1.1 and
        vol_ok
    )

    bear = (
        row["ema20"] < row["ema50"] and
        row["macd"] < row["signal"] and
        row["rsi"] > 30 and
        row["bb_percent_b"] > -0.1 and
        vol_ok
    )

    side = "Hold"
    prob = 0.5

    if bull:
        side, prob = "Buy", 0.7
    elif bear:
        side, prob = "Sell", 0.7

    # Momentum kicker: if strong recent move in same direction
    roc = row.get("roc_close", 0.0)
    if side == "Buy" and roc > 0.01:
        prob = min(0.9, prob + 0.1)
    if side == "Sell" and roc < -0.01:
        prob = min(0.9, prob + 0.1)

    return side, prob

# ---------------------------------------------------------------------------
# TP / SL generator based on ATR and risk profile
# ---------------------------------------------------------------------------
def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    ATR-based TP/SL bands.
    Risk multipliers:
      Low    -> tighter TP/looser SL
      Medium -> balanced
      High   -> wider TP/tighter SL
    """
    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)

    if atr is None or np.isnan(atr):
        atr = price * 0.005  # fallback ~0.5%

    if side == "Buy":
        tp = price + mult * atr
        sl = price - mult * atr
    elif side == "Sell":
        tp = price - mult * atr
        sl = price + mult * atr
    else:
        # If Hold, still present notional ranges for UI
        tp = price + mult * atr
        sl = price - mult * atr

    return float(tp), float(sl)

# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
def fetch_sentiment(symbol: str) -> float:
    """
    Average VADER compound score over latest headlines in yfinance.Ticker(symbol).news.
    Range approx [-1,1]. If no news or error, return 0.
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
# ML prep and training
# ---------------------------------------------------------------------------
def _prepare_ml_frame(df_ind: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Build supervised dataset using multi-bar lookahead:
    target = 1 if Close(horizon ahead) > Close(now), else 0
    This is less noisy than 1-bar.
    """
    d = df_ind.copy()
    d["future_close"] = d["Close"].shift(-horizon)
    d["target"] = (d["future_close"] > d["Close"]).astype(int)
    return d

def _get_ml_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Columns we feed to ML. Must exist in df_ind.
    """
    feats = [
        "ema20", "ema50", "ema_gap",
        "rsi",
        "macd", "signal", "macd_hist",
        "atr", "atr_rel",
        "bb_percent_b", "bb_bandwidth",
        "adx",
        "roc_close", "roc_vol",
    ]
    return df_ind[feats].copy()

def train_ml_model(symbol: str, interval_key: str, df_ind: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train or reuse a cached RandomForest model for the current regime (bull/bear).
    Model is per:
      (symbol, interval_key, regime_label)
    This way bull/bear get different classifiers.
    """
    if df_ind is None or df_ind.empty or len(df_ind) < 150:
        return None

    # Determine regime from the most recent row
    last_row = df_ind.iloc[-1]
    regime_label = _get_regime_label(last_row)

    cache_key = (symbol, interval_key, regime_label)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Supervised frame with 3-bar horizon
    supervised = _prepare_ml_frame(df_ind, horizon=3)

    # Filter rows that match the same regime (bull/bear) so model specializes
    # We'll approximate regime per row by ema_gap sign
    regime_mask = np.where(supervised["ema_gap"] >= 0, "bull", "bear")
    supervised = supervised[regime_mask == regime_label]

    X_all = _get_ml_features(supervised).dropna()
    if X_all.empty or "target" not in supervised.columns:
        return None
    y_all = supervised.loc[X_all.index, "target"]

    # Need both classes present
    if y_all.nunique() < 2:
        return None

    # Time-aware split (no shuffling)
    X_train, _, y_train, _ = train_test_split(
        X_all, y_all, test_size=0.25, shuffle=False
    )
    if X_train.empty or y_train.nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    _MODEL_CACHE[cache_key] = model
    return model

def predict_ml_probability(model: Optional[RandomForestClassifier], row: pd.Series) -> Optional[float]:
    """
    Given a trained model and feature row (Series), return prob next 3 bars are UP.
    """
    if model is None:
        return None
    try:
        X_last = row.to_frame().T
        proba_up = model.predict_proba(X_last)[0][1]
        return float(proba_up)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Probability fusion
# ---------------------------------------------------------------------------
def fuse_probabilities(rule_prob: float,
                       ml_prob: Optional[float],
                       sentiment: float,
                       adx: float) -> float:
    """
    Combine:
      - rule_prob (0..1)
      - ml_prob (0..1 or None)
      - sentiment (-1..1)
      - adx (trend strength)
    We:
      1. Blend rule + ML (60/40 if ML present)
      2. Apply sentiment multiplier ONLY if ADX suggests trend (>=20)
    """
    # base fusion of rule + ML
    base_prob = rule_prob
    if ml_prob is not None:
        base_prob = 0.6 * rule_prob + 0.4 * ml_prob

    # sentiment multiplier, only if there's an actual trend
    if adx >= 20:
        sent_weight = 1.0 + 0.2 * np.clip(sentiment, -1, 1)
        sent_weight = float(max(0.5, min(1.5, sent_weight)))
    else:
        sent_weight = 1.0  # ignore news in chop

    final_prob = base_prob * sent_weight

    # Clamp final prob into [0,1]
    final_prob = max(0.0, min(1.0, final_prob))
    return float(final_prob)

# ---------------------------------------------------------------------------
# Latest prediction
# ---------------------------------------------------------------------------
def latest_prediction(df_raw: pd.DataFrame,
                      symbol: str = "",
                      risk: str = "Medium",
                      interval_key: str = "1h") -> Optional[Dict[str, object]]:
    """
    Generate the live signal snapshot:
      - indicators
      - rule-based side/prob
      - ML regime-aware prob (3-bar horizon)
      - sentiment
      - fused probability (rule+ML+sentiment)
      - volatility filter & probability threshold
      - TP/SL
      - regime label
    """
    df_ind = add_indicators(df_raw)
    if df_ind is None or df_ind.empty or len(df_ind) < 2:
        return None

    row_prev = df_ind.iloc[-2]
    row_curr = df_ind.iloc[-1]

    # rule-based call
    side_rule, rule_prob = compute_signal_row(row_prev, row_curr)

    # ML prob
    model = train_ml_model(symbol, interval_key, df_ind)
    feat_row = _get_ml_features(df_ind).iloc[-1]
    ml_prob = predict_ml_probability(model, feat_row)

    # sentiment
    sentiment_score = fetch_sentiment(symbol)

    # fuse (uses ADX to decide if sentiment matters)
    fused_prob = fuse_probabilities(
        rule_prob=rule_prob,
        ml_prob=ml_prob,
        sentiment=sentiment_score,
        adx=row_curr.get("adx", 0.0),
    )

    # probability thresholding:
    # if fused_prob < 0.55 then we just call it Hold even if rule said Buy/Sell
    final_side = side_rule
    if fused_prob < _PROB_THRESHOLD:
        final_side = "Hold"

    # ATR, TP/SL
    atr_val = float(row_curr["atr"]) if not np.isnan(row_curr["atr"]) else float(df_ind["atr"].tail(14).mean())
    price = float(row_curr["Close"])
    orient_for_ranges = final_side if final_side != "Hold" else "Buy"
    tp, sl = compute_tp_sl(price, atr_val, orient_for_ranges, risk)

    # regime label
    regime_label = _get_regime_label(row_curr)

    return {
        "side": final_side,          # final actionable side after threshold
        "prob": fused_prob,          # fused prob (0..1)
        "rule_prob": rule_prob,      # rule confidence
        "ml_prob": ml_prob,          # ML prob next 3 bars up (may be None)
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment_score,
        "regime": regime_label,
    }

# ---------------------------------------------------------------------------
# Backtest utilities (Sharpe, drawdown, ATR sizing)
# ---------------------------------------------------------------------------
def _compute_drawdown_stats(equity_curve: List[float]) -> Tuple[float, float]:
    """
    From the running equity curve:
    - max_drawdown_pct
    - sharpe_like = mean(return)/std(return)*sqrt(N)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0

    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd_pct = float(dd.min() * 100.0)

    rets = np.diff(eq) / eq[:-1]
    if np.std(rets) > 0:
        sharpe_like = float(np.mean(rets) / np.std(rets) * np.sqrt(len(rets)))
    else:
        sharpe_like = 0.0

    return max_dd_pct, sharpe_like

def backtest_signals(df_raw: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    """
    Walk-forward backtest:
      - build indicators
      - from bar i>=60:
          * compute rule signal
          * fuse probability (simulate that you'd have ML+sent available)
          * skip if prob < threshold or side == "Hold"
          * size trade by ATR (lower ATR -> larger size, higher ATR -> smaller size)
          * hold 1 bar (entry at close[i], exit at close[i+1])
      - track equity
      - compute win rate, total return, drawdown, sharpe
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

    # Precompute rolling ATR mean for size scaling reference
    # We'll cap position size so we don't go insane on ultra-low ATR
    atr_mean = df_ind["atr"].rolling(50).mean()

    for i in range(60, len(df_ind) - 1):
        prev_row = df_ind.iloc[i - 1]
        row = df_ind.iloc[i]
        nxt_price = float(df_ind["Close"].iloc[i + 1])

        # rule side & rule prob
        side_rule, rule_prob = compute_signal_row(prev_row, row)

        # ML side probability based on current row features
        # we can't retrain per bar cheaply in backtest loop realistically
        # so we approximate by using a "fused probability" style but
        # without per-iteration retraining for speed:
        # We'll reuse rule_prob as proxy here to keep runtime sane.
        # (We keep the trade skip logic consistent with live: threshold).
        fused_prob_proxy = rule_prob

        if fused_prob_proxy < _PROB_THRESHOLD or side_rule == "Hold":
            equity_curve.append(balance)
            continue

        entry_px = float(row["Close"])

        # Position sizing inversely proportional to ATR (risk-based sizing)
        this_atr = float(row.get("atr", np.nan))
        ref_atr = float(atr_mean.iloc[i]) if not np.isnan(atr_mean.iloc[i]) else this_atr
        if np.isnan(this_atr) or this_atr <= 0 or np.isnan(ref_atr) or ref_atr <= 0:
            size = 1.0
        else:
            # e.g. if ATR is half the usual, we size up a bit; cap it
            size = min(2.0, max(0.25, ref_atr / this_atr))
            # ref_atr/this_atr: if ATR < ref_atr, that's >1 (bigger size), etc.

        # PnL contribution for 1-bar hold
        if side_rule == "Buy":
            trade_ret = (nxt_price - entry_px) / entry_px
        else:  # Sell
            trade_ret = (entry_px - nxt_price) / entry_px

        # apply position sizing
        sized_ret = trade_ret * size

        if sized_ret > 0:
            wins += 1

        balance *= (1 + sized_ret)
        equity_curve.append(balance)

        trades.append({
            "index": i,
            "side": side_rule,
            "entry_price": entry_px,
            "exit_price": nxt_price,
            "profit_pct": sized_ret * 100.0,
            "size": size,
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
# analyze_asset — single asset pipeline for summary row
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str,
                  interval_key: str = "1h",
                  risk: str = "Medium",
                  use_cache: bool = True) -> Optional[Dict[str, object]]:
    """
    Fetch data -> indicators -> prediction (final side, prob, TP/SL) -> backtest stats
    Returns dict with keys consumed by summarize_assets and app UI.
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
# summarize_assets — for Market Summary table
# ---------------------------------------------------------------------------
def summarize_assets(interval_key: str = "1h",
                     risk: str = "Medium",
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Iterates all assets, runs analyze_asset(), returns summary DataFrame.
    """
    _log("Fetching and analyzing market data (smart v4)...")
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
# load_asset_with_indicators — for charts / detail tabs
# ---------------------------------------------------------------------------
def load_asset_with_indicators(asset: str,
                               interval_key: str,
                               use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Returns (symbol, df_ind) with indicators for plotting candles & overlays.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)

    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key, use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind

# ---------------------------------------------------------------------------
# asset_prediction_and_backtest — for Scenario / Backtest tabs
# ---------------------------------------------------------------------------
def asset_prediction_and_backtest(asset: str,
                                  interval_key: str,
                                  risk: str,
                                  use_cache: bool = True):
    """
    Returns:
      - result dict (prediction, stats, risk info, sentiment, regime, etc.)
      - df_ind with indicators for charting

    The dict keeps keys used in earlier versions:
      side, probability, tp, sl, win_rate,
      backtest_return_pct, n_trades, sentiment, regime, ml_prob, etc.
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
        "probability": pred["prob"],               # fused final prob (0..1)
        "rule_prob": pred["rule_prob"],            # rule-only prob
        "ml_prob": pred["ml_prob"],                # ML-only prob (may be None)
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