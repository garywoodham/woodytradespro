# utils.py — WoodyTradesPro Smart v5
# ---------------------------------------------------------------------------
# This module provides:
# - Safe normalized OHLCV fetching with cache (fixes 2D column bugs)
# - Technical feature set (EMA/RSI/MACD/ATR/Bollinger/ADX/ROC/etc.)
# - Regime detection (Bull / Bear)
# - Volatility filter (atr_rel, ADX floor)
# - Sentiment scoring via headlines
# - Regime-aware ML model predicting a 3-bar-ahead move
# - Probability fusion (rule + ML + sentiment gated by ADX)
# - Probability thresholding to skip weak setups
# - TP/SL bands from ATR and risk level
# - NEW: Reward/Risk gating (must have decent RR or we refuse the trade)
# - NEW: Exhaustion / no-chase filter (don't enter into blowoff extensions)
# - NEW: Regime-aware position sizing in backtest (pro-trend gets size boost)
# - Backtesting with Sharpe-like + Max Drawdown
# - Streamlit tab helpers
#
# Maintains compatibility with app.py:
#   summarize_assets
#   analyze_asset
#   load_asset_with_indicators
#   asset_prediction_and_backtest
#
# Keeps Smart v4 outputs (win_rate, probability, TP, SL, trades, etc.)
# Nothing removed; only enhanced.

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
# Quiet mode (suppress noisy libs)
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Could not infer format")

# ---------------------------------------------------------------------------
# Global config
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

# Minimum final conviction required to allow a trade
_PROB_THRESHOLD = 0.55

# Minimum reward/risk required to allow a trade
_MIN_RR = 1.5  # e.g. TP distance must be >= 1.5x SL distance

# ML cache (now regime-aware)
# cache key: (symbol, interval_key, regime_label)
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}

# ---------------------------------------------------------------------------
# Internal logging helper
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Data normalization helper
# ---------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV like:
    - Flatten MultiIndex columns
    - Keep {Open,High,Low,Close,Adj Close,Volume}
    - Force 1D numeric for each column (no shape (n,1), no arrays-of-arrays)
    - DatetimeIndex ascending
    - Fill small gaps and drop full NaN rows
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

        # flatten (n,1) -> (n,)
        if isinstance(vals, np.ndarray) and getattr(vals, "ndim", 1) > 1:
            vals = vals.reshape(-1)

        # flatten arrays per cell
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
# Fetch data with cache, normalize it
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """
    1. Try local CSV cache
    2. Else fetch via yfinance
    3. Normalize with _normalize_ohlcv
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
# Technical indicators / engineered features
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - ema20 / ema50
    - rsi (14)
    - macd / signal / macd_hist
    - atr (14)
    - atr_mean_50 (rolling mean of atr)
    - atr_rel = atr / atr_mean_50  (volatility regime factor)
    - Bollinger percent_b, bandwidth
    - ADX (14)
    - roc_close, roc_vol (10-bar momentum)
    - ema_gap = (ema20-ema50)/ema50
    Plus:
    - close_above_ema20_atr: how many ATRs price is above EMA20
    - trend_age: consecutive bars ema20>ema50 (or < for bear)
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

    # Momentum / volume thrust
    out["roc_close"] = close.pct_change(10)
    if "Volume" in out.columns:
        out["roc_vol"] = out["Volume"].pct_change(10)
    else:
        out["roc_vol"] = np.nan

    # Regime strength
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    # Extension above EMA20 in ATR units (used for "no chase" filter)
    out["close_above_ema20_atr"] = (out["Close"] - out["ema20"]) / out["atr"]

    # Trend age = how long has ema20 been above ema50 (bull streak) or below
    # We'll compute streak in a vectorized way:
    is_bull = out["ema20"] > out["ema50"]
    streak = np.zeros(len(out), dtype=int)
    run = 0
    for i, bull_now in enumerate(is_bull):
        if i == 0:
            run = 1
        else:
            if bull_now == is_bull.iloc[i - 1]:
                run += 1
            else:
                run = 1
        streak[i] = run
    out["trend_age"] = streak  # note: in a bear regime this is bear streak length

    # cleanup
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    out = out.reset_index(drop=True)

    return out

# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def _get_regime_label(row: pd.Series) -> str:
    """ 'bull' if ema_gap >= 0, else 'bear' """
    return "bull" if row.get("ema_gap", 0) >= 0 else "bear"

# ---------------------------------------------------------------------------
# Rule-based side with volatility gate
# ---------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Return (side, base_confidence) using:
    - ema20 vs ema50
    - macd vs signal
    - rsi boundaries
    - bb_percent_b extremes
    - adx > 15 and atr_rel >= 1 (vol/trend filter)
    - momentum kicker
    """

    vol_ok = (
        row.get("atr_rel", 1.0) >= 1.0 and
        row.get("adx", 0.0) > 15
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

    roc = row.get("roc_close", 0.0)
    if side == "Buy" and roc > 0.01:
        prob = min(0.9, prob + 0.1)
    if side == "Sell" and roc < -0.01:
        prob = min(0.9, prob + 0.1)

    return side, prob

# ---------------------------------------------------------------------------
# TP / SL logic (unchanged from v4)
# ---------------------------------------------------------------------------
def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    ATR-based TP/SL bands with risk multipliers.
    """
    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)

    if atr is None or np.isnan(atr):
        atr = price * 0.005  # fallback

    if side == "Buy":
        tp = price + mult * atr
        sl = price - mult * atr
    elif side == "Sell":
        tp = price - mult * atr
        sl = price + mult * atr
    else:
        tp = price + mult * atr
        sl = price - mult * atr

    return float(tp), float(sl)

# ---------------------------------------------------------------------------
# Sentiment scoring (unchanged core logic)
# ---------------------------------------------------------------------------
def fetch_sentiment(symbol: str) -> float:
    """
    Average VADER compound score over recent headlines.
    """
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", [])
        if not news:
            return 0.0

        analyzer = SentimentIntensityAnalyzer()
        vals = []
        for item in news[:5]:
            title = item.get("title", "")
            if not title:
                continue
            s = analyzer.polarity_scores(title)["compound"]
            vals.append(s)

        if not vals:
            return 0.0
        return float(np.mean(vals))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# ML prep/training — regime-aware, 3-bar horizon (like v4)
# ---------------------------------------------------------------------------
def _prepare_ml_frame(df_ind: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    target = 1 if Close(horizon ahead) > Close(now), else 0
    """
    d = df_ind.copy()
    d["future_close"] = d["Close"].shift(-horizon)
    d["target"] = (d["future_close"] > d["Close"]).astype(int)
    return d

def _get_ml_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set to feed ML.
    """
    feats = [
        "ema20", "ema50", "ema_gap",
        "rsi",
        "macd", "signal", "macd_hist",
        "atr", "atr_rel",
        "bb_percent_b", "bb_bandwidth",
        "adx",
        "roc_close", "roc_vol",
        "close_above_ema20_atr",
        "trend_age",
    ]
    return df_ind[feats].copy()

def train_ml_model(symbol: str, interval_key: str, df_ind: pd.DataFrame) -> Optional[RandomForestClassifier]:
    """
    Train (or reuse) model specialized to the *current regime* ('bull' or 'bear').
    """
    if df_ind is None or df_ind.empty or len(df_ind) < 150:
        return None

    last_row = df_ind.iloc[-1]
    regime_label = _get_regime_label(last_row)

    cache_key = (symbol, interval_key, regime_label)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    supervised = _prepare_ml_frame(df_ind, horizon=3)

    # Filter rows to same regime
    regime_mask = np.where(supervised["ema_gap"] >= 0, "bull", "bear")
    supervised = supervised[regime_mask == regime_label]

    X_all = _get_ml_features(supervised).dropna()
    if X_all.empty or "target" not in supervised.columns:
        return None
    y_all = supervised.loc[X_all.index, "target"]

    if y_all.nunique() < 2:
        return None

    # time-aware split
    X_train, _, y_train, _ = train_test_split(
        X_all, y_all,
        test_size=0.25,
        shuffle=False
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
    Return probability that price is higher 3 bars ahead than now.
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
# Probability fusion (unchanged logic except comments)
# ---------------------------------------------------------------------------
def fuse_probabilities(rule_prob: float,
                       ml_prob: Optional[float],
                       sentiment: float,
                       adx: float) -> float:
    """
    Blend:
    - rule_prob (0..1)
    - ml_prob (0..1 or None)
    - sentiment (-1..1), only if ADX>=20 (trending)
    """
    base_prob = rule_prob
    if ml_prob is not None:
        base_prob = 0.6 * rule_prob + 0.4 * ml_prob

    if adx >= 20:
        sent_weight = 1.0 + 0.2 * np.clip(sentiment, -1, 1)
        sent_weight = float(max(0.5, min(1.5, sent_weight)))
    else:
        sent_weight = 1.0

    final_prob = base_prob * sent_weight
    final_prob = max(0.0, min(1.0, final_prob))
    return float(final_prob)

# ---------------------------------------------------------------------------
# NEW: Reward/Risk calculation
# ---------------------------------------------------------------------------
def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Compute reward/risk ratio based on TP/SL levels.
    Buy: RR = (tp - price) / (price - sl)
    Sell: RR = (price - tp) / (sl - price)
    Hold: treat like Buy for shape, it's only used if we consider entry anyway.
    Guard against div-by-zero.
    """
    if side == "Sell":
        reward = price - tp
        risk = sl - price
    else:  # Buy or Hold fallback
        reward = tp - price
        risk = price - sl

    if risk <= 0:
        return 0.0
    return float(reward / risk)

# ---------------------------------------------------------------------------
# NEW: Exhaustion / no-chase filter
# ---------------------------------------------------------------------------
def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    We don't want to chase vertical moves.
    Conditions:
    - If trying to Buy:
        - price is already far above ema20 in ATR terms
        - AND trend_age is large (trend already running)
    - If trying to Sell:
        - price is far below ema20
        - AND trend_age is large
    """
    stretch = row.get("close_above_ema20_atr", 0.0)
    age = row.get("trend_age", 0)

    # Heuristic thresholds:
    # "far" = > +2 ATR above ema20 for longs, < -2 ATR for shorts
    # "mature" trend = streak > 30 bars
    mature_trend = age > 30

    if side == "Buy":
        if (stretch is not None) and (stretch > 2.0) and mature_trend:
            return True
    if side == "Sell":
        if (stretch is not None) and (stretch < -2.0) and mature_trend:
            return True

    return False

# ---------------------------------------------------------------------------
# Latest prediction (Smart v5 core)
# ---------------------------------------------------------------------------
def latest_prediction(df_raw: pd.DataFrame,
                      symbol: str = "",
                      risk: str = "Medium",
                      interval_key: str = "1h") -> Optional[Dict[str, object]]:
    """
    Steps:
    1. indicators
    2. rule-based side/prob
    3. ML probability (regime-aware, 3-bar horizon)
    4. sentiment
    5. fuse them -> fused_prob
    6. apply:
        - probability threshold (_PROB_THRESHOLD)
        - reward/risk threshold (_MIN_RR)
        - exhaustion / no-chase filter
    7. build TP/SL, regime, ATR
    """
    df_ind = add_indicators(df_raw)
    if df_ind is None or df_ind.empty or len(df_ind) < 2:
        return None

    row_prev = df_ind.iloc[-2]
    row_curr = df_ind.iloc[-1]

    # rule signal
    side_rule, rule_prob = compute_signal_row(row_prev, row_curr)

    # ML prob
    model = train_ml_model(symbol, interval_key, df_ind)
    feat_row = _get_ml_features(df_ind).iloc[-1]
    ml_prob = predict_ml_probability(model, feat_row)

    # sentiment
    sentiment_score = fetch_sentiment(symbol)

    # fused confidence
    fused_prob = fuse_probabilities(
        rule_prob=rule_prob,
        ml_prob=ml_prob,
        sentiment=sentiment_score,
        adx=row_curr.get("adx", 0.0),
    )

    # ATR + TP/SL pre-calc
    atr_val = float(row_curr["atr"]) if not np.isnan(row_curr["atr"]) else float(df_ind["atr"].tail(14).mean())
    price = float(row_curr["Close"])

    # We'll propose TP/SL in the direction of side_rule first (not final_side yet)
    orient_initial = side_rule if side_rule != "Hold" else "Buy"
    tp_raw, sl_raw = compute_tp_sl(price, atr_val, orient_initial, risk)
    rr = _calc_rr(price, tp_raw, sl_raw, orient_initial)

    # base final_side from side_rule
    final_side = side_rule

    # 1) probability gating
    if fused_prob < _PROB_THRESHOLD:
        final_side = "Hold"

    # 2) reward/risk gating
    if rr < _MIN_RR:
        final_side = "Hold"

    # 3) exhaustion / no-chase gating
    if _is_exhausted(row_curr, final_side):
        final_side = "Hold"

    # after gating, recompute TP/SL aligned with final_side
    orient_for_ranges = final_side if final_side != "Hold" else "Buy"
    tp, sl = compute_tp_sl(price, atr_val, orient_for_ranges, risk)

    # final regime label
    regime_label = _get_regime_label(row_curr)

    return {
        "side": final_side,
        "prob": fused_prob,          # fused probability 0..1
        "rule_prob": rule_prob,      # rule-only probability
        "ml_prob": ml_prob,          # ML-only prob (may be None)
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "sentiment": sentiment_score,
        "regime": regime_label,
        "rr": rr,                    # reward/risk before gating
    }

# ---------------------------------------------------------------------------
# Backtest with regime-aware sizing and ATR sizing
# ---------------------------------------------------------------------------
def _compute_drawdown_stats(equity_curve: List[float]) -> Tuple[float, float]:
    """
    Returns:
    - max_drawdown_pct
    - sharpe_like
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
    Backtest logic:
    - Build indicators
    - For each bar i >= 60:
        * Get rule signal + rule_prob
        * Build ATR info
        * Estimate fused_prob_proxy from rule_prob
        * Compute RR for that bar
        * Apply Smart v5 gating:
            - prob threshold
            - RR threshold
            - exhaustion filter
        * Trade for one bar (entry close[i], exit close[i+1])
        * Position size:
            - inversely proportional to ATR (like v4)
            - boosted/reduced if trade direction matches regime
    - Track equity, trades, Sharpe-like, MaxDD%
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

    atr_mean = df_ind["atr"].rolling(50).mean()

    for i in range(60, len(df_ind) - 1):
        row_prev = df_ind.iloc[i - 1]
        row = df_ind.iloc[i]
        nxt_price = float(df_ind["Close"].iloc[i + 1])

        # rule-only
        side_rule, rule_prob = compute_signal_row(row_prev, row)

        # approximate fused prob in backtest loop using rule_prob
        fused_prob_proxy = rule_prob

        # skip if no direction or weak probability
        if side_rule == "Hold" or fused_prob_proxy < _PROB_THRESHOLD:
            equity_curve.append(balance)
            continue

        # compute ATR / TP/SL / RR
        this_atr = float(row.get("atr", np.nan))
        if np.isnan(this_atr) or this_atr <= 0:
            this_atr = float(df_ind["atr"].iloc[max(0, i-14):i+1].mean())

        entry_px = float(row["Close"])

        orient_initial = side_rule if side_rule != "Hold" else "Buy"
        tp_raw, sl_raw = compute_tp_sl(entry_px, this_atr, orient_initial, risk)
        rr_here = _calc_rr(entry_px, tp_raw, sl_raw, orient_initial)

        # reward/risk gate
        if rr_here < _MIN_RR:
            equity_curve.append(balance)
            continue

        # exhaustion / no-chase gate
        if _is_exhausted(row, side_rule):
            equity_curve.append(balance)
            continue

        # Size by ATR (like v4)
        ref_atr = float(atr_mean.iloc[i]) if not np.isnan(atr_mean.iloc[i]) else this_atr
        if np.isnan(ref_atr) or ref_atr <= 0:
            size = 1.0
        else:
            size = min(2.0, max(0.25, ref_atr / this_atr))

        # Regime-aware sizing:
        regime_now = _get_regime_label(row)
        # If we're trading *with* the regime, boost size a bit.
        # If against, reduce size.
        if regime_now == "bull":
            if side_rule == "Buy":
                size *= 1.25  # pro-trend long
            elif side_rule == "Sell":
                size *= 0.75  # counter-trend short
        else:  # bear regime
            if side_rule == "Sell":
                size *= 1.25  # pro-trend short
            elif side_rule == "Buy":
                size *= 0.75  # counter-trend long

        # PnL for 1 bar
        if side_rule == "Buy":
            trade_ret = (nxt_price - entry_px) / entry_px
        else:
            trade_ret = (entry_px - nxt_price) / entry_px

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
            "rr_at_entry": rr_here,
            "regime": regime_now,
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
# analyze_asset — used in summary rows
# ---------------------------------------------------------------------------
def analyze_asset(symbol: str,
                  interval_key: str = "1h",
                  risk: str = "Medium",
                  use_cache: bool = True) -> Optional[Dict[str, object]]:
    """
    High-level snapshot for one symbol.
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
# summarize_assets — multi-asset overview table
# ---------------------------------------------------------------------------
def summarize_assets(interval_key: str = "1h",
                     risk: str = "Medium",
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Loop assets, run analyze_asset, assemble table.
    """
    _log("Fetching and analyzing market data (smart v5)...")
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
# load_asset_with_indicators — for charts in Asset Analysis tab
# ---------------------------------------------------------------------------
def load_asset_with_indicators(asset: str,
                               interval_key: str,
                               use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    """
    Returns (symbol, df_with_indicators) for plotting.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)

    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key, use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind

# ---------------------------------------------------------------------------
# asset_prediction_and_backtest — for Backtest tab / Scenario tab
# ---------------------------------------------------------------------------
def asset_prediction_and_backtest(asset: str,
                                  interval_key: str,
                                  risk: str,
                                  use_cache: bool = True):
    """
    Returns:
      result dict (latest signal snapshot + backtest stats)
      df_ind for plotting
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
        "probability": pred["prob"],               # 0..1 fused prob
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
        "atr": pred["atr"],
        "rr": pred["rr"],
    }

    return result, df_ind

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------