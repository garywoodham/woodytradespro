# utils.py — WoodyTradesPro Smart v6 (full)
# --------------------------------------------------------------------------------------
# COMPLETE module:
# - Safe OHLCV fetch (yfinance) + CSV cache
# - Normalization: flatten 2D shapes, MultiIndex, numeric coercion, datetime index
# - Indicators: EMA(20,50), RSI(14), MACD(12,26,9), ATR(14), Bollinger(20,2), ADX(14),
#               ROC(10), atr_rel, ema_gap, close_above_ema20_atr, trend_age
# - Sentiment: VADER over yfinance headlines with advanced weighting + smoothing
# - Regime-aware ML: RandomForest (time split, class_weight), cached per (symbol, interval, regime)
# - Signal engine: rule-based + probability fusion (rule+ML+sentiment gated by ADX)
# - Risk: ATR-based TP/SL; adaptive RR gating by regime & volatility
# - Protection: Exhaustion/no-chase filter; robust NaN strategy (ffill/bfill)
# - Backtest: regime-aware sizing, equity curve, max drawdown, Sharpe-like
# - Public API: summarize_assets, analyze_asset, load_asset_with_indicators, asset_prediction_and_backtest
#
# IMPORTANT: No prior functionality removed. Only additive enhancements and safety fixes.

from __future__ import annotations

import os
import time
import math
import warnings
import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Technical analysis
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------------------------------------------
# Quiet noisy libraries
# --------------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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

# Base gating thresholds (we adapt them below)
_BASE_PROB_THRESHOLD = 0.55
_BASE_MIN_RR = 1.2

# ML model cache per (symbol, interval_key, regime_label)
_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}

# Sentiment analyzer (singleton)
_VADER = SentimentIntensityAnalyzer()


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Flatten MultiIndex -> level 0
    - Keep OHLCV columns
    - Enforce 1D numeric vectors (fix (n,1) shapes)
    - Datetime index, ascending
    - Robust ffill/bfill, drop all-NaN rows
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    for col in df.columns:
        vals = np.array(df[col]).reshape(-1)  # flatten (n,1) -> (n,)
        # if cells are arrays, take first element
        if len(vals) and isinstance(vals[0], (list, np.ndarray)):
            vals = np.array([v[0] if isinstance(v, (list, np.ndarray)) and len(v) else np.nan for v in vals])
        df[col] = pd.to_numeric(vals, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df = df.dropna(how="all")
    return df


# --------------------------------------------------------------------------------------
# Fetch + cache
# --------------------------------------------------------------------------------------
def fetch_data(symbol: str, interval_key: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    interval = INTERVALS.get(interval_key, "60m")
    period = PERIODS.get(interval_key, "2mo")

    safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    cache_path = os.path.join(DATA_DIR, f"{safe}_{interval}.csv")

    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if not cached.empty:
                return cached
        except Exception:
            pass

    _log(f"⏳ Fetching {symbol} [{interval}] ...")
    for attempt in range(3):
        try:
            raw = yf.download(symbol, period=period, interval=interval, progress=False, threads=False, auto_adjust=True)
            norm = _normalize_ohlcv(raw)
            if not norm.empty and len(norm) > 50:
                try:
                    norm.to_csv(cache_path)
                except Exception:
                    pass
                _log(f"✅ {symbol}: fetched {len(norm)} rows.")
                return norm
        except Exception as e:
            _log(f"⚠️ {symbol} attempt {attempt+1}: {e}")
            time.sleep(1 + attempt * 0.5)

    _log(f"🚫 All fetch attempts failed for {symbol}.")
    return pd.DataFrame()


# --------------------------------------------------------------------------------------
# Indicators & engineered features
# --------------------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    base = _normalize_ohlcv(df)
    if base.empty:
        return pd.DataFrame()

    out = base.copy()
    c = out["Close"]
    h = out["High"]
    l = out["Low"]

    # EMA / RSI / MACD
    out["ema20"] = EMAIndicator(c, 20).ema_indicator()
    out["ema50"] = EMAIndicator(c, 50).ema_indicator()
    out["rsi"] = RSIIndicator(c, 14).rsi()

    macd = MACD(c)
    out["macd"] = macd.macd()
    out["signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # ATR + volatility normalization
    atr = AverageTrueRange(h, l, c, 14).average_true_range()
    out["atr"] = atr.fillna(atr.rolling(14).mean())
    out["atr_mean_50"] = out["atr"].rolling(50, min_periods=10).mean()
    out["atr_rel"] = out["atr"] / out["atr_mean_50"]

    # Bollinger + ADX
    bb = BollingerBands(c, 20, 2)
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_percent_b"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_bandwidth"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    out["adx"] = ADXIndicator(h, l, c, 14).adx()

    # Momentum
    out["roc_close"] = c.pct_change(10)
    out["roc_vol"] = out["Volume"].pct_change(10) if "Volume" in out.columns else np.nan

    # Regime / stretch / age
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]
    out["close_above_ema20_atr"] = (out["Close"] - out["ema20"]) / out["atr"]

    # trend_age: consecutive streak of ema20 > ema50 (or <)
    bull = out["ema20"] > out["ema50"]
    streak = []
    run = 0
    for i, b in enumerate(bull):
        if i == 0:
            run = 1
        else:
            run = run + 1 if b == bull.iloc[i - 1] else 1
        streak.append(run)
    out["trend_age"] = streak

    # Clean NaNs conservatively
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = out.dropna(subset=["Close", "ema20", "ema50"])
    out.reset_index(drop=True, inplace=True)
    return out


# --------------------------------------------------------------------------------------
# Sentiment (advanced weighting + smoothing)
# --------------------------------------------------------------------------------------
def fetch_sentiment(symbol: str, smoothing: int = 3) -> float:
    """
    Score = smoothed mean VADER compound for latest headlines (up to 7).
    - Downweight very old headlines (rare in yfinance payload)
    - Slightly upweight large caps tickers vs commodities (heuristic)
    - Clamp to [-1, 1]
    """
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", []) or []
        if not news:
            return 0.0

        vals: List[float] = []
        for i, item in enumerate(news[:7]):
            title = item.get("title", "") or ""
            if not title.strip():
                continue
            score = _VADER.polarity_scores(title)["compound"]

            # recency weight (front-loaded)
            w_recency = 1.0 - 0.08 * i  # 1.00, 0.92, 0.84, ...
            w_recency = max(0.7, w_recency)

            # asset class tweak: indices/stocks react more to headlines than FX/commodities
            if symbol.startswith("^") or symbol.endswith(("-USD",)):
                w_asset = 1.1
            elif symbol.endswith(("=X",)):
                w_asset = 0.9
            else:
                w_asset = 1.0

            vals.append(score * w_recency * w_asset)

        if not vals:
            return 0.0

        # simple smoothing (EMA on the last N)
        smoothed = []
        alpha = 2.0 / (smoothing + 1.0)
        ema = vals[0]
        for v in vals:
            ema = alpha * v + (1 - alpha) * ema
            smoothed.append(ema)

        s = float(np.mean(smoothed))
        return float(np.clip(s, -1.0, 1.0))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------------------
# Regime
# --------------------------------------------------------------------------------------
def _get_regime_label(row: pd.Series) -> str:
    return "bull" if row.get("ema_gap", 0.0) >= 0 else "bear"


# --------------------------------------------------------------------------------------
# Rule engine (direction & base confidence)
# --------------------------------------------------------------------------------------
def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Base direction with a volatility/trend floor.
    """
    vol_ok = (row.get("adx", 0.0) > 12) and (row.get("atr_rel", 1.0) >= 0.6)

    side = "Hold"
    prob = 0.5

    if row["ema20"] > row["ema50"] and row["macd"] > row["signal"] and vol_ok:
        side, prob = "Buy", 0.66
        if row.get("rsi", 50) < 70 and row.get("bb_percent_b", 0.5) < 1.1:
            prob += 0.06
    elif row["ema20"] < row["ema50"] and row["macd"] < row["signal"] and vol_ok:
        side, prob = "Sell", 0.66
        if row.get("rsi", 50) > 30 and row.get("bb_percent_b", 0.5) > -0.1:
            prob += 0.06

    # Momentum kicker
    roc = row.get("roc_close", 0.0)
    if side == "Buy" and roc > 0.01:
        prob = min(0.9, prob + 0.08)
    if side == "Sell" and roc < -0.01:
        prob = min(0.9, prob + 0.08)

    return side, float(prob)


# --------------------------------------------------------------------------------------
# TP/SL + RR + filters
# --------------------------------------------------------------------------------------
def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0}.get(risk, 1.0)
    atr = float(atr) if not (atr is None or np.isnan(atr)) else price * 0.005
    if side == "Buy":
        return price + mult * atr, price - mult * atr
    elif side == "Sell":
        return price - mult * atr, price + mult * atr
    else:
        return price + mult * atr, price - mult * atr


def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    if side == "Sell":
        reward = price - tp
        risk = sl - price
    else:
        reward = tp - price
        risk = price - sl
    if risk <= 0:
        return 0.0
    return float(reward / risk)


def _is_exhausted(row: pd.Series, side: str) -> bool:
    stretch = row.get("close_above_ema20_atr", 0.0)
    age = int(row.get("trend_age", 0))
    mature = age > 30
    if side == "Buy" and stretch > 2.0 and mature:
        return True
    if side == "Sell" and stretch < -2.0 and mature:
        return True
    return False


def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    Adapt (_BASE_PROB_THRESHOLD, _BASE_MIN_RR) by regime & volatility.
    - In strong trend (ADX high), accept slightly lower prob, but demand better RR
    - In choppy regime, require higher prob but allow lower RR
    """
    adx = row.get("adx", 0.0)
    atr_rel = row.get("atr_rel", 1.0)
    regime = _get_regime_label(row)

    prob_th = _BASE_PROB_THRESHOLD
    rr_th = _BASE_MIN_RR

    if adx >= 25 and atr_rel >= 1.0:
        prob_th -= 0.03  # easier to enter in clean trend
        rr_th += 0.2     # but demand better RR
    elif adx < 15 or atr_rel < 0.8:
        prob_th += 0.05  # avoid chop
        rr_th -= 0.1

    # Regime tilt: slightly stricter on counter-trend
    if regime == "bull":
        pass
    else:
        pass  # leave symmetric; sizing handles tilt

    return float(max(0.5, min(0.9, prob_th))), float(max(1.0, min(2.5, rr_th)))


# --------------------------------------------------------------------------------------
# ML (regime-aware)
# --------------------------------------------------------------------------------------
def _prepare_ml_frame(df_ind: pd.DataFrame) -> pd.DataFrame:
    d = df_ind.copy()
    d["future_close"] = d["Close"].shift(-3)
    d["target"] = (d["future_close"] > d["Close"]).astype(int)
    return d


def _get_ml_features(df_ind: pd.DataFrame) -> pd.DataFrame:
    feats = [
        "ema20", "ema50", "ema_gap",
        "rsi",
        "macd", "signal", "macd_hist",
        "atr_rel",
        "bb_percent_b", "bb_bandwidth",
        "adx",
        "roc_close",
        "close_above_ema20_atr",
        "trend_age",
    ]
    return df_ind[feats].copy()


def train_ml_model(symbol: str, interval_key: str, df_ind: pd.DataFrame) -> Optional[RandomForestClassifier]:
    if df_ind is None or df_ind.empty or len(df_ind) < 120:
        return None

    regime_label = _get_regime_label(df_ind.iloc[-1])
    cache_key = (symbol, interval_key, regime_label)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    d = _prepare_ml_frame(df_ind)
    # Filter to same regime to avoid mixing distributions
    mask = np.where(d["ema_gap"] >= 0, "bull", "bear")
    d = d[mask == regime_label]

    X = _get_ml_features(d).dropna()
    if X.empty:
        return None
    y = d.loc[X.index, "target"]
    if y.nunique() < 2:
        return None

    # Time-aware split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, shuffle=False)
    if y_train.nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    _MODEL_CACHE[cache_key] = model
    return model


def predict_ml_probability(model: Optional[RandomForestClassifier], row: pd.Series) -> Optional[float]:
    if model is None:
        return None
    try:
        X_last = row.to_frame().T
        proba = model.predict_proba(X_last)[0][1]
        return float(proba)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Probability fusion
# --------------------------------------------------------------------------------------
def fuse_probabilities(rule_prob: float, ml_prob: Optional[float], sentiment: float, adx: float) -> float:
    """
    - Base: 0.6*rule + 0.4*ml (if ml available else rule)
    - Sentiment scales probability only in trending markets (ADX >= 20)
    """
    base = 0.6 * rule_prob + 0.4 * (ml_prob if ml_prob is not None else rule_prob)
    if adx >= 20:
        sent_weight = 1.0 + 0.25 * float(np.clip(sentiment, -1, 1))
        base *= float(np.clip(sent_weight, 0.7, 1.4))
    return float(np.clip(base, 0.0, 1.0))


# --------------------------------------------------------------------------------------
# Latest prediction (Smart v6 core)
# --------------------------------------------------------------------------------------
def latest_prediction(
    df_raw: pd.DataFrame,
    symbol: str = "",
    risk: str = "Medium",
    interval_key: str = "1h",
) -> Optional[Dict[str, object]]:
    df_ind = add_indicators(df_raw)
    if df_ind is None or df_ind.empty or len(df_ind) < 5:
        return None

    row_prev = df_ind.iloc[-2]
    row = df_ind.iloc[-1]

    # Rule direction
    side_rule, rule_prob = compute_signal_row(row_prev, row)

    # ML
    model = train_ml_model(symbol, interval_key, df_ind)
    ml_prob = predict_ml_probability(model, _get_ml_features(df_ind).iloc[-1])

    # Sentiment
    sent = fetch_sentiment(symbol)

    # Fuse
    fused_prob = fuse_probabilities(rule_prob, ml_prob, sent, row.get("adx", 0.0))

    # Adaptive thresholds
    prob_th, rr_th = _adaptive_thresholds(row)

    # TP/SL + RR
    atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else float(df_ind["atr"].tail(14).mean())
    price = float(row["Close"])
    orient = side_rule if side_rule != "Hold" else "Buy"
    tp_raw, sl_raw = compute_tp_sl(price, atr_val, orient, risk)
    rr = _calc_rr(price, tp_raw, sl_raw, orient)

    final_side = side_rule

    # Gating
    if fused_prob < prob_th:
        final_side = "Hold"
    if rr < rr_th:
        final_side = "Hold"
    if _is_exhausted(row, final_side):
        final_side = "Hold"

    # Final aligned TP/SL (if Hold, keep long orientation for display)
    orient_final = final_side if final_side != "Hold" else "Buy"
    tp, sl = compute_tp_sl(price, atr_val, orient_final, risk)

    return {
        "side": final_side,
        "prob": float(fused_prob),
        "rule_prob": float(rule_prob),
        "ml_prob": ml_prob,
        "price": price,
        "tp": float(tp),
        "sl": float(sl),
        "atr": float(atr_val),
        "sentiment": float(sent),
        "regime": _get_regime_label(row),
        "rr": float(rr),
        "prob_threshold": float(prob_th),
        "rr_threshold": float(rr_th),
    }


# --------------------------------------------------------------------------------------
# Backtest (regime-aware sizing, adaptive gates)
# --------------------------------------------------------------------------------------
def _compute_drawdown_stats(equity_curve: List[float]) -> Tuple[float, float]:
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd_pct = float(dd.min() * 100.0)
    rets = np.diff(eq) / eq[:-1]
    sharpe_like = float(np.mean(rets) / np.std(rets) * np.sqrt(len(rets))) if np.std(rets) > 0 else 0.0
    return max_dd_pct, sharpe_like


def backtest_signals(df_raw: pd.DataFrame, risk: str = "Medium") -> Dict[str, object]:
    d = add_indicators(df_raw)
    if d is None or len(d) < 40:
        return {"win_rate": 0.0, "total_return_pct": 0.0, "n_trades": 0, "trades": [], "max_drawdown_pct": 0.0, "sharpe_like": 0.0}

    balance = 1.0
    wins = 0
    trades: List[Dict[str, object]] = []
    equity_curve = [balance]

    for i in range(30, len(d) - 1):
        rp, r = d.iloc[i - 1], d.iloc[i]
        side_rule, rule_prob = compute_signal_row(rp, r)

        # adaptive thresholds per-bar
        prob_th, rr_th = _adaptive_thresholds(r)

        if side_rule == "Hold" or rule_prob < prob_th:
            equity_curve.append(balance)
            continue

        atr = float(r.get("atr", np.nan))
        if np.isnan(atr) or atr <= 0:
            atr = float(d["atr"].iloc[max(0, i - 14):i + 1].mean())

        entry_px = float(r["Close"])
        tp, sl = compute_tp_sl(entry_px, atr, side_rule, risk)
        rr_here = _calc_rr(entry_px, tp, sl, side_rule)

        # RR gate
        if rr_here < rr_th:
            equity_curve.append(balance)
            continue

        # No-chase
        if _is_exhausted(r, side_rule):
            equity_curve.append(balance)
            continue

        # Sizing: base 1.0, pro-trend boost, counter-trend cut
        regime = _get_regime_label(r)
        size = 1.0
        if regime == "bull" and side_rule == "Buy":
            size *= 1.25
        elif regime == "bear" and side_rule == "Sell":
            size *= 1.25
        elif regime == "bull" and side_rule == "Sell":
            size *= 0.75
        elif regime == "bear" and side_rule == "Buy":
            size *= 0.75

        # 1-bar PnL (quick backtest)
        next_px = float(d["Close"].iloc[i + 1])
        ret = (next_px - entry_px) / entry_px if side_rule == "Buy" else (entry_px - next_px) / entry_px
        pnl = ret * size
        if pnl > 0:
            wins += 1

        balance *= (1.0 + pnl)
        equity_curve.append(balance)

        trades.append({
            "index": int(i),
            "side": side_rule,
            "entry": entry_px,
            "exit": next_px,
            "profit_pct": float(pnl * 100.0),
            "rr_at_entry": float(rr_here),
            "regime": regime,
            "prob_gate": float(prob_th),
            "rr_gate": float(rr_th),
        })

    n = len(trades)
    win_rate = float(100.0 * wins / n) if n > 0 else 0.0
    total_return_pct = float((balance - 1.0) * 100.0)
    max_dd_pct, sharpe_like = _compute_drawdown_stats(equity_curve)

    return {
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "n_trades": n,
        "trades": trades,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_like": sharpe_like,
    }


# --------------------------------------------------------------------------------------
# Public API (used by app.py)
# --------------------------------------------------------------------------------------
def analyze_asset(symbol: str, interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> Optional[Dict[str, object]]:
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
        "tp": float(pred["tp"]) if pred["tp"] is not None else None,
        "sl": float(pred["sl"]) if pred["sl"] is not None else None,
        "win_rate": bt["win_rate"],
        "return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "max_drawdown_pct": bt["max_drawdown_pct"],
        "sharpe_like": bt["sharpe_like"],
    }


def summarize_assets(interval_key: str = "1h", risk: str = "Medium", use_cache: bool = True) -> pd.DataFrame:
    _log("Fetching and analyzing market data (smart v6)...")
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
    return pd.DataFrame(rows)


def load_asset_with_indicators(asset: str, interval_key: str, use_cache: bool = True) -> Tuple[str, pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)
    symbol = ASSET_SYMBOLS[asset]
    df_raw = fetch_data(symbol, interval_key, use_cache)
    df_ind = add_indicators(df_raw)
    return symbol, df_ind


def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache: bool = True) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
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
        "probability": float(pred["prob"]),         # fused 0..1
        "rule_prob": float(pred["rule_prob"]),
        "ml_prob": pred["ml_prob"],
        "tp": float(pred["tp"]) if pred["tp"] is not None else None,
        "sl": float(pred["sl"]) if pred["sl"] is not None else None,
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "sentiment": pred["sentiment"],
        "regime": pred["regime"],
        "max_drawdown_pct": bt["max_drawdown_pct"],
        "sharpe_like": bt["sharpe_like"],
        "trades": bt["trades"],
        "atr": float(pred["atr"]),
        "rr": float(pred["rr"]),
        "prob_threshold": float(pred["prob_threshold"]),
        "rr_threshold": float(pred["rr_threshold"]),
    }
    return result, df_ind


# --------------------------------------------------------------------------------------
# END OF FILE (manual quick check mode)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    for asset in ["Gold", "NASDAQ 100", "Bitcoin"]:
        sym = ASSET_SYMBOLS[asset]
        df = fetch_data(sym, "1h")
        pred = latest_prediction(df, sym, "Medium", "1h")
        print(f"{asset}: side={pred['side']}, prob={pred['prob']:.2f}, rr={pred['rr']:.2f}, regime={pred['regime']}")