# utils.py - FINAL MERGED VERSION (v1 + v2)
# ---------------------------------------------------------------------------
# Includes:
# - Robust data fetch, caching, indicators
# - v1: basic technical signal engine (EMA, RSI, MACD, ATR)
# - v1: basic backtest and pipelines
# - v2: sentiment, market regime detection, ML classifier (RandomForest)
# - v2: fused adaptive signal + improved backtest + summary
#
# This file is ASCII-only (no curly quotes / smart dashes), Python 3.13 safe.
# ---------------------------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Technical indicators (ta)
try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = MACD = RSIIndicator = AverageTrueRange = None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d", "min_rows": 150},
    "1h":  {"interval": "60m", "period": "2mo", "min_rows": 200},
    "4h":  {"interval": "240m", "period": "3mo", "min_rows": 200},
    "1d":  {"interval": "1d", "period": "1y", "min_rows": 150},
    "1wk": {"interval": "1wk", "period": "5y", "min_rows": 100},
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

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def _cache_path(symbol: str, interval_key: str) -> Path:
    safe_sym = (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return DATA_DIR / f"{safe_sym}_{interval_key}.csv"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns like ('Open','GC=F')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # keep common OHLCV columns
    keep = [
        c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    if not keep:
        # fallback: capitalize columns
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        keep = [
            c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if c in df.columns
        ]
    df = df[keep].copy()

    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    # coerce numeric and flatten any (n,1) arrays
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # clean NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="all")
    return df


def _yahoo_try_download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            threads=False
        )
        return _normalize_ohlcv(raw)
    except Exception as e:
        _log(f"Fetch error for {symbol}: {e}")
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
        _log(f"Mirror fetch error for {symbol}: {e}")
        return pd.DataFrame()


def fetch_data(
    symbol: str,
    interval_key: str = "1h",
    use_cache: bool = True,
    max_retries: int = 4,
    backoff_range: Tuple[float, float] = (3.5, 12.5),
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol/interval with caching and retry fallback.
    """
    if interval_key not in INTERVALS:
        raise KeyError(f"Unknown interval_key {interval_key}")

    interval = INTERVALS[interval_key]["interval"]
    period = INTERVALS[interval_key]["period"]
    min_rows = INTERVALS[interval_key]["min_rows"]

    cache_fp = _cache_path(symbol, interval_key)
    _log(f"Fetching {symbol} [{interval}] for {period}...")

    # 1. Read cache if valid
    if use_cache and cache_fp.exists():
        try:
            cached = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if len(cached) >= min_rows:
                _log(f"Using cached {symbol} ({len(cached)} rows).")
                return cached
        except Exception as e:
            _log(f"Cache read failed: {e}")

    # 2. Try live download with retries
    for attempt in range(1, max_retries + 1):
        df = _yahoo_try_download(symbol, interval, period)
        if not df.empty and len(df) >= min_rows:
            _log(f"{symbol}: fetched {len(df)} rows.")
            try:
                df.to_csv(cache_fp)
            except Exception:
                pass
            return df
        _log(f"Retry {attempt} failed for {symbol} ({len(df)} rows).")
        time.sleep(np.random.uniform(*backoff_range))

    # 3. Fallback mirror fetch
    _log(f"Trying mirror fetch for {symbol}...")
    df = _yahoo_mirror_history(symbol, interval, period)
    if not df.empty and len(df) >= min_rows:
        try:
            df.to_csv(cache_fp)
        except Exception:
            pass
        return df

    _log(f"All fetch attempts failed for {symbol}.")
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ema20, ema50
      - RSI (14)
      - MACD, MACD signal
      - ATR(14)
    Ensures columns are numeric and NaNs are handled.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EMA
    try:
        df["ema20"] = EMAIndicator(df["Close"], 20).ema_indicator()
        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    except Exception:
        df["ema20"] = df["Close"].ewm(span=20).mean()
        df["ema50"] = df["Close"].ewm(span=50).mean()

    # RSI
    try:
        df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    except Exception:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["RSI"]

    # MACD
    try:
        macd = MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception:
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # ATR
    try:
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14)
        df["atr"] = atr.average_true_range()
    except Exception:
        tr1 = (df["High"] - df["Low"]).abs()
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"] - df["Close"].shift(1)).abs()
        df["atr"] = (
            pd.concat([tr1, tr2, tr3], axis=1)
            .max(axis=1)
            .rolling(14)
            .mean()
        )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ---------------------------------------------------------------------------
# SIGNAL ENGINE (v1)
# ---------------------------------------------------------------------------

def compute_signal_row(row_prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Original rule-based signal:
    - EMA20 vs EMA50 trend bias
    - RSI oversold/overbought
    - MACD cross
    Returns (side, confidence)
    """
    score, votes = 0.0, 0

    # EMA trend
    if pd.notna(row.get("ema20")) and pd.notna(row.get("ema50")):
        votes += 1
        score += 1 if row["ema20"] > row["ema50"] else -1

    # RSI context
    if pd.notna(row.get("RSI")):
        votes += 1
        if row["RSI"] < 40:
            score += 1
        elif row["RSI"] > 60:
            score -= 1

    # MACD cross info
    if (
        pd.notna(row.get("macd"))
        and pd.notna(row.get("macd_signal"))
        and pd.notna(row_prev.get("macd"))
        and pd.notna(row_prev.get("macd_signal"))
    ):
        votes += 1
        crossed_up = (
            row_prev["macd"] <= row_prev["macd_signal"]
            and row["macd"] > row["macd_signal"]
        )
        crossed_dn = (
            row_prev["macd"] >= row_prev["macd_signal"]
            and row["macd"] < row["macd_signal"]
        )
        if crossed_up:
            score += 1
        elif crossed_dn:
            score -= 1

    # normalize
    conf = 0 if votes == 0 else abs(score) / votes

    # decision
    if score > 0:
        return "Buy", conf
    elif score < 0:
        return "Sell", conf
    return "Hold", 0.2

# ---------------------------------------------------------------------------
# TP/SL AND PREDICTION (v1)
# ---------------------------------------------------------------------------

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Translate ATR into suggested TP/SL levels for display.
    """
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    if side not in ("Buy", "Sell"):
        side = "Buy"
    tp = price + m["tp_atr"] * atr if side == "Buy" else price - m["tp_atr"] * atr
    sl = price - m["sl_atr"] * atr if side == "Buy" else price + m["sl_atr"] * atr
    return float(tp), float(sl)


def latest_prediction(df: pd.DataFrame, risk: str = "Medium") -> Optional[Dict[str, object]]:
    """
    v1 prediction:
    - uses compute_signal_row on last 2 rows
    - returns side/probability/tp/sl/atr
    """
    if df is None or df.empty or len(df) < 30:
        return None

    df = add_indicators(df)
    if df.empty:
        return None

    row_prev, row = df.iloc[-2], df.iloc[-1]
    side, conf_raw = compute_signal_row(row_prev, row)

    atr_val = (
        float(row["atr"])
        if ("atr" in row and pd.notna(row["atr"]))
        else float(df["atr"].tail(14).mean())
    )
    price_now = float(row["Close"])

    tp_val, sl_val = compute_tp_sl(price_now, atr_val, side, risk)

    # UI prob: never below 5%
    prob_pct = max(5.0, round(conf_raw * 100.0, 2))

    return {
        "side": side,
        "prob": prob_pct / 100.0,  # store internally as 0..1
        "price": price_now,
        "tp": tp_val,
        "sl": sl_val,
        "atr": atr_val,
    }

# ---------------------------------------------------------------------------
# BACKTEST (v1)
# ---------------------------------------------------------------------------

def backtest_signals(
    df: pd.DataFrame,
    risk: str = "Medium",
    hold_allowed: bool = True,
) -> Dict[str, object]:
    """
    Simple backtest with v1 rules.
    Enter on Buy/Sell, flip on opposite, force close at end.
    """
    out = {
        "win_rate": 0.0,
        "total_return_pct": 0.0,
        "n_trades": 0,
        "trades": [],
    }

    if df is None or df.empty or len(df) < 60:
        return out

    df = add_indicators(df)
    if df.empty:
        return out

    prev = df.iloc[0]
    position = None  # (side, entry_px, tp, sl, entry_ts)
    trades: List[Dict[str, object]] = []
    wins = 0
    total_ret = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        side, _ = compute_signal_row(prev, row)

        atr_here = (
            float(row["atr"])
            if pd.notna(row.get("atr"))
            else float(df["atr"].iloc[max(0, i - 14):i + 1].mean())
        )
        tp, sl = compute_tp_sl(row["Close"], atr_here, side, risk)

        if position is None and side in ("Buy", "Sell"):
            position = (side, float(row["Close"]), tp, sl, row.name)

        elif position is not None:
            pos_side, entry_px, _, _, entry_ts = position
            if side in ("Buy", "Sell") and side != pos_side:
                exit_px = float(row["Close"])
                ret = (exit_px - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
                total_ret += ret
                if ret > 0:
                    wins += 1
                trades.append({
                    "entry_time": entry_ts,
                    "exit_time": row.name,
                    "side": pos_side,
                    "entry": entry_px,
                    "exit": exit_px,
                    "reason": "Flip",
                    "return_pct": ret * 100.0,
                })
                position = None

        prev = row

    # force close final position
    if position is not None:
        pos_side, entry_px, _, _, entry_ts = position
        last_close = float(df["Close"].iloc[-1])
        last_ts = df.index[-1]
        ret = (last_close - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
        total_ret += ret
        if ret > 0:
            wins += 1
        trades.append({
            "entry_time": entry_ts,
            "exit_time": last_ts,
            "side": pos_side,
            "entry": entry_px,
            "exit": last_close,
            "reason": "EoS",
            "return_pct": ret * 100.0,
        })

    n = len(trades)
    out["n_trades"] = n
    out["win_rate"] = 100.0 * wins / n if n else 0.0
    out["total_return_pct"] = 100.0 * total_ret if n else 0.0
    out["trades"] = trades
    return out

# ---------------------------------------------------------------------------
# PIPELINES FOR APP (v1)
# ---------------------------------------------------------------------------

def analyze_asset(
    symbol: str,
    interval_key: str,
    risk: str = "Medium",
    use_cache: bool = True,
) -> Optional[Dict[str, object]]:
    """
    v1 pipeline, used in the original app:
      fetch -> indicators -> latest_prediction -> backtest_signals
    """
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None

    df = add_indicators(df)
    pred = latest_prediction(df, risk)
    bt = backtest_signals(df, risk)

    if not pred:
        return None

    return {
        "symbol": symbol,
        "interval_key": interval_key,
        "risk": risk,
        "last_price": float(df["Close"].iloc[-1]),
        "signal": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],
        "win_rate": bt["win_rate"],
        "total_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "df": df,
        "trades": bt["trades"],
    }


def summarize_assets(
    interval_key: str = "1h",
    risk: str = "Medium",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build summary table across all assets using v1 logic.
    """
    rows = []
    _log("Fetching and analyzing market data... please wait")
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"Analyzing {asset} ({symbol}) ...")
        try:
            res = analyze_asset(symbol, interval_key, risk, use_cache)
            if res:
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
            _log(f"Error analyzing {asset}: {e}")

    if not rows:
        return pd.DataFrame()

    cols = [
        "Asset",
        "Symbol",
        "Interval",
        "Price",
        "Signal",
        "Probability_%",
        "TP",
        "SL",
        "WinRate_%",
        "BacktestReturn_%",
        "Trades",
    ]
    return pd.DataFrame(rows)[cols]


def load_asset_with_indicators(
    asset: str,
    interval_key: str,
    use_cache: bool = True
) -> Tuple[str, pd.DataFrame]:
    """
    For tabs where one asset is selected and plotted, etc.
    """
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset {asset}")
    symbol = ASSET_SYMBOLS[asset]
    df = fetch_data(symbol, interval_key, use_cache)
    df = add_indicators(df)
    return symbol, df


def asset_prediction_and_backtest(
    asset: str,
    interval_key: str,
    risk: str,
    use_cache: bool = True
) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
    """
    For scenario / drilldown tabs in v1.
    """
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()

    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None, pd.DataFrame()

    df = add_indicators(df)
    pred = latest_prediction(df, risk)
    bt = backtest_signals(df, risk)

    if not pred:
        return None, df

    pred_out = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "trades": bt["trades"],
    }
    return pred_out, df

# ---------------------------------------------------------------------------
# v2 ENGINE: SENTIMENT, REGIME, ML, FUSED SIGNAL
# ---------------------------------------------------------------------------

_sentiment_analyzer = SentimentIntensityAnalyzer()

def fetch_recent_headline_sentiment(symbol: str, max_headlines: int = 15) -> float:
    """
    Pull recent headlines via yfinance.Ticker(...).news and compute
    mean VADER compound sentiment in [-1,1].
    If no headlines, return 0.0 (neutral).
    """
    try:
        tk = yf.Ticker(symbol)
        news_items = getattr(tk, "news", [])
        if not news_items:
            return 0.0
        scores = []
        for item in news_items[:max_headlines]:
            title = item.get("title", "")
            if not title:
                continue
            vs = _sentiment_analyzer.polarity_scores(title)
            scores.append(vs["compound"])
        if not scores:
            return 0.0
        return float(np.mean(scores))
    except Exception as e:
        _log(f"sentiment fetch failed for {symbol}: {e}")
        return 0.0


def attach_sentiment_feature(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add a constant 'sentiment' column to df representing current headline tone.
    """
    df = df.copy()
    sent_score = fetch_recent_headline_sentiment(symbol)
    df["sentiment"] = sent_score
    return df


def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Decide if the market is 'trend' or 'range' using EMA separation and dominance.
    """
    if df.empty or "ema20" not in df or "ema50" not in df:
        return "range"

    spread = (df["ema20"] - df["ema50"]).abs()
    spread_norm = spread / df["Close"]
    spread_mean = float(spread_norm.tail(100).mean())

    above = (df["ema20"] > df["ema50"]).tail(100)
    dominance = max(above.mean(), 1.0 - above.mean())

    if spread_mean > 0.002 and dominance > 0.7:
        return "trend"
    else:
        return "range"


def train_direction_model(df: pd.DataFrame) -> Tuple[Optional[RandomForestClassifier], pd.DataFrame]:
    """
    Train a random forest to predict if next bar's close is up (>0 return).
    Returns (model, df_augmented).
    """
    use_cols = [
        "ema20", "ema50", "RSI", "macd", "macd_signal",
        "atr", "sentiment"
    ]

    df_feat = df.copy()
    df_feat["fwd_ret"] = df_feat["Close"].pct_change().shift(-1)
    df_feat["up_next"] = (df_feat["fwd_ret"] > 0).astype(int)

    X = df_feat[use_cols].shift(1)
    y = df_feat["up_next"]

    valid_mask = X.notna().all(axis=1) & y.notna()
    X_train = X[valid_mask]
    y_train = y[valid_mask]

    if len(X_train) < 200:
        return None, df_feat

    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model, df_feat
    except Exception as e:
        _log(f"model training failed: {e}")
        return None, df_feat


def predict_direction_prob(model: Optional[RandomForestClassifier], last_row: pd.Series) -> float:
    """
    Given last_row of features, estimate probability price goes UP next bar.
    Fallback 0.5 if not confident / no model.
    """
    if model is None:
        return 0.5

    feat_vec = np.array([[
        last_row.get("ema20", np.nan),
        last_row.get("ema50", np.nan),
        last_row.get("RSI", np.nan),
        last_row.get("macd", np.nan),
        last_row.get("macd_signal", np.nan),
        last_row.get("atr", np.nan),
        last_row.get("sentiment", 0.0),
    ]], dtype=float)

    if np.isnan(feat_vec).any():
        return 0.5

    try:
        proba = model.predict_proba(feat_vec)[0, 1]
        return float(proba)
    except Exception:
        return 0.5


def fused_signal(df: pd.DataFrame, symbol: str, risk: str = "Medium") -> Optional[Dict[str, object]]:
    """
    Produce final trading view for latest bar using:
      - technical rule (compute_signal_row)
      - regime (trend or range)
      - sentiment
      - ML probability of up move

    Returns:
      {
        "side": "Buy"/"Sell"/"Hold",
        "prob": 0..1,
        "price": float,
        "tp": float,
        "sl": float,
        "atr": float,
        "regime": "trend"/"range",
        "sentiment": float,
        "ml_prob_up": float,
        "rule_side": str
      }
    """
    if df is None or df.empty or len(df) < 60:
        return None

    # indicators should already exist, but be safe
    df = add_indicators(df)

    # attach sentiment column
    df = attach_sentiment_feature(df, symbol)

    # market regime
    regime = detect_market_regime(df)

    # train ML on full df
    model, df_aug = train_direction_model(df)
    last_row_prev = df_aug.iloc[-2]
    last_row = df_aug.iloc[-1]

    # rule-based bias
    side_rule, conf_rule = compute_signal_row(last_row_prev, last_row)

    # sentiment bias
    sent_score = float(last_row.get("sentiment", 0.0))
    if sent_score > 0.1:
        sent_bias = "Buy"
    elif sent_score < -0.1:
        sent_bias = "Sell"
    else:
        sent_bias = "Neutral"

    # model probability for "next bar up"
    prob_up = predict_direction_prob(model, last_row)

    # vote
    votes_buy = 0.0
    votes_sell = 0.0

    if side_rule == "Buy":
        votes_buy += 1
    elif side_rule == "Sell":
        votes_sell += 1

    if sent_bias == "Buy":
        votes_buy += 1
    elif sent_bias == "Sell":
        votes_sell += 1

    if prob_up > 0.55:
        votes_buy += 1
    elif prob_up < 0.45:
        votes_sell += 1

    # regime weighting
    if regime == "trend":
        # trust the momentum side_rule more in trends
        if side_rule == "Buy":
            votes_buy += 0.5
        elif side_rule == "Sell":
            votes_sell += 0.5
    else:
        # in ranges, RSI extremes matter
        rsi_last = float(last_row.get("RSI", 50.0))
        if rsi_last < 35:
            votes_buy += 0.5
        elif rsi_last > 65:
            votes_sell += 0.5

    # final side
    if votes_buy - votes_sell > 0.5:
        final_side = "Buy"
    elif votes_sell - votes_buy > 0.5:
        final_side = "Sell"
    else:
        final_side = "Hold"

    # final probability/confidence in 0..1, blended from multiple signals
    margin = abs(votes_buy - votes_sell) / 3.0  # rough normalization
    ml_conf = abs(prob_up - 0.5) * 2.0          # 0..1
    blended_conf = 0.4 * conf_rule + 0.4 * ml_conf + 0.2 * margin
    blended_conf = max(0.05, min(1.0, blended_conf))

    # TP/SL
    atr_val = float(last_row.get("atr", np.nan))
    if np.isnan(atr_val):
        atr_val = float(df_aug["atr"].tail(14).mean())
    price_now = float(last_row["Close"])

    disp_side_for_levels = final_side if final_side in ["Buy", "Sell"] else "Buy"
    tp_val, sl_val = compute_tp_sl(price_now, atr_val, disp_side_for_levels, risk)

    return {
        "side": final_side,
        "prob": blended_conf,
        "price": price_now,
        "tp": float(tp_val),
        "sl": float(sl_val),
        "atr": float(atr_val),
        "regime": regime,
        "sentiment": sent_score,
        "ml_prob_up": prob_up,
        "rule_side": side_rule,
    }

# ---------------------------------------------------------------------------
# BACKTEST (v2, fused)
# ---------------------------------------------------------------------------

def backtest_fused(
    df: pd.DataFrame,
    symbol: str,
    risk: str = "Medium"
) -> Dict[str, object]:
    """
    Walk forward through time, recomputing fused_signal() with only
    data up to that bar.
    Enter long if side=Buy, short if side=Sell.
    Flip on reversal. Force close at the end.
    """
    results = {
        "win_rate": 0.0,
        "total_return_pct": 0.0,
        "n_trades": 0,
        "trades": [],
    }

    if df is None or df.empty or len(df) < 120:
        return results

    df_full = df.copy()
    df_full = add_indicators(df_full)

    position = None  # (side, entry_px, entry_time)
    trades: List[Dict[str, object]] = []
    wins = 0
    equity_sum = 0.0

    start_i = 60
    for i in range(start_i, len(df_full)):
        window = df_full.iloc[: i+1].copy()

        fused = fused_signal(window, symbol, risk=risk)
        if fused is None:
            continue

        this_side = fused["side"]
        now_px = float(window["Close"].iloc[-1])
        now_ts = window.index[-1]

        if position is None:
            if this_side in ("Buy", "Sell"):
                position = (this_side, now_px, now_ts)
        else:
            pos_side, entry_px, entry_ts = position
            # flip if signal changes
            if this_side in ("Buy", "Sell") and this_side != pos_side:
                ret = (now_px - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
                equity_sum += ret
                if ret > 0:
                    wins += 1
                trades.append({
                    "entry_time": entry_ts,
                    "exit_time": now_ts,
                    "side": pos_side,
                    "entry": entry_px,
                    "exit": now_px,
                    "reason": "Flip",
                    "return_pct": ret * 100.0,
                })
                position = (this_side, now_px, now_ts)

    # force close at end
    if position is not None:
        pos_side, entry_px, entry_ts = position
        last_px = float(df_full["Close"].iloc[-1])
        last_ts = df_full.index[-1]
        ret = (last_px - entry_px) / entry_px * (1 if pos_side == "Buy" else -1)
        equity_sum += ret
        if ret > 0:
            wins += 1
        trades.append({
            "entry_time": entry_ts,
            "exit_time": last_ts,
            "side": pos_side,
            "entry": entry_px,
            "exit": last_px,
            "reason": "EoS",
            "return_pct": ret * 100.0,
        })

    n = len(trades)
    results["n_trades"] = n
    results["win_rate"] = 100.0 * wins / n if n else 0.0
    results["total_return_pct"] = 100.0 * equity_sum if n else 0.0
    results["trades"] = trades
    return results

# ---------------------------------------------------------------------------
# PIPELINES FOR APP (v2)
# ---------------------------------------------------------------------------

def analyze_asset_v2(
    symbol: str,
    interval_key: str,
    risk: str = "Medium",
    use_cache: bool = True,
) -> Optional[Dict[str, object]]:
    """
    High-level per-asset analysis for dashboard rows (v2).
    Uses fused_signal() and backtest_fused() instead of v1 logic.
    """
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None

    df = add_indicators(df)

    pred = fused_signal(df, symbol, risk=risk)
    bt = backtest_fused(df, symbol, risk=risk)

    if pred is None:
        return None

    out = {
        "symbol": symbol,
        "interval_key": interval_key,
        "risk": risk,
        "last_price": float(df["Close"].iloc[-1]),

        "signal": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],

        "regime": pred["regime"],
        "sentiment": pred["sentiment"],
        "ml_prob_up": pred["ml_prob_up"],
        "rule_side": pred["rule_side"],

        "win_rate": bt["win_rate"],
        "total_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "df": df,
        "trades": bt["trades"],
    }
    return out


def summarize_assets_v2(
    interval_key: str = "1h",
    risk: str = "Medium",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build a table across all assets using the smarter fused model.
    You can render this instead of summarize_assets() in the Overview tab.
    """
    rows = []
    _log("Fetching and analyzing market data (v2)...")
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"[v2] {asset} ({symbol}) ...")
        try:
            res = analyze_asset_v2(symbol, interval_key, risk, use_cache)
            if res:
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
                    "Regime": res["regime"],
                    "Sentiment": res["sentiment"],
                    "ML_ProbUp_%": round(float(res["ml_prob_up"]) * 100.0, 2),
                })
        except Exception as e:
            _log(f"v2 analysis failed for {asset}: {e}")

    if not rows:
        return pd.DataFrame()

    cols = [
        "Asset",
        "Symbol",
        "Interval",
        "Price",
        "Signal",
        "Probability_%",
        "TP",
        "SL",
        "WinRate_%",
        "BacktestReturn_%",
        "Trades",
        "Regime",
        "Sentiment",
        "ML_ProbUp_%",
    ]
    return pd.DataFrame(rows)[cols]


def asset_prediction_and_backtest_v2(
    asset: str,
    interval_key: str,
    risk: str,
    use_cache: bool = True,
) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
    """
    For scenario / drilldown tabs in v2.
    Includes sentiment, regime, ML prob, fused trades.
    """
    symbol = ASSET_SYMBOLS.get(asset)
    if not symbol:
        return None, pd.DataFrame()

    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None, pd.DataFrame()

    df = add_indicators(df)
    pred = fused_signal(df, symbol, risk=risk)
    bt = backtest_fused(df, symbol, risk=risk)

    if pred is None:
        return None, df

    pred_out = {
        "asset": asset,
        "symbol": symbol,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),

        "side": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],

        "regime": pred["regime"],
        "sentiment": pred["sentiment"],
        "ml_prob_up": round(float(pred["ml_prob_up"]) * 100.0, 2),
        "rule_side": pred["rule_side"],

        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "trades": bt["trades"],
    }
    return pred_out, df

# ---------------------------------------------------------------------------
# END OF MODULE
# ---------------------------------------------------------------------------