# utils.py — FINAL SMART (v2) VERSION
# ---------------------------------------------------------------------------
# WoodyTradesPro Forecast Utilities
# ---------------------------------------------------------------------------
# Features:
#   • Robust data fetch + cache (yfinance)
#   • Technical indicators (EMA, RSI, MACD, ATR)
#   • Vader sentiment on Yahoo Finance headlines
#   • Market regime detection (trend vs range)
#   • RandomForest ML classifier for adaptive bias
#   • Fused signal (technicals + sentiment + ML)
#   • Backtest with win rate & total return
#   • Compatible with app.py imports (no changes needed)
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

try:
    from ta.trend import EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = MACD = RSIIndicator = AverageTrueRange = None

# ---------------------------------------------------------------------------
# CONFIGURATION
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
    safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return DATA_DIR / f"{safe}_{interval_key}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf,-np.inf],np.nan).ffill().bfill().dropna()


def fetch_data(symbol: str, interval_key: str="1h", use_cache=True) -> pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(interval_key)
    interval = INTERVALS[interval_key]["interval"]
    period = INTERVALS[interval_key]["period"]
    min_rows = INTERVALS[interval_key]["min_rows"]
    cache_fp = _cache_path(symbol, interval_key)
    if use_cache and cache_fp.exists():
        try:
            df = pd.read_csv(cache_fp, index_col=0, parse_dates=True)
            if len(df) >= min_rows:
                return _normalize(df)
        except Exception:
            pass
    for _ in range(3):
        try:
            raw = yf.download(symbol, period=period, interval=interval,
                              progress=False, threads=False, auto_adjust=True)
            df = _normalize(raw)
            if len(df) >= min_rows:
                df.to_csv(cache_fp)
                return df
        except Exception:
            time.sleep(2)
    try:
        tk = yf.Ticker(symbol)
        raw = tk.history(period=period, interval=interval, auto_adjust=True)
        df = _normalize(raw)
        if not df.empty:
            df.to_csv(cache_fp)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    try:
        df["ema20"] = EMAIndicator(df["Close"], 20).ema_indicator()
        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    except Exception:
        df["ema20"] = df["Close"].ewm(span=20).mean()
        df["ema50"] = df["Close"].ewm(span=50).mean()
    try:
        df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    except Exception:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
    try:
        macd = MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception:
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
    try:
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14)
        df["atr"] = atr.average_true_range()
    except Exception:
        tr = pd.concat([
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
    return df.ffill().bfill()

# ---------------------------------------------------------------------------
# SENTIMENT + REGIME + MACHINE LEARNING
# ---------------------------------------------------------------------------

_sent = SentimentIntensityAnalyzer()

def fetch_sentiment(symbol: str) -> float:
    try:
        tk = yf.Ticker(symbol)
        news = tk.news
        if not news:
            return 0.0
        scores = [_sent.polarity_scores(n["title"])["compound"] for n in news if "title" in n]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def detect_regime(df: pd.DataFrame) -> str:
    if "ema20" not in df or "ema50" not in df:
        return "range"
    spread = (df["ema20"] - df["ema50"]).abs() / df["Close"]
    dom = max((df["ema20"] > df["ema50"]).mean(), (df["ema20"] < df["ema50"]).mean())
    return "trend" if spread.tail(100).mean() > 0.002 and dom > 0.7 else "range"


def train_rf(df: pd.DataFrame) -> Optional[RandomForestClassifier]:
    cols = ["ema20","ema50","RSI","macd","macd_signal","atr","sentiment"]
    df = df.copy()
    df["fwd"] = df["Close"].pct_change().shift(-1)
    df["up"] = (df["fwd"] > 0).astype(int)
    X = df[cols].shift(1)
    y = df["up"]
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 200:
        return None
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X[mask], y[mask])
    return model

# ---------------------------------------------------------------------------
# SIGNAL ENGINE + BACKTEST
# ---------------------------------------------------------------------------

def compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float,float]:
    m = RISK_MULT.get(risk, RISK_MULT["Medium"])
    if side == "Buy":
        return price + m["tp_atr"] * atr, price - m["sl_atr"] * atr
    else:
        return price - m["tp_atr"] * atr, price + m["sl_atr"] * atr


def fused_signal(df: pd.DataFrame, symbol: str, risk: str="Medium") -> Optional[Dict[str,object]]:
    if df.empty:
        return None
    df = add_indicators(df)
    sent = fetch_sentiment(symbol)
    df["sentiment"] = sent
    regime = detect_regime(df)
    model = train_rf(df)

    last = df.iloc[-1]
    score = 0
    if last["ema20"] > last["ema50"]:
        score += 1
    if last["RSI"] < 40:
        score += 1
    if last["RSI"] > 60:
        score -= 1
    if last["macd"] > last["macd_signal"]:
        score += 1
    elif last["macd"] < last["macd_signal"]:
        score -= 1
    side_rule = "Buy" if score > 0 else "Sell" if score < 0 else "Hold"

    prob_up = 0.5
    if model is not None:
        feat = pd.DataFrame([{
            "ema20": last.get("ema20"),
            "ema50": last.get("ema50"),
            "RSI": last.get("RSI"),
            "macd": last.get("macd"),
            "macd_signal": last.get("macd_signal"),
            "atr": last.get("atr"),
            "sentiment": sent,
        }])
        prob_up = float(model.predict_proba(feat)[0, 1])

    votes_buy = votes_sell = 0
    if side_rule == "Buy":
        votes_buy += 1
    elif side_rule == "Sell":
        votes_sell += 1
    if sent > 0.1:
        votes_buy += 1
    elif sent < -0.1:
        votes_sell += 1
    if prob_up > 0.55:
        votes_buy += 1
    elif prob_up < 0.45:
        votes_sell += 1

    final_side = "Buy" if votes_buy > votes_sell else "Sell" if votes_sell > votes_buy else "Hold"
    conf = min(1.0, abs(votes_buy - votes_sell) / 3 + abs(prob_up - 0.5) * 2)

    atr = float(last["atr"])
    price = float(last["Close"])
    tp, sl = compute_tp_sl(price, atr, final_side, risk)
    return {
        "side": final_side,
        "prob": conf,
        "price": price,
        "tp": tp,
        "sl": sl,
        "atr": atr,
        "regime": regime,
        "sentiment": sent,
        "ml_prob_up": prob_up,
    }


def backtest(df: pd.DataFrame, symbol: str, risk: str="Medium") -> Dict[str,object]:
    res = {"win_rate": 0, "total_return_pct": 0, "n_trades": 0, "trades": []}
    if len(df) < 120:
        return res
    df = add_indicators(df)
    df["sentiment"] = fetch_sentiment(symbol)
    pos = None
    wins = ret_sum = 0
    trades = []
    for i in range(60, len(df)):
        sig = fused_signal(df.iloc[:i+1], symbol, risk)
        if not sig:
            continue
        side = sig["side"]
        px = float(df["Close"].iloc[i])
        ts = df.index[i]
        if pos is None and side in ("Buy","Sell"):
            pos = (side, px, ts)
        elif pos is not None:
            ps, ep, ets = pos
            if side in ("Buy","Sell") and side != ps:
                r = (px - ep) / ep * (1 if ps == "Buy" else -1)
                ret_sum += r
                if r > 0:
                    wins += 1
                trades.append({
                    "entry_time": ets,
                    "exit_time": ts,
                    "side": ps,
                    "entry": ep,
                    "exit": px,
                    "return_pct": r * 100,
                })
                pos = (side, px, ts)
    if pos is not None:
        ps, ep, ets = pos
        lp = float(df["Close"].iloc[-1])
        lts = df.index[-1]
        r = (lp - ep) / ep * (1 if ps == "Buy" else -1)
        ret_sum += r
        if r > 0:
            wins += 1
        trades.append({
            "entry_time": ets,
            "exit_time": lts,
            "side": ps,
            "entry": ep,
            "exit": lp,
            "return_pct": r * 100,
        })
    n = len(trades)
    res.update({
        "n_trades": n,
        "win_rate": 100 * wins / n if n else 0,
        "total_return_pct": 100 * ret_sum,
        "trades": trades,
    })
    return res

# ---------------------------------------------------------------------------
# PIPELINES (APP INTERFACE)
# ---------------------------------------------------------------------------

def analyze_asset(symbol: str, interval_key: str, risk: str="Medium", use_cache=True) -> Optional[Dict[str,object]]:
    df = fetch_data(symbol, interval_key, use_cache)
    if df.empty:
        return None
    df = add_indicators(df)
    pred = fused_signal(df, symbol, risk)
    bt = backtest(df, symbol, risk)
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
        "regime": pred["regime"],
        "sentiment": pred["sentiment"],
        "ml_prob_up": round(pred["ml_prob_up"] * 100, 2),
        "win_rate": bt["win_rate"],
        "total_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "df": df,
        "trades": bt["trades"],
    }


def summarize_assets(interval_key: str="1h", risk: str="Medium", use_cache=True) -> pd.DataFrame:
    rows = []
    _log("Fetching and analyzing market data (smart v2)...")
    for asset, symbol in ASSET_SYMBOLS.items():
        _log(f"{asset} ({symbol})...")
        try:
            r = analyze_asset(symbol, interval_key, risk, use_cache)
            if r:
                rows.append({
                    "Asset": asset,
                    "Symbol": symbol,
                    "Interval": interval_key,
                    "Price": r["last_price"],
                    "Signal": r["signal"],
                    "Probability_%": r["probability"],
                    "TP": r["tp"],
                    "SL": r["sl"],
                    "WinRate_%": r["win_rate"],
                    "BacktestReturn_%": r["total_return_pct"],
                    "Trades": r["n_trades"],
                    "Regime": r["regime"],
                    "Sentiment": r["sentiment"],
                    "ML_ProbUp_%": r["ml_prob_up"],
                })
        except Exception as e:
            _log(f"Error analyzing {asset}: {e}")
    return pd.DataFrame(rows)


def load_asset_with_indicators(asset: str, interval_key: str, use_cache=True) -> Tuple[str, pd.DataFrame]:
    if asset not in ASSET_SYMBOLS:
        raise KeyError(asset)
    sym = ASSET_SYMBOLS[asset]
    df = fetch_data(sym, interval_key, use_cache)
    return sym, add_indicators(df)


def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str, use_cache=True) -> Tuple[Optional[Dict[str,object]], pd.DataFrame]:
    sym = ASSET_SYMBOLS.get(asset)
    if not sym:
        return None, pd.DataFrame()
    df = fetch_data(sym, interval_key, use_cache)
    if df.empty:
        return None, df
    df = add_indicators(df)
    pred = fused_signal(df, sym, risk)
    bt = backtest(df, sym, risk)
    if not pred:
        return None, df
    return {
        "asset": asset,
        "symbol": sym,
        "interval": interval_key,
        "price": float(df["Close"].iloc[-1]),
        "side": pred["side"],
        "probability": round(pred["prob"] * 100, 2),
        "tp": pred["tp"],
        "sl": pred["sl"],
        "atr": pred["atr"],
        "regime": pred["regime"],
        "sentiment": pred["sentiment"],
        "ml_prob_up": round(pred["ml_prob_up"] * 100, 2),
        "win_rate": bt["win_rate"],
        "backtest_return_pct": bt["total_return_pct"],
        "n_trades": bt["n_trades"],
        "trades": bt["trades"],
    }, df

# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------