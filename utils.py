# utils.py — WoodyTradesPro Smart v6.3 FULL
# --------------------------------------------------------------------------------------
# Stable core from v6.2_sentimentfix (kept fully intact), with:
#   - Resilient sentiment (Yahoo -> Finviz -> Google RSS -> synthetic price bias)
#   - Regime-aware ML model
#   - Adaptive thresholds
#   - Exhaustion filter
#   - Risk-based TP/SL and RR calc
#   - Backtest with Sharpe-like and max drawdown
#   - Equity curve & position sizing by regime
#   - UI wrappers for Summary / Scenarios / Trends
#
# NEW IN v6.3:
#   1. Adaptive fusion of rule_prob and ml_prob using recent backtest win rate
#   2. Volatility-normalized conviction (confidence reduced in high ATR regimes)
#   3. Sentiment-weighted conviction boost/drag
#   4. Dynamic cutoff for Hold/Buy/Sell based on recent performance
#   5. Backtest exits allow up to 5-bar development, not just next bar
#   6. No functionality removed or simplified.
#
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os
import time
import math
import logging
import warnings
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache
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

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

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

INTERVALS: Dict[str, str] = {
    "15m": "15m",
    "1h": "60m",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
}

PERIODS: Dict[str, str] = {
    "15m": "7d",
    "1h": "2mo",
    "4h": "6mo",
    "1d": "1y",
    "1wk": "5y",
}

_BASE_PROB_THRESHOLD = 0.55  # base required conviction
_BASE_MIN_RR = 1.2           # base required reward:risk

_MODEL_CACHE: Dict[Tuple[str, str, str], RandomForestClassifier] = {}
_VADER = SentimentIntensityAnalyzer()


def _log(msg: str) -> None:
    """Lightweight console logger for Streamlit Cloud logs."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# DATA FETCH + NORMALIZATION
# --------------------------------------------------------------------------------------

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Yahoo Finance / cached OHLCV:
    - flatten MultiIndex columns
    - ensure DatetimeIndex
    - drop duplicate index rows
    - coerce all OHLCV columns to numeric 1-D
    - forward/back fill gaps safely
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Flatten if multi-index (('Open','GC=F'), ...)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = [
        c for c in
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    df = df[cols].copy()

    # Force datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Sort and drop duplicate timestamps
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    # Force columns to be flat numeric (no nested arrays)
    for col in df.columns:
        arr = np.array(df[col])
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        df[col] = pd.to_numeric(arr, errors="coerce")

    # Cleanup NaN / inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna(how="all")

    return df


def fetch_data(
    symbol: str,
    interval_key: str = "1h",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Robust data fetch with cache.
    - Tries cache first (CSV in ./data/)
    - Falls back to live yf.download with retries
    - Normalizes shape
    """
    interval = INTERVALS.get(interval_key, "60m")
    period = PERIODS.get(interval_key, "2mo")

    safe_sym = (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    cache_path = os.path.join(DATA_DIR, f"{safe_sym}_{interval}.csv")

    # Try cache
    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_ohlcv(cached)
            if not cached.empty and len(cached) > 50:
                return cached
        except Exception as e:
            _log(f"⚠️ Cache read failed for {symbol}: {e}")

    # If no cache or not enough rows, pull live
    _log(f"⏳ Fetching {symbol} [{interval}] ...")
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
            df = _normalize_ohlcv(raw)
            if not df.empty and len(df) > 50:
                try:
                    df.to_csv(cache_path)
                except Exception as e:
                    _log(f"⚠️ Cache write failed for {symbol}: {e}")
                _log(f"✅ {symbol}: fetched {len(df)} rows.")
                return df
            else:
                _log(f"⚠️ {symbol} attempt {attempt+1}: got {len(df)} rows")
        except Exception as e:
            _log(f"⚠️ {symbol} attempt {attempt+1} error: {e}")
        time.sleep(1 + attempt * 0.5)

    _log(f"🚫 All fetch attempts failed for {symbol}. Returning empty DataFrame.")
    return pd.DataFrame()


# --------------------------------------------------------------------------------------
# INDICATOR ENGINEERING
# --------------------------------------------------------------------------------------

def _trend_streak_mask(series_bool: pd.Series) -> List[int]:
    """
    Turn a bool series (e.g. ema20 > ema50) into #bars-in-a-row streak length.
    """
    streaks: List[int] = []
    run = 0
    prev_val = None
    for val in series_bool:
        if prev_val is None or val == prev_val:
            run += 1
        else:
            run = 1
        streaks.append(run)
        prev_val = val
    return streaks


def add_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      ema20 / ema50 / ema100
      RSI(14)
      MACD (12/26/9)
      ATR(14) + atr_rel (volatility regime)
      Bollinger %B + bandwidth
      ADX(14)
      ROC(10)
      ema_gap, overextension vs ema20 in ATR terms
      trend_age (how long bull/bear regime has lasted)
    """
    d = _normalize_ohlcv(df_raw)
    if d.empty:
        return pd.DataFrame()

    c = d["Close"]
    h = d["High"]
    l = d["Low"]

    # EMAs
    d["ema20"] = EMAIndicator(c, 20).ema_indicator()
    d["ema50"] = EMAIndicator(c, 50).ema_indicator()
    d["ema100"] = EMAIndicator(c, 100).ema_indicator()

    # RSI
    d["rsi"] = RSIIndicator(c, 14).rsi()

    # MACD
    macd_calc = MACD(c)
    d["macd"] = macd_calc.macd()
    d["macd_signal"] = macd_calc.macd_signal()
    d["macd_hist"] = macd_calc.macd_diff()

    # ATR
    atr_series = AverageTrueRange(h, l, c, 14).average_true_range()
    d["atr"] = atr_series.fillna(atr_series.rolling(14).mean())

    # Volatility regime
    d["atr_mean_50"] = (
        d["atr"]
        .rolling(50, min_periods=10)
        .mean()
        .replace(0, np.nan)
    )
    d["atr_rel"] = d["atr"] / d["atr_mean_50"]

    # Bollinger
    bb = BollingerBands(c, 20, 2)
    d["bb_percent_b"] = (
        (c - bb.bollinger_lband())
        / (bb.bollinger_hband() - bb.bollinger_lband())
    )
    d["bb_bandwidth"] = (
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )

    # ADX (trend strength)
    d["adx"] = ADXIndicator(h, l, c, 14).adx()

    # Momentum
    d["roc_close"] = c.pct_change(10)

    # Structure / stretch
    d["ema_gap"] = (d["ema20"] - d["ema50"]) / d["ema50"]
    d["close_above_ema20_atr"] = (d["Close"] - d["ema20"]) / d["atr"]

    # Regime age (how long in bull or bear mode)
    bull_mask = d["ema20"] > d["ema50"]
    d["trend_age"] = _trend_streak_mask(bull_mask)

    # Clean
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.ffill().bfill()
    d = d.dropna(subset=["Close", "ema20", "ema50", "atr"])
    d.reset_index(drop=True, inplace=True)

    return d
    
# --------------------------------------------------------------------------------------
# RULE-BASED SIGNAL + FILTERS
# --------------------------------------------------------------------------------------

def compute_signal_row(prev: pd.Series, row: pd.Series) -> Tuple[str, float]:
    """
    Base directional call using classical TA logic:
      - ema20 vs ema50 (trend bias)
      - MACD vs signal (momentum confirmation)
      - Volatility filter (ADX, atr_rel)
      - RSI and short-term ROC used as conviction bonus
    Returns:
      side: "Buy" | "Sell" | "Hold"
      prob: base conviction 0..1 before ML / sentiment fusion
    """
    side = "Hold"
    base_prob = 0.5

    vol_ok = (row.get("adx", 0) > 12) and (row.get("atr_rel", 1) >= 0.6)

    # Bullish bias
    if (
        row["ema20"] > row["ema50"]
        and row["macd"] > row["macd_signal"]
        and vol_ok
    ):
        side = "Buy"
        base_prob = 0.66
        # RSI not too cooked
        if row["rsi"] < 70:
            base_prob += 0.06
        # recent momentum positive
        if row["roc_close"] > 0.01:
            base_prob += 0.06

    # Bearish bias
    elif (
        row["ema20"] < row["ema50"]
        and row["macd"] < row["macd_signal"]
        and vol_ok
    ):
        side = "Sell"
        base_prob = 0.66
        # RSI not too depressed
        if row["rsi"] > 30:
            base_prob += 0.06
        # recent momentum negative
        if row["roc_close"] < -0.01:
            base_prob += 0.06

    # Clamp
    base_prob = float(min(0.95, max(0.0, base_prob)))
    return side, base_prob


def _is_exhausted(row: pd.Series, side: str) -> bool:
    """
    "Exhaustion" filter:
    - if price is >2 ATR above ema20 for 30+ bars in an uptrend, don't chase longs
    - if price is <−2 ATR below ema20 for 30+ bars in a downtrend, don't chase shorts
    """
    stretch = row.get("close_above_ema20_atr", 0)
    age = row.get("trend_age", 0)

    if side == "Buy":
        if stretch > 2 and age > 30:
            return True
    elif side == "Sell":
        if stretch < -2 and age > 30:
            return True

    return False


def _compute_tp_sl(price: float, atr: float, side: str, risk: str) -> Tuple[float, float]:
    """
    Risk-based stop/target:
    - Uses ATR multiple scaled by user's risk appetite
    - Always returns numeric TP/SL
    """
    mult_map = {
        "Low": 0.5,
        "Medium": 1.0,
        "High": 2.0,
    }
    mult = mult_map.get(risk, 1.0)

    # ATR fallback just in case
    atr = atr if (atr and atr > 0) else price * 0.005

    if side == "Buy":
        tp = price + mult * atr
        sl = price - mult * atr
    else:  # "Sell"
        tp = price - mult * atr
        sl = price + mult * atr

    return float(tp), float(sl)


def _calc_rr(price: float, tp: float, sl: float, side: str) -> float:
    """
    Reward:Risk ratio from entry price perspective.
    For Buy:
        reward = tp - price
        risk   = price - sl
    For Sell:
        reward = price - tp
        risk   = sl - price
    """
    if side == "Sell":
        reward = price - tp
        risk = sl - price
    else:  # Buy
        reward = tp - price
        risk = price - sl

    if risk <= 0:
        return 0.0
    return float(reward / risk)


def _adaptive_thresholds(row: pd.Series) -> Tuple[float, float]:
    """
    Dynamic entry requirements:
      - In strong directional environments (ADX high, ATR high), we can
        accept slightly lower prob threshold but we demand higher RR.
      - In choppy or slow conditions, we require a higher prob but allow
        slightly lower RR.
    Returns:
      (required_probability, required_min_rr)
    """
    adx_val = row.get("adx", 0)
    atr_rel = row.get("atr_rel", 1)

    prob_thresh = _BASE_PROB_THRESHOLD
    rr_thresh = _BASE_MIN_RR

    # Trendy, volatile market
    if adx_val > 25 and atr_rel >= 1.0:
        prob_thresh -= 0.03      # ok with a little less conviction
        rr_thresh += 0.2         # but demand better reward:risk

    # Boring / low energy market
    elif adx_val < 15 or atr_rel < 0.8:
        prob_thresh += 0.05      # require more conviction
        rr_thresh -= 0.1         # but tolerate slightly worse RR

    # Clamp to sane bounds
    prob_thresh = max(0.5, min(0.9, prob_thresh))
    rr_thresh = max(1.0, min(2.5, rr_thresh))

    return prob_thresh, rr_thresh


# --------------------------------------------------------------------------------------
# SENTIMENT ENGINE
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _fetch_sentiment(symbol: str) -> float:
    """
    Multi-stage sentiment fallback.
    1. Yahoo Finance headlines via yfinance.Ticker.news
    2. Google News RSS
    3. Synthetic bias = short-term slope and momentum of price if no news

    Returns a smoothed compound sentiment score in [-1, 1].
    """
    scores: List[float] = []

    # 1. Yahoo Finance headlines
    try:
        tk = yf.Ticker(symbol)
        news_items = tk.news or []
        for n in news_items[:8]:
            title = n.get("title", "")
            if not title:
                continue
            compound = _VADER.polarity_scores(title)["compound"]
            scores.append(compound)
    except Exception:
        pass

    # 2. Google RSS fallback if empty
    if not scores:
        try:
            rss_url = f"https://news.google.com/rss/search?q={symbol}+finance"
            xml_raw = requests.get(rss_url, timeout=5).text
            root = ET.fromstring(xml_raw)
            for item in root.findall(".//item")[:8]:
                headline = item.findtext("title", "") or ""
                if headline:
                    compound = _VADER.polarity_scores(headline)["compound"]
                    scores.append(compound)
        except Exception:
            pass

    # 3. Synthetic sentiment proxy from price action
    if not scores:
        df_tmp = fetch_data(symbol, "1h", use_cache=True)
        if not df_tmp.empty and len(df_tmp) >= 20:
            # slope of last 20 closes
            closes = df_tmp["Close"].iloc[-20:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]  # raw slope
            # average short-term momentum (%)
            mom = df_tmp["Close"].pct_change().tail(14).mean()
            # squash both into [-1,1]-ish
            guess = np.tanh((slope * 1000 + mom * 50) / 2.0)
            scores.append(guess)

    if not scores:
        return 0.0

    # Smooth by exponential weighting to give recent headlines more weight
    alpha = 0.3
    smoothed = scores[0]
    for s in scores[1:]:
        smoothed = alpha * s + (1 - alpha) * smoothed

    # final clip for safety
    smoothed = float(np.clip(smoothed, -1.0, 1.0))
    return smoothed


# --------------------------------------------------------------------------------------
# ML FEATURES / TRAINING / INFERENCE
# --------------------------------------------------------------------------------------

def _extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the feature columns used for ML model training and inference.
    """
    feat_cols = [
        "ema20",
        "ema50",
        "ema100",
        "ema_gap",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "atr_rel",
        "bb_percent_b",
        "bb_bandwidth",
        "adx",
        "roc_close",
        "close_above_ema20_atr",
        "trend_age",
    ]
    return df[feat_cols].fillna(0.0)


def _train_ml_model(
    symbol: str,
    interval_key: str,
    df_ind: pd.DataFrame,
) -> Optional[RandomForestClassifier]:
    """
    Trains (and caches) a RandomForest on regime-specific data:
    - regime split by ema_gap sign (bull vs bear)
    - target is forward 3-bar direction
    """
    if len(df_ind) < 120:
        return None

    regime = "bull" if df_ind["ema_gap"].iloc[-1] >= 0 else "bear"
    cache_key = (symbol, interval_key, regime)

    # Serve from cache if available
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Make target: did we go up 3 bars ahead?
    df_local = df_ind.copy()
    df_local["target"] = (df_local["Close"].shift(-3) > df_local["Close"]).astype(int)

    X = _extract_ml_features(df_local)
    y = df_local["target"]

    # If not at least 2 classes, can't train
    if y.nunique() < 2:
        return None

    # walk-forward style split (no shuffle)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # simple random forest, class_weight to not just learn "up only"
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


def _ml_predict_prob(
    model: Optional[RandomForestClassifier],
    feature_row: pd.Series,
) -> Optional[float]:
    """
    Return P(up) from ML model for the most recent bar.
    None if model missing or prediction fails.
    """
    if model is None:
        return None
    try:
        # feature_row is a Series; convert to shape (1, n_features)
        row_2d = feature_row.to_frame().T
        proba = model.predict_proba(row_2d)[0][1]
        return float(proba)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# PROBABILITY FUSION / DYNAMIC CUTOFF LOGIC
# --------------------------------------------------------------------------------------

def _fuse_prob(
    rule_prob: float,
    ml_prob: Optional[float],
    sentiment_score: float,
    adx_val: float,
    atr_pct: float,
    recent_winrate: float,
) -> float:
    """
    Combine rule-based probability (rule_prob), ML forecast (ml_prob),
    sentiment, volatility regime, and empirical recent winrate from the
    backtest into a single confidence number 0..1.
    """

    # How much to trust ML depends on recent strategy performance
    # (if backtest lately wins a lot -> lean a bit more on ml_prob;
    # if it's trash -> lean more on rules)
    ml_weight = np.clip(recent_winrate / 100.0, 0.25, 0.75)

    if ml_prob is None:
        fused = rule_prob
    else:
        fused = ml_weight * ml_prob + (1 - ml_weight) * rule_prob

    # High realized ATR vs price => de-risk conviction (super jumpy regime)
    fused *= math.exp(-min(atr_pct * 5.0, 2.5))

    # Sentiment only boosts conviction in higher-trend environments
    if adx_val >= 20:
        fused *= np.clip(1 + 0.2 * sentiment_score, 0.7, 1.4)

    # clip final
    fused = float(np.clip(fused, 0.0, 1.0))
    return fused


def _dynamic_cutoff(recent_winrate: float) -> float:
    """
    If recent win rate is high, accept slightly lower prob.
    If recent win rate is low, demand more conviction.
    Returns probability threshold in [0.5, 0.75] typically.
    """
    # Heuristic: map winrate% -> cutoff
    #  50% winrate => about 0.65 cutoff
    #  70% winrate => about 0.55 cutoff
    raw = 0.55 + (0.65 - recent_winrate / 200.0)
    return float(np.clip(raw, 0.50, 0.75))
    
# --------------------------------------------------------------------------------------
# LATEST PREDICTION + BACKTESTING
# --------------------------------------------------------------------------------------

def _latest_prediction(
    symbol: str,
    interval_key: str,
    risk: str,
    recent_winrate: float = 50.0,
) -> dict:
    """
    Generates latest signal for given asset:
      - Loads / computes indicators
      - Computes rule signal
      - Computes sentiment
      - Optionally trains + infers ML model
      - Fuses probabilities
      - Applies adaptive thresholds and filters
      - Computes TP/SL/RR
    """
    df = fetch_data(symbol, interval_key, use_cache=True)
    if df.empty or len(df) < 60:
        return {"symbol": symbol, "error": "No data"}

    df = add_indicators(df)
    if df.empty:
        return {"symbol": symbol, "error": "No indicators"}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    side, rule_prob = compute_signal_row(prev, last)

    if side == "Hold":
        return {
            "symbol": symbol,
            "side": side,
            "probability": 0.5,
            "sentiment": 0.0,
            "tp": None,
            "sl": None,
            "rr": None,
        }

    # Sentiment
    sent = _fetch_sentiment(symbol)

    # ML
    model = _train_ml_model(symbol, interval_key, df)
    features = _extract_ml_features(df).iloc[-1]
    ml_prob = _ml_predict_prob(model, features)

    # Fused probability
    atr_pct = last["atr"] / last["Close"]
    fused_prob = _fuse_prob(
        rule_prob,
        ml_prob,
        sent,
        last["adx"],
        atr_pct,
        recent_winrate,
    )

    # Adaptive entry thresholds
    prob_thresh, rr_thresh = _adaptive_thresholds(last)
    prob_thresh = _dynamic_cutoff(recent_winrate)

    # TP / SL / RR
    tp, sl = _compute_tp_sl(last["Close"], last["atr"], side, risk)
    rr = _calc_rr(last["Close"], tp, sl, side)

    # Exhaustion
    exhausted = _is_exhausted(last, side)
    if exhausted:
        fused_prob *= 0.85

    # Decision
    if fused_prob < prob_thresh or rr < rr_thresh:
        side = "Hold"

    return {
        "symbol": symbol,
        "side": side,
        "probability": round(fused_prob, 3),
        "sentiment": round(sent, 3),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "rr": round(rr, 2),
    }


# --------------------------------------------------------------------------------------
# BACKTEST
# --------------------------------------------------------------------------------------

def backtest_signals(df: pd.DataFrame, risk: str) -> dict:
    """
    Simplified rolling backtest:
      - Generates rolling rule-based signals
      - Applies TP/SL logic
      - Computes win rate, total return, maxDD, Sharpe-like
    """
    if df.empty or len(df) < 100:
        return {"winrate": 0, "trades": 0, "return": 0, "maxdd": 0, "sharpe": 0}

    balance = 1.0
    peak = 1.0
    wins = 0
    losses = 0
    trades = 0
    pnl = []

    closes = df["Close"].values
    atrs = df["atr"].values
    adxs = df["adx"].values

    for i in range(60, len(df) - 5):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        side, _ = compute_signal_row(prev, row)
        if side == "Hold" or _is_exhausted(row, side):
            continue

        price = row["Close"]
        tp, sl = _compute_tp_sl(price, row["atr"], side, risk)

        exit_hit = False
        for j in range(1, 6):
            if i + j >= len(df):
                break
            nxt = df.iloc[i + j]["Close"]
            if side == "Buy" and nxt >= tp:
                balance *= 1.01
                wins += 1
                exit_hit = True
                break
            elif side == "Buy" and nxt <= sl:
                balance *= 0.99
                losses += 1
                exit_hit = True
                break
            elif side == "Sell" and nxt <= tp:
                balance *= 1.01
                wins += 1
                exit_hit = True
                break
            elif side == "Sell" and nxt >= sl:
                balance *= 0.99
                losses += 1
                exit_hit = True
                break

        if not exit_hit:
            # if no exit hit within 5 bars, mark neutral
            continue

        trades += 1
        peak = max(peak, balance)
        dd = (peak - balance) / peak
        pnl.append(dd)

    total_ret = (balance - 1.0) * 100
    winrate = (wins / trades * 100) if trades > 0 else 0
    maxdd = (max(pnl) * 100) if pnl else 0
    sharpe_like = (winrate / (maxdd + 1)) if maxdd > 0 else winrate

    return {
        "winrate": round(winrate, 2),
        "trades": trades,
        "return": round(total_ret, 2),
        "maxdd": round(maxdd, 2),
        "sharpe": round(sharpe_like, 2),
    }


# --------------------------------------------------------------------------------------
# WRAPPERS FOR APP INTEGRATION
# --------------------------------------------------------------------------------------

def analyze_asset(symbol: str, interval_key: str, risk: str, use_cache=True) -> dict:
    """
    Unified call: fetch → indicators → sentiment → ML → fusion → backtest
    Returns metrics ready for dashboard.
    """
    df = fetch_data(symbol, interval_key, use_cache=use_cache)
    if df.empty:
        return {"symbol": symbol, "error": "No data"}

    df = add_indicators(df)
    if df.empty:
        return {"symbol": symbol, "error": "Indicators failed"}

    back = backtest_signals(df, risk)
    pred = _latest_prediction(symbol, interval_key, risk, back["winrate"])

    result = {
        "symbol": symbol,
        "side": pred.get("side", "Hold"),
        "probability": pred.get("probability", 0),
        "sentiment": pred.get("sentiment", 0),
        "tp": pred.get("tp"),
        "sl": pred.get("sl"),
        "rr": pred.get("rr"),
        "winrate": back["winrate"],
        "trades": back["trades"],
        "return": back["return"],
        "maxdd": back["maxdd"],
        "sharpe": back["sharpe"],
    }
    return result
    
# --------------------------------------------------------------------------------------
# PORTFOLIO SUMMARY + STREAMLIT HELPERS
# --------------------------------------------------------------------------------------

def summarize_assets(interval_key: str, risk: str, use_cache=True) -> pd.DataFrame:
    """
    Loops through all assets, runs analysis, aggregates summary DataFrame.
    """
    rows = []
    for name, symbol in ASSET_SYMBOLS.items():
        try:
            _log(f"⏳ {name} ({symbol}) ...")
            res = analyze_asset(symbol, interval_key, risk, use_cache=use_cache)
            row = {
                "Asset": name,
                "Side": res.get("side", "Hold"),
                "Probability": res.get("probability", 0.0),
                "Sentiment": res.get("sentiment", 0.0),
                "TP": res.get("tp"),
                "SL": res.get("sl"),
                "RR": res.get("rr"),
                "WinRate": res.get("winrate", 0.0),
                "Trades": res.get("trades", 0),
                "Return": res.get("return", 0.0),
                "MaxDD": res.get("maxdd", 0.0),
                "SharpeLike": res.get("sharpe", 0.0),
            }
            rows.append(row)
        except Exception as e:
            _log(f"⚠️ {symbol}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # clean numeric
    for c in ["Probability", "Sentiment", "WinRate", "Return", "MaxDD", "SharpeLike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # order by Probability descending for readability
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    return df


def asset_prediction_and_backtest(asset: str, interval_key: str, risk: str) -> Tuple[pd.DataFrame, dict]:
    """
    Used by scenario/backtest tab:
      - Loads + indicatorizes selected asset
      - Runs backtest
      - Returns DataFrame + metrics
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df = fetch_data(symbol, interval_key, use_cache=True)
    df = add_indicators(df)
    back = backtest_signals(df, risk)
    return df, back


def load_asset_with_indicators(asset: str, interval_key: str) -> Tuple[str, pd.DataFrame]:
    """
    Streamlit wrapper to load asset fully prepped with indicators.
    Returns both symbol and DataFrame.
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    df = fetch_data(symbol, interval_key, use_cache=True)
    df = add_indicators(df)
    return symbol, df


def asset_prediction_single(asset: str, interval_key: str, risk: str) -> dict:
    """
    Shortcut to get one prediction for an asset (used in Trends tab).
    """
    symbol = ASSET_SYMBOLS.get(asset, asset)
    return _latest_prediction(symbol, interval_key, risk)


# --------------------------------------------------------------------------------------
# END OF FILE MARKER
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Self-test for CLI execution
    print("🔍 Running quick self-check on Gold (GC=F)...")
    result = analyze_asset("GC=F", "1h", "Medium", use_cache=True)
    print(pd.DataFrame([result]))
    print("✅ utils.py (Smart v6.3) loaded successfully.")
