import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import time
import random

# ───────────────────────────────
# GLOBAL CONFIGURATION
# ───────────────────────────────
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
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "1d": {"interval": "1d", "period": "6mo"},
}

RISK_MULT = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.8
}

# ───────────────────────────────
# FETCH DATA (with fixes for YFinance)
# ───────────────────────────────
def fetch_data(symbol, interval="1h", period="1mo", retries=4, delay=4):
    """Fetch OHLCV data with retry logic and dimension fix."""
    for attempt in range(1, retries + 1):
        try:
            print(f"📊 Fetching {symbol} [{interval}] for {period} (Attempt {attempt})...")
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
                auto_adjust=True
            )

            # Fix: Yahoo returns 2D arrays for single columns sometimes
            for col in df.columns:
                if isinstance(df[col].iloc[0], (np.ndarray, list)):
                    df[col] = [float(x[0]) for x in df[col]]

            if df.empty or len(df) < 20:
                raise ValueError("No data returned or too few rows")

            df = df.dropna()
            df["Return"] = df["Close"].pct_change()
            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            if attempt < retries:
                time.sleep(delay + random.random() * 3)
            else:
                print(f"🚫 All retries failed for {symbol}. Returning empty DataFrame.")
                return pd.DataFrame()

# ───────────────────────────────
# ADD TECHNICAL INDICATORS
# ───────────────────────────────
def add_indicators(df):
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    bb = BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df = df.dropna()
    return df

# ───────────────────────────────
# TRADING THEORY OVERLAY
# ───────────────────────────────
def apply_trading_theory(pred, df):
    latest = df.iloc[-1]
    ema_trend = latest["EMA_20"] > latest["EMA_50"]
    rsi_ok = 40 < latest["RSI"] < 70
    macd_conf = latest["MACD"] > latest["Signal_Line"]
    bb_breakout = latest["Close"] > latest["BB_High"] or latest["Close"] < latest["BB_Low"]

    score = sum([ema_trend, rsi_ok, macd_conf, bb_breakout])
    if pred == "buy" and score >= 2:
        return "buy", 0.95
    elif pred == "sell" and not ema_trend and score >= 2:
        return "sell", 0.95
    else:
        return "neutral", 0.6

# ───────────────────────────────
# MODEL TRAINING & PREDICTION
# ───────────────────────────────
def train_and_predict(df, horizon="1h", risk="Medium"):
    df = add_indicators(df)
    if len(df) < 60:
        return None

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    features = ["EMA_20", "EMA_50", "RSI", "MACD", "Signal_Line", "Return"]
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X[:-1], y[:-1])
    latest = X.iloc[-1:].values
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]
    raw = "buy" if pred == 1 else "sell"

    adjusted_pred, conf_adj = apply_trading_theory(raw, df)
    conf = min(1.0, prob * conf_adj)

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    mult = RISK_MULT.get(risk, 1.0)
    price = df["Close"].iloc[-1]

    tp = price + (atr * 1.5 * mult if adjusted_pred == "buy" else -atr * 1.5 * mult)
    sl = price - (atr * 1.0 * mult if adjusted_pred == "buy" else -atr * 1.0 * mult)

    return {
        "prediction": adjusted_pred,
        "probability": conf,
        "accuracy": model.score(X, y),
        "tp": tp,
        "sl": sl
    }

# ───────────────────────────────
# BACKTESTING ENGINE
# ───────────────────────────────
def backtest_signals(df, pred):
    if df is None or df.empty or pred is None:
        return {"winrate": 0, "total_return": 0, "equity_curve": pd.Series(dtype=float)}

    df = df.copy()
    sig = 1 if pred["prediction"] == "buy" else -1 if pred["prediction"] == "sell" else 0
    df["Signal"] = sig
    df["Strat_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strat_Return"]).cumprod()
    winrate = (df["Strat_Return"] > 0).sum() / max(1, len(df))
    total_ret = df["Equity"].iloc[-1] - 1
    return {"winrate": winrate, "total_return": total_ret, "equity_curve": df["Equity"]}

# ───────────────────────────────
# MULTI-ASSET SUMMARY
# ───────────────────────────────
def summarize_assets():
    results = []
    for asset, symbol in ASSET_SYMBOLS.items():
        df = fetch_data(symbol, "1h", "1mo")
        if df.empty:
            print(f"⚠️ Skipping {asset} (no data)")
            continue
        pred = train_and_predict(df)
        if not pred:
            continue
        back = backtest_signals(df, pred)
        results.append({
            "Asset": asset,
            "Prediction": pred["prediction"],
            "Confidence": round(pred["probability"] * 100, 2),
            "Win Rate": round(back["winrate"] * 100, 2),
            "Return": round(back["total_return"] * 100, 2)
        })
    return pd.DataFrame(results)