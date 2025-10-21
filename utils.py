# ─────────────────────────────────────────────────────────────────────────────
# Utilities: data fetch, indicators, ML predict, plotting, backtests, sentiment
# No TensorFlow; uses scikit-learn RandomForest + technicals + (optional) news
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Memory
import plotly.graph_objects as go

# Optional libs
try:
    import ta
except:
    ta = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except:
    _VADER = None

# ── Caching (on disk in /tmp for Streamlit Cloud) ────────────────────────────
memory = Memory(location="/tmp/woody_cache", verbose=0)

# ── Asset universe & intervals ───────────────────────────────────────────────
ASSET_SYMBOLS = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Copper (HG=F)": "HG=F",
    "US 100 (^NDX)": "^NDX",
    "S&P 500 (^GSPC)": "^GSPC",
    "Dow Jones (^DJI)": "^DJI",
    "FTSE 100 (^FTSE)": "^FTSE",
}

INTERVALS = {
    "15m": {"yf_interval": "15m", "yf_period": "60d", "horizon_steps": 4},   # next 1h
    "1h":  {"yf_interval": "1h",  "yf_period": "60d", "horizon_steps": 4},   # next 4h
    "1d":  {"yf_interval": "1d",  "yf_period": "2y",  "horizon_steps": 1},   # next 1d
}

RISK_MULT = {"Low": 0.5, "Medium": 1.0, "High": 1.5}

# ── Fetch & clean ────────────────────────────────────────────────────────────
@memory.cache
def fetch_data(symbol: str, interval_key: str = "1h") -> pd.DataFrame:
    cfg = INTERVALS[interval_key]
    df = yf.download(
        symbol,
        interval=cfg["yf_interval"],
        period=cfg["yf_period"],
        progress=False,
        prepost=False
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    # Flatten multiindex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].capitalize() for c in df.columns]
    else:
        df.columns = [c.capitalize() for c in df.columns]

    # Ensure necessary cols
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # tz-naive index
    try:
        df.index = pd.to_datetime(df.index).tz_convert(None)
    except Exception:
        df.index = pd.to_datetime(df.index).tz_localize(None)

    df = df.dropna(subset=["Open","High","Low","Close"])
    return df

# ── Indicators ───────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return_1"] = out["Close"].pct_change()
    out["Volatility_20"] = out["Return_1"].rolling(20).std()

    # ATR (manual, to avoid extra deps)
    tr = np.maximum(out["High"]-out["Low"],
                    np.maximum((out["High"]-out["Close"].shift()).abs(),
                               (out["Low"]-out["Close"].shift()).abs()))
    out["ATR_14"] = pd.Series(tr).rolling(14).mean()

    if ta is not None:
        out["RSI_14"] = ta.momentum.RSIIndicator(close=out["Close"], window=14).rsi()
        macd = ta.trend.MACD(close=out["Close"])
        out["MACD"] = macd.macd()
        out["MACD_Signal"] = macd.macd_signal()
        out["EMA_20"] = ta.trend.EMAIndicator(close=out["Close"], window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(close=out["Close"], window=20, window_dev=2)
        out["BB_W"] = bb.bollinger_wband()
    else:
        out["RSI_14"] = out["Return_1"].rolling(14).apply(lambda r: 50, raw=True)
        out["MACD"] = out["Return_1"].ewm(span=12).mean() - out["Return_1"].ewm(span=26).mean()
        out["MACD_Signal"] = out["MACD"].ewm(span=9).mean()
        out["EMA_20"] = out["Close"].ewm(span=20).mean()
        out["BB_W"] = (out["Close"].rolling(20).std()/out["Close"].rolling(20).mean()).fillna(0)

    out["EMA_Dist"] = (out["Close"]/out["EMA_20"] - 1.0)
    out["Volume_Change"] = out["Volume"].pct_change().fillna(0)
    out = out.dropna()
    return out

# ── Sentiment (optional) via Yahoo Finance news in yfinance ──────────────────
def _news_sentiment(symbol: str, when: pd.DatetimeIndex) -> pd.Series:
    if _VADER is None:
        return pd.Series(0.0, index=when)  # neutral

    try:
        tk = yf.Ticker(symbol)
        news = tk.news  # list of dicts with 'title' and 'providerPublishTime'
        if not news:
            return pd.Series(0.0, index=when)

        rows = []
        for n in news:
            title = n.get("title") or ""
            t = n.get("providerPublishTime")
            if not title or t is None:
                continue
            ts = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(tz=None).date()
            score = _VADER.polarity_scores(title)["compound"]
            rows.append((ts, score))

        if not rows:
            return pd.Series(0.0, index=when)

        s = pd.DataFrame(rows, columns=["date","score"]).groupby("date")["score"].mean()
        idx_dates = pd.Index([d.date() for d in when], name="date")
        aligned = s.reindex(idx_dates, method="ffill").fillna(0.0)
        aligned.index = when
        return aligned
    except Exception:
        return pd.Series(0.0, index=when)

# ── Feature prep & label ─────────────────────────────────────────────────────
FEATURES = ["Return_1","Volatility_20","ATR_14","RSI_14","MACD","MACD_Signal","EMA_Dist","BB_W","Volume_Change","Sentiment"]

def prepare_ml_frame(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    X = add_indicators(df)
    X["Sentiment"] = _news_sentiment(symbol="SPY" if len(df)>0 else "", when=X.index)  # fallback if news mapping fails
    # Forward return label
    X["FwdRet"] = X["Close"].pct_change(horizon_steps).shift(-horizon_steps)
    # Class label: 1=BUY (up), -1=SELL (down), 0=HOLD (flat)
    thresh = X["FwdRet"].abs().median() * 0.2  # small deadzone
    X["Y"] = 0
    X.loc[X["FwdRet"] >  thresh, "Y"] = 1
    X.loc[X["FwdRet"] < -thresh, "Y"] = -1
    X = X.dropna(subset=FEATURES+["Y","Close","ATR_14"])
    return X

# ── Train/predict (lightweight) ──────────────────────────────────────────────
def train_and_predict(df: pd.DataFrame, interval_key: str, risk: str = "Medium"):
    if df is None or df.empty:
        return None, None, {"signal":"HOLD","prob":0.0,"risk":"Low","tp":None,"sl":None,"accuracy":None}

    horizon = INTERVALS[interval_key]["horizon_steps"]
    X = prepare_ml_frame(df, horizon_steps=horizon)
    if X.empty or X.shape[0] < 200:
        return X, None, {"signal":"HOLD","prob":0.0,"risk":"Low","tp":None,"sl":None,"accuracy":None}

    X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    clf.fit(X_train[FEATURES], X_train["Y"])
    yhat = clf.predict(X_test[FEATURES])
    acc = accuracy_score(X_test["Y"], yhat)

    # Latest prediction + probability
    last_row = X.iloc[[-1]]
    proba = clf.predict_proba(last_row[FEATURES])[0]
    classes = clf.classes_.tolist()   # e.g. [-1, 0, 1]
    p_buy = proba[classes.index(1)] if 1 in classes else 0.0
    p_sell= proba[classes.index(-1)] if -1 in classes else 0.0

    if p_buy > max(0.5, p_sell + 0.1):
        signal = "BUY"; prob = float(p_buy)
    elif p_sell > max(0.5, p_buy + 0.1):
        signal = "SELL"; prob = float(p_sell)
    else:
        signal = "HOLD"; prob = float(max(p_buy, p_sell))

    # TP/SL based on ATR and risk
    atr = float(last_row["ATR_14"].iloc[0])
    price = float(last_row["Close"].iloc[0])
    mult = RISK_MULT.get(risk, 1.0)
    sl, tp = None, None
    if signal == "BUY":
        sl = price - mult * 1.2 * atr
        tp = price + mult * 2.0 * atr
    elif signal == "SELL":
        sl = price + mult * 1.2 * atr
        tp = price - mult * 2.0 * atr

    out = {
        "signal": signal,
        "prob": prob,
        "risk": risk,
        "tp": tp,
        "sl": sl,
        "accuracy": float(acc)
    }
    return X, clf, out

# ── Backtest simple strategy (marker overlay & PnL) ──────────────────────────
def backtest_signals(X: pd.DataFrame, signal_col="Y"):
    df = X.copy()
    df["Signal"] = df[signal_col].replace({-1:-1, 0:0, 1:1})
    df["NextRet"] = df["Close"].pct_change().shift(-1).fillna(0)
    df["StrategyRet"] = df["NextRet"] * df["Signal"].shift().fillna(0)
    equity = (1 + df["StrategyRet"]).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    trades = (df["Signal"].diff().abs() > 0).sum()
    winrate = float((df["StrategyRet"] > 0).sum() / max(1,(df["StrategyRet"] != 0).sum()))
    return {
        "total_return": total,
        "num_trades": int(trades),
        "winrate": winrate,
        "equity_curve": equity
    }

# ── Plotting ─────────────────────────────────────────────────────────────────
def make_candles(df: pd.DataFrame, title: str, max_points: int = 500,
                 buys: pd.Index | None = None, sells: pd.Index | None = None,
                 sl: float | None = None, tp: float | None = None):
    if df is None or df.empty:
        return go.Figure()

    data = df.copy()
    if len(data) > max_points:
        data = data.iloc[-max_points:].copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price"
    ))

    if buys is not None and len(buys)>0:
        here = data.index.intersection(buys)
        fig.add_trace(go.Scatter(
            x=here, y=data.loc[here,"Close"],
            mode="markers", marker=dict(size=8, symbol="triangle-up"),
            name="Buy"
        ))
    if sells is not None and len(sells)>0:
        here = data.index.intersection(sells)
        fig.add_trace(go.Scatter(
            x=here, y=data.loc[here,"Close"],
            mode="markers", marker=dict(size=8, symbol="triangle-down"),
            name="Sell"
        ))
    if tp is not None:
        fig.add_hline(y=tp, line_dash="dot", annotation_text="TP", annotation_position="top left")
    if sl is not None:
        fig.add_hline(y=sl, line_dash="dot", annotation_text="SL", annotation_position="bottom left")

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10,r=10,t=35,b=10),
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#e6e6e6"),
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222"),
        legend=dict(orientation="h", y=1.05, x=0)
    )
    return fig

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmtpct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except:
        return "—"

def guard_float(x):
    try:
        return float(x)
    except:
        return None