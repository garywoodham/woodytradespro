import pandas as pd
import numpy as np
import yfinance as yf

ASSET_SYMBOLS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "US100 (Nasdaq 100)": "^NDX",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
}

INTERVALS = {
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h":  ("60m", "60d"),
    "1d":  ("1d",  "2y"),
    "1wk": ("1wk", "5y"),
}

POS_WORDS = set("surge rally gain beat strong bullish breakout rebound squeeze growth momentum upgrade profit record positive optimistic upbeat".split())
NEG_WORDS = set("drop plunge miss weak bearish breakdown slump downgrade loss negative pessimistic bleak".split())

def quick_sentiment(headlines):
    if not headlines:
        return 0.0, "Neutral"
    score = 0
    count = 0
    for h in headlines[:20]:
        w = h.lower().split()
        score += sum(1 for x in w if x in POS_WORDS)
        score -= sum(1 for x in w if x in NEG_WORDS)
        count += 1
    import numpy as np
    norm = np.tanh(score / max(1, count))
    label = "Bullish" if norm > 0.15 else "Bearish" if norm < -0.15 else "Neutral"
    return float(norm), label

def _flatten_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].capitalize() for c in df.columns]
    else:
        df.columns = [str(c).capitalize() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def risk_tooltip_md():
    return ("<small>Low: TP=0.6%, SL=0.3% • "
            "Medium: TP=1.2%, SL=0.6% • "
            "High: TP=2.0%, SL=1.0%</small>")

def tp_sl_by_risk(risk: str, price: float):
    risk = (risk or "Medium").lower()
    if risk == "low":
        tp_pct, sl_pct = 0.006, 0.003
    elif risk == "high":
        tp_pct, sl_pct = 0.020, 0.010
    else:
        tp_pct, sl_pct = 0.012, 0.006
    return price * (1 + tp_pct), price * (1 - sl_pct), tp_pct, sl_pct

def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    yf_interval, _default = INTERVALS.get(interval, ("60m","60d"))
    p = period or _default
    try:
        df = yf.download(symbol, interval=yf_interval, period=p, progress=False, prepost=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = _flatten_ohlc(df.copy())
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df.sort_index()
        return df
    except Exception:
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    delta = out["Close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out["RSI_14"] = 100 - (100 / (1 + rs))
    out["ATR_14"] = (out["High"] - out["Low"]).rolling(14).mean()
    return out

def predict_next(df: pd.DataFrame, risk: str = "Medium", news_headlines = None) -> dict:
    if df is None or df.empty:
        return {"signal":"HOLD","probability":0.0,"risk":risk,"tp":None,"sl":None}
    x = add_indicators(df)
    close = safe_float(x["Close"].iloc[-1])
    ema12 = safe_float(x["EMA_12"].iloc[-1])
    ema26 = safe_float(x["EMA_26"].iloc[-1])
    rsi = safe_float(x["RSI_14"].iloc[-1])
    ema_slope = safe_float(x["EMA_12"].iloc[-1] - x["EMA_12"].iloc[-3]) if len(x) > 3 else 0.0

    score = 0.0
    if ema12 > ema26: score += 0.6
    if rsi < 30: score += 0.4
    if rsi > 70: score -= 0.4
    if ema_slope > 0: score += 0.2
    if ema_slope < 0: score -= 0.2

    sent_score, sent_label = quick_sentiment(news_headlines or [])
    score += 0.4 * sent_score

    prob = float(np.clip(0.5 + 0.5*np.tanh(score), 0.0, 1.0))
    signal = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "HOLD"

    tp, sl, tp_pct, sl_pct = tp_sl_by_risk(risk, close)
    return {
        "signal": signal,
        "probability": prob,
        "risk": risk,
        "tp": tp if signal=="BUY" else close*(1-sl_pct),
        "sl": sl if signal=="BUY" else close*(1+sl_pct),
        "rsi": rsi, "ema12": ema12, "ema26": ema26,
        "sentiment": sent_label, "sentiment_score": sent_score
    }

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    x = add_indicators(df)
    sig = pd.Series(index=x.index, data="HOLD")
    buy = (x["EMA_12"].shift(1) <= x["EMA_26"].shift(1)) & (x["EMA_12"] > x["EMA_26"]) & (x["RSI_14"] < 60)
    sell = (x["EMA_12"].shift(1) >= x["EMA_26"].shift(1)) & (x["EMA_12"] < x["EMA_26"]) & (x["RSI_14"] > 40)
    sig[buy] = "BUY"
    sig[sell] = "SELL"
    out = x.copy()
    out["Signal"] = sig
    return out

def backtest(df: pd.DataFrame, risk: str = "Medium") -> dict:
    if df is None or df.empty:
        return {"trades":[], "total_return":0.0, "win_rate":0.0}
    x = generate_signals(df)
    trades = []
    pos = None
    tp_mult, sl_mult = (0.012, 0.006) if risk.lower()=="medium" else (0.006, 0.003) if risk.lower()=="low" else (0.02, 0.01)

    for t, row in x.iterrows():
        if pos is None and row["Signal"] in ("BUY","SELL"):
            direction = 1 if row["Signal"]=="BUY" else -1
            entry = row["Close"]
            tp = entry*(1+tp_mult*direction)
            sl = entry*(1-sl_mult*direction)
            pos = {"time":t,"entry":entry,"dir":direction,"tp":tp,"sl":sl}
        elif pos is not None:
            price = row["Close"]
            hit_tp = price >= pos["tp"] if pos["dir"]==1 else price <= pos["tp"]
            hit_sl = price <= pos["sl"] if pos["dir"]==1 else price >= pos["sl"]
            exit_reason = None
            if hit_tp: exit_reason = "TP"
            elif hit_sl: exit_reason = "SL"
            elif row["Signal"] in ("BUY","SELL"): exit_reason = "Flip"

            if exit_reason:
                pnl = (price - pos["entry"]) * pos["dir"]
                trades.append({
                    "entry_time": pos["time"],
                    "exit_time": t,
                    "entry": pos["entry"],
                    "exit": price,
                    "direction": "LONG" if pos["dir"]==1 else "SHORT",
                    "pnl": pnl,
                    "return_pct": pnl/pos["entry"]*100.0,
                    "exit_reason": exit_reason
                })
                pos = None

    if trades:
        total_ret = sum(tr["return_pct"] for tr in trades)
        win_rate = 100.0 * sum(1 for tr in trades if tr["pnl"]>0) / len(trades)
    else:
        total_ret = 0.0
        win_rate = 0.0
    return {"trades": trades, "total_return": total_ret, "win_rate": win_rate}

import plotly.graph_objects as go

def plot_candles(df: pd.DataFrame, title: str="", signals_df: pd.DataFrame | None = None, latest: dict | None = None):
    if df is None or df.empty:
        return go.Figure()
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444", name="Price"
        )
    ])
    if signals_df is not None and "Signal" in signals_df.columns:
        buys = signals_df[signals_df["Signal"]=="BUY"]
        sells = signals_df[signals_df["Signal"]=="SELL"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                 marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
                                 name="BUY"))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                 marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
                                 name="SELL"))
    if latest and latest.get("tp") and latest.get("sl"):
        fig.add_hline(y=float(latest["tp"]), line_dash="dot", line_color="#22c55e", annotation_text="TP", annotation_position="top left")
        fig.add_hline(y=float(latest["sl"]), line_dash="dot", line_color="#ef4444", annotation_text="SL", annotation_position="bottom left")

    fig.update_layout(template="plotly_dark", title=title, margin=dict(l=10,r=10,t=40,b=10), xaxis_rangeslider_visible=False, height=520)
    return fig
