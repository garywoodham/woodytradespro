# utils.py v8.3 (Smart Strategy Modes Edition)
# NOTE: This is the runnable engine version with:
# - Strategy modes (9 styles)
# - Accuracy/Performance plumbing
# - 5-fold CV ML confidence
# - Adaptive TP/SL with regime scaling
# - Weekend stale handling
# - PF / EV% / WinRate backtest stats
# - Signal markers for plotting
#
# Public API used by app.py:
#   summarize_assets(...)
#   asset_prediction_and_backtest(...)
#   load_asset_with_indicators(...)
#   ASSET_SYMBOLS
#   INTERVALS
#
# This version is compacted for delivery. Some comments trimmed,
# but logic matches our design.

from __future__ import annotations

import json, math, time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import KFold
except Exception:
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    KFold = None

try:
    from ta.trend import EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
except Exception:
    EMAIndicator = MACD = ADXIndicator = RSIIndicator = AverageTrueRange = BollingerBands = None

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
CALIBRATION_PATH = DATA_DIR / "calibration_cache.json"
_CAL_HISTORY_LEN = 5

INTERVALS: Dict[str, Dict[str, object]] = {
    "15m": {"interval": "15m", "period": "5d",  "min_rows": 150},
    "1h":  {"interval": "1h",  "period": "2mo", "min_rows": 250},
    "4h":  {"interval": "4h",  "period": "6mo", "min_rows": 250},
    "1d":  {"interval": "1d",  "period": "1y",  "min_rows": 200},
    "1wk": {"interval": "1wk", "period": "5y",  "min_rows": 150},
}

RISK_MULT = {
    "Low":    {"tp_atr": 1.0, "sl_atr": 1.5},
    "Medium": {"tp_atr": 1.5, "sl_atr": 1.0},
    "High":   {"tp_atr": 2.0, "sl_atr": 0.8},
}

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

# Tunables
RSI_WINDOW = 100
RSI_Q_LOW = 0.2
RSI_Q_HIGH = 0.8
ADX_MIN_TREND_DEFAULT = 20.0
VOL_REGIME_MIN_ATR_PCT_DEFAULT = 0.3
CONF_EXEC_THRESHOLD_DEFAULT = 0.6

# Initialize dynamic globals with defaults
ADX_MIN_TREND = ADX_MIN_TREND_DEFAULT
VOL_REGIME_MIN_ATR_PCT = VOL_REGIME_MIN_ATR_PCT_DEFAULT
CONF_EXEC_THRESHOLD = CONF_EXEC_THRESHOLD_DEFAULT

HIGHER_TF_MAP = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1wk",
    "1wk": "1wk",
}

ML_RECENT_WINDOW = 500
CV_FOLDS = 5
FORCED_TRADE_PROB = 0.02
FORCED_CONF_MIN = 0.55
ATR_FLOOR_MULT = 0.0005
STALE_MULTIPLIER_HOURS = 48.0

_TP_SL_PROFILES = {
    "Off":        {"trend_min": 1.0, "trend_max": 1.0, "vol_min": 1.0, "vol_max": 1.0},
    "Normal":     {"trend_min": 0.8, "trend_max": 1.4, "vol_min": 0.8, "vol_max": 1.4},
    "Aggressive": {"trend_min": 0.5, "trend_max": 2.0, "vol_min": 0.5, "vol_max": 2.0},
}

# structure detection params
STRUCT_WINDOW = 20
SUP_PROX_PCT = 0.003
RSI_DIP_THRESHOLD = 40.0
RANGE_LOOKBACK = 50
SWING_LOOKBACK = 5
BB_WINDOW = 20
BB_WIDTH_PCT_THRESH = 0.01

def _log(msg:str)->None:
    try: print(msg, flush=True)
    except: pass

# ---------------- Calibration memory ----------------

def _load_calib()->Dict[str,Any]:
    if not CALIBRATION_PATH.exists(): return {}
    try:
        data=json.load(open(CALIBRATION_PATH,"r"))
        return data if isinstance(data,dict) else {}
    except Exception:
        return {}

def _save_calib(c:Dict[str,Any])->None:
    try: json.dump(c, open(CALIBRATION_PATH,"w"))
    except Exception: pass

def _record_calib(c:Dict[str,Any], sym:str, winrate:float)->Dict[str,Any]:
    now_str=pd.Timestamp.utcnow().isoformat()+"Z"
    e=c.get(sym,{"winrates":[],"last_ts":now_str})
    hist=e.get("winrates",[]); hist=hist if isinstance(hist,list) else []
    hist.append(float(winrate))
    hist=hist[-_CAL_HISTORY_LEN:]
    e["winrates"]=hist; e["last_ts"]=now_str
    c[sym]=e
    return c

def _symbol_wr_stats(c:Dict[str,Any], sym:str)->Dict[str,float]:
    e=c.get(sym,{}); hist=e.get("winrates",[])
    if not hist: return {"avg_wr":50.0,"bias":0.0}
    avg_wr=float(np.mean(hist))
    bias=(avg_wr-50.0)/100.0
    bias=max(-0.1,min(0.1,bias))
    return {"avg_wr":avg_wr,"bias":bias}

def _calib_bias_for_symbol(c:Dict[str,Any], sym:str, enabled:bool)->Dict[str,float]:
    if not enabled:
        return {"conf_mult":1.0,"adx_mult":1.0,"avg_wr":50.0}
    st=_symbol_wr_stats(c,sym)
    bias=st["bias"]
    conf_mult=1.0-bias
    adx_mult =1.0-bias
    conf_mult=max(0.8,min(1.2,conf_mult))
    adx_mult =max(0.8,min(1.2,adx_mult))
    return {"conf_mult":conf_mult,"adx_mult":adx_mult,"avg_wr":st["avg_wr"]}

def _apply_filter_level(filter_level:str, calib_bias:Dict[str,float]):
    global ADX_MIN_TREND, VOL_REGIME_MIN_ATR_PCT, CONF_EXEC_THRESHOLD
    if filter_level=="Loose":
        base_adx=15.0; base_atr=0.2; base_conf=0.5
    elif filter_level=="Strict":
        base_adx=25.0; base_atr=0.4; base_conf=0.7
    else:
        base_adx=20.0; base_atr=0.3; base_conf=0.6
    cm=calib_bias.get("conf_mult",1.0)
    am=calib_bias.get("adx_mult",1.0)
    ADX_MIN_TREND = max(10.0,min(35.0,base_adx*am))
    VOL_REGIME_MIN_ATR_PCT = max(0.1,min(1.0,base_atr))
    CONF_EXEC_THRESHOLD = max(0.4,min(0.8,base_conf*cm))

# ---------------- Fetch / normalize -----------------

def _cache_path(sym:str, ikey:str)->Path:
    safe=sym.replace("^","").replace("=","_").replace("/","_").replace("-","_")
    return DATA_DIR / f"{safe}_{ikey}.csv"

def _normalize_ohlcv(df:pd.DataFrame)->pd.DataFrame:
    if df is None or not isinstance(df,pd.DataFrame) or df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    keep=[c for c in["Open","High","Low","Close","Adj Close","Volume"]if c in df.columns]
    if not keep:
        df=df.rename(columns={c:c.capitalize() for c in df.columns})
        keep=[c for c in["Open","High","Low","Close","Adj Close","Volume"]if c in df.columns]
    df=df[keep].copy()
    if not isinstance(df.index,pd.DatetimeIndex):
        df.index=pd.to_datetime(df.index)
    df=df.sort_index()
    for col in df.columns:
        vals=df[col].values
        if isinstance(vals,np.ndarray) and getattr(vals,"ndim",1)>1:
            df[col]=pd.Series(vals.ravel(),index=df.index)
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.replace([np.inf,-np.inf],np.nan).ffill().bfill()
    df=df.dropna(how="all")
    df=df[~df.index.duplicated(keep="last")]
    return df

def _yf_dl(sym:str,interval:str,period:str)->pd.DataFrame:
    try:
        raw=yf.download(sym,period=period,interval=interval,progress=False,threads=False)
        return _normalize_ohlcv(raw)
    except Exception:
        return pd.DataFrame()

def _yf_hist(sym:str,interval:str,period:str)->pd.DataFrame:
    try:
        tk=yf.Ticker(sym)
        raw=tk.history(period=period,interval=interval,auto_adjust=True,prepost=False)
        df=_normalize_ohlcv(raw)
        if not df.empty: return df
        raw=tk.history(period=period,interval=interval,auto_adjust=False,prepost=False)
        return _normalize_ohlcv(raw)
    except Exception:
        return pd.DataFrame()

def fetch_data(symbol:str, interval_key:str="1h", use_cache:bool=True,
               max_retries:int=4, backoff_range:Tuple[float,float]=(2.5,6.0))->pd.DataFrame:
    if interval_key not in INTERVALS:
        raise KeyError(f"Bad interval_key {interval_key}")
    meta=INTERVALS[interval_key]
    interval=str(meta["interval"]); period=str(meta["period"]); min_rows=int(meta["min_rows"])
    cache_fp=_cache_path(symbol,interval_key)
    if use_cache and cache_fp.exists():
        try:
            cached=_normalize_ohlcv(pd.read_csv(cache_fp,index_col=0,parse_dates=True))
            if len(cached)>=min_rows: return cached
        except Exception: pass
    for _a in range(max_retries):
        live=_yf_dl(symbol,interval,period)
        if not live.empty and len(live)>=min_rows:
            try: live.to_csv(cache_fp)
            except Exception: pass
            return live
        time.sleep(np.random.uniform(*backoff_range))
    mir=_yf_hist(symbol,interval,period)
    if not mir.empty and len(mir)>=min_rows:
        try: mir.to_csv(cache_fp)
        except Exception: pass
        return mir
    return pd.DataFrame()

# -------------- stale / weekend awareness -----------

def _interval_hours(ikey:str)->float:
    return {"15m":0.25,"1h":1.0,"4h":4.0,"1d":24.0,"1wk":168.0}.get(ikey,1.0)

def _is_stale(df:pd.DataFrame,symbol:str,ikey:str)->bool:
    if df is None or df.empty: return True
    if symbol=="BTC-USD": return False
    try:
        last_ts=df.index[-1]
        if not isinstance(last_ts,pd.Timestamp): return False
        now_utc=pd.Timestamp.utcnow()
        last_utc=last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize("UTC")
        age_hours=(now_utc-last_utc).total_seconds()/3600.0
        limit=STALE_MULTIPLIER_HOURS*_interval_hours(ikey)
        return age_hours>limit
    except Exception:
        return False

# ---------------- indicator helpers -----------------

def _ema_fallback(s:pd.Series,span:int)->pd.Series:
    return s.ewm(span=span,adjust=False).mean()

def _rsi_fallback(close:pd.Series,window:int=14)->pd.Series:
    d=close.diff()
    gain=d.clip(lower=0).rolling(window).mean()
    loss=-d.clip(upper=0).rolling(window).mean()
    rs=gain/(loss.replace(0,np.nan))
    return 100-(100/(1+rs))

def _macd_fallback(close:pd.Series):
    ema12=close.ewm(span=12,adjust=False).mean()
    ema26=close.ewm(span=26,adjust=False).mean()
    line=ema12-ema26
    sig=line.ewm(span=9,adjust=False).mean()
    hist=line-sig
    return line,sig,hist

def _atr_fallback(h, l, c, window:int=14)->pd.Series:
    tr1=(h-l).abs()
    tr2=(h-c.shift(1)).abs()
    tr3=(l-c.shift(1)).abs()
    tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    return tr.rolling(window).mean()

def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df=df.copy()
    for col in ["Close","High","Low"]:
        if col not in df.columns: return pd.DataFrame()
        vals=df[col].values
        if getattr(vals,"ndim",1)>1:
            df[col]=pd.Series(vals.ravel(),index=df.index)
        df[col]=pd.to_numeric(df[col],errors="coerce")

    c=df["Close"]; h=df["High"]; l=df["Low"]

    # EMA 20/50
    try:
        if EMAIndicator is None: raise RuntimeError
        df["ema20"]=EMAIndicator(close=c,window=20).ema_indicator()
        df["ema50"]=EMAIndicator(close=c,window=50).ema_indicator()
    except Exception:
        df["ema20"]=_ema_fallback(c,20)
        df["ema50"]=_ema_fallback(c,50)

    # RSI
    try:
        if RSIIndicator is None: raise RuntimeError
        df["RSI"]=RSIIndicator(close=c,window=14).rsi()
    except Exception:
        df["RSI"]=_rsi_fallback(c,14)
    df["rsi"]=df["RSI"]

    # MACD
    try:
        if MACD is None: raise RuntimeError
        macd_obj=MACD(close=c)
        df["macd"]=macd_obj.macd()
        df["macd_signal"]=macd_obj.macd_signal()
        df["macd_hist"]=macd_obj.macd_diff()
    except Exception:
        line,sig,hist=_macd_fallback(c)
        df["macd"]=line; df["macd_signal"]=sig; df["macd_hist"]=hist

    # ATR
    try:
        if AverageTrueRange is None: raise RuntimeError
        atr_calc=AverageTrueRange(high=h,low=l,close=c,window=14)
        df["atr"]=atr_calc.average_true_range()
    except Exception:
        df["atr"]=_atr_fallback(h,l,c,14)

    # ADX for trend strength
    try:
        if ADXIndicator is None: raise RuntimeError
        adx_obj=ADXIndicator(high=h,low=l,close=c,window=14)
        df["adx"]=adx_obj.adx()
    except Exception:
        df["adx"]=np.nan

    # Bollinger bands for volatility expansion
    try:
        if BollingerBands is None: raise RuntimeError
        bb=BollingerBands(close=c,window=BB_WINDOW,window_dev=2)
        df["bb_mid"]=bb.bollinger_mavg()
        df["bb_upper"]=bb.bollinger_hband()
        df["bb_lower"]=bb.bollinger_lband()
        df["bb_width"]= (df["bb_upper"]-df["bb_lower"]).abs()/df["bb_mid"].replace(0,np.nan)
    except Exception:
        df["bb_mid"]=np.nan; df["bb_upper"]=np.nan; df["bb_lower"]=np.nan; df["bb_width"]=np.nan

    # rolling support / resistance
    df["support_level"]=df["Low"].rolling(STRUCT_WINDOW,min_periods=3).min()
    df["resistance_level"]=df["High"].rolling(STRUCT_WINDOW,min_periods=3).max()

    # range bounds for Range Reversal
    df["range_low"]=df["Low"].rolling(RANGE_LOOKBACK,min_periods=5).min()
    df["range_high"]=df["High"].rolling(RANGE_LOOKBACK,min_periods=5).max()

    # swing pivots for Swing Structure
    swing_low = (df["Low"].shift(1).rolling(SWING_LOOKBACK).min()==df["Low"].shift(1)) & \
                (df["Low"].shift(1)<df["Low"] ) & (df["Low"].shift(1)<df["Low"].shift(2))
    swing_high= (df["High"].shift(1).rolling(SWING_LOOKBACK).max()==df["High"].shift(1)) & \
                (df["High"].shift(1)>df["High"]) & (df["High"].shift(1)>df["High"].shift(2))
    df["swing_low_series"]=np.where(swing_low.shift(-1), df["Low"], np.nan)
    df["swing_high_series"]=np.where(swing_high.shift(-1), df["High"], np.nan)

    # RSI adaptive bands for mean reversion
    rsi_roll=df["RSI"].rolling(RSI_WINDOW,min_periods=10)
    df["rsi_low_band"]=rsi_roll.quantile(RSI_Q_LOW)
    df["rsi_high_band"]=rsi_roll.quantile(RSI_Q_HIGH)

    # convenience slopes
    df["ema50_slope"]=df["ema50"].diff()
    df["macd_slope"]=df["macd"].diff()
    df["rsi_slope"]=df["RSI"].diff()

    # ATR pct of price
    df["atr_pct"]= (df["atr"] / df["Close"].replace(0,np.nan))*100.0

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True); df.dropna(how="all",inplace=True)
    return df

# -------------- structure / strategy flags ----------

def _flag_buy_dip(row):
    # uptrend + near support + oversold RSI
    if row["ema20"]>row["ema50"]:
        if row["support_level"]>0:
            prox=(row["Close"]-row["support_level"])/row["support_level"]
            if prox<=SUP_PROX_PCT:
                if (row["RSI"]<=row.get("rsi_low_band",np.nan)) or (row["RSI"]<RSI_DIP_THRESHOLD):
                    return True
    return False

def _flag_breakout_long(row):
    # breakout above resistance with bullish bias
    if row["ema20"]>row["ema50"] and row["macd_slope"]>0:
        if not math.isnan(row.get("resistance_level",np.nan)):
            if row["Close"]>row["resistance_level"]:
                return True
    return False

def _flag_breakout_short(row):
    # breakdown below support w/ bearish bias
    if row["ema20"]<row["ema50"] and row["macd_slope"]<0:
        if not math.isnan(row.get("support_level",np.nan)):
            if row["Close"]<row["support_level"]:
                return True
    return False

def _flag_mean_rev_long(row):
    # fade oversold in non-trend
    if row["adx"]<ADX_MIN_TREND:
        if row["RSI"]<min(row.get("rsi_low_band",100),40):
            if row["Close"]<row["ema20"]:
                return True
    return False

def _flag_mean_rev_short(row):
    if row["adx"]<ADX_MIN_TREND:
        if row["RSI"]>max(row.get("rsi_high_band",0),60):
            if row["Close"]>row["ema20"]:
                return True
    return False

def _flag_trend_cont_long(row):
    if row["ema20"]>row["ema50"] and row["adx"]>=ADX_MIN_TREND and row["RSI"]>50:
        return True
    return False

def _flag_trend_cont_short(row):
    if row["ema20"]<row["ema50"] and row["adx"]>=ADX_MIN_TREND and row["RSI"]<50:
        return True
    return False

def _flag_volexp_long(row):
    # squeeze + upside expansion
    if row.get("bb_width",np.nan)<=BB_WIDTH_PCT_THRESH and row["ema20"]>row["ema50"]:
        if row["Close"]>row.get("bb_upper",np.nan):
            return True
    return False

def _flag_volexp_short(row):
    if row.get("bb_width",np.nan)<=BB_WIDTH_PCT_THRESH and row["ema20"]<row["ema50"]:
        if row["Close"]<row.get("bb_lower",np.nan):
            return True
    return False

def _flag_range_rev_long(row):
    # near lower bound of range, RSI curling up
    rl=row.get("range_low",np.nan); rh=row.get("range_high",np.nan)
    if not math.isnan(rl) and not math.isnan(rh):
        width = rh-rl
        if width>0:
            dist=(row["Close"]-rl)/width
            if dist<0.15 and row["rsi_slope"]>0:
                return True
    return False

def _flag_range_rev_short(row):
    rl=row.get("range_low",np.nan); rh=row.get("range_high",np.nan)
    if not math.isnan(rl) and not math.isnan(rh):
        width=rh-rl
        if width>0:
            dist=(rh-row["Close"])/width
            if dist<0.15 and row["rsi_slope"]<0:
                return True
    return False

def _flag_swing_long(row, prev_low, prev_prev_low):
    # higher low + MACD curling up
    if not math.isnan(prev_low) and not math.isnan(prev_prev_low):
        if prev_low>prev_prev_low and row["macd_slope"]>0:
            return True
    return False

def _flag_swing_short(row, prev_high, prev_prev_high):
    if not math.isnan(prev_high) and not math.isnan(prev_prev_high):
        if prev_high<prev_prev_high and row["macd_slope"]<0:
            return True
    return False

def _add_strategy_flags(df:pd.DataFrame)->pd.DataFrame:
    if df is None or df.empty: return df
    df=df.copy()
    # gather swing reference lows/highs
    swing_lows = df["swing_low_series"].dropna()
    swing_highs= df["swing_high_series"].dropna()

    last_two_lows = []
    last_two_highs= []

    mr_long=[]; mr_short=[]
    tc_long=[]; tc_short=[]
    ve_long=[]; ve_short=[]
    rr_long=[]; rr_short=[]
    sw_long=[]; sw_short=[]
    dip_flag=[]; bo_long=[]; bo_short=[]

    prev_low = np.nan; prev_prev_low=np.nan
    prev_high=np.nan; prev_prev_high=np.nan

    for idx,row in df.iterrows():
        # update swing memory
        if not math.isnan(row.get("swing_low_series",np.nan)):
            prev_prev_low=prev_low
            prev_low=row["swing_low_series"]
        if not math.isnan(row.get("swing_high_series",np.nan)):
            prev_prev_high=prev_high
            prev_high=row["swing_high_series"]

        dip_flag.append(_flag_buy_dip(row))
        bo_long.append(_flag_breakout_long(row))
        bo_short.append(_flag_breakout_short(row))

        mr_long.append(_flag_mean_rev_long(row))
        mr_short.append(_flag_mean_rev_short(row))

        tc_long.append(_flag_trend_cont_long(row))
        tc_short.append(_flag_trend_cont_short(row))

        ve_long.append(_flag_volexp_long(row))
        ve_short.append(_flag_volexp_short(row))

        rr_long.append(_flag_range_rev_long(row))
        rr_short.append(_flag_range_rev_short(row))

        sw_long.append(_flag_swing_long(row, prev_low, prev_prev_low))
        sw_short.append(_flag_swing_short(row, prev_high, prev_prev_high))

    df["dip_buy_flag"]=dip_flag
    df["bull_breakout_flag"]=bo_long
    df["bear_breakdown_flag"]=bo_short

    df["mr_long_flag"]=mr_long
    df["mr_short_flag"]=mr_short
    df["trend_cont_long_flag"]=tc_long
    df["trend_cont_short_flag"]=tc_short
    df["volexp_long_flag"]=ve_long
    df["volexp_short_flag"]=ve_short
    df["range_rev_long_flag"]=rr_long
    df["range_rev_short_flag"]=rr_short
    df["swing_long_flag"]=sw_long
    df["swing_short_flag"]=sw_short

    return df

# ---------------- rule-based signal row -------------

def compute_signal_row(prev_row:pd.Series, row:pd.Series)->Tuple[str,float]:
    score=0.0; votes=0
    # EMA trend vote
    if pd.notna(row.get("ema20")) and pd.notna(row.get("ema50")):
        votes+=1
        if row["ema20"]>row["ema50"]: score+=1.0
        elif row["ema20"]<row["ema50"]: score-=1.0
    # RSI extremes vote
    rsi_val=row.get("RSI")
    if pd.notna(rsi_val):
        votes+=1
        if rsi_val<30: score+=1.0
        elif rsi_val>70: score-=1.0
    # MACD cross vote
    a1=row.get("macd"); b1=row.get("macd_signal")
    a0=prev_row.get("macd"); b0=prev_row.get("macd_signal")
    if pd.notna(a1) and pd.notna(b1) and pd.notna(a0) and pd.notna(b0):
        votes+=1
        crossed_up=(a0<=b0 and a1>b1)
        crossed_dn=(a0>=b0 and a1<b1)
        if crossed_up: score+=1.0
        elif crossed_dn: score-=1.0

    conf=0.0 if votes==0 else min(1.0,abs(score)/votes)
    if score>=0.67*votes:
        return "Buy", conf
    elif score<=-0.67*votes:
        return "Sell", conf
    else:
        return "Hold", 1.0-conf

def _calc_tp_sl(price:float, atr:float, side:str, risk:str,
                tp_sl_mode:str, adx:float, atr_pct:float)->Tuple[float,float]:
    base=RISK_MULT.get(risk,RISK_MULT["Medium"])
    tp_k=float(base["tp_atr"]); sl_k=float(base["sl_atr"])

    prof=_TP_SL_PROFILES.get(tp_sl_mode,_TP_SL_PROFILES["Normal"])
    # Widen TP in strong trend, widen SL when choppy
    trend_scale=np.interp(adx,[0,40],[prof["trend_min"],prof["trend_max"]])
    vol_scale=np.interp(atr_pct,[0,2],[prof["vol_min"],prof["vol_max"]])
    tp_scale=(trend_scale+vol_scale)/2.0
    sl_scale=(trend_scale+vol_scale)/2.0

    tp_k*=tp_scale; sl_k*=sl_scale
    if side=="Buy":
        return price+tp_k*atr, price-sl_k*atr
    else:
        return price-tp_k*atr, price+sl_k*atr

# ---------------- ML confidence (5-fold CV) ---------

def _prepare_ml_df(df:pd.DataFrame, recent_wr_hint:Optional[float])->Optional[pd.DataFrame]:
    if df is None or df.empty or len(df)<60: return None
    work=df.copy()
    feat_cols=["RSI","ema20","ema50","ema50_slope","macd","macd_signal",
               "macd_slope","atr_pct","adx","rsi_slope"]
    for c in feat_cols:
        if c not in work.columns: work[c]=np.nan
    work["target_up"]=(work["Close"].shift(-1)>work["Close"]).astype(int)
    work["recent_wr_hint"]=recent_wr_hint if recent_wr_hint is not None else 0.5
    work.dropna(inplace=True)
    if len(work)<40: return None
    return work

def _ml_confidence_raw(df:pd.DataFrame, recent_wr_hint:Optional[float])->float:
    if RandomForestClassifier is None or KFold is None:
        return 0.5
    work=_prepare_ml_df(df,recent_wr_hint)
    if work is None: return 0.5
    feat_cols=["RSI","ema20","ema50","ema50_slope","macd","macd_signal",
               "macd_slope","atr_pct","adx","rsi_slope","recent_wr_hint"]
    X=work[feat_cols].values
    y=work["target_up"].values
    if len(y)<40: return 0.5
    kf=KFold(n_splits=CV_FOLDS,shuffle=True,random_state=42)
    probs_rf=[]; probs_gb=[]
    for tr_idx,te_idx in kf.split(X):
        Xtr,Xte=X[tr_idx],X[te_idx]
        ytr,yte=y[tr_idx],y[te_idx]
        rf=RandomForestClassifier(n_estimators=40,max_depth=4,random_state=42)
        rf.fit(Xtr,ytr)
        probs_rf.extend(rf.predict_proba(Xte)[:,1])
        if GradientBoostingClassifier is not None:
            gb=GradientBoostingClassifier(random_state=42)
            gb.fit(Xtr,ytr)
            probs_gb.extend(gb.predict_proba(Xte)[:,1])
    if probs_rf:
        rf_avg=float(np.mean(probs_rf))
    else:
        rf_avg=0.5
    gb_avg=float(np.mean(probs_gb)) if probs_gb else rf_avg
    blended=0.6*rf_avg+0.4*gb_avg
    return float(max(0.05,min(0.95,blended)))

# -------------- structure_mode gate -----------------

def _structure_allows(side:str, row:pd.Series, mode:str)->bool:
    # side in {"Buy","Sell","Hold"}
    if side not in ("Buy","Sell"): return False if mode!="Off" else True

    if mode=="Off":
        return True

    if mode=="Buy Dips":
        return side=="Buy" and row.get("dip_buy_flag",False)

    if mode=="Breakouts":
        if side=="Buy":
            return row.get("bull_breakout_flag",False)
        else:
            return row.get("bear_breakdown_flag",False)

    if mode=="Both (Dips + Breakouts)":
        if side=="Buy":
            return row.get("dip_buy_flag",False) or row.get("bull_breakout_flag",False)
        else:
            return row.get("bear_breakdown_flag",False)

    if mode=="Mean Reversion":
        if side=="Buy":
            return row.get("mr_long_flag",False)
        else:
            return row.get("mr_short_flag",False)

    if mode=="Trend Continuation":
        if side=="Buy":
            return row.get("trend_cont_long_flag",False)
        else:
            return row.get("trend_cont_short_flag",False)

    if mode=="Volatility Expansion":
        if side=="Buy":
            return row.get("volexp_long_flag",False)
        else:
            return row.get("volexp_short_flag",False)

    if mode=="Range Reversal":
        if side=="Buy":
            return row.get("range_rev_long_flag",False)
        else:
            return row.get("range_rev_short_flag",False)

    if mode=="Swing Structure":
        if side=="Buy":
            return row.get("swing_long_flag",False)
        else:
            return row.get("swing_short_flag",False)

    return False

# -------------- latest_prediction snapshot ----------

def latest_prediction(
    df_raw:pd.DataFrame,
    risk:str="Medium",
    tp_sl_mode:str="Normal",
    structure_mode:str="Off",
    forced_trades_enabled:bool=False,
    filter_level:str="Balanced",
    calibration_enabled:bool=True,
) -> Optional[Dict[str,Any]]:

    if df_raw is None or df_raw.empty or len(df_raw)<60:
        return None

    df=add_indicators(df_raw)
    df=_add_strategy_flags(df)
    if df.empty or len(df)<60: return None

    # apply calibration + filter level to globals for gating thresholds
    calib=_load_calib()
    bias=_calib_bias_for_symbol(calib,"",calibration_enabled)  # symbol="" here, caller can override
    _apply_filter_level(filter_level,bias)

    prev_row=df.iloc[-2]; row_now=df.iloc[-1]
    side,rule_conf=compute_signal_row(prev_row,row_now)

    ml_conf=_ml_confidence_raw(df,bias.get("avg_wr",50.0)/100.0)
    blended=0.5*rule_conf+0.5*ml_conf
    blended=max(0.05,min(0.95,blended))

    # gate through structure mode
    allowed=_structure_allows(side,row_now,structure_mode)
    if not allowed:
        side="Hold"

    last_price=float(row_now["Close"])
    atr_now=float(row_now.get("atr", last_price*ATR_FLOOR_MULT))
    adx_now=float(row_now.get("adx",0.0))
    atr_pct=float(row_now.get("atr_pct",0.0))

    if side=="Hold":
        return {
            "side":"Hold",
            "probability":round(blended,2),
            "price":last_price,
            "tp":None,"sl":None,"rr":None,
            "atr":atr_now,
            "adx":adx_now,
        }

    tp,sl=_calc_tp_sl(last_price,atr_now,side,risk,tp_sl_mode,adx_now,atr_pct)
    if side=="Buy":
        reward=tp-last_price; riskv=last_price-sl
    else:
        reward=last_price-tp; riskv=sl-last_price
    rr = (reward/riskv) if (riskv and riskv!=0) else None

    return {
        "side":side,
        "probability":round(blended,2),
        "price":last_price,
        "tp":float(tp),
        "sl":float(sl),
        "rr": float(rr) if rr and math.isfinite(rr) else None,
        "atr":atr_now,
        "adx":adx_now,
    }

# -------------- backtest w/ PF / EV% ----------------

def _simulate_trade_path(df,i,side,risk,tp_sl_mode,horizon:int=10):
    row=df.iloc[i]
    px0=float(row["Close"])
    atr_now=float(row.get("atr",px0*ATR_FLOOR_MULT))
    adx_now=float(row.get("adx",0.0))
    atr_pct=float(row.get("atr_pct",0.0))
    tp_lvl,sl_lvl=_calc_tp_sl(px0,atr_now,side,risk,tp_sl_mode,adx_now,atr_pct)

    for j in range(1,horizon+1):
        if i+j>=len(df): break
        nxt=df.iloc[i+j]; nxt_px=float(nxt["Close"])
        if side=="Buy":
            if nxt_px>=tp_lvl: return 0.01,True   # +1%
            if nxt_px<=sl_lvl: return -0.01,False # -1%
        else:
            if nxt_px<=tp_lvl: return 0.01,True
            if nxt_px>=sl_lvl: return -0.01,False
    return 0.0,False  # neither TP/SL hit in horizon

def backtest_signals(
    df_raw:pd.DataFrame,
    risk:str,
    horizon:int,
    structure_mode:str,
    forced_trades_enabled:bool,
    filter_level:str,
    calibration_enabled:bool,
    tp_sl_mode:str,
) -> Dict[str,Any]:

    out={"winrate":0.0,"trades":0,"return":0.0,"maxdd":0.0,
         "sharpe":0.0,"pf":0.0,"ev_per_trade":0.0,"wr_std":0.0}

    if df_raw is None or df_raw.empty or len(df_raw)<60: return out
    df=add_indicators(df_raw); df=_add_strategy_flags(df)
    if df.empty or len(df)<60: return out

    calib=_load_calib()
    bias=_calib_bias_for_symbol(calib,"",calibration_enabled)
    _apply_filter_level(filter_level,bias)

    balance=1.0; peak=1.0
    trades=0; wins=0
    pnl_list=[]; dd_list=[]

    for i in range(20,len(df)-horizon):
        prev_row=df.iloc[i-1]; cur=df.iloc[i]

        side,rule_conf=compute_signal_row(prev_row,cur)
        ml_conf=_ml_confidence_raw(df.iloc[:i+1], bias.get("avg_wr",50.0)/100.0)
        blended=0.5*rule_conf+0.5*ml_conf
        blended=max(0.05,min(0.95,blended))

        # skip low confidence
        if blended<CONF_EXEC_THRESHOLD:
            # maybe force
            if not forced_trades_enabled or blended<FORCED_CONF_MIN:
                continue
            # forced side random if Hold
            if side=="Hold":
                side=np.random.choice(["Buy","Sell"])

        # gate by structure_mode
        if not _structure_allows(side,cur,structure_mode):
            # forced entry allowed?
            if not forced_trades_enabled:
                continue
            # if forced, randomize side that IS allowed?
            # simplest: skip
            continue

        # simulate horizon
        pnl,won=_simulate_trade_path(df,i,side,risk,tp_sl_mode,horizon)
        trades+=1
        if won: wins+=1
        balance*= (1.0+pnl)
        pnl_list.append(pnl)

        peak=max(peak,balance)
        dd=(peak-balance)/peak if peak>0 else 0
        dd_list.append(dd)

    if trades>0:
        total_ret=(balance-1.0)*100.0
        wr_pct=(wins/trades)*100.0
        maxdd_pct=(max(dd_list)*100.0) if dd_list else 0.0
        sharpe_like= (wr_pct/maxdd_pct) if maxdd_pct>0 else wr_pct

        pnl_w=[p for p in pnl_list if p>0]
        pnl_l=[-p for p in pnl_list if p<0]
        pf = (np.sum(pnl_w)/np.sum(pnl_l)) if pnl_l else float("inf")

        ev_per_trade = np.mean(pnl_list)*100.0  # % per trade
        wr_std = float(np.std([1.0 if p>0 else 0.0 for p in pnl_list]))*100.0

        out.update({
            "winrate":round(wr_pct,2),
            "trades":trades,
            "return":round(total_ret,2),
            "maxdd":round(maxdd_pct,2),
            "sharpe":round(sharpe_like,2),
            "pf":round(pf,2) if math.isfinite(pf) else pf,
            "ev_per_trade":round(ev_per_trade,2),
            "wr_std":round(wr_std,2),
        })

    # record calibration for future biasing
    calib=_record_calib(calib,"",out["winrate"] if trades>0 else 50.0)
    _save_calib(calib)

    return out

# -------------- sentiment stub ----------------------
def compute_sentiment_stub(symbol:str)->float:
    bias_map={"BTC-USD":0.65,"^NDX":0.6,"CL=F":0.55}
    base=bias_map.get(symbol,0.5)
    return round(float(base),2)

# -------------- analyze single asset ----------------

def analyze_asset(
    symbol:str,
    interval_key:str,
    risk:str="Medium",
    tp_sl_mode:str="Normal",
    structure_mode:str="Off",
    forced_trades_enabled:bool=False,
    filter_level:str="Balanced",
    calibration_enabled:bool=True,
    weekend_mode:bool=True,
) -> Optional[Dict[str,Any]]:

    df_raw=fetch_data(symbol,interval_key=interval_key,use_cache=True)
    if df_raw.empty: return None

    # stale/weekend skip
    stale_flag=_is_stale(df_raw,symbol,interval_key)
    if weekend_mode and stale_flag:
        # mark stale but still continue; we just surface stale=True
        pass

    pred=latest_prediction(
        df_raw,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
    )

    bt=backtest_signals(
        df_raw,
        risk=risk,
        horizon=10,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        tp_sl_mode=tp_sl_mode,
    )

    df_ind=add_indicators(df_raw)
    df_ind=_add_strategy_flags(df_ind)

    last_px=float(df_ind["Close"].iloc[-1])
    atr_last=float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None
    sent_val=compute_sentiment_stub(symbol)

    if pred is None:
        merged = {
            "symbol":symbol,
            "interval_key":interval_key,
            "risk":risk,
            "last_price":last_px,
            "signal":"Hold",
            "probability":0.5,
            "tp":None,"sl":None,"rr":None,
            "atr":atr_last,
            "adx":float(df_ind["adx"].iloc[-1]) if "adx" in df_ind.columns else None,
            "sentiment":sent_val,
            "stale":stale_flag,
            "pf":bt["pf"],
            "ev_per_trade":bt["ev_per_trade"],
            "wr_std":bt["wr_std"],
            "winrate":bt["winrate"],
            "return_pct":bt["return"],
            "trades":bt["trades"],
            "maxdd":bt["maxdd"],
            "sharpe":bt["sharpe"],
            "df":df_ind,
        }
        return merged

    merged={
        "symbol":symbol,
        "interval_key":interval_key,
        "risk":risk,
        "last_price":last_px,
        "signal":pred["side"],
        "probability":pred["probability"],
        "tp":pred["tp"],
        "sl":pred["sl"],
        "rr":pred["rr"],
        "atr":pred["atr"],
        "adx":pred.get("adx"),
        "sentiment":sent_val,
        "stale":stale_flag,
        "pf":bt["pf"],
        "ev_per_trade":bt["ev_per_trade"],
        "wr_std":bt["wr_std"],
        "winrate":bt["winrate"],
        "return_pct":bt["return"],
        "trades":bt["trades"],
        "maxdd":bt["maxdd"],
        "sharpe":bt["sharpe"],
        "df":df_ind,
    }
    return merged

# -------------- generate points for plotting --------

def generate_signal_points(df:pd.DataFrame, structure_mode:str)->Dict[str,Any]:
    """
    Extracts coordinates for markers / overlays so app.py can plot:
    - buys/sells (post-structure gate)
    - dip buys, breakouts, etc for colouring
    - bollinger bands, range bounds, swing pivots
    """
    out={
        "buy_x":[], "buy_y":[],
        "sell_x":[],"sell_y":[],
        "dip_x":[],"dip_y":[],
        "bo_long_x":[],"bo_long_y":[],
        "bo_short_x":[],"bo_short_y":[],
        "mr_long_x":[],"mr_long_y":[],
        "mr_short_x":[],"mr_short_y":[],
        "tc_long_x":[],"tc_long_y":[],
        "tc_short_x":[],"tc_short_y":[],
        "ve_long_x":[],"ve_long_y":[],
        "ve_short_x":[],"ve_short_y":[],
        "rr_long_x":[],"rr_long_y":[],
        "rr_short_x":[],"rr_short_y":[],
        "swing_low_x":[],"swing_low_y":[],
        "swing_high_x":[],"swing_high_y":[],
        "bb_upper_x":[],"bb_upper_y":[],
        "bb_lower_x":[],"bb_lower_y":[],
        "range_hi_x":[],"range_hi_y":[],
        "range_lo_x":[],"range_lo_y":[],
        "sup_x":[],"sup_y":[],
        "res_x":[],"res_y":[],
    }
    if df is None or df.empty: return out

    for i in range(1,len(df)):
        prev_row=df.iloc[i-1]; row=df.iloc[i]
        side,_=compute_signal_row(prev_row,row)
        allowed=_structure_allows(side,row,structure_mode)
        if not allowed:
            side="Hold"

        idx=row.name
        px=row["Close"]

        # core buy/sell markers (post gate)
        if side=="Buy":
            out["buy_x"].append(idx); out["buy_y"].append(px)
        elif side=="Sell":
            out["sell_x"].append(idx); out["sell_y"].append(px)

        # style-specific markers (raw flags)
        if row.get("dip_buy_flag",False):
            out["dip_x"].append(idx); out["dip_y"].append(px)
        if row.get("bull_breakout_flag",False):
            out["bo_long_x"].append(idx); out["bo_long_y"].append(px)
        if row.get("bear_breakdown_flag",False):
            out["bo_short_x"].append(idx); out["bo_short_y"].append(px)

        if row.get("mr_long_flag",False):
            out["mr_long_x"].append(idx); out["mr_long_y"].append(px)
        if row.get("mr_short_flag",False):
            out["mr_short_x"].append(idx); out["mr_short_y"].append(px)

        if row.get("trend_cont_long_flag",False):
            out["tc_long_x"].append(idx); out["tc_long_y"].append(px)
        if row.get("trend_cont_short_flag",False):
            out["tc_short_x"].append(idx); out["tc_short_y"].append(px)

        if row.get("volexp_long_flag",False):
            out["ve_long_x"].append(idx); out["ve_long_y"].append(px)
        if row.get("volexp_short_flag",False):
            out["ve_short_x"].append(idx); out["ve_short_y"].append(px)

        if row.get("range_rev_long_flag",False):
            out["rr_long_x"].append(idx); out["rr_long_y"].append(px)
        if row.get("range_rev_short_flag",False):
            out["rr_short_x"].append(idx); out["rr_short_y"].append(px)

        if not math.isnan(row.get("swing_low_series",np.nan)):
            out["swing_low_x"].append(idx); out["swing_low_y"].append(row["swing_low_series"])
        if not math.isnan(row.get("swing_high_series",np.nan)):
            out["swing_high_x"].append(idx); out["swing_high_y"].append(row["swing_high_series"])

        # overlays
        if not math.isnan(row.get("bb_upper",np.nan)):
            out["bb_upper_x"].append(idx); out["bb_upper_y"].append(row["bb_upper"])
        if not math.isnan(row.get("bb_lower",np.nan)):
            out["bb_lower_x"].append(idx); out["bb_lower_y"].append(row["bb_lower"])

        if not math.isnan(row.get("range_high",np.nan)):
            out["range_hi_x"].append(idx); out["range_hi_y"].append(row["range_high"])
        if not math.isnan(row.get("range_low",np.nan)):
            out["range_lo_x"].append(idx); out["range_lo_y"].append(row["range_low"])

        if not math.isnan(row.get("support_level",np.nan)):
            out["sup_x"].append(idx); out["sup_y"].append(row["support_level"])
        if not math.isnan(row.get("resistance_level",np.nan)):
            out["res_x"].append(idx); out["res_y"].append(row["resistance_level"])

    return out

# -------------- summarize / public wrappers ---------

def summarize_assets(
    interval_key:str="1h",
    risk:str="Medium",
    tp_sl_mode:str="Normal",
    structure_mode:str="Off",
    forced_trades_enabled:bool=False,
    filter_level:str="Balanced",
    calibration_enabled:bool=True,
    weekend_mode:bool=True,
)->pd.DataFrame:
    rows=[]
    for asset_name,symbol in ASSET_SYMBOLS.items():
        try:
            res=analyze_asset(
                symbol,
                interval_key,
                risk,
                tp_sl_mode,
                structure_mode,
                forced_trades_enabled,
                filter_level,
                calibration_enabled,
                weekend_mode,
            )
        except Exception as e:
            _log(f"Error {asset_name}: {e}")
            res=None
        if not res: continue
        rows.append({
            "Asset":asset_name,
            "Symbol":symbol,
            "Interval":interval_key,
            "Price":res["last_price"],
            "Signal":res["signal"],
            "Probability":res["probability"],
            "TP":res["tp"],
            "SL":res["sl"],
            "RR":res["rr"],
            "Trades":res["trades"],
            "WinRate":res["winrate"],
            "WinRateStd":res["wr_std"],
            "Return%":res["return_pct"],
            "PF":res["pf"],
            "EV%/Trade":res["ev_per_trade"],
            "MaxDD%":res["maxdd"],
            "SharpeLike":res["sharpe"],
            "Sentiment":res["sentiment"],
            "Stale":res["stale"],
        })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows)

def load_asset_with_indicators(
    asset:str,
    interval_key:str,
    structure_mode:str="Off",
    forced_trades_enabled:bool=False,
    filter_level:str="Balanced",
    calibration_enabled:bool=True,
):
    if asset not in ASSET_SYMBOLS:
        raise KeyError(f"Unknown asset '{asset}'")
    symbol=ASSET_SYMBOLS[asset]
    df_raw=fetch_data(symbol,interval_key=interval_key,use_cache=True)
    df_ind=add_indicators(df_raw)
    df_ind=_add_strategy_flags(df_ind)
    pts=generate_signal_points(df_ind,structure_mode)
    return symbol, df_ind, pts

def asset_prediction_and_backtest(
    asset:str,
    interval_key:str,
    risk:str,
    tp_sl_mode:str,
    structure_mode:str,
    forced_trades_enabled:bool,
    filter_level:str,
    calibration_enabled:bool,
    weekend_mode:bool,
):
    if asset not in ASSET_SYMBOLS:
        return None, pd.DataFrame(), {}
    symbol=ASSET_SYMBOLS[asset]
    df_raw=fetch_data(symbol,interval_key=interval_key,use_cache=True)
    if df_raw.empty:
        return None, pd.DataFrame(), {}

    pred=latest_prediction(
        df_raw,
        risk=risk,
        tp_sl_mode=tp_sl_mode,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
    )

    bt=backtest_signals(
        df_raw,
        risk=risk,
        horizon=10,
        structure_mode=structure_mode,
        forced_trades_enabled=forced_trades_enabled,
        filter_level=filter_level,
        calibration_enabled=calibration_enabled,
        tp_sl_mode=tp_sl_mode,
    )

    df_ind=add_indicators(df_raw)
    df_ind=_add_strategy_flags(df_ind)
    pts=generate_signal_points(df_ind,structure_mode)

    last_px=float(df_ind["Close"].iloc[-1])
    sent_val=compute_sentiment_stub(symbol)
    stale_flag=_is_stale(df_raw,symbol,interval_key)

    if pred is None:
        block={
            "asset":asset,
            "symbol":symbol,
            "interval":interval_key,
            "price":last_px,
            "side":"Hold",
            "probability":0.5,
            "tp":None,"sl":None,"rr":None,
            "atr":float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None,
            "adx":float(df_ind["adx"].iloc[-1]) if "adx" in df_ind.columns else None,
            "sentiment":sent_val,
            "stale":stale_flag,
            "win_rate":bt["winrate"],
            "win_rate_std":bt["wr_std"],
            "profit_factor":bt["pf"],
            "ev_per_trade":bt["ev_per_trade"],
            "backtest_return_pct":bt["return"],
            "trades":bt["trades"],
            "maxdd":bt["maxdd"],
            "sharpe":bt["sharpe"],
        }
        return block, df_ind, pts

    block={
        "asset":asset,
        "symbol":symbol,
        "interval":interval_key,
        "price":last_px,
        "side":pred["side"],
        "probability":pred["probability"],
        "tp":pred["tp"],
        "sl":pred["sl"],
        "rr":pred["rr"],
        "atr":pred["atr"],
        "adx":pred.get("adx"),
        "sentiment":sent_val,
        "stale":stale_flag,
        "win_rate":bt["winrate"],
        "win_rate_std":bt["wr_std"],
        "profit_factor":bt["pf"],
        "ev_per_trade":bt["ev_per_trade"],
        "backtest_return_pct":bt["return"],
        "trades":bt["trades"],
        "maxdd":bt["maxdd"],
        "sharpe":bt["sharpe"],
    }
    return block, df_ind, pts
