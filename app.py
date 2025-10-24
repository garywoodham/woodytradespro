from __future__ import annotations

import os
import json
import math
import pickle
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def get_logger(name: str = "forecast", level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger with a standard format.
    Ensures we don't attach duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


log = get_logger()


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Central config you can save with each trained model so inference
    can rebuild preprocessing identically.

    Extend this with any hyperparams you care about.
    """
    target_col: str = "gold_price"
    timestamp_col: str = "date"
    feature_cols: Optional[List[str]] = None  # set after feature eng
    lookback: int = 60        # timesteps fed into model
    horizon: int = 1          # forecast horizon (t+1 by default)
    scaler_type: str = "standard"  # "standard" or "minmax"
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test is whatever is left
    freq: Optional[str] = None  # e.g. "D" if you'd like to resample
    # add model-specific stuff here if needed:
    model_type: str = "LSTM"
    notes: str = ""


def save_config(cfg: RunConfig, path: str) -> None:
    """
    Persist config as json (human-readable + git diff friendly).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log.info(f"Config saved -> {path}")


def load_config(path: str) -> RunConfig:
    """
    Load config json and return RunConfig.
    """
    with open(path, "r") as f:
        data = json.load(f)
    cfg = RunConfig(**data)
    log.info(f"Config loaded <- {path}")
    return cfg


# -----------------------------------------------------------------------------
# Paths / filesystem helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """
    Create directory if not exists.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        log.info(f"Created directory: {path}")


# -----------------------------------------------------------------------------
# Data loading & cleaning
# -----------------------------------------------------------------------------

def load_price_data(
    path: str,
    date_col: str = "date",
    price_col: str = "gold_price",
    parse_dates: bool = True,
    sort: bool = True,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load raw price data from CSV (or feather/parquet based on extension),
    clean column names, parse datetime, sort, and optional resample.

    Expected columns: at least [date_col, price_col].
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
    elif ext in [".feather", ".ft"]:
        df = pd.read_feather(path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path}")

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # rename to expected if needed (be forgiving about naming)
    if date_col.lower() != "date" and "date" in df.columns:
        df.rename(columns={"date": date_col}, inplace=True)
    if price_col.lower() != "gold_price" and "gold_price" in df.columns:
        df.rename(columns={"gold_price": price_col}, inplace=True)

    # datetime
    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # drop rows with invalid date or missing price
    df = df.dropna(subset=[date_col, price_col])
    # sort
    if sort:
        df = df.sort_values(date_col)

    # optional resample to regular frequency
    if freq:
        df = (
            df.set_index(date_col)
              .resample(freq)
              .last()  # last known price in period
              .ffill()
              .reset_index()
        )

    df = df.reset_index(drop=True)
    log.info(
        f"Loaded data: {path} -> {len(df)} rows, "
        f"{df[date_col].min()} to {df[date_col].max()}"
    )
    return df


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def add_returns_and_lags(
    df: pd.DataFrame,
    price_col: str,
    lags: Iterable[int] = (1, 2, 3, 5, 10, 20),
    pct_change_windows: Iterable[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """
    Add lagged prices and % returns.
    """
    out = df.copy()

    # % change features
    for win in pct_change_windows:
        out[f"ret_{win}"] = out[price_col].pct_change(win)

    # lag features
    for lag in lags:
        out[f"{price_col}_lag_{lag}"] = out[price_col].shift(lag)

    return out


def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str,
    windows: Iterable[int] = (5, 10, 20, 50, 100, 200),
) -> pd.DataFrame:
    """
    Add common technical indicators (SMA, EMA, rolling vol, RSI-lite).
    This is intentionally self-contained (no ta-lib dependency).
    """
    out = df.copy()

    # simple moving averages & rolling std (volatility)
    for w in windows:
        out[f"sma_{w}"] = out[price_col].rolling(w).mean()
        out[f"vol_{w}"] = (
            out[price_col].pct_change().rolling(w).std()
        )  # realized vol proxy

    # exponential moving averages
    for w in windows:
        out[f"ema_{w}"] = out[price_col].ewm(span=w, adjust=False).mean()

    # RSI (Relative Strength Index) style calc (14 default-ish)
    period = 14
    delta = out[price_col].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return out


def forward_target(
    df: pd.DataFrame,
    price_col: str,
    horizon: int = 1,
    target_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create the prediction target: price at t+horizon (or return to t+horizon).
    Default: raw future price.
    """
    out = df.copy()
    tgt_col = target_name or f"{price_col}_future_{horizon}"
    out[tgt_col] = out[price_col].shift(-horizon)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    After feature engineering + forward_target, split into:
    X (features), y (target), t (timestamps).
    """
    # Keep timestamp separate so we don't scale it.
    t = df[timestamp_col].copy()

    # y
    y = df[target_col].copy()

    # X = everything numeric except timestamp & target
    ignore_cols = {timestamp_col, target_col}
    numeric_cols = [
        c for c in df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[numeric_cols].copy()

    if dropna:
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)
        t = t.loc[valid_mask].reset_index(drop=True)

    return X, y, t


# -----------------------------------------------------------------------------
# Time-based split
# -----------------------------------------------------------------------------

def train_val_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, pd.DataFrame]:
    """
    Deterministic chronological split.
    """
    assert len(X) == len(y) == len(t), "Length mismatch"
    n = len(X)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_test <= 0:
        raise ValueError("Not enough data left for test split. Adjust ratios.")

    idx_train_end = n_train
    idx_val_end = n_train + n_val

    split = {
        "X_train": X.iloc[:idx_train_end].reset_index(drop=True),
        "y_train": y.iloc[:idx_train_end].reset_index(drop=True),
        "t_train": t.iloc[:idx_train_end].reset_index(drop=True),

        "X_val": X.iloc[idx_train_end:idx_val_end].reset_index(drop=True),
        "y_val": y.iloc[idx_train_end:idx_val_end].reset_index(drop=True),
        "t_val": t.iloc[idx_train_end:idx_val_end].reset_index(drop=True),

        "X_test": X.iloc[idx_val_end:].reset_index(drop=True),
        "y_test": y.iloc[idx_val_end:].reset_index(drop=True),
        "t_test": t.iloc[idx_val_end:].reset_index(drop=True),
    }

    log.info(
        f"Split sizes -> train:{len(split['X_train'])} "
        f"val:{len(split['X_val'])} test:{len(split['X_test'])}"
    )

    return split


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------

@dataclass
class Scalers:
    feature_scaler: object
    target_scaler: object


def fit_scalers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler_type: str = "standard",
) -> Scalers:
    """
    Fit feature & target scalers on TRAIN ONLY.
    """
    if scaler_type == "standard":
        feat_scaler = StandardScaler()
        tgt_scaler = StandardScaler()
    elif scaler_type == "minmax":
        feat_scaler = MinMaxScaler()
        tgt_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    feat_scaler.fit(X_train.values)
    tgt_scaler.fit(y_train.values.reshape(-1, 1))

    return Scalers(feat_scaler, tgt_scaler)


def apply_scalers(
    scalers: Scalers,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Transform any split using pre-fit scalers.
    """
    X_scaled = scalers.feature_scaler.transform(X.values)
    y_scaled = None
    if y is not None:
        y_scaled = scalers.target_scaler.transform(y.values.reshape(-1, 1)).ravel()
    return X_scaled, y_scaled


def inverse_target(
    scalers: Scalers,
    y_scaled: np.ndarray,
) -> np.ndarray:
    """
    Undo scaling for predictions or eval.
    """
    return scalers.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# -----------------------------------------------------------------------------
# Sequence builder (for RNN/LSTM/Transformer style models)
# -----------------------------------------------------------------------------

def make_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn tabular time series into supervised sequences.

    Example:
        lookback = 60, horizon = 1
        For index i, X_seq[i] = X_arr[i-lookback : i]
                       y_seq[i] = y_arr[i + horizon - 1]

    We only create sequences where we have full lookback history
    AND full horizon future.
    """
    n_samples = len(X_arr)
    X_seqs = []
    y_seqs = []

    # last usable index for input is n_samples - horizon
    # but we also need i - lookback >= 0  -> i >= lookback
    # so i runs from lookback to n_samples - horizon (inclusive-1)
    max_i = n_samples - horizon
    for i in range(lookback, max_i):
        x_window = X_arr[i - lookback : i, :]
        y_point = y_arr[i + horizon - 1]  # scalar
        X_seqs.append(x_window)
        y_seqs.append(y_point)

    X_seqs = np.stack(X_seqs, axis=0)  # [N, lookback, features]
    y_seqs = np.array(y_seqs)          # [N]

    return X_seqs, y_seqs


def align_sequences_with_timestamps(
    t: pd.Series,
    lookback: int,
    horizon: int,
) -> pd.Series:
    """
    When you build sequences with make_sequences(), you're effectively
    dropping the first <lookback> timestamps and the last <horizon> timestamps.

    This helper returns the timestamps that correspond to each y in y_seqs
    so you can plot pred vs actual on real dates.
    """
    n = len(t)
    usable = []
    max_i = n - horizon
    for i in range(lookback, max_i):
        usable.append(t.iloc[i + horizon - 1])
    return pd.Series(usable).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Metrics / evaluation
# -----------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Common evaluation metrics.
    """
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
    }


def plot_predictions(
    timestamps: Iterable[pd.Timestamp],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    figsize: Tuple[int, int] = (10, 4),
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Simple time series prediction plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(timestamps, y_true, label="Actual")
    plt.plot(timestamps, y_pred, label="Predicted", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        log.info(f"Saved plot -> {save_path}")

    if show:
        plt.show()

    plt.close()


# -----------------------------------------------------------------------------
# Walk-forward / rolling backtest scaffold
# -----------------------------------------------------------------------------

def walk_forward_backtest(
    X_full: np.ndarray,
    y_full: np.ndarray,
    t_full: Iterable[pd.Timestamp],
    train_size: int,
    step_size: int,
    fit_fn,
    predict_fn,
    inverse_fn=None,
) -> pd.DataFrame:
    """
    Rolling-origin evaluation:
    - Take the first `train_size` points, fit model
    - Predict the next `step_size`
    - Slide forward by `step_size`
    - Repeat

    fit_fn(X_train, y_train) should *train and return* a model object
    predict_fn(model, X_input) should return predictions
    inverse_fn(optional) to invert scaling on preds and y

    Returns dataframe with columns:
    ['timestamp','y_true','y_pred']
    """

    results = []

    start = 0
    n = len(X_full)

    while True:
        train_end = start + train_size
        test_end = train_end + step_size
        if test_end > n:
            break  # not enough future data left

        X_train = X_full[start:train_end]
        y_train = y_full[start:train_end]

        X_test = X_full[train_end:test_end]
        y_test = y_full[train_end:test_end]

        t_test = list(t_full[train_end:test_end])

        model = fit_fn(X_train, y_train)
        y_hat = predict_fn(model, X_test)

        if inverse_fn is not None:
            y_test_inv = inverse_fn(y_test)
            y_hat_inv = inverse_fn(y_hat)
        else:
            y_test_inv = y_test
            y_hat_inv = y_hat

        for ts, yt, yp in zip(t_test, y_test_inv, y_hat_inv):
            results.append(
                {
                    "timestamp": ts,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )

        start += step_size

    out_df = pd.DataFrame(results)
    return out_df


# -----------------------------------------------------------------------------
# Model persistence
# -----------------------------------------------------------------------------

def save_model(
    model,
    scalers: Optional[Scalers],
    cfg: Optional[RunConfig],
    out_dir: str,
    model_name: str = "model",
) -> None:
    """
    Save model weights (pickle or model.state_dict()), scalers, and config
    into <out_dir>.

    You can adapt this for Keras/PyTorch:
      - For Keras: model.save(os.path.join(out_dir, f"{model_name}.h5"))
      - For PyTorch: torch.save(model.state_dict(), path)
    For now we assume a pickle-able model.
    """
    ensure_dir(out_dir)

    # model
    model_path = os.path.join(out_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved model -> {model_path}")

    # scalers
    if scalers is not None:
        scalers_path = os.path.join(out_dir, "scalers.pkl")
        with open(scalers_path, "wb") as f:
            pickle.dump(scalers, f)
        log.info(f"Saved scalers -> {scalers_path}")

    # config
    if cfg is not None:
        cfg_path = os.path.join(out_dir, "config.json")
        save_config(cfg, cfg_path)


def load_model(
    out_dir: str,
    model_name: str = "model",
    load_scalers: bool = True,
    load_cfg: bool = True,
) -> Tuple[object, Optional[Scalers], Optional[RunConfig]]:
    """
    Load model + scalers + config from disk.
    """
    model_path = os.path.join(out_dir, f"{model_name}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    log.info(f"Loaded model <- {model_path}")

    scalers_obj = None
    if load_scalers:
        scalers_path = os.path.join(out_dir, "scalers.pkl")
        if os.path.exists(scalers_path):
            with open(scalers_path, "rb") as f:
                scalers_obj = pickle.load(f)
            log.info(f"Loaded scalers <- {scalers_path}")
        else:
            log.warning("No scalers.pkl found")

    cfg_obj = None
    if load_cfg:
        cfg_path = os.path.join(out_dir, "config.json")
        if os.path.exists(cfg_path):
            cfg_obj = load_config(cfg_path)
            log.info(f"Loaded config <- {cfg_path}")
        else:
            log.warning("No config.json found")

    return model, scalers_obj, cfg_obj


# -----------------------------------------------------------------------------
# Convenience end-to-end prep helper
# -----------------------------------------------------------------------------

def prepare_timeseries_supervised(
    raw_df: pd.DataFrame,
    cfg: RunConfig,
) -> Dict[str, object]:
    """
    High-level convenience:
    1. Feature engineering
    2. Forward target
    3. Build X,y,t
    4. Chronological split
    5. Fit scalers on train only
    6. Scale all splits
    7. Build sequences for each split

    Returns a dict with:
    {
        'splits': {...original splits with unscaled y...},
        'scalers': Scalers,
        'seq': { 'train': (X_seq_train, y_seq_train, t_seq_train), ... }
    }

    NOTE:
    - cfg.feature_cols will get set here to lock in columns.
    """

    # 1. engineered features (lags, indicators)
    df_feat = raw_df.copy()
    df_feat = add_returns_and_lags(df_feat, price_col=cfg.target_col)
    df_feat = add_technical_indicators(df_feat, price_col=cfg.target_col)
    df_feat = forward_target(df_feat, price_col=cfg.target_col, horizon=cfg.horizon)

    target_name = f"{cfg.target_col}_future_{cfg.horizon}"

    # 2. build matrices
    X, y, t = build_feature_matrix(
        df_feat,
        timestamp_col=cfg.timestamp_col,
        target_col=target_name,
        dropna=True,
    )

    # persist which columns we actually used
    cfg.feature_cols = list(X.columns)

    # 3. split
    splits_tab = train_val_test_split_time(
        X, y, t,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    # 4. scalers from TRAIN ONLY
    scalers = fit_scalers(
        splits_tab["X_train"],
        splits_tab["y_train"],
        scaler_type=cfg.scaler_type,
    )

    # 5. apply scalers to each split
    Xtr_s, ytr_s = apply_scalers(scalers, splits_tab["X_train"], splits_tab["y_train"])
    Xva_s, yva_s = apply_scalers(scalers, splits_tab["X_val"], splits_tab["y_val"])
    Xte_s, yte_s = apply_scalers(scalers, splits_tab["X_test"], splits_tab["y_test"])

    # 6. sequences
    Xtr_seq, ytr_seq = make_sequences(Xtr_s, ytr_s, cfg.lookback, cfg.horizon)
    ttr_seq = align_sequences_with_timestamps(
        splits_tab["t_train"], cfg.lookback, cfg.horizon
    )

    Xva_seq, yva_seq = make_sequences(Xva_s, yva_s, cfg.lookback, cfg.horizon)
    tva_seq = align_sequences_with_timestamps(
        splits_tab["t_val"], cfg.lookback, cfg.horizon
    )

    Xte_seq, yte_seq = make_sequences(Xte_s, yte_s, cfg.lookback, cfg.horizon)
    tte_seq = align_sequences_with_timestamps(
        splits_tab["t_test"], cfg.lookback, cfg.horizon
    )

    out = {
        "splits": splits_tab,
        "scalers": scalers,
        "seq": {
            "train": (Xtr_seq, ytr_seq, ttr_seq),
            "val":   (Xva_seq, yva_seq, tva_seq),
            "test":  (Xte_seq, yte_seq, tte_seq),
        },
        "config": cfg,
    }

    return out