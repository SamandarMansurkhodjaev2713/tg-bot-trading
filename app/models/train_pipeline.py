import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from joblib import dump
from services.data_loader import discover_project_data as dl_discover
from services.preprocess import build_dataset as pp_build_dataset, preprocess as pp_preprocess
from services.evaluation import evaluate as ev_evaluate, fine_tune as ev_fine_tune
from utils.logging import save_logs as ut_save_logs
from models.finworld import FinWorldAnalyst as MFinWorld
from models.flag_trader import FlagTraderAnalyst as MFlagTrader
from models.finbloom import FinBloom7BAnalyst as MFinBloom


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {
        "UTC": "timestamp",
        "Timestamp": "timestamp",
        "Date": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=cols)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", dayfirst=True)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def discover_project_data() -> Dict[str, pd.DataFrame]:
    roots = ["data", "data/custom", "data/external", "models", "logs"]
    files = []
    for r in roots:
        files.extend(glob.glob(os.path.join(r, "**", "*.csv"), recursive=True))
        files.extend(glob.glob(os.path.join(r, "**", "*.json"), recursive=True))
    datasets: Dict[str, pd.DataFrame] = {}
    for p in files:
        try:
            if p.lower().endswith(".csv"):
                df = _read_csv(p)
            else:
                df = pd.read_json(p)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna().sort_values("timestamp").reset_index(drop=True)
            if {"open", "high", "low", "close"}.issubset(df.columns):
                key = os.path.basename(p)
                datasets[key] = df
        except Exception:
            continue
    return datasets


def _winsorize_returns(df: pd.DataFrame) -> pd.DataFrame:
    r = df["close"].pct_change().fillna(0)
    q1 = r.quantile(0.01)
    q99 = r.quantile(0.99)
    r = r.clip(q1, q99)
    df["ret1"] = r
    return df.dropna()


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal
    low = df["low"]
    high = df["high"]
    k = (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9)
    df["stoch_k"] = 100 * k
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df.dropna()


def build_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name, df in datasets.items():
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue
        df = df.copy()
        df = _winsorize_returns(df)
        df = _indicators(df)
        df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1.0
        df["cls_target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df["source"] = name
        frames.append(df.dropna())
    if not frames:
        raise RuntimeError("No datasets found")
    data = pd.concat(frames, axis=0, ignore_index=True)
    cols_keep = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret1",
        "sma_10",
        "sma_20",
        "ema_10",
        "ema_20",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
        "future_return_5",
        "cls_target",
    ]
    data = data[cols_keep].dropna()
    return data


def preprocess(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = data.copy()
    X = data.drop(["future_return_5", "cls_target"], axis=1)
    y_cls = data["cls_target"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    dump(scaler, "models/scaler_main.pkl")
    X_train, X_val, y_train, y_val = train_test_split(Xs, y_cls, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val


class FinWorldAnalyst:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    def save(self, path: str) -> None:
        dump(self.model, path)


class FlagTraderAnalyst:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    def save(self, path: str) -> None:
        dump(self.model, path)


class FinBloom7BAnalyst:
    def __init__(self):
        self.model = GradientBoostingRegressor()
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    def save(self, path: str) -> None:
        dump(self.model, path)


def evaluate(models: Dict[str, object], X_val: np.ndarray, y_val_cls: np.ndarray, X_train: np.ndarray, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        try:
            if hasattr(m, "predict"):
                y_pred = m.predict(X_val)
                if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
                    mae = mean_absolute_error(data["future_return_5"].iloc[-len(y_pred):].values, y_pred)
                    rmse = mean_squared_error(data["future_return_5"].iloc[-len(y_pred):].values, y_pred, squared=False)
                    metrics[name] = {"mae": float(mae), "rmse": float(rmse)}
                else:
                    acc = accuracy_score(y_val_cls, y_pred)
                    prec = precision_score(y_val_cls, y_pred, zero_division=0)
                    rec = recall_score(y_val_cls, y_pred, zero_division=0)
                    metrics[name] = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec)}
        except Exception:
            continue
    return metrics


def fine_tune(models: Dict[str, object], metrics: Dict[str, Dict[str, float]], X_train: np.ndarray, y_train_cls: np.ndarray, data: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    updated = dict(models)
    acc_threshold = 0.60
    rmse_threshold = 0.015
    fw_m = metrics.get("FinWorld", {})
    if fw_m.get("accuracy", 1.0) < acc_threshold and isinstance(updated.get("FinWorld"), FinWorldAnalyst):
        m = FinWorldAnalyst()
        m.fit(X_train, y_train_cls)
        m.save("models/finworld_ft.pkl")
        updated["FinWorld"] = m
    fl_m = metrics.get("FLAG-Trader", {})
    if fl_m.get("accuracy", 1.0) < acc_threshold and isinstance(updated.get("FLAG-Trader"), FlagTraderAnalyst):
        m = FlagTraderAnalyst()
        m.fit(X_train, y_train_cls)
        m.save("models/flag_trader_ft.pkl")
        updated["FLAG-Trader"] = m
    fb_m = metrics.get("FinBloom-7B", {})
    if fb_m.get("rmse", 0.0) > rmse_threshold and isinstance(updated.get("FinBloom-7B"), FinBloom7BAnalyst):
        y_train_reg = data["future_return_5"].values[: len(X_train)]
        m = FinBloom7BAnalyst()
        m.fit(X_train, y_train_reg)
        m.save("models/finbloom_7b_ft.pkl")
        updated["FinBloom-7B"] = m
    X_val = X_train[-max(1, int(0.2 * len(X_train))):]
    y_val_cls = y_train_cls[-len(X_val):]
    new_metrics = evaluate(updated, X_val, y_val_cls, X_train, data)
    return updated, new_metrics


def save_logs(metrics: Dict[str, Dict[str, float]]) -> None:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    with open("reports/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open("logs/train.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] TRAINED: {lines}\n")
    with open("logs/test.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] EVAL: {lines}\n")


def main() -> None:
    os.makedirs("models", exist_ok=True)
    datasets = dl_discover()
    data = pp_build_dataset(datasets)
    X_train, X_val, y_train, y_val = pp_preprocess(data)
    y_train_reg = data["future_return_5"].values[: len(X_train)]
    finworld = MFinWorld()
    finworld.fit(X_train, y_train)
    finworld.save("models/finworld.pkl")
    flag_trader = MFlagTrader()
    flag_trader.fit(X_train, y_train)
    flag_trader.save("models/flag_trader.pkl")
    finbloom_7b = MFinBloom()
    finbloom_7b.fit(X_train, y_train_reg)
    finbloom_7b.save("models/finbloom_7b.pkl")
    models = {"FinWorld": finworld, "FLAG-Trader": flag_trader, "FinBloom-7B": finbloom_7b}
    metrics = ev_evaluate(models, X_val, y_val, X_train, data)
    models, metrics_ft = ev_fine_tune(models, metrics, X_train, y_train, data)
    ut_save_logs({"initial": metrics, "fine_tuned": metrics_ft})


if __name__ == "__main__":
    main()
