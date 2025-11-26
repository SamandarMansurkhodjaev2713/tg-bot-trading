import os
import glob
import pandas as pd

def _read_csv(path):
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
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    return df

def discover_project_data():
    roots = ["data", "data/custom", "data/external", "models", "logs"]
    files = []
    for r in roots:
        files.extend(glob.glob(os.path.join(r, "**", "*.csv"), recursive=True))
        files.extend(glob.glob(os.path.join(r, "**", "*.json"), recursive=True))
    datasets = {}
    for p in files:
        try:
            if p.lower().endswith(".csv"):
                df = _read_csv(p)
            else:
                df = pd.read_json(p)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", dayfirst=True)
                df = df.dropna().sort_values("timestamp").reset_index(drop=True)
            if {"open", "high", "low", "close"}.issubset(df.columns):
                key = os.path.basename(p)
                datasets[key] = df
        except Exception:
            continue
    return datasets
