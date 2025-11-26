import os
import json
import pandas as pd
from typing import Dict

def load_sentiment(path: str = "data/sentiment") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(path):
        return out
    for fn in os.listdir(path):
        p = os.path.join(path, fn)
        if fn.lower().endswith(".csv"):
            df = pd.read_csv(p)
        elif fn.lower().endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            continue
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        key = os.path.splitext(fn)[0].upper()
        out[key] = df
    return out

def get_sentiment(pair: str, ts, cache: Dict[str, pd.DataFrame]) -> float:
    if ts is None or not cache:
        return 0.0
    pair = str(pair).upper()
    score = 0.0
    cnt = 0
    for key, df in cache.items():
        if "value" not in df.columns:
            continue
        s = df.iloc[(df["timestamp"] - ts).abs().values.argmin()]
        val = float(s["value"]) if "value" in s else 0.0
        if "FEAR_GREED" in key and "BTC" in pair:
            val = (val - 50.0)/50.0
        elif "EXPERTS" in key:
            val = (val - 0.5)*2.0
        elif "WHALES" in key:
            val = (val - 0.5)*2.0
        else:
            val = max(min(val, 1.0), -1.0)
        score += val
        cnt += 1
    if cnt == 0:
        return 0.0
    return max(min(score / cnt, 1.0), -1.0)
