import os
import pandas as pd
from typing import Dict, List

def load_calendar(path: str = "data/news") -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    if not os.path.isdir(path):
        return out
    for fn in os.listdir(path):
        if not fn.lower().endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(path, fn))
        if df is None or len(df) == 0:
            continue
        if "timestamp" not in df.columns or "impact" not in df.columns:
            continue
        sym = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) > 0 else os.path.splitext(fn)[0]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        events = []
        for _, r in df.iterrows():
            events.append({
                "timestamp": r["timestamp"],
                "impact": str(r["impact"]).lower(),
                "symbol": str(r.get("symbol", sym)).upper()
            })
        out[sym.upper()] = events
    return out

def is_blocked(ts, pair: str, calendar: Dict[str, List[Dict]], window_minutes: int = 60) -> bool:
    if ts is None:
        return False
    pair = str(pair).upper()
    for key, events in calendar.items():
        if key not in pair and pair not in key:
            continue
        for ev in events:
            evts = ev["timestamp"]
            if evts is None:
                continue
            dt = abs((ts - evts).total_seconds())/60.0
            if dt <= window_minutes and ev["impact"] in {"high","medium"}:
                return True
    return False
