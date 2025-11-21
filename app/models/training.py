import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from ..risk.risk import position_size
from ..exness.config import COSTS
from ..features.indicators import compute_features
from ..services.quotes import get_ohlc

def _label(df: pd.DataFrame):
    r = df["close"].pct_change().shift(-1)
    y = np.where(r > 0.0002, 1, np.where(r < -0.0002, -1, 0))
    return pd.Series(y, index=df.index)

def analyze_signal(pair: str, tf: str, df: pd.DataFrame, feats: pd.DataFrame):
    y = _label(feats)
    X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
    tscv = TimeSeriesSplit(n_splits=3)
    clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    best = None
    for train_idx, test_idx in tscv.split(X):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = clf.predict(X.iloc[test_idx])
        a = accuracy_score(y.iloc[test_idx], p)
        best = clf
    last = X.iloc[-1:]
    pred = int(best.predict(last)[0])
    size = position_size(pair, tf, feats)
    sl = float(df["close"].iloc[-1] - feats["atr_14"].iloc[-1] * 1.5) if pred == 1 else float(df["close"].iloc[-1] + feats["atr_14"].iloc[-1] * 1.5)
    tp = float(df["close"].iloc[-1] + feats["atr_14"].iloc[-1] * 2.0) if pred == 1 else float(df["close"].iloc[-1] - feats["atr_14"].iloc[-1] * 2.0)
    return {"pair": pair, "tf": tf, "action": "buy" if pred==1 else "sell" if pred==-1 else "hold", "size": size, "sl": sl, "tp": tp, "explanation": {"indicators": {"rsi": float(feats["rsi_14"].iloc[-1]), "adx": float(feats["adx_14"].iloc[-1])}}}

def train_models(pairs: list[str], tf: str):
    reports = []
    for p in pairs:
        df = get_ohlc(p, tf, 2000)
        feats = compute_features(df)
        y = _label(feats)
        X = feats.drop(columns=[c for c in ["open","high","low","close","adj_close","volume"] if c in feats.columns])
        tscv = TimeSeriesSplit(n_splits=3)
        clf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
        scores = []
        for tr, te in tscv.split(X):
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[te])
            scores.append(accuracy_score(y.iloc[te], pred))
        reports.append({"pair": p, "accuracy": float(np.mean(scores))})
    return {"results": reports}